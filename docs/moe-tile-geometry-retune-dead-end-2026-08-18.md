# MoE tile-geometry retune: confirmed dead end at production's real shape (2026-08-18)

> **CORRECTION 2026-08-19**: the "62.6% of dense ceiling" / "~37% gap"
> figures below use the WRONG denominator (dense fp16 GEMM, uncorrected
> for the ~13% mxfp4 quant tax that applies even with zero MoE routing).
> Against the correct denominator (dense mxfp4-quantized matmul at the
> same shape), true MoE-specific efficiency is **72.0-72.5% at L=2048**
> (28% gap, not 37%) -- see
> `docs/moe-vs-dense-qmm-isolation-2026-08-19.md`. This doc's core
> conclusion (tile-geometry retune is a dead end) is UNCHANGED by the
> correction -- larger tiles still measured worse, independent of which
> denominator you use for the headline percentage. Only the "~37% gap"
> framing below is stale; treat "~28% gap, driven by per-expert-run
> raggedness (median M=7-14), not tile geometry" as the corrected
> framing.

## Context

Following up on the discovery that the MoE GEMM kernels (~97% of real MoE
cost per `docs/moe-per-stage-gpu-breakdown-2026-08-18.md`) account for the
unexplained ~37% gap between measured efficiency (62.6% of dense ceiling)
and 100%. A prior July 2026 tile-geometry sweep
(`bench/qmm_tile_sweep.py`) had concluded "default 16,32,32,1,2 is local
optimum... falsified tile-retune" -- but that sweep tested a
non-representative UNIFORM 6-rows-per-expert distribution (256 tokens),
not production's real ragged/skewed shape (2048-token chunks, median 14
rows/expert, range 1-1653). This investigation re-examined whether the
July conclusion actually transfers to production's real shape, or whether
it was an artifact of testing an unrepresentative distribution.

## Method

1. **K1 (kernel source read)**: traced the actual dispatch path for
   production's real MoE call shape (M=1 after `expand_dims`, B/E=48) and
   confirmed it hits `gather_qmm_rhs` (`quantized.cpp:1313`, dispatched at
   `quantized.cpp:1914-1931`) -- a DIFFERENT function from the one the July
   sweep and its env override (`MLX_GATHER_QMM_RHS_LHS_TILE`) target
   (`gather_qmm_rhs_lhs`, dispatched separately at `quantized.cpp:1940` for
   `M>=16` calls, which production's `M=1` shape never reaches). This
   session independently re-traced `gather_qmm_rhs`'s actual kernel body
   (`affine_gather_qmm_rhs`, `quantized.h:2568+`) and confirmed it has the
   IDENTICAL run-boundary-scan-then-full-tile-GEMM mechanism K1 found in
   the sibling kernel: a `while (n < tgp_bm)` loop scans the sorted
   `indices` array for expert-boundary changes within each `BM`-row tile,
   and executes one full `BM x BN x K` `BlockMMA` per distinct expert-run
   found inside that tile (`quantized.h:2650-2662` boundary scan,
   `quantized.h:2665+` per-run MMA). Tile geometry is HARDCODED
   (`bm=16, bn=32, bk=32`, `quantized.cpp:1373`) -- not wired to any env
   override, so a real sweep of this specific kernel would require a code
   change, not just a build/config change.
2. **K2 (analytic waste prediction)**: extended the existing production-
   accurate tile-waste model (`bench/moe_run_length_tile_waste.py`,
   already validated: reproduces the documented 20.7% waste at bm=16
   bit-for-bit) to bm=32 and bm=64, holding bn=32/bk=32 fixed, across 8
   random seeds at production's real ragged routing distribution.

## Results

| bm | mean waste | std | range across 8 seeds |
|---|---|---|---|
| 16 (current) | 20.4% | 0.73% | 19.5-21.5% |
| 32 | 34.8% | 1.03% | 33.2-36.4% |
| 64 | 52.1% | 1.10% | 50.5-53.7% |

Larger tiles make padding waste monotonically WORSE, with tight variance
across seeds (std ~1%) -- this is a structural result of the distribution
shape, not a lucky/unlucky draw. Mechanism: production's median expert
gets ~8-14 rows, already below the current bm=16 tile height. Doubling
tile height doubles the wasted-capacity floor for the ~200 small experts
per chunk, while the few large experts (400-1600+ rows) that would
benefit from bigger tiles are a small minority of the 256-expert
population and don't offset the loss.

Row-weighted sensitivity confirms this isn't distribution-draw-dependent:
the fraction of total ROWS coming from experts whose run-length is a
near-multiple of the tile size (i.e., experts that wouldn't be hurt by a
larger tile) collapses from 69.2% at bm=16 to just 36.4% at bm=64.

## Conclusion

**Tile-geometry retuning is confirmed dead at production's real shape --
this is not an inherited stale conclusion, it's independently re-verified
against the correct distribution and the correct (actually-dispatched)
kernel.** The July sweep's "falsified" verdict happened to be directionally
correct despite testing the wrong shape and (per K1's initial read) the
wrong kernel variant -- but that was not something to assume; it required
this session's re-verification to actually confirm, since the uniform-6
shape and the ragged-production shape are different enough in principle
that the same conclusion did NOT have to hold. It does.

If anything, the math points toward SMALLER tiles (bm=8) as the only
plausibly-productive direction, not larger ones -- but that direction was
not investigated this session (smaller tiles reduce padding waste per this
model, but may hit other costs: more distinct tile launches, lower
per-launch compute-to-overhead ratio, and interacts with a still-open
question about whether the existing small-M `gather_qmv_rhs` kernel
already covers the truly-small-run population better than any tiled
kernel could -- see the earlier hybrid-dispatch NO-GO finding, which found
even a hypothetical zero-cost partition of short runs caps out under 5.6%
end-to-end). Not pursued further tonight -- flagging as the one loose
thread from this specific investigation if a future session wants to chase
it, but expectations should be modest given the hybrid-dispatch ceiling
already found.

## Files

- `bench/moe_tile_size_sweep.py` -- committed, reusable, imports the
  existing `moe_run_length_tile_waste.py` model rather than reimplementing
  it (kept in sync automatically).

## Status of the MoE-kernel-efficiency investigation thread as a whole

This closes the third and final concrete idea explored tonight for the
~37% GEMM-kernel efficiency gap (after: hybrid short/long-run dispatch --
NO-GO, capped <5.6%; gather/scatter overhead -- ruled out, only ~4-5% of
real cost; tile geometry -- now also ruled out, larger tiles predicted
34.8-52.1% worse). No further concrete, cheaply-testable ideas remain from
tonight's investigation for closing this specific gap. Whatever explains
the remaining efficiency shortfall is likely intrinsic to the kernel's
per-run dequant/compute overhead at small-to-medium run sizes rather than
a geometry or dispatch-routing problem -- closing this would require
either a genuinely new kernel design (not a parameter retune of the
existing one) or accepting this as close to the practical ceiling for
this quantization scheme's ragged-MoE access pattern on this hardware.
