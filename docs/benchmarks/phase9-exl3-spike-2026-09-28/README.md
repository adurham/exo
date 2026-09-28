# Phase 9 — EXL3 MoE kernel spike (day 1 + day-2 closure)

Date: 2026-09-28 (day 1 of the 2-3 day window)
Machine: macstudio-m4-1, isolated `~/phase1-exl3` venv, live cluster untouched
Status: **CLOSED (day 2, p19 + p20)** — kernel thread ended at ~1.9-2.0x
full-width vs mxfp4; see "Day-2 closure" at the end. Day-1 summary: the 2.65-3.1x gap was diagnosed to
its source, two bit-identical fixes were found, applied and verified; the
remaining headroom was then measured to be small. Path A's prospects
materially improved; the 1.25x plan gate is provably out of reach for this
kernel design (floor ~1.55-1.68x).

## Bottom line

The gap is NOT the trellis format's memory pattern and NOT the kernel
skeleton — it is the kernel's own per-codeword ALU chains (codebook decode +
cross-lane broadcast). Two rewrites of those chains were proven
**bit-identical** (md5 of outputs unchanged at every measured shape) and are
now the library default on node 1:

- mul1 codebook decode: 4-iteration byte-sum loop -> SWAR packed half-sums
- cross-lane broadcast (`simd_shuffle`) -> direct indexed read

| shape | before (control) | after | ratio vs prod mxfp4 |
|---|---:|---:|---:|
| decode R=1 | 1.031 ms (2.667x) | **0.733 ms (1.909x)** | gate 1.25: still MISS |
| verify R=4 | 3.169 ms (3.005x) | **2.086 ms (1.979x)** | gate 1.25: still MISS |
| verify R=6 | 4.609 ms (3.094x) | **3.046 ms (2.048x)** |  |
| verify R=8 | 6.042 ms (3.087x) | **3.938 ms (2.046x)** |  |

40-layer MoE-only decode ceiling: **24.1 -> 34.0 tok/s**. The gap is now
~1.9-2.0x (was ~2.65-3.1x) and the diagnosis says the remainder is further
ALU work, not format overhead — with the idealized floor (2-op decode, no
shuffle, loads live) measured at **1.55-1.68x**, i.e. the remaining headroom
is real and bounded.

Day-2 target for the A-vs-B decision: push toward the ~1.5x gate region;
stop at 1.7x+ only if the next two candidate classes also fail.

## Step 0 — fresh baseline (reproduces phase 2 exactly)

| rows | EXL3 ms | mxfp4 ms | ratio | (2026-09-27 ref) |
|---:|---:|---:|---:|---|
| 1 | 1.033 | 0.387 | 2.666 | 1.029 / 2.649 |
| 4 | 3.168 | 1.023 | 3.096 | 3.171 / 3.035 |
| 8 | 6.049 | 1.933 | 3.129 | 6.050 / 3.114 |
| 256 | 147.649 | 39.109 | 3.775 | 149.588 / 3.773 |

## Experiment 1 — geometry sweep (NEGATIVE, closed)

20 configs of `EXL3_MOE_A2_TILES` x `EXL3_MOE_CHUNK`, 60 reps each, fresh
process per config. R=1 medians:

| config | R=1 | | config | R=1 |
|---|---:|---|---|---:|
| **T1-C0 (default)** | **1.037** | | T4-C0 | 1.117 |
| T1-C128 | 1.051 | | T4-C1152 | 1.244 |
| T1-C384 | 1.057 | | T8-C0 | 1.175 |
| T1-C768 | 1.038 | | T8-C1152 | 1.303 |
| T1-C1152 | 1.164 | | T2-C0 | 1.036 |

Default is the optimum; every knob is neutral-to-worse. Re-ran on the patched
library (5 configs): still T1-C0 (0.758 vs 0.770-0.905 for the rest). Geometry
is exhausted — do not revisit.

## Experiment 2 — dispatch-path alternatives (NEGATIVE, closed)

`EXL3_MOE_FUSED=0` (v1 unfused): R=1 0.936 vs 1.033 — better at R=1 alone, but
R=4/8 collapse to the prefill sort machinery (11-12x). On the patched library:
2.317x at R=1 but 11.861x at R=4. Not adoptable. `EXL3_MOE_V2=0`: worse
(1.298). Both closed.

## Experiment 3 — stub probe: where the time goes (decisive)

Diagnostic stubs (wrong values on purpose, `p14_stub_probe.py`):

| arm | R=1 | ratio | R=4 | R=8 |
|---|---:|---:|---:|---:|
| none (control) | 1.030 | 2.650 | 3.170 | 6.044 |
| decode-stubbed | 0.690 | 1.786 | 2.036 | 3.860 |
| shuffle-stubbed | 0.732 | 1.950 | 2.005 | 3.766 |
| **both stubbed** | **0.290** | **0.779** | 0.549 | 0.937 |

With both chains stubbed the kernel is **faster than production mxfp4** —
so the skeleton, dispatch structure and trellis read pattern are not the
problem. The two chains are each worth ~30% and overlap (removing one helps
by more than half the pair).

## Experiment 4 — bandwidth probe (rules out the layout)

Same bytes through a trivial reduction: contiguous fp32 408 GB/s vs the real
trellis uint32 views **451-469 GB/s** (at 304-607 MB). The trellis layout
streams at full device bandwidth; the earlier "78 GB/s achieved" was the
kernel's effective rate, not the layout's.

## Experiment 5 — idealized floor (bounds the remaining work)

`p15_chain_probe.py` replaces the decode with ~2 ops (loads still live via
data dependency) and optionally drops the shuffle:

| arm | R=1 | ratio | R=4 | R=6 | R=8 |
|---|---:|---:|---:|---:|---:|
| cheap (2-op decode, shuffle kept) | 0.866 | 2.233 | 2.551 | 3.709 | 4.843 |
| cheapshuf (2-op + no shuffle) | 0.653 | **1.677** | 1.636 | 2.383 | 3.060 |

Floor estimate: ~1.55-1.68x at R=1 with an ideal decode — the current
adopted state (1.91x) is within ~15-20% of that floor.

## Experiment 6 — the two adopted, bit-identical rewrites

`p16_direct_probe.py`:

1. **SWAR decode** — `dq_sum = 0x6400 + sum(bytes of cw*K)` as two packed
   half-sums instead of a 4-iteration loop. Integer add is associative; max
   0x6400+4*0xFF cannot overflow. Standalone: R=1 1.031 -> 1.000 ms.
2. **Direct read** — `simd_shuffle(x_lane, ushort(row_[jw]))` broadcasts lane
   `row_[jw]`'s `x_lane`, which holds `x[..., (row_[jw] & 15)]`; a direct
   indexed read is the same fp16 value with no cross-lane op. Standalone:
   R=1 -> 0.823 ms.

Combined: **R=1 0.756 ms (1.937x)**, R=4 2.088 (1.986x), R=6 3.044 (2.049x),
R=8 3.941 (2.015x) — **md5-identical at every shape**.

### Rejected candidates (day 1)

- **uchar4 byte-sum** (`p17`): slower (0.804 vs 0.733). Rejected on timing.
- **threadgroup LUT for the decode tail** (`p18`): slower (0.829) AND
  md5-MISMATCHED — two independent rejection signals. Rejected.
- FUSED=0, V2=0, all geometry knobs: closed above.

## Adoption + verification (the state on node 1 now)

Applied by `apply_v4_alu_optim.py` (backups `.bak-v4`, revert switch, patch
file `~/exl3-moe-v4-alu-optim.patch`). Both changes are env-gated, default ON:
`EXL3_DECODE_SWAR`, `EXL3_XDIRECT`.

Verification (same harness, same day):

| measurement | library defaults (ON) | gates forced OFF |
|---|---:|---:|
| p2_final decode R=1 | **1.899x** | 2.655x |
| p2_final verify R=4 | **1.988x** | 3.022x |
| p2_final prefill R=2048 | 1.662x | 1.655x |
| p16 R=1 | 0.733 ms, md5 match | 1.031 ms, md5 match |

md5 of outputs: identical to the pre-change reference at R=1/4/6/8 in BOTH
configurations.

Package tests (scratch venv, pytest via uv): `moe_activation` 2 passed 1
skipped, `mlx_gemv` 17 passed 6 skipped, `codebook` 17 passed,
`gemma4_model` 2 passed 3 skipped — **38 passed, 10 skipped, 0 failed** on
the patched tree.

## Chunk-verify divergence (diagnostics, not fixed today)

Running `p6b_chunk_equiv.py` under the FORK mlx (node1 exo venv) with
`MLX_GEMV_BATCH_INVARIANT` / `MLX_STEEL_BATCH_INVARIANT` in three arms:
divergence persists in all three (flags shift individual token values;
`p6c_localize` shows the fork+flags arm converges the FINAL ARGMAX at the
short prompt but layer-0 hiddens still differ at ~bf16-ulp scale). Conclusion:
this is a batch-shape numerics property of the port's body, not a flag away
from fixed — consistent with production gating its own batched verify off at
short context (`EXO_DSV4_VERIFY_BATCH_MIN_CTX`). The deployable conclusion
stands: rowseq = validation instrument, chunk = deployment shape with the
acceptance-rate cost measured separately.

## Files (all in this directory)

- `scripts/` — p14/p15/p16/p17/p18 probes, apply script, arm runners,
  resweep/verify/libcheck/test runners, bwprobe, p6b
- `raw/` — every log referenced above (step0, sweep, sweep2, stubs, chains,
  direct, uchar4, lut, verify, libcheck, tests, bwprobe)
- `exl3-moe-v4-alu-optim.patch` — the adopted diff (86 lines)

Kernel-side artifacts live on node 1: patch `~/exl3-moe-v4-alu-optim.patch`,
pre-state backups `ponyexl3/mlx/{gemv_metal,exl3_moe}.py.bak-v4`; v3 patch
`~/exl3-moe-v3-chunked-prologue.patch` unchanged.

## Day-2 candidates (ranked)

1. **Decode-chain restructure**: the idealized floor says ~0.65 ms is
   reachable; current 0.73. The next structural idea: fold the decode tail
   into the FMA via a different representation, or cut the number of decode
   instances per tile (share across the 8 accumulator lanes).
2. **Shuffle-chain residue**: direct-read removed the cross-lane op; re-run
   the stub probe on the PATCHED kernel to see what the chain costs now
   (it may have moved or changed character).
3. **A2/B2 boundary**: re-evaluate whether the A2 raw-output -> B2 prologue
   handoff is still optimal now that the decode is cheaper.

## Addendum — patched-kernel chain re-probe (p19, same night)

Re-ran the two-arm chain probe ON the patched kernel (library defaults):

| arm | R=1 | ratio | R=4 | R=6 | R=8 |
|---|---:|---:|---:|---:|---:|
| control (patched) | 0.744 | 1.916 | 2.088 | 3.044 | 3.946 |
| cheap (ideal decode, loads live) | 0.688 | 1.774 | 1.852 | 2.692 | 3.472 |

**Decode ALU is nearly exhausted**: an IDEAL decode buys only ~8% at R=1
and ~11% at R=4 over the adopted state. Cross-check: the cheap arm's md5s
(`4a50c65d...`, `75b1b289...`) are byte-identical to p15's cheap arm
yesterday — the measurement is stable across kernels and days.

Consequence for the fork decision: the residual headroom in this kernel
design is now bounded at roughly **1.77x (R=1) / 1.76x (R=4)** even with
infinite decode cleverness, and the realizable number is ~1.9x. The plan's
1.25x gate is NOT reachable by ALU work on this architecture — a different
kernel architecture (or accepting B) is the only route past ~1.8x.

Ranked day-2 options, updated:
1. **Accept ~1.9x** and re-cost path A with the REAL numbers (decode
   1.909x / verify 1.979x) — the consult's "move it toward ~1.5x" test is
   answered: it moved 2.65 -> 1.91, not to 1.5, and the floor says it can't.
2. **One structural experiment**: the A2/B2 boundary + dispatch count
   (the only remaining lever with unknown size — not ALU).
3. If neither lands, the kernel says B (streamed experts) is the build;
   A stays available as the slow-but-resident fallback at ~1.9x MoE cost
   (which still nets a viable end-to-end V4.1 — see the re-cost in phase 8
   terms).

## Day-2 closure — x-access probe (p20) — KERNEL THREAD CLOSED

p19 bounded the decode-ALU residue (above). p20 sized the last named lever:
the per-jw x ACCESS (device indexed read, adopted `EXL3_XDIRECT`) vs a
register value. Arms: `xdir` = library defaults (md5 must match REF);
`imm` = register value substituted — a diagnostic FLOOR with deliberately
wrong values (md5 mismatch expected).

| rows | xdir ms (ratio) | imm floor ms (ratio) | prize |
|---:|---:|---:|---:|
| 1 | 0.744 (1.920x) md5 REF | 0.700 (1.803x) | ~6% |
| 4 | 2.087 (1.986x) md5 REF | 1.787 (1.699x) | ~14% |
| 6 | 3.034 (2.046x) md5 REF | 2.590 (1.743x) | ~15% |
| 8 | 3.942 (2.021x) md5 REF | 3.319 (1.699x) | ~16% |

Reading: at decode (R=1) the device read is effectively free (L1). At verify
shapes a ~14-16% prize exists on paper, but reaching it needs a
register/shared-memory select that the tile loop's lane layout does not allow
(the value needed lives in another lane — that is exactly why the original
kernel used `simd_shuffle`). No realizable form was found. xdir numbers
reproduce the day-1 adoption (0.733-0.744 / 2.086-2.087) — stable.

**Verdict: kernel work stops at ~1.9-2.0x full-width.** The plan-B day-2 gate
(phase 10) then chose path A (EXL3-resident), and phase 11's serving-geometry
addendum re-priced the MoE at the real TP=2 half-width shape (1.61-1.74x,
g5 ~25-31 tok/s). Raw: `raw/p20-xacc-{xdir,imm,driver}.log`; scripts:
`scripts/p20_xaccess_probe.py`, `scripts/p20_xacc_arms.sh`.
