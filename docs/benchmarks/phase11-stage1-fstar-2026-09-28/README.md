# Phase 11 -- Stage-1 f* (post-gate pivot) + soak characterization (p24/p25)

Date: 2026-09-28
Host: hermes-gw-01 (analysis) + macstudio-m4-1 / -2 (measurement)
Status: **Stage-1 answered**; soak question **resolved** -- transient mid-run
device dip on BOTH nodes with full recovery; node2 parity confirmed.

## Bottom line

1. **Stage-1: the 25 tok/s bar is not reachable with EXL3 experts on this GPU.**
   Meeting it would need the expert kernels 2-4x faster than anything measured
   *or bounded* (best-known EXL3/production ratios 1.91-2.05), and the non-expert
   path at 0.07-0.46 ms/layer against 1.3-2.1 ms/layer derived from production.
   Expected landing: **mid-teens, central ~15-19 tok/s** (MTP gamma=5), matching
   the independent port-kernel projection (16-19 tok/s). Both build paths (A
   resident, B streamed) land in the same band; A gets there with no I/O cliff.
2. **The soak does not degrade permanently, and node2 is not slow.** Both nodes
   hold ~6.2-6.3 GB/s; in long runs BOTH show a transient dip (~15-25 min in;
   60 s-mean floors ~3.5-4.8 GB/s; full recovery by ~30-32 min). p24 had been
   **cut mid-dip** -- its terminal "node2 ~half speed" reading was that artifact
   (retracted). p25 (full 39 min): medians **n2 6243** vs n1 6076 MB/s;
   last-300 s 6269 vs 5769.
3. **No decision change; adds to A's case.** A streamed build pinned at
   ~5.8 GB/s would starve during dip windows; resident A reads no experts from
   the SSD during decode. Mechanism unidentified (SSD-internal thermal /
   housekeeping, or macOS memory-pressure stalls); not chased further unless a
   streamed build is revived.

## 1. Stage-1 -- the expert-speed factor A would need (f*)

Method: measured expert costs (p20 mxfp4 probe at V4.1 shapes: 0.388 / 1.051 /
1.483 ms/layer at R=1/4/6) x adopted EXL3 ratios (1.909 / 1.979 / 2.048) x 40
layers; per-cycle budgets at 25 tok/s (plain 40.0 ms; MTP gamma=3 86.0 ms; MTP
gamma=5 140.0 ms); non-expert (T_N) anchored on production's own cycle (22.09
tok/s; 2.056 tok/cycle -> 93.1 ms/cycle) and scaled for V4.1 -- DERIVED, flagged;
no direct per-component measurement exists (Prometheus exports none).

| shape | EXL3 MoE, 40L | budget @25 | T_N allowed | vs derived T_N (53-83 ms) |
|---|---:|---:|---:|---|
| plain R=1 | 29.6 ms | 40.0 | <= 10.4 ms (0.26 ms/L) | infeasible |
| MTP g3 (R=4) | 83.2 ms | 86.0 | <= 2.8 ms (0.07 ms/L) | infeasible |
| MTP g5 (R=6) | 121.5 ms | 140.0 | <= 18.5 ms (0.46 ms/L) | 3-4.5x over |

Even granting the idealized-decode bounds (p19 cheap arm: 1.774 R=1 / 1.852 R=4;
nothing better known at R=6), g3 still needs T_N <= 8.1 ms and plain needs
T_N <= 12.5 ms -- still absurd.

Expected landing (current kernels):
- g5, 3.5 tok/cycle: T_N40 53-90 -> **16.5-20.1 tok/s**
- g3, 2.15 tok/cycle: T_N40 53-80 -> 13.2-15.8 tok/s

=> mid-teens band, central **~15-19 tok/s**, before engine/comm overheads; agrees
with the port-kernel projection (16-19). The main software lever left on path A
is a lean, production-grade non-expert path for V4.1 -- it does not exist yet.

Note: the port's own non-expert timing (p40, session record: 5.1-5.8 ms/layer)
is reference-grade (fp32 debug cache; unoptimized attention) and is excluded
from pricing; its output artifact was not preserved (inline output).

## 2. Device soak -- p24 (cut) + p25 (full run)

`iostat` per-second is ground truth; the soak's process-level rate series is
cache-inflated (upper bound) and is never evidence.

**p24** (35 min planned; cut at 25 min to release the 40-layer trace gate):
- both nodes plateau ~6.2-6.3 GB/s (medians 6210 n1 / 6289 n2) through ~t19 min
- dip onsets t~19.8 min (n2) / t~20.5 min (n1); still mid-dip at the cut --
  final samples 2.6-2.9 (n1) / 2.8-3.0 (n2) GB/s
- signature: request size constant (~410 KB/t), IOPS fell to ~42-45% of plateau
- the phase-10 device table (medians 6206 / 6284 MB/s) remains correct for its
  pre-dip core window; this section adds the post-20-min behavior

**p25** (same harness, full 39 min; per-minute swap/vm_stat/thermal sampler):

| phase (t since soak start) | macstudio-m4-1 | macstudio-m4-2 |
|---|---|---|
| plateau (0-720 s) | ~6.16 GB/s (Q1 mean) | ~6.14 GB/s (Q1 mean) |
| soft wobble (~840-1260 s) | 5.2-5.9 GB/s | full speed until ~1260 s |
| deep dip (~1260-1680 s) | 3.8-5.6; 60 s-mean floor **3512 MB/s** (t~1495) | 3.8-5.2; floor **3469 MB/s** (t~1406) |
| recovery (~1680-1920 s) | ~5.5-6.1 GB/s | ~6.28 GB/s |
| last 300 s | mean 5769 MB/s | mean 6269 MB/s |
| run median / p10 / p90 | 6076 / 2768 / 6218 MB/s | 6243 / 2975 / 6292 MB/s |

Environment (sampler): free pages pinned near the floor (4.8-8 k pages,
75-130 MB) from early in both runs; n1 swap flat (36 494 / 36 864 MB), n2 +26 MB
over the run; no thermal or performance warnings (`pmset -g therm`); nothing
else running on either node. No environment correlate to the dip was found;
remaining candidates are SSD-internal (thermal throttling cycles, or media
housekeeping at multi-TB cumulative reads) or macOS memory-pressure stalls.
Left open -- it matters only to a streamed design.

## 3. Implications for the build decision

- **B (streamed native)**: steady-state margin was already 1.08x at the 25 tok/s
  bar (worse with MTP); dip windows (~3.5-4.8 GB/s for ~5-15 min, seen in both
  runs on both nodes) would additionally starve a decoder needing ~5.8 GB/s.
- **A (EXL3-resident)**: decode touches no experts on the SSD; the dips are
  irrelevant beyond one-time load. No decision change: A remains the build.

## Artifacts

- `raw/p41-stage1-fstar.md` -- Stage-1 arithmetic
- `raw/p24-curves-analysis.md`, `raw/p25-curves-analysis.md` -- curve statistics
- `raw/p25-soak-macstudio-m4-1.log`, `raw/p25-soak-macstudio-m4-2.log`
- `raw/p25-iostat-macstudio-m4-1.txt`, `raw/p25-iostat-macstudio-m4-2.txt`
- `raw/p25-env-macstudio-m4-1.log`, `raw/p25-env-macstudio-m4-2.log`
- `raw/p25-driver-macstudio-m4-1.log`, `raw/p25-driver-macstudio-m4-2.log`
- `scripts/p25_driver.sh`, `scripts/p25_env.sh`, `scripts/p25_launch.sh`,
  `scripts/p23_ssd_soak2.py`
