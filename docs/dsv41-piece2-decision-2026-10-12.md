# DSv4.1 — indexer piece-2 DECISION (Phase 3 closeout) — 2026-10-12

**Provenance:** Phase 3 of `docs/dsv41-indexer-abscost-plan-2026-10-12.md` (Fable consult #5), executed same-day.
**Inputs (all MEASURED unless labeled):** profile doc `docs/dsv41-indexer-profile-2026-10-12.md` §0–§1E (splits, crossover, occupancy, replication audit, live window) + the Phase-3 first experiment (`idxprof4/REPORT.md` — reduced-precision score-path probe, ≤2 h, offline, 0 boots).

## 1. Filled decision table (plan §Phase 3)

| Condition (deep, 100K+) | Measured input | Row fires? |
|---|---|---|
| GPU-busy < 80 % during prefill | **92.6–96.8 %** prefill / 100 % decode (§1E) | No — kernels are the wall |
| I < 15 % of chunk time | **~22 % @105K** (~10 % @50K; §1A over §1E walls) | No |
| Top-k/select ≥ 40 % of indexer time → Lever A | select **13.7–14.8 %** of the REF path; HIER select-ish (exact rescore + blocks) 15–32 % | No — below bar |
| K-cache read BW-bound and ≥ 40 % → Lever B | K-read **0.3–1.3 %** (§1A) | No — **Lever B dead** (contradicts its premise) |
| Score GEMM compute-bound & dominant → compute-ceiling clause | dominant **yes** (85 % of REF; coarse 67–83 % of HIER); compute-bound **no** (31 % of fp16 peak) | Probe = the clause's check → **FAIL (§2)** |
| Mixed / no sub-op ≥ 40 % → larger modeled gain, A preferred | nothing reaches its bar; the dominant term has no available faster path | — |

## 2. First-experiment results — reduced-precision score probe (`idxprof4/REPORT.md`)

**Verdict: no int8 (or other reduced-precision) MLX path clears the 1.5× bar at 100K/130K. Best is a plain
bf16/fp16 einsum at 1.34–1.35× wall (1.48–1.49× GPU-time — still below the bar). int8 adds nothing.**

Full tile loop, rows=1024, tile=504, h=32, d=128 (median of 7, ABAB, 17 arms; ms MEASURED):

| variant | 50K ms (×) | 100K ms (×) | 130K ms (×) |
|---|---:|---:|---:|
| ref_fp32 (production `_score_body`, compiled) | 97.68 (1.00) | 192.09 (1.00) | 250.83 (1.00) |
| bf16_full / fp16_full | 71.2–71.6 (**1.37**) | 143.0 (**1.34**) | 185.9–186.1 (**1.35**) |
| q8g64_fp16 (`quantized_matmul`, affine int8 K) | 72.20 (1.35) | 144.69 (1.33) | 187.40 (1.34) |
| mxfp8 / q4g64 (reference) | ~72.3–73.1 (1.34–1.35) | ~144.4–145.0 (1.33) | ~187.5–188.1 (1.33–1.34) |
| qqmm_mxfp8 (both operands quantized) | 348.9 (**0.28**) | 694.1 (**0.28**) | 901.0 (**0.28**) |
| *GEMM only, no epilogue:* fp32 / fp16 / q8g64 | 1.85 / **2.27** / 1.93 | 1.86 / **2.28** / 1.92 | 1.87 / **2.29** / 1.93 |

- **Why it fails (MEASURED ablation):** the epilogue (`relu·w → head-sum` over the materialised `[n,h,t]`
  transient) costs ~40 % of the fp16 variant's time and is precision-independent; and the GEMM is not the
  bottleneck — even an fp16 GEMM alone (2.27–2.29×) barely beats 2×, while the int8 GEMM alone (1.92×) is
  **slower** than the fp16 GEMM (dequant is inside the GEMM). The cost driver is the fp32 materialisation
  (~135–147 GB/s achieved of 448 measured), not the MACs.
- **What MLX on this build (0.32.3.dev) actually supports:** `mx.matmul` int8×int8 → **unsupported**
  ("Only inexact types"); `quantized_matmul` works for affine 2–8 bit / mxfp8 but dequantizes inside an
  fp16/bf16 GEMM (**no int8 math**); `qqmm` (mxfp8×mxfp8) runs at 0.28×; `scaled_dot_product_attention_quant`
  is decode-only and a different expression. There is no hardware int8/fp8 GEMM path on the M4 Max worth using.
- **Numerics (reference: exact fp32 row; for the record, not a ship gate):** int8 g64 — 0/424 real rows
  top-512-identical, recall 0.991 mean / 0.981 min (worse than the production bf16 row: 33/424, 0.994);
  fp16 GEMM + fp32 epilogue — **414/424 identical, recall 0.99995 (0.998)**, near-exact class. q/k are
  already fp4-rounded, so operand-storage precision is not the lever; accumulation/epilogue dtype is.
- **Reproduction checks:** loop cells reproduce the §1A comparators within +5.4–7.7 % (✅, 146→135 GB/s
  consistent); the §1A per-tile body cell (1.102 ms) did **not** reproduce at this measurement style
  (1.55–2.82 ms; launch/cache-dominated among 17 rotating arms) — flagged non-representative; the gate is
  decided on the loop cells, which are the 1A comparator and the production shapes.
- **MODELED e2e band (for the record only):** naive 1.35× × (score 85 % of REF) × (indexer ~22 % @105K) →
  ≤4.8 % of a 105K chunk. **Overstated**: production already runs HIER whose coarse sweep is already bf16 —
  most of any reference-path bf16 gain is already banked. No live increment is claimed.

## 3. Decision

**STOP piece-2.** All three candidate levers are closed by measurement:
1. small-batch bypass — EV 0.015 % vs the 1.0 % gate (bypass skipped, §1);
2. K-cache lower precision (Lever B) — the K-read is 0.3–1.3 % of the indexer (dead on arrival);
3. int8/reduced-precision score path — probe gate **FAIL** (best 1.34–1.35× wall / 1.48–1.49× GPU vs 1.5×;
   no int8 GEMM exists on this build; the precision-independent epilogue + materialisation dominate).

No live work follows from this decision; no further experiment is budgeted under this plan. The workstream's
**0 boots, live untouched** throughout (`60008939` + `6a5c78f4`, runner PIDs 73006/77318, API healthy at every checkpoint).

## 4. Follow-up candidates (each needs its OWN plan and go; NOT started, NOT recommended without one)

1. **Fused GEMM + `relu·w` + head-sum Metal kernel.** Ceiling ≈ **2.27–2.29× (BOUND)** for the GEMM part —
   the only variant family that exceeded 2× in the probe. Must be (a) compared against the HIER **coarse
   sweep** (where most of the modeled gain already lives), not the reference path, and (b) assigned an
   exactness class (fp32 accumulate is near-exact: 0.99995 recall measured for fp16-GEMM+fp32-epilogue).
   A real kernel project, not a microbench.
2. **Route/budget tuning (exact; no exactness argument needed).** §1B/§1D findings: the `tiled()` 128 MB
   budget heuristic costs 13.8× at (rows=8, nb=130K) and 2.4–2.9× at (rows=78, nb≥16K) vs the bit-identical
   row-whole variant. Small, bounded, but requires identity checks (D3-class route rules) + its own plan/go
   before any production change.

## 5. Artifacts

- This doc; plan `docs/dsv41-indexer-abscost-plan-2026-10-12.md`; profile `docs/dsv41-indexer-profile-2026-10-12.md`.
- Probe: `scratch/idxprof4/` (REPORT.md, raw/{loop,numerics,tile,tile_noclear,smoke}.json, idle.log, logs/);
  harness `bench/score_precision_probe.py` on the mlx-lm branch `perf/indexer-profile`.
- Phase 1: `scratch/idxprof1/`, `scratch/idxprof2/`; 1E: `scratch/idxprof3/`; harnesses on the same branch.
- No production change of any kind across the workstream; boots used: **0**.
