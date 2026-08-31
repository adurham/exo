# P07c: argpartition vs argsort A/B for DSv4-Flash indexer top-k (2026-08-30)

Node: macstudio-m4-1, standalone process beside live runner (PID 59909 verified
unchanged before+after, started Sun Aug 30 19:34:04 2026).
MLX 0.32.1.dev20260822, applegpu_g16s. Script: `p07c_topk_argpartition_vs_argsort.py`,
raw: `results_topk_ab.json`.

Production expressions (deepseek_v4.py:4055/:4059), bf16, (1, L_band=1024, P), k=512,
distinct-input rotation banks (>= 164 MB working set), fresh graph + mx.eval per call,
MLX_GPU_TIME=1 GPU-bracketed, median of 5 passes.

## Results (GPU µs/call, median)

| P | ~ctx | argpartition | argsort | ratio | idx sets equal |
|---|------|-------------|---------|-------|----------------|
| 5000 | 20K | 1003.1 | 1003.0 | 1.0002 | YES |
| 12500 | 50K | 2871.6 | 2873.4 | 0.9994 | YES |
| 25000 | 100K | 6336.8 | 6337.2 | 1.0000 | YES |
| 55000 | 220K | 15396.0 | 15402.5 | 0.9996 | YES |
| 125000 | 500K | 39149.3 | 39174.1 | 0.9994 | YES |

Correctness: set-equality of selected indices AND multiset-equality of selected
scores pass at every P, on random AND forced-tie (coarse-quantized, ~100% of rows
with boundary ties) inputs. Equivalence claim holds.

L2 sanity: bank1 vs bankN identical within noise (P=55000: 15474 vs 15442
argpartition; 15408 vs 15417 argsort) — op is bandwidth-dominated; no L2-resident
artifact.

## VERDICT: NEUTRAL (dead heat) at real production P — production's default-ON argpartition is NOT a pessimization

- Ratio is 1.000 +/- 0.001 across the entire sweep, P=12500..125000. Neither path
  wins by more than ~0.1%.
- At P=55000: argpartition is 6.5 µs/call FASTER (0.04%). At P=125000:
  24.8 µs/call FASTER (0.06%). Both trivially inside run-to-run spread
  (pass spreads 0.1-3.6%); effectively zero.
- Small-P probe added post-hoc (single reused tensor, L2-resident): argpartition
  only diverges at P<=2000: P=2000 argpart 465.1 us vs argsort 438.4 us (+6.1%);
  P=1000: 211.5 vs 211.3 (equal); P<=1000 identical to noise. The historical
  "argpartition much slower at small P (295->163 tok/s at P=500)" is NOT
  reproduced on the current MLX build at L=1024 prefill shapes — at P=512 argpart
  98.4 us == argsort 98.4 us. (The old P=500 measurement was likely at different
  L or a much older MLX.) The existing EXO_DSV4_ARGPARTITION_MIN_P gate is
  harmless but unnecessary on today's build.
- Ref A/B third reference (mx.sort, no index gather): ~3% faster than both at
  large P (14915 vs 15396 at P=55000) — consistent with both paths paying an
  index side-channel cost; irrelevant since output format differs.

Physical plausibility: per-call time scales ~O(P) (1003 us @ P=5000 -> 39149 us
@ P=125000 = 39x for 25x P, superlinear tail but consistent across both paths);
16-30 GB/s effective read+write streaming on a 546 GB/s part is low but these
are sort/partition kernels (pointer-chasing, poor locality), not streams.
Identical dispatch counts per call between the two paths (7/9/11/13/15 by P) on
a build where both lower to the same radix-sort kernel family.

No crossover exists at production P: the two are the same cost. The
production default (argpartition ON) is fine as-is; the code comment
"~5% faster on Metal" for the argsort fallback is FALSE at prefill shapes on
the current build (measured equal), though it did hold at tiny P historically
under a different measurement.