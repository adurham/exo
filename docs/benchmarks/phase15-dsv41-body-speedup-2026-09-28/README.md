# Phase 15 -- DSv4.1 body speedup: 9.3 -> 16.3 tok/s plain decode

Date: 2026-09-28. mlx-lm `feat/dsv41-exl3` @ `14d1c74` (signed, pushed).
Production stopped cleanly for the two-node run and restored (fresh pids,
env verified, real completion OK).

## Bottom line

Plain greedy decode on both nodes went from **9.25 to 16.32 tok/s**
(107 -> 61 ms/step, p10-p90 60.2-62.1), 104.9 GB/rank, coherent text.
No speculative decoding yet.

## Method

8-layer single-node harness (`scripts/p48_bench.py`, layers 0,1,2,3,20,21,
24,25 = every layer kind), unfenced timing, teacher-forced inputs, logits
compared bit-for-bit to the reference path each change. Real costs came from
removing one component at a time, not from the fenced profile (which
inflated experts/shared-expert ~3x).

## Changes and effect (8-layer harness, ms/step)

| change | ms/step | output |
|---|---:|---|
| baseline | 26.0 | reference |
| fused hc kernels | 22.6 | cos >= 0.99983/layer, NLL 1.0028 (= ref) |
| engram: parallel preads + prefetch, no GPU sync | 18.2 -> 14.6 with async | bit-identical |
| async eval per block | (in the row above) | bit-identical |
| same-input EXL3 projections in one launch | 13.4 | bit-identical |
| compiled small-batch EXL3 linear | 13.4 | bit-identical |

Then wider TP: attention heads / wo_a groups / wo_b input, shared expert,
and vocab head split across ranks (128-block Hadamard-aligned slices;
simulated both ranks vs full: cos >= 0.9999946). End-to-end: 16.3 tok/s.

## Findings

- The real bottleneck was graph building + GPU syncs, not kernels: the
  engram lookup forced a full GPU sync every token; the reference hc path
  issued ~140 launches per sub-layer.
- Shared expert and in-situ experts were NOT pathological -- the fenced
  profile overstated them ~3x.
- Plain `mx.compile` over the reference hc_mixes CHANGES results (80% argmax
  agreement) -- rejected; the fused kernel is exact to reference math.
- Rejected: affine re-quantization of dense weights (faster, changes outputs),
  EXL3 M=1 split retuning (no gain, not bit-exact).
- JACCL all_gather failed at the head size (wc.status=1); head uses all_sum
  of zero-padded slices instead.

## Not done yet

- Compiled rms/rope/fake-quant (bit-exact, -0.7 ms on the harness): measured,
  not yet moved into the model code.
- Fused sparse attention / indexer kernels; gate fusion.
- Speculative decoding (MTP/DSpark) -- the next big multiplier.

## Artifacts

`raw/p47-r0-16tok.log`, `scripts/p48_bench.py`, `scripts/p55_tp_sim.py`,
`scripts/p51_dense.py`, `scripts/p52_dense_each.py`.
