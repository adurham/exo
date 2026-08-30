# P03 — First per-kernel GPU capture of the decode "small-op bucket" (2026-08-30)

**Scope: REAL CAPTURE WORK on m4-1** (standalone process beside the live
production runner — the p01-proven recipe; **zero cluster relaunches**, the
live runner PID 31066 kept its 9h+ uptime throughout). Method:
`MLX_GPU_TIME=1` bracketing (median-of-3 passes, fresh graph per timed call,
`mx.eval` per call — p01 `time_stage` pattern) + `MLX_DISPATCH_COUNT=1` +
`METAL_CAPTURE_ENABLED=1` gputraces of the top contributors. Real weights for
gate/HC/norm/lm_head/markov/rope-freqs (runtime dtype/shape assertion battery
against checkpoint headers — all pass); synthetic same-dtype/shape rotation
banks (16-64 sets) for L2 defeat on gate/shared/HC (p01 precedent: values
don't affect bandwidth-bound timing). Kernel-selection env parity with the
production runner (`MLX_GEMV_BATCH_INVARIANT=1`, `MLX_STEEL_BATCH_INVARIANT=1`,
`MLX_MAX_OPS_PER_BUFFER=200`, `MLX_MAX_MB_PER_BUFFER=200`, both HC kernel
flags ON). Artifacts: `tmp/p03-20260830/` (script, results.json, stdout log,
4 clean per-op gputrace bundles).

## 1. The question

Phase (d) (`docs/p02d-roofline-occupancy-kernel-reconciliation-2026-08-29.md`)
identified the small-op bucket (moe.gate, shared_experts, post_combine, norms,
HC residuals, rope, lm_head) as **27-31% of GPU-busy decode time at an implied
16-23% of spec bandwidth (~6-10 ms/token) — never measured per-kernel**. Two
possible outcomes: irreducible per-op dispatch latency (decode tuning done)
or a concrete fusion target. This phase measured it.

## 2. Four measurement-integrity traps found and fixed (live, this session)

These are recorded because each silently produces plausible garbage:

1. **`HyperConnection` defaults to `training=True`.** First run measured
   92-95 dispatches / ~217 µs per HC call — the pure-MLX Sinkhorn loop. The
   production loader calls `model.eval()` (mlx_lm utils), which flips HC to
   the fused-kernel path: **4 dispatches / 24.7 µs at L=1 (ops path), 2
   dispatches / 60.9 µs at L=4 (fused precursor+collapse)**. Verified live
   both ways before re-running. Any standalone microbench of this model's
   modules MUST `.eval()` them first.
2. **`* 0` constant-folding DCE.** My first sampler unit used
   `logsumexp(lp) * 0 + lp.max()` to force dependency — MLX's compiler folded
   the ENTIRE lm_head graph away (measured 1 dispatch / 2.1 µs, "497,890
   GB/s"). Fixed to the production sampler form
   (`logprobs = lp - logsumexp; argmax`) → 5 dispatches / 2135 µs. Never use
   arithmetic-that-cancels to force evaluation in MLX.
3. **The 70 µs "shared recheck" was a harness bug**, not model variance: the
   recheck lambda rebuilt fresh random input tensors inside the timed call
   (22.3 dispatches vs 4.0). The three clean shared measurements (34.3 /
   34.1 / 35.1 µs cold-rotated) agree within 3% and stand.
4. **gputrace filename collision**: `start_capture` fails (not overwrites)
   when the bundle exists; second-run skeleton captures failed silently
   until renamed. Per-op evidence re-captured with fresh names.

## 3. L2-defeat sanity (consult-mandated before trusting any GB/s)

2-set vs 64-set gate rotation: **80.2 → 75.5 GB/s** (no meaningful drop —
gate's 2 MB weight + tiny outputs don't benefit from L2 anyway; its cost is
latency, see §4). 2-set vs 16-set shared rotation: **438.4 → 381.8 GB/s**
(the 13 MB weight set is partially L2-resident at 2 sets — confirms rotation
was necessary and 16 sets (210 MB) is past the effective cache boundary).
All isolated-op GB/s below use the rotated (DRAM-real) numbers.

## 4. Per-kernel table — SPEC-OFF (L=1), the Phase (d) comparability regime

GPU-busy µs/call, median of 3 passes; GB/s vs modeled bytes; both ceilings
shown per campaign convention. "disp" = Metal kernel dispatches/call.

| op (calls/token) | µs/call | disp | bytes | GB/s | %spec | %real(424) | binding mode |
|---|---|---|---|---|---|---|---|
| lm_head (1) | 2094.4 | 1 | 1059 MB | 505.8 | **92.6%** | 119.3% | bandwidth — near-optimal |
| shared_experts (43) | 34.3 | 4 | 13.0 MB | 378.9 | 69.4% | 89.4% | bandwidth — good |
| moe.gate routed (40) | 27.8 | 7 | 2.1 MB | 75.9 | 13.9% | 17.9% | **latency** (7 dispatches) |
| moe.gate hash (3) | 16.7 | 6 | 2.1 MB | 126.3 | 23.1% | 29.8% | latency |
| hc_collapse ops-path (86) | 24.7 | 4 | 1.7 MB | 78.0 | 14.3% | 18.4% | **latency** (fn 1.5 MB + Sinkhorn-20 barriers) |
| hc_expand fused (86) | 4.4 | 1 | 0.2 MB | 16.6 | 3.0% | 3.9% | latency floor |
| post_combine (43) | 7.5 | 3 | 0.06 MB | 8.8 | 1.6% | 2.1% | latency floor |
| rmsnorm (87) | 3.5 | 1 | 0.01 MB | 4.7 | 0.9% | 1.1% | latency floor |
| rope q (86) | 2.8 | 1 | 0.13 MB | 46.3 | 8.5% | 10.9% | latency floor |
| rope kv (43) | 2.6 | 1 | 2 KB | — | — | — | latency floor |
| rope indexer-q (20) | 2.8 | 1 | 33 KB | 11.8 | 2.2% | 2.8% | latency floor |
| hc_head (1) | 28.6 | 10 | 0.4 MB | 19.2 | 3.5% | 4.6% | latency (10 dispatches) |
| lm_head+logsumexp+argmax sampler unit | 2135.6 | 5 | 1059 MB | 496.8 | 91.0% | 117.2% | bandwidth |

**Spec-off bucket total (sum of parts): 8.34 ms/token** — dead center of
Phase (d)'s implied 6-10 ms band. Share: lm_head-family 25.9%, hc_collapse
25.4%, shared 17.7%, gate 13.9%, rope 4.9%, hc_expand 4.6%, combine 3.8%,
norms 3.6%.

**Chained cross-check** (the fusion-verdict instrument, consult-mandated):
a skeleton chaining the REAL per-layer bucket sequence (43 layers ×
hc→norm→[attn excluded]→expand→hc→norm→gate→[switch excluded]→shared→combine
→expand, + tail) under ONE `mx.eval` measures **7.76 ms GPU-busy** vs 8.34 ms
sum-of-isolated → **ratio 0.93**. Sum-of-parts is valid (slight overlap
benefit in-chain, opposite direction from p01's 1.24 chained/serial ratio).
The skeleton issues **1314 dispatches/token**; its wall-vs-gpu gap (11.0 −
7.76 = 3.2 ms) implies **~2.4 µs per dispatch** — the honest per-dispatch
cost that production partially hides via async-eval + 200-op command buffers.

## 5. Two byte-model corrections to Phase (d)

1. **lm_head is UNQUANTIZED BF16: 1.059 GB/rank** (checkpoint `head.weight`
   BF16 129280×4096; `make_quantization_config` has no lm_head category;
   runner-log repr shows plain `Linear`). Phase (d) carried ~0.55 GB (±0.28
   sensitivity) — **the real tensor is 1.9x the central estimate**, and it is
   REPLICATED (not sharded). This was caught by the runtime assertion
   battery, exactly as the consult predicted ("this is what catches Phase (d)
   assumption errors").
2. **HC `fn` matrices were never in any byte model**: 1.57 MB fp32 × 86
   calls/token ≈ 135 MB/token of reads (16% of the whole B_true attention
   path, for comparison). They are inside the measured hc_collapse numbers.

## 6. Spec-ON (production verbon3) re-basing

Production cycle = DSpark draft (3 stages @ L=3, per-row shared, markov ×3)
+ batched verify forward @ L=4 (γ=3, `EXO_DSV4_VERIFY_BATCH=1`, fused HC at
L%4==0) + lm_head ×2 + samplers. Measured per cycle (GPU-busy):

| component | µs/cycle | % of cycle |
|---|---|---|
| verify hc_collapse fused ×86 @61µs | 5240 | 30.4% |
| verify shared per-row ×43 groups @82µs | 3525 | 20.5% |
| verify tail (hc_head+norm+lm_head L4+sampler) | 2282 | 13.3% |
| draft tail (hc_head+norm+lm_head L3) | 2098 | 12.2% |
| verify gate ×43 @30µs | 1300 | 7.6% |
| draft 3-stage bodies (hc/norms/gate/shared/combine/expand/rope) | 656 | 3.8% |
| verify rope | 632 | 3.7% |
| draft markov ×3 @154µs | 461 | 2.7% |
| verify expand/post_combine/norms | 1028 | 6.0% |
| **total** | **17.2 ms/cycle** | |

**Per token (÷3.2 committed tokens/cycle): 5.38 ms/token** — the re-based
number Phase (d) asked for. The spec-ON bucket is LOWER per token than
spec-off (8.34) because verify-batch amortization is real: fused HC L=4
(60.9 µs) beats 4× L=1 ops-path (98.7 µs) by 38%; batched shared L=4 (44.4
µs) beats the per-row group (82.0 µs) by 45%. Skeleton cross-checks agree
(verify_L4_perrow 12.94 ms vs sum-of-parts 14.0 ms; batched-shared skeleton
11.73 ms — the 1.2 ms/cycle shared-batching delta is directly visible).

## 7. Verdict: NOT irreducible — three concrete targets, one blocked

The implied "16-23% of spec" was an averaging artifact: **no op runs at
16-23%**. The bucket is a bimodal blend — near-optimal bandwidth ops
(lm_head 92.6% spec, shared 69-90%, markov 80.7%) and genuinely
latency-floor ops (norms/rope/expand/combine, 1-8 µs each at 1-3
dispatches, no headroom). The headroom is in specific, nameable places:

1. **lm_head + markov_w2 are BF16; quantize them (largest lever).** The
   lm_head family (verify L=4 + draft L=3 + samplers + markov ×3) is
   ~4.6 ms/cycle ≈ 1.44 ms/token ≈ 27% of the spec-ON bucket, running at
   92-94% of spec — i.e. byte-limited, not fixable by fusion, but the BYTES
   are 2x what they need to be. An mxfp8 lm_head (the same packing the
   attention path already uses, and the same `mxfp8_qmv_fast` kernel family
   measured at ~380-440 GB/s in this very capture for shared/markov) would
   cut it to ~2.3 ms/cycle (−0.7 ms/token). Needs the needle quality gate;
   logits-side 8-bit is generally near-lossless but must be verified.
2. **HC Sinkhorn: 20 iterations of 4×4 matrix normalization per call, ~61 µs
   at L=4, 30.4% of the spec-ON cycle.** The fused collapse kernel's time is
   dominated by 20 sequential row/col-normalization passes with threadgroup
   barriers (tiny 1024-thread grid — pure latency). Hyper-Connections'
   own design says Sinkhorn converges fast; truncating to ~4-5 iterations
   (numerics-gated) projects the family from 5.24 → ~1.5-2 ms/cycle. This
   is the single biggest GPU-busy reduction available in the bucket.
3. **shared_experts verify batching: −1.2 ms/cycle measured directly**
   (44.4 µs batched L=4 vs 82.0 µs per-row group, confirmed in the skeleton
   A/B) — **but BLOCKED**: per-row shared (`EXO_DSV4_MOE_PARTS_ROWSEQ=shared`)
   is the 2026-08-04 numerics-divergence fix (batched shared was the
   isolated divergence source; 0.023% vs 0% floor). Revisit only with the
   offline `EXO_DSV4_MOE_ISOLATION_DUMP` bisect harness, not by flipping the
   flag.
4. **Genuinely irreducible remainder**: norms, rope, hc_expand, combine
   (~1.9 ms/token spec-off combined) sit at the 1-3-dispatch latency floor
   with µs-scale kernels; gate (7 dispatches, 27.8 µs) has maybe 10-15 µs
   of fusible dispatch overhead (custom score+topk kernel) — real but
   third-order. Decode tuning on these is done.

## 8. Regimes tested & why

Both, per task instruction: spec-off L=1 (Phase (d) comparability — its
anchors/census/occupancy are spec-off era) AND spec-ON production shapes
(verify L=4 batched + draft L=3 + markov, `ROWSEQ=shared` per-row pattern
reproduced, both HC kernel paths measured at their production L's). Gamma
varies 1-3 with confidence-τ in production; full γ=3 measured (the
upper-cost case). Below-ctx-8192 verify runs per-row L=1 ×4 instead of
batched — not separately measured (production steady state at real context
is the batched path).

## 9. Cluster state (unchanged by design)

**Zero relaunches used** (cap was 2). The capture ran as a standalone
process on m4-1 beside the live runner; `ps eww` before AND after confirms
both runners kept all 5 verbon3 production flags with ZERO capture-related
env vars, uptime continuous ~9h20m through the work. Real generation
smoke-test passed post-capture: coherent two-part answer ("Paris … Louvre")
with real `finish_reason: length`, 80 completion tokens, on
`deepseek-ai/DeepSeek-V4-Flash-0731`.

## 10. Limitations

- Isolated per-op GPU times measured with per-call `mx.eval` (p01 pattern);
  the 0.93 skeleton ratio shows in-chain overlap slightly REDUCES real cost
  vs sum-of-parts — numbers are conservative by ~7%.
- Single node (m4-1), like p01; rank-1 identical silicon.
- Byte models are documented approximations (weight + activation r/w);
  GB/s for sub-10-µs ops carry ±µs timing quantization — their verdict
  (latency floor) does not depend on the byte model.
- Skeleton omits attention + switch_mlp bodies (characterized in p01 /
  worker C census) — it is the bucket-only chain, as designed.
- wall-µs in isolated benches includes ~190 µs harness sync floor; use the
  skeleton wall numbers for dispatch accounting (§4).
- Production `hc_sinkhorn_iters=20` comes from the checkpoint config; the
  convergence-truncation projection (target #2) is arithmetic, not yet
  numerics-validated.

## 11. Artifacts

- `tmp/p03-20260830/p03_smallop_capture.py` — main capture script
- `tmp/p03-20260830/preflight.py` — API/weights preflight
- `tmp/p03-20260830/recapture_evidence.py` — clean per-op gputrace capture
- `tmp/p03-20260830/results.json` — all measurements (machine-readable)
- `tmp/p03-20260830/capture_stdout.log` — full run log
- `tmp/p03-20260830/captures/v2_{hc_L1_ops,hc_L4_fused,gate_L1,lmhead_L4}.gputrace`
  — 4 clean per-op evidence bundles (1.1 GB each, eval-mode HC)