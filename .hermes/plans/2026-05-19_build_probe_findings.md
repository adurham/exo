# Build Probe Findings — Where Verify's 46ms GPU Time Goes

**Date:** 2026-05-19 ~14:00 CDT
**Plan:** `.hermes/plans/2026-05-19_structural_to_35tps.md` (refined)
**Probes:** `MLX_BUILD_PROBE=1 MLX_BUILD_PROBE_LOG_EVERY=20`

## TL;DR

DSv4 decode @ TOPK=512 FENCE=43 has **50.18 ms per forward**. The FFN
section (MoE expert dispatch + all_sum + fence eval) is **92.7% of
that = 46.51 ms**. Attention is 5.9% (2.94 ms). Other sections
combined are <1%.

**Verify cycle bottleneck = the GPU compute for the MoE.** Not jaccl,
not Python overhead, not attention.

## Decomposition (steady-state decode, 100K context, c=1, gamma=2)

```
PER FORWARD (~50ms):
  attn_pre   (compiled: hc_attn + attn_norm):     0.17 ms (0.3%)
  attn       (LocalAttention/Sparse + own all_sum): 2.94 ms (5.9%)
  post_attn  (compiled: hc_expand):                 0.12 ms (0.2%)
  ffn_pre    (compiled: hc_ffn + ffn_norm):         0.18 ms (0.4%)
  ffn        (MoE: gate + switch_mlp + post_combine 46.51 ms (92.7%)
              + all_sum + mx.eval fence)
  post_ffn   (compiled: hc_expand):                 0.12 ms (0.2%)

PER LAYER (43 layers per forward):
  ffn:       1.082 ms per layer
  attn:      0.068 ms per layer
  others:    <0.005 ms per layer each
```

Note: `ffn` at the fence-layer (only layer 42 at FENCE=43) blocks for
all 43 layers of accumulated GPU work. The 1.082 ms per-layer AVERAGE
hides this: layers 0-41 take ~50us of lazy graph build, layer 42 takes
~46ms of GPU completion wait.

## Cross-reference with other probes

| Probe | Reading | What it measures |
|---|---|---|
| BUILD_PROBE ffn | 46.51 ms / forward | CPU wall waiting for FFN section |
| ALLSUM_PROBE layer 42 | 37 ms p50 | CPU wall around the mx.eval(y) fence |
| JACCL_POLL_INSTRUMENT | 8 us / call avg | RDMA poll-loop wall |
| MTP-PROF verify | 57 ms | Total verify cycle wall |

Reconciliation:
- 46.51 ms (BUILD_PROBE ffn) ~= 37 ms (ALLSUM eval) + ~9ms Python overhead
- 700 us total RDMA (43 layers × 8us × 2 subcalls) = 1.5% of forward
- MTP-PROF verify (57 ms) > BUILD_PROBE total forward (50 ms) by ~7 ms
  which is the Python prep work in `_speculative_next` (draft_concat,
  verify_input concat, etc.)

## Where the 46ms of GPU compute actually goes

Per layer (~1.07ms GPU work per layer × 43):

The MoE forward in `DeepseekV4MoE._compiled_forward` does:
1. `self.gate(x, input_ids)` — softmax over 256 experts, select top-6
2. `self.switch_mlp(x, indices)` = BatchedSwitchGLU(SwitchGLU):
   - Fused gate+up: 1× `mx.gather_qmm` (concatenated weights)
   - Activation (SwiGLU)
   - down_proj: 1× `mx.gather_qmm`
3. `self.shared_experts(x)` — 2× `mx.quantized_matmul` (fused gate+up
   then down)
4. `_moe_post_combine(y, scores, shared_out)` — weighted_reduce +
   shared+y add (compiled)
5. `mx.distributed.all_sum(y, group=...)` — ~8us
6. `mx.eval(y)` at layer 42 only — drains entire chain

The 1.07 ms/layer of GPU work is mostly the gather_qmm dispatches.
At Mac Studio M4 Max ~400GB/s memory bandwidth (shared across 2
ranks), each layer's expert weights pull:
- 6 routed experts × (4096 hidden × 4096 inter × 8bit ≈ 16 MB)
  but at TP=2 each rank holds half the experts = ~8 MB/layer/rank
- Plus shared_experts (1 always-active small MLP)
- Total ~10 MB/layer/rank

At 400 GB/s that's ~25us per layer of bandwidth-bound work. We measure
~1000us per layer (~40× above bandwidth floor). So **kernel launch
overhead + indexer/sparse-attn + post-combine + Python glue** are
eating the rest.

## Real optimization targets (refined)

### A. Indexer + sparse pooled attention (in `attn` section, 68us/layer)

The 68us/layer for `attn` covers Indexer (`_indexer_score` + topk) +
SparseCompressedAttention's `_sparse_pooled_attention`. The May-18
attempts to optimize this (Lever 1/2) both failed. But this section
is only 6% of the cycle — not a big lever.

### B. MoE kernel launch reduction (in `ffn` section, 46ms/forward)

This is THE big opportunity. ~43 layers × 4 dispatches/layer (gate, 
switch_mlp gather_qmm fused, switch_mlp down, shared_experts) = 172
Metal kernel dispatches. Each ~50-100us launch overhead = 8-17ms of
pure dispatch cost.

**To reduce: write a single fused-MoE Metal kernel** that does:
- gate (sigmoid + topk) → expert routing
- gather + matmul gate+up (already fused) → SwiGLU
- gather + matmul down
- weighted_combine + shared_experts_add
All in ONE Metal dispatch per layer. Could save ~50% of the 46ms
= **+2-3 t/s lift**.

Reference work: `src/exo/worker/engines/mlx/patches/qwen3_5_moe/`
already has batched_moe.py + custom kernels. DSv4 doesn't currently
use that pattern.

### C. Reduce per-layer mx.compile boundary overhead (in `attn_pre +
post_attn + ffn_pre + post_ffn`, ~0.59 ms/forward)

Total per-forward overhead for these 4 small compiled chunks: 0.59ms.
Not a big lever (1.2% of forward).

### D. Concurrency (c>1) — doesn't reduce per-stream but aggregates

At c=2, MTP claims 2.7× throughput per the docstring. At 30 t/s per
stream × 2 streams × 2.7× scaling factor ~= aggregate >>100 t/s.
If user accepts aggregate metric, this is the easy win.

## Recommended next experiment

**Write a fused-MoE kernel.** This is real mlx kernel work:
1. Start from the existing `BatchedSwitchGLU` (already fuses gate+up).
2. Look at qwen3_5_moe patches for kernel-fusion reference.
3. Write a Metal kernel that does gate softmax + topk + gather_qmm
   chain + post_combine in one launch.
4. Validate quality (needle probe MUST pass).
5. Bench against current production baseline.

Estimated effort: 1-2 days of focused mlx kernel work.

Alternative (easier, may not get all the way to 35):
- Profile EACH of the 4 dispatches separately to find the dominant
  one. If gate softmax or down_proj is disproportionate, target that
  first.

## Current cluster state

- Production baseline restored (TOPK=512, FENCE=43, GAMMA=2, MTP=1, no probes)
- mlx-lm: 6dcdd40a (mc_ping removed, allsum probe code present but inactive)
- mlx: facbed9a (mlx@main, NO ack_sync_pre/QP fix)
- exo: c7032932 (post probe + mc_ping removal)
- Inference probe passes
- 30.1 t/s scored on a 3-iter quick bench post-mc_ping
