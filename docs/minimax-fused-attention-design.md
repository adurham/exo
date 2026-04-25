# MiniMax fused attention — design

Architecture decision record for the C++ MLX kernel project that lands
Option D from `docs/minimax-decode-optimization.md`. Captures the
constraints that force a two-primitive split (vs the original
single-mega-kernel framing) so future sessions don't re-derive them.

Status: design committed 2026-04-24. Implementation pending.

## Goal

Cut MiniMax-M2.7-5bit decode dispatches from ~21/layer to ~5/layer to
clear the M4 Ultra discrete-boundary threshold (see
`memory/cluster_dispatch_savings_curve.md`). Predicted cluster decode
gain: +3–5 % above the current 24.26 tok/s baseline at 50K context.

## Why two primitives, not one

The cluster runs MiniMax with `use_qk_norm=True`. The current path
(`auto_parallel.py:1059–1162`) pipes Q/K through `_fused_sharded_qk_norm`
between the QKV projection and RoPE. That helper fires a cross-rank
`mx.distributed.all_sum` on the partial sum-of-squares. MLX has no
GPU-resident `all_sum`, so the collective MUST happen between two
separately-dispatched compute steps — a single Metal kernel cannot
straddle it.

So the fused work splits:

```
   x (post pre-RMSNorm)
       │
       ▼
  ┌──────────────────────────┐
  │ Pre  primitive           │  1 dispatch
  │   QKV projection         │
  │   partial Q/K SS         │
  └──────────────────────────┘
       │   ┌───────────────┐
       ├──▶│ all_sum (CPU) │  1 collective (unchanged)
       │   └───────┬───────┘
       ▼           │
  ┌────────────────▼─────────┐
  │ Post primitive           │  3 dispatches
  │   q/k norm rsqrt+scale   │
  │   RoPE on Q,K            │
  │   K/V quantize + scatter │
  │   2-pass SDPA            │
  │   o_proj GEMV            │
  └──────────────────────────┘
       │
       ▼  residual output
```

Per-layer dispatch budget: **5 (1 pre + 1 all_sum + 3 post) vs 21**
in the current `WrappedMiniMaxAttention` path. Savings ~16/layer.

## Internal Post-primitive dispatches

3 Metal kernel dispatches inside the Post primitive:

1. `minimax_attn_post_norm_rope_kvquant_<dtype>_<bits>_<group>_<dim>`
   — norm rsqrt + scale + RoPE on Q,K + 8-bit quantize K,V into the
   cache scatter-write tile.
2. `sdpa_vector_2pass_1_quant_<...>` — reuse existing kernel
   unchanged.
3. `sdpa_vector_2pass_2_oproj_<dtype>_<v_dim>_<hidden>` — extends the
   existing `sdpa_vector_2pass_2` reduction kernel to fold in the o_proj
   GEMV (multiply by `W_o` and write residual into output buffer).

The 2-pass SDPA reduction is preserved as-is (no `blocks=1` regression
on `MLX_SDPA_BLOCKS=88`). The fused o_proj epilogue rides on top of
pass-2, which already has the per-output-row reduce in place.

## Writable-cache convention

The Post primitive needs to mutate the `QuantizedKVCache`'s underlying
packed/scales/biases buffers in place at the cache offset. MLX
primitives don't mutate inputs by default; we use the `Custom`
primitive's input-donation mechanism so the same `array` slot serves
as both input and output, mirroring how `slice_update` lowers.

Required exo-side change (small): expose
`QuantizedKVCache.{keys,values}_packed`/`scales`/`biases` as writable
arrays and the `offset` as an int input to the primitive. The primitive
gets passed the *full* preallocated cache buffer (not a slice), plus the
write offset, and writes a single 1-token row in the kernel.

This is the only invasive part. Without it, "cache scatter-write inline"
is a lie — we'd be re-issuing the 6 cache-write dispatches the project
exists to eliminate.

## Public API (MLX side)

```cpp
namespace mlx::core::fast {

// Pre kernel: x → (Q, K, V, partial_qk_ss).
std::vector<array> minimax_attn_pre_norm(
    const array& x,
    const array& w_qkv_packed,    // [out_q + 2*out_kv, in_packed]
    const array& w_qkv_scales,    // [out_q + 2*out_kv, in / group_size]
    const std::optional<array>& w_qkv_biases,
    int n_q_heads,
    int n_kv_heads,
    int head_dim,
    int group_size,
    int bits,
    StreamOrDevice s = {});

// Post kernel: (Q, K, V, qk_ss_global, ...) → residual.
array minimax_attn_post_norm(
    const array& q,                        // [B, n_q,  1, D]
    const array& k_new,                    // [B, n_kv, 1, D]
    const array& v_new,                    // [B, n_kv, 1, D]
    const array& qk_ss_global,             // [B, 1, 2]
    const array& q_norm_weight,            // [D] (per-rank, joined-heads scalar)
    const array& k_norm_weight,
    float q_norm_eps,
    float k_norm_eps,
    int joined_q_dim,                      // D * group.size() — for rsqrt scale
    int joined_k_dim,
    const array& rope_cos,                 // [max_offset+1, D/2]
    const array& rope_sin,
    int rope_offset,
    array& k_cache_packed,                 // mutated (donated input)
    array& k_cache_scales,                 // mutated
    array& k_cache_biases,                 // mutated
    array& v_cache_packed,                 // mutated
    array& v_cache_scales,                 // mutated
    array& v_cache_biases,                 // mutated
    int cache_offset,
    const array& w_o,                      // o_proj weight, packed/quant
    const array& w_o_scales,
    const std::optional<array>& w_o_biases,
    float sdpa_scale,
    int group_size,
    int bits,
    const std::optional<array>& sinks,
    StreamOrDevice s = {});

}
```

## use_fallback gates

v1 supports only the live cluster config:

| param | v1 |
| ----- | -- |
| q_seq_len | 1 (decode) |
| head_dim | 128 |
| bits | 8 |
| group_size | 64 |
| KV layout | `QuantizedKVCache` writable |
| n_q_heads / n_kv_heads | any GQA factor (24 / 4 on the cluster) |

Anything else falls back to the existing path
(WrappedMiniMaxAttention's current 21-dispatch flow).

## Exo integration

New env gate `EXO_MINIMAX_FUSED_KERNEL=1`, parallel to
`EXO_MINIMAX_FUSED_ATTN=1`. The two are mutually exclusive — fused
kernel mode supersedes the python-level fused-QKV path.

`WrappedMiniMaxAttention.__call__` gains a third branch:

```python
if _MINIMAX_FUSED_KERNEL and minimax_fused_kernel_is_installed(layer):
    q, k, v, qk_ss = mx.fast.minimax_attn_pre_norm(...)
    qk_ss = mx.distributed.all_sum(qk_ss, group=self.group)
    output = mx.fast.minimax_attn_post_norm(q, k, v, qk_ss, ..., cache.state, ...)
    return output
elif _MINIMAX_FUSED_ATTN and fused_qkv_is_installed(layer):
    # existing python-level fused-QKV path
    ...
else:
    # vanilla path
    ...
```

`minimax_fused_kernel_is_installed` checks for the merged W_qkv (already
attached by `install_fused_qkv`) plus that o_proj is the supported type.

## Per-week deliverables

- **Week 1 — scaffold (this branch):**
  - C++ primitive class declarations in `mlx/fast_primitives.h`.
  - Public API in `mlx/fast.h` + `mlx/fast.cpp` with full input
    validation, `use_fallback`, and `eval_gpu` stubs that compose
    existing MLX ops (correct, no dispatch savings).
  - Python binding in `python/src/fast.cpp`.
  - Correctness test
    `python/tests/test_fast_minimax_fused_attn.py` exercising both
    primitives at the cluster config.
  - Exo wrapper integration behind `EXO_MINIMAX_FUSED_KERNEL=1`,
    default off, default skipped if `EXO_MINIMAX_FUSED_ATTN=1`.
- **Week 2 — Metal kernels:** drop in the three real Metal kernels
  (`pre`, `post_norm_rope_kvquant`, `pass_2_oproj`). Re-run correctness
  test. Tolerance 2e-2 abs/rel for bf16.
- **Week 3 — cluster A/B:** ASK before deploying. Bench at 50K context
  + Huihui scouts on. Need ≥+3 % vs 24.26 tok/s.

## Exit criteria

- Ship if cluster decode ≥ 25 tok/s at 50K (+3 % over current 24.26)
  AND token-level output matches reference within bf16 budget AND no
  prefill regression.
- Abort if 3 cluster deploys show <2 % cumulative gain — the remaining
  headroom is below kernel-fusion's reach on this hardware tier; reroute
  to other levers.

## Things this design rules out

- A single mega-kernel covering everything (qk-norm all_sum is in the
  middle).
- `blocks=1` SDPA (regresses the +6.5 % `MLX_SDPA_BLOCKS=88` win).
- bf16 KV cache (busts the 50K + Huihui co-residency memory budget).
- Speculative decoding for MiniMax (user-ruled, see
  `memory/feedback_no_minimax_speculation.md`).
- Single-dispatch shaving (cluster curve flat at <5 saves/layer, see
  `memory/cluster_dispatch_savings_curve.md`).
