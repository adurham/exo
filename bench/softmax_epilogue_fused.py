"""Route 1: Fuse the softmax overhead chain into one Metal kernel.

The overhead chain in _sparse_pooled_attention_inner (deepseek_v4.py:1292-1313):
  1. local_scores = _apply_score_mask(local_scores, local_mask)     [where]
  2. normalizer = logsumexp(local_scores, -1, keepdims)             [reduce]
  3. pooled_scores = _apply_score_mask(pooled_scores, pooled_mask)  [where]
  4. normalizer = logaddexp(normalizer, logsumexp(pooled_scores))   [reduce+add]
  5. normalizer = logaddexp(normalizer, sinks)                      [add]
  6. local_weights = exp(local_scores - normalizer)                 [elementwise]
  7. pooled_weights = exp(pooled_scores - normalizer)               [elementwise]

This is 5-6 separate kernel launches with intermediate materialization (normalizer
tensor, masked scores, etc). A single fused kernel reads local_scores + pooled_scores
+ masks + sinks, computes the normalizer online, and writes local_weights + pooled_weights.

No simdgroup_matrix needed — pure elementwise + reduction. No indexed loads.
Dodges the D=512 tiling challenge entirely.

Production shape per tile:
  local_scores:  (B, H, L, sw)  = (1, 64, 128, 128) bf16
  pooled_scores: (B, H, L, k)   = (1, 64, 128, 512) bf16
  local_mask:    (B, 1, L, sw)  bool (broadcast over H)
  pooled_mask:   (B, 1, L, k)   bool (broadcast over H)
  sinks:         (H,)           bf16
  local_weights:  (B, H, L, sw) bf16
  pooled_weights: (B, H, L, k)  bf16

Kernel: per (b, h, l) threadgroup, compute normalizer over [local | pooled | sink],
then write exp(score - normalizer) for each element.
"""
from __future__ import annotations

import statistics
import time

import mlx.core as mx

B, H, L, D, SW, K = 1, 64, 128, 512, 128, 512
N_ITERS, N_WARMUP = 50, 10
DTYPE = mx.bfloat16

# Kernel: one threadgroup per (b, h, l). Each threadgroup processes
# N = sw + k = 640 score elements, computes the logsumexp normalizer
# (with sink), then writes exp(score - normalizer) for each.
# Use thread_position_in_grid.x (MLX quirk: thread_position_in_threadgroup.x always 0).
# grid = (B * H * L * BD, 1, 1) where BD=32 threads per simdgroup.
# Each simdgroup handles one (b, h, l). Each thread handles N/BD = 20 elements.
KERNEL_SOURCE = r"""
uint gid      = thread_position_in_grid.x;
uint simd_lid = gid % 32;            // 0..31 lane within simdgroup
uint triple   = gid / 32;            // which (b, h, l) pair
uint l        = triple % L_;
uint bh       = triple / L_;
uint h        = bh % H_;
uint b        = bh / H_;

constexpr int BD = 32;
constexpr int N = SW_ + K_;
constexpr int elems_per_thread = (N + BD - 1) / BD;  // ceil(640/32) = 20

// Phase 1: load scores, apply mask, compute local max
float scores_r[elems_per_thread];
float local_max = -3.3895313892515355e+38f;  // bf16 min

// Load local scores (first SW elements)
for (int i = 0; i < elems_per_thread; i++) {
    uint idx = simd_lid * elems_per_thread + i;
    float s = -3.3895313892515355e+38f;
    if (idx < N) {
        if (idx < SW_) {
            // local_scores[b, h, l, idx]
            s = float(local_scores[((b * H_ + h) * L_ + l) * SW_ + idx]);
            // local_mask[b, 0, l, idx] — broadcast over H
            if (!lmask[((b * L_ + l) * SW_ + idx)]) {
                s = -3.3895313892515355e+38f;
            }
        } else {
            uint k_idx = idx - SW_;
            // pooled_scores[b, h, l, k_idx]
            s = float(pooled_scores[((b * H_ + h) * L_ + k_idx)]);
            // pooled_mask[b, 0, l, k_idx]
            if (!pmask[((b * L_ + l) * K_ + k_idx)]) {
                s = -3.3895313892515355e+38f;
            }
        }
    }
    scores_r[i] = s;
    if (s > local_max) local_max = s;
}

// Phase 2: global max via simd_max
float global_max = simd_max(local_max);
// Include sink in the max
float sink_val = float(sinks[h]);
global_max = fmax(global_max, sink_val);

// Phase 3: compute exp sum (online softmax normalizer)
float exp_sum = 0.0f;
float sink_exp = metal::exp(sink_val - global_max);
exp_sum += sink_exp;
for (int i = 0; i < elems_per_thread; i++) {
    scores_r[i] = metal::exp(scores_r[i] - global_max);
    exp_sum += scores_r[i];
}
float total_exp = simd_sum(exp_sum);
float inv_total = 1.0f / total_exp;

// Phase 4: write weights
for (int i = 0; i < elems_per_thread; i++) {
    uint idx = simd_lid * elems_per_thread + i;
    if (idx >= N) continue;
    float w = scores_r[i] * inv_total;
    if (idx < SW_) {
        local_weights[((b * H_ + h) * L_ + l) * SW_ + idx] = T(w);
    } else {
        uint k_idx = idx - SW_;
        pooled_weights[((b * H_ + h) * L_ + k_idx)] = T(w);
    }
}
"""


def build_kernel():
    header = f"""
constant uint B_ = {B};
constant uint H_ = {H};
constant uint L_ = {L};
constant uint D_ = {D};
constant uint SW_ = {SW};
constant uint K_ = {K};
"""
    return mx.fast.metal_kernel(
        name="softmax_fused_epilogue",
        input_names=["local_scores", "pooled_scores", "lmask", "pmask", "sinks"],
        output_names=["local_weights", "pooled_weights"],
        source=KERNEL_SOURCE,
        header=header,
        ensure_row_contiguous=True,
    )


def make_inputs():
    mx.random.seed(42)
    local_scores = mx.random.normal((B, H, L, SW), dtype=DTYPE)
    pooled_scores = mx.random.normal((B, H, L, K), dtype=DTYPE)
    lmask = mx.ones((B, 1, L, SW), dtype=mx.bool_)
    pmask = mx.ones((B, 1, L, K), dtype=mx.bool_)
    sinks = mx.random.normal((H,), dtype=DTYPE) * 0.1
    return local_scores, pooled_scores, lmask, pmask, sinks


def current_unfused(local_scores, pooled_scores, lmask, pmask, sinks):
    """The current overhead chain (5-6 launches with intermediates)."""
    # Mask
    local_scores = mx.where(lmask, local_scores, mx.finfo(DTYPE).min)
    pooled_scores = mx.where(pmask, pooled_scores, mx.finfo(DTYPE).min)
    # Normalizer
    normalizer = mx.logsumexp(local_scores, -1, keepdims=True)
    normalizer = mx.logaddexp(normalizer, mx.logsumexp(pooled_scores, -1, keepdims=True))
    sinks_exp = sinks[None, :, None, None]
    normalizer = mx.logaddexp(normalizer, sinks_exp)
    # Weights
    local_weights = mx.exp(local_scores - normalizer)
    pooled_weights = mx.exp(pooled_scores - normalizer)
    return local_weights, pooled_weights


def fused_kernel(kernel, local_scores, pooled_scores, lmask, pmask, sinks):
    """Single Metal kernel."""
    # Broadcast masks to (B, L, SW) and (B, L, K) for the kernel
    lm = mx.broadcast_to(lmask, (B, 1, L, SW)).reshape(B, L, SW)
    pm = mx.broadcast_to(pmask, (B, 1, L, K)).reshape(B, L, K)
    outs = kernel(
        inputs=[local_scores, pooled_scores, lm, pm, sinks.astype(DTYPE)],
        template=[("T", DTYPE)],
        grid=(B * H * L * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(B, H, L, SW), (B, H, L, K)],
        output_dtypes=[DTYPE, DTYPE],
    )
    return outs[0], outs[1]


def time_fn(fn, *args):
    for _ in range(N_WARMUP):
        out = fn(*args)
        if isinstance(out, tuple):
            out = out[0]
        mx.eval(out)
        mx.synchronize()
    samples = []
    for _ in range(N_ITERS):
        mx.synchronize()
        t0 = time.perf_counter()
        out = fn(*args)
        if isinstance(out, tuple):
            out = out[0]
        mx.eval(out)
        mx.synchronize()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples) * 1e6


def main():
    print("=" * 76)
    print("Route 1: Fused softmax epilogue kernel")
    print(f"  Shape: B={B} H={H} L={L} sw={SW} k={K} N={SW+K}")
    print(f"  grid=({B*H*L*32}, 1, 1), threadgroup=(32, 1, 1)")
    print("=" * 76)

    args = make_inputs()
    mx.eval(*args)
    kernel = build_kernel()

    # Numerical check
    ref_lw, ref_pw = current_unfused(*args)
    cand_lw, cand_pw = fused_kernel(kernel, *args)
    mx.eval(ref_lw, ref_pw, cand_lw, cand_pw)
    diff_l = float(mx.max(mx.abs(ref_lw - cand_lw)))
    diff_p = float(mx.max(mx.abs(ref_pw - cand_pw)))
    print(f"  Numerical: local max|d|={diff_l:.4e} pooled max|d|={diff_p:.4e}")
    print("    (bf16 noise expected: ~0.01-0.05)")

    # Timing
    t_cur = time_fn(current_unfused, *args)
    t_fused = time_fn(fused_kernel, kernel, *args)
    print(f"\n  Current unfused (5-6 launches):  {t_cur:8.0f} µs")
    print(f"  Fused epilogue kernel:           {t_fused:8.0f} µs  ({t_cur/t_fused:.2f}x)")
    print("\n  Scaled to full layer (x16 tiles):")
    print(f"    current:   {t_cur*16:.0f} µs/layer")
    print(f"    fused:     {t_fused*16:.0f} µs/layer")
    print(f"    savings:   {(t_cur - t_fused)*16:.0f} µs/layer")
    # e2e estimate: savings / (full inner per layer) × module share × ...
    full_inner = 15478  # µs/layer from microbench
    savings_pct = (t_cur - t_fused) * 16 / full_inner
    e2e = savings_pct * 0.75 * 0.225  # inner share × module share
    print(f"    e2e estimate: {e2e*100:.1f}% = 334 × {1+e2e:.3f} = {334*(1+e2e):.0f} tok/s")
    print("=" * 76)


if __name__ == "__main__":
    main()