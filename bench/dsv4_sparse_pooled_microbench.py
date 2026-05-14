"""DSv4 sparse_pooled_attention microbench - v2.

Tests THREE paths:
  (A) unfused (current cluster path):  gather + kv-concat + mask-concat + mx.fast.SDPA
  (B) gather-fused-SDPA:                metal_kernel for gather only, then mx.fast.SDPA
  (C) full-fused metal_kernel:          gather + SDPA + sinks in one kernel (simd-reduce)

Why (B): mx.fast.SDPA with sinks is hard to replicate numerically. If we just
fuse the gather (eliminating the 4 take_along_axis + concat dispatches) and
hand the result to Apple's SDPA, we keep its hyper-tuned compute path and
only fight dispatch overhead. Lower ceiling, higher floor.
"""

from __future__ import annotations

import time
import statistics

import mlx.core as mx

B = 1
H = 32
L = 1
D = 512
K = 160
SW = 128
P = 25000
SCALE = D ** -0.5
N_LAYERS = 21
N_ITERS = 300
N_WARMUP = 30
SEED = 42
DTYPE = mx.bfloat16


def make_inputs(seed: int):
    mx.random.seed(seed)
    q = mx.random.normal((B, H, L, D), dtype=DTYPE)
    local_kv = mx.random.normal((B, 1, SW, D), dtype=DTYPE)
    pooled = mx.random.normal((B, P, D), dtype=DTYPE)
    topk = mx.random.randint(0, P, (B, L, K), dtype=mx.int32)
    local_mask = mx.zeros((B, H, L, SW), dtype=DTYPE)
    sparse_mask = mx.zeros((B, H, L, K), dtype=DTYPE)
    sinks = mx.random.normal((H,), dtype=DTYPE) * 0.1
    return q, local_kv, pooled, topk, local_mask, sparse_mask, sinks


# ─────────────── Path A: unfused (current cluster code) ───────────────
def unfused_path(q, local_kv, pooled, topk, local_mask, sparse_mask, sinks):
    idx = topk[:, None, :, :, None]
    pooled_gathered = mx.take_along_axis(
        mx.broadcast_to(pooled[:, None, None], (B, 1, L, P, D)),
        mx.broadcast_to(idx, idx.shape[:-1] + (D,)),
        axis=3,
    )
    pooled_kv = pooled_gathered.squeeze(2)
    combined_kv = mx.concatenate([local_kv, pooled_kv], axis=2)
    lm = mx.broadcast_to(local_mask, (B, H, L, SW))
    pm = mx.broadcast_to(sparse_mask, (B, H, L, K))
    combined_mask = mx.concatenate([lm, pm], axis=-1)
    return mx.fast.scaled_dot_product_attention(
        q, combined_kv, combined_kv, scale=SCALE, mask=combined_mask, sinks=sinks,
    )


# ─────────────── Path B: gather-fused-only (keep SDPA) ───────────────
# Single metal_kernel that, for each (b, k, d), reads pooled[b, topk[b,0,k], d]
# and writes to gathered[b, 0, k, d]. Then concat + SDPA.
# grid: (K, D, B) threadgroups of 1×1×1 — actually let's use (K*D, B, 1) with TG=(D, 1, 1)
GATHER_KERNEL_SOURCE = r"""
uint d  = thread_position_in_threadgroup.x;
uint k  = threadgroup_position_in_grid.x;
uint b  = threadgroup_position_in_grid.y;
// topk shape (B, L=1, K)
uint p_idx = uint(topk[b * K_ + k]);
// pooled (B, P, D), gathered (B, 1, K, D)
gathered[b * K_ * D_ + k * D_ + d] = pooled[b * P_ * D_ + p_idx * D_ + d];
"""

def _build_gather_kernel():
    return mx.fast.metal_kernel(
        name="dsv4_sparse_gather_v1",
        input_names=["pooled", "topk"],
        output_names=["gathered"],
        source=GATHER_KERNEL_SOURCE,
        header=f"""
        constant uint K_ = {K};
        constant uint D_ = {D};
        constant uint P_ = {P};
        """,
        ensure_row_contiguous=True,
    )


def gather_fused_path(gather_kernel, q, local_kv, pooled, topk, local_mask, sparse_mask, sinks):
    """Fused gather + Apple SDPA. Concat still happens but only one gather dispatch."""
    outs = gather_kernel(
        inputs=[pooled, topk],
        grid=(K * D, B, 1),
        threadgroup=(D, 1, 1),
        output_shapes=[(B, 1, K, D)],
        output_dtypes=[DTYPE],
    )
    pooled_kv = outs[0]
    combined_kv = mx.concatenate([local_kv, pooled_kv], axis=2)
    lm = mx.broadcast_to(local_mask, (B, H, L, SW))
    pm = mx.broadcast_to(sparse_mask, (B, H, L, K))
    combined_mask = mx.concatenate([lm, pm], axis=-1)
    return mx.fast.scaled_dot_product_attention(
        q, combined_kv, combined_kv, scale=SCALE, mask=combined_mask, sinks=sinks,
    )


# ─────────────── Path C: full-fused (simd-reduce, fixed mem) ───────────────
# Strategy: per (b, h, l) threadgroup, use simd_sum across 32-lane subgroup for
# fast q·k dot product. D=512 → 16 simdgroups per threadgroup. Each simdgroup
# handles 32 dims, simd_sum gives the partial, atomic add to a final accumulator.
#
# But simpler: D=512, tg=512 threads, organize as 16 simdgroups of 32.
# Each thread holds q[tid]. For each score s ∈ [0, SW+K):
#   - Each thread computes prod = q[tid] * k[s][tid]
#   - simd_sum gets partial sum within each 32-lane simdgroup
#   - simdgroup 0 collects partials from all 16 simdgroups via threadgroup mem
#   - thread 0 of simdgroup 0 writes final score to scratch[s]
#
# Then softmax + sinks in thread 0. Then value pass, same pattern.

FULL_FUSED_KERNEL_SOURCE = r"""
threadgroup float scratch[SW_PLUS_K_];      // scores then weights
threadgroup float simd_partials[16];        // 16 simdgroups per TG
threadgroup float m_buf[1];
threadgroup float s_buf[1];

uint tid          = thread_position_in_threadgroup.x;
uint simd_id      = tid / 32;
uint simd_lane_id = tid % 32;
uint bl           = threadgroup_position_in_grid.x;
uint h            = threadgroup_position_in_grid.y;
uint b            = bl / L_;
uint l            = bl % L_;

uint q_off  = b * (H_ * L_ * D_) + h * (L_ * D_) + l * D_;
uint lk_off = b * (SW_ * D_);
uint po_off = b * (P_ * D_);
uint tk_off = b * (L_ * K_) + l * K_;
uint lm_off = b * (H_ * L_ * SW_) + h * (L_ * SW_) + l * SW_;
uint sm_off = b * (H_ * L_ * K_) + h * (L_ * K_) + l * K_;
uint out_off = b * (H_ * L_ * D_) + h * (L_ * D_) + l * D_;

float q_val = float(q[q_off + tid]);
float out_acc = 0.0f;

// ──────── Phase A: scores for SW local positions ────────
for (uint s = 0; s < SW_; ++s) {
    float kv_val = float(local_kv[lk_off + s * D_ + tid]);
    float prod = q_val * kv_val;
    float partial = simd_sum(prod);
    if (simd_lane_id == 0) {
        simd_partials[simd_id] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {
        float acc = (simd_lane_id < 16) ? simd_partials[simd_lane_id] : 0.0f;
        acc = simd_sum(acc);
        if (simd_lane_id == 0) {
            scratch[s] = acc * SCALE_ + float(local_mask[lm_off + s]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// ──────── Phase B: scores for K pool positions ────────
for (uint k = 0; k < K_; ++k) {
    uint p_idx = uint(topk[tk_off + k]);
    float kv_val = float(pooled[po_off + p_idx * D_ + tid]);
    float prod = q_val * kv_val;
    float partial = simd_sum(prod);
    if (simd_lane_id == 0) {
        simd_partials[simd_id] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {
        float acc = (simd_lane_id < 16) ? simd_partials[simd_lane_id] : 0.0f;
        acc = simd_sum(acc);
        if (simd_lane_id == 0) {
            scratch[SW_ + k] = acc * SCALE_ + float(sparse_mask[sm_off + k]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// ──────── Softmax with sinks (thread 0 only) ────────
if (tid == 0) {
    float m = float(sinks[h]);
    for (uint i = 0; i < SW_PLUS_K_; ++i) {
        if (scratch[i] > m) m = scratch[i];
    }
    m_buf[0] = m;
    float s_acc = exp(float(sinks[h]) - m);
    for (uint i = 0; i < SW_PLUS_K_; ++i) {
        float w = exp(scratch[i] - m);
        scratch[i] = w;
        s_acc += w;
    }
    s_buf[0] = s_acc;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

float inv_s = 1.0f / s_buf[0];

// ──────── Value pass (V == K in MLA) ────────
for (uint s = 0; s < SW_; ++s) {
    float kv_val = float(local_kv[lk_off + s * D_ + tid]);
    out_acc += scratch[s] * kv_val;
}
for (uint k = 0; k < K_; ++k) {
    uint p_idx = uint(topk[tk_off + k]);
    float kv_val = float(pooled[po_off + p_idx * D_ + tid]);
    out_acc += scratch[SW_ + k] * kv_val;
}

out[out_off + tid] = T(out_acc * inv_s);
"""


def _build_full_kernel():
    return mx.fast.metal_kernel(
        name="dsv4_sparse_full_fused_v2",
        input_names=["q", "local_kv", "pooled", "topk", "local_mask", "sparse_mask", "sinks"],
        output_names=["out"],
        source=FULL_FUSED_KERNEL_SOURCE,
        header=f"""
        constant uint B_ = {B};
        constant uint H_ = {H};
        constant uint L_ = {L};
        constant uint D_ = {D};
        constant uint K_ = {K};
        constant uint SW_ = {SW};
        constant uint P_ = {P};
        constant uint SW_PLUS_K_ = {SW + K};
        constant float SCALE_ = {SCALE}f;
        """,
        ensure_row_contiguous=True,
    )


def full_fused_path(kernel, q, local_kv, pooled, topk, local_mask, sparse_mask, sinks):
    outs = kernel(
        inputs=[q, local_kv, pooled, topk, local_mask, sparse_mask, sinks],
        template=[("T", mx.bfloat16)],
        grid=(B * L * D, H, 1),
        threadgroup=(D, 1, 1),
        output_shapes=[(B, H, L, D)],
        output_dtypes=[DTYPE],
    )
    return outs[0]


# ─────────────── Checks ───────────────
def check_equiv(label, ref_fn, cand_fn, n_seeds: int = 5):
    max_diff = 0.0
    max_rel = 0.0
    for seed in range(n_seeds):
        args = make_inputs(seed)
        out_ref = ref_fn(*args)
        out_cand = cand_fn(*args)
        mx.eval(out_ref, out_cand)
        diff = mx.abs(out_ref - out_cand)
        ref = mx.abs(out_ref) + 1e-6
        max_diff = max(max_diff, float(mx.max(diff)))
        max_rel = max(max_rel, float(mx.max(diff / ref)))
    ok = max_rel < 5e-2
    print(f"  {label}: max|d|={max_diff:.2e} max|d|/|r|={max_rel:.2e} {'✓' if ok else '✗'}")
    return ok


def time_per_call(fn, *args, n_iters: int = N_ITERS, n_warmup: int = N_WARMUP):
    for _ in range(n_warmup):
        out = fn(*args)
        mx.eval(out)
    samples = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        out = fn(*args)
        mx.eval(out)
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples) * 1e6


def time_pipelined(fn, args_list, n_iters: int = N_ITERS, n_warmup: int = N_WARMUP):
    def _chain():
        return [fn(*a) for a in args_list]
    for _ in range(n_warmup):
        outs = _chain()
        mx.eval(*outs)
    samples = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        outs = _chain()
        mx.eval(*outs)
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples) * 1e6


def main():
    print("=" * 70)
    print(f"DSv4 sparse_pooled_attention microbench v2")
    print(f"  Shape: B={B} H={H} L={L} D={D} K={K} SW={SW} P={P}")
    print(f"  Layers: {N_LAYERS}, iters: {N_ITERS}, dtype: {DTYPE}")
    print(f"  Device: {mx.default_device()}")
    print("=" * 70)
    print()

    gather_kernel = _build_gather_kernel()
    full_kernel = _build_full_kernel()

    print("Numerical equivalence vs unfused (5 seeds):")
    print(f"  Path B (gather-fused, Apple SDPA):")
    check_equiv("    B vs A", unfused_path,
                lambda *a: gather_fused_path(gather_kernel, *a))
    print(f"  Path C (full-fused metal_kernel):")
    check_equiv("    C vs A", unfused_path,
                lambda *a: full_fused_path(full_kernel, *a))

    print()
    print("Per-call timing (median µs):")

    # Pre-bake inputs once
    args = make_inputs(SEED)
    mx.eval(*args)

    t_a = time_per_call(unfused_path, *args)
    t_b = time_per_call(lambda *a: gather_fused_path(gather_kernel, *a), *args)
    t_c = time_per_call(lambda *a: full_fused_path(full_kernel, *a), *args)
    print(f"  A unfused (current path):    {t_a:7.2f} µs")
    print(f"  B gather-fused + Apple SDPA: {t_b:7.2f} µs  ({t_a/t_b:.2f}x)")
    print(f"  C full-fused metal_kernel:   {t_c:7.2f} µs  ({t_a/t_c:.2f}x)")

    print()
    print(f"Pipelined {N_LAYERS}-call chain timing (median µs):")

    args_list = []
    for layer in range(N_LAYERS):
        a = make_inputs(SEED + layer)
        mx.eval(*a)
        args_list.append(a)

    p_a = time_pipelined(unfused_path, args_list)
    p_b = time_pipelined(lambda *a: gather_fused_path(gather_kernel, *a), args_list)
    p_c = time_pipelined(lambda *a: full_fused_path(full_kernel, *a), args_list)
    print(f"  A unfused:      {p_a:8.1f} µs total = {p_a/N_LAYERS:5.1f} µs/call")
    print(f"  B gather-fused: {p_b:8.1f} µs total = {p_b/N_LAYERS:5.1f} µs/call  ({p_a/p_b:.2f}x)")
    print(f"  C full-fused:   {p_c:8.1f} µs total = {p_c/N_LAYERS:5.1f} µs/call  ({p_a/p_c:.2f}x)")

    print()
    print("=" * 70)
    print("Decision gate (need >= 1.7x pipelined speedup):")
    for label, ratio in [("B gather-fused", p_a/p_b), ("C full-fused", p_a/p_c)]:
        if ratio >= 1.7:
            verdict = "✓ ABOVE GATE - proceed"
        elif ratio >= 1.3:
            verdict = "~ MARGINAL"
        elif ratio >= 1.0:
            verdict = "✗ BELOW GATE"
        else:
            verdict = "✗✗ SLOWER"
        print(f"  {label}: {ratio:.2f}x  {verdict}")

    print()
    print("Cluster projection (21 layers/token, baseline 28.5 ms = 29.2 t/s):")
    for label, total in [("B", p_b), ("C", p_c)]:
        saved_ms = (p_a - total) / 1000.0
        new_wall = 28.5 - saved_ms
        new_tps = 1000.0 / new_wall if new_wall > 0 else 0
        print(f"  {label}: save {saved_ms:+.2f} ms → {new_tps:.1f} t/s")
    print("=" * 70)


if __name__ == "__main__":
    main()
