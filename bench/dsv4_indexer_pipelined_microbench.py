"""DSv4 Indexer pipelined microbench — phase-1 spike for indexer-fused-kernel plan.

Compares the current `_indexer_score` chain (5-7 mx.* dispatches per call)
against a candidate fused metal_kernel that does
score-matmul + ReLU + weights-scale + H-reduce in a single Metal dispatch.

Argsort is left OUTSIDE the kernel (Apple's tuned implementation).

Production shape: B=1, H=64, L=1, D=128, K=160, P=25000.
21 sparse layers per token at 100K context.

Critical: runs BOTH per-call AND pipelined 21-call chain. Per the May 14
sparse_attn lesson, per-call alone can lie if MLX is already pipelining
dispatches at the chain level.
"""

from __future__ import annotations
import time
import statistics

import mlx.core as mx

# ─────────────── Production shapes ───────────────
B = 1
H = 64
L = 1
D = 128
K = 160
P_BASE = 25000   # pool size at 100K context

N_LAYERS = 21    # number of sparse_attn / indexer calls per token at 100K
N_ITERS = 300
N_WARMUP = 30
SEED = 42

DTYPE = mx.bfloat16
SCALE = D ** -0.5
N_HEADS_INV_SQRT = H ** -0.5


def make_inputs(seed: int, pool_size: int = P_BASE):
    """Build representative inputs for ONE indexer call."""
    mx.random.seed(seed)
    q = mx.random.normal((B, H, L, D), dtype=DTYPE)
    pooled = mx.random.normal((B, pool_size, D), dtype=DTYPE)
    weights = mx.random.normal((B, L, H), dtype=DTYPE) * 0.1
    return q, pooled, weights


# ─────────────── Path A: current _indexer_score (production code) ───────────────
def current_indexer(q, pooled, weights_x, scale, n_heads_inv_sqrt):
    """Verbatim copy of `_indexer_score` from deepseek_v4.py."""
    pf = pooled[:, None]
    scores = q @ pf.swapaxes(-1, -2)                          # (B, H, L, P)
    scores = mx.maximum(scores, 0)                            # ReLU
    w = (weights_x * (scale * n_heads_inv_sqrt))[..., None]   # (B, L, H, 1)
    scores_blph = scores.transpose(0, 2, 3, 1)                # (B, L, P, H)
    return (scores_blph @ w).squeeze(-1)                      # (B, L, P)


def current_topk(q, pooled, weights_x):
    scores = current_indexer(q, pooled, weights_x, SCALE, N_HEADS_INV_SQRT)
    return mx.argsort(-scores, axis=-1)[..., :K]


# ─────────────── Path B: candidate fused metal_kernel ───────────────
# One threadgroup per pool-position chunk. Each thread computes the per-head
# dot product for one (head, pool_position) pair, then we collapse across H
# with the scaled weights.
#
# Strategy: launch (P, B*L, 1) threadgroups of (H, 1, 1) threads.
# Each threadgroup p computes scores[b, :, l, p] = q[b, :, l, :] @ pooled[b, p, :]
# Then reduces across H via weights[b, l, :] * scale * head_inv_sqrt.
# Writes single scalar score[b, l, p] to global.

FUSED_KERNEL_SOURCE = r"""
// grid: (P, B*L, 1), threadgroup: (H_, 1, 1)
uint h     = thread_position_in_threadgroup.x;
uint p     = threadgroup_position_in_grid.x;
uint bl    = threadgroup_position_in_grid.y;
uint b     = bl / L_;
uint l     = bl % L_;

// q: (B, H, L, D), strides = (H*L*D, L*D, D, 1)
// pooled: (B, P, D), strides = (P*D, D, 1)
// weights: (B, L, H), strides = (L*H, H, 1)
// out: (B, L, P), strides = (L*P, P, 1)

uint q_off     = b * (H_ * L_ * D_) + h * (L_ * D_) + l * D_;
uint pool_off  = b * (P_ * D_) + p * D_;
uint w_off     = b * (L_ * H_) + l * H_;
uint out_off   = b * (L_ * P_) + l * P_;

// Dot product q[b, h, l, :] · pooled[b, p, :]
float acc = 0.0f;
for (uint d = 0; d < D_; ++d) {
    acc += float(q[q_off + d]) * float(pooled[pool_off + d]);
}

// ReLU
float relu_score = (acc > 0.0f) ? acc : 0.0f;

// Per-head contribution: relu_score * weights[b, l, h] * (scale * head_inv_sqrt)
float scaled_w = float(weights[w_off + h]) * SCALE_W_;
float contrib = relu_score * scaled_w;

// Reduce across H within threadgroup via threadgroup memory
threadgroup float h_partials[H_];
h_partials[h] = contrib;
threadgroup_barrier(mem_flags::mem_threadgroup);

// Thread 0 sums and writes
if (h == 0) {
    float total = 0.0f;
    for (uint hh = 0; hh < H_; ++hh) {
        total += h_partials[hh];
    }
    out[out_off + p] = T(total);
}
"""


def _build_fused_kernel(pool_size: int):
    return mx.fast.metal_kernel(
        name="dsv4_indexer_score_fused_v1",
        input_names=["q", "pooled", "weights"],
        output_names=["out"],
        source=FUSED_KERNEL_SOURCE,
        header=f"""
        constant uint B_ = {B};
        constant uint H_ = {H};
        constant uint L_ = {L};
        constant uint D_ = {D};
        constant uint P_ = {pool_size};
        constant float SCALE_W_ = {SCALE * N_HEADS_INV_SQRT}f;
        """,
        ensure_row_contiguous=True,
    )


def fused_indexer(kernel, q, pooled, weights, pool_size):
    """Fused score kernel; argsort still external."""
    outs = kernel(
        inputs=[q, pooled, weights],
        template=[("T", mx.bfloat16)],
        grid=(H * pool_size, B * L, 1),
        threadgroup=(H, 1, 1),
        output_shapes=[(B, L, pool_size)],
        output_dtypes=[DTYPE],
    )
    return outs[0]


def fused_topk(kernel, q, pooled, weights, pool_size):
    scores = fused_indexer(kernel, q, pooled, weights, pool_size)
    return mx.argsort(-scores, axis=-1)[..., :K]


# ─────────────── Numerical equivalence ───────────────
def check_equiv(kernel, n_seeds=5):
    print("Numerical equivalence (5 seeds):")
    overlap_counts = []
    for seed in range(n_seeds):
        q, pooled, weights = make_inputs(seed)
        s_cur = current_indexer(q, pooled, weights, SCALE, N_HEADS_INV_SQRT)
        s_fu  = fused_indexer(kernel, q, pooled, weights, P_BASE)
        mx.eval(s_cur, s_fu)
        diff = mx.abs(s_cur - s_fu)
        ref = mx.abs(s_cur) + 1e-6
        max_abs = float(mx.max(diff))
        norm_diff = float(mx.sqrt(mx.sum(diff * diff)))
        norm_ref = float(mx.sqrt(mx.sum(s_cur * s_cur)))
        rel_norm = norm_diff / max(norm_ref, 1e-9)

        # Top-K agreement
        topk_cur = current_topk(q, pooled, weights)
        topk_fu  = fused_topk(kernel, q, pooled, weights, P_BASE)
        mx.eval(topk_cur, topk_fu)
        set_cur = set(topk_cur.tolist()[0][0])
        set_fu  = set(topk_fu.tolist()[0][0])
        overlap = len(set_cur.intersection(set_fu))
        overlap_counts.append(overlap)
        print(f"  seed={seed}: max_abs={max_abs:.4e} norm_rel={rel_norm:.4e} top-{K} overlap={overlap}/{K}")
    return overlap_counts


# ─────────────── Timing harness ───────────────
def time_per_call(fn, args, n_iters=N_ITERS, n_warmup=N_WARMUP):
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


def time_pipelined(fn, args_list, n_iters=N_ITERS, n_warmup=N_WARMUP):
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


# ─────────────── Main ───────────────
def main():
    print("=" * 70)
    print(f"DSv4 Indexer pipelined microbench")
    print(f"  Shape: B={B} H={H} L={L} D={D} K={K} P={P_BASE}")
    print(f"  Layers in pipelined chain: {N_LAYERS}")
    print(f"  Iters: {N_ITERS} (warmup {N_WARMUP}), dtype={DTYPE}")
    print(f"  Device: {mx.default_device()}")
    print("=" * 70)

    print("\nBuilding fused kernel...")
    kernel = _build_fused_kernel(P_BASE)
    print("  ✓")

    overlap_counts = check_equiv(kernel)
    avg_overlap = sum(overlap_counts) / len(overlap_counts)
    print(f"  Avg top-{K} overlap: {avg_overlap:.1f}/{K}")
    if avg_overlap < K * 0.95:
        print(f"  ✗ WARNING: top-K overlap below 95% threshold")
    else:
        print(f"  ✓ Top-K overlap acceptable")

    # ─── Per-call timing ───
    print(f"\nPer-call timing (median µs over {N_ITERS} iters):")
    q, pooled, weights = make_inputs(SEED)
    mx.eval(q, pooled, weights)
    args = (q, pooled, weights)

    def cur_call():
        return current_topk(*args)

    def fused_call():
        return fused_topk(kernel, *args, P_BASE)

    t_cur = time_per_call(lambda: cur_call(), ())
    t_fu  = time_per_call(lambda: fused_call(), ())
    print(f"  current chain (score + argsort):   {t_cur:7.2f} µs")
    print(f"  fused score kernel + argsort:      {t_fu:7.2f} µs  ({t_cur/t_fu:.2f}x)")

    # ─── Pipelined 21-call timing ───
    print(f"\nPipelined {N_LAYERS}-call chain (median µs):")
    args_list = []
    for layer in range(N_LAYERS):
        a = make_inputs(SEED + layer)
        mx.eval(*a)
        args_list.append(a)

    def cur_chain():
        return [current_topk(*a) for a in args_list]
    def fused_chain():
        return [fused_topk(kernel, *a, P_BASE) for a in args_list]

    def time_chain(chain_fn):
        for _ in range(N_WARMUP):
            outs = chain_fn()
            mx.eval(*outs)
        samples = []
        for _ in range(N_ITERS):
            t0 = time.perf_counter()
            outs = chain_fn()
            mx.eval(*outs)
            samples.append(time.perf_counter() - t0)
        return statistics.median(samples) * 1e6

    p_cur = time_chain(cur_chain)
    p_fu  = time_chain(fused_chain)
    print(f"  current chain total:   {p_cur:8.1f} µs = {p_cur/N_LAYERS:5.1f} µs/call")
    print(f"  fused chain total:     {p_fu:8.1f} µs = {p_fu/N_LAYERS:5.1f} µs/call")
    print(f"  speedup:               {p_cur/p_fu:.2f}x")

    # ─── Decision gate ───
    print()
    print("=" * 70)
    speedup = p_cur / p_fu
    print(f"PIPELINED SPEEDUP: {speedup:.2f}x")
    if speedup >= 1.7:
        print(f"  ✓ ABOVE 1.7x GATE — proceed to phase 2 (integration)")
    elif speedup >= 1.3:
        print(f"  ~ MARGINAL — consider further fusion (argsort, RoPE)")
    elif speedup >= 1.0:
        print(f"  ✗ BELOW 1.3x — kernel doesn't help; chain already pipelined")
    else:
        print(f"  ✗✗ FUSED IS SLOWER — kernel design wrong, fix or abandon")

    saved_us = max(0, p_cur - p_fu)
    saved_ms_per_token = saved_us / 1000.0
    baseline_wall_ms = 28.5
    projected_tps = 1000.0 / max(0.1, baseline_wall_ms - saved_ms_per_token)
    print(f"\nCluster projection (21 calls/token, baseline 28.5 ms/token = 29.2 t/s):")
    print(f"  saved per token:  {saved_ms_per_token:+.2f} ms")
    print(f"  projected:        {projected_tps:.1f} t/s")
    print("=" * 70)


if __name__ == "__main__":
    main()
