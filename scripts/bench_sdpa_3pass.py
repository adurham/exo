#!/usr/bin/env python3
"""Test whether 3-pass attention (separate score/max + V accumulate) is faster
than fused online-softmax SDPA on M4 Max.

Pass 1: Q @ K^T → scores + max (still has max dependency but no V or exp)
Pass 2: softmax(scores) @ V   (no dependency — max is known)

If unfused is faster despite kernel launch and intermediate overhead,
a fused 3-pass kernel will definitely be worth writing.
"""
import time
import mlx.core as mx

NUM_Q_HEADS = 32
NUM_KV_HEADS = 2
HEAD_DIM = 128
GQA_FACTOR = NUM_Q_HEADS // NUM_KV_HEADS
SCALE = 1.0 / (HEAD_DIM ** 0.5)

CONTEXTS = [8192, 16384, 32768, 45000, 65536]
WARMUP = 5
ITERS = 30


def bench(fn, warmup=WARMUP, iters=ITERS) -> float:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        mx.synchronize()
        t0 = time.perf_counter_ns()
        fn()
        mx.synchronize()
        times.append((time.perf_counter_ns() - t0) / 1e6)
    return sum(times) / len(times)


def main() -> None:
    print(f"Config: {NUM_Q_HEADS}Q/{NUM_KV_HEADS}KV heads, dim={HEAD_DIM}, GQA={GQA_FACTOR}")
    print()
    print(f"{'Ctx':>6}  {'fused SDPA':>10}  {'unfused 3p':>10}  {'speedup':>8}  {'intermediate':>12}")
    print(f"{'':>6}  {'ms':>10}  {'ms':>10}  {'ratio':>8}  {'MB':>12}")
    print("-" * 60)

    for n_ctx in CONTEXTS:
        q = mx.random.normal((1, NUM_Q_HEADS, 1, HEAD_DIM)).astype(mx.bfloat16)
        k = mx.random.normal((1, NUM_KV_HEADS, n_ctx, HEAD_DIM)).astype(mx.bfloat16)
        v = mx.random.normal((1, NUM_KV_HEADS, n_ctx, HEAD_DIM)).astype(mx.bfloat16)
        mx.eval(q, k, v)

        # Intermediate size for unfused path (scores matrix)
        intermediate_mb = NUM_Q_HEADS * n_ctx * 4 / 1e6  # float32 scores

        # 1. Fused SDPA (current)
        def fused():
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=SCALE)
            mx.eval(out)
        t_fused = bench(fused)

        # 2. Unfused 3-pass: separate Q@K, softmax, attn@V
        # Reshape for GQA: group Q heads with their KV head
        q_gqa = mx.reshape(q, (1, NUM_KV_HEADS, GQA_FACTOR, 1, HEAD_DIM))
        k_t = mx.expand_dims(k, axis=2)  # (1, kv_heads, 1, N, D)
        k_t = mx.transpose(k_t, (0, 1, 2, 4, 3))  # (1, kv_heads, 1, D, N)
        v_gqa = mx.expand_dims(v, axis=2)  # (1, kv_heads, 1, N, D)
        mx.eval(q_gqa, k_t, v_gqa)

        def unfused():
            # Pass 1: scores + max (Q @ K^T)
            scores = (q_gqa @ k_t) * SCALE  # (1, kv_heads, gqa, 1, N)
            # Pass 2: softmax (max is computed internally by softmax)
            weights = mx.softmax(scores, axis=-1)
            # Pass 3: weighted sum (weights @ V)
            out = weights @ v_gqa  # (1, kv_heads, gqa, 1, D)
            out = mx.reshape(out, (1, NUM_Q_HEADS, 1, HEAD_DIM))
            mx.eval(out)
        t_unfused = bench(unfused)

        speedup = t_fused / t_unfused

        print(f"{n_ctx:>6}  {t_fused:>10.3f}  {t_unfused:>10.3f}  {speedup:>8.2f}x  {intermediate_mb:>12.1f}")

    print()
    print("speedup > 1.0 means unfused 3-pass is FASTER than fused SDPA")
    print("(if true, a fused 3-pass kernel will be even faster)")


if __name__ == "__main__":
    main()
