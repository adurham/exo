#!/usr/bin/env python3
"""Diagnose SDPA bandwidth bottleneck: is it online softmax or hardware?

Compares:
1. Raw memory copy (mx.sum over KV) — achievable bandwidth ceiling
2. Q·K dot product only (no softmax, no V) — dot product overhead
3. Full SDPA — current performance

If (1) >> (3) but (2) ≈ (1), softmax is the bottleneck → two-pass rewrite helps.
If (1) ≈ (3), hardware ceiling → nothing helps at kernel level.
If (1) >> (2) >> (3), both dot product and softmax add overhead.
"""
import time
import mlx.core as mx

NUM_Q_HEADS = 32
NUM_KV_HEADS = 2
HEAD_DIM = 128
SCALE = 1.0 / (HEAD_DIM ** 0.5)
PEAK_BW = 546  # GB/s

CONTEXTS = [8192, 32768, 45000, 65536]
WARMUP = 5
ITERS = 30


def kv_bytes_bf16(n_ctx: int) -> int:
    return NUM_KV_HEADS * n_ctx * HEAD_DIM * 2 * 2  # K+V, 2 bytes each


def bench(fn, warmup=WARMUP, iters=ITERS) -> float:
    """Returns average time in ms."""
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
    print(f"Config: {NUM_Q_HEADS}Q/{NUM_KV_HEADS}KV heads, dim={HEAD_DIM}")
    print(f"All times are TOTAL across the data (not per-layer)")
    print()
    print(f"{'Ctx':>6}  {'raw sum':>8}  {'matmul':>8}  {'SDPA':>8}  "
          f"{'raw BW':>8}  {'mm BW':>8}  {'sdpa BW':>8}  "
          f"{'mm/raw':>7}  {'sdpa/raw':>8}")
    print(f"{'':>6}  {'ms':>8}  {'ms':>8}  {'ms':>8}  "
          f"{'GB/s':>8}  {'GB/s':>8}  {'GB/s':>8}  "
          f"{'ratio':>7}  {'ratio':>8}")
    print("-" * 95)

    for n_ctx in CONTEXTS:
        q = mx.random.normal((1, NUM_Q_HEADS, 1, HEAD_DIM)).astype(mx.bfloat16)
        k = mx.random.normal((1, NUM_KV_HEADS, n_ctx, HEAD_DIM)).astype(mx.bfloat16)
        v = mx.random.normal((1, NUM_KV_HEADS, n_ctx, HEAD_DIM)).astype(mx.bfloat16)
        mx.eval(q, k, v)

        data_bytes = kv_bytes_bf16(n_ctx)

        # 1. Raw memory read: just sum K+V (forces full read, minimal compute)
        def raw_sum():
            r = mx.sum(k) + mx.sum(v)
            mx.eval(r)
        t_raw = bench(raw_sum)

        # 2. Q·K matmul only (dot products, no softmax, no V)
        # Expand Q to match KV heads via GQA
        q_for_mm = mx.reshape(q, (1, NUM_KV_HEADS, NUM_Q_HEADS // NUM_KV_HEADS, 1, HEAD_DIM))
        k_t = mx.transpose(k, (0, 1, 3, 2))  # (1, kv_heads, D, N)
        k_t = mx.expand_dims(k_t, axis=2)  # (1, kv_heads, 1, D, N)
        mx.eval(q_for_mm, k_t)

        def matmul_only():
            scores = q_for_mm @ k_t  # (1, kv_heads, gqa, 1, N)
            mx.eval(scores)
        t_mm = bench(matmul_only)

        # 3. Full SDPA
        def full_sdpa():
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=SCALE)
            mx.eval(out)
        t_sdpa = bench(full_sdpa)

        # Compute bandwidths
        raw_bw = (data_bytes / (t_raw / 1000)) / 1e9
        # matmul reads K only (half the data) + Q (negligible)
        k_bytes = NUM_KV_HEADS * n_ctx * HEAD_DIM * 2
        mm_bw = (k_bytes / (t_mm / 1000)) / 1e9
        sdpa_bw = (data_bytes / (t_sdpa / 1000)) / 1e9

        mm_ratio = t_mm / t_raw if t_raw > 0 else 0
        sdpa_ratio = t_sdpa / t_raw if t_raw > 0 else 0

        print(f"{n_ctx:>6}  {t_raw:>8.2f}  {t_mm:>8.2f}  {t_sdpa:>8.2f}  "
              f"{raw_bw:>8.1f}  {mm_bw:>8.1f}  {sdpa_bw:>8.1f}  "
              f"{mm_ratio:>7.2f}  {sdpa_ratio:>8.2f}")

    print()
    print("raw sum = mx.sum(K) + mx.sum(V) — pure memory read ceiling")
    print("matmul  = Q @ K^T — dot products only, no softmax or V")
    print("SDPA    = full scaled_dot_product_attention")
    print("BW      = data read / time (higher = better)")
    print("ratio   = time relative to raw sum (1.0 = same speed)")


if __name__ == "__main__":
    main()
