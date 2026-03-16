#!/usr/bin/env python3
"""Test impact of MLX_MAX_OPS_PER_BUFFER on decode performance.

MLX flushes the GPU pipeline every N ops. Default is 50 for M4 Max,
meaning ~13 pipeline flushes per decode token (658 ops / 50).
Each flush commits the command buffer and waits for completion.

Usage:
  for ops in 50 100 200 500 1000; do
    echo "=== MLX_MAX_OPS_PER_BUFFER=$ops ==="
    MLX_MAX_OPS_PER_BUFFER=$ops uv run python scripts/bench_buffer_ops.py
  done
"""
import os
import time
import mlx.core as mx

NUM_Q_HEADS = 32
NUM_KV_HEADS = 2
HEAD_DIM = 128
NUM_LAYERS = 94
SCALE = 1.0 / (HEAD_DIM ** 0.5)
WARMUP = 3
ITERS = 15


def simulate_one_layer(q, k, v, w1, w2):
    """Simulate one transformer layer: SDPA + 2 matmuls + activations."""
    # Attention
    attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=SCALE)

    # Simulate FFN: matmul + silu + matmul + residual
    x = mx.reshape(attn, (1, -1))  # flatten
    h = x @ w1
    h = mx.maximum(h, 0)  # ReLU as proxy for SiLU
    h = h @ w2
    h = h + x  # residual
    return mx.reshape(h, q.shape), k, v


def bench_multi_layer(n_ctx: int, n_layers: int) -> float:
    """Benchmark n_layers of SDPA + FFN. Returns ms."""
    hidden = NUM_Q_HEADS * HEAD_DIM  # 4096
    ffn_dim = 1280  # approximate

    q = mx.random.normal((1, NUM_Q_HEADS, 1, HEAD_DIM)).astype(mx.bfloat16)
    k = mx.random.normal((1, NUM_KV_HEADS, n_ctx, HEAD_DIM)).astype(mx.bfloat16)
    v = mx.random.normal((1, NUM_KV_HEADS, n_ctx, HEAD_DIM)).astype(mx.bfloat16)
    w1 = mx.random.normal((hidden, ffn_dim)).astype(mx.bfloat16)
    w2 = mx.random.normal((ffn_dim, hidden)).astype(mx.bfloat16)
    mx.eval(q, k, v, w1, w2)

    def run():
        x = q
        for _ in range(n_layers):
            x, _, _ = simulate_one_layer(x, k, v, w1, w2)
        mx.eval(x)

    for _ in range(WARMUP):
        run()

    times = []
    for _ in range(ITERS):
        mx.synchronize()
        t0 = time.perf_counter_ns()
        run()
        mx.synchronize()
        times.append((time.perf_counter_ns() - t0) / 1e6)

    return sum(times) / len(times)


def main():
    max_ops = os.environ.get("MLX_MAX_OPS_PER_BUFFER", "50")
    max_mb = os.environ.get("MLX_MAX_MB_PER_BUFFER", "50")
    print(f"MLX_MAX_OPS_PER_BUFFER={max_ops}, MLX_MAX_MB_PER_BUFFER={max_mb}")
    print()

    # Test with different layer counts to see scaling
    for n_ctx in [8192, 45000]:
        print(f"Context={n_ctx}:")
        for n_layers in [1, 10, 94]:
            t = bench_multi_layer(n_ctx, n_layers)
            ops_estimate = n_layers * 7  # ~7 kernel dispatches per layer
            flushes = ops_estimate // int(max_ops)
            print(f"  {n_layers:>3} layers: {t:>8.1f}ms  (~{ops_estimate} ops, ~{flushes} flushes)")
        print()


if __name__ == "__main__":
    main()
