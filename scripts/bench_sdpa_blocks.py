#!/usr/bin/env python3
"""Benchmark SDPA 2-pass kernel at various context lengths and threadgroup targets.

Usage:
  # Default (MLX_SDPA_MAX_TG=320):
  uv run python scripts/bench_sdpa_blocks.py

  # Test specific target:
  MLX_SDPA_MAX_TG=160 uv run python scripts/bench_sdpa_blocks.py

  # Sweep multiple targets (run on cluster node):
  for tg in 80 160 320 640 1280; do
    echo "=== MLX_SDPA_MAX_TG=$tg ==="
    MLX_SDPA_MAX_TG=$tg uv run python scripts/bench_sdpa_blocks.py
  done
"""
import os
import time
import mlx.core as mx

# Model config matching Qwen3-235B-A22B with TP=2
NUM_Q_HEADS = 32   # 64 total / 2 nodes
NUM_KV_HEADS = 2   # 4 total / 2 nodes
HEAD_DIM = 128
NUM_LAYERS = 94
SCALE = 1.0 / (HEAD_DIM ** 0.5)

CONTEXT_LENGTHS = [1024, 4096, 8192, 16384, 32768, 45000, 65536]
WARMUP = 3
ITERS = 20


def bench_sdpa(n_ctx: int) -> float:
    """Returns average SDPA time in ms for a single layer at given context length."""
    q = mx.random.normal((1, NUM_Q_HEADS, 1, HEAD_DIM))
    k = mx.random.normal((1, NUM_KV_HEADS, n_ctx, HEAD_DIM))
    v = mx.random.normal((1, NUM_KV_HEADS, n_ctx, HEAD_DIM))
    mx.eval(q, k, v)

    # Warmup
    for _ in range(WARMUP):
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=SCALE)
        mx.eval(out)

    # Timed runs
    times = []
    for _ in range(ITERS):
        mx.synchronize()
        t0 = time.perf_counter_ns()
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=SCALE)
        mx.eval(out)
        mx.synchronize()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e6)  # ns -> ms

    return sum(times) / len(times)


def main() -> None:
    max_tg = os.environ.get("MLX_SDPA_MAX_TG", "320")
    print(f"MLX_SDPA_MAX_TG={max_tg}")
    print(f"Config: {NUM_Q_HEADS}Q/{NUM_KV_HEADS}KV heads, dim={HEAD_DIM}")
    print(f"  gqa_factor={NUM_Q_HEADS // NUM_KV_HEADS}, n_simds={NUM_Q_HEADS // NUM_KV_HEADS}")
    print(f"  tg_per_block={NUM_KV_HEADS * 1}")
    print()
    print(f"{'Context':>8}  {'SDPA/layer':>12}  {'×94 layers':>12}  {'per 1K ctx':>12}")
    print(f"{'':>8}  {'(μs)':>12}  {'(ms)':>12}  {'(μs/1K)':>12}")
    print("-" * 52)

    base_time = None
    for n_ctx in CONTEXT_LENGTHS:
        t_ms = bench_sdpa(n_ctx)
        t_us = t_ms * 1000
        total_ms = t_ms * NUM_LAYERS
        per_1k = (t_us / n_ctx) * 1000 if n_ctx > 0 else 0

        if base_time is None:
            base_time = t_ms

        print(f"{n_ctx:>8}  {t_us:>12.1f}  {total_ms:>12.1f}  {per_1k:>12.1f}")

    print()
    if base_time is not None and len(CONTEXT_LENGTHS) > 1:
        last_t = bench_sdpa(CONTEXT_LENGTHS[-1])
        delta_ms = (last_t - base_time) * NUM_LAYERS
        delta_ctx = CONTEXT_LENGTHS[-1] - CONTEXT_LENGTHS[0]
        slope = delta_ms / (delta_ctx / 1000) if delta_ctx > 0 else 0
        print(f"Scaling: {slope:.2f} ms per 1K context tokens (across {NUM_LAYERS} layers)")


if __name__ == "__main__":
    main()
