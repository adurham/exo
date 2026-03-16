#!/usr/bin/env python3
"""Benchmark SDPA kernel: float vs bfloat16 vs quantized KV, across context lengths.

Usage:
  uv run python scripts/bench_sdpa_blocks.py

Compares three modes:
  1. float32 KV (current benchmark baseline)
  2. bfloat16 KV (actual inference dtype)
  3. int8 quantized KV (via mx.fast.quantized_scaled_dot_product_attention)

Shows bandwidth utilization to identify where we're leaving performance on the table.
"""
import time
import mlx.core as mx

# Model config matching Qwen3-235B-A22B with TP=2
NUM_Q_HEADS = 32   # 64 total / 2 nodes
NUM_KV_HEADS = 2   # 4 total / 2 nodes
HEAD_DIM = 128
NUM_LAYERS = 94
SCALE = 1.0 / (HEAD_DIM ** 0.5)
PEAK_BW_GBS = 546  # M4 Max peak memory bandwidth

CONTEXT_LENGTHS = [1024, 4096, 8192, 16384, 32768, 45000, 65536]
WARMUP = 5
ITERS = 20
GROUP_SIZE = 64


def bench_sdpa_float(n_ctx: int, dtype: mx.Dtype) -> float:
    """Benchmark standard SDPA with given dtype. Returns avg ms per layer."""
    q = mx.random.normal((1, NUM_Q_HEADS, 1, HEAD_DIM)).astype(dtype)
    k = mx.random.normal((1, NUM_KV_HEADS, n_ctx, HEAD_DIM)).astype(dtype)
    v = mx.random.normal((1, NUM_KV_HEADS, n_ctx, HEAD_DIM)).astype(dtype)
    mx.eval(q, k, v)

    for _ in range(WARMUP):
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=SCALE)
        mx.eval(out)

    times = []
    for _ in range(ITERS):
        mx.synchronize()
        t0 = time.perf_counter_ns()
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=SCALE)
        mx.eval(out)
        mx.synchronize()
        times.append((time.perf_counter_ns() - t0) / 1e6)

    return sum(times) / len(times)


def bench_sdpa_quantized(n_ctx: int) -> float:
    """Benchmark quantized SDPA (int8 KV). Returns avg ms per layer."""
    q = mx.random.normal((1, NUM_Q_HEADS, 1, HEAD_DIM)).astype(mx.bfloat16)
    k_full = mx.random.normal((1, NUM_KV_HEADS, n_ctx, HEAD_DIM)).astype(mx.bfloat16)
    v_full = mx.random.normal((1, NUM_KV_HEADS, n_ctx, HEAD_DIM)).astype(mx.bfloat16)

    # Quantize K and V to int8 (matches QuantizedKVCache behavior)
    k_data, k_scales, k_biases = mx.quantize(k_full, group_size=GROUP_SIZE, bits=8)
    v_data, v_scales, v_biases = mx.quantize(v_full, group_size=GROUP_SIZE, bits=8)
    mx.eval(q, k_data, k_scales, k_biases, v_data, v_scales, v_biases)

    for _ in range(WARMUP):
        out = mx.fast.quantized_scaled_dot_product_attention(
            q, k_data, k_scales, k_biases, v_data, v_scales, v_biases,
            scale=SCALE, group_size=GROUP_SIZE,
        )
        mx.eval(out)

    times = []
    for _ in range(ITERS):
        mx.synchronize()
        t0 = time.perf_counter_ns()
        out = mx.fast.quantized_scaled_dot_product_attention(
            q, k_data, k_scales, k_biases, v_data, v_scales, v_biases,
            scale=SCALE, group_size=GROUP_SIZE,
        )
        mx.eval(out)
        mx.synchronize()
        times.append((time.perf_counter_ns() - t0) / 1e6)

    return sum(times) / len(times)


def kv_bytes(n_ctx: int, dtype_bytes: int) -> int:
    """Total KV bytes read per layer (K + V, all KV heads)."""
    return NUM_KV_HEADS * n_ctx * HEAD_DIM * 2 * dtype_bytes  # 2 for K+V


def main() -> None:
    print(f"Config: {NUM_Q_HEADS}Q/{NUM_KV_HEADS}KV heads, dim={HEAD_DIM}, {NUM_LAYERS} layers")
    print(f"Peak bandwidth: {PEAK_BW_GBS} GB/s")
    print()

    # Header
    print(f"{'Ctx':>6}  {'float32':>9}  {'bf16':>9}  {'int8':>9}  "
          f"{'bf16 BW':>9}  {'int8 BW':>9}  {'bf16 util':>9}  {'int8 util':>9}  "
          f"{'int8 vs':>8}")
    print(f"{'':>6}  {'×94 ms':>9}  {'×94 ms':>9}  {'×94 ms':>9}  "
          f"{'GB/s':>9}  {'GB/s':>9}  {'%peak':>9}  {'%peak':>9}  "
          f"{'bf16':>8}")
    print("-" * 100)

    for n_ctx in CONTEXT_LENGTHS:
        t_f32 = bench_sdpa_float(n_ctx, mx.float32)
        t_bf16 = bench_sdpa_float(n_ctx, mx.bfloat16)
        t_int8 = bench_sdpa_quantized(n_ctx)

        total_f32 = t_f32 * NUM_LAYERS
        total_bf16 = t_bf16 * NUM_LAYERS
        total_int8 = t_int8 * NUM_LAYERS

        # Bandwidth utilization for bf16
        bf16_bytes = kv_bytes(n_ctx, 2)  # 2 bytes per bf16
        bf16_bw = (bf16_bytes / (t_bf16 / 1000)) / 1e9 if t_bf16 > 0 else 0
        bf16_util = bf16_bw / PEAK_BW_GBS * 100

        # Bandwidth utilization for int8 (1 byte data + scales/biases overhead)
        int8_bytes_data = NUM_KV_HEADS * n_ctx * HEAD_DIM * 2 // 4  # packed uint32
        int8_bytes_meta = NUM_KV_HEADS * n_ctx * (HEAD_DIM // GROUP_SIZE) * 2 * 2 * 4  # scales+biases, K+V, float32
        int8_total_bytes = int8_bytes_data * 4 + int8_bytes_meta  # uint32 = 4 bytes each
        int8_bw = (int8_total_bytes / (t_int8 / 1000)) / 1e9 if t_int8 > 0 else 0
        int8_util = int8_bw / PEAK_BW_GBS * 100

        speedup = total_bf16 / total_int8 if total_int8 > 0 else 0

        print(f"{n_ctx:>6}  {total_f32:>9.1f}  {total_bf16:>9.1f}  {total_int8:>9.1f}  "
              f"{bf16_bw:>9.1f}  {int8_bw:>9.1f}  {bf16_util:>8.1f}%  {int8_util:>8.1f}%  "
              f"{speedup:>7.2f}x")

    print()
    print("BW = effective memory bandwidth (KV data read / SDPA time)")
    print("int8 vs bf16 = speedup from quantized KV (>1.0 = int8 is faster)")


if __name__ == "__main__":
    main()
