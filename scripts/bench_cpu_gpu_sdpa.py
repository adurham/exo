#!/usr/bin/env python3
"""Prototype: CPU+GPU hybrid attention on Apple Silicon unified memory.

Tests whether splitting attention between GPU (MLX) and CPU (numpy/Accelerate)
can improve total throughput by utilizing unused memory bandwidth.

GPU achieves 118 GB/s of 546 GB/s — the CPU can potentially use the remaining
bandwidth in parallel since they share the same unified memory controller.

Usage:
  uv run python scripts/bench_cpu_gpu_sdpa.py

  # Limit CPU threads (avoid starving the system):
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 uv run python scripts/bench_cpu_gpu_sdpa.py
"""
import os
import time
import threading
import numpy as np
import mlx.core as mx

NUM_Q_HEADS = 32
NUM_KV_HEADS = 2
HEAD_DIM = 128
GQA_FACTOR = NUM_Q_HEADS // NUM_KV_HEADS
SCALE = 1.0 / (HEAD_DIM ** 0.5)

WARMUP = 3
ITERS = 15


def cpu_attention_partial(q_np, k_np, v_np, scale):
    """Compute attention on CPU using numpy (backed by Accelerate/AMX).

    Fully vectorized — uses batched matmul for all Q heads at once.
    Returns (output, max_scores, sum_exps) for merging with GPU results.
    q: (num_q_heads, 1, D)
    k: (num_kv_heads, N_cpu, D)
    v: (num_kv_heads, N_cpu, D)
    """
    n_kv = k_np.shape[0]
    n_q_per_kv = q_np.shape[0] // n_kv
    D = q_np.shape[-1]

    # Reshape Q for GQA: (kv_heads, gqa_factor, D)
    q_gqa = q_np[:, 0, :].reshape(n_kv, n_q_per_kv, D)

    # Batched Q @ K^T: (kv_heads, gqa, D) @ (kv_heads, D, N) → (kv_heads, gqa, N)
    scores = np.matmul(q_gqa, k_np.transpose(0, 2, 1)) * scale

    # Softmax per Q head
    max_scores = scores.max(axis=-1)  # (kv_heads, gqa)
    exp_scores = np.exp(scores - max_scores[..., None])
    sum_exps = exp_scores.sum(axis=-1)  # (kv_heads, gqa)

    # Weighted V: (kv_heads, gqa, N) @ (kv_heads, N, D) → (kv_heads, gqa, D)
    outputs = np.matmul(exp_scores, v_np)

    # Reshape back to (num_q_heads, 1, D)
    outputs = outputs.reshape(q_np.shape[0], 1, D)
    max_scores = max_scores.reshape(q_np.shape[0])
    sum_exps = sum_exps.reshape(q_np.shape[0])

    return outputs, max_scores, sum_exps


def merge_partial_results(out1, max1, sum1, out2, max2, sum2):
    """Merge two partial attention results using online softmax merge.

    Same math as the 2pass_2 kernel.
    """
    # Find global max
    global_max = np.maximum(max1, max2)

    # Rescale sums
    factor1 = np.exp(max1 - global_max)
    factor2 = np.exp(max2 - global_max)
    new_sum1 = sum1 * factor1
    new_sum2 = sum2 * factor2
    total_sum = new_sum1 + new_sum2

    # Rescale outputs and combine
    f1 = (factor1 / total_sum)[:, None, None]
    f2 = (factor2 / total_sum)[:, None, None]
    merged = out1 * f1 + out2 * f2

    return merged


def bench_gpu_only(q, k, v, n_ctx):
    """Benchmark: full GPU SDPA."""
    def run():
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=SCALE)
        mx.eval(out)

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


def bench_cpu_only(q_np, k_np, v_np):
    """Benchmark: full CPU attention."""
    for _ in range(WARMUP):
        cpu_attention_partial(q_np, k_np, v_np, SCALE)

    times = []
    for _ in range(ITERS):
        t0 = time.perf_counter_ns()
        cpu_attention_partial(q_np, k_np, v_np, SCALE)
        times.append((time.perf_counter_ns() - t0) / 1e6)
    return sum(times) / len(times)


def bench_hybrid(q_mx, k_mx, v_mx, q_np, k_np, v_np, split_pos):
    """Benchmark: GPU handles [0, split_pos), CPU handles [split_pos, N)."""
    N = k_mx.shape[2]

    # Pre-split arrays
    k_gpu = k_mx[:, :, :split_pos, :]
    v_gpu = v_mx[:, :, :split_pos, :]
    k_cpu = k_np[:, split_pos:, :]  # (kv_heads, N_cpu, D)
    v_cpu = v_np[:, split_pos:, :]
    mx.eval(k_gpu, v_gpu)

    # Reshape q for CPU
    q_cpu = q_np  # (q_heads, 1, D)

    def run():
        result = {}

        def gpu_work():
            out = mx.fast.scaled_dot_product_attention(
                q_mx, k_gpu, v_gpu, scale=SCALE)
            mx.eval(out)
            result['gpu'] = out

        def cpu_work():
            out, maxs, sums = cpu_attention_partial(q_cpu, k_cpu, v_cpu, SCALE)
            result['cpu'] = (out, maxs, sums)

        # Launch both
        gpu_thread = threading.Thread(target=gpu_work)
        gpu_thread.start()
        cpu_work()  # CPU runs on main thread
        gpu_thread.join()

        return result

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
    # Check thread count
    import subprocess
    result = subprocess.run(
        ["sysctl", "-n", "hw.perflevel0.logicalcpu"],
        capture_output=True, text=True)
    p_cores = result.stdout.strip() if result.returncode == 0 else "?"
    omp = os.environ.get("OMP_NUM_THREADS", "all")

    print(f"CPU P-cores: {p_cores}, OMP_NUM_THREADS={omp}")
    print(f"Config: {NUM_Q_HEADS}Q/{NUM_KV_HEADS}KV heads, dim={HEAD_DIM}, GQA={GQA_FACTOR}")
    print()

    print(f"{'Ctx':>6}  {'GPU only':>9}  {'CPU only':>9}  {'Hybrid':>9}  "
          f"{'GPU/Hyb':>8}  {'CPU frac':>8}  {'Split':>6}")
    print(f"{'':>6}  {'ms':>9}  {'ms':>9}  {'ms':>9}  "
          f"{'speedup':>8}  {'of total':>8}  {'pos':>6}")
    print("-" * 66)

    for n_ctx in [4096, 8192, 16384, 32768, 45000, 65536]:
        # Create data (float32 for numpy compat)
        q_mx = mx.random.normal((1, NUM_Q_HEADS, 1, HEAD_DIM))
        k_mx = mx.random.normal((1, NUM_KV_HEADS, n_ctx, HEAD_DIM))
        v_mx = mx.random.normal((1, NUM_KV_HEADS, n_ctx, HEAD_DIM))
        mx.eval(q_mx, k_mx, v_mx)

        # Convert to numpy (copy — bfloat16 not supported by numpy)
        q_np = np.array(q_mx[0])  # (q_heads, 1, D)
        k_np = np.array(k_mx[0])  # (kv_heads, N, D)
        v_np = np.array(v_mx[0])  # (kv_heads, N, D)

        # GPU-only baseline
        t_gpu = bench_gpu_only(q_mx, k_mx, v_mx, n_ctx)

        # CPU-only throughput measurement (use 4K sample, extrapolate)
        sample_n = min(n_ctx, 4096)
        t_cpu_sample = bench_cpu_only(q_np, k_np[:, :sample_n, :], v_np[:, :sample_n, :])
        cpu_pos_per_ms = sample_n / t_cpu_sample if t_cpu_sample > 0 else 1
        t_cpu_est = n_ctx / cpu_pos_per_ms

        # Compute optimal split based on measured throughputs
        gpu_pos_per_ms = n_ctx / t_gpu if t_gpu > 0 else 1
        # CPU fraction = cpu_rate / (cpu_rate + gpu_rate)
        optimal_cpu_frac = cpu_pos_per_ms / (cpu_pos_per_ms + gpu_pos_per_ms)
        # Clamp to reasonable range
        cpu_frac = max(0.05, min(0.40, optimal_cpu_frac))
        split_pos = int(n_ctx * (1 - cpu_frac))

        # Test multiple splits to find best
        best_t = float('inf')
        best_frac = cpu_frac
        for frac in [0.05, 0.10, 0.15, 0.20, cpu_frac, 0.30]:
            sp = int(n_ctx * (1 - frac))
            if sp < 100 or sp >= n_ctx - 100:
                continue
            t = bench_hybrid(q_mx, k_mx, v_mx, q_np, k_np, v_np, sp)
            if t < best_t:
                best_t = t
                best_frac = frac

        speedup = t_gpu / best_t if best_t > 0 else 0

        print(f"{n_ctx:>6}  {t_gpu:>9.2f}  {t_cpu_est:>9.1f}  {best_t:>9.2f}  "
              f"{speedup:>8.2f}x  {best_frac:>7.0%}  {int(n_ctx*(1-best_frac)):>6}")

    print()
    print("GPU/Hyb > 1.0 means hybrid is faster")
    print("CPU runs numpy (backed by Accelerate/AMX on Apple Silicon)")


if __name__ == "__main__":
    main()
