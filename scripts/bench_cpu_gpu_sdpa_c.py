#!/usr/bin/env python3
"""CPU+GPU hybrid attention using C/Accelerate for CPU portion.

Compares:
1. GPU-only SDPA (baseline)
2. CPU-only via C/Accelerate (to measure raw CPU throughput)
3. Hybrid: GPU + CPU in parallel with auto-calibrated split
"""
import ctypes
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
ITERS = 20

# Load C library
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LIB_PATH = os.path.join(SCRIPT_DIR, "..", "src", "exo", "worker", "engines", "mlx", "cpu_attention.dylib")
try:
    _lib = ctypes.CDLL(LIB_PATH)
    _lib.cpu_attention_f32.argtypes = [
        ctypes.c_void_p,  # q
        ctypes.c_void_p,  # k
        ctypes.c_void_p,  # v
        ctypes.c_void_p,  # out
        ctypes.c_void_p,  # max_out
        ctypes.c_void_p,  # sum_out
        ctypes.c_float,   # scale
        ctypes.c_int,     # num_q_heads
        ctypes.c_int,     # num_kv_heads
        ctypes.c_int,     # N
        ctypes.c_int,     # D
    ]
    _lib.cpu_attention_f32.restype = None
    HAS_C_LIB = True
    print("C/Accelerate library loaded")
except OSError as e:
    HAS_C_LIB = False
    print(f"C library not found: {e}")
    print("Build with: clang -O3 -shared -DACCELERATE_NEW_LAPACK -o cpu_attention.dylib cpu_attention.c -framework Accelerate")


def cpu_attention_c(q_np, k_np, v_np, scale):
    """CPU attention via C/Accelerate. Returns (out, max, sum_exp)."""
    n_q = q_np.shape[0]
    n_kv = k_np.shape[0]
    N = k_np.shape[1]
    D = k_np.shape[2]

    # Ensure contiguous float32
    q_f32 = np.ascontiguousarray(q_np.reshape(n_q, D), dtype=np.float32)
    k_f32 = np.ascontiguousarray(k_np, dtype=np.float32)
    v_f32 = np.ascontiguousarray(v_np, dtype=np.float32)

    out = np.zeros((n_q, D), dtype=np.float32)
    max_out = np.zeros(n_q, dtype=np.float32)
    sum_out = np.zeros(n_q, dtype=np.float32)

    _lib.cpu_attention_f32(
        q_f32.ctypes.data,
        k_f32.ctypes.data,
        v_f32.ctypes.data,
        out.ctypes.data,
        max_out.ctypes.data,
        sum_out.ctypes.data,
        ctypes.c_float(scale),
        ctypes.c_int(n_q),
        ctypes.c_int(n_kv),
        ctypes.c_int(N),
        ctypes.c_int(D),
    )
    return out.reshape(n_q, 1, D), max_out, sum_out


def cpu_attention_numpy(q_np, k_np, v_np, scale):
    """CPU attention via numpy (Accelerate-backed batched matmul)."""
    n_kv = k_np.shape[0]
    n_q_per_kv = q_np.shape[0] // n_kv
    D = q_np.shape[-1]

    q_gqa = q_np[:, 0, :].reshape(n_kv, n_q_per_kv, D)
    scores = np.matmul(q_gqa, k_np.transpose(0, 2, 1)) * scale
    max_scores = scores.max(axis=-1)
    exp_scores = np.exp(scores - max_scores[..., None])
    sum_exps = exp_scores.sum(axis=-1)
    outputs = np.matmul(exp_scores, v_np)

    return (outputs.reshape(q_np.shape[0], 1, D),
            max_scores.reshape(q_np.shape[0]),
            sum_exps.reshape(q_np.shape[0]))


def bench(fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        fn()
        times.append((time.perf_counter_ns() - t0) / 1e6)
    return sum(times) / len(times)


def bench_gpu(q, k, v):
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


def bench_hybrid(q_mx, k_mx, v_mx, q_np, k_np, v_np, split_pos, cpu_fn):
    N = k_mx.shape[2]
    k_gpu = k_mx[:, :, :split_pos, :]
    v_gpu = v_mx[:, :, :split_pos, :]
    k_cpu = k_np[:, split_pos:, :]
    v_cpu = v_np[:, split_pos:, :]
    mx.eval(k_gpu, v_gpu)

    def run():
        result = {}

        def gpu_work():
            out = mx.fast.scaled_dot_product_attention(
                q_mx, k_gpu, v_gpu, scale=SCALE)
            mx.eval(out)
            result['gpu'] = out

        def cpu_work():
            result['cpu'] = cpu_fn(q_np, k_cpu, v_cpu, SCALE)

        gpu_thread = threading.Thread(target=gpu_work)
        gpu_thread.start()
        cpu_work()
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
    import subprocess
    result = subprocess.run(
        ["sysctl", "-n", "hw.perflevel0.logicalcpu"],
        capture_output=True, text=True)
    p_cores = result.stdout.strip() if result.returncode == 0 else "?"
    print(f"CPU P-cores: {p_cores}")
    print(f"Config: {NUM_Q_HEADS}Q/{NUM_KV_HEADS}KV heads, dim={HEAD_DIM}")
    print()

    # Header
    cpu_label = "C/Accel" if HAS_C_LIB else "numpy"
    print(f"{'Ctx':>6}  {'GPU':>7}  {cpu_label:>7}  {'numpy':>7}  "
          f"{'Hybrid':>7}  {'Speedup':>7}  {'Frac':>5}  {'Save×94':>7}")
    print("-" * 68)

    for n_ctx in [4096, 8192, 16384, 32768, 45000, 65536]:
        q_mx = mx.random.normal((1, NUM_Q_HEADS, 1, HEAD_DIM))
        k_mx = mx.random.normal((1, NUM_KV_HEADS, n_ctx, HEAD_DIM))
        v_mx = mx.random.normal((1, NUM_KV_HEADS, n_ctx, HEAD_DIM))
        mx.eval(q_mx, k_mx, v_mx)

        q_np = np.array(q_mx[0])
        k_np = np.array(k_mx[0])
        v_np = np.array(v_mx[0])

        # GPU baseline
        t_gpu = bench_gpu(q_mx, k_mx, v_mx)

        # CPU-only (C and numpy)
        sample_n = min(n_ctx, 4096)
        k_s, v_s = k_np[:, :sample_n, :], v_np[:, :sample_n, :]

        t_numpy = bench(lambda: cpu_attention_numpy(q_np, k_s, v_s, SCALE))
        t_numpy_est = t_numpy * (n_ctx / sample_n)

        if HAS_C_LIB:
            t_c = bench(lambda: cpu_attention_c(q_np, k_s, v_s, SCALE))
            t_c_est = t_c * (n_ctx / sample_n)
            cpu_fn = cpu_attention_c
            cpu_rate = sample_n / t_c  # positions per ms
        else:
            t_c_est = t_numpy_est
            cpu_fn = cpu_attention_numpy
            cpu_rate = sample_n / t_numpy

        # Find best split
        gpu_rate = n_ctx / t_gpu
        optimal_frac = cpu_rate / (cpu_rate + gpu_rate)
        optimal_frac = max(0.02, min(0.40, optimal_frac))

        best_t = float('inf')
        best_frac = optimal_frac
        for frac in [0.02, 0.05, 0.10, 0.15, 0.20, optimal_frac, 0.30]:
            sp = int(n_ctx * (1 - frac))
            if sp < 100 or sp >= n_ctx - 100:
                continue
            t = bench_hybrid(q_mx, k_mx, v_mx, q_np, k_np, v_np, sp, cpu_fn)
            if t < best_t:
                best_t = t
                best_frac = frac

        speedup = t_gpu / best_t
        save_ms = (t_gpu - best_t) * 94

        print(f"{n_ctx:>6}  {t_gpu:>7.2f}  {t_c_est:>7.1f}  {t_numpy_est:>7.1f}  "
              f"{best_t:>7.2f}  {speedup:>7.2f}x  {best_frac:>4.0%}  {save_ms:>7.1f}")

    print()
    print(f"CPU impl: {cpu_label}")
    print("Speedup > 1.0 means hybrid is faster. Save×94 = ms saved per token.")


if __name__ == "__main__":
    main()
