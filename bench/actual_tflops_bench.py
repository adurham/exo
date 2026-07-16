"""Measure ACTUAL M4 Max dense GEMM TFLOPS — the empirical compute ceiling.
Not a spec number. Measure at multiple shapes to find the true peak."""
import statistics
import time

import mlx.core as mx

shapes_fp16 = [
    (1536, 2048, 1024),
    (4096, 2048, 1024),
    (8192, 2048, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (8192, 4096, 4096),
]

print("=== ACTUAL M4 Max dense fp16 GEMM TFLOPS ===")
print(f"{'M':>6} {'K':>5} {'N':>5} {'time_us':>8} {'TFLOPS':>8}")
for M, K, N in shapes_fp16:
    mx.random.seed(42)
    a = mx.random.normal((M, K), dtype=mx.float16)
    b = mx.random.normal((K, N), dtype=mx.float16)
    mx.eval(a, b)
    for _ in range(5):
        c = a @ b; mx.eval(c); mx.synchronize()
    samples = []
    for _ in range(20):
        mx.synchronize(); t0 = time.perf_counter()
        c = a @ b; mx.eval(c); mx.synchronize()
        samples.append((time.perf_counter() - t0) * 1e6)
    t = statistics.median(samples)
    flops = 2 * M * K * N
    tflops = flops / (t * 1e-6) / 1e12
    print(f"{M:6d} {K:5d} {N:5d} {t:8.0f} {tflops:8.1f}")

print()
print("=== bf16 (model uses bf16 activations) ===")
for M, K, N in [(1536, 2048, 1024), (4096, 2048, 1024), (8192, 2048, 1024)]:
    a = mx.random.normal((M, K), dtype=mx.bfloat16)
    b = mx.random.normal((K, N), dtype=mx.bfloat16)
    mx.eval(a, b)
    for _ in range(5):
        c = a @ b; mx.eval(c); mx.synchronize()
    samples = []
    for _ in range(20):
        mx.synchronize(); t0 = time.perf_counter()
        c = a @ b; mx.eval(c); mx.synchronize()
        samples.append((time.perf_counter() - t0) * 1e6)
    t = statistics.median(samples)
    flops = 2 * M * K * N
    tflops = flops / (t * 1e-6) / 1e12
    print(f"bf16 {M:6d} {K:5d} {N:5d} {t:8.0f} {tflops:8.1f}")

print()
print("=== Now measure the ACTUAL gather_qmm at production shape ===")
print("The 141 GB/s number is from the July 1 handoff. Let me re-measure TODAY.")
import numpy as np

# Production shape: 256 experts, top-6, chunk=2048
# B/E = 48, but use 32 experts to fit memory
N_EXP = 32
M_PER = 48
B = N_EXP * M_PER  # 1536
K = 2048
N = 1024

w_fp = mx.random.normal((N_EXP, N, K), dtype=mx.float16)
q = mx.quantize(w_fp, bits=8, group_size=64)
w_q, w_s, w_b = q[0], q[1], q[2]
x = mx.random.normal((B, K), dtype=mx.float16)
inds = mx.array(np.repeat(np.arange(N_EXP), M_PER), dtype=mx.int32)
mx.eval(w_q, w_s, w_b, x, inds)

# Measure gather_qmm with sorted indices (production path)
def gather_sorted():
    return mx.gather_qmm(
        x[None], w_q[None], w_s[None], w_b[None],
        rhs_indices=inds[None],
        sorted_indices=True, transpose=True,
        group_size=64, bits=8, mode="affine"
    )

try:
    for _ in range(5):
        out = gather_sorted(); mx.eval(out); mx.synchronize()
    samples = []
    for _ in range(20):
        mx.synchronize(); t0 = time.perf_counter()
        out = gather_sorted(); mx.eval(out); mx.synchronize()
        samples.append((time.perf_counter() - t0) * 1e6)
    t_qmm = statistics.median(samples)
    flops = 2 * B * N * K
    tflops_qmm = flops / (t_qmm * 1e-6) / 1e12
    weight_bytes = N_EXP * N * K * 1  # 8-bit
    gbps = weight_bytes / (t_qmm * 1e-6) / 1e9
    print(f"gather_qmm sorted: {t_qmm:.0f} us, {tflops_qmm:.1f} TFLOPS, {gbps:.0f} GB/s")
except Exception as e:
    print(f"gather_qmm sorted FAILED: {str(e)[:200]}")

# Also try unsorted for comparison
def gather_unsorted():
    return mx.gather_qmm(
        x[None], w_q[None], w_s[None], w_b[None],
        rhs_indices=inds[None],
        sorted_indices=False, transpose=True,
        group_size=64, bits=8, mode="affine"
    )

try:
    for _ in range(3):
        out = gather_unsorted(); mx.eval(out); mx.synchronize()
    samples = []
    for _ in range(10):
        mx.synchronize(); t0 = time.perf_counter()
        out = gather_unsorted(); mx.eval(out); mx.synchronize()
        samples.append((time.perf_counter() - t0) * 1e6)
    t_uns = statistics.median(samples)
    flops = 2 * B * N * K
    tflops_uns = flops / (t_uns * 1e-6) / 1e12
    gbps_uns = weight_bytes / (t_uns * 1e-6) / 1e9
    print(f"gather_qmm unsorted: {t_uns:.0f} us, {tflops_uns:.1f} TFLOPS, {gbps_uns:.0f} GB/s")
except Exception as e:
    print(f"gather_qmm unsorted FAILED: {str(e)[:200]}")