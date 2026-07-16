"""Dense quantized_matmul TFLOPS — the TRUE ceiling for gather_qmm."""
import statistics
import time

import mlx.core as mx

print("=== Dense quantized_matmul TFLOPS (8-bit, group_size=64) ===")
print(f"{'M':>6} {'K':>5} {'N':>5} {'time_us':>8} {'TFLOPS':>8}")

for M, K, N in [
    (48, 2048, 1024),
    (1536, 2048, 1024),
    (4096, 2048, 1024),
    (8192, 2048, 1024),
    (48, 2048, 2048),
    (1536, 2048, 2048),
    (8192, 2048, 2048),
]:
    w_fp = mx.random.normal((K, N), dtype=mx.float16)
    q = mx.quantize(w_fp, bits=8, group_size=64)
    w_q, w_s, w_b = q[0], q[1], q[2]
    x = mx.random.normal((M, K), dtype=mx.float16)
    mx.eval(w_q, w_s, w_b, x)
    try:
        for _ in range(5):
            out = mx.quantized_matmul(x, w_q, w_s, w_b, group_size=64, bits=8)
            mx.eval(out)
            mx.synchronize()
        samples = []
        for _ in range(20):
            mx.synchronize()
            t0 = time.perf_counter()
            out = mx.quantized_matmul(x, w_q, w_s, w_b, group_size=64, bits=8)
            mx.eval(out)
            mx.synchronize()
            samples.append((time.perf_counter() - t0) * 1e6)
        t = statistics.median(samples)
        flops = 2 * M * K * N
        tflops = flops / (t * 1e-6) / 1e12
        print(f"{M:6d} {K:5d} {N:5d} {t:8.0f} {tflops:8.1f}")
    except Exception as e:
        print(f"{M:6d} {K:5d} {N:5d} FAILED: {str(e)[:80]}")

print()
print("=== For comparison: fp16 dense GEMM (measured earlier) ===")
print("M=48: 1.0 TFLOPS, M=1536: 9.4, M=8192: 13.6, Peak: 15.0")
print()
print("The qmm ceiling is the REAL ceiling for gather_qmm.")
print("If qmm at M=1536 is ~5-6 TFLOPS (Fable's estimate), the MoE gap is ~2x not 5x.")