"""Test 2: Dense qmm at M=48/96/192/384 — the go/no-go gate.
K=2048 (works with quantized_matmul), N=2048.
Gate: M=192 >= 10 TFLOPS → restructure worth it.
"""
import statistics
import time

import mlx.core as mx

K = 2048
N = 2048

print("=" * 60)
print("Dense qmm TFLOPS (8-bit, gs=64) — M scaling")
print(f"K={K}, N={N}")
print(f"Gate: M=192 >= 10 TFLOPS → GO")
print("=" * 60)

for M in [48, 96, 192, 384, 768, 1536]:
    w_fp = mx.random.normal((K, N), dtype=mx.float16)
    q = mx.quantize(w_fp, bits=8, group_size=64)
    w_q, w_s, w_b = q[0], q[1], q[2]
    x = mx.random.normal((M, K), dtype=mx.float16)
    mx.eval(w_q, w_s, w_b, x)
    try:
        for _ in range(5):
            out = mx.quantized_matmul(x, w_q, w_s, w_b, group_size=64, bits=8)
            mx.eval(out); mx.synchronize()
        samples = []
        for _ in range(20):
            mx.synchronize(); t0 = time.perf_counter()
            out = mx.quantized_matmul(x, w_q, w_s, w_b, group_size=64, bits=8)
            mx.eval(out); mx.synchronize()
            samples.append((time.perf_counter() - t0) * 1e6)
        t = statistics.median(samples)
        flops = 2 * M * K * N
        tflops = flops / (t * 1e-6) / 1e12
        gate = " ← GATE (>=10 = GO)" if M == 192 else ""
        print(f"  M={M:4d}: {t:6.0f} us = {tflops:5.1f} TFLOPS{gate}")
    except Exception as e:
        print(f"  M={M:4d}: FAILED: {str(e)[:80]}")

print()
print("=== AMDAHL MATH ===")
print("MoE = 26.7% of wall at 500K")
print("Current M=48: 7.3 TFLOPS (from spans)")
print("If M=192 achieves X TFLOPS:")
print("  speedup = X / 7.3")
print("  wall reduction = 26.7% × (1 - 1/speedup)")
print("  new tps = 339 / (1 - wall_reduction)")