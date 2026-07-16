"""Item 3: Stream-overlap microbenchmark.
Can a bandwidth-bound gather + a compute-bound GEMM overlap on separate
mx.new_stream(mx.gpu) streams? If yes, double-buffer the gather behind the
attention GEMMs. If no (serialized), the double-buffer idea dies for $0.
"""
import statistics
import time

import mlx.core as mx
import numpy as np

B, L, D, K, P = 1, 2048, 512, 512, 1024
mx.random.seed(42)
pooled = mx.random.normal((B, P, D), dtype=mx.bfloat16)
topk = mx.array(np.random.randint(0, P, (B, L, K), dtype=np.int32))
q = mx.random.normal((B, 64, L, D), dtype=mx.bfloat16)
kv = mx.random.normal((B, 1, 128, D), dtype=mx.bfloat16)
mx.eval(pooled, topk, q, kv)


def gemm_work():
    # 50 attention-shaped GEMMs: (B,H,L,D) @ (B,1,D,sw) = (B,H,L,sw)
    # Use kv (B,1,128,D) -> swap to (B,1,D,128) for matmul
    kv_t = kv.swapaxes(-1, -2)  # (B,1,D,128)
    scores = q @ kv_t  # (B,64,L,128)
    for _ in range(49):
        scores = q @ kv_t
    return scores


def gather_work():
    pooled_flat = pooled.reshape(B * P, D)
    offset = (mx.arange(B) * P).reshape(B, 1, 1)
    topk_flat = (topk + offset).reshape(-1)
    g = pooled_flat[topk_flat].reshape(B, L, K, D)
    for _ in range(49):
        g = pooled_flat[topk_flat].reshape(B, L, K, D)
    return g


def bench(fn, n=30, w=8):
    for _ in range(w):
        out = fn()
        mx.eval(out)
        mx.synchronize()
    samples = []
    for _ in range(n):
        mx.synchronize()
        t0 = time.perf_counter()
        out = fn()
        mx.eval(out)
        mx.synchronize()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples) * 1e6


t_gemm = bench(gemm_work)
t_gather = bench(gather_work)
print(f"GEMM only (50 iters):    {t_gemm:.0f} us")
print(f"Gather only (50 iters):  {t_gather:.0f} us")
print(f"Sum (if serialized):     {t_gemm + t_gather:.0f} us")

s_gpu = mx.new_stream(mx.gpu)
print("GPU stream created")


def serialized():
    s = gemm_work()
    g = gather_work()
    mx.eval(s, g)
    mx.synchronize()


def overlapped():
    with mx.stream(s_gpu):
        g = gather_work()
    s = gemm_work()
    mx.eval(s, g)
    mx.synchronize()


t_ser = bench(serialized)
t_ovl = bench(overlapped)
print(f"\nSerialized (same stream):     {t_ser:.0f} us")
print(f"Overlapped (separate streams): {t_ovl:.0f} us")
print(f"Overlap factor: {t_ser / t_ovl:.2f}x (>1.0 means they overlap)")
if t_ovl < t_ser * 0.9:
    print("  -> STREAMS OVERLAP on Metal — double-buffer is viable!")
elif t_ovl < t_ser * 0.95:
    print("  -> PARTIAL overlap — some hiding possible")
else:
    print("  -> NO overlap — MLX serializes the streams on Metal. Double-buffer dies.")