#!/usr/bin/env python3
"""MLX_GEMV_BATCH_INVARIANT validation (mlx fork 1fe020ed).

Checks, for M in [2,8] across dtypes/shapes/layouts:
  1. CORRECTNESS: batched matmul == fp32 reference (tolerance).
  2. BITEXACT: batched matmul rows == per-row (M=1) matmul, bitwise.
  3. Leading batch dims (B, M, K) still correct.
The same script run WITHOUT the env is the control: bf16 large shapes
are expected to show gemv/gemm drift (proves the test detects it).

Run:  MLX_GEMV_BATCH_INVARIANT=1 .venv/bin/python gemv_bi_test.py
      .venv/bin/python gemv_bi_test.py            # control
"""
import os
import time

import mlx.core as mx
import numpy as np

ON = os.environ.get("MLX_GEMV_BATCH_INVARIANT") == "1"
mx.random.seed(3)

SHAPES = [  # (K, N) — DSv4-relevant + generic
    (4096, 256),    # MoE router gate
    (16384, 24),    # hc fn mixes
    (16384, 4),     # hc_head
    (4096, 4096),   # generic large (the 6e-5 repro shape)
    (512, 1024),
    (1000, 333),    # odd sizes
]
DTYPES = [mx.bfloat16, mx.float16, mx.float32]

n_bitfail = 0
n_corrfail = 0
n_cases = 0
for dt in DTYPES:
    for K, N in SHAPES:
        w = (mx.random.normal((N, K)) * 0.05).astype(dt)   # Linear layout
        wp = (mx.random.normal((K, N)) * 0.05).astype(dt)  # plain layout
        mx.eval(w, wp)
        for M in (2, 3, 4, 8):
            for tag, f in (
                ("xWT", lambda x: x @ w.T),
                ("xW", lambda x: x @ wp),
            ):
                x = (mx.random.normal((M, K)) * 0.5).astype(dt)
                mx.eval(x)
                yb = f(x)
                yr = mx.concatenate([f(x[j : j + 1]) for j in range(M)], axis=0)
                # fp32 reference for correctness
                xf = np.asarray(x.astype(mx.float32))
                wf = np.asarray((w if tag == "xWT" else wp).astype(mx.float32))
                ref = xf @ (wf.T if tag == "xWT" else wf)
                mx.eval(yb, yr)
                ybf = np.asarray(yb.astype(mx.float32))
                n_cases += 1
                atol = {mx.bfloat16: 2e-1, mx.float16: 5e-2, mx.float32: 1e-4}[dt]
                scale = np.abs(ref).max() + 1.0
                if np.abs(ybf - ref).max() > atol * scale:
                    n_corrfail += 1
                    print(
                        f"CORRECTNESS FAIL {dt} K{K} N{N} M{M} {tag}: "
                        f"max|d|={np.abs(ybf - ref).max():.4e}"
                    )
                if not bool(mx.all(yb == yr).item()):
                    n_bitfail += 1
                    d = float(
                        mx.abs(yb.astype(mx.float32) - yr.astype(mx.float32)).max()
                    )
                    if ON:
                        print(f"BITEXACT FAIL {dt} K{K} N{N} M{M} {tag} d={d:.2e}")

# Leading batch dims: (B, M, K) @ (K, N)
for dt in (mx.bfloat16, mx.float32):
    K, N, B, M = 4096, 512, 2, 3
    wp = (mx.random.normal((K, N)) * 0.05).astype(dt)
    x = (mx.random.normal((B, M, K)) * 0.5).astype(dt)
    mx.eval(wp, x)
    yb = x @ wp
    yr = mx.concatenate(
        [
            mx.concatenate([x[b : b + 1, j : j + 1] @ wp for j in range(M)], axis=1)
            for b in range(B)
        ],
        axis=0,
    )
    mx.eval(yb, yr)
    ref = np.asarray(x.astype(mx.float32)) @ np.asarray(wp.astype(mx.float32))
    n_cases += 1
    if np.abs(np.asarray(yb.astype(mx.float32)) - ref).max() > 0.2:
        n_corrfail += 1
        print(f"CORRECTNESS FAIL batched-dims {dt}")
    if not bool(mx.all(yb == yr).item()):
        n_bitfail += 1
        if ON:
            print(f"BITEXACT FAIL batched-dims {dt}")

print(
    f"\nenv={'ON' if ON else 'OFF (control)'}: {n_cases} cases, "
    f"correctness fails={n_corrfail}, bitexact fails={n_bitfail}"
)
if ON:
    print("EXPECT: 0 / 0" )
else:
    print("EXPECT: correctness 0, bitexact > 0 (drift detectable)")

# Perf: worst case (large shape) + DSv4 shapes
def bench(f, iters=200):
    for _ in range(20):
        mx.eval(f())
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        y = f()
    mx.eval(y)
    mx.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6

print("\nperf (us/call, M=3 bf16):")
for K, N in ((4096, 4096), (4096, 256), (16384, 24)):
    w = (mx.random.normal((N, K)) * 0.05).astype(mx.bfloat16)
    x = (mx.random.normal((3, K)) * 0.5).astype(mx.bfloat16)
    mx.eval(w, x)
    print(f"  K{K} N{N}: {bench(lambda: x @ w.T):.1f}")
