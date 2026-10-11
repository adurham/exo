#!/usr/bin/env python3
"""Sustained GPU load for Task D powermetrics sampling (~40 s dense bf16 matmul loop).

Offline, studio2 only. Prints heartbeat; runs mx.matmul back-to-back for the
requested duration (default 40 s). No model, no generation.
"""
import os, sys, time
os.environ.setdefault("MTL_DISABLE_TIMEOUT", "1")

import mlx.core as mx

SECS = float(sys.argv[1]) if len(sys.argv) > 1 else 40.0
# large enough to keep the GPU busy but memory-light enough for the tight node
T, M, N = 4096, 8192, 8192
a = mx.random.normal((T, M)).astype(mx.bfloat16)
b = mx.random.normal((M, N)).astype(mx.bfloat16)
mx.eval(a, b)
print(f"[load] start {SECS:.0f}s T={T} M={M} N={N}", flush=True)
t0 = time.perf_counter()
n = 0
while time.perf_counter() - t0 < SECS:
    c = mx.matmul(a, b)
    mx.eval(c)
    n += 1
mx.synchronize()
dt = time.perf_counter() - t0
tf = 2 * T * M * N * n / dt / 1e12
print(f"[load] done {n} matmuls in {dt:.1f}s -> {tf:.2f} TF sustained", flush=True)
