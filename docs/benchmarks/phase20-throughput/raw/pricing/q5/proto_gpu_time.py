#!/usr/bin/env python3
"""Q5 prototype: verify MLX_GPU_TIME per-command-buffer GPU-time capture on
the node build (mlx 0.32.3.dev20260918+603f16eb7, M4 Max).

Mechanism under test (from local mlx source HEAD ac73d0c):
  mlx/backend/metal/eval.cpp:94-109  accumulate_gpu_time_if_enabled()
    reads cbuf->GPUStartTime()/GPUEndTime() in completion handler
  mlx/backend/metal/device.cpp:896  gpu_time_enabled() gated on MLX_GPU_TIME
  python/src/metal.cpp:118          mx.metal.gpu_time_ns() / reset_gpu_time()

Goal: show per-command-buffer GPU ms for isolated kernels. Not per-kernel
name attribution (no such API) — that is the finding.
"""
import os, statistics, time

# Must be set BEFORE importing mlx (gpu_time_enabled() caches on first call).
assert os.environ.get("MLX_GPU_TIME") == "1", "run with MLX_GPU_TIME=1"
import mlx.core as mx

print("mlx:", mx.__version__)
print("gpu_time_enabled exposed:", hasattr(mx.metal, "gpu_time_ns"),
      hasattr(mx.metal, "reset_gpu_time"), hasattr(mx.metal, "dispatch_count"))

# --- sanity: is the counter actually wired (nonzero) on THIS build? ---
mx.metal.reset_gpu_time()
x = mx.random.normal((1024, 1024), dtype=mx.float16)
w = mx.random.normal((1024, 1024), dtype=mx.float16)
mx.eval(x, w)
mx.synchronize()
mx.metal.reset_gpu_time()
y = x @ w
mx.eval(y)
mx.synchronize()
wired_ns = mx.metal.gpu_time_ns()
print(f"WIRED_CHECK gpu_time_ns after one 1024^3 fp16 matmul = {wired_ns} ns "
      f"({wired_ns/1e3:.1f} us)  -> {'EXPOSED' if wired_ns else 'STILL 0 / NOT WIRED'}")


def gpu_ms(fn, n=20, w=5):
    """Per-command-buffer GPU ms: reset, build+eval (drains via synchronize),
    read accumulated GPUEndTime-GPUStartTime. Matches item2_opclass_breakdown.py
    methodology but drains with mx.synchronize() so the completion handler runs."""
    for _ in range(w):
        mx.eval(fn()); mx.synchronize()
    samples = []
    for _ in range(n):
        mx.metal.reset_gpu_time()
        mx.eval(fn())
        mx.synchronize()
        samples.append(mx.metal.gpu_time_ns())
    return statistics.median(samples) / 1e6  # ns -> ms


def wall_ms(fn, n=20, w=5):
    for _ in range(w):
        mx.eval(fn()); mx.synchronize()
    samples = []
    for _ in range(n):
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


# --- several isolated kernels, distinct shapes so we can see them differ ---
cases = {
    "matmul_1024x1024x1024_fp16": lambda: mx.random.normal((1024, 1024), dtype=mx.float16) @ w,
    "matmul_4096x4096x4096_fp16": lambda: mx.random.normal((4096, 4096), dtype=mx.float16)
        @ mx.random.normal((4096, 4096), dtype=mx.float16),
    "elementwise_8M_fp16": lambda: mx.random.normal((8_000_000,), dtype=mx.float16) * 2.0,
    "quantized_matmul_mxfp4": None,  # filled below
}
# mxfp4 quantized matmul (matches the EXL3/MoE family class)
wq, sc = mx.quantize(mx.random.normal((1024, 4096), dtype=mx.float32),
                     group_size=32, bits=4, mode="mxfp4")
mx.eval(wq, sc)
xh = mx.random.normal((64, 4096), dtype=mx.bfloat16)
mx.eval(xh)
cases["quantized_matmul_mxfp4_64x1024x4096"] = (
    lambda: mx.quantized_matmul(xh, wq, sc, transpose=True, group_size=32,
                                bits=4, mode="mxfp4"))

print("\n%-38s %10s %10s %8s" % ("case", "gpu_ms", "wall_ms", "gpu/wall"))
results = {}
for name, fn in cases.items():
    if fn is None:
        continue
    g = gpu_ms(fn)
    wl = wall_ms(fn)
    results[name] = {"gpu_ms": g, "wall_ms": wl, "ratio": g / wl if wl else 0}
    print("%-38s %10.3f %10.3f %8.3f" % (name, g, wl, g / wl if wl else 0))

# --- dispatch count cross-check (same build exposes it) ---
mx.metal.reset_dispatch_count()
mx.eval(cases["matmul_1024x1024x1024_fp16"]())
mx.synchronize()
print("\ndispatch_count for one 1024^3 matmul =", mx.metal.dispatch_count())

import json
out = {"mlx": mx.__version__, "wired_ns": wired_ns, "results": results,
       "dispatch_count_1024matmul": int(mx.metal.dispatch_count())}
with open("/tmp/q5_proto.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nwrote /tmp/q5_proto.json")
