#!/usr/bin/env python3
"""Q5 prototype #2: characterise the GRANULARITY + OVERHEAD of the existing
MLX_GPU_TIME per-command-buffer capture, to fix the gap to per-kernel.

Findings targeted:
  A. per-command-buffer != per-kernel: one eval commits ONE buffer that may
     hold many dispatches; the counter sums the whole buffer's GPU span.
  B. per-op attribution IS possible by reset/eval/synchronize per op, but it
     forces a commit+sync per op (changes the overlapped pipeline).
  C. compiled / custom-kernel dispatches share the same CommandEncoder and
     ARE counted (same completion handler at commit).
  D. reading without synchronize undercounts (async).
"""
import os
assert os.environ.get("MLX_GPU_TIME") == "1"
import statistics, time
import mlx.core as mx

MOPS = os.environ.get("MLX_MAX_OPS_PER_BUFFER", "default")
print("mlx:", mx.__version__, " MLX_MAX_OPS_PER_BUFFER=", MOPS)


def one(fn, n=15, w=5):
    for _ in range(w):
        mx.eval(fn()); mx.synchronize()
    g, wl = [], []
    for _ in range(n):
        mx.metal.reset_gpu_time()
        t0 = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        wl.append((time.perf_counter() - t0) * 1e3)
        g.append(mx.metal.gpu_time_ns() / 1e6)
    return statistics.median(g), statistics.median(wl)


A = mx.random.normal((2048, 2048), dtype=mx.float16)
B = mx.random.normal((2048, 2048), dtype=mx.float16)
mx.eval(A, B)


def op1():
    return A @ B
def op2():
    return mx.maximum(A @ B, 0.0)
def op3():
    return mx.exp(A @ B)


print("\n[A/B] per-op attribution + dispatch count per op")
for name, fn in [("single matmul", op1), ("matmul+relu", op2), ("matmul+exp", op3)]:
    mx.metal.reset_dispatch_count()
    g, wl = one(fn)
    dc = mx.metal.dispatch_count()
    print(f"  {name:16s} gpu={g:7.3f}ms wall={wl:7.3f}ms  dispatches={dc}")

# C. compiled path
cf = mx.compile(lambda a, b: mx.maximum(a @ b, 0.0))
mx.eval(cf(A, B))
mx.metal.reset_gpu_time(); mx.eval(cf(A, B)); mx.synchronize()
print(f"\n[C] mx.compile(matmul+relu) gpu={mx.metal.gpu_time_ns()/1e6:.3f}ms "
      f"(counted={'YES' if mx.metal.gpu_time_ns() else 'NO'})")

# D. no-sync read
mx.metal.reset_gpu_time()
r = A @ B; mx.eval(r)
nosync = mx.metal.gpu_time_ns()
mx.synchronize()
withsync = mx.metal.gpu_time_ns()
print(f"\n[D] read-before-sync={nosync} ns  read-after-sync={withsync} ns "
      f"-> {'undercount' if nosync < withsync else 'equal'}")

# B. overhead of per-op sync method vs batched
def batched(k=20):
    outs = [A @ B for _ in range(k)]
    mx.eval(*outs)
mx.metal.reset_dispatch_count()
mx.eval(batched())
mx.synchronize()
dc_batched = mx.metal.dispatch_count()
for _ in range(3): batched(); mx.synchronize()
t0 = time.perf_counter(); mx.eval(batched()); mx.synchronize()
wall_batched = (time.perf_counter()-t0)*1e3
g_single, w_single = one(op1)
print(f"\n[B] 20-op batched: wall={wall_batched:.2f}ms ({wall_batched/20:.3f}ms/op) "
      f"dispatches={dc_batched}")
print(f"    per-op sync method: wall={w_single:.3f}ms/op  gpu={g_single:.3f}ms/op")
print(f"    per-op-sync overhead factor = {w_single/(wall_batched/20):.2f}x")
