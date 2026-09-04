#!/usr/bin/env python3
"""Prove WHICH Metal kernel each bit width dispatches, via the JIT cache.

MLX compiles each quantized kernel variant on demand and names it
"<mode>_<func>_<type>_gs_<gs>_b_<bits>". The compiled-kernel names are
observable in MLX's on-disk Metal library cache: run a gather_qmm at a
given bit width in a FRESH process with an empty cache dir, then list
what got compiled. Whatever appears IS the kernel that ran -- this does
not depend on reading the C++ dispatch correctly.
"""
import os
import shutil
import subprocess
import sys

bits = int(sys.argv[1])
cache = f"/tmp/i11_mlxcache_b{bits}"
shutil.rmtree(cache, ignore_errors=True)
os.makedirs(cache, exist_ok=True)

prog = f'''
import mlx.core as mx, numpy as np
HIDDEN, INTER, E, TOPK, M, gs, bits = 4096, 1024, 8, 6, 4, 32, {bits}
scale = (1.0/HIDDEN) ** 0.5
mx.random.seed(0)
def q(o, i):
    s = mx.random.uniform(low=-scale, high=scale, shape=(E, o, i)).astype(mx.bfloat16)
    mx.eval(s); p = mx.quantize(s, group_size=gs, bits=bits, mode="affine"); mx.eval(p); return p
gate, up, down = q(INTER, HIDDEN), q(INTER, HIDDEN), q(HIDDEN, INTER)
x = mx.random.normal(shape=(M, HIDDEN)).astype(mx.bfloat16)
rng = np.random.default_rng(0)
idx = mx.array(rng.integers(0, E, size=(M, TOPK)).astype(np.uint32))
mx.eval(x, idx)
xe = mx.expand_dims(x, (-2, -3))
def gq(p, inp):
    return mx.gather_qmm(inp, p[0], p[1], p[2] if len(p) > 2 else None,
        rhs_indices=idx, transpose=True, group_size=gs, bits=bits,
        mode="affine", sorted_indices=False)
a = gq(up, xe); g = gq(gate, xe)
o = gq(down, mx.sigmoid(g) * g * a)
mx.eval(o)
print("ran bits={bits} ok, out", o.shape)
'''
env = dict(os.environ, MLX_METAL_CACHE_DIR=cache, MLX_METAL_KERNEL_CACHE=cache)
r = subprocess.run([sys.executable, "-c", prog], env=env, capture_output=True, text=True)
print(r.stdout.strip() or r.stderr.strip()[-400:])

found = set()
for root, _, files in os.walk(cache):
    for f in files:
        p = os.path.join(root, f)
        try:
            raw = open(p, "rb").read()
        except Exception:
            continue
        out = subprocess.run(["strings"], input=raw, capture_output=True).stdout.decode(errors="ignore")
        for tok in out.split():
            if "gather_q" in tok and "_b_" in tok:
                found.add(tok)
print(f"bits={bits} compiled gather kernels in cache:")
for n in sorted(found):
    print("   ", n)
if not found:
    print("    (cache dir empty -- this mlx build may not use an on-disk kernel cache)")
