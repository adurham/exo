#!/usr/bin/env python3
"""p44 -- full-scale EXL3 layer: all 384 experts, real serving geometry.

Closes the last derived gap in the Phase-11 Stage-1 arithmetic:
  - the mxfp4 baseline (p20) was measured at E=384,
  - the EXL3/prod ratios (p2) at E=128,
  - the stage-1 doc multiplied them as if per-layer cost were E-independent.

This measures EXL3 on layer 1 of the real checkpoint at:
  A) E=128  (control, matches the p2 ratio measurement)
  B) E=384  (full layer, single-node geometry)
  C) E=384, rank=0/world=2 (the 2-node serving geometry: half intermediate)
and times R=1 (decode), R=4 (MTP gamma=3 verify), R=6 (gamma=5 verify).

Also records load time + resident bytes per variant, and vm_stat before/after.
"""
import gc
import os
import sys
import time

import numpy as np
import mlx.core as mx

HOME = os.path.expanduser("~")
MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
sys.path.insert(0, HOME + "/exl3-vendor-test")
os.environ.setdefault("EXL3_MM_MAX_ROWS", "100000")

from exl3.loader import Exl3Checkpoint, load_experts  # noqa: E402

LAYER = 1
K_SEL = 6
ACT = "silu_clamp"


def vm():
    out = os.popen("vm_stat").read()
    d = {}
    for line in out.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            v = v.strip().rstrip(".")
            if v.isdigit():
                d[k.strip()] = int(v)
    page = 16384
    free = (d.get("Pages free", 0) + d.get("Pages inactive", 0)) * page / 2**30
    return f"free+inactive={free:.1f} GiB"


def bytes_of(m):
    tot = 0
    for a in (m._gu_trellis, m._gu_suh, m._gu_svh, m._dn_trellis, m._dn_suh, m._dn_svh):
        tot += int(np.prod(a.shape)) * a.itemsize
    return tot


def bench(m, R, reps, warm=8):
    x = mx.array(np.linspace(-0.4, 0.4, R * m.input_dims, dtype=np.float16)
                 ).reshape(1, R, m.input_dims)
    ind = mx.array(np.stack([np.random.RandomState(5 + R).choice(m.num_experts, K_SEL,
                                                                 replace=False)
                             for _ in range(R)]).reshape(1, R, K_SEL).astype(np.int32))
    for _ in range(warm):
        mx.eval(m(x, ind))
    mx.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(m(x, ind))
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2], ts[0]


def drop(m):
    del m
    gc.collect()
    mx.clear_cache()


ckpt = Exl3Checkpoint(MODEL)
print(f"[p44] node vm before: {vm()}", flush=True)

variants = [
    ("A E=128 single  ", dict(n_experts=128)),
    ("B E=384 single  ", dict(n_experts=None)),
    ("C E=384 r0/w2  ", dict(n_experts=None, rank=0, world=2)),
]
results = {}
for name, kw in variants:
    t0 = time.time()
    m = load_experts(ckpt, LAYER, activation=ACT, **kw)
    tl = time.time() - t0
    nb = bytes_of(m)
    print(f"[p44] {name} loaded in {tl:.1f}s  bytes={nb/2**20:.0f} MiB  "
          f"H={m.hidden_dims} D={m.input_dims} E={m.num_experts}  vm={vm()}", flush=True)
    row = {}
    for R, reps in ((1, 150), (4, 80), (6, 50)):
        med, mn = bench(m, R, reps)
        row[R] = (med, mn)
        print(f"[p44] {name} R={R}: median={med:.3f} ms  min={mn:.3f} ms", flush=True)
    results[name] = (tl, nb, row)
    drop(m)

print(flush=True)
print("[p44] === implied 40-layer MoE time (ms/token or ms/cycle) ===")
for name, (tl, nb, row) in results.items():
    parts = " ".join(f"R{R}={row[R][0]*40:.1f}" for R in (1, 4, 6))
    print(f"  {name}  load={tl:.1f}s bytes={nb/2**20:.0f}MiB  {parts}")

# ratio view vs the E=128 control (same tensors)
a, b, c = (results[n][2] for n, _ in variants)
print()
print("[p44] ratios vs A(E=128):")
for R in (1, 4, 6):
    print(f"  R={R}: B/A={b[R][0]/a[R][0]:.3f}  C/A={c[R][0]/a[R][0]:.3f}")

print()
print("[p44] production mxfp4 reference (p20, E=384): R=1 0.388 / R=4 1.051 / R=6 1.483 ms")
for R in (1, 4, 6):
    p20 = {1: 0.388, 4: 1.051, 6: 1.483}[R]
    print(f"  R={R}: B ratio vs prod = {b[R][0]/p20:.3f}  (stage-1 doc assumed 1.909/1.979/2.048)")
print("[p44] DONE", flush=True)
