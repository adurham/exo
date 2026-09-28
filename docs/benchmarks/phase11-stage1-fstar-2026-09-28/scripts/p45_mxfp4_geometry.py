#!/usr/bin/env python3
"""p45 -- mxfp4 (production format) at serving geometry: full vs half width.

Companion to p44. The phase-11 Stage-1 arithmetic prices EXL3 from single-rank
FULL-width per-layer costs x 40 layers; the build serves a 2-way split
(phase-12 recipe: both ranks hold all experts at half intermediate width).
This measures the production format (mxfp4 4/32) at both geometries so the
like-for-like ratio can be read at the geometry that actually serves.

Arms (layer 1, E=384, V4.1 dims):
  FULL   mxfp4 full width             (control vs p20: expect 0.388/1.051/1.483)
  H0/H1  mxfp4 rank-0 / rank-1 slice at world=2 (half intermediate width)

Checks:
  rank-sum: y0 + y1 == y_full (fp16 rounding) at R=1 and R=4
  H0/F cost ratio; x40L projections against p44's EXL3 C arm

Build follows p20_xaccess_probe.py (Cheap(QuantizedSwitchLinear) +
CheapSwitch(SwitchGLU); weights via EXL3 reconstruct -> mx.quantize mxfp4
gs=32). Slice convention matches the vendored loader's _slice_intermediate:
gate/up on the OUT (intermediate) axis, down on the IN axis, first half for
rank 0.
"""
import gc
import os
import sys
import time

import numpy as np
import mlx.core as mx
import mlx.nn as nn

HOME = os.path.expanduser("~")
MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
sys.path.insert(0, HOME + "/exl3-vendor-test")
os.environ.setdefault("EXL3_MM_MAX_ROWS", "100000")

from exl3.loader import Exl3Checkpoint, load_dense_layer  # noqa: E402
from exl3.reconstruct import reconstruct_public_mlx  # noqa: E402
from mlx_lm.models.switch_layers import (  # noqa: E402
    QuantizedSwitchLinear, SwiGLU, SwitchGLU)

LAYER = 1
E = 384
HID, INT = 5120, 2304
KK = 6
WORLD = 2
HS = INT // WORLD

ckpt = Exl3Checkpoint(MODEL)
pre = f"layers.{LAYER}.ffn.experts."


def deq(prefix):
    """Expert projection as [out, in] fp16 (loader convention: no trailing dot)."""
    lay = load_dense_layer(ckpt, prefix)
    return np.array(reconstruct_public_mlx(lay)).T.astype(np.float16)


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


class Cheap(QuantizedSwitchLinear):
    def __init__(self, W, bits, gs, mode):
        nn.Module.__init__(self)
        self.group_size, self.bits, self.mode = gs, bits, mode
        q = mx.quantize(mx.array(W), group_size=gs, bits=bits, mode=mode)
        self.weight, self.scales = q[0], q[1]
        if len(q) > 2:
            self.biases = q[2]
        mx.eval(self.weight, self.scales)


class CheapSwitch(SwitchGLU):
    def __init__(self, mods):
        nn.Module.__init__(self)
        self.gate_proj, self.up_proj, self.down_proj = (mods["gate"], mods["up"],
                                                        mods["down"])
        self.activation = SwiGLU()


print(f"[p45] checkpoint: {MODEL}", flush=True)
print(f"[p45] vm before: {vm()}", flush=True)
t_all = time.time()
FULL, H0, H1 = {}, {}, {}
for nm, pr in (("gate", "w1"), ("up", "w3"), ("down", "w2")):
    exp = (INT, HID) if nm != "down" else (HID, INT)
    full_acc = np.empty((E,) + exp, dtype=np.float16)
    if nm == "down":
        h_accs = [np.empty((E, HID, HS), dtype=np.float16) for _ in range(WORLD)]
    else:
        h_accs = [np.empty((E, HS, HID), dtype=np.float16) for _ in range(WORLD)]
    for e in range(E):
        a = deq(f"{pre}{e}.{pr}")
        if e == 0:
            assert a.shape == exp, f"{nm}: e0 shape {a.shape} != {exp}"
        full_acc[e] = a
        if nm == "down":
            for r in range(WORLD):
                h_accs[r][e] = a[:, r * HS:(r + 1) * HS]
        else:
            for r in range(WORLD):
                h_accs[r][e] = a[r * HS:(r + 1) * HS, :]
    FULL[nm] = Cheap(full_acc, 4, 32, "mxfp4")
    del full_acc
    gc.collect()
    mx.clear_cache()
    H0[nm] = Cheap(h_accs[0], 4, 32, "mxfp4")
    H1[nm] = Cheap(h_accs[1], 4, 32, "mxfp4")
    del h_accs
    gc.collect()
    mx.clear_cache()
    print(f"[p45] {nm} built (full + 2 halves)  vm={vm()}", flush=True)

FULLM = CheapSwitch(FULL)
H0M = CheapSwitch(H0)
H1M = CheapSwitch(H1)
print(f"[p45] all modules built in {time.time() - t_all:.1f}s  vm={vm()}", flush=True)


def bench(fn, x, ind, reps, warm=8):
    for _ in range(warm):
        mx.eval(fn(x, ind))
    mx.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(fn(x, ind))
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2], ts[0]


np.random.seed(777)
mx.random.seed(777)
res = {}
for R, reps in ((1, 200), (4, 100), (6, 80)):
    x = (mx.random.normal((1, R, HID)).astype(mx.float16) * 0.1)
    ind = mx.array(np.stack([np.random.choice(E, KK, replace=False)
                             for _ in range(R)]).reshape(1, R, KK).astype(np.int32))
    mx.eval(x, ind)
    if R in (1, 4):
        yF = FULLM(x, ind)
        y0 = H0M(x, ind)
        y1 = H1M(x, ind)
        mx.eval(yF, y0, y1)
        a = np.array(yF, np.float32)
        b = np.array(y0 + y1, np.float32)
        maxd = float(np.abs(a - b).max())
        cos = float((a.ravel() @ b.ravel()) / (np.linalg.norm(a) * np.linalg.norm(b)))
        tag = "OK" if cos > 0.9999 else "BAD"
        print(f"[p45] rank-sum R={R}: [{tag}] max|diff|={maxd:.4g} "
              f"cos={cos:.7f}", flush=True)
    tF, minF = bench(FULLM, x, ind, reps)
    tH, _ = bench(H0M, x, ind, reps)
    res[R] = (tF, tH)
    print(f"[p45] R={R}: FULL median={tF:.3f} (min {minF:.3f})  "
          f"H0 median={tH:.3f}  H0/F={tH / tF:.3f}", flush=True)

print()
print("[p45] control F vs p20 (full width): expected 0.388 / 1.051 / 1.483")
for R in (1, 4, 6):
    exp_ = {1: 0.388, 4: 1.051, 6: 1.483}[R]
    got = res[R][0]
    print(f"  R={R}: got {got:.3f} vs p20 {exp_:.3f}  ratio={got / exp_:.3f}")

print()
print("[p45] x40L projections (per-rank wall time at serving geometry):")
for R in (1, 4, 6):
    f, h = res[R]
    print(f"  R={R}: FULL x40 = {f * 40:.1f} ms   H0 x40 = {h * 40:.1f} ms")

print()
print("[p45] serving geometry: EXL3 (p44 C: 0.430/1.040/1.459) vs mxfp4 H0:")
for R in (1, 4, 6):
    ex = {1: 0.430, 4: 1.040, 6: 1.459}[R]
    h = res[R][1]
    print(f"  R={R}: EXL3 {ex:.3f} / mxfp4H {h:.3f} = {ex / h:.3f}x"
          f"   (full-width adopted: 1.909/1.979/2.048)")
print("[p45] DONE", flush=True)
