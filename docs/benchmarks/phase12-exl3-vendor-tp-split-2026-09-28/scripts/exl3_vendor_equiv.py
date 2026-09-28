#!/usr/bin/env python3
"""Vendored-exl3 equivalence test.

Gate: the vendored tree (`~/exl3-vendor-test/exl3`, import-rewrite + headers vs
upstream PonyExl3) must produce BIT-IDENTICAL outputs to upstream ponyexl3 on
real DeepSeek-V4.1 EXL3 checkpoint tensors, across:
  - EXL3SwitchGLU forward (R=1/4/8) for activation "silu" and "silu_clamp"
  - EXL3Linear forward on the quantized head group (k=6)
  - reconstruct_public_mlx on one expert w1 group
Deterministic inputs (no RNG in x; fixed-seed index draw).
"""
import hashlib
import json
import os
import sys

import numpy as np
import mlx.core as mx

MODEL = os.path.expanduser(
    "~/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw")
LAYER = int(os.environ.get("BENCH_LAYER", "1"))
E = int(os.environ.get("BENCH_EXPERTS", "16"))
KK = 6
HID = 5120

sys.path.insert(0, os.path.expanduser("~/exl3-vendor-test"))
import exl3 as V  # vendored package (import surface: EXL3Linear, EXL3SwitchGLU)
from exl3.ref.codebook import codebook_mode_from_flags as V_CBMODE
from exl3.ref.layer import EXL3Layer as V_Layer
from exl3.reconstruct import reconstruct_public_mlx as V_REC

from ponyexl3.mlx.exl3_moe import EXL3SwitchGLU as O_Switch
from ponyexl3.mlx.exl3_linear import EXL3Linear as O_Linear
from ponyexl3.ref.codebook import codebook_mode_from_flags as O_CBMODE
from ponyexl3.ref.layer import EXL3Layer as O_Layer
from ponyexl3.mlx.reconstruct import reconstruct_public_mlx as O_REC

fails = []

def md5_of(arr):
    a = np.array(arr)
    return hashlib.md5(a.tobytes()).hexdigest(), str(a.dtype), tuple(a.shape), int(np.isnan(a).sum())

def cmp_pair(name, a, b):
    ma = md5_of(a); mb = md5_of(b)
    eq = bool(mx.array_equal(a, b))
    ok = eq and ma[0] == mb[0]
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: equal={eq} md5 {ma[0]} vs {mb[0]} "
          f"dtype={ma[1]} shape={ma[2]} nan={ma[3]}/{mb[3]}")
    if not ok:
        d = np.abs(np.array(a, copy=False).astype(np.float32) - np.array(b, copy=False).astype(np.float32))
        print(f"         max|diff| = {float(d.max())}")
        fails.append(name)
    return ok

print(f"== loading E={E} experts of layer {LAYER} ==")
idx = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))["weight_map"]
SUF = ("w1.trellis", "w1.suh", "w1.svh", "w2.trellis", "w2.suh", "w2.svh",
       "w3.trellis", "w3.suh", "w3.svh")
by_shard = {}
for e in range(E):
    for s in SUF:
        k = f"layers.{LAYER}.ffn.experts.{e}.{s}"
        by_shard.setdefault(idx[k], []).append(k)

from safetensors import safe_open
tens = {}
for s, ks in sorted(by_shard.items()):
    with safe_open(os.path.join(MODEL, s), framework="np") as f:
        for k in ks:
            tens[k] = f.get_tensor(k)
print(f"  loaded {len(tens)} tensors")

CB_V = V_CBMODE(mcg=False, mul1=True)
CB_O = O_CBMODE(mcg=False, mul1=True)

gu, up, gs_, gv, dn, ds, dv = [], [], [], [], [], [], []
for e in range(E):
    p = f"layers.{LAYER}.ffn.experts.{e}."
    gu.append(mx.array(tens[p + "w1.trellis"]))
    up.append(mx.array(tens[p + "w3.trellis"]))
    gs_.append(mx.stack([mx.array(tens[p + "w1.suh"]), mx.array(tens[p + "w3.suh"])]))
    gv.append(mx.concatenate([mx.array(tens[p + "w1.svh"]), mx.array(tens[p + "w3.svh"])]))
    dn.append(mx.array(tens[p + "w2.trellis"]))
    ds.append(mx.array(tens[p + "w2.suh"]))
    dv.append(mx.array(tens[p + "w2.svh"]))

GU_T = mx.concatenate(gu + up, axis=1).view(mx.uint16)
GU_SUH = mx.stack(gs_).astype(mx.float16)
GU_SVH = mx.stack(gv).astype(mx.float16)
DN_T = mx.concatenate(dn, axis=1).view(mx.uint16)
DN_SUH = mx.stack(ds).astype(mx.float16)
DN_SVH = mx.stack(dv).astype(mx.float16)
K_BITS = int(tens[f"layers.{LAYER}.ffn.experts.0.w1.trellis"].shape[2] * 16 // 256)
print(f"  K_BITS = {K_BITS}")

mx.eval(GU_T, GU_SUH, GU_SVH, DN_T, DN_SUH, DN_SVH)

print()
print("== EXL3SwitchGLU: vendored vs upstream ==")
for act in ("silu", "silu_clamp"):
    vmod = V.EXL3SwitchGLU(gu_trellis=GU_T, gu_suh=GU_SUH, gu_svh=GU_SVH,
                           dn_trellis=DN_T, dn_suh=DN_SUH, dn_svh=DN_SVH,
                           k=K_BITS, cb=CB_V, activation=act)
    omod = O_Switch(gu_trellis=GU_T, gu_suh=GU_SUH, gu_svh=GU_SVH,
                    dn_trellis=DN_T, dn_suh=DN_SUH, dn_svh=DN_SVH,
                    k=K_BITS, cb=CB_O, activation=act)
    for R in (1, 4, 8):
        x = mx.array(np.linspace(-0.5, 0.5, R * HID, dtype=np.float16)).reshape(1, R, HID)
        ind = mx.array(np.stack([np.random.RandomState(42 + R).choice(E, KK, replace=False)
                                 for _ in range(R)]).reshape(1, R, KK).astype(np.int32))
        yv = vmod(x, ind)
        yo = omod(x, ind)
        mx.eval(yv, yo)
        cmp_pair(f"switch act={act} R={R}", yv, yo)

print()
print("== EXL3Linear (head group, k=6): vendored vs upstream ==")
HEAD = "head."
ht = {s: tens.get(HEAD + s) for s in ("trellis", "suh", "svh")}
if ht["trellis"] is None:
    # fall back: first dense .trellis key that has companions
    for k in sorted(idx):
        if k.endswith(".trellis") and "experts" not in k:
            base = k[:-len("trellis")]
            if (base + "suh") in idx and (base + "svh") in idx:
                HEAD = base
                ht = {s: None for s in ("trellis", "suh", "svh")}
                for s in ht:
                    with safe_open(os.path.join(MODEL, idx[base + s]), framework="np") as f:
                        ht[s] = f.get_tensor(base + s)
                break
print(f"  group = {HEAD} shapes: " + ", ".join(f"{s}{ht[s].shape}" for s in ht))
hk = int(ht["trellis"].shape[2] * 16 // 256)
vlay = V_Layer(key=HEAD, in_features=len(ht["suh"]), out_features=len(ht["svh"]),
               k=hk, trellis=ht["trellis"], suh=ht["suh"], svh=ht["svh"], mul1=True)
olay = O_Layer(key=HEAD, in_features=len(ht["suh"]), out_features=len(ht["svh"]),
               k=hk, trellis=ht["trellis"], suh=ht["suh"], svh=ht["svh"], mul1=True)
vlin = V.EXL3Linear(vlay)
olin = O_Linear(olay)
for R in (1, 4):
    x = mx.array(np.linspace(-0.5, 0.5, R * len(ht["suh"]), dtype=np.float16)
                 ).reshape(1, R, len(ht["suh"]))
    yv = vlin(x)
    yo = olin(x)
    mx.eval(yv, yo)
    cmp_pair(f"linear head R={R}", yv, yo)

print()
print("== reconstruct_public_mlx (expert w1 of expert 0): vendored vs upstream ==")
p0 = f"layers.{LAYER}.ffn.experts.0.w1."
vl0 = V_Layer(key=p0, in_features=len(tens[p0 + "suh"]), out_features=len(tens[p0 + "svh"]),
              k=K_BITS, trellis=tens[p0 + "trellis"], suh=tens[p0 + "suh"],
              svh=tens[p0 + "svh"], mul1=True)
ol0 = O_Layer(key=p0, in_features=len(tens[p0 + "suh"]), out_features=len(tens[p0 + "svh"]),
              k=K_BITS, trellis=tens[p0 + "trellis"], suh=tens[p0 + "suh"],
              svh=tens[p0 + "svh"], mul1=True)
wv = V_REC(vl0)
wo = O_REC(ol0)
mx.eval(wv, wo)
cmp_pair("reconstruct w1 expert0", wv, wo)

print()
if fails:
    print(f"VERDICT: FAIL ({len(fails)} mismatches): {fails}")
    sys.exit(1)
print("VERDICT: PASS -- vendored tree is bit-identical to upstream on all tested paths")
