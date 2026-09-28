#!/usr/bin/env python3
"""p43 -- loader.py gate: checkpoint -> module via mlx_lm.models.exl3.loader.

C1: loader.load_experts(E) == hand-built (p2-style) stacked module -- bit-identical.
C2: loader rank slices: sum(world=2 partials) == full (loader).
C3: loader.load_dense_linear(head) == reconstruct_public_mlx path on 2 R values.
"""
import os
import sys

import numpy as np
import mlx.core as mx

HOME = os.path.expanduser("~")
MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
sys.path.insert(0, HOME + "/exl3-vendor-test")   # the vendored tree under test
import exl3  # noqa: E402
from exl3.loader import Exl3Checkpoint, load_experts, load_dense_layer, load_dense_linear  # noqa: E402
from exl3.ref.codebook import codebook_mode_from_flags  # noqa: E402
from exl3.ref.layer import EXL3Layer  # noqa: E402
from exl3.reconstruct import reconstruct_public_mlx  # noqa: E402
from exl3.exl3_moe import EXL3SwitchGLU  # noqa: E402

LAYER = int(os.environ.get("BENCH_LAYER", "1"))
E = int(os.environ.get("BENCH_EXPERTS", "16"))
K_SEL = 6
ACT = "silu_clamp"

fails = []

ckpt = Exl3Checkpoint(MODEL)
print(f"== checkpoint open: {ckpt.n_experts(LAYER)} experts in layer {LAYER} ==", flush=True)

print("== C1: hand-built reference vs loader (E=%d) ==" % E, flush=True)
# hand-built (p2-style): read raw arrays, stack by hand, construct directly
pre = f"layers.{LAYER}.ffn.experts."
gu_l, up_l, dn_l, gs_l, gv_l, ds_l, dv_l = [], [], [], [], [], [], []
for e in range(E):
    p = pre + f"{e}."
    gu_l.append(ckpt.np(p + "w1.trellis"))
    up_l.append(ckpt.np(p + "w3.trellis"))
    dn_l.append(ckpt.np(p + "w2.trellis"))
    gs_l.append(np.stack([ckpt.np(p + "w1.suh"), ckpt.np(p + "w3.suh")]))
    gv_l.append(np.concatenate([ckpt.np(p + "w1.svh"), ckpt.np(p + "w3.svh")]))
    ds_l.append(ckpt.np(p + "w2.suh"))
    dv_l.append(ckpt.np(p + "w2.svh"))

GU_T = np.concatenate(gu_l + up_l, axis=1)
GU_SUH = np.stack(gs_l)
GU_SVH = np.stack(gv_l)
DN_T = np.concatenate(dn_l, axis=1)
DN_SUH = np.stack(ds_l)
DN_SVH = np.stack(dv_l)
K = int(GU_T.shape[2] * 16 // 256)
CB = codebook_mode_from_flags(mcg=False, mul1=True)

REF = EXL3SwitchGLU(
    gu_trellis=mx.array(GU_T).view(mx.uint16), gu_suh=mx.array(GU_SUH),
    gu_svh=mx.array(GU_SVH), dn_trellis=mx.array(DN_T).view(mx.uint16),
    dn_suh=mx.array(DN_SUH), dn_svh=mx.array(DN_SVH),
    k=K, cb=CB, activation=ACT)
LOADED = load_experts(ckpt, LAYER, n_experts=E, activation=ACT)
mx.eval(REF._gu_trellis, LOADED._gu_trellis)

D_ALL = LOADED.input_dims
for R in (1, 4, 8):
    x = mx.array(np.linspace(-0.4, 0.4, R * D_ALL, dtype=np.float16)).reshape(1, R, D_ALL)
    ind = mx.array(np.stack([np.random.RandomState(11 + R).choice(E, K_SEL, replace=False)
                             for _ in range(R)]).reshape(1, R, K_SEL).astype(np.int32))
    y_ref = REF(x, ind)
    y_new = LOADED(x, ind)
    mx.eval(y_ref, y_new)
    eq = bool(mx.array_equal(y_ref, y_new))
    print(f"  [{'PASS' if eq else 'FAIL'}] C1 R={R}: bit-equal={eq}", flush=True)
    if not eq:
        a, b = np.array(y_ref, np.float32), np.array(y_new, np.float32)
        print("      max|diff| =", float(np.abs(a - b).max()), flush=True)
        fails.append(f"C1-R{R}")

print("== C2: rank slices (world=2) vs full (loader) ==", flush=True)
FULL = load_experts(ckpt, LAYER, n_experts=E, activation=ACT)
R0 = load_experts(ckpt, LAYER, n_experts=E, rank=0, world=2, activation=ACT)
R1 = load_experts(ckpt, LAYER, n_experts=E, rank=1, world=2, activation=ACT)
mx.eval(FULL._gu_trellis, R0._gu_trellis, R1._gu_trellis)
assert R0.hidden_dims + R1.hidden_dims == FULL.hidden_dims
for R in (1, 4):
    x = mx.array(np.linspace(-0.4, 0.4, R * D_ALL, dtype=np.float16)).reshape(1, R, D_ALL)
    ind = mx.array(np.stack([np.random.RandomState(23 + R).choice(E, K_SEL, replace=False)
                             for _ in range(R)]).reshape(1, R, K_SEL).astype(np.int32))
    yf = FULL(x, ind)
    ys = R0(x, ind) + R1(x, ind)
    mx.eval(yf, ys)
    a, b = np.array(yf, np.float32), np.array(ys, np.float32)
    maxd = float(np.abs(a - b).max())
    cos = float((a.ravel() @ b.ravel()) / (np.linalg.norm(a) * np.linalg.norm(b)))
    ok = cos > 0.99995 and maxd <= max(1e-2, 1e-3 * float(np.abs(a).max()))
    print(f"  [{'PASS' if ok else 'FAIL'}] C2 R={R}: max|diff|={maxd:.4g} cos={cos:.7f}", flush=True)
    if not ok:
        fails.append(f"C2-R{R}")

print("== C3: dense path (head group) ==", flush=True)
lay = load_dense_layer(ckpt, "head")
print(f"  head layer: in={lay.in_features} out={lay.out_features} k={lay.k}", flush=True)
lin = load_dense_linear(ckpt, "head")
ref_w = reconstruct_public_mlx(EXL3Layer(
    key="head", in_features=lay.in_features, out_features=lay.out_features, k=lay.k,
    trellis=lay.trellis, suh=lay.suh, svh=lay.svh, mul1=True))
mx.eval(ref_w)
for R in (1, 4):
    x = mx.array(np.linspace(-0.4, 0.4, R * lay.in_features, dtype=np.float16)
                 ).reshape(1, R, lay.in_features)
    y_lin = lin(x)
    y_ref = (x.astype(mx.float16) @ ref_w).astype(mx.float16)
    mx.eval(y_lin, y_ref)
    a, b = np.array(y_lin, np.float32), np.array(y_ref, np.float32)
    maxd = float(np.abs(a - b).max())
    cos = float((a.ravel() @ b.ravel()) / (np.linalg.norm(a) * np.linalg.norm(b)))
    ok = cos > 0.9999 and maxd <= max(5e-2, 2e-3 * float(np.abs(a).max()))
    print(f"  [{'PASS' if ok else 'FAIL'}] C3 R={R}: max|diff|={maxd:.4g} cos={cos:.7f}", flush=True)
    if not ok:
        fails.append(f"C3-R{R}")

print(flush=True)
if fails:
    print("VERDICT: FAIL", fails)
    sys.exit(1)
print("VERDICT: PASS -- loader gates C1/C2/C3 all green")
