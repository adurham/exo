#!/usr/bin/env python3
"""p42 -- TP split-equivalence for EXL3 experts (the plan's "main integration edit").

Question: can EXL3SwitchGLU be tensor-parallel-split across 2 ranks so that
    full(x) == partial_rank0(x) + partial_rank1(x)
(up to fp16 accumulation order), with:
  rank r keeps gate/up OUT-tiles for intermediate H range [r*H/2, (r+1)*H/2)
  rank r keeps dn K-tiles for the same H range
  gu_suh / dn_svh replicated; gu_svh / dn_suh sliced on H.

If yes: decode splits by intermediate width (both ranks hold all experts, half
width) and the MoE wrapper all_sums the partials -- same shape as the V4 TP
convention, no division.

Runs on one node, in-process (no distributed needed): builds the FULL module and
two SLICED modules from the same real checkpoint tensors, compares outputs.
"""
import hashlib
import json
import os
import struct
import sys
import time

import numpy as np
import mlx.core as mx

HOME = os.path.expanduser("~")
MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
sys.path.insert(0, HOME + "/repos/ref/PonyExl3")
os.environ.setdefault("EXL3_MM_MAX_ROWS", "100000")

from ponyexl3.ref.codebook import codebook_mode_from_flags  # noqa: E402
from ponyexl3.mlx.exl3_moe import EXL3SwitchGLU  # noqa: E402

LAYER = int(os.environ.get("BENCH_LAYER", "1"))
E = int(os.environ.get("BENCH_EXPERTS", "16"))
K_SEL = 6
WORLD = 2

_NP = {"F16": np.float16, "F32": np.float32, "I16": np.int16, "I32": np.int32,
       "U8": np.uint8, "I8": np.int8, "U16": np.uint16, "U32": np.uint32, "I64": np.int64}


class ST:
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            self.hdr = json.loads(f.read(n))
        self.base = 8 + n
        self.fd = None

    def np_arr(self, name):
        o = self.hdr[name]["data_offsets"]
        dt = self.hdr[name]["dtype"]
        shp = self.hdr[name]["shape"]
        if self.fd is None:
            self.fd = os.open(self.path, os.O_RDONLY)
        buf = os.pread(self.fd, o[1] - o[0], self.base + o[0])
        if dt == "BF16":
            u = np.frombuffer(buf, np.uint16).astype(np.uint32) << 16
            return u.view(np.float32).reshape(shp)
        return np.frombuffer(buf, _NP[dt]).reshape(shp)


IDX = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))["weight_map"]
_st = {}


def np_t(name):
    p = os.path.join(MODEL, IDX[name])
    if p not in _st:
        _st[p] = ST(p)
    return _st[p].np_arr(name)


print(f"== loading E={E} experts of layer {LAYER} ==", flush=True)
p = f"layers.{LAYER}.ffn.experts."
gu_l, up_l, dn_l, gs_l, gv_l, ds_l, dv_l = [], [], [], [], [], [], []
for e in range(E):
    pre = p + f"{e}."
    gu_l.append(np_t(pre + "w1.trellis"))
    up_l.append(np_t(pre + "w3.trellis"))
    dn_l.append(np_t(pre + "w2.trellis"))
    gs_l.append(np.stack([np_t(pre + "w1.suh"), np_t(pre + "w3.suh")]))
    gv_l.append(np.concatenate([np_t(pre + "w1.svh"), np_t(pre + "w3.svh")]))
    ds_l.append(np_t(pre + "w2.suh"))
    dv_l.append(np_t(pre + "w2.svh"))

GU_T = np.concatenate(gu_l + up_l, axis=1)      # (in_tiles, 2*E*gu_tiles, P)
GU_SUH = np.stack(gs_l).astype(np.float16)      # (E, 2, in)
GU_SVH = np.stack(gv_l).astype(np.float16)      # (E, 2*H)
DN_T = np.concatenate(dn_l, axis=1)             # (hid_tiles, E*out_tiles, P)
DN_SUH = np.stack(ds_l).astype(np.float16)      # (E, H)
DN_SVH = np.stack(dv_l).astype(np.float16)      # (E, in)
K_BITS = int(GU_T.shape[2] * 16 // 256)

in_tiles, gucols, P = GU_T.shape
gu_tiles = gucols // (2 * E)         # tiles per gate (== per up) per expert
hid_tiles, dncols, _ = DN_T.shape
dn_tiles = dncols // E               # out tiles per expert (== D/16)
H_ALL = gu_tiles * 16
D_ALL = dn_tiles * 16
print(f"  K={K_BITS} in_tiles={in_tiles} gu_tiles={gu_tiles} (H={H_ALL}) "
      f"hid_tiles={hid_tiles} dn_tiles={dn_tiles} (D={D_ALL}) P={P}", flush=True)

CB = codebook_mode_from_flags(mcg=False, mul1=True)
ACT = os.environ.get("TRACE_ACT", "silu_clamp")


def build_full():
    return EXL3SwitchGLU(
        gu_trellis=mx.array(GU_T).view(mx.uint16),
        gu_suh=mx.array(GU_SUH),
        gu_svh=mx.array(GU_SVH),
        dn_trellis=mx.array(DN_T).view(mx.uint16),
        dn_suh=mx.array(DN_SUH),
        dn_svh=mx.array(DN_SVH),
        k=K_BITS, cb=CB, activation=ACT)


def slice_buffers(rank, world=WORLD):
    """Contiguous intermediate split. Returns (gu_t, gu_suh, gu_svh, dn_t, dn_suh, dn_svh)."""
    hpt = gu_tiles // world            # gate tiles per rank
    kpt = hid_tiles // world           # dn K tiles per rank
    assert gu_tiles % world == 0 and hid_tiles % world == 0
    t0, t1 = rank * hpt, (rank + 1) * hpt
    k0, k1 = rank * kpt, (rank + 1) * kpt
    gu_v = GU_T.reshape(in_tiles, 2, E, gu_tiles, P)
    gu_s = gu_v[:, :, :, t0:t1, :].reshape(in_tiles, 2 * E * hpt, P).copy()
    svh_v = GU_SVH.reshape(E, 2, H_ALL)
    svh_s = svh_v[:, :, t0 * 16:(t1) * 16].reshape(E, 2 * (hpt * 16)).copy()
    dn_s = DN_T[k0:k1, :, :].copy()
    dsuh_s = DN_SUH[:, k0 * 16:k1 * 16].copy()
    return gu_s, GU_SUH.copy(), svh_s, dn_s, dsuh_s, DN_SVH.copy()


def build_rank(rank):
    gu_s, gu_suh, gu_svh, dn_s, dn_suh, dn_svh = slice_buffers(rank)
    return EXL3SwitchGLU(
        gu_trellis=mx.array(gu_s).view(mx.uint16),
        gu_suh=mx.array(gu_suh),
        gu_svh=mx.array(gu_svh),
        dn_trellis=mx.array(dn_s).view(mx.uint16),
        dn_suh=mx.array(dn_suh),
        dn_svh=mx.array(dn_svh),
        k=K_BITS, cb=CB, activation=ACT)


print("== building full module ==", flush=True)
t0 = time.time()
FULL = build_full()
mx.eval(FULL._gu_trellis, FULL._dn_trellis)
print(f"  built in {time.time()-t0:.1f}s; local H={FULL.hidden_dims} D={FULL.input_dims}", flush=True)

print("== building rank modules (sliced) ==", flush=True)
R0, R1 = build_rank(0), build_rank(1)
mx.eval(R0._gu_trellis, R1._gu_trellis)
print(f"  rank0 H={R0.hidden_dims} rank1 H={R1.hidden_dims} (full H={FULL.hidden_dims})", flush=True)
assert R0.hidden_dims + R1.hidden_dims == FULL.hidden_dims, "slice width mismatch"

fails = []
for R in (1, 4):
    x = mx.array(np.linspace(-0.4, 0.4, R * D_ALL, dtype=np.float16)).reshape(1, R, D_ALL)
    ind = mx.array(np.stack([np.random.RandomState(7 + R).choice(E, K_SEL, replace=False)
                             for _ in range(R)]).reshape(1, R, K_SEL).astype(np.int32))
    y_full = FULL(x, ind)
    y0 = R0(x, ind)
    y1 = R1(x, ind)
    mx.eval(y_full, y0, y1)
    y_sum = y0 + y1
    mx.eval(y_full, y_sum)
    a = np.array(y_full, dtype=np.float32)
    b = np.array(y_sum, dtype=np.float32)
    maxd = float(np.abs(a - b).max())
    denom = float(np.abs(a).max())
    cos = float((a.ravel() @ b.ravel()) / (np.linalg.norm(a) * np.linalg.norm(b)))
    ok = cos > 0.99995 and maxd <= max(1e-2, 1e-3 * denom)
    print(f"  [{'PASS' if ok else 'FAIL'}] R={R}: max|diff|={maxd:.4g} "
          f"(max|full|={denom:.4g}) cos={cos:.7f}", flush=True)
    if not ok:
        fails.append(R)

# control: two rank module outputs must NOT each equal the full output (they are partials)
R = 1
x = mx.array(np.linspace(-0.4, 0.4, R * D_ALL, dtype=np.float16)).reshape(1, R, D_ALL)
ind = mx.array(np.random.RandomState(99).choice(E, K_SEL, replace=False).reshape(1, 1, K_SEL).astype(np.int32))
yf = np.array(FULL(x, ind), dtype=np.float32)
y0 = np.array(R0(x, ind), dtype=np.float32)
ratio = float(np.abs(y0 - yf).max() / (np.abs(yf).max() + 1e-9))
print(f"  control: rank0-alone deviates from full by {ratio:.4f} of max (expect >0.1)",
      "OK" if ratio > 0.1 else "SUSPECT", flush=True)

print(flush=True)
if fails:
    print(f"VERDICT: FAIL at R={fails}")
    sys.exit(1)
print("VERDICT: PASS -- sum of 2-way split partials reproduces the full EXL3SwitchGLU")
