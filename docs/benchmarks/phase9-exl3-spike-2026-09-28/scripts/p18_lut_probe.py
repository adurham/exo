#!/usr/bin/env python3
"""p18_lut_probe.py -- threadgroup LUT for the decode tail (candidate #3).

Current library default (post phase-9): SWAR byte-sum + affine tail, ~13 ops
per codeword. The tail -- ushort trunc, bitcast, half mul, half add, cvt --
is 4-5 ops whose result depends ONLY on the integer byte-sum s in [0, 1020]:

    dq_val(s) = float(as_type<half>(ushort(0x6400 + s)) * K_INV + K_BIAS)

So a 1021-entry threadgroup fp32 table + one shared load replaces it.
The kernel keeps the SWAR sum; only the tail changes:

    float dq_val = dq_lut[dq_sum - 0x6400u];

LUT init is 8 iterations of 128 threads + one barrier, once per threadgroup.
The table values are computed with THE EXACT SAME expression as stock, so
outputs must be bit-identical (md5 guard vs REF).

Arm: EXL3_LUT=1. Control = library defaults (SWAR+XDIRECT, md5-verified at
0.733 ms R=1 in p13-libcheck-adopted.log).
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import time

import numpy as np
import mlx.core as mx
import mlx.nn as nn

os.environ.setdefault("EXL3_MM_MAX_ROWS", "100000")

MODEL = os.path.expanduser(
    "~/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw")
LAYER = int(os.environ.get("BENCH_LAYER", "1"))
E = int(os.environ.get("BENCH_EXPERTS", "384"))
KK = 6
HID, INT = 5120, 2304
LUT = os.environ.get("EXL3_LUT", "0") == "1"

REF = {1: "f859beeeb0408d6e8d5bb248c9c75c4b",
       4: "08a16bf60b2c8225f8b2585d93e99662",
       6: "27cd2b1f4f3ef900f4637b8501c97f33",
       8: "cff548f7d15a522b0fc32388508f9fe0"}


def log(*a):
    print(*a, flush=True)


idx = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))[
    "weight_map"]
SUF = ("w1.trellis", "w1.suh", "w1.svh", "w2.trellis", "w2.suh", "w2.svh",
       "w3.trellis", "w3.suh", "w3.svh")
by_shard: dict = {}
for e in range(E):
    for s in SUF:
        k = f"layers.{LAYER}.ffn.experts.{e}.{s}"
        by_shard.setdefault(idx[k], []).append(k)

from safetensors import safe_open

tens: dict = {}
for s, ks in sorted(by_shard.items()):
    with safe_open(os.path.join(MODEL, s), framework="np") as f:
        for k in ks:
            tens[k] = f.get_tensor(k)

import ponyexl3.mlx.gemv_metal as G  # noqa: E402
import ponyexl3.mlx.exl3_moe as M  # noqa: E402
from ponyexl3.ref.codebook import CodebookMode  # noqa: E402

if LUT:
    _od = G._decode_expr

    def _lut_dec(cb, cw_in="cw"):
        if cb == CodebookMode.MUL1:
            return f"""
            uint dq_cw = {cw_in} * 0x83DCD12Du;
            uint dq_t = (dq_cw & 0x00FF00FFu) + ((dq_cw >> 8u) & 0x00FF00FFu);
            uint dq_sum = 0x6400u + (dq_t & 0xFFFFu) + (dq_t >> 16u);
            float dq_val = dq_lut[dq_sum - 0x6400u];
"""
        return _od(cb, cw_in=cw_in)
    G._decode_expr = _lut_dec

    _LUT_DECL = """
    threadgroup float dq_lut[1021];
    for (uint li = tid; li < 1021u; li += 128u) {
        half lh = as_type<half>(ushort(li));
        dq_lut[li] = float(lh * as_type<half>(ushort(0x1EEEu))
                           + as_type<half>(ushort(0xC931u)));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
"""
    _MARKER = "    threadgroup float tg_w[4][256];"
    _o = {n: getattr(M, n) for n in (
        "_moe_gateup_source", "_moe_down_source",
        "_moe_gateup2_source", "_moe_down2_source")}

    def _mk(orig):
        def _w(*a, **k):
            s = orig(*a, **k)
            if _MARKER not in s:
                raise AssertionError("tg_w marker not found in kernel source")
            return s.replace(_MARKER, _MARKER + _LUT_DECL, 1)
        return _w
    for n, fn in _o.items():
        setattr(M, n, _mk(fn))

log(f"[probe] LUT={int(LUT)} (XDIRECT+SWAR at library defaults)")

from ponyexl3.mlx.exl3_moe import EXL3SwitchGLU  # noqa: E402
from ponyexl3.ref.layer import EXL3Layer  # noqa: E402
from ponyexl3.mlx.reconstruct import reconstruct_public_mlx  # noqa: E402
from mlx_lm.models.switch_layers import (  # noqa: E402
    QuantizedSwitchLinear, SwiGLU, SwitchGLU)

CB = CodebookMode.MUL1
p0 = f"layers.{LAYER}.ffn.experts.0."
K_BITS = int(tens[p0 + "w1.trellis"].shape[2] * 16 // 256)

gu, up, gs_, gv, dn, ds, dv = [], [], [], [], [], [], []
for e in range(E):
    p = f"layers.{LAYER}.ffn.experts.{e}."
    gu.append(mx.array(tens[p + "w1.trellis"]))
    up.append(mx.array(tens[p + "w3.trellis"]))
    gs_.append(mx.stack([mx.array(tens[p + "w1.suh"]),
                         mx.array(tens[p + "w3.suh"])]))
    gv.append(mx.concatenate([mx.array(tens[p + "w1.svh"]),
                              mx.array(tens[p + "w3.svh"])]))
    dn.append(mx.array(tens[p + "w2.trellis"]))
    ds.append(mx.array(tens[p + "w2.suh"]))
    dv.append(mx.array(tens[p + "w2.svh"]))

exl3 = EXL3SwitchGLU(
    gu_trellis=mx.concatenate(gu + up, axis=1).view(mx.uint16),
    gu_suh=mx.stack(gs_).astype(mx.float16),
    gu_svh=mx.stack(gv).astype(mx.float16),
    dn_trellis=mx.concatenate(dn, axis=1).view(mx.uint16),
    dn_suh=mx.stack(ds).astype(mx.float16),
    dn_svh=mx.stack(dv).astype(mx.float16),
    k=K_BITS, cb=CB, activation="silu")
mx.eval(exl3._gu_trellis, exl3._gu_suh, exl3._gu_svh,
        exl3._dn_trellis, exl3._dn_suh, exl3._dn_svh)
log(f"[exl3] built k={K_BITS} E={E} _v2_ok()={exl3._v2_ok()}")


def deq(prefix):
    lay = EXL3Layer(key=prefix, in_features=len(tens[prefix + "suh"]),
                    out_features=len(tens[prefix + "svh"]), k=K_BITS,
                    trellis=tens[prefix + "trellis"], suh=tens[prefix + "suh"],
                    svh=tens[prefix + "svh"], mul1=True)
    return reconstruct_public_mlx(lay)


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
        self.gate_proj, self.up_proj, self.down_proj = (
            mods["gate"], mods["up"], mods["down"])
        self.activation = SwiGLU()


def build(bits, gs, mode):
    m = {}
    for nm, pre in (("gate", "w1"), ("up", "w3"), ("down", "w2")):
        out = None
        for e in range(E):
            a = np.array(deq(f"layers.{LAYER}.ffn.experts.{e}.{pre}.")).T
            if out is None:
                out = np.empty((E,) + a.shape, dtype=np.float16)
            out[e] = a
        m[nm] = Cheap(out, bits, gs, mode)
        del out
        gc.collect()
        mx.clear_cache()
    return CheapSwitch(m)


log("[build] mxfp4 4/32 (PRODUCTION) ...")
P4 = build(4, 32, "mxfp4")


def bench(fn, x, ind, reps, warmup=8):
    for _ in range(warmup):
        mx.eval(fn(x, ind))
    mx.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(fn(x, ind))
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


log("")
log("=" * 92)
log(f"EXL3 LUT PROBE (LUT={int(LUT)}) -- E={E}, layer={LAYER}, top-{KK}")
log("=" * 92)
log(f"{'rows':>6} {'EXL3 ms':>10} {'mxfp4 ms':>10} {'ex/PROD':>9} {'EXL3 tok/s':>11} "
    f"{'md5':>8} {'match':>6}")
np.random.seed(777)
mx.random.seed(777)
allok = True
for R, reps in ((1, 250), (4, 250), (6, 200), (8, 200)):
    x = (mx.random.normal((1, R, HID)).astype(mx.float16) * 0.1)
    ind = mx.array(np.stack([np.random.choice(E, KK, replace=False)
                             for _ in range(R)]).reshape(1, R, KK).astype(np.int32))
    mx.eval(x, ind)
    y = exl3(x, ind)
    mx.eval(y)
    h = hashlib.md5(np.array(y).tobytes()).hexdigest()
    ok = (h == REF[R])
    allok &= ok
    t_ex = bench(exl3, x, ind, reps)
    t_p4 = bench(lambda a, b: P4(a, b), x, ind, reps)
    log(f"{R:>6} {t_ex:>10.3f} {t_p4:>10.3f} {t_ex / t_p4:>9.3f} "
        f"{1000 / (t_ex * 40):>11.1f} {h[:8]:>8} {('YES' if ok else 'NO'):>6}")

log("")
log(f"ALL-MD5-MATCH={allok}")
log(f"LUT-PROBE-DONE lut={int(LUT)}")
