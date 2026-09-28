#!/usr/bin/env python3
"""p19_chain2_probe.py -- chain re-probe ON THE PATCHED kernel.

After adopting SWAR decode + direct-read (library defaults), how much does
the decode chain still cost? Re-run the stub comparison against the patched
kernel:

  control : patched defaults (SWAR + XDIRECT ON)
  cheap   : decode replaced by float(cw)*5.086263e-5f + 2.2f -- keeps the
            trellis loads LIVE (cw feeds the result), so it measures the
            ceiling of any further decode-ALU optimization.

Compare with p15 floors (measured on the UNPATCHED kernel):
  control(unpatched) 1.031 | cheap+shuffle 0.866 | cheapshuf 0.653  (R=1)
If `cheap` on the patched kernel lands near 0.65-0.70 (R=1) / ~1.6-1.7
(R=4), decode ALU is still where the remaining gap lives. If it barely
moves from the patched control, decode cutting is exhausted and the next
target is structural (A2/B2 boundary, dispatch count).

md5 guard: control must match REF; cheap arm md5 will differ (diagnostic).
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
ARM = os.environ.get("EXL3_CHAIN2", "control")

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
from ponyexl3.ref.codebook import CodebookMode  # noqa: E402

if ARM == "cheap":
    G._decode_expr = lambda cb, cw_in="cw": (
        f"            float dq_val = float({cw_in}) * 5.086263e-5f + 2.2f;")
elif ARM != "control":
    raise SystemExit(f"unknown EXL3_CHAIN2={ARM}")
log(f"[probe] CHAIN2={ARM} (library SWAR+XDIRECT defaults apply)")

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
log(f"PATCHED-KERNEL CHAIN RE-PROBE (ARM={ARM}) -- E={E}, layer={LAYER}")
log("=" * 92)
log(f"{'rows':>6} {'EXL3 ms':>10} {'mxfp4 ms':>10} {'ex/PROD':>9} {'EXL3 tok/s':>11} "
    f"{'md5':>8} {'ref?':>5}")
np.random.seed(777)
mx.random.seed(777)
for R, reps in ((1, 250), (4, 250), (6, 200), (8, 200)):
    x = (mx.random.normal((1, R, HID)).astype(mx.float16) * 0.1)
    ind = mx.array(np.stack([np.random.choice(E, KK, replace=False)
                             for _ in range(R)]).reshape(1, R, KK).astype(np.int32))
    mx.eval(x, ind)
    y = exl3(x, ind)
    mx.eval(y)
    h = hashlib.md5(np.array(y).tobytes()).hexdigest()
    ok = (h == REF[R])
    t_ex = bench(exl3, x, ind, reps)
    t_p4 = bench(lambda a, b: P4(a, b), x, ind, reps)
    log(f"{R:>6} {t_ex:>10.3f} {t_p4:>10.3f} {t_ex / t_p4:>9.3f} "
        f"{1000 / (t_ex * 40):>11.1f} {h[:8]:>8} {('YES' if ok else 'no'):>5}")

log("")
log(f"CHAIN2-PROBE-DONE arm={ARM}")
