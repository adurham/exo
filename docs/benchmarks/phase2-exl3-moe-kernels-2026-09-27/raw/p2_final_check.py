#!/usr/bin/env python3
"""FINAL corrected baseline: EXL3 vs the REAL production expert format.

CORRECTION TO THE EARLIER RUNS: production V4-Flash experts are **mxfp4
4-bit, group_size 32** (mlx-lm/utils.py: "packing (mxfp4 experts / mxfp8
attention+shared+MTP)"), NOT mxfp8 8-bit. Confirmed by shape arithmetic on the
real tensors: gate/up [out=2048, in=4096] stored as I8 [2048, 2048] = 2 values
per byte = 4 bits/value, with E8M0 scales [2048, 128] = one per 32 inputs.

An mxfp8 "baseline" has 2x the bytes of the real thing, so it made EXL3 look
~2x better than it is. This measures the honest comparison.

Baselines:
  mxfp4 gs=32 bits=4   <- PRODUCTION (what the cluster actually runs)
  affine gs=64 bits=4  <- format control
  affine gs=32 bits=4  <- group-size control at production's group size
"""
import gc
import json
import os
import time

import numpy as np
import mlx.core as mx
import mlx.nn as nn

os.environ.setdefault("EXL3_MM_MAX_ROWS", "100000")   # must precede import

MODEL = os.path.expanduser(
    "~/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw")
LAYER = int(os.environ.get("BENCH_LAYER", "1"))
E = int(os.environ.get("BENCH_EXPERTS", "128"))
KK = 6
HID, INT = 5120, 2304


def log(*a):
    print(*a, flush=True)


idx = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))["weight_map"]
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

from ponyexl3.mlx.exl3_moe import EXL3SwitchGLU
from ponyexl3.ref.codebook import codebook_mode_from_flags
from ponyexl3.ref.layer import EXL3Layer
from ponyexl3.mlx.reconstruct import reconstruct_public_mlx
from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwiGLU, SwitchGLU

CB = codebook_mode_from_flags(mcg=False, mul1=True)
p0 = f"layers.{LAYER}.ffn.experts.0."
K_BITS = int(tens[p0 + "w1.trellis"].shape[2] * 16 // 256)

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
        self._bytes = sum(int(np.prod(a.shape)) * a.itemsize
                          for a in (self.weight, self.scales))


class CheapSwitch(SwitchGLU):
    def __init__(self, mods):
        nn.Module.__init__(self)
        self.gate_proj, self.up_proj, self.down_proj = mods["gate"], mods["up"], mods["down"]
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
log("[build] affine 4/64 ...")
A46 = build(4, 64, "affine")
log("[build] affine 4/32 ...")
A43 = build(4, 32, "affine")

exl3_bytes = (int(np.prod(exl3._gu_trellis.shape)) * 2
              + int(np.prod(exl3._gu_suh.shape)) * 2
              + int(np.prod(exl3._gu_svh.shape)) * 2
              + int(np.prod(exl3._dn_trellis.shape)) * 2
              + int(np.prod(exl3._dn_suh.shape)) * 2
              + int(np.prod(exl3._dn_svh.shape)) * 2)
log(f"[bytes] EXL3 E={E}: {exl3_bytes / 2**20:.1f} MiB "
    f"({exl3_bytes / (E * (HID * INT * 2 + INT * HID)) * 8:.2f} bpw effective)")


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
log("=" * 100)
log("EXL3 vs PRODUCTION mxfp4 4-bit/gs32  (mx.eval per rep; MLX is lazy)")
log("=" * 100)
log(f"{'rows':>6} {'EXL3 ms':>10} {'mxfp4 ms':>10} {'a4/64 ms':>10} {'a4/32 ms':>10} "
    f"{'ex/PROD':>9} {'ex/a4/64':>9} {'EXL3 tok/s':>11}")
res = {}
for R, reps in ((1, 250), (4, 250), (8, 200), (256, 12), (512, 10),
                (1024, 6), (2048, 4)):
    x = mx.random.normal((1, R, HID)).astype(mx.float16) * 0.1
    ind = mx.array(np.stack([np.random.choice(E, KK, replace=False)
                             for _ in range(R)]).reshape(1, R, KK).astype(np.int32))
    mx.eval(x, ind)
    t_ex = bench(exl3, x, ind, reps)
    t_p4 = bench(lambda a, b: P4(a, b), x, ind, reps)
    t_46 = bench(lambda a, b: A46(a, b), x, ind, reps)
    t_43 = bench(lambda a, b: A43(a, b), x, ind, reps)
    res[R] = (t_ex, t_p4, t_46, t_43)
    log(f"{R:>6} {t_ex:>10.3f} {t_p4:>10.3f} {t_46:>10.3f} {t_43:>10.3f} "
        f"{t_ex / t_p4:>9.3f} {t_ex / t_46:>9.3f} {1000 / (t_ex * 40):>11.1f}")

log("")
log("VERDICT vs plan gates, against the PRODUCTION baseline:")
for R, gate, lbl in ((1, 1.25, "decode R=1"), (4, 1.25, "verify R=4 (DSpark)"),
                     (8, 1.25, "verify R=8"),
                     (512, 1.7, "prefill R=512"), (2048, 1.7, "prefill R=2048")):
    r = res[R][0] / res[R][1]
    log(f"  {lbl:<22} exl3/prod = {r:>6.3f}  gate {gate}  "
        f"{'PASS' if r <= gate else 'MISS'}")

log("")
log("If EXL3 replaced ALL 40 MoE layers (upper bound on the speed cost):")
log(f"  decode: {res[1][0]:.3f} ms/layer x 40 = {res[1][0] * 40:.1f} ms/token "
    f"-> {1000 / (res[1][0] * 40):.1f} tok/s (EXL3-only, MoE only)")
log(f"  prod  : {res[1][1]:.3f} ms/layer x 40 = {res[1][1] * 40:.1f} ms/token "
    f"-> {1000 / (res[1][1] * 40):.1f} tok/s (mxfp4, MoE only)")

with open(os.path.expanduser("~/p2_final_results.json"), "w") as f:
    json.dump({str(k): {"exl3_ms": v[0], "prod_mxfp4_ms": v[1],
                        "aff4_64_ms": v[2], "aff4_32_ms": v[3],
                        "ratio_vs_prod": v[0] / v[1]} for k, v in res.items()},
              f, indent=1)
log("\nwrote ~/p2_final_results.json")
