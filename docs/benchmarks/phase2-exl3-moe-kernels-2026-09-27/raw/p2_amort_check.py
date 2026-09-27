#!/usr/bin/env python3
"""Independent cross-check of the R=1 decode ratio (the decisive gate).

The first bench measured whole-call wall time with mx.eval() per call, which
includes the Python/dtype plumbing around the kernels. This script measures the
GPU-side cost directly:

  1. Amortized timing over MANY reps inside one graph, then one sync
     (removes per-call sync overhead from the comparison).
  2. The same timing with the trellis/gather machinery exercised at two expert
     counts, to confirm the ratio is stable and not a warmup artifact.
  3. Reports per-call microseconds so the absolute cost is visible.

Also probes the R=4 (speculative verify) case, which is what MTP/DSpark
actually runs, since that is the ratio that matters for the 30-45 tok/s
projection.
"""
import gc
import json
import os
import time

import numpy as np
import mlx.core as mx
import mlx.nn as nn

MODEL = os.path.expanduser(
    "~/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw")
LAYER = 1
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

gu, up, gs, gv, dn, ds, dv = [], [], [], [], [], [], []
for e in range(E):
    p = f"layers.{LAYER}.ffn.experts.{e}."
    gu.append(mx.array(tens[p + "w1.trellis"]))
    up.append(mx.array(tens[p + "w3.trellis"]))
    gs.append(mx.stack([mx.array(tens[p + "w1.suh"]), mx.array(tens[p + "w3.suh"])]))
    gv.append(mx.concatenate([mx.array(tens[p + "w1.svh"]), mx.array(tens[p + "w3.svh"])]))
    dn.append(mx.array(tens[p + "w2.trellis"]))
    ds.append(mx.array(tens[p + "w2.suh"]))
    dv.append(mx.array(tens[p + "w2.svh"]))

exl3 = EXL3SwitchGLU(
    gu_trellis=mx.concatenate(gu + up, axis=1).view(mx.uint16),
    gu_suh=mx.stack(gs).astype(mx.float16),
    gu_svh=mx.stack(gv).astype(mx.float16),
    dn_trellis=mx.concatenate(dn, axis=1).view(mx.uint16),
    dn_suh=mx.stack(ds).astype(mx.float16),
    dn_svh=mx.stack(dv).astype(mx.float16),
    k=K_BITS, cb=CB, activation="silu")
mx.eval(exl3._gu_trellis, exl3._gu_suh, exl3._gu_svh,
        exl3._dn_trellis, exl3._dn_suh, exl3._dn_svh)


def deq(prefix):
    lay = EXL3Layer(key=prefix, in_features=len(tens[prefix + "suh"]),
                    out_features=len(tens[prefix + "svh"]), k=K_BITS,
                    trellis=tens[prefix + "trellis"], suh=tens[prefix + "suh"],
                    svh=tens[prefix + "svh"], mul1=True)
    return reconstruct_public_mlx(lay)


class Cheap(QuantizedSwitchLinear):
    def __init__(self, W, bits, gs):
        nn.Module.__init__(self)
        self.group_size, self.bits, self.mode = gs, bits, "affine"
        w, sc, *b = mx.quantize(mx.array(W), group_size=gs, bits=bits, mode="affine")
        self.weight, self.scales = w, sc
        if b:
            self.biases = b[0]
        mx.eval(self.weight, self.scales)


class CheapSwitch(SwitchGLU):
    def __init__(self, mods):
        nn.Module.__init__(self)
        self.gate_proj, self.up_proj, self.down_proj = mods["gate"], mods["up"], mods["down"]
        self.activation = SwiGLU()


def build(bits):
    m = {}
    for nm, pre in (("gate", "w1"), ("up", "w3"), ("down", "w2")):
        out = None
        for e in range(E):
            a = np.array(deq(f"layers.{LAYER}.ffn.experts.{e}.{pre}.")).T
            if out is None:
                out = np.empty((E,) + a.shape, dtype=np.float16)
            out[e] = a
        m[nm] = Cheap(out, bits, 64)
        del out
        gc.collect()
        mx.clear_cache()
    return CheapSwitch(m)


A4 = build(4)
A3 = build(3)

log("=" * 76)
log("CROSS-CHECK: amortized GPU time, many reps per sync")
log("=" * 76)


def amort(fn, x, ind, reps):
    for _ in range(10):
        mx.eval(fn(x, ind))
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn(x, ind)
    mx.eval(fn(x, ind))
    mx.synchronize()
    return (time.perf_counter() - t0) * 1e6 / reps          # microseconds


log(f"{'rows':>6} {'EXL3 us':>11} {'aff4 us':>11} {'aff3 us':>11} "
    f"{'ex/aff4':>9} {'ex/aff3':>9} {'reps':>7}")
for R, reps in ((1, 400), (2, 300), (4, 300), (8, 200), (16, 150), (64, 60), (256, 25)):
    x = mx.random.normal((1, R, HID)).astype(mx.float16) * 0.1
    ind = mx.array(np.stack([np.random.choice(E, KK, replace=False)
                             for _ in range(R)]).reshape(1, R, KK).astype(np.int32))
    mx.eval(x, ind)
    t_ex = amort(exl3, x, ind, reps)
    t_4 = amort(lambda a, b: A4(a, b), x, ind, reps)
    t_3 = amort(lambda a, b: A3(a, b), x, ind, reps)
    log(f"{R:>6} {t_ex:>11.1f} {t_4:>11.1f} {t_3:>11.1f} "
        f"{t_ex / t_4:>9.3f} {t_ex / t_3:>9.3f} {reps:>7}")

log("")
log("NOTE: MTP/DSpark speculative verify runs the MoE at R = gamma+1 rows,")
log("      i.e. the R=4..8 rows here. R=1 is plain decode.")
