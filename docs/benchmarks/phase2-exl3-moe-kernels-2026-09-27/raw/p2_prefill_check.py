#!/usr/bin/env python3
"""Prefill-rows focus: EXL3 vs the PRODUCTION mxfp8 baseline at R=512/1024/2048.

The first bench compared prefill against affine-4, but production loads mxfp8
8-bit (start_cluster.sh converts the fp8 experts to mxfp8 gs=32 bits=8 at
load), and mxfp8 measures ~1.5-1.8x SLOWER than affine-4 at the same rows.
So the affine-4 ratio overstated the gap. This measures the honest baseline.

Also: the default _MM_MAX_ROWS=9216 means the segmented-GEMM path is NOT taken
above N = rows*top_k = 1536 rows at top_k=6, and the fallback (decode-all +
gather_mm) tries to materialize E*out*in*2 bytes -- 3.02 GB at E=128, an int32
overflow crash. We therefore set EXL3_MM_MAX_ROWS explicitly (env, before
import) and confirm the segmented path is what runs.
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

import ponyexl3.mlx.exl3_moe as _m
from ponyexl3.mlx.exl3_moe import EXL3SwitchGLU
from ponyexl3.ref.codebook import codebook_mode_from_flags
from ponyexl3.ref.layer import EXL3Layer
from ponyexl3.mlx.reconstruct import reconstruct_public_mlx
from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwiGLU, SwitchGLU

log(f"[env] _MOE_MM={_m._MOE_MM} _MM_MAX_ROWS={_m._MM_MAX_ROWS} "
    f"_SEG_BM={_m._SEG_BM}")

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


log("[build] production baseline mxfp8 8/32 ...")
P8 = build(8, 32, "mxfp8")
log("[build] affine 4/64 ...")
A4 = build(4, 64, "affine")


def bench(fn, x, ind, reps, warmup=5):
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
log("PREFILL rows: EXL3 vs PRODUCTION mxfp8 8-bit (the honest baseline)")
log("=" * 92)
log(f"{'rows':>6} {'N=rows*k':>9} {'EXL3 ms':>11} {'mxfp8 ms':>11} {'aff4 ms':>11} "
    f"{'ex/p8':>8} {'ex/a4':>8} {'gate1.7':>9}")
for R, reps in ((256, 12), (512, 10), (1024, 6), (2048, 4)):
    x = mx.random.normal((1, R, HID)).astype(mx.float16) * 0.1
    ind = mx.array(np.stack([np.random.choice(E, KK, replace=False)
                             for _ in range(R)]).reshape(1, R, KK).astype(np.int32))
    mx.eval(x, ind)
    N = R * KK
    t_ex = bench(exl3, x, ind, reps)
    t_p8 = bench(lambda a, b: P8(a, b), x, ind, reps)
    t_a4 = bench(lambda a, b: A4(a, b), x, ind, reps)
    ok = "PASS" if t_ex / t_p8 <= 1.7 else "MISS"
    log(f"{R:>6} {N:>9} {t_ex:>11.3f} {t_p8:>11.3f} {t_a4:>11.3f} "
        f"{t_ex / t_p8:>8.3f} {t_ex / t_a4:>8.3f} {ok:>9}")
log("")
log("If the segmented-GEMM path is disabled (EXL3_MOE_MM=0) the fallback tries")
log("to materialize E*out*in*2 bytes and crashes on int32 overflow at E=128:")
log("  128 * 5120 * 2304 * 2 = 3019898880 > INT32_MAX")
log("  (E=384 -- the real V4.1 count -- would be 9.06 GB, so the fallback is")
log("   never usable at full scale; the segmented path is mandatory.)")
