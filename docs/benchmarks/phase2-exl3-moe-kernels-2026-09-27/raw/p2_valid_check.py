#!/usr/bin/env python3
"""Cross-check with the VALID timing method + the PRODUCTION baseline.

Method note: MLX evaluation is lazy. Building graphs in a tight loop and
dropping the results does NOT execute them, so "amortized reps/sync" timing
measures almost nothing (it can be 100x+ off). The only valid method is
mx.eval(per call) around every measured call. That is what this script does.

Baselines:
  mxfp8 gs=32 bits=8  <- WHAT PRODUCTION ACTUALLY LOADS. start_cluster.sh
                         converts the fp8 experts to mxfp8 8-bit at load
                         (group=32, bits=8). This is the honest baseline for
                         "is EXL3 faster or slower than what we run today".
  affine 4/64, affine 8/64  <- format controls

Also quantifies the R=2..8 path problem: EXL3SwitchGLU._decode_fused2 (the
fused multi-row decode) requires hidden_dims <= 512, and V4.1 is 2304, so
R=2..8 falls through to _prefill which decodes the WHOLE stacked trellis to
transient fp16. We measure the cost of that, and the cost of the obvious
workaround (loop the R=1 fused path per row).
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
log(f"[exl3] built; _v2_ok()={exl3._v2_ok()} (False => R=2..8 uses _prefill)")


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


log("[build] production baseline mxfp8 gs=32 bits=8 ...")
P8 = build(8, 32, "mxfp8")
log("[build] affine 4/64 ...")
A4 = build(4, 64, "affine")
log("[build] affine 8/64 ...")
A8 = build(8, 64, "affine")


def bench(fn, x, ind, reps, warmup=10):
    """VALID method: mx.eval around every call (MLX is lazy)."""
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


def rowloop(x, ind):
    """Workaround probe: run the R=1 fused path once per row."""
    B, S, kk = ind.shape
    outs = [exl3(x[:, i:i + 1], ind[:, i:i + 1]) for i in range(S)]
    return mx.concatenate(outs, axis=1)


log("")
log("=" * 96)
log("VALID per-call timing (mx.eval per rep). Ratios vs the PRODUCTION mxfp8 baseline.")
log("=" * 96)
log(f"{'rows':>6} {'EXL3 ms':>10} {'mxfp8 ms':>10} {'aff4 ms':>10} {'aff8 ms':>10} "
    f"{'ex/p8':>8} {'ex/a4':>8} {'ex/a8':>8} {'rowloop':>9}")
for R, reps in ((1, 300), (4, 300), (8, 200)):
    x = mx.random.normal((1, R, HID)).astype(mx.float16) * 0.1
    ind = mx.array(np.stack([np.random.choice(E, KK, replace=False)
                             for _ in range(R)]).reshape(1, R, KK).astype(np.int32))
    mx.eval(x, ind)
    t_ex = bench(exl3, x, ind, reps)
    t_p8 = bench(lambda a, b: P8(a, b), x, ind, reps)
    t_a4 = bench(lambda a, b: A4(a, b), x, ind, reps)
    t_a8 = bench(lambda a, b: A8(a, b), x, ind, reps)
    t_rl = bench(rowloop, x, ind, max(20, reps // 10)) if R > 1 else float("nan")
    log(f"{R:>6} {t_ex:>10.3f} {t_p8:>10.3f} {t_a4:>10.3f} {t_a8:>10.3f} "
        f"{t_ex / t_p8:>8.3f} {t_ex / t_a4:>8.3f} {t_ex / t_a8:>8.3f} "
        f"{t_rl:>9.3f}")

log("")
log("Context: one V4.1 token = 40 MoE layers. If EVERY layer were EXL3:")
for R, reps in ((1, 300),):
    x = mx.random.normal((1, R, HID)).astype(mx.float16) * 0.1
    ind = mx.array(np.stack([np.random.choice(E, KK, replace=False)
                             for _ in range(R)]).reshape(1, R, KK).astype(np.int32))
    mx.eval(x, ind)
    t = bench(exl3, x, ind, reps)
    log(f"  decode: {t:.3f} ms/layer x 40 layers = {t * 40:.1f} ms/token "
        f"-> {1000 / (t * 40):.1f} tok/s ceiling (single rank, MoE only)")
log("  (the plan streams only a FRACTION of layers as EXL3; the rest stay")
log("   resident affine, so the blend is what matters -- see the report.)")
