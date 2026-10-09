#!/usr/bin/env python3
"""p2_cached_w.py -- decisive route test: cached decoded-W matmul vs fused trellis.

Reading-1's route (bf16 dequant-cache) does NOT rest on a "240 GB/s effective"
claim (that is refuted by the trellis measurement); its real mechanism would be
to trade the ~53 GB/s trellis-stream GEMM for a native-matmul stream over a
persistent decoded W. This script measures that directly:

  * fused trellis (production m=1 simd GEMV / m=4 devx GEMM)         [baseline]
  * cached-W  (decode W ONCE, keep resident, then only the matmul)

Runs ON a node with the node's exo venv python. Reports per-shape ms and the
trellis-equivalent GB/s (= trellis_bytes / time, apples-to-apples with the
p2_dense_real numbers) plus the resident W bytes (RAM cost of the cache).

Env: PD_LAYER (20) PD_REPS (9) PD_JSON PD_PKG.
"""
import json, os, sys, time

HOME = os.path.expanduser("~")
_pkg = os.environ.get("PD_PKG")
if _pkg:
    sys.path.insert(0, _pkg)

import numpy as np
import mlx.core as mx

from mlx_lm.models.exl3.gemv_metal import decode_full_mlx
from mlx_lm.models.exl3.exl3_linear import EXL3Linear
from mlx_lm.models.exl3.loader import Exl3Checkpoint, load_dense_layer

CK = os.environ.get(
    "PD_CK", HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
)
LAYER = int(os.environ.get("PD_LAYER", "20"))
REPS = int(os.environ.get("PD_REPS", "9"))
OUT = os.environ.get("PD_JSON", f"{HOME}/p5-dense-ws/p2_cached_w_layer{LAYER}.json")

ROSTER = [
    "attn.wq_a", "attn.wq_b", "attn.wkv",
    "attn.wo_a.slice.0", "attn.wo_a.slice.1", "attn.wo_a.slice.2", "attn.wo_a.slice.3",
    "attn.wo_a.slice.4", "attn.wo_a.slice.5", "attn.wo_a.slice.6", "attn.wo_a.slice.7",
    "attn.wo_b", "ffn.shared_experts.w1", "ffn.shared_experts.w2", "ffn.shared_experts.w3",
]
SHARD = {"attn.wq_b": "out", "attn.wo_b": "in",
         "ffn.shared_experts.w1": "out", "ffn.shared_experts.w2": "in",
         "ffn.shared_experts.w3": "out"}


def log(*a):
    print("[p2cw]", *a, flush=True)


def timeit(fn, reps=REPS, warm=1):
    for _ in range(warm):
        mx.eval(fn())
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


ck = Exl3Checkpoint(CK)
shapes = {}
for nm in ROSTER:
    full = f"layers.{LAYER}.{nm}"
    if ck.has(full + ".trellis"):
        shapes[nm] = EXL3Linear(load_dense_layer(ck, full))

res = {"layer": LAYER, "reps": REPS, "shapes": {}}
for nm, lin in shapes.items():
    rt = lin._rt
    tb = lin._exl3.trellis.size * lin._exl3.trellis.itemsize
    if SHARD.get(nm):
        tb //= 2
    # decode once, keep resident
    w = decode_full_mlx(rt.trellis, rt.k, rt.cb)
    mx.eval(w)
    w_dtype = str(w.dtype)
    w_bytes_full = w.size * w.itemsize
    w_bytes_rank = w_bytes_full // 2 if SHARD.get(nm) else w_bytes_full
    ent = {"in": lin.in_features, "out": lin.out_features, "k": lin._exl3.k,
           "shard": SHARD.get(nm), "trellis_bytes_rank0": tb,
           "cachedW_dtype": w_dtype, "cachedW_bytes_rank0": w_bytes_rank,
           "cachedW_over_trellis": w_bytes_rank / tb}
    for m, tag in ((1, "m1"), (4, "m4")):
        x = mx.array(np.random.RandomState(m).randn(m, lin.in_features).astype(np.float16))
        mx.eval(x)
        ent[f"fused_ms_{tag}"] = timeit(lambda: lin(x)) * 1e3
        ent[f"cached_ms_{tag}"] = timeit(lambda: rt.finish_y((rt.prepare_xh(x) @ w).astype(mx.float16))) * 1e3
        ent[f"fused_gbs_{tag}"] = tb / (ent[f"fused_ms_{tag}"] * 1e-3 * 1e9)
        ent[f"cached_gbs_{tag}"] = tb / (ent[f"cached_ms_{tag}"] * 1e-3 * 1e9)
        ent[f"cached_native_gbs_{tag}"] = w_bytes_rank / (ent[f"cached_ms_{tag}"] * 1e-3 * 1e9)
    res["shapes"][nm] = ent
    log(f"{nm:22s} tB={tb/1e6:6.2f}MB  fused m1/m4={ent['fused_ms_m1']:.3f}/{ent['fused_ms_m4']:.3f}  "
        f"cached m1/m4={ent['cached_ms_m1']:.3f}/{ent['cached_ms_m4']:.3f} ms  "
        f"cachedW={w_bytes_rank/1e6:.1f}MB({w_dtype}, x{w_bytes_rank/tb:.2f})")

for tag in ("m1", "m4"):
    tb = sum(e["trellis_bytes_rank0"] for e in res["shapes"].values())
    wb = sum(e["cachedW_bytes_rank0"] for e in res["shapes"].values())
    tf = sum(e[f"fused_ms_{tag}"] for e in res["shapes"].values())
    tc = sum(e[f"cached_ms_{tag}"] for e in res["shapes"].values())
    res[f"sum_trellis_bytes_rank0_{tag}"] = tb
    res[f"sum_cachedW_bytes_rank0_{tag}"] = wb
    res[f"sum_fused_ms_{tag}"] = tf
    res[f"sum_cached_ms_{tag}"] = tc
    res[f"fused_gbs_{tag}"] = tb / (tf * 1e9)
    res[f"cached_trellis_equiv_gbs_{tag}"] = tb / (tc * 1e9)
    res[f"cached_native_gbs_{tag}"] = wb / (tc * 1e9)
    res[f"speedup_{tag}"] = tf / tc
    log(f"SUM {tag}: fused {tf:.3f}ms ({tb/(tf*1e9):.1f} GB/s trellis-equiv) | "
        f"cached {tc:.3f}ms ({tb/(tc*1e9):.1f} GB/s trellis-equiv, {wb/(tc*1e9):.1f} GB/s native) "
        f"-> speedup x{tf/tc:.2f}")

# whole-slice single-eval (production fused-graph shape; buries the eval floor)
pre = {}
for nm, lin in shapes.items():
    w = decode_full_mlx(lin._rt.trellis, lin._rt.k, lin._rt.cb)
    mx.eval(w)
    pre[nm] = (lin, w, [mx.array(np.random.RandomState(m).randn(m, lin.in_features).astype(np.float16))
                        for m in (1, 4)])
    for x in pre[nm][2]:
        mx.eval(x)
for mi, tag in ((0, "m1"), (1, "m4")):
    def whole_fused():
        return [lin(xs[mi]) for lin, w, xs in pre.values()]
    def whole_cached():
        return [lin._rt.finish_y((lin._rt.prepare_xh(xs[mi]) @ w).astype(mx.float16))
                for lin, w, xs in pre.values()]
    for name, fn in (("fused", whole_fused), ("cached", whole_cached)):
        mx.eval(*fn())
        ts = []
        for _ in range(max(REPS, 7)):
            t0 = time.perf_counter()
            mx.eval(*fn())
            ts.append(time.perf_counter() - t0)
        ts.sort()
        tb = sum(e["trellis_bytes_rank0"] for e in res["shapes"].values())
        wb = sum(e["cachedW_bytes_rank0"] for e in res["shapes"].values())
        res[f"whole_{name}_ms_{tag}"] = ts[len(ts) // 2] * 1e3
        res[f"whole_{name}_gbs_{tag}"] = tb / (ts[len(ts) // 2] * 1e9)
        log(f"WHOLE-SLICE {name:6s} {tag}: {ts[len(ts)//2]*1e3:.3f} ms -> "
            f"{tb/(ts[len(ts)//2]*1e9):.1f} GB/s trellis-equiv"
            + (f", {wb/(ts[len(ts)//2]*1e9):.1f} GB/s native" if name == "cached" else ""))

json.dump(res, open(OUT, "w"), indent=1)
log("wrote", OUT)
