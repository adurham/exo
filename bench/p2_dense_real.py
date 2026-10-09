#!/usr/bin/env python3
"""p2_dense_real.py -- isolated PRODUCTION-SHAPE dense EXL3 GEMV/GEMM on REAL weights.

Runs ON a node with that node's exo venv python (the INSTALLED production mlx_lm,
the exl3 dispatch at mlx-lm 3bf8316). Loads real dense/shared/attn EXL3 projections
from the node's real checkpoint and times, at the production shapes:

  * m=1   GEMV  (decode / draft path)           -> rows==1 branch (simd GEMV)
  * m=4   GEMM  (verify path, gamma=3 -> m=g+1)  -> rows<=16 branch (v20 devx GEMM)

by calling the EXL3Linear module itself (prepare_xh + inner_gemv_mlx /
inner_gemm_mlx + finish_y), so the measured per-projection time IS the production
per-projection cost at the production shapes.

The mx.eval launch floor (~0.2 ms) is (a) buried per projection by K-batching
(K projections built into one lazy graph, one eval, time/K) and (b) reported for
the whole dense slice in a SINGLE eval (the production fused-graph shape).

Measured at FULL (unsharded) trellis shape: GB/s = trellis_bytes / time is
invariant to a world-2 rank slice (bytes and work both halve), so the /2 rank
sharding is applied only in the byte accounting (exact census printed below).

Env: PD_LAYER (20) PD_REPS (9) PD_K (8) PD_JSON PD_PKG.
"""
import json, os, sys, time

HOME = os.path.expanduser("~")
_pkg = os.environ.get("PD_PKG")
if _pkg:
    sys.path.insert(0, _pkg)

import numpy as np
import mlx.core as mx

from mlx_lm.models.exl3.gemv_metal import decode_full_mlx, inner_gemm_mlx
from mlx_lm.models.exl3.exl3_linear import EXL3Linear
from mlx_lm.models.exl3.layer_state import stripe_weight_mlx
from mlx_lm.models.exl3.loader import Exl3Checkpoint, load_dense_layer
from mlx_lm.models.exl3.stripe import DEFAULT_STRIPE_COLS

CK = os.environ.get(
    "PD_CK", HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
)
LAYER = int(os.environ.get("PD_LAYER", "20"))
REPS = int(os.environ.get("PD_REPS", "9"))
K = int(os.environ.get("PD_K", "8"))
OUT = os.environ.get("PD_JSON", f"{HOME}/p5-dense-ws/p2_dense_real_layer{LAYER}.json")

ROSTER = [
    "attn.wq_a", "attn.wq_b", "attn.wkv",
    "attn.wo_a.slice.0", "attn.wo_a.slice.1", "attn.wo_a.slice.2", "attn.wo_a.slice.3",
    "attn.wo_a.slice.4", "attn.wo_a.slice.5", "attn.wo_a.slice.6", "attn.wo_a.slice.7",
    "attn.wo_b", "attn.compressor.wkv", "attn.indexer.wk", "attn.indexer.wq_b",
    "ffn.shared_experts.w1", "ffn.shared_experts.w2", "ffn.shared_experts.w3",
]
# world-2 sharded axis for the rank-0 byte census. None = full/replicated.
SHARD = {"attn.wq_b": "out", "attn.wo_b": "in",
         "ffn.shared_experts.w1": "out", "ffn.shared_experts.w2": "in",
         "ffn.shared_experts.w3": "out"}


def log(*a):
    print("[p2d]", *a, flush=True)


def trellis_bytes(lay, shard):
    n = lay.trellis.size * lay.trellis.itemsize
    if shard:
        n //= 2
    return n


def timeit(fn, reps=REPS, warm=1):
    for _ in range(warm):
        mx.eval(fn())
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2], ts[0], ts[-1]


def timeit_k(fn, k=K, reps=REPS, warm=1):
    """K calls into one lazy graph, one eval, time/k -> eval floor buried."""
    for _ in range(warm):
        mx.eval(*[fn() for _ in range(k)])
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        outs = [fn() for _ in range(k)]
        mx.eval(*outs)
        ts.append((time.perf_counter() - t0) / k)
    ts.sort()
    return ts[len(ts) // 2], ts[0], ts[-1]


def path_fullw(lin, x2d):
    rt = lin._rt
    w = decode_full_mlx(rt.trellis, rt.k, rt.cb)
    return rt.finish_y((rt.prepare_xh(x2d) @ w).astype(mx.float16))


def path_stripe(lin, x2d, cols=DEFAULT_STRIPE_COLS):
    rt = lin._rt
    xh32 = rt.prepare_xh(x2d).astype(mx.float32)
    ch = []
    for n0 in range(0, lin.out_features, cols):
        n = min(cols, lin.out_features - n0)
        w = stripe_weight_mlx(lin._exl3, n0, n, use_cache=True)
        ch.append((xh32 @ w.astype(mx.float32)).astype(mx.float16))
    return rt.finish_y(mx.concatenate(ch, axis=1))


def path_fused(lin, x2d):
    rt = lin._rt
    return rt.finish_y(inner_gemm_mlx(rt.prepare_xh(x2d), rt.trellis, rt.k, rt.cb).astype(mx.float16))


log(f"loading layer {LAYER} from {CK}")
ck = Exl3Checkpoint(CK)
shapes = {}
for nm in ROSTER:
    full = f"layers.{LAYER}.{nm}"
    if not ck.has(full + ".trellis"):
        log("skip missing", full)
        continue
    shapes[nm] = EXL3Linear(load_dense_layer(ck, full))

fl = mx.array(np.zeros((1, 4096), np.float16))
mx.eval(fl)
floor, *_ = timeit(lambda: fl.sum(), reps=max(REPS, 9))
log(f"layer {LAYER} reps {REPS} K {K} shapes={len(shapes)} eval_floor_ms={floor*1e3:.3f}")

res = {"layer": LAYER, "reps": REPS, "k_batch": K, "floor_ms": floor * 1e3,
       "shapes": {}}

for nm, lin in shapes.items():
    shard = SHARD.get(nm)
    wb = trellis_bytes(lin._exl3, shard)
    ent = {"in": lin.in_features, "out": lin.out_features, "k": lin._exl3.k,
           "shard": shard, "trellis_bytes_rank0": wb,
           "trellis_bytes_full": lin._exl3.trellis.size * lin._exl3.trellis.itemsize}
    for m, tag in ((1, "m1"), (4, "m4")):
        x = mx.array(np.random.RandomState(m).randn(m, lin.in_features).astype(np.float16))
        mx.eval(x)
        med, lo, hi = timeit_k(lambda: lin(x), k=K)
        ent[f"ms_{tag}"] = med * 1e3
        ent[f"ms_{tag}_min"] = lo * 1e3
        ent[f"ms_{tag}_max"] = hi * 1e3
        ent[f"gbs_{tag}"] = wb / (med * 1e9)
    res["shapes"][nm] = ent
    log(f"{nm:24s} {lin.in_features:5d}x{lin.out_features:6d} k{lin._exl3.k} "
        f"tB={wb/1e6:7.2f}MB  m1={ent['ms_m1']:7.3f}ms({ent['gbs_m1']:6.1f}GB/s)  "
        f"m4={ent['ms_m4']:7.3f}ms({ent['gbs_m4']:6.1f}GB/s)")

# aggregate over the layer's dense slice (rank0)
for tag in ("m1", "m4"):
    b = sum(e["trellis_bytes_rank0"] for e in res["shapes"].values())
    t = sum(e[f"ms_{tag}"] for e in res["shapes"].values())
    res[f"sum_bytes_rank0"] = b
    res[f"sum_ms_{tag}"] = t
    res[f"effective_gbs_{tag}"] = b / (t * 1e9)
    log(f"SUM  rank0 {tag}: {b/1e6:.2f} MB, {t:.3f} ms -> {b/(t*1e9):.1f} GB/s "
        f"(floor-corrected {b/((t - floor)*1e9):.1f})")

# whole-slice SINGLE-eval (production fused-graph shape, rank0)
pre = {}
for m, tag in ((1, "m1"), (4, "m4")):
    pre[tag] = [(lin, mx.array(np.random.RandomState(m).randn(m, lin.in_features).astype(np.float16)))
                for lin in shapes.values()]
    for _, x in pre[tag]:
        mx.eval(x)

for tag in ("m1", "m4"):
    def whole():
        return [lin(x) for lin, x in pre[tag]]
    mx.eval(*whole())
    ts = []
    for _ in range(max(REPS, 7)):
        t0 = time.perf_counter()
        mx.eval(*whole())
        ts.append(time.perf_counter() - t0)
    ts.sort()
    b = sum(e["trellis_bytes_rank0"] for e in res["shapes"].values())
    res[f"whole_slice_ms_{tag}"] = ts[len(ts) // 2] * 1e3
    res[f"whole_slice_gbs_{tag}"] = b / (ts[len(ts) // 2] * 1e9)
    log(f"WHOLE-SLICE single-eval {tag}: {ts[len(ts)//2]*1e3:.3f} ms -> "
        f"{b/(ts[len(ts)//2]*1e9):.1f} GB/s")

# kernel-parity audit: production vs the three microbench variants
audit = {}
for nm in ("attn.wq_b", "attn.wo_b", "ffn.shared_experts.w2", "attn.wq_a"):
    if nm not in shapes:
        continue
    lin = shapes[nm]
    a = {}
    for m, tag in ((1, "m1"), (4, "m4")):
        x = mx.array(np.random.RandomState(7 + m).randn(m, lin.in_features).astype(np.float16))
        mx.eval(x)
        a[f"prod_{tag}"] = timeit_k(lambda: lin(x), k=K)[0] * 1e3
        a[f"fullw_{tag}"] = timeit_k(lambda: path_fullw(lin, x), k=2)[0] * 1e3
        a[f"stripe_{tag}"] = timeit_k(lambda: path_stripe(lin, x), k=2)[0] * 1e3
        a[f"fused_{tag}"] = timeit_k(lambda: path_fused(lin, x), k=K)[0] * 1e3
    audit[nm] = a
res["kernel_parity_audit"] = audit
log("kernel-parity audit: " + json.dumps(audit))

json.dump(res, open(OUT, "w"), indent=1)
log("wrote", OUT)
