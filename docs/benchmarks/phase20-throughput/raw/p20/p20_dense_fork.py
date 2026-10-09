#!/usr/bin/env python3
"""p20_dense_fork.py -- PHASE-1 EXPERIMENT 1: RAW no-dequant isolation (THE FORK).

Decides whether the dense/shared EXL3 slice's 9.3x gap to the ~497 GB/s read roof is
  (a) ALU/issue-bound in the trellis decode itself  -> near-structural, CLOSE the track
  (b) latency/occupancy/scheduling-bound at m=4      -> split-K / geometry / prefetch can harvest

Runs ON a node (studio2) with the node's exo venv python and the INSTALLED production
mlx_lm (exl3 dispatch at mlx-lm 16830e1).  REAL weights, layer-20 dense roster, production
per-rank (TP=2) shapes.  No relaunch, no server restart, no API POST -- local disk reads only.

ARMS (symmetric: whole-slice single-eval = production fused shape; K-batched = per-call,
floor-buried p95):
  R0   pure-read control : mx.sum over a bf16 buffer of the slice byte volume + 512 MB.
  R2   production kernel : EXL3Linear(x)  -> fused decode+matmul (the baseline).
  R1   RAW no-dequant    : native `x @ W` on DECODED W (decode NOT in the timed region).
  R3   decode-only       : decode_full_mlx(trellis) -> pure decode throughput.

THE FORK (from ROUND-DENSE-EXL3.md S6; sharpened by the decode/matmul-share ratios):
  decode_share = R3_time / R2_time at m=4.
    >= 0.80  -> decode alone ≈ the whole fused cost -> (a) ALU/issue-bound -> CLOSE.
  native_free  = R1_time / R2_time at m=4 (same shapes, more bytes).
  read_roof    = R0 read GB/s.
  If pure-read of the slice volume is far below T_fused AND decode_share < 0.5 -> (b) latency.

Env: PD_LAYER (20) PD_REPS (9) PD_K (8) PD_JSON PD_READ_MB (512).
"""
import json, os, sys, time, statistics

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
    "PD_CK", HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw")
LAYER = int(os.environ.get("PD_LAYER", "20"))
REPS = int(os.environ.get("PD_REPS", "9"))
K = int(os.environ.get("PD_K", "8"))
OUT = os.environ.get("PD_JSON", f"{HOME}/p5-dense-ws/p20_dense_fork_layer{LAYER}.json")
READ_MB = int(os.environ.get("PD_READ_MB", "512"))

ROSTER = [
    "attn.wq_a", "attn.wq_b", "attn.wkv",
    "attn.wo_a.slice.0", "attn.wo_a.slice.1", "attn.wo_a.slice.2", "attn.wo_a.slice.3",
    "attn.wo_a.slice.4", "attn.wo_a.slice.5", "attn.wo_a.slice.6", "attn.wo_a.slice.7",
    "attn.wo_b", "attn.compressor.wkv", "attn.indexer.wk", "attn.indexer.wq_b",
    "ffn.shared_experts.w1", "ffn.shared_experts.w2", "ffn.shared_experts.w3",
]
SHARD = {"attn.wq_b": "out", "attn.wo_b": "in",
         "ffn.shared_experts.w1": "out", "ffn.shared_experts.w2": "in",
         "ffn.shared_experts.w3": "out"}


def log(*a):
    print("[fork]", *a, flush=True)


def p95(vals):
    vs = sorted(vals)
    return vs[min(len(vs) - 1, int(round(0.95 * (len(vs) - 1))))]


def _stats(ts):
    return (statistics.median(ts) * 1e3, p95(ts) * 1e3, min(ts) * 1e3, max(ts) * 1e3)


def timeit(fn, reps=REPS, warm=3):
    """Whole-graph single eval (production fused shape). Discard 1st timed rep."""
    for _ in range(warm):
        mx.eval(fn())
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    if len(ts) > 2:
        ts = ts[1:]
    return _stats(ts)


def timeit_k(fn, k=K, reps=REPS, warm=2):
    """K whole-graph calls into one lazy graph, one eval, /k -> eval floor buried."""
    for _ in range(warm):
        mx.eval(*[fn() for _ in range(k)])
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(*[fn() for _ in range(k)])
        ts.append((time.perf_counter() - t0) / k)
    if len(ts) > 2:
        ts = ts[1:]
    return _stats(ts)


log(f"mlx {mx.__version__} loadavg={tuple(round(x,2) for x in os.getloadavg())}")
ck = Exl3Checkpoint(CK)
entries, sum_tb_rank, sum_tb_full = [], 0, 0
for nm in ROSTER:
    full = f"layers.{LAYER}.{nm}"
    if not ck.has(full + ".trellis"):
        log("skip missing", full)
        continue
    lin = EXL3Linear(load_dense_layer(ck, full))
    rt = lin._rt
    tb_full = rt.trellis.size * rt.trellis.itemsize
    tb_rank = tb_full // 2 if SHARD.get(nm) else tb_full
    sum_tb_full += tb_full
    sum_tb_rank += tb_rank
    mx.eval(rt.trellis)
    entries.append(dict(nm=nm, lin=lin, rt=rt, tb_full=tb_full, tb_rank=tb_rank,
                        in_f=lin.in_features, out_f=lin.out_features))
log(f"{len(entries)} dense linears; trellis full={sum_tb_full/1e6:.1f} MB rank0={sum_tb_rank/1e6:.1f} MB")

res = dict(layer=LAYER, reps=REPS, k_batch=K, mlx=mx.__version__,
           sum_trellis_full_bytes=sum_tb_full, sum_trellis_rank0_bytes=sum_tb_rank)

# R0 -- pure-read controls ------------------------------------------------
log("R0 pure-read")
def read_rate(nbytes, reps=REPS):
    n = nbytes // 2
    a = mx.random.normal((n,)).astype(mx.float16); mx.eval(a)
    med, p, mn, mx_ = timeit(lambda: mx.sum(a), warm=3)
    return dict(arr_MB=nbytes / 1e6, ms_median=med, ms_p95=p, ms_min=mn, ms_max=mx_,
                read_gbs=nbytes / (med * 1e-3) / 1e9, read_gbs_p95=nbytes / (p * 1e-3) / 1e9)
res["read_slice"] = read_rate(sum_tb_rank)
res["read_512mb"] = read_rate(READ_MB * 1024 * 1024)
log(f"  read slice {sum_tb_rank/1e6:.1f}MB: {res['read_slice']['read_gbs']:.1f} GB/s   "
    f"read {READ_MB}MB: {res['read_512mb']['read_gbs']:.1f} GB/s (roof)")

# build decoded W (decode NOT in R1/R2 timed region) + fresh inputs ---------
log("decoding W (for R1) + fresh x")
xs = {}
sum_W = 0
for e in entries:
    w = decode_full_mlx(e["rt"].trellis, e["rt"].k, e["rt"].cb)
    mx.eval(w)
    e["W"] = w
    e["W_bytes"] = w.size * w.itemsize
    sum_W += e["W_bytes"]
    e["xs"] = {}
    for m in (1, 4):
        x = mx.array(np.random.RandomState(m * 977 + e["in_f"]).randn(m, e["in_f"]).astype(np.float16))
        mx.eval(x)
        e["xs"][m] = x
res["W_dtype"] = str(entries[0]["W"].dtype)
res["sum_W_bytes"] = sum_W
log(f"  decoded W total = {sum_W/1e6:.1f} MB  ({res['W_dtype']})  W/trellis = {sum_W/sum_tb_rank:.2f}x")

res["arms"] = {}
def arm(tag, fn, tb):
    med, p, mn, mx_ = timeit(fn)
    kmed, kp, kmn, kmx = timeit_k(fn)
    r = dict(ms_median=med, ms_p95=p, ms_min=mn, ms_max=mx_,
             kmed_percall=kmed, kp95_percall=kp,
             trellis_gbs=tb / (med * 1e-3) / 1e9,
             trellis_gbs_kp95=tb / (kp * 1e-3) / 1e9)
    res["arms"][tag] = r
    log(f"{tag:14s} whole {med:7.3f} ms ({tb/(med*1e-3)/1e9:5.1f} GB/s tB) | "
        f"K-batched/call med {kmed:7.3f} p95 {kp:7.3f} ms ({tb/(kp*1e-3)/1e9:5.1f} GB/s tB)")
    return r

for m in (1, 4):
    arm(f"prod_m{m}", (lambda mm: (lambda: [e["lin"](e["xs"][mm]) for e in entries]))(m), sum_tb_rank)
for m in (1, 4):
    arm(f"native_m{m}", (lambda mm: (lambda: [e["xs"][mm] @ e["W"] for e in entries]))(m), sum_tb_rank)
arm("decode_only", lambda: [decode_full_mlx(e["rt"].trellis, e["rt"].k, e["rt"].cb) for e in entries],
    sum_tb_rank)

# FORK --------------------------------------------------------------------
a = res["arms"]
dec = a["decode_only"]["kmed_percall"]
p4 = a["prod_m4"]["kmed_percall"]
n4 = a["native_m4"]["kmed_percall"]
p1 = a["prod_m1"]["kmed_percall"]
n1 = a["native_m1"]["kmed_percall"]
res["fork"] = dict(
    read_roof_gbs=res["read_512mb"]["read_gbs"],
    decode_share_m4=dec / p4, native_over_prod_m4=n4 / p4, native_over_prod_m1=n1 / p1,
    prod_m4_kmed=p4, native_m4_kmed=n4, decode_kmed=dec,
    native_m4_trellis_gbs=a["native_m4"]["trellis_gbs_kp95"],
    prod_m4_trellis_gbs=a["prod_m4"]["trellis_gbs_kp95"])
if res["fork"]["decode_share_m4"] >= 0.80:
    v = "CLOSE_DECODE_ALU_BOUND"
elif res["fork"]["decode_share_m4"] <= 0.50 and res["fork"]["native_over_prod_m4"] < 1.6:
    v = "PROCEED_LATENCY_BOUND"
else:
    v = "AMBIGUOUS_RUN_ALL"
res["fork"]["verdict"] = v
log(f"FORK: {v}  decode_share_m4={res['fork']['decode_share_m4']:.2f}  "
    f"native/prod m4={res['fork']['native_over_prod_m4']:.2f}  roof={res['fork']['read_roof_gbs']:.0f} GB/s")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, "w"), indent=1)
log(f"JSON -> {OUT}")
print("JSON_BEGIN"); print(json.dumps(res, indent=1)); print("JSON_END")
