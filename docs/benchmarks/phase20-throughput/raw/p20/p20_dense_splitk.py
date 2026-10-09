#!/usr/bin/env python3
"""p20_dense_splitk.py -- PHASE-1 EXPERIMENT 3: split-K (n_splits) sweep on the
PRODUCTION kernel via a CALL-TIME env override (P20_XSPLIT / P20_MIN_SPLIT_TILES).

In scope: this tunes the EXISTING kernel's launch parameter (grid.z / partial-sum
reduction) IN PLACE -- no new shader.  The numerics change only in the partial-sum
ORDER, so a win here is battery-gated (owner ruling), not identity-gated.

Requires the patched gemv_metal (apply_p20_nsplits.py) to be importable FIRST:
run with PYTHONPATH pointed at a WT whose mlx_lm carries the hook, OR set P20_PKG.
Reports per-call p95 ms and trellis-equiv GB/s at m=1 and m=4 for each split setting,
plus a cosine sanity check vs the stock arm (the kernel must still be correct).

Env: PD_LAYER (20) PD_REPS (9) PD_K (8) PD_JSON P20_ARMS ("0,32,16,8,4").
"""
import json, os, sys, time, statistics

HOME = os.path.expanduser("~")
for _p in (os.environ.get("P20_PKG"), os.environ.get("PD_PKG")):
    if _p:
        sys.path.insert(0, _p)

import numpy as np
import mlx.core as mx

import mlx_lm
from mlx_lm.models.exl3.exl3_linear import EXL3Linear
from mlx_lm.models.exl3.loader import Exl3Checkpoint, load_dense_layer

assert "P20_MIN_SPLIT_TILES" in open(mlx_lm.models.exl3.gemv_metal.__file__).read(), \
    "gemv_metal has no P20 hook -- run apply_p20_nsplits.py on this package first"

CK = os.environ.get("PD_CK", HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw")
LAYER = int(os.environ.get("PD_LAYER", "20"))
REPS = int(os.environ.get("PD_REPS", "9"))
K = int(os.environ.get("PD_K", "8"))
OUT = os.environ.get("PD_JSON", f"{HOME}/p5-dense-ws/p20_dense_splitk_layer{LAYER}.json")
ARMS = [int(x) for x in os.environ.get("P20_ARMS", "0,32,16,8,4").split(",")]  # 0 = stock

ROSTER = ["attn.wq_a", "attn.wq_b", "attn.wkv",
          "attn.wo_a.slice.0", "attn.wo_a.slice.1", "attn.wo_a.slice.2", "attn.wo_a.slice.3",
          "attn.wo_a.slice.4", "attn.wo_a.slice.5", "attn.wo_a.slice.6", "attn.wo_a.slice.7",
          "attn.wo_b", "attn.compressor.wkv", "attn.indexer.wk", "attn.indexer.wq_b",
          "ffn.shared_experts.w1", "ffn.shared_experts.w2", "ffn.shared_experts.w3"]
SHARD = {"attn.wq_b": "out", "attn.wo_b": "in", "ffn.shared_experts.w1": "out",
         "ffn.shared_experts.w2": "in", "ffn.shared_experts.w3": "out"}


def log(*a):
    print("[splitk]", *a, flush=True)


def p95(vals):
    vs = sorted(vals)
    return vs[min(len(vs) - 1, int(round(0.95 * (len(vs) - 1))))]


def timeit_k(fn, k=K, reps=REPS, warm=2):
    for _ in range(warm):
        mx.eval(*[fn() for _ in range(k)])
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); mx.eval(*[fn() for _ in range(k)]); ts.append((time.perf_counter() - t0) / k)
    if len(ts) > 2:
        ts = ts[1:]
    return statistics.median(ts) * 1e3, p95(ts) * 1e3


log(f"mlx {mx.__version__} pkg={mlx_lm.__file__}")
ck = Exl3Checkpoint(CK)
entries, sum_tb = [], 0
for nm in ROSTER:
    full = f"layers.{LAYER}.{nm}"
    if not ck.has(full + ".trellis"):
        continue
    lin = EXL3Linear(load_dense_layer(ck, full))
    tb = lin._rt.trellis.size * lin._rt.trellis.itemsize
    tb = tb // 2 if SHARD.get(nm) else tb
    sum_tb += tb
    mx.eval(lin._rt.trellis)
    entries.append((nm, lin, tb))

xs, refs = {}, {}
for m in (1, 4):
    xs[m] = {}
    refs[m] = {}
    for nm, lin, tb in entries:
        x = mx.array(np.random.RandomState(m * 977 + lin.in_features).randn(m, lin.in_features).astype(np.float16))
        mx.eval(x); xs[m][nm] = x

res = dict(layer=LAYER, reps=REPS, k=K, arms={}, split_gbs={}, sanity={})

# reference outputs at STOCK splits
os.environ.pop("P20_XSPLIT", None); os.environ.pop("P20_MIN_SPLIT_TILES", None)
for m in (1, 4):
    for nm, lin, tb in entries:
        y = lin(xs[m][nm]).astype(mx.float32); mx.eval(y); refs[m][nm] = y

def run(m):
    return [lin(xs[m][nm]) for nm, lin, tb in entries]

for arm in ARMS:
    os.environ.pop("P20_XSPLIT", None); os.environ.pop("P20_MIN_SPLIT_TILES", None)
    if arm > 0:
        os.environ["P20_XSPLIT"] = str(arm)
    res["arms"][str(arm)] = {}
    for m in (1, 4):
        kmed, kp = timeit_k(lambda mm=m: run(mm))
        gbs = sum_tb / (kp * 1e-3) / 1e9
        res["arms"][str(arm)][f"m{m}"] = dict(kmed_percall=kmed, kp95_percall=kp, trellis_gbs_p95=gbs)
        log(f"arm XSPLIT={arm:3d} m={m}: K/call p95 {kp:7.3f} ms  {gbs:5.1f} GB/s")
    # sanity: cosine vs stock on a big tile
    nm0 = "attn.wo_b"
    e0 = next(e for e in entries if e[0] == nm0)
    for m in (1, 4):
        y = e0[1](xs[m][nm0]).astype(mx.float32); mx.eval(y)
        cos = (mx.sum(y * refs[m][nm0]) / (mx.linalg.norm(y) * mx.linalg.norm(refs[m][nm0]))).item()
        res["sanity"][f"arm{arm}_m{m}_cos"] = round(cos, 8)
    log(f"  sanity cos m1={res['sanity'][f'arm{arm}_m1_cos']:.7f} m4={res['sanity'][f'arm{arm}_m4_cos']:.7f}")

os.environ.pop("P20_XSPLIT", None)
json.dump(res, open(OUT, "w"), indent=1)
log(f"JSON -> {OUT}")
print("JSON_BEGIN"); print(json.dumps(res, indent=1)); print("JSON_END")
