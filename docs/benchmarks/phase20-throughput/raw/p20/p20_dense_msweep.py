#!/usr/bin/env python3
"""p20_dense_msweep.py -- PHASE-1 EXPERIMENT 2: m-sweep + auto-split-K report.

Measures the PRODUCTION EXL3 kernel (EXL3Linear / _run_inner_gem) over the layer-20
dense roster at m in {1,2,3,4,5,6,8,16} (whole-slice single-eval + K-batched per-call p95),
and reports the kernel's AUTO-SELECTED n_splits per projection (replicated from the
in-source formula at gemv_metal.py L1435-1443) so the split-K state is visible without a
patch.  Pure production path, REAL weights, no relaunch, no POST.

Purpose (given the fork already returned decode-ALU-bound): show the m-curve, and whether
m=4 is specifically degraded vs m=1, and whether the auto-tune leaves the GPU under-filled.

Env: PD_LAYER (20) PD_REPS (9) PD_K (8) PD_JSON.
"""
import json, os, sys, time, statistics

HOME = os.path.expanduser("~")
_pkg = os.environ.get("PD_PKG")
if _pkg:
    sys.path.insert(0, _pkg)

import numpy as np
import mlx.core as mx

from mlx_lm.models.exl3.exl3_linear import EXL3Linear
from mlx_lm.models.exl3.loader import Exl3Checkpoint, load_dense_layer

CK = os.environ.get("PD_CK", HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw")
LAYER = int(os.environ.get("PD_LAYER", "20"))
REPS = int(os.environ.get("PD_REPS", "9"))
K = int(os.environ.get("PD_K", "8"))
OUT = os.environ.get("PD_JSON", f"{HOME}/p5-dense-ws/p20_dense_msweep_layer{LAYER}.json")
MS = [int(x) for x in os.environ.get("PD_MS", "1,2,3,4,5,6,8,16").split(",")]

ROSTER = ["attn.wq_a", "attn.wq_b", "attn.wkv",
          "attn.wo_a.slice.0", "attn.wo_a.slice.1", "attn.wo_a.slice.2", "attn.wo_a.slice.3",
          "attn.wo_a.slice.4", "attn.wo_a.slice.5", "attn.wo_a.slice.6", "attn.wo_a.slice.7",
          "attn.wo_b", "attn.compressor.wkv", "attn.indexer.wk", "attn.indexer.wq_b",
          "ffn.shared_experts.w1", "ffn.shared_experts.w2", "ffn.shared_experts.w3"]
SHARD = {"attn.wq_b": "out", "attn.wo_b": "in", "ffn.shared_experts.w1": "out",
         "ffn.shared_experts.w2": "in", "ffn.shared_experts.w3": "out"}


def log(*a):
    print("[msweep]", *a, flush=True)


def p95(vals):
    vs = sorted(vals)
    return vs[min(len(vs) - 1, int(round(0.95 * (len(vs) - 1))))]


def auto_splits(in_tiles, out_tiles, batch):
    """Replica of gemv_metal.py L1410-1443 (simd/devx path)."""
    mt = 1 if batch == 1 else (2 if batch == 2 else (4 if batch <= 4 else 8))
    devx_ok = 1 < batch <= 16
    m_groups = (batch + mt - 1) // mt
    use_simd = True
    min_split_tiles = 128 if mt == 1 else 64
    if use_simd and mt > 1 and out_tiles <= 8:
        min_split_tiles = 8
    n_splits = 1
    while out_tiles * m_groups * n_splits < 8192 and in_tiles // (n_splits * 2) >= min_split_tiles:
        n_splits *= 2
    return n_splits, mt, min_split_tiles


ck = Exl3Checkpoint(CK)
entries, sum_tb_rank = [], 0
for nm in ROSTER:
    full = f"layers.{LAYER}.{nm}"
    if not ck.has(full + ".trellis"):
        continue
    lin = EXL3Linear(load_dense_layer(ck, full))
    tb = lin._rt.trellis.size * lin._rt.trellis.itemsize
    tb = tb // 2 if SHARD.get(nm) else tb
    sum_tb_rank += tb
    mx.eval(lin._rt.trellis)
    it, ot, _ = lin._rt.trellis.shape
    entries.append(dict(nm=nm, lin=lin, tb=tb, in_f=lin.in_features, out_f=lin.out_features,
                        in_tiles=it, out_tiles=ot))

res = dict(layer=LAYER, reps=REPS, k=K, sum_trellis_rank0_bytes=sum_tb_rank, arms={})

def timeit(fn, reps=REPS, warm=3):
    for _ in range(warm):
        mx.eval(fn())
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); mx.eval(fn()); ts.append(time.perf_counter() - t0)
    if len(ts) > 2:
        ts = ts[1:]
    return statistics.median(ts) * 1e3, p95(ts) * 1e3, min(ts) * 1e3

def timeit_k(fn, k=K, reps=REPS, warm=2):
    for _ in range(warm):
        mx.eval(*[fn() for _ in range(k)])
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); mx.eval(*[fn() for _ in range(k)]); ts.append((time.perf_counter() - t0) / k)
    if len(ts) > 2:
        ts = ts[1:]
    return statistics.median(ts) * 1e3, p95(ts) * 1e3

xs = {}
for m in MS:
    xs[m] = {e["nm"]: mx.array(np.random.RandomState(m * 977 + e["in_f"]).randn(m, e["in_f"]).astype(np.float16))
             for e in entries}
    for v in xs[m].values():
        mx.eval(v)

for m in MS:
    def run(mm=m):
        return [e["lin"](xs[mm][e["nm"]]) for e in entries]
    med, p, mn = timeit(run)
    kmed, kp = timeit_k(run)
    res["arms"][str(m)] = dict(ms_median=med, ms_p95=p, ms_min=mn, kmed_percall=kmed, kp95_percall=kp,
                               trellis_gbs=kmed and sum_tb_rank / (kmed * 1e-3) / 1e9 or None)
    log(f"m={m:2d}  whole {med:7.3f} ms ({sum_tb_rank/(med*1e-3)/1e9:5.1f} GB/s) | "
        f"K/call med {kmed:7.3f} p95 {kp:7.3f} ms ({sum_tb_rank/(kp*1e-3)/1e9:5.1f} GB/s)")

# auto-split report per shape at m=1 and m=4
res["auto_splits"] = {}
for e in entries:
    a1 = auto_splits(e["in_tiles"], e["out_tiles"], 1)
    a4 = auto_splits(e["in_tiles"], e["out_tiles"], 4)
    res["auto_splits"][e["nm"]] = dict(in_tiles=e["in_tiles"], out_tiles=e["out_tiles"],
                                       n_splits_m1=a1[0], n_splits_m4=a4[0],
                                       threadgroups_m4=e["out_tiles"] * a4[0])
    log(f"  split {e['nm']:22s} in_t={e['in_tiles']:4d} out_t={e['out_tiles']:4d} "
        f"nsplit m1={a1[0]:3d} m4={a4[0]:3d}  grid_m4={e['out_tiles']*a4[0]}")

json.dump(res, open(OUT, "w"), indent=1)
log(f"JSON -> {OUT}")
print("JSON_BEGIN"); print(json.dumps(res, indent=1)); print("JSON_END")
