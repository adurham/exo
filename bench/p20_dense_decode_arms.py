#!/usr/bin/env python3
"""p20_dense_decode_arms.py -- PHASE-1 EXPERIMENT 4 (adapted): decode-MECHANISM env arms.

The dense dispatch exposes no prefetch/double-buffer parameter (v19b already measured
double-buffering as noise in the mm path), so R2-as-written is N/A.  What DOES exist are
in-scope, existing ENV knobs that change the DECODE mechanism itself -- the actual
remaining decode-side levers.  If NONE of them moves the plateau, the decode is at its
structural optimum and the ALU-bound close is confirmed from a second direction.

Arms (all stock production kernel, one env flip each; process re-execs per arm so the
module-level env read is fresh):
  stock            all defaults
  swar_off         EXL3_DECODE_SWAR=0   (the non-SWAR mul1 decode)
  lut_on           EXL3_GEMV_LUT=1      (LUT-based decode)
  fuse_post_on     EXL3_FUSE_POST=1     (post-Hadamard+svh fused, one dispatch fewer)
  simd_off         EXL3_GEMV_SIMD=0     (staged fallback GEMV; sanity, expected slower)

Reports per-call p95 ms and trellis-equiv GB/s at m=1 and m=4 for each arm.
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
OUT = os.environ.get("PD_JSON", f"{HOME}/p5-dense-ws/p20_dense_decode_arms_layer{LAYER}.json")
ARM = os.environ.get("P20_ARM", "stock")  # this process measures ONE arm; driver loops

ROSTER = ["attn.wq_a", "attn.wq_b", "attn.wkv",
          "attn.wo_a.slice.0", "attn.wo_a.slice.1", "attn.wo_a.slice.2", "attn.wo_a.slice.3",
          "attn.wo_a.slice.4", "attn.wo_a.slice.5", "attn.wo_a.slice.6", "attn.wo_a.slice.7",
          "attn.wo_b", "attn.compressor.wkv", "attn.indexer.wk", "attn.indexer.wq_b",
          "ffn.shared_experts.w1", "ffn.shared_experts.w2", "ffn.shared_experts.w3"]
SHARD = {"attn.wq_b": "out", "attn.wo_b": "in", "ffn.shared_experts.w1": "out",
         "ffn.shared_experts.w2": "in", "ffn.shared_experts.w3": "out"}


def log(*a):
    print(f"[decarm {ARM}]", *a, flush=True)


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

xs = {}
for m in (1, 4):
    xs[m] = {nm: mx.array(np.random.RandomState(m * 977 + lin.in_features).randn(m, lin.in_features).astype(np.float16))
             for nm, lin, tb in entries}
    for v in xs[m].values():
        mx.eval(v)

res = dict(layer=LAYER, reps=REPS, k=K, arm=ARM, split_gbs=None, arms={}
           )
for m in (1, 4):
    kmed, kp = timeit_k(lambda mm=m: [lin(xs[mm][nm]) for nm, lin, tb in entries])
    res["arms"][f"m{m}"] = dict(kmed_percall=kmed, kp95_percall=kp,
                                trellis_gbs_p95=sum_tb / (kp * 1e-3) / 1e9)
    log(f"m={m}: K/call p95 {kp:7.3f} ms  {sum_tb/(kp*1e-3)/1e9:5.1f} GB/s")
json.dump(res, open(OUT, "w"), indent=1)
log(f"JSON -> {OUT}")
print("JSON_BEGIN"); print(json.dumps(res, indent=1)); print("JSON_END")
