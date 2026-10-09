#!/usr/bin/env python3
"""p20_pricing_q1_dense.py -- PRICING Q1: EXL3-fused dense kernel vs MLX NATIVE
quantized_matmul (q4/q5/q6, group_size=64) at the REAL DSv4.1 layer-20 dense
shapes on studio2, at m=1 and m=4.  REAL weights.  TP-SHARDED rank-0 geometry.

WHAT THIS ANSWERS (the question the prior 'dense/EXL3 structurally bound' closure
never asked): the prior fork compared the fused EXL3 kernel against an
UNQUANTIZED bf16 x@W, which streams ~5x the bytes.  The real alternative is a
PLAIN native-quant kernel (q4/q5/q6, hardware dequant, group_size=64) -- fewer
bpw-equivalent and no trellis ALU.  We PRICE it here (measure only; nothing is
re-quantized in production, the live model is untouched).

ARMS (all on the same per-rank sharded shapes):
  prod_m{m}        : e['lin'](x)                       EXL3-fused decode+matmul (+Hadamard)
  native_m{m}_B{b} : mx.quantized_matmul(x, q,s,b, transpose=True)  RAW kernel (no Hadamard)
  nativehad_m{m}_B{b}: prepare_xh -> qmm -> finish_y    FAIR drop-in (Hadamards kept)

TIMING (adopted from the pre-dispatch review; per-call-at-m=1 alone is
launch-overhead-dominated and can falsely show 'no difference'):
  whole   = ALL dense matmuls chained into ONE lazy graph, ONE mx.eval, wall time
            (per-call = whole / n_calls).
  K-batch = K whole-graph copies into one eval, /K -> per-call (eval floor buried);
            p95 reported, comparable to the prior 1.048 ms/layer figure.
  warm>=3, reps>=9.

QUALITY READ (fidelity to bf16, NOT model quality): cosine(y_q, x@W_decoded)
averaged over the roster, per B.  A qN re-quant of an ALREADY-2.9bpw-lossy tensor
measures fidelity to EXL3, not to the original model.

Runs ON a node (studio2) with the node's venv python + installed production mlx_lm.
No relaunch, no server restart, no POST, no writes to ~/repos/exo.

Env: PD_LAYER (20) PD_REPS (9) PD_K (8) PD_JSON PD_STDOUT PD_BITS (4,5,6).
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
from mlx_lm.models.exl3.gemv_metal import decode_full_mlx
from mlx_lm.models.deepseek_v41.exl3_build import _slice_dense

CK = os.environ.get(
    "PD_CK", HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw")
LAYER = int(os.environ.get("PD_LAYER", "20"))
REPS = int(os.environ.get("PD_REPS", "9"))
K = int(os.environ.get("PD_K", "8"))
BITS = [int(x) for x in os.environ.get("PD_BITS", "4,5,6").split(",")]
OUT = os.environ.get("PD_JSON", f"/tmp/pricing/p20_pricing_q1_dense_layer{LAYER}.json")
STDOUT = os.environ.get("PD_STDOUT", OUT + ".stdout.txt")
WORLD = 2          # TP=2
RANK = 0           # rank-0 geometry (matches production per-rank reads)

ROSTER = [
    "attn.wq_a", "attn.wq_b", "attn.wkv",
    "attn.wo_a.slice.0", "attn.wo_a.slice.1", "attn.wo_a.slice.2", "attn.wo_a.slice.3",
    "attn.wo_a.slice.4", "attn.wo_a.slice.5", "attn.wo_a.slice.6", "attn.wo_a.slice.7",
    "attn.wo_b", "attn.compressor.wkv", "attn.indexer.wk", "attn.indexer.wq_b",
    "ffn.shared_experts.w1", "ffn.shared_experts.w2", "ffn.shared_experts.w3",
]
# TP=2 rank-0 sharding (mirrors deepseek_v41/exl3_build.build_block):
#   attn.wq_b -> out, attn.wo_b -> in  (attn_tp)
#   ffn.shared_experts.w1 -> out, w2 -> in, w3 -> out  (shared_tp)
SHARD = {"attn.wq_b": "out", "attn.wo_b": "in",
         "ffn.shared_experts.w1": "out", "ffn.shared_experts.w2": "in",
         "ffn.shared_experts.w3": "out"}

_logf = open(STDOUT, "w")


def log(*a):
    line = "[q1] " + " ".join(str(x) for x in a)
    print(line, flush=True)
    _logf.write(line + "\n")
    _logf.flush()


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


def cos(a, b):
    a = np.asarray(a).reshape(-1).astype(np.float32)
    b = np.asarray(b).reshape(-1).astype(np.float32)
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ------------------------------------------------------------------ load ----
log(f"mlx {mx.__version__} loadavg={tuple(round(x, 2) for x in os.getloadavg())}")
ck = Exl3Checkpoint(CK)
entries, skipped = [], []
sum_tb_rank, sum_tb_full = 0, 0
for nm in ROSTER:
    full = f"layers.{LAYER}.{nm}"
    if not ck.has(full + ".trellis"):
        log("skip missing", full)
        continue
    lay = load_dense_layer(ck, full)
    tb_full = lay.trellis.size * lay.trellis.itemsize
    if SHARD.get(nm):
        lay = _slice_dense(lay, axis=SHARD[nm], rank=RANK, world=WORLD)
    lin = EXL3Linear(lay)
    rt = lin._rt
    tb_rank = rt.trellis.size * rt.trellis.itemsize
    sign_bytes = 0
    for s in (rt.suh, rt.svh):
        if s is not None:
            sign_bytes += s.size * s.itemsize
    sum_tb_full += tb_full
    sum_tb_rank += tb_rank
    mx.eval(rt.trellis)
    entries.append(dict(nm=nm, lin=lin, rt=rt, tb_full=tb_full, tb_rank=tb_rank,
                        sign_bytes=sign_bytes, in_f=lin.in_features, out_f=lin.out_features,
                        sharded=SHARD.get(nm)))
log(f"{len(entries)} dense linears; trellis full={sum_tb_full/1e6:.1f} MB "
    f"rank0={sum_tb_rank/1e6:.1f} MB")

res = dict(layer=LAYER, reps=REPS, k_batch=K, mlx=mx.__version__, world=WORLD, rank=RANK,
           bits=BITS, sum_trellis_full_bytes=sum_tb_full, sum_trellis_rank0_bytes=sum_tb_rank)

# -------------------------------------------------- decode W + quantize -----
log("decoding W + building native-quant arms")
sum_quant_bytes = {B: 0 for B in BITS}
sum_W_bytes = 0
for e in entries:
    w = decode_full_mlx(e["rt"].trellis, e["rt"].k, e["rt"].cb)   # (in, out) fp16
    mx.eval(w)
    e["W"] = w
    e["W_bytes"] = w.size * w.itemsize
    sum_W_bytes += e["W_bytes"]
    # quantized_matmul(transpose=True) wants (out, in); group_size groups the last dim (=in)
    wt = mx.contiguous(w.T)                                        # (out, in)
    e["Wt"] = wt
    e["q"] = {}
    for B in BITS:
        if wt.shape[-1] % 64 != 0:
            skipped.append(dict(nm=e["nm"], bits=B, reason=f"in_features {wt.shape[-1]} % 64 != 0"))
            continue
        q, s, b = mx.quantize(wt, group_size=64, bits=B)
        mx.eval(q, s, b)
        nb = q.nbytes + s.nbytes + b.nbytes
        e["q"][B] = dict(q=q, s=s, b=b, nbytes=nb)
        sum_quant_bytes[B] += nb
res["sum_W_bytes"] = sum_W_bytes
res["sum_quant_bytes"] = sum_quant_bytes
res["skipped"] = skipped
log(f"decoded W total={sum_W_bytes/1e6:.1f} MB ; "
    f"quant(q4)={sum_quant_bytes.get(4,0)/1e6:.1f} MB q5={sum_quant_bytes.get(5,0)/1e6:.1f} MB "
    f"q6={sum_quant_bytes.get(6,0)/1e6:.1f} MB ; skipped={len(skipped)}")

# fresh inputs per m (no Hadamard prep here; the arms apply their own)
xs = {}
for m in (1, 4):
    xs[m] = {}
    for e in entries:
        x = mx.array(np.random.RandomState(m * 977 + e["in_f"]).randn(m, e["in_f"]).astype(np.float16))
        mx.eval(x)
        xs[m][e["nm"]] = x

# ------------------------------------------------------------------ arms ----
ncalls = len(entries)
res["arms"] = {}


def arm(tag, fn, bytes_moved, trellis_bytes):
    med, p, mn, mx_ = timeit(fn)
    kmed, kp, kmn, kmx = timeit_k(fn)
    r = dict(ms_median=med, ms_p95=p, ms_min=mn, ms_max=mx_,
             per_call_median=med / ncalls, kmed_percall=kmed, kp95_percall=kp,
             bytes_moved=bytes_moved, trellis_rank0_bytes=trellis_bytes,
             eff_gbs=bytes_moved / (med * 1e-3) / 1e9,
             eff_gbs_kp95=bytes_moved / (kp * 1e-3) / 1e9,
             trellis_gbs_kp95=trellis_bytes / (kp * 1e-3) / 1e9)
    res["arms"][tag] = r
    log(f"{tag:20s} whole {med:7.3f} ms ({r['eff_gbs']:5.1f} GB/s eff) | "
        f"K/call med {kmed:7.3f} p95 {kp:7.3f} ms (tB {r['trellis_gbs_kp95']:5.1f} GB/s)")
    return r


for m in (1, 4):
    arm(f"prod_m{m}", (lambda mm: (lambda: [e["lin"](xs[mm][e["nm"]]) for e in entries]))(m),
        sum_tb_rank, sum_tb_rank)

for m in (1, 4):
    for B in BITS:
        if not all(B in e["q"] for e in entries):
            continue
        # raw native kernel (as specified): no Hadamard
        arm(f"native_m{m}_B{B}",
            (lambda mm, bb: (lambda: [mx.quantized_matmul(xs[mm][e["nm"]], e["q"][bb]["q"],
                                                         e["q"][bb]["s"], e["q"][bb]["b"],
                                                         transpose=True, group_size=64, bits=bb)
                                     for e in entries]))(m, B),
            sum_quant_bytes[B], sum_tb_rank)
        # fair drop-in: keep the EXL3 pre/post Hadamard rotations
        arm(f"nativehad_m{m}_B{B}",
            (lambda mm, bb: (lambda: [e["rt"].finish_y(
                mx.quantized_matmul(e["rt"].prepare_xh(xs[mm][e["nm"]]), e["q"][bb]["q"],
                                    e["q"][bb]["s"], e["q"][bb]["b"],
                                    transpose=True, group_size=64, bits=bb))
                for e in entries]))(m, B),
            sum_quant_bytes[B], sum_tb_rank)

# ----------------------------------------------------------- cosine read ----
log("cosine fidelity (qN output vs bf16 x@W_decoded; NOT model quality)")
res["cosine"] = {}
for m in (1, 4):
    for B in BITS:
        if not all(B in e["q"] for e in entries):
            continue
        cs, cs_had = [], []
        for e in entries:
            ref = xs[m][e["nm"]] @ e["W"]                     # bf16/fp16 x@W (no Hadamard)
            yq = mx.quantized_matmul(xs[m][e["nm"]], e["q"][B]["q"], e["q"][B]["s"],
                                     e["q"][B]["b"], transpose=True, group_size=64, bits=B)
            mx.eval(ref, yq)
            cs.append(cos(yq, ref))
        res["cosine"][f"m{m}_B{B}"] = dict(mean=float(np.mean(cs)), min=float(np.min(cs)),
                                           max=float(np.max(cs)))
        log(f"  cos m{m} B{B}: mean {np.mean(cs):.5f} min {np.min(cs):.5f} max {np.max(cs):.5f}")

# ------------------------------------------------------ rate ratio + price --
log("rate ratios + projected ms/round")
DENSE_PER_ROUND = os.environ.get("PD_PER_ROUND", "40")
N_LAYERS = int(DENSE_PER_ROUND)
a = res["arms"]
res["price"] = dict(dense_layers_per_round=N_LAYERS)
per_round = {}
for m in (1, 4):
    p_whole = a[f"prod_m{m}"]["ms_median"]
    per_round[f"prod_m{m}"] = p_whole * N_LAYERS
    entry = dict(prod_whole_ms=p_whole, prod_km_percall_ms=a[f"prod_m{m}"]["kmed_percall"])
    for B in BITS:
        for armk in (f"native_m{m}_B{B}", f"nativehad_m{m}_B{B}"):
            if armk not in a:
                continue
            w = a[armk]["ms_median"]
            km = a[armk]["kmed_percall"]
            entry[armk + "_whole_ms"] = w
            entry[armk + "_km_percall_ms"] = km
            entry[armk + "_rate_ratio_whole"] = p_whole / w
            entry[armk + "_rate_ratio_kmean"] = a[f"prod_m{m}"]["kmed_percall"] / km
            per_round[armk] = w * N_LAYERS
    res["price"][f"m{m}"] = entry
res["price"]["projected_dense_ms_per_round"] = {k: v for k, v in per_round.items()}

# owner-decision trigger: qN rate at m=4 >= 2x EXL3 rate
trig_raw = max((res["price"]["m4"].get(f"native_m4_B{B}_rate_ratio_kmean", 0) for B in BITS), default=0)
trig_had = max((res["price"]["m4"].get(f"nativehad_m4_B{B}_rate_ratio_kmean", 0) for B in BITS), default=0)
res["owner_decision"] = dict(
    trigger_threshold_x=2.0,
    max_ratio_m4_native_raw=trig_raw,
    max_ratio_m4_native_hadamard=trig_had,
    triggered=bool(max(trig_raw, trig_had) >= 2.0),
    basis="per-call K-batched mean at m=4 (dense slice, rank-0 shapes)",
    note="quant-format change is QUALITY-GATED; NOT to be acted on. Measurement only.")
log(f"OWNER-DECISION: triggered={res['owner_decision']['triggered']} "
    f"(max ratio m4 raw={trig_raw:.2f}x hadamard={trig_had:.2f}x)")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, "w"), indent=1, default=str)
log(f"JSON -> {OUT}")
_logf.close()
print("JSON_BEGIN"); print(json.dumps(res, indent=1, default=str)); print("JSON_END")
