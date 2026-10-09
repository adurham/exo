#!/usr/bin/env python3
"""p20_q1_recheck_dense.py -- PRICING Q1 RECHECK on the CONVERTED / affine-path
tensors at the TP=2 rank-0 SHARDED shapes, on a node.  OFFLINE ONLY (no engine,
no cluster, no deploy).  Adds THE ACTUAL AFFINE-PATH ARM (what the engine's
AffineProj runs), which the prior Q1 run did not measure.

ARMS (per-rank sharded shapes, real layer-20 weights from the EXL3 ckpt):
  prod_m{m}      : e['lin'](x)                                EXL3 fused (baseline)
  native_m{m}_B{b}: mx.quantized_matmul(x, q,s,b, transpose)  RAW native, no Hadamard
  nativehad_m{m}_B{b}: prepare_xh -> qmm -> finish_y          FAIR drop-in (Hadamards kept)
  affine_m{m}_B{b}: reconstruct_public_mlx(sharded).T -> mx.quantize(gs=64,B) ->
                    plain mx.quantized_matmul(transpose=True)  == AffineProj (THE engine arm)

Geometry: layer-20 dense roster, TP=2 rank-0, EXACTLY as exl3_build.build_block:
  attn.wq_b  -> _slice_dense axis='out'    attn.wo_b -> axis='in'
  ffn.shared_experts.w1/w3 -> 'out', w2 -> 'in'
  attn.wo_a  -> HEAD-SPLIT: rank0 owns groups [0:gpr] (whole group slices, no cut)
(14 linears).  A FULL-18 roster (all 8 wo_a, prior script's roster) is ALSO run
for prod+affine6 to cross-check the prior 0.945 ms/layer baseline.

TIMING: whole-slice = all linears chained into ONE lazy graph + ONE mx.eval.
K-batch = K whole-graph copies /K -> per-call.  warm>=3 (whole) / 2 (K), reps>=9.
COSINE: per arm vs the matching bf16 reference: raw decode (no Hadamard) for
native-raw; reconstruct_public_mlx (rotations folded) for prod/nativehad/affine.

Env: PD_LAYER(20) PD_REPS(9) PD_K(8) PD_BITS(5,6) PD_CK PD_JSON PD_STDOUT.
"""
import json, os, sys, time, statistics

HOME = os.path.expanduser("~")
import numpy as np
import mlx.core as mx

from mlx_lm.models.exl3.exl3_linear import EXL3Linear
from mlx_lm.models.exl3.loader import Exl3Checkpoint, load_dense_layer
from mlx_lm.models.exl3.gemv_metal import decode_full_mlx
from mlx_lm.models.exl3.reconstruct import reconstruct_public_mlx
from mlx_lm.models.deepseek_v41.exl3_build import _slice_dense

CK = os.environ.get("PD_CK", HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw")
LAYER = int(os.environ.get("PD_LAYER", "20"))
REPS = int(os.environ.get("PD_REPS", "9"))
K = int(os.environ.get("PD_K", "8"))
BITS = [int(x) for x in os.environ.get("PD_BITS", "5,6").split(",")]
OUT = os.environ.get("PD_JSON", f"/tmp/pricing/q1_recheck_layer{LAYER}.json")
STDOUT = os.environ.get("PD_STDOUT", OUT + ".stdout.txt")
WORLD, RANK = 2, 0
NHEADS_DEFAULT_GROUPS = 8

# ---- rank-0 engine geometry (build_block rank!=world>1, DENSE_MODE=exl3) ----
GPR = 4  # o_groups(8) // world(2)
RANK0 = [
    ("attn.wq_a", None), ("attn.wq_b", "out"), ("attn.wkv", None),
] + [(f"attn.wo_a.slice.{i}", None) for i in range(GPR)] + [
    ("attn.wo_b", "in"), ("attn.compressor.wkv", None),
    ("attn.indexer.wk", None), ("attn.indexer.wq_b", None),
    ("ffn.shared_experts.w1", "out"), ("ffn.shared_experts.w2", "in"),
    ("ffn.shared_experts.w3", "out"),
]
# ---- prior-script roster (all 8 wo_a) for cross-check vs 0.945 baseline ----
FULL18 = [
    ("attn.wq_a", None), ("attn.wq_b", "out"), ("attn.wkv", None),
] + [(f"attn.wo_a.slice.{i}", None) for i in range(8)] + [
    ("attn.wo_b", "in"), ("attn.compressor.wkv", None),
    ("attn.indexer.wk", None), ("attn.indexer.wq_b", None),
    ("ffn.shared_experts.w1", "out"), ("ffn.shared_experts.w2", "in"),
    ("ffn.shared_experts.w3", "out"),
]

_logf = open(STDOUT, "w")


def log(*a):
    line = "[q1r] " + " ".join(str(x) for x in a)
    print(line, flush=True)
    _logf.write(line + "\n")
    _logf.flush()


def p95(vals):
    vs = sorted(vals)
    return vs[min(len(vs) - 1, int(round(0.95 * (len(vs) - 1))))]


def _stats(ts):
    return (statistics.median(ts) * 1e3, p95(ts) * 1e3, min(ts) * 1e3, max(ts) * 1e3)


def timeit(fn, reps=REPS, warm=3):
    for _ in range(warm):
        mx.eval(fn())
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); mx.eval(fn()); ts.append(time.perf_counter() - t0)
    if len(ts) > 2:
        ts = ts[1:]
    return _stats(ts)


def timeit_k(fn, k=K, reps=REPS, warm=2):
    for _ in range(warm):
        mx.eval(*[fn() for _ in range(k)])
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); mx.eval(*[fn() for _ in range(k)])
        ts.append((time.perf_counter() - t0) / k)
    if len(ts) > 2:
        ts = ts[1:]
    return _stats(ts)


def cosf(a, b):
    a = np.asarray(a).reshape(-1).astype(np.float32)
    b = np.asarray(b).reshape(-1).astype(np.float32)
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


log(f"mlx {mx.__version__} loadavg={tuple(round(x, 2) for x in os.getloadavg())}")
ck = Exl3Checkpoint(CK)

# ---------------------------------------------------------------- load ------
_FULLCACHE = {}


def load_full(name):
    if name not in _FULLCACHE:
        _FULLCACHE[name] = load_dense_layer(ck, f"layers.{LAYER}.{name}")
    return _FULLCACHE[name]


def build_entries(roster):
    entries, skipped = [], []
    for nm, axis in roster:
        full = f"layers.{LAYER}.{nm}"
        if not ck.has(full + ".trellis"):
            skipped.append(dict(nm=nm, reason="missing trellis")); continue
        full_lay = load_full(nm)
        lay = full_lay
        if axis in ("in", "out"):
            lay = _slice_dense(full_lay, axis=axis, rank=RANK, world=WORLD)
        try:
            lin = EXL3Linear(lay)
        except Exception as ex:
            skipped.append(dict(nm=nm, reason=f"EXL3Linear: {ex}")); continue
        rt = lin._rt
        e = dict(nm=nm, axis=axis, lay=lay, full_lay=full_lay, lin=lin, rt=rt,
                 in_f=lin.in_features, out_f=lin.out_features)
        entries.append(e)
    return entries, skipped


def decode_W_dec(e):
    w = decode_full_mlx(e["rt"].trellis, e["rt"].k, e["rt"].cb)   # (in,out) fp16, no Hadamard
    mx.eval(w)
    return w


def recon_W_rec(lay):
    w = reconstruct_public_mlx(lay)                               # (in,out) fp16, rotations folded
    mx.eval(w)
    return w


# =========================================================== RANK0 roster ====
entries, skipped = build_entries(RANK0)
log(f"RANK0 roster: {len(entries)} dense linears (wo_a head-split -> {GPR} groups); skipped={len(skipped)}")
res = dict(layer=LAYER, reps=REPS, k_batch=K, mlx=mx.__version__, world=WORLD, rank=RANK,
           bits=BITS, roster="rank0_engine_geometry", gpr=GPR, skipped_rank0=skipped)

# quantization arms
log("decoding W (raw) + reconstructing W (rotations folded) + quantizing arms")
sum_W_bytes = 0
sum_q = {B: 0 for B in BITS}
sum_qa = {B: 0 for B in BITS}
for e in entries:
    e["W_dec"] = decode_W_dec(e)                                 # raw decode ref
    e["W_rec"] = recon_W_rec(e["lay"])                           # reconstruct ref (== AffineProj source)
    if e["W_dec"].shape != e["W_rec"].shape:
        log(f"  !! shape mismatch {e['nm']}: dec {e['W_dec'].shape} rec {e['W_rec'].shape}")
    sum_W_bytes += e["W_rec"].size * e["W_rec"].itemsize
    wt = mx.contiguous(e["W_dec"].T)                             # native arms: quantize raw decode
    wrt = mx.contiguous(e["W_rec"].T)                            # affine arm: quantize reconstruct
    e["q"], e["qa"] = {}, {}
    for B in BITS:
        if wrt.shape[-1] % 64:
            skipped.append(dict(nm=e["nm"], bits=B, reason=f"in {wrt.shape[-1]} %64")); continue
        q, s, b = mx.quantize(wt, group_size=64, bits=B)
        mx.eval(q, s, b)
        e["q"][B] = (q, s, b); sum_q[B] += q.nbytes + s.nbytes + b.nbytes
        qa, sa, ba = mx.quantize(wrt, group_size=64, bits=B)     # == AffineProj.__init__
        mx.eval(qa, sa, ba)
        e["qa"][B] = (qa, sa, ba); sum_qa[B] += qa.nbytes + sa.nbytes + ba.nbytes
res["sum_Wrec_bytes"] = sum_W_bytes
res["sum_q_native_bytes"] = sum_q
res["sum_qa_affine_bytes"] = sum_qa
res["skipped_quant"] = skipped
log(f"reconstruct W total={sum_W_bytes/1e6:.1f} MB ; native q6={sum_q.get(6,0)/1e6:.1f} MB ; "
    f"affine q6={sum_qa.get(6,0)/1e6:.1f} MB ; skipped={len(skipped)}")

# equivalence check: reconstruct-FULL-then-slice  ==  slice-then-reconstruct
log("equiv check: reconstruct_public_mlx(full) sliced  vs  reconstruct(sliced)")
eq_ok = {}
for nm in ("attn.wq_b", "attn.wo_b", "ffn.shared_experts.w2"):
    try:
        fl = load_full(nm)
        wf = recon_W_rec(fl)
        if nm in ("ffn.shared_experts.w2",):
            ax, cut = "in", wf.shape[0] // 2
            wf_sl = wf[0:cut, :]
        else:
            ax = "out" if nm == "attn.wq_b" else "in"
            wf_sl = wf[:, 0:wf.shape[1] // 2] if ax == "out" else wf[0:wf.shape[0] // 2, :]
            cut = wf_sl.shape[1] if ax == "out" else wf_sl.shape[0]
        sl = _slice_dense(fl, axis=ax, rank=RANK, world=WORLD)
        ws = recon_W_rec(sl)
        eq_ok[nm] = dict(max_abs_rel=float((np.abs(np.asarray(wf_sl) - np.asarray(ws)) /
                        (np.abs(np.asarray(ws)) + 1e-3)).max()),
                        allclose=bool(np.allclose(np.asarray(wf_sl), np.asarray(ws), atol=2e-3, rtol=2e-2)))
    except Exception as ex:
        eq_ok[nm] = dict(error=str(ex))
res["equiv_reconstruct_slice"] = eq_ok
log("  equiv:", json.dumps(eq_ok))

# inputs
xs = {}
for m in (1, 4):
    xs[m] = {e["nm"]: mx.array(np.random.RandomState(m * 977 + e["in_f"]).randn(m, e["in_f"]).astype(np.float16))
             for e in entries}
    for e in entries:
        mx.eval(xs[m][e["nm"]])

ncalls = len(entries)
res["arms"] = {}


def arm(tag, fn, bytes_moved):
    try:
        med, p, mn, mx_ = timeit(fn)
        kmed, kp, kmn, kmx = timeit_k(fn)
    except Exception as ex:
        res["arms"][tag] = dict(error=str(ex)); log(f"{tag:22s} FAILED: {ex}"); return None
    r = dict(ms_median=med, ms_p95=p, ms_min=mn, ms_max=mx_,
             per_call_median=med / ncalls, kmed_percall=kmed, kp95_percall=kp,
             bytes_moved=bytes_moved, eff_gbs=bytes_moved / (med * 1e-3) / 1e9)
    res["arms"][tag] = r
    log(f"{tag:22s} whole {med:7.3f} ms ({r['eff_gbs']:5.1f} GB/s) | "
        f"K/call med {kmed:7.3f} p95 {kp:7.3f} ms")
    return r


def qmm_native(mm, bb):
    return lambda: [mx.quantized_matmul(xs[mm][e["nm"]], e["q"][bb][0], e["q"][bb][1],
                                        e["q"][bb][2], transpose=True, group_size=64, bits=bb) for e in entries]


def qmm_had(mm, bb):
    return lambda: [e["rt"].finish_y(mx.quantized_matmul(e["rt"].prepare_xh(xs[mm][e["nm"]]),
                                      e["q"][bb][0], e["q"][bb][1], e["q"][bb][2],
                                      transpose=True, group_size=64, bits=bb)) for e in entries]


def qmm_affine(mm, bb):
    return lambda: [mx.quantized_matmul(xs[mm][e["nm"]], e["qa"][bb][0], e["qa"][bb][1],
                                        e["qa"][bb][2], transpose=True, group_size=64, bits=bb) for e in entries]


for m in (1, 4):
    arm(f"prod_m{m}", (lambda mm: (lambda: [e["lin"](xs[mm][e["nm"]]) for e in entries]))(m), sum_W_bytes)

for m in (1, 4):
    for B in BITS:
        arm(f"native_m{m}_B{B}", qmm_native(m, B), sum_q.get(B, 0))
        arm(f"nativehad_m{m}_B{B}", qmm_had(m, B), sum_q.get(B, 0))
        arm(f"affine_m{m}_B{B}", qmm_affine(m, B), sum_qa.get(B, 0))

# ------------------------------------------------------------ cosine --------
log("cosine fidelity (arm output vs matching bf16 reference)")
res["cosine"] = {}
for m in (1, 4):
    for B in BITS:
        acc = {"prod_vs_rec": [], "native_vs_dec": [], "native_vs_rec": [],
               "nativehad_vs_rec": [], "affine_vs_rec": [], "affine_vs_dec": []}
        for e in entries:
            x = xs[m][e["nm"]]
            ref_dec = x @ e["W_dec"]; ref_rec = x @ e["W_rec"]; mx.eval(ref_dec, ref_rec)
            prod = e["lin"](x)
            nraw = mx.quantized_matmul(x, e["q"][B][0], e["q"][B][1], e["q"][B][2], transpose=True, group_size=64, bits=B)
            nhad = e["rt"].finish_y(mx.quantized_matmul(e["rt"].prepare_xh(x), e["q"][B][0], e["q"][B][1], e["q"][B][2], transpose=True, group_size=64, bits=B))
            aff = mx.quantized_matmul(x, e["qa"][B][0], e["qa"][B][1], e["qa"][B][2], transpose=True, group_size=64, bits=B)
            mx.eval(prod, nraw, nhad, aff)
            acc["prod_vs_rec"].append(cosf(prod, ref_rec))
            acc["native_vs_dec"].append(cosf(nraw, ref_dec))
            acc["native_vs_rec"].append(cosf(nraw, ref_rec))
            acc["nativehad_vs_rec"].append(cosf(nhad, ref_rec))
            acc["affine_vs_rec"].append(cosf(aff, ref_rec))
            acc["affine_vs_dec"].append(cosf(aff, ref_dec))
        res["cosine"][f"m{m}_B{B}"] = {k: dict(mean=float(np.mean(v)), min=float(np.min(v)),
                                              max=float(np.max(v))) for k, v in acc.items()}
        log(f"  m{m} B{B}: prod~rec {np.mean(acc['prod_vs_rec']):.5f} | native~dec {np.mean(acc['native_vs_dec']):.5f} | "
            f"nativehad~rec {np.mean(acc['nativehad_vs_rec']):.5f} | affine~rec {np.mean(acc['affine_vs_rec']):.5f} | "
            f"affine~dec {np.mean(acc['affine_vs_dec']):.5f}")

# --------------------------------------------------------- ratios + gate -----
a = res["arms"]
res["ratios_m4"] = {}
entry = {}
for B in BITS:
    for armk in (f"native_m4_B{B}", f"nativehad_m4_B{B}", f"affine_m4_B{B}"):
        if armk not in a or "error" in a[armk]:
            continue
        entry[armk + "_whole_ms"] = a[armk]["ms_median"]
        entry[armk + "_km_percall_ms"] = a[armk]["kmed_percall"]
        entry[armk + "_ratio_whole"] = a["prod_m4"]["ms_median"] / a[armk]["ms_median"]
        entry[armk + "_ratio_kmean"] = a["prod_m4"]["kmed_percall"] / a[armk]["kmed_percall"]
res["ratios_m4"] = entry
best_aff = max((entry.get(f"affine_m4_B{B}_ratio_whole", 0) for B in BITS), default=0)
best_aff_k = max((entry.get(f"affine_m4_B{B}_ratio_kmean", 0) for B in BITS), default=0)
best_had = max((entry.get(f"nativehad_m4_B{B}_ratio_whole", 0) for B in BITS), default=0)
res["gate_GA"] = dict(threshold_x=2.0, best_affine_ratio_whole=best_aff,
                      best_affine_ratio_kmean=best_aff_k, best_nativehad_ratio_whole=best_had,
                      passed_whole=bool(best_aff >= 2.0), passed_kmean=bool(best_aff_k >= 2.0),
                      basis="affine-path arm (AffineProj-equivalent) vs prod EXL3, m=4, rank-0 shapes")
log(f"GATE G-A: affine m4 whole max ratio={best_aff:.2f}x  Kmean={best_aff_k:.2f}x  "
    f"(nativehad {best_had:.2f}x) -> {'PASS' if max(best_aff, best_aff_k) >= 2.0 else 'FAIL'}")

# ==================================================== FULL18 cross-check =====
log("FULL-18 roster cross-check (prior script's roster: all 8 wo_a) -> prod + affine6")
res["full18"] = {}
try:
    e18, sk18 = build_entries(FULL18)
    res["full18"]["n"] = len(e18); res["full18"]["skipped"] = sk18
    for e in e18:
        e["W_dec"] = decode_W_dec(e)
        e["W_rec"] = recon_W_rec(e["lay"])
        e["qa"] = {}
        wrt = mx.contiguous(e["W_rec"].T)
        B = 6
        e["qa"][B] = mx.quantize(wrt, group_size=64, bits=B); mx.eval(e["qa"][B])
    n18 = len(e18)
    x18 = {m: {e["nm"]: mx.array(np.random.RandomState(m * 977 + e["in_f"]).randn(m, e["in_f"]).astype(np.float16)) for e in e18} for m in (1, 4)}
    for m in (1, 4):
        for e in e18:
            mx.eval(x18[m][e["nm"]])
    rp = timeit(lambda: [e["lin"](x18[4][e["nm"]]) for e in e18])
    rp_k = timeit_k(lambda: [e["lin"](x18[4][e["nm"]]) for e in e18])
    ra = timeit(lambda: [mx.quantized_matmul(x18[4][e["nm"]], e["qa"][6][0], e["qa"][6][1], e["qa"][6][2], transpose=True, group_size=64, bits=6) for e in e18])
    ra_k = timeit_k(lambda: [mx.quantized_matmul(x18[4][e["nm"]], e["qa"][6][0], e["qa"][6][1], e["qa"][6][2], transpose=True, group_size=64, bits=6) for e in e18])
    res["full18"]["prod_m4"] = dict(whole_ms=rp[0], kmed_percall=rp_k[0])
    res["full18"]["affine6_m4"] = dict(whole_ms=ra[0], kmed_percall=ra_k[0])
    res["full18"]["affine6_ratio_whole"] = rp[0] / ra[0]
    res["full18"]["affine6_ratio_kmean"] = rp_k[0] / ra_k[0]
    log(f"  FULL18 prod_m4 whole {rp[0]:.3f} ms (per-call {rp[0]/n18:.4f}); affine6 whole {ra[0]:.3f} "
        f"(per-call {ra[0]/n18:.4f}) ratio_whole {rp[0]/ra[0]:.2f}x Km {rp_k[0]/ra_k[0]:.2f}x")
except Exception as ex:
    res["full18"]["error"] = str(ex); log(f"  FULL18 cross-check failed: {ex}")

res["loadavg_after"] = list(os.getloadavg())
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, "w"), indent=1, default=str)
log(f"JSON -> {OUT} ; loadavg_after={tuple(round(x,2) for x in os.getloadavg())}")
_logf.close()
print("JSON_BEGIN"); print(json.dumps(res, indent=1, default=str)); print("JSON_END")
