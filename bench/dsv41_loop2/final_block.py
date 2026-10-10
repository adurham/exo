#!/usr/bin/env python3
"""Final block-level headline: one layer, 192 experts, production chunk (N=12288
pairs), current segmented path vs dequant+dense arm, uniform (production-like,
per pDE_segment) and skewed routing.  x40 extrapolation to per-chunk."""
import os, sys, json, time
import numpy as np
os.environ.setdefault("EXL3_MM_SEG", "v19d")
HOME = os.path.expanduser("~")
sys.path.insert(0, HOME + "/repos/exo"); sys.path.insert(0, HOME + "/repos/exo/mlx-lm")
import mlx.core as mx
from mlx_lm.models.exl3 import gemv_metal as G
from mlx_lm.models.exl3 import exl3_moe as MOE
from mlx_lm.models.exl3.loader import Exl3Checkpoint, load_experts
from mlx_lm.models.exl3.gemv_metal import decode_full_mlx, inner_mm_seg_mlx

CK = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
E_LOAD = int(os.environ.get("MB_E_BLOCK", "192")); KK = 6; REPS = int(os.environ.get("MB_REPS", "7"))
OUT = os.environ.get("MB_JSON", "/tmp/moe_final_block.json")
R = {"runs": [], "ceiling": {}}

def log(*a): print("[fin]", *a, flush=True)
def drain(v):
    if isinstance(v, (tuple, list)): mx.eval(*v)
    else: mx.eval(v)
def tmed(fn, reps=REPS, warm=3):
    for _ in range(warm): drain(fn())
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); drain(fn()); ts.append(time.perf_counter()-t0)
    ts.sort(); return ts[len(ts)//2], ts[len(ts)//4], ts[max(0, (3*len(ts))//4-1)]

def make_idx(E, N, rng, mode):
    if mode == "uniform":
        ids = np.repeat(np.arange(E), int(np.ceil(N/E)))[:N].astype(np.int32)
        rng.shuffle(ids)
    else:
        nhot = max(1, int(round(E*0.35))); w = np.zeros(E)
        w[:nhot] = 1.0/(np.arange(1, nhot+1)**0.9); w[nhot:] = w[nhot-1]*0.25
        w = w/w.sum(); ids = rng.choice(E, size=N, replace=True, p=w).astype(np.int32)
    tokens = (ids.size+KK-1)//KK; pad = tokens*KK-ids.size
    if pad: ids = np.concatenate([ids, np.zeros(pad, np.int32)])
    return ids.reshape(1, tokens, KK)

def _sc(g, u, lim=10.0):
    g = mx.minimum(g, lim); u = mx.clip(u, -lim, lim)
    return (g/(1.0+mx.exp(-g)))*u

def arm_dense(x, idx, sg):
    E, D, H = sg.num_experts, sg.input_dims, sg.hidden_dims
    k, cb, gt, dt = sg._k, sg._cb, sg._gu_tiles, sg._dn_tiles
    xf = x.reshape(-1, D); flat = np.asarray(idx).reshape(-1).astype(np.int64)
    order = mx.argsort(mx.array(flat.astype(np.uint32)))
    sid_np = np.asarray(mx.array(flat.astype(np.uint32))[order]).astype(np.int64)
    tok = (mx.arange(flat.size, dtype=mx.uint32)//KK)[order]
    xs = xf[tok]; outs = []
    for e in np.unique(flat):
        e = int(e); lo = int(np.searchsorted(sid_np, e, "left")); hi = int(np.searchsorted(sid_np, e, "right"))
        if hi <= lo: continue
        Re = hi-lo; xe = xs[lo:hi]
        xg = MOE._rows_prep()(xe, mx.broadcast_to(sg._gu_suh[e, 0][None], (Re, D)))
        xu = MOE._rows_prep()(xe, mx.broadcast_to(sg._gu_suh[e, 1][None], (Re, D)))
        wg = decode_full_mlx(sg._gu_trellis, k, cb, tn_offset=e*gt, tn_count=gt)
        wu = decode_full_mlx(sg._gu_trellis, k, cb, tn_offset=(E+e)*gt, tn_count=gt)
        g = MOE._rows_finish()(xg@wg, mx.broadcast_to(sg._gu_svh[e, :H][None], (Re, H)))
        u = MOE._rows_finish()(xu@wu, mx.broadcast_to(sg._gu_svh[e, H:][None], (Re, H)))
        h = _sc(g.astype(mx.float32), u.astype(mx.float32)).astype(mx.float16)
        xh = MOE._rows_prep()(h, mx.broadcast_to(sg._dn_suh[e][None], (Re, H)))
        wd = decode_full_mlx(sg._dn_trellis, k, cb, tn_offset=e*dt, tn_count=dt)
        outs.append(MOE._rows_finish()(xh@wd, mx.broadcast_to(sg._dn_svh[e][None], (Re, D))))
        del wg, wu, wd
    return mx.concatenate(outs, 0)[mx.argsort(order)].reshape(1, idx.shape[1], KK, D)

ck = Exl3Checkpoint(CK)
sg = load_experts(ck, 0, n_experts=E_LOAD, rank=0, world=2, activation="silu_clamp")
E, D, H = sg.num_experts, sg.input_dims, sg.hidden_dims
k, cb, gt, dt = sg._k, sg._cb, sg._gu_tiles, sg._dn_tiles
peb = (sg._gu_trellis.size + sg._dn_trellis.size)*2/E
F = 6.0*D*H
log("E=%d D=%d H=%d k=%d gt=%d dt=%d per_expert=%.2fMB" % (E, D, H, k, gt, dt, peb/1e6))
# ceilings
A = mx.random.normal((4096, 4096)).astype(mx.bfloat16); Bm = mx.random.normal((4096, 4096)).astype(mx.bfloat16)
mx.eval(A, Bm); R["ceiling"]["TFLOPS"] = 2.0*4096**3/tmed(lambda: A@Bm, 5, 2)[0]*1e-12
nb2 = 64*1024*1024; a = mx.ones((nb2,), mx.float32); b = mx.ones_like(a); mx.eval(a, b)
R["ceiling"]["GBps"] = 3.0*nb2*4/tmed(lambda: mx.add(a, b), 5, 2)[0]/1e9
log("ceiling %.2f TFLOPS / %.0f GB/s" % (R["ceiling"]["TFLOPS"], R["ceiling"]["GBps"]))

NPROD = 12288
for mode in ("uniform", "skew"):
    rng = np.random.RandomState(2026)
    idx = make_idx(E, NPROD, rng, mode)
    B, S, kk = idx.shape; Np = B*S*kk
    x = mx.random.normal((B, S, D)).astype(mx.float16); m = mx.array(idx, dtype=mx.uint32); mx.eval(x, m)
    counts = np.bincount(np.asarray(m).reshape(-1), minlength=E)
    nblk = int(np.sum((counts+63)//64))
    row = {"mode": mode, "N": int(Np), "tokens": int(S), "E": E, "blocks": nblk,
           "max_rows_per_expert": int(counts.max()), "avg_rows_per_expert": float(Np/int((counts>0).sum())),
           "waste": float(nblk*64/Np), "model_E_prod": 384}
    # correctness
    try:
        ya = sg._prefill(x, m); yb = arm_dense(x, m, sg); mx.eval(ya, yb)
        row["rel_max_err"] = float(mx.max(mx.abs(ya.astype(mx.float32)-yb.astype(mx.float32))) /
                                    (mx.max(mx.abs(ya.astype(mx.float32)))+1e-9))
        del ya, yb; mx.clear_cache()
    except Exception as e:
        row["chk_err"] = str(e)
    # A/B interleaved
    fa = lambda: sg._prefill(x, m); fb = lambda: arm_dense(x, m, sg)
    drain(fa()); drain(fb())
    ta, tb = [], []
    for _ in range(REPS):
        t0 = time.perf_counter(); drain(fa()); ta.append(time.perf_counter()-t0)
        t0 = time.perf_counter(); drain(fb()); tb.append(time.perf_counter()-t0)
    ta.sort(); tb.sort()
    am, bm = ta[len(ta)//2], tb[len(tb)//2]
    ps = NPROD/float(Np)
    row.update(seg_ms=am*1e3, seg_iqr_ms=[ta[len(ta)//4]*1e3, ta[(3*len(ta))//4]*1e3],
               seg_TFLOPS=F*Np/am/1e12, seg_GBps=peb*nblk/1e9/am,
               dense_ms=bm*1e3, dense_TFLOPS=F*Np/bm/1e12,
               seg_over_dense=bm/am,
               seg_x40_ms_chunk=am*1e3*ps*40, dense_x40_ms_chunk=bm*1e3*ps*40,
               seg_pct_of_ceiling=F*Np/am/1e12/R["ceiling"]["TFLOPS"]*100)
    R["runs"].append(row)
    log("[%s] blocks=%d waste=%.2f maxM=%d | seg %.2f ms (%.2fTF, %.0f%%ceil, %.0fGB/s) | "
        "dense %.2f ms (%.2fTF) | seg/dense=%.2fx | x40 seg=%.0fms" %
        (mode, nblk, row["waste"], int(counts.max()), row["seg_ms"], row["seg_TFLOPS"],
         row["seg_pct_of_ceiling"], row["seg_GBps"], row["dense_ms"], row["dense_TFLOPS"],
         row["seg_over_dense"], row["seg_x40_ms_chunk"]))
    del x, m; mx.clear_cache()
json.dump(R, open(OUT, "w"), indent=1, default=float)
log("wrote", OUT)
