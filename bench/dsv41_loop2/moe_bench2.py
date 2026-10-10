#!/usr/bin/env python3
"""EXL3 routed-expert (MoE) prefill microbench -- macstudio-m4-2 node venv.

Bench ONLY (no relaunch/kill). Loads a subset of the REAL checkpoint's experts
for ONE MoE layer and measures:
  Phase 0  ceilings: peak dense bf16 mx.matmul TFLOPS + peak GB/s
  Phase 1/2 M-curve on the production segmented path (_prefill -> inner_mm_seg_mlx)
  Phase 2b decode-only floor
  Phase 3  arms at M=64/M=128: (a) current segmented, (b) dequant-to-fp16 in-region
           + dense per-expert matmul (rotations applied), interleaved A/B/A/B
  Phase 4  block-level mock, one layer, production-skewed routing, x40 extrapolation
Wall clock only (gpu_time_ns is dead on this MLX build); every timed region ends
in mx.eval.  Incremental JSON so partial results survive.
"""
import os, sys, json, time, traceback, subprocess
import numpy as np

os.environ.setdefault("EXL3_MM_SEG", "v19d")   # production default
HOME = os.path.expanduser("~")
sys.path.insert(0, HOME + "/repos/exo")
sys.path.insert(0, HOME + "/repos/exo/mlx-lm")

import mlx.core as mx
from mlx_lm.models.exl3 import gemv_metal as G
from mlx_lm.models.exl3 import exl3_moe as MOE
from mlx_lm.models.exl3.loader import Exl3Checkpoint, load_experts
from mlx_lm.models.exl3.gemv_metal import decode_full_mlx

CK    = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
LAYER = int(os.environ.get("MB_LAYER", "0"))    # k=3 (2.9bpw-nominal) layer
REPS  = int(os.environ.get("MB_REPS", "7"))
JSON  = os.environ.get("MB_JSON", "/tmp/moe_bench.json")
E_LOAD  = int(os.environ.get("MB_E_LOAD", "64"))
E_BLOCK = int(os.environ.get("MB_E_BLOCK", "192"))
E_ARM   = int(os.environ.get("MB_E_ARM", "32"))
KK = 6

R = {"started": time.strftime("%Y-%m-%d %H:%M:%S"),
     "host": subprocess.run(["hostname"], capture_output=True, text=True).stdout.strip(),
     "env": {"seg": G._MM_SEG_VERSION, "BM": G._MM_BM, "BN": G._MM_BN,
             "MM_MAX_ROWS": MOE._MM_MAX_ROWS, "layer": LAYER,
             "E_LOAD": E_LOAD, "E_BLOCK": E_BLOCK, "E_ARM": E_ARM, "reps": REPS},
     "cal": {}, "geometry": {}, "layer_kmap": {}, "curve": [], "curve_traffic": [],
     "arms": {}, "block": {}, "diag": {}, "notes": []}

def log(*a): print("[bench]", *a, flush=True)
def flush():
    try: json.dump(R, open(JSON, "w"), indent=1, default=float)
    except Exception as e: log("json flush failed", e)
def drain(v):
    if isinstance(v, (tuple, list)): mx.eval(*v)
    else: mx.eval(v)
def tmed(fn, reps=None, warm=3):
    reps = reps or REPS
    for _ in range(warm): drain(fn())
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); drain(fn()); ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts)//2], ts[len(ts)//4], ts[max(0, (3*len(ts))//4 - 1)]

# ---------------------------------------------------------------- phase 0
def calib():
    log("=== PHASE 0: ceilings ===")
    nb = 128 * 1024 * 1024
    a = mx.ones((nb,), mx.float32); b = mx.ones_like(a)
    mx.eval(a, b)
    t, _, _ = tmed(lambda: mx.add(a, b)); R["cal"]["add_GBps"] = 3.0*nb*4/t/1e9
    t, _, _ = tmed(lambda: mx.multiply(a, 1.0)); R["cal"]["copy_GBps"] = 2.0*nb*4/t/1e9
    t, _, _ = tmed(lambda: mx.sum(a)); R["cal"]["reduce_GBps"] = 1.0*nb*4/t/1e9
    del a, b; mx.clear_cache()
    log("  BW add=%.0f copy=%.0f reduce=%.0f GB/s" % (R["cal"]["add_GBps"],
        R["cal"]["copy_GBps"], R["cal"]["reduce_GBps"]))
    best = 0.0; mm = []
    for (M, N, K) in [(1024,4096,4096),(2048,4096,4096),(4096,4096,4096),
                      (8192,4096,4096),(2048,8192,8192),(4096,8192,8192)]:
        A = mx.random.normal((M, K)).astype(mx.bfloat16)
        B = mx.random.normal((K, N)).astype(mx.bfloat16); mx.eval(A, B)
        t, _, _ = tmed(lambda: A @ B, reps=max(3, REPS-4), warm=2)
        tf = 2.0*M*N*K/t/1e12; best = max(best, tf)
        mm.append({"M": M, "N": N, "K": K, "ms": t*1e3, "TFLOPS": tf})
        log("  dense bf16 %dx%dx%d: %8.3f ms  %6.2f TFLOPS" % (M, N, K, t*1e3, tf))
        del A, B; mx.clear_cache()
    R["cal"]["matmul"] = mm; R["cal"]["peak_TFLOPS"] = best
    flush()

def layer_kmap(ck):
    log("=== layer bit-width map ===")
    km = {}
    for L in range(40):
        try:
            p = ck.header("layers.%d.ffn.experts.0.w1.trellis" % L)["shape"][2]
            km[L] = p * 16 // 256
        except Exception:
            km[L] = None
    R["layer_kmap"] = km
    log("  k per layer:", {L: k for L, k in km.items()})
    flush()

# ---------------------------------------------------------------- helpers
def build_idx(E_sel, M, rng):
    ids = np.repeat(np.arange(E_sel, dtype=np.int32), M)
    rng.shuffle(ids)
    tokens = (ids.size + KK - 1) // KK
    pad = tokens*KK - ids.size
    if pad: ids = np.concatenate([ids, np.zeros(pad, np.int32)])
    return ids.reshape(1, tokens, KK)

def build_idx_skew(E_sel, N, rng, alpha=0.9, hot_frac=0.35):
    nhot = max(1, int(round(E_sel*hot_frac)))
    w = np.zeros(E_sel); w[:nhot] = 1.0/(np.arange(1, nhot+1)**alpha)
    w[nhot:] = w[nhot-1]*0.25 if nhot < E_sel else 0.0
    w = w/w.sum()
    ids = rng.choice(E_sel, size=N, replace=True, p=w).astype(np.int32)
    tokens = (N + KK - 1)//KK
    pad = tokens*KK - N
    if pad: ids = np.concatenate([ids, np.zeros(pad, np.int32)])
    return ids.reshape(1, tokens, KK)

def stats(idx, E):
    flat = np.asarray(idx).reshape(-1)
    c = np.bincount(flat, minlength=E)
    nb = int(np.sum((c + G._MM_BM - 1)//G._MM_BM))
    live = int((c > 0).sum())
    return {"pairs": int(flat.size), "tokens": int(idx.shape[1]), "experts_hit": live,
            "avg_rows_per_expert": float(flat.size/max(1, live)),
            "max_rows_per_expert": int(c.max()), "blocks_bm64": nb,
            "block_rows": nb*G._MM_BM, "pad_waste": float(nb*G._MM_BM/flat.size)}

def per_expert_bytes(sg):
    E = sg.num_experts
    return {"gu": sg._gu_trellis.size*2/E, "dn": sg._dn_trellis.size*2/E,
            "tot": (sg._gu_trellis.size + sg._dn_trellis.size)*2/E}

def fpp(D, H):   # FLOP per (token,slot) pair: gate+up+down = 6*D*H
    return 6.0*D*H

# ---------------------------------------------------------------- phase 1/2
def curve(sg, rng):
    log("=== PHASE 1/2: geometry + M-curve ===")
    E, D, H = sg.num_experts, sg.input_dims, sg.hidden_dims
    peb = per_expert_bytes(sg)
    R["geometry"] = {"E_loaded": E, "D": D, "H": H, "k": sg._k,
                     "gu_tiles": sg._gu_tiles, "dn_tiles": sg._dn_tiles,
                     "activation": sg._activation,
                     "per_expert_MB": {k2: v/1e6 for k2, v in peb.items()},
                     "working_set_MB": peb["tot"]*E/1e6}
    log("  loaded E=%d D=%d H=%d k=%d gu_tiles=%d dn_tiles=%d per_expert=%.2fMB WS=%.0fMB"
        % (E, D, H, sg._k, sg._gu_tiles, sg._dn_tiles, peb["tot"]/1e6, peb["tot"]*E/1e6))
    F = fpp(D, H)
    for M, E_sel in [(16, E), (32, E), (64, E), (128, E), (256, E), (2048, min(E, 8))]:
        idx = build_idx(E_sel, M, rng); st = stats(idx, E)
        x = mx.random.normal((1, idx.shape[1], D)).astype(mx.float16)
        m = mx.array(idx, dtype=mx.uint32); mx.eval(x, m)
        try:
            drain(sg._prefill(x, m))
            t, q1, q3 = tmed(lambda: sg._prefill(x, m))
            dec_bytes = peb["tot"]*st["blocks_bm64"]
            rec = {"target_M": M, "E_sel": E_sel, **st, "ms": t*1e3,
                   "iqr_ms": [q1*1e3, q3*1e3], "TFLOPS": F*st["pairs"]/t/1e12,
                   "GBps_decode_traffic": dec_bytes/1e9/t,
                   "decode_traffic_MB": dec_bytes/1e6,
                   "redecode_factor": st["blocks_bm64"]/max(1, st["experts_hit"]),
                   "us_per_expert": t*1e6/max(1, st["experts_hit"])}
            R["curve"].append(rec)
            log("  M~%5d E_sel=%3d pairs=%7d avgM=%6.1f blk=%5d waste=%.2fx redec=%.1fx "
                "-> %8.2f ms %6.2f TF %6.0f decGB/s" % (M, E_sel, st["pairs"],
                st["avg_rows_per_expert"], st["blocks_bm64"], st["pad_waste"],
                rec["redecode_factor"], rec["ms"], rec["TFLOPS"], rec["GBps_decode_traffic"]))
        except Exception as e:
            log("  M~%d FAILED: %s: %s" % (M, type(e).__name__, e)); R["notes"].append("curve M=%d: %s" % (M, e))
        del x, m; mx.clear_cache(); flush()

def decode_floor(sg):
    log("=== PHASE 2b: decode-only floor ===")
    k, cb = sg._k, sg._cb
    for tag, tr, tpe in (("gu", sg._gu_trellis, sg._gu_tiles),
                         ("dn", sg._dn_trellis, sg._dn_tiles)):
        try:
            drain(decode_full_mlx(tr, k, cb, tn_offset=0, tn_count=2*tpe))
            t, _, _ = tmed(lambda: decode_full_mlx(tr, k, cb, tn_offset=0, tn_count=2*tpe),
                           reps=5, warm=1)
            src = tr.size*2/1e6
            out = (tr.shape[0]*16)*(2*tpe*16)*2/1e6
            R["diag"]["decode_%s_ms_2exp" % tag] = t*1e3
            R["diag"]["decode_%s_src_GBps" % tag] = src/1e3/t
            R["diag"]["decode_%s_out_GBps" % tag] = out/1e3/t
            log("  decode %s 2 experts: %.3f ms  src %.0f GB/s  out %.0f GB/s"
                % (tag, t*1e3, src/1e3/t, out/1e3/t))
        except Exception as e:
            log("  decode %s failed: %s" % (tag, e)); R["notes"].append("decode_floor %s: %s" % (tag, e))
        mx.clear_cache()
    flush()

# ---------------------------------------------------------------- phase 3
def _silu_clamp(g, u, lim=10.0):
    g = mx.minimum(g, lim); u = mx.clip(u, -lim, lim)
    return (g/(1.0 + mx.exp(-g)))*u

def arm_dense(x, idx, sg):
    """Dequant-to-fp16 inside the timed region + dense per-expert matmul, with the
    EXL3 Hadamard rotations applied exactly as production _prefill does."""
    E, D, H = sg.num_experts, sg.input_dims, sg.hidden_dims
    k, cb = sg._k, sg._cb
    gt, dt = sg._gu_tiles, sg._dn_tiles
    xf = x.reshape(-1, D)
    flat = np.asarray(idx).reshape(-1).astype(np.int64)
    order = mx.argsort(mx.array(flat.astype(np.uint32)))
    sidx = mx.array(flat.astype(np.uint32))[order]
    sid_np = np.asarray(sidx).astype(np.int64)
    tok = (mx.arange(flat.size, dtype=mx.uint32)//KK)[order]
    xg_all = xf[tok]
    outs = []
    for e in np.unique(flat):
        e = int(e)
        lo = int(np.searchsorted(sid_np, e, "left")); hi = int(np.searchsorted(sid_np, e, "right"))
        if hi <= lo: continue
        Re = hi - lo
        xe = xg_all[lo:hi]
        xrg = MOE._rows_prep()(xe, mx.broadcast_to(sg._gu_suh[e, 0][None, :], (Re, D)))
        xru = MOE._rows_prep()(xe, mx.broadcast_to(sg._gu_suh[e, 1][None, :], (Re, D)))
        wg = decode_full_mlx(sg._gu_trellis, k, cb, tn_offset=e*gt, tn_count=gt)
        wu = decode_full_mlx(sg._gu_trellis, k, cb, tn_offset=(E + e)*gt, tn_count=gt)
        g = MOE._rows_finish()(xrg @ wg, mx.broadcast_to(sg._gu_svh[e, :H][None, :], (Re, H)))
        u = MOE._rows_finish()(xru @ wu, mx.broadcast_to(sg._gu_svh[e, H:][None, :], (Re, H)))
        h = _silu_clamp(g.astype(mx.float32), u.astype(mx.float32)).astype(mx.float16)
        xrh = MOE._rows_prep()(h, mx.broadcast_to(sg._dn_suh[e][None, :], (Re, H)))
        wd = decode_full_mlx(sg._dn_trellis, k, cb, tn_offset=e*dt, tn_count=dt)
        y = MOE._rows_finish()(xrh @ wd, mx.broadcast_to(sg._dn_svh[e][None, :], (Re, D)))
        outs.append(y)
        del wg, wu, wd
    y = mx.concatenate(outs, axis=0)[mx.argsort(order)]
    return y.reshape(1, int(idx.shape[1]), KK, D)

def arms(sg, rng):
    log("=== PHASE 3: A/B arms (M=64, M=128; interleaved A/B/A/B) ===")
    E, D, H = sg.num_experts, sg.input_dims, sg.hidden_dims
    F = fpp(D, H)
    for M in (64, 128):
        E_sel = min(E, E_ARM)
        idx = build_idx(E_sel, M, rng); st = stats(idx, E)
        x = mx.random.normal((1, idx.shape[1], D)).astype(mx.float16)
        m = mx.array(idx, dtype=mx.uint32); mx.eval(x, m)
        fa = lambda: sg._prefill(x, m)
        fb = lambda: arm_dense(x, m, sg)
        try:
            ya = sg._prefill(x, m); yb = arm_dense(x, m, sg); mx.eval(ya, yb)
            rel = float(mx.max(mx.abs(ya.astype(mx.float32) - yb.astype(mx.float32))) /
                        (mx.max(mx.abs(ya.astype(mx.float32))) + 1e-9))
            R["arms"].setdefault("M%d" % M, {})["rel_max_err"] = rel
            log("  M%d: rel max err (seg vs dense) = %.3e" % (M, rel))
        except Exception as e:
            log("  M%d: correctness check failed %s" % (M, e)); R["notes"].append("arm M=%d chk: %s" % (M, e))
        del ya, yb; mx.clear_cache()
        for _ in range(2): drain(fa()); drain(fb())
        ta, tb = [], []
        for _ in range(max(5, REPS)):
            t0 = time.perf_counter(); drain(fa()); ta.append(time.perf_counter()-t0)
            t0 = time.perf_counter(); drain(fb()); tb.append(time.perf_counter()-t0)
        ta.sort(); tb.sort()
        iqr = lambda v: [v[len(v)//4]*1e3, v[(3*len(v))//4]*1e3]
        am, bm = ta[len(ta)//2], tb[len(tb)//2]
        d = {"M": M, "E_sel": E_sel, **st,
             "seg_ms": am*1e3, "seg_iqr_ms": iqr(ta), "seg_TFLOPS": F*st["pairs"]/am/1e12,
             "dense_ms": bm*1e3, "dense_iqr_ms": iqr(tb), "dense_TFLOPS": F*st["pairs"]/bm/1e12,
             "seg_over_dense": bm/am}
        R["arms"].setdefault("M%d" % M, {}).update(d)
        log("  M%d: seg %7.2f ms (%5.2f TF) | dense %7.2f ms (%5.2f TF) | seg/dense=%.2fx"
            % (M, d["seg_ms"], d["seg_TFLOPS"], d["dense_ms"], d["dense_TFLOPS"], d["seg_over_dense"]))
        del x, m; mx.clear_cache(); flush()

# ---------------------------------------------------------------- phase 4
def block_level(sg, rng):
    log("=== PHASE 4: block-level mock (one layer, production-skewed routing) ===")
    E, D, H = sg.num_experts, sg.input_dims, sg.hidden_dims
    F = fpp(D, H); peb = per_expert_bytes(sg)["tot"]
    variants = {"meanM32": E*32, "pairs12288": 12288}
    out = {}
    for name, N in variants.items():
        idx = build_idx_skew(E, N, rng); st = stats(idx, E)
        x = mx.random.normal((1, idx.shape[1], D)).astype(mx.float16)
        m = mx.array(idx, dtype=mx.uint32); mx.eval(x, m)
        rec = {"E": E, **st, "model_E_prod": 384}
        pair_scale = 12288.0/st["pairs"]
        try:
            drain(sg._prefill(x, m))
            t, q1, q3 = tmed(lambda: sg._prefill(x, m))
            rec.update(seg_ms=t*1e3, seg_iqr_ms=[q1*1e3, q3*1e3],
                       seg_TFLOPS=F*st["pairs"]/t/1e12,
                       seg_GBps_decode=peb*st["blocks_bm64"]/1e9/t)
            rec["pair_scale_to_prod_per_rank"] = pair_scale
            rec["seg_ms_prod_per_rank_layer"] = t*1e3*pair_scale
            rec["seg_x40_ms_per_chunk"] = t*1e3*pair_scale*40
            log("  [%s] seg %.2f ms/layer-mock %.2f TF | x%.2f(pairs)x40 = %.1f ms/chunk"
                % (name, t*1e3, rec["seg_TFLOPS"], pair_scale, rec["seg_x40_ms_per_chunk"]))
        except Exception as e:
            rec["seg_err"] = str(e); R["notes"].append("block seg %s: %s" % (name, e)); log("  seg failed", e)
        try:
            drain(arm_dense(x, m, sg))
            t2, q1, q3 = tmed(lambda: arm_dense(x, m, sg), reps=5, warm=1)
            rec.update(dense_ms=t2*1e3, dense_iqr_ms=[q1*1e3, q3*1e3],
                       dense_TFLOPS=F*st["pairs"]/t2/1e12)
            rec["dense_ms_prod_per_rank_layer"] = t2*1e3*pair_scale
            rec["dense_x40_ms_per_chunk"] = t2*1e3*pair_scale*40
            if "seg_ms" in rec: rec["seg_over_dense"] = t2/t
            log("  [%s] dense %.2f ms/layer-mock %.2f TF | x40 = %.1f ms/chunk"
                % (name, t2*1e3, rec["dense_TFLOPS"], rec["dense_x40_ms_per_chunk"]))
        except Exception as e:
            rec["dense_err"] = str(e); R["notes"].append("block dense %s: %s" % (name, e)); log("  dense failed", e)
        out[name] = rec
        del x, m; mx.clear_cache(); flush()
    R["block"] = out

# ---------------------------------------------------------------- main
def main():
    log("checkpoint:", CK)
    calib()
    ck = Exl3Checkpoint(CK)
    layer_kmap(ck)
    try:
        sg = load_experts(ck, LAYER, n_experts=E_LOAD, rank=0, world=2, activation="silu_clamp")
        rng = np.random.RandomState(1234)
        curve(sg, rng); decode_floor(sg); arms(sg, rng)
        del sg; mx.clear_cache(); flush()
    except Exception as e:
        log("E_LOAD stage failed:", type(e).__name__, e); traceback.print_exc()
        R["notes"].append("E_LOAD stage: %s" % e)
    try:
        sg2 = load_experts(ck, LAYER, n_experts=E_BLOCK, rank=0, world=2, activation="silu_clamp")
        block_level(sg2, np.random.RandomState(99))
        del sg2; mx.clear_cache()
    except Exception as e:
        log("E_BLOCK stage failed:", type(e).__name__, e); traceback.print_exc()
        R["notes"].append("E_BLOCK stage: %s" % e)
    R["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    flush(); log("DONE ->", JSON)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("FATAL", type(e).__name__, e); traceback.print_exc()
        R["notes"].append("fatal: %s" % e); flush()
