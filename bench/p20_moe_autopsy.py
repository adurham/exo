#!/usr/bin/env python3
"""Phase-20 MoE expert-GEMM autopsy (Loop-3 prescribed) -- OFFLINE, single node.

Steps 1-4 of docs/PERFORMANCE_HISTORY.md:11394-11406. Runs in the node venv
(~/repos/exo/.venv/bin/python) with a SCRATCH copy of the instrumented mlx_lm
on PYTHONPATH. Never touches the exo server; checkpoint loads are read-only
(Exl3Checkpoint uses os.pread). Wall clock only (gpu_time_ns is dead on this
build); every timed region ends in an mx.eval drain; warmups drained.

  STAGE=1  ceilings + Step-1 production-chunk segmentation capture (real trace,
           E=384) + uniform/skew E=192 in-session timing + per-launch wall
  STAGE=3  Step-3 dequant-excluded arm (pre-converted fp16 W vs in-loop)
  STAGE=4  Step-4 balanced-assignment synthetic test
"""
import os, sys, json, time, socket, traceback
import numpy as np

os.environ.setdefault("EXL3_MM_SEG", "v19d")   # production default
HOME = os.path.expanduser("~")
# Scratch instrumented mlx_lm FIRST so neither the node src tree nor
# site-packages shadows it (production is never touched).
P20_PKG = os.environ.get("P20_PKG", "")
if P20_PKG:
    sys.path.insert(0, P20_PKG)
sys.path.insert(0, HOME + "/repos/exo")

import mlx.core as mx
from mlx_lm.models.exl3 import gemv_metal as G
from mlx_lm.models.exl3 import exl3_moe as MOE
from mlx_lm.models.exl3.loader import Exl3Checkpoint, load_experts
from mlx_lm.models.exl3.gemv_metal import decode_full_mlx, inner_mm_seg_mlx

CK = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
TRACE = os.environ.get("MB_TRACE", "/tmp/p20moe/p30_exl3_trace_clamp.json")
STAGE = os.environ.get("STAGE", "1")
KK = 6
REPS = int(os.environ.get("MB_REPS", "7"))
E_BLOCK = int(os.environ.get("MB_E_BLOCK", "192"))
NPROD = 12288
OUT = os.environ.get("MB_JSON", "/tmp/p20moe/p20_moe_autopsy_stage%s.json" % STAGE)

R = {"stage": STAGE, "host": socket.gethostname(),
     "started": time.strftime("%Y-%m-%d %H:%M:%S"),
     "env": {"seg": G._MM_SEG_VERSION, "BM": G._MM_BM, "BN": G._MM_BN,
             "GEM_THREADS": G._GEM_THREADS, "MM_MAX_ROWS": MOE._MM_MAX_ROWS,
             "reps": REPS, "E_BLOCK": E_BLOCK, "NPROD": NPROD},
     "ceil": {}, "notes": []}


def log(*a):
    print("[autopsy]", *a, flush=True)


def flushjson():
    try:
        json.dump(R, open(OUT, "w"), indent=1, default=float)
    except Exception as e:
        log("jsonflush fail", e)


def drain(v):
    if isinstance(v, (tuple, list)):
        mx.eval(*v)
    else:
        mx.eval(v)


def tmed(fn, reps=None, warm=3):
    reps = reps or REPS
    for _ in range(warm):
        drain(fn())
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        drain(fn())
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2], ts[len(ts) // 4], ts[max(0, (3 * len(ts)) // 4 - 1)]


# ---------------------------------------------------------------- roofline
def roofline(M, bits, dense_tf, bw_gbps):
    """Per-segment shape roofline. FLOPs/byte = 16*M/bits (M rows amortize the
    b-bit weight stream). ceiling = min(dense, BW*FLOPs_per_byte)."""
    fpb = 16.0 * M / bits
    bw_tf = bw_gbps * fpb / 1000.0
    return min(dense_tf, bw_tf), fpb, bw_tf


def crossover_M(bits, dense_tf, bw_gbps):
    return dense_tf * 1000.0 * bits / (16.0 * bw_gbps)


def seg_analysis(tab_np, nb, out_e, in_tiles, bits, dense_tf, bw_gbps):
    """Per-segment roofline table + FLOP-weighted mix ceiling from a blk_tab."""
    K = int(in_tiles) * 16
    N = int(out_e)
    ln = tab_np[2, :nb].astype(np.int64) if nb else np.zeros(0, np.int64)
    M_hist = {}
    for v in ln.tolist():
        M_hist[int(v)] = M_hist.get(int(v), 0) + 1
    # per-M-class roofline
    per_M = []
    for M in sorted(M_hist):
        ceil_tf, fpb, bw_tf = roofline(M, bits, dense_tf, bw_gbps)
        per_M.append({"M": M, "blocks": M_hist[M], "fpb": fpb, "bw_ceiling_tf": bw_tf,
                      "shape_ceiling_tf": ceil_tf,
                      "bound": "compute" if bw_tf >= dense_tf else "bw"})
    total_flops = float(np.sum(2.0 * ln * N * K)) if nb else 0.0
    # FLOP-weighted mix ceiling = total_flops / sum_i(flops_i/ceil_i)
    denom = 0.0
    for M in M_hist:
        ceil_tf, _, _ = roofline(M, bits, dense_tf, bw_gbps)
        denom += (2.0 * M * M_hist[M] * N * K) / (ceil_tf * 1e12)
    mix_ceiling_tf = total_flops / (denom * 1e12) if denom > 0 else 0.0
    weight_bytes = float(np.sum(ln) * 0) + nb * N * K * bits / 8.0
    return {"K": K, "N": N, "nb_real": nb, "M_hist": M_hist, "M_sum": int(ln.sum()) if nb else 0,
            "M_max": int(ln.max()) if nb else 0,
            "per_M_roofline": per_M, "crossover_M": crossover_M(bits, dense_tf, bw_gbps),
            "total_flops": total_flops, "mix_ceiling_tf": mix_ceiling_tf,
            "weight_bytes": weight_bytes,
            "pad_waste": (nb * 64.0 / float(ln.sum())) if ln.sum() else 0.0}


# ---------------------------------------------------------------- ceilings
def ceilings():
    log("=== ceilings (in-session) ===")
    nb = 64 * 1024 * 1024
    a = mx.ones((nb,), mx.float32); b = mx.ones_like(a)
    mx.eval(a, b)
    R["ceil"]["add_GBps"] = 3.0 * nb * 4 / tmed(lambda: mx.add(a, b))[0] / 1e9
    R["ceil"]["copy_GBps"] = 2.0 * nb * 4 / tmed(lambda: mx.multiply(a, 1.0))[0] / 1e9
    del a, b; mx.clear_cache()
    best = 0.0; mm = []
    for (M, N, K) in [(2048, 4096, 4096), (4096, 4096, 4096), (8192, 4096, 4096),
                      (4096, 8192, 8192)]:
        A = mx.random.normal((M, K)).astype(mx.bfloat16)
        B = mx.random.normal((K, N)).astype(mx.bfloat16); mx.eval(A, B)
        t = tmed(lambda: A @ B, reps=max(3, REPS - 4), warm=2)[0]
        tf = 2.0 * M * N * K / t / 1e12; best = max(best, tf)
        mm.append({"M": M, "N": N, "K": K, "ms": t * 1e3, "TFLOPS": tf})
        log("  dense bf16 %dx%dx%d: %.3f ms %.2f TF" % (M, N, K, t * 1e3, tf))
        del A, B; mx.clear_cache()
    R["ceil"]["matmul"] = mm
    R["ceil"]["dense_TFLOPS"] = best
    R["ceil"]["BW_GBps"] = max(R["ceil"]["add_GBps"], R["ceil"]["copy_GBps"])
    log("  ceiling: %.2f TF dense-bf16 / %.0f GB/s BW" % (best, R["ceil"]["BW_GBps"]))
    flushjson()


# ---------------------------------------------------------------- idx builders
def build_idx_uniform(E, N, rng):
    ids = np.repeat(np.arange(E), int(np.ceil(N / E)))[:N].astype(np.int32)
    rng.shuffle(ids)
    return _pack(ids)


def build_idx_skew(E, N, rng, alpha=0.9, hot_frac=0.35):
    nhot = max(1, int(round(E * hot_frac)))
    w = np.zeros(E); w[:nhot] = 1.0 / (np.arange(1, nhot + 1) ** alpha)
    w[nhot:] = w[nhot - 1] * 0.25 if nhot < E else 0.0
    w = w / w.sum()
    ids = rng.choice(E, size=N, replace=True, p=w).astype(np.int32)
    return _pack(ids)


def build_idx_skew_balanced(E, N, rng, alpha=0.9, hot_frac=0.35):
    """Same skew shape, but every expert's count is a multiple of 64 -> no
    partial tail blocks (the balanced-assignment arm for Step 4)."""
    nhot = max(1, int(round(E * hot_frac)))
    w = np.zeros(E); w[:nhot] = 1.0 / (np.arange(1, nhot + 1) ** alpha)
    w[nhot:] = w[nhot - 1] * 0.25 if nhot < E else 0.0
    w = w / w.sum()
    nblk_tot = N // 64
    nblk = np.maximum(np.round(w * nblk_tot).astype(np.int64), 0)
    # fix rounding so total blocks == nblk_tot
    diff = nblk_tot - int(nblk.sum())
    i = 0
    while diff != 0 and i < len(nblk):
        if diff > 0 and w[i] > 0:
            nblk[i] += 1; diff -= 1
        elif diff < 0 and nblk[i] > 0:
            nblk[i] -= 1; diff += 1
        i += 1
    counts = nblk * 64
    ids = np.concatenate([np.full(int(c), e, np.int32) for e, c in enumerate(counts) if c > 0])
    rng.shuffle(ids)
    return _pack(ids)


def _pack(ids):
    tokens = (ids.size + KK - 1) // KK
    pad = tokens * KK - ids.size
    if pad:
        ids = np.concatenate([ids, np.zeros(pad, np.int32)])
    return ids.reshape(1, tokens, KK)


def real_chunk_idx(trace_path, layer, n_tokens, rng):
    d = json.load(open(trace_path))
    per_layer = {}
    for rec in d["records"]:
        L, experts = rec[0], rec[1]
        per_layer.setdefault(int(L), []).append(np.asarray(experts, np.int32))
    rows = per_layer[layer]
    sel = rng.randint(0, len(rows), size=n_tokens)
    ids = np.stack([rows[i] for i in sel]).reshape(-1).astype(np.int32)  # n_tokens*topk
    return _pack(ids), len(rows)


def idx_stats(idx, E):
    flat = np.asarray(idx).reshape(-1)
    c = np.bincount(flat, minlength=E)
    nb = int(np.sum((c + 63) // 64))
    return {"pairs": int(flat.size), "tokens": int(idx.shape[1]), "experts_hit": int((c > 0).sum()),
            "max_rows_per_expert": int(c.max()), "avg_rows_per_expert": float(flat.size / max(1, (c > 0).sum())),
            "blocks_bm64": nb, "pad_waste": float(nb * 64 / flat.size)}


# ---------------------------------------------------------------- prefill parts
def seg_parts(x, idx, sg, want_parts=False):
    """Reproduce EXL3SwitchGLU._prefill's segmented path, returning the per-launch
    inputs so each of the 3 launches can be timed separately."""
    E, D, H = sg.num_experts, sg.input_dims, sg.hidden_dims
    B, S, kk = idx.shape
    N = B * S * kk
    flat = idx.reshape(-1)
    order = mx.argsort(flat); inv = mx.argsort(order)
    sidx = flat[order]
    tok = (mx.arange(N, dtype=mx.uint32) // kk)[order]
    tok_x = mx.concatenate([tok, mx.zeros((64,), dtype=tok.dtype)])
    sidx_x = mx.concatenate([sidx, mx.zeros((64,), dtype=sidx.dtype)])
    x_pairs = x.reshape(B * S, D)[tok_x]
    xg = MOE._rows_prep()(x_pairs, sg._gu_suh[sidx_x, 0])
    xu = MOE._rows_prep()(x_pairs, sg._gu_suh[sidx_x, 1])
    nb_max = N // 64 + E + 1
    tab, nbr = MOE._seg_table_fn(E, nb_max, 64)(sidx)
    gt, dt = sg._gu_tiles, sg._dn_tiles
    g = inner_mm_seg_mlx(xg, sg._gu_trellis, sg._k, sg._cb, tab, nbr,
                         n_rows=N, tn_base=0, tiles_per_e=gt, out_e=H)
    u = inner_mm_seg_mlx(xu, sg._gu_trellis, sg._k, sg._cb, tab, nbr,
                         n_rows=N, tn_base=E * gt, tiles_per_e=gt, out_e=H)
    g = MOE._rows_finish()(g.reshape(N, H).astype(mx.float16), sg._gu_svh[sidx, :H])
    u = MOE._rows_finish()(u.reshape(N, H).astype(mx.float16), sg._gu_svh[sidx, H:])
    h = MOE._moe_gate_activation(g, u, sg._activation)
    h_pad = mx.concatenate([h, mx.zeros((64, H), dtype=h.dtype)])
    xd = MOE._rows_prep()(h_pad, sg._dn_suh[sidx_x])
    yl = inner_mm_seg_mlx(xd, sg._dn_trellis, sg._k, sg._cb, tab, nbr,
                          n_rows=N, tn_base=0, tiles_per_e=dt, out_e=D)
    y = MOE._rows_finish()(yl.reshape(N, D).astype(mx.float16), sg._dn_svh[sidx])
    y = y[inv].reshape(B, S, kk, D)
    if want_parts:
        return {"g": g, "u": u, "y": y, "tab": tab, "nbr": nbr, "N": N,
                "xg": xg, "xu": xu, "xd": xd, "H": H, "D": D, "gt": gt, "dt": dt}
    return y


def time_launches(x, idx, sg):
    """Per-launch wall (host perf_counter around a drained eval) for the 3
    inner_mm_seg_mlx launches. Per-segment GPU cycles are not exposed (single
    dispatch); this is the launch-level wall."""
    p = seg_parts(x, idx, sg, want_parts=True)
    E, D, H, N = sg.num_experts, sg.input_dims, sg.hidden_dims, p["N"]
    k, cb, gt, dt = sg._k, sg._cb, p["gt"], p["dt"]
    tab, nbr = p["tab"], p["nbr"]
    f_g = lambda: inner_mm_seg_mlx(p["xg"], sg._gu_trellis, k, cb, tab, nbr,
                                   n_rows=N, tn_base=0, tiles_per_e=gt, out_e=H)
    f_u = lambda: inner_mm_seg_mlx(p["xu"], sg._gu_trellis, k, cb, tab, nbr,
                                   n_rows=N, tn_base=E * gt, tiles_per_e=gt, out_e=H)
    f_y = lambda: inner_mm_seg_mlx(p["xd"], sg._dn_trellis, k, cb, tab, nbr,
                                   n_rows=N, tn_base=0, tiles_per_e=dt, out_e=D)
    res = {}
    for tag, f in (("gu_gate", f_g), ("gu_up", f_u), ("dn", f_y)):
        t, q1, q3 = tmed(f)
        res[tag] = {"ms": t * 1e3, "iqr_ms": [q1 * 1e3, q3 * 1e3]}
    full = tmed(lambda: sg._prefill(x, idx))
    res["_full_prefill_ms"] = full[0] * 1e3
    return res, p


# ---------------------------------------------------------------- STAGE 1
def stage1():
    ceilings()
    dense_tf = R["ceil"]["dense_TFLOPS"]; bw = R["ceil"]["BW_GBps"]
    ck = Exl3Checkpoint(CK)
    R["stage1"] = {}

    # -- (a) REAL production chunk, all 384 experts, layer 0 (k=3)
    try:
        log("=== Step1(a): REAL production chunk (trace layer 0), E=384 ===")
        sg = load_experts(ck, 0, n_experts=None, rank=0, world=2, activation="silu_clamp")
        E, D, H = sg.num_experts, sg.input_dims, sg.hidden_dims
        k, gt, dt = sg._k, sg._gu_tiles, sg._dn_tiles
        peb = (sg._gu_trellis.size + sg._dn_trellis.size) * 2 / E
        R["stage1"]["real_geometry"] = {"E": E, "D": D, "H": H, "k": k, "gu_tiles": gt,
                                        "dn_tiles": dt, "per_expert_MB": peb / 1e6,
                                        "layer": 0}
        log("  E=%d D=%d H=%d k=%d gt=%d dt=%d per_exp=%.2fMB" % (E, D, H, k, gt, dt, peb / 1e6))
        rng = np.random.RandomState(20261010)
        idx, ntr = real_chunk_idx(TRACE, 0, 2048, rng)
        st = idx_stats(idx, E)
        x = mx.random.normal((1, idx.shape[1], D)).astype(mx.float16)
        m = mx.array(idx, dtype=mx.uint32); mx.eval(x, m)
        # capture (LOG on) -- single pass, eval sync only here
        G._MM_SEG_LOG_ROWS.clear()
        G._MM_SEG_LOG = True
        y = sg._prefill(x, m); mx.eval(y)
        G._MM_SEG_LOG = False
        caps = [dict(r) for r in G._MM_SEG_LOG_ROWS]
        R["stage1"]["real"] = {"idx_stats": st, "trace_tokens": ntr,
                               "launch_caps": caps, "num_launches_captured": len(caps)}
        log("  captured %d launches" % len(caps))
        # timing: full prefill + per-launch wall (LOG off)
        tl, _ = time_launches(x, m, sg)
        F = 6.0 * D * H
        R["stage1"]["real"]["timing"] = tl
        R["stage1"]["real"]["full_prefill_TFLOPS"] = F * st["pairs"] / (tl["_full_prefill_ms"] / 1e3) / 1e12
        log("  full prefill %.2f ms  %.2f TF" % (tl["_full_prefill_ms"],
            R["stage1"]["real"]["full_prefill_TFLOPS"]))
        del x, m, sg, y; mx.clear_cache(); flushjson()
    except Exception as e:
        log("Step1(a) FAILED", type(e).__name__, e); traceback.print_exc()
        R["notes"].append("stage1 real: %s" % e); flushjson()

    # -- (b)/(c) uniform + skew E=192 N=12288 (in-session, prior-comparable)
    try:
        log("=== Step1(b/c): uniform+skew E=%d N=%d ===" % (E_BLOCK, NPROD))
        sg = load_experts(ck, 0, n_experts=E_BLOCK, rank=0, world=2, activation="silu_clamp")
        E, D, H = sg.num_experts, sg.input_dims, sg.hidden_dims
        k, gt, dt = sg._k, sg._gu_tiles, sg._dn_tiles
        F = 6.0 * D * H
        rows = {}
        for mode, builder in (("uniform", build_idx_uniform), ("skew", build_idx_skew)):
            rng = np.random.RandomState(2026)
            idx = builder(E, NPROD, rng)
            st = idx_stats(idx, E)
            x = mx.random.normal((1, idx.shape[1], D)).astype(mx.float16)
            m = mx.array(idx, dtype=mx.uint32); mx.eval(x, m)
            G._MM_SEG_LOG_ROWS.clear(); G._MM_SEG_LOG = True
            mx.eval(sg._prefill(x, m)); G._MM_SEG_LOG = False
            caps = [dict(r) for r in G._MM_SEG_LOG_ROWS]
            tl, _ = time_launches(x, m, sg)
            t = tl["_full_prefill_ms"] / 1e3
            rec = {"mode": mode, "E": E, "D": D, "H": H, "k": k, "idx_stats": st,
                   "launch_caps": caps, "timing": tl,
                   "seg_ms": tl["_full_prefill_ms"], "TFLOPS": F * st["pairs"] / t / 1e12,
                   "pct_of_dense_ceiling": F * st["pairs"] / t / 1e12 / dense_tf * 100}
            rows[mode] = rec
            log("  [%s] blocks=%d waste=%.3f maxM=%d | %.2f ms %.2f TF (%.0f%% ceil)"
                % (mode, st["blocks_bm64"], st["pad_waste"], st["max_rows_per_expert"],
                   rec["seg_ms"], rec["TFLOPS"], rec["pct_of_dense_ceiling"]))
            del x, m; mx.clear_cache(); flushjson()
        R["stage1"]["block_192"] = rows
        # per-launch segmentation capture for both modes, gu(dn) shape
        for mode in rows:
            caps = rows[mode]["launch_caps"]
            rows[mode]["seg_analysis"] = [_cap_roofline(c, dense_tf, bw) for c in caps]
        flushjson()
        del sg; mx.clear_cache()
    except Exception as e:
        log("Step1(b/c) FAILED", type(e).__name__, e); traceback.print_exc()
        R["notes"].append("stage1 block: %s" % e)

    R["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    flushjson(); log("DONE ->", OUT)


def _cap_roofline(cap, dense_tf, bw):
    """Roofline summary for one captured launch, from its M histogram."""
    k = cap["k_bits"]; N = cap["N"]; K = cap["K"]
    M_hist = {int(a): int(b) for a, b in cap["M_hist"].items()}
    per_M = []
    for M in sorted(M_hist):
        ceil_tf, fpb, bw_tf = roofline(M, k, dense_tf, bw)
        per_M.append({"M": M, "blocks": M_hist[M], "fpb": round(fpb, 2),
                      "bw_ceiling_tf": round(bw_tf, 2), "shape_ceiling_tf": round(ceil_tf, 3),
                      "bound": "compute" if bw_tf >= dense_tf else "bw"})
    total_flops = float(sum(2.0 * M * M_hist[M] * N * K for M in M_hist))
    denom = 0.0
    for M in M_hist:
        ceil_tf, _, _ = roofline(M, k, dense_tf, bw)
        denom += (2.0 * M * M_hist[M] * N * K) / (ceil_tf * 1e12)
    mix_ceil = total_flops / (denom * 1e12) if denom > 0 else 0.0
    wb = cap["nb_real"] * N * K * k / 8.0
    return {"tn_base": cap["tn_base"], "k_bits": k, "N": N, "K": K,
            "nb_real": cap["nb_real"], "M_sum": cap["M_sum"], "M_max": cap["M_max"],
            "n_experts_touched": cap.get("n_experts_touched"),
            "pad_waste": cap.get("blocks_per_full_row"),
            "per_M_roofline": per_M, "crossover_M": round(crossover_M(k, dense_tf, bw), 2),
            "total_flops": total_flops, "mix_ceiling_tf": mix_ceil,
            "weight_bytes": wb}


# ---------------------------------------------------------------- STAGE 3
def arm_dense_inloop(x, idx, sg):
    """In-loop dequant + per-expert dense matmul (the prior killed arm)."""
    E, D, H = sg.num_experts, sg.input_dims, sg.hidden_dims
    k, cb, gt, dt = sg._k, sg._cb, sg._gu_tiles, sg._dn_tiles
    xf = x.reshape(-1, D); flat = np.asarray(idx).reshape(-1).astype(np.int64)
    order = mx.argsort(mx.array(flat.astype(np.uint32)))
    sid_np = np.asarray(mx.array(flat.astype(np.uint32))[order]).astype(np.int64)
    tok = (mx.arange(flat.size, dtype=mx.uint32) // KK)[order]
    xs = xf[tok]; outs = []
    for e in np.unique(flat):
        e = int(e); lo = int(np.searchsorted(sid_np, e, "left")); hi = int(np.searchsorted(sid_np, e, "right"))
        if hi <= lo:
            continue
        Re = hi - lo; xe = xs[lo:hi]
        xg = MOE._rows_prep()(xe, mx.broadcast_to(sg._gu_suh[e, 0][None], (Re, D)))
        xu = MOE._rows_prep()(xe, mx.broadcast_to(sg._gu_suh[e, 1][None], (Re, D)))
        wg = decode_full_mlx(sg._gu_trellis, k, cb, tn_offset=e * gt, tn_count=gt)
        wu = decode_full_mlx(sg._gu_trellis, k, cb, tn_offset=(E + e) * gt, tn_count=gt)
        g = MOE._rows_finish()(xg @ wg, mx.broadcast_to(sg._gu_svh[e, :H][None], (Re, H)))
        u = MOE._rows_finish()(xu @ wu, mx.broadcast_to(sg._gu_svh[e, H:][None], (Re, H)))
        h = _sc(g.astype(mx.float32), u.astype(mx.float32)).astype(mx.float16)
        xh = MOE._rows_prep()(h, mx.broadcast_to(sg._dn_suh[e][None], (Re, H)))
        wd = decode_full_mlx(sg._dn_trellis, k, cb, tn_offset=e * dt, tn_count=dt)
        outs.append(MOE._rows_finish()(xh @ wd, mx.broadcast_to(sg._dn_svh[e][None], (Re, D))))
        del wg, wu, wd
    return mx.concatenate(outs, 0)[mx.argsort(order)].reshape(1, idx.shape[1], KK, D)


def build_wcache(sg, experts):
    """Decode each expert's fp16 W ONCE (outside the timed region)."""
    E, gt, dt, k, cb = sg.num_experts, sg._gu_tiles, sg._dn_tiles, sg._k, sg._cb
    cache = {}
    for e in experts:
        e = int(e)
        cache[e] = (
            decode_full_mlx(sg._gu_trellis, k, cb, tn_offset=e * gt, tn_count=gt),
            decode_full_mlx(sg._gu_trellis, k, cb, tn_offset=(E + e) * gt, tn_count=gt),
            decode_full_mlx(sg._dn_trellis, k, cb, tn_offset=e * dt, tn_count=dt),
        )
    mx.eval(*[t for v in cache.values() for t in v])
    return cache


def arm_dense_pre(x, idx, sg, cache):
    """GEMM from PRE-converted fp16 weights (no dequant in the timed region)."""
    E, D, H = sg.num_experts, sg.input_dims, sg.hidden_dims
    xf = x.reshape(-1, D); flat = np.asarray(idx).reshape(-1).astype(np.int64)
    order = mx.argsort(mx.array(flat.astype(np.uint32)))
    sid_np = np.asarray(mx.array(flat.astype(np.uint32))[order]).astype(np.int64)
    tok = (mx.arange(flat.size, dtype=mx.uint32) // KK)[order]
    xs = xf[tok]; outs = []
    for e in np.unique(flat):
        e = int(e); lo = int(np.searchsorted(sid_np, e, "left")); hi = int(np.searchsorted(sid_np, e, "right"))
        if hi <= lo:
            continue
        Re = hi - lo; xe = xs[lo:hi]
        wg, wu, wd = cache[e]
        xg = MOE._rows_prep()(xe, mx.broadcast_to(sg._gu_suh[e, 0][None], (Re, D)))
        xu = MOE._rows_prep()(xe, mx.broadcast_to(sg._gu_suh[e, 1][None], (Re, D)))
        g = MOE._rows_finish()(xg @ wg, mx.broadcast_to(sg._gu_svh[e, :H][None], (Re, H)))
        u = MOE._rows_finish()(xu @ wu, mx.broadcast_to(sg._gu_svh[e, H:][None], (Re, H)))
        h = _sc(g.astype(mx.float32), u.astype(mx.float32)).astype(mx.float16)
        xh = MOE._rows_prep()(h, mx.broadcast_to(sg._dn_suh[e][None], (Re, H)))
        outs.append(MOE._rows_finish()(xh @ wd, mx.broadcast_to(sg._dn_svh[e][None], (Re, D))))
    return mx.concatenate(outs, 0)[mx.argsort(order)].reshape(1, idx.shape[1], KK, D)


def _sc(g, u, lim=10.0):
    g = mx.minimum(g, lim); u = mx.clip(u, -lim, lim)
    return (g / (1.0 + mx.exp(-g))) * u


def stage3():
    if not R["ceil"]:
        ceilings()
    dense_tf = R["ceil"]["dense_TFLOPS"]; bw = R["ceil"]["BW_GBps"]
    ck = Exl3Checkpoint(CK)
    E_SUB = int(os.environ.get("MB_E_ARM", "32")); M_ARM = int(os.environ.get("MB_M_ARM", "64"))
    sg = load_experts(ck, 0, n_experts=E_SUB, rank=0, world=2, activation="silu_clamp")
    E, D, H = sg.num_experts, sg.input_dims, sg.hidden_dims
    k, gt, dt = sg._k, sg._gu_tiles, sg._dn_tiles
    F = 6.0 * D * H
    log("=== Step3: dequant-excluded arm E_sub=%d ===" % E)
    res = {"E_sub": E, "D": D, "H": H, "k": k, "arms": {}}
    for M in (M_ARM,):
        rng = np.random.RandomState(7)
        idx = _pack(np.repeat(np.arange(E, dtype=np.int32), M))
        # shuffle within
        flat = np.asarray(idx).reshape(-1).copy(); rng.shuffle(flat)
        idx = _pack(flat)
        st = idx_stats(idx, E)
        x = mx.random.normal((1, idx.shape[1], D)).astype(mx.float16)
        m = mx.array(idx, dtype=mx.uint32); mx.eval(x, m)
        experts = np.unique(np.asarray(idx).reshape(-1))
        log("  building fp16 W cache for %d experts ..." % len(experts))
        cache = build_wcache(sg, experts)
        nbytes = sum(t.size * t.dtype.size for v in cache.values() for t in v)
        # correctness: pre == inloop
        ya = arm_dense_pre(x, m, sg, cache); yb = arm_dense_inloop(x, m, sg); mx.eval(ya, yb)
        rel = float(mx.max(mx.abs(ya.astype(mx.float32) - yb.astype(mx.float32))) /
                    (mx.max(mx.abs(yb.astype(mx.float32))) + 1e-9))
        del ya, yb; mx.clear_cache()
        # interleaved timing: seg, pre, inloop
        f_seg = lambda: seg_parts(x, m, sg)
        f_pre = lambda: arm_dense_pre(x, m, sg, cache)
        f_inl = lambda: arm_dense_inloop(x, m, sg)
        for f in (f_seg, f_pre, f_inl):
            drain(f())
        ts = {"seg": [], "pre": [], "inloop": []}
        for _ in range(max(5, REPS)):
            t0 = time.perf_counter(); drain(f_seg()); ts["seg"].append(time.perf_counter() - t0)
            t0 = time.perf_counter(); drain(f_pre()); ts["pre"].append(time.perf_counter() - t0)
            t0 = time.perf_counter(); drain(f_inl()); ts["inloop"].append(time.perf_counter() - t0)
        med = {kk2: sorted(v)[len(v) // 2] for kk2, v in ts.items()}
        r2 = {"M": M, "E_sub": E, **st, "wcache_MB": nbytes / 1e6, "rel_max_err_pre_vs_inloop": rel}
        for kk2 in med:
            r2["%s_ms" % kk2] = med[kk2] * 1e3
            r2["%s_TFLOPS" % kk2] = F * st["pairs"] / med[kk2] / 1e12
            r2["%s_pct_ceiling" % kk2] = r2["%s_TFLOPS" % kk2] / dense_tf * 100
        r2["pre_over_inloop"] = med["inloop"] / med["pre"]
        r2["pre_over_seg"] = med["seg"] / med["pre"]
        cap = _cap_roofline({"k_bits": k, "N": H, "K": D,
                             "M_hist": {M: E}, "nb_real": E, "M_sum": E * M, "M_max": M,
                             "tn_base": 0, "blocks_per_full_row": 1.0}, dense_tf, bw)
        r2["gu_roofline"] = cap["per_M_roofline"]
        res["arms"]["M%d" % M] = r2
        log("  M=%d E=%d: seg %.2fTF(%.0f%%) pre %.2fTF(%.0f%%) inloop %.2fTF(%.0f%%) | pre/inloop=%.2fx | roofline(M=%d)=%.2fTF"
            % (M, E, r2["seg_TFLOPS"], r2["seg_pct_ceiling"], r2["pre_TFLOPS"], r2["pre_pct_ceiling"],
               r2["inloop_TFLOPS"], r2["inloop_pct_ceiling"], r2["pre_over_inloop"], M, cap["per_M_roofline"][0]["shape_ceiling_tf"]))
        del x, m, cache; mx.clear_cache(); flushjson()
    R["stage3"] = res
    R["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    flushjson(); log("DONE ->", OUT)


# ---------------------------------------------------------------- STAGE 4
def stage4():
    if not R["ceil"]:
        ceilings()
    dense_tf = R["ceil"]["dense_TFLOPS"]; bw = R["ceil"]["BW_GBps"]
    ck = Exl3Checkpoint(CK)
    sg = load_experts(ck, 0, n_experts=E_BLOCK, rank=0, world=2, activation="silu_clamp")
    E, D, H = sg.num_experts, sg.input_dims, sg.hidden_dims
    k = sg._k; F = 6.0 * D * H
    log("=== Step4: uniform vs skew vs skew-balanced E=%d N=%d ===" % (E, NPROD))
    res = {}
    variants = (("uniform", build_idx_uniform),
                ("skew", build_idx_skew),
                ("skew_balanced", build_idx_skew_balanced))
    # two rounds interleaved to average thermal drift
    acc = {n: [] for n, _ in variants}
    for rnd in range(2):
        for name, builder in variants:
            rng = np.random.RandomState(2026 + rnd)
            idx = builder(E, NPROD, rng)
            x = mx.random.normal((1, idx.shape[1], D)).astype(mx.float16)
            m = mx.array(idx, dtype=mx.uint32); mx.eval(x, m)
            t = tmed(lambda: sg._prefill(x, m), reps=max(5, REPS))[0]
            acc[name].append((t, idx_stats(idx, E)))
            del x, m; mx.clear_cache()
    for name, _ in variants:
        ts = [a[0] for a in acc[name]]
        st = acc[name][0][1]
        tmed_t = sorted(ts)[len(ts) // 2]
        res[name] = {"idx_stats": st, "ms": tmed_t * 1e3, "ms_all": [a * 1e3 for a in ts],
                     "TFLOPS": F * st["pairs"] / tmed_t / 1e12,
                     "pct_ceiling": F * st["pairs"] / tmed_t / 1e12 / dense_tf * 100,
                     "pad_waste": st["pad_waste"]}
        log("  [%s] blocks=%d waste=%.3f maxM=%d | %.2f ms %.2f TF (%.0f%%)"
            % (name, st["blocks_bm64"], st["pad_waste"], st["max_rows_per_expert"],
               res[name]["ms"], res[name]["TFLOPS"], res[name]["pct_ceiling"]))
    base = res["uniform"]["TFLOPS"]; sk = res["skew"]["TFLOPS"]; bl = res["skew_balanced"]["TFLOPS"]
    res["_derived"] = {
        "uniform_to_skew_drop_pct": (1 - sk / base) * 100,
        "skew_balanced_recovery_pct": (bl / sk - 1) * 100,
        "skew_balanced_pct_of_uniform": bl / base * 100,
        "skew_pad_waste": res["skew"]["pad_waste"],
        "skew_balanced_pad_waste": res["skew_balanced"]["pad_waste"],
    }
    R["stage4"] = res
    R["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    flushjson(); log("DONE ->", OUT)


def main():
    log("stage", STAGE, "ck", CK)
    try:
        if STAGE == "1":
            stage1()
        elif STAGE == "3":
            stage3()
        elif STAGE == "4":
            stage4()
        else:
            raise SystemExit("unknown STAGE=%s" % STAGE)
    except Exception as e:
        log("FATAL", type(e).__name__, e); traceback.print_exc()
        R["notes"].append("fatal: %s" % e); flushjson()


if __name__ == "__main__":
    main()
