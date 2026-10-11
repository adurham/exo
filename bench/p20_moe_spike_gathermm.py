#!/usr/bin/env python3
"""Phase-20 MoE spike: Dir-D price -- decode-once fp16 -> MLX steel gather GEMM.

QUESTION. On the real dsv41 MoE prefill mix (E=384, 2048 tok x top-6 = 12288
(token,slot) pairs/rank; gu K=5120 N=1152, dn K=1152 N=5120) can MLX's own
batched GATHERED GEMM on once-decoded fp16 weights beat the fused EXL3 seg
kernel (v19d, in-kernel trellis decode) once the decode is amortized?

PRIMITIVE (verified present in this mlx; see "primitive" block in the JSON):
  mx.gather_mm(a, b, lhs_indices=None, rhs_indices=None, *, sorted_indices=False)
    out[i...] = a[lhs[i...]] @ b[rhs[i...]]   (batch-level gather, NOT per-row)
    b is (..., K, N) -- the per-expert weight in (in, out) layout.
  Metal impl: steel_gather_mm_* generic batched gather, bm=64/bn=64/bk=16
    (mlx/backend/metal/matmul.cpp:2835, dispatch in GatherMM::eval_gpu :2974).
    The sorted fast path (gather_mm_rhs :2308 / _nax :2441) is M==1-only -- it
    is what mlx_lm EXL3SwitchGLU._prefill's existing fallback uses
    (exl3_moe.py:920-927), i.e. a per-(token,slot) M=1 GEMV with contiguous-run
    weight reuse, NOT a batched M>1 GEMM. Dir D prices the batched form.
  mx.gather_qmm / mx.segmented_mm also exist; gather_qmm is the quantized
    variant (no 3-bit-exl3 mode) and segmented_mm splits the K dimension --
    neither expresses ragged per-expert rows.

DIR-D FORMULATION (the ragged runs cannot be a single uniform-M batch):
  * rows are expert-sorted (contiguous runs, partition buildnote sec 5 item 3);
    each expert's run is cut into contiguous tiles of Mt rows; the last tile of
    an expert is zero-padded.  lhs=arange(T), rhs=expert-of-tile (sorted) ->
    one batched steel gather per projection.
  * gu: decode_full_eg_mlx(gu_trellis, tiles_per_e=H/16) -> (2E, D, H)
    ([all gates | all ups], same layout the fork's fallback consumes);
    gate = w_gu[:E], up = w_gu[E:]  (two gather_mm calls, as the fork does).
  * dn: decode_full_eg_mlx(dn_trellis, tiles_per_e=D/16) -> (E, H, D), one call.
  * pad rows are multiplied by 0 so they cannot poison the dn input.

METRIC (identical to bench/p20_moe_partition_bench.py:13):
  TF = 6*D*H*pairs / wall, pairs = (token,slot) pairs = 12288.
  Wall variants reported per mix:
    seg3        = the 3 inner_mm_seg_mlx launches          [launch-only]
    seg_pipe    = seg3 + gate activation                    [glue-incl]
    steel3      = the 3 gather_mm launches                  [launch-only]
    steel_pipe  = tile-build + 3 gathers + act + unpad-take [glue-incl]
    +decode_incl = steel_pipe + one full gu+dn decode (all experts)
  The verdict ratio uses steel_pipe / seg_pipe ON THE SAME HOST (prep/finish/
  sort/take are shared by both production paths and cancel); it is measured
  INTERLEAVED (seg and steel alternate, median of per-round ratios) so slow
  drift from GPU co-tenants cancels.  A ratio within +-0.10 of 1.0 counts as
  marginal and is decided by the decode-included price (the decode is
  intrinsic to Dir D).  The decode cost is also reported amortized over the
  routed-expert fraction and flagged as once-per-forward reusable.
  Reference (studio2, cross-host): seg 6.63 TF real / 8.85 TF uniform E=192,
  mix roofline 14.96 TF.

WEIGHTS. GM_CK default = node checkpoint path; REAL_CK=auto uses it when it
exists, else synthetic trellis (weights=synthetic in stdout/JSON -- the label
is in every record).  Synthetic trellis = bench/p20_moe_partition_common
.synth_layer (random trellis words exercise every decode path; kernel cost is
value-independent).

VALIDATION. `selfcheck()` (E=8 synthetic) cross-checks the tiled gather path
END-TO-END against the seg kernel on the same weights+rows (row-mapping,
gate/up layout, padding, dn chaining) and reports the relative difference.

Env:  GM_JSON (/tmp/p20moe_spike/p20_moe_spike_gathermm.json), GM_CK,
      REAL_CK=auto|1|0, GM_MIXES=real,uniform192, GM_MTS=64,32,16,128,
      GM_EBATCH=auto|N (experts decoded+GEMMed per chunk; auto clamps to the
      mlx int32 fast-kernel output cap -- REQUIRED at these dims for E>~182,
      one-shot whole-E decode raises OverflowError; a smaller N also bounds
      RAM), GM_REPS=5, GM_REPS_DECODE=3, MB_TRACE, P20_PKG,
      GM_CONFIRM=1|0, GM_CONFIRM_ROUNDS=5 (interleaved drift-cancelled
      confirm: alternate seg and steel pipes, median of per-round ratios).

Node run (real weights, unchanged command, scratch pkg on PYTHONPATH):
      P20_PKG=<scratch-mlx-lm> GM_MIXES=real,uniform192 \\
        ~/repos/exo/.venv/bin/python bench/p20_moe_spike_gathermm.py
Laptop smoke (checkpoint absent -> synthetic): same without P20_PKG/REAL_CK.
"""
from __future__ import annotations
import os, sys, json, time, socket, traceback
sys.dont_write_bytecode = True  # never write .pyc into shared checkouts

os.environ.setdefault("EXL3_MM_SEG", "v19d")  # production default
HOME = os.path.expanduser("~")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
P20_PKG = os.environ.get("P20_PKG", "")
if P20_PKG:
    sys.path.insert(0, P20_PKG)
elif os.path.isdir(os.path.join(ROOT, "mlx-lm", "mlx_lm")):
    sys.path.insert(0, os.path.join(ROOT, "mlx-lm"))
else:
    sys.path.insert(0, HOME + "/repos/exo/mlx-lm")  # read-only import
sys.path.insert(0, HERE)

import numpy as np
import mlx.core as mx
from mlx_lm.models.exl3 import gemv_metal as G
from mlx_lm.models.exl3 import exl3_moe as MOE
import p20_moe_partition_common as C

decode_full_eg_mlx = G.decode_full_eg_mlx  # local alias (chunked + whole paths)

GM_JSON = os.environ.get("GM_JSON", "/tmp/p20moe_spike/p20_moe_spike_gathermm.json")
GM_CK = os.environ.get("GM_CK",
                       HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw")
REAL_CK = os.environ.get("REAL_CK", "auto")            # auto|1|0
MIXES = [m for m in os.environ.get("GM_MIXES", "real,uniform192").split(",") if m]
MTS = [int(x) for x in os.environ.get("GM_MTS", "64,32,16,128").split(",")]
_EB = os.environ.get("GM_EBATCH", "auto").strip().lower()
EBATCH = -1 if _EB in ("", "auto") else int(_EB)   # -1 = auto-clamp (see price_mix)
REPS = int(os.environ.get("GM_REPS", "5"))
REPS_DEC = int(os.environ.get("GM_REPS_DECODE", "3"))
KK = 6
E_UNIFORM = 192
PUBLISHED = {"seg_real_TF": 6.63, "seg_uniform192_TF": 8.85,
             "mix_roofline_TF": 14.96, "host": "studio2",
             "note": "cross-host reference; the same-host ratio is authoritative"}

os.makedirs(os.path.dirname(GM_JSON), exist_ok=True)
R = {"script": "bench/p20_moe_spike_gathermm.py", "host": socket.gethostname(),
     "mlx": mx.__version__, "gemv_metal": G.__file__, "seg_version": G._MM_SEG_VERSION,
     "tf_convention": "TF = 6*D*H*pairs / wall (pairs = token,slot pairs = 12288); "
                      "walls: seg3/seg_pipe (prod seg launches, launch-only/pipe), "
                      "steel3/steel_pipe (gather_mm launches/pipe+glue), +decode_incl",
     "started": time.strftime("%Y-%m-%d %H:%M:%S"), "env": {
         "GM_MIXES": MIXES, "GM_MTS": MTS, "GM_EBATCH": EBATCH, "GM_REPS": REPS,
         "GM_REPS_DECODE": REPS_DEC, "REAL_CK": REAL_CK, "GM_CK": GM_CK},
     "primitive": {
         "gather_mm": "out[i] = a[lhs[i]] @ b[rhs[i]] (batch-level gather)",
         "signature": "mx.gather_mm(a, b, lhs_indices=None, rhs_indices=None, "
                      "*, sorted_indices=False) -> array",
         "py_stub": "mlx/python/mlx/core/__init__.pyi:3458",
         "binding": "mlx/python/src/ops.cpp:5146 (array gather_mm ... :6147)",
         "metal": "mlx/backend/metal/matmul.cpp:2835 steel_gather_mm_* "
                  "(generic batched gather, bm=64 bn=64 bk=16); sorted M==1 "
                  "rhs-only path :2308 (gather_mm_rhs) / :2441 (_nax)",
         "also_present": "mx.gather_qmm (ops.cpp:4837, affine modes only), "
                         "mx.segmented_mm (ops.cpp:6280, splits K not rows)"},
     "mixes": {}, "selfcheck": {}, "notes": []}


def log(*a):
    print("[gathermm]", *a, flush=True)


def flushjson():
    try:
        json.dump(R, open(GM_JSON, "w"), indent=1, default=float)
    except Exception as e:
        log("jsonflush fail", e)


def drain(v):
    if isinstance(v, (tuple, list)):
        mx.eval(*v)
    else:
        mx.eval(v)


def tmed(fn, reps=REPS, warm=3):
    for _ in range(warm):
        drain(fn())
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        drain(fn())
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


def seg_launches(I, L, E, D, H, N, xd_pad, seg_version=None):
    """The 3 production seg launches (bench/p20_moe_autopsy.py:258 time_launches)."""
    if seg_version:
        G._MM_SEG_VERSION = seg_version
    k, cb, gt, dt = L["k"], L["cb"], L["gt"], L["dt"]
    tab, nbr = I["tab"], I["nbr"]
    f_g = lambda: G.inner_mm_seg_mlx(I["x"], L["gu"], k, cb, tab, nbr,
                                     n_rows=N, tn_base=0, tiles_per_e=gt, out_e=H)
    f_u = lambda: G.inner_mm_seg_mlx(I["x"], L["gu"], k, cb, tab, nbr,
                                     n_rows=N, tn_base=E * gt, tiles_per_e=gt, out_e=H)
    f_y = lambda: G.inner_mm_seg_mlx(xd_pad, L["dn"], k, cb, tab, nbr,
                                     n_rows=N, tn_base=0, tiles_per_e=dt, out_e=D)
    tg, tu, ty = tmed(f_g), tmed(f_u), tmed(f_y)
    g = f_g(); u = f_u()
    mx.eval(g, u)
    t_act = tmed(lambda: MOE._moe_gate_activation(g, u, "silu_clamp"))
    return {"launch_gate_ms": tg * 1e3, "launch_up_ms": tu * 1e3,
            "launch_dn_ms": ty * 1e3, "seg3_ms": (tg + tu + ty) * 1e3,
            "act_ms": t_act * 1e3, "seg_pipe_ms": (tg + tu + ty + t_act) * 1e3}


def tile_tables(c, starts, e0, e1, Mt):
    """Rows-sorted tile decomposition for experts [e0,e1).  Returns per-tile
    local expert ids, per-slot global sorted-row source (pad slots use src 0)
    and the slot validity mask."""
    eids, srcs, valid = [], [], []
    for e in range(e0, e1):
        m = int(c[e])
        if m == 0:
            continue
        for t in range((m + Mt - 1) // Mt):
            base = starts[e] + t * Mt
            take = min(Mt, starts[e] + m - base)
            eids.append(e - e0)
            srcs.extend(range(base, base + take))
            srcs.extend([0] * (Mt - take))
            valid.extend([True] * take + [False] * (Mt - take))
    return (np.asarray(eids, np.int32), np.asarray(srcs, np.int32),
            np.asarray(valid, bool))


def decode_chunk(trellis, k, cb, tpe, tile_lo, tile_hi, needs_copy):
    sl = trellis[:, tile_lo:tile_hi, :]
    if needs_copy:
        sl = mx.array(np.ascontiguousarray(np.array(sl)))
    return G.decode_full_eg_mlx(sl, k, cb, tiles_per_e=tpe)


def price_steel(I, L, E, D, H, N, sidx_np, Mt, weights, ebatch):
    """Time the tiled batched gather path for one Mt.  Decode is EXCLUDED here
    (priced once in price_mix).  ``weights``=(wg,wu,wd) whole-E fp16 arrays from
    a single decode_full_eg_mlx pair, or None -> decode per expert chunk
    internally (sliced-trellis decode + np copy: the memory-bounded path)."""
    k, cb, gt, dt = L["k"], L["cb"], L["gt"], L["dt"]
    c = np.bincount(sidx_np.astype(np.int64), minlength=E)
    starts = np.concatenate([[0], np.cumsum(c)])
    x_rows = I["x"][:N]
    act = lambda g3, u3: MOE._moe_gate_activation(g3, u3, "silu_clamp")
    acc = {"g": 0.0, "u": 0.0, "y": 0.0, "act": 0.0, "build": 0.0, "unpad": 0.0}
    tiles = pads = mach = 0
    nchunk = (E + ebatch - 1) // ebatch if ebatch else 1
    for ci in range(nchunk):
        e0 = ci * ebatch if ebatch else 0
        e1 = min(E, e0 + ebatch) if ebatch else E
        if weights is not None:
            wg, wu, wd = weights
        else:
            wg = decode_chunk(L["gu"], k, cb, gt, e0 * gt, e1 * gt, True)
            wu = decode_chunk(L["gu"], k, cb, gt, E * gt + e0 * gt, E * gt + e1 * gt, True)
            wd = decode_chunk(L["dn"], k, cb, dt, e0 * dt, e1 * dt, True)
            mx.eval(wg, wu, wd)
        assert wg.shape == (e1 - e0, D, H) and wu.shape == (e1 - e0, D, H) \
            and wd.shape == (e1 - e0, H, D), (wg.shape, wu.shape, wd.shape)
        eids, srcs, valid = tile_tables(c, starts, e0, e1, Mt)
        T = int(eids.size)
        if T == 0:
            continue
        tiles += T
        pads += T * Mt
        mach += T * Mt if Mt >= 64 else T * 64
        srcs_mx = mx.array(srcs)
        mask_mx = mx.array(valid.astype(np.float16))
        valid_pos = mx.array(np.flatnonzero(valid).astype(np.int32))
        lhs = mx.arange(T, dtype=mx.int32)
        rhs = mx.array(eids)
        xt = (x_rows[srcs_mx] * mask_mx[:, None]).astype(mx.float16)
        mx.eval(xt)
        xt3 = xt.reshape(T, Mt, D)
        acc["build"] += tmed(lambda: x_rows[srcs_mx] * mask_mx[:, None]) * 1e3
        acc["g"] += tmed(lambda: mx.gather_mm(xt3, wg, lhs, rhs)) * 1e3
        acc["u"] += tmed(lambda: mx.gather_mm(xt3, wu, lhs, rhs)) * 1e3
        g3 = mx.gather_mm(xt3, wg, lhs, rhs)
        u3 = mx.gather_mm(xt3, wu, lhs, rhs)
        mx.eval(g3, u3)
        acc["act"] += tmed(lambda: act(g3, u3)) * 1e3
        h3 = act(g3, u3)
        mx.eval(h3)
        acc["y"] += tmed(lambda: mx.gather_mm(h3, wd, lhs, rhs)) * 1e3
        y3 = mx.gather_mm(h3, wd, lhs, rhs)
        mx.eval(y3)
        yflat = y3.reshape(T * Mt, D)
        acc["unpad"] += tmed(lambda: yflat[valid_pos]) * 1e3
        if weights is None:
            del wg, wu, wd, g3, u3, h3, y3, yflat, xt, xt3
            mx.clear_cache()
    F = 6.0 * D * H
    steel3 = acc["g"] + acc["u"] + acc["y"]
    pipe = steel3 + acc["act"] + acc["build"] + acc["unpad"]
    return {"Mt": Mt, "n_chunks": nchunk, "tiles": tiles, "rows_padded": pads,
            "pad_waste": pads / N, "machine_bm64_rows": mach,
            "ms": acc, "steel3_ms": steel3, "steel_pipe_ms": pipe,
            "steel3_TF": F * N / (steel3 / 1e3) / 1e12,
            "steel_pipe_TF": F * N / (pipe / 1e3) / 1e12}, {}


def price_m1(I, L, E, D, H, N, sidx_np, weights, ebatch):
    """The fork's shipping fallback form (exl3_moe.py:920-927): M=1 per pair,
    rhs-only sorted gather (steel gather_mm_rhs/_nax), weight runs reused.
    Decode EXCLUDED; rows are raw (values do not affect kernel time).
    ``weights`` or None -> per-chunk decode, same convention as price_steel."""
    k, cb, gt, dt = L["k"], L["cb"], L["gt"], L["dt"]
    c = np.bincount(sidx_np.astype(np.int64), minlength=E)
    starts = np.concatenate([[0], np.cumsum(c)])
    x_rows = I["x"][:N]
    F = 6.0 * D * H
    t = {"g": 0.0, "u": 0.0, "y": 0.0, "act": 0.0}
    nchunk = (E + ebatch - 1) // ebatch if ebatch else 1
    for ci in range(nchunk):
        e0 = ci * ebatch if ebatch else 0
        e1 = min(E, e0 + ebatch) if ebatch else E
        if weights is not None:
            wg, wu, wd = weights
        else:
            wg = decode_chunk(L["gu"], k, cb, gt, e0 * gt, e1 * gt, True)
            wu = decode_chunk(L["gu"], k, cb, gt, E * gt + e0 * gt, E * gt + e1 * gt, True)
            wd = decode_chunk(L["dn"], k, cb, dt, e0 * dt, e1 * dt, True)
            mx.eval(wg, wu, wd)
        r0, r1 = int(starts[e0]), int(starts[e1])
        Rc = r1 - r0
        a1 = x_rows[r0:r1].reshape(Rc, 1, 1, D)
        idx = mx.array((sidx_np[r0:r1] - e0).reshape(-1, 1).astype(np.uint32))
        t["g"] += tmed(lambda: mx.gather_mm(a1, wg, None, idx, sorted_indices=True)) * 1e3
        t["u"] += tmed(lambda: mx.gather_mm(a1, wu, None, idx, sorted_indices=True)) * 1e3
        g = mx.gather_mm(a1, wg, None, idx, sorted_indices=True)
        u = mx.gather_mm(a1, wu, None, idx, sorted_indices=True)
        mx.eval(g, u)
        t["act"] += tmed(lambda: MOE._moe_gate_activation(
            g.reshape(Rc, H), u.reshape(Rc, H), "silu_clamp")) * 1e3
        h = MOE._moe_gate_activation(g.reshape(Rc, H), u.reshape(Rc, H), "silu_clamp")
        mx.eval(h)
        t["y"] += tmed(lambda: mx.gather_mm(h.reshape(Rc, 1, 1, H), wd, None, idx,
                                            sorted_indices=True)) * 1e3
        del g, u, h
        if weights is None:
            del wg, wu, wd
    pipe = t["g"] + t["u"] + t["y"] + t["act"]
    return {"launch_gate_ms": t["g"], "launch_up_ms": t["u"],
            "launch_dn_ms": t["y"], "act_ms": t["act"], "m1_pipe_ms": pipe,
            "m1_TF": F * N / (pipe / 1e3) / 1e12, "n_chunks": nchunk}


def price_mix(name, ids, E, L, wlabel, real_sg=None):
    D, H = L["D"], L["H"]
    N = int(ids.size)
    I = C.launch_inputs(mx, MOE, ids, E, D)
    sidx_np = np.asarray(I["sidx"])
    c = np.bincount(sidx_np.astype(np.int64), minlength=E)
    st = {"pairs": N, "experts_hit": int((c > 0).sum()), "max_rows": int(c.max()),
          "avg_rows": float(N / max(1, (c > 0).sum()))}
    log("%s: E=%d pairs=%d experts_hit=%d maxM=%d weights=%s"
        % (name, E, N, st["experts_hit"], st["max_rows"], wlabel))
    rec = {"E": E, "D": D, "H": H, "k": L["k"], "cb": L["cb"], "weights": wlabel,
           "idx_stats": st}
    k, cb, gt, dt = L["k"], L["cb"], L["gt"], L["dt"]

    # ---------------- seg baseline (same host)
    xd_pad = mx.random.normal((N + 64, H), key=mx.random.key(3)).astype(mx.float16)
    mx.eval(xd_pad)
    rec["seg"] = seg_launches(I, L, E, D, H, N, xd_pad)
    F = 6.0 * D * H
    for tag, ms in (("seg3", rec["seg"]["seg3_ms"]),
                    ("seg_pipe", rec["seg"]["seg_pipe_ms"])):
        rec["seg"][tag + "_TF"] = F * N / (ms / 1e3) / 1e12
    log("  seg3 %.2f ms (%.2f TF)  seg_pipe %.2f ms (%.2f TF)"
        % (rec["seg"]["seg3_ms"], rec["seg"]["seg3_TF"],
           rec["seg"]["seg_pipe_ms"], rec["seg"]["seg_pipe_TF"]))
    if real_sg is not None:
        try:
            xin = mx.random.normal((1, N // KK, D), key=mx.random.key(4)).astype(mx.float16)
            midx = mx.array(ids.reshape(1, N // KK, KK).astype(np.uint32))
            mx.eval(xin, midx)
            t_full = tmed(lambda: real_sg._prefill(xin, midx))
            rec["seg"]["full_prefill_ms"] = t_full * 1e3
            rec["seg"]["full_prefill_TF"] = F * N / t_full / 1e12
            log("  full _prefill %.2f ms (%.2f TF)"
                % (rec["seg"]["full_prefill_ms"], rec["seg"]["full_prefill_TF"]))
        except Exception as e:
            log("  full _prefill timing failed:", type(e).__name__, e)
    flushjson()

    # ---------------- decode once (decode-EXCLUDED prices below; the fork
    # decodes gu with ONE whole-array call for both gate and up -- mirrored
    # here; the chunked variant slices the stacked trellis and np-copies the
    # slice because the decode kernel reads a contiguous tiling).
    # INT32 CAP: mlx 0.32.0.dev fast.metal_kernel output shapes must fit
    # int32 -- observed OverflowError at 2E*D*H elements (=2.265e9 at E=192)
    # and the fork's own _mm_fallback_viable checks the byte-size form of the
    # same limit (exl3_moe.py:867-878).  One-shot whole-E decode is therefore
    # IMPOSSIBLE at DSv4.1 dims for E >~182; Dir D requires chunking.
    lim_whole = (2 ** 31 - 1) // (D * H * 2)
    ebatch = EBATCH if EBATCH > 0 else (min(E, lim_whole) if E > lim_whole else 0)
    dec_bytes = (2 * E * D * H + E * H * D) * 2
    ranges = ([(i * ebatch, min(E, (i + 1) * ebatch))
               for i in range((E + ebatch - 1) // ebatch)] if ebatch else [(0, E)])
    if ebatch:
        t_dec = 0.0
        for e0, e1 in ranges:
            t_dec += (
                tmed(lambda: decode_chunk(L["gu"], k, cb, gt, e0 * gt, e1 * gt, True),
                     reps=REPS_DEC, warm=1)
                + tmed(lambda: decode_chunk(L["gu"], k, cb, gt, E * gt + e0 * gt,
                                            E * gt + e1 * gt, True),
                       reps=REPS_DEC, warm=1)
                + tmed(lambda: decode_chunk(L["dn"], k, cb, dt, e0 * dt, e1 * dt, True),
                       reps=REPS_DEC, warm=1))
        weights = None
    else:
        t_dec = (tmed(lambda: decode_full_eg_mlx(L["gu"], k, cb, tiles_per_e=gt),
                      reps=REPS_DEC, warm=1)
                 + tmed(lambda: decode_full_eg_mlx(L["dn"], k, cb, tiles_per_e=dt),
                        reps=REPS_DEC, warm=1))
        W_gu = decode_full_eg_mlx(L["gu"], k, cb, tiles_per_e=gt)
        wd = decode_full_eg_mlx(L["dn"], k, cb, tiles_per_e=dt)
        wg, wu = W_gu[:E], W_gu[E:]
        mx.eval(wg, wu, wd)
        assert wg.shape == (E, D, H) and wu.shape == (E, D, H) and wd.shape == (E, H, D), \
            (wg.shape, wu.shape, wd.shape)
        weights = (wg, wu, wd)
    rec["decode"] = {"chunked_at": ebatch, "chunks": len(ranges),
                     "GM_EBATCH": EBATCH, "whole_e_limit": lim_whole,
                     "total_ms": t_dec * 1e3, "fp16_bytes": dec_bytes,
                     "GBps": dec_bytes / t_dec / 1e9 if t_dec else 0.0}
    log("  decode once: %.2f ms (%.2f GB fp16, %.0f GB/s)%s"
        % (rec["decode"]["total_ms"], dec_bytes / 1e9, rec["decode"]["GBps"],
           "" if not ebatch else " [%d chunks of %d, int32 cap]" % (len(ranges), ebatch)))
    flushjson()

    # ---------------- tiled batched gather price per Mt
    rec["steel"] = {}
    for Mt in MTS:
        r, _ = price_steel(I, L, E, D, H, N, sidx_np, Mt, weights, ebatch)
        rec["steel"]["Mt%d" % Mt] = r
        log("  Mt=%-3d tiles=%-4d pad=%.2fx steel3 %.2f TF pipe %.2f TF (build %.1f act %.1f unpad %.1f ms)"
            % (Mt, r["tiles"], r["pad_waste"], r["steel3_TF"], r["steel_pipe_TF"],
               r["ms"]["build"], r["ms"]["act"], r["ms"]["unpad"]))
        flushjson()

    # ---------------- fork's shipping M=1 fallback form
    try:
        rec["m1_fallback"] = price_m1(I, L, E, D, H, N, sidx_np, weights, ebatch)
        log("  m1 fallback (fork's form): %.2f TF"
            % rec["m1_fallback"]["m1_TF"])
    except Exception as e:
        log("  m1 fallback failed:", type(e).__name__, e)
        R["notes"].append("m1 fallback %s: %s" % (name, e))

    # ---------------- drift-cancelled confirm: alternate seg vs steel pipes
    # (this host shares the GPU with other live sessions; phase-separated
    # medians can drift.  Interleaved rounds cancel slow drift, as in
    # bench/p20_moe_partition_bench.py.)
    best = max(rec["steel"].values(), key=lambda r: r["steel_pipe_TF"]) \
        if rec["steel"] else None
    if best and os.environ.get("GM_CONFIRM", "1") == "1":
        try:
            nrounds = int(os.environ.get("GM_CONFIRM_ROUNDS", "5"))
            pairs = []
            for _ in range(nrounds):
                s_ms = seg_launches(I, L, E, D, H, N, xd_pad)["seg_pipe_ms"]
                t_ms = price_steel(I, L, E, D, H, N, sidx_np, best["Mt"],
                                   weights, ebatch)[0]["steel_pipe_ms"]
                pairs.append((s_ms, t_ms))
            s_med = float(np.median([p[0] for p in pairs]))
            t_med = float(np.median([p[1] for p in pairs]))
            r_med = float(np.median([p[1] / p[0] for p in pairs]))
            rec["confirm_interleaved"] = {
                "Mt": best["Mt"], "rounds": nrounds,
                "seg_pipe_ms": s_med, "steel_pipe_ms": t_med,
                "seg_pipe_ms_all": [p[0] for p in pairs],
                "steel_pipe_ms_all": [p[1] for p in pairs],
                "ratio_per_round": [p[1] / p[0] for p in pairs],
                "ratio": r_med,
                "seg_pipe_TF": F * N / (s_med / 1e3) / 1e12,
                "steel_pipe_TF": F * N / (t_med / 1e3) / 1e12,
                "note": "ratio = median of per-round steel/seg pipe ratios"}
            log("  confirm (interleaved x%d): steel %.2f TF vs seg %.2f TF -> "
                "%.2fx (medians) / %.2fx (median-of-rounds)"
                % (nrounds, rec["confirm_interleaved"]["steel_pipe_TF"],
                   rec["confirm_interleaved"]["seg_pipe_TF"],
                   (F * N / (t_med / 1e3)) / (F * N / (s_med / 1e3)), r_med))
        except Exception as e:
            log("  confirm failed:", type(e).__name__, e)
            R["notes"].append("confirm %s: %s" % (name, e))
        flushjson()

    # ---------------- price summary
    best = max(rec["steel"].values(), key=lambda r: r["steel_pipe_TF"]) \
        if rec["steel"] else None
    if best:
        conf = rec.get("confirm_interleaved")
        seg_pipe_tf = conf["seg_pipe_TF"] if conf else rec["seg"]["seg_pipe_TF"]
        steel_pipe_tf = conf["steel_pipe_TF"] if conf else best["steel_pipe_TF"]
        pipe = conf["steel_pipe_ms"] if conf else best["steel_pipe_ms"]
        ratio_head = conf["ratio"] if conf else \
            best["steel_pipe_TF"] / rec["seg"]["seg_pipe_TF"]
        dec_ms = rec["decode"]["total_ms"]
        pipe_inc = pipe + dec_ms
        present_frac = st["experts_hit"] / E
        rec["price"] = {
            "best_Mt": best["Mt"], "weights": wlabel,
            "headline": "interleaved_confirm" if conf else "phase_separated",
            "steel_pipe_TF": steel_pipe_tf, "seg_pipe_TF": seg_pipe_tf,
            "ratio_vs_seg_pipe": ratio_head,
            "phase_separated": {"steel_pipe_TF": best["steel_pipe_TF"],
                                "seg_pipe_TF": rec["seg"]["seg_pipe_TF"],
                                "ratio": best["steel_pipe_TF"] / rec["seg"]["seg_pipe_TF"]},
            "ratio_vs_seg3": best["steel3_TF"] / rec["seg"]["seg3_TF"],
            "steel_pipe_decode_incl_ms": pipe_inc,
            "steel_pipe_decode_incl_TF": F * N / (pipe_inc / 1e3) / 1e12,
            "ratio_decode_incl_vs_seg_pipe": (F * N / (pipe_inc / 1e3) / 1e12)
                                             / seg_pipe_tf,
            "decode_share_of_pipe_pct": dec_ms / pipe * 100,
            "decode_amortized_routed_frac": {"frac": present_frac,
                "ms": dec_ms * present_frac,
                "ratio_est": (F * N / ((pipe + dec_ms * present_frac) / 1e3) / 1e12)
                             / seg_pipe_tf},
            "published_ref": PUBLISHED if name == "real" else
                             {"seg_uniform192_TF": PUBLISHED["seg_uniform192_TF"]},
        }
        raw_win = ratio_head > 1.0
        marginal = abs(ratio_head - 1.0) <= 0.10
        dec_ratio = rec["price"]["ratio_decode_incl_vs_seg_pipe"]
        # Dir D carries the decode intrinsically; a GEMM-side ratio within
        # noise of 1.0 cannot absorb it, so the decode-included number decides.
        final_win = raw_win and not (marginal and dec_ratio < 1.0)
        rec["verdict_rule"] = ("WIN iff interleaved steel/seg ratio > 1 and "
                               "(|ratio-1| > 0.10 or decode-included ratio > 1); "
                               "otherwise LOSS")
        rec["price"]["marginal_within_noise"] = bool(marginal)
        rec["price"]["raw_ratio_win"] = bool(raw_win)
        rec["verdict"] = ("P3: steel-gather price %.2f TF = %.2fx vs seg kernel %.2f TF => %s%s"
                          % (steel_pipe_tf, ratio_head, seg_pipe_tf,
                             "WIN" if final_win else "LOSS",
                             (" [marginal, |ratio-1|<=0.10; decode-included %.2fx]"
                              % dec_ratio) if marginal else ""))
    return rec


def selfcheck():
    """E=8 synthetic: tiled gather path vs seg kernel, same weights+rows."""
    E, Mt = 8, 32
    Lc = C.synth_layer(mx, E, seed=5)
    D, H = Lc["D"], Lc["H"]
    k, cb, gt, dt = Lc["k"], 2, Lc["gt"], Lc["dt"]
    counts = np.array([70, 5, 0, 33, 64, 2, 100, 238])  # total 512, multi-tile+pad
    ids = C.counts_ids(counts, seed=3)
    N = int(ids.size)
    I = C.launch_inputs(mx, MOE, ids, E, D)
    wg = G.decode_full_eg_mlx(Lc["gu"], k, cb, tiles_per_e=gt)
    wd = G.decode_full_eg_mlx(Lc["dn"], k, cb, tiles_per_e=dt)
    mx.eval(wg, wd)
    # seg reference (chained, same h)
    g_s = G.inner_mm_seg_mlx(I["x"], Lc["gu"], k, cb, I["tab"], I["nbr"],
                             n_rows=N, tn_base=0, tiles_per_e=gt, out_e=H)
    u_s = G.inner_mm_seg_mlx(I["x"], Lc["gu"], k, cb, I["tab"], I["nbr"],
                             n_rows=N, tn_base=E * gt, tiles_per_e=gt, out_e=H)
    mx.eval(g_s, u_s)
    h_s = MOE._moe_gate_activation(g_s, u_s, "silu_clamp")
    h_s_pad = mx.concatenate([h_s, mx.zeros((64, H), dtype=h_s.dtype)])
    y_s = G.inner_mm_seg_mlx(h_s_pad, Lc["dn"], k, cb, I["tab"], I["nbr"],
                             n_rows=N, tn_base=0, tiles_per_e=dt, out_e=D)
    mx.eval(h_s, y_s)
    # tiled path
    c = np.bincount(np.asarray(I["sidx"]).astype(np.int64), minlength=E)
    starts = np.concatenate([[0], np.cumsum(c)])
    eids, srcs, valid = tile_tables(c, starts, 0, E, Mt)
    T = int(eids.size)
    x_rows = I["x"][:N]
    xt = (x_rows[mx.array(srcs)] * mx.array(valid.astype(np.float16))[:, None]).astype(mx.float16)
    xt3 = xt.reshape(T, Mt, D)
    lhs, rhs = mx.arange(T, dtype=mx.int32), mx.array(eids)
    wg_gate, wg_up = wg[:E], wg[E:]      # stacked as [all gates | all ups]
    g_t = mx.gather_mm(xt3, wg_gate, lhs, rhs)
    u_t = mx.gather_mm(xt3, wg_up, lhs, rhs)
    mx.eval(g_t, u_t)
    h_t = MOE._moe_gate_activation(g_t, u_t, "silu_clamp")
    mx.eval(h_t)
    y_t = mx.gather_mm(h_t, wd, lhs, rhs)
    mx.eval(y_t)
    vp = mx.array(np.flatnonzero(valid).astype(np.int32))
    y_sorted = y_t.reshape(T * Mt, D)[vp]
    mx.eval(y_sorted)

    def rel(a, b):
        a = np.asarray(a.astype(mx.float32)); b = np.asarray(b.astype(mx.float32))
        return float(np.abs(a - b).max() / (np.abs(a).max() + 1e-9))
    out = {"E": E, "N": N, "Mt": Mt, "tiles": T, "rows_padded": T * Mt,
           "gate_rel": rel(g_s.reshape(N, H), (g_t.reshape(T * Mt, H)[vp])),
           "dn_rel": rel(y_s, y_sorted)}
    # chunked-decode path correctness: sliced+np-copied decode must be
    # bit-exact vs the whole-array decode of the same tile range (the raw
    # strided slice is also probed to document whether the copy is needed)
    Wg_whole = decode_full_eg_mlx(Lc["gu"], k, cb, tiles_per_e=gt)
    sl_c = decode_chunk(Lc["gu"], k, cb, gt, 0, 2 * gt, True)
    mx.eval(Wg_whole, sl_c)
    out["chunk_slice_bitdiff_copy"] = int(
        (np.asarray(Wg_whole[:2].view(mx.uint16)) != np.asarray(sl_c.view(mx.uint16))).sum())
    try:
        sl_r = G.decode_full_eg_mlx(Lc["gu"][:, 0:2 * gt, :], k, cb, tiles_per_e=gt)
        mx.eval(sl_r)
        out["chunk_slice_bitdiff_raw"] = int(
            (np.asarray(Wg_whole[:2].view(mx.uint16)) != np.asarray(sl_r.view(mx.uint16))).sum())
        del sl_r
    except Exception as e:
        out["chunk_slice_bitdiff_raw"] = "raised: %s" % type(e).__name__
    out["pass"] = bool(out["gate_rel"] < 0.02 and out["dn_rel"] < 0.02
                       and out["chunk_slice_bitdiff_copy"] == 0)
    log("selfcheck: gate_rel=%.4f dn_rel=%.4f slice_copy_bitdiff=%d slice_raw_bitdiff=%s -> %s"
        % (out["gate_rel"], out["dn_rel"], out["chunk_slice_bitdiff_copy"],
           out["chunk_slice_bitdiff_raw"], "PASS" if out["pass"] else "FAIL"))
    R["selfcheck"] = out
    del Lc, I, wg, wd, g_s, u_s, h_s, h_s_pad, y_s, g_t, u_t, h_t, y_t, y_sorted, Wg_whole, sl_c
    mx.clear_cache()
    return out["pass"]


def main():
    log("start mlx=%s gemv_metal=%s seg=%s" % (mx.__version__, G.__file__,
                                               G._MM_SEG_VERSION))
    R["primitive"]["gather_mm_doc"] = (mx.gather_mm.__doc__ or "").strip().splitlines()[0]
    flushjson()
    ok = selfcheck()
    flushjson()

    ck_present = os.path.isdir(GM_CK)
    use_real = ck_present and REAL_CK != "0"
    if REAL_CK == "1" and not ck_present:
        log("REAL_CK=1 but checkpoint missing at %s -- synthetic fallback" % GM_CK)
    R["real_ck"] = use_real
    R["checkpoint"] = GM_CK
    log("weights=%s (ckpt %s present=%s)" % ("real" if use_real else "synthetic",
                                             GM_CK, ck_present))
    synths, sg_cache = {}, {}
    for name in MIXES:
        try:
            if name == "real":
                tp = C.env_trace_path(ROOT)
                if not os.path.exists(tp):
                    log("skip real: trace missing %s" % tp)
                    R["notes"].append("real: trace missing %s" % tp)
                    continue
                ids = C.real_chunk_ids(tp)
                E = 384
            elif name.startswith("uniform"):
                ids = C.uniform_ids(E_UNIFORM, 12288)
                E = E_UNIFORM
            elif name.startswith("skew"):
                ids = C.skew_ids(E_UNIFORM, 12288)
                E = E_UNIFORM
            else:
                log("skip unknown mix", name)
                continue
            if use_real:
                from mlx_lm.models.exl3.loader import Exl3Checkpoint, load_experts
                ck = sg_cache.get("ck") or Exl3Checkpoint(GM_CK)
                sg_cache["ck"] = ck
                sg = load_experts(ck, 0, n_experts=None if E == 384 else E,
                                  rank=0, world=2, activation="silu_clamp")
                L = {"gu": sg._gu_trellis, "dn": sg._dn_trellis, "gt": sg._gu_tiles,
                     "dt": sg._dn_tiles, "D": sg.input_dims, "H": sg.hidden_dims,
                     "k": sg._k, "cb": int(sg._cb), "E": sg.num_experts}
                wlabel = "real"
            else:
                if E not in synths:
                    synths[E] = C.synth_layer(mx, E)
                ss = synths[E]
                L = {"gu": ss["gu"], "dn": ss["dn"], "gt": ss["gt"], "dt": ss["dt"],
                     "D": ss["D"], "H": ss["H"], "k": ss["k"], "cb": 2, "E": E}
                sg, wlabel = None, "synthetic"
            if E == 384 and wlabel == "synthetic":
                wlabel = "synthetic"
            R["mixes"][name] = price_mix(name, ids, E, L,
                                         wlabel if wlabel == "synthetic" else "real",
                                         real_sg=sg)
            flushjson()
            del L
            if sg is not None:
                del sg
            mx.clear_cache()
        except Exception as e:
            log("mix %s FAILED" % name, type(e).__name__, e)
            traceback.print_exc()
            R["notes"].append("mix %s: %s: %s" % (name, type(e).__name__, e))
            try:
                mx.clear_cache()   # don't leak Metal buffers into the next mix
            except Exception:
                pass
            flushjson()
    R["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    flushjson()
    # ---------------- verdict
    real = R["mixes"].get("real", {})
    line = None
    if real.get("verdict"):
        line = real["verdict"]
        p = real["price"]
        log(line)
        log("  (decode included once: %.2f TF = %.2fx; decode %.1f ms = %.0f%% of pipe; "
            "amortized over routed %.0f%%: %.2fx)"
            % (p["steel_pipe_decode_incl_TF"], p["ratio_decode_incl_vs_seg_pipe"],
               real["decode"]["total_ms"], p["decode_share_of_pipe_pct"],
               p["decode_amortized_routed_frac"]["frac"] * 100,
               p["decode_amortized_routed_frac"]["ratio_est"]))
    else:
        for nm, rr in R["mixes"].items():
            if rr.get("verdict"):
                line = "%s [%s]" % (rr["verdict"], nm)
                break
    R["verdict"] = line or "P3: no priced mix (see notes)"
    R["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    flushjson()
    print(R["verdict"])
    log("DONE ->", GM_JSON)


if __name__ == "__main__":
    main()
