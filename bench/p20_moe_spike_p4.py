#!/usr/bin/env python3
"""Phase-20 MoE spike P4 -- `wblk` DOUBLE-BUFFERED (ping-pong) v19d variant (v19p).

QUESTION (pre-registered gate #2 of the spike): the v19d segmented MoE GEMM
serialises trellis decode and simdgroup mma inside each 32-deep k-stage:
[barrier; decode ks; barrier; mma ks].  no_tload measured the mma as stalled on
weight-operand supply, so P4 asks: does a software-pipelined PING-PONG wblk --
decode stage ks+1 into buffer (ks+1)&1 while stage ks's guarded mma runs on
buffer ks&1, ONE barrier per stage -- beat v19d by >=5% on the CHAINED
_full_prefill_ms while staying bit-exact?

STRUCTURE emitted (source-to-source transform of the guarded v19d source):

    threadgroup half wblk[2u * 32u * 72u];   // v19p ping-pong: 2 x (32x72)
    ...
    uint stages = in_tiles >> 1u;             // in_tiles is even (>=32)
    threadgroup_barrier(mem_flags::mem_threadgroup);
    <decode stage 0 -> buffer 0>              // v19d decode verbatim + buf idx
    for (uint ks = 0u; ks < stages; ks++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (ks + 1u < stages) {
            <decode stage ks+1 -> buffer (ks+1)&1>
        }
        <v19d guarded mma block VERBATIM; wrow = &wblk[(ks&1)*32u*72u + ...]>
    }

The mma block, the decode expression, the wblk element layout and the
accumulation order (ks asc, k2 asc, r asc) are IDENTICAL to v19d: the ping-pong
only changes WHEN operand bytes are produced, so results must be 0-diff
fp16-as-uint16.  Any bitdiff != 0 ABORTS before timing (valid negative result)
and is reported with sample differing offsets.

MEASUREMENTS in this script
  1. BIT-EXACT: v19d vs v19p as RAW kernels on identical inputs (gu gate/up +
     dn shapes, real E=384 trace ids + uniform E=192 ids, plus an E=8 smoke)
     AND the full CHAINED `_prefill` output on the same sg/x/idx -- fp16-as-
     uint16 0-diff + maxabs.
  2. IN-CHAIN: `_full_prefill_ms` via A.time_launches for v19d / v19p / v19dnd
     (v19d built with EXL3_MM_NODECODE=1 at build time) on real (E=384 trace)
     and uniform (E=192) mixes; P4_ROUNDS alternating-order rounds; median +
     sigma + delta%% + gate (>=5%% AND >3*sigma of the real v19d band); the
     isolated per-launch (seg3) numbers are reported alongside.
  3. One-line verdict + JSON to P4_JSON.

NON-INVASIVE: kernels are pre-populated into `G._mm_seg_kernels` under new
version tags; gemv_metal.py is never edited.

Env:
  P20_PKG=<scratch mlx_lm pkg>   node only; gemv_metal origin is asserted
  P4_ROUNDS=5                   rounds per (mix, version); order alternates
  P4_MIXES=real,uniform         real (E=384 trace) / uniform|uniform192 / skew
  P4_VERS=v19d,v19p,v19dnd      v19dnd only used when P4_NODECODE=1
  P4_NODECODE=1                 include the v19d+EXL3_MM_NODECODE=1 cross-check
  P4_REPS                       override A.REPS (tmed reps; default MB_REPS=7)
  P4_JSON=/tmp/p20moe_spike/p20_moe_spike_p4.json
  MB_TRACE=<trace json>  MB_REPS  REAL_CK=auto|1|0  (real ckpt auto-detected)

Laptop smoke (synthetic trellis; checkpoint is node-only):
  env -u PYTHONPATH ~/repos/exo/.venv/bin/python bench/p20_moe_spike_p4.py
Node (authoritative; real weights + scratch pkg):
  P20_PKG=<scratch mlx_lm> REAL_CK=1 ~/repos/exo/.venv/bin/python \
      bench/p20_moe_spike_p4.py
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import sys
import time

sys.dont_write_bytecode = True  # never write .pyc into shared checkouts
os.environ.setdefault("EXL3_MM_SEG", "v19d")

HOME = os.path.expanduser("~")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
P20_PKG = os.environ.get("P20_PKG", "")
if P20_PKG:
    PIN = P20_PKG
elif os.path.isdir(os.path.join(ROOT, "mlx-lm", "mlx_lm")):
    PIN = os.path.join(ROOT, "mlx-lm")
else:
    PIN = HOME + "/repos/exo/mlx-lm"  # read-only import; never edited here
sys.path.insert(0, PIN)
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import mlx.core as mx  # noqa: E402
from mlx_lm.models.exl3 import gemv_metal as G  # noqa: E402
from mlx_lm.models.exl3 import exl3_moe as MOE  # noqa: E402
import p20_moe_partition_common as C  # noqa: E402
import p20_moe_autopsy as A  # noqa: E402

assert str(G.__file__).startswith(PIN), \
    "gemv_metal shadowed: %s (pin %s)" % (G.__file__, PIN)

HDR = "#include <metal_simdgroup_matrix>\n#include <metal_stdlib>\nusing namespace metal;\n"
K = 3                       # 3-bit trellis (dsv41)
ROUNDS = int(os.environ.get("P4_ROUNDS", "5"))
MIXES = [m for m in os.environ.get("P4_MIXES", "real,uniform").split(",") if m]
VERS = [v for v in os.environ.get("P4_VERS", "v19d,v19p,v19dnd").split(",") if v]
DO_NODECODE = os.environ.get("P4_NODECODE", "1") == "1"
OUT = os.environ.get("P4_JSON", "/tmp/p20moe_spike/p20_moe_spike_p4.json")
REAL_CK = os.environ.get("REAL_CK", "auto")  # auto|1|0
CK = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
GATE_PCT = 5.0
os.makedirs(os.path.dirname(OUT), exist_ok=True)
if os.environ.get("P4_REPS"):
    A.REPS = int(os.environ["P4_REPS"])  # time_launches' tmed reads A.REPS

# The v19d reference must be the TRUE production source: drop any stray
# EXL3_MM_NODECODE from the shell (the v19dnd build sets it locally, below).
if os.environ.pop("EXL3_MM_NODECODE", None) is not None:
    print("[p4] warning: cleared stray EXL3_MM_NODECODE from the environment",
          flush=True)

R = {"script": "bench/p20_moe_spike_p4.py", "host": socket.gethostname(),
     "mlx": mx.__version__, "gemv_metal": G.__file__, "pin": PIN,
     "gemv_metal_sha256": hashlib.sha256(open(G.__file__, "rb").read()).hexdigest(),
     "seg_version_default": G._MM_SEG_VERSION,
     "rounds": ROUNDS, "mixes": MIXES, "vers": VERS, "nodecode": DO_NODECODE,
     "gate": {"pct": GATE_PCT, "rule": "delta>=5%% AND delta>3*sigma(real/v19d)"},
     "tf_convention": "TF = 6*D*H*pairs / _full_prefill_ms",
     "started": time.strftime("%Y-%m-%d %H:%M:%S"),
     "res": {}, "bitexact": {"cases": [], "fail": 0}, "notes": []}


def log(*a):
    print("[p4]", *a, flush=True)


def flushjson():
    json.dump(R, open(OUT, "w"), indent=1, default=float)


def drain(v):
    if isinstance(v, (tuple, list)):
        mx.eval(*v)
    else:
        mx.eval(v)


# ------------------------------------------------------------------ v19p source
_A_WBLK_DECL = "    threadgroup half wblk[32u * 72u];   // 2 in-tiles x 4 out-tiles"
_A_WBLK_DECL_P = ("    threadgroup half wblk[2u * 32u * 72u];"
                  "   // v19p ping-pong: 2 x (2 in-tiles x 4 out-tiles)")
_A_LOOP = "    for (uint ks = 0u; ks < stages; ks++) {"
_A_BAR = "        threadgroup_barrier(mem_flags::mem_threadgroup);"
_A_DEC = "        for (uint it2 = 0u; it2 < 2u; it2++) {"
_A_STORE = "    // store: fp32 frags -> tg scratch -> half device rows within the segment"
_A_WSTORE = "wblk[(it2 * 16u + 2u * rp + j) * 72u + lt * 16u + dcol]"
_A_WROW = "&wblk[k2 * 8u * 72u + sgc * 32u];"
_A_TK = "            uint tk = ks * 2u + it2;"
_BUF = "32u * 72u"


def _dedent4(text):
    return "\n".join((l[4:] if l.startswith("    ") else l) for l in text.split("\n"))


def _indent4(text):
    return "\n".join(("    " + l if l.strip() else l) for l in text.split("\n"))


def _with_buf(text, buf):
    """Add the ping-pong buffer index to the decode's wblk store (layout kept)."""
    new = ("wblk[(%s) * %s + (it2 * 16u + 2u * rp + j) * 72u + lt * 16u + dcol]"
           % (buf, _BUF))
    assert text.count(_A_WSTORE) == 1
    return text.replace(_A_WSTORE, new)


def _with_wrow(text, buf):
    """Add the buffer index to every mma B-fragment row pointer (5 sites)."""
    new = "&wblk[(%s) * %s + k2 * 8u * 72u + sgc * 32u];" % (buf, _BUF)
    assert text.count(_A_WROW) == 5, text.count(_A_WROW)
    return text.replace(_A_WROW, new)


def v19p_from_guarded(src: str) -> str:
    """Re-emit the guarded v19d main loop as a double-buffered pipeline.

    v19d (in):   for ks { barrier; decode ks -> wblk; barrier; mma ks on wblk }
    v19p (out):  barrier; decode 0 -> buf0
                 for ks { barrier; [if ks+1<stages: decode ks+1 -> buf (ks+1)&1];
                          mma ks on buf ks&1 }          (v19d mma block verbatim)
    """
    assert src.count(_A_WBLK_DECL) == 1
    assert src.count(_A_LOOP) == 1
    assert src.count(_A_STORE) == 1
    i_loop, i_store = src.index(_A_LOOP), src.index(_A_STORE)
    pre, loop, post = src[:i_loop], src[i_loop:i_store], src[i_store:]
    b1 = loop.index(_A_BAR)
    d0 = loop.index(_A_DEC, b1)
    b2 = loop.index(_A_BAR, d0)
    dec = loop[d0:b2]                       # decode block, verbatim, ends "\n"
    tail = loop[b2 + len(_A_BAR):].rstrip()
    assert tail.endswith("    }"), repr(tail[-40:])
    mma = tail[: len(tail) - len("    }")].rstrip() + "\n"   # guarded mma, verbatim
    # anchors: the pieces must be exactly the shapes v19d has
    assert dec.count(_A_WSTORE) == 1 and dec.count(_A_TK) == 1
    assert mma.count(_A_WROW) == 5 and mma.count("ks * 32u") == 5
    assert "ks" not in pre[pre.index("uint stages"):]

    dec0 = _with_buf(dec, "0u").replace(_A_TK, "            uint tk = it2;   // stage 0")
    dec1 = _with_buf(dec, "((ks + 1u) & 1u)").replace(
        _A_TK, "            uint tk = (ks + 1u) * 2u + it2;")
    mma_k = _with_wrow(mma, "(ks & 1u)")
    prologue = (_dedent4(_A_BAR) + "\n" + _dedent4(dec0))
    body = (_A_LOOP + "\n"
            + _A_BAR + "\n"
            + "        if (ks + 1u < stages) {\n" + _indent4(dec1) + "        }\n"
            + mma_k
            + "    }\n")
    out = (pre.replace(_A_WBLK_DECL, _A_WBLK_DECL_P, 1)
           + "    // v19p (P4): decode(ks+1) into buffer (ks+1)&1 overlaps mma(ks)\n"
           + "    // on buffer ks&1; ONE barrier per stage. Accumulation order and\n"
           + "    // the guarded mma block are v19d's, so the result is 0-diff.\n"
           + prologue + body + "\n" + post)
    return out


def v19p_source(k, cb) -> str:
    return v19p_from_guarded(G._mm_seg_guarded(G._mm_seg_source(k, cb)))


# ------------------------------------------------------------ kernel management
_BUILT = {}


def kern_for(tag, cb_i):
    """Build a tagged seg kernel and pre-populate G._mm_seg_kernels so that
    inner_mm_seg_mlx (and therefore EXL3SwitchGLU._prefill) runs it unchanged."""
    if (tag, cb_i) in _BUILT:
        return _BUILT[(tag, cb_i)]
    if tag == "v19d":
        src = G._mm_seg_guarded(G._mm_seg_source(K, G.CodebookMode(cb_i)))
    elif tag == "v19p":
        src = v19p_source(K, G.CodebookMode(cb_i))
    elif tag == "v19dnd":
        os.environ["EXL3_MM_NODECODE"] = "1"      # must precede _mm_seg_source
        try:
            src = G._mm_seg_guarded(G._mm_seg_source(K, G.CodebookMode(cb_i)))
        finally:
            os.environ.pop("EXL3_MM_NODECODE", None)
    else:
        raise ValueError("unknown tag %r" % tag)
    name = "p4_seg_k%d_cb%d_%s" % (K, cb_i, tag)
    kern = mx.fast.metal_kernel(
        name=name,
        input_names=["xh", "trellis", "inv_perm", "blk_tab", "nbr", "dims"],
        output_names=["out"], source=src, header=HDR)
    G._mm_seg_kernels[(K, cb_i, tag)] = kern
    _BUILT[(tag, cb_i)] = (kern, src)
    log("built tag=%s cb=%d (src %d B, name %s)" % (tag, cb_i, len(src), name))
    return _BUILT[(tag, cb_i)]


def raw_run(kern, x, tr, I, Nout, tpe, tn_base):
    """The exact dispatch inner_mm_seg_mlx issues, with a caller-chosen kernel."""
    in_tiles, src_tiles, _ = tr.shape
    dims = mx.array([in_tiles, tpe, src_tiles, tn_base, Nout, I["nb_max"]],
                    dtype=mx.uint32)
    return kern(
        inputs=[x.reshape(-1), tr.reshape(-1).view(mx.uint32), G._inv_perm_u32(),
                I["tab"].reshape(-1), I["nbr"], dims],
        template=[("T", mx.float16)],
        grid=((Nout // 64) * 128, I["nb_max"], 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(I["N"], Nout)], output_dtypes=[mx.float16])[0]


def cmp_fp16(name, ref, got, **extra):
    mx.eval(ref, got)
    a = np.asarray(ref.view(mx.uint16))
    b = np.asarray(got.view(mx.uint16))
    nd = int((a != b).sum())
    fa = np.asarray(ref.astype(mx.float32)).reshape(-1)
    fb = np.asarray(got.astype(mx.float32)).reshape(-1)
    # synthetic random trellis words decode to arbitrary fp16 bit patterns and
    # can be non-finite; a bit-equal non-finite pair is still an exact match, so
    # maxabs is taken over the finite pairs only (nan-vs-nan handled by bitdiff).
    fin = np.isfinite(fa) & np.isfinite(fb)
    mad = float(np.max(np.abs(fa[fin] - fb[fin]))) if fin.any() else 0.0
    rec = {"case": name, "n_elem": int(a.size), "n_bitdiff": nd, "max_abs_diff": mad,
           "n_nonfinite_ref": int((~np.isfinite(fa)).sum()),
           "n_nonfinite_got": int((~np.isfinite(fb)).sum())}
    if nd:
        first = np.flatnonzero((a != b).reshape(-1))[:8].tolist()
        rec["first_diff_flat"] = first
    rec.update(extra)
    R["bitexact"]["cases"].append(rec)
    if nd:
        R["bitexact"]["fail"] += 1
    log("%-44s %s bitdiff=%d maxabs=%g elems=%d nonfinite=%d/%d" % (
        name, "OK " if nd == 0 else "FAIL", nd, mad, a.size,
        rec["n_nonfinite_ref"], rec["n_nonfinite_got"]))
    return nd


def run_prefill(sg, ver, x, m):
    G._MM_SEG_VERSION = ver
    y = sg._prefill(x, m)
    mx.eval(y)
    return y


def synth_sg(E, seed=0):
    """Full EXL3SwitchGLU on synthetic trellis + +-1 hadamard scale vectors, so
    the CHAINED `_prefill` (and therefore `_full_prefill_ms`) runs on the laptop.
    Weights are synthetic (values differ from the checkpoint; timing does not)."""
    L = C.synth_layer(mx, E, seed=seed)
    D, H = L["D"], L["H"]
    key = mx.random.key(1000 + E)
    ks = mx.random.split(key, 4)
    sign = lambda sh, k, lo, hi: (
        (mx.random.uniform(low=lo, high=hi, shape=sh, key=k) < 0.5) * 2 - 1
    ).astype(mx.float16)
    sg = MOE.EXL3SwitchGLU(
        gu_trellis=L["gu"], gu_suh=sign((E, 2, D), ks[0], 0, 1),
        gu_svh=sign((E, 2 * H), ks[1], 0, 1),
        dn_trellis=L["dn"], dn_suh=sign((E, H), ks[2], 0, 1),
        dn_svh=sign((E, D), ks[3], 0, 1),
        k=L["k"], cb=G.CodebookMode(2),  # MUL1 == the production checkpoint mode
        activation="silu_clamp")
    return sg, L


def build_mixes(use_real, ck_holder):
    """[(name, ids_flat, packed_idx, E, sg, L, cb_i)] for each requested mix."""
    out = []
    for name in MIXES:
        if name == "real":
            tp = C.env_trace_path(ROOT)
            if not os.path.exists(tp):
                log("skip real: trace missing %s" % tp)
                R["notes"].append("real: trace missing %s" % tp)
                continue
            rng = np.random.RandomState(20261010)
            packed, ntr = A.real_chunk_idx(tp, 0, 2048, rng)
            E = 384
        elif name in ("uniform", "uniform192"):
            name = "uniform"
            packed = A.build_idx_uniform(192, A.NPROD, np.random.RandomState(2026))
            E = 192
        elif name in ("skew", "skew192"):
            name = "skew"
            packed = A.build_idx_skew(192, A.NPROD, np.random.RandomState(2026))
            E = 192
        else:
            log("skip unknown mix", name)
            continue
        if use_real:
            ck = ck_holder.get("ck")
            if ck is None:
                from mlx_lm.models.exl3.loader import Exl3Checkpoint
                ck = ck_holder["ck"] = Exl3Checkpoint(CK)
            sg = ck_holder.setdefault("sg_by_e", {}).get(E)
            if sg is None:
                from mlx_lm.models.exl3.loader import load_experts
                sg = load_experts(ck, 0, n_experts=None if E == 384 else E,
                                  rank=0, world=2, activation="silu_clamp")
                ck_holder["sg_by_e"][E] = sg
            L, wlabel = None, "real"
        else:
            sg, L = synth_sg(E)
            wlabel = "synthetic"
        out.append((name, np.asarray(packed).reshape(-1).astype(np.int32), packed,
                    E, sg, L, int(sg._cb), wlabel))
    return out


def bitexact_mix(name, ids_flat, packed, E, sg, L, cb_i, label):
    """Raw-kernel + chained 0-diff for one mix (v19d reference)."""
    kern_d, _ = kern_for("v19d", cb_i)
    kern_p, _ = kern_for("v19p", cb_i)
    if L is not None:
        gu, dn, gt, dt, D, H = L["gu"], L["dn"], L["gt"], L["dt"], L["D"], L["H"]
    else:
        gu, dn = sg._gu_trellis, sg._dn_trellis
        gt, dt, D, H = sg._gu_tiles, sg._dn_tiles, sg.input_dims, sg.hidden_dims
    cases = (("gate", gu, 0), ("up", gu, E * gt), ("dn", dn, 0))
    for bname, tr, tn_base in cases:
        Kx = D if bname != "dn" else H
        Nout = H if bname != "dn" else D
        tpe = gt if bname != "dn" else dt
        I = C.launch_inputs(mx, MOE, ids_flat, E, Kx)
        ref = raw_run(kern_d, I["x"], tr, I, Nout, tpe, tn_base)
        got = raw_run(kern_p, I["x"], tr, I, Nout, tpe, tn_base)
        cmp_fp16("raw/%s/%s %s v19p==v19d" % (label, name, bname), ref, got,
                 E=E, N=I["N"], out_e=Nout, tn_base=tn_base, cb=cb_i)
        del I, ref, got
        mx.clear_cache()
    # chained: the whole EXL3SwitchGLU._prefill on identical x/idx
    x = mx.random.normal((1, packed.shape[1], D), key=mx.random.key(4)).astype(mx.float16)
    m = mx.array(packed, dtype=mx.uint32)
    mx.eval(x, m)
    ref = run_prefill(sg, "v19d", x, m)
    got = run_prefill(sg, "v19p", x, m)
    cmp_fp16("chain/%s/%s v19p==v19d" % (label, name), ref, got, E=E,
             N=int(m.size), shape="full _prefill", cb=cb_i)
    del x, m, ref, got
    mx.clear_cache()


def bench_mix(name, sg, packed, E, label, cb_i):
    """SAME alternating-rounds interleave as bench/p20_moe_spike_bench.py:
    every version runs ROUNDS rounds (order flips each round), each round is
    A.time_launches -> tmed() of the CHAINED `_full_prefill_ms` + the 3
    isolated seg launches."""
    E_, D, H = sg.num_experts, sg.input_dims, sg.hidden_dims
    F = 6.0 * D * H
    st = A.idx_stats(packed, E_)
    x = mx.random.normal((1, packed.shape[1], D), key=mx.random.key(11)).astype(mx.float16)
    m = mx.array(packed, dtype=mx.uint32)
    mx.eval(x, m)
    vers = [v for v in VERS if (v != "v19dnd" or DO_NODECODE)]
    for v in vers:                      # one untimed warmup per version
        kern_for(v, cb_i)
        G._MM_SEG_VERSION = v
        mx.eval(sg._prefill(x, m))
    acc = {v: {"full": [], "gu_gate": [], "gu_up": [], "dn": []} for v in vers}
    for rnd in range(ROUNDS):
        order = vers if rnd % 2 == 0 else vers[::-1]
        for v in order:
            G._MM_SEG_VERSION = v
            tl, _ = A.time_launches(x, m, sg)
            acc[v]["full"].append(tl["_full_prefill_ms"])
            for kk in ("gu_gate", "gu_up", "dn"):
                acc[v][kk].append(tl[kk]["ms"])
            log("  %s r%d %-6s full %.2f ms | seg3 gate %.2f up %.2f dn %.2f" % (
                name, rnd, v, tl["_full_prefill_ms"], tl["gu_gate"]["ms"],
                tl["gu_up"]["ms"], tl["dn"]["ms"]))
    rec = {"E": E_, "D": D, "H": H, "F": F, "weights": label, "cb": cb_i,
           "idx_stats": st, "vers": {}}
    for v in vers:
        fa = [float(z) for z in acc[v]["full"]]
        med = float(np.median(fa))
        seg = {kk: float(np.median(acc[v][kk])) for kk in ("gu_gate", "gu_up", "dn")}
        seg["seg3_ms"] = seg["gu_gate"] + seg["gu_up"] + seg["dn"]
        rec["vers"][v] = {
            "full_prefill_ms_all": fa, "full_prefill_ms": med,
            "full_prefill_ms_min": float(np.min(fa)),
            "full_prefill_ms_max": float(np.max(fa)),
            "full_prefill_ms_sigma": float(np.std(fa, ddof=1)) if len(fa) > 1 else 0.0,
            "TFLOPS": F * st["pairs"] / (med / 1e3) / 1e12,
            "isolated": seg,
            "isolated_seg3_TFLOPS": F * st["pairs"] / (seg["seg3_ms"] / 1e3) / 1e12,
        }
        log("  %s %-6s med %.2f ms (sigma %.3f) TF %.2f | seg3 %.2f ms (TF %.2f)" % (
            name, v, med, rec["vers"][v]["full_prefill_ms_sigma"],
            rec["vers"][v]["TFLOPS"], seg["seg3_ms"], rec["vers"][v]["isolated_seg3_TFLOPS"]))
    del x, m
    mx.clear_cache()
    return rec


def gate_report():
    real = R["res"].get("real")
    if not real or "v19d" not in real["vers"]:
        R["p0_band"] = None
        log("gate: no real/v19d band; using per-mix sigma")
        t3 = None
    else:
        fa = real["vers"]["v19d"]["full_prefill_ms_all"]
        mu, sd = float(np.mean(fa)), (float(np.std(fa, ddof=1)) if len(fa) > 1 else 0.0)
        R["p0_band"] = {"mix": "real", "vers": "v19d", "n": len(fa),
                        "mean_ms": mu, "sigma_ms": sd,
                        "sigma_pct": 100.0 * sd / mu, "three_sigma_ms": 3.0 * sd,
                        "three_sigma_pct": 100.0 * 3.0 * sd / mu}
        t3 = 3.0 * sd
        print("P0 noise band: v19d real sigma=%.3f ms (%.2f%%) 3sigma=%.3f ms (%.2f%%)"
              % (sd, 100.0 * sd / mu, t3, 100.0 * t3 / mu), flush=True)
    g = {}
    for name, rec in R["res"].items():
        if "v19d" not in rec["vers"]:
            continue
        ref = rec["vers"]["v19d"]["full_prefill_ms"]
        t3m = t3 if t3 is not None else 3.0 * rec["vers"]["v19d"]["full_prefill_ms_sigma"]
        g[name] = {}
        for v, vr in rec["vers"].items():
            if v == "v19d":
                continue
            got = vr["full_prefill_ms"]
            dms, dpct = ref - got, 100.0 * (ref - got) / ref
            clears = bool(dpct >= GATE_PCT and dms > t3m)
            sr = vr["isolated"]
            sd_ = rec["vers"]["v19d"]["isolated"]
            seg3_d = 100.0 * (sd_["seg3_ms"] - sr["seg3_ms"]) / sd_["seg3_ms"]
            g[name][v] = {"ref_v19d_full_ms": ref, "cand_full_ms": got,
                          "delta_ms": dms, "delta_pct": dpct,
                          "gt_3sigma": bool(dms > t3m), "ge_5pct": bool(dpct >= GATE_PCT),
                          "clears_both": clears,
                          "seg3_delta_pct": seg3_d,
                          "ref_seg3_ms": sd_["seg3_ms"], "cand_seg3_ms": sr["seg3_ms"]}
            print("gate [%s] %s vs v19d: full %+.2f%% (%+.2f ms) seg3 %+.2f%% (%s); "
                  "3sigma=%.3f ms -> clears: %s" % (
                      name, v, dpct, dms, seg3_d,
                      "decode-ALU ablation" if v == "v19dnd" else "pin-pong",
                      t3m, "YES" if clears else "NO"), flush=True)
    R["gate_tab"] = g
    return g


def main():
    log("start mlx=%s gemv_metal=%s (pin %s)" % (mx.__version__, G.__file__, PIN))
    flushjson()
    ck_present = os.path.isdir(CK)
    use_real = ck_present and REAL_CK != "0"
    if REAL_CK == "1" and not ck_present:
        log("REAL_CK=1 but checkpoint missing at %s -- synthetic fallback" % CK)
    R["real_ck"] = use_real
    R["checkpoint"] = CK
    log("weights=%s (ckpt %s present=%s) rounds=%d mixes=%s" % (
        "real" if use_real else "synthetic", CK, ck_present, ROUNDS, MIXES))

    ck_holder = {"sg_by_e": {}}
    mixes = build_mixes(use_real, ck_holder)

    # -- 1. bit-exact (raw E=8 smoke, then every mix: raw gu/dn + chained) ----
    for v in VERS:
        kern_for(v, int(G.CodebookMode(2)))
    _, _ = kern_for("v19d", 2)
    try:  # provenance for review: the exact v19p source the run used
        _, src_p = kern_for("v19p", int(G.CodebookMode(2)))
        R["v19p_source_sha256"] = hashlib.sha256(src_p.encode()).hexdigest()
        with open(os.path.join(os.path.dirname(OUT), "p20_moe_spike_p4_v19p.metal"),
                  "w") as fh:
            fh.write(src_p)
    except Exception as e:  # never let provenance break a bench run
        log("v19p source dump failed:", type(e).__name__, e)
    I8 = C.launch_inputs(mx, MOE, np.zeros(48, np.int32), 8, 5120)
    L8 = C.synth_layer(mx, 8, seed=7)
    ref = raw_run(kern_for("v19d", 2)[0], I8["x"], L8["gu"], I8, L8["H"], L8["gt"], 0)
    got = raw_run(kern_for("v19p", 2)[0], I8["x"], L8["gu"], I8, L8["H"], L8["gt"], 0)
    cmp_fp16("raw/synthetic-smokeE8 gu v19p==v19d", ref, got, E=8, N=I8["N"])
    del I8, L8, ref, got
    mx.clear_cache()
    flushjson()

    for name, ids_flat, packed, E, sg, L, cb_i, label in mixes:
        bitexact_mix(name, ids_flat, packed, E, sg, L, cb_i, label)
        flushjson()

    if R["bitexact"]["fail"]:
        R["verdict"] = ("P4: v19p -- vs v19d in-chain [MISMATCH]; gate#2 FAILS "
                        "(bit-exactness required before any pricing)")
        R["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        flushjson()
        print(R["verdict"], flush=True)
        return

    # -- 2./3. in-chain timing + NODECODE cross-check --------------------------
    for name, ids_flat, packed, E, sg, L, cb_i, label in mixes:
        for v in VERS:
            kern_for(v, cb_i)
        R["res"][name] = bench_mix(name, sg, packed, E, label, cb_i)
        flushjson()

    g = gate_report()
    verdict = "P4: v19p n/a vs v19d in-chain [bit-exact]"
    if g.get("real") and "v19p" in g["real"]:
        d = g["real"]["v19p"]
        verdict = ("P4: v19p %+.2f%% vs v19d in-chain [bit-exact]; gate#2 %s" % (
            d["delta_pct"], "CLEARS" if d["clears_both"] else "FAILS"))
    elif g and "v19p" in next(iter(g.values())):
        nm = next(iter(g))
        d = g[nm]["v19p"]
        verdict = ("P4: v19p %+.2f%% vs v19d in-chain (%s) [bit-exact]; gate#2 %s"
                    % (d["delta_pct"], nm, "CLEARS" if d["clears_both"] else "FAILS"))
    R["verdict"] = verdict
    R["bitexact"]["n_cases"] = len(R["bitexact"]["cases"])
    R["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    flushjson()
    print(verdict, flush=True)
    print("DONE -> %s" % OUT, flush=True)


if __name__ == "__main__":
    main()
