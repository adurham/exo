#!/usr/bin/env python3
"""Phase-20 MoE spike ablation sweep -- OFFLINE (laptop or node).

Template: bench/p20_moe_partition_ablate.py. Diagnostic, NOT numerics-preserving
(timing only). Rewrites the segmented-GEMM kernel SOURCE per variant to bound
where the per-threadgroup time goes for the spike's v19d / v19e / v20 kernels.
Each version's base source is built the SAME way the kernel does
(gemv_metal.inner_mm_seg_mlx):
  v19d -> G._mm_seg_guarded(G._mm_seg_source(k, cb))
  v19e -> G._mm_seg_hoisted(G._mm_seg_source(k, cb))
  v20  -> G._mm_seg_source_v20(k, cb)          (node pkg only; hasattr-guarded)

Variants (each applied independently to the version's base source):
  full     -- base source unchanged
  no_mma   -- every simdgroup_multiply_accumulate line dropped
  no_tload -- the SWAR trellis word pair replaced by an id-derived constant
  no_aload -- EVERY A-fragment load ``simdgroup_load(<tgt>, xrow + ...)``
              replaced by a fill ``make_filled_simdgroup_matrix<half,8,8>``
              (half(0.5f));``. Matches all three A-load shapes: the guarded
              per-row ``a``, the hoisted ``af[k2]`` and the full-block ``a``.
              Purpose: bound the A-load share so we can decide whether hoisting
              it (v19e) can help at all. Diagnostic -- not numerics-preserving.

Shapes (synthetic random trellis): gu (K=5120, N=1152) and dn (K=1152, N=5120).

Env:
  SPIKE_REPS=9                        median-of-REPS wall, 3 warmups
  SPIKE_ABL_VERS=v19d,v19e,v20
  SPIKE_ABL_MIXES=real,uniform192     (real needs MB_TRACE; skipped if missing)
  SPIKE_ABL_SHAPES=gu,dn
  P20_PKG=<scratch mlx_lm pkg>        (node)
  ABL_JSON=/tmp/p20moe_spike/p20_moe_spike_ablate.json
"""
import os, sys, time, json, re
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
P20_PKG = os.environ.get("P20_PKG", "")
sys.path.insert(0, P20_PKG or os.path.join(ROOT, "mlx-lm"))
sys.path.insert(0, HERE)
import numpy as np
import mlx.core as mx
from mlx_lm.models.exl3 import gemv_metal as G
from mlx_lm.models.exl3 import exl3_moe as MOE
import p20_moe_partition_common as C

REPS = int(os.environ.get("SPIKE_REPS", "9"))
HDR = "#include <metal_simdgroup_matrix>\n#include <metal_stdlib>\nusing namespace metal;\n"
VERS = [v for v in os.environ.get("SPIKE_ABL_VERS", "v19d,v19e,v20").split(",") if v]
MIXES = [m for m in os.environ.get("SPIKE_ABL_MIXES", "real,uniform192").split(",") if m]
SHAPES = [s for s in os.environ.get("SPIKE_ABL_SHAPES", "gu,dn").split(",") if s]
OUT = os.environ.get("ABL_JSON", "/tmp/p20moe_spike/p20_moe_spike_ablate.json")
A_FILL = "make_filled_simdgroup_matrix<half,8,8>(half(0.5f))"

# A-fragment loads only: the second argument starts with ``xrow +``. B loads
# read ``wrow`` and are left untouched. Covers guarded ``a``, hoisted ``af[k2]``
# and full-block ``a`` (address spans lines; re.S).
A_LOAD_RE = re.compile(
    r"\bsimdgroup_load\s*\(\s*([A-Za-z_]\w*(?:\s*\[[^\]]*\])?)\s*,\s*(xrow\s*\+.*?)\)\s*;",
    re.S)


def tmed(fn, reps=REPS, warm=3):
    for _ in range(warm):
        mx.eval(fn())
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); mx.eval(fn()); ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


def variants(base):
    out = {"full": base}
    # no mma: drop every simdgroup_multiply_accumulate
    out["no_mma"] = "\n".join(l for l in base.splitlines()
                              if "simdgroup_multiply_accumulate" not in l)
    # no trellis load: cw from thread/stage ids
    out["no_tload"] = base.replace("((ulong)words[i0_[j]] << 32) | (ulong)words[i1_[j]]",
                                   "((ulong)(tk * 977u + lt) << 32) | (ulong)(tn0 + j)")
    # no A-load: every xrow + ... fragment load becomes a fill (diagnostic)
    out["no_aload"] = A_LOAD_RE.sub(r"\1 = " + A_FILL + ";", base)
    return out


def base_for(ver, k=3, cb=0):
    """Version base source, built exactly as inner_mm_seg_mlx builds it."""
    src = G._mm_seg_source(k, G.CodebookMode(cb))
    if ver == "v19d":
        return G._mm_seg_guarded(src)
    if ver == "v19e":
        return G._mm_seg_hoisted(src)
    if ver == "v20":
        if not hasattr(G, "_mm_seg_source_v20"):
            return None
        return G._mm_seg_source_v20(k, G.CodebookMode(cb))
    raise ValueError(ver)


def build_mixes():
    mixes = {}
    if "real" in MIXES:
        tp = C.env_trace_path(ROOT)
        if os.path.exists(tp):
            mixes["real"] = (C.real_chunk_ids(tp), 384)
        else:
            print("skip mix real: trace missing %s" % tp, flush=True)
    if "uniform192" in MIXES:
        mixes["uniform192"] = (C.uniform_ids(192, 12288), 192)
    if "skew192" in MIXES:
        mixes["skew192"] = (C.skew_ids(192, 12288), 192)
    return mixes


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    L = C.synth_layer(mx, 384)
    mixes = build_mixes()
    print("ablate start gemv_metal=%s reps=%d vers=%s mixes=%s shapes=%s" % (
        G.__file__, REPS, VERS, list(mixes), SHAPES), flush=True)
    res = {"gemv_metal": G.__file__, "mlx": mx.__version__, "reps": REPS,
           "vers": VERS, "mixes": list(mixes), "shapes": SHAPES, "ms": {}}
    for ver in VERS:
        base = base_for(ver)
        if base is None:
            print("skip %s: G._mm_seg_source_v20 unavailable (node pkg only)" % ver,
                  flush=True)
            res["ver_available_%s" % ver] = False
            continue
        res["ver_available_%s" % ver] = True
        for vname, vsrc in variants(base).items():
            kern = mx.fast.metal_kernel(
                name="spike_abl_%s_%s" % (ver, vname),
                input_names=["xh", "trellis", "inv_perm", "blk_tab", "nbr", "dims"],
                output_names=["out"], source=vsrc, header=HDR)
            for mname, (ids, E) in mixes.items():
                for shp in SHAPES:
                    if shp == "gu":
                        K, Nout, tr, tpe = L["D"], L["H"], L["gu"], L["gt"]
                    else:
                        K, Nout, tr, tpe = L["H"], L["D"], L["dn"], L["dt"]
                    I = C.launch_inputs(mx, MOE, ids, E, K)
                    tab, nbr = I["tab"], I["nbr"]
                    if ver == "v20" and hasattr(G, "_seg_pair_sched"):
                        tab, nbr = G._seg_pair_sched(tab, nbr)
                    in_tiles, src_tiles, _ = tr.shape
                    dims = mx.array([in_tiles, tpe, src_tiles, 0, Nout, I["nb_max"]],
                                    dtype=mx.uint32)
                    f = lambda: kern(
                        inputs=[I["x"].reshape(-1), tr.reshape(-1).view(mx.uint32),
                                G._inv_perm_u32(), tab.reshape(-1), nbr, dims],
                        template=[("T", mx.float16)],
                        grid=((Nout // 64) * 128, I["nb_max"], 1),
                        threadgroup=(128, 1, 1),
                        output_shapes=[(I["N"], Nout)], output_dtypes=[mx.float16])[0]
                    t = tmed(f)
                    key = "%s/%s/%s/%s" % (ver, vname, mname, shp)
                    res["ms"][key] = t * 1e3
                    print("%-5s %-9s %-12s %-3s %8.3f ms" % (
                        ver, vname, mname, shp, t * 1e3), flush=True)
    json.dump(res, open(OUT, "w"), indent=1, default=float)
    print("DONE -> %s" % OUT, flush=True)


if __name__ == "__main__":
    main()
