#!/usr/bin/env python3
"""Diagnostic ablation (NOT numerics-preserving; timing only): where does the
real-mix per-threadgroup time go in v19d / v20? Variants rewrite the kernel
source: no-mma, no-trellis-load, no-store. Synthetic trellis, gu shape."""
import os, sys, time, json
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
P20_PKG = os.environ.get("P20_PKG", "")
sys.path.insert(0, P20_PKG or os.path.join(ROOT, "mlx-lm")); sys.path.insert(0, HERE)
import numpy as np
import mlx.core as mx
from mlx_lm.models.exl3 import gemv_metal as G
from mlx_lm.models.exl3 import exl3_moe as MOE
import p20_moe_partition_common as C

REPS = int(os.environ.get("MB_REPS", "9"))
HDR = "#include <metal_simdgroup_matrix>\n#include <metal_stdlib>\nusing namespace metal;\n"


def tmed(fn, reps=REPS, warm=3):
    for _ in range(warm):
        mx.eval(fn())
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); mx.eval(fn()); ts.append(time.perf_counter() - t0)
    ts.sort(); return ts[len(ts) // 2]


def variants(base):
    out = {"full": base}
    s = base
    # no mma: drop every simdgroup_multiply_accumulate
    out["no_mma"] = "\n".join(l for l in s.splitlines() if "simdgroup_multiply_accumulate" not in l)
    # no trellis load: cw from thread/stage ids
    out["no_tload"] = s.replace("((ulong)words[i0_[j]] << 32) | (ulong)words[i1_[j]]",
                                "((ulong)(tk * 977u + lt) << 32) | (ulong)(tn0 + j)")
    return out


def main():
    L = C.synth_layer(mx, 384)
    mixes = {"real": (C.real_chunk_ids(C.env_trace_path(ROOT)), 384),
             "uniform192": (C.uniform_ids(192, 12288), 192)}
    res = {}
    for ver in ("v19d", "v20"):
        base = G._mm_seg_source(3, G.CodebookMode(0))
        base = G._mm_seg_guarded(base) if ver == "v19d" else G._mm_seg_source_v20(3, G.CodebookMode(0))
        for vname, src in variants(base).items():
            kern = mx.fast.metal_kernel(name="abl_%s_%s" % (ver, vname),
                                        input_names=["xh", "trellis", "inv_perm", "blk_tab", "nbr", "dims"],
                                        output_names=["out"], source=src, header=HDR)
            for mname, (ids, E) in mixes.items():
                K, Nout, tr, tpe = L["D"], L["H"], L["gu"], L["gt"]
                I = C.launch_inputs(mx, MOE, ids, E, K)
                tab, nbr = I["tab"], I["nbr"]
                if ver == "v20":
                    tab, nbr = G._seg_pair_sched(tab, nbr)
                in_tiles, src_tiles, _ = tr.shape
                dims = mx.array([in_tiles, tpe, src_tiles, 0, Nout, I["nb_max"]], dtype=mx.uint32)
                f = lambda: kern(inputs=[I["x"].reshape(-1), tr.reshape(-1).view(mx.uint32), G._inv_perm_u32(),
                                         tab.reshape(-1), nbr, dims], template=[("T", mx.float16)],
                                 grid=((Nout // 64) * 128, I["nb_max"], 1), threadgroup=(128, 1, 1),
                                 output_shapes=[(I["N"], Nout)], output_dtypes=[mx.float16])[0]
                t = tmed(f)
                res["%s/%s/%s" % (ver, vname, mname)] = t * 1e3
                print("%-5s %-9s %-10s %8.3f ms" % (ver, vname, mname, t * 1e3), flush=True)
    if os.environ.get("ABL_JSON"):
        json.dump(res, open(os.environ["ABL_JSON"], "w"), indent=1)


if __name__ == "__main__":
    main()
