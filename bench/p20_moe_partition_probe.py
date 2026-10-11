#!/usr/bin/env python3
"""Exploratory probe (laptop or node): per-launch wall for the real/uniform/
skew mixes under each EXL3_MM_SEG version, synthetic trellis, gu shape.
Optional NODECODE ablation (EXL3_MM_NODECODE) to separate decode cost.
Not the authoritative measurement (that is p20_moe_partition_bench.py)."""
import os, sys, time, json
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

assert G.__file__.startswith(P20_PKG or os.path.join(ROOT, "mlx-lm")), G.__file__
VERS = os.environ.get("VERS", "v19d").split(",")
REPS = int(os.environ.get("MB_REPS", "9"))
SHAPES = os.environ.get("SHAPES", "gu,dn").split(",")


def tmed(fn, reps=REPS, warm=3):
    for _ in range(warm):
        mx.eval(fn())
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); mx.eval(fn()); ts.append(time.perf_counter() - t0)
    ts.sort(); return ts[len(ts) // 2]


def main():
    L = C.synth_layer(mx, 384)
    mixes = {"real": C.real_chunk_ids(C.env_trace_path(ROOT))}
    mixes["uniform192"] = C.uniform_ids(192, 12288)
    mixes["skew192"] = C.skew_ids(192, 12288)
    out = {}
    for name, ids in mixes.items():
        E = 384 if name == "real" else 192
        for shp in SHAPES:
            if shp == "gu":
                K, Nout, tr, tpe = L["D"], L["H"], L["gu"], L["gt"]
            else:
                K, Nout, tr, tpe = L["H"], L["D"], L["dn"], L["dt"]
            I = C.launch_inputs(mx, MOE, ids, E, K)
            for v in VERS:
                G._MM_SEG_VERSION = v
                f = lambda: G.inner_mm_seg_mlx(I["x"], tr, 3, int(os.environ.get("CB", "2")), I["tab"], I["nbr"], n_rows=I["N"],
                                               tn_base=0, tiles_per_e=tpe, out_e=Nout)
                t = tmed(f)
                tf = 2.0 * I["N"] * K * Nout / t / 1e12
                out["%s/%s/%s" % (name, shp, v)] = {"ms": t * 1e3, "TF": tf}
                print("%-12s %-3s %-6s %8.3f ms %6.2f TF" % (name, shp, v, t * 1e3, tf), flush=True)
    if os.environ.get("PROBE_JSON"):
        json.dump(out, open(os.environ["PROBE_JSON"], "w"), indent=1)


if __name__ == "__main__":
    main()
