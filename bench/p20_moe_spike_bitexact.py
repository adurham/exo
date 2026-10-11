#!/usr/bin/env python3
"""Phase-20 MoE spike bit-exactness: v19d vs v19e (and v20 where available).

fp16-as-uint16 0-diff convention: every output element's fp16 bit pattern must
match exactly (-0/+0 and NaN payloads included). Weights are synthetic random
trellis by default; REAL_CK=1 additionally loads the real dsv41 layer-0 experts
(node only -- the checkpoint is not on the laptop).

Cache note: the segmented kernel caches compiled kernels by
``(k, int(cb), _MM_SEG_VERSION)`` in ``G._mm_seg_kernels``, so comparing two
versions in one process only needs ``G._MM_SEG_VERSION`` set before each call --
distinct keys, nothing cleared -- exactly the pattern
bench/p20_moe_partition_bitexact.py uses.

That harness already selects its compare-versions from the ``BX_VERS`` env
list, so the equivalent node command (v19d reference vs v19e and v20, real
checkpoint, full _prefill end-to-end) is:

    BX_VERS=v19e,v20 REAL_CK=1 P20_PKG=<scratch pkg> \\
        ~/repos/exo/.venv/bin/python bench/p20_moe_partition_bitexact.py

This standalone script is the focused v19e comparator and runs on the laptop on
synthetic trellis (v20 auto-skipped when the node pkg is absent).

Env:
  SPIKE_BX_VERS=v19e,v20            compare-versions (reference is always v19d)
  SPIKE_BX_MIXES=real,uniform192,skew192,adv
  SPIKE_BX_SHAPES=gu,dn
  REAL_CK=1                        also load the real checkpoint (node)
  P20_PKG / BX_JSON
"""
import os, sys, json, time, socket
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

OUT = os.environ.get("BX_JSON", "/tmp/p20moe_spike/p20_moe_spike_bitexact.json")
REAL_CK = os.environ.get("REAL_CK", "0") == "1"
WANT = [v for v in os.environ.get("SPIKE_BX_VERS", "v19e,v20").split(",") if v]
MIXES = [m for m in os.environ.get("SPIKE_BX_MIXES",
                                   "real,uniform192,skew192").split(",") if m]
SHAPES = [s for s in os.environ.get("SPIKE_BX_SHAPES", "gu,dn").split(",") if s]
CK = os.path.expanduser("~/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw")
# v19c/v19d/v19e are always built; v20 exists only in the node pkg.
AVAIL = [v for v in WANT if v in ("v19c", "v19d", "v19e")
         or (v == "v20" and hasattr(G, "_mm_seg_source_v20"))]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
R = {"host": socket.gethostname(), "mlx": mx.__version__, "gemv_metal": G.__file__,
     "started": time.strftime("%Y-%m-%d %H:%M:%S"), "want": WANT, "compared": AVAIL,
     "real_ck": REAL_CK, "cases": [], "fail": 0}


def run(ver, x, tr, tab, nbr, N, tn_base, tpe, out_e, k=3, cb=2):
    # cache key (k, int(cb), _MM_SEG_VERSION): setting the version selects a
    # distinct compiled kernel; nothing is cleared between versions.
    G._MM_SEG_VERSION = ver
    y = G.inner_mm_seg_mlx(x, tr, k, cb, tab, nbr, n_rows=N, tn_base=tn_base,
                           tiles_per_e=tpe, out_e=out_e)
    mx.eval(y)
    return y


def cmp(name, ref, got, extra):
    a = np.asarray(ref.view(mx.uint16)); b = np.asarray(got.view(mx.uint16))
    nd = int((a != b).sum())
    fa = np.asarray(ref.astype(mx.float32)); fb = np.asarray(got.astype(mx.float32))
    mad = float(np.nanmax(np.abs(fa - fb))) if fa.size else 0.0
    R["cases"].append({"case": name, "n_elem": int(a.size), "n_bitdiff": nd,
                       "max_abs_diff": mad, **extra})
    if nd:
        R["fail"] += 1
    print("%-52s %s  bitdiff=%d maxabs=%g elems=%d" % (
        name, "OK " if nd == 0 else "FAIL", nd, mad, a.size), flush=True)


def check(ids, E, L, tag, cb=2):
    for shp in SHAPES:
        if shp == "gu":
            K, Nout, tr, tpe = L["D"], L["H"], L["gu"], L["gt"]
            bases = (("gate", 0), ("up", E * tpe))
        else:
            K, Nout, tr, tpe = L["H"], L["D"], L["dn"], L["dt"]
            bases = (("down", 0),)
        I = C.launch_inputs(mx, MOE, ids, E, K)
        for bname, tb in bases:
            ref = run("v19d", I["x"], tr, I["tab"], I["nbr"], I["N"], tb, tpe, Nout, cb=cb)
            extra = {"cb": int(cb), "E": E, "N": I["N"], "shape": shp, "launch": bname,
                     "tag": tag, "ref": "v19d"}
            for pv in AVAIL:
                got = run(pv, I["x"], tr, I["tab"], I["nbr"], I["N"], tb, tpe, Nout, cb=cb)
                cmp("%s/cb%d/%s/%s %s==v19d" % (tag, cb, shp, bname, pv), ref, got,
                    dict(extra, patched=pv))
                del got
        del I
        mx.clear_cache()


def main():
    print("bitexact start gemv_metal=%s compare=%s real_ck=%s" % (
        G.__file__, AVAIL, REAL_CK), flush=True)
    L = C.synth_layer(mx, 384)
    mixes = []
    if "real" in MIXES:
        tp = C.env_trace_path(ROOT)
        if os.path.exists(tp):
            mixes.append(("real_E384", C.real_chunk_ids(tp), 384))
        else:
            print("skip real: trace missing %s" % tp, flush=True)
    if "uniform192" in MIXES:
        mixes.append(("uniform_E192", C.uniform_ids(192, 12288), 192))
    if "skew192" in MIXES:
        mixes.append(("skew_E192", C.skew_ids(192, 12288), 192))
    if "adv" in MIXES:
        for nm, cnts in list(C.adversarial_mixes(384).items())[:3]:
            mixes.append(("adv_%s" % nm, C.counts_ids(cnts, seed=len(nm)), 384))
    for name, ids, E in mixes:
        check(ids, E, L, name, cb=2)
        json.dump(R, open(OUT, "w"), indent=1, default=float)
    del L
    mx.clear_cache()
    if REAL_CK:
        from mlx_lm.models.exl3.loader import Exl3Checkpoint, load_experts
        ck = Exl3Checkpoint(CK)
        sg = load_experts(ck, 0, n_experts=None, rank=0, world=2, activation="silu_clamp")
        E = sg.num_experts
        Lr = {"gu": sg._gu_trellis, "dn": sg._dn_trellis, "gt": sg._gu_tiles,
              "dt": sg._dn_tiles, "D": sg.input_dims, "H": sg.hidden_dims, "k": sg._k}
        assert sg._k == 3, sg._k
        R["real_ck_codebook"] = int(sg._cb)
        for name, ids, Em in mixes:
            check(ids, E, Lr, "REAL_W/" + name, cb=int(sg._cb))
            json.dump(R, open(OUT, "w"), indent=1, default=float)
        del sg, Lr
        mx.clear_cache()
    R["n_cases"] = len(R["cases"])
    R["verdict"] = "0-diff" if R["fail"] == 0 else "FAIL"
    R["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(R, open(OUT, "w"), indent=1, default=float)
    print("VERDICT %s cases=%d fail=%d -> %s" % (
        R["verdict"], R["n_cases"], R["fail"], OUT), flush=True)
    sys.exit(0 if R["fail"] == 0 else 1)


if __name__ == "__main__":
    main()
