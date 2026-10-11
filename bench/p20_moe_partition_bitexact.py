#!/usr/bin/env python3
"""PATCHED (EXL3_MM_SEG=v20) vs UNPATCHED (v19d production default) bit-exact
harness for inner_mm_seg_mlx. Exact equality on every output element
(fp16 bit patterns compared as uint16 -> catches -0/+0 and NaN payloads too).

Shapes: (a) real routing mix (trace layer 0, 2048 tok x top-6 = 12288
pairs, E=384) on BOTH gu (K=5120,N=1152,tpe=72, gate+up tn_base) and dn
(K=1152,N=5120,tpe=320); (b) autopsy uniform/skew E=192 N=12288 controls;
(c) adversarial ragged mixes (p20_moe_partition_common.adversarial_mixes),
E=384, gu+dn. Also v20 vs v19c (stock) and a full EXL3SwitchGLU._prefill
end-to-end equality when REAL_CK is set (studio2: real checkpoint weights).

Weights: synthetic random trellis by default (numerics are a pure function
of trellis bits, x and blk_tab); REAL_CK=1 additionally loads the real
dsv41 layer-0 experts (studio2 only).
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

assert G.__file__.startswith(P20_PKG or os.path.join(ROOT, "mlx-lm")), G.__file__
OUT = os.environ.get("BX_JSON", os.path.join(HERE, "p20_moe_partition_bitexact.json"))
REAL_CK = os.environ.get("REAL_CK", "0") == "1"
PATCHED = os.environ.get("BX_VERS", "v20").split(",")
CK = os.path.expanduser("~/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw")
R = {"host": socket.gethostname(), "mlx": mx.__version__, "gemv_metal": G.__file__,
     "started": time.strftime("%Y-%m-%d %H:%M:%S"), "cases": [], "fail": 0}


def run(ver, x, tr, tab, nbr, N, tn_base, tpe, out_e, k=3, cb=0):
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
    finite = bool(np.isfinite(fa).all())
    rec = {"case": name, "n_elem": int(a.size), "n_bitdiff": nd, "max_abs_diff": mad,
           "ref_finite": finite, "ref_absmax": float(np.abs(fa).max()) if fa.size else 0.0, **extra}
    R["cases"].append(rec)
    if nd:
        R["fail"] += 1
    print("%-46s %s  bitdiff=%d maxabs=%g elems=%d" % (name, "OK " if nd == 0 else "FAIL", nd, mad, a.size), flush=True)
    return nd


def check_mix(tag, ids, E, L, layer_tag="synth", cb=None):
    cb = L.get("cb", 2) if cb is None else cb
    for shp in ("gu", "dn"):
        if shp == "gu":
            K, Nout, tr, tpe = L["D"], L["H"], L["gu"], L["gt"]
            bases = (("gate", 0), ("up", E * tpe))
        else:
            K, Nout, tr, tpe = L["H"], L["D"], L["dn"], L["dt"]
            bases = (("down", 0),)
        I = C.launch_inputs(mx, MOE, ids, E, K, seed=hash(tag) % 1000)
        cnt = np.bincount(ids, minlength=E)
        nb = int(((cnt + 63) // 64).sum())
        for bname, tb in bases:
            ref = run("v19d", I["x"], tr, I["tab"], I["nbr"], I["N"], tb, tpe, Nout, cb=cb)
            s, nts = G._seg_pair_sched(I["tab"], I["nbr"])
            extra = {"cb": int(cb), "E": E, "N": I["N"], "shape": shp, "launch": bname, "weights": layer_tag,
                     "blocks_v19d": nb, "tgs_v20": int(np.asarray(nts)[0]),
                     "experts_hit": int((cnt > 0).sum())}
            stock = None
            if bname in ("gate", "down"):
                stock = run("v19c", I["x"], tr, I["tab"], I["nbr"], I["N"], tb, tpe, Nout, cb=cb)
            for pv in PATCHED:
                got = run(pv, I["x"], tr, I["tab"], I["nbr"], I["N"], tb, tpe, Nout, cb=cb)
                cmp("%s/cb%d/%s/%s %s==v19d" % (tag, cb, shp, bname, pv), ref, got, dict(extra, patched=pv))
                if stock is not None:
                    cmp("%s/cb%d/%s/%s %s==v19c" % (tag, cb, shp, bname, pv), stock, got, dict(extra, patched=pv))
                del got
        del I
        mx.clear_cache()


def check_sched(ids, E):
    """Independent host-side check of the v20 schedule: every live row of
    every v19d block is covered exactly once, by the right expert, at the
    v19d 32-row-window alignment, never mixing experts within a half."""
    N = ids.size
    I = C.launch_inputs(mx, MOE, ids, E, 64)
    tab = np.asarray(I["tab"]).astype(np.int64); nb = int(np.asarray(I["nbr"])[0])
    sidx = np.asarray(I["sidx"]).astype(np.int64)
    starts = set(int(r) for r in tab[1, :nb])
    out = {}
    for ver, fn, hmax in (("v20", G._seg_pair_sched, 32),):
        s, nts = fn(I["tab"], I["nbr"]); s = np.asarray(s).astype(np.int64)
        nts = int(np.asarray(nts)[0])
        cover = np.zeros(N, np.int64)
        for t in range(nts):
            for h in range(2):
                e, r0, ln = s[3 * h, t], s[3 * h + 1, t], s[3 * h + 2, t]
                assert ln <= hmax, (ver, t, h, ln)
                if ln:
                    assert (sidx[r0:r0 + ln] == e).all(), (ver, t, h)
                    # windows start on a v19d block start (+32 for v20 half 1)
                    assert r0 in starts or (r0 - 32) in starts, (ver, t, h, r0)
                    cover[r0:r0 + ln] += 1
        assert (cover == 1).all(), (ver, "coverage")
        assert (s[:, nts:] == 0).all(), (ver, "dead columns must be zero")
        out[ver] = nts
    nl = int((tab[2, :nb] > 32).sum()); ns = nb - nl
    assert out["v20"] == nl + (ns + 1) // 2
    return out


def main():
    tr_path = C.env_trace_path(ROOT)
    mixes = [("real_trace_L0_E384", C.real_chunk_ids(tr_path), 384),
             ("uniform_E192_N12288", C.uniform_ids(192, 12288), 192),
             ("skew_E192_N12288", C.skew_ids(192, 12288), 192)]
    for name, cnts in C.adversarial_mixes(384).items():
        mixes.append(("adv_%s" % name, C.counts_ids(cnts, seed=len(name)), 384))
    for name, ids, E in mixes:
        n = check_sched(ids, E)
        print("sched ok %-30s tgs=%s" % (name, n), flush=True)
    R["sched_checks"] = len(mixes)
    L = C.synth_layer(mx, 384)
    # production codebook is MUL1 (cb=2) -> every mix; DEFAULT/MCG on the real mix
    for name, ids, E in mixes:
        check_mix(name, ids, E, L, cb=2)
        json.dump(R, open(OUT, "w"), indent=1)
    for cbx in (0, 1):
        check_mix(mixes[0][0], mixes[0][1], mixes[0][2], L, cb=cbx)
        json.dump(R, open(OUT, "w"), indent=1)
    del L; mx.clear_cache()
    if REAL_CK:
        sys.path.insert(0, os.path.expanduser("~/repos/exo"))
        from mlx_lm.models.exl3.loader import Exl3Checkpoint, load_experts
        ck = Exl3Checkpoint(CK)
        for Esub, sub in ((None, mixes[:1] + mixes[3:]), (192, mixes[1:3])):
            sg = load_experts(ck, 0, n_experts=Esub, rank=0, world=2, activation="silu_clamp")
            E = sg.num_experts
            Lr = {"gu": sg._gu_trellis, "dn": sg._dn_trellis, "gt": sg._gu_tiles,
                  "dt": sg._dn_tiles, "D": sg.input_dims, "H": sg.hidden_dims, "E": E, "k": sg._k}
            assert sg._k == 3, sg._k
            Lr["cb"] = int(sg._cb)
            R.setdefault("real_ck_codebook", int(sg._cb))
            for name, ids, Em in sub:
                assert Em == E
                check_mix("REAL_W/" + name, ids, E, Lr, layer_tag="real_ck_L0")
                # full _prefill end to end (gate+up+act+down+finish), v20 vs v19d
                idx = mx.array(C.pack(ids), dtype=mx.uint32)
                x = mx.random.normal((1, idx.shape[1], sg.input_dims), key=mx.random.key(3)).astype(mx.float16)
                G._MM_SEG_VERSION = "v19d"; y0 = sg._prefill(x, idx); mx.eval(y0)
                for pv in PATCHED:
                    G._MM_SEG_VERSION = pv; y1 = sg._prefill(x, idx); mx.eval(y1)
                    cmp("REAL_W/%s/_prefill e2e %s==v19d" % (name, pv), y0, y1,
                        {"E": E, "N": int(ids.size), "weights": "real_ck_L0", "shape": "prefill", "patched": pv})
                json.dump(R, open(OUT, "w"), indent=1)
            del sg, Lr; mx.clear_cache()
    R["n_cases"] = len(R["cases"])
    R["verdict"] = "0-diff" if R["fail"] == 0 else "FAIL"
    R["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(R, open(OUT, "w"), indent=1)
    print("VERDICT %s  cases=%d fail=%d -> %s" % (R["verdict"], R["n_cases"], R["fail"], OUT), flush=True)
    sys.exit(0 if R["fail"] == 0 else 1)


if __name__ == "__main__":
    main()
