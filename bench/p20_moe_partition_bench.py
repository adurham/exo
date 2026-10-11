#!/usr/bin/env python3
"""Phase-20 MoE partition (v20) measurement -- OFFLINE, single node (studio2).

Reuses the autopsy harness (bench/p20_moe_autopsy.py: ceilings, real-chunk /
uniform / skew builders, seg_parts / per-launch timing, metric conventions)
and runs v19d (production default) vs v20 INTERLEAVED in one process on the
SAME shapes and real checkpoint weights:

  real    : layer-0 experts E=384, real routing trace chunk 2048 x top-6
  uniform : E=192, N=12288 (autopsy control)
  skew    : E=192, N=12288, Zipf-0.9 hot-35% (autopsy control)

Metric (identical to the autopsy): TF = 6*D*H*pairs / full EXL3SwitchGLU
._prefill wall; per-launch (gate/up/down) wall reported too. Every timed
region ends in a drained mx.eval; warmups drained; medians over REPS; rounds
alternate the version order to cancel drift. Wall clock only.
"""
import os, sys, json, time, socket
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("STAGE", "partition")
import p20_moe_autopsy as A  # sets sys.path (P20_PKG first, ~/repos/exo)
import numpy as np
import mlx.core as mx
from mlx_lm.models.exl3 import gemv_metal as G
from mlx_lm.models.exl3.loader import Exl3Checkpoint, load_experts

P20_PKG = os.environ.get("P20_PKG", "")
assert P20_PKG and G.__file__.startswith(P20_PKG), G.__file__
VERS = os.environ.get("VERS", "v19d,v20").split(",")
ROUNDS = int(os.environ.get("MB_ROUNDS", "2"))
OUT = os.environ.get("MB_JSON", "/tmp/p20moe_partition/p20_moe_partition_bench.json")
R = A.R
R.update({"stage": "partition", "vers": VERS, "rounds": ROUNDS, "gemv_metal": G.__file__,
          "mlx": mx.__version__, "res": {}})
A.OUT = OUT
BASE = {"real": (6.63, 14.96), "uniform": (8.85, 15.15), "skew": (7.55, 15.10)}


def log(*a):
    print("[partition]", *a, flush=True)


def bench_mix(name, sg, idx):
    E, D, H = sg.num_experts, sg.input_dims, sg.hidden_dims
    F = 6.0 * D * H
    st = A.idx_stats(idx, E)
    x = mx.random.normal((1, idx.shape[1], D), key=mx.random.key(11)).astype(mx.float16)
    m = mx.array(idx, dtype=mx.uint32); mx.eval(x, m)
    acc = {v: {"full": [], "gu_gate": [], "gu_up": [], "dn": []} for v in VERS}
    for rnd in range(ROUNDS):
        order = VERS if rnd % 2 == 0 else VERS[::-1]
        for v in order:
            G._MM_SEG_VERSION = v
            tl, _ = A.time_launches(x, m, sg)
            acc[v]["full"].append(tl["_full_prefill_ms"])
            for k in ("gu_gate", "gu_up", "dn"):
                acc[v][k].append(tl[k]["ms"])
            log("  %s r%d %-5s full %.2f ms  gate %.2f up %.2f dn %.2f" % (
                name, rnd, v, tl["_full_prefill_ms"], tl["gu_gate"]["ms"], tl["gu_up"]["ms"], tl["dn"]["ms"]))
    rec = {"idx_stats": st, "E": E, "D": D, "H": H}
    mix_ceil = None
    for v in VERS:
        full = float(np.median(acc[v]["full"]))
        tf = F * st["pairs"] / (full / 1e3) / 1e12
        seg = sum(float(np.median(acc[v][k])) for k in ("gu_gate", "gu_up", "dn"))
        rec[v] = {"full_prefill_ms": full, "full_all": acc[v]["full"], "TFLOPS": tf,
                  "seg3_ms": seg, "seg3_TFLOPS": F * st["pairs"] / (seg / 1e3) / 1e12,
                  **{k + "_ms": float(np.median(acc[v][k])) for k in ("gu_gate", "gu_up", "dn")}}
    return rec


def main():
    log("start", G.__file__, "vers", VERS)
    A.ceilings()
    dense = R["ceil"]["dense_TFLOPS"]
    ck = Exl3Checkpoint(A.CK)
    # real chunk, E=384
    sg = load_experts(ck, 0, n_experts=None, rank=0, world=2, activation="silu_clamp")
    rng = np.random.RandomState(20261010)
    idx, _ = A.real_chunk_idx(A.TRACE, 0, 2048, rng)
    R["res"]["real"] = bench_mix("real", sg, idx); A.flushjson()
    del sg; mx.clear_cache()
    sg = load_experts(ck, 0, n_experts=A.E_BLOCK, rank=0, world=2, activation="silu_clamp")
    for name, b in (("uniform", A.build_idx_uniform), ("skew", A.build_idx_skew)):
        idx = b(sg.num_experts, A.NPROD, np.random.RandomState(2026))
        R["res"][name] = bench_mix(name, sg, idx); A.flushjson()
    del sg; mx.clear_cache()
    # summary: %-of-mix-roofline uses the autopsy mix ceilings scaled by the
    # in-session dense ceiling (mix/dense ratio is a pure function of the
    # M-histogram, unchanged by v20 -- same rows, same weights)
    summ = {}
    for name, (base_tf, base_mix) in BASE.items():
        r = R["res"][name]
        mix = base_mix / 15.146 * dense
        s = {"autopsy_baseline_TF": base_tf, "autopsy_mix_roofline_TF": base_mix,
             "in_session_mix_roofline_TF": mix}
        for v in VERS:
            s[v + "_TF"] = r[v]["TFLOPS"]; s[v + "_pct_mix"] = r[v]["TFLOPS"] / mix * 100
        if "v19d" in VERS and "v20" in VERS:
            s["v20_vs_v19d_pct"] = (r["v20"]["TFLOPS"] / r["v19d"]["TFLOPS"] - 1) * 100
            s["v20_vs_v19d_full_ms_saved"] = r["v19d"]["full_prefill_ms"] - r["v20"]["full_prefill_ms"]
        summ[name] = s
        log("[%s] " % name + " ".join("%s=%.3f" % kv for kv in s.items() if isinstance(kv[1], float)))
    R["summary"] = summ
    R["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    A.flushjson(); log("DONE ->", OUT)


if __name__ == "__main__":
    main()
