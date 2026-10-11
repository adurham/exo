#!/usr/bin/env python3
"""Phase-20 MoE spike: in-chain P0+P2 runner -- OFFLINE, single node (studio2).

Gate #2 of the pre-registered spike kill gate: does ANY variant beat v19d by
>=5% MoE throughput IN THE PIPELINED CHAIN -- i.e. on ``EXL3SwitchGLU._prefill``
(the chained gate+up+dn+glue wall that the autopsy exposes as
``_full_prefill_ms``), NOT the isolated per-launch number -- with bit-exact
live rows?

Template: bench/p20_moe_partition_bench.py. Same reuse of
bench/p20_moe_autopsy.py (A.ceilings / real_chunk_idx / build_idx_uniform /
build_idx_skew / idx_stats / time_launches), same real/uniform/skew mixes, same
TF = 6*D*H*pairs / full_prefill formula, same wall-clock-only drained-eval
convention. Each round runs the version list in ALTERNATING order to cancel
monotone drift; SPIKE_ROUNDS rounds per (mix, version) give the P0 noise band
the gate is judged against.

Env:
  VERS=v19d,v19e,v20               versions (v20 needs the node pkg via P20_PKG)
  SPIKE_ROUNDS=5                   rounds per (mix, version); order alternates
  SPIKE_MIXES=real,uniform,skew
  P20_PKG=<scratch mlx_lm pkg>     (node only; enforces gemv_metal origin)
  MB_JSON=/tmp/p20moe_spike/p20_moe_spike_bench.json

Every timed region ends in a drained mx.eval; warmups drained; medians over
rounds. Wall clock only (gpu_time_ns is dead on this build).
"""
import os, sys, json, time, socket
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("STAGE", "spike")
import p20_moe_autopsy as A  # sets sys.path (P20_PKG first, ~/repos/exo)
import mlx.core as mx
from mlx_lm.models.exl3 import gemv_metal as G
from mlx_lm.models.exl3.loader import Exl3Checkpoint, load_experts

P20_PKG = os.environ.get("P20_PKG", "")
# On the node P20_PKG points at the scratch pkg that carries v20: enforce it
# when set, so a stray site-packages mlx_lm can never shadow the patched one.
# Left unset (local import smoke) it falls through to the venv's mlx_lm.
if P20_PKG:
    assert G.__file__.startswith(P20_PKG), G.__file__

VERS = [v for v in os.environ.get("VERS", "v19d,v19e,v20").split(",") if v]
ROUNDS = int(os.environ.get("SPIKE_ROUNDS", "5"))
MIXES = [m for m in os.environ.get("SPIKE_MIXES", "real,uniform,skew").split(",") if m]
OUT = os.environ.get("MB_JSON", "/tmp/p20moe_spike/p20_moe_spike_bench.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

R = A.R
R.update({"stage": "spike_bench", "vers": VERS, "rounds": ROUNDS, "mixes": MIXES,
          "gemv_metal": G.__file__, "mlx": mx.__version__, "res": {}})
A.OUT = OUT
# autopsy baselines (MoE TFLOPS, mix roofline TFLOPS) for the context lines
BASE = {"real": (6.63, 14.96), "uniform": (8.85, 15.15), "skew": (7.55, 15.10)}
GATE_PCT = 5.0


def log(*a):
    print("[spike]", *a, flush=True)


def bench_mix(name, sg, idx):
    """Run every version, ROUNDS rounds, alternating order; record the chained
    _full_prefill_ms (THE gate-#2 number) plus the isolated per-launch walls."""
    E, D, H = sg.num_experts, sg.input_dims, sg.hidden_dims
    F = 6.0 * D * H
    st = A.idx_stats(idx, E)
    x = mx.random.normal((1, idx.shape[1], D), key=mx.random.key(11)).astype(mx.float16)
    m = mx.array(idx, dtype=mx.uint32)
    mx.eval(x, m)
    acc = {v: {"full": [], "gu_gate": [], "gu_up": [], "dn": []} for v in VERS}
    for rnd in range(ROUNDS):
        order = VERS if rnd % 2 == 0 else VERS[::-1]  # alternate -> cancel drift
        for v in order:
            G._MM_SEG_VERSION = v
            tl, _ = A.time_launches(x, m, sg)
            acc[v]["full"].append(tl["_full_prefill_ms"])
            for k in ("gu_gate", "gu_up", "dn"):
                acc[v][k].append(tl[k]["ms"])
            log("  %s r%d %-5s full %.2f ms  gate %.2f up %.2f dn %.2f" % (
                name, rnd, v, tl["_full_prefill_ms"], tl["gu_gate"]["ms"],
                tl["gu_up"]["ms"], tl["dn"]["ms"]))
    rec = {"idx_stats": st, "E": E, "D": D, "H": H, "F": F}
    for v in VERS:
        fa = [float(z) for z in acc[v]["full"]]
        med = float(np.median(fa))
        rec[v] = {
            "full_prefill_ms_all": fa,
            "full_prefill_ms": med,
            "full_prefill_ms_min": float(np.min(fa)),
            "full_prefill_ms_max": float(np.max(fa)),
            "full_prefill_ms_mean": float(np.mean(fa)),
            "full_prefill_ms_stdev": float(np.std(fa, ddof=1)) if len(fa) > 1 else 0.0,
            "TFLOPS": F * st["pairs"] / (med / 1e3) / 1e12,
            **{k + "_ms": float(np.median(acc[v][k])) for k in ("gu_gate", "gu_up", "dn")},
        }
    return rec


def gate2():
    """P0 noise band (real / v19d) + gate-#2 lines on the CHAINED _full_prefill_ms.

    The 3*sigma band measured on real/v19d is THE significance bar: a variant
    delta must clear BOTH >= GATE_PCT% and > 3*sigma (in ms) to count."""
    real = R["res"].get("real")
    if not real or "v19d" not in real:
        R["p0_band"] = None
        R["gate2"] = None
        log("gate#2: SKIPPED (real mix or v19d missing)")
        return
    fa = real["v19d"]["full_prefill_ms_all"]
    mu = float(np.mean(fa))
    sd = float(np.std(fa, ddof=1)) if len(fa) > 1 else 0.0
    sig_pct = 100.0 * sd / mu
    t3 = 3.0 * sd
    t3_pct = 100.0 * t3 / mu
    R["p0_band"] = {"mix": "real", "vers": "v19d", "n": len(fa), "mean_ms": mu,
                    "sigma_ms": sd, "sigma_pct": sig_pct, "three_sigma_ms": t3,
                    "three_sigma_pct": t3_pct}
    print("P0 noise band: v19d real sigma=%.3f ms (%.2f%%) 3sigma=%.3f ms (%.2f%%)" % (
        sd, sig_pct, t3, t3_pct), flush=True)
    g2 = {}
    for name in MIXES:
        rec = R["res"].get(name)
        if not rec or "v19d" not in rec:
            continue
        ref = rec["v19d"]["full_prefill_ms"]
        g2[name] = {}
        for v in VERS:
            if v == "v19d" or v not in rec:
                continue
            got = rec[v]["full_prefill_ms"]
            dms = ref - got
            dpct = 100.0 * dms / ref
            clears = bool(dpct >= GATE_PCT and dms > t3)
            g2[name][v] = {"ref_v19d_full_ms": ref, "cand_full_ms": got,
                           "delta_ms": dms, "delta_pct": dpct,
                           "gt_3sigma": bool(dms > t3),
                           "ge_5pct": bool(dpct >= GATE_PCT),
                           "clears_both": clears}
            print("gate#2 [%s] %s vs v19d: full delta=%+.2f%% (%+.2f ms; need >=%.1f%%)  "
                  "3sigma=%.3f ms -> clears(>=5%% & >3sigma): %s" % (
                      name, v, dpct, dms, GATE_PCT, t3, "YES" if clears else "NO"),
                  flush=True)
    R["gate2"] = g2


def main():
    log("start", G.__file__, "vers", VERS, "rounds", ROUNDS, "mixes", MIXES)
    A.ceilings()
    ck = Exl3Checkpoint(A.CK)
    if "real" in MIXES:
        sg = load_experts(ck, 0, n_experts=None, rank=0, world=2, activation="silu_clamp")
        rng = np.random.RandomState(20261010)
        idx, _ = A.real_chunk_idx(A.TRACE, 0, 2048, rng)
        R["res"]["real"] = bench_mix("real", sg, idx)
        A.flushjson()
        del sg; mx.clear_cache()
    if any(m in MIXES for m in ("uniform", "skew")):
        sg = load_experts(ck, 0, n_experts=A.E_BLOCK, rank=0, world=2, activation="silu_clamp")
        for name, b in (("uniform", A.build_idx_uniform), ("skew", A.build_idx_skew)):
            if name not in MIXES:
                continue
            idx = b(sg.num_experts, A.NPROD, np.random.RandomState(2026))
            R["res"][name] = bench_mix(name, sg, idx)
            A.flushjson()
        del sg; mx.clear_cache()
    gate2()
    # context: %-of-mix-roofline, using the autopsy mix ceiling scaled by the
    # in-session dense ceiling (the mix/dense ratio is a pure function of the
    # M-histogram -> unchanged by the kernel variant; same rows, same weights).
    dense = R["ceil"]["dense_TFLOPS"]
    summ = {}
    for name in MIXES:
        rec = R["res"].get(name)
        if not rec:
            continue
        base_tf, base_mix = BASE.get(name, (None, None))
        mix = base_mix / 15.146 * dense if base_mix else None
        s = {"autopsy_baseline_TF": base_tf, "in_session_mix_roofline_TF": mix}
        for v in VERS:
            if v in rec:
                s[v + "_TF"] = rec[v]["TFLOPS"]
                if mix:
                    s[v + "_pct_mix"] = rec[v]["TFLOPS"] / mix * 100
        summ[name] = s
        log("[%s] " % name + " ".join("%s=%.3f" % kv for kv in s.items()
                                      if isinstance(kv[1], float)))
    R["summary"] = summ
    R["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    A.flushjson()
    log("DONE ->", OUT)


if __name__ == "__main__":
    main()
