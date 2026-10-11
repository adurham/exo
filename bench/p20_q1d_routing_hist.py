#!/usr/bin/env python3
"""p20_q1d_routing_hist.py -- Q1d: TRUE per-layer unique-expert routing histogram
of the DSv4.1 MoE (40 layers, 384 experts, topk6) from the REAL gate.

CPU / numpy ONLY -- no mlx model load, no GPU, no engine touch.  Run with the
node venv python on studio1 (staged under /tmp/q1d_hist, nothing written under
~/repos/exo).

Two arms:

  A. REAL TRACE ARM.  Reads an EXISTING captured real routing trace if present
     (~/p30_exl3_trace.json -- the phase-10 full 40-layer trace of the real EXL3
     checkpoint, 531 tokens x 40 layers, real gate top-6; a committed copy lives
     at docs/benchmarks/phase10-planb-gate-2026-09-28/raw/p30_exl3_trace_clamp.json).
     Groups R consecutive tokens per layer (R = the spec-verify row count) and
     measures the unique-expert count + per-expert row-count (m) histogram.  This
     is REAL routing, no synthetic hidden states.

  B. SYNTHETIC CORRELATED-ROWS ARM.  For every layer 0..39, for R in {1,4,8},
     over many seeds and correlation strengths corr in {0.01,0.03,0.1,...}, builds
     correlated draft rows (base N(0,1)[D] + corr*N(0,1)[R,D]) and runs the REAL
     gate (sqrt(softplus(x@W.T/temp)) + bias-only top-k, argpartition).  This is
     identical to bench/p20_q1c_opclass.py::make_rows + route_idx.

  C. SANITY.  Compares the numpy route_idx below against the REAL production
     deepseek_v41/moe.py Gate (imported CPU-side, mx.set_default_device(mx.cpu)).

Settles the N1 "[4,4,4,4,4,4] 6 experts x 4 rows" claim.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import socket
import statistics
import sys
import time

import numpy as np

REPO = os.path.expanduser("~/repos/exo")
MLXLM = os.path.join(REPO, "mlx-lm")
MODEL = os.path.expanduser(
    "~/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw")
TRACE_DEFAULT = os.path.expanduser("~/p30_exl3_trace.json")

N_MOE = 40
D_EXP, E_EXP, TOPK = 5120, 384, 6
ROUTE_SCALE = 1.5
TEMP = 1.0

CORR_VALUES = [0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 1.0]  # required {0.01,0.03,0.1} + match set
R_VALUES = [1, 4, 8]


# ---------------------------------------------------------------------------
# real gate: sqrt(softplus(x@W.T/temp)); bias-only top-k (matches production
# deepseek_v41/moe.py::Gate for text/decode rows, score_func="sqrtsoftplus")
# ---------------------------------------------------------------------------
def make_rows(R, D, seed, corr):
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(D).astype(np.float32)
    if R == 1:
        return base[None, :].copy()
    rows = base[None, :] + rng.standard_normal((R, D)).astype(np.float32) * corr
    rows[0] = base
    return rows


def route_idx(gw, gb, rows, topk, temp=1.0):
    s = rows @ gw.T / temp
    s = np.sqrt(np.logaddexp(s, 0.0))          # sqrt(softplus(z)), stable form
    biased = s + gb[None, :]                   # bias reorders, does not scale
    idx = np.argpartition(-biased, topk - 1, axis=-1)[:, :topk]
    return idx.astype(np.int32)


def uniq_and_hist(idx):
    flat = idx.reshape(-1).tolist()
    c = collections.Counter(flat)
    return len(c), tuple(sorted(c.values(), reverse=True))


def summarise_uniq(vals):
    a = np.asarray(vals, dtype=np.int64)
    hist = {int(k): int(v) for k, v in sorted(collections.Counter(a.tolist()).items())}
    return {
        "n_groups": int(a.size),
        "mean": round(float(a.mean()), 4),
        "median": float(np.median(a)),
        "p5": float(np.percentile(a, 5)),
        "p95": float(np.percentile(a, 95)),
        "min": int(a.min()),
        "max": int(a.max()),
        "hist": hist,
    }


def m_distribution(hists):
    c = collections.Counter()
    for h in hists:
        for m in h:
            c[int(m)] += 1
    tot = sum(c.values()) or 1
    share = {str(m): round(v / tot, 4) for m, v in sorted(c.items())}
    mean_m = sum(m * v for m, v in c.items()) / tot
    frac_ge2 = sum(v for m, v in c.items() if m >= 2) / tot
    return {"share": share, "mean_m": round(mean_m, 4),
            "frac_m_ge2": round(frac_ge2, 4), "max_m": int(max(c) if c else 0)}


# ---------------------------------------------------------------------------
# Arm A: real captured trace
# ---------------------------------------------------------------------------
def analyse_trace(path):
    if not os.path.exists(path):
        return {"found": False, "path": path}
    raw = open(path, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    TR = json.loads(raw)
    per = collections.defaultdict(list)
    for layer, idx in TR["records"]:
        per[int(layer)].append(tuple(int(x) for x in idx))
    layers = sorted(per)
    out = {
        "found": True, "path": path, "sha256": sha,
        "n_steps": TR.get("n_steps"), "prompt_tokens": TR.get("prompt_tokens"),
        "n_layers": len(layers), "topk": len(per[layers[0]][0]) if layers else None,
        "rows_per_layer": len(per[layers[0]]) if layers else 0,
        "per_layer_distinct_over_all_tokens": {
            str(L): len({e for t in per[L] for e in t}) for L in layers},
    }
    for R in R_VALUES:
        uniq, hists = [], []
        for L in layers:
            seq = per[L]
            for i in range(0, len(seq) - R + 1):
                flat = [e for t in seq[i:i + R] for e in t]
                c = collections.Counter(flat)
                uniq.append(len(c))
                hists.append(tuple(sorted(c.values(), reverse=True)))
        cell = summarise_uniq(uniq)
        cell["m_distribution"] = m_distribution(hists)
        cell["n6x4"] = sum(1 for h in hists if h == (4,) * 6) if R == 4 else None
        cell["n_uniq_eq_6"] = sum(1 for u in uniq if u == 6)
        # per-layer mean unique
        pl = {}
        for L in layers:
            seq = per[L]
            us = []
            for i in range(0, len(seq) - R + 1):
                flat = [e for t in seq[i:i + R] for e in t]
                us.append(len(set(flat)))
            pl[str(L)] = round(statistics.mean(us), 3)
        cell["per_layer_mean_unique"] = pl
        out[f"R{R}"] = cell
    return out


# ---------------------------------------------------------------------------
# Arm B: synthetic correlated rows, real gate, all layers
# ---------------------------------------------------------------------------
def analyse_gate(ckpt, n_seeds):
    by_corr = {}
    for corr in CORR_VALUES:
        by_corr[str(corr)] = {}
        # accumulate across all layers/seeds
        acc = {R: {"uniq": [], "hists": []} for R in R_VALUES}
        per_layer_R4 = {str(L): {"uniq": []} for L in range(N_MOE)}
        for L in range(N_MOE):
            gw = ckpt.np(f"layers.{L}.ffn.gate.weight").astype(np.float32)
            gb = ckpt.np(f"layers.{L}.ffn.gate.bias").astype(np.float32)
            for R in R_VALUES:
                for sd in range(n_seeds):
                    rows = make_rows(R, D_EXP, 7000 + sd, corr)
                    idx = route_idx(gw, gb, rows, TOPK, TEMP)
                    nu, h = uniq_and_hist(idx)
                    acc[R]["uniq"].append(nu)
                    acc[R]["hists"].append(h)
                    if R == 4:
                        per_layer_R4[str(L)]["uniq"].append(nu)
        for R in R_VALUES:
            cell = summarise_uniq(acc[R]["uniq"])
            cell["m_distribution"] = m_distribution(acc[R]["hists"])
            if R == 4:
                cell["n6x4"] = sum(1 for h in acc[R]["hists"]
                                   if h == (4,) * 6)
                cell["n_uniq_eq_6"] = sum(1 for u in acc[R]["uniq"] if u == 6)
                cell["per_layer_mean_unique"] = {
                    L: round(statistics.mean(v["uniq"]), 3)
                    for L, v in per_layer_R4.items()}
            by_corr[str(corr)][f"R{R}"] = cell
    return by_corr


# ---------------------------------------------------------------------------
# Arm C: sanity vs the real production Gate
# ---------------------------------------------------------------------------
def sanity(ckpt, layers=(0, 20, 39)):
    res = {"method": "numpy route_idx vs real deepseek_v41/moe.py Gate "
                     "(imported CPU-side, mx.set_default_device(mx.cpu))"}
    try:
        import mlx.core as mx
        mx.set_default_device(mx.cpu)
        from mlx_lm.models.deepseek_v41.moe import Gate

        class _A:
            pass

        rows = make_rows(4, D_EXP, 12345, 0.05)
        checks = []
        for L in layers:
            gw = ckpt.np(f"layers.{L}.ffn.gate.weight").astype(np.float32)
            gb = ckpt.np(f"layers.{L}.ffn.gate.bias").astype(np.float32)
            a = _A()
            a.n_routed_experts = E_EXP
            a.n_activated_experts = TOPK
            a.score_func = "sqrtsoftplus"
            a.gate_temp = TEMP
            a.norm_topk_prob = True
            a.route_scale = ROUTE_SCALE
            a.dim = D_EXP
            g = Gate(a)
            g.weight = mx.array(gw)
            g.bias = mx.array(gb)
            w, i = g(mx.array(rows.astype(np.float32)))
            mx.eval(w, i)
            midx = np.asarray(i)
            nidx = route_idx(gw, gb, rows, TOPK, TEMP)
            set_ok = all(set(midx[r].tolist()) == set(nidx[r].tolist())
                         for r in range(rows.shape[0]))
            exact = bool((np.sort(midx, axis=1) == np.sort(nidx, axis=1)).all())
            checks.append({"layer": L, "set_match": bool(set_ok),
                           "sorted_exact": exact,
                           "mlx_row0": sorted(int(x) for x in midx[0]),
                           "np_row0": sorted(int(x) for x in nidx[0])})
        res["import"] = "ok"
        res["checks"] = checks
        res["all_set_match"] = all(c["set_match"] for c in checks)
    except Exception as exc:  # noqa: BLE001
        res["import"] = "FAILED"
        res["error"] = repr(exc)
        res["formula_documented"] = (
            "scores = (x.fp32 @ W.fp32.T)/gate_temp; "
            "scores = sqrt(softplus(scores))  [score_func=sqrtsoftplus]; "
            "biased = scores + gate.bias; "
            "indices = argpartition(-biased, topk-1)[..., :topk]. "
            "Bias reorders only (weights read from UNBIASED scores).")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--trace", default=TRACE_DEFAULT)
    ap.add_argument("--out", default="/tmp/q1d_hist/q1d_routing_hist.json")
    args = ap.parse_args()

    sys.path.insert(0, MLXLM)
    from mlx_lm.models.exl3.loader import Exl3Checkpoint  # noqa: E402

    t0 = time.perf_counter()
    ckpt = Exl3Checkpoint(MODEL)

    # --- provenance of the two synthetic make_rows/route_idx (q1c parity) ---
    q1c_parity = None
    gw0 = ckpt.np("layers.20.ffn.gate.weight").astype(np.float32)
    gb0 = ckpt.np("layers.20.ffn.gate.bias").astype(np.float32)
    rows = make_rows(4, D_EXP, 1000, 0.03)
    idx = route_idx(gw0, gb0, rows, TOPK, TEMP)
    nu, h = uniq_and_hist(idx)
    q1c_parity = {"layer": 20, "corr": 0.03, "seed": 1000,
                  "unique": nu, "hist": list(h),
                  "matches_q1c": (nu == 6 and h == (4,) * 6)}
    del gw0, gb0

    result = {
        "meta": {
            "bench": "p20_q1d_routing_hist",
            "node": socket.gethostname(),
            "n_moe_layers": N_MOE, "E": E_EXP, "D": D_EXP, "topk": TOPK,
            "gate_temp": TEMP, "route_scale": ROUTE_SCALE,
            "numpy": np.__version__, "seeds_per_cell": args.seeds,
            "corr_values": CORR_VALUES, "R_values": R_VALUES,
            "method": "real gate sqrt(softplus(x@W.T/temp))+bias-only argpartition top-6",
        },
        "q1c_parity_check": q1c_parity,
        "real_trace": analyse_trace(args.trace),
        "synthetic_gate": analyse_gate(ckpt, args.seeds),
        "sanity_check": sanity(ckpt),
        "load_s": round(time.perf_counter() - t0, 1),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[q1d] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
