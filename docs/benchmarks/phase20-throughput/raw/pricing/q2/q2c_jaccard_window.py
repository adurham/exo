#!/usr/bin/env python
"""q2c — hot-set Jaccard / turnover vs window length (DESK, CPU-numpy only).

Source: the real EXL3 production-clamp routing trace
  docs/benchmarks/phase10-planb-gate-2026-09-28/raw/p30_exl3_trace_clamp.json
  sha256 b4f1b142a5c5ef779e87b8eb3f041f4e822fdc5de8fe858ea9c8fba1bde91a11
  531 tokens x 40 layers = 21240 real gate decisions (topk6), layer-major.

Definitions (frozen):
  batch  = one R=4 consecutive-token spec-verify group (gamma=3 -> verify R=gamma+1=4).
           531 tokens -> nb = floor(531/4) = 132 full batches/layer (last 3 tokens dropped).
  window = W consecutive batches (W*4 tokens). Tiling into nwin = 132 // W non-overlapping
           windows per layer; adjacent pairs = nwin-1.
  hot set = top-k experts by pick frequency within the window (stable argsort desc,
           ties -> lowest expert id, matching Q1E c_trace_mining.py::top_set).
  Jaccard(A,B) = |A n B| / |A u B| between hot sets of ADJACENT windows.
  turnover   = 1 - |A n B| / k = fraction of the previous window's hot-k not in the current
           (k = hot-set size, both sets size k).
Controls:
  random baseline    = E[Jaccard] of two independent random k-subsets of E=384
                       = (k^2/E) / (2k - k^2/E)  (analytic; + empirical MC check).
  parity control     = even-index tokens vs odd-index tokens (Q1E's interleaved control),
                       top-k over each half of that layer's 531 picks.

CPU/numpy only. No engine, no cluster, no mlx.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

E = 384
TOPK = 6
R = 4  # batch = R consecutive tokens (spec-verify group, gamma 3)
WS = [1, 2, 4, 8, 16, 32]
KS = [8, 24, 48]
TRACE_SHA = "b4f1b142a5c5ef779e87b8eb3f041f4e822fdc5de8fe858ea9c8fba1bde91a11"


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: str):
    with open(path) as f:
        d = json.load(f)
    n = int(d["n_steps"])
    nL = len(d["gate_meta"])
    recs = d["records"]
    assert len(recs) == n * nL, (len(recs), n * nL)
    picks = np.empty((nL, n, TOPK), dtype=np.int32)
    for idx, (L, ids) in enumerate(recs):
        picks[idx // n, idx % n] = ids
    assert picks.min() >= 0 and picks.max() < E
    return d, picks, n, nL


def top_set(counts: np.ndarray, k: int) -> set:
    order = np.argsort(-counts, kind="stable")
    return set(int(x) for x in order[:k])


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def qstats(x) -> dict:
    x = np.asarray(x, dtype=float)
    q25 = float(np.percentile(x, 25))
    q75 = float(np.percentile(x, 75))
    return {
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "q25": q25,
        "q75": q75,
        "iqr": q75 - q25,
        "min": float(x.min()),
        "max": float(x.max()),
        "n": int(x.size),
    }


def rand_baseline(k: int) -> float:
    # analytic E[Jaccard] for two independent random k-subsets of E
    r = k * k / E
    return r / (2 * k - r)


def rand_baseline_mc(k: int, draws_sets: int = 200000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    a = np.array([set(rng.choice(E, k, replace=False).tolist()) for _ in range(draws_sets)])
    b = np.array([set(rng.choice(E, k, replace=False).tolist()) for _ in range(draws_sets)])
    js = [jaccard(a[i], b[i]) for i in range(draws_sets)]
    return float(np.mean(js))


def window_sets(b_pick: np.ndarray, nb: int, W: int, k: int):
    """b_pick: (nb, R*TOPK) flattened picks per batch. Returns list of hot sets per window."""
    nwin = nb // W
    sets = []
    for i in range(nwin):
        chunk = b_pick[i * W:(i + 1) * W].reshape(-1)
        cnt = np.bincount(chunk, minlength=E)
        sets.append(top_set(cnt, k))
    return sets


def window_sets_capped(b_pick: np.ndarray, nb: int, W: int, k: int):
    """Hot set = top-min(k, D) experts, D = #distinct seen in window (no zero-count filler)."""
    nwin = nb // W
    sets, dist = [], []
    for i in range(nwin):
        chunk = b_pick[i * W:(i + 1) * W].reshape(-1)
        cnt = np.bincount(chunk, minlength=E)
        D = int((cnt > 0).sum())
        sets.append(top_set(cnt, min(k, D)))
        dist.append(D)
    return sets, dist


def curve_from_sets(sets, k: int):
    js, to = [], []
    for i in range(len(sets) - 1):
        j = jaccard(sets[i], sets[i + 1])
        js.append(j)
        to.append(1.0 - len(sets[i] & sets[i + 1]) / k)
    return js, to


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", default=None)
    args = ap.parse_args()

    sha = sha256(args.trace)
    d, picks, n, nL = load(args.trace)
    nb = n // R  # 132

    # flattened per-batch picks per layer: (nL, nb, R*TOPK)
    bflat = picks[:, : nb * R].reshape(nL, nb, R, TOPK).reshape(nL, nb, R * TOPK)

    # ---------- primary: batch-window curve, per-layer pooled ----------
    batch_curve = {f"k{k}": {} for k in KS}
    perlayer_median = {f"k{k}": {} for k in KS}
    for k in KS:
        for W in WS:
            allj, allto, perL = [], [], []
            for L in range(nL):
                sets = window_sets(bflat[L], nb, W, k)
                js, to = curve_from_sets(sets, k)
                allj += js
                allto += to
                if js:
                    perL.append(float(np.median(js)))
            s = qstats(allj)
            batch_curve[f"k{k}"][str(W)] = {
                "jaccard_median": s["median"],
                "jaccard_iqr": s["iqr"],
                "jaccard_q25": s["q25"],
                "jaccard_q75": s["q75"],
                "jaccard_mean": s["mean"],
                "jaccard_min": s["min"],
                "jaccard_max": s["max"],
                "turnover_mean": float(np.mean(allto)) if allto else None,
                "turnover_median": float(np.median(allto)) if allto else None,
                "n_pairs": s["n"],
                "n_windows_per_layer": nb // W,
                "n_pairs_per_layer": (nb // W) - 1,
            }
            perlayer_median[f"k{k}"][str(W)] = {
                "median_of_perlayer_medians": float(np.median(perL)),
                "min_perlayer_median": float(np.min(perL)),
                "max_perlayer_median": float(np.max(perL)),
            }

    # ---------- sensitivity: distinct-capped hot set (no zero-count filler) ----------
    capped_curve = {f"k{k}": {} for k in KS}
    for k in KS:
        for W in WS:
            allj, allto, alld, nsat = [], [], [], 0
            for L in range(nL):
                sets, dist = window_sets_capped(bflat[L], nb, W, k)
                alld += dist
                for i in range(len(sets) - 1):
                    a, b = sets[i], sets[i + 1]
                    j = jaccard(a, b)
                    denom = max(len(a), len(b), 1)
                    allj.append(j)
                    allto.append(1.0 - len(a & b) / denom)
                    nsat += 1 if (len(a) == k and len(b) == k) else 0
            s = qstats(allj)
            capped_curve[f"k{k}"][str(W)] = {
                "jaccard_median": s["median"], "jaccard_iqr": s["iqr"],
                "jaccard_q25": s["q25"], "jaccard_q75": s["q75"],
                "jaccard_min": s["min"], "jaccard_max": s["max"],
                "turnover_mean": float(np.mean(allto)) if allto else None,
                "n_pairs": s["n"],
                "mean_distinct_per_window": float(np.mean(alld)),
                "frac_pairs_both_windows_saturated_at_k": (nsat / s["n"]) if s["n"] else None,
            }

    # ---------- global (union across layers): window tokens pooled over all 40 layers ----------
    global_curve = {f"k{k}": {} for k in KS}
    for k in KS:
        for W in WS:
            nwin = nb // W
            tokw = W * R
            sets = []
            for i in range(nwin):
                toks = np.arange(i * tokw, (i + 1) * tokw)
                chunk = picks[:, toks, :].reshape(-1)
                cnt = np.bincount(chunk, minlength=E)
                sets.append(top_set(cnt, k))
            js, to = curve_from_sets(sets, k)
            s = qstats(js) if js else {kk: None for kk in
                                       ("median", "iqr", "q25", "q75", "mean", "min", "max", "n")}
            global_curve[f"k{k}"][str(W)] = {
                "jaccard_median": s["median"],
                "jaccard_iqr": s["iqr"],
                "jaccard_q25": s["q25"],
                "jaccard_q75": s["q75"],
                "jaccard_mean": s["mean"],
                "jaccard_min": s["min"],
                "jaccard_max": s["max"],
                "turnover_mean": float(np.mean(to)) if to else None,
                "n_pairs": s["n"],
            }

    # ---------- per-token adjacent curve (window = 1 token) ----------
    # per token the hot set = the (<=6 distinct) experts the token picks; Jaccard of the
    # adjacent-token 6-expert sets. Independent of k>6 (single token picks only topk=6).
    tok_j, tok_to, tok_nuniq = [], [], []
    for L in range(nL):
        sets = [set(int(x) for x in picks[L, t]) for t in range(n)]
        for t in range(n - 1):
            tok_j.append(jaccard(sets[t], sets[t + 1]))
            tok_to.append(1.0 - len(sets[t] & sets[t + 1]) / max(1, len(sets[t])))
        for t in range(n):
            tok_nuniq.append(len(sets[t]))
    sj = qstats(tok_j)
    per_token_curve = {
        "definition": "window = 1 token; hot set = the topk=6 picked experts (<=6 distinct).",
        "jaccard_median": sj["median"], "jaccard_iqr": sj["iqr"],
        "jaccard_q25": sj["q25"], "jaccard_q75": sj["q75"],
        "jaccard_mean": sj["mean"], "jaccard_min": sj["min"], "jaccard_max": sj["max"],
        "turnover_mean": float(np.mean(tok_to)),
        "n_pairs": sj["n"],
        "mean_unique_experts_per_token": float(np.mean(tok_nuniq)),
    }

    # ---------- parity control (even vs odd token halves), per k ----------
    parity = {}
    for k in KS:
        js, to = [], []
        for L in range(nL):
            ev = picks[L, 0::2, :].reshape(-1)
            od = picks[L, 1::2, :].reshape(-1)
            te = top_set(np.bincount(ev, minlength=E), k)
            td = top_set(np.bincount(od, minlength=E), k)
            js.append(jaccard(te, td))
            to.append(1.0 - len(te & td) / k)
        s = qstats(js)
        parity[f"k{k}"] = {
            "jaccard_median": s["median"], "jaccard_iqr": s["iqr"],
            "jaccard_q25": s["q25"], "jaccard_q75": s["q75"],
            "jaccard_mean": s["mean"], "jaccard_min": s["min"], "jaccard_max": s["max"],
            "turnover_mean": float(np.mean(to)), "n_pairs_layers": len(js),
        }

    # ---------- sliding-window variant (step = 1 batch), robustness at large W ----------
    sliding_curve = {f"k{k}": {} for k in KS}
    for k in KS:
        for W in WS:
            allj, npr = [], 0
            for L in range(nL):
                nwin = nb - W + 1
                # build sliding sets explicitly (step = 1 batch)
                sets = []
                for i in range(nwin):
                    chunk = bflat[L, i:i + W].reshape(-1)
                    sets.append(top_set(np.bincount(chunk, minlength=E), k))
                js, _to = curve_from_sets(sets, k)
                allj += js
                npr += len(js)
            s = qstats(allj) if allj else {"median": None, "iqr": None, "n": 0}
            sliding_curve[f"k{k}"][str(W)] = {
                "jaccard_median": s["median"], "jaccard_iqr": s["iqr"], "n_pairs": s["n"],
            }

    # ---------- random baselines ----------
    rand = {}
    for k in KS:
        rand[f"k{k}"] = {
            "analytic": rand_baseline(k),
            "empirical_mc": rand_baseline_mc(k),
            "n_subsets_each": 200000,
            "E": E,
        }

    out = {
        "meta": {
            "bench": "q2c_jaccard_window",
            "mode": "desk-cpu-numpy",
            "trace_path": os.path.basename(args.trace),
            "trace_sha256": sha,
            "trace_sha256_expected": TRACE_SHA,
            "sha_match": sha == TRACE_SHA,
            "n_steps": n, "n_layers": nL, "E": E, "topk": TOPK,
            "R": R, "n_batches_per_layer": nb,
            "records": len(d["records"]),
            "ordering": "layer-major: records[L*531:(L+1)*531] = layer L, token t=0..530",
            "numpy_version": np.__version__,
        },
        "definitions": {
            "batch": "one R=4 consecutive-token spec-verify group (gamma=3 -> verify R=gamma+1). "
                     "531 tokens -> nb=floor(531/4)=132 full batches/layer; last 3 tokens dropped.",
            "window": "W consecutive batches (W*4 tokens); tiling into nwin=132//W non-overlapping "
                      "windows/layer; adjacent pairs = nwin-1.",
            "hot_set": "top-k experts by pick frequency within the window (stable desc argsort, "
                       "ties -> lowest id; matches Q1E top_set).",
            "jaccard": "|A n B| / |A u B| between hot sets of ADJACENT windows.",
            "turnover": "1 - |A n B|/k = fraction of the previous window's hot-k not in the current.",
            "parity_control": "even-index tokens vs odd-index tokens (Q1E interleaved control).",
            "random_baseline": "(k^2/E)/(2k - k^2/E) = E[Jaccard] of two independent random "
                               "k-subsets of E=384.",
        },
        "batch_curve_perlayer_pooled": batch_curve,
        "batch_curve_distinct_capped_sensitivity": capped_curve,
        "perlayer_median_of_medians": perlayer_median,
        "batch_curve_global_union_across_layers": global_curve,
        "per_token_curve": per_token_curve,
        "parity_control": parity,
        "random_baseline": rand,
        "sliding_curve_step1_robustness": sliding_curve,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print("trace sha:", sha, "match:", sha == TRACE_SHA)
    print("nb/layer:", nb, "layers:", nL)
    for k in KS:
        row = batch_curve[f"k{k}"]
        print(f"--- k={k} (per-layer pooled) ---")
        for W in WS:
            r = row[str(W)]
            print(f"  W={W:2d}: median={r['jaccard_median']:.4f} iqr={r['jaccard_iqr']:.4f} "
                  f"min={r['jaccard_min']:.4f} max={r['jaccard_max']:.4f} "
                  f"turnover={r['turnover_mean']:.4f} n={r['n_pairs']}")
    print("--- parity ---")
    for k in KS:
        p = parity[f"k{k}"]
        print(f"  k={k}: median={p['jaccard_median']:.4f} iqr={p['jaccard_iqr']:.4f} "
              f"mean={p['jaccard_mean']:.4f} turnover={p['turnover_mean']:.4f}")
    print("--- per-token (window=1 token, 6-expert set) ---")
    print(" ", json.dumps({kk: per_token_curve[kk] for kk in
                           ("jaccard_median", "jaccard_iqr", "jaccard_mean",
                            "turnover_mean", "n_pairs")}))
    print("--- random baseline ---")
    for k in KS:
        print(f"  k={k}: analytic={rand[f'k{k}']['analytic']:.4f} mc={rand[f'k{k}']['empirical_mc']:.4f}")
    print("--- distinct-capped sensitivity (no filler) ---")
    for k in KS:
        row = capped_curve[f"k{k}"]
        print(f"  k={k}: " + " | ".join(
            f"W{W}={row[str(W)]['jaccard_median']:.4f}(satu {row[str(W)]['frac_pairs_both_windows_saturated_at_k']:.2f},D {row[str(W)]['mean_distinct_per_window']:.0f})"
            for W in WS))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
