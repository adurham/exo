#!/usr/bin/env python3
"""q1e — mine the saved 531x40 EXL3 clamp routing trace (local CPU / numpy only).

Read-only on the trace. Produces c_trace_mining.json + c-trace-mining.md.
Four analyses:
  1. per-layer unique-expert union + frequency-rank curve (coverage points)
  2. hot-set concentration (top-k share) + STABILITY (window Jaccard etc.)
  3. per-rank (2-way 192/192 EP split) touched-expert imbalance for R=4 / R=1
  4. batch-to-batch (R=4) hot-set Jaccard overlap
"""
import json, hashlib, os
import numpy as np

TRACE = "/private/tmp/phase20-campaign/docs/benchmarks/phase10-planb-gate-2026-09-28/raw/p30_exl3_trace_clamp.json"
OUTDIR = os.path.dirname(os.path.abspath(__file__))
E = 384
TOPK = 6
NLAYERS = 40

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def load():
    with open(TRACE) as f:
        d = json.load(f)
    recs = d["records"]
    n = d["n_steps"]
    layers = sorted(int(k) for k in d["gate_meta"].keys())
    gm = d["gate_meta"]
    # build picks[layer] -> int array (n, 6), layer-major
    picks = {}
    for L in layers:
        block = recs[L * n:(L + 1) * n]
        arr = np.array([r[1] for r in block], dtype=np.int64)
        lyrs = set(r[0] for r in block)
        assert lyrs == {L}, f"layer block {L} mixed: {lyrs}"
        picks[L] = arr
    return d, picks, n, layers, gm

# ---------- helpers ----------
def counts_of(arr):
    """bincount over expert ids -> dense array len E."""
    return np.bincount(arr.ravel(), minlength=E)

def coverage_k(cnt, thresh):
    """smallest k (number of experts, sorted desc by count) whose cumulative
    count >= thresh * total picks. Returns (k, k_as_frac_of_touched)."""
    total = int(cnt.sum())
    order = np.argsort(-cnt, kind="stable")
    c = cnt[order]
    cum = np.cumsum(c)
    need = thresh * total
    k = int(np.searchsorted(cum, need, side="left")) + 1
    k = min(k, int((cnt > 0).sum()))
    touched = int((cnt > 0).sum())
    return k, (k / touched if touched else float("nan"))

def top_set(cnt, k):
    order = np.argsort(-cnt, kind="stable")
    return set(order[:k].tolist())

def jaccard(a, b):
    if not a and not b:
        return 1.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0

def quantiles(x):
    x = np.asarray(x, dtype=float)
    return {
        "mean": float(np.mean(x)), "median": float(np.median(x)),
        "p5": float(np.percentile(x, 5)), "p95": float(np.percentile(x, 95)),
        "min": float(np.min(x)), "max": float(np.max(x)), "n": int(x.size),
    }

# =================== MAIN ===================
def main():
    d, picks, n, layers, gm = load()
    sha = sha256(TRACE)
    assert max(int(picks[L].max()) for L in layers) < E
    assert min(int(picks[L].min()) for L in layers) >= 0
    picks_total_per_layer = n * TOPK  # 3186

    result = {"meta": {
        "bench": "q1e_c_trace_mining",
        "trace_path": TRACE,
        "trace_sha256": sha,
        "n_steps": n, "prompt_tokens": d["prompt_tokens"],
        "n_layers": len(layers), "E": E, "topk": TOPK,
        "picks_per_layer": picks_total_per_layer,
        "records": len(d["records"]),
        "ordering": "layer-major: records[L*531:(L+1)*531] = layer L, token t=0..530",
        "gate_meta": {k: gm[k] for k in sorted(gm, key=int)},
        "gate_meta_note": "per-layer keys '0'..'39' each hold ONLY {n_experts:384, topk:6}; "
                          "no temperature/bias/scale fields present",
        "rank_split_assumption": "rank0 = expert ids 0..191, rank1 = 192..383 (2-way EP, "
                                 "even contiguous split); MODELING ASSUMPTION - the trace "
                                 "does not encode rank assignment. NOTE: this is NOT the "
                                 "production TP partition (see analysis3."
                                 "production_geometry_widthshard) - it is an EP counterfactual",
        "production_tp_partition": "intermediate-WIDTH sharding; both ranks hold all 384 "
                                   "experts at half width; no expert-identity partition "
                                   "(auto_parallel.py:1164-1175)",
        "numpy": np.__version__,
    }}

    # ------------- Analysis 1 -------------
    a1 = {"per_layer": {}, "overall": {}}
    all_ids = np.concatenate([picks[L].ravel() for L in layers])
    cnt_all = np.bincount(all_ids, minlength=E)
    a1["overall"] = {
        "distinct_experts": int((cnt_all > 0).sum()),
        "of_E": E,
        "total_picks": int(cnt_all.sum()),
        "top10_counts": [int(x) for x in np.sort(cnt_all)[::-1][:10]],
        "coverage_k": {f"{t}": coverage_k(cnt_all, t / 100.0)[0]
                       for t in (50, 80, 90, 95, 99)},
        "expert_with_min_count": int(cnt_all.min()),
        "experts_never_hit": int((cnt_all == 0).sum()),
    }
    union_sizes = []
    cov_ks = {t: [] for t in (50, 80, 90, 95, 99)}
    top10_all = []
    for L in layers:
        cnt = counts_of(picks[L])
        touched = int((cnt > 0).sum())
        union_sizes.append(touched)
        curve = np.sort(cnt)[::-1]
        top10_all.append([int(x) for x in curve[:10]])
        per = {"distinct_experts": touched,
               "top10_counts": [int(x) for x in curve[:10]],
               "max_count": int(curve[0]), "min_pos_count": int(cnt[cnt > 0].min())}
        for t in (50, 80, 90, 95, 99):
            k, kf = coverage_k(cnt, t / 100.0)
            cov_ks[t].append(k)
            per[f"k_cov_{t}"] = k
            per[f"kfrac_cov_{t}"] = round(kf, 4)
        a1["per_layer"][str(L)] = per
    us = np.array(union_sizes)
    a1["union_sizes_summary"] = {
        "min": int(us.min()), "median": float(np.median(us)), "max": int(us.max()),
        "mean": float(us.mean()), "of_E": E,
        "argmin_layer": int(layers[int(np.argmin(us))]),
        "argmax_layer": int(layers[int(np.argmax(us))]),
    }
    aCo = {}
    for t in (50, 80, 90, 95, 99):
        arr = np.array(cov_ks[t], dtype=float)
        aCo[f"{t}"] = {"min": int(arr.min()), "median": float(np.median(arr)),
                       "max": int(arr.max()), "mean": float(arr.mean()),
                       "as_frac_of_union_mean": round(float(arr.mean() / us.mean()), 4)}
    a1["coverage_k_summary"] = aCo
    result["analysis1_union_and_rank_curve"] = a1

    # ------------- Analysis 2 -------------
    a2 = {"topk_share": {}, "stability": {}}
    KS = [1, 2, 4, 8, 16, 24, 48, 96]
    share = {k: [] for k in KS}
    for L in layers:
        cnt = counts_of(picks[L]); total = cnt.sum()
        order = np.argsort(-cnt, kind="stable")
        cum = np.cumsum(cnt[order])
        for k in KS:
            kk = min(k, len(order))
            share[k].append(float(cum[kk - 1] / total))
    a2["topk_share"] = {
        f"top{k}": {
            "mean": float(np.mean(share[k])), "median": float(np.median(share[k])),
            "min": float(np.min(share[k])), "max": float(np.max(share[k])),
        } for k in KS}
    a2["topk_share_note"] = ("fraction of the 3186 per-layer token-picks covered by the "
                             "k most-used experts of that layer (per-layer, then "
                             "aggregated across 40 layers)")

    TOPK_HOT = 24
    def windows_of(a, parts):
        return np.array_split(np.arange(a.shape[0]), parts)
    stab = {"jaccard_top24": {}, "later_in_earlier_top24": {}, "note": ""}
    # 2 windows
    j2, f2_1in0 = [], []
    j3_01, j3_12, j3_02, f3 = [[] for _ in range(4)]
    jp, fp = [], []  # parity
    for L in layers:
        P = picks[L]
        # 2 windows (P[idx] -> actual picks; windows_of returns row indices)
        w = windows_of(P, 2)
        W = [P[i] for i in w]
        c0, c1 = counts_of(W[0]), counts_of(W[1])
        t0, t1 = top_set(c0, TOPK_HOT), top_set(c1, TOPK_HOT)
        j2.append(jaccard(t0, t1))
        f2_1in0.append(float(np.isin(W[1].ravel(), list(t0)).mean()))
        # 3 windows
        w3 = windows_of(P, 3)
        W3 = [P[i] for i in w3]
        c = [counts_of(x) for x in W3]
        t = [top_set(ci, TOPK_HOT) for ci in c]
        j3_01.append(jaccard(t[0], t[1])); j3_12.append(jaccard(t[1], t[2]))
        j3_02.append(jaccard(t[0], t[2]))
        f3.append((float(np.isin(W3[1].ravel(), list(t[0])).mean()),
                   float(np.isin(W3[2].ravel(), list(t[1])).mean()),
                   float(np.isin(W3[2].ravel(), list(t[0])).mean())))
        # parity
        ev = np.arange(0, P.shape[0], 2); od = np.arange(1, P.shape[0], 2)
        te, to = top_set(counts_of(P[ev]), TOPK_HOT), top_set(counts_of(P[od]), TOPK_HOT)
        jp.append(jaccard(te, to))
        fp.append(float(np.isin(P[od].ravel(), list(te)).mean()))
    stab["jaccard_top24"] = {
        "two_windows_265_266": quantiles(j2),
        "three_windows": {"w0_w1": quantiles(j3_01), "w1_w2": quantiles(j3_12),
                          "w0_w2": quantiles(j3_02)},
        "parity_even_odd": quantiles(jp),
    }
    stab["later_in_earlier_top24"] = {
        "two_windows_w1in_w0top24": quantiles(f2_1in0),
        "three_windows": {"w1in_w0": quantiles([x[0] for x in f3]),
                          "w2in_w1": quantiles([x[1] for x in f3]),
                          "w2in_w0": quantiles([x[2] for x in f3])},
        "parity_oddin_eventop24": quantiles(fp),
    }
    stab["note"] = ("top-24 expert set per window (window = contiguous token slice of "
                    "that layer's 531 picks); Jaccard = |A n B|/|A u B|. "
                    "'later_in_earlier_top24' = fraction of the later window's 3186-picks "
                    "whose expert id is in the earlier window's top-24.")
    # baseline: random top-24 of 384 -> expected Jaccard ~ 24*23/(384*... ) ~ 0.047
    a2["stability"] = stab
    _k, _E = float(TOPK_HOT), float(E)
    _pj = (_k * _k / _E) / (2 * _k - _k * _k / _E)  # rand 24-subset Jaccard of 384
    a2["stability"]["random_baseline_note"] = (
        "expected Jaccard of two independent random 24-subsets of 384 experts = ~%.4f"
        % _pj)
    result["analysis2_hotset_and_stability"] = a2

    # ------------- Analysis 3 -------------
    R0 = np.arange(0, 192); R1 = np.arange(192, 384)
    a3 = {}
    def rank_batches(P, R):
        ntok = P.shape[0]
        nb = ntok // R
        rows = []
        for b in range(nb):
            blk = P[b * R:(b + 1) * R].ravel()
            t0 = len(set(blk[np.isin(blk, R0)].tolist()))
            t1 = len(set(blk[np.isin(blk, R1)].tolist()))
            p0 = int(np.isin(blk, R0).sum()); p1 = int(np.isin(blk, R1).sum())
            rows.append((t0, t1, p0, p1))
        return np.array(rows, dtype=float)
    for R in (4, 1):
        per_layer_rows = [rank_batches(picks[L], R) for L in layers]
        allrows = np.concatenate(per_layer_rows, axis=0)
        t0, t1 = allrows[:, 0], allrows[:, 1]
        pooled = np.concatenate([t0, t1])
        mx = np.maximum(t0, t1); mn = np.minimum(t0, t1)
        ratio_mxmean = mx / np.where(((t0 + t1) / 2) > 0, (t0 + t1) / 2, np.nan)
        ratio_mxmn = mx / np.where(mn > 0, mn, np.nan)
        # pick-count imbalance
        pmx = np.maximum(allrows[:, 2], allrows[:, 3])
        ratio_pick_mxmn = pmx / np.where(np.minimum(allrows[:, 2], allrows[:, 3]) > 0,
                                         np.minimum(allrows[:, 2], allrows[:, 3]), np.nan)
        a3[f"R{R}"] = {
            "n_batches": int(allrows.shape[0]),
            "rank0_touched": quantiles(t0), "rank1_touched": quantiles(t1),
            "pooled_touched": quantiles(pooled),
            "max_rank_touched_ratio_to_mean": {
                "p50": float(np.nanmedian(ratio_mxmean)),
                "p95": float(np.nanpercentile(ratio_mxmean, 95)),
                "max": float(np.nanmax(ratio_mxmean)),
                "frac_ge_2x": float(np.nanmean(ratio_mxmean >= 2.0)),
                "frac_ge_1.5x": float(np.nanmean(ratio_mxmean >= 1.5)),
                "n_ge_2x": int(np.nansum(ratio_mxmean >= 2.0)),
            },
            "max_rank_touched_ratio_to_min": {
                "p50": float(np.nanmedian(ratio_mxmn)),
                "p95": float(np.nanpercentile(ratio_mxmn, 95)),
                "max": float(np.nanmax(ratio_mxmn)),
                "frac_ge_2x": float(np.nanmean(ratio_mxmn >= 2.0)),
                "n_ge_2x": int(np.nansum(ratio_mxmn >= 2.0)),
            },
            "pickcount_imbalance_max_over_min": {
                "p50": float(np.nanmedian(ratio_pick_mxmn)),
                "p95": float(np.nanpercentile(ratio_pick_mxmn, 95)),
                "max": float(np.nanmax(ratio_pick_mxmn)),
                "frac_ge_2x": float(np.nanmean(ratio_pick_mxmn >= 2.0)),
            },
            "mean_rank0_touched": float(t0.mean()), "mean_rank1_touched": float(t1.mean()),
        }
    a3["assumption"] = ("rank0=ids 0..191, rank1=192..383 (2-way even contiguous EP split). "
                        "MODELING ASSUMPTION - trace has no rank labels. Per batch = R "
                        "consecutive tokens; 'touched' = distinct experts in that rank's "
                        "slice of the R*6=6R picks.")
    # Production reality: DSv4.1 TP is INTERMEDIATE-WIDTH sharding, not expert-id EP.
    # Both ranks hold ALL experts at half width -> per-rank touched set is identical
    # (= the batch union) for every token -> imbalance is 1.0 by construction.
    prod_union = []
    for L in layers:
        P = picks[L]; nb = P.shape[0] // 4
        for b in range(nb):
            prod_union.append(len(set(P[b * 4:(b + 1) * 4].ravel().tolist())))
    a3["production_geometry_widthshard"] = {
        "mechanism": ("Verified in src/exo/worker/engines/mlx/auto_parallel.py:1164-1175 and "
                      "bench/section108_tp_expert_locality_analysis.md: DSv4.1 TP shards each "
                      "expert's weight matrix along the INTERMEDIATE-WIDTH axis (gate/up on "
                      "ndim-2, down on -1), NOT along the num_experts axis. Both ranks hold "
                      "ALL 384 experts at half width; there is no expert-ownership."),
        "consequence": ("per-rank distinct-experts-touched is IDENTICAL on both ranks for every "
                         "token (both compute every routed expert), so cross-rank routing-skew "
                         "imbalance under the REAL architecture = 1.0 exactly, for all batches."),
        "rank0_touched_equals_rank1_equals_batch_union": True,
        "R4_batch_union_mean": float(np.mean(prod_union)),
        "R4_imbalance_ratio_under_real_arch": 1.0,
        "verdict": ("The 192/192 expert-id split below is a COUNTERFACTUAL for a hypothetical "
                    "future expert-parallel (EP) scheme, NOT the production partition. Under "
                    "production width-sharding, no routing skew can create rank imbalance."),
    }
    result["analysis3_rank_imbalance"] = a3

    # ------------- Analysis 4 -------------
    a4 = {}
    R = 4
    per_layer_j = []
    all_j = []
    overall_union_per_batch = []
    for L in layers:
        P = picks[L]
        nb = P.shape[0] // R
        sets = [set(P[b * R:(b + 1) * R].ravel().tolist()) for b in range(nb)]
        js = [jaccard(sets[b], sets[b + 1]) for b in range(nb - 1)]
        per_layer_j.append(float(np.median(js)))
        all_j.extend(js)
        overall_union_per_batch.append(float(np.mean([len(s) for s in sets])))
    all_j = np.array(all_j)
    a4["R4_batch_overlap"] = {
        "n_batches_per_layer": int(picks[layers[0]].shape[0] // R),
        "union_per_batch_mean_unique": float(np.mean(overall_union_per_batch)),
        "consecutive_batch_jaccard": quantiles(all_j),
        "per_layer_median_jaccard": {
            "mean": float(np.mean(per_layer_j)), "median": float(np.median(per_layer_j)),
            "min": float(np.min(per_layer_j)), "max": float(np.max(per_layer_j)),
        },
        "trend_late_minus_early": None,
        "note": ("batch = 4 consecutive tokens; set = unique experts in the batch; "
                 "Jaccard between consecutive batches' sets, per layer, pooled here."),
    }
    # trend: early half vs late half of batches (per layer median, then mean)
    early, late = [], []
    for L in layers:
        P = picks[L]; nb = P.shape[0] // R
        sets = [set(P[b * R:(b + 1) * R].ravel().tolist()) for b in range(nb)]
        js = np.array([jaccard(sets[b], sets[b + 1]) for b in range(nb - 1)])
        h = len(js) // 2
        early.append(float(np.median(js[:h]))); late.append(float(np.median(js[h:])))
    a4["R4_batch_overlap"]["trend_late_minus_early"] = {
        "early_half_mean": float(np.mean(early)), "late_half_mean": float(np.mean(late)),
        "delta": float(np.mean(late) - np.mean(early))}
    result["analysis4_batch_overlap"] = a4

    # write json
    json_path = os.path.join(OUTDIR, "c_trace_mining.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=1)
    return result, json_path

if __name__ == "__main__":
    res, jp = main()
    # tiny stdout digest
    a1 = res["analysis1_union_and_rank_curve"]
    a2 = res["analysis2_hotset_and_stability"]
    a3 = res["analysis3_rank_imbalance"]
    a4 = res["analysis4_batch_overlap"]
    print("WROTE", jp)
    print("union per-layer min/med/max:", a1["union_sizes_summary"])
    print("overall distinct:", a1["overall"]["distinct_experts"], "of", E)
    print("coverage_k_summary:", json.dumps(a1["coverage_k_summary"]))
    print("topk_share:", json.dumps(a2["topk_share"]))
    print("stab top24 2win:", json.dumps(a2["stability"]["jaccard_top24"]["two_windows_265_266"]))
    print("stab top24 3win w0w1:", json.dumps(a2["stability"]["jaccard_top24"]["three_windows"]["w0_w1"]))
    print("stab top24 parity:", json.dumps(a2["stability"]["jaccard_top24"]["parity_even_odd"]))
    for R in ("R4", "R1"):
        print(f"{R} rank0/rank1 touched mean:",
              a3[R]["mean_rank0_touched"], a3[R]["mean_rank1_touched"],
              "mx/mean:", json.dumps(a3[R]["max_rank_touched_ratio_to_mean"]))
    print("R4 batch jaccard:", json.dumps(a4["R4_batch_overlap"]["consecutive_batch_jaccard"]))
