#!/usr/bin/env python3
"""Task 0 regression: decompose the ~0.55s unexplained per-request residual.

Pure-python (statistics + itertools only), runs under /usr/bin/python3, no
external deps. Reads the real-usage capture dataset and recomputes the
residual identity from raw fields (does NOT trust any stored residual
column), then runs the pre-registered regressions from
tmp/perf-campaign-2/round11/PREDICTION.md §1.

Usage: /usr/bin/python3 tmp/perf-campaign-2/round11/task0_regression.py
Run from repo root (paths below are relative to repo root).
"""

from __future__ import annotations

import json
import statistics
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
REQUESTS_PATH = REPO_ROOT / "tmp/real-usage-capture-20260902/phase1/requests.jsonl"
PARTITION_PATH = REPO_ROOT / "tmp/real-usage-capture-20260902/partition_verified.json"
OUT_JSON = REPO_ROOT / "tmp/perf-campaign-2/round11/task0_regression.json"

TS_FMT = "%Y-%m-%d %H:%M:%S.%f"


def unwrap(rec: dict) -> dict:
    """Unwrap {field: {value, provenance}} -> {field: value}."""
    return {k: v["value"] for k, v in rec.items()}


def parse_ts(s: str) -> float:
    """Parse 'YYYY-MM-DD HH:MM:SS.mmm' to epoch seconds (float, ms precision)."""
    return datetime.strptime(s, TS_FMT).timestamp()


def load_rows() -> list[dict]:
    rows = []
    with open(REQUESTS_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(unwrap(json.loads(line)))
    return rows


def is_main_row(r: dict) -> bool:
    """The 55 main rows: exclude the 2 aux rows (prompt 18/17, null client fields)."""
    return r.get("wall_seconds_client") is not None and r.get("client_started_ts") is not None


def compute_residual(r: dict) -> dict:
    """Recompute the identity: wall = prefill_uncached + decode + residual."""
    prompt_tokens = r["prompt_tokens"]
    cached_tokens = r["cached_tokens"]
    completion_tokens = r["completion_tokens"]
    prompt_tps = r["prompt_tps"]
    generation_tps = r["generation_tps"]
    wall = r["wall_seconds_client"]

    prefill_uncached = (prompt_tokens - cached_tokens) / prompt_tps
    decode = completion_tokens / generation_tps
    residual = wall - prefill_uncached - decode

    server_ts = parse_ts(r["server_received_ts"])
    client_ts = parse_ts(r["client_started_ts"])
    transit = server_ts - client_ts
    residual_ex_transit = residual - transit

    return {
        "prefill_uncached": prefill_uncached,
        "decode": decode,
        "residual": residual,
        "transit": transit,
        "residual_ex_transit": residual_ex_transit,
    }


# ---------------------------------------------------------------------------
# Regression primitives (pure python)
# ---------------------------------------------------------------------------


def ols(xs: list[float], ys: list[float]) -> dict:
    """Simple OLS y = a + b*x. Returns slope, intercept, r2, n, se_slope, ci95."""
    n = len(xs)
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    syy = sum((y - mean_y) ** 2 for y in ys)

    slope = sxy / sxx
    intercept = mean_y - slope * mean_x

    # residual sum of squares
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = syy
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # standard error of slope
    dof = n - 2
    if dof > 0 and sxx > 0:
        sigma2 = ss_res / dof
        se_slope = (sigma2 / sxx) ** 0.5
        # t critical value for 95% CI, dof degrees of freedom
        t_crit = t_critical_95(dof)
        ci_half = t_crit * se_slope
    else:
        se_slope = float("nan")
        ci_half = float("nan")

    return {
        "n": n,
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "se_slope": se_slope,
        "ci95_low": slope - ci_half,
        "ci95_high": slope + ci_half,
    }


def t_critical_95(dof: int) -> float:
    """Approximate two-sided 95% t critical value.

    Uses a small lookup table for common dof ranges (our n is always in the
    30-55 range so dof is 28-53), falling back to the normal approximation
    (1.96) for large dof. This avoids a scipy dependency.
    """
    table = {
        20: 2.086, 25: 2.060, 28: 2.048, 29: 2.045, 30: 2.042,
        35: 2.030, 40: 2.021, 45: 2.014, 50: 2.009, 53: 2.006,
        55: 2.004, 60: 2.000, 80: 1.990, 100: 1.984,
    }
    if dof >= 100:
        return 1.984
    # nearest key <= dof, else nearest above
    keys = sorted(table.keys())
    best = keys[-1]
    for k in keys:
        if k >= dof:
            best = k
            break
    return table[best]


def pearson_r(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    sxx = sum((x - mean_x) ** 2 for x in xs)
    syy = sum((y - mean_y) ** 2 for y in ys)
    return sxy / ((sxx * syy) ** 0.5)


def theil_sen(xs: list[float], ys: list[float]) -> dict:
    """Median-of-pairwise-slopes estimator (robust to leverage outliers)."""
    n = len(xs)
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[j] - xs[i]
            if dx == 0:
                continue
            slopes.append((ys[j] - ys[i]) / dx)
    slope = statistics.median(slopes)
    # intercept: median of (y_i - slope * x_i)
    intercepts = [y - slope * x for x, y in zip(xs, ys)]
    intercept = statistics.median(intercepts)
    return {"slope": slope, "intercept": intercept, "n_pairs": len(slopes)}


def describe(vals: list[float]) -> dict:
    return {
        "median": statistics.median(vals),
        "min": min(vals),
        "max": max(vals),
        "mean": statistics.mean(vals),
        "n": len(vals),
    }


def main() -> None:
    rows = load_rows()
    main_rows = [r for r in rows if is_main_row(r)]

    print(f"Loaded {len(rows)} total records, {len(main_rows)} main rows (client fields present).")

    # ---- validation against partition_verified.json ----
    with open(PARTITION_PATH) as f:
        partition = json.load(f)

    computed = [compute_residual(r) for r in main_rows]
    residuals = [c["residual"] for c in computed]
    res_stats = describe(residuals)

    pub = partition["residual_per_request_s"]
    tol = 0.01
    validation = {
        "recomputed_median": res_stats["median"],
        "recomputed_min": res_stats["min"],
        "recomputed_max": res_stats["max"],
        "published_median": pub["median"],
        "published_min": pub["min"],
        "published_max": pub["max"],
        "median_diff": abs(res_stats["median"] - pub["median"]),
        "min_diff": abs(res_stats["min"] - pub["min"]),
        "max_diff": abs(res_stats["max"] - pub["max"]),
        "pass": (
            abs(res_stats["median"] - pub["median"]) <= tol
            and abs(res_stats["min"] - pub["min"]) <= tol
            and abs(res_stats["max"] - pub["max"]) <= tol
        ),
    }
    print("Validation vs partition_verified.json:", json.dumps(validation, indent=2))
    if not validation["pass"]:
        print("STOP: recomputed residual does not match published numbers within tolerance.")
        raise SystemExit(1)
    print(f"n main rows = {len(main_rows)} (expect 55)")
    assert len(main_rows) == 55, f"expected 55 main rows, got {len(main_rows)}"

    # ---- exclusions bookkeeping (E) ----
    none_hit_rows = [r for r in main_rows if r.get("prefix_cache_hit") == "none"]
    print(f"Rows with prefix_cache_hit == 'none' among 55 main rows: {len(none_hit_rows)}")
    for r in none_hit_rows:
        print(
            f"  task_id={r['task_id']} prompt_tokens={r['prompt_tokens']} "
            f"cached_tokens={r['cached_tokens']} prefix_cache_hit={r['prefix_cache_hit']}"
        )

    # The pre-registration's single "cold request" leverage outlier is defined
    # as prompt_tokens 92594 / cached_tokens 0 / ~222s prefill.
    cold_row = None
    for r, c in zip(main_rows, computed):
        if r["prompt_tokens"] == 92594 and r["cached_tokens"] == 0:
            cold_row = (r, c)
    assert cold_row is not None, "could not find the pre-registered cold outlier row"
    print(
        f"Cold outlier: task_id={cold_row[0]['task_id']} prompt_tokens={cold_row[0]['prompt_tokens']} "
        f"prefill_uncached={cold_row[1]['prefill_uncached']:.1f}s"
    )

    # ---- fit data ----
    prompt_tokens = [r["prompt_tokens"] for r in main_rows]
    cached_tokens = [r["cached_tokens"] for r in main_rows]
    residual_vals = [c["residual"] for c in computed]
    transit_vals = [c["transit"] for c in computed]
    residual_ex_transit_vals = [c["residual_ex_transit"] for c in computed]

    cold_idx = None
    for i, r in enumerate(main_rows):
        if r["prompt_tokens"] == 92594 and r["cached_tokens"] == 0:
            cold_idx = i
    assert cold_idx is not None

    def drop_cold(lst):
        return [v for i, v in enumerate(lst) if i != cold_idx]

    # (A) primary fit: residual_ex_transit ~ prompt_tokens, WITH and WITHOUT cold outlier
    fit_A_with_cold = ols(prompt_tokens, residual_ex_transit_vals)
    fit_A_without_cold = ols(drop_cold(prompt_tokens), drop_cold(residual_ex_transit_vals))

    # convert slope units: residual_ex_transit is in seconds, prompt_tokens in tokens
    # slope [s/token] * 1e6 = [us/token]
    def to_us_per_tok(fit):
        return {
            **fit,
            "slope_us_per_tok": fit["slope"] * 1e6,
            "ci95_low_us_per_tok": fit["ci95_low"] * 1e6,
            "ci95_high_us_per_tok": fit["ci95_high"] * 1e6,
        }

    fit_A_with_cold = to_us_per_tok(fit_A_with_cold)
    fit_A_without_cold = to_us_per_tok(fit_A_without_cold)

    # (B) secondary fit: raw residual ~ prompt_tokens (without cold outlier, matching A's primary)
    fit_B_with_cold = to_us_per_tok(ols(prompt_tokens, residual_vals))
    fit_B_without_cold = to_us_per_tok(ols(drop_cold(prompt_tokens), drop_cold(residual_vals)))

    # (C) raw residual ~ cached_tokens
    fit_C_with_cold = to_us_per_tok(ols(cached_tokens, residual_vals))
    fit_C_without_cold = to_us_per_tok(ols(drop_cold(cached_tokens), drop_cold(residual_vals)))

    # (D) collinearity
    collinearity_with_cold = pearson_r(prompt_tokens, cached_tokens)
    collinearity_without_cold = pearson_r(drop_cold(prompt_tokens), drop_cold(cached_tokens))

    # (G) Theil-Sen robustness check for fit (A), with and without cold outlier
    ts_A_with_cold = theil_sen(prompt_tokens, residual_ex_transit_vals)
    ts_A_with_cold["slope_us_per_tok"] = ts_A_with_cold["slope"] * 1e6
    ts_A_without_cold = theil_sen(drop_cold(prompt_tokens), drop_cold(residual_ex_transit_vals))
    ts_A_without_cold["slope_us_per_tok"] = ts_A_without_cold["slope"] * 1e6

    # (F) descriptive stats
    descriptive = {
        "residual_s": describe(residual_vals),
        "transit_s": describe(transit_vals),
        "residual_ex_transit_s": describe(residual_ex_transit_vals),
        "prompt_tokens": describe(prompt_tokens),
    }

    # ---- verdict logic ----
    def verdict_for_slope(slope_us_per_tok: float) -> str:
        if 1.0 <= slope_us_per_tok <= 2.0:
            return "O(context) work dominates (tokenization/trie/restore)"
        elif abs(slope_us_per_tok) < 0.3:
            return "flat -> IPC/polling ticks dominate"
        else:
            return "MIXED -- naive fit cannot rank hypotheses"

    verdict_A_without_cold = verdict_for_slope(fit_A_without_cold["slope_us_per_tok"])
    verdict_A_with_cold = verdict_for_slope(fit_A_with_cold["slope_us_per_tok"])

    # ---- pre-registered prediction scoring ----
    # predictions (from PREDICTION.md): slope 0.5-2.0 us/tok (fit B, secondary/raw),
    # r2 of raw fit < 0.35, intercept 0.4-0.8s, r2 of ex-transit fit (A) > 0.4
    # We score against the WITHOUT-cold-outlier fits since that is the fit the
    # pre-registration treats as primary/valid for ranking (cold row is leverage
    # outlier, excluded from primary interpretation).
    pred_slope_ok = 0.5 <= fit_B_without_cold["slope_us_per_tok"] <= 2.0
    pred_r2_raw_ok = fit_B_without_cold["r2"] < 0.35
    pred_intercept_ok = 0.4 <= fit_B_without_cold["intercept"] <= 0.8
    pred_r2_ex_transit_ok = fit_A_without_cold["r2"] > 0.4

    results = {
        "interpreter": "/usr/bin/python3",
        "n_total_records": len(rows),
        "n_main_rows": len(main_rows),
        "validation_vs_partition_verified": validation,
        "exclusions": {
            "aux_rows_excluded": len(rows) - len(main_rows),
            "prefix_cache_hit_none_rows_in_55": len(none_hit_rows),
            "prefix_cache_hit_none_task_ids": [r["task_id"] for r in none_hit_rows],
            "cold_outlier_task_id": cold_row[0]["task_id"],
            "cold_outlier_prompt_tokens": cold_row[0]["prompt_tokens"],
            "cold_outlier_cached_tokens": cold_row[0]["cached_tokens"],
            "cold_outlier_prefill_uncached_s": cold_row[1]["prefill_uncached"],
            "note": (
                "partition_verified.json reports prefix_cache={'partial':54,'none':3} "
                "across ALL captured records (57), not just the 55 main rows. "
                "Reconciliation: 2 of those 3 'none' rows are the aux rows (prompt "
                "18/17, excluded from the 55 main rows because client fields are "
                "null). Only 1 'none'-hit row remains among the 55 main rows, and "
                "it is exactly the pre-registered cold outlier (prompt 92594, "
                "cached 0). No further exclusion beyond the 2 aux rows + the "
                "single cold-outlier sensitivity check was needed."
            ),
        },
        "fit_A_residual_ex_transit_vs_prompt_tokens": {
            "with_cold_outlier": fit_A_with_cold,
            "without_cold_outlier": fit_A_without_cold,
        },
        "fit_B_residual_vs_prompt_tokens": {
            "with_cold_outlier": fit_B_with_cold,
            "without_cold_outlier": fit_B_without_cold,
        },
        "fit_C_residual_vs_cached_tokens": {
            "with_cold_outlier": fit_C_with_cold,
            "without_cold_outlier": fit_C_without_cold,
        },
        "collinearity_prompt_vs_cached_tokens": {
            "pearson_r_with_cold_outlier": collinearity_with_cold,
            "pearson_r_without_cold_outlier": collinearity_without_cold,
        },
        "theil_sen_fit_A": {
            "with_cold_outlier": ts_A_with_cold,
            "without_cold_outlier": ts_A_without_cold,
        },
        "descriptive_stats": descriptive,
        "verdict": {
            "fit_A_without_cold_outlier": verdict_A_without_cold,
            "fit_A_with_cold_outlier": verdict_A_with_cold,
        },
        "prediction_scoring": {
            "slope_0.5_to_2.0_us_per_tok": {
                "predicted": "0.5-2.0 us/token",
                "observed_us_per_tok": fit_B_without_cold["slope_us_per_tok"],
                "correct": pred_slope_ok,
            },
            "r2_raw_fit_lt_0.35": {
                "predicted": "< 0.35",
                "observed": fit_B_without_cold["r2"],
                "correct": pred_r2_raw_ok,
            },
            "intercept_0.4_to_0.8s": {
                "predicted": "0.4-0.8 s",
                "observed_s": fit_B_without_cold["intercept"],
                "correct": pred_intercept_ok,
            },
            "r2_ex_transit_fit_gt_0.4": {
                "predicted": "> 0.4",
                "observed": fit_A_without_cold["r2"],
                "correct": pred_r2_ex_transit_ok,
            },
        },
    }

    OUT_JSON.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {OUT_JSON}")
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
