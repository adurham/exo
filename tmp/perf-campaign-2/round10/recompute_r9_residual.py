#!/usr/bin/env python3
"""Round-10 R9-only residual recompute (zero cluster cost).

Reads round9/results/<LABEL>_{short,2k}_r*.json and recomputes the governing
statistic fixed in round10/PRE-REGISTRATION.md section 1:

    residual_ms = prefill_s*1000 - ((prompt_tokens - 1) / prompt_tps) * 1000

where prompt_tps = server_stats.prompt_tps. Formula/logic mirrors
round9/summarize.py exactly (read-only reuse, not modified).

NO outlier exclusion at any stage - every rep present is used.

Outputs:
  - tmp/perf-campaign-2/round10/r9_residual_recompute.json  (structured)
  - tmp/perf-campaign-2/round10/R9-RESIDUAL-RECOMPUTE.md    (readable report)
"""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # tmp/perf-campaign-2
RES = ROOT / "round9" / "results"
OUT_DIR = Path(__file__).resolve().parent  # tmp/perf-campaign-2/round10

LABELS = ["A", "Z1", "B", "Z2"]
RV200_LABELS = ["A", "B"]
RV0_LABELS = ["Z1", "Z2"]
KINDS = [("short", 10), ("2k", 5)]

# Acceptance oracle values from R9 REPORT.md section 2.2
ORACLE_SHORT = {"A": 686, "Z1": 485, "B": 634, "Z2": 469}
ORACLE_2K = {"A": 697, "Z1": 431, "B": 675, "Z2": 400}


def load(label, kind, n):
    rows = []
    for i in range(1, n + 1):
        p = RES / f"{label}_{kind}_r{i}.json"
        if not p.exists():
            rows.append({"file": p.name, "MISSING": True})
            continue
        d = json.loads(p.read_text())
        ss = d.get("server_stats") or {}
        ptok = d.get("prompt_tokens")
        ptps = ss.get("prompt_tps")
        ttft = d["prefill_s"] * 1000.0
        resid = None
        if ptok and ptps:
            resid = ttft - ((ptok - 1) / ptps) * 1000.0
        rows.append({
            "file": p.name,
            "tag": d.get("tag"),
            "ttft_ms": round(ttft, 3),
            "prompt_tokens": ptok,
            "prompt_tps": ptps,
            "resid_ms": round(resid, 3) if resid is not None else None,
            "prefix_cache_hit": ss.get("prefix_cache_hit", "MISSING"),
            "completion_tokens": d.get("completion_tokens"),
        })
    return rows


def summarize(label, kind, n):
    rows = load(label, kind, n)
    present = [r for r in rows if not r.get("MISSING")]
    missing = [r["file"] for r in rows if r.get("MISSING")]
    t = [r["ttft_ms"] for r in present]
    pt = [r["prompt_tokens"] for r in present if r["prompt_tokens"]]
    rs = [r["resid_ms"] for r in present if r["resid_ms"] is not None]
    tps = [r["prompt_tps"] for r in present if r["prompt_tps"]]
    hits = sorted({str(r["prefix_cache_hit"]) for r in present})
    return {
        "label": label,
        "kind": kind,
        "n": len(present),
        "missing_files": missing,
        "resid_median": round(statistics.median(rs), 1) if rs else None,
        "resid_range": [min(rs), max(rs)] if rs else None,
        "resid_all": rs,
        "ttft_median": round(statistics.median(t), 1) if t else None,
        "ttft_range": [min(t), max(t)] if t else None,
        "ptok_range": [min(pt), max(pt)] if pt else None,
        "prompt_tps_median": round(statistics.median(tps), 2) if tps else None,
        "cache_hits": hits,
        "rows": present,
    }


def main():
    out = {}
    for lb in LABELS:
        for kind, n in KINDS:
            out[f"{lb}_{kind}"] = summarize(lb, kind, n)

    # ---- oracle check ----
    oracle_report = {"all_matched": True, "mismatches": []}
    for lb, expected in ORACLE_SHORT.items():
        got = out[f"{lb}_short"]["resid_median"]
        diff = None if got is None else abs(got - expected)
        ok = diff is not None and diff <= 1.0
        if not ok:
            oracle_report["all_matched"] = False
            oracle_report["mismatches"].append({
                "label": lb, "kind": "short", "expected": expected, "got": got, "diff": diff,
            })
    for lb, expected in ORACLE_2K.items():
        got = out[f"{lb}_2k"]["resid_median"]
        diff = None if got is None else abs(got - expected)
        ok = diff is not None and diff <= 1.0
        if not ok:
            oracle_report["all_matched"] = False
            oracle_report["mismatches"].append({
                "label": lb, "kind": "2k", "expected": expected, "got": got, "diff": diff,
            })

    # ---- prefix_cache_hit audit across all 60 reps ----
    cache_audit = {"all_none": True, "violations": []}
    for lb in LABELS:
        for kind, n in KINDS:
            for r in out[f"{lb}_{kind}"]["rows"]:
                if str(r["prefix_cache_hit"]).lower() != "none":
                    cache_audit["all_none"] = False
                    cache_audit["violations"].append({
                        "file": r["file"], "prefix_cache_hit": r["prefix_cache_hit"],
                    })

    # ---- STEP 3: SHORT instrument (governing) C1/C2 ----
    def build_c1c2(kind):
        short_medians = {lb: out[f"{lb}_{kind}"]["resid_median"] for lb in LABELS}
        rv200_medians = [short_medians[lb] for lb in RV200_LABELS]
        rv0_medians = [short_medians[lb] for lb in RV0_LABELS]
        spread_rv200 = max(rv200_medians) - min(rv200_medians)

        c1_lhs = min(rv200_medians) - max(rv0_medians)
        c1_rhs = spread_rv200
        c1_pass = c1_lhs > c1_rhs

        pooled_rv200 = []
        pooled_rv0 = []
        for lb in RV200_LABELS:
            pooled_rv200.extend(out[f"{lb}_{kind}"]["resid_all"])
        for lb in RV0_LABELS:
            pooled_rv0.extend(out[f"{lb}_{kind}"]["resid_all"])
        pooled_rv200_median = statistics.median(pooled_rv200)
        pooled_rv0_median = statistics.median(pooled_rv0)
        signed_gap = pooled_rv200_median - pooled_rv0_median  # expect positive: RV0 lower
        pooled_gap_mag = abs(signed_gap)
        sign_direction = "RV=0 LOWER" if signed_gap > 0 else ("RV=0 HIGHER" if signed_gap < 0 else "EQUAL")
        c2_in_band = 150.0 <= pooled_gap_mag <= 250.0
        c2_pass = c2_in_band and sign_direction == "RV=0 LOWER"

        return {
            "kind": kind,
            "boot_medians": short_medians,
            "rv200_medians": {lb: short_medians[lb] for lb in RV200_LABELS},
            "rv0_medians": {lb: short_medians[lb] for lb in RV0_LABELS},
            "spread_rv200": round(spread_rv200, 1),
            "c1": {
                "lhs_min_rv200_minus_max_rv0": round(c1_lhs, 1),
                "rhs_spread_rv200": round(c1_rhs, 1),
                "condition": "min(RV200 medians) - max(RV0 medians) > spread(RV200)",
                "result": "PASS" if c1_pass else "FAIL",
            },
            "c2": {
                "pooled_rv200_median": round(pooled_rv200_median, 1),
                "pooled_rv0_median": round(pooled_rv0_median, 1),
                "pooled_gap_magnitude": round(pooled_gap_mag, 1),
                "sign_direction": sign_direction,
                "n_rv200_pooled": len(pooled_rv200),
                "n_rv0_pooled": len(pooled_rv0),
                "in_band_150_250": c2_in_band,
                "result": "PASS" if c2_pass else "FAIL",
            },
        }

    short_c1c2 = build_c1c2("short")
    k2_c1c2 = build_c1c2("2k")

    full_out = {
        "summaries": out,
        "oracle_check": oracle_report,
        "prefix_cache_hit_audit": cache_audit,
        "short_c1c2_governing": short_c1c2,
        "2k_c1c2_secondary_diagnostic": k2_c1c2,
    }

    (OUT_DIR / "r9_residual_recompute.json").write_text(json.dumps(full_out, indent=2))

    write_markdown(full_out)
    print(json.dumps({
        "oracle_all_matched": oracle_report["all_matched"],
        "cache_all_none": cache_audit["all_none"],
        "short_medians": short_c1c2["boot_medians"],
        "short_spread_rv200": short_c1c2["spread_rv200"],
        "short_c1": short_c1c2["c1"],
        "short_c2": short_c1c2["c2"],
        "2k_medians": k2_c1c2["boot_medians"],
        "2k_spread_rv200": k2_c1c2["spread_rv200"],
        "2k_c1": k2_c1c2["c1"],
        "2k_c2": k2_c1c2["c2"],
    }, indent=2))


def write_markdown(full_out):
    out = full_out["summaries"]
    oracle = full_out["oracle_check"]
    cache_audit = full_out["prefix_cache_hit_audit"]
    short_c1c2 = full_out["short_c1c2_governing"]
    k2_c1c2 = full_out["2k_c1c2_secondary_diagnostic"]

    lines = []
    lines.append("# R9-Only Residual Recompute (Round 10, zero cluster cost)")
    lines.append("")
    lines.append("Recomputes the round-10 governing statistic (the RESIDUAL, PRE-REGISTRATION.md ")
    lines.append("section 1) from round 9's 60 raw rep JSONs. This is the 'R9-only recompute, ")
    lines.append("4 boots' sub-analysis required by PRE-REGISTRATION section 4.1. NO outlier ")
    lines.append("exclusion at any stage.")
    lines.append("")
    lines.append("```")
    lines.append("residual_ms = prefill_s*1000 - ((prompt_tokens - 1) / prompt_tps) * 1000")
    lines.append("```")
    lines.append("")

    lines.append("## Acceptance oracle check (vs R9 REPORT.md section 2.2)")
    lines.append("")
    if oracle["all_matched"]:
        lines.append("**ALL 8 published residual medians MATCHED to within ±1 ms.**")
    else:
        lines.append("**MISMATCH DETECTED — DO NOT TRUST DOWNSTREAM NUMBERS WITHOUT INVESTIGATING:**")
        for m in oracle["mismatches"]:
            lines.append(f"- {m['label']} {m['kind']}: expected {m['expected']}, got {m['got']} (diff {m['diff']})")
    lines.append("")

    lines.append("## prefix_cache_hit audit (all 60 reps)")
    lines.append("")
    if cache_audit["all_none"]:
        lines.append("**All 60 reps have prefix_cache_hit == none.**")
    else:
        lines.append("**VIOLATIONS FOUND:**")
        for v in cache_audit["violations"]:
            lines.append(f"- {v['file']}: prefix_cache_hit = {v['prefix_cache_hit']}")
    lines.append("")

    lines.append("## Per-boot / per-instrument summary")
    lines.append("")
    lines.append("| Label | Kind | n | Resid median (ms) | Resid range (ms) | TTFT median (ms) | TTFT range (ms) | ptok range | prompt_tps median | cache_hits |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for lb in LABELS:
        for kind, _n in KINDS:
            s = out[f"{lb}_{kind}"]
            lines.append(
                f"| {lb} | {kind} | {s['n']} | {s['resid_median']} | "
                f"[{s['resid_range'][0]:.1f}, {s['resid_range'][1]:.1f}] | "
                f"{s['ttft_median']} | [{s['ttft_range'][0]:.1f}, {s['ttft_range'][1]:.1f}] | "
                f"{s['ptok_range']} | {s['prompt_tps_median']} | {', '.join(s['cache_hits'])} |"
            )
    lines.append("")

    lines.append("## Per-rep residual values (auditable)")
    lines.append("")
    for lb in LABELS:
        for kind, _n in KINDS:
            s = out[f"{lb}_{kind}"]
            lines.append(f"### {lb} {kind} (n={s['n']})")
            lines.append("")
            lines.append("| file | resid_ms | ttft_ms | prompt_tokens | prompt_tps | prefix_cache_hit |")
            lines.append("|---|---|---|---|---|---|")
            for r in s["rows"]:
                lines.append(
                    f"| {r['file']} | {r['resid_ms']} | {r['ttft_ms']} | {r['prompt_tokens']} | "
                    f"{r['prompt_tps']} | {r['prefix_cache_hit']} |"
                )
            lines.append("")
            lines.append(f"resid_all = {s['resid_all']}")
            lines.append("")

    lines.append("## SHORT instrument (GOVERNING) — C1 / C2")
    lines.append("")
    lines.append(f"Boot short residual medians: A={short_c1c2['boot_medians']['A']}, "
                  f"Z1={short_c1c2['boot_medians']['Z1']}, B={short_c1c2['boot_medians']['B']}, "
                  f"Z2={short_c1c2['boot_medians']['Z2']}")
    lines.append("")
    lines.append(f"spread(RV200) = max - min across A,B short residual medians = **{short_c1c2['spread_rv200']} ms**")
    lines.append("")
    c1 = short_c1c2["c1"]
    lines.append(f"**C1**: {c1['condition']}")
    lines.append(f"- LHS = min(RV200 medians) - max(RV0 medians) = {c1['lhs_min_rv200_minus_max_rv0']} ms")
    lines.append(f"- RHS = spread(RV200) = {c1['rhs_spread_rv200']} ms")
    lines.append(f"- **C1 result: {c1['result']}**")
    lines.append("")
    c2 = short_c1c2["c2"]
    lines.append(f"**C2**: pooled short residual gap = median(all RV200 reps, n={c2['n_rv200_pooled']}) - "
                  f"median(all RV0 reps, n={c2['n_rv0_pooled']})")
    lines.append(f"- pooled RV200 median = {c2['pooled_rv200_median']} ms")
    lines.append(f"- pooled RV0 median = {c2['pooled_rv0_median']} ms")
    lines.append(f"- pooled gap magnitude = **{c2['pooled_gap_magnitude']} ms**, sign direction = {c2['sign_direction']}")
    lines.append(f"- in [150, 250] band: {c2['in_band_150_250']}")
    lines.append(f"- **C2 result: {c2['result']}**")
    lines.append("")

    lines.append("## 2K instrument (SECONDARY DIAGNOSTIC — non-governing) — C1 / C2")
    lines.append("")
    lines.append(f"Boot 2K residual medians: A={k2_c1c2['boot_medians']['A']}, "
                  f"Z1={k2_c1c2['boot_medians']['Z1']}, B={k2_c1c2['boot_medians']['B']}, "
                  f"Z2={k2_c1c2['boot_medians']['Z2']}")
    lines.append("")
    lines.append(f"spread(RV200)_2k = **{k2_c1c2['spread_rv200']} ms**")
    lines.append("")
    c1k = k2_c1c2["c1"]
    lines.append(f"**C1 (2K, non-governing)**: LHS = {c1k['lhs_min_rv200_minus_max_rv0']} ms, "
                  f"RHS = {c1k['rhs_spread_rv200']} ms -> **{c1k['result']}**")
    lines.append("")
    c2k = k2_c1c2["c2"]
    lines.append(f"**C2 (2K, non-governing)**: pooled gap magnitude = {c2k['pooled_gap_magnitude']} ms, "
                  f"sign = {c2k['sign_direction']}, in [150,250] band: {c2k['in_band_150_250']} -> "
                  f"**{c2k['result']}**")
    lines.append("")

    lines.append("---")
    lines.append("*This report covers R9-only, 4 boots. It is one of three required breakdowns ")
    lines.append("per PRE-REGISTRATION section 4.1 (the other two — fresh-pair-only 2 boots, and ")
    lines.append("the full 6-boot set — require round-10 fresh data and are out of scope here).*")

    (OUT_DIR / "R9-RESIDUAL-RECOMPUTE.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
