#!/usr/bin/env python3
import json
import os
import math

MARGINS_DIR = "/Users/adam.durham/repos/exo/tmp/p05-lmhead-mxfp8-20260830/real_margins"
OUTPUT_DIR = MARGINS_DIR

KERNEL = {
    "band0": 0.406, # [0, 1.44)
    "band1": 0.125, # [1.44, 3.62)
    "band2": 0.0,   # [3.62, 7.5)
    "band3": 0.0    # [7.5, inf)
}

def get_band(m):
    if m < 1.44: return 0
    if m < 3.62: return 1
    if m < 7.5: return 2
    return 3

def analyze_margins(margins_list):
    if not margins_list:
        return None
    
    n = len(margins_list)
    sorted_m = sorted(margins_list)
    
    counts = [0] * 4
    for val in sorted_m:
        counts[get_band(val)] += 1
    
    fractions = [c / n for c in counts]
    
    # Flip rate estimate
    flip_rate = sum(f * k for f, k in zip(fractions, [KERNEL["band0"], KERNEL["band1"], KERNEL["band2"], KERNEL["band3"]]))
    
    def percentile(data, p):
        idx = (len(data) - 1) * p
        low = math.floor(idx)
        high = math.ceil(idx)
        if low == high:
            return data[int(idx)]
        return data[low] * (high - idx) + data[high] * (idx - low)

    return {
        "n": n,
        "counts": counts,
        "fractions": fractions,
        "mean": sum(sorted_m) / n,
        "median": percentile(sorted_m, 0.5),
        "p10": percentile(sorted_m, 0.1),
        "p90": percentile(sorted_m, 0.9),
        "frac_below_3_62": sum(fractions[:2]),
        "implied_flip_rate": flip_rate
    }

def main():
    results = {}
    all_margins = []
    
    files = [f for f in os.listdir(MARGINS_DIR) if f.startswith("margins_") and f.endswith(".json")]
    
    for f_name in files:
        label = f_name.replace("margins_", "").replace(".json", "")
        with open(os.path.join(MARGINS_DIR, f_name), "r") as f:
            data = json.load(f)
            margins = [p["margin"] for p in data["positions"]]
            results[label] = analyze_margins(margins)
            all_margins.extend(margins)
    
    results["pooled"] = analyze_margins(all_margins)
    
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    md = "# Margin Analysis Report\n\n"
    md += "## Summary Table\n\n"
    md += "| Context | n | Mean | Median | p10 | p90 | Frac < 3.62 | Flip Rate |\n"
    md += "|---|---|---|---|---|---|---|---|\n"
    
    for label in sorted(results.keys()):
        if label == "pooled": continue
        r = results[label]
        md += f"| {label} | {r['n']} | {r['mean']:.3f} | {r['median']:.3f} | {r['p10']:.3f} | {r['p90']:.3f} | {r['frac_below_3_62']:.3%} | {r['implied_flip_rate']:.3%} |\n"
    
    p = results["pooled"]
    md += f"| **Pooled** | **{p['n']}** | **{p['mean']:.3f}** | **{p['median']:.3f}** | **{p['p10']:.3f}** | **{p['p90']:.3f}** | **{p['frac_below_3_62']:.3%}** | **{p['implied_flip_rate']:.3%}** |\n\n"
    
    md += "## Verdict\n\n"
    md += f"- **Measured Frac < 3.62:** {p['frac_below_3_62']:.3%} (Asserted: ~58%)\n"
    md += f"- **Implied Flip Rate:** {p['implied_flip_rate']:.3%} (Asserted: ~16%)\n\n"
    
    frac_ok = abs(p['frac_below_3_62'] - 0.58) < 0.10
    flip_ok = abs(p['implied_flip_rate'] - 0.16) < 0.05
    
    verdict = "CONFIRMED" if (frac_ok and flip_ok) else "REFUTED"
    md += f"**Verdict: {verdict}**\n\n"
    md += "*Note: Flip rate is an estimate via synthetic-band kernel, not a direct measurement.*\n"
    
    with open(os.path.join(OUTPUT_DIR, "ANALYSIS.md"), "w") as f:
        f.write(md)

    print(f"Pooled Frac < 3.62: {p['frac_below_3_62']:.3%}")
    print(f"Pooled Flip Rate: {p['implied_flip_rate']:.3%}")
    print(f"Verdict: {verdict}")

if __name__ == "__main__":
    main()
