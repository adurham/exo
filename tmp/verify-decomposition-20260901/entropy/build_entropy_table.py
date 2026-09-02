#!/usr/bin/env python3
"""Build the 4-arm entropy A/B table from probe JSON + [MTP-PROF] harvests.

Reads artifacts produced by run_entropy_ab.sh under entropy/raw/ and emits
entropy/raw/entropy_table.json plus a readable markdown table on stdout.

Anchoring rule (critical): the profiler's `n` is cumulative over the runner's
LIFETIME, not the measurement window. The first dump inside the harvest window
carries ~N pre-window idle cycles at their cumulative mean. We therefore anchor
on the LAST dump STRICTLY BEFORE the window (from prof_anchor_<tag>_n{1,2}.txt)
so only in-window cycles are counted. This is the same arithmetic that
reproduces the V3 published figure (56.087 ms verify / 650 cycles).

rb_pool_restores is a COUNT mislabeled with an 'ms' suffix by a unit-blind
formatter -- analyze_prof.py tags it unit='count'; we respect that and never
report it as a time.
"""
from __future__ import annotations
import json, os, sys, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)                      # verify-decomposition-20260901
RAW = os.path.join(HERE, "raw")                  # entropy/raw
sys.path.insert(0, OUT)                          # parent dir -> analyze_prof importable
from analyze_prof import analyze, anchors_from   # noqa: E402

# The four arms, in run order. `mode` is what the probe was invoked with;
# `tag` is the artifact tag (repetitive_recheck re-runs mode 'repetitive').
ARMS = [
    {"tag": "repetitive",         "mode": "repetitive", "words": 75000},
    {"tag": "natural",            "mode": "natural",    "words": 65646},
    {"tag": "random",             "mode": "random",     "words": 23525},
    {"tag": "repetitive_recheck", "mode": "repetitive", "words": 75000},
]

# Phases that constitute the MTP cycle (verify/draft/accept/rollback family).
# Confirmed against real profiler output (raw/prof_089k_n1.txt and V3 run1):
#   draft verify accept rollback total rb_drain rb_gate rb_pool
#   rb_pool_restores rb_ring rb_snap rb_tail
# The cycle = verify + draft + accept + rollback. `total` is the profiler's own
# sum of those four; we recompute it from the four to be explicit.
CYCLE_PHASES = ["verify", "draft", "accept", "rollback"]

G4_THRESHOLD = 0.5   # % max spread of achieved_prompt_tokens across arms
G2_THRESHOLD = 10.0 # % relative drift between repetitive and repetitive_recheck


def load_probe(tag: str):
    """Return the probe summary dict, or None if the arm is missing/failed."""
    path = os.path.join(RAW, f"entropy_{tag}.json")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    try:
        d = json.load(open(path))
    except Exception:
        return None
    s = d.get("summary")
    if not s:
        return None
    return {
        "mode": d.get("config", {}).get("mode"),
        "achieved_prompt_tokens": d.get("achieved_prompt_tokens"),
        "decode_tps_mean": s.get("generation_tps_mean"),
        "decode_tps_median": s.get("generation_tps_median"),
        "decode_tps_min": s.get("generation_tps_min"),
        "decode_tps_max": s.get("generation_tps_max"),
        "prefill_tps_mean": s.get("prompt_tps_mean"),
        "per_iteration_decode": [r.get("generation_tps") for r in d.get("iterations", [])],
    }


def load_prof(tag: str, node: str):
    """De-aggregate [MTP-PROF] for one arm/node, anchored on the pre-window dump."""
    ent = os.path.join(RAW, f"prof_ent_{tag}_{node}.txt")
    anc = os.path.join(RAW, f"prof_anchor_{tag}_{node}.txt")
    if not os.path.exists(ent) or os.path.getsize(ent) == 0:
        return None
    anchors = {}
    if os.path.exists(anc) and os.path.getsize(anc) > 0:
        anchors = anchors_from(anc)
    return analyze(ent, anchors=anchors)


def cycle_time_ms(prof: dict):
    """Sum the TIME-unit cycle phases. Returns None if any is missing."""
    if not prof:
        return None
    total = 0.0
    for ph in CYCLE_PHASES:
        v = prof.get(ph)
        if not v or v.get("unit") != "ms":
            return None
        total += v["interval_weighted_mean"]
    return round(total, 4)


def main():
    rows = []
    for arm in ARMS:
        tag = arm["tag"]
        b = load_probe(tag)
        nodes = {}
        for node in ("n1", "n2"):
            p = load_prof(tag, node)
            if p:
                nodes[node] = p
        row = {
            "tag": tag,
            "mode": arm["mode"],
            "words": arm["words"],
            "probe": b,
            "nodes": nodes,
            "cycle_time_ms": {},
        }
        for node, p in nodes.items():
            row["cycle_time_ms"][node] = cycle_time_ms(p)
        rows.append(row)

    # ---- G4: matched-depth check (max % spread of achieved_prompt_tokens) ----
    toks = [r["probe"]["achieved_prompt_tokens"] for r in rows
            if r["probe"] and r["probe"]["achieved_prompt_tokens"]]
    g4 = None
    if toks:
        lo, hi = min(toks), max(toks)
        g4 = {
            "min": lo, "max": hi,
            "max_spread_pct": round((hi - lo) / lo * 100.0, 4) if lo else None,
            "threshold_pct": G4_THRESHOLD,
            "pass": (hi - lo) / lo * 100.0 <= G4_THRESHOLD if lo else False,
        }

    # ---- Effect sizes (R-N)/R and (R-X)/R ----
    def mean_of(tag):
        for r in rows:
            if r["tag"] == tag and r["probe"] and r["probe"]["decode_tps_mean"] is not None:
                return r["probe"]["decode_tps_mean"]
        return None

    R = mean_of("repetitive")
    N = mean_of("natural")
    X = mean_of("random")
    effect = {
        "R": R, "N": N, "X": X,
        "R_minus_N_over_R_pct": round((R - N) / R * 100.0, 2) if R and N is not None else None,
        "R_minus_X_over_R_pct": round((R - X) / R * 100.0, 2) if R and X is not None else None,
    }

    # ---- G2: drift check between repetitive and repetitive_recheck ----
    Rr = mean_of("repetitive_recheck")
    g2 = None
    if R and Rr is not None:
        g2 = {
            "repetitive_mean": R, "recheck_mean": Rr,
            "delta_pct": round((Rr - R) / R * 100.0, 2),
            "threshold_pct": G2_THRESHOLD,
            "pass": abs((Rr - R) / R * 100.0) <= G2_THRESHOLD,
        }

    out = {
        "arms": rows,
        "gates": {"G2_drift": g2, "G4_matched_depth": g4},
        "effect_sizes": effect,
        "cycle_phases": CYCLE_PHASES,
    }
    with open(os.path.join(RAW, "entropy_table.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    # ---- Readable markdown table ----
    print("## Entropy A/B — decode throughput (t/s) & cycle time (ms)")
    print()
    print("| arm | mode | prompt tok | decode mean | med | min | max | prefill t/s | cycle n1 ms | cycle n2 ms |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        b = r["probe"]
        if b is None:
            print(f"| {r['tag']} | {r['mode']} | **MISSING/FAILED** | | | | | | | |")
            continue
        c1 = r["cycle_time_ms"].get("n1")
        c2 = r["cycle_time_ms"].get("n2")
        print(f"| {r['tag']} | {r['mode']} | {b['achieved_prompt_tokens']} | "
              f"{b['decode_tps_mean']:.2f} | {b['decode_tps_median']:.2f} | "
              f"{b['decode_tps_min']:.2f} | {b['decode_tps_max']:.2f} | "
              f"{b['prefill_tps_mean']:.1f} | "
              f"{c1 if c1 is not None else 'n/a'} | {c2 if c2 is not None else 'n/a'} |")
    print()
    print("### Per-phase [MTP-PROF] (interval-weighted means, ms unless noted)")
    for r in rows:
        if not r["nodes"]:
            print(f"\n**{r['tag']}** — no profiler data")
            continue
        print(f"\n**{r['tag']}**")
        phases = sorted({ph for p in r["nodes"].values() for ph in p})
        print("| phase | unit | n1 mean | n1 cycles | n2 mean | n2 cycles |")
        print("|---|---|---|---|---|---|")
        for ph in phases:
            n1 = r["nodes"].get("n1", {}).get(ph)
            n2 = r["nodes"].get("n2", {}).get(ph)
            unit = (n1 or n2 or {}).get("unit", "ms")
            print(f"| {ph} | {unit} | "
                  f"{n1['interval_weighted_mean'] if n1 else 'n/a'} | {n1['cycles'] if n1 else 'n/a'} | "
                  f"{n2['interval_weighted_mean'] if n2 else 'n/a'} | {n2['cycles'] if n2 else 'n/a'} |")
    print()
    print("### Gates")
    print(f"- **G2 drift** (repetitive vs repetitive_recheck, threshold {G2_THRESHOLD}%): "
          f"{g2 if g2 else 'n/a'}")
    print(f"- **G4 matched depth** (max spread, threshold {G4_THRESHOLD}%): "
          f"{g4 if g4 else 'n/a'}")
    print(f"- **Effect sizes**: (R-N)/R = {effect['R_minus_N_over_R_pct']}%, "
          f"(R-X)/R = {effect['R_minus_X_over_R_pct']}%")
    print()
    print(f"Wrote {os.path.join(RAW, 'entropy_table.json')}")


if __name__ == "__main__":
    main()
