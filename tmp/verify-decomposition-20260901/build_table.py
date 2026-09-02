#!/usr/bin/env python3
"""Build the context-depth table from bench JSON + [MTP-PROF] harvests.

Anchoring rule (critical): the profiler's `n` is cumulative over the runner's
LIFETIME, not the measurement window. The first dump inside the harvest window
(e.g. n=9000) carries ~9000 pre-window idle cycles at their cumulative mean.
We therefore use that first in-window dump AS THE ANCHOR and only count
intervals after it. Cost: up to 50 cycles of the window are discarded. Benefit:
zero contamination from pre-bench idle cycles. (Validated: this procedure
reproduces v3/RESULTS.md's published 56.087 ms verify / 650 cycles exactly.)

Acceptance is DERIVED, not logged: EXO_DSV4_MTP_LOG_INTERVAL is unset on the
live runners, so the `[MTP] mean_accept=` line never fires. With gamma=3 the
cycle emits 1 bonus + n_accepted tokens, so:
    tokens_per_cycle = total_generation_tokens / cycles
    mean_acceptance  = tokens_per_cycle - 1
"""
from __future__ import annotations
import json, os, sys, glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_prof import parse, deaggregate  # noqa: E402

BASE = "/Users/adam.durham/repos/exo/tmp/verify-decomposition-20260901/raw"
PHASES = ["draft", "verify", "accept", "rollback", "total", "rb_snap"]


def prof_for(tag: str, node: str):
    path = os.path.join(BASE, f"prof_{tag}_{node}.txt")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    series = parse(path)
    if "total" not in series:
        return None
    # anchor = first in-window dump per phase
    out = {}
    for phase, pts in series.items():
        pts_sorted = sorted(pts, key=lambda x: x[0])
        if len(pts_sorted) < 2:
            continue
        anchor = pts_sorted[0]
        wmean, cycles, _ = deaggregate(pts_sorted, anchor=anchor)
        if wmean is None:
            continue
        out[phase] = {"ms": round(wmean, 3), "cycles": cycles}
    return out


def bench_for(tag: str):
    path = os.path.join(BASE, f"bench_{tag}.json")
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    lvl = d["by_concurrency"]["1"]
    iters = lvl["iterations"]
    scored = [i for i in iters if not i.get("warmup")]
    reqs_all = [r for i in iters for r in i["per_request"]]
    reqs_scored = [r for i in scored for r in i["per_request"]]
    return {
        "prompt_tokens": reqs_scored[0]["prompt_tokens"] if reqs_scored else None,
        "decode_tps_mean": lvl["summary"]["agg_tps_mean"],
        "decode_tps_median": lvl["summary"]["agg_tps_median"],
        "decode_tps_min": lvl["summary"]["agg_tps_min"],
        "decode_tps_max": lvl["summary"]["agg_tps_max"],
        "prefill_tps_mean": sum(r["prompt_tps"] for r in reqs_scored) / len(reqs_scored),
        "wall_mean_s": lvl["summary"]["wall_mean_s"],
        "gen_tokens_all_iters": sum(r["generation_tokens"] for r in reqs_all),
        "n_scored": len(scored),
        "bad_rate": lvl["summary"]["bad_iter_rate"],
    }


def main():
    tags = sorted({os.path.basename(f).split("_")[1].split(".")[0]
                   for f in glob.glob(os.path.join(BASE, "bench_*.json"))})
    rows = []
    for tag in tags:
        b = bench_for(tag)
        if not b:
            continue
        row = {"tag": tag, **b, "nodes": {}}
        for node in ("n1", "n2"):
            p = prof_for(tag, node)
            if p:
                row["nodes"][node] = p
        # derive acceptance from the node profile (both ranks are in lockstep)
        cyc = None
        for node in ("n1", "n2"):
            if node in row["nodes"] and "total" in row["nodes"][node]:
                cyc = row["nodes"][node]["total"]["cycles"]
                break
        if cyc:
            tpc = b["gen_tokens_all_iters"] / cyc
            row["cycles_in_window"] = cyc
            row["tokens_per_cycle"] = round(tpc, 4)
            row["derived_acceptance"] = round(tpc - 1, 4)
            row["implied_tps_from_profile"] = round(tpc / (row["nodes"]["n1"]["total"]["ms"] / 1000.0), 2) if "n1" in row["nodes"] else None
        rows.append(row)

    print(json.dumps(rows, indent=1))

    # Compact human table
    print("\n" + "=" * 108)
    hdr = f"{'depth':>7} {'prompt_tok':>11} {'decode t/s':>11} {'prefill t/s':>12} {'verify ms':>10} {'draft':>7} {'accept':>7} {'rb':>6} {'total ms':>9} {'acc':>6}"
    print(hdr); print("-" * 108)
    for r in rows:
        n1 = r["nodes"].get("n1", {})
        g = lambda k: (f"{n1[k]['ms']:.2f}" if k in n1 else "n/a")
        print(f"{r['tag']:>7} {r['prompt_tokens'] or 0:>11} "
              f"{r['decode_tps_mean']:>11.2f} {r['prefill_tps_mean']:>12.1f} "
              f"{g('verify'):>10} {g('draft'):>7} {g('accept'):>7} {g('rollback'):>6} {g('total'):>9} "
              f"{r.get('derived_acceptance', float('nan')):>6.2f}")
    print("=" * 108)


if __name__ == "__main__":
    main()
