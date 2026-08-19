#!/usr/bin/env python3
"""Analyze gpu_util_needle_probe output: correlate GPU-busy % against the
server-log `Prefill progress:` bracket and hunt for sustained idle bubbles."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean

NODES = ["macstudio-m4-1", "macstudio-m4-2"]
LOG_RE = re.compile(
    r"\[ (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}).*Prefill progress: (\d+)/(\d+)"
)
LOCAL_TZ_OFFSET = timedelta(hours=-5)  # CDT; exo.log timestamps are local


def load_samples(path: Path) -> list[tuple[datetime, float, float]]:
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts = datetime.fromisoformat(d["timestamp"])
        out.append((ts, d["gpu_usage"][1], d.get("gpu_power", 0.0)))
    return out


def prefill_bracket(host: str, after: datetime) -> tuple[datetime, datetime, int]:
    raw = subprocess.run(
        ["ssh", host, "grep -a 'Prefill progress' ~/exo.log | tail -5000"],
        capture_output=True, text=True, check=True).stdout
    start = end = None
    total = 0
    for line in raw.splitlines():
        m = LOG_RE.search(line)
        if not m:
            continue
        ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f").replace(
            tzinfo=timezone(LOCAL_TZ_OFFSET))
        if ts < after:
            continue
        done, tot = int(m.group(2)), int(m.group(3))
        if tot < 1000:
            continue
        total = tot
        if start is None:
            start = ts
        end = ts
    assert start and end, "no prefill progress lines found in window"
    return start, end, total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/tmp/gpu_util_probe")
    ap.add_argument("--low-threshold", type=float, default=0.90)
    args = ap.parse_args()

    run = Path(args.dir)
    meta = json.loads((run / "meta.json").read_text())
    t0 = datetime.fromisoformat(meta["request_start_utc"])

    start, end, total = prefill_bracket(NODES[0], t0 - timedelta(seconds=30))
    print(f"prefill window: {start.isoformat()} -> {end.isoformat()} "
          f"({(end-start).total_seconds():.1f}s, {total} tokens)")

    report = {"prefill_start": start.isoformat(), "prefill_end": end.isoformat(),
              "prefill_seconds": (end - start).total_seconds(),
              "prefill_tokens": total, "nodes": {}}

    for node in NODES:
        samples = load_samples(run / f"gpu_util_{node}.jsonl")
        win = [(ts, u, p) for ts, u, p in samples if start <= ts <= end]
        if not win:
            print(f"{node}: NO SAMPLES IN WINDOW")
            continue
        utils = [u for _, u, _ in win]
        lows = [(ts, u) for ts, u, _ in win if u < args.low_threshold]
        # longest contiguous run below threshold
        longest, cur = 0, 0
        for _, u, _ in win:
            cur = cur + 1 if u < args.low_threshold else 0
            longest = max(longest, cur)
        node_rep = {
            "samples_in_window": len(win),
            "mean_util": mean(utils),
            "min_util": min(utils),
            "max_util": max(utils),
            "p05_util": sorted(utils)[max(0, len(utils) // 20)],
            "mean_gpu_power_w": mean(p for _, _, p in win),
            f"pct_below_{args.low_threshold}": 100.0 * len(lows) / len(win),
            f"longest_contiguous_below_{args.low_threshold}_samples": longest,
            "low_samples": [(ts.isoformat(), round(u, 4)) for ts, u in lows][:50],
        }
        report["nodes"][node] = node_rep
        print(f"\n=== {node} ===")
        for k, v in node_rep.items():
            if k != "low_samples":
                print(f"  {k}: {v}")
        if lows:
            print(f"  first low samples: {node_rep['low_samples'][:10]}")

    (run / "analysis.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {run/'analysis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
