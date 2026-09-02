#!/usr/bin/env python3
"""De-aggregate cumulative [MTP-PROF] running means into per-interval values.

The profiler emits CUMULATIVE running means: n grows 50,100,150... and `mean`
is the average over ALL cycles so far. Taking statistics.mean over dump lines
double-counts early cycles (v3/RESULTS.md §1). Correct arithmetic:

    sum_k    = mean_k * n_k
    interval = (sum_k - sum_{k-1}) / (n_k - n_{k-1})

then weight each interval by its cycle count.

Also handles the profiler's per-series independent n (rb_* series only fire on
restore cycles, so their n advances at a different rate than draft/verify).
"""
from __future__ import annotations
import re, json, sys, glob, os
from collections import defaultdict

LINE = re.compile(
    r"\[MTP-PROF\]\s+B=(?P<b>\d+)\s+(?P<phase>\S+)\s+mean=\s*(?P<mean>[-\d.]+)m?s?\s+"
    r"min=\s*(?P<min>[-\d.]+)m?s?\s+max=\s*(?P<max>[-\d.]+)m?s?\s+n=(?P<n>\d+)"
)

# rb_pool_restores is an integer COUNT mislabeled with an 'ms' suffix by a
# unit-blind formatter (v3/RESULTS.md §2). Never report it as a time.
COUNT_SERIES = {"rb_pool_restores"}


def parse(path: str):
    series = defaultdict(list)  # phase -> [(n, mean), ...] in file order
    with open(path, errors="replace") as fh:
        for line in fh:
            m = LINE.search(line)
            if not m:
                continue
            series[m.group("phase")].append((int(m.group("n")), float(m.group("mean"))))
    return series


def deaggregate(points, anchor=None):
    """points: [(n, cumulative_mean)] -> (weighted_interval_mean, total_cycles, intervals)

    CRITICAL: `n` is cumulative over the runner's whole lifetime, not the
    measurement window. If the first in-window dump is n=1050, treating it as a
    single interval silently imports all 1050 pre-window (idle) cycles at their
    cumulative mean. That reproduces the very double-counting v3/RESULTS.md §1
    warns about, just one level up.

    `anchor` = (n, cumulative_mean) of the LAST dump strictly BEFORE the window.
    It seeds prev_n/prev_sum so only in-window cycles are counted. Without an
    anchor we start at (0, 0.0), which is correct ONLY if the window truly
    begins at runner start.
    """
    pts = sorted(points, key=lambda x: x[0])
    # Drop non-monotonic restarts (a runner restart resets n); keep longest run
    clean, prev_n = [], -1
    for n, mean in pts:
        if n > prev_n:
            clean.append((n, mean))
            prev_n = n
    intervals = []
    if anchor is not None:
        prev_n, prev_sum = anchor[0], anchor[1] * anchor[0]
        clean = [(n, m) for n, m in clean if n > prev_n]
    else:
        prev_n, prev_sum = 0, 0.0
    for n, mean in clean:
        cur_sum = mean * n
        dn = n - prev_n
        if dn > 0:
            intervals.append((dn, (cur_sum - prev_sum) / dn))
        prev_n, prev_sum = n, cur_sum
    if not intervals:
        return None, 0, []
    tot = sum(dn for dn, _ in intervals)
    wmean = sum(dn * v for dn, v in intervals) / tot if tot else None
    return wmean, tot, intervals


def analyze(path: str, anchors=None):
    """anchors: dict phase -> (n, cumulative_mean) from the last pre-window dump."""
    anchors = anchors or {}
    out = {}
    for phase, pts in parse(path).items():
        wmean, cycles, _ = deaggregate(pts, anchor=anchors.get(phase))
        if wmean is None:
            continue
        out[phase] = {
            "interval_weighted_mean": round(wmean, 4),
            "cycles": cycles,
            "unit": "count" if phase in COUNT_SERIES else "ms",
            "n_dumps": len(pts),
            "anchored": phase in anchors,
        }
    return out


def anchors_from(path: str):
    """Last (n, mean) per phase from a file of PRE-window dumps."""
    return {ph: sorted(pts)[-1] for ph, pts in parse(path).items() if pts}


if __name__ == "__main__":
    base = "/Users/adam.durham/repos/exo/tmp/verify-decomposition-20260901/raw"
    results = {}
    for f in sorted(glob.glob(os.path.join(base, "prof_*.txt"))):
        tag = os.path.basename(f).replace("prof_", "").replace(".txt", "")
        if os.path.getsize(f) == 0:
            results[tag] = {"error": "empty"}
            continue
        results[tag] = analyze(f)
    print(json.dumps(results, indent=1))
