#!/usr/bin/env python3
"""Derive real ms/cycle for the DSv4 MTP speculative loop from runner logs.

WHY: end-to-end tok/s is too blunt an instrument for the Phase C question.
P03 projected the shared_experts verify-batching win at -1.2 ms/cycle, which
at ~3.2 accepted tokens/cycle is only ~+1.3% on decode tok/s -- comfortably
inside the run-to-run spread of a 100K probe (measured spread was ~2.4
tok/s peak-to-peak on an unchanged config). Measuring a 1.3% effect with an
instrument whose noise is larger than the effect can only produce a
coin-flip.

The runner already logs one line per speculative cycle with a millisecond
timestamp and a monotonically increasing cycle counter. Differencing
consecutive lines gives the actual wall time per cycle directly, in the same
unit P03's projection was expressed in, with thousands of samples per run
instead of three.

Run this ON the studio node (the log is large; don't ship it over ssh --
a plain `cat` of 6000 lines got silently truncated to 266 when piped
through the ssh transport).

Usage:  python3 mtp_cycle_time.py [LOGPATH] [--tag NAME] [--last N]
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys

LINE = re.compile(
    r"(\d\d):(\d\d):(\d\d)\.(\d+).*?\[MTP\] cycles=(\d+)\s+mean_accept=([\d.]+)"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?", default="/Users/adam.durham/exo_verbon3.log")
    ap.add_argument("--tag", default="untagged")
    ap.add_argument("--last", type=int, default=4000,
                    help="consider only the last N matching cycle lines")
    ap.add_argument("--out")
    args = ap.parse_args()

    rows: list[tuple[float, int, float]] = []
    with open(args.log, "rb") as fh:
        for raw in fh:
            line = raw.decode("utf-8", "replace")
            m = LINE.search(line)
            if not m:
                continue
            h, mi, s, ms, cyc, acc = m.groups()
            # log ms field can be 3 or 6 digits; normalise to seconds
            frac = float("0." + ms)
            t = int(h) * 3600 + int(mi) * 60 + int(s) + frac
            rows.append((t, int(cyc), float(acc)))

    rows = rows[-args.last:]
    if len(rows) < 20:
        print(json.dumps({"tag": args.tag, "error": "too few cycle lines",
                          "n": len(rows)}))
        return 1

    # Difference consecutive cycles. Only keep steps where the cycle counter
    # advanced by exactly 1 (a gap means lines were rotated away or the
    # generation restarted, and would otherwise fabricate a huge "cycle").
    deltas: list[float] = []
    for (t0, c0, _), (t1, c1, _) in zip(rows, rows[1:]):
        if c1 - c0 == 1:
            dt = (t1 - t0) * 1000.0
            # Guard against midnight wrap and inter-request idle gaps: a real
            # speculative cycle at 100K context is single-digit-to-tens of ms.
            if 0.0 < dt < 500.0:
                deltas.append(dt)

    deltas_sorted = sorted(deltas)
    n = len(deltas_sorted)
    # Trimmed statistics: the tail includes cross-request boundaries and
    # scheduler hiccups that are not the steady-state cycle we care about.
    lo, hi = int(n * 0.05), int(n * 0.95)
    trimmed = deltas_sorted[lo:hi] or deltas_sorted

    accs = [a for _, _, a in rows]
    result = {
        "tag": args.tag,
        "n_cycles": n,
        "cycle_range": [rows[0][1], rows[-1][1]],
        "ms_per_cycle_median": round(statistics.median(deltas_sorted), 3),
        "ms_per_cycle_mean_trimmed": round(statistics.fmean(trimmed), 3),
        "ms_per_cycle_p25": round(deltas_sorted[int(n * 0.25)], 3),
        "ms_per_cycle_p75": round(deltas_sorted[int(n * 0.75)], 3),
        "ms_per_cycle_stdev": round(statistics.pstdev(trimmed), 3),
        "mean_accept_last": accs[-1],
    }
    print(json.dumps(result, indent=1))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
