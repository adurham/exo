#!/usr/bin/env python3
"""
Standalone parser for the untimestamped per-cycle MTP acceptance lines in the
runner stderr.log. NOT a modification of bench/mtp_cycle_time.py.

Segments the [MTP] lines by the resetting `cycles` counter (cycles=1 marks a new
segment). Reports per-segment final converged mean_accept.

NOTE: mean_accept is CUMULATIVE within a segment (converges as cycles increases).
The correct per-segment value is the FINAL converged mean_accept at the segment's
max cycle, NOT a naive average of the cumulative values.

Usage:
    python3 parse_mtp_segments.py <acceptance_raw_file> [--csv out.csv]
"""
import re
import sys

LINE = re.compile(r"\[MTP\] cycles=(\d+)\s+mean_accept=([\d.]+)/(\d+)")


def parse(path):
    segs = []
    cur = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("#"):
                continue
            m = LINE.match(line)
            if not m:
                continue
            c = int(m.group(1))
            a = float(m.group(2))
            d = int(m.group(3))
            if c == 1 and cur:
                segs.append(cur)
                cur = []
            cur.append((c, a, d))
    if cur:
        segs.append(cur)
    return segs


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    path = sys.argv[1]
    segs = parse(path)
    print(f"total MTP lines: {sum(len(s) for s in segs)}")
    print(f"num segments: {len(segs)}")
    print("seg | len | max_cycles | final_mean_accept | gamma")
    for i, s in enumerate(segs):
        maxc = s[-1][0]
        final = s[-1][1]
        gamma = s[-1][2]
        print(f"{i:3d} | {len(s):4d} | {maxc:5d} | {final} | {gamma}")
    if len(sys.argv) >= 3 and sys.argv[2] == "--csv":
        out = sys.argv[3] if len(sys.argv) >= 4 else "segments.csv"
        with open(out, "w") as fh:
            fh.write("seg,len,max_cycles,final_mean_accept,gamma\n")
            for i, s in enumerate(segs):
                fh.write(f"{i},{len(s)},{s[-1][0]},{s[-1][1]},{s[-1][2]}\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
