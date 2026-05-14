"""Summarize routing histograms collected by ROUTE_HIST probe."""
from __future__ import annotations
import argparse, glob, os, re, sys
from collections import defaultdict
import numpy as np


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="/tmp/dsv4_route_hist")
    p.add_argument("--n-layers", type=int, default=43)
    p.add_argument("--n-experts", type=int, default=256)
    p.add_argument("--csv", default="")
    args = p.parse_args()

    if not os.path.isdir(args.dir):
        print(f"ERROR: {args.dir} does not exist", file=sys.stderr)
        return 2

    files = sorted(glob.glob(f"{args.dir}/L*.npy"))
    if not files:
        print(f"ERROR: no L*.npy files in {args.dir}", file=sys.stderr)
        return 2

    pat = re.compile(r"L(\d+)_pid\d+\.npy$")
    by_layer = defaultdict(lambda: np.zeros(args.n_experts, dtype=np.int64))
    for f in files:
        m = pat.search(f)
        if not m:
            continue
        li = int(m.group(1))
        arr = np.load(f)
        if arr.shape[0] != args.n_experts:
            print(f"WARN: {f} has {arr.shape[0]} experts", file=sys.stderr)
            continue
        by_layer[li] += arr

    if not by_layer:
        print("ERROR: no valid data", file=sys.stderr)
        return 2

    print("=" * 80)
    print(f"DSv4 Expert Routing Histogram Summary")
    print(f"  Dir: {args.dir}")
    print(f"  Files: {len(files)}    Layers: {len(by_layer)} of {args.n_layers}")
    print("=" * 80)

    global_counts = np.zeros(args.n_experts, dtype=np.int64)
    for counts in by_layer.values():
        global_counts += counts
    total = int(global_counts.sum())

    print(f"\nTotal expert activations: {total:,}")
    if total == 0:
        return 1

    sorted_global = np.sort(global_counts)[::-1]
    cum = np.cumsum(sorted_global)
    pct = cum / max(1, total) * 100.0

    print("\nGlobal expert concentration (top-K share of routings):")
    for k in [1, 4, 8, 16, 32, 64, 128]:
        if k <= args.n_experts:
            print(f"  top-{k:3d}: {pct[k-1]:5.1f}%")

    avg = total / args.n_experts
    cold = int((global_counts < avg * 0.1).sum())
    hot = int((global_counts > avg * 3).sum())
    print(f"\nMean per expert: {avg:.0f}    Max: {global_counts.max()} (E{global_counts.argmax()})    Min: {global_counts.min()}")
    print(f"Cold experts (<10% mean): {cold}/{args.n_experts}   Hot (>3x mean): {hot}/{args.n_experts}")

    print(f"\nPer-layer concentration:")
    print(f"  {'Layer':>5}  {'Total':>10}  {'Top-1':>7}  {'Top-8':>7}  {'Top-32':>7}  {'Cold':>5}  {'Hot':>5}")
    rows = []
    for li in sorted(by_layer.keys()):
        c = by_layer[li]
        tot = int(c.sum())
        if tot == 0:
            continue
        sc = np.sort(c)[::-1]
        cc = np.cumsum(sc) / tot * 100.0
        avl = tot / args.n_experts
        cld = int((c < avl * 0.1).sum())
        hot_l = int((c > avl * 3).sum())
        print(f"  {li:5d}  {tot:10,}  {cc[0]:6.2f}%  {cc[7]:6.2f}%  {cc[31]:6.2f}%  {cld:5d}  {hot_l:5d}")
        rows.append((li, cc[7]))

    rows.sort(key=lambda t: -t[1])
    print(f"\nLayers ranked by top-8 concentration (best co-location candidates):")
    for li, conc in rows[:10]:
        print(f"  L{li:02d}: top-8 = {conc:.2f}%")

    print("\nCross-layer hot-expert reuse (which experts are hot in MANY layers):")
    expert_layer_count = np.zeros(args.n_experts, dtype=np.int64)
    for c in by_layer.values():
        top32 = np.argsort(c)[::-1][:32]
        expert_layer_count[top32] += 1
    print(f"  Experts in top-32 of  2+ layers: {int((expert_layer_count >=  2).sum())}")
    print(f"  Experts in top-32 of  5+ layers: {int((expert_layer_count >=  5).sum())}")
    print(f"  Experts in top-32 of 10+ layers: {int((expert_layer_count >= 10).sum())}")
    print(f"  Experts in top-32 of 20+ layers: {int((expert_layer_count >= 20).sum())}")

    # Top-N most-universally-hot experts
    print("\nTop-20 experts by number of layers where they are top-32:")
    universal = np.argsort(expert_layer_count)[::-1][:20]
    for e in universal:
        print(f"  E{e:3d}: hot in {int(expert_layer_count[e]):2d} layers, {int(global_counts[e]):,} total routings")

    if args.csv:
        with open(args.csv, "w") as f:
            f.write("layer,expert,count\n")
            for li in sorted(by_layer.keys()):
                for eid, cnt in enumerate(by_layer[li]):
                    f.write(f"{li},{eid},{cnt}\n")
        print(f"\nFull table -> {args.csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
