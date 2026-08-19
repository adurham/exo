#!/usr/bin/env python3
"""K2: analytic tile-size sweep (bm in {16,32,64}) for DeepSeek-V4-Flash MoE prefill.

Companion to bench/moe_run_length_tile_waste.py -- reuses that script's
routing-generation (Gumbel-argmax over a lognormal expert-popularity prior,
tuned to production: median ~14, mean ~48, max ~1.6k, ~181/256 active) and its
tile-accounting model (flat sorted (token,expert) row array, tiled every bm
rows, one full BM x BN x K MMA pass per DISTINCT expert segment appearing in a
tile). That accounting is equivalent to segment-aligned per-expert tiling:
    passes == sum_e (# tiles expert e's rows touch) ~= sum_e ceil(run_e / bm)

*** VALIDITY CAVEAT ***
This model assumes the kernel pads EACH EXPERT SEGMENT to a full tile boundary.
If the real steel kernel instead packs mixed-expert tiles WITHOUT per-segment
padding, this model overestimates waste and these predictions do not apply.
See task K1 (kernel-source determination).

Usage: python bench/moe_tile_size_sweep.py
"""

import numpy as np

from moe_run_length_tile_waste import NUM_EXPERTS, routing_run_lengths, tile_passes

BMS = (16, 32, 64)
SEEDS = tuple(range(8))


def clean_row_fraction(runs, bm, slack=0):
    """Fraction of TOTAL ROWS coming from experts whose run length is already a
    clean multiple of bm (slack=0) or within `slack` rows of one."""
    runs = runs[runs > 0]
    rem = runs % bm
    clean = (rem == 0) | (rem >= bm - slack) if slack else (rem == 0)
    return runs[clean].sum() / runs.sum()


def insensitive_row_fraction(runs, bm, thresh=0.05):
    """Fraction of total rows from experts whose per-expert padding overhead
    (ceil(run/bm)*bm - run)/run is <= thresh -- i.e. experts big/clean enough
    that tile size barely matters for them."""
    runs = runs[runs > 0]
    pad = np.ceil(runs / bm) * bm - runs
    return runs[(pad / runs) <= thresh].sum() / runs.sum()


def main():
    print("seeds:", SEEDS)
    per_bm = {bm: [] for bm in BMS}
    rowfrac = {bm: {"clean": [], "insens": []} for bm in BMS}
    for seed in SEEDS:
        runs = routing_run_lengths(seed=seed)
        active = runs[runs > 0]
        total_rows = int(runs.sum())
        print(f"\nseed {seed}: active {len(active)}/{NUM_EXPERTS}  rows {total_rows}"
              f"  median {np.median(active):.0f}  mean {active.mean():.1f}"
              f"  min/max {active.min()}/{active.max()}")
        for bm in BMS:
            total, passes, _ = tile_passes(runs, bm=bm)
            dense = int(np.ceil(total / bm))
            useful = total / (passes * bm)
            per_bm[bm].append((dense, passes, useful, 1 - useful))
            rowfrac[bm]["clean"].append(clean_row_fraction(runs, bm))
            rowfrac[bm]["insens"].append(insensitive_row_fraction(runs, bm))
            print(f"   bm={bm:<3} dense_min_passes={dense:<5} actual_passes={passes:<5}"
                  f" (+{passes/dense-1:6.1%})  useful={useful:6.1%}  waste={1-useful:6.1%}")

    print("\n=== SUMMARY across %d seeds ===" % len(SEEDS))
    print("{:>4} {:>12} {:>12} {:>10} {:>10} {:>16} {:>18}".format(
        "bm", "mean waste", "std waste", "min", "max", "clean-mult rows", "pad<=5% rows"))
    for bm in BMS:
        w = np.array([r[3] for r in per_bm[bm]])
        c = np.mean(rowfrac[bm]["clean"])
        i = np.mean(rowfrac[bm]["insens"])
        print("{:>4} {:>11.1%} {:>12.2%} {:>9.1%} {:>10.1%} {:>15.1%} {:>17.1%}".format(
            bm, w.mean(), w.std(), w.min(), w.max(), c, i))


if __name__ == "__main__":
    main()
