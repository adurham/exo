#!/usr/bin/env python3
"""Upper-bound analysis: hybrid per-run dispatch (gather_qmv_rhs) vs the
call-level B/E gate for DeepSeek-V4-Flash MoE prefill.

Models the ACTUAL steel gather_qmm_rhs(_lhs) kernel geometry read from
mlx/backend/metal/kernels/quantized.h:2769 + quantized.cpp:1494:
  grid = (ceil(N/bn), ceil(M/bm)) over the FLAT SORTED (token,expert) rows,
  bm = 16 (MLX_GQMM_RHS_LHS_BM default).
Each threadgroup owns a 16-row slice and loops over the DISTINCT expert
segments inside it, doing a FULL BM x BN x K MMA per segment (Xs rows beyond
group_rows are zero-filled). So cost is measured in full-tile MMA passes:
    passes = sum over tiles of (#distinct experts in that tile)
and useful fraction = total_rows / (passes * bm).

Usage: python bench/moe_run_length_tile_waste.py
"""

import numpy as np

TOKENS = 2048
TOPK = 6
NUM_EXPERTS = 256
BM = 16
MOE_SHARE_OF_PREFILL = 0.269  # span profile 2026-08-18: moe.switch_mlp
THRESHOLDS = [4, 6, 8, 16, 32]


def routing_run_lengths(seed=0, skew=2.2):
    """Top-6-of-256 routing with realistic skew (Gumbel-argmax over
    non-uniform expert logits). Tuned to match production stats:
    median ~14, mean ~48, max ~1.6k, ~181/256 experts active."""
    rng = np.random.default_rng(seed)
    # Heavy-tailed expert popularity prior -> real routing skew
    prior = rng.lognormal(mean=0.0, sigma=skew, size=NUM_EXPERTS)
    logits = np.log(prior)[None, :] + rng.gumbel(size=(TOKENS, NUM_EXPERTS))
    top = np.argpartition(-logits, TOPK, axis=1)[:, :TOKENS and TOPK]
    counts = np.bincount(top.ravel(), minlength=NUM_EXPERTS)
    return counts


def tile_passes(runs, bm=BM):
    """Simulate the sorted flat row array and count full-tile MMA passes."""
    runs = runs[runs > 0]
    # expert id per sorted row
    eid = np.repeat(np.arange(len(runs)), runs)
    total = len(eid)
    passes = 0
    seg_rows = []  # (expert_local_run_len, rows_of_that_expert_in_this_tile)
    for start in range(0, total, bm):
        tile = eid[start:start + bm]
        ids, cnts = np.unique(tile, return_counts=True)
        passes += len(ids)
        for i, c in zip(ids, cnts):
            seg_rows.append((runs[i], c))
    return total, passes, seg_rows


def main():
    runs = routing_run_lengths()
    active = runs[runs > 0]
    total_rows = int(runs.sum())
    print("=== routing distribution ===")
    print(f"active experts : {len(active)}/{NUM_EXPERTS}")
    print(f"total rows     : {total_rows} (expect {TOKENS*TOPK})")
    print(f"median / mean  : {np.median(active):.0f} / {active.mean():.1f}")
    print(f"min / max      : {active.min()} / {active.max()}")
    print(f"call-level B/E : {total_rows/NUM_EXPERTS:.1f}")

    total, passes, seg_rows = tile_passes(runs)
    tile_capacity = passes * BM
    overall_useful = total / tile_capacity
    print("\n=== steel tile geometry (bm=%d) ===" % BM)
    print(f"dense tiles (M/bm)      : {int(np.ceil(total/BM))}")
    print(f"actual MMA tile passes  : {passes}  (+{passes/np.ceil(total/BM)-1:.1%})")
    print(f"useful row fraction     : {overall_useful:.1%}")
    print(f"TOTAL padding waste     : {1-overall_useful:.1%} of launched tile capacity")

    print("\n=== per-threshold buckets ===")
    hdr = ("T", "rows<=T", "row frac", "passes<=T", "%passes",
           "pad waste in bucket", "iso speedup", "e2e speedup")
    print("{:>3} {:>8} {:>9} {:>10} {:>8} {:>20} {:>12} {:>12}".format(*hdr))
    rows_out = []
    for T in THRESHOLDS:
        bucket = [(r, c) for r, c in seg_rows if r <= T]
        b_passes = len(bucket)
        b_rows = sum(c for _, c in bucket)
        b_cap = b_passes * BM
        pad_frac = 1 - b_rows / b_cap if b_cap else 0.0
        # Isolated: best case = bucket's wasted capacity removed entirely from
        # the MoE matmul's launched tile capacity.
        wasted = b_cap - b_rows
        iso = wasted / tile_capacity           # frac of MoE matmul work removed
        e2e = iso * MOE_SHARE_OF_PREFILL       # frac of prefill wall time
        rows_out.append((T, b_rows, b_rows/total, b_passes,
                         b_passes/passes, pad_frac, iso, e2e))
        print("{:>3} {:>8} {:>8.1%} {:>10} {:>7.1%} {:>19.1%} {:>11.1%} {:>11.1%}"
              .format(T, b_rows, b_rows/total, b_passes, b_passes/passes,
                      pad_frac, iso, e2e))

    print("\nNote: 'iso speedup' = fraction of MoE matmul tile-capacity work removed")
    print("assuming 100% of that bucket's padding vanishes AND the replacement")
    print("kernel is at least as fast per useful row (optimistic on both counts).")
    print("'e2e speedup' scales by moe.switch_mlp = %.1f%% of prefill." % (MOE_SHARE_OF_PREFILL*100))
    return rows_out


if __name__ == "__main__":
    main()
