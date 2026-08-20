#!/usr/bin/env python3
"""LEVER 1: real headroom in the LIVE MoE dispatch path at small M.

Production facts encoded (verified from the tree, not assumed):
  * DSv4-Flash MoE experts = mxfp4 g=32 (mlx-lm utils.py per-layer override
    over the affine-8 top level); hidden=4096, moe_intermediate=2048 ->
    INTER=1024 per TP rank; 256 routed experts, top_k=6.
  * SwitchGLU's expand_dims collapses GatherQMM's M to 1, so every production
    MoE call hits an M==1 gate: gather_qmv_rhs (B/E in [2, MAXBE]) or
    gather_qmm_rhs (B/E>=4), else generic gather_qmm.

Three timings per shape, all streaming the SAME expert-weight bytes:
  live    : mx.gather_qmm at the SwitchGLU convention -- whatever kernel the
            dispatcher actually picks (A/B via MLX_GATHER_QMV_RHS etc.)
  ceiling : dense mx.quantized_matmul against the experts' weights viewed as
            ONE (E*INTER, HIDDEN) matrix, with M = mean rows/expert. Same
            weight bytes read exactly once, same nominal FLOPs, zero gather,
            zero raggedness. This is the achievable roofline for this data.
  bwfloor : same dense qmm with M=1 -- pure weight-streaming rate, the
            hardware's ceiling when there is no arithmetic intensity at all.

Routing is REALISTIC and RAGGED: expert popularity is drawn from a Dirichlet-
like power law so the run-length histogram matches production (median ~7-14,
min 1, long tail) rather than the uniform ~L*6/256 a naive random top-k gives.

Usage:
  .venv/bin/python bench/lever1_smallm_headroom.py --ls 256,512,1024,2048
  MLX_GATHER_QMV_RHS=0 .venv/bin/python bench/lever1_smallm_headroom.py ...
"""

from __future__ import annotations

import argparse
import json
import os
import time

import mlx.core as mx

N_EXPERTS = 256
HIDDEN = 4096
INTER = 1024  # per TP rank
TOP_K = 6
GROUP = 32
BITS = 4
MODE = "mxfp4"


def bench(fn, iters: int, warm: int) -> float:
    for _ in range(warm):
        mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - t0) / iters


def ragged_pairs(L: int, skew: float, seed: int):
    """Sorted (token,expert) expert ids with a production-like ragged run
    histogram. skew=0 -> uniform; larger -> heavier tail."""
    mx.random.seed(seed)
    # power-law expert popularity
    pop = mx.exp(mx.random.normal((N_EXPERTS,)) * skew)
    logits = mx.log(pop)[None, :] + mx.random.gumbel((L, N_EXPERTS))
    idx = mx.argpartition(-logits, TOP_K, axis=-1)[:, :TOP_K]
    pairs = mx.sort(idx.reshape(-1).astype(mx.uint32))
    mx.eval(pairs)
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ls", default="256,512,1024,2048")
    ap.add_argument("--skew", type=float, default=1.1)
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--warm", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    mx.random.seed(args.seed)
    w, s = mx.quantize(
        mx.random.normal((N_EXPERTS, INTER, HIDDEN), dtype=mx.float32) * 0.02,
        group_size=GROUP, bits=BITS, mode=MODE,
    )
    mx.eval(w, s)
    # flat view: all experts as one (E*INTER, HIDDEN) quantized matrix
    wf = w.reshape(N_EXPERTS * INTER, -1)
    sf = s.reshape(N_EXPERTS * INTER, -1)
    mx.eval(wf, sf)

    bytes_per_expert = INTER * HIDDEN * (BITS / 8) + (INTER * HIDDEN // GROUP) * 1.0
    all_bytes = N_EXPERTS * bytes_per_expert

    rows = []
    for L in [int(v) for v in args.ls.split(",")]:
        pairs = ragged_pairs(L, args.skew, args.seed + L)
        B = int(pairs.size)
        counts = mx.zeros((N_EXPERTS,), mx.int32)
        counts = counts.at[pairs.astype(mx.int32)].add(mx.ones((B,), mx.int32))
        mx.eval(counts)
        cl = counts.tolist()
        used = sorted(c for c in cl if c > 0)
        used_n = len(used)
        med = used[used_n // 2]
        gb_used = used_n * bytes_per_expert / 1e9
        mean_m = max(1, round(B / used_n))

        x = (mx.random.normal((B, HIDDEN)) * 0.1).astype(mx.bfloat16)
        xc = x[:mean_m]
        x1 = x[:1]
        mx.eval(x, xc, x1)
        xg = mx.expand_dims(x, -2)  # SwitchGLU convention -> M collapses to 1

        def live():
            return mx.gather_qmm(
                xg, w, s, rhs_indices=pairs, transpose=True,
                group_size=GROUP, bits=BITS, mode=MODE, sorted_indices=True,
            )

        def ceiling():
            return mx.quantized_matmul(
                xc, wf, sf, transpose=True,
                group_size=GROUP, bits=BITS, mode=MODE,
            )

        def bwfloor():
            return mx.quantized_matmul(
                x1, wf, sf, transpose=True,
                group_size=GROUP, bits=BITS, mode=MODE,
            )

        # Round-robin interleaved timing, MIN over iterations. Two separate
        # hazards to defeat:
        #  1. sequential tier-by-tier timing lets GPU thermal drift + page-in
        #     alias onto the tier comparison (docs/moe-vs-dense-qmm-isolation
        #     -2026-08-19.md methodology note) -> interleave.
        #  2. this laptop runs other work concurrently (load avg 6+ observed),
        #     which inflates MEAN wall time per call unpredictably -> take the
        #     MINIMUM, which is the uncontended cost.
        fns = {"live": live, "ceil": ceiling, "bw": bwfloor}
        for f in fns.values():
            for _ in range(args.warm):
                mx.eval(f())
        mx.synchronize()
        best = {k: float("inf") for k in fns}
        for _ in range(args.iters):
            for k, f in fns.items():
                t0 = time.perf_counter()
                mx.eval(f())
                mx.synchronize()
                best[k] = min(best[k], time.perf_counter() - t0)
        t_live, t_ceil, t_bw = best["live"], best["ceil"], best["bw"]

        live_gbs = gb_used / t_live
        ceil_gbs = all_bytes / 1e9 / t_ceil
        bw_gbs = all_bytes / 1e9 / t_bw
        row = {
            "L": L, "B": B, "B_over_E": B / N_EXPERTS, "experts_used": used_n,
            "mean_M": mean_m, "median_M": med, "min_M": used[0],
            "p90_M": used[int(0.9 * used_n)], "max_M": used[-1],
            "live_ms": t_live * 1e3, "live_gbs": live_gbs,
            "ceiling_ms": t_ceil * 1e3, "ceiling_gbs": ceil_gbs,
            "bwfloor_gbs": bw_gbs,
            "headroom_x": t_live / t_ceil,
            "live_frac_of_ceiling": live_gbs / ceil_gbs,
            "live_frac_of_bwfloor": live_gbs / bw_gbs,
        }
        rows.append(row)
        print(
            f"L={L:5d} B={B:6d} B/E={row['B_over_E']:5.1f} used={used_n:3d} "
            f"M med/p90/max {med:3d}/{row['p90_M']:4d}/{used[-1]:5d} | "
            f"live {t_live*1e3:7.2f}ms {live_gbs:6.1f} GB/s | "
            f"ceil {t_ceil*1e3:7.2f}ms {ceil_gbs:6.1f} | bwfloor {bw_gbs:6.1f} | "
            f"headroom {row['headroom_x']:.2f}x  live/ceil {row['live_frac_of_ceiling']*100:5.1f}%"
        )

    env = {k: os.environ.get(k) for k in
           ("MLX_GATHER_QMV_RHS", "MLX_GATHER_QMV_RHS_MAXBE",
            "MLX_GATHER_QMV_RHS_TILE", "MLX_GATHER_QMV_RHS_RPS")}
    blob = {"device": mx.device_info()["device_name"], "env": env,
            "skew": args.skew,
            "shape": {"experts": N_EXPERTS, "hidden": HIDDEN, "inter": INTER,
                      "top_k": TOP_K, "group": GROUP, "bits": BITS,
                      "mode": MODE},
            "rows": rows}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(blob, f, indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
