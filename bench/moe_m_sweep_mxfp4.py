#!/usr/bin/env python3
"""M1: per-expert M sweep on the production mxfp4 switch-linear kernel.

Three ways to do the SAME math at each per-expert M (tokens/expert):

  A_prod : production prefill layout used by QuantizedSwitchLinear --
           x (E*M, 1, H), one expert index per row, sorted_indices=True.
           Dispatch -> mxfp4 gather_qmm_rhs (M==1 per pair).
  A_mb   : M-batched layout -- x (E, M, H), one index per expert.
           Dispatch -> generic gather_qmm (sorted_indices=False forced,
           because fp_gather_qmm_rhs_lhs is NOT instantiated in this build).
  B      : dense quantized_matmul, single expert weight, x (E*M, H).
           = "achievable dense ceiling" at this total row count.
  C      : bf16 dense GEMM, same shape = compute ceiling.

Shapes from the DSv4 production prefill path: 256 experts, hidden 4096,
moe_intermediate 1024, mxfp4 group 32.
"""

from __future__ import annotations

import argparse
import json
import time

import mlx.core as mx

N_EXPERTS = 256
HIDDEN = 4096
INTER = 1024
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


def try_bench(fn, iters, warm):
    try:
        return bench(fn, iters, warm)
    except Exception as exc:  # noqa: BLE001
        print(f"    [skip] {type(exc).__name__}: {str(exc)[:110]}")
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--warm", type=int, default=4)
    ap.add_argument("--out", type=str, default="bench/results_moe_m_sweep_mxfp4.json")
    ap.add_argument("--ms", type=str,
                    default="4,8,16,24,32,48,64,96,128,192,256,384,512")
    args = ap.parse_args()
    ms = [int(v) for v in args.ms.split(",")]

    mx.random.seed(0)
    wq = mx.quantize(
        mx.random.normal((N_EXPERTS, INTER, HIDDEN), dtype=mx.float32) * 0.02,
        group_size=GROUP, bits=BITS, mode=MODE,
    )
    wq1 = tuple(a[0] for a in wq)
    w_bf16 = mx.random.normal((INTER, HIDDEN), dtype=mx.bfloat16) * 0.02
    mx.eval(*wq, *wq1, w_bf16)

    n_w = N_EXPERTS * INTER * HIDDEN
    expert_bytes = n_w * 0.5 + (n_w // GROUP) * 1.0          # all experts once
    dense_bytes = (INTER * HIDDEN) * 0.5 + ((INTER * HIDDEN) // GROUP) * 1.0

    idx_mb = mx.arange(N_EXPERTS, dtype=mx.uint32)
    rows = []
    hdr = (f"{'M':>5} {'rows':>7} | {'prod us':>9} {'GB/s':>6} {'TF':>5} "
           f"| {'mb us':>8} {'GB/s':>6} {'TF':>5} "
           f"| {'dense us':>9} {'TF':>5} | {'bf16 TF':>7} "
           f"| {'prod/dns':>8} {'mb/dns':>7}")
    print(hdr)
    print("-" * len(hdr))
    for M in ms:
        tot = N_EXPERTS * M
        flops = 2.0 * tot * INTER * HIDDEN

        x_mb = mx.random.normal((N_EXPERTS, M, HIDDEN), dtype=mx.bfloat16)
        x_flat = x_mb.reshape(tot, HIDDEN)
        x_prod = x_flat[:, None, :]
        idx_prod = mx.repeat(mx.arange(N_EXPERTS, dtype=mx.uint32), M)
        mx.eval(x_mb, x_flat, x_prod, idx_prod)

        def run_prod():
            return mx.gather_qmm(x_prod, *wq, rhs_indices=idx_prod,
                                 transpose=True, group_size=GROUP, bits=BITS,
                                 mode=MODE, sorted_indices=True)

        def run_mb():
            return mx.gather_qmm(x_mb, *wq, rhs_indices=idx_mb,
                                 transpose=True, group_size=GROUP, bits=BITS,
                                 mode=MODE, sorted_indices=False)

        def run_dense():
            return mx.quantized_matmul(x_flat, *wq1, transpose=True,
                                       group_size=GROUP, bits=BITS, mode=MODE)

        def run_bf16():
            return x_flat @ w_bf16.T

        tp = try_bench(run_prod, args.iters, args.warm)
        tm = try_bench(run_mb, args.iters, args.warm)
        td = try_bench(run_dense, args.iters, args.warm)
        tc = try_bench(run_bf16, args.iters, args.warm)

        def tf(t):
            return None if t is None else flops / t / 1e12

        row = dict(
            M=M, rows=tot,
            prod_us=None if tp is None else tp * 1e6,
            prod_gbs=None if tp is None else expert_bytes / tp / 1e9,
            prod_tflops=tf(tp),
            mb_us=None if tm is None else tm * 1e6,
            mb_gbs=None if tm is None else expert_bytes / tm / 1e9,
            mb_tflops=tf(tm),
            dense_us=None if td is None else td * 1e6,
            dense_gbs=None if td is None else dense_bytes / td / 1e9,
            dense_tflops=tf(td),
            bf16_us=None if tc is None else tc * 1e6,
            bf16_tflops=tf(tc),
        )
        row["prod_over_dense"] = (
            None if not (tp and td) else row["prod_tflops"] / row["dense_tflops"])
        row["mb_over_dense"] = (
            None if not (tm and td) else row["mb_tflops"] / row["dense_tflops"])
        rows.append(row)

        def f(v, spec):
            return ("n/a".rjust(int(spec.split('.')[0].lstrip('>')))
                    if v is None else format(v, spec))
        print(
            f"{M:>5} {tot:>7} | {f(row['prod_us'],'>9.0f')} "
            f"{f(row['prod_gbs'],'>6.0f')} {f(row['prod_tflops'],'>5.2f')} "
            f"| {f(row['mb_us'],'>8.0f')} {f(row['mb_gbs'],'>6.0f')} "
            f"{f(row['mb_tflops'],'>5.2f')} "
            f"| {f(row['dense_us'],'>9.0f')} {f(row['dense_tflops'],'>5.2f')} "
            f"| {f(row['bf16_tflops'],'>7.2f')} "
            f"| {f(row['prod_over_dense'],'>8.2f')} "
            f"{f(row['mb_over_dense'],'>7.2f')}"
        )
        del x_mb, x_flat, x_prod, idx_prod

    with open(args.out, "w") as fh:
        json.dump({
            "device": str(mx.device_info()["device_name"]),
            "shape": dict(n_experts=N_EXPERTS, hidden=HIDDEN, inter=INTER,
                          group=GROUP, bits=BITS, mode=MODE),
            "expert_bytes": expert_bytes,
            "rows": rows,
        }, fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
