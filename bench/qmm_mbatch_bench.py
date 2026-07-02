#!/usr/bin/env python3
"""Does M-batching same-expert rows recover gather_qmm bandwidth?

Variant A (production): x (1536, 1, 4096), rhs_indices (1536,) — M=1/pair.
Variant B (M-batched): x (256, 6, 4096), rhs_indices (256,) — M=6/expert.
Same total math. Asserts bit-identical outputs (row reshuffle only), then
compares wall. If B ~3x faster, segmented dispatch is the win.
Run ON A STUDIO: .venv/bin/python /tmp/qmm_mbatch_bench.py
"""
import time

import mlx.core as mx

TOKENS = 256
TOPK = 6
N_EXPERTS = 256
HIDDEN = 4096
INTER = 1024
GROUP = 32
BITS = 4
MODE = "mxfp4"
N_ITERS = 30
N_WARM = 5


def bench(fn):
    for _ in range(N_WARM):
        mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - t0) / N_ITERS


def main():
    mx.random.seed(0)
    n_pairs = TOKENS * TOPK  # 1536
    reps = n_pairs // N_EXPERTS  # 6 rows per expert, uniform for the bench
    w = mx.quantize(
        mx.random.normal((N_EXPERTS, INTER, HIDDEN), dtype=mx.float32) * 0.02,
        group_size=GROUP, bits=BITS, mode=MODE,
    )
    x_rows = mx.random.normal((n_pairs, HIDDEN), dtype=mx.bfloat16)

    # A: production layout — (n_pairs, 1, hidden), one expert index per row
    idx_a = mx.sort(mx.concatenate([mx.arange(N_EXPERTS, dtype=mx.uint32)] * reps))
    x_a = x_rows[:, None, :]

    def run_a():
        return mx.gather_qmm(x_a, *w, rhs_indices=idx_a, transpose=True,
                             group_size=GROUP, bits=BITS, mode=MODE,
                             sorted_indices=True)

    # B: M-batched — (n_experts, reps, hidden), one index per expert
    x_b = x_rows.reshape(N_EXPERTS, reps, HIDDEN)  # rows already expert-sorted
    idx_b = mx.arange(N_EXPERTS, dtype=mx.uint32)

    def run_b():
        return mx.gather_qmm(x_b, *w, rhs_indices=idx_b, transpose=True,
                             group_size=GROUP, bits=BITS, mode=MODE,
                             sorted_indices=True)

    # Bit-exactness: same rows, same experts, same per-row dot products.
    out_a = run_a()  # (n_pairs, 1, INTER)
    out_b = run_b()  # (n_experts, reps, INTER)
    mx.eval(out_a, out_b)
    same = bool(mx.all(out_a.reshape(N_EXPERTS, reps, INTER) == out_b).item())
    print(f"bit-identical: {same}")
    if not same:
        diff = mx.max(mx.abs(out_a.reshape(N_EXPERTS, reps, INTER).astype(mx.float32)
                             - out_b.astype(mx.float32))).item()
        print(f"  max abs diff: {diff:.3e}")

    dt_a = bench(run_a)
    dt_b = bench(run_b)
    wbytes = sum(a.nbytes for a in w)
    print(f"A (M=1 x {n_pairs} pairs):  {dt_a*1000:7.2f} ms  "
          f"{wbytes/dt_a/1e9:5.0f} GB/s")
    print(f"B (M={reps} x {N_EXPERTS} experts): {dt_b*1000:7.2f} ms  "
          f"{wbytes/dt_b/1e9:5.0f} GB/s")
    print(f"speedup: {dt_a/dt_b:.2f}x")


if __name__ == "__main__":
    main()
