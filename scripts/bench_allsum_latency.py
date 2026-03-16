#!/usr/bin/env python3
"""Measure per-operation all_sum latency breakdown.

Must be run on a cluster node with TP active (needs mx.distributed group).
If no group available, measures the MLX dispatch overhead in isolation.

Usage (on a cluster node):
  uv run python scripts/bench_allsum_latency.py
"""
import time
import mlx.core as mx

HIDDEN = 4096  # Qwen3-235B hidden size
ITERS = 200
WARMUP = 20


def bench_eval_overhead():
    """Measure just mx.eval() overhead for a tiny operation."""
    x = mx.ones((1, HIDDEN), dtype=mx.bfloat16)
    mx.eval(x)

    times = []
    for _ in range(WARMUP):
        y = x + 0
        mx.eval(y)

    for _ in range(ITERS):
        y = x + 0  # trivial op
        mx.synchronize()
        t0 = time.perf_counter_ns()
        mx.eval(y)
        mx.synchronize()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1000)  # ns → μs

    times.sort()
    trimmed = times[10:-10]
    avg = sum(trimmed) / len(trimmed)
    p50 = trimmed[len(trimmed) // 2]
    p99 = trimmed[int(len(trimmed) * 0.99)]
    print(f"  mx.eval (trivial op):  avg={avg:.1f}μs  p50={p50:.1f}μs  p99={p99:.1f}μs")


def bench_matmul_eval():
    """Measure eval overhead for a small matmul (similar to what happens before all_sum)."""
    x = mx.ones((1, HIDDEN), dtype=mx.bfloat16)
    w = mx.ones((HIDDEN, HIDDEN), dtype=mx.bfloat16)
    mx.eval(x, w)

    times = []
    for _ in range(WARMUP):
        y = x @ w
        mx.eval(y)

    for _ in range(ITERS):
        mx.synchronize()
        t0 = time.perf_counter_ns()
        y = x @ w  # graph construction
        mx.eval(y)  # dispatch + execute
        mx.synchronize()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1000)

    times.sort()
    trimmed = times[10:-10]
    avg = sum(trimmed) / len(trimmed)
    p50 = trimmed[len(trimmed) // 2]
    print(f"  matmul (4096×4096):    avg={avg:.1f}μs  p50={p50:.1f}μs")


def bench_sequential_evals():
    """Measure cost of N sequential eval calls (simulates 188 all_sum dispatches)."""
    x = mx.ones((1, HIDDEN), dtype=mx.bfloat16)
    mx.eval(x)

    for count in [1, 10, 94, 188]:
        times = []
        for _ in range(WARMUP):
            for _ in range(count):
                y = x + 0
                mx.eval(y)
                x = y

        for _ in range(5):  # fewer outer iters for large counts
            x = mx.ones((1, HIDDEN), dtype=mx.bfloat16)
            mx.eval(x)
            mx.synchronize()
            t0 = time.perf_counter_ns()
            for _ in range(count):
                y = x + 0
                mx.eval(y)
                x = y
            mx.synchronize()
            t1 = time.perf_counter_ns()
            times.append((t1 - t0) / 1000)

        avg = sum(times) / len(times)
        per_op = avg / count
        print(f"  {count:>3} sequential evals:  total={avg:.0f}μs  per_op={per_op:.1f}μs")


def bench_allsum_if_available():
    """If distributed group exists, measure actual all_sum latency."""
    try:
        group = mx.distributed.init()
        if group.size() <= 1:
            print("\n  No distributed group (single node) — skipping all_sum benchmark")
            return

        print(f"\n  Distributed group: rank={group.rank()}, size={group.size()}")

        x = mx.ones((1, HIDDEN), dtype=mx.bfloat16)
        mx.eval(x)

        times = []
        for _ in range(WARMUP):
            y = mx.distributed.all_sum(x, group=group)
            mx.eval(y)

        for _ in range(ITERS):
            mx.synchronize()
            t0 = time.perf_counter_ns()
            y = mx.distributed.all_sum(x, group=group)
            mx.eval(y)
            mx.synchronize()
            t1 = time.perf_counter_ns()
            times.append((t1 - t0) / 1000)

        times.sort()
        trimmed = times[10:-10]
        avg = sum(trimmed) / len(trimmed)
        p50 = trimmed[len(trimmed) // 2]
        p99 = trimmed[int(len(trimmed) * 0.99)]
        print(f"  all_sum (8KB bf16):    avg={avg:.1f}μs  p50={p50:.1f}μs  p99={p99:.1f}μs")

        # Measure 188 sequential all_sums (full decode simulation)
        for count in [94, 188]:
            run_times = []
            for _ in range(3):
                x = mx.ones((1, HIDDEN), dtype=mx.bfloat16)
                mx.eval(x)
                mx.synchronize()
                t0 = time.perf_counter_ns()
                for _ in range(count):
                    x = mx.distributed.all_sum(x, group=group)
                    mx.eval(x)
                mx.synchronize()
                t1 = time.perf_counter_ns()
                run_times.append((t1 - t0) / 1000)
            avg = sum(run_times) / len(run_times)
            per_op = avg / count
            print(f"  {count:>3} sequential all_sum: total={avg/1000:.1f}ms  per_op={per_op:.1f}μs")

    except Exception as e:
        print(f"\n  Distributed not available: {e}")


def main():
    print("=== MLX Dispatch Overhead Benchmark ===")
    print(f"Hidden size: {HIDDEN}, dtype: bfloat16")
    print()

    print("1. Eval overhead (graph construction + dispatch + sync):")
    bench_eval_overhead()
    print()

    print("2. Matmul eval (small compute + overhead):")
    bench_matmul_eval()
    print()

    print("3. Sequential eval chain (simulates layer-by-layer dispatch):")
    bench_sequential_evals()
    print()

    print("4. All_sum latency (if distributed group available):")
    bench_allsum_if_available()


if __name__ == "__main__":
    main()
