"""PHASE 0b GATE, part 2: realistic overlap conditions.

Part 1 used a PRE-EVALUATED collective input. Real code does
    y = all_sum(f(x))   where f runs on the GPU
which inserts a cross-device fence. This probe asks whether INDEPENDENT GPU
compute still overlaps with the in-flight collective in that shape, and
sweeps payload size down to the few-MB regime real MoE/TP all_sums use.

Run: .venv/bin/mlx.launch -n 2 --backend ring .venv/bin/python bench/phase0b_collective_overlap_probe2.py
"""

import argparse
import statistics
import time

import mlx.core as mx


def chain(x: mx.array, depth: int) -> mx.array:
    a = x
    for _ in range(depth):
        a = (a @ x) * 1.0001
    return a


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, default=2048)
    p.add_argument("--depth", type=int, default=24)
    p.add_argument("--reps", type=int, default=7)
    args = p.parse_args()

    world = mx.distributed.init()
    rank, size = world.rank(), world.size()
    if size < 2:
        print("INVALID: need >=2 ranks")
        raise SystemExit(2)

    mm = mx.random.normal((args.dim, args.dim)).astype(mx.float32) / (args.dim**0.5)
    mx.eval(mm)
    med = statistics.median

    def barrier() -> None:
        mx.eval(mx.distributed.all_sum(mx.zeros((1,)), group=world))

    print(f"--- rank {rank}/{size} | matmul {args.dim}^2 x{args.depth} | reps={args.reps} ---")
    print(f"{'MB':>8} {'gpuprod+comm':>13} {'compute':>9} {'BOTH':>9} {'max':>9} {'sum':>9} {'b/max':>7} {'b/sum':>7}")

    for mb in (1.0, 4.0, 16.0, 64.0):
        n = int(mb * 1024 * 1024 / 4)
        src = mx.ones((n,), dtype=mx.float32)
        mx.eval(src)

        # collective whose input is produced on the GPU -> cross-device fence
        def comm_dep(src: mx.array = src) -> mx.array:
            gpu_made = src * 1.0  # GPU elementwise, must finish before all_sum
            return mx.distributed.all_sum(gpu_made, group=world)

        def t(fn) -> float:
            t0 = time.perf_counter()
            fn()
            return time.perf_counter() - t0

        def only_comm() -> None:
            mx.eval(comm_dep())

        def only_compute() -> None:
            mx.eval(chain(mm, args.depth))

        def both() -> None:
            m = comm_dep()
            c = chain(mm, args.depth)
            mx.eval(m, c)

        for _ in range(3):
            only_comm()
            only_compute()
            both()
        barrier()

        ms, cs, bs = [], [], []
        for _ in range(args.reps):
            barrier()
            ms.append(t(only_comm))
            barrier()
            cs.append(t(only_compute))
            barrier()
            bs.append(t(both))

        tm, tc, tb = med(ms), med(cs), med(bs)
        print(
            f"{mb:8.1f} {tm*1e3:13.2f} {tc*1e3:9.2f} {tb*1e3:9.2f} "
            f"{max(tm,tc)*1e3:9.2f} {(tm+tc)*1e3:9.2f} {tb/max(tm,tc):7.3f} {tb/(tm+tc):7.3f}"
        )


if __name__ == "__main__":
    main()
