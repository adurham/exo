"""PHASE 0b GATE, part 3: interleaved A/B isolating the ONE variable.

Part 1 (pre-evaluated collective input) showed near-perfect overlap.
Part 2 (collective input produced by a GPU op) showed full serialization.
Those were separate processes/runs. This probe runs BOTH arms interleaved in
one process at the same payload sizes, so the only difference is whether the
all_sum's input is already evaluated or is a GPU-produced array.

Arms:
  PREEVAL : all_sum(src)          where src was mx.eval'd beforehand
  GPUPROD : all_sum(src * 1.0)    GPU elementwise feeds the collective
  NULLDEP : all_sum(src) but a same-cost GPU elementwise is eval'd in the
            same graph WITHOUT feeding the collective -- controls for the
            elementwise op's own GPU cost.

Run: .venv/bin/mlx.launch -n 2 --backend ring .venv/bin/python bench/phase0b_collective_overlap_probe3.py
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
    p.add_argument("--reps", type=int, default=9)
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

    def t(fn) -> float:
        t0 = time.perf_counter()
        fn()
        return time.perf_counter() - t0

    print(f"--- rank {rank}/{size} | matmul {args.dim}^2 x{args.depth} | reps={args.reps} ---")
    hdr = f"{'MB':>6} {'arm':>8} {'comm':>8} {'compute':>8} {'BOTH':>8} {'max':>8} {'sum':>8} {'b/max':>7} {'b/sum':>7}"
    print(hdr)

    for mb in (16.0, 64.0):
        n = int(mb * 1024 * 1024 / 4)
        src = mx.ones((n,), dtype=mx.float32)
        mx.eval(src)

        arms = {
            "PREEVAL": (
                lambda s=src: mx.eval(mx.distributed.all_sum(s, group=world)),
                lambda s=src: mx.eval(
                    mx.distributed.all_sum(s, group=world), chain(mm, args.depth)
                ),
            ),
            "GPUPROD": (
                lambda s=src: mx.eval(mx.distributed.all_sum(s * 1.0, group=world)),
                lambda s=src: mx.eval(
                    mx.distributed.all_sum(s * 1.0, group=world), chain(mm, args.depth)
                ),
            ),
            "NULLDEP": (
                lambda s=src: mx.eval(mx.distributed.all_sum(s, group=world), s * 1.0),
                lambda s=src: mx.eval(
                    mx.distributed.all_sum(s, group=world), s * 1.0, chain(mm, args.depth)
                ),
            ),
        }

        compute_only = lambda: mx.eval(chain(mm, args.depth))  # noqa: E731

        for _ in range(3):
            compute_only()
            for c, b in arms.values():
                c()
                b()
        barrier()

        cs: list[float] = []
        res: dict[str, tuple[list[float], list[float]]] = {k: ([], []) for k in arms}
        for _ in range(args.reps):
            barrier()
            cs.append(t(compute_only))
            for name, (c, b) in arms.items():
                barrier()
                res[name][0].append(t(c))
                barrier()
                res[name][1].append(t(b))

        tc = med(cs)
        for name in arms:
            tm, tb = med(res[name][0]), med(res[name][1])
            print(
                f"{mb:6.0f} {name:>8} {tm*1e3:8.2f} {tc*1e3:8.2f} {tb*1e3:8.2f} "
                f"{max(tm,tc)*1e3:8.2f} {(tm+tc)*1e3:8.2f} {tb/max(tm,tc):7.3f} {tb/(tm+tc):7.3f}"
            )


if __name__ == "__main__":
    main()
