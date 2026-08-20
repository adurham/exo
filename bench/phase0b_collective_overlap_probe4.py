"""PHASE 0b GATE, part 4: is the GPUPROD serialization a DEVICE-wide drain,
or merely single-GPU-stream FIFO ordering that a second stream can escape?

Part 3 proved: all_sum whose input is GPU-produced serializes against
independent GPU compute (b/sum ~= 1.01), while a pre-evaluated input overlaps
perfectly (b/max ~= 1.00). If the cause is that the cross-device fence sits
behind the matmul chain in ONE GPU stream's FIFO, then issuing the matmul on a
SECOND GPU stream should restore overlap. If it is a true device-wide drain,
a second stream changes nothing.

Arms at each size:
  GPUPROD_SAME  : all_sum(src*1.0) + matmul, both on the default GPU stream
  GPUPROD_ALT   : all_sum(src*1.0) on default; matmul on a second GPU stream

Run: .venv/bin/mlx.launch -n 2 --backend ring .venv/bin/python bench/phase0b_collective_overlap_probe4.py
"""

import argparse
import statistics
import time

import mlx.core as mx


def chain(x: mx.array, depth: int, stream: mx.Stream | None = None) -> mx.array:
    a = x
    for _ in range(depth):
        a = mx.multiply(mx.matmul(a, x, stream=stream), 1.0001, stream=stream)
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

    gpu_alt = mx.new_stream(mx.gpu)
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
    print(f"{'MB':>6} {'arm':>14} {'comm':>8} {'compute':>8} {'BOTH':>8} {'max':>8} {'sum':>8} {'b/max':>7} {'b/sum':>7}")

    for mb in (16.0, 64.0):
        n = int(mb * 1024 * 1024 / 4)
        src = mx.ones((n,), dtype=mx.float32)
        mx.eval(src)

        def comm(src: mx.array = src) -> mx.array:
            return mx.distributed.all_sum(src * 1.0, group=world)

        only_comm = lambda: mx.eval(comm())  # noqa: E731
        only_compute = lambda: mx.eval(chain(mm, args.depth))  # noqa: E731
        both_same = lambda: mx.eval(comm(), chain(mm, args.depth))  # noqa: E731
        both_alt = lambda: mx.eval(comm(), chain(mm, args.depth, gpu_alt))  # noqa: E731

        for _ in range(3):
            only_comm()
            only_compute()
            both_same()
            both_alt()
        barrier()

        ms, cs, bsame, balt = [], [], [], []
        for _ in range(args.reps):
            barrier()
            ms.append(t(only_comm))
            barrier()
            cs.append(t(only_compute))
            barrier()
            bsame.append(t(both_same))
            barrier()
            balt.append(t(both_alt))

        tm, tc = med(ms), med(cs)
        for name, vals in (("GPUPROD_SAME", bsame), ("GPUPROD_ALT", balt)):
            tb = med(vals)
            print(
                f"{mb:6.0f} {name:>14} {tm*1e3:8.2f} {tc*1e3:8.2f} {tb*1e3:8.2f} "
                f"{max(tm,tc)*1e3:8.2f} {(tm+tc)*1e3:8.2f} {tb/max(tm,tc):7.3f} {tb/(tm+tc):7.3f}"
            )


if __name__ == "__main__":
    main()
