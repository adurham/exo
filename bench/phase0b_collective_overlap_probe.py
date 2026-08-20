"""PHASE 0b GATE: can GPU compute overlap an in-flight mx.distributed.all_sum?

Run with:  .venv/bin/mlx.launch -n 2 --backend ring bench/phase0b_collective_overlap_probe.py

Design
------
Three timed conditions, all on PRE-EVALUATED inputs so no fence dependency
exists between the compute graph and the collective graph:

  COMPUTE_ONLY   t_c  = eval(matmul chain)
  COMM_ONLY      t_m  = eval(all_sum(big))
  BOTH           t_b  = eval(matmul chain, all_sum(big))     [independent DAGs]

Decision rule (the gate):
  t_b ~= max(t_c, t_m)          -> OVERLAP WORKS
  t_b ~= t_c + t_m              -> DEVICE-WIDE DRAIN / serialization

Reported as overlap_ratio = t_b / max(t_c, t_m) and
serial_ratio = t_b / (t_c + t_m).

Everything is replicated n>=5 with median reported; the two ranks share ONE
GPU under loopback ring, which is a confound -- see the printed caveat.
"""

import argparse
import os
import statistics
import time

import mlx.core as mx


def _sync(*arrays: mx.array) -> None:
    mx.eval(*arrays)


def build_compute(x: mx.array, depth: int) -> mx.array:
    """A GPU-bound matmul chain. Serial dependency so it cannot be reordered."""
    a = x
    for _ in range(depth):
        a = a @ x
        a = a * 1.0001
    return a


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matmul-dim", type=int, default=2048)
    parser.add_argument("--matmul-depth", type=int, default=24)
    parser.add_argument("--comm-mb", type=float, default=64.0)
    parser.add_argument("--reps", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    world = mx.distributed.init()
    rank, size = world.rank(), world.size()
    if size < 2:
        print("INVALID: need >=2 ranks (all_sum is a no-op at size 1)")
        raise SystemExit(2)

    n_elem = int(args.comm_mb * 1024 * 1024 / 4)
    comm_src = mx.ones((n_elem,), dtype=mx.float32)
    mm_src = mx.random.normal((args.matmul_dim, args.matmul_dim)).astype(mx.float32)
    mm_src = mm_src / (args.matmul_dim**0.5)
    _sync(comm_src, mm_src)

    def t_compute() -> float:
        t0 = time.perf_counter()
        out = build_compute(mm_src, args.matmul_depth)
        _sync(out)
        return time.perf_counter() - t0

    def t_comm() -> float:
        t0 = time.perf_counter()
        out = mx.distributed.all_sum(comm_src, group=world)
        _sync(out)
        return time.perf_counter() - t0

    def t_both() -> float:
        t0 = time.perf_counter()
        c = build_compute(mm_src, args.matmul_depth)
        m = mx.distributed.all_sum(comm_src, group=world)
        _sync(c, m)
        return time.perf_counter() - t0

    # correctness proof from the same process as the measurement
    proof = mx.distributed.all_sum(comm_src, group=world)
    _sync(proof)
    ok = bool(mx.all(proof == float(size)).item())
    if not ok:
        print(f"INVALID rank{rank}: all_sum did not produce {size}s")
        raise SystemExit(2)

    for _ in range(args.warmup):
        t_compute()
        t_comm()
        t_both()
        # keep ranks in lockstep between phases
        _sync(mx.distributed.all_sum(mx.zeros((1,)), group=world))

    cs, ms, bs = [], [], []
    for _ in range(args.reps):
        _sync(mx.distributed.all_sum(mx.zeros((1,)), group=world))
        cs.append(t_compute())
        _sync(mx.distributed.all_sum(mx.zeros((1,)), group=world))
        ms.append(t_comm())
        _sync(mx.distributed.all_sum(mx.zeros((1,)), group=world))
        bs.append(t_both())

    med = statistics.median
    tc, tm, tb = med(cs), med(ms), med(bs)
    spread = lambda v: (max(v) - min(v)) / med(v)  # noqa: E731

    print(f"--- rank {rank}/{size}  pid={os.getpid()} ---")
    print(f"  matmul {args.matmul_dim}^2 x{args.matmul_depth} | comm {args.comm_mb} MB | reps={args.reps}")
    print(f"  COMPUTE_ONLY  median {tc*1e3:8.2f} ms   spread {spread(cs)*100:5.1f}%")
    print(f"  COMM_ONLY     median {tm*1e3:8.2f} ms   spread {spread(ms)*100:5.1f}%")
    print(f"  BOTH          median {tb*1e3:8.2f} ms   spread {spread(bs)*100:5.1f}%")
    print(f"  ideal-overlap max(c,m) = {max(tc,tm)*1e3:8.2f} ms")
    print(f"  full-serial   c+m      = {(tc+tm)*1e3:8.2f} ms")
    print(f"  overlap_ratio (b/max)  = {tb/max(tc,tm):.3f}   (1.0 = perfect overlap)")
    print(f"  serial_ratio  (b/sum)  = {tb/(tc+tm):.3f}   (1.0 = full serialization)")
    saved = (tc + tm) - tb
    print(f"  time saved vs serial   = {saved*1e3:8.2f} ms  ({saved/(tc+tm)*100:5.1f}%)")


if __name__ == "__main__":
    main()
