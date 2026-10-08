#!/usr/bin/env python3
"""Read-bandwidth canary for the exo DSv4.1 campaign — the READ-side analogue of
gpu_canary2.py (which measures the 15.14 TFLOPS dense-matmul ceiling).

WHY. The Phase-4 P5 bytes-roofline (docs/benchmarks/phase20-throughput/
PHASE4-P5-ROOFLINE.md) divides a verify round's bytes by "~450 GB/s". That 450 is a
**triad** (read+write) figure from the loop-2 node calibration
(PERFORMANCE_HISTORY.md:11145). A pure-read kernel (what the verify weight-stream
actually is) can sit lower. This canary measures the achievable READ bandwidth so the
denominator is a measurement, not an anchor, on each node.

RUN LATER, ON AN IDLE CLUSTER NODE. Do not run while a production measurement is live —
any GPU work biases both this canary and the measurement. Pure mlx.core: no network, no
cluster, no file writes; everything goes to stdout.

METHOD (mirrors gpu_canary2.py's amortisation, adapted to a memory-bound kernel):
  * Arrays are large (default >= 1 GB working set) so caches are dwarfed and the traffic
    is DRAM/SLC-miss — the same regime as the verify path's weight stream.
  * Each timed eval enqueues K independent identical calls, so the ~150-250 us host/eval
    floor (phase3-kernel-microbench.md:36-46) divides out. Without this the small-kernel
    timing is the eval floor, not the kernel.
  * Three arms:
      1. pure READ      : mx.sum(x) over a big fp16 array           -> bytes read once
      2. read+write     : y = a*x + y over big arrays (triad-like)   -> reproduces 450
      3. GEMV-shaped    : x @ W, x is 1 row, W large (verify M=1)    -> the read the model does
  * Warm up 3 iters, then take the median of N reps.
  * Health gate: DEGRADED if pure-read < 0.6 * triad (stuck/thermally-throttled node).

Output: one JSON line so two nodes can be diffed.

Usage:
  python3 read_bw_canary.py                 # default sizes (>=1 GB set), K=64, N=5
  python3 read_bw_canary.py --mb 2048 --K 128 --reps 7
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import mlx.core as mx


def _median_gbps(byte_count: int, fn, K: int, reps: int) -> float:
    """Median GB/s of ``K`` amortised calls to ``fn`` per timed eval.

    ``fn`` returns a lazy mx array whose construction enqueues the kernel(s); ``mx.eval``
    drains. GB/s = byte_count * K / (median eval wall)."""
    # warm-up (JIT build + first-touch residency)
    for _ in range(3):
        mx.eval(fn())
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        for _ in range(K):
            out = fn()
        mx.eval(out)
        ts.append(byte_count * K / (time.perf_counter() - t) / 1e9)
    return statistics.median(ts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", type=int, default=1024,
                    help="per-array size in MB (default 1024 -> >=2 GB working set)")
    ap.add_argument("--K", type=int, default=64, help="amortisation iters per timed eval")
    ap.add_argument("--reps", type=int, default=5, help="timed reps (median taken)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    n_el = max(1, a.mb * 1024 * 1024 // 2)          # fp16 => 2 B/element
    side = 1
    while side * side < n_el:                        # square-ish for the matmul arm
        side += 1
    side = min(side, 16384)
    print(f"# read_bw_canary: {a.mb} MB/array, side={side}, K={a.K}, reps={a.reps}, "
          f"mlx={mx.__version__}")

    mx.random.seed(a.seed)
    x = mx.random.normal((side * side,)).astype(mx.float16)
    mx.eval(x)
    nbytes = x.size * x.itemsize

    # ---- arm 1: pure read (reduction) ----
    a1 = _median_gbps(nbytes, lambda: mx.sum(x), a.K, a.reps)

    # ---- arm 2: read+write (triad-like) ----
    y = mx.zeros_like(x)
    mx.eval(y)
    a2 = _median_gbps(3 * nbytes, lambda: x + y, a.K, a.reps)   # 2 reads + 1 write

    # ---- arm 3: GEMV-shaped read (verify M=1: x[1,D] @ W[D,N]) ----
    D, N = side, side
    W = mx.random.normal((D, N)).astype(mx.float16)
    xr = mx.random.normal((1, D)).astype(mx.float16)
    mx.eval(W, xr)
    wbytes = W.size * W.itemsize
    a3 = _median_gbps(wbytes, lambda: xr @ W, a.K, a.reps)

    healthy = a1 >= 0.6 * a2
    rec = {"arm1_read_gbps": round(a1, 1), "arm2_triad_gbps": round(a2, 1),
           "arm3_gemv_gbps": round(a3, 1), "read_over_triad": round(a1 / a2, 3),
           "healthy": healthy, "mb_per_array": a.mb, "K": a.K, "reps": a.reps}
    print(json.dumps(rec))
    print(f"# arm1 pure-read {a1:.1f} GB/s | arm2 triad {a2:.1f} | arm3 gemv {a3:.1f} "
          f"| read/triad {a1/a2:.2f} | {'HEALTHY' if healthy else 'DEGRADED (<0.6x triad)'}")
    print("# => use arm1 (pure read) as the P5 roofline denominator; 450 GB/s is arm2.")
    return 0 if healthy else 2


if __name__ == "__main__":
    raise SystemExit(main())
