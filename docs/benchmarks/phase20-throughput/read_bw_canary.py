#!/usr/bin/env python3
"""Read-bandwidth canary for the exo DSv4.1 campaign (READ-side, v2 — CSE-fixed).

WHY: the P5 bytes-roofline needs a MEASURED read bandwidth, not an assumed anchor.
The loop-2 "~450 GB/s" is a TRIAD (read+write) figure; a pure-read kernel can differ.

v1 BUG (fixed here): v1 amortised K calls of the SAME `mx.sum(x)` expression per timed
eval, which MLX constant-folds/reuses -> physically impossible ~30 TB/s. v2 times ONE
fresh `mx.eval` of a full-array reduction per rep (the array is far larger than the
~150-250 us eval floor, so no amortisation is needed or wanted).

Arms:
  1. pure READ      : mx.sum(x) over a large fp16 array            (bytes = nbytes)
  2. read+write     : y = x + y over large arrays                  (bytes = 2*nbytes)
  3. GEMV-shaped    : xr[1,D] @ W[D,N], W large (verify M=1 read)  (bytes = W bytes)

Run ON AN IDLE NODE (no production measurement in flight).
"""
from __future__ import annotations
import argparse, json, statistics, time
import mlx.core as mx


def _time_median(nbytes: int, fn, reps: int) -> float:
    for _ in range(2):
        mx.eval(fn())
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t)
    dt = statistics.median(ts)
    return nbytes / dt / 1e9


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    n_el = max(1, a.mb * 1024 * 1024 // 2)
    mx.random.seed(a.seed)
    x = mx.random.normal((n_el,)).astype(mx.float16); mx.eval(x)
    nb = x.size * x.itemsize

    r1 = _time_median(nb, lambda: mx.sum(x), a.reps)          # arm 1: pure read
    y = mx.zeros_like(x); mx.eval(y)
    r2 = _time_median(2 * nb, lambda: x + y, a.reps)          # arm 2: read x, write y
    D = 16384
    N = max(1024, nb // (D * 2))
    W = mx.random.normal((D, N)).astype(mx.float16)
    xr = mx.random.normal((1, D)).astype(mx.float16); mx.eval(W, xr)
    wb = W.size * W.itemsize
    r3 = _time_median(wb, lambda: xr @ W, a.reps)             # arm 3: GEMV read (M=1)

    sane = all(0 < v < 5000 for v in (r1, r2, r3))
    rec = {"arm1_read_gbps": round(r1, 1), "arm2_readwrite_gbps": round(r2, 1),
           "arm3_gemv_gbps": round(r3, 1), "mb_per_array": a.mb, "reps": a.reps,
           "sane": sane}
    print(f"# read_bw_canary v2: {a.mb} MB/array, reps={a.reps}, mlx={mx.__version__}")
    print(json.dumps(rec))
    print(f"# pure-read {r1:.1f} GB/s | read+write {r2:.1f} | gemv-read {r3:.1f} GB/s")
    print(f"# => P5 denominator: use arm1 (pure read)={r1:.0f} and arm3 (gemv)={r3:.0f} GB/s")
    return 0 if sane else 2


if __name__ == "__main__":
    raise SystemExit(main())
