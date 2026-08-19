#!/usr/bin/env python3
"""Allocator A/B: does EXO_PREFILL_CLEAR_CACHE_INTERVAL=1 cost more per
token at L=2048 than L=1024?

At L=2048 the per-chunk peak transient (indexer topk over (B,L,P)) is
exactly 2x larger (measured 1.99GB vs 1.00GB at 190K ctx). With
clear_cache firing EVERY chunk, the Metal allocator must release and
re-acquire that whole working set each chunk. Half as many clears, but
each one re-acquires 2x the bytes -- net per-token cost is what matters.

Simulates one prefill chunk: allocate the big transient, do the topk,
then clear_cache (interval=1) or not (interval=0).
"""
import argparse
import time
import mlx.core as mx

D, TOPK = 128, 512


def chunk_cycle(L, P, clear):
    scores = mx.random.normal((1, L, P)).astype(mx.float32)
    idx = mx.argsort(-scores, axis=-1)[..., :min(TOPK, P)]
    mx.eval(idx)
    if clear:
        mx.clear_cache()
    return idx


def bench(L, P, clear, chunks):
    for _ in range(2):
        chunk_cycle(L, P, clear)
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(chunks):
        chunk_cycle(L, P, clear)
    mx.synchronize()
    return (time.perf_counter() - t0) / (chunks * L) * 1e6  # us/token


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=190000)
    ap.add_argument("--chunks", type=int, default=8)
    args = ap.parse_args()
    P = args.ctx // 4
    print(f"ctx={args.ctx} P={P}  ({args.chunks} chunks/config)\n")
    print(f"{'L':>6}{'clear=1 us/tok':>17}{'clear=0 us/tok':>17}"
          f"{'clear overhead':>16}")
    res = {}
    for L in (1024, 2048):
        c1 = bench(L, P, True, args.chunks)
        c0 = bench(L, P, False, args.chunks)
        res[L] = (c1, c0)
        print(f"{L:>6}{c1:>17.2f}{c0:>17.2f}{(c1-c0):>15.2f}u")
    print()
    for tag, i in (("clear=1", 0), ("clear=0", 1)):
        a, b = res[1024][i], res[2048][i]
        print(f"{tag}: L=2048 vs L=1024 per-token -> {(b/a-1)*100:+.1f}%")


if __name__ == "__main__":
    main()
