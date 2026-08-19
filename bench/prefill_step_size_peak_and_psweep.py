#!/usr/bin/env python3
"""Peak-memory + P-sweep follow-up to prefill_step_size_stage_ab.py.

Q1: does any indexer stage's per-TOKEN cost cross over and become worse at
    L=2048 as P grows toward the end of a long prefill?
Q2: how much does the per-chunk PEAK allocation grow at L=2048 vs L=1024?
    (relevant because EXO_PREFILL_CLEAR_CACHE_INTERVAL=1 drops the pool
    every chunk, so peak is re-acquired from the allocator each time.)
"""
import argparse
import time
import mlx.core as mx

D, H, TOPK = 128, 64, 512
R4, R128 = 21, 20


def timeit(fn, reps=5):
    for _ in range(2):
        mx.eval(fn())
    mx.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


def run(L, P, reps):
    q = mx.random.normal((1, H, L, D)).astype(mx.bfloat16)
    pooled = mx.random.normal((1, P, D)).astype(mx.bfloat16)
    wx = mx.random.normal((1, L, H)).astype(mx.bfloat16)
    mx.eval(q, pooled, wx)

    def score():
        w = mx.sigmoid(wx) * (D ** -0.5 * H ** -0.5)
        qw = (w[..., None] * q.transpose(0, 2, 1, 3)).sum(axis=2)
        return qw @ pooled.swapaxes(-1, -2)

    t_score = timeit(score, reps)
    scores = score()
    mx.eval(scores)
    k = min(TOPK, P)

    mx.clear_cache()
    mx.reset_peak_memory()
    t_topk = timeit(lambda: mx.argsort(-scores, axis=-1)[..., :k], reps)
    peak = mx.get_peak_memory() / 1e9
    del scores
    mx.clear_cache()
    return t_score, t_topk, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    print(f"{'ctx':>8}{'P':>8}{'L':>6}{'score us/tok':>14}"
          f"{'topk us/tok':>13}{'topk peak GB':>14}")
    for ctx in (50000, 100000, 190000, 300000):
        P = ctx // 4
        for L in (1024, 2048):
            s, t, pk = run(L, P, args.reps)
            print(f"{ctx:>8}{P:>8}{L:>6}{s/L*1e6:>14.2f}"
                  f"{t/L*1e6:>13.2f}{pk:>14.3f}")
        print()


if __name__ == "__main__":
    main()
