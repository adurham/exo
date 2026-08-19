#!/usr/bin/env python3
"""Isolated per-stage A/B of EXO_PREFILL_STEP_SIZE 2048 vs 4096.

Measures, at a REALISTIC pooled-context P (partway through a ~190K prefill),
the per-TOKEN cost of each candidate stage at the two real per-rank chunk
sizes (step_size // 2 on a 2-rank cluster: 1024 vs 2048).

Because doubling L halves the chunk COUNT at fixed total context, the honest
metric is cost-per-token, not cost-per-chunk. A stage only explains an
end-to-end regression if its cost-per-token goes UP at the larger L.

Stages:
  indexer_score  -- folded (B,L,D)@(B,D,P) GEMM (post-OPT-6; the old
                    (B,H,L,P) transient no longer exists)
  indexer_pmask  -- tail-restricted causal pool mask apply
  indexer_topk   -- argsort/argpartition over (B,L,P)
  allgather_pay  -- SEQ_SPLIT reconstruction payload (contiguous copy of
                    the (B,L,hidden) activation; local stand-in for the
                    per-call collective payload)
  clear_cache    -- mx.clear_cache() amortized per token at the real
                    per-chunk cadence (EXO_PREFILL_CLEAR_CACHE_INTERVAL=1)

Usage:  python bench/prefill_step_size_stage_ab.py [--ctx 190000] [--reps 5]
"""
import argparse
import time
import mlx.core as mx

D = 128          # index_head_dim
H = 64           # index_n_heads
HIDDEN = 4096
TOPK = 512
RATIO4_LAYERS = 21   # compress_ratio == 4   -> P = ctx/4
RATIO128_LAYERS = 20  # compress_ratio == 128 -> P = ctx/128


def timeit(fn, reps, warmup=2):
    for _ in range(warmup):
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


def bench_indexer(L, P, reps):
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

    # tail-restricted pmask: only ~L/ratio+1 columns are row-dependent
    band = L // 4 + 1
    vis_min = max(P // 2, 0)
    vis_max = min(vis_min + band, P)
    pm = mx.random.uniform(shape=(L, vis_max - vis_min)) > 0.5

    def pmask():
        neg = mx.finfo(scores.dtype).min
        parts = [scores[..., :vis_min],
                 mx.where(pm[None], scores[..., vis_min:vis_max], neg)]
        if P > vis_max:
            parts.append(mx.full((1, L, P - vis_max), neg, dtype=scores.dtype))
        return mx.concatenate(parts, axis=-1)

    t_pmask = timeit(pmask, reps)

    k = min(TOPK, P)
    t_topk = timeit(lambda: mx.argsort(-scores, axis=-1)[..., :k], reps)
    del scores
    mx.clear_cache()
    return t_score, t_pmask, t_topk


def bench_allgather_payload(L, reps):
    a = mx.random.normal((1, L, HIDDEN)).astype(mx.bfloat16)
    mx.eval(a)
    # SEQ_SPLIT reconstruction = concatenating the two half-bands back
    half = L // 2
    x0, x1 = a[:, :half], a[:, half:]
    mx.eval(x0, x1)
    return timeit(lambda: mx.concatenate([x0, x1], axis=1), reps)


def bench_clear_cache(L, reps):
    # realistic working set for one chunk, then clear
    def f():
        buf = mx.random.normal((1, L, HIDDEN)).astype(mx.bfloat16)
        mx.eval(buf)
        mx.clear_cache()
        return mx.array(0)
    return timeit(f, reps, warmup=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=190000)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()

    ctx = args.ctx
    P4 = ctx // 4
    P128 = ctx // 128
    print(f"ctx={ctx}  P(ratio4)={P4}  P(ratio128)={P128}")
    print("Per-rank chunk L = EXO_PREFILL_STEP_SIZE // 2 (2-rank cluster)\n")

    rows = {}
    for step, L in ((2048, 1024), (4096, 2048)):
        s4, m4, k4 = bench_indexer(L, P4, args.reps)
        s128, m128, k128 = bench_indexer(L, P128, args.reps)
        ag = bench_allgather_payload(L, args.reps)
        cc = bench_clear_cache(L, args.reps)

        # per-chunk totals across all layers
        score = s4 * RATIO4_LAYERS + s128 * RATIO128_LAYERS
        pmask = m4 * RATIO4_LAYERS + m128 * RATIO128_LAYERS
        topk = k4 * RATIO4_LAYERS + k128 * RATIO128_LAYERS
        rows[step] = dict(L=L, indexer_score=score, indexer_pmask=pmask,
                          indexer_topk=topk, allgather_pay=ag,
                          clear_cache=cc)

    print(f"{'stage':<16}{'ms/tok @2048':>14}{'ms/tok @4096':>14}{'delta':>10}")
    print("-" * 54)
    for stage in ("indexer_score", "indexer_pmask", "indexer_topk",
                  "allgather_pay", "clear_cache"):
        a = rows[2048][stage] / rows[2048]["L"] * 1000
        b = rows[4096][stage] / rows[4096]["L"] * 1000
        print(f"{stage:<16}{a:>14.4f}{b:>14.4f}{(b/a - 1)*100:>9.1f}%")

    ta = sum(rows[2048][s] for s in rows[2048] if s != "L") / rows[2048]["L"]
    tb = sum(rows[4096][s] for s in rows[4096] if s != "L") / rows[4096]["L"]
    print("-" * 54)
    print(f"{'MEASURED TOTAL':<16}{ta*1000:>14.4f}{tb*1000:>14.4f}"
          f"{(tb/ta - 1)*100:>9.1f}%")


if __name__ == "__main__":
    main()
