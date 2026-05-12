#!/usr/bin/env python3
"""Microbench for DSv4 _indexer_score hot path.

Times the score+collapse computation at realistic decode shapes for c=1, 100K ctx:
  q:       [1, 64, 3, 128]    bf16  (B, n_heads, L_q, head_dim)
  pooled:  [1, 25000, 128]    bf16  (B, L_pool, head_dim)
  weights: [1, 3, 64]         bf16  (B, L_q, n_heads)

Each decode cycle invokes this on ~21 indexer-equipped layers.

Usage:
  uv run python3 bench/indexer_score_microbench.py
  uv run python3 bench/indexer_score_microbench.py --pool-size 25000 --n-trials 200
"""
from __future__ import annotations

import argparse
import time

import mlx.core as mx


def baseline_score(
    q: mx.array,
    pooled: mx.array,
    weights_x: mx.array,
    scale: float,
    n_heads_inv_sqrt: float,
) -> mx.array:
    """Current implementation from mlx_lm/models/deepseek_v4.py:_indexer_score."""
    qf = q.astype(mx.float32)
    pf = pooled[:, None].astype(mx.float32)
    scores = qf @ pf.swapaxes(-1, -2)
    scores = mx.maximum(scores, 0) * scale
    w = weights_x.astype(mx.float32) * n_heads_inv_sqrt
    return (scores * w.swapaxes(-1, -2)[..., None]).sum(axis=1)


def variant_bf16_score(
    q: mx.array,
    pooled: mx.array,
    weights_x: mx.array,
    scale: float,
    n_heads_inv_sqrt: float,
) -> mx.array:
    """Drop fp32 cast — keep bf16 throughout.

    Hypothesis: the .astype(mx.float32) at every step forces upcast and
    consumes both compute and memory bandwidth. bf16 should be sufficient
    for ranking-by-score (we only care about argpartition order).
    """
    pf = pooled[:, None]
    scores = q @ pf.swapaxes(-1, -2)
    scores = mx.maximum(scores, 0) * scale
    w = weights_x * n_heads_inv_sqrt
    return (scores * w.swapaxes(-1, -2)[..., None]).sum(axis=1)


def variant_einsum(
    q: mx.array,
    pooled: mx.array,
    weights_x: mx.array,
    scale: float,
    n_heads_inv_sqrt: float,
) -> mx.array:
    """Fuse the matmul + weighted-sum into a single einsum.

    score[b, q, p] = sum_{h,d} q[b,h,q,d] * pooled[b,p,d] * scale
                              * weights_x[b,q,h] * n_heads_inv_sqrt * max-relu-mask

    We can't easily fuse the maximum(scores, 0) into einsum, but we can
    fuse the .sum(axis=1) over heads with the weights multiplication.
    """
    qf = q.astype(mx.float32)
    pf = pooled.astype(mx.float32)
    # scores[b, h, q, p]
    scores = mx.einsum("bhqd,bpd->bhqp", qf, pf)
    scores = mx.maximum(scores, 0) * scale
    w = weights_x.astype(mx.float32) * n_heads_inv_sqrt
    # Final: sum over h: scores[b,h,q,p] * w[b,q,h]
    return mx.einsum("bhqp,bqh->bqp", scores, w)


def variant_einsum_bf16(
    q: mx.array,
    pooled: mx.array,
    weights_x: mx.array,
    scale: float,
    n_heads_inv_sqrt: float,
) -> mx.array:
    """Einsum fusion + bf16 throughout (both optimizations combined)."""
    scores = mx.einsum("bhqd,bpd->bhqp", q, pooled)
    scores = mx.maximum(scores, 0) * scale
    w = weights_x * n_heads_inv_sqrt
    return mx.einsum("bhqp,bqh->bqp", scores, w)


@mx.compile
def variant_compile_bf16(
    q: mx.array,
    pooled: mx.array,
    weights_x: mx.array,
    scale: float,
    n_heads_inv_sqrt: float,
) -> mx.array:
    """mx.compile-wrapped bf16 einsum — let MLX fuse if it can."""
    scores = mx.einsum("bhqd,bpd->bhqp", q, pooled)
    scores = mx.maximum(scores, 0) * scale
    w = weights_x * n_heads_inv_sqrt
    return mx.einsum("bhqp,bqh->bqp", scores, w)


def variant_single_matmul(
    q: mx.array,
    pooled: mx.array,
    weights_x: mx.array,
    scale: float,
    n_heads_inv_sqrt: float,
) -> mx.array:
    """Pre-multiply weights into q to collapse the head dimension early.

    Math:
      score[b,q,p] = sum_h w[b,q,h] * relu(scale * sum_d q[b,h,q,d] * p[b,p,d])
                   = sum_h w[b,q,h] * relu(scale * (q @ p^T)[b,h,q,p])

    The relu prevents perfectly clean fusion, but if we compute relu in
    bf16 right after the matmul we still avoid the upcast hop.

    Actually this is essentially what einsum+bf16 does, just spelled out.
    Keep for sanity-check.
    """
    pf = pooled[:, None]  # [B, 1, L_pool, head_dim]
    scores = q @ pf.swapaxes(-1, -2)
    scores = mx.maximum(scores, 0) * scale
    w = weights_x * n_heads_inv_sqrt
    return (scores * w.swapaxes(-1, -2)[..., None]).sum(axis=1)


def time_variant(fn, q, pooled, weights_x, scale, nhi, n_trials: int, n_warmup: int = 10):
    # Warmup
    for _ in range(n_warmup):
        out = fn(q, pooled, weights_x, scale, nhi)
        mx.eval(out)
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        out = fn(q, pooled, weights_x, scale, nhi)
        mx.eval(out)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return out, times


def numerical_check(ref, alt, name: str, atol: float = 0.01) -> bool:
    ref_f = ref.astype(mx.float32)
    alt_f = alt.astype(mx.float32)
    diff = (ref_f - alt_f).abs()
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    rel = max_diff / max(float(ref_f.abs().max()), 1e-9)
    ok = rel < atol
    print(f"  numerical_check {name}: max={max_diff:.4g} mean={mean_diff:.4g} rel={rel:.4g} {'OK' if ok else 'MISMATCH'}")
    # Bigger thing: do argpartition results match?
    k = 192
    ref_topk = mx.argpartition(-ref_f, kth=k - 1, axis=-1)[..., :k]
    alt_topk = mx.argpartition(-alt_f, kth=k - 1, axis=-1)[..., :k]
    ref_set = set(ref_topk.flatten().tolist())
    alt_set = set(alt_topk.flatten().tolist())
    overlap = len(ref_set & alt_set)
    total = len(ref_set | alt_set)
    print(f"  topk-{k} overlap: {overlap}/{total} = {100*overlap/total:.1f}%")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-size", type=int, default=25000, help="L_pool (100K ctx / compress_ratio=4 = 25000)")
    ap.add_argument("--n-trials", type=int, default=100)
    ap.add_argument("--n-warmup", type=int, default=10)
    args = ap.parse_args()

    B = 1
    n_heads = 64
    head_dim = 128
    L_q = 3  # MTP verify (gamma+1)
    L_pool = args.pool_size

    # Match the shapes from _indexer_score callers
    q = mx.random.normal(shape=(B, n_heads, L_q, head_dim)).astype(mx.bfloat16)
    pooled = mx.random.normal(shape=(B, L_pool, head_dim)).astype(mx.bfloat16)
    weights_x = mx.random.normal(shape=(B, L_q, n_heads)).astype(mx.bfloat16)
    scale = head_dim ** -0.5
    nhi = n_heads ** -0.5

    print(f"Shapes: q={q.shape} pooled={pooled.shape} weights_x={weights_x.shape}")
    print(f"Trials: {args.n_trials} warmup={args.n_warmup}")
    print()

    variants = [
        ("baseline (fp32)", baseline_score),
        ("bf16 throughout", variant_bf16_score),
        ("einsum (fp32)", variant_einsum),
        ("einsum+bf16", variant_einsum_bf16),
        ("compile(einsum+bf16)", variant_compile_bf16),
        ("single-matmul+bf16", variant_single_matmul),
    ]

    # Run baseline first to get reference
    ref_out, ref_times = time_variant(baseline_score, q, pooled, weights_x, scale, nhi, args.n_trials, args.n_warmup)
    baseline_mean = sum(ref_times) / len(ref_times)
    baseline_p50 = sorted(ref_times)[len(ref_times) // 2]
    print(f"{variants[0][0]:25s}  mean={baseline_mean:6.3f}ms  p50={baseline_p50:6.3f}ms  speedup=1.00x")

    for name, fn in variants[1:]:
        out, times = time_variant(fn, q, pooled, weights_x, scale, nhi, args.n_trials, args.n_warmup)
        mean = sum(times) / len(times)
        p50 = sorted(times)[len(times) // 2]
        speedup = baseline_mean / mean if mean > 0 else 0
        print(f"{name:25s}  mean={mean:6.3f}ms  p50={p50:6.3f}ms  speedup={speedup:.2f}x")
        numerical_check(ref_out, out, name)


if __name__ == "__main__":
    main()
