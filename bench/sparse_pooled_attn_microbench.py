#!/usr/bin/env python3
"""Microbench/correctness check for _sparse_pooled_attention rewrite.

Goal: find a shapeless-compatible rewrite of the take_along_axis pattern
in _sparse_pooled_attention so the whole function can be @mx.compile'd
(saves ~15-20 op constructions per call × ~21 indexer layers per cycle
× 256 decode steps = many CPU dispatch ops).

Compares:
  A. Current implementation (no compile)
  B. Current + @mx.compile shapeless (the May 9 attempt that crashed
     with a shapeless broadcast bug)
  C. Rewritten take_along_axis variant (broadcast-free) + @mx.compile
     shapeless
"""
from __future__ import annotations

import time
from functools import partial
from typing import Optional

import mlx.core as mx


# === Helper functions copied/adapted from deepseek_v4.py ===

def _apply_score_mask(scores, mask):
    if mask is None:
        return scores
    if mask.dtype == mx.bool_:
        return mx.where(mask, scores, mx.finfo(scores.dtype).min)
    return scores + mask.astype(scores.dtype)


@partial(mx.compile, shapeless=True)
def _split_softmax(log_normalizer, logits_a, logits_b, sinks=None):
    if sinks is not None:
        log_normalizer = mx.logaddexp(log_normalizer, sinks)
    weights_a = mx.exp(logits_a - log_normalizer)
    weights_b = mx.exp(logits_b - log_normalizer)
    return weights_a, weights_b


# === Variants ===

def baseline(q, local_kv, pooled, topk, local_mask, pooled_mask, scale, sinks):
    """Current implementation, no compile."""
    B, H, L, D = q.shape
    idx = topk[:, None, :, :, None]
    pooled = mx.take_along_axis(
        mx.broadcast_to(pooled[:, None, None], (B, 1, L, pooled.shape[1], D)),
        mx.broadcast_to(idx, idx.shape[:-1] + (D,)),
        axis=3,
    )
    q_scaled = q * scale
    local_scores = q_scaled @ local_kv.swapaxes(-1, -2)
    local_scores = _apply_score_mask(local_scores, local_mask)
    normalizer = mx.logsumexp(local_scores, -1, keepdims=True)
    pooled_sq = pooled.squeeze(1)
    q_bl = q_scaled.transpose(0, 2, 1, 3)
    pooled_scores = q_bl @ pooled_sq.swapaxes(-1, -2)
    pooled_scores = pooled_scores.transpose(0, 2, 1, 3)
    pooled_scores = _apply_score_mask(pooled_scores, pooled_mask)
    normalizer = mx.logaddexp(
        normalizer, mx.logsumexp(pooled_scores, -1, keepdims=True)
    )
    local_weights, pooled_weights = _split_softmax(
        normalizer, local_scores, pooled_scores,
        sinks[None, :, None, None] if sinks is not None else None,
    )
    out = local_weights @ local_kv
    pw_bl = pooled_weights.transpose(0, 2, 1, 3)
    out = out + (pw_bl @ pooled_sq).transpose(0, 2, 1, 3)
    return out.astype(q.dtype)


@partial(mx.compile, shapeless=True)
def compiled_shapeless(q, local_kv, pooled, topk, local_mask, pooled_mask, scale, sinks):
    """Same as baseline but @mx.compile shapeless — the May 9 attempt."""
    B, H, L, D = q.shape
    idx = topk[:, None, :, :, None]
    pooled = mx.take_along_axis(
        mx.broadcast_to(pooled[:, None, None], (B, 1, L, pooled.shape[1], D)),
        mx.broadcast_to(idx, idx.shape[:-1] + (D,)),
        axis=3,
    )
    q_scaled = q * scale
    local_scores = q_scaled @ local_kv.swapaxes(-1, -2)
    local_scores = _apply_score_mask(local_scores, local_mask)
    normalizer = mx.logsumexp(local_scores, -1, keepdims=True)
    pooled_sq = pooled.squeeze(1)
    q_bl = q_scaled.transpose(0, 2, 1, 3)
    pooled_scores = q_bl @ pooled_sq.swapaxes(-1, -2)
    pooled_scores = pooled_scores.transpose(0, 2, 1, 3)
    pooled_scores = _apply_score_mask(pooled_scores, pooled_mask)
    normalizer = mx.logaddexp(
        normalizer, mx.logsumexp(pooled_scores, -1, keepdims=True)
    )
    local_weights, pooled_weights = _split_softmax(
        normalizer, local_scores, pooled_scores,
        sinks[None, :, None, None] if sinks is not None else None,
    )
    out = local_weights @ local_kv
    pw_bl = pooled_weights.transpose(0, 2, 1, 3)
    out = out + (pw_bl @ pooled_sq).transpose(0, 2, 1, 3)
    return out.astype(q.dtype)


def rewritten_inner(q, local_kv, pooled_gathered, local_mask, pooled_mask, scale, sinks):
    """Inner kernel after pooled_gathered is pre-gathered.
    
    Once topk gather is done OUTSIDE compile, all shapes are static, so
    @mx.compile non-shapeless is safe.
    """
    B, H, L, D = q.shape
    q_scaled = q * scale
    local_scores = q_scaled @ local_kv.swapaxes(-1, -2)
    local_scores = _apply_score_mask(local_scores, local_mask)
    normalizer = mx.logsumexp(local_scores, -1, keepdims=True)
    pooled_sq = pooled_gathered.squeeze(1)
    q_bl = q_scaled.transpose(0, 2, 1, 3)
    pooled_scores = q_bl @ pooled_sq.swapaxes(-1, -2)
    pooled_scores = pooled_scores.transpose(0, 2, 1, 3)
    pooled_scores = _apply_score_mask(pooled_scores, pooled_mask)
    normalizer = mx.logaddexp(
        normalizer, mx.logsumexp(pooled_scores, -1, keepdims=True)
    )
    local_weights, pooled_weights = _split_softmax(
        normalizer, local_scores, pooled_scores,
        sinks[None, :, None, None] if sinks is not None else None,
    )
    out = local_weights @ local_kv
    pw_bl = pooled_weights.transpose(0, 2, 1, 3)
    out = out + (pw_bl @ pooled_sq).transpose(0, 2, 1, 3)
    return out.astype(q.dtype)


# Wrap inner with non-shapeless compile (k is fixed per workload at 192)
rewritten_inner_compiled = mx.compile(rewritten_inner)


def variant_static_inner(q, local_kv, pooled, topk, local_mask, pooled_mask, scale, sinks):
    """Pull take_along_axis OUTSIDE compile boundary, compile the rest."""
    B, H, L, D = q.shape
    idx = topk[:, None, :, :, None]
    pooled_gathered = mx.take_along_axis(
        mx.broadcast_to(pooled[:, None, None], (B, 1, L, pooled.shape[1], D)),
        mx.broadcast_to(idx, idx.shape[:-1] + (D,)),
        axis=3,
    )
    return rewritten_inner_compiled(q, local_kv, pooled_gathered, local_mask, pooled_mask, scale, sinks)


# === Benchmarking ===

def time_variant(fn, args, n_trials=50, n_warmup=10):
    try:
        for _ in range(n_warmup):
            out = fn(*args)
            mx.eval(out)
    except Exception as e:
        import traceback
        return None, None, f"{type(e).__name__}: {e}\n{traceback.format_exc()[-500:]}"
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        out = fn(*args)
        mx.eval(out)
        times.append((time.perf_counter() - t0) * 1000)
    return out, times, None


def main():
    # Realistic decode shapes for c=1 100K on a compress_ratio=4 layer
    B, n_heads, head_dim = 1, 64, 128
    L_q = 1  # decode step (or 3 under MTP)
    sliding_window = 128  # local KV size
    k = 192  # topk
    L_pool = 25000  # 100K context / compress_ratio=4

    q = mx.random.normal(shape=(B, n_heads, L_q, head_dim)).astype(mx.bfloat16)
    local_kv = mx.random.normal(shape=(B, 1, sliding_window, head_dim)).astype(mx.bfloat16)
    pooled = mx.random.normal(shape=(B, L_pool, head_dim)).astype(mx.bfloat16)
    topk = mx.random.randint(0, L_pool, shape=(B, L_q, k))
    local_mask = None
    pooled_mask = None
    scale = head_dim ** -0.5
    sinks = mx.zeros((n_heads,), dtype=mx.float32)

    args = (q, local_kv, pooled, topk, local_mask, pooled_mask, scale, sinks)

    print(f"Shapes: q={q.shape} local_kv={local_kv.shape} pooled={pooled.shape} topk={topk.shape}")
    print()

    variants = [
        ("baseline (no compile)", baseline),
        ("compile shapeless (May 9 attempt)", compiled_shapeless),
        ("static_inner compile (rewrite)", variant_static_inner),
    ]

    ref_out, ref_times, err = time_variant(baseline, args)
    if err:
        print(f"{'baseline (no compile)':36s}  ERROR: {err[:500]}")
        return
    bm = sum(ref_times) / len(ref_times)
    print(f"{'baseline (no compile)':36s}  mean={bm:6.3f}ms speedup=1.00x")

    for name, fn in variants[1:]:
        out, times, err = time_variant(fn, args)
        if err:
            print(f"{name:36s}  ERROR: {err[:80]}")
            continue
        m = sum(times) / len(times)
        sp = bm / m if m > 0 else 0
        diff = (ref_out.astype(mx.float32) - out.astype(mx.float32)).abs()
        max_d = float(diff.max())
        print(f"{name:36s}  mean={m:6.3f}ms speedup={sp:.2f}x  max_diff={max_d:.4g}")


if __name__ == "__main__":
    main()
