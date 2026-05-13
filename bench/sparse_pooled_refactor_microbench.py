#!/usr/bin/env python3
"""SDPA refactor microbench v2 — investigate numerical mismatch.

The v1 microbench showed 1.25x speedup but ~2% relative diff vs current
code. This version isolates WHERE the diff comes from:
  1. bf16 rounding noise (expected, small)
  2. softmax-implementation difference (one-shot vs split + logaddexp)
  3. anything else surprising

Also tests with realistic attn_sink values (not all zeros) since
zero sinks are a degenerate case.
"""
from __future__ import annotations
import argparse
import time
import mlx.core as mx
from functools import partial


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


# Same as production
@partial(mx.compile, shapeless=True)
def current_sparse_pooled(q_scaled, local_kv, pooled_gathered, local_mask, pooled_mask, sinks_expanded):
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
        normalizer, local_scores, pooled_scores, sinks_expanded,
    )

    out = local_weights @ local_kv
    pw_bl = pooled_weights.transpose(0, 2, 1, 3)
    out = out + (pw_bl @ pooled_sq).transpose(0, 2, 1, 3)
    return out.astype(q_scaled.dtype)


def proposed_sdpa_single(q, local_kv, pooled_gathered, local_mask, pooled_mask, scale, sinks):
    """L_q=1 fast path."""
    B, H, L_q, D = q.shape
    assert L_q == 1
    pooled_kv = pooled_gathered.squeeze(2)
    combined_kv = mx.concatenate([local_kv, pooled_kv], axis=2)
    combined_mask = None
    if local_mask is not None or pooled_mask is not None:
        sw = local_kv.shape[2]
        k = pooled_kv.shape[2]
        lm = local_mask if local_mask is not None else mx.zeros((B, H, L_q, sw), dtype=q.dtype)
        pm = pooled_mask if pooled_mask is not None else mx.zeros((B, H, L_q, k), dtype=q.dtype)
        combined_mask = mx.concatenate([lm, pm], axis=-1)
    return mx.fast.scaled_dot_product_attention(
        q, combined_kv, combined_kv,
        scale=scale,
        mask=combined_mask,
        sinks=sinks,
    )


def reference_singlematmul(q, local_kv, pooled_gathered, scale, sinks):
    """Reference: hand-rolled fp32 single-softmax over combined K/V.
    
    This is what fast.sdpa SHOULD be computing. Comparing this against the
    current implementation tells us if the current code itself differs from
    a "true" single-softmax math.
    """
    B, H, L_q, D = q.shape
    pooled_kv = pooled_gathered.squeeze(2)
    combined_kv = mx.concatenate([local_kv, pooled_kv], axis=2)  # (B, 1, sw+k, D)
    
    # Manual SDPA in fp32 for max precision
    q32 = (q * scale).astype(mx.float32)
    k32 = combined_kv.astype(mx.float32)
    v32 = combined_kv.astype(mx.float32)
    
    # scores: (B, H, L_q, sw+k)
    scores = q32 @ k32.swapaxes(-1, -2)
    # Add sinks as a virtual extra column to the softmax denominator
    if sinks is not None:
        # sinks shape (H,) -> (1, H, 1, 1) -> append to scores last dim
        s = sinks.astype(mx.float32)[None, :, None, None]  # (1, H, 1, 1)
        s_broadcast = mx.broadcast_to(s, scores.shape[:-1] + (1,))
        ext = mx.concatenate([scores, s_broadcast], axis=-1)  # (B, H, L_q, sw+k+1)
        weights = mx.softmax(ext, axis=-1)
        weights = weights[..., :-1]  # drop the sink column from weights for output
    else:
        weights = mx.softmax(scores, axis=-1)
    
    out = weights @ v32
    return out.astype(q.dtype)


def time_variant(fn, args, n_warmup=10, n_trials=100):
    for _ in range(n_warmup):
        out = fn(*args)
        mx.eval(out)
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        out = fn(*args)
        mx.eval(out)
        times.append((time.perf_counter() - t0) * 1000)
    return out, times


def diff_stats(label, ref, alt):
    diff = (ref.astype(mx.float32) - alt.astype(mx.float32)).abs()
    max_d = float(diff.max())
    mean_d = float(diff.mean())
    ref_scale = float(ref.astype(mx.float32).abs().max())
    rel = max_d / max(ref_scale, 1e-9)
    print(f"  {label:35s} max={max_d:.4g} mean={mean_d:.4g} rel={rel:.4g} ref_scale={ref_scale:.4g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=200)
    ap.add_argument("--n-warmup", type=int, default=20)
    ap.add_argument("--sinks-mode", choices=["zeros", "small", "realistic"], default="realistic")
    args = ap.parse_args()
    
    B, H, L_q, D = 1, 64, 1, 128
    sw = 128
    k = 160
    
    mx.random.seed(42)  # reproducibility
    q = mx.random.normal(shape=(B, H, L_q, D)).astype(mx.bfloat16)
    local_kv = mx.random.normal(shape=(B, 1, sw, D)).astype(mx.bfloat16)
    pooled_gathered = mx.random.normal(shape=(B, 1, L_q, k, D)).astype(mx.bfloat16)
    
    if args.sinks_mode == "zeros":
        sinks_bf16 = mx.zeros((H,), dtype=mx.bfloat16)
        sinks_fp32 = mx.zeros((H,), dtype=mx.float32)
    elif args.sinks_mode == "small":
        sinks_fp32 = (mx.random.normal(shape=(H,)) * 0.01).astype(mx.float32)
        sinks_bf16 = sinks_fp32.astype(mx.bfloat16)
    else:  # realistic — DSv4 attn_sink starts at zeros but learnable, so values can be ~1
        sinks_fp32 = (mx.random.normal(shape=(H,)) * 0.5).astype(mx.float32)
        sinks_bf16 = sinks_fp32.astype(mx.bfloat16)
    
    scale = D ** -0.5
    q_scaled = q * scale
    sinks_expanded_fp32 = sinks_fp32[None, :, None, None]
    
    print(f"Shapes: q={q.shape} local_kv={local_kv.shape} pooled={pooled_gathered.shape}")
    print(f"sinks_mode={args.sinks_mode}")
    print(f"Trials: {args.n_trials} warmup={args.n_warmup}\n")
    
    # Reference fp32 math
    out_ref, _ = time_variant(
        reference_singlematmul,
        (q, local_kv, pooled_gathered, scale, sinks_fp32),
        5, 5,
    )
    
    # Current
    out_a, times_a = time_variant(
        current_sparse_pooled,
        (q_scaled, local_kv, pooled_gathered, None, None, sinks_expanded_fp32),
        args.n_warmup, args.n_trials,
    )
    mean_a = sum(times_a) / len(times_a)
    p50_a = sorted(times_a)[len(times_a) // 2]
    print(f"{'current (split-softmax bf16)':32s} mean={mean_a:.4f}ms p50={p50_a:.4f}ms speedup=1.00x")
    
    # Proposed
    out_b, times_b = time_variant(
        proposed_sdpa_single,
        (q, local_kv, pooled_gathered, None, None, scale, sinks_bf16),
        args.n_warmup, args.n_trials,
    )
    mean_b = sum(times_b) / len(times_b)
    p50_b = sorted(times_b)[len(times_b) // 2]
    sp = mean_a / mean_b
    print(f"{'proposed (fast.sdpa)':32s} mean={mean_b:.4f}ms p50={p50_b:.4f}ms speedup={sp:.2f}x")
    
    print(f"\n=== Numerical analysis vs fp32 reference ===")
    diff_stats("current vs fp32 ref", out_ref, out_a)
    diff_stats("proposed vs fp32 ref", out_ref, out_b)
    diff_stats("current vs proposed", out_a, out_b)
    
    # bf16 epsilon ~ 1/256 ~ 0.004; relative diff at this scale is expected
    print(f"\nbf16 epsilon ≈ 0.004 (1/256)")
    print(f"max abs diff between current and proposed is bf16-noise-scale if rel ~ 0.01-0.05")


if __name__ == "__main__":
    main()
