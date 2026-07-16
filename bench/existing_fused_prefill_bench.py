"""Benchmark the EXISTING _sparse_fused_sdpa kernel at prefill scale (L=128).

The existing kernel (deepseek_v4.py:1538) does online-softmax fusion for L<=16.
Fable's question: can it scale to prefill L=128 at D=512?

This benchmark calls the existing kernel at L=128 and compares to the current
unfused path (_sparse_pooled_attention_inner) to see if the fusion wins at
prefill scale.

The kernel's constraint: sw+k_sel <= 768 (s_exp register array, 24 keys/simdgroup).
Production: sw=128, k=512, N=640 <= 768. OK.
L gate: currently <=16. We test at L=128 by calling the kernel directly.
"""
from __future__ import annotations

import statistics
import time

import mlx.core as mx

B, H, D, SW, K = 1, 64, 512, 128, 512
L_TILE = 128  # prefill tile (_SPARSE_SDPA_TILE)
SCALE = D ** -0.5
N_ITERS, N_WARMUP = 30, 8
DTYPE = mx.bfloat16

# Import the existing kernel builder from the production code
import sys

sys.path.insert(0, "/Users/adam.durham/repos/exo/mlx-lm")
import mlx_lm.models.deepseek_v4 as dsv4


def make_inputs():
    mx.random.seed(42)
    q = mx.random.normal((B, H, L_TILE, D), dtype=DTYPE)
    local_kv = mx.random.normal((B, 1, SW, D), dtype=DTYPE)
    P = 1024  # production pool size
    pooled = mx.random.normal((B, P, D), dtype=DTYPE)
    topk = mx.random.randint(0, P, (B, L_TILE, K), dtype=mx.int32)
    local_mask = mx.ones((B, 1, L_TILE, SW), dtype=mx.bool_)
    pooled_mask = mx.ones((B, H, L_TILE, K), dtype=mx.bool_)
    sinks = mx.random.normal((H,), dtype=DTYPE) * 0.1
    return q, local_kv, pooled, topk, local_mask, pooled_mask, sinks


def current_unfused(q, local_kv, pooled, topk, local_mask, pooled_mask, sinks):
    """The current prefill path: gather + _sparse_pooled_attention_inner."""
    # Gather
    P = pooled.shape[1]
    pooled_flat = pooled.reshape(B * P, D)
    offset = (mx.arange(B) * P).reshape(B, 1, 1)
    topk_flat = (topk + offset).reshape(-1)
    pooled_gathered = pooled_flat[topk_flat].reshape(B, L_TILE, K, D)
    pooled_gathered = pooled_gathered[:, None, :, :, :]  # (B, 1, L, k, D)
    sinks_exp = sinks[None, :, None, None]
    # Inner kernel (replicated)
    q_scaled = q * SCALE
    local_scores = q_scaled @ local_kv.swapaxes(-1, -2)
    local_scores = mx.where(local_mask, local_scores, mx.finfo(DTYPE).min)
    normalizer = mx.logsumexp(local_scores, -1, keepdims=True)
    pooled_sq = pooled_gathered.squeeze(1)
    q_bl = q_scaled.transpose(0, 2, 1, 3)
    pooled_scores = q_bl @ pooled_sq.swapaxes(-1, -2)
    pooled_scores = pooled_scores.transpose(0, 2, 1, 3)
    pooled_scores = mx.where(pooled_mask, pooled_scores, mx.finfo(DTYPE).min)
    normalizer = mx.logaddexp(normalizer, mx.logsumexp(pooled_scores, -1, keepdims=True))
    normalizer = mx.logaddexp(normalizer, sinks_exp)
    local_w = mx.exp(local_scores - normalizer)
    pooled_w = mx.exp(pooled_scores - normalizer)
    out = local_w @ local_kv
    pw_bl = pooled_w.transpose(0, 2, 1, 3)
    out = out + (pw_bl @ pooled_sq).transpose(0, 2, 1, 3)
    return out.astype(DTYPE)


def existing_fused(q, local_kv, pooled, topk, local_mask, pooled_mask, sinks):
    """Call the EXISTING _sparse_fused_sdpa kernel at L=128."""
    # The kernel needs: q (B,H,L,D), local_kv (B,1,sw,D), pooled (B,P,D),
    # topk (B,L,k), local_mask (B,1,L,sw) bool, pooled_mask (B,H,L,k) bool, scale, sinks
    return dsv4._sparse_fused_sdpa(
        q, local_kv, pooled, topk, local_mask, pooled_mask, SCALE, sinks
    )


def time_fn(fn, *args):
    for _ in range(N_WARMUP):
        out = fn(*args)
        mx.eval(out)
        mx.synchronize()
    samples = []
    for _ in range(N_ITERS):
        mx.synchronize()
        t0 = time.perf_counter()
        out = fn(*args)
        mx.eval(out)
        mx.synchronize()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples) * 1e6


def main():
    print("=" * 76)
    print("Existing fused kernel at prefill scale (L=128)")
    print(f"  Shape: B={B} H={H} L={L_TILE} D={D} sw={SW} k={K}")
    print(f"  N=sw+k={SW+K} (<=768 constraint: {'OK' if SW+K<=768 else 'FAIL'})")
    print("=" * 76)

    args = make_inputs()
    mx.eval(*args)

    # Numerical check
    ref = current_unfused(*args)
    cand = existing_fused(*args)
    mx.eval(ref, cand)
    if cand is None:
        print("  EXISTING KERNEL RETURNED NONE — contract miss at L=128")
        print("  (the kernel may have a shape guard that rejects L>16)")
        return
    diff = mx.abs(ref - cand)
    max_diff = float(mx.max(diff))
    ref_abs = mx.abs(ref) + 1e-5
    max_rel = float(mx.max(diff / ref_abs))
    print(f"  Numerical: max|d|={max_diff:.4e} max|d|/|r|={max_rel:.4e}")
    print("    (bf16 noise expected: max|d|~0.25, max|d|/|r|~0.07)")

    # Timing
    t_cur = time_fn(current_unfused, *args)
    t_fused = time_fn(existing_fused, *args)
    print(f"\n  Current unfused (gather+inner):  {t_cur:8.0f} µs")
    print(f"  Existing fused kernel (L=128):   {t_fused:8.0f} µs  ({t_cur/t_fused:.2f}x)")
    print("\n  Scaled to full layer (x16 tiles of 128 = 2048 L_q):")
    print(f"    current:   {t_cur*16:.0f} µs/layer")
    print(f"    fused:     {t_fused*16:.0f} µs/layer")
    print(f"    350 gate:  {10000*16:.0f} µs/layer (full layer)")
    print(f"    break-even: {12455*16:.0f} µs/layer")
    print()
    if t_fused < t_cur:
        print(f"  -> FUSED WINS at L=128: {t_cur/t_fused:.2f}x speedup")
        print("     This is the softmax-fusion path Fable identified!")
    else:
        print(f"  -> FUSED LOSES at L=128: {t_fused/t_cur:.2f}x slower")
        print("     The existing kernel does not scale to prefill L=128.")
    print("=" * 76)


if __name__ == "__main__":
    main()