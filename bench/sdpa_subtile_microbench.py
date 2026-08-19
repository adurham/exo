#!/usr/bin/env python3
"""Isolated SDPA sub-tiling microbenchmark (DECISION GATE for "Option A").

Question: when SEQ_SPLIT's per-rank query-row length L doubles 1024 -> 2048,
attn.sdpa costs 1.78x more ms/token and attn.sdpa.compressed 2.00x more
(live-cluster A/B, matched prompts). Is that a KERNEL-SHAPE effect that can be
recovered by tiling one L=2048 call into two sequential L=1024 calls, or is it
INTRINSIC (total FLOPs / memory traffic)?

Everything here is standalone MLX -- no exo, no model weights.

Shapes are taken from the real DeepSeek-V4-Flash config
(mlx-community/DeepSeek-V4-Flash config.json):
  num_attention_heads=64  -> 32 per TP rank (heads sharded by TP, NOT by
                             SEQ_SPLIT, which splits query ROWS only)
  head_dim=512, num_key_value_heads=1 (MLA single shared KV head)
  index_topk=512, sliding_window=128
  compress_ratios in {4, 128}

Two attention classes are benchmarked, each replicating the ACTUAL production
call site in mlx_lm/models/deepseek_v4.py:

1. SPARSE  (span "attn.sdpa", SparseCompressedAttention, pooled > index_topk):
   NOT mx.fast.scaled_dot_product_attention. It is `_sparse_pooled_attention`:
   an explicit matmul chain over (a) the 128-row local sliding window and
   (b) a per-query-row gathered top-512 pooled KV, joined by a split softmax.
   Production ALREADY tiles this over query rows at
   EXO_DSV4_SPARSE_SDPA_TILE=128, with a SINGLE gather for the whole L_q
   (EXO_DSV4_SINGLE_GATHER=1). Both variants are measured.

2. COMPRESSED (span "attn.sdpa.compressed", CompressedAttention):
   a single mx.fast.scaled_dot_product_attention over
   kv = concat(local sliding window, pooled) with an array mask.

Usage:  ~/repos/exo/.venv/bin/python bench/sdpa_subtile_microbench.py
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Optional

import mlx.core as mx

DTYPE = mx.bfloat16
N_HEADS_PER_RANK = 32  # 64 total // 2 TP ranks
HEAD_DIM = 512
INDEX_TOPK = 512
SLIDING_WINDOW = 128
SCALE = HEAD_DIM**-0.5


# ───────────────────────────── timing helper ─────────────────────────────
def timeit(fn, warmup: int = 6, iters: int = 25) -> float:
    """Median ms of `fn`, fully synchronized per iteration."""
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    samples = []
    for _ in range(iters):
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        samples.append((time.perf_counter() - t0) * 1e3)
    samples.sort()
    return samples[len(samples) // 2]


# ─────────────────────── sparse path (production copy) ───────────────────
def _sparse_inner(q_scaled, local_kv, pooled_gathered, sinks_expanded):
    """Byte-for-byte structure of _sparse_pooled_attention_inner (masks None)."""
    local_scores = q_scaled @ local_kv.swapaxes(-1, -2)
    normalizer = mx.logsumexp(local_scores, -1, keepdims=True)

    pooled_sq = pooled_gathered.squeeze(1)
    q_bl = q_scaled.transpose(0, 2, 1, 3)
    pooled_scores = q_bl @ pooled_sq.swapaxes(-1, -2)
    pooled_scores = pooled_scores.transpose(0, 2, 1, 3)
    normalizer = mx.logaddexp(
        normalizer, mx.logsumexp(pooled_scores, -1, keepdims=True)
    )
    if sinks_expanded is not None:
        normalizer = mx.logaddexp(normalizer, sinks_expanded)
    local_weights = mx.exp(local_scores - normalizer)
    pooled_weights = mx.exp(pooled_scores - normalizer)

    out = local_weights @ local_kv
    pw_bl = pooled_weights.transpose(0, 2, 1, 3)
    out = out + (pw_bl @ pooled_sq).transpose(0, 2, 1, 3)
    return out.astype(q_scaled.dtype)


def sparse_call(q, local_kv, pooled, topk, sinks, tile: int, single_gather: bool):
    """Replicates the production attn.sdpa sparse branch for one q block."""
    B, _, Lq, D = q.shape
    P = pooled.shape[1]
    k = topk.shape[2]
    q_scaled = q * SCALE

    def gather(topk_blk, lq):
        pooled_flat = pooled.reshape(B * P, D)
        off = (mx.arange(B) * P).reshape(B, 1, 1)
        return pooled_flat[(topk_blk + off).reshape(-1)].reshape(B, lq, k, D)

    if tile <= 0 or Lq <= tile:
        pg = gather(topk, Lq)[:, None]
        return _sparse_inner(q_scaled, local_kv, pg, sinks)

    if single_gather:
        pg_full = gather(topk, Lq)
        parts = [
            _sparse_inner(
                q_scaled[:, :, s : s + tile, :],
                local_kv,
                pg_full[:, s : s + tile, :, :][:, None],
                sinks,
            )
            for s in range(0, Lq, tile)
        ]
    else:
        parts = [
            _sparse_inner(
                q_scaled[:, :, s : s + tile, :],
                local_kv,
                gather(topk[:, s : s + tile, :], min(tile, Lq - s))[:, None],
                sinks,
            )
            for s in range(0, Lq, tile)
        ]
    return mx.concatenate(parts, axis=2)


def sparse_flops(Lq: int) -> float:
    """2 GEMM pairs: local (window) + pooled (topk), fwd + weighted-sum."""
    h, d = N_HEADS_PER_RANK, HEAD_DIM
    local = 2 * 2 * h * Lq * SLIDING_WINDOW * d
    pooled = 2 * 2 * h * Lq * INDEX_TOPK * d
    return float(local + pooled)


def make_sparse(Lq: int, pooled_len: int):
    q = mx.random.normal((1, N_HEADS_PER_RANK, Lq, HEAD_DIM)).astype(DTYPE)
    local_kv = mx.random.normal((1, 1, SLIDING_WINDOW, HEAD_DIM)).astype(DTYPE)
    pooled = mx.random.normal((1, pooled_len, HEAD_DIM)).astype(DTYPE)
    topk = mx.random.randint(0, pooled_len, (1, Lq, INDEX_TOPK)).astype(mx.int32)
    sinks = mx.zeros((1, N_HEADS_PER_RANK, 1, 1), dtype=mx.float32)
    mx.eval(q, local_kv, pooled, topk, sinks)
    return q, local_kv, pooled, topk, sinks


# ───────────────────── compressed path (production copy) ─────────────────
def compressed_call(q, kv, mask, sinks):
    return mx.fast.scaled_dot_product_attention(
        q, kv, kv, scale=SCALE, mask=mask, sinks=sinks
    )


def compressed_flops(Lq: int, kv_len: int) -> float:
    return float(2 * 2 * N_HEADS_PER_RANK * Lq * kv_len * HEAD_DIM)


def make_compressed(Lq: int, kv_len: int, offset: int):
    q = mx.random.normal((1, N_HEADS_PER_RANK, Lq, HEAD_DIM)).astype(DTYPE)
    kv = mx.random.normal((1, 1, kv_len, HEAD_DIM)).astype(DTYPE)
    sinks = mx.zeros((N_HEADS_PER_RANK,), dtype=DTYPE)
    mx.eval(q, kv, sinks)
    return q, kv, sinks


def causal_mask(Lq: int, kv_len: int, row_offset: int) -> mx.array:
    """Array mask, matching the production _extend_mask output shape."""
    qi = mx.arange(row_offset, row_offset + Lq)[:, None]
    ki = mx.arange(kv_len)[None, :]
    return (qi + (kv_len - (row_offset + Lq))) >= ki


# ─────────────────────────────── benchmark ───────────────────────────────
def bench_case(name: str, kv_len: int, one, two_a, two_b, flops_1024, flops_2048,
               warmup, iters):
    t1 = timeit(one, warmup, iters)
    ta = timeit(two_a, warmup, iters)
    tb = timeit(two_b, warmup, iters)
    t2 = ta + tb
    tref = timeit(two_a, warmup, iters)
    row = {
        "attention_type": name,
        "kv_length": kv_len,
        "single_call_2048_ms": round(t1, 4),
        "two_call_1024x2_ms": round(t2, 4),
        "single_call_1024_ms": round(tref, 4),
        "ratio_two_over_one": round(t2 / t1, 4),
        "ratio_2048_over_1024": round(t1 / tref, 4),
        "tflops_single_2048": round(flops_2048 / (t1 * 1e-3) / 1e12, 2),
        "tflops_two_1024x2": round(2 * flops_1024 / (t2 * 1e-3) / 1e12, 2),
        "tflops_single_1024": round(flops_1024 / (tref * 1e-3) / 1e12, 2),
    }
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    rows = []

    # ---- SPARSE (attn.sdpa) ------------------------------------------------
    # pooled length only affects the gather source; the SDPA math is over the
    # gathered top-512. Sweep a couple of pool depths to confirm.
    for pooled_len in (10250, 41000):
        for tile, sg, label in (
            (0, False, "sparse/untiled"),
            (128, True, "sparse/prod-tile128-singlegather"),
            (128, False, "sparse/tile128-pertile-gather"),
        ):
            q2, lkv, pool, tk2, sk = make_sparse(2048, pooled_len)
            q1a, tk1a = q2[:, :, :1024, :], tk2[:, :1024, :]
            q1b, tk1b = q2[:, :, 1024:, :], tk2[:, 1024:, :]
            mx.eval(q1a, q1b, tk1a, tk1b)

            one = lambda: sparse_call(q2, lkv, pool, tk2, sk, tile, sg)
            twa = lambda: sparse_call(q1a, lkv, pool, tk1a, sk, tile, sg)
            twb = lambda: sparse_call(q1b, lkv, pool, tk1b, sk, tile, sg)
            rows.append(
                bench_case(label, pooled_len, one, twa, twb,
                           sparse_flops(1024), sparse_flops(2048),
                           args.warmup, args.iters)
            )
            del q2, lkv, pool, tk2, sk, q1a, q1b, tk1a, tk1b
            mx.clear_cache()

    # ---- COMPRESSED (attn.sdpa.compressed) ---------------------------------
    # kv = local sliding window (128) + pooled (ctx/compress_ratio).
    for pooled_len in (2000, 5000, 10000, 20000):
        kv_len = pooled_len + SLIDING_WINDOW
        q2, kv, sk = make_compressed(2048, kv_len, 0)
        m2 = causal_mask(2048, kv_len, 0)
        q1a, q1b = q2[:, :, :1024, :], q2[:, :, 1024:, :]
        m1a, m1b = m2[:1024, :], m2[1024:, :]
        mx.eval(m2, q1a, q1b, m1a, m1b)

        one = lambda: compressed_call(q2, kv, m2, sk)
        twa = lambda: compressed_call(q1a, kv, m1a, sk)
        twb = lambda: compressed_call(q1b, kv, m1b, sk)
        rows.append(
            bench_case("compressed", kv_len, one, twa, twb,
                       compressed_flops(1024, kv_len),
                       compressed_flops(2048, kv_len),
                       args.warmup, args.iters)
        )
        del q2, kv, sk, m2, q1a, q1b, m1a, m1b
        mx.clear_cache()

    hdr = (f"{'attention_type':<38}{'KV':>7}{'1x2048ms':>11}{'2x1024ms':>11}"
           f"{'ratio':>8}{'1x1024ms':>10}{'2048/1024':>11}"
           f"{'TF@2048':>9}{'TF@2x1024':>11}{'TF@1024':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['attention_type']:<38}{r['kv_length']:>7}"
              f"{r['single_call_2048_ms']:>11.3f}{r['two_call_1024x2_ms']:>11.3f}"
              f"{r['ratio_two_over_one']:>8.3f}{r['single_call_1024_ms']:>10.3f}"
              f"{r['ratio_2048_over_1024']:>11.3f}"
              f"{r['tflops_single_2048']:>9.2f}{r['tflops_two_1024x2']:>11.2f}"
              f"{r['tflops_single_1024']:>9.2f}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
