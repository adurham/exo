#!/usr/bin/env python3
"""Bit-exact gate for OPT-12 (tail-restricted pmask apply in Indexer).

Reproduces both paths of the pmask-apply logic on synthetic score tensors
using the REAL PoolingCache.make_mask row rule, and asserts elementwise
equality (not just top-k set) between the full-P where() and the banded
apply, across chunk offsets, ratios, band slices and ragged shapes.

Run: uv run python bench/tail_pmask_bitexact.py
"""
from __future__ import annotations

import sys

sys.path.insert(0, "mlx-lm")

import mlx.core as mx  # noqa: E402

FAILURES = 0


def make_pmask(P: int, L: int, offset: int, ratio: int) -> mx.array:
    """PoolingCache.make_mask rule: pool_idx < (offset+1+j)//ratio."""
    pool_idx = mx.arange(P)
    query_idx = mx.arange(offset + 1, offset + L + 1)
    return pool_idx < query_idx[:, None] // ratio


def apply_full(scores: mx.array, pmask: mx.array) -> mx.array:
    return mx.where(pmask[None], scores, mx.finfo(scores.dtype).min)


def apply_banded(scores: mx.array, pmask: mx.array, q_off: int, ratio: int) -> mx.array:
    P_len = scores.shape[-1]
    L_rows = scores.shape[1]
    vis_min = min((q_off + 1) // ratio, P_len)
    vis_max = min((q_off + L_rows) // ratio + 1, P_len)
    neg = mx.finfo(scores.dtype).min
    parts = [scores[..., :vis_min]]
    if vis_max > vis_min:
        parts.append(mx.where(
            pmask[None, :, vis_min:vis_max],
            scores[..., vis_min:vis_max],
            neg,
        ))
    if P_len > vis_max:
        parts.append(mx.full(
            (scores.shape[0], L_rows, P_len - vis_max), neg, dtype=scores.dtype,
        ))
    return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=-1)


def check(P: int, L: int, offset: int, ratio: int, band_lo: int | None, seed: int) -> None:
    """band_lo simulates seq-split: pmask sliced to [band_lo, band_lo+L)."""
    global FAILURES
    mx.random.seed(seed)
    scores = mx.random.normal((1, L, P)).astype(mx.float32)
    if band_lo is None:
        pmask = make_pmask(P, L, offset, ratio)
        q_off = offset
    else:
        full = make_pmask(P, band_lo + L, offset, ratio)
        pmask = full[band_lo:band_lo + L, :]
        q_off = offset + band_lo

    ref = apply_full(scores, pmask)
    got = apply_banded(scores, pmask, q_off, ratio)
    mx.eval(ref, got)
    equal = bool(mx.all(ref == got).item())
    tag = f"P={P} L={L} off={offset} ratio={ratio} band_lo={band_lo}"
    if equal:
        print(f"[OK]   {tag}")
    else:
        diff = mx.sum(ref != got).item()
        print(f"[FAIL] {tag} — {diff} mismatched elements")
        FAILURES += 1


def main() -> int:
    for ratio in (4, 128):
        for P, L, offset in (
            (1000, 256, 3968),        # early prefill
            (25_000, 256, 99_840),    # 100K ctx
            (123_800, 256, 495_000),  # 495K ctx
            (25_000, 128, 99_840),    # chunk 128
            (25_000, 37, 99_877),     # ragged remainder chunk
            (25_000, 256, 0),         # zero offset (pool ahead of queries)
            (10, 256, 99_840),        # pool shorter than cutoffs (vis_max clamp)
        ):
            for band_lo in (None, 0, 128):
                if band_lo == 128 and L < 256:
                    continue
                check(P, L, offset, ratio, band_lo, seed=P + L + offset + ratio)
    print("=" * 50)
    print("ALL OK" if FAILURES == 0 else f"{FAILURES} FAILURES")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
