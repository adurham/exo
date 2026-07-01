#!/usr/bin/env python3
"""Bit-exact gate for the L>1 fused top-K kernel variant (OPT-11).

Validates _fused_topk at L in {1, 3} (decode + MTP verify widths) against
the argsort reference: top-k SET overlap must be 1.0 wherever scores have
no exact ties at the boundary, and any disagreement must be an exact-tie
swap (same score value). Mirrors the indexer_tiled_p_bitexact.py pattern.

Run: uv run python bench/topk_fused_verify_bitexact.py
"""
from __future__ import annotations

import sys

sys.path.insert(0, "mlx-lm")

import mlx.core as mx  # noqa: E402

from mlx_lm.models.deepseek_v4 import _fused_topk  # noqa: E402

FAILURES = 0


def check(B: int, L: int, P: int, K: int, seed: int) -> None:
    global FAILURES
    mx.random.seed(seed)
    # Match production score character: post-mask fp32-ish scores from the
    # indexer collapse. Use float32 like the real scores tensor after the
    # (B,L,D)@(B,D,P) bf16 GEMM output (bf16 values, so ties are common —
    # exactly the hard case for set-equality).
    scores = mx.random.normal((B, L, P)).astype(mx.bfloat16).astype(mx.float32)

    ref = mx.argsort(-scores, axis=-1)[..., :K]
    fused = _fused_topk(scores, K)
    assert fused is not None, "kernel unavailable for this K"
    mx.eval(ref, fused)

    ok = True
    for b in range(B):
        for l in range(L):
            ref_set = set(ref[b, l].tolist())
            fu_set = set(fused[b, l].tolist())
            if ref_set == fu_set:
                continue
            # Disagreements must be exact-score ties at the K boundary.
            only_ref = ref_set - fu_set
            only_fu = fu_set - ref_set
            row = scores[b, l]
            ref_vals = sorted(float(row[i].item()) for i in only_ref)
            fu_vals = sorted(float(row[i].item()) for i in only_fu)
            if ref_vals != fu_vals:
                ok = False
                print(
                    f"  FAIL b={b} l={l}: non-tie divergence "
                    f"ref-only={ref_vals[:5]} fused-only={fu_vals[:5]}"
                )
            else:
                print(
                    f"  note b={b} l={l}: {len(only_ref)} exact-tie swaps "
                    f"(quality-equivalent)"
                )
    status = "OK" if ok else "FAIL"
    print(f"[{status}] B={B} L={L} P={P} K={K} seed={seed}")
    if not ok:
        FAILURES += 1


def main() -> int:
    for P in (25_000, 90_000, 250_000):
        for L in (1, 3):
            for seed in (0, 1, 2):
                check(B=1, L=L, P=P, K=512, seed=seed)
    # gamma sweep widths
    for L in (2, 4, 8):
        check(B=1, L=L, P=90_000, K=512, seed=7)
    print("=" * 50)
    print("ALL OK" if FAILURES == 0 else f"{FAILURES} FAILURES")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
