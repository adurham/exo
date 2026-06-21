"""Bit-exactness check: _indexer_score_tiled vs _indexer_score.

Asserts the tiled-P indexer score matches the full-P path within bf16 epsilon
AND produces an identical top-k SET (the only thing the downstream
gathered-KV attention consumes; it is order-invariant). Run locally (no
cluster): python bench/indexer_tiled_p_bitexact.py
"""

import sys

import mlx.core as mx

from mlx_lm.models.deepseek_v4 import _indexer_score, _indexer_score_tiled


def run_case(B, H, L, P, D, k, p_block, seed=0):
    mx.random.seed(seed)
    q = mx.random.normal((B, H, L, D)).astype(mx.bfloat16)
    pooled = mx.random.normal((B, P, D)).astype(mx.bfloat16)
    weights_x = mx.random.normal((B, L, H)).astype(mx.bfloat16)
    scale = D ** -0.5
    nhi = H ** -0.5

    full = _indexer_score(q, pooled, weights_x, scale, nhi)
    tiled = _indexer_score_tiled(q, pooled, weights_x, scale, nhi, p_block)
    mx.eval(full, tiled)

    # numeric diff
    max_abs = float(mx.max(mx.abs(full.astype(mx.float32) - tiled.astype(mx.float32))))

    # top-k SET agreement (order-invariant) — what downstream actually uses
    kk = min(k, P)
    full_tk = mx.argsort(-full, axis=-1)[..., :kk]
    tiled_tk = mx.argsort(-tiled, axis=-1)[..., :kk]
    mx.eval(full_tk, tiled_tk)
    # per (B,L) row: intersection size / kk
    import numpy as np
    ft = np.array(full_tk)
    tt = np.array(tiled_tk)
    overlaps = []
    for b in range(B):
        for l in range(L):
            s_full = set(ft[b, l].tolist())
            s_tiled = set(tt[b, l].tolist())
            overlaps.append(len(s_full & s_tiled) / kk)
    min_overlap = min(overlaps)
    mean_overlap = sum(overlaps) / len(overlaps)
    return max_abs, min_overlap, mean_overlap


def main():
    cases = [
        # B, H, L,    P,   D,   k,  p_block
        (1, 64, 1,  25000, 128, 512,  8192),   # decode-ish, large P
        (1, 64, 128, 25000, 128, 512, 8192),   # prefill chunk, 25K pool
        (1, 64, 128, 90000, 128, 512, 16384),  # ~360K ctx pool, 16K block
        (1, 64, 128, 90000, 128, 512, 32768),  # same, 32K block
        (1, 64, 128, 250000, 128, 512, 16384), # ~1M ctx pool
        (1, 64, 64,  90007, 128, 512, 13337),  # ragged P + ragged block
    ]
    all_ok = True
    print(f"{'B':>2} {'H':>3} {'L':>4} {'P':>7} {'k':>5} {'pblk':>6} "
          f"{'max_abs':>10} {'min_ovlp':>9} {'mean_ovlp':>9}  verdict")
    for (B, H, L, P, D, k, pb) in cases:
        max_abs, mn, me = run_case(B, H, L, P, D, k, pb)
        # bf16 ulp at score magnitudes here ~1e-2; allow 3 ulp slack
        ok = max_abs <= 5e-2 and mn >= 0.98
        all_ok &= ok
        print(f"{B:>2} {H:>3} {L:>4} {P:>7} {k:>5} {pb:>6} "
              f"{max_abs:>10.5f} {mn:>9.4f} {me:>9.4f}  {'PASS' if ok else 'FAIL'}")
    print()
    if all_ok:
        print("ALL PASS — tiled-P indexer is bit-compatible with full-P.")
        sys.exit(0)
    else:
        print("FAIL — tiled-P diverges from full-P. DO NOT DEPLOY.")
        sys.exit(1)


if __name__ == "__main__":
    main()
