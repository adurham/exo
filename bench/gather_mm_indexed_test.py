"""Test if mx.gather_mm can do the compact-buffer indexed matmul.

gather_mm: "Matrix multiplication with matrix-level gather. Gathers operands
with indices then does a batched matmul. More efficient than take + matmul."

The compact staging pattern needs: for each query row l, compute
  scores[l] = q[l] @ compact[remap[l]].T
where compact is (U, D) and remap[l] is (k,) indices into [0, U).

Can gather_mm express this? gather_mm does BATCH-level gather (select which
matrices from a batch), not column-level gather within a matrix. So:
  - a = q reshaped to (L, H, D) — batch L, matrix (H, D)
  - b = compact — needs to be (L, k, D) with per-l different rows selected
  - This requires PER-L different b matrices, which is a column-gather, not batch-gather

So gather_mm CANNOT directly express the compact staging pattern. But let me
test it anyway to confirm, and test the closest approximation:
  - Materialize compact[remap] -> (L, k, D) then gather_mm with identity indices
  - vs current full materialize path

If gather_mm can't help, this confirms the custom Metal kernel is required.
"""
from __future__ import annotations

import glob
import os
import statistics
import time

import mlx.core as mx
import numpy as np

B, H, L_Q, D, K_SEL = 1, 64, 2048, 512, 512
N_ITERS, N_WARMUP = 30, 8
DTYPE = mx.bfloat16
DUMP_DIR = "/tmp/topk_dumps_500k"


def load_real_topk():
    files = sorted(glob.glob(os.path.join(DUMP_DIR, "*.npy")))
    chunks = [np.load(f)[0] for f in files[:16]]
    real = np.concatenate(chunks, axis=0)[:L_Q]
    return mx.array(real[None], dtype=mx.int32)


def make_inputs(topk):
    mx.random.seed(42)
    q = mx.random.normal((B, H, L_Q, D), dtype=DTYPE)
    P = int(topk.max()) + 1
    pooled = mx.random.normal((B, P, D), dtype=DTYPE)
    return q, pooled, topk


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


def current_path(q, pooled, topk):
    """Current: full materialize (B,L,k,D) then dense matmul."""
    P = pooled.shape[1]
    pooled_flat = pooled.reshape(B * P, D)
    offset = (mx.arange(B) * P).reshape(B, 1, 1)
    topk_flat = (topk + offset).reshape(-1)
    gathered = pooled_flat[topk_flat].reshape(B, L_Q, K_SEL, D)
    q_bl = q.transpose(0, 2, 1, 3)  # (B, L, H, D)
    scores = q_bl @ gathered.transpose(0, 1, 3, 2)  # (B, L, H, k)
    return scores.transpose(0, 2, 1, 3)


def gather_mm_path(q, pooled, topk):
    """Try gather_mm: materialize compact[remap] then gather_mm."""
    P = pooled.shape[1]
    pooled_flat = pooled.reshape(B * P, D)
    offset = (mx.arange(B) * P).reshape(B, 1, 1)
    topk_flat = (topk + offset).reshape(-1)
    gathered = pooled_flat[topk_flat].reshape(B, L_Q, K_SEL, D)
    # gather_mm needs batch dims. Reshape q to (L, H, D) and gathered to (L, D, k)
    q_bl = q.transpose(0, 2, 1, 3).reshape(L_Q, H, D)  # (L, H, D)
    g = gathered.reshape(L_Q, D, K_SEL).transpose(0, 2, 1)  # (L, k, D) -> (L, D, k)
    # gather_mm(a, b, lhs_indices, rhs_indices) — identity indices
    idx = mx.arange(L_Q, dtype=mx.int32)
    scores = mx.gather_mm(q_bl, g.transpose(0, 2, 1), idx, idx)  # (L, H, k)
    return scores.reshape(B, H, L_Q, K_SEL)


def compact_gather_mm_path(q, pooled, topk):
    """Two-phase with gather_mm: compact buffer + remapped gather."""
    P = pooled.shape[1]
    topk_np = np.array(topk[0])
    all_unique = sorted(set(topk_np.flatten().tolist()))
    U = len(all_unique)
    union_mx = mx.array(np.array(all_unique, dtype=np.int32))
    pooled_flat = pooled.reshape(B * P, D)
    compact = pooled_flat[union_mx]  # (U, D)

    # Remap
    idx_map = {v: i for i, v in enumerate(all_unique)}
    remapped = np.array([[idx_map[v] for v in row] for row in topk_np], dtype=np.int32)
    remapped_mx = mx.array(remapped)  # (L, k)

    # Gather from compact: compact[remapped] -> (L, k, D) — STILL materializes
    compact_gathered = compact[remapped_mx.reshape(-1)].reshape(L_Q, K_SEL, D)
    q_bl = q.transpose(0, 2, 1, 3).reshape(L_Q, H, D)
    idx = mx.arange(L_Q, dtype=mx.int32)
    scores = mx.gather_mm(q_bl, compact_gathered, idx, idx)  # (L, H, k)
    return scores.reshape(B, H, L_Q, K_SEL), U


def main():
    print("=" * 76)
    print("gather_mm test for compact-buffer indexed matmul")
    print(f"  Shape: B={B} H={H} L={L_Q} D={D} k={K_SEL}")
    print("=" * 76)

    topk = load_real_topk()
    P = int(topk.max()) + 1
    print(f"  Pool size: {P}, union: {len(set(np.array(topk[0]).flatten()))} = {len(set(np.array(topk[0]).flatten()))/K_SEL:.1f}x k_sel")

    q, pooled, topk = make_inputs(topk)
    mx.eval(q, pooled, topk)

    print("\nTiming (median µs):")
    t_cur = time_fn(current_path, q, pooled, topk)
    print(f"  A) current (full materialize + matmul):  {t_cur:8.0f} µs")

    t_gmm = time_fn(gather_mm_path, q, pooled, topk)
    print(f"  B) gather_mm (materialize + gather_mm):   {t_gmm:8.0f} µs  ({t_cur/t_gmm:.2f}x)")

    def wrap(*a):
        out, U = compact_gather_mm_path(*a)
        return out
    t_cgm = time_fn(wrap, q, pooled, topk)
    # Re-run to get U
    _, U = compact_gather_mm_path(q, pooled, topk)
    mx.eval(_)
    print(f"  C) compact + gather_mm (2-phase):         {t_cgm:8.0f} µs  ({t_cur/t_cgm:.2f}x, U={U})")

    print(f"\n  Break-even: {t_cur + 4790:.0f} µs (current + gather cost)")
    print(f"  350-target: {10000} µs")
    print(f"  Ideal: {8500} µs")

    if t_cgm <= 10000:
        print("  -> gather_mm 2-phase MEETS 350 target — may not need custom Metal!")
    elif t_cgm <= 12455:
        print("  -> gather_mm 2-phase above break-even but below 350 target — marginal")
    else:
        print("  -> gather_mm 2-phase above break-even — custom Metal still needed")


if __name__ == "__main__":
    main()