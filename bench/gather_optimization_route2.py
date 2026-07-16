"""Route 2: Optimize the gather — single union gather vs 16 per-tile gathers.

The production code (deepseek_v4.py:4079) tiles the sparse SDPA over 128-row
sub-chunks. EACH tile call enters _sparse_pooled_attention and does its OWN
gather of (B, 128, 512, 512) = 67 MB. 16 tiles = 1.07 GB total gather traffic.

A single union gather: dedup all 2048 rows' topk into a union (~1024 entries),
gather 1024×512×2 = 1 MB once. 1000x less traffic. Then each tile's inner kernel
indexes into the compact buffer via per-row remap.

This test measures: how much faster is a single union gather vs 16 per-tile gathers?
The gather itself is memory-bound (82% DRAM peak). Reducing traffic 1000x should
cut the 4790µs/layer gather dramatically.

NOTE: This does NOT require fusing the gather into the GEMM (which hit the MMA wall).
The GEMM still reads from the materialized gathered tensor. The difference is whether
that tensor is (L, k, D) = 1.07 GB (current, re-gathered per tile) or (U, D) = 1 MB
(union, gathered once). The GEMM input is still dense either way.
"""
from __future__ import annotations

import glob
import os
import statistics
import time

import mlx.core as mx
import numpy as np

B, H, L_Q, D, K = 1, 64, 2048, 512, 512
L_TILE = 128
N_TILES = L_Q // L_TILE  # 16
N_ITERS, N_WARMUP = 30, 8
DTYPE = mx.bfloat16
DUMP_DIR = "/tmp/topk_redump"


def load_real_topk():
    files = sorted(glob.glob(os.path.join(DUMP_DIR, "*.npy")))
    if not files:
        # Fallback: synthetic topk with Jaccard ~0.65 (matching production locality)
        np.random.seed(42)
        topk_np = np.zeros((B, L_Q, K), dtype=np.int32)
        topk_np[0, 0] = np.random.choice(1024, K, replace=False)
        n_keep = int(K * 0.65)
        for i in range(1, L_Q):
            keep_idx = np.random.choice(K, n_keep, replace=False)
            topk_np[0, i, keep_idx] = topk_np[0, i - 1, keep_idx]
            new_idx = [j for j in range(K) if j not in keep_idx]
            topk_np[0, i, new_idx] = np.random.choice(1024, len(new_idx), replace=False)
        return mx.array(topk_np, dtype=mx.int32)  # (B, L_Q, K)
    chunks = [np.load(f)[0] for f in files[:16]]
    real = np.concatenate(chunks, axis=0)[:L_Q]
    return mx.array(real[None], dtype=mx.int32)


def make_inputs(topk):
    mx.random.seed(42)
    P = int(topk.max()) + 1
    pooled = mx.random.normal((B, P, D), dtype=DTYPE)
    return pooled, topk


def current_per_tile_gather(pooled, topk):
    """16 per-tile gathers, each (B, 128, 512, D) = 67 MB."""
    P = pooled.shape[1]
    pooled_flat = pooled.reshape(B * P, D)
    offset = (mx.arange(B) * P).reshape(B, 1, 1)
    gathered_all = []
    for t in range(N_TILES):
        s = t * L_TILE
        e = s + L_TILE
        topk_tile = topk[:, s:e, :]
        topk_flat = (topk_tile + offset).reshape(-1)
        gathered = pooled_flat[topk_flat].reshape(B, L_TILE, K, D)
        gathered_all.append(gathered)
    return mx.concatenate(gathered_all, axis=1)  # (B, L_Q, K, D)


def single_full_gather(pooled, topk):
    """Single gather for all 2048 rows at once (B, 2048, 512, D) = 1.07 GB."""
    P = pooled.shape[1]
    pooled_flat = pooled.reshape(B * P, D)
    offset = (mx.arange(B) * P).reshape(B, 1, 1)
    topk_flat = (topk + offset).reshape(-1)
    return pooled_flat[topk_flat].reshape(B, L_Q, K, D)


def union_gather(pooled, topk):
    """Single union gather: dedup all topk, gather only unique entries."""
    P = pooled.shape[1]
    topk_np = np.array(topk[0])
    all_unique = sorted(set(topk_np.flatten().tolist()))
    U = len(all_unique)
    union_arr = np.array(all_unique, dtype=np.int32)
    pooled_flat = pooled.reshape(B * P, D)
    compact = pooled_flat[mx.array(union_arr)]  # (U, D) = 1 MB
    # Remap for each row
    idx_map = {v: i for i, v in enumerate(all_unique)}
    remapped = np.array([[idx_map[v] for v in row] for row in topk_np], dtype=np.int32)
    remapped_mx = mx.array(remapped[None])
    # Still need to materialize (L, k, D) for the GEMM — but from compact (L2-resident)
    gathered = compact[remapped_mx.reshape(-1)].reshape(B, L_Q, K, D)
    return gathered, U


def time_fn(fn, *args):
    for _ in range(N_WARMUP):
        out = fn(*args)
        if isinstance(out, tuple):
            out = out[0]
        mx.eval(out)
        mx.synchronize()
    samples = []
    for _ in range(N_ITERS):
        mx.synchronize()
        t0 = time.perf_counter()
        out = fn(*args)
        if isinstance(out, tuple):
            out = out[0]
        mx.eval(out)
        mx.synchronize()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples) * 1e6


def main():
    print("=" * 76)
    print("Route 2: Gather optimization — per-tile vs single vs union")
    print(f"  Shape: B={B} L_Q={L_Q} D={D} k={K} tiles={N_TILES}")
    print("=" * 76)

    topk = load_real_topk()
    P = int(topk.max()) + 1
    print(f"  Pool size: {P}")
    pooled, topk = make_inputs(topk)
    mx.eval(pooled, topk)

    # Verify all produce the same gathered tensor
    ref = current_per_tile_gather(pooled, topk)
    full = single_full_gather(pooled, topk)
    union, U = union_gather(pooled, topk)
    mx.eval(ref, full, union)
    diff1 = float(mx.max(mx.abs(ref - full)))
    diff2 = float(mx.max(mx.abs(ref - union)))
    print(f"  Numerical: per-tile vs full max|d|={diff1:.6f}, per-tile vs union max|d|={diff2:.6f}")
    print(f"  Union size: {U} = {U/K:.1f}x k_sel (compact = {U*D*2/1e6:.1f} MB)")

    # Timing
    t_per_tile = time_fn(current_per_tile_gather, pooled, topk)
    t_full = time_fn(single_full_gather, pooled, topk)
    t_union, _ = time_fn(union_gather, pooled, topk), 0
    # Re-run to get U
    _, U = union_gather(pooled, topk)

    print("\n  Timing (median µs):")
    print(f"  A) 16 per-tile gathers:  {t_per_tile:8.0f} µs  (current production)")
    print(f"  B) single full gather:   {t_full:8.0f} µs  ({t_per_tile/t_full:.2f}x)")
    print(f"  C) union gather:         {t_union:8.0f} µs  ({t_per_tile/t_union:.2f}x)")
    print(f"     (union compact = {U*D*2/1e6:.1f} MB vs per-tile total = {B*L_Q*K*D*2/1e9:.2f} GB)")

    print("\n  Scaled to full layer (21 sparse layers):")
    print(f"    per-tile:  {t_per_tile*21:.0f} µs/layer")
    print(f"    full:      {t_full*21:.0f} µs/layer")
    print(f"    union:     {t_union*21:.0f} µs/layer")
    savings = (t_per_tile - t_union) * 21
    print(f"    savings:   {savings:.0f} µs/layer")
    # e2e: gather is 23.6% of module, module is 22.5% of wall
    # The gather savings as fraction of wall = savings / (full prefill time per layer)
    # From microbench: full inner = 15478µs/layer, gather was 4790µs (separate from inner)
    # Actually the gather is OUTSIDE the inner kernel (at :2321), so:
    # module = gather (4790) + inner (15478) = ~20268 µs/layer
    # wall at 500K = module / 0.225 = ~90213 µs/layer = 2048/90213e-6 = ~22.7 tok/s? No...
    # Let me just compute: savings / total_wall_per_layer
    # At 334 tok/s, 2048 tokens/chunk: 2048/334 = 6.13s per chunk = 6131ms per 2048 tokens
    # Per layer (43 layers, 21 sparse): 6131/21 = 292ms per sparse layer
    # savings = (t_per_tile - t_union) * 21 µs per sparse layer
    e2e_savings = savings / (292 * 1000)  # fraction of per-sparse-layer wall
    print(f"    e2e estimate: {e2e_savings*100:.1f}% = 334 × {1+e2e_savings:.3f} = {334*(1+e2e_savings):.0f} tok/s")
    print("=" * 76)


if __name__ == "__main__":
    main()