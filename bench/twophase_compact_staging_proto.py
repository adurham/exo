"""Two-phase compact staging prototype for DSv4 prefill sparse attention.

THE CONCEPT (Fable's recommended design):
Current path materializes the full gathered tensor: pooled_flat[topk_flat] -> (B, L, k, D) = 1.07 GB.
This is 128 rows × 512 k_sel × 512 D = the dominant cost, written then re-read by matmuls B and D.

Two-phase compact staging:
1. Per 128-row tile: dedup the topk indices into a union list (~1024 entries, the measured 2x k_sel)
2. Gather ONLY the union entries: 1024 × D = 1 MB (not 1.07 GB) — 1000x less write traffic
3. Run matmuls B and D against the compact buffer, using per-row remapped indices into [0, |U|)

This prototype tests whether MLX can do step 3 efficiently (indexed matmul) or whether
it requires a custom Metal kernel. It compares:
  A) current path: full materialize + dense matmul
  B) two-phase: dedup + compact gather + indexed matmul (via take_along_axis on the compact buffer)
  C) two-phase with mx.compile (does the compiler fuse the indexed access?)

Production shape: B=1, H=64, L_q=2048, D=512, sw=128, k_sel=512, P=1024 (measured pool size)
Uses REAL topk indices from the 500K dumps to get authentic locality.
"""
from __future__ import annotations

import glob
import os
import statistics
import time

import mlx.core as mx
import numpy as np

# Production shape
B = 1
H = 64
L_Q = 2048
D = 512
SW = 128
K_SEL = 512
SCALE = D ** -0.5
N_LAYERS = 21
N_ITERS = 30
N_WARMUP = 8
DTYPE = mx.bfloat16

# Load real topk from 500K dumps for authentic locality
DUMP_DIR = "/tmp/topk_dumps_500k"


def load_real_topk():
    """Load real topk indices from the 500K dumps, tile to L_Q=2048."""
    files = sorted(glob.glob(os.path.join(DUMP_DIR, "*.npy")))
    if not files:
        raise FileNotFoundError(f"no dumps in {DUMP_DIR}")
    # Each dump is (1, 128, 512). Concatenate to get L=2048 (16 dumps)
    chunks = []
    for f in files[:16]:
        a = np.load(f)
        chunks.append(a[0])  # (128, 512)
    real_topk = np.concatenate(chunks, axis=0)[:L_Q]  # (2048, 512)
    return mx.array(real_topk[None], dtype=mx.int32)  # (1, 2048, 512)


def make_inputs(topk):
    """Make inputs matching production _sparse_pooled_attention_inner shape."""
    mx.random.seed(42)
    q_scaled = mx.random.normal((B, H, L_Q, D), dtype=DTYPE) * SCALE
    local_kv = mx.random.normal((B, 1, SW, D), dtype=DTYPE)
    # pooled: (B, P, D) — P=1024 (measured pool size)
    P = int(topk.max()) + 1
    pooled = mx.random.normal((B, P, D), dtype=DTYPE)
    local_mask = mx.ones((B, 1, L_Q, SW), dtype=mx.bool_)
    pooled_mask = mx.ones((B, H, L_Q, K_SEL), dtype=mx.bool_)
    sinks = mx.random.normal((H,), dtype=DTYPE) * 0.1
    sinks_expanded = sinks[None, :, None, None]
    return q_scaled, local_kv, pooled, topk, local_mask, pooled_mask, sinks_expanded


# ─── Path A: current (full materialize) ───
def current_path(q_scaled, local_kv, pooled, topk, local_mask, pooled_mask, sinks_exp):
    P = pooled.shape[1]
    pooled_flat = pooled.reshape(B * P, D)
    offset = (mx.arange(B) * P).reshape(B, 1, 1)
    topk_flat = (topk + offset).reshape(-1)
    pooled_gathered = pooled_flat[topk_flat].reshape(B, L_Q, K_SEL, D)
    pooled_sq = pooled_gathered[:, None, :, :, :].squeeze(1)  # (B, L, k, D) — wait, need (B,1,L,k,D)
    # Match inner kernel: pooled_gathered is (B, 1, L, k, D), squeeze(1) -> (B, L, k, D)
    pooled_sq = pooled_gathered  # (B, L, k, D)
    # local scores
    local_scores = q_scaled @ local_kv.swapaxes(-1, -2)
    # pooled scores: q_bl @ pooled_sq.swapaxes
    q_bl = q_scaled.transpose(0, 2, 1, 3)  # (B, L, H, D)
    pooled_scores = q_bl @ pooled_sq.transpose(0, 1, 3, 2)  # (B, L, H, k)
    pooled_scores = pooled_scores.transpose(0, 2, 1, 3)  # (B, H, L, k)
    return pooled_scores


# ─── Path B: two-phase compact staging ───
def twophase_path(q_scaled, local_kv, pooled, topk, local_mask, pooled_mask, sinks_exp, tile=128):
    """Two-phase: dedup per tile, gather union, indexed matmul."""
    P = pooled.shape[1]
    pooled_flat = pooled.reshape(B * P, D)  # (P, D)

    # Phase 1: per-tile dedup + compact gather
    # For simplicity, dedup the ENTIRE L_q's topk at once (not per-tile)
    # In production this would be per 128-row tile, but the union across 2048 rows
    # with 2x k_sel locality should be ~4-8x k_sel = 2048-4096 entries
    topk_np = np.array(topk[0])  # (L, k)
    all_unique = sorted(set(topk_np.flatten().tolist()))
    union_arr = np.array(all_unique, dtype=np.int32)
    U = len(union_arr)

    # Compact gather: only the union entries
    union_mx = mx.array(union_arr)
    compact = pooled_flat[union_mx]  # (U, D) — tiny vs (L*k, D)

    # Per-row remap: map each topk entry to its index in the union list
    idx_map = {v: i for i, v in enumerate(all_unique)}
    remapped = np.array([[idx_map[v] for v in row] for row in topk_np], dtype=np.int32)
    remapped_mx = mx.array(remapped[None])  # (1, L, k)

    # Phase 2: indexed matmul against compact buffer
    # Instead of gathering (L, k, D) and doing q_bl @ gathered, we:
    #   compact_remapped = compact[remapped]  -> (L, k, D)  [but this IS the gather we're avoiding]
    # The REAL win is a kernel that reads compact[remapped[i,j]] directly during the matmul.
    # In Python/MLX we can't avoid the materialization for the matmul input —
    # but we CAN measure whether the compact gather + remapped matmul is faster
    # than the full gather, by checking if the compact buffer is L2-resident.

    # Actually, the Python path still materializes (L, k, D) for the matmul.
    # The win would only materialize in a Metal kernel that indexes during the GEMM.
    # So this Python prototype measures: is the compact gather + remap overhead
    # small enough that a Metal kernel indexing compact[] during the GEMM would win?

    # Gather from compact (same size output, but compact is L2-resident)
    compact_gathered = compact[remapped_mx.reshape(-1)].reshape(B, L_Q, K_SEL, D)

    q_bl = q_scaled.transpose(0, 2, 1, 3)
    pooled_scores = q_bl @ compact_gathered.transpose(0, 1, 3, 2)
    pooled_scores = pooled_scores.transpose(0, 2, 1, 3)
    return pooled_scores, U


def time_fn(fn, *args, n_iters=N_ITERS, n_warmup=N_WARMUP):
    last_meta = None
    for _ in range(n_warmup):
        out = fn(*args)
        if isinstance(out, tuple):
            last_meta = out[1:]
            out = out[0]
        mx.eval(out)
        mx.synchronize()
    samples = []
    for _ in range(n_iters):
        mx.synchronize()
        t0 = time.perf_counter()
        out = fn(*args)
        if isinstance(out, tuple):
            last_meta = out[1:]
            out = out[0]
        mx.eval(out)
        mx.synchronize()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples) * 1e6, last_meta


def main():
    print("=" * 76)
    print("Two-phase compact staging prototype (DSv4 prefill sparse attention)")
    print(f"  Shape: B={B} H={H} L_q={L_Q} D={D} sw={SW} k_sel={K_SEL}")
    print(f"  Using REAL topk from 500K dumps")
    print("=" * 76)

    topk = load_real_topk()
    print(f"  Real topk: shape={topk.shape}, pool size={int(topk.max())+1}")
    args = make_inputs(topk)
    mx.eval(*args)

    # Measure union size
    topk_np = np.array(topk[0])
    all_unique = set(topk_np.flatten().tolist())
    print(f"  Union across all {L_Q} rows: {len(all_unique)} entries = {len(all_unique)/K_SEL:.1f}x k_sel")
    for tile in [128, 256, 512]:
        unions = []
        for s in range(0, L_Q - tile + 1, tile):
            u = set()
            for i in range(s, min(s + tile, L_Q)):
                u.update(topk_np[i].tolist())
            unions.append(len(u))
        print(f"  Union per {tile}-tile: avg={statistics.mean(unions):.0f} = {statistics.mean(unions)/K_SEL:.1f}x k_sel")

    print()
    print("Timing (median µs, forced eval+sync):")

    # Path A: current (full materialize)
    t_a, _ = time_fn(current_path, *args)
    print(f"  A) current (full materialize):     {t_a:8.0f} µs")

    # Path B: two-phase
    t_b, meta = time_fn(twophase_path, *args)
    U = meta[0] if meta else 0
    print(f"  B) two-phase (compact gather):     {t_b:8.0f} µs  (union={U}={U/K_SEL:.1f}x k_sel)")
    print(f"     speedup: {t_a/t_b:.2f}x")

    # Path C: two-phase with mx.compile
    twophase_compiled = mx.compile(twophase_path)
    def twophase_compiled_wrapper(*a):
        out, U = twophase_compiled(*a)
        return out, U

    try:
        t_c, _ = time_fn(lambda *a: twophase_compiled_wrapper(*a), *args)
        print(f"  C) two-phase + mx.compile:         {t_c:8.0f} µs")
        print(f"     speedup vs A: {t_a/t_c:.2f}x")
    except Exception as e:
        print(f"  C) two-phase + mx.compile: FAILED ({type(e).__name__}: {e})")

    print()
    print("=" * 76)
    print("ANALYSIS:")
    print(f"  Current path materializes (B,L,k,D) = {B*L_Q*K_SEL*D*2/1e9:.2f} GB")
    print(f"  Two-phase compact buffer = (U,D) = {U*D*2/1e6:.1f} MB")
    print(f"  Traffic reduction: {B*L_Q*K_SEL*D*2 / (U*D*2):.0f}x")
    print()
    if t_b < t_a:
        print(f"  -> Python two-phase is {t_a/t_b:.2f}x FASTER — promising for Metal kernel")
        print("     A Metal kernel indexing compact[] during the GEMM would eliminate the")
        print("     (L,k,D) materialization entirely, capturing the full traffic reduction.")
    else:
        print(f"  -> Python two-phase is {t_b/t_a:.2f}x SLOWER (expected — still materializes)")
        print("     The Python path still does compact[remapped] -> (L,k,D) for the matmul.")
        print("     The win requires a Metal kernel that indexes compact[] DURING the GEMM,")
        print("     avoiding the (L,k,D) materialization. This confirms the scope: custom Metal.")
    print("=" * 76)


if __name__ == "__main__":
    main()