"""Indexed GEMM PoC: Metal kernel for compact-buffer score matmul.

THE KERNEL: scores = q @ compact[remap].T
  q: (L, H, D) — queries, L=128 per tile, H=64 heads, D=512
  compact: (U, D) — deduplicated pooled entries, U~1024 (2x k_sel), L2-resident
  remap: (L, k) — per-row indices into [0, U), k=512
  scores: (L, H, k) — output

DESIGN (Fable): "load remap row once per tile, then stream compact[] with
regular access within the tile." The indexed lookup (remap[l][j]) is hoisted
out of the D-reduction inner loop. For each (l, h) threadgroup:
  1. Load remap[l] into threadgroup memory (k=512 int32 = 2KB)
  2. For each j in [0, k): load compact[remap[l][j]] (D=512 bf16 = 1KB)
  3. Dot product q[l][h] · compact[remap[l][j]] via simd_sum over D
  4. Write scores[l][h][j]

This is a GEMV-style kernel (M=H=64 small, K=D=512, N=k=512). The alternative
is simdgroup_matrix MMA tiling, but with per-row indexed N, the N dimension
can't be tiled across threadgroups (each row has different N indices). So the
per-(l,h) threadgroup approach with simd-reduce over D is the right structure.

KILL CRITERIA (Fable):
  Dense baseline: 7665µs. Gather eliminated: 4790µs.
  Break-even: <= 12455µs. 350-target: <= 10000µs. Ideal: <= 8500µs.

Usage (m4-2): .venv/bin/python /tmp/indexed_gemm_poc.py
"""
from __future__ import annotations

import glob
import os
import statistics
import time

import mlx.core as mx
import numpy as np

B, H, L_Q, D, K_SEL = 1, 64, 2048, 512, 512
L_TILE = 128  # the _SPARSE_SDPA_TILE, one tile at a time
N_ITERS, N_WARMUP = 30, 8
DTYPE = mx.bfloat16
DUMP_DIR = "/tmp/topk_dumps_500k"

# The kernel: uses thread_position_in_grid.x (MLX metal_kernel quirk —
# thread_position_in_threadgroup.x always returns 0, but thread_position_in_grid.x
# works and simd_sum reduces across the 32-lane simdgroup within the grid).
# grid = (L_TILE * H * BN * BD, 1, 1) where BN=32 simdgroups per (l,h), BD=32 lanes.
# Each 32-thread simdgroup handles one (l, h, key_batch) triple.
# key_batch iterates K/BN = 16 times, each simdgroup processing one key per iter.
KERNEL_SOURCE = r"""
uint gid      = thread_position_in_grid.x;
uint simd_lid = gid % 32;            // 0..31 — lane within simdgroup
uint triple   = gid / 32;            // which (l, h, key_batch) simdgroup
uint key_batch = triple % BN_;       // 0..BN-1 — which key in current iteration
uint pair     = triple / BN_;        // which (l, h) pair
uint l        = pair / H_;
uint h        = pair % H_;

constexpr int BD = 32;
constexpr int qk_per_thread = D_ / BD;  // 512/32 = 16

// Load q[l, h, :] — each lane loads its 16 D-elements
float q_r[qk_per_thread];
const uint q_off = (l * H_ + h) * D_ + simd_lid * qk_per_thread;
for (int i = 0; i < qk_per_thread; i++) {
    q_r[i] = float(q[q_off + i]);
}

// Iterate K/BN key batches — each simdgroup handles one key per iteration
for (uint j_base = 0; j_base < K_; j_base += BN_) {
    uint j = j_base + key_batch;
    if (j >= K_) continue;

    uint p_idx = uint(remap[l * K_ + j]);
    const uint k_off = p_idx * D_ + simd_lid * qk_per_thread;
    float partial = 0.0f;
    for (int i = 0; i < qk_per_thread; i++) {
        partial += q_r[i] * float(compact[k_off + i]);
    }
    float score = simd_sum(partial);  // reduce across 32 lanes -> full D=512 dot
    if (simd_lid == 0) {
        out[(l * H_ + h) * K_ + j] = T(score);
    }
}
"""


def build_kernel(k: int, h: int, d: int):
    header = f"""
constant uint H_ = {h};
constant uint D_ = {d};
constant uint K_ = {k};
constant uint BN_ = 8;
"""
    return mx.fast.metal_kernel(
        name=f"indexed_gemm_poc_k{k}_h{h}_d{d}",
        input_names=["q", "compact", "remap"],
        output_names=["out"],
        source=KERNEL_SOURCE,
        header=header,
        ensure_row_contiguous=True,
    )


def load_real_topk():
    files = sorted(glob.glob(os.path.join(DUMP_DIR, "*.npy")))
    # Use the worst-case 4x dump (dump_400) and a typical 2x dump
    typical = np.load(files[0])[0]  # (128, 512)
    worst = None
    for f in files:
        a = np.load(f)
        if int(a.max()) >= 2047:  # the 4x union case
            worst = a[0]
            break
    return typical, worst


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


def dense_baseline(q_tile, compact, remap_tile):
    """Dense: materialize compact[remap] then matmul."""
    gathered = compact[remap_tile.reshape(-1)].reshape(L_TILE, K_SEL, D)
    q_bl = q_tile.reshape(L_TILE, H, D)
    scores = q_bl @ gathered.transpose(0, 2, 1)  # (L, H, k)
    return scores


def indexed_kernel(kernel, q_tile, compact, remap_tile):
    """Custom Metal kernel with indexed loads."""
    # q_tile: (L_TILE, H, D), compact: (U, D), remap_tile: (L_TILE, K_SEL)
    outs = kernel(
        inputs=[q_tile, compact, remap_tile.astype(mx.uint32)],
        template=[("T", mx.bfloat16)],
        grid=(L_TILE * H * 8 * 32, 1, 1),  # BN=8 simdgroups per (l,h) × BD=32 lanes
        threadgroup=(32, 1, 1),
        output_shapes=[(L_TILE, H, K_SEL)],
        output_dtypes=[mx.bfloat16],
    )
    return outs[0]


def main():
    print("=" * 76)
    print("Indexed GEMM PoC — Metal kernel for compact-buffer score matmul")
    print(f"  Shape: L_TILE={L_TILE} H={H} D={D} k={K_SEL}")
    print(f"  grid=({L_TILE}*{H}*8*32, 1, 1) = {L_TILE*H*8*32} threads, threadgroup=(32, 1, 1)")
    print("=" * 76)

    typical_topk, worst_topk = load_real_topk()
    print(f"  Typical dump: pool={int(typical_topk.max())+1}")
    if worst_topk is not None:
        print(f"  Worst-case dump: pool={int(worst_topk.max())+1}")

    kernel = build_kernel(K_SEL, H, D)

    for label, topk_np in [("typical (2x)", typical_topk), ("worst (4x)", worst_topk)]:
        if topk_np is None:
            continue
        # Build compact buffer + remap for this tile
        all_unique = sorted(set(topk_np.flatten().tolist()))
        U = len(all_unique)
        idx_map = {v: i for i, v in enumerate(all_unique)}
        remap = np.array([[idx_map[v] for v in row] for row in topk_np], dtype=np.int32)

        mx.random.seed(42)
        q_tile = mx.random.normal((L_TILE, H, D), dtype=DTYPE)
        compact = mx.random.normal((U, D), dtype=DTYPE)
        remap_mx = mx.array(remap)
        mx.eval(q_tile, compact, remap_mx)

        print(f"\n  {label}: U={U} = {U/K_SEL:.1f}x k_sel, compact={U*D*2/1e6:.1f}MB")

        # Numerical check
        ref = dense_baseline(q_tile, compact, remap_mx)
        cand = indexed_kernel(kernel, q_tile, compact, remap_mx)
        mx.eval(ref, cand)
        diff = mx.abs(ref - cand)
        max_diff = float(mx.max(diff))
        ref_abs = mx.abs(ref) + 1e-5
        max_rel = float(mx.max(diff / ref_abs))
        print(f"    numerical: max|d|={max_diff:.4e} max|d|/|r|={max_rel:.4e}")

        # Timing
        t_dense = time_fn(dense_baseline, q_tile, compact, remap_mx)
        t_kernel = time_fn(indexed_kernel, kernel, q_tile, compact, remap_mx)
        print(f"    dense (materialize+matmul): {t_dense:8.0f} µs")
        print(f"    indexed Metal kernel:       {t_kernel:8.0f} µs  ({t_dense/t_kernel:.2f}x)")
        print(f"    KILL GATES: break-even={t_dense+4790:.0f} 350-target=10000 ideal=8500")

        # Scale to full layer (16 tiles of 128 = 2048 L_q)
        t_dense_layer = t_dense * 16
        t_kernel_layer = t_kernel * 16
        print(f"    scaled to 2048 L_q (x16 tiles): dense={t_dense_layer:.0f}µs kernel={t_kernel_layer:.0f}µs")

    print(f"\n{'='*76}")
    print("VERDICT per Fable kill criteria:")
    print("  (compare kernel time to: break-even 12455µs, 350-target 10000µs, ideal 8500µs)")
    print("  Note: times above are PER-TILE (128 rows). Full layer = x16.)")
    print(f"  Full-layer break-even: {12455*16:.0f}µs, 350-target: {10000*16:.0f}µs")
    print("=" * 76)


if __name__ == "__main__":
    main()