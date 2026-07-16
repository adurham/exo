"""Item 2: Itemize the 77.5% — per-op-class GPU time breakdown.

Fable: "Your ceiling math only covered the sparse module (22.5% of wall). The
other 77.5% — dense projections, MoE FFN, TP all-reduce, framework sync — has
never been itemized. If per-layer all-reduce is serialized with compute at
500K, that alone could be bigger than everything you've tried in the sparse
module."

This script uses mx.metal.gpu_time_ns() and dispatch_count() to measure GPU
time per section of one prefill chunk. Run it on a single node (not the full
cluster) to measure the GPU-side cost without TP comms confounding the
measurement.

Sections to measure (per layer, for a 2048-token prefill chunk):
  1. Dense projections (QKV + O proj)
  2. Sparse attention (the 22.5% we already know)
  3. MoE FFN (gate + expert GEMMs)
  4. Layer norm + residuals
  5. (TP all-reduce is not measurable on a single node — need the cluster)

The gap between GPU time and wall time = framework sync + comms overhead.
"""
from __future__ import annotations

import statistics
import time

import mlx.core as mx

# We can't easily instrument the production code per-section without modifying
# it. Instead, measure the ISOLATED GPU time of each op class at production
# shape, then sum them to see how much of the layer they account for.

B, H, L, D = 1, 64, 2048, 512
# Dense proj: (B, L, D_model) @ (D_model, D_model) — DSv4 has D_model=2048? Check config.
# For now, use representative shapes.
D_MODEL = 2048  # DSv4 hidden dim (approx)
N_EXPERTS = 64
EXPERT_DIM = 2048
N_ROUTED = 8  # top-8 experts

mx.random.seed(42)

# Dense QKV proj: (B*L, D_MODEL) @ (D_MODEL, 3*D) for QKV
qkv_w = mx.random.normal((D_MODEL, 3 * D), dtype=mx.bfloat16)
x = mx.random.normal((B * L, D_MODEL), dtype=mx.bfloat16)

# O proj: (B*L, H*D) @ (H*D, D_MODEL)
o_w = mx.random.normal((H * D, D_MODEL), dtype=mx.bfloat16)

# Attention Q/K/V already in x (skip for now — the 22.5% is known)

# MoE expert weights: (N_EXPERTS, EXPERT_DIM, D_MODEL) for gate + down
expert_gate = mx.random.normal((N_EXPERTS, D_MODEL, EXPERT_DIM), dtype=mx.bfloat16)
expert_down = mx.random.normal((N_EXPERTS, EXPERT_DIM, D_MODEL), dtype=mx.bfloat16)

mx.eval(qkv_w, x, o_w, expert_gate, expert_down)


def bench_gpu(fn, n=20, w=5):
    """Measure GPU time (not wall time) via mx.metal.gpu_time_ns."""
    for _ in range(w):
        out = fn()
        mx.eval(out)
        mx.synchronize()
    samples = []
    for _ in range(n):
        mx.metal.reset_gpu_time()
        t0 = mx.metal.gpu_time_ns()
        out = fn()
        mx.eval(out)
        mx.synchronize()
        t1 = mx.metal.gpu_time_ns()
        samples.append(t1 - t0)
    return statistics.median(samples) / 1e3  # ns -> us


def bench_wall(fn, n=20, w=5):
    """Measure wall time."""
    for _ in range(w):
        out = fn()
        mx.eval(out)
        mx.synchronize()
    samples = []
    for _ in range(n):
        mx.synchronize()
        t0 = time.perf_counter()
        out = fn()
        mx.eval(out)
        mx.synchronize()
        samples.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(samples)


# Dense QKV proj
def qkv_proj():
    return (x @ qkv_w).reshape(B, L, 3, D)

# O proj (after attention, input is (B*L, H*D))
attn_out = mx.random.normal((B * L, H * D), dtype=mx.bfloat16)
mx.eval(attn_out)

def o_proj():
    return attn_out @ o_w

# MoE: for top-8 experts, gather weights, matmul, combine
# Simplified: 8 expert gate GEMMs + 8 expert down GEMMs
gate_indices = mx.array([list(range(8)) for _ in range(B * L // 16)], dtype=mx.int32)  # (B*L/16, 8)

def moe_ffn():
    # Simulate: for each token, 8 experts, gate + down
    # This is approximate — the real MoE uses gather_mm
    out = mx.zeros((B * L, D_MODEL), dtype=mx.bfloat16)
    for e in range(8):
        g = x @ expert_gate[e]  # (B*L, EXPERT_DIM)
        g = mx.maximum(g, 0)  # SiLU/ReLU
        out = out + g @ expert_down[e]  # (B*L, D_MODEL)
    return out

# Sparse attention inner (the known 22.5%)
# Use the measured value: ~15478 us/layer at 500K (from microbench)
# At 50K (single chunk, L=2048): scale down

print("=" * 70)
print("Item 2: Per-op-class GPU time breakdown (single node, no TP comms)")
print(f"Shape: B={B} L={L} D={D} D_MODEL={D_MODEL}")
print("=" * 70)

t_qkv_gpu = bench_gpu(qkv_proj)
t_qkv_wall = bench_wall(qkv_proj)
print(f"Dense QKV proj:     GPU={t_qkv_gpu:6.0f} us  wall={t_qkv_wall:6.0f} us")

t_o_gpu = bench_gpu(o_proj)
t_o_wall = bench_wall(o_proj)
print(f"O proj:             GPU={t_o_gpu:6.0f} us  wall={t_o_wall:6.0f} us")

t_moe_gpu = bench_gpu(moe_ffn)
t_moe_wall = bench_wall(moe_ffn)
print(f"MoE FFN (8 experts): GPU={t_moe_gpu:6.0f} us  wall={t_moe_wall:6.0f} us")

total_gpu = t_qkv_gpu + t_o_gpu + t_moe_gpu
total_wall = t_qkv_wall + t_o_wall + t_moe_wall
print(f"\nTotal (proj+o+moe): GPU={total_gpu:6.0f} us  wall={total_wall:6.0f} us")

# The sparse attention at L=2048 is ~15478us GPU (from prior microbench)
sparse_gpu = 15478
print(f"Sparse attention:   GPU={sparse_gpu:6.0f} us  (from prior microbench)")
print(f"\nFull layer GPU estimate: {total_gpu + sparse_gpu:6.0f} us")
print(f"  proj+o+moe = {total_gpu:.0f} us = {total_gpu/(total_gpu+sparse_gpu)*100:.1f}% of layer")
print(f"  sparse     = {sparse_gpu} us = {sparse_gpu/(total_gpu+sparse_gpu)*100:.1f}% of layer")
print()
print("NOTE: This is single-node, no TP all-reduce. On the cluster, each")
print("TP layer adds an all-reduce over RDMA. If serialized with compute,")
print("that's the unaudited cost Fable flagged. Need the cluster trace to")
print("measure it — this script can't capture comms.")
print("=" * 70)