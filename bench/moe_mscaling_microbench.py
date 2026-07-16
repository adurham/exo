"""MoE microbench: M-scaling sweep to confirm memory-bound diagnosis.

Fable: "At 128-token chunks x top-6 = 768 token-expert / 256 experts = ~3 tokens/expert.
Each chunk streams ENTIRE 256-expert weight set per layer for M≈3 GEMMs.
M≈3 is way below the ridge (M≈16) — deeply memory-bound.
A 4x chunk increase (128→512) raises M≈3→12, approaching the ridge."

This microbench measures the MoE expert GEMM throughput at M=1,2,4,8,16,32,64,128
and compares to:
  (a) the pure streaming bound (total expert bytes / 546 GB/s)
  (b) a dense qmm of equivalent FLOPs at large M

If throughput scales near-linearly with M up to ~64, the memory-bound diagnosis
is confirmed and the two-level chunking lever is quantified.

Model config: 256 routed experts, top-6 per token, moe_intermediate_size=2048.
Expert GEMM: (M, 2048) @ (2048, D_model) per expert — but in TP, each node holds
a shard. For this single-node microbench, use the full expert weight.
"""
from __future__ import annotations

import statistics
import time

import mlx.core as mx

# Model config: 256 experts, top-6, moe_intermediate_size=2048
# D_model from config — DSv4 has hidden_size... let me check
# For the microbench, use representative shapes
N_EXPERTS = 256
EXPERT_DIM = 2048  # moe_intermediate_size
D_MODEL = 2048  # hidden_size (approximate — will confirm)

# Expert weights: gate (D_MODEL -> EXPERT_DIM) and down (EXPERT_DIM -> D_MODEL)
# In 4-bit or 8-bit quant — start with bf16 for the baseline measurement
mx.random.seed(42)
expert_gate = mx.random.normal((N_EXPERTS, D_MODEL, EXPERT_DIM), dtype=mx.bfloat16)
expert_down = mx.random.normal((N_EXPERTS, EXPERT_DIM, D_MODEL), dtype=mx.bfloat16)
mx.eval(expert_gate, expert_down)

# Total expert weight bytes (one node's shard in TP — for now, full)
expert_bytes = N_EXPERTS * (D_MODEL * EXPERT_DIM + EXPERT_DIM * D_MODEL) * 2  # bf16
print(f"Expert weights: {N_EXPERTS} experts x 2 x {D_MODEL}x{EXPERT_DIM} bf16 = {expert_bytes/1e9:.2f} GB")
print("M4 Max DRAM: ~546 GB/s")
print(f"Streaming bound (all experts, once): {expert_bytes/546e9*1e6:.0f} us")
print()


def bench_expert_gemms(M: int, n=20, w=5):
    """Time M-token expert GEMMs across all 256 experts (gate + down).
    This simulates one layer's MoE FFN for M tokens distributed across experts.
    """
    # M tokens, each assigned to top-6 experts. For simplicity, distribute
    # M tokens round-robin across experts (M/256 per expert on average).
    # Each expert does: gate GEMM (M_per_expert, D_MODEL) @ (D_MODEL, EXPERT_DIM)
    #               + down GEMM (M_per_expert, EXPERT_DIM) @ (EXPERT_DIM, D_MODEL)
    x = mx.random.normal((M, D_MODEL), dtype=mx.bfloat16)
    mx.eval(x)

    # Simulate: each expert gets ceil(M/N_EXPERTS) tokens (at least 1)
    # For M < N_EXPERTS, only M experts are active (the rest get 0 tokens)
    M_per_expert = max(1, (M + N_EXPERTS - 1) // N_EXPERTS)
    n_active = min(M, N_EXPERTS)
    # Pad x to fit: n_active experts each with M_per_expert tokens
    x_padded = mx.zeros((n_active * M_per_expert, D_MODEL), dtype=mx.bfloat16)
    x_padded[:M] = x[:M]
    mx.eval(x_padded)

    def moe_layer():
        # Reshape into (n_active, M_per_expert, D_MODEL)
        x_expert = x_padded.reshape(n_active, M_per_expert, D_MODEL)
        # Gate GEMM: (n_active, M_per, D) @ (n_active, D, E) -> (n_active, M_per, E)
        # Use only the first n_active experts
        gate_w = expert_gate[:n_active]  # (n_active, D, E)
        gate_out = mx.matmul(x_expert, gate_w)
        # Activation (SiLU)
        gate_out = gate_out * (1.0 / (1.0 + mx.exp(-gate_out)))
        # Down GEMM: (n_active, M_per, E) @ (n_active, E, D) -> (n_active, M_per, D)
        down_w = expert_down[:n_active]  # (n_active, E, D)
        down_out = mx.matmul(gate_out, down_w)
        return down_out.reshape(n_active * M_per_expert, D_MODEL)

    for _ in range(w):
        out = moe_layer()
        mx.eval(out)
        mx.synchronize()
    samples = []
    for _ in range(n):
        mx.synchronize()
        t0 = time.perf_counter()
        out = moe_layer()
        mx.eval(out)
        mx.synchronize()
        samples.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(samples)


def bench_dense_gemm(M: int, n=20, w=5):
    """Dense GEMM of equivalent total FLOPs for comparison.
    Total MoE FLOPs = N_EXPERTS * M_per_expert * 2 * 2 * D_MODEL * EXPERT_DIM
    Dense equivalent: (M_total, D_MODEL) @ (D_MODEL, EXPERT_DIM*N_EXPERTS) — one big GEMM
    """
    # Equivalent: (M, D_MODEL) @ (D_MODEL, N_EXPERTS*EXPERT_DIM)
    # This is what the MoE would be if it were a single dense FFN
    big_w = mx.random.normal((D_MODEL, N_EXPERTS * EXPERT_DIM), dtype=mx.bfloat16)
    mx.eval(big_w)
    x = mx.random.normal((M, D_MODEL), dtype=mx.bfloat16)
    mx.eval(x)

    def dense_ffn():
        out = x @ big_w
        out = out * (1.0 / (1.0 + mx.exp(-out)))
        # Down: (M, N_EXPERTS*EXPERT_DIM) @ (N_EXPERTS*EXPERT_DIM, D_MODEL)
        down_w = mx.random.normal((N_EXPERTS * EXPERT_DIM, D_MODEL), dtype=mx.bfloat16)
        return out @ down_w

    for _ in range(w):
        o = dense_ffn()
        mx.eval(o)
        mx.synchronize()
    samples = []
    for _ in range(n):
        mx.synchronize()
        t0 = time.perf_counter()
        o = dense_ffn()
        mx.eval(o)
        mx.synchronize()
        samples.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(samples)


print("=" * 80)
print("MoE M-scaling sweep (256 experts, top-6, D_MODEL=2048, EXPERT_DIM=2048)")
print("=" * 80)
print(f"{'M_total':>8s} {'M/exp':>6s} {'MoE us':>8s} {'Dense us':>10s} {'MoE/Dense':>10s} {'Streaming':>10s}")
print(f"{'':>8s} {'':>6s} {'':>8s} {'':>10s} {'':>10s} {'bound us':>10s}")
print("-" * 80)

for M in [128, 256, 512, 1024, 2048, 4096]:
    M_per = max(1, M // N_EXPERTS)
    t_moe = bench_expert_gemms(M)
    t_dense = bench_dense_gemm(M) if M <= 2048 else None
    # Streaming bound: how long to stream all expert weights once
    streaming_us = expert_bytes / 546e9 * 1e6
    dense_str = f"{t_dense:.0f}" if t_dense else "N/A"
    ratio = f"{t_moe/t_dense:.2f}x" if t_dense else "N/A"
    print(f"{M:8d} {M_per:6d} {t_moe:8.0f} {dense_str:>10s} {ratio:>10s} {streaming_us:10.0f}")

print("-" * 80)
print()
print("INTERPRETATION:")
print("  If MoE time is ~constant regardless of M (memory-bound): streaming the")
print("  weights dominates, M doesn't matter. Chunk size increase won't help much.")
print("  If MoE time scales DOWN with M (compute amortization): small M is")
print("  inefficient, larger M amortizes weight streaming. Two-level chunking wins.")
print("  Dense time should be much faster at large M (compute-bound at peak).")
print("  MoE/Dense ratio shows the overhead of the grouped/expert structure.")
print()
print(f"Streaming bound (all 256 experts, bf16, once): {expert_bytes/546e9*1e6:.0f} us")
print("  If MoE time ≈ streaming bound at small M, it's purely memory-bound.")
print("  If MoE time >> streaming bound at small M, there's dispatch/gather overhead too.")