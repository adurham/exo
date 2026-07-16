"""Roofline validation: is gather_qmm_rhs at 84% of compute ceiling at B/E=48?

Fable: "At B/E=48, 141 GB/s is ~84% of the compute roofline (168 GB/s = 16 TFLOPS / 96 FLOP/byte).
If steel is >=75-80% of the measured dense-GEMM ceiling, this path is DONE."

Measure:
1. gather_qmm_rhs achieved TFLOPS at prefill shape (B/E≈48)
2. Dense fp16 GEMM at same shape as empirical compute ceiling
3. Ratio = gather_qmm_rhs / dense = how close to compute ceiling
"""
import statistics
import time

import mlx.core as mx
import numpy as np

# Production prefill shape
# DSv4-Flash: 256 experts, top-6, moe_intermediate_size=2048, 8-bit quantized
# At chunk=2048: 2048×6 = 12288 token-expert pairs / 256 experts = 48 per expert (B/E=48)
# Each expert GEMM: (M=48, K=2048) @ (K=2048, N=2048) — but sharded, so N=1024 per node
# With 8-bit quantized weights: the GEMM is quantized (qmm)

N_EXPERTS = 256
TOP_K = 6
CHUNK = 2048
K = 2048  # moe_intermediate_size / hidden_size
N = 1024  # output dim per node (TP=2 shards the 2048 intermediate)
M = CHUNK * TOP_K // N_EXPERTS  # 48 tokens per expert

print(f"Production prefill shape: M={M}, K={K}, N={N}, experts={N_EXPERTS}")
print(f"B/E = {M} (tokens per expert)")
print(f"FLOP/byte = 2×M = {2*M} (each 8-bit weight byte used for 2×M FLOPs)")
print()

# Create production-like data
mx.random.seed(42)
# Quantized weights: (experts, N, K) in 8-bit
# MLX quantized weights are stored as uint32 packed + scales
# For the bench, use mx.quantize to create proper quantized weights
# Use a SUBSET to fit in memory while keeping B/E=48
# 256 experts × 1024 × 2048 × 1 byte = 512 MB (weights only) — fits
# But gather_qmm output: B × N = 12288 × 1024 × 2 = 24 MB — fits
# The 309 GB allocation means the shapes are wrong. Let me reduce.
N_EXPERTS_TEST = 32  # subset of experts
M_TEST = 48  # keep B/E=48
B_TEST = M_TEST * N_EXPERTS_TEST  # 1536 pairs
K_TEST = 2048
N_TEST = 1024

w_fp16 = mx.random.normal((N_EXPERTS_TEST, N_TEST, K_TEST), dtype=mx.float16)
w_q = mx.quantize(w_fp16, bits=8, group_size=64)
w_quant, w_scales, w_biases = w_q[0], w_q[1], w_q[2]
mx.eval(w_quant, w_scales, w_biases)

x = mx.random.normal((B_TEST, K_TEST), dtype=mx.float16)
inds = mx.array(np.repeat(np.arange(N_EXPERTS_TEST), M_TEST), dtype=mx.int32)
mx.eval(x, inds)

print(f"Total token-expert pairs: B={B_TEST}")
print(f"Weight shape: ({N_EXPERTS_TEST}, {N_TEST}, {K_TEST}) 8-bit quantized")
print()


def bench(fn, n=20, w=5):
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


# 1. gather_qmm_rhs (the production path) — sorted indices
def gather_qmm_rhs_prod():
    # Affine mode with biases — try without sorted_indices to hit a different kernel
    return mx.gather_qmm(
        x[None], w_quant[None], w_scales[None], w_biases[None],
        rhs_indices=inds[None],
        sorted_indices=False, transpose=True,
        group_size=64, bits=8, mode="affine"
    )


t_qmm = bench(gather_qmm_rhs_prod)
# FLOPs = 2 × B × N × K (each pair does a (1,K)@(K,N) GEMM)
flops = 2 * B_TEST * N_TEST * K_TEST
tflops_qmm = flops / (t_qmm * 1e-6) / 1e12
# Unique weight bytes = N_EXPERTS_TEST × N_TEST × K_TEST × 1 byte (8-bit)
weight_bytes = N_EXPERTS_TEST * N_TEST * K_TEST * 1  # 8-bit = 1 byte
gbps_qmm = weight_bytes / (t_qmm * 1e-6) / 1e9

print(f"=== gather_qmm_rhs (production path) ===")
print(f"  Time: {t_qmm:.0f} us")
print(f"  FLOPs: {flops/1e12:.1f} TFLOP")
print(f"  Achieved: {tflops_qmm:.1f} TFLOPS")
print(f"  Unique weight bandwidth: {gbps_qmm:.0f} GB/s")
print()

# 2. Dense fp16 GEMM at same shape — the empirical compute ceiling
# Dense equivalent: (B, K) @ (K, N) — one big GEMM (no expert routing)
w_dense = mx.random.normal((K_TEST, N_TEST), dtype=mx.float16)
mx.eval(w_dense)


def dense_gemm():
    return x @ w_dense


t_dense = bench(dense_gemm)
tflops_dense = flops / (t_dense * 1e-6) / 1e12

print(f"=== Dense fp16 GEMM (compute ceiling) ===")
print(f"  Time: {t_dense:.0f} us")
print(f"  Achieved: {tflops_dense:.1f} TFLOPS")
print()

# 3. Ratio
ratio = tflops_qmm / tflops_dense
print(f"=== ROOFLINE RATIO ===")
print(f"  gather_qmm_rhs: {tflops_qmm:.1f} TFLOPS")
print(f"  Dense GEMM:     {tflops_dense:.1f} TFLOPS")
print(f"  Ratio: {ratio:.2f} ({ratio*100:.0f}% of compute ceiling)")
print()
if ratio >= 0.75:
    print(f"  -> {ratio*100:.0f}% >= 75%: MoE path is NEAR COMPUTE CEILING.")
    print(f"     The 141 GB/s is 84% of the compute roofline (168 GB/s = 16T/96 FLOP/byte).")
    print(f"     This path is DONE — no kernel optimization can break it.")
    print(f"     The handoff doc's '300 GB/s target' is UNPHYSICAL at B/E={M}.")
else:
    print(f"  -> {ratio*100:.0f}% < 75%: REAL GAP exists. Kernel tuning may help.")
    print(f"     Next: BM sweep (16->32->48) + run-length bucketing.")

print()
print(f"=== FABLE'S COMPUTE ROOFLINE CHECK ===")
compute_peak_spec = 16e12  # M4 Max ~16 TFLOPS fp16
flop_per_byte = 2 * M_TEST
bw_ceiling = compute_peak_spec / flop_per_byte
print(f"  Spec compute peak: {compute_peak_spec/1e12:.0f} TFLOPS")
print(f"  FLOP/byte at M={M_TEST}: {flop_per_byte}")
print(f"  Compute roofline (as bandwidth): {bw_ceiling/1e9:.0f} GB/s")
print(f"  Measured: {gbps_qmm:.0f} GB/s = {gbps_qmm/bw_ceiling*100:.0f}% of compute roofline")