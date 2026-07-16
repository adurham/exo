"""Test 1 (fixed): MoE-in-isolation via SwitchGLU — does 8192 tokens work?
Test 2: Dense qmm at M=192 — the go/no-go gate.
"""
import statistics
import time

import mlx.core as mx
import numpy as np
import sys
sys.path.insert(0, "/Users/adam.durham/repos/exo/mlx-lm")
from mlx_lm.models.switch_layers import SwitchGLU

# === TEST 2 FIRST (simpler, no SwitchGLU needed) ===
print("=" * 70)
print("TEST 2: Dense qmm TFLOPS at M=192 (go/no-go gate)")
print("=" * 70)

K = 4096
N = 1024  # per node (TP=2)

for M in [48, 96, 192, 384, 768, 1536]:
    w_fp = mx.random.normal((K, N), dtype=mx.float16)
    q = mx.quantize(w_fp, bits=8, group_size=64)
    w_q, w_s, w_b = q[0], q[1], q[2]
    x = mx.random.normal((M, K), dtype=mx.float16)
    mx.eval(w_q, w_s, w_b, x)
    try:
        for _ in range(5):
            out = mx.quantized_matmul(x, w_q, w_s, w_b, group_size=64, bits=8)
            mx.eval(out); mx.synchronize()
        samples = []
        for _ in range(20):
            mx.synchronize(); t0 = time.perf_counter()
            out = mx.quantized_matmul(x, w_q, w_s, w_b, group_size=64, bits=8)
            mx.eval(out); mx.synchronize()
            samples.append((time.perf_counter() - t0) * 1e6)
        t = statistics.median(samples)
        flops = 2 * M * K * N
        tflops = flops / (t * 1e-6) / 1e12
        gate = " ← GATE (>=10 = GO)" if M == 192 else ""
        print(f"  M={M:4d}: {t:6.0f} us = {tflops:5.1f} TFLOPS{gate}")
    except Exception as e:
        print(f"  M={M:4d}: FAILED: {str(e)[:80]}")

# === TEST 1: MoE via SwitchGLU ===
print()
print("=" * 70)
print("TEST 1: MoE via SwitchGLU at 8192 tokens")
print("=" * 70)

N_EXP = 256
HIDDEN = 4096  # K
INTER = 2048  # moe_intermediate (N before TP shard)

# Create a SwitchGLU with quantized weights
sglu = SwitchGLU(HIDDEN, INTER, N_EXP)

# Quantize the weights manually (SwitchGLU stores them as QuantizedSwitchLinear)
# Actually, SwitchGLU's sub-modules use QuantizedSwitchLinear which has
# w (packed), scales, biases, group_size, bits
# Let me just create the quantized weights and inject them

# Simpler approach: create the activations and indices, run through the
# actual production SwitchGLU.__call__ path
# SwitchGLU.__call__(x, indices, sorted_indices=True)
# x: (B, K) where B = tokens * top_k
# indices: (B,) expert assignments

TOP_K = 6
CHUNK = 2048
N_CHUNKS = 4
B_2048 = CHUNK * TOP_K  # 12288
B_8192 = B_2048 * 4    # 49152

# Need to quantize SwitchGLU's weights to 8-bit
# SwitchGLU has gate_proj, up_proj, down_proj as SwitchLinear modules
# Each has a .weight that needs to be quantized
# Let's create with the right shapes and quantize

print(f"  Creating SwitchGLU with {N_EXP} experts, hidden={HIDDEN}, inter={INTER}")
print(f"  Weights: {N_EXP} x {INTER} x {HIDDEN} = {N_EXP*INTER*HIDDEN*1/1e9:.1f} GB (8-bit)")

# Can't easily create quantized SwitchGLU from scratch — too much memory
# Instead, test gather_qmm directly but with the RIGHT shapes
# The issue was x[None] creating a batch dim that triggers broadcast
# SwitchGLU calls: mx.gather_qmm(x, self["weight"], self["scales"], self.get("biases"),
#                                rhs_indices=indices, transpose=True, ...)
# x is (B, K), weight is (experts, N, K_packed), indices is (B,)

# Let me use a SUBSET of experts to fit memory
N_EXP_TEST = 64
N_TEST = 1024  # per node

w_fp = mx.random.normal((N_EXP_TEST, N_TEST, HIDDEN), dtype=mx.float16)
q = mx.quantize(w_fp, bits=8, group_size=64)
w_q, w_s, w_b = q[0], q[1], q[2]
mx.eval(w_q, w_s, w_b)

print(f"\n  Testing with {N_EXP_TEST} experts (subset) to fit memory")

# Test at 2048 tokens (6 experts each = 12288 pairs)
for B in [12288, 24576, 49152]:  # 2048, 4096, 8192 tokens worth
    x = mx.random.normal((B, HIDDEN), dtype=mx.float16)
    # Assign tokens to experts (sorted)
    inds = mx.array(np.sort(np.random.choice(N_EXP_TEST, B)), dtype=mx.int32)
    mx.eval(x, inds)

    try:
        out = mx.gather_qmm(
            x, w_q, w_s, w_b,
            rhs_indices=inds,
            transpose=True,
            group_size=64, bits=8, mode="affine",
            sorted_indices=True
        )
        mx.eval(out)
        has_nan = bool(mx.isnan(out).any())
        has_inf = bool(mx.isinf(out).any())
        tokens = B // TOP_K
        print(f"  B={B:6d} ({tokens:4d} tokens): OK, shape={out.shape}, NaN={has_nan}, Inf={has_inf}")
    except Exception as e:
        tokens = B // TOP_K
        print(f"  B={B:6d} ({tokens:4d} tokens): FAILED: {str(e)[:150]}")
        break

print()
print("=== DECISION ===")
print("Test 2 gate: M=192 qmm TFLOPS >= 10 → restructure worth it")
print("Test 1: if 8192-token gather_qmm produces valid output → MoE has no hidden 2048 limit")