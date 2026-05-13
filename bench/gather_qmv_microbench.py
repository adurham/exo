#!/usr/bin/env python3
"""gather_qmv microbench at DSv4-Flash decode shape.

c=1 decode, top-6 expert routing, mxfp4 weights (bits=4, group_size=32):
  x:        (1, 1, K=4096)        bf16  per-expert query
  w:        (E=256, N=2048, K=4096 / 8)  uint32  packed mxfp4 weights
  scales:   (E=256, N=2048, K/32)        bf16
  lhs_idx:  (1, B=6)               int32  token-to-batch mapping
  rhs_idx:  (1, B=6)               int32  expert indices

The actual op exercised by SwitchGLU is mx.gather_qmm. We can call it
directly via mx.fast.quantized_matmul... wait no, gather variant is
different. Let me just use SwitchLinear from mlx_lm to drive the kernel
naturally; that hits gather_qmm under the hood via QuantizedSwitchLinear.

Or use a dummy SwitchLinear instance, quantize it to mxfp4 group=32 bits=4,
and time forward at decode shape. This is the most accurate microbench
because it hits the EXACT dispatch path the runner uses.
"""
from __future__ import annotations
import time
import argparse
import mlx.core as mx
import mlx.nn as nn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=200)
    ap.add_argument("--n-warmup", type=int, default=20)
    args = ap.parse_args()
    
    # DSv4-Flash params
    K = 4096          # hidden_size
    N = 2048          # moe_intermediate_size
    E = 256           # n_routed_experts
    B = 6             # num_experts_per_tok at decode
    M = 1             # M=1 decode
    
    # Build a SwitchLinear layer with mxfp4 quantization
    from mlx_lm.models.switch_layers import SwitchLinear
    sl = SwitchLinear(K, N, E, bias=False)
    # Quantize to mxfp4 bits=4 group=32
    nn.quantize(sl, group_size=32, bits=4, mode="mxfp4")
    
    # Input shape that exercises the gather_qmm path
    x = mx.random.normal(shape=(1, M, K)).astype(mx.bfloat16)
    # Top-B random expert indices per token
    # SwitchLinear expects indices broadcasting to output shape
    indices = mx.random.randint(0, E, shape=(1, B))
    
    # Need to reshape x to (1, 1, K) -> (1, B, 1, K) via expand_dims like SwitchGLU does
    x_in = mx.expand_dims(x, (-2, -3))  # (1, 1, 1, 1, K)
    print(f"x_in shape: {x_in.shape}")
    print(f"indices shape: {indices.shape}")
    
    # Try it
    out = sl(x_in, indices, sorted_indices=False)
    mx.eval(out)
    print(f"out shape: {out.shape}")
    
    # Warmup
    for _ in range(args.n_warmup):
        out = sl(x_in, indices, sorted_indices=False)
        mx.eval(out)
    
    # Time it
    times = []
    for _ in range(args.n_trials):
        t0 = time.perf_counter()
        out = sl(x_in, indices, sorted_indices=False)
        mx.eval(out)
        times.append((time.perf_counter() - t0) * 1000)
    
    mean_ms = sum(times) / len(times)
    p50_ms = sorted(times)[len(times) // 2]
    p99_ms = sorted(times)[int(len(times) * 0.99)]
    print(f"\nSwitchLinear(mxfp4, b4 g32) @ M=1 B=6 N=2048 K=4096:")
    print(f"  mean={mean_ms:.4f}ms  p50={p50_ms:.4f}ms  p99={p99_ms:.4f}ms")
    print(f"  trials={args.n_trials}")
    
    # Per call vs cluster expectation:
    # ffn=391ms/step / 43 layers / 3 matmuls per layer = ~3.0ms per matmul expected
    print(f"\n  Cluster BUILD_PROBE expects ~3-4.5ms per gather_qmv call")


if __name__ == "__main__":
    main()
