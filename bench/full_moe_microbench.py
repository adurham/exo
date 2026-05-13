#!/usr/bin/env python3
"""Full SwitchGLU + post-combine microbench at DSv4-Flash decode shape.

Replicates one MoE layer's forward at c=1 decode shape:
  - gate (router) — dense matmul → topk routing
  - SwitchGLU: gate_proj + up_proj + down_proj (3 gather_qmm)
  - shared_experts (dense MLP, 3 matmul)
  - post-combine (weighted reduce + add)

Per-layer cost should be ~9ms (BUILD_PROBE per_layer ffn=9.0ms).
If we hit that target, microbench is faithful.
"""
from __future__ import annotations
import time
import argparse
import mlx.core as mx
import mlx.nn as nn


class FullMoEBench(nn.Module):
    def __init__(self):
        super().__init__()
        K = 4096
        N = 2048
        E = 256
        n_shared = 1  # shared experts count for DSv4-Flash
        
        # Routed expert weights (mxfp4 b4 g32)
        from mlx_lm.models.switch_layers import SwitchGLU, SwitchLinear
        from mlx_lm.models.deepseek_v4 import LimitedSwiGLU
        self.switch_mlp = SwitchGLU(K, N, E, activation=LimitedSwiGLU(7.0))
        
        # Shared experts (affine b8 g64)
        self.shared_gate = nn.Linear(K, N * n_shared, bias=False)
        self.shared_up = nn.Linear(K, N * n_shared, bias=False)
        self.shared_down = nn.Linear(N * n_shared, K, bias=False)
        
        # Router
        self.gate_router = nn.Linear(K, E, bias=False)


def quantize_module(mod, K, N, E, n_shared):
    # Quantize routed experts to mxfp4 b4 g32
    nn.quantize(mod.switch_mlp, group_size=32, bits=4, mode="mxfp4")
    # Quantize shared experts + router to affine b8 g64
    nn.quantize(mod.shared_gate, group_size=64, bits=8, mode="affine")
    nn.quantize(mod.shared_up, group_size=64, bits=8, mode="affine")
    nn.quantize(mod.shared_down, group_size=64, bits=8, mode="affine")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=200)
    ap.add_argument("--n-warmup", type=int, default=20)
    args = ap.parse_args()
    
    K = 4096
    N = 2048
    E = 256
    n_shared = 1
    B = 6  # num_experts_per_tok
    M = 1
    
    mod = FullMoEBench()
    quantize_module(mod, K, N, E, n_shared)
    mx.eval(mod.parameters())
    
    x = mx.random.normal(shape=(1, M, K)).astype(mx.bfloat16)
    
    def forward(x):
        # router scores
        scores = mod.gate_router(x)
        # topk routing (simulated, just pick first B)
        topk_idx = mx.random.randint(0, E, shape=(1, M, B))
        topk_w = mx.random.normal(shape=(1, M, B)).astype(mx.bfloat16)
        
        # switch_mlp expects (..., 1, K)
        y_routed = mod.switch_mlp(x.reshape(1, M, K), topk_idx.reshape(1, M, B))
        # y_routed shape: (1, M, B, K)
        
        # weighted combine
        y_combined = (y_routed * mx.expand_dims(topk_w, -1)).sum(axis=-2)
        
        # shared experts: gate-up-swiglu-down
        sg = mod.shared_gate(x)
        su = mod.shared_up(x)
        sh = mod.shared_down(nn.silu(sg) * su)
        
        return y_combined + sh
    
    # Warmup
    for _ in range(args.n_warmup):
        out = forward(x)
        mx.eval(out)
    
    times = []
    for _ in range(args.n_trials):
        t0 = time.perf_counter()
        out = forward(x)
        mx.eval(out)
        times.append((time.perf_counter() - t0) * 1000)
    
    mean_ms = sum(times) / len(times)
    p50_ms = sorted(times)[len(times) // 2]
    p99_ms = sorted(times)[int(len(times) * 0.99)]
    print(f"Full MoE forward @ M=1 B=6 K=4096 N=2048 E=256:")
    print(f"  mean={mean_ms:.4f}ms  p50={p50_ms:.4f}ms  p99={p99_ms:.4f}ms")
    print(f"  trials={args.n_trials}")
    print(f"\n  Cluster BUILD_PROBE per-layer ffn = ~9ms expected")
    print(f"  Note: this microbench is SINGLE NODE (no all_sum cross-rank wait)")


if __name__ == "__main__":
    main()
