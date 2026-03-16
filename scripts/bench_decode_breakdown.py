#!/usr/bin/env python3
"""Full decode pipeline breakdown — where does time ACTUALLY go?

Measures each component of one decode step at various context lengths:
1. Attention (SDPA) — reading KV cache
2. Feed-forward / MoE — reading expert weights
3. RoPE + norms + residuals — lightweight ops
4. Total forward pass

This tells us whether SDPA is really the bottleneck or if there's
low-hanging fruit elsewhere in the pipeline.
"""
import time
import mlx.core as mx
import mlx.nn as nn

NUM_Q_HEADS = 32
NUM_KV_HEADS = 2
HEAD_DIM = 128
HIDDEN = 4096  # Qwen3-235B hidden size
NUM_LAYERS = 94
SCALE = 1.0 / (HEAD_DIM ** 0.5)

# MoE config (Qwen3-235B)
NUM_EXPERTS_ACTIVE = 8
EXPERT_INTERMEDIATE = 2560  # moe_intermediate_size per expert
# With TP=2, each node has half the expert width
EXPERT_INTERMEDIATE_TP = EXPERT_INTERMEDIATE // 2

CONTEXTS = [1024, 8192, 32768, 45000, 65536]
WARMUP = 5
ITERS = 20


def bench(fn):
    for _ in range(WARMUP):
        fn()
    times = []
    for _ in range(ITERS):
        mx.synchronize()
        t0 = time.perf_counter_ns()
        fn()
        mx.synchronize()
        times.append((time.perf_counter_ns() - t0) / 1e6)
    return sum(times) / len(times)


def main():
    print(f"Qwen3-235B TP=2 decode breakdown")
    print(f"hidden={HIDDEN}, {NUM_Q_HEADS}Q/{NUM_KV_HEADS}KV, {NUM_EXPERTS_ACTIVE} active experts")
    print(f"expert_intermediate={EXPERT_INTERMEDIATE_TP} (per node after TP)")
    print()

    # Simulate model weights for ONE layer (MoE)
    # gate_proj + up_proj: hidden → intermediate (per active expert)
    # down_proj: intermediate → hidden
    # These are quantized 6-bit, but we'll use bf16 for bandwidth measurement
    gate_up = mx.random.normal((NUM_EXPERTS_ACTIVE, HIDDEN, EXPERT_INTERMEDIATE_TP)).astype(mx.bfloat16)
    down = mx.random.normal((NUM_EXPERTS_ACTIVE, EXPERT_INTERMEDIATE_TP, HIDDEN)).astype(mx.bfloat16)

    # Attention projections (per layer, TP sharded)
    q_proj = mx.random.normal((HIDDEN, NUM_Q_HEADS * HEAD_DIM // 2)).astype(mx.bfloat16)  # TP/2
    k_proj = mx.random.normal((HIDDEN, NUM_KV_HEADS * HEAD_DIM // 2)).astype(mx.bfloat16)
    v_proj = mx.random.normal((HIDDEN, NUM_KV_HEADS * HEAD_DIM // 2)).astype(mx.bfloat16)
    o_proj = mx.random.normal((NUM_Q_HEADS * HEAD_DIM // 2, HIDDEN)).astype(mx.bfloat16)

    mx.eval(gate_up, down, q_proj, k_proj, v_proj, o_proj)

    # Weight sizes
    attn_weight_bytes = (q_proj.nbytes + k_proj.nbytes + v_proj.nbytes + o_proj.nbytes)
    moe_weight_bytes = (gate_up.nbytes + down.nbytes)

    print(f"Per-layer weight sizes:")
    print(f"  Attention projections: {attn_weight_bytes / 1e6:.1f} MB")
    print(f"  MoE weights ({NUM_EXPERTS_ACTIVE} experts): {moe_weight_bytes / 1e6:.1f} MB")
    print(f"  Total per layer: {(attn_weight_bytes + moe_weight_bytes) / 1e6:.1f} MB")
    print(f"  Total {NUM_LAYERS} layers: {(attn_weight_bytes + moe_weight_bytes) * NUM_LAYERS / 1e6:.1f} MB")
    print()

    print(f"{'Ctx':>6}  {'SDPA':>8}  {'Attn proj':>9}  {'MoE':>8}  {'KV bytes':>9}  {'Wt bytes':>9}")
    print(f"{'':>6}  {'×94 ms':>8}  {'×94 ms':>9}  {'×94 ms':>8}  {'MB/layer':>9}  {'MB/layer':>9}")
    print("-" * 60)

    for n_ctx in CONTEXTS:
        # --- SDPA benchmark ---
        q = mx.random.normal((1, NUM_Q_HEADS, 1, HEAD_DIM)).astype(mx.bfloat16)
        k = mx.random.normal((1, NUM_KV_HEADS, n_ctx, HEAD_DIM)).astype(mx.bfloat16)
        v = mx.random.normal((1, NUM_KV_HEADS, n_ctx, HEAD_DIM)).astype(mx.bfloat16)
        mx.eval(q, k, v)

        def sdpa():
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=SCALE)
            mx.eval(out)
        t_sdpa = bench(sdpa)

        # --- Attention projection benchmark (QKV + O projection for 1 token) ---
        x = mx.random.normal((1, 1, HIDDEN)).astype(mx.bfloat16)
        mx.eval(x)

        def attn_proj():
            q_out = x @ q_proj
            k_out = x @ k_proj
            v_out = x @ v_proj
            # After SDPA, output projection
            attn_out = mx.random.normal((1, 1, NUM_Q_HEADS * HEAD_DIM // 2)).astype(mx.bfloat16)
            o_out = attn_out @ o_proj
            mx.eval(q_out, k_out, v_out, o_out)
        t_attn_proj = bench(attn_proj)

        # --- MoE benchmark (8 active experts, gate+up+down) ---
        def moe():
            # Simplified: batch matmul through active experts
            expert_in = mx.random.normal((NUM_EXPERTS_ACTIVE, 1, HIDDEN)).astype(mx.bfloat16)
            # gate_proj + up_proj (fused as one read in practice)
            gate_out = expert_in @ gate_up  # (8, 1, intermediate)
            up_out = expert_in @ gate_up    # reuse for simplicity
            hidden = nn.silu(gate_out) * up_out
            out = hidden @ down  # (8, 1, hidden)
            mx.eval(out)
        t_moe = bench(moe)

        kv_bytes_mb = NUM_KV_HEADS * n_ctx * HEAD_DIM * 2 * 2 / 1e6
        wt_bytes_mb = (attn_weight_bytes + moe_weight_bytes) / 1e6

        print(f"{n_ctx:>6}  {t_sdpa * NUM_LAYERS:>8.1f}  {t_attn_proj * NUM_LAYERS:>9.1f}  "
              f"{t_moe * NUM_LAYERS:>8.1f}  {kv_bytes_mb:>9.1f}  {wt_bytes_mb:>9.1f}")

    print()
    print("Note: Real model uses 6-bit quantized weights (3.75× smaller reads)")
    print("SDPA time scales with context. Proj/MoE time is constant.")


if __name__ == "__main__":
    main()
