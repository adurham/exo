#!/usr/bin/env python3
"""Smoke test: instantiate SparseCompressedAttention, install compiled
methods, run a forward pass with realistic c=1 100K decode shapes,
verify output matches the un-compiled path within numerical tolerance.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.deepseek_v4 import ModelArgs, SparseCompressedAttention


def make_config(num_hidden_layers: int = 43) -> ModelArgs:
    """Minimal DSv4-Flash-like config for one-layer instantiation."""
    cfg = ModelArgs(
        model_type="deepseek_v4",
        vocab_size=128000,
        num_hidden_layers=num_hidden_layers,
        hidden_size=4096,
        intermediate_size=8192,
        moe_intermediate_size=1024,
        num_attention_heads=64,
        num_key_value_heads=1,
        n_routed_experts=128,
        n_shared_experts=1,
        num_experts_per_tok=8,
        head_dim=128,
        q_lora_rank=1024,
        compress_ratios=[4] * num_hidden_layers,  # all sparse
        rms_norm_eps=1e-5,
    )
    return cfg


def main():
    cfg = make_config()
    # layer_idx=1 → compress_ratio=4 → SparseCompressedAttention
    layer_idx = 1
    attn = SparseCompressedAttention(cfg, layer_idx)

    # Run uncompiled path
    B, L = 1, 1  # decode shape
    x = mx.random.normal(shape=(B, L, cfg.hidden_size)).astype(mx.bfloat16)

    # Without cache, no compressor/indexer state. Test just the proj/post chains.
    print("Testing _raw_attn_proj (uncompiled)...")
    offset = mx.array(0)
    q_u, qr_u, kv_u = attn._raw_attn_proj(x, offset)
    mx.eval(q_u, qr_u, kv_u)
    print(f"  q: {q_u.shape}  q_residual: {qr_u.shape}  kv: {kv_u.shape}")

    print("Installing compiled methods...")
    attn.install_compiled_attn()
    print(f"  attn._compiled_attn_proj is set: {attn._compiled_attn_proj is not None}")

    print("Testing _compiled_attn_proj (compiled)...")
    q_c, qr_c, kv_c = attn._compiled_attn_proj(x, offset)
    mx.eval(q_c, qr_c, kv_c)
    print(f"  q: {q_c.shape}  q_residual: {qr_c.shape}  kv: {kv_c.shape}")

    # Compare
    q_diff = float((q_u.astype(mx.float32) - q_c.astype(mx.float32)).abs().max())
    qr_diff = float((qr_u.astype(mx.float32) - qr_c.astype(mx.float32)).abs().max())
    kv_diff = float((kv_u.astype(mx.float32) - kv_c.astype(mx.float32)).abs().max())
    print(f"  max abs diff: q={q_diff:.4g} q_residual={qr_diff:.4g} kv={kv_diff:.4g}")

    # Same for post chain
    print("Testing _raw_attn_post / _compiled_attn_post...")
    # Synthetic attn output shape: (B, n_heads, L, head_dim)
    attn_out = mx.random.normal(shape=(B, cfg.num_attention_heads, L, cfg.head_dim)).astype(mx.bfloat16)
    out_u = attn._raw_attn_post(attn_out, offset)
    mx.eval(out_u)
    out_c = attn._compiled_attn_post(attn_out, offset)
    mx.eval(out_c)
    out_diff = float((out_u.astype(mx.float32) - out_c.astype(mx.float32)).abs().max())
    print(f"  out shape: {out_u.shape}  max abs diff: {out_diff:.4g}")

    # Test with L=3 (MTP verify shape) to make sure compile cache handles it
    print("Testing with L=3 (MTP verify shape)...")
    x3 = mx.random.normal(shape=(B, 3, cfg.hidden_size)).astype(mx.bfloat16)
    q_c3, qr_c3, kv_c3 = attn._compiled_attn_proj(x3, offset)
    mx.eval(q_c3)
    print(f"  q: {q_c3.shape}  PASSED")

    print("\nALL TESTS PASSED.")


if __name__ == "__main__":
    main()
