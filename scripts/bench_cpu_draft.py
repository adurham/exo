#!/usr/bin/env python3
"""CPU-only draft model inference via numpy/Accelerate.

Dequantizes Qwen3-0.6B weights using MLX, converts to numpy float32,
then runs forward pass entirely on CPU. Measures speed for speculative decode viability.
"""
import time
import numpy as np
import mlx.core as mx
from mlx_lm.utils import load


def to_np32(arr):
    """MLX array → numpy float32."""
    return np.array(arr.astype(mx.float32))


def extract_weights(model):
    """Extract all weights as numpy float32 using MLX dequantization."""
    inner = model.model
    args = model.args

    # Embedding: call it with each token ID to get dequantized vectors
    # Or dequantize the weight directly
    emb = inner.embed_tokens
    emb_weight = to_np32(mx.dequantize(
        emb.weight, emb.scales, emb.biases, emb.group_size, emb.bits))

    layers = []
    for layer in inner.layers:
        attn = layer.self_attn
        mlp = layer.mlp
        ld = {}

        ld['in_norm'] = to_np32(layer.input_layernorm.weight)
        ld['post_norm'] = to_np32(layer.post_attention_layernorm.weight)

        # Dequantize attention projections
        for name, proj in [('q', attn.q_proj), ('k', attn.k_proj),
                           ('v', attn.v_proj), ('o', attn.o_proj)]:
            w = mx.dequantize(proj.weight, proj.scales,
                              getattr(proj, 'biases', None),
                              proj.group_size, proj.bits)
            ld[name] = to_np32(w)

        # Q/K norms
        if hasattr(attn, 'q_norm'):
            ld['q_norm'] = to_np32(attn.q_norm.weight)
            ld['k_norm'] = to_np32(attn.k_norm.weight)

        # Dequantize MLP
        for name, proj in [('gate', mlp.gate_proj), ('up', mlp.up_proj),
                           ('down', mlp.down_proj)]:
            w = mx.dequantize(proj.weight, proj.scales,
                              getattr(proj, 'biases', None),
                              proj.group_size, proj.bits)
            ld[name] = to_np32(w)

        ld['n_heads'] = attn.n_heads
        ld['n_kv'] = attn.n_kv_heads
        ld['scale'] = attn.scale
        ld['head_dim'] = args.head_dim

        layers.append(ld)

    final_norm = to_np32(inner.norm.weight)

    # LM head (tied to embedding for 0.6B)
    if args.tie_word_embeddings:
        lm_head = emb_weight
    else:
        lm = model.lm_head
        lm_head = to_np32(mx.dequantize(
            lm.weight, lm.scales, getattr(lm, 'biases', None),
            lm.group_size, lm.bits))

    return emb_weight, layers, final_norm, lm_head


def rms_norm(x, w, eps=1e-6):
    return x / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps) * w


def silu(x):
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -88, 88))))


def rope(x, offset, head_dim, theta=1000000.0):
    """Apply RoPE. x: (n_heads, 1, head_dim)"""
    half = head_dim // 2
    freqs = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    angle = float(offset) * freqs  # (half,)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * cos_a - x2 * sin_a, x1 * sin_a + x2 * cos_a], axis=-1)


def forward(tok_id, emb, layers, fnorm, lm_head, cache, offset):
    """Single-token forward pass. Returns logits, updates cache in-place."""
    h = emb[tok_id:tok_id+1]  # (1, hidden)

    for i, ld in enumerate(layers):
        n_h, n_kv, hd = ld['n_heads'], ld['n_kv'], ld['head_dim']
        gqa = n_h // n_kv

        # Attention
        x = rms_norm(h, ld['in_norm'])
        q = (x @ ld['q'].T).reshape(n_h, 1, hd)
        k = (x @ ld['k'].T).reshape(n_kv, 1, hd)
        v = (x @ ld['v'].T).reshape(n_kv, 1, hd)

        if 'q_norm' in ld:
            q = rms_norm(q, ld['q_norm'])
            k = rms_norm(k, ld['k_norm'])

        q = rope(q, offset, hd)
        k = rope(k, offset, hd)

        # KV cache
        if cache[i] is None:
            cache[i] = (k, v)
        else:
            pk, pv = cache[i]
            cache[i] = (np.concatenate([pk, k], axis=1),
                        np.concatenate([pv, v], axis=1))
        ck, cv = cache[i]

        # GQA attention
        out = np.zeros((n_h, 1, hd), dtype=np.float32)
        for kh in range(n_kv):
            kk, vv = ck[kh], cv[kh]  # (seq, hd)
            for g in range(gqa):
                qh = q[kh * gqa + g]  # (1, hd)
                s = (qh @ kk.T) * ld['scale']
                s = np.exp(s - s.max(axis=-1, keepdims=True))
                s = s / s.sum(axis=-1, keepdims=True)
                out[kh * gqa + g] = s @ vv

        h = h + out.reshape(1, n_h * hd) @ ld['o'].T

        # MLP
        x = rms_norm(h, ld['post_norm'])
        h = h + (silu(x @ ld['gate'].T) * (x @ ld['up'].T)) @ ld['down'].T

    h = rms_norm(h, fnorm)
    return h @ lm_head.T


def main():
    print("Loading and dequantizing Qwen3-0.6B...")
    t0 = time.perf_counter()
    model, tok = load('mlx-community/Qwen3-0.6B-8bit')
    mx.eval(model.parameters())
    emb, layers, fnorm, lm_head = extract_weights(model)
    del model; mx.clear_cache()
    print(f"Ready in {time.perf_counter()-t0:.1f}s")
    print(f"Layers: {len(layers)}, Hidden: {emb.shape[1]}, Vocab: {lm_head.shape[0]}")

    # Warmup
    cache = [None] * len(layers)
    for i in range(3):
        forward(1, emb, layers, fnorm, lm_head, cache, i)

    # Benchmark
    cache = [None] * len(layers)
    times = []
    for i in range(10):
        t0 = time.perf_counter_ns()
        logits = forward(1, emb, layers, fnorm, lm_head, cache, i)
        times.append((time.perf_counter_ns() - t0) / 1e6)

    avg = sum(times) / len(times)
    print(f"\nCPU forward pass: {avg:.1f}ms per token")
    print(f"3 draft tokens: {avg*3:.1f}ms")
    draft_3 = avg * 3
    verify = 40  # ms
    acceptance = 0.63
    tok_per_step = 1 + acceptance + acceptance**2 + acceptance**3
    total = draft_3 + verify
    effective = total / tok_per_step
    print(f"\nWith 63% acceptance, 3 draft tokens:")
    print(f"  Draft: {draft_3:.0f}ms + Verify: {verify}ms = {total:.0f}ms")
    print(f"  Tokens/step: {tok_per_step:.2f}")
    print(f"  Effective: {effective:.1f}ms/tok")
    print(f"  Baseline: 35ms/tok")
    print(f"  Speedup: {35/effective:.2f}×")


if __name__ == "__main__":
    main()
