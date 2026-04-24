"""Dispatch-count microbench for MiniMax decode-layer attention.

Counts Metal kernel dispatches per decode-token attention block call via the
patched ``mx.metal.dispatch_count()`` counter. Goal: validate Phase 3's
assumption that the attention block issues ~5-7 dispatches per layer, and
confirm the dispatch-count-reduction hypothesis that underpins the fused
kernel project.

Runs locally against a single synthesized ``MiniMaxAttention`` layer at
per-rank shapes (n_q=24, n_kv=4, head_dim=128, hidden=6144) — dispatch
counts are weight-independent, only the op graph matters.
"""

from __future__ import annotations

import time

import mlx.core as mx
from mlx_lm.models.cache import KVCache, QuantizedKVCache
from mlx_lm.models.minimax import MiniMaxAttention, ModelArgs


def make_args(use_qk_norm: bool = True) -> ModelArgs:
    return ModelArgs(
        model_type="minimax",
        hidden_size=3072,  # per-rank (48/2 × 128). Using per-rank shape keeps projection-matmul dispatches representative of the cluster.
        intermediate_size=1536,  # unused here
        num_attention_heads=24,  # per-rank
        num_key_value_heads=4,   # per-rank
        max_position_embeddings=200000,
        num_experts_per_tok=8,
        num_local_experts=256,
        num_hidden_layers=1,
        rms_norm_eps=1e-5,
        rope_theta=10_000_000.0,
        rotary_dim=64,
        vocab_size=200064,
        head_dim=128,
        use_qk_norm=use_qk_norm,
    )


def prefill_cache(attn: MiniMaxAttention, cache, x_ctx: mx.array) -> None:
    y = attn(x_ctx, mask=None, cache=cache)
    mx.eval(y)


def measure(cache_type: str, bits: int, seq_ctx: int, warmup: int, repeats: int):
    args = make_args()
    attn = MiniMaxAttention(args)
    mx.eval(attn.parameters())

    if cache_type == "quant":
        cache = QuantizedKVCache(group_size=64, bits=bits)
    elif cache_type == "bf16":
        cache = KVCache()
    else:
        raise ValueError(cache_type)

    x_ctx = mx.random.normal((1, seq_ctx, args.hidden_size)).astype(mx.bfloat16)
    prefill_cache(attn, cache, x_ctx)

    x_dec = mx.random.normal((1, 1, args.hidden_size)).astype(mx.bfloat16)

    for _ in range(warmup):
        y = attn(x_dec, mask=None, cache=cache)
        mx.eval(y)

    samples = []
    for _ in range(repeats):
        mx.metal.reset_dispatch_count()
        t0 = time.perf_counter()
        y = attn(x_dec, mask=None, cache=cache)
        mx.eval(y)
        dt = time.perf_counter() - t0
        samples.append((mx.metal.dispatch_count(), dt))

    return samples, cache.offset


def measure_sub_ops(cache_type: str, bits: int, seq_ctx: int):
    """Per-sub-op dispatch breakdown for one decode-step attention block.

    Manually reproduces ``MiniMaxAttention.__call__`` so we can eval() at
    each sub-op boundary and delta the dispatch counter. This mirrors the
    5 sub-spans already tracked in ``auto_parallel.py``:
      qkv_proj, qk_norm, reshape_rope_cache, sdpa, o_proj.
    """
    from mlx_lm.models.base import scaled_dot_product_attention

    args = make_args()
    attn = MiniMaxAttention(args)
    mx.eval(attn.parameters())
    if cache_type == "quant":
        cache = QuantizedKVCache(group_size=64, bits=bits)
    else:
        cache = KVCache()
    prefill_cache(attn, cache, mx.random.normal((1, seq_ctx, args.hidden_size)).astype(mx.bfloat16))

    x = mx.random.normal((1, 1, args.hidden_size)).astype(mx.bfloat16)

    # Warmup
    for _ in range(3):
        y = attn(x, mask=None, cache=cache)
        mx.eval(y)

    def delta(fn):
        mx.metal.reset_dispatch_count()
        t0 = time.perf_counter()
        out = fn()
        if out is not None:
            mx.eval(out)
        dt = time.perf_counter() - t0
        return out, mx.metal.dispatch_count(), dt

    b, seq_len, _d = x.shape

    # 1. q/k/v projections (three matmuls)
    def _qkv():
        return (attn.q_proj(x), attn.k_proj(x), attn.v_proj(x))
    (q, k, v), qkv_dc, qkv_dt = delta(_qkv)

    # 2. qk_norm
    def _qkn():
        return attn.q_norm(q), attn.k_norm(k)
    (qn, kn), qkn_dc, qkn_dt = delta(_qkn)

    # 3. reshape + RoPE + cache update
    def _rope_cache():
        queries = qn.reshape(b, seq_len, attn.num_attention_heads, -1).transpose(0, 2, 1, 3)
        keys = kn.reshape(b, seq_len, attn.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        values = v.reshape(b, seq_len, attn.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        queries = attn.rope(queries, offset=cache.offset)
        keys = attn.rope(keys, offset=cache.offset)
        keys, values = cache.update_and_fetch(keys, values)
        return queries, keys, values
    (queries, keys, values), rc_dc, rc_dt = delta(_rope_cache)

    # 4. SDPA
    def _sdpa():
        return scaled_dot_product_attention(queries, keys, values, cache=cache, scale=attn.scale, mask=None)
    (sdpa_out,), sdpa_dc, sdpa_dt = (lambda: delta(lambda: (_sdpa(),)))()

    # 5. reshape + o_proj
    def _op():
        o = sdpa_out.transpose(0, 2, 1, 3).reshape(b, seq_len, -1)
        return attn.o_proj(o)
    _, op_dc, op_dt = delta(_op)

    return {
        "qkv_proj": (qkv_dc, qkv_dt),
        "qk_norm": (qkn_dc, qkn_dt),
        "reshape_rope_cache": (rc_dc, rc_dt),
        "sdpa": (sdpa_dc, sdpa_dt),
        "o_proj": (op_dc, op_dt),
    }


def measure_prefill_dispatches(cache_type: str, bits: int, seq_ctx: int):
    args = make_args()
    attn = MiniMaxAttention(args)
    mx.eval(attn.parameters())
    if cache_type == "quant":
        cache = QuantizedKVCache(group_size=64, bits=bits)
    else:
        cache = KVCache()
    x_ctx = mx.random.normal((1, seq_ctx, args.hidden_size)).astype(mx.bfloat16)

    mx.metal.reset_dispatch_count()
    t0 = time.perf_counter()
    y = attn(x_ctx, mask=None, cache=cache)
    mx.eval(y)
    dt = time.perf_counter() - t0
    return mx.metal.dispatch_count(), dt


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-ctx", type=int, default=8192)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    print(f"seq_ctx={args.seq_ctx}  warmup={args.warmup}  repeats={args.repeats}")
    print("Per-rank shapes: hidden=3072 n_q=24 n_kv=4 head_dim=128 rotary_dim=64")
    print()

    configs = [
        ("quant", 8, "QuantKV bits=8 (cluster config)"),
        ("quant", 4, "QuantKV bits=4"),
        ("bf16", 0, "bf16 KV"),
    ]
    for cache_type, bits, label in configs:
        prefill_dc, prefill_dt = measure_prefill_dispatches(cache_type, bits, args.seq_ctx)
        decode_samples, offset = measure(
            cache_type=cache_type, bits=bits, seq_ctx=args.seq_ctx,
            warmup=args.warmup, repeats=args.repeats,
        )

        print(f"--- {label}  (cache offset={offset}) ---")
        print(f"  prefill ({args.seq_ctx}-tok, 1 attn block): {prefill_dc} dispatches   {prefill_dt*1000:.2f} ms")
        for i, (dc, dt) in enumerate(decode_samples):
            print(f"  decode run {i+1}: {dc:>4} dispatches   {dt*1000:>7.2f} ms")
        avg_dc = sum(d for d, _ in decode_samples) / len(decode_samples)
        avg_dt = sum(d for _, d in decode_samples) / len(decode_samples) * 1000
        print(f"  decode avg:   {avg_dc:.1f} dispatches   {avg_dt:.2f} ms")
        print()

    print("=" * 70)
    print("Per-sub-op decode dispatch breakdown (cluster config: QuantKV bits=8)")
    print("=" * 70)
    sub = measure_sub_ops("quant", 8, args.seq_ctx)
    total_dc = sum(dc for dc, _ in sub.values())
    total_dt = sum(dt for _, dt in sub.values()) * 1000
    for name, (dc, dt) in sub.items():
        pct = 100 * dc / total_dc if total_dc else 0
        print(f"  {name:<22s} {dc:>3} dispatches  ({pct:>5.1f}%)   {dt*1000:>7.3f} ms")
    print(f"  {'total':<22s} {total_dc:>3} dispatches             {total_dt:>7.3f} ms")


if __name__ == "__main__":
    main()
