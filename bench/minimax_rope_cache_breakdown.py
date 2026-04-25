"""Per-step dispatch breakdown for the reshape_rope_cache sub-op.

The Phase-3 sub-op profile said reshape_rope_cache = 12 dispatches per
attention block per decode token (vs sdpa = 2, qkv_proj = 6, qk_norm = 2,
o_proj = 1). Before deciding what to fuse in Week 3, measure exactly which
of those 12 come from where.

Output:
    reshape Q                  X dispatches
    reshape K
    reshape V
    transpose Q
    transpose K
    transpose V
    rope Q
    rope K
    cache.update_and_fetch
        quantize K
        quantize V
        scatter K + V

Requires MLX_DISPATCH_COUNT=1 in the env.
"""

from __future__ import annotations

import os

os.environ.setdefault("MLX_DISPATCH_COUNT", "1")

import time

import mlx.core as mx

from mlx_lm.models.cache import QuantizedKVCache
from mlx_lm.models.minimax import MiniMaxAttention, ModelArgs


def make_args() -> ModelArgs:
    return ModelArgs(
        model_type="minimax",
        hidden_size=3072,
        intermediate_size=1536,
        num_attention_heads=24,
        num_key_value_heads=4,
        max_position_embeddings=200000,
        num_experts_per_tok=8,
        num_local_experts=256,
        num_hidden_layers=1,
        rms_norm_eps=1e-5,
        rope_theta=10_000_000.0,
        rotary_dim=64,
        vocab_size=200064,
        head_dim=128,
        use_qk_norm=True,
    )


def delta(label: str, fn):
    mx.metal.reset_dispatch_count()
    t0 = time.perf_counter()
    out = fn()
    if isinstance(out, tuple):
        mx.eval(*out)
    else:
        mx.eval(out)
    dt = time.perf_counter() - t0
    dc = mx.metal.dispatch_count()
    print(f"  {label:<32s} {dc:>3} dispatches  {dt*1000:>7.3f} ms")
    return out, dc


def main() -> None:
    args = make_args()
    attn = MiniMaxAttention(args)
    mx.eval(attn.parameters())
    cache = QuantizedKVCache(group_size=64, bits=8)

    seq_ctx = 8192
    x_ctx = mx.random.normal((1, seq_ctx, args.hidden_size)).astype(mx.bfloat16)
    y = attn(x_ctx, mask=None, cache=cache)
    mx.eval(y)

    x = mx.random.normal((1, 1, args.hidden_size)).astype(mx.bfloat16)

    # Warmup
    for _ in range(3):
        y = attn(x, mask=None, cache=cache)
        mx.eval(y)

    print(f"Per-step dispatches (MiniMax cluster shape, decode L=1, ctx={seq_ctx}, KV bits=8):\n")

    # Pre-projection so we have q/k/v that match what enters reshape_rope_cache
    q_pre = attn.q_proj(x)
    k_pre = attn.k_proj(x)
    v_pre = attn.v_proj(x)
    if attn.use_qk_norm:
        q_pre = attn.q_norm(q_pre)
        k_pre = attn.k_norm(k_pre)
    mx.eval(q_pre, k_pre, v_pre)

    b, seq_len, _ = x.shape
    n_q = attn.num_attention_heads
    n_kv = attn.num_key_value_heads

    # 1-3: per-array reshape — likely views (no dispatch)
    (q,), _ = delta("reshape Q",
                    lambda: (q_pre.reshape(b, seq_len, n_q, -1),))
    (k,), _ = delta("reshape K",
                    lambda: (k_pre.reshape(b, seq_len, n_kv, -1),))
    (v,), _ = delta("reshape V",
                    lambda: (v_pre.reshape(b, seq_len, n_kv, -1),))

    # 4-6: transposes
    (qT,), _ = delta("transpose Q (0,2,1,3)",
                     lambda: (q.transpose(0, 2, 1, 3),))
    (kT,), _ = delta("transpose K (0,2,1,3)",
                     lambda: (k.transpose(0, 2, 1, 3),))
    (vT,), _ = delta("transpose V (0,2,1,3)",
                     lambda: (v.transpose(0, 2, 1, 3),))

    # 7-8: RoPE
    offset = cache.offset
    (qR,), _ = delta("rope Q",
                     lambda: (attn.rope(qT, offset=offset),))
    (kR,), _ = delta("rope K",
                     lambda: (attn.rope(kT, offset=offset),))

    # 9: cache.update_and_fetch (likely 4 dispatches — quantize K, quantize V, scatter K, scatter V)
    delta("cache.update_and_fetch",
          lambda: cache.update_and_fetch(kR, vT))

    print("\nNow Week 3 candidate: joined K+V transpose, joined Q+K rope:")

    # Reset cache so we don't double-append
    cache_alt = QuantizedKVCache(group_size=64, bits=8)
    y = attn(x_ctx, mask=None, cache=cache_alt)
    mx.eval(y)
    for _ in range(3):
        y = attn(x, mask=None, cache=cache_alt)
        mx.eval(y)
    q_pre = attn.q_proj(x)
    k_pre = attn.k_proj(x)
    v_pre = attn.v_proj(x)
    if attn.use_qk_norm:
        q_pre = attn.q_norm(q_pre)
        k_pre = attn.k_norm(k_pre)
    mx.eval(q_pre, k_pre, v_pre)

    # Fuse Q+K into one tensor along head dim, transpose once
    def _joined_qk_transpose():
        # Put Q and K side-by-side as a single (B, L, n_q+n_kv, head_dim)
        # tensor, transpose to (B, n_q+n_kv, L, head_dim) in one call.
        qk = mx.concatenate([q_pre, k_pre], axis=-1)  # (B, L, (n_q+n_kv)*head_dim)
        return qk.reshape(b, seq_len, n_q + n_kv, -1).transpose(0, 2, 1, 3)
    (qkT,), _ = delta("joined Q+K reshape+transpose", lambda: (_joined_qk_transpose(),))

    # One RoPE on the whole thing
    offset = cache_alt.offset
    (qkR,), _ = delta("joined RoPE on Q+K", lambda: (attn.rope(qkT, offset=offset),))

    # Split back. Slice along axis=1 — should be free (view).
    (qR_alt, kR_alt), _ = delta("split Q,K from joined",
                                lambda: (qkR[:, :n_q, :, :], qkR[:, n_q:, :, :]))

    # V transpose alone
    (vT_alt,), _ = delta("transpose V alone",
                         lambda: (v_pre.reshape(b, seq_len, n_kv, -1).transpose(0, 2, 1, 3),))

    # Numerical equivalence
    qref = attn.rope(q_pre.reshape(b, seq_len, n_q, -1).transpose(0, 2, 1, 3), offset=offset)
    kref = attn.rope(k_pre.reshape(b, seq_len, n_kv, -1).transpose(0, 2, 1, 3), offset=offset)
    mx.eval(qref, kref, qR_alt, kR_alt)
    print(f"\n  numerical delta Q: {float(mx.max(mx.abs(qref - qR_alt)).item()):.2e}")
    print(f"  numerical delta K: {float(mx.max(mx.abs(kref - kR_alt)).item()):.2e}")


if __name__ == "__main__":
    main()
