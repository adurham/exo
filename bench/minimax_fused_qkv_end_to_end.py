"""End-to-end dispatch-count comparison for the fused-QKV installer.

Exercises the *production* forward path — the one ``WrappedMiniMaxAttention``
runs after ``install_fused_qkv`` has attached the merged weight. We can't
call the real wrapper from a single-process bench because it needs an
``mx.distributed.Group``, so we run the same op sequence inline.

Requires ``MLX_DISPATCH_COUNT=1`` in the env (mlx commit 22ef1101).
"""

from __future__ import annotations

import os

os.environ.setdefault("MLX_DISPATCH_COUNT", "1")

import time

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.cache import QuantizedKVCache
from mlx_lm.models.minimax import MiniMaxAttention, ModelArgs

from exo.worker.engines.mlx.patches.minimax.fused_qkv import (
    fused_qkv_is_installed,
    fused_qkv_proj,
    install_fused_qkv,
)


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


def _quantize_projs(attn: MiniMaxAttention, *, bits: int, group_size: int = 64) -> None:
    attn.q_proj = nn.QuantizedLinear.from_linear(attn.q_proj, group_size=group_size, bits=bits)
    attn.k_proj = nn.QuantizedLinear.from_linear(attn.k_proj, group_size=group_size, bits=bits)
    attn.v_proj = nn.QuantizedLinear.from_linear(attn.v_proj, group_size=group_size, bits=bits)
    mx.eval(attn.parameters())


def _attn_block(attn: MiniMaxAttention, x: mx.array, cache: QuantizedKVCache) -> mx.array:
    """Minimal reproduction of ``WrappedMiniMaxAttention.__call__`` — takes
    the fused QKV path when installed, else calls the three projections
    separately. Everything downstream of QKV is identical to the existing
    wrapper."""
    b, seq_len, _ = x.shape

    if fused_qkv_is_installed(attn):
        queries, keys, values = fused_qkv_proj(attn, x)
    else:
        queries = attn.q_proj(x)
        keys = attn.k_proj(x)
        values = attn.v_proj(x)

    if attn.use_qk_norm:
        queries = attn.q_norm(queries)
        keys = attn.k_norm(keys)

    queries = queries.reshape(b, seq_len, attn.num_attention_heads, -1).transpose(0, 2, 1, 3)
    keys = keys.reshape(b, seq_len, attn.num_key_value_heads, -1).transpose(0, 2, 1, 3)
    values = values.reshape(b, seq_len, attn.num_key_value_heads, -1).transpose(0, 2, 1, 3)

    queries = attn.rope(queries, offset=cache.offset)
    keys = attn.rope(keys, offset=cache.offset)
    keys, values = cache.update_and_fetch(keys, values)

    out = scaled_dot_product_attention(
        queries, keys, values, cache=cache, scale=attn.scale, mask=None
    )
    out = out.transpose(0, 2, 1, 3).reshape(b, seq_len, -1)
    return attn.o_proj(out)


def run(proj_label: str, *, bits: int | None, seq_ctx: int = 8192, repeats: int = 5, warmup: int = 3) -> None:
    args = make_args()
    attn = MiniMaxAttention(args)
    mx.eval(attn.parameters())
    if bits is not None:
        _quantize_projs(attn, bits=bits)

    cache = QuantizedKVCache(group_size=64, bits=8)
    x_ctx = mx.random.normal((1, seq_ctx, args.hidden_size)).astype(mx.bfloat16)
    y = _attn_block(attn, x_ctx, cache)
    mx.eval(y)

    x_dec = mx.random.normal((1, 1, args.hidden_size)).astype(mx.bfloat16)

    def bench(label: str) -> tuple[float, float]:
        for _ in range(warmup):
            z = _attn_block(attn, x_dec, cache)
            mx.eval(z)

        sum_dc = 0
        sum_dt = 0.0
        for _ in range(repeats):
            mx.metal.reset_dispatch_count()
            t0 = time.perf_counter()
            z = _attn_block(attn, x_dec, cache)
            mx.eval(z)
            sum_dt += time.perf_counter() - t0
            sum_dc += mx.metal.dispatch_count()
        return sum_dc / repeats, sum_dt / repeats * 1000

    print(f"--- {proj_label} ---")
    dc_unfused, dt_unfused = bench("unfused")
    print(f"    unfused       {dc_unfused:>5.1f} dispatches   {dt_unfused:>7.3f} ms")

    installed = install_fused_qkv(attn)
    # Cache state is independent of the projection path, but re-prefill so
    # both measurements start from the same offset.
    cache = QuantizedKVCache(group_size=64, bits=8)
    y = _attn_block(attn, x_ctx, cache)
    mx.eval(y)
    if not installed:
        print("    (install_fused_qkv: skipped — bench will show no change)")

    dc_fused, dt_fused = bench("fused  ")
    delta_dc = dc_unfused - dc_fused
    delta_dt = dt_unfused - dt_fused
    print(f"    fused         {dc_fused:>5.1f} dispatches   {dt_fused:>7.3f} ms")
    print(f"    delta         {delta_dc:+5.1f} dispatches   {delta_dt:+7.3f} ms")
    print()


def main() -> None:
    print("Per-rank shapes: hidden=3072 n_q=24 n_kv=4 head_dim=128 rotary_dim=64")
    print("(one attention block — multiply savings by ~62 for MiniMax per-token)\n")
    run("bf16 projections", bits=None)
    for bits in (4, 5, 8):
        run(f"QuantizedLinear bits={bits}", bits=bits)


if __name__ == "__main__":
    main()
