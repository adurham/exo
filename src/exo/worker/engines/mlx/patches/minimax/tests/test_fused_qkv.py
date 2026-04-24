"""Tests for Phase 3 Week-1 fused-QKV projection skeleton."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from exo.worker.engines.mlx.patches.minimax.fused_qkv import (
    fused_qkv_is_installed,
    fused_qkv_proj,
    install_fused_qkv,
)


class _MockMiniMaxAttn(nn.Module):
    """Minimal stand-in for MiniMaxAttention's projection surface."""

    def __init__(self, hidden: int, q_out: int, kv_out: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden, q_out, bias=False)
        self.k_proj = nn.Linear(hidden, kv_out, bias=False)
        self.v_proj = nn.Linear(hidden, kv_out, bias=False)


def _make_attn(hidden: int = 3072, q_out: int = 3072, kv_out: int = 512) -> _MockMiniMaxAttn:
    attn = _MockMiniMaxAttn(hidden, q_out, kv_out)
    mx.eval(attn.parameters())
    return attn


def test_install_fused_qkv_creates_merged_weight() -> None:
    attn = _make_attn()
    assert not fused_qkv_is_installed(attn)
    ok = install_fused_qkv(attn)
    assert ok is True
    assert fused_qkv_is_installed(attn)


def test_fused_output_matches_three_linears_bitwise() -> None:
    attn = _make_attn()
    x = mx.random.normal((1, 1, 3072)).astype(mx.bfloat16)
    mx.eval(x)

    # Reference: three separate matmuls
    q_ref = attn.q_proj(x)
    k_ref = attn.k_proj(x)
    v_ref = attn.v_proj(x)
    mx.eval(q_ref, k_ref, v_ref)

    install_fused_qkv(attn)
    q, k, v = fused_qkv_proj(attn, x)
    mx.eval(q, k, v)

    # Same math ⇒ bitwise identical is the goal, but bf16 GEMM can have
    # tile-order differences. Allow a small absolute tolerance.
    max_q = float(mx.max(mx.abs(q - q_ref)).item())
    max_k = float(mx.max(mx.abs(k - k_ref)).item())
    max_v = float(mx.max(mx.abs(v - v_ref)).item())
    assert max_q < 2e-2, f"Q delta {max_q} exceeds tolerance"
    assert max_k < 2e-2, f"K delta {max_k} exceeds tolerance"
    assert max_v < 2e-2, f"V delta {max_v} exceeds tolerance"


def test_fused_output_works_at_prefill_shapes() -> None:
    attn = _make_attn()
    x = mx.random.normal((1, 128, 3072)).astype(mx.bfloat16)
    mx.eval(x)

    q_ref, k_ref, v_ref = attn.q_proj(x), attn.k_proj(x), attn.v_proj(x)
    mx.eval(q_ref, k_ref, v_ref)

    install_fused_qkv(attn)
    q, k, v = fused_qkv_proj(attn, x)
    mx.eval(q, k, v)

    assert q.shape == q_ref.shape
    assert k.shape == k_ref.shape
    assert v.shape == v_ref.shape
    assert float(mx.max(mx.abs(q - q_ref)).item()) < 2e-2
    assert float(mx.max(mx.abs(k - k_ref)).item()) < 2e-2
    assert float(mx.max(mx.abs(v - v_ref)).item()) < 2e-2


def test_install_skips_quantized_projection() -> None:
    attn = _make_attn()
    attn.q_proj = nn.QuantizedLinear.from_linear(attn.q_proj, bits=4)
    mx.eval(attn.parameters())
    ok = install_fused_qkv(attn)
    assert ok is False, "Quantized Q projection must not be fused in Week 1"
    assert not fused_qkv_is_installed(attn)


def test_install_skips_when_bias_present() -> None:
    attn = _MockMiniMaxAttn(3072, 3072, 512)
    attn.q_proj = nn.Linear(3072, 3072, bias=True)
    mx.eval(attn.parameters())
    ok = install_fused_qkv(attn)
    assert ok is False
    assert not fused_qkv_is_installed(attn)


def test_fused_dispatch_count_is_lower_than_three_linears() -> None:
    """Sanity check — the whole point of the fusion. Requires
    ``MLX_DISPATCH_COUNT=1`` in the env (see mlx patch ``22ef1101``)."""
    import os

    if os.environ.get("MLX_DISPATCH_COUNT") != "1":
        pytest.skip("MLX_DISPATCH_COUNT=1 not set — skipping dispatch-count check")

    attn = _make_attn()
    x = mx.random.normal((1, 1, 3072)).astype(mx.bfloat16)
    mx.eval(x)

    # Warm caches on both paths
    q = attn.q_proj(x)
    k = attn.k_proj(x)
    v = attn.v_proj(x)
    mx.eval(q, k, v)

    mx.metal.reset_dispatch_count()
    q = attn.q_proj(x)
    k = attn.k_proj(x)
    v = attn.v_proj(x)
    mx.eval(q, k, v)
    ref_dc = mx.metal.dispatch_count()

    install_fused_qkv(attn)
    # Warm the fused path too
    qf, kf, vf = fused_qkv_proj(attn, x)
    mx.eval(qf, kf, vf)

    mx.metal.reset_dispatch_count()
    qf, kf, vf = fused_qkv_proj(attn, x)
    mx.eval(qf, kf, vf)
    fused_dc = mx.metal.dispatch_count()

    assert fused_dc < ref_dc, (
        f"Fused QKV expected fewer dispatches than separate; "
        f"got fused={fused_dc} ref={ref_dc}"
    )
