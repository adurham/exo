"""Fused Q/K/V projection for MiniMax decode (Phase 3 Weeks 1-2).

``install_fused_qkv(attn)`` replaces ``attn.q_proj/k_proj/v_proj``'s three
separate matmuls with a single matmul against a pre-concatenated weight
matrix. The concat happens at install time (during sharding); the forward
path is ``x @ merged_Wqkv.T`` (bf16) or ``mx.quantized_matmul`` (packed
5/8-bit), followed by ``mx.split`` on the output.

Dispatch-count effect measured on M4 Max, (1, 1, 3072) bf16 input:

    three separate matmuls   →  3 dispatches
    one merged matmul+split  →  1 dispatch

The same 3 → 1 ratio holds for ``nn.QuantizedLinear`` at bits ∈ {4, 5, 8}
with group_size=64 (see ``bench/minimax_quant_concat_probe.py``).

Scope:

* Supports plain :class:`mlx.nn.Linear` and :class:`mlx.nn.QuantizedLinear`
  Q/K/V projections. Mixed types (e.g. bf16 Q, quant K/V) are refused.
* Biases on the projection are refused (MiniMax has ``bias=False``).
* Quantisation config must match across the three projections (same
  ``bits``, ``group_size``, ``mode``).
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn
from loguru import logger

# Sentinel attributes set on the attention module after a successful
# install. ``WrappedMiniMaxAttention`` checks ``_INSTALLED_ATTR`` on each
# forward pass to decide whether to take the fused path. The scales /
# biases / mode / bits / group_size attrs are present iff the install
# used the quantised path.
_INSTALLED_ATTR = "_exo_fused_qkv_weight"
_Q_OUT_ATTR = "_exo_fused_qkv_q_out"
_KV_OUT_ATTR = "_exo_fused_qkv_kv_out"
_SCALES_ATTR = "_exo_fused_qkv_scales"
_BIASES_ATTR = "_exo_fused_qkv_biases"
_BITS_ATTR = "_exo_fused_qkv_bits"
_GROUP_SIZE_ATTR = "_exo_fused_qkv_group_size"
_MODE_ATTR = "_exo_fused_qkv_mode"


def fused_qkv_is_installed(attn: nn.Module) -> bool:
    return hasattr(attn, _INSTALLED_ATTR)


def install_fused_qkv(attn: nn.Module) -> bool:
    """Attach a merged QKV weight to ``attn`` so ``fused_qkv_proj`` can run.

    Returns ``True`` on success, ``False`` if the projections aren't the
    right shape / type to fuse (in which case the attention module is
    untouched).
    """
    q_proj = getattr(attn, "q_proj", None)
    k_proj = getattr(attn, "k_proj", None)
    v_proj = getattr(attn, "v_proj", None)
    if q_proj is None or k_proj is None or v_proj is None:
        logger.warning(
            "install_fused_qkv: attn missing q_proj/k_proj/v_proj; skipping"
        )
        return False

    # Refuse biases — MiniMax is bias=False, and merging a bias would
    # require an extra add after the matmul that we'd rather not fold in
    # until the bias case actually shows up.
    for name, p in (("q_proj", q_proj), ("k_proj", k_proj), ("v_proj", v_proj)):
        if hasattr(p, "bias") and "bias" in p:
            logger.info(
                f"install_fused_qkv: {name} has a bias; MiniMax is bias=False, "
                "refusing to install"
            )
            return False

    all_quant = all(
        isinstance(p, nn.QuantizedLinear) for p in (q_proj, k_proj, v_proj)
    )
    all_bf16 = all(
        isinstance(p, nn.Linear) and not isinstance(p, nn.QuantizedLinear)
        for p in (q_proj, k_proj, v_proj)
    )
    if all_quant:
        return _install_quantized(attn, q_proj, k_proj, v_proj)
    if all_bf16:
        return _install_bf16(attn, q_proj, k_proj, v_proj)

    types = [type(p).__name__ for p in (q_proj, k_proj, v_proj)]
    logger.info(
        f"install_fused_qkv: mixed or unsupported projection types {types} — "
        "fused QKV path disabled for this layer"
    )
    return False


def _install_bf16(
    attn: nn.Module, q_proj: nn.Linear, k_proj: nn.Linear, v_proj: nn.Linear
) -> bool:
    wq = q_proj["weight"]
    wk = k_proj["weight"]
    wv = v_proj["weight"]
    if not (wq.shape[1] == wk.shape[1] == wv.shape[1]):
        logger.warning(
            f"install_fused_qkv: input-dim mismatch "
            f"(q={wq.shape[1]}, k={wk.shape[1]}, v={wv.shape[1]}); skipping"
        )
        return False
    if not (wq.dtype == wk.dtype == wv.dtype):
        logger.warning(
            f"install_fused_qkv: dtype mismatch "
            f"(q={wq.dtype}, k={wk.dtype}, v={wv.dtype}); skipping"
        )
        return False

    merged = mx.concatenate([wq, wk, wv], axis=0)
    mx.eval(merged)
    setattr(attn, _INSTALLED_ATTR, merged)
    setattr(attn, _Q_OUT_ATTR, int(wq.shape[0]))
    setattr(attn, _KV_OUT_ATTR, int(wk.shape[0]))
    return True


def _install_quantized(
    attn: nn.Module,
    q_proj: nn.QuantizedLinear,
    k_proj: nn.QuantizedLinear,
    v_proj: nn.QuantizedLinear,
) -> bool:
    projs: list[nn.QuantizedLinear] = [q_proj, k_proj, v_proj]
    bits_set = {int(p.bits) for p in projs}
    group_set = {int(p.group_size) for p in projs}
    mode_set = {str(p.mode) for p in projs}
    if len(bits_set) != 1 or len(group_set) != 1 or len(mode_set) != 1:
        logger.warning(
            f"install_fused_qkv (quantized): config mismatch "
            f"bits={bits_set} group_size={group_set} mode={mode_set}; skipping"
        )
        return False
    mode = next(iter(mode_set))
    if mode != "affine":
        logger.info(
            f"install_fused_qkv (quantized): mode={mode!r} not supported "
            "(only 'affine' for now); skipping"
        )
        return False

    wq, wk, wv = q_proj["weight"], k_proj["weight"], v_proj["weight"]
    if not (wq.shape[1] == wk.shape[1] == wv.shape[1]):
        logger.warning(
            f"install_fused_qkv (quantized): packed input-dim mismatch "
            f"(q={wq.shape[1]}, k={wk.shape[1]}, v={wv.shape[1]}); skipping"
        )
        return False

    sq, sk, sv = q_proj["scales"], k_proj["scales"], v_proj["scales"]
    bq, bk, bv = (
        q_proj.get("biases", None),
        k_proj.get("biases", None),
        v_proj.get("biases", None),
    )
    has_biases = all(x is not None for x in (bq, bk, bv))
    has_none = all(x is None for x in (bq, bk, bv))
    if not (has_biases or has_none):
        logger.warning(
            "install_fused_qkv (quantized): inconsistent biases presence "
            f"(q={bq is not None}, k={bk is not None}, v={bv is not None}); skipping"
        )
        return False

    merged_w = mx.concatenate([wq, wk, wv], axis=0)
    merged_s = mx.concatenate([sq, sk, sv], axis=0)
    merged_b = mx.concatenate([bq, bk, bv], axis=0) if has_biases else None
    mx.eval(merged_w, merged_s)
    if merged_b is not None:
        mx.eval(merged_b)

    setattr(attn, _INSTALLED_ATTR, merged_w)
    setattr(attn, _SCALES_ATTR, merged_s)
    setattr(attn, _BIASES_ATTR, merged_b)
    setattr(attn, _BITS_ATTR, int(next(iter(bits_set))))
    setattr(attn, _GROUP_SIZE_ATTR, int(next(iter(group_set))))
    setattr(attn, _MODE_ATTR, mode)
    setattr(attn, _Q_OUT_ATTR, int(wq.shape[0]))
    setattr(attn, _KV_OUT_ATTR, int(wk.shape[0]))
    return True


def fused_qkv_proj(attn: nn.Module, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
    """Single-matmul Q/K/V projection. Caller must check ``fused_qkv_is_installed``."""
    w = getattr(attn, _INSTALLED_ATTR)
    q_out: int = getattr(attn, _Q_OUT_ATTR)
    kv_out: int = getattr(attn, _KV_OUT_ATTR)

    scales: Any = getattr(attn, _SCALES_ATTR, None)
    if scales is not None:
        biases: Any = getattr(attn, _BIASES_ATTR, None)
        bits: int = getattr(attn, _BITS_ATTR)
        group_size: int = getattr(attn, _GROUP_SIZE_ATTR)
        mode: str = getattr(attn, _MODE_ATTR)
        qkv = mx.quantized_matmul(
            x,
            w,
            scales=scales,
            biases=biases,
            transpose=True,
            group_size=group_size,
            bits=bits,
            mode=mode,
        )
    else:
        qkv = x @ w.T

    q, k, v = mx.split(qkv, [q_out, q_out + kv_out], axis=-1)
    return q, k, v
