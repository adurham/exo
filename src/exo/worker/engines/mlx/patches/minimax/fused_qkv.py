"""Fused Q/K/V projection for MiniMax decode (Phase 3 Week 1).

``install_fused_qkv(attn)`` replaces ``attn.q_proj/k_proj/v_proj``'s three
separate matmuls with a single matmul against a pre-concatenated weight
matrix. The concat happens at install time (during sharding); the forward
path is ``x @ merged_Wqkv.T`` plus an ``mx.split``.

Dispatch-count effect measured on M4 Max:

    3× nn.Linear on (1,1,3072) bf16   →  6 dispatches
    1× x @ merged.T + split           →  1 dispatch

Scope of Week 1:

* Only engaged for plain :class:`mlx.nn.Linear` Q/K/V projections. If any
  of the three projections is a :class:`mlx.nn.QuantizedLinear` the
  installer is a no-op (logs a warning) and the existing per-proj code
  path stays active. Concat'ing packed uint32 quantized weights along the
  output-dim requires matching scale/bias groups and is a Week 2
  deliverable.
* Biases are assumed absent (MiniMax has ``bias=False``); asserted at
  install time.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
from loguru import logger

# Sentinel attribute set on the attention module after a successful
# install. ``WrappedMiniMaxAttention`` checks it on each forward pass.
_INSTALLED_ATTR = "_exo_fused_qkv_weight"
_Q_OUT_ATTR = "_exo_fused_qkv_q_out"
_KV_OUT_ATTR = "_exo_fused_qkv_kv_out"


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

    # Bail on quantized projections — concat'ing packed weights is Week 2.
    for name, p in (("q_proj", q_proj), ("k_proj", k_proj), ("v_proj", v_proj)):
        if not isinstance(p, nn.Linear) or isinstance(p, nn.QuantizedLinear):
            logger.info(
                f"install_fused_qkv: {name} is {type(p).__name__}, not nn.Linear — "
                "fused QKV path disabled for this layer"
            )
            return False
        if "bias" in p:
            logger.info(
                f"install_fused_qkv: {name} has a bias; MiniMax is bias=False, "
                "refusing to install"
            )
            return False

    # Weight shape for nn.Linear is (out_dims, in_dims).
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


def fused_qkv_proj(attn: nn.Module, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
    """Single-matmul Q/K/V projection. Caller must check ``fused_qkv_is_installed``."""
    w = getattr(attn, _INSTALLED_ATTR)
    q_out: int = getattr(attn, _Q_OUT_ATTR)
    kv_out: int = getattr(attn, _KV_OUT_ATTR)
    qkv = x @ w.T
    q, k, v = mx.split(qkv, [q_out, q_out + kv_out], axis=-1)
    return q, k, v
