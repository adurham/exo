"""MiniMax-specific fused-attention patches (Phase 3 skeleton).

Enabled by setting ``EXO_MINIMAX_FUSED_ATTN=1``. When off the existing
``WrappedMiniMaxAttention`` path runs unchanged.

This module is the scaffold for the Phase 3 fused-attention Metal kernel
project. Weeks 1-2 ship one fusion:

    q_proj(x), k_proj(x), v_proj(x)   →   x @ merged_Wqkv, then split

which collapses 3 separate matmul dispatches into 1 and demonstrates the
load-time weight-merge + attention-wrapper wiring that later phases will
reuse for RMSNorm+RoPE+cache+SDPA fusion.

Both plain :class:`mlx.nn.Linear` (bf16) and :class:`mlx.nn.QuantizedLinear`
(affine 4/5/8-bit) projections are supported; the production 5-bit
MiniMax checkpoint hits the quantised path. Mixed-type or biased
projections bail silently, leaving the per-proj code path active.
"""

from .fused_qkv import fused_qkv_is_installed, install_fused_qkv

__all__ = ["install_fused_qkv", "fused_qkv_is_installed"]
