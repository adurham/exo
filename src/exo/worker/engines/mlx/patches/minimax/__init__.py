"""MiniMax-specific fused-attention patches (Phase 3 skeleton).

Enabled by setting ``EXO_MINIMAX_FUSED_ATTN=1``. When off the existing
``WrappedMiniMaxAttention`` path runs unchanged.

This module is the scaffold for the Phase 3 fused-attention Metal kernel
project. Week 1 ships one fusion:

    q_proj(x), k_proj(x), v_proj(x)   →   x @ merged_Wqkv, then split

which collapses 3 separate matmul dispatches into 1 and demonstrates the
load-time weight-merge + attention-wrapper wiring that later phases will
reuse for RMSNorm+RoPE+cache+SDPA fusion.

The fused path is currently restricted to plain ``nn.Linear`` projections
(i.e. bf16 weights). Quantised ``QuantizedLinear`` projections — the
production cluster config — silently stay on the existing code path; a
future week extends the fusion to quantised weights via a custom Metal
kernel.
"""

from .fused_qkv import fused_qkv_is_installed, install_fused_qkv

__all__ = ["install_fused_qkv", "fused_qkv_is_installed"]
