"""MLX engine constants and configuration.

This module defines constants used by the MLX engine for model execution,
including KV cache settings, quantization, and token limits.
"""

KV_GROUP_SIZE: int | None = 32
"""Group size for KV cache quantization."""
KV_BITS: int | None = None
"""Bits per element for KV cache quantization."""
ATTENTION_KV_BITS: int | None = 4
"""Bits per element for attention KV cache."""
MAX_TOKENS: int = 8192
"""Maximum number of tokens per request."""
MAX_KV_SIZE: int | None = 3200
"""Maximum KV cache size."""
KEEP_KV_SIZE: int | None = 1600
"""KV cache size to keep for prefix matching."""
QUANTIZE_MODEL_MODE: str | None = "affine"
"""Quantization mode for model weights."""
CACHE_GROUP_SIZE: int = 64
"""Group size for cache quantization."""
KV_CACHE_BITS: int | None = 8
"""Bits per element for KV cache."""
TEMPERATURE: float = 1.0
"""Default sampling temperature."""
TRUST_REMOTE_CODE: bool = True
"""Whether to trust remote code when loading models (required for some models like Kimi)."""
