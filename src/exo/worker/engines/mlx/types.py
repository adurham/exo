"""Shared types for MLX-related functionality."""

from collections.abc import Sequence

from mlx import core as mx
from mlx import nn as nn
from mlx_lm.models.cache import (
    ArraysCache,
    BatchPoolingCache,
    BatchRotatingKVCache,
    CacheList,
    KVCache,
    PoolingCache,
    QuantizedKVCache,
    RotatingKVCache,
)

# This list contains one cache entry per transformer layer.
# DeepSeek-V4 layers no longer use a unified DeepseekV4Cache (removed in
# Blaizzy PR #1192's cache refactor). Each layer now uses a CacheList of
# (RotatingKVCache + 2× PoolingCache) constructed by the model's
# make_cache() method. Batched generation swaps these for the Batch* variants
# in mlx_lm.generate._make_cache.
KVCacheType = Sequence[
    KVCache
    | RotatingKVCache
    | BatchRotatingKVCache
    | QuantizedKVCache
    | ArraysCache
    | CacheList
    | PoolingCache
    | BatchPoolingCache
]


# Model is a wrapper function to fix the fact that mlx is not strongly typed in the same way that EXO is.
# For example - MLX has no guarantee of the interface that nn.Module will expose. But we need a guarantee that it has a __call__() function
class Model(nn.Module):
    layers: list[nn.Module]

    def __call__(
        self,
        x: mx.array,
        cache: KVCacheType | None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array: ...
