"""KV cache management with prefix caching.

This module provides KVPrefixCache for caching KV states of prompt prefixes
to speed up inference when prompts share common prefixes.
"""

# type: ignore
# TODO: Fix this file, including types!
from copy import deepcopy
from typing import Callable

import mlx.core as mx
from mlx_lm import stream_generate
from mlx_lm.models.cache import _BaseCache, trim_prompt_cache
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.worker.engines.mlx import Model
from exo.worker.engines.mlx.constants import KEEP_KV_SIZE, KV_BITS, KV_GROUP_SIZE
from exo.worker.engines.mlx.utils_mlx import make_kv_cache


class KVPrefixCache:
    """Cache for KV states of prompt prefixes.

    Maintains a cache of prompt prefixes and their corresponding KV caches
    to speed up inference when new prompts share common prefixes with
    previously seen prompts.

    Attributes:
        prompts: List of cached prompt token arrays.
        caches: List of corresponding KV caches.
    """

    def __init__(self) -> None:
        """Initialize an empty prefix cache."""
        self.prompts: list[mx.array] = []
        self.caches: list[list[_BaseCache]] = []

    def add_kv_cache(
        self, tokenizer: TokenizerWrapper, prompt: str, cache: list[_BaseCache]
    ) -> None:
        """Add a prompt and its KV cache to the cache.

        Args:
            tokenizer: Tokenizer for encoding the prompt.
            prompt: Prompt text to cache.
            cache: KV cache corresponding to the prompt.
        """
        tokenized_prompt = self.encode_prompt(tokenizer, prompt)
        self.prompts.append(tokenized_prompt)
        self.caches.append(deepcopy(cache))

    def get_kv_cache(
        self,
        model: Model,
        tokenizer: TokenizerWrapper,
        sampler: Callable[[mx.array], mx.array],
        prompt: str,
    ) -> list[_BaseCache]:
        """Get or compute KV cache for a prompt, using prefix matching.

        Checks if the prompt shares a prefix with any cached prompt. If so,
        reuses the cached KV state and only computes the remaining tokens.
        Otherwise, computes the full KV cache.

        Args:
            model: Model to use for computation.
            tokenizer: Tokenizer for encoding the prompt.
            sampler: Sampling function (unused, required by stream_generate).
            prompt: Prompt text to get cache for.

        Returns:
            KV cache for the prompt.
        """
        tokenized_prompt = self.encode_prompt(tokenizer, prompt)
        max_length = len(tokenized_prompt)

        best_snapshot_index, best_snapshot_length = None, 0

        for i, cached_prompt in enumerate(self.prompts):
            length = _get_prefix_length(tokenized_prompt, cached_prompt)

            if length == max_length:
                return self.caches[i]

            if length > best_snapshot_length:
                best_snapshot_index, best_snapshot_length = i, length

        if best_snapshot_index is not None:
            prompt_cache = deepcopy(self.caches[best_snapshot_index])
            trim_prompt_cache(prompt_cache, max_length - best_snapshot_length)
            tokenized_prompt = tokenized_prompt[best_snapshot_index:]

        else:
            prompt_cache = make_kv_cache(
                model,
                # max_kv_size=MAX_KV_SIZE,
                # keep=KEEP_KV_SIZE
            )

        prefill(model, tokenizer, sampler, tokenized_prompt, prompt_cache)

        return prompt_cache

    def encode_prompt(self, tokenizer: TokenizerWrapper, prompt: str) -> mx.array:
        """Encode a prompt using the tokenizer.

        Args:
            tokenizer: Tokenizer to use.
            prompt: Prompt text.

        Returns:
            Encoded prompt as MLX array.
        """
        add_special_tokens = tokenizer.bos_token is None or not prompt.startswith(
            tokenizer.bos_token
        )
        tokenized_prompt = tokenizer.encode(
            prompt, add_special_tokens=add_special_tokens
        )
        return mx.array(tokenized_prompt)


def _get_prefix_length(prompt: mx.array, cached_prompt: mx.array) -> int:
    """Calculate the length of the common prefix between two prompts.

    Args:
        prompt: First prompt array.
        cached_prompt: Second prompt array.

    Returns:
        Length of the common prefix (up to KEEP_KV_SIZE).
    """
    n = min(int(prompt.shape[0]), int(cached_prompt.shape[0]), KEEP_KV_SIZE)
    if n == 0:
        return 0

    equal = (prompt[:n] == cached_prompt[:n]).astype(mx.int32)
    prefix_mask = mx.cumprod(equal)  # stays 1 until first mismatch, then 0 forever
    return int(mx.sum(prefix_mask).item())


def prefill(
    model: Model,
    tokenizer: TokenizerWrapper,
    sampler: Callable[[mx.array], mx.array],
    prompt: mx.array,
    cache: list[_BaseCache],
) -> None:
    """Prefill the KV cache for a prompt.

    Runs the model through the prompt tokens to populate the KV cache.

    Args:
        model: Model to use.
        tokenizer: Tokenizer (unused, required by stream_generate).
        sampler: Sampling function (unused, required by stream_generate).
        prompt: Prompt tokens to process.
        cache: KV cache to populate.
    """
    for _ in stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=0,
        sampler=sampler,
        prompt_cache=cache,
        prefill_step_size=2048,
        kv_group_size=KV_GROUP_SIZE,
        kv_bits=KV_BITS,
    ):
        pass
