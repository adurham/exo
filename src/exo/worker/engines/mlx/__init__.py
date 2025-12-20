"""MLX engine type stubs and wrappers.

This module provides type stubs for MLX components to ensure type safety
across the EXO codebase, since MLX's types are not as strict as EXO requires.

These are protocol-style definitions that match MLX's actual interfaces.
"""

from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache


class Model(nn.Module):
    """Type stub for MLX model.

    Provides a strongly-typed interface for MLX models, ensuring
    they have the expected __call__ signature.
    """

    layers: list[nn.Module]

    def __call__(
        self,
        x: mx.array,
        cache: list[KVCache] | None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array: ...


class Detokenizer:
    """Type stub for MLX detokenizer.

    Provides a strongly-typed interface for detokenizing model outputs.
    """

    def reset(self) -> None: ...
    def add_token(self, token: int) -> None: ...
    def finalize(self) -> None: ...

    @property
    def last_segment(self) -> str: ...


class TokenizerWrapper:
    """Type stub for MLX tokenizer wrapper.

    Provides a strongly-typed interface for tokenizing inputs and
    applying chat templates.
    """

    bos_token: str | None
    eos_token_ids: list[int]
    detokenizer: Detokenizer

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]: ...

    def apply_chat_template(
        self,
        messages_dicts: list[dict[str, Any]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str: ...
