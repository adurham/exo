"""Chunk type definitions for streaming generation responses.

This module defines types for chunks of generated content that are
streamed back during inference (tokens, images, etc.).
"""

from enum import Enum

from exo.utils.pydantic_ext import TaggedModel

from .api import FinishReason
from .models import ModelId


class ChunkType(str, Enum):
    """Type of generation chunk.

    Values:
        Token: Text token chunk.
        Image: Image data chunk.
    """

    Token = "Token"
    Image = "Image"


class BaseChunk(TaggedModel):
    """Base class for generation chunks.

    Attributes:
        idx: Sequential index of this chunk in the generation.
        model: Model identifier that generated this chunk.
    """

    idx: int
    model: ModelId


class TokenChunk(BaseChunk):
    """Chunk containing a generated text token.

    Attributes:
        text: The token text.
        token_id: Numeric token ID.
        finish_reason: Reason why generation finished (if this is the final chunk).
    """

    text: str
    token_id: int
    finish_reason: FinishReason | None = None


class ImageChunk(BaseChunk):
    """Chunk containing generated image data.

    Attributes:
        data: Raw image bytes.
    """

    data: bytes


GenerationChunk = TokenChunk | ImageChunk
"""Discriminated union of all generation chunk types.

Used for type checking and pattern matching over chunks.
"""
