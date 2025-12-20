"""Response types from runner processes.

This module defines types for responses sent by runner processes back
to the worker supervisor during inference.
"""

from exo.shared.types.api import FinishReason
from exo.utils.pydantic_ext import TaggedModel


class BaseRunnerResponse(TaggedModel):
    """Base class for runner responses.

    All responses from runners inherit from this class for discriminated
    union support.
    """

    pass


class TokenizedResponse(BaseRunnerResponse):
    """Response indicating prompt tokenization completed.

    Attributes:
        prompt_tokens: Number of tokens in the prompt.
    """

    prompt_tokens: int


class GenerationResponse(BaseRunnerResponse):
    """Response containing a generated token.

    Attributes:
        text: Generated token text.
        token: Token ID.
        finish_reason: Reason why generation finished (if this is the final token).
    """

    text: str
    token: int
    finish_reason: FinishReason | None = None


class FinishedResponse(BaseRunnerResponse):
    """Response indicating inference has completed.

    Sent after the final token is generated.
    """

    pass
