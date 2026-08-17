"""Ground-truth token counting (design point 5).

A benchmark script once inflated prompt token counts 1.42x by trusting the
server's reported ``prompt_tokens`` instead of tokenizing the prompt itself,
silently corrupting a whole investigation arc. Any prompt-size-dependent
measurement must therefore tokenize the *actual* prompt string offline and
cross-check the server's number against it. A mismatch beyond a small epsilon
FAILS the run - it is never downgraded to a warning.
"""

from __future__ import annotations

from typing import Protocol, final, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "TokenCountMismatchError",
    "TokenGroundTruth",
    "Tokenizer",
    "cross_check_token_count",
]


@runtime_checkable
class Tokenizer(Protocol):
    """Minimal offline tokenizer interface (HF fast tokenizers satisfy this)."""

    def encode(self, text: str, /) -> list[int]: ...


class TokenCountMismatchError(RuntimeError):
    """Raised when a server-reported prompt length contradicts real tokenization."""


@final
class TokenGroundTruth(BaseModel):
    """Real offline token count cross-checked against the server's claim."""

    model_config = ConfigDict(frozen=True, strict=True)

    tokenizer_name: str
    offline_token_count: int = Field(ge=0)
    server_reported_token_count: int = Field(ge=0)
    tolerance_tokens: int = Field(ge=0)

    @property
    def absolute_difference(self) -> int:
        return abs(self.offline_token_count - self.server_reported_token_count)

    @property
    def ratio(self) -> float:
        if self.offline_token_count == 0:
            return 1.0 if self.server_reported_token_count == 0 else float("inf")
        return self.server_reported_token_count / self.offline_token_count

    @property
    def agrees(self) -> bool:
        return self.absolute_difference <= self.tolerance_tokens


def cross_check_token_count(
    *,
    tokenizer: Tokenizer,
    tokenizer_name: str,
    prompt: str,
    server_reported_token_count: int,
    tolerance_tokens: int = 2,
) -> TokenGroundTruth:
    """Tokenize ``prompt`` for real and fail loudly if the server disagrees.

    ``tolerance_tokens`` covers legitimate chat-template/BOS bookkeeping only.
    Anything larger is the 1.42x bug class and raises.
    """
    offline_count = len(tokenizer.encode(prompt))
    ground_truth = TokenGroundTruth(
        tokenizer_name=tokenizer_name,
        offline_token_count=offline_count,
        server_reported_token_count=server_reported_token_count,
        tolerance_tokens=tolerance_tokens,
    )
    if not ground_truth.agrees:
        raise TokenCountMismatchError(
            f"prompt token count mismatch: offline={offline_count} "
            f"server={server_reported_token_count} "
            f"(ratio {ground_truth.ratio:.3f}, tolerance {tolerance_tokens}). "
            "Refusing to emit a prompt-size-dependent measurement."
        )
    return ground_truth
