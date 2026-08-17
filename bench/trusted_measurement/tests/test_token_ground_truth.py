"""Regression test for the 1.42x server-estimate token-inflation bug class."""

from __future__ import annotations

import math
from typing import final

import pytest

from trusted_measurement.token_truth import (
    TokenCountMismatchError,
    cross_check_token_count,
)


@final
class WordTokenizer:
    """Deterministic stand-in for a real offline tokenizer: one token per word."""

    def encode(self, text: str, /) -> list[int]:
        return [hash(word) & 0xFFFF for word in text.split()]


def _prompt(words: int) -> str:
    return " ".join(f"word{index}" for index in range(words))


def test_agreeing_counts_pass() -> None:
    ground_truth = cross_check_token_count(
        tokenizer=WordTokenizer(),
        tokenizer_name="word",
        prompt=_prompt(1000),
        server_reported_token_count=1000,
    )
    assert ground_truth.offline_token_count == 1000
    assert ground_truth.agrees
    assert math.isclose(ground_truth.ratio, 1.0, rel_tol=1e-9)


def test_small_template_overhead_within_tolerance_passes() -> None:
    ground_truth = cross_check_token_count(
        tokenizer=WordTokenizer(),
        tokenizer_name="word",
        prompt=_prompt(1000),
        server_reported_token_count=1002,
        tolerance_tokens=2,
    )
    assert ground_truth.agrees


def test_fabricated_server_count_fails_the_run() -> None:
    """The actual historical bug: server reported 1.42x the real token count."""
    with pytest.raises(TokenCountMismatchError) as excinfo:
        _ = cross_check_token_count(
            tokenizer=WordTokenizer(),
            tokenizer_name="word",
            prompt=_prompt(1000),
            server_reported_token_count=1420,
        )
    assert "1.420" in str(excinfo.value)
    assert "Refusing to emit" in str(excinfo.value)


def test_undercount_also_fails() -> None:
    with pytest.raises(TokenCountMismatchError):
        _ = cross_check_token_count(
            tokenizer=WordTokenizer(),
            tokenizer_name="word",
            prompt=_prompt(1000),
            server_reported_token_count=700,
        )


def test_mismatch_is_never_downgraded_to_a_warning() -> None:
    """There is no non-raising path that returns a disagreeing ground truth."""
    with pytest.raises(TokenCountMismatchError):
        _ = cross_check_token_count(
            tokenizer=WordTokenizer(),
            tokenizer_name="word",
            prompt=_prompt(100),
            server_reported_token_count=103,
            tolerance_tokens=2,
        )
