"""Tests for the decode-side cache-release knob (EXO_DSV41_DECODE_CLEAR_CACHE).

The knob must be OFF by default and must not disturb the round when unset: the
engine carries a multi-GB allocator cache through a deep decode turn, macOS
compresses it under the wired cap, and the turn pays decompression bursts — the
knob exists so that can be measured, not so it changes default behaviour.
"""

from __future__ import annotations

import pytest

import exo.worker.engines.mlx.dsv41.rounds as rounds_mod


def test_the_knob_is_off_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EXO_DSV41_DECODE_CLEAR_CACHE", raising=False)
    assert rounds_mod._decode_clear_cache_every() == 0


@pytest.mark.parametrize("raw, expected", [("0", 0), ("1", 1), ("8", 8), ("64", 64)])
def test_the_knob_parses_a_round_interval(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: int
) -> None:
    monkeypatch.setenv("EXO_DSV41_DECODE_CLEAR_CACHE", raw)
    assert rounds_mod._decode_clear_cache_every() == expected


@pytest.mark.parametrize("raw", ["", "abc", "-4", "8.0", " "])
def test_junk_values_fall_back_to_off(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    monkeypatch.setenv("EXO_DSV41_DECODE_CLEAR_CACHE", raw)
    assert rounds_mod._decode_clear_cache_every() == 0
