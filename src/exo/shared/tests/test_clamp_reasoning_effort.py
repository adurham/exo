"""Tests for the reasoning-effort ladder clamp helper.

Hermes / OpenAI-style clients send effort levels ABOVE this wire's ceiling
(``max``, ``ultra``). The wire ceiling is ``xhigh`` -- the level the DeepSeek-V4
encoder maps to its strongest ``max`` prompt. Over-ceiling levels must clamp
DOWN to the ceiling instead of tripping the ``ReasoningEffort`` Literal with a
422. Unknown names pass through so genuine typos still 422 with the valid set.
"""

import pytest

from exo.shared.types.text_generation import (
    REASONING_EFFORT_CEILING,
    clamp_reasoning_effort,
)

# The six levels the ``ReasoningEffort`` wire Literal accepts.
_WIRE_LEVELS = ("none", "minimal", "low", "medium", "high", "xhigh")


@pytest.mark.parametrize("effort", _WIRE_LEVELS)
def test_wire_levels_pass_through_unchanged(effort: str) -> None:
    assert clamp_reasoning_effort(effort) == effort


@pytest.mark.parametrize("effort", ["max", "ultra"])
def test_over_ceiling_levels_clamp_to_ceiling(effort: str) -> None:
    assert clamp_reasoning_effort(effort) == REASONING_EFFORT_CEILING


def test_unknown_name_passes_through_unchanged() -> None:
    # A genuine typo is NOT in the ladder, so it must survive the clamp and be
    # rejected downstream by the pydantic Literal (which names the valid set).
    assert clamp_reasoning_effort("hgih") == "hgih"
