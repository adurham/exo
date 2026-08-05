"""Tests for pp_batched_decode_eligibility.py -- the pure gate
deciding whether a request may use the batched-decode path."""

from __future__ import annotations

import pytest

from exo.worker.engines.mlx.pp_batched_decode_eligibility import (
    EligibilityResult,
    is_eligible_for_batched_decode,
)

_ALL_ELIGIBLE_KWARGS: dict[str, bool] = {
    "has_images": False,
    "has_tools": False,
    "uses_speculative_decode": False,
    "is_prefix_cache_hit": False,
    "sharding_is_pipeline": True,
    "batched_decode_enabled": True,
}


def test_fully_eligible_request_returns_eligible_true_no_reason() -> None:
    result = is_eligible_for_batched_decode(**_ALL_ELIGIBLE_KWARGS)
    assert result.eligible is True
    assert result.reason is None


@pytest.mark.parametrize(
    "override,expected_reason_substring",
    [
        ({"batched_decode_enabled": False}, "batched_decode_disabled"),
        ({"sharding_is_pipeline": False}, "not_pipeline_sharding"),
        ({"has_images": True}, "has_images"),
        ({"has_tools": True}, "has_tools"),
        ({"uses_speculative_decode": True}, "uses_speculative_decode"),
        ({"is_prefix_cache_hit": True}, "is_prefix_cache_hit"),
    ],
)
def test_each_single_disqualifying_condition_reports_its_own_reason(
    override: dict[str, bool], expected_reason_substring: str
) -> None:
    kwargs = dict(_ALL_ELIGIBLE_KWARGS)
    kwargs.update(override)
    result = is_eligible_for_batched_decode(**kwargs)
    assert result.eligible is False
    assert result.reason is not None
    assert expected_reason_substring in result.reason


def test_flag_off_takes_priority_over_every_other_condition() -> None:
    """The opt-in flag is checked FIRST -- even a request that would
    otherwise fail 5 other checks reports ONLY the flag-off reason,
    matching this module's documented "first applicable reason"
    contract (deterministic, not whichever check happens to run
    last)."""
    result = is_eligible_for_batched_decode(
        has_images=True,
        has_tools=True,
        uses_speculative_decode=True,
        is_prefix_cache_hit=True,
        sharding_is_pipeline=False,
        batched_decode_enabled=False,
    )
    assert result.eligible is False
    assert result.reason is not None
    assert "batched_decode_disabled" in result.reason


def test_speculative_decode_reason_explains_the_real_incompatibility() -> None:
    """The speculative-decode rejection reason must explain WHY (not
    just restate the flag name) -- this is the single most likely
    thing a future integrator hits first (MTP is the default
    speculative path when EXO_SPECULATIVE=1), so the reason string is
    load-bearing documentation, not just a log tag."""
    kwargs = dict(_ALL_ELIGIBLE_KWARGS)
    kwargs["uses_speculative_decode"] = True
    result = is_eligible_for_batched_decode(**kwargs)
    assert result.reason is not None
    assert "VARIABLE" in result.reason or "variable" in result.reason.lower()


def test_eligibility_result_rejects_ineligible_with_no_reason() -> None:
    """Constructing EligibilityResult(eligible=False, reason=None)
    directly (bypassing the gate function) must fail loudly -- the
    module docstring's entire operational argument for this
    dataclass is that ineligibility is NEVER silent."""
    with pytest.raises(ValueError, match="MUST carry a reason"):
        EligibilityResult(eligible=False, reason=None)


def test_eligibility_result_rejects_eligible_with_a_reason() -> None:
    with pytest.raises(ValueError, match="must not carry a reason"):
        EligibilityResult(eligible=True, reason="some reason")


def test_eligibility_result_eligible_true_no_reason_constructs_cleanly() -> None:
    result = EligibilityResult(eligible=True)
    assert result.eligible is True
    assert result.reason is None


def test_eligibility_result_eligible_false_with_reason_constructs_cleanly() -> None:
    result = EligibilityResult(eligible=False, reason="some reason")
    assert result.eligible is False
    assert result.reason == "some reason"
