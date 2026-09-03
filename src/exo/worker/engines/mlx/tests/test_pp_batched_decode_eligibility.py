"""Tests for pp_batched_decode_eligibility.py -- the pure gate
deciding whether a request may use the batched-decode path."""

from __future__ import annotations

import inspect

import pytest

from exo.worker.engines.mlx.pp_batched_decode_eligibility import (
    EligibilityResult,
    is_eligible_for_batched_decode,
)

_ALL_ELIGIBLE_KWARGS: dict[str, bool] = {
    "has_images": False,
    "has_tools": False,
    "uses_speculative_decode": False,
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
    otherwise fail 4 other checks reports ONLY the flag-off reason,
    matching this module's documented "first applicable reason"
    contract (deterministic, not whichever check happens to run
    last)."""
    result = is_eligible_for_batched_decode(
        has_images=True,
        has_tools=True,
        uses_speculative_decode=True,
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


# --- 2026-08-06 cross-rank divergence fix (bug #6) regression tests ---


def test_signature_has_no_per_rank_mutable_state_input() -> None:
    """Structural guarantee against the bug #6 (cross-rank eligibility
    divergence) regression: the function signature must never again
    accept a per-rank-mutable-state input (like is_prefix_cache_hit)
    that a caller could compute differently on rank 0 vs rank 1 from
    each rank's independent KVPrefixCache trie. This test enforces
    the invariant by inspecting the actual parameter list -- if
    somebody adds a divergence-capable input back, this fails loudly
    at the signature level before any runtime path can hit the bug.
    """
    sig = inspect.signature(is_eligible_for_batched_decode)
    param_names = set(sig.parameters.keys())
    forbidden = {
        "is_prefix_cache_hit",
        "prefix_cache_hit_length",
        "local_hit_length",
        "kv_prefix_cache",
        "kv_cache",
        "cache",
    }
    assert forbidden.isdisjoint(param_names), (
        f"is_eligible_for_batched_decode must not take per-rank mutable "
        f"state as input (bug #6 cross-rank divergence regression). "
        f"Forbidden params found in signature: "
        f"{forbidden.intersection(param_names)}"
    )
    # The invariant, positively stated: every parameter is either
    # request-derived or static startup config.
    allowed = {
        "has_images",
        "has_tools",
        "uses_speculative_decode",
        "sharding_is_pipeline",
        "batched_decode_enabled",
    }
    assert param_names == allowed, (
        f"is_eligible_for_batched_decode signature drift: expected "
        f"exactly {allowed}, got {param_names}. If adding a new input, "
        f"confirm it is request-derived or static startup config -- "
        f"NEVER per-rank mutable state -- and update this assertion."
    )


def test_two_simulated_ranks_compute_identical_verdicts_regardless_of_per_rank_state() -> (
    None
):
    """Simulates the bug #6 scenario: two ranks with genuinely divergent
    per-rank state (e.g. different KVPrefixCache trie contents leading
    to different local hit lengths) MUST now compute identical batched-
    decode verdicts, because the function no longer accepts any input
    that could reflect per-rank state. This is proven by construction:
    both ranks call the function with the identical request/config
    inputs and get the identical result. The old signature took
    `is_prefix_cache_hit`, which each rank computed from its own trie
    and which could genuinely differ; the new signature has no such
    input, so divergence is structurally impossible.
    """
    # Common inputs: request-derived + static config (identical on both ranks)
    common = dict(_ALL_ELIGIBLE_KWARGS)
    # Both ranks see the same request/config and compute the same verdict.
    rank0_verdict = is_eligible_for_batched_decode(**common)
    rank1_verdict = is_eligible_for_batched_decode(**common)
    assert rank0_verdict == rank1_verdict
    assert rank0_verdict.eligible is True

    # Repeat for every ineligibility case: identical inputs -> identical
    # verdict on both ranks.
    for override in [
        {"batched_decode_enabled": False},
        {"sharding_is_pipeline": False},
        {"has_images": True},
        {"has_tools": True},
        {"uses_speculative_decode": True},
    ]:
        kwargs = dict(common)
        kwargs.update(override)
        r0 = is_eligible_for_batched_decode(**kwargs)
        r1 = is_eligible_for_batched_decode(**kwargs)
        assert r0 == r1, f"cross-rank divergence for override {override}"
        assert r0.eligible is False
