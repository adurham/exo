"""Phase 1 batched-PP eligibility gate: pure, fully-testable logic
deciding whether a real ``TextGenerationTaskParams`` request is
eligible for the batched-decode path
(``pp_batched_decode_runtime.py``'s ``BatchedDecodeSession``) or must
fall back to today's existing single-request serial path.

Per this session's `consult`-reviewed feature-interaction matrix
(``docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md`` Section 9,
Phase 1 step 9's addendum): rather than requiring
``ExoBatchGenerator``'s integrator to invent per-feature routing logic
case by case, gate eligibility explicitly and loudly here, once. Only
DECODE-ONLY (already-prefilled), no-speculation, no-vision, no-tools,
cold-cache requests are eligible for the batched path -- matching
Phase 1's own already-confirmed scope (decode-only, no speculative
decode). Any request outside this scope automatically and silently
falls back to the existing single-request path; it never gets a
degraded/wrong batched execution.

Deliberately has ZERO MLX/cluster/wire dependency -- pure logic over
plain Python types, fuzzable/unit-testable at Python-object speed,
matching this session's own established design principle (see
``pp_scheduler_protocol.py``'s module docstring point 1) for exactly
the same reason: the thing that decides "is this request allowed
into the new, less-battle-tested path" needs to be dead simple to
verify exhaustively, since a false positive here means a request with
an unverified feature interaction (Section 9's matrix) silently runs
through code that was never tested against it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EligibilityResult:
    """Outcome of ``is_eligible_for_batched_decode``. ``eligible=False``
    always carries a ``reason`` -- never a bare boolean -- so a caller
    (or a log line) can say WHY a request fell back, not just that it
    did. This matters operationally: a request that's ALWAYS
    ineligible for a reason nobody notices (e.g. a client always sets
    an unused ``tools=[]`` empty list that this gate's caller
    mishandles) would silently defeat the whole batched path forever
    without anyone knowing why throughput never improved."""

    eligible: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        if not self.eligible and self.reason is None:
            raise ValueError(
                "EligibilityResult(eligible=False, reason=None) -- an "
                "ineligible result MUST carry a reason; silent "
                "ineligibility defeats this gate's entire operational "
                "purpose (see module docstring)"
            )
        if self.eligible and self.reason is not None:
            raise ValueError(
                f"EligibilityResult(eligible=True, reason={self.reason!r}) "
                f"-- an ELIGIBLE result must not carry a reason (that's "
                f"only for explaining ineligibility); this looks like a "
                f"caller bug"
            )


def is_eligible_for_batched_decode(
    *,
    has_images: bool,
    has_tools: bool,
    uses_speculative_decode: bool,
    is_prefix_cache_hit: bool,
    sharding_is_pipeline: bool,
    batched_decode_enabled: bool,
) -> EligibilityResult:
    """Decide whether ONE request may use the batched-decode path.

    All inputs are plain booleans the caller has ALREADY computed from
    real request/runtime state (this function never reaches into
    ``TextGenerationTaskParams``/env vars itself -- keeping it a pure
    function of explicit inputs is what makes it exhaustively
    testable without constructing real task/model objects). Checked
    in a fixed, documented order so the FIRST applicable reason is
    always the one reported (a request can fail multiple checks at
    once; reporting only the first keeps log lines and tests
    deterministic rather than reporting "whichever check the
    implementation happened to run last").
    """
    if not batched_decode_enabled:
        return EligibilityResult(
            eligible=False, reason="batched_decode_disabled (opt-in flag is off)"
        )
    if not sharding_is_pipeline:
        return EligibilityResult(
            eligible=False,
            reason="not_pipeline_sharding (batched decode is PP-specific; "
            "TP already supports real concurrency without this path)",
        )
    if has_images:
        return EligibilityResult(
            eligible=False,
            reason="has_images (vision path not verified against the "
            "batched session -- see design doc's feature-interaction matrix)",
        )
    if has_tools:
        return EligibilityResult(
            eligible=False,
            reason="has_tools (tool-call parser state not verified against "
            "the batched session -- see design doc's feature-interaction "
            "matrix)",
        )
    if uses_speculative_decode:
        return EligibilityResult(
            eligible=False,
            reason="uses_speculative_decode (MTP/DSpark accept a VARIABLE "
            "token count per step; the batched scheduler protocol assumes "
            "exactly 1 token per active slot per step -- KNOWN "
            "INCOMPATIBLE, not a missing test. See design doc Section 6.2 "
            "item 7 / Phase 4 and the feature-interaction matrix.)",
        )
    if is_prefix_cache_hit:
        return EligibilityResult(
            eligible=False,
            reason="is_prefix_cache_hit (KVPrefixCache's serial "
            "snapshot/restore lifecycle vs BatchedCacheRouter's slot-based "
            "lifecycle has not been analyzed for compatibility -- see "
            "design doc's feature-interaction matrix)",
        )
    return EligibilityResult(eligible=True)
