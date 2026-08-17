"""Harness liveness canary (design point 8) - the most important piece.

The real lesson from the cancellation-test failure is not "add a content
check". It is that **the harness itself reported success while broken**. So
before any real measurement in a session is certified, the harness must prove
it can still detect a rigged measurement.

The canary builds a set of records that are each deliberately broken in one
specific way, plus one positive control that is fully valid, and runs them
through the same envelope validator the real probes use:

* every deliberately-broken scenario MUST come back RED (violations found);
* the positive control MUST come back GREEN (no violations).

If any scenario comes back the wrong colour, the session is NOT certified and
:meth:`CanaryReport.require_certified` halts. Nothing else measured in that
session may be presented as trustworthy.

The validator is injected (:data:`RecordJudge`), which is what makes the canary
meta-testable: feed it a rubber-stamp judge and the canary must itself go red.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import final

from pydantic import BaseModel, ConfigDict

from trusted_measurement.content_check import (
    CONTENT_CHECK_DOMAIN,
    ContentCheckResult,
    build_needle_haystack,
    run_needle_check,
)
from trusted_measurement.depth_matrix import ALL_DEPTH_CELLS, DepthMatrixCell
from trusted_measurement.fingerprint import Fingerprint, LinkTopology
from trusted_measurement.proof import ProofToken
from trusted_measurement.record import EnvelopeViolation, MeasurementRecord
from trusted_measurement.replication import aggregate_replicates
from trusted_measurement.runtime_mode import RuntimeModeRecorder
from trusted_measurement.token_truth import TokenGroundTruth

__all__ = [
    "CanaryReport",
    "CanaryScenarioResult",
    "HarnessNotCertifiedError",
    "RecordJudge",
    "default_judge",
    "run_liveness_canary",
]

RecordJudge = Callable[[MeasurementRecord], tuple[EnvelopeViolation, ...]]
"""The thing under test: maps a record to the reasons it may not be trusted."""


def default_judge(record: MeasurementRecord) -> tuple[EnvelopeViolation, ...]:
    """The production validator - what the real probes rely on."""
    return record.validate_envelope()


class HarnessNotCertifiedError(RuntimeError):
    """Raised when the harness failed to detect its own rigged scenarios."""


_SYNTHETIC_ENV: Mapping[str, str | None] = {
    "MLX_JACCL_SHARDING_MODE": "tp",
    "MLX_JACCL_ACK_RETRANSMIT_US": "2000",
}


def _synthetic_fingerprint() -> Fingerprint:
    """A complete, offline fingerprint. The canary never shells out to git."""
    return Fingerprint(
        exo_commit="0" * 40,
        mlx_commit="1" * 40,
        exo_dirty=False,
        mlx_dirty=False,
        registered_env=dict(_SYNTHETIC_ENV),
        link_topology=LinkTopology(
            thunderbolt_link_count=6,
            link_descriptors=("canary-link-0",),
            source="canary-synthetic",
        ),
        hostname="canary-host",
    )


def _full_matrix_cells() -> tuple[DepthMatrixCell, ...]:
    return tuple(
        DepthMatrixCell(
            context_depth=depth,
            thermal_state=thermal,
            prompt_tokens=1024,
        )
        for depth, thermal in ALL_DEPTH_CELLS
    )


def _passing_content_check() -> ContentCheckResult:
    haystack = build_needle_haystack(filler_sentences=4, seed=17)
    return run_needle_check(haystack, f"The code is {haystack.needle}.")


def _valid_record(
    probe_name: str = "canary_positive_control", **overrides: object
) -> MeasurementRecord:
    """Build a canary record by CONSTRUCTION so field constraints always run.

    ``model_copy(update=...)`` would skip pydantic validation, which is exactly
    the kind of silent bypass this package exists to prevent.
    """
    recorder = RuntimeModeRecorder()
    recorder.emit("tp_allreduce", detail="canary", count=61)
    fields: dict[str, object] = dict(
        probe_name=probe_name,
        value=aggregate_replicates(
            metric_name="canary_throughput",
            unit="tok/s",
            replicates=(100.0, 101.0, 99.5),
        ),
        content_check=_passing_content_check(),
        fingerprint=_synthetic_fingerprint(),
        runtime_mode_markers=recorder.markers(),
        required_runtime_modes=("tp_allreduce",),
        depth_matrix_cells=_full_matrix_cells(),
        claims_default_safe=True,
        token_ground_truth=TokenGroundTruth(
            tokenizer_name="canary",
            offline_token_count=1024,
            server_reported_token_count=1024,
            tolerance_tokens=2,
        ),
        prompt_size_dependent=True,
        canary_session_certified=True,
    )
    fields.update(overrides)
    return MeasurementRecord(**fields)  # pyright: ignore[reportArgumentType]


# --------------------------------------------------------------- rigged records


def _rigged_forged_content_proof() -> MeasurementRecord:
    """Content check claims it passed, but the proof was hand-written."""
    forged = ContentCheckResult(
        check_name="needle_in_haystack",
        passed=True,
        expected="123456",
        observed_excerpt="123456",
        proof=ProofToken(
            domain=CONTENT_CHECK_DOMAIN, digest="0" * 64, session_id="forged"
        ),
    )
    return _valid_record("canary_forged_content_proof", content_check=forged)


def _rigged_failed_content_check() -> MeasurementRecord:
    """Real check, honestly reporting that the output was wrong."""
    haystack = build_needle_haystack(filler_sentences=4, seed=23)
    return _valid_record(
        "canary_failed_content_check",
        content_check=run_needle_check(haystack, "I do not know."),
    )


def _rigged_missing_runtime_marker() -> MeasurementRecord:
    """Config implies TP, but the path emitted no marker."""
    return _valid_record("canary_missing_runtime_marker", runtime_mode_markers=())


def _rigged_non_reproducing() -> MeasurementRecord:
    """Three replicates that flatly disagree."""
    return _valid_record(
        "canary_non_reproducing",
        value=aggregate_replicates(
            metric_name="canary_throughput",
            unit="tok/s",
            replicates=(100.0, 42.0, 180.0),
        ),
    )


def _rigged_single_replicate() -> MeasurementRecord:
    """One glorious unreplicated run."""
    return _valid_record(
        "canary_single_replicate",
        value=aggregate_replicates(
            metric_name="canary_throughput",
            unit="tok/s",
            replicates=(137.0,),
        ),
    )


def _rigged_token_inflation() -> MeasurementRecord:
    """The 1.42x server-estimate bug, reproduced on purpose."""
    return _valid_record(
        "canary_token_inflation",
        token_ground_truth=TokenGroundTruth(
            tokenizer_name="canary",
            offline_token_count=1024,
            server_reported_token_count=1454,
            tolerance_tokens=2,
        ),
    )


def _rigged_shallow_only_generalisation() -> MeasurementRecord:
    """Validated only at shallow/warm, presented as default-safe."""
    return _valid_record(
        "canary_shallow_only_generalisation",
        depth_matrix_cells=(
            DepthMatrixCell(
                context_depth="shallow",
                thermal_state="warm",
                prompt_tokens=256,
            ),
        ),
        claims_default_safe=True,
    )


def _rigged_unknown_link_topology() -> MeasurementRecord:
    """Thunderbolt link count unknown - the 3-to-6 blind spot."""
    fingerprint = Fingerprint(
        exo_commit="0" * 40,
        mlx_commit="1" * 40,
        exo_dirty=False,
        mlx_dirty=False,
        registered_env=dict(_SYNTHETIC_ENV),
        link_topology=LinkTopology(
            thunderbolt_link_count=0, link_descriptors=(), source="unavailable"
        ),
        hostname="canary-host",
    )
    return _valid_record("canary_unknown_link_topology", fingerprint=fingerprint)


def _rigged_uncertified_session() -> MeasurementRecord:
    """A record from a session whose canary never certified it."""
    return _valid_record("canary_uncertified_session", canary_session_certified=False)


_RIGGED_SCENARIOS: tuple[tuple[str, Callable[[], MeasurementRecord]], ...] = (
    ("forged_content_proof", _rigged_forged_content_proof),
    ("failed_content_check", _rigged_failed_content_check),
    ("missing_runtime_marker", _rigged_missing_runtime_marker),
    ("non_reproducing_replicates", _rigged_non_reproducing),
    ("single_replicate", _rigged_single_replicate),
    ("token_count_inflation", _rigged_token_inflation),
    ("shallow_only_generalisation", _rigged_shallow_only_generalisation),
    ("unknown_link_topology", _rigged_unknown_link_topology),
    ("uncertified_session", _rigged_uncertified_session),
)


# ------------------------------------------------------------------- reporting


@final
class CanaryScenarioResult(BaseModel):
    """Outcome of one canary scenario."""

    model_config = ConfigDict(frozen=True, strict=True)

    name: str
    expected_red: bool
    was_red: bool
    violations: tuple[str, ...]

    @property
    def behaved_correctly(self) -> bool:
        return self.was_red == self.expected_red

    def describe(self) -> str:
        colour = "RED" if self.was_red else "GREEN"
        status = "ok" if self.behaved_correctly else "HARNESS BROKEN"
        return f"  [{status}] {self.name}: {colour}"


@final
class CanaryReport(BaseModel):
    """Whether this session's harness may certify any result at all."""

    model_config = ConfigDict(frozen=True, strict=True)

    results: tuple[CanaryScenarioResult, ...]

    @property
    def certified(self) -> bool:
        return bool(self.results) and all(
            result.behaved_correctly for result in self.results
        )

    @property
    def failures(self) -> tuple[CanaryScenarioResult, ...]:
        return tuple(result for result in self.results if not result.behaved_correctly)

    def require_certified(self) -> None:
        """Halt the session unless the harness proved it can detect rigged runs."""
        if not self.certified:
            broken = ", ".join(result.name for result in self.failures)
            raise HarnessNotCertifiedError(
                "harness liveness canary failed on: "
                f"{broken or '<no scenarios ran>'}. The harness cannot be trusted "
                "to detect a bad measurement, so no result from this session may "
                "be certified."
            )

    def describe(self) -> str:
        header = (
            "harness liveness canary: CERTIFIED"
            if self.certified
            else "harness liveness canary: NOT CERTIFIED"
        )
        lines = [header, *(result.describe() for result in self.results)]
        return "\n".join(lines)


def run_liveness_canary(judge: RecordJudge = default_judge) -> CanaryReport:
    """Prove the harness still rejects rigged measurements and accepts good ones."""
    results: list[CanaryScenarioResult] = []
    for name, build in _RIGGED_SCENARIOS:
        violations = judge(build())
        results.append(
            CanaryScenarioResult(
                name=name,
                expected_red=True,
                was_red=bool(violations),
                violations=tuple(str(violation) for violation in violations),
            )
        )
    control_violations = judge(_valid_record())
    results.append(
        CanaryScenarioResult(
            name="positive_control_valid_record",
            expected_red=False,
            was_red=bool(control_violations),
            violations=tuple(str(violation) for violation in control_violations),
        )
    )
    return CanaryReport(results=tuple(results))
