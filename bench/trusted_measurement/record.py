"""The Record: a measurement that refuses to be trusted without its envelope.

A :class:`MeasurementRecord` is constructible only with every envelope field
present (pydantic enforces that structurally), and is only *valid* -
:meth:`MeasurementRecord.validate_envelope` returning no violations - when:

1. the content-correctness check ran in this process and passed;
2. the fingerprint is complete (both commits, registered env snapshot, known
   link topology);
3. every required runtime-mode marker was emitted in-process with a live proof;
4. replication met the minimum n and passed the dispersion gate;
5. prompt-size-dependent metrics carry cross-checked ground-truth token counts;
6. the depth-matrix cells covered are declared, and general-validity claims are
   only allowed when the whole matrix was covered.

Anything short of that yields an ``EnvelopeViolation`` and the number is not
reportable.
"""

from __future__ import annotations

from typing import final

from pydantic import BaseModel, ConfigDict, Field

from trusted_measurement.content_check import ContentCheckResult
from trusted_measurement.depth_matrix import DepthMatrixCell, coverage_gaps
from trusted_measurement.fingerprint import Fingerprint
from trusted_measurement.replication import ReplicatedValue
from trusted_measurement.runtime_mode import RuntimeModeMarker
from trusted_measurement.token_truth import TokenGroundTruth

__all__ = [
    "EnvelopeViolation",
    "MeasurementRecord",
    "UntrustedMeasurementError",
]


@final
class EnvelopeViolation(BaseModel):
    """One reason a record may not be trusted."""

    model_config = ConfigDict(frozen=True, strict=True)

    field: str
    reason: str

    def __str__(self) -> str:
        return f"{self.field}: {self.reason}"


class UntrustedMeasurementError(RuntimeError):
    """Raised when a caller tries to read a value off an invalid record."""


@final
class MeasurementRecord(BaseModel):
    """A measurement plus the complete validity envelope that earns its trust."""

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    probe_name: str = Field(min_length=1)
    """Which probe produced this (jaccl transport, prefill, decode, ...)."""

    value: ReplicatedValue
    """The measured quantity, already replicated and dispersion-gated."""

    content_check: ContentCheckResult
    """Point 1 - output-content proof from the same process/request."""

    fingerprint: Fingerprint
    """Point 2 - build, registered env, link topology."""

    runtime_mode_markers: tuple[RuntimeModeMarker, ...]
    """Point 3 - what actually executed."""

    required_runtime_modes: tuple[str, ...]
    """Modes the probe asserts must have executed for this number to mean anything."""

    depth_matrix_cells: tuple[DepthMatrixCell, ...] = Field(min_length=1)
    """Point 6 - the cells this result is scoped to."""

    claims_default_safe: bool = False
    """Whether this is being presented as a general result (needs full matrix)."""

    token_ground_truth: TokenGroundTruth | None = None
    """Point 5 - required when :attr:`prompt_size_dependent` is set."""

    prompt_size_dependent: bool
    """Whether the metric varies with prompt length (prefill-style measurements)."""

    canary_session_certified: bool
    """Point 8 - the session's liveness canary came back RED as designed."""

    notes: str = ""

    # ---------------------------------------------------------------- validity

    def validate_envelope(self) -> tuple[EnvelopeViolation, ...]:
        """Return every reason this record may not be trusted (empty == valid)."""
        violations: list[EnvelopeViolation] = []
        violations.extend(self._content_violations())
        violations.extend(self._fingerprint_violations())
        violations.extend(self._runtime_mode_violations())
        violations.extend(self._replication_violations())
        violations.extend(self._token_violations())
        violations.extend(self._depth_violations())
        if not self.canary_session_certified:
            violations.append(
                EnvelopeViolation(
                    field="canary_session_certified",
                    reason="harness liveness canary did not certify this session; "
                    "no result from it may be trusted",
                )
            )
        return tuple(violations)

    def _content_violations(self) -> list[EnvelopeViolation]:
        if not self.content_check.proof_is_valid():
            return [
                EnvelopeViolation(
                    field="content_check",
                    reason="content-check proof did not verify in this process "
                    "(fabricated, or carried over from an earlier run)",
                )
            ]
        if not self.content_check.passed:
            return [
                EnvelopeViolation(
                    field="content_check",
                    reason=f"content check {self.content_check.check_name!r} failed; "
                    "timing from an incorrect output is not a measurement",
                )
            ]
        return []

    def _fingerprint_violations(self) -> list[EnvelopeViolation]:
        violations: list[EnvelopeViolation] = []
        fingerprint = self.fingerprint
        if fingerprint.link_topology.source == "unavailable":
            violations.append(
                EnvelopeViolation(
                    field="fingerprint.link_topology",
                    reason="link topology unavailable; Thunderbolt link count is a "
                    "known measurement-shifting variable and must be recorded",
                )
            )
        if not fingerprint.registered_env:
            violations.append(
                EnvelopeViolation(
                    field="fingerprint.registered_env",
                    reason="registered env snapshot is empty",
                )
            )
        return violations

    def _runtime_mode_violations(self) -> list[EnvelopeViolation]:
        violations: list[EnvelopeViolation] = []
        observed: dict[str, RuntimeModeMarker] = {
            marker.mode: marker for marker in self.runtime_mode_markers
        }
        for marker in self.runtime_mode_markers:
            if not marker.proof_is_valid():
                violations.append(
                    EnvelopeViolation(
                        field="runtime_mode_markers",
                        reason=f"marker {marker.mode!r} carries an unverifiable proof",
                    )
                )
        for mode in self.required_runtime_modes:
            marker = observed.get(mode)
            if marker is None:
                violations.append(
                    EnvelopeViolation(
                        field="runtime_mode_markers",
                        reason=f"required runtime mode {mode!r} emitted no in-process "
                        "marker; config alone does not prove the path executed",
                    )
                )
        return violations

    def _replication_violations(self) -> list[EnvelopeViolation]:
        if self.value.is_trustworthy:
            return []
        return [
            EnvelopeViolation(
                field="value",
                reason=f"replication verdict {self.value.verdict!r} "
                f"(n={len(self.value.replicates)}, "
                f"dispersion={self.value.dispersion_ratio:.3f} vs threshold "
                f"{self.value.dispersion_threshold})",
            )
        ]

    def _token_violations(self) -> list[EnvelopeViolation]:
        if not self.prompt_size_dependent:
            return []
        if self.token_ground_truth is None:
            return [
                EnvelopeViolation(
                    field="token_ground_truth",
                    reason="prompt-size-dependent metric without offline "
                    "tokenization ground truth",
                )
            ]
        if not self.token_ground_truth.agrees:
            return [
                EnvelopeViolation(
                    field="token_ground_truth",
                    reason="offline token count disagrees with server-reported "
                    f"count (ratio {self.token_ground_truth.ratio:.3f})",
                )
            ]
        return []

    def _depth_violations(self) -> list[EnvelopeViolation]:
        gaps = coverage_gaps(self.depth_matrix_cells)
        if self.claims_default_safe and gaps:
            labels = ", ".join(f"{depth}/{thermal}" for depth, thermal in gaps)
            return [
                EnvelopeViolation(
                    field="claims_default_safe",
                    reason="result claims general/default-safe validity but the "
                    f"depth matrix has untested cells: {labels}",
                )
            ]
        return []

    # ----------------------------------------------------------------- reading

    @property
    def is_trusted(self) -> bool:
        return not self.validate_envelope()

    def trusted_value(self) -> float:
        """The number, readable only when the envelope is complete."""
        violations = self.validate_envelope()
        if violations:
            joined = "; ".join(str(violation) for violation in violations)
            raise UntrustedMeasurementError(
                f"{self.probe_name}: refusing to emit a trusted measurement - {joined}"
            )
        return self.value.reportable_value()

    def scope_label(self) -> str:
        """Human-facing scope this result may be quoted within."""
        cells = ", ".join(cell.label() for cell in self.depth_matrix_cells)
        suffix = "default-safe" if not coverage_gaps(self.depth_matrix_cells) else cells
        return f"{self.probe_name} [{suffix}]"
