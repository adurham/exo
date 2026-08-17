"""Paired A/B runner discipline (design point 7).

Two rules, enforced structurally rather than by convention:

* **One variable, one deployment.** The two arms must differ in exactly one
  registered env var and share an identical build fingerprint. Comparing across
  separate rebuilds/relaunches is rejected, because that is how a build
  difference gets attributed to a config change.
* **Interleaved arms.** Long runs on this hardware drift thermally, so the
  runner emits an A,B,A,B... schedule. A blocked A,A,A,B,B,B schedule is
  rejected outright.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, final

from pydantic import BaseModel, ConfigDict, Field

from trusted_measurement.fingerprint import Fingerprint
from trusted_measurement.record import MeasurementRecord

__all__ = [
    "ArmLabel",
    "PairedComparison",
    "PairedComparisonError",
    "interleaved_schedule",
    "compare_arms",
]

ArmLabel = Literal["A", "B"]


class PairedComparisonError(RuntimeError):
    """Raised when an A/B comparison violates toggle discipline."""


def interleaved_schedule(replicates_per_arm: int) -> tuple[ArmLabel, ...]:
    """A,B,A,B,... schedule for ``replicates_per_arm`` replicates of each arm."""
    if replicates_per_arm < 1:
        raise ValueError("replicates_per_arm must be >= 1")
    schedule: list[ArmLabel] = []
    for _ in range(replicates_per_arm):
        schedule.append("A")
        schedule.append("B")
    return tuple(schedule)


def _is_interleaved(schedule: Sequence[ArmLabel]) -> bool:
    if len(schedule) < 2:
        return False
    return all(
        schedule[index] != schedule[index + 1] for index in range(len(schedule) - 1)
    )


def _build_identity(fingerprint: Fingerprint) -> tuple[str, str, bool, bool, str]:
    return (
        fingerprint.exo_commit,
        fingerprint.mlx_commit,
        fingerprint.exo_dirty,
        fingerprint.mlx_dirty,
        fingerprint.link_topology.summary(),
    )


@final
class PairedComparison(BaseModel):
    """A verified single-variable, interleaved A/B result."""

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    toggled_variable: str
    arm_a_value: str | None
    arm_b_value: str | None
    arm_a: MeasurementRecord
    arm_b: MeasurementRecord
    execution_schedule: tuple[ArmLabel, ...] = Field(min_length=2)

    @property
    def ratio_b_over_a(self) -> float:
        return self.arm_b.trusted_value() / self.arm_a.trusted_value()

    def describe(self) -> str:
        return (
            f"{self.toggled_variable}: {self.arm_a_value!r} -> {self.arm_b_value!r} "
            f"gives {self.ratio_b_over_a:.3f}x "
            f"({self.arm_a.value.describe()} vs {self.arm_b.value.describe()})"
        )


def compare_arms(
    *,
    arm_a: MeasurementRecord,
    arm_b: MeasurementRecord,
    execution_schedule: Sequence[ArmLabel],
) -> PairedComparison:
    """Validate toggle discipline and build the comparison, or raise."""
    if not _is_interleaved(execution_schedule):
        raise PairedComparisonError(
            "A/B arms must be interleaved (A,B,A,B,...) to control for thermal "
            f"drift; got {''.join(execution_schedule)}"
        )
    if _build_identity(arm_a.fingerprint) != _build_identity(arm_b.fingerprint):
        raise PairedComparisonError(
            "A/B arms have different build/host fingerprints; one variable must be "
            "flipped on the SAME live deployment, never across rebuilds"
        )
    differences = arm_a.fingerprint.env_diff(arm_b.fingerprint)
    if len(differences) != 1:
        raise PairedComparisonError(
            "A/B arms must differ in exactly one registered env var; "
            f"differing: {sorted(differences) or ['<none>']}"
        )
    variable, (value_a, value_b) = next(iter(differences.items()))
    return PairedComparison(
        toggled_variable=variable,
        arm_a_value=value_a,
        arm_b_value=value_b,
        arm_a=arm_a,
        arm_b=arm_b,
        execution_schedule=tuple(execution_schedule),
    )
