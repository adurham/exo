"""Replication and the dispersion gate (design point 4).

Single unreplicated "fast" numbers on this cluster have repeatedly failed to
reproduce. A value is trusted only when at least ``minimum_replicates`` runs
agree: if max/min across replicates exceeds ``dispersion_threshold``, the
aggregate is FLAGGED as non-reproducing. It is never quietly reduced to the
best or the mean.
"""

from __future__ import annotations

import statistics
from typing import Final, Literal, final

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "DEFAULT_DISPERSION_THRESHOLD",
    "DEFAULT_MINIMUM_REPLICATES",
    "ReplicationVerdict",
    "ReplicatedValue",
    "aggregate_replicates",
]

DEFAULT_MINIMUM_REPLICATES: Final[int] = 3
DEFAULT_DISPERSION_THRESHOLD: Final[float] = 1.15

ReplicationVerdict = Literal[
    "reproducing", "non_reproducing", "insufficient_replicates"
]


@final
class ReplicatedValue(BaseModel):
    """An aggregate over replicates, carrying its own reproducibility verdict."""

    model_config = ConfigDict(frozen=True, strict=True)

    metric_name: str
    unit: str
    replicates: tuple[float, ...]
    minimum_replicates: int = Field(ge=1)
    dispersion_threshold: float = Field(gt=0.0)
    verdict: ReplicationVerdict

    @property
    def dispersion_ratio(self) -> float:
        smallest = min(self.replicates)
        largest = max(self.replicates)
        if smallest <= 0.0:
            return float("inf") if largest > 0.0 else 1.0
        return largest / smallest

    @property
    def median(self) -> float:
        return statistics.median(self.replicates)

    @property
    def mean(self) -> float:
        return statistics.fmean(self.replicates)

    @property
    def is_trustworthy(self) -> bool:
        return self.verdict == "reproducing"

    def reportable_value(self) -> float:
        """The only number callers may quote - and only if reproducing.

        Deliberately raises rather than returning a best/mean fallback, because
        quietly reporting the best of three disagreeing runs is the exact
        failure this gate exists to prevent.
        """
        if not self.is_trustworthy:
            raise ValueError(
                f"{self.metric_name}: verdict={self.verdict} "
                f"(n={len(self.replicates)}, dispersion="
                f"{self.dispersion_ratio:.3f} > {self.dispersion_threshold}); "
                "no value may be reported."
            )
        return self.median

    def describe(self) -> str:
        return (
            f"{self.metric_name}={self.median:.4g}{self.unit} "
            f"n={len(self.replicates)} dispersion={self.dispersion_ratio:.3f} "
            f"verdict={self.verdict}"
        )


def aggregate_replicates(
    *,
    metric_name: str,
    unit: str,
    replicates: tuple[float, ...],
    minimum_replicates: int = DEFAULT_MINIMUM_REPLICATES,
    dispersion_threshold: float = DEFAULT_DISPERSION_THRESHOLD,
) -> ReplicatedValue:
    """Aggregate replicates and assign a reproducibility verdict."""
    if not replicates:
        raise ValueError("at least one replicate is required")
    if len(replicates) < minimum_replicates:
        verdict: ReplicationVerdict = "insufficient_replicates"
    else:
        smallest = min(replicates)
        largest = max(replicates)
        ratio = float("inf") if smallest <= 0.0 else largest / smallest
        verdict = "reproducing" if ratio <= dispersion_threshold else "non_reproducing"
    return ReplicatedValue(
        metric_name=metric_name,
        unit=unit,
        replicates=replicates,
        minimum_replicates=minimum_replicates,
        dispersion_threshold=dispersion_threshold,
        verdict=verdict,
    )
