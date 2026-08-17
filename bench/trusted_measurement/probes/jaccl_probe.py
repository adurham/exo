"""jaccl / soft-RC transport probe scaffold (Phase 2).

WHAT THIS MEASURES
------------------
The transport layer under the model: round-trip latency distribution,
retransmit counts, and receive-pool occupancy. These are the quantities that
move when the ``MLX_JACCL_*`` reliability knobs are tuned, and they are the
ones that have historically been "tuned at shallow context, shipped as a
global default, broke generation at depth".

REAL TODAY (typed, tested, no cluster required)
-----------------------------------------------
* :class:`JacclTransportSample` / :class:`RoundTripHistogram` - the data model
  a jaccl measurement must carry, including percentile arithmetic.
* :func:`summarise_samples` - folds N per-replicate samples into one metric
  series ready for :func:`~trusted_measurement.replication.aggregate_replicates`.
* :func:`jaccl_arm_environments` - A/B toggle plumbing: derives the two arm
  environments from one base environment by flipping exactly one registered
  variable, so the resulting fingerprints differ in exactly one var and the
  Phase 1 :func:`~trusted_measurement.ab_runner.compare_arms` gate passes.
* :func:`build_jaccl_record` - depth-matrix loop structure and full
  :class:`~trusted_measurement.record.MeasurementRecord` construction, calling
  :func:`~trusted_measurement.fingerprint.capture_fingerprint` through an
  injected command runner.
* :func:`build_argument_parser` / :func:`main` - argument parsing.

INTERFACE STUB (needs real-hardware / other-session coordination)
-----------------------------------------------------------------
* :class:`JacclLogSource` - yields raw jaccl log text for one measurement
  window. The real implementation tails the runner log on each node. NOT
  implemented here: the log line format is owned by the transport code that a
  different session is actively editing.
* :func:`parse_jaccl_transport_sample` - the parser that turns that raw text
  into a :class:`JacclTransportSample`. It currently accepts only the
  documented, self-describing ``key=value`` contract below; extending it to the
  real emitted format is a follow-up task. The contract is deliberately strict:
  an unparseable window raises rather than silently yielding zeros, because a
  zero retransmit count that actually means "we could not read the log" is the
  exact failure this harness exists to prevent.

TODO(phase-3, needs coordination): the runtime-mode markers this probe requires
must be emitted from inside the jaccl transport path itself. Emitting them
means editing shared transport code, which is owned by another session; this
module therefore only *consumes* markers handed to it.
"""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Mapping, Sequence
from typing import Protocol, final, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from trusted_measurement.content_check import ContentCheckResult
from trusted_measurement.depth_matrix import DepthMatrixCell
from trusted_measurement.fingerprint import (
    CommandRunner,
    Fingerprint,
    capture_fingerprint,
    registered_env_names,
)
from trusted_measurement.record import MeasurementRecord
from trusted_measurement.replication import (
    DEFAULT_DISPERSION_THRESHOLD,
    DEFAULT_MINIMUM_REPLICATES,
    aggregate_replicates,
)
from trusted_measurement.runtime_mode import RuntimeModeMarker

__all__ = [
    "JACCL_REQUIRED_RUNTIME_MODES",
    "JacclLogSource",
    "JacclTransportSample",
    "RoundTripHistogram",
    "StaticJacclLogSource",
    "UnparseableJacclWindowError",
    "build_argument_parser",
    "build_jaccl_record",
    "jaccl_arm_environments",
    "main",
    "parse_jaccl_transport_sample",
    "summarise_samples",
]

JACCL_REQUIRED_RUNTIME_MODES: tuple[str, ...] = ("jaccl_reliable_data",)
"""Modes a jaccl transport number is meaningless without.

The marker for this mode must be emitted from inside the transport path. See
the module-level TODO: that call site is a separate, coordinated task.
"""


class UnparseableJacclWindowError(RuntimeError):
    """Raised when a measurement window's log text cannot be parsed.

    Never downgraded to a warning and never defaulted to zeros: an unreadable
    window is a missing measurement, not a clean one.
    """


@final
class RoundTripHistogram(BaseModel):
    """Round-trip latency distribution for one measurement window."""

    model_config = ConfigDict(frozen=True, strict=True)

    samples_microseconds: tuple[float, ...] = Field(min_length=1)

    @property
    def count(self) -> int:
        return len(self.samples_microseconds)

    @property
    def median_microseconds(self) -> float:
        return statistics.median(self.samples_microseconds)

    def percentile_microseconds(self, percentile: float) -> float:
        """Nearest-rank percentile; ``percentile`` in [0, 100]."""
        if not 0.0 <= percentile <= 100.0:
            raise ValueError("percentile must be within [0, 100]")
        ordered = sorted(self.samples_microseconds)
        if percentile == 0.0:
            return ordered[0]
        rank = max(
            1, min(len(ordered), int(-(-percentile / 100.0 * len(ordered) // 1)))
        )
        return ordered[rank - 1]

    @property
    def tail_ratio(self) -> float:
        """p99 / median - the shape metric that distinguishes jitter from load."""
        median = self.median_microseconds
        if median <= 0.0:
            return float("inf")
        return self.percentile_microseconds(99.0) / median


@final
class JacclTransportSample(BaseModel):
    """One measurement window of soft-RC/transport-layer observations."""

    model_config = ConfigDict(frozen=True, strict=True)

    window_label: str = Field(min_length=1)
    round_trip: RoundTripHistogram
    retransmits: int = Field(ge=0)
    messages_sent: int = Field(ge=0)
    recv_pool_capacity: int = Field(gt=0)
    recv_pool_peak_occupancy: int = Field(ge=0)
    stalls_detected: int = Field(ge=0)

    @property
    def retransmit_rate(self) -> float:
        """Retransmits per message sent; 0.0 when nothing was sent."""
        if self.messages_sent == 0:
            return 0.0
        return self.retransmits / self.messages_sent

    @property
    def recv_pool_occupancy_ratio(self) -> float:
        return self.recv_pool_peak_occupancy / self.recv_pool_capacity

    def metric(self, metric_name: str) -> float:
        """Read one named scalar off this sample."""
        match metric_name:
            case "median_rtt_us":
                return self.round_trip.median_microseconds
            case "p99_rtt_us":
                return self.round_trip.percentile_microseconds(99.0)
            case "rtt_tail_ratio":
                return self.round_trip.tail_ratio
            case "retransmit_rate":
                return self.retransmit_rate
            case "recv_pool_occupancy_ratio":
                return self.recv_pool_occupancy_ratio
            case "stalls_detected":
                return float(self.stalls_detected)
            case _:
                raise KeyError(f"unknown jaccl metric {metric_name!r}")


@runtime_checkable
class JacclLogSource(Protocol):
    """INTERFACE STUB - yields raw jaccl log text for one measurement window.

    The real implementation tails each node's runner log over the measurement
    window and returns the text emitted between window start and window end.
    It must be side-effect free with respect to the measured run.
    """

    def read_window(self, window_label: str) -> str: ...


@final
class StaticJacclLogSource:
    """Test/offline log source backed by a fixed mapping of window -> text."""

    def __init__(self, windows: Mapping[str, str]) -> None:
        self._windows: Mapping[str, str] = dict(windows)

    def read_window(self, window_label: str) -> str:
        try:
            return self._windows[window_label]
        except KeyError as error:
            raise UnparseableJacclWindowError(
                f"no log text captured for window {window_label!r}"
            ) from error


_REQUIRED_LOG_KEYS: tuple[str, ...] = (
    "rtt_us",
    "retransmits",
    "messages_sent",
    "recv_pool_capacity",
    "recv_pool_peak",
    "stalls",
)


def parse_jaccl_transport_sample(
    window_label: str, raw_text: str
) -> JacclTransportSample:
    """Parse one window of log text into a sample.

    INTERFACE STUB. The accepted format is the *contract* a future real parser
    must satisfy, not the format jaccl emits today: whitespace-separated
    ``key=value`` pairs anywhere in the text, where ``rtt_us`` is a
    comma-separated list of microsecond samples and every key in
    :data:`_REQUIRED_LOG_KEYS` is present exactly once. Anything else raises
    :class:`UnparseableJacclWindowError`.
    """
    fields: dict[str, str] = {}
    for token in raw_text.split():
        if "=" not in token:
            continue
        key, _, value = token.partition("=")
        if key in fields:
            raise UnparseableJacclWindowError(
                f"window {window_label!r}: duplicate key {key!r}"
            )
        fields[key] = value
    missing = [key for key in _REQUIRED_LOG_KEYS if key not in fields]
    if missing:
        raise UnparseableJacclWindowError(
            f"window {window_label!r}: missing keys {missing}; "
            "refusing to substitute zeros for an unreadable window"
        )
    try:
        latencies = tuple(
            float(part) for part in fields["rtt_us"].split(",") if part != ""
        )
        sample = JacclTransportSample(
            window_label=window_label,
            round_trip=RoundTripHistogram(samples_microseconds=latencies),
            retransmits=int(fields["retransmits"]),
            messages_sent=int(fields["messages_sent"]),
            recv_pool_capacity=int(fields["recv_pool_capacity"]),
            recv_pool_peak_occupancy=int(fields["recv_pool_peak"]),
            stalls_detected=int(fields["stalls"]),
        )
    except (ValueError, TypeError) as error:
        raise UnparseableJacclWindowError(
            f"window {window_label!r}: malformed values ({error})"
        ) from error
    return sample


def summarise_samples(
    samples: Sequence[JacclTransportSample],
    *,
    metric_name: str,
    unit: str,
    minimum_replicates: int = DEFAULT_MINIMUM_REPLICATES,
    dispersion_threshold: float = DEFAULT_DISPERSION_THRESHOLD,
):
    """Fold per-window samples into a replication-gated value."""
    if not samples:
        raise ValueError("at least one sample is required")
    return aggregate_replicates(
        metric_name=metric_name,
        unit=unit,
        replicates=tuple(sample.metric(metric_name) for sample in samples),
        minimum_replicates=minimum_replicates,
        dispersion_threshold=dispersion_threshold,
    )


def jaccl_arm_environments(
    base_environment: Mapping[str, str],
    *,
    variable: str,
    arm_a_value: str,
    arm_b_value: str,
) -> tuple[dict[str, str], dict[str, str]]:
    """Derive two A/B arm environments differing in exactly one registered var.

    Rejects unregistered variables outright: an A/B toggle on a variable the
    fingerprint does not capture produces two arms whose fingerprints are
    identical, and Phase 1's ``compare_arms`` would then reject the pair with a
    confusing "must differ in exactly one var" error. Failing here says why.
    """
    if variable not in registered_env_names():
        raise KeyError(
            f"{variable!r} is not in FINGERPRINT_ENV_REGISTRY; a toggle the "
            "fingerprint cannot see is not an A/B comparison"
        )
    if arm_a_value == arm_b_value:
        raise ValueError("A/B arm values must differ")
    arm_a = dict(base_environment)
    arm_b = dict(base_environment)
    arm_a[variable] = arm_a_value
    arm_b[variable] = arm_b_value
    return arm_a, arm_b


def build_jaccl_record(
    *,
    metric_name: str,
    unit: str,
    samples_by_cell: Sequence[tuple[DepthMatrixCell, Sequence[JacclTransportSample]]],
    content_check: ContentCheckResult,
    runtime_mode_markers: Sequence[RuntimeModeMarker],
    canary_session_certified: bool,
    exo_repo: str,
    mlx_repo: str,
    command_runner: CommandRunner,
    environ: Mapping[str, str],
    fingerprint: Fingerprint | None = None,
    claims_default_safe: bool = False,
    notes: str = "",
) -> MeasurementRecord:
    """Build the record for a jaccl transport measurement.

    The depth-matrix loop is the caller's: ``samples_by_cell`` is an ordered
    sequence of ``(cell, windows measured at that cell)`` pairs, and every
    cell's samples are pooled into the replicate series so the dispersion gate
    sees cross-cell disagreement rather than hiding it behind a per-cell
    average. A sequence of pairs rather than a mapping because
    :class:`DepthMatrixCell` is a pydantic model and measurement order matters.
    """
    if not samples_by_cell:
        raise ValueError("at least one depth-matrix cell must be measured")
    cells = tuple(cell for cell, _ in samples_by_cell)
    pooled: list[JacclTransportSample] = []
    for cell, cell_samples in samples_by_cell:
        if not cell_samples:
            raise ValueError(f"cell {cell.label()} has no samples")
        pooled.extend(cell_samples)
    value = summarise_samples(pooled, metric_name=metric_name, unit=unit)
    captured = (
        capture_fingerprint(exo_repo, mlx_repo, runner=command_runner, environ=environ)
        if fingerprint is None
        else fingerprint
    )
    return MeasurementRecord(
        probe_name="jaccl_transport",
        value=value,
        content_check=content_check,
        fingerprint=captured,
        runtime_mode_markers=tuple(runtime_mode_markers),
        required_runtime_modes=JACCL_REQUIRED_RUNTIME_MODES,
        depth_matrix_cells=cells,
        claims_default_safe=claims_default_safe,
        token_ground_truth=None,
        prompt_size_dependent=False,
        canary_session_certified=canary_session_certified,
        notes=notes,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """CLI surface for the jaccl probe."""
    parser = argparse.ArgumentParser(prog="jaccl_probe")
    _ = parser.add_argument("--exo-repo", default=".")
    _ = parser.add_argument("--mlx-repo", default="./mlx")
    _ = parser.add_argument(
        "--metric",
        default="median_rtt_us",
        choices=[
            "median_rtt_us",
            "p99_rtt_us",
            "rtt_tail_ratio",
            "retransmit_rate",
            "recv_pool_occupancy_ratio",
            "stalls_detected",
        ],
    )
    _ = parser.add_argument(
        "--replicates", type=int, default=DEFAULT_MINIMUM_REPLICATES
    )
    _ = parser.add_argument(
        "--toggle-variable",
        default=None,
        help="registered env var to flip between the A and B arms",
    )
    _ = parser.add_argument("--arm-a-value", default=None)
    _ = parser.add_argument("--arm-b-value", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Refuses to run: the log source is not implemented yet."""
    _ = build_argument_parser().parse_args(None if argv is None else list(argv))
    raise NotImplementedError(
        "jaccl_probe has no live JacclLogSource yet. The log format is owned by "
        "the transport code; implement JacclLogSource.read_window and extend "
        "parse_jaccl_transport_sample against the real format first."
    )
