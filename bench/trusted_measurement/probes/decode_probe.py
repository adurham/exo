"""Per-token decode latency probe scaffold, with the stall watchdog (Phase 2).

REAL TODAY (typed, tested, no cluster required)
-----------------------------------------------
* :func:`gaps_from_timestamps` / :func:`decode_latencies_milliseconds` - per-token
  gap arithmetic from a raw token-arrival timestamp sequence.
* :class:`StallWatchdog` - the full trigger logic Fable's design called for: a
  per-token gap threshold, expressed both as an absolute millisecond bound and
  as a multiple of the running median, with a consecutive-gap requirement and a
  cooldown so one stall does not fire a diagnostic storm. Entirely testable
  with synthetic timestamp sequences.
* :func:`build_decode_record` - record construction over a depth-matrix sweep.

INTERFACE STUB (needs real hardware)
------------------------------------
* :class:`DecodeClient` - the one network call; streams tokens and yields their
  arrival timestamps.
* :class:`DiagnosticAction` - what the watchdog *does* when it fires. On real
  hardware this captures a stack sample of the wedged runner (``sample``/
  faulthandler dump on both nodes). Here it is an interface;
  :class:`RecordingDiagnosticAction` records invocations for tests.
"""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Protocol, final, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from trusted_measurement.content_check import (
    ContentCheckResult,
    NeedleHaystack,
    run_needle_check,
)
from trusted_measurement.depth_matrix import DepthMatrixCell
from trusted_measurement.fingerprint import (
    CommandRunner,
    Fingerprint,
    capture_fingerprint,
)
from trusted_measurement.record import MeasurementRecord
from trusted_measurement.replication import (
    DEFAULT_DISPERSION_THRESHOLD,
    DEFAULT_MINIMUM_REPLICATES,
    aggregate_replicates,
)
from trusted_measurement.runtime_mode import RuntimeModeMarker

__all__ = [
    "DECODE_REQUIRED_RUNTIME_MODES",
    "DecodeClient",
    "DecodeProbeConfig",
    "DecodeStreamSample",
    "DiagnosticAction",
    "RecordingDiagnosticAction",
    "StallEvent",
    "StallWatchdog",
    "WatchdogPolicy",
    "build_argument_parser",
    "build_decode_record",
    "decode_latencies_milliseconds",
    "gaps_from_timestamps",
    "main",
]

DECODE_REQUIRED_RUNTIME_MODES: tuple[str, ...] = ("decode_step",)
"""TODO(phase-3, needs coordination): emitted from the decode loop itself."""


def gaps_from_timestamps(timestamps: Sequence[float]) -> tuple[float, ...]:
    """Inter-arrival gaps in the timestamps' own unit (seconds).

    Requires a strictly non-decreasing sequence: a token that appears to have
    arrived before its predecessor means the clock or the capture is wrong, and
    a negative gap must never be quietly folded into a latency distribution.
    """
    if len(timestamps) < 2:
        return ()
    gaps: list[float] = []
    for earlier, later in pairwise(timestamps):
        gap = later - earlier
        if gap < 0.0:
            raise ValueError(
                "token arrival timestamps decreased; refusing to compute "
                "latencies from a non-monotonic capture"
            )
        gaps.append(gap)
    return tuple(gaps)


def decode_latencies_milliseconds(timestamps: Sequence[float]) -> tuple[float, ...]:
    """Per-token decode latencies in milliseconds."""
    return tuple(gap * 1000.0 for gap in gaps_from_timestamps(timestamps))


@final
class WatchdogPolicy(BaseModel):
    """When a per-token gap counts as a stall.

    A gap trips only when it exceeds BOTH bounds that are configured:
    ``absolute_threshold_ms`` and (once ``warmup_tokens`` gaps have been seen)
    ``median_multiple`` times the running median. Requiring both keeps a
    genuinely slow-but-steady regime from firing continuously, while
    ``median_multiple`` catches a stall that is small in absolute terms but
    enormous relative to this run's own baseline.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    absolute_threshold_ms: float = Field(gt=0.0)
    median_multiple: float = Field(gt=1.0, default=5.0)
    consecutive_gaps_required: int = Field(ge=1, default=1)
    warmup_tokens: int = Field(ge=1, default=8)
    cooldown_gaps: int = Field(ge=0, default=16)


@final
class StallEvent(BaseModel):
    """One watchdog firing."""

    model_config = ConfigDict(frozen=True, strict=True)

    token_index: int = Field(ge=0)
    gap_ms: float = Field(ge=0.0)
    running_median_ms: float = Field(ge=0.0)
    consecutive_gaps: int = Field(ge=1)

    @property
    def median_multiple(self) -> float:
        if self.running_median_ms <= 0.0:
            return float("inf")
        return self.gap_ms / self.running_median_ms


@runtime_checkable
class DiagnosticAction(Protocol):
    """INTERFACE STUB - what to do when the watchdog fires.

    On real hardware: capture a stack sample of both runner processes (and the
    GPU state) while the stall is still in progress. Must be fast and must not
    perturb the run more than the stall already has.
    """

    def capture(self, event: StallEvent) -> None: ...


@final
class RecordingDiagnosticAction:
    """Test double: records every stall it was asked to diagnose."""

    def __init__(self) -> None:
        self.events: list[StallEvent] = []

    def capture(self, event: StallEvent) -> None:
        self.events.append(event)


@final
class StallWatchdog:
    """Per-token gap watchdog. Fully implemented and independently testable.

    Feed it gaps in millisecond order via :meth:`observe_gap` (or a whole
    timestamp sequence via :meth:`scan_timestamps`). It returns the
    :class:`StallEvent` for a firing gap, or ``None``.
    """

    def __init__(
        self, policy: WatchdogPolicy, action: DiagnosticAction | None = None
    ) -> None:
        self._policy: WatchdogPolicy = policy
        self._action: DiagnosticAction | None = action
        self._gaps: list[float] = []
        self._consecutive: int = 0
        self._cooldown_remaining: int = 0
        self._events: list[StallEvent] = []

    @property
    def policy(self) -> WatchdogPolicy:
        return self._policy

    @property
    def events(self) -> tuple[StallEvent, ...]:
        return tuple(self._events)

    @property
    def observed_gaps(self) -> int:
        return len(self._gaps)

    def running_median_ms(self) -> float:
        if not self._gaps:
            return 0.0
        return statistics.median(self._gaps)

    def _exceeds(self, gap_ms: float) -> bool:
        if gap_ms <= self._policy.absolute_threshold_ms:
            return False
        if len(self._gaps) < self._policy.warmup_tokens:
            # Not enough history for a relative bound; absolute alone decides.
            return True
        median = statistics.median(self._gaps)
        if median <= 0.0:
            return True
        return gap_ms > median * self._policy.median_multiple

    def observe_gap(self, gap_ms: float, *, token_index: int) -> StallEvent | None:
        """Feed one per-token gap; returns the stall event if this gap fired."""
        if gap_ms < 0.0:
            raise ValueError("gap must be non-negative")
        # The median baseline is the *pre-stall* history, so a stall never
        # inflates the very baseline used to judge the stall after it.
        median_before = self.running_median_ms()
        exceeded = self._exceeds(gap_ms)
        if exceeded:
            self._consecutive += 1
        else:
            self._consecutive = 0
            self._gaps.append(gap_ms)
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return None
        if not exceeded:
            return None
        if self._consecutive < self._policy.consecutive_gaps_required:
            return None
        event = StallEvent(
            token_index=token_index,
            gap_ms=gap_ms,
            running_median_ms=median_before,
            consecutive_gaps=self._consecutive,
        )
        self._events.append(event)
        self._cooldown_remaining = self._policy.cooldown_gaps
        if self._action is not None:
            self._action.capture(event)
        return event

    def scan_timestamps(self, timestamps: Sequence[float]) -> tuple[StallEvent, ...]:
        """Run the watchdog over a whole token-arrival timestamp sequence."""
        for index, gap_ms in enumerate(decode_latencies_milliseconds(timestamps)):
            _ = self.observe_gap(gap_ms, token_index=index + 1)
        return self.events


@final
class DecodeStreamSample(BaseModel):
    """One decode stream: token arrival times plus the text produced."""

    model_config = ConfigDict(frozen=True, strict=True)

    token_arrival_seconds: tuple[float, ...] = Field(min_length=2)
    completion_text: str

    @property
    def latencies_ms(self) -> tuple[float, ...]:
        return decode_latencies_milliseconds(self.token_arrival_seconds)

    @property
    def median_latency_ms(self) -> float:
        return statistics.median(self.latencies_ms)

    @property
    def throughput_tokens_per_second(self) -> float:
        span = self.token_arrival_seconds[-1] - self.token_arrival_seconds[0]
        if span <= 0.0:
            raise ValueError("decode stream has zero duration")
        return (len(self.token_arrival_seconds) - 1) / span


@runtime_checkable
class DecodeClient(Protocol):
    """INTERFACE STUB - streams a decode and returns arrival timestamps."""

    def stream_decode(self, prompt: str, max_tokens: int) -> DecodeStreamSample: ...


@final
class DecodeProbeConfig(BaseModel):
    """Decode sweep configuration."""

    model_config = ConfigDict(frozen=True, strict=True)

    max_tokens: int = Field(gt=1)
    replicates_per_cell: int = Field(ge=1)
    watchdog: WatchdogPolicy


def build_decode_record(
    *,
    client: DecodeClient,
    prompts_by_cell: Sequence[tuple[DepthMatrixCell, NeedleHaystack]],
    config: DecodeProbeConfig,
    runtime_mode_markers: Sequence[RuntimeModeMarker],
    canary_session_certified: bool,
    exo_repo: str,
    mlx_repo: str,
    command_runner: CommandRunner,
    environ: Mapping[str, str],
    diagnostic_action: DiagnosticAction | None = None,
    fingerprint: Fingerprint | None = None,
    claims_default_safe: bool = False,
    notes: str = "",
) -> tuple[MeasurementRecord, tuple[StallEvent, ...]]:
    """Run the decode sweep; return the record and every stall observed.

    Stalls are returned rather than folded into the value: a stalled run is
    still a real observation of the system, but the caller must decide whether
    the number is describing steady-state decode or a wedge.
    """
    if not prompts_by_cell:
        raise ValueError("at least one depth-matrix cell must be measured")
    watchdog = StallWatchdog(config.watchdog, diagnostic_action)
    replicates: list[float] = []
    content_check: ContentCheckResult | None = None
    for _cell, haystack in prompts_by_cell:
        for _ in range(config.replicates_per_cell):
            sample = client.stream_decode(haystack.prompt, config.max_tokens)
            _ = watchdog.scan_timestamps(sample.token_arrival_seconds)
            content_check = run_needle_check(haystack, sample.completion_text)
            replicates.append(sample.median_latency_ms)
    if content_check is None:
        raise ValueError("decode config produced no measured replicates")
    value = aggregate_replicates(
        metric_name="decode_per_token_latency",
        unit="ms",
        replicates=tuple(replicates),
        minimum_replicates=DEFAULT_MINIMUM_REPLICATES,
        dispersion_threshold=DEFAULT_DISPERSION_THRESHOLD,
    )
    captured = (
        capture_fingerprint(exo_repo, mlx_repo, runner=command_runner, environ=environ)
        if fingerprint is None
        else fingerprint
    )
    record = MeasurementRecord(
        probe_name="decode_latency",
        value=value,
        content_check=content_check,
        fingerprint=captured,
        runtime_mode_markers=tuple(runtime_mode_markers),
        required_runtime_modes=DECODE_REQUIRED_RUNTIME_MODES,
        depth_matrix_cells=tuple(cell for cell, _ in prompts_by_cell),
        claims_default_safe=claims_default_safe,
        token_ground_truth=None,
        prompt_size_dependent=False,
        canary_session_certified=canary_session_certified,
        notes=notes,
    )
    return record, watchdog.events


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="decode_probe")
    _ = parser.add_argument("--exo-repo", default=".")
    _ = parser.add_argument("--mlx-repo", default="./mlx")
    _ = parser.add_argument("--base-url", default="http://localhost:52415")
    _ = parser.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash-0731")
    _ = parser.add_argument("--max-tokens", type=int, default=256)
    _ = parser.add_argument(
        "--replicates", type=int, default=DEFAULT_MINIMUM_REPLICATES
    )
    _ = parser.add_argument("--stall-threshold-ms", type=float, default=2000.0)
    _ = parser.add_argument("--stall-median-multiple", type=float, default=5.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Refuses to run: no live DecodeClient exists yet."""
    _ = build_argument_parser().parse_args(None if argv is None else list(argv))
    raise NotImplementedError(
        "decode_probe has no live DecodeClient or DiagnosticAction yet. "
        "Implement both against the cluster API / a real stack sampler."
    )
