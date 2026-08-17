"""Decode probe: gap arithmetic, the stall watchdog trigger logic, record."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import final

import pytest

from trusted_measurement.content_check import NeedleHaystack, build_needle_haystack
from trusted_measurement.depth_matrix import ALL_DEPTH_CELLS, DepthMatrixCell
from trusted_measurement.probes.decode_probe import (
    DECODE_REQUIRED_RUNTIME_MODES,
    DecodeClient,
    DecodeProbeConfig,
    DecodeStreamSample,
    DiagnosticAction,
    RecordingDiagnosticAction,
    StallWatchdog,
    WatchdogPolicy,
    build_argument_parser,
    build_decode_record,
    decode_latencies_milliseconds,
    gaps_from_timestamps,
    main,
)
from trusted_measurement.probes.tests.builders import (
    fake_command_runner,
    markers_for,
)

# ------------------------------------------------------------ gap arithmetic


def test_gaps_from_timestamps() -> None:
    assert gaps_from_timestamps([0.0, 1.0, 1.5, 4.0]) == (1.0, 0.5, 2.5)
    assert decode_latencies_milliseconds([0.0, 0.05]) == (50.0,)


def test_gaps_need_at_least_two_timestamps() -> None:
    assert gaps_from_timestamps([]) == ()
    assert gaps_from_timestamps([1.0]) == ()


def test_non_monotonic_capture_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-monotonic"):
        _ = gaps_from_timestamps([0.0, 2.0, 1.0])


# ------------------------------------------------- watchdog: trigger logic
#
# Every case below is a synthetic gap sequence. Steady baseline = 20ms.


def _steady(count: int, value: float = 20.0) -> list[float]:
    return [value] * count


def _policy(**overrides: object) -> WatchdogPolicy:
    fields: dict[str, object] = dict(
        absolute_threshold_ms=200.0,
        median_multiple=5.0,
        consecutive_gaps_required=1,
        warmup_tokens=8,
        cooldown_gaps=0,
    )
    fields.update(overrides)
    return WatchdogPolicy(**fields)  # pyright: ignore[reportArgumentType]


def _run(policy: WatchdogPolicy, gaps: Sequence[float]) -> StallWatchdog:
    watchdog = StallWatchdog(policy)
    for index, gap in enumerate(gaps):
        _ = watchdog.observe_gap(gap, token_index=index)
    return watchdog


def test_steady_stream_never_fires() -> None:
    watchdog = _run(_policy(), _steady(50))
    assert watchdog.events == ()
    assert watchdog.running_median_ms() == 20.0


def test_a_clear_stall_fires() -> None:
    watchdog = _run(_policy(), [*_steady(20), 3000.0, *_steady(10)])
    assert len(watchdog.events) == 1
    event = watchdog.events[0]
    assert event.token_index == 20
    assert event.gap_ms == 3000.0
    assert event.running_median_ms == 20.0
    assert math.isclose(event.median_multiple, 150.0, rel_tol=1e-9)


def test_gap_below_the_absolute_threshold_never_fires_even_if_relatively_huge() -> None:
    # 199ms is ~10x the 20ms median, but under the 200ms absolute bound.
    watchdog = _run(_policy(), [*_steady(20), 199.0, *_steady(5)])
    assert watchdog.events == ()


def test_gap_above_the_absolute_threshold_but_not_relatively_large_does_not_fire() -> (
    None
):
    # Slow-but-steady regime: 500ms baseline, one 600ms gap. Over the DEFAULT
    # absolute bound (200ms) but only 1.2x the median once warmup is past --
    # not a stall, just this deployment's normal speed. A real deployment
    # calibrates absolute_threshold_ms to its own expected per-token latency
    # rather than an arbitrary global default, so this policy is calibrated
    # for the 500ms baseline it is actually testing (well above the baseline,
    # so the warmup window's absolute-alone rule doesn't itself misfire on
    # the baseline before any history exists).
    watchdog = _run(
        _policy(absolute_threshold_ms=1000.0),
        [*_steady(20, 500.0), 600.0, *_steady(5, 500.0)],
    )
    assert watchdog.events == ()


def test_boundary_gap_exactly_at_the_absolute_threshold_does_not_fire() -> None:
    watchdog = _run(_policy(), [*_steady(20), 200.0])
    assert watchdog.events == ()


def test_boundary_gap_exactly_at_the_median_multiple_does_not_fire() -> None:
    # median 100ms, multiple 5 -> exactly 500ms is not "greater than". Absolute
    # threshold set above the 100ms baseline so warmup's absolute-alone rule
    # doesn't fire on the baseline itself before history accumulates.
    watchdog = _run(_policy(absolute_threshold_ms=150.0), [*_steady(20, 100.0), 500.0])
    assert watchdog.events == ()


def test_one_microsecond_over_both_bounds_fires() -> None:
    watchdog = _run(
        _policy(absolute_threshold_ms=150.0), [*_steady(20, 100.0), 500.001]
    )
    assert len(watchdog.events) == 1


def test_during_warmup_the_absolute_bound_alone_decides() -> None:
    # Only 3 gaps of history (< warmup_tokens=8): no reliable median yet, so a
    # gap over the absolute bound fires immediately rather than being masked.
    watchdog = _run(_policy(), [20.0, 20.0, 20.0, 5000.0])
    assert len(watchdog.events) == 1
    assert watchdog.events[0].consecutive_gaps == 1


def test_consecutive_requirement_suppresses_an_isolated_spike() -> None:
    policy = _policy(consecutive_gaps_required=3)
    watchdog = _run(policy, [*_steady(20), 3000.0, *_steady(10)])
    assert watchdog.events == ()


def test_consecutive_requirement_fires_on_a_sustained_stall() -> None:
    policy = _policy(consecutive_gaps_required=3)
    watchdog = _run(policy, [*_steady(20), 3000.0, 3000.0, 3000.0, 3000.0])
    assert len(watchdog.events) == 2
    assert watchdog.events[0].consecutive_gaps == 3
    assert watchdog.events[1].consecutive_gaps == 4


def test_cooldown_suppresses_a_diagnostic_storm() -> None:
    policy = _policy(cooldown_gaps=5)
    watchdog = _run(policy, [*_steady(20), *([3000.0] * 4), *_steady(10)])
    assert len(watchdog.events) == 1


def test_without_cooldown_every_stalled_gap_fires() -> None:
    watchdog = _run(_policy(cooldown_gaps=0), [*_steady(20), *([3000.0] * 4)])
    assert len(watchdog.events) == 4


def test_stall_gaps_do_not_poison_the_baseline_median() -> None:
    watchdog = _run(_policy(), [*_steady(20), *([3000.0] * 30)])
    # Stalled gaps are excluded from the running history, so the median still
    # describes healthy decode rather than drifting up to meet the stall.
    assert watchdog.running_median_ms() == 20.0
    assert watchdog.observed_gaps == 20


def test_negative_gap_is_rejected() -> None:
    with pytest.raises(ValueError):
        _ = StallWatchdog(_policy()).observe_gap(-1.0, token_index=0)


def test_median_multiple_of_a_zero_baseline_is_infinite() -> None:
    watchdog = StallWatchdog(_policy())
    event = watchdog.observe_gap(5000.0, token_index=0)
    assert event is not None
    assert event.running_median_ms == 0.0
    assert event.median_multiple == float("inf")


def test_scan_timestamps_matches_gap_by_gap_feeding() -> None:
    timestamps = [0.0]
    for gap_ms in [*_steady(20), 3000.0, *_steady(5)]:
        timestamps.append(timestamps[-1] + gap_ms / 1000.0)
    scanned = StallWatchdog(_policy()).scan_timestamps(timestamps)
    fed = _run(_policy(), [*_steady(20), 3000.0, *_steady(5)]).events
    assert [event.gap_ms for event in scanned] == [event.gap_ms for event in fed]


def test_policy_rejects_a_median_multiple_of_one_or_less() -> None:
    with pytest.raises(ValueError):
        _ = WatchdogPolicy(absolute_threshold_ms=10.0, median_multiple=1.0)


# ------------------------------------------------------- diagnostic action


def test_diagnostic_action_is_invoked_once_per_firing() -> None:
    action = RecordingDiagnosticAction()
    assert isinstance(action, DiagnosticAction)
    watchdog = StallWatchdog(_policy(), action)
    for index, gap in enumerate([*_steady(20), 3000.0, 20.0, 4000.0]):
        _ = watchdog.observe_gap(gap, token_index=index)
    assert len(action.events) == 2
    assert [event.gap_ms for event in action.events] == [3000.0, 4000.0]


# -------------------------------------------------------------- fake client


@final
class FakeDecodeClient:
    """Replays a fixed per-token gap pattern and answers the needle."""

    def __init__(self, *, gaps_ms: Sequence[float], needle: str) -> None:
        self._gaps: tuple[float, ...] = tuple(gaps_ms)
        self._needle: str = needle
        self.calls: int = 0

    def stream_decode(self, prompt: str, max_tokens: int) -> DecodeStreamSample:
        _ = (prompt, max_tokens)
        self.calls += 1
        timestamps = [0.0]
        for gap in self._gaps:
            timestamps.append(timestamps[-1] + gap / 1000.0)
        return DecodeStreamSample(
            token_arrival_seconds=tuple(timestamps),
            completion_text=f"The secret access code is {self._needle}.",
        )


def test_stream_sample_derived_metrics() -> None:
    sample = DecodeStreamSample(
        token_arrival_seconds=(0.0, 0.02, 0.04, 0.06), completion_text="x"
    )
    assert len(sample.latencies_ms) == 3
    for actual, expected in zip(sample.latencies_ms, (20.0, 20.0, 20.0), strict=True):
        assert math.isclose(actual, expected, rel_tol=1e-9)
    assert math.isclose(sample.median_latency_ms, 20.0, rel_tol=1e-9)
    assert math.isclose(sample.throughput_tokens_per_second, 50.0, rel_tol=1e-9)


# ------------------------------------------------------------------ record


def _cells_with_prompts() -> list[tuple[DepthMatrixCell, NeedleHaystack]]:
    haystack = build_needle_haystack(filler_sentences=6, seed=42)
    return [
        (
            DepthMatrixCell(
                context_depth=depth, thermal_state=thermal, prompt_tokens=2048
            ),
            haystack,
        )
        for depth, thermal in ALL_DEPTH_CELLS
    ]


def test_decode_record_is_trusted_and_reports_no_stalls_on_a_clean_run() -> None:
    pairs = _cells_with_prompts()
    needle = pairs[0][1].needle
    client = FakeDecodeClient(gaps_ms=_steady(30), needle=needle)
    assert isinstance(client, DecodeClient)
    record, stalls = build_decode_record(
        client=client,
        prompts_by_cell=pairs,
        config=DecodeProbeConfig(
            max_tokens=32, replicates_per_cell=3, watchdog=_policy()
        ),
        runtime_mode_markers=markers_for(*DECODE_REQUIRED_RUNTIME_MODES),
        canary_session_certified=True,
        exo_repo="/exo",
        mlx_repo="/mlx",
        command_runner=fake_command_runner(),
        environ={"MLX_JACCL_SHARDING_MODE": "tp"},
        claims_default_safe=True,
    )
    assert record.validate_envelope() == ()
    assert math.isclose(record.trusted_value(), 20.0, rel_tol=1e-9)
    assert stalls == ()
    assert client.calls == 18


def test_decode_record_surfaces_stalls_without_hiding_them_in_the_value() -> None:
    pairs = _cells_with_prompts()
    needle = pairs[0][1].needle
    action = RecordingDiagnosticAction()
    client = FakeDecodeClient(
        gaps_ms=[*_steady(20), 5000.0, *_steady(10)], needle=needle
    )
    record, stalls = build_decode_record(
        client=client,
        prompts_by_cell=pairs,
        config=DecodeProbeConfig(
            max_tokens=32,
            replicates_per_cell=3,
            watchdog=_policy(cooldown_gaps=0),
        ),
        runtime_mode_markers=markers_for(*DECODE_REQUIRED_RUNTIME_MODES),
        canary_session_certified=True,
        exo_repo="/exo",
        mlx_repo="/mlx",
        command_runner=fake_command_runner(),
        environ={},
        diagnostic_action=action,
        claims_default_safe=True,
    )
    assert record.validate_envelope() == ()
    # One stall per replicate stream; the median latency is unaffected.
    assert len(stalls) == 18
    assert len(action.events) == 18
    assert math.isclose(record.trusted_value(), 20.0, rel_tol=1e-9)


def test_decode_record_requires_at_least_one_cell() -> None:
    with pytest.raises(ValueError, match="at least one depth-matrix cell"):
        _ = build_decode_record(
            client=FakeDecodeClient(gaps_ms=_steady(5), needle="1"),
            prompts_by_cell=[],
            config=DecodeProbeConfig(
                max_tokens=8, replicates_per_cell=3, watchdog=_policy()
            ),
            runtime_mode_markers=(),
            canary_session_certified=True,
            exo_repo="/exo",
            mlx_repo="/mlx",
            command_runner=fake_command_runner(),
            environ={},
        )


def test_uncertified_session_makes_the_record_untrusted() -> None:
    pairs = _cells_with_prompts()
    client = FakeDecodeClient(gaps_ms=_steady(30), needle=pairs[0][1].needle)
    record, _ = build_decode_record(
        client=client,
        prompts_by_cell=pairs,
        config=DecodeProbeConfig(
            max_tokens=32, replicates_per_cell=3, watchdog=_policy()
        ),
        runtime_mode_markers=markers_for(*DECODE_REQUIRED_RUNTIME_MODES),
        canary_session_certified=False,
        exo_repo="/exo",
        mlx_repo="/mlx",
        command_runner=fake_command_runner(),
        environ={},
    )
    assert any(
        violation.field == "canary_session_certified"
        for violation in record.validate_envelope()
    )


# --------------------------------------------------------------------- CLI


def test_argument_parser_defaults() -> None:
    arguments = build_argument_parser().parse_args([])
    assert arguments.stall_threshold_ms == 2000.0  # pyright: ignore[reportAny]


def test_main_is_an_honest_stub() -> None:
    with pytest.raises(NotImplementedError, match="DecodeClient"):
        _ = main([])
