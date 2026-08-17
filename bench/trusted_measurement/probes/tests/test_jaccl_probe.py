"""jaccl transport probe: data model, parsing contract, A/B plumbing, record."""

from __future__ import annotations

import math

import pytest

from trusted_measurement.content_check import build_needle_haystack, run_needle_check
from trusted_measurement.depth_matrix import ALL_DEPTH_CELLS, DepthMatrixCell
from trusted_measurement.probes.jaccl_probe import (
    JACCL_REQUIRED_RUNTIME_MODES,
    JacclTransportSample,
    RoundTripHistogram,
    StaticJacclLogSource,
    UnparseableJacclWindowError,
    build_argument_parser,
    build_jaccl_record,
    jaccl_arm_environments,
    main,
    parse_jaccl_transport_sample,
    summarise_samples,
)
from trusted_measurement.probes.tests.builders import (
    fake_command_runner,
    markers_for,
)

# --------------------------------------------------------------- histogram


def test_histogram_percentiles_and_tail_ratio() -> None:
    histogram = RoundTripHistogram(
        samples_microseconds=tuple(float(value) for value in range(1, 101))
    )
    assert histogram.count == 100
    assert math.isclose(histogram.median_microseconds, 50.5, rel_tol=1e-9)
    assert histogram.percentile_microseconds(0.0) == 1.0
    assert histogram.percentile_microseconds(50.0) == 50.0
    assert histogram.percentile_microseconds(99.0) == 99.0
    assert histogram.percentile_microseconds(100.0) == 100.0
    assert math.isclose(histogram.tail_ratio, 99.0 / 50.5, rel_tol=1e-9)


def test_histogram_rejects_out_of_range_percentile() -> None:
    histogram = RoundTripHistogram(samples_microseconds=(1.0, 2.0))
    with pytest.raises(ValueError):
        _ = histogram.percentile_microseconds(101.0)


def test_histogram_requires_at_least_one_sample() -> None:
    with pytest.raises(ValueError):
        _ = RoundTripHistogram(samples_microseconds=())


# ------------------------------------------------------------------ sample


def _sample(label: str = "w0", *, retransmits: int = 5) -> JacclTransportSample:
    return JacclTransportSample(
        window_label=label,
        round_trip=RoundTripHistogram(samples_microseconds=(10.0, 20.0, 30.0)),
        retransmits=retransmits,
        messages_sent=1000,
        recv_pool_capacity=256,
        recv_pool_peak_occupancy=64,
        stalls_detected=0,
    )


def test_sample_derived_metrics() -> None:
    sample = _sample()
    assert math.isclose(sample.retransmit_rate, 0.005, rel_tol=1e-9)
    assert math.isclose(sample.recv_pool_occupancy_ratio, 0.25, rel_tol=1e-9)
    assert sample.metric("median_rtt_us") == 20.0
    assert math.isclose(sample.metric("retransmit_rate"), 0.005, rel_tol=1e-9)
    assert sample.metric("stalls_detected") == 0.0


def test_sample_retransmit_rate_is_zero_when_nothing_was_sent() -> None:
    sample = JacclTransportSample(
        window_label="idle",
        round_trip=RoundTripHistogram(samples_microseconds=(1.0,)),
        retransmits=0,
        messages_sent=0,
        recv_pool_capacity=8,
        recv_pool_peak_occupancy=0,
        stalls_detected=0,
    )
    assert sample.retransmit_rate == 0.0


def test_unknown_metric_name_raises() -> None:
    with pytest.raises(KeyError):
        _ = _sample().metric("not_a_metric")


# ------------------------------------------------------------------ parsing

_GOOD_LINE = (
    "jaccl window rtt_us=10,20,30 retransmits=4 messages_sent=800 "
    "recv_pool_capacity=256 recv_pool_peak=32 stalls=1"
)


def test_parses_the_documented_contract() -> None:
    sample = parse_jaccl_transport_sample("w0", _GOOD_LINE)
    assert sample.window_label == "w0"
    assert sample.round_trip.count == 3
    assert sample.retransmits == 4
    assert sample.stalls_detected == 1


def test_missing_key_raises_rather_than_defaulting_to_zero() -> None:
    truncated = _GOOD_LINE.replace(" retransmits=4", "")
    with pytest.raises(UnparseableJacclWindowError, match="retransmits"):
        _ = parse_jaccl_transport_sample("w0", truncated)


def test_duplicate_key_raises() -> None:
    with pytest.raises(UnparseableJacclWindowError, match="duplicate"):
        _ = parse_jaccl_transport_sample("w0", _GOOD_LINE + " stalls=2")


def test_malformed_value_raises() -> None:
    broken = _GOOD_LINE.replace("retransmits=4", "retransmits=many")
    with pytest.raises(UnparseableJacclWindowError, match="malformed"):
        _ = parse_jaccl_transport_sample("w0", broken)


def test_static_log_source_round_trip() -> None:
    source = StaticJacclLogSource({"w0": _GOOD_LINE})
    sample = parse_jaccl_transport_sample("w0", source.read_window("w0"))
    assert sample.messages_sent == 800
    with pytest.raises(UnparseableJacclWindowError):
        _ = source.read_window("missing")


# -------------------------------------------------------------- summarise


def test_summarise_flags_non_reproducing_windows() -> None:
    samples = [
        JacclTransportSample(
            window_label=f"w{index}",
            round_trip=RoundTripHistogram(samples_microseconds=(latency,)),
            retransmits=0,
            messages_sent=10,
            recv_pool_capacity=8,
            recv_pool_peak_occupancy=1,
            stalls_detected=0,
        )
        for index, latency in enumerate((10.0, 10.2, 400.0))
    ]
    value = summarise_samples(samples, metric_name="median_rtt_us", unit="us")
    assert value.verdict == "non_reproducing"
    with pytest.raises(ValueError):
        _ = value.reportable_value()


def test_summarise_requires_samples() -> None:
    with pytest.raises(ValueError):
        _ = summarise_samples([], metric_name="median_rtt_us", unit="us")


# ------------------------------------------------------------- A/B plumbing


def test_arm_environments_flip_exactly_one_registered_variable() -> None:
    base = {"MLX_JACCL_SHARDING_MODE": "tp", "EXO_PREFILL_STEP_SIZE": "2048"}
    arm_a, arm_b = jaccl_arm_environments(
        base,
        variable="MLX_JACCL_ACK_RETRANSMIT_US",
        arm_a_value="2000",
        arm_b_value="4000",
    )
    differing = {key for key in arm_a if arm_a[key] != arm_b.get(key)}
    assert differing == {"MLX_JACCL_ACK_RETRANSMIT_US"}
    assert arm_a["MLX_JACCL_SHARDING_MODE"] == "tp"


def test_arm_environments_reject_unregistered_variable() -> None:
    with pytest.raises(KeyError, match="FINGERPRINT_ENV_REGISTRY"):
        _ = jaccl_arm_environments(
            {}, variable="TOTALLY_UNDECLARED", arm_a_value="0", arm_b_value="1"
        )


def test_arm_environments_reject_identical_values() -> None:
    with pytest.raises(ValueError):
        _ = jaccl_arm_environments(
            {},
            variable="MLX_JACCL_ACK_RETRANSMIT_US",
            arm_a_value="1",
            arm_b_value="1",
        )


# ----------------------------------------------------------------- record


def _full_cells() -> tuple[DepthMatrixCell, ...]:
    return tuple(
        DepthMatrixCell(context_depth=depth, thermal_state=thermal, prompt_tokens=4096)
        for depth, thermal in ALL_DEPTH_CELLS
    )


def _good_content_check():
    haystack = build_needle_haystack(filler_sentences=4, seed=7)
    return run_needle_check(haystack, f"code is {haystack.needle}")


def test_build_jaccl_record_is_trusted_when_the_envelope_is_complete() -> None:
    cells = _full_cells()
    record = build_jaccl_record(
        metric_name="median_rtt_us",
        unit="us",
        samples_by_cell=[(cell, [_sample(cell.label())]) for cell in cells],
        content_check=_good_content_check(),
        runtime_mode_markers=markers_for(*JACCL_REQUIRED_RUNTIME_MODES),
        canary_session_certified=True,
        exo_repo="/exo",
        mlx_repo="/mlx",
        command_runner=fake_command_runner(),
        environ={"MLX_JACCL_SHARDING_MODE": "tp"},
        claims_default_safe=True,
    )
    assert record.validate_envelope() == ()
    assert record.trusted_value() == 20.0
    assert record.fingerprint.link_topology.thunderbolt_link_count == 6
    assert record.fingerprint.registered_env["MLX_JACCL_SHARDING_MODE"] == "tp"


def test_build_jaccl_record_refuses_without_the_required_marker() -> None:
    cells = _full_cells()
    record = build_jaccl_record(
        metric_name="median_rtt_us",
        unit="us",
        samples_by_cell=[(cell, [_sample(cell.label())]) for cell in cells],
        content_check=_good_content_check(),
        runtime_mode_markers=markers_for("some_other_mode"),
        canary_session_certified=True,
        exo_repo="/exo",
        mlx_repo="/mlx",
        command_runner=fake_command_runner(),
        environ={"MLX_JACCL_SHARDING_MODE": "tp"},
    )
    violations = record.validate_envelope()
    assert any("jaccl_reliable_data" in str(violation) for violation in violations)


def test_partial_matrix_may_not_claim_default_safe() -> None:
    cell = DepthMatrixCell(
        context_depth="shallow", thermal_state="warm", prompt_tokens=1024
    )
    record = build_jaccl_record(
        metric_name="median_rtt_us",
        unit="us",
        samples_by_cell=[(cell, [_sample(), _sample(), _sample()])],
        content_check=_good_content_check(),
        runtime_mode_markers=markers_for(*JACCL_REQUIRED_RUNTIME_MODES),
        canary_session_certified=True,
        exo_repo="/exo",
        mlx_repo="/mlx",
        command_runner=fake_command_runner(),
        environ={"MLX_JACCL_SHARDING_MODE": "tp"},
        claims_default_safe=True,
    )
    violations = record.validate_envelope()
    assert any(violation.field == "claims_default_safe" for violation in violations)


def test_empty_cell_is_rejected() -> None:
    cell = DepthMatrixCell(
        context_depth="mid", thermal_state="warm", prompt_tokens=1024
    )
    with pytest.raises(ValueError, match="no samples"):
        _ = build_jaccl_record(
            metric_name="median_rtt_us",
            unit="us",
            samples_by_cell=[(cell, [])],
            content_check=_good_content_check(),
            runtime_mode_markers=markers_for(*JACCL_REQUIRED_RUNTIME_MODES),
            canary_session_certified=True,
            exo_repo="/exo",
            mlx_repo="/mlx",
            command_runner=fake_command_runner(),
            environ={},
        )


def test_no_cells_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one depth-matrix cell"):
        _ = build_jaccl_record(
            metric_name="median_rtt_us",
            unit="us",
            samples_by_cell=[],
            content_check=_good_content_check(),
            runtime_mode_markers=(),
            canary_session_certified=True,
            exo_repo="/exo",
            mlx_repo="/mlx",
            command_runner=fake_command_runner(),
            environ={},
        )


# --------------------------------------------------------------------- CLI


def test_argument_parser_accepts_the_documented_flags() -> None:
    arguments = build_argument_parser().parse_args(
        [
            "--metric",
            "retransmit_rate",
            "--replicates",
            "5",
            "--toggle-variable",
            "MLX_JACCL_ACK_RETRANSMIT_US",
        ]
    )
    assert arguments.metric == "retransmit_rate"  # pyright: ignore[reportAny]
    assert arguments.replicates == 5  # pyright: ignore[reportAny]


def test_main_is_an_honest_stub() -> None:
    with pytest.raises(NotImplementedError, match="JacclLogSource"):
        _ = main([])
