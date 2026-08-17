"""Dispersion gate: disagreeing replicates must be FLAGGED, never averaged away."""

from __future__ import annotations

import math

import pytest

from trusted_measurement.replication import (
    DEFAULT_DISPERSION_THRESHOLD,
    aggregate_replicates,
)


def test_three_agreeing_replicates_pass() -> None:
    value = aggregate_replicates(
        metric_name="prefill_throughput",
        unit="tok/s",
        replicates=(1000.0, 1010.0, 995.0),
    )
    assert value.verdict == "reproducing"
    assert value.is_trustworthy
    assert math.isclose(value.dispersion_ratio, 1010.0 / 995.0, rel_tol=1e-9)
    assert math.isclose(value.reportable_value(), 1000.0, rel_tol=1e-9)


def test_three_disagreeing_replicates_are_flagged_non_reproducing() -> None:
    value = aggregate_replicates(
        metric_name="prefill_throughput",
        unit="tok/s",
        replicates=(1000.0, 400.0, 1800.0),
    )
    assert value.verdict == "non_reproducing"
    assert not value.is_trustworthy
    assert math.isclose(value.dispersion_ratio, 4.5, rel_tol=1e-9)


def test_non_reproducing_refuses_to_report_the_best_or_the_mean() -> None:
    """The exact failure mode: quoting the fast run out of three."""
    value = aggregate_replicates(
        metric_name="decode_throughput",
        unit="tok/s",
        replicates=(12.0, 13.0, 30.0),
    )
    with pytest.raises(ValueError, match="no value may be reported"):
        _ = value.reportable_value()


def test_fewer_than_three_replicates_is_insufficient() -> None:
    for replicates in ((42.0,), (42.0, 42.1)):
        value = aggregate_replicates(
            metric_name="decode_throughput", unit="tok/s", replicates=replicates
        )
        assert value.verdict == "insufficient_replicates"
        assert not value.is_trustworthy


def test_threshold_is_configurable() -> None:
    replicates = (100.0, 130.0, 120.0)
    strict = aggregate_replicates(
        metric_name="m", unit="u", replicates=replicates, dispersion_threshold=1.1
    )
    loose = aggregate_replicates(
        metric_name="m", unit="u", replicates=replicates, dispersion_threshold=1.5
    )
    assert strict.verdict == "non_reproducing"
    assert loose.verdict == "reproducing"


def test_boundary_at_default_threshold_passes() -> None:
    value = aggregate_replicates(
        metric_name="m",
        unit="u",
        replicates=(100.0, 100.0, 100.0 * DEFAULT_DISPERSION_THRESHOLD),
    )
    assert value.verdict == "reproducing"


def test_zero_replicate_is_treated_as_infinite_dispersion() -> None:
    value = aggregate_replicates(
        metric_name="m", unit="u", replicates=(0.0, 10.0, 11.0)
    )
    assert value.verdict == "non_reproducing"


def test_empty_replicates_rejected() -> None:
    with pytest.raises(ValueError, match="at least one replicate"):
        _ = aggregate_replicates(metric_name="m", unit="u", replicates=())
