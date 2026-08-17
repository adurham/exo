"""Meta-test: can our own canary catch a broken harness?

The canary exists because the harness once reported green while broken. So the
canary itself must be tested against deliberately broken judges - if a
rubber-stamp validator does not make the canary go red, the canary is
decorative.
"""

from __future__ import annotations

import pytest

from trusted_measurement.canary import (
    CanaryReport,
    HarnessNotCertifiedError,
    default_judge,
    run_liveness_canary,
)
from trusted_measurement.record import EnvelopeViolation, MeasurementRecord


def _rubber_stamp_judge(record: MeasurementRecord) -> tuple[EnvelopeViolation, ...]:
    """A broken harness: everything is fine, always."""
    _ = record
    return ()


def _paranoid_judge(record: MeasurementRecord) -> tuple[EnvelopeViolation, ...]:
    """A different broken harness: everything is a violation, even valid records."""
    _ = record
    return (EnvelopeViolation(field="everything", reason="paranoid"),)


def _blind_to_token_inflation(
    record: MeasurementRecord,
) -> tuple[EnvelopeViolation, ...]:
    """A subtly broken harness: one specific check silently disabled."""
    return tuple(
        violation
        for violation in record.validate_envelope()
        if violation.field != "token_ground_truth"
    )


def test_real_harness_certifies() -> None:
    report = run_liveness_canary()
    assert report.certified, report.describe()
    report.require_certified()


def test_every_rigged_scenario_comes_back_red() -> None:
    report = run_liveness_canary()
    rigged = [result for result in report.results if result.expected_red]
    assert len(rigged) >= 9
    for result in rigged:
        assert result.was_red, f"{result.name} should have been RED"
        assert result.violations


def test_positive_control_comes_back_green() -> None:
    report = run_liveness_canary()
    control = next(result for result in report.results if not result.expected_red)
    assert not control.was_red
    assert control.violations == ()


def test_canary_detects_a_rubber_stamp_harness() -> None:
    """THE meta-test: a harness that approves everything must fail the canary."""
    report = run_liveness_canary(judge=_rubber_stamp_judge)
    assert not report.certified
    failures = {result.name for result in report.failures}
    assert "forged_content_proof" in failures
    assert "token_count_inflation" in failures
    assert "non_reproducing_replicates" in failures
    with pytest.raises(HarnessNotCertifiedError, match="cannot be trusted"):
        report.require_certified()


def test_canary_detects_a_harness_that_rejects_everything() -> None:
    report = run_liveness_canary(judge=_paranoid_judge)
    assert not report.certified
    assert "positive_control_valid_record" in {
        result.name for result in report.failures
    }


def test_canary_detects_one_silently_disabled_check() -> None:
    report = run_liveness_canary(judge=_blind_to_token_inflation)
    assert not report.certified
    assert {result.name for result in report.failures} == {"token_count_inflation"}


def test_empty_report_is_not_certified() -> None:
    """No scenarios ran is NOT the same as everything passed."""
    report = CanaryReport(results=())
    assert not report.certified
    with pytest.raises(HarnessNotCertifiedError):
        report.require_certified()


def test_describe_names_the_broken_scenarios() -> None:
    report = run_liveness_canary(judge=_rubber_stamp_judge)
    described = report.describe()
    assert "NOT CERTIFIED" in described
    assert "HARNESS BROKEN" in described


def test_default_judge_is_the_production_validator() -> None:
    from trusted_measurement.tests.builders import valid_record

    record = valid_record()
    assert default_judge(record) == record.validate_envelope()


def test_cli_canary_exits_zero() -> None:
    from trusted_measurement.__main__ import main

    assert main(["canary"]) == 0
