"""Every required envelope field, absent independently, must invalidate."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from trusted_measurement.content_check import (
    CONTENT_CHECK_DOMAIN,
    ContentCheckResult,
    build_needle_haystack,
    run_needle_check,
)
from trusted_measurement.depth_matrix import DepthMatrixCell
from trusted_measurement.fingerprint import LinkTopology
from trusted_measurement.proof import ProofToken
from trusted_measurement.record import MeasurementRecord, UntrustedMeasurementError
from trusted_measurement.replication import aggregate_replicates
from trusted_measurement.runtime_mode import RUNTIME_MODE_DOMAIN, RuntimeModeMarker
from trusted_measurement.tests.builders import (
    full_matrix_cells,
    good_content_check,
    good_fingerprint,
    good_markers,
    good_value,
    valid_record,
)
from trusted_measurement.token_truth import TokenGroundTruth


def test_baseline_record_is_trusted() -> None:
    record = valid_record()
    assert record.validate_envelope() == ()
    assert record.is_trusted
    assert math.isclose(record.trusted_value(), 20.0, rel_tol=1e-9)


@pytest.mark.parametrize(
    "missing_field",
    [
        "probe_name",
        "value",
        "content_check",
        "fingerprint",
        "runtime_mode_markers",
        "required_runtime_modes",
        "depth_matrix_cells",
        "prompt_size_dependent",
        "canary_session_certified",
    ],
)
def test_record_cannot_be_constructed_without_each_required_field(
    missing_field: str,
) -> None:
    """Structural absence of any envelope field is a construction error."""
    kwargs: dict[str, object] = {
        "probe_name": "test_probe",
        "value": good_value(),
        "content_check": good_content_check(),
        "fingerprint": good_fingerprint(),
        "runtime_mode_markers": good_markers(),
        "required_runtime_modes": ("tp_allreduce",),
        "depth_matrix_cells": full_matrix_cells(),
        "prompt_size_dependent": False,
        "canary_session_certified": True,
    }
    del kwargs[missing_field]
    with pytest.raises(ValidationError):
        _ = MeasurementRecord(**kwargs)  # pyright: ignore[reportArgumentType]


def test_empty_probe_name_rejected() -> None:
    with pytest.raises(ValidationError):
        _ = valid_record(probe_name="")


def test_empty_depth_matrix_rejected() -> None:
    with pytest.raises(ValidationError):
        _ = valid_record(depth_matrix_cells=())


def test_failed_content_check_invalidates() -> None:
    haystack = build_needle_haystack(filler_sentences=3, seed=9)
    record = valid_record(content_check=run_needle_check(haystack, "no idea"))
    violations = record.validate_envelope()
    assert [violation.field for violation in violations] == ["content_check"]
    with pytest.raises(UntrustedMeasurementError):
        _ = record.trusted_value()


def test_forged_content_check_proof_invalidates() -> None:
    """A hand-written 'it passed' cannot substitute for the check running."""
    forged = ContentCheckResult(
        check_name="needle_in_haystack",
        passed=True,
        expected="999999",
        observed_excerpt="999999",
        proof=ProofToken(
            domain=CONTENT_CHECK_DOMAIN, digest="f" * 64, session_id="not-this-session"
        ),
    )
    violations = valid_record(content_check=forged).validate_envelope()
    assert any("proof did not verify" in violation.reason for violation in violations)


def test_unknown_link_topology_invalidates() -> None:
    fingerprint = good_fingerprint(
        link_topology=LinkTopology(
            thunderbolt_link_count=0, link_descriptors=(), source="unavailable"
        )
    )
    violations = valid_record(fingerprint=fingerprint).validate_envelope()
    assert any(
        violation.field == "fingerprint.link_topology" for violation in violations
    )


def test_empty_registered_env_invalidates() -> None:
    violations = valid_record(
        fingerprint=good_fingerprint(registered_env={})
    ).validate_envelope()
    assert any(
        violation.field == "fingerprint.registered_env" for violation in violations
    )


def test_uncertified_session_invalidates() -> None:
    violations = valid_record(canary_session_certified=False).validate_envelope()
    assert any(
        violation.field == "canary_session_certified" for violation in violations
    )


def test_prompt_size_dependent_without_token_truth_invalidates() -> None:
    violations = valid_record(
        prompt_size_dependent=True, token_ground_truth=None
    ).validate_envelope()
    assert any(violation.field == "token_ground_truth" for violation in violations)


def test_prompt_size_dependent_with_agreeing_token_truth_is_valid() -> None:
    record = valid_record(
        prompt_size_dependent=True,
        token_ground_truth=TokenGroundTruth(
            tokenizer_name="test",
            offline_token_count=2048,
            server_reported_token_count=2049,
            tolerance_tokens=2,
        ),
    )
    assert record.is_trusted


def test_shallow_only_result_cannot_claim_default_safe() -> None:
    shallow_only = (
        DepthMatrixCell(
            context_depth="shallow", thermal_state="warm", prompt_tokens=128
        ),
    )
    record = valid_record(depth_matrix_cells=shallow_only, claims_default_safe=True)
    violations = record.validate_envelope()
    assert any(violation.field == "claims_default_safe" for violation in violations)

    scoped = valid_record(depth_matrix_cells=shallow_only, claims_default_safe=False)
    assert scoped.is_trusted
    assert scoped.scope_label() == "test_probe [shallow/warm]"


def test_forged_runtime_marker_proof_invalidates() -> None:
    forged = RuntimeModeMarker(
        mode="tp_allreduce",
        detail="fabricated",
        observed_count=61,
        proof=ProofToken(
            domain=RUNTIME_MODE_DOMAIN, digest="0" * 64, session_id="elsewhere"
        ),
    )
    violations = valid_record(runtime_mode_markers=(forged,)).validate_envelope()
    assert any("unverifiable proof" in violation.reason for violation in violations)


def test_non_reproducing_value_invalidates() -> None:
    record = valid_record(
        value=aggregate_replicates(
            metric_name="decode_throughput",
            unit="tok/s",
            replicates=(20.0, 9.0, 33.0),
        )
    )
    violations = record.validate_envelope()
    assert any(violation.field == "value" for violation in violations)
