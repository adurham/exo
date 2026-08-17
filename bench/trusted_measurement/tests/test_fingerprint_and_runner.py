"""Fingerprint registry, link topology, proof tokens, content checks, A/B runner."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence

import pytest

from trusted_measurement.ab_runner import (
    PairedComparisonError,
    compare_arms,
    interleaved_schedule,
)
from trusted_measurement.content_check import build_needle_haystack, run_needle_check
from trusted_measurement.depth_matrix import (
    DepthMatrixCell,
    coverage_gaps,
    is_default_safe,
)
from trusted_measurement.fingerprint import (
    FINGERPRINT_ENV_REGISTRY,
    capture_fingerprint,
    capture_registered_env,
    probe_link_topology,
    registered_env_names,
)
from trusted_measurement.proof import mint_proof, verify_proof
from trusted_measurement.record import MeasurementRecord
from trusted_measurement.tests.builders import good_fingerprint, valid_record

# --------------------------------------------------------------- env registry


def test_registry_is_the_single_declaration_point() -> None:
    names = registered_env_names()
    assert len(names) == len(set(names)), "duplicate entries in the registry"
    assert len(names) == len(FINGERPRINT_ENV_REGISTRY)
    groups = {spec.group for spec in FINGERPRINT_ENV_REGISTRY}
    assert {"jaccl", "sharding", "dsv4", "runner"} <= groups
    for spec in FINGERPRINT_ENV_REGISTRY:
        assert spec.why_it_matters, f"{spec.name} has no rationale"


def test_known_high_risk_knobs_are_registered() -> None:
    names = set(registered_env_names())
    for required in (
        "MLX_JACCL_ACK_RETRANSMIT_US",
        "MLX_JACCL_SHARDING_MODE",
        "MLX_JACCL_RELIABLE_DATA",
        "EXO_PREFILL_STEP_SIZE",
        "EXO_DSV4_MTP",
    ):
        assert required in names


def test_unset_variables_are_recorded_as_none_not_omitted() -> None:
    snapshot = capture_registered_env({"MLX_JACCL_SHARDING_MODE": "tp"})
    assert set(snapshot) == set(registered_env_names())
    assert snapshot["MLX_JACCL_SHARDING_MODE"] == "tp"
    assert snapshot["MLX_JACCL_RELIABLE_DATA"] is None


def test_unregistered_variables_are_not_captured() -> None:
    snapshot = capture_registered_env({"TOTALLY_UNDECLARED_KNOB": "1"})
    assert "TOTALLY_UNDECLARED_KNOB" not in snapshot


# -------------------------------------------------------------- link topology


def test_link_topology_parses_system_profiler_output() -> None:
    payload: dict[str, object] = {
        "SPThunderboltDataType": [
            {
                "_name": "thunderbolt_bus",
                "_items": [
                    {"_name": "link 0", "receptacle_1_tag": {}},
                    {"_name": "link 1", "receptacle_1_tag": {}},
                    {"_name": "link 2", "receptacle_1_tag": {}},
                ],
            }
        ]
    }

    def runner(command: Sequence[str]) -> str:
        assert "system_profiler" in command[0]
        return json.dumps(payload)

    topology = probe_link_topology(runner)
    assert topology.source == "system_profiler"
    assert topology.thunderbolt_link_count == 3
    assert "3 TB links" in topology.summary()


def test_link_topology_failure_is_explicit_not_silent() -> None:
    def runner(command: Sequence[str]) -> str:
        _ = command
        raise OSError("system_profiler unavailable")

    topology = probe_link_topology(runner)
    assert topology.source == "unavailable"
    assert topology.thunderbolt_link_count == 0


def test_capture_fingerprint_uses_injected_runner() -> None:
    def runner(command: Sequence[str]) -> str:
        parts = list(command)
        if parts[0] == "git" and parts[-1] == "HEAD":
            return ("a" if parts[2] == "/exo" else "b") * 40 + "\n"
        if parts[0] == "git":
            return ""
        return json.dumps({"x": [{"_name": "link 0", "receptacle_1_tag": {}}]})

    fingerprint = capture_fingerprint(
        "/exo",
        "/mlx",
        runner=runner,
        environ={"MLX_JACCL_SHARDING_MODE": "tp"},
        hostname="fake-host",
    )
    assert fingerprint.exo_commit == "a" * 40
    assert fingerprint.mlx_commit == "b" * 40
    assert not fingerprint.exo_dirty
    assert fingerprint.link_topology.thunderbolt_link_count == 1
    assert fingerprint.hostname == "fake-host"


# --------------------------------------------------------------- proof tokens


def test_proof_verifies_only_for_its_own_payload() -> None:
    payload: dict[str, object] = {"a": 1}
    token = mint_proof("domain", payload)
    assert verify_proof(token, "domain", payload)
    assert not verify_proof(token, "domain", {"a": 2})
    assert not verify_proof(token, "other_domain", payload)


# -------------------------------------------------------------- content check


def test_needle_check_passes_when_needle_present() -> None:
    haystack = build_needle_haystack(filler_sentences=20, seed=5)
    result = run_needle_check(haystack, f"The code is {haystack.needle}.")
    assert result.passed
    assert result.proof_is_valid()


def test_needle_check_fails_when_needle_absent() -> None:
    haystack = build_needle_haystack(filler_sentences=20, seed=5)
    result = run_needle_check(haystack, "I could not find it.")
    assert not result.passed
    assert result.proof_is_valid()


def test_needle_haystack_is_deterministic_for_a_seed() -> None:
    first = build_needle_haystack(filler_sentences=10, seed=42)
    second = build_needle_haystack(filler_sentences=10, seed=42)
    assert first == second


# --------------------------------------------------------------- depth matrix


def test_coverage_gaps_and_default_safety() -> None:
    shallow = (
        DepthMatrixCell(
            context_depth="shallow", thermal_state="warm", prompt_tokens=128
        ),
    )
    assert len(coverage_gaps(shallow)) == 5
    assert not is_default_safe(shallow)


# ------------------------------------------------------------------ A/B runner


def test_interleaved_schedule_alternates() -> None:
    assert interleaved_schedule(3) == ("A", "B", "A", "B", "A", "B")


def _arm(value: str, **overrides: object) -> MeasurementRecord:
    environment = {
        "MLX_JACCL_SHARDING_MODE": "tp",
        "MLX_JACCL_ACK_RETRANSMIT_US": value,
    }
    fingerprint = good_fingerprint(registered_env=environment, **overrides)
    return valid_record(fingerprint=fingerprint)


def test_single_variable_interleaved_comparison_is_accepted() -> None:
    comparison = compare_arms(
        arm_a=_arm("2000"),
        arm_b=_arm("8000"),
        execution_schedule=interleaved_schedule(3),
    )
    assert comparison.toggled_variable == "MLX_JACCL_ACK_RETRANSMIT_US"
    assert comparison.arm_a_value == "2000"
    assert math.isclose(comparison.ratio_b_over_a, 1.0, rel_tol=1e-9)


def test_blocked_schedule_is_rejected() -> None:
    with pytest.raises(PairedComparisonError, match="interleaved"):
        _ = compare_arms(
            arm_a=_arm("2000"),
            arm_b=_arm("8000"),
            execution_schedule=("A", "A", "A", "B", "B", "B"),
        )


def test_comparison_across_different_builds_is_rejected() -> None:
    with pytest.raises(PairedComparisonError, match="different build"):
        _ = compare_arms(
            arm_a=_arm("2000"),
            arm_b=_arm("8000", exo_commit="c" * 40),
            execution_schedule=interleaved_schedule(2),
        )


def test_comparison_with_two_toggled_variables_is_rejected() -> None:
    arm_a = valid_record(
        fingerprint=good_fingerprint(
            registered_env={"MLX_JACCL_SHARDING_MODE": "tp", "EXO_NO_BATCH": "0"}
        )
    )
    arm_b = valid_record(
        fingerprint=good_fingerprint(
            registered_env={"MLX_JACCL_SHARDING_MODE": "pp", "EXO_NO_BATCH": "1"}
        )
    )
    with pytest.raises(PairedComparisonError, match="exactly one"):
        _ = compare_arms(
            arm_a=arm_a, arm_b=arm_b, execution_schedule=interleaved_schedule(2)
        )


def test_comparison_with_no_toggled_variable_is_rejected() -> None:
    with pytest.raises(PairedComparisonError, match="exactly one"):
        _ = compare_arms(
            arm_a=_arm("2000"),
            arm_b=_arm("2000"),
            execution_schedule=interleaved_schedule(2),
        )
