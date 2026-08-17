"""Shared fixtures/builders for the trusted-measurement core regression suite."""

from __future__ import annotations

from trusted_measurement.content_check import (
    ContentCheckResult,
    build_needle_haystack,
    run_needle_check,
)
from trusted_measurement.depth_matrix import ALL_DEPTH_CELLS, DepthMatrixCell
from trusted_measurement.fingerprint import Fingerprint, LinkTopology
from trusted_measurement.record import MeasurementRecord
from trusted_measurement.replication import ReplicatedValue, aggregate_replicates
from trusted_measurement.runtime_mode import RuntimeModeMarker, RuntimeModeRecorder

__all__ = [
    "full_matrix_cells",
    "good_content_check",
    "good_fingerprint",
    "good_markers",
    "good_value",
    "valid_record",
]


def good_fingerprint(**overrides: object) -> Fingerprint:
    """Construct (never ``model_copy``) so field constraints actually run."""
    fields: dict[str, object] = dict(
        exo_commit="a" * 40,
        mlx_commit="b" * 40,
        exo_dirty=False,
        mlx_dirty=False,
        registered_env={
            "MLX_JACCL_SHARDING_MODE": "tp",
            "MLX_JACCL_ACK_RETRANSMIT_US": "2000",
        },
        link_topology=LinkTopology(
            thunderbolt_link_count=6,
            link_descriptors=("link-0", "link-1"),
            source="system_profiler",
        ),
        hostname="test-host",
    )
    fields.update(overrides)
    return Fingerprint(**fields)  # pyright: ignore[reportArgumentType]


def good_content_check() -> ContentCheckResult:
    haystack = build_needle_haystack(filler_sentences=5, seed=1)
    return run_needle_check(haystack, f"The secret access code is {haystack.needle}")


def good_markers() -> tuple[RuntimeModeMarker, ...]:
    recorder = RuntimeModeRecorder()
    recorder.emit("tp_allreduce", detail="2 ops/layer", count=122)
    return recorder.markers()


def good_value() -> ReplicatedValue:
    return aggregate_replicates(
        metric_name="decode_throughput",
        unit="tok/s",
        replicates=(20.0, 20.4, 19.8),
    )


def full_matrix_cells() -> tuple[DepthMatrixCell, ...]:
    return tuple(
        DepthMatrixCell(context_depth=depth, thermal_state=thermal, prompt_tokens=2048)
        for depth, thermal in ALL_DEPTH_CELLS
    )


def valid_record(**overrides: object) -> MeasurementRecord:
    """A fully valid record; ``overrides`` break exactly one thing at a time."""
    fields: dict[str, object] = dict(
        probe_name="test_probe",
        value=good_value(),
        content_check=good_content_check(),
        fingerprint=good_fingerprint(),
        runtime_mode_markers=good_markers(),
        required_runtime_modes=("tp_allreduce",),
        depth_matrix_cells=full_matrix_cells(),
        claims_default_safe=True,
        token_ground_truth=None,
        prompt_size_dependent=False,
        canary_session_certified=True,
    )
    fields.update(overrides)
    return MeasurementRecord(**fields)  # pyright: ignore[reportArgumentType]
