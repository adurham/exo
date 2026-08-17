"""Trusted-measurement core for the exo cluster campaign (Phase 1).

A throughput or latency number is only trustworthy when it arrives with a
complete validity envelope. This package makes that structural: measurements
are :class:`~trusted_measurement.record.MeasurementRecord` objects that refuse
to yield a value unless every envelope component is present and verified.

Domain probes (jaccl/soft-RC transport, prefill, decode) are built on top of
this core in later phases; nothing here talks to the cluster.
"""

from __future__ import annotations

from trusted_measurement.ab_runner import (
    ArmLabel,
    PairedComparison,
    PairedComparisonError,
    compare_arms,
    interleaved_schedule,
)
from trusted_measurement.canary import (
    CanaryReport,
    CanaryScenarioResult,
    HarnessNotCertifiedError,
    RecordJudge,
    default_judge,
    run_liveness_canary,
)
from trusted_measurement.content_check import (
    ContentCheckResult,
    NeedleHaystack,
    build_needle_haystack,
    run_needle_check,
)
from trusted_measurement.depth_matrix import (
    ALL_DEPTH_CELLS,
    ContextDepth,
    DepthMatrixCell,
    ThermalState,
    coverage_gaps,
    is_default_safe,
)
from trusted_measurement.fingerprint import (
    FINGERPRINT_ENV_REGISTRY,
    EnvVarSpec,
    Fingerprint,
    LinkTopology,
    capture_fingerprint,
    capture_registered_env,
    probe_link_topology,
    registered_env_names,
)
from trusted_measurement.proof import SESSION_ID, ProofToken, mint_proof, verify_proof
from trusted_measurement.record import (
    EnvelopeViolation,
    MeasurementRecord,
    UntrustedMeasurementError,
)
from trusted_measurement.replication import (
    DEFAULT_DISPERSION_THRESHOLD,
    DEFAULT_MINIMUM_REPLICATES,
    ReplicatedValue,
    ReplicationVerdict,
    aggregate_replicates,
)
from trusted_measurement.runtime_mode import RuntimeModeMarker, RuntimeModeRecorder
from trusted_measurement.token_truth import (
    TokenCountMismatchError,
    TokenGroundTruth,
    Tokenizer,
    cross_check_token_count,
)

__all__ = [
    "ALL_DEPTH_CELLS",
    "DEFAULT_DISPERSION_THRESHOLD",
    "DEFAULT_MINIMUM_REPLICATES",
    "FINGERPRINT_ENV_REGISTRY",
    "SESSION_ID",
    "ArmLabel",
    "CanaryReport",
    "CanaryScenarioResult",
    "ContentCheckResult",
    "ContextDepth",
    "DepthMatrixCell",
    "EnvVarSpec",
    "EnvelopeViolation",
    "Fingerprint",
    "HarnessNotCertifiedError",
    "LinkTopology",
    "MeasurementRecord",
    "NeedleHaystack",
    "PairedComparison",
    "PairedComparisonError",
    "ProofToken",
    "RecordJudge",
    "ReplicatedValue",
    "ReplicationVerdict",
    "RuntimeModeMarker",
    "RuntimeModeRecorder",
    "ThermalState",
    "TokenCountMismatchError",
    "TokenGroundTruth",
    "Tokenizer",
    "UntrustedMeasurementError",
    "aggregate_replicates",
    "build_needle_haystack",
    "capture_fingerprint",
    "capture_registered_env",
    "compare_arms",
    "coverage_gaps",
    "cross_check_token_count",
    "default_judge",
    "interleaved_schedule",
    "is_default_safe",
    "mint_proof",
    "probe_link_topology",
    "registered_env_names",
    "run_liveness_canary",
    "run_needle_check",
    "verify_proof",
]
