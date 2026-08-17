"""Phase 2 registry reconciliation against start_cluster.sh."""

from __future__ import annotations

import re
from pathlib import Path

from trusted_measurement.fingerprint import (
    FINGERPRINT_ENV_REGISTRY,
    capture_registered_env,
    registered_env_names,
)

_START_CLUSTER = Path(__file__).resolve().parents[4] / "start_cluster.sh"

# Vars start_cluster.sh sets that are deliberately EXCLUDED from the registry:
# pure logging / tracing / dump / debug-probe knobs that record what happened
# without changing what happens, plus non-measurement plumbing (peer addresses,
# namespaces, output paths). Listed explicitly so a future reviewer sees the
# judgement rather than a silent omission.
DELIBERATELY_EXCLUDED = frozenset(
    {
        "EXO_ARRAYSCACHE_DIAG",
        "EXO_CACHE_EVICT_TIMING_LOG",
        "EXO_CACHE_EVICT_TIMING_MS",
        "EXO_CMDBUF_RING_DIAG",
        "EXO_DECODE_PHASE_TRACE",
        "EXO_DECODE_PROBE",
        "EXO_DECODE_PROBE_EVERY",
        "EXO_DISCOVERY_PEERS",
        "EXO_DSV4_ACT_PROBE",
        "EXO_DSV4_ALLSUM_PROBE",
        "EXO_DSV4_ALLSUM_PROBE_LOG_EVERY",
        "EXO_DSV4_BATCHED_PREFILL_DEBUG",
        "EXO_DSV4_C2_TRACE",
        "EXO_DSV4_DEGEN_PROBE",
        "EXO_DSV4_DSPARK_DIR",
        "EXO_DSV4_FP32_COLL_LOG",
        "EXO_DSV4_HEAPCENSUS",
        "EXO_DSV4_LAYER_HASH_DUMP",
        "EXO_DSV4_LAYER_HASH_MAX_POS",
        "EXO_DSV4_LAYER_HASH_SUBOPS",
        "EXO_DSV4_MOE_ISOLATION_DUMP",
        "EXO_DSV4_MTP_C2_GATE_DEBUG",
        "EXO_DSV4_MTP_CYCLE_STATS",
        "EXO_DSV4_MTP_DUMP_TOPK",
        "EXO_DSV4_MTP_LOG_INTERVAL",
        "EXO_DSV4_MTP_NO_BROADCAST",
        "EXO_DSV4_MTP_PROFILE",
        "EXO_DSV4_MTP_REFCHECK",
        "EXO_DSV4_MTP_REFCHECK_ALL",
        "EXO_DSV4_MTP_REFCHECK_BATCH",
        "EXO_DSV4_MTP_TIE_REVERIFY_LOG",
        "EXO_DSV4_MTP_TRANSITION_TRACE",
        "EXO_DSV4_MTP_VERIFY_AUDIT",
        "EXO_DSV4_PSCACHE_DEBUG",
        "EXO_DSV4_RB_PROFILE",
        "EXO_DSV4_ROUTE_HIST",
        "EXO_DSV4_ROUTE_HIST_DECODE_ONLY",
        "EXO_DSV4_SECTION_TIME",
        "EXO_DSV4_SECTION_TIME_LOG_EVERY",
        "EXO_DSV4_SPEC_RB_LOG",
        "EXO_DSV4_SPEC_TRACE",
        "EXO_DSV4_TOPK_OVERLAP_LOG",
        "EXO_DSV4_TREE_ALPHA_PROBE",
        "EXO_DSV4_TREE_DEBUG",
        "EXO_DSV4_VERIFY_DIAG",
        "EXO_DSV4_VERIFY_TRACE",
        "EXO_DSV4_WEDGE_INJECT",
        "EXO_DSV4_WEDGE_TRACE",
        "EXO_DISABLE_METAL_TIMEOUT",
        "EXO_JIT_LOAD_TIMEOUT_SECONDS",
        "EXO_JIT_PLACEMENT_WAIT_SECONDS",
        "EXO_MEMORY_PROFILE_INTERVAL",
        "EXO_MEMORY_PROFILE_PATH",
        "EXO_MOE_EXPERT_HIST_DIAG",
        "EXO_MOE_GPUTRACE_DIAG",
        "EXO_MTP_DRIFT_DUMP",
        "EXO_PP_DEBUG",
        "EXO_PP_DSPARK_DRAFT_AHEAD_LOG",
        "EXO_PP_DSPARK_VERIFY_MARGIN_LOG",
        "EXO_PP_DSPARK_WIDTH_SWEEP",
        "EXO_PP_SPEC_FINISH_LOG",
        "EXO_PREFILL_GPU_TIME",
        "EXO_PREFILL_GPU_TRACE",
        "EXO_PREFIX_CACHE_DIAG",
        "EXO_RECLAIM_CHECK",
        "EXO_RECLAIM_DEADLINE_SECONDS",
        "EXO_RECLAIM_RESIDUAL_MAX_GB",
        "EXO_RUNNER_COREDUMP",
        "EXO_RUNNER_HANG_TIMEOUT_SECONDS",
        "EXO_SPEC_STATE_SPLIT_DIAG",
        "EXO_STALL_SAMPLER_SECONDS",
        "EXO_TRACEMALLOC_INTERVAL",
        "EXO_TRACEMALLOC_PATH",
        "EXO_TRACEMALLOC_TOP_N",
        "EXO_TRACING_ENABLED",
        "EXO_ZENOH_NAMESPACE",
        "JACCL_POLL_INSTRUMENT",
        "JACCL_POLL_INSTRUMENT_THRESHOLD_US",
        "JACCL_TRACE_CALLS",
        "JACCL_TRACE_HASH",
        "JACCL_TRACE_PROGRESS",
        "JACCL_TRACE_SPLIT",
        "JACCL_TRACE_STEP",
        "MLX_BUILD_PROBE",
        "MLX_BUILD_PROBE_LOG_EVERY",
        "MLX_DIAG_HOLD_WEDGE",
        "MLX_GPU_TIME",
        "MLX_GPU_TIME_LOG_EVERY",
        "MLX_LOG_ARRAY_DESC_COUNT_INTERVAL",
        "MLX_LOG_NEW_BUFFER_PATH",
        "MLX_OP_PROBE",
        "MLX_PER_TYPE_DUMP_INTERVAL",
        "MLX_PER_TYPE_TRACK",
        "MLX_SIGNAL_PROBE",
        "MTL_CAPTURE_ENABLED",
        "PYTHONFAULTHANDLER",
        "PYTHONUNBUFFERED",
        "QWEN36_MIN_P",
        "QWEN36_PRESENCE_PENALTY",
        "QWEN36_REPETITION_PENALTY",
        "QWEN36_TEMPERATURE",
        "QWEN36_TOP_K",
        "QWEN36_TOP_P",
    }
)


def _start_cluster_variables() -> set[str]:
    source = _START_CLUSTER.read_text(encoding="utf-8")
    names: set[str] = set(re.findall(r'^\s*:\s*"\$\{([A-Z0-9_]+)[:=]', source, re.M))
    names |= set(re.findall(r"^\s*export\s+([A-Z0-9_]+)=", source, re.M))
    for line in source.splitlines():
        if "EXO_ENV=" in line:
            names |= set(re.findall(r"\b([A-Z][A-Z0-9_]{3,})=", line))
    return {
        name
        for name in names
        if not name.startswith(("EXO_ENV", "LAUNCH", "RELAUNCH", "NODE_PEERS"))
    }


def test_start_cluster_script_is_where_we_think_it_is() -> None:
    assert _START_CLUSTER.is_file(), _START_CLUSTER


def test_every_start_cluster_variable_is_registered_or_explicitly_excluded() -> None:
    """The reconciliation gate: no silent third category.

    A var start_cluster.sh sets must either be in the fingerprint registry or
    named in DELIBERATELY_EXCLUDED with the reasoning documented above it. A
    new knob added to the launcher fails this test until someone judges it.
    """
    registered = set(registered_env_names())
    unaccounted = sorted(
        _start_cluster_variables() - registered - DELIBERATELY_EXCLUDED
    )
    assert not unaccounted, (
        "start_cluster.sh sets variables the harness neither captures nor "
        f"explicitly excludes: {unaccounted}"
    )


def test_exclusion_list_does_not_overlap_the_registry() -> None:
    assert not (DELIBERATELY_EXCLUDED & set(registered_env_names()))


def test_phase2_measurement_relevant_variables_are_captured() -> None:
    """Spot-check the highest-consequence Phase 2 additions."""
    snapshot = capture_registered_env(
        {
            "DSV4_SHARDING": "Tensor",
            "DSV4_MODEL_ID": "deepseek-ai/DeepSeek-V4-Flash-0731",
            "EXO_DSV4_INDEXER_WINDOW": "8192",
            "EXO_DSV4_VERIFY_ROWSEQ": "1",
            "EXO_DSV4_MOE_PARTS_ROWSEQ": "shared",
            "MLX_STEEL_BATCH_INVARIANT": "1",
            "EXO_MAX_ACTIVE_TASKS": "5",
            "EXO_KV_CACHE_BITS": "0",
            "DSV4_REPETITION_PENALTY": "1.05",
            "EXO_TB_MTU": "65520",
            "LOG_LEVEL": "INFO",
        }
    )
    assert snapshot["DSV4_SHARDING"] == "Tensor"
    assert snapshot["DSV4_MODEL_ID"] == "deepseek-ai/DeepSeek-V4-Flash-0731"
    assert snapshot["EXO_DSV4_INDEXER_WINDOW"] == "8192"
    assert snapshot["EXO_DSV4_VERIFY_ROWSEQ"] == "1"
    assert snapshot["EXO_DSV4_MOE_PARTS_ROWSEQ"] == "shared"
    assert snapshot["MLX_STEEL_BATCH_INVARIANT"] == "1"
    assert snapshot["EXO_MAX_ACTIVE_TASKS"] == "5"
    assert snapshot["EXO_KV_CACHE_BITS"] == "0"
    assert snapshot["DSV4_REPETITION_PENALTY"] == "1.05"
    assert snapshot["EXO_TB_MTU"] == "65520"
    assert snapshot["LOG_LEVEL"] == "INFO"


def test_pure_logging_knobs_stay_out_of_the_registry() -> None:
    registered = set(registered_env_names())
    for noisy in (
        "EXO_DSV4_VERIFY_TRACE",
        "JACCL_TRACE_STEP",
        "MLX_GPU_TIME",
        "EXO_TRACEMALLOC_PATH",
        "PYTHONUNBUFFERED",
    ):
        assert noisy not in registered


def test_phase2_groups_are_declared_with_rationale() -> None:
    groups = {spec.group for spec in FINGERPRINT_ENV_REGISTRY}
    assert {"prefill", "decode_spec", "kernel", "sampling", "model"} <= groups
    for spec in FINGERPRINT_ENV_REGISTRY:
        assert spec.why_it_matters.strip(), spec.name


def test_registry_has_no_duplicates_after_phase2() -> None:
    names = [spec.name for spec in FINGERPRINT_ENV_REGISTRY]
    assert len(names) == len(set(names)), sorted(
        {name for name in names if names.count(name) > 1}
    )
