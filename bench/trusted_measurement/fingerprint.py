"""Centralised fingerprint registry (design point 2).

Everything that can silently change a throughput number between two runs must
be declared in ONE place, here. Probes never read ``os.environ`` directly for
config that matters; they call :func:`capture_fingerprint`, which walks the
registry below. If a new jaccl/DSv4/sharding knob is added to the system and
not added to :data:`FINGERPRINT_ENV_REGISTRY`, it is invisible to the harness —
so the registry is the single review point for that class of drift.

The registry also captures **link topology**, which is a live blind spot: the
number of Thunderbolt link descriptors on this hardware recently went from 3 to
6, and every benchmark taken before that change was implicitly taken on a
different machine.
"""

from __future__ import annotations

import json
import os
import subprocess  # noqa: S404 - git/system_profiler introspection is the point
from collections.abc import Callable, Mapping, Sequence
from typing import Final, cast, final

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "FINGERPRINT_ENV_REGISTRY",
    "CommandRunner",
    "EnvVarSpec",
    "Fingerprint",
    "LinkTopology",
    "capture_fingerprint",
    "capture_registered_env",
    "probe_link_topology",
    "registered_env_names",
]

CommandRunner = Callable[[Sequence[str]], str]
"""Runs a command and returns stdout. Injectable so tests never shell out."""


@final
class EnvVarSpec(BaseModel):
    """One declared environment variable that can move a measurement."""

    model_config = ConfigDict(frozen=True, strict=True)

    name: str
    group: str
    why_it_matters: str


def _spec(group: str, name: str, why: str) -> EnvVarSpec:
    return EnvVarSpec(name=name, group=group, why_it_matters=why)


FINGERPRINT_ENV_REGISTRY: Final[tuple[EnvVarSpec, ...]] = (
    # --- jaccl / soft-RC transport -----------------------------------------
    _spec("jaccl", "MLX_JACCL_RELIABLE_DATA", "soft-RC reliability on/off"),
    _spec("jaccl", "MLX_JACCL_RELIABLE_INFLIGHT", "in-flight window depth"),
    _spec("jaccl", "MLX_JACCL_RELIABLE_MAX_SZ", "max reliable chunk size"),
    _spec("jaccl", "MLX_JACCL_RELIABLE_IDLE_US", "idle poll backoff"),
    _spec("jaccl", "MLX_JACCL_RELIABLE_OPTIMISTIC", "optimistic-send mode"),
    _spec("jaccl", "MLX_JACCL_RELIABLE_SMALL_CHUNKS", "small-chunk split policy"),
    _spec(
        "jaccl",
        "MLX_JACCL_ACK_RETRANSMIT_US",
        "retransmit timer - depth-sensitive, has broken generation when tuned "
        "at one context depth and shipped as a global default",
    ),
    _spec("jaccl", "MLX_JACCL_ACK_RETRANSMIT_MAX", "retransmit attempt ceiling"),
    _spec("jaccl", "MLX_JACCL_ACK_SYNC_PRE", "pre-barrier ack sync"),
    _spec("jaccl", "MLX_JACCL_P2P_DRAIN_QUIET_US", "drain quiet period"),
    _spec("jaccl", "MLX_JACCL_STALL_TIMEOUT_US", "stall detection threshold"),
    _spec("jaccl", "MLX_JACCL_CONFIRMED_BARRIER", "confirmed-barrier mode"),
    _spec("jaccl", "MLX_JACCL_CONFIRMED_BARRIER_PRE", "pre-op confirmed barrier"),
    _spec("jaccl", "MLX_JACCL_CONFIRMED_BARRIER_POST", "post-op confirmed barrier"),
    _spec("jaccl", "MLX_JACCL_DATA_RECV_POOL", "receive pool sizing"),
    _spec("jaccl", "MLX_JACCL_RECV_RETRY_DEADLINE_SECS", "recv retry deadline"),
    _spec("jaccl", "MLX_JACCL_P2P_RECV_RETRY_DEADLINE_SECS", "p2p recv deadline"),
    _spec("jaccl", "MLX_JACCL_P2P_RETRY_STALL_TIMEOUT_SECS", "p2p stall timeout"),
    _spec("jaccl", "MLX_JACCL_COORD_RECV_TIMEOUT_SECS", "coordinator recv timeout"),
    _spec("jaccl", "MLX_JACCL_RECONNECT_FRESH", "fresh reconnect behaviour"),
    _spec("jaccl", "MLX_JACCL_RING", "ring vs mesh collective topology"),
    _spec("jaccl", "MLX_IBV_DEVICES", "which RDMA devices are bound"),
    _spec("jaccl", "MLX_HOSTFILE", "rank-to-host mapping"),
    # --- sharding ----------------------------------------------------------
    _spec("sharding", "MLX_JACCL_SHARDING_MODE", "TP vs PP - the headline axis"),
    _spec("sharding", "EXO_PP_LAYER_SPLIT", "pipeline layer split point"),
    _spec("sharding", "EXO_DSV4_SEQ_SPLIT", "sequence-dimension split"),
    _spec("sharding", "EXO_COMPUTE_DTYPE", "compute precision"),
    # --- DSv4-Flash model path ---------------------------------------------
    _spec("dsv4", "EXO_DSV4_BATCHED_PREFILL", "batched prefill path on/off"),
    _spec("dsv4", "EXO_PREFILL_STEP_SIZE", "prefill chunk size"),
    _spec("dsv4", "EXO_BATCHED_PREFILL_RENDEZVOUS_MS", "prefill rendezvous window"),
    _spec("dsv4", "EXO_DSV4_FUSED_MOE", "fused MoE kernel"),
    _spec("dsv4", "EXO_DSV4_MOE_FUSED_GATE_UP", "fused gate/up projection"),
    _spec("dsv4", "EXO_DSV4_COMPILE_LAYER", "layer-level mx.compile"),
    _spec("dsv4", "EXO_DSV4_COMPILE_FFN", "FFN-level mx.compile"),
    _spec("dsv4", "EXO_DSV4_INDEX_TOPK", "sparse indexer top-k"),
    _spec("dsv4", "EXO_DSV4_FENCE_EVERY_N_LAYERS", "fence cadence"),
    _spec("dsv4", "EXO_DSV4_FENCE_ASYNC_C2", "async C2 fencing"),
    _spec("dsv4", "EXO_DSV4_MTP", "multi-token prediction on/off"),
    _spec("dsv4", "EXO_DSV4_MTP_MAX_CTX", "MTP context ceiling - depth-sensitive"),
    _spec("dsv4", "EXO_DSV4_MTP_C2_MAX_CTX", "MTP C2 context ceiling"),
    _spec("dsv4", "EXO_DSV4_DSPARK", "DSpark speculative mechanism"),
    _spec("dsv4", "EXO_SPECULATIVE", "generic speculative decode on/off"),
    _spec("dsv4", "EXO_SPECULATIVE_GAMMA", "speculative draft length"),
    _spec("dsv4", "EXO_TURBOQUANT", "turboquant path"),
    _spec("dsv4", "EXO_TURBOQUANT_BITS", "turboquant bit width"),
    # --- runner / scheduling ------------------------------------------------
    _spec("runner", "EXO_NO_BATCH", "disables request batching"),
    _spec("runner", "EXO_PP_BATCHED_DECODE", "batched decode path"),
    _spec("runner", "EXO_RUNNER_QOS", "process QoS class - thermal/perf impact"),
    _spec("runner", "EXO_RECLAIM_ON_IDLE", "idle memory reclaim"),
    _spec("runner", "EXO_PREFIX_CACHE_TRACE", "prefix-cache instrumentation"),
    _spec("runner", "EXO_PROFILER", "profiler attach - perturbs timing"),
    _spec("runner", "EXO_PROFILER_LEVEL", "profiler verbosity"),
    # === Phase 2 reconciliation against start_cluster.sh =====================
    # Everything below was set (or explicitly plumbed into EXO_ENV) by
    # start_cluster.sh but invisible to the harness. Selection rule: a var is
    # registered when changing it can move a throughput/latency number or the
    # tokens produced. Pure logging/tracing/dump/diagnostic knobs are
    # deliberately NOT registered (see excluded list in the Phase 2 notes),
    # except where the instrumentation itself perturbs timing.
    # --- jaccl / transport / interconnect -----------------------------------
    _spec("jaccl", "EXO_JACCL_ACK_PRE_FASTSKIP", "skips the pre-op ack barrier"),
    _spec("jaccl", "JACCL_SPLIT_FRESH_CTX", "fresh context per comm split"),
    _spec("jaccl", "JACCL_SPLIT_PARENT_STREAM", "split inherits the parent stream"),
    _spec("jaccl", "EXO_TB_MTU", "Thunderbolt MTU - directly sets wire framing"),
    _spec("jaccl", "IBV_FORK_SAFE", "RDMA fork-safety mode"),
    # --- sharding / topology ------------------------------------------------
    _spec("sharding", "DSV4_SHARDING", "Tensor vs Pipeline for the DSv4 instance"),
    _spec("sharding", "EXO_PP_METAFRAME", "metadata-framed PP transport path"),
    _spec("sharding", "EXO_PP_NO_COORD_COLLECTIVE", "disables coord collectives"),
    _spec("sharding", "EXO_DSV4_SEQSPLIT_BALANCED", "balanced causal seq-split"),
    _spec("sharding", "EXO_DSV4_SEQ_SPLIT_MIN_L", "seq-split arming threshold"),
    # --- model identity / co-tenancy ---------------------------------------
    _spec(
        "model",
        "DSV4_MODEL_ID",
        "preview vs production checkpoint - different "
        "layer splits, so different cross-rank parity",
    ),
    _spec("model", "DSV4_ENABLED", "whether DSv4 is resident at all"),
    _spec("model", "QWEN36_ENABLED", "aux model co-tenancy - memory/GPU contention"),
    _spec("model", "QWEN36_MODEL_ID", "which aux checkpoint co-hosts"),
    # --- prefill ------------------------------------------------------------
    _spec("prefill", "DSV4_PREFILL_STEP_SIZE", "per-instance prefill chunk override"),
    _spec("prefill", "QWEN36_PREFILL_STEP_SIZE", "aux instance prefill chunk"),
    _spec("prefill", "EXO_PREFILL_STEP_SIZE_HIGH_CTX", "adaptive high-ctx chunk"),
    _spec("prefill", "EXO_PREFILL_STEP_SIZE_CROSSOVER", "adaptive chunk crossover"),
    _spec("prefill", "EXO_PREFILL_CLEAR_CACHE_INTERVAL", "allocator clear cadence"),
    _spec("prefill", "EXO_PREFILL_EVAL_INTERVAL", "prefill eval cadence"),
    _spec("prefill", "EXO_PREFILL_ASYNC_EVAL", "async prefill eval path"),
    _spec(
        "prefill",
        "EXO_DSV4_PREFILL_ARGPARTITION",
        "argpartition indexer top-k - the reason chunk 256 beats 128",
    ),
    # --- decode / speculation ----------------------------------------------
    _spec("decode_spec", "EXO_DSV4_MTP_EAGLE_K", "MTP soft-emb mixture width"),
    _spec("decode_spec", "EXO_DSV4_MTP_EAGLE_T", "MTP mixture temperature"),
    _spec("decode_spec", "EXO_DSV4_MTP_DEDICATED", "dedicated MTP head path"),
    _spec("decode_spec", "EXO_DSV4_MTP_MIN_P", "MTP draft min-p"),
    _spec("decode_spec", "EXO_DSV4_MTP_ACCEPT_LOGPROBS", "logprob accept rule"),
    _spec(
        "decode_spec",
        "EXO_DSV4_MTP_TIEBREAK_FIX",
        "tie-break fix - corrupts output on the affine checkpoint",
    ),
    _spec("decode_spec", "EXO_DSV4_MTP_TIEBREAK_EPS", "tie-break epsilon"),
    _spec("decode_spec", "EXO_DSV4_MTP_TIE_REVERIFY", "trim+refeed re-verify"),
    _spec("decode_spec", "EXO_DSV4_MTP_TIE_REVERIFY_EPS", "re-verify epsilon"),
    _spec("decode_spec", "EXO_DSV4_VERIFY_ROWSEQ", "row-sequential verify forward"),
    _spec("decode_spec", "EXO_DSV4_VERIFY_ROWSEQ_MIN_CTX", "row-seq arming depth"),
    _spec("decode_spec", "EXO_DSV4_VERIFY_ROWSEQ_MAX_L", "row-seq block ceiling"),
    _spec("decode_spec", "EXO_DSV4_VERIFY_ROWSEQ_VEC", "vectorised row-seq"),
    _spec("decode_spec", "EXO_DSV4_VERIFY_ROWSEQ_VEC_ROWSDPA", "row-SDPA variant"),
    _spec("decode_spec", "EXO_DSV4_ROWSEQ_ROWMASK", "per-row decode masks"),
    _spec("decode_spec", "EXO_DSV4_ROWSEQ_FULLBLOCK", "full-block row-seq"),
    _spec(
        "decode_spec",
        "EXO_DSV4_ROWSEQ_FULLBLOCK_MOE",
        "per-row whole MoE - correct but caps DSpark at sequential-decode parity",
    ),
    _spec("decode_spec", "EXO_DSV4_MOE_PARTS_ROWSEQ", "which MoE parts go per-row"),
    _spec("decode_spec", "EXO_DSV4_SPEC_STATE_RESTORE", "unified spec-state rollback"),
    _spec("decode_spec", "EXO_DSV4_SPEC_CACHE_ROLLBACK", "spec cache rollback"),
    _spec("decode_spec", "EXO_DSV4_SPEC_CACHE_ROLLBACK_C2", "c=2 spec rollback"),
    _spec("decode_spec", "EXO_DSV4_POOL_SNAPSHOT_BATCH", "batched pool snapshot"),
    _spec("decode_spec", "EXO_DSV4_POOL_RESTORE_AFTER_TRIM", "restore-after-trim fix"),
    _spec("decode_spec", "EXO_DSV4_POOL_DEFER_COPY_MAX_BYTES", "deferred pool copy"),
    _spec("decode_spec", "EXO_DSV4_DSPARK_NATIVE", "DSpark native draft head"),
    _spec("decode_spec", "EXO_DSV4_DSPARK_CONF_TAU", "DSpark confidence threshold"),
    _spec("decode_spec", "EXO_DSV4_BS_MIN_ACCEPT", "minimum accept length"),
    _spec("decode_spec", "EXO_DSV4_TREE_DRAFT", "tree drafting on/off"),
    _spec("decode_spec", "EXO_DSV4_TREE_K", "tree draft width"),
    _spec("decode_spec", "EXO_DSV4_TREE_GREEDY", "greedy tree selection"),
    _spec("decode_spec", "EXO_QWEN_SPECULATIVE_GAMMA", "aux-model draft length"),
    _spec("decode_spec", "EXO_PP_MTP_CHAIN_K", "chained-MTP depth"),
    _spec(
        "decode_spec",
        "EXO_PP_DRAFT_MODEL",
        "classic draft model path - non-empty used to silently arm PP speculation",
    ),
    _spec("decode_spec", "EXO_DRAFT_KV_WINDOW", "draft-model KV window"),
    _spec("decode_spec", "EXO_PP_DSPARK_VERIFY_WIDTH", "verify-width truncation"),
    _spec("decode_spec", "EXO_PP_DSPARK_DRAFT_AHEAD", "draft-ahead tagging"),
    _spec(
        "decode_spec",
        "EXO_PP_DSPARK_DRAFT_AHEAD_EXECUTE",
        "speculative forward against live KV - confirmed to cause 15-70s stalls",
    ),
    _spec("decode_spec", "EXO_PP_DSPARK_DRAFT_AHEAD_YIELD", "draft-ahead yield"),
    _spec("decode_spec", "EXO_PP_DSPARK_BATCHED_SNAPSHOT_EVAL", "batched snapshot"),
    # --- model numerics / kernels ------------------------------------------
    _spec(
        "kernel",
        "EXO_DSV4_INDEXER_WINDOW",
        "indexer sliding-window - unbounded halves decode rate by ~20K tokens",
    ),
    _spec("kernel", "EXO_DSV4_INDEXER_WINDOW_LATE", "late-phase indexer window"),
    _spec("kernel", "EXO_DSV4_INDEXER_PBLOCK", "indexer pooled block size"),
    _spec("kernel", "EXO_DSV4_SPARSE_SDPA_TILE", "sparse SDPA tile size"),
    _spec("kernel", "EXO_DSV4_FUSED_SOFTMAX", "fused softmax epilogue"),
    _spec("kernel", "EXO_DSV4_TOPK_FUSED", "fused top-k"),
    _spec("kernel", "EXO_DSV4_SINGLE_GATHER", "single-gather MoE path"),
    _spec("kernel", "EXO_DSV4_ATTN_ALLSUM", "attention all-sum placement"),
    _spec("kernel", "EXO_DSV4_FENCE_ASYNC", "async fencing"),
    _spec("kernel", "EXO_DSV4_FP32_ACT", "fp32 activations"),
    _spec("kernel", "EXO_DSV4_LMHEAD_LASTROW", "last-row-only LM head"),
    _spec("kernel", "EXO_DSV4_BATCH_INVARIANT_MM", "batch-invariant matmul"),
    _spec("kernel", "EXO_DSV4_BATCH_INVARIANT_MM_MAX_M", "batch-invariant M ceiling"),
    _spec("kernel", "EXO_DSV4_ARGPARTITION_MIN_P", "argpartition arming threshold"),
    _spec("kernel", "MLX_SDPA_BLOCKS", "2-pass SDPA block count"),
    _spec("kernel", "MLX_SDPA_D512_FUSED", "fused d=512 SDPA"),
    _spec("kernel", "MLX_SDPA_D512_BQ16", "d=512 SDPA query blocking"),
    _spec("kernel", "MLX_GEMV_BATCH_INVARIANT", "GEMV batch invariance"),
    _spec(
        "kernel",
        "MLX_STEEL_BATCH_INVARIANT",
        "Steel batch invariance - ~5% c=1 "
        "decode cost, required for bitexact spec at c>=2",
    ),
    _spec("kernel", "MLX_GATHER_QMV_RHS", "gathered QMV RHS path"),
    _spec("kernel", "MLX_GATHER_QMV_RHS_RPS", "gathered QMV rows per split"),
    _spec("kernel", "MLX_GATHER_QMV_RHS_TILE", "gathered QMV tile"),
    _spec("kernel", "MLX_GQMM_RHS_LHS_BK", "gathered QMM tile K"),
    _spec("kernel", "MLX_GQMM_RHS_LHS_BM", "gathered QMM tile M"),
    _spec("kernel", "MLX_GQMM_RHS_LHS_BN", "gathered QMM tile N"),
    _spec("kernel", "MLX_GQMM_RHS_LHS_WM", "gathered QMM warp M"),
    _spec("kernel", "MLX_GQMM_RHS_LHS_WN", "gathered QMM warp N"),
    _spec("kernel", "MLX_LM_SDPA_ROWSPLIT", "mlx-lm SDPA row split"),
    _spec("kernel", "MLX_DISABLE_COMPILE", "disables mx.compile globally"),
    # --- memory / scheduling / thermal -------------------------------------
    _spec("runner", "EXO_MAX_ACTIVE_TASKS", "GPU command-buffer queue depth"),
    _spec("runner", "EXO_MAX_CONCURRENT_REQUESTS", "server-side concurrency cap"),
    _spec("runner", "EXO_FAST_SYNCH", "fast synchronisation path"),
    _spec("runner", "EXO_LAYER_EVAL_INTERVAL", "per-layer eval cadence"),
    _spec("runner", "EXO_MLX_CLEAR_CACHE_INTERVAL", "Metal cache clear cadence"),
    _spec("runner", "EXO_GC_COLLECT_INTERVAL", "forced GC cadence"),
    _spec("runner", "EXO_MALLOC_RELIEF_INTERVAL", "malloc relief cadence"),
    _spec("runner", "EXO_LEAF_SNAPSHOT_RETENTION", "prefix-cache snapshot retention"),
    _spec("runner", "EXO_HC_USE_OPS", "health-check op path"),
    _spec("runner", "EXO_KV_CACHE_BITS", "KV quantisation - quality and bandwidth"),
    _spec("runner", "DSV4_KV_CACHE_BITS", "per-instance KV quantisation"),
    _spec("runner", "QWEN36_KV_CACHE_BITS", "aux instance KV quantisation"),
    _spec("runner", "DSV4_MAX_KV_TOKENS", "KV budget - bounds reachable depth"),
    _spec("runner", "DSV4_MAX_PREFIX_SESSIONS", "prefix-cache session count"),
    _spec("runner", "DSV4_MAX_PREFIX_BYTES", "prefix-cache byte budget"),
    _spec("runner", "QWEN36_MAX_KV_TOKENS", "aux KV budget - steals headroom"),
    _spec("runner", "QWEN36_MAX_PREFIX_SESSIONS", "aux prefix sessions"),
    _spec("runner", "QWEN36_MAX_PREFIX_BYTES", "aux prefix byte budget"),
    _spec("runner", "EXO_JIT_ENABLED", "JIT load/unload can move a model mid-run"),
    _spec("runner", "EXO_JIT_IDLE_UNLOAD_SECONDS", "idle unload window"),
    _spec("runner", "EXO_JIT_MEMORY_RESERVE_GB", "JIT memory reserve"),
    _spec("runner", "MLX_MAX_OPS_PER_BUFFER", "command-buffer op bound"),
    _spec("runner", "MLX_MAX_MB_PER_BUFFER", "command-buffer byte bound"),
    _spec("runner", "MLX_LM_EAGER_EVAL_CACHES", "eager cache eval"),
    _spec("runner", "MLX_LM_SYNC_AFTER_STEP", "synchronise after each step"),
    _spec("runner", "MLX_LM_CLEAR_COMPILE_CACHE_INTERVAL", "compile cache cadence"),
    _spec("runner", "MLX_EAGER_COMMIT_BEFORE_CPU_COLLECTIVE", "eager commit"),
    _spec("runner", "MLX_EVENT_WAIT_SPIN", "event-wait spin vs block"),
    _spec("runner", "MLX_EVENT_WAIT_POLL_US", "event-wait poll interval"),
    _spec("runner", "MLX_EVENT_WAIT_TIMEOUT_MS", "event-wait timeout"),
    _spec("runner", "MLX_STREAM_QOS", "Metal stream QoS class"),
    _spec("runner", "MLX_STREAM_RT", "realtime stream scheduling"),
    _spec("runner", "MLX_STREAM_RT_PERIOD_US", "realtime period"),
    _spec("runner", "MLX_STREAM_RT_COMPUTATION_US", "realtime computation budget"),
    _spec("runner", "MLX_STREAM_RT_CONSTRAINT_US", "realtime constraint"),
    _spec("runner", "MTL_DISABLE_TIMEOUT", "Metal watchdog disable"),
    _spec("runner", "MTL_COMMAND_BUFFER_TIMEOUT", "Metal command-buffer timeout"),
    _spec("runner", "AGX_RELAX_CDM_CTXSTORE_TIMEOUT", "GPU context-store timeout"),
    _spec("runner", "LOG_LEVEL", "DEBUG measurably slows serving"),
    _spec(
        "runner",
        "EXO_PROFILER_SYNC_SPANS",
        "synchronous profiler spans perturb the very timings being measured",
    ),
    # --- sampling (decides how many/which tokens are produced) --------------
    _spec(
        "sampling",
        "DSV4_TEMPERATURE",
        "temperature changes the token trajectory "
        "and therefore speculative accept rates",
    ),
    _spec("sampling", "DSV4_TOP_P", "nucleus cutoff"),
    _spec("sampling", "DSV4_TOP_K", "top-k cutoff"),
    _spec("sampling", "DSV4_MIN_P", "min-p cutoff"),
    _spec("sampling", "DSV4_PRESENCE_PENALTY", "presence penalty"),
    _spec(
        "sampling",
        "DSV4_REPETITION_PENALTY",
        "repetition penalty - changes "
        "generation length, which changes tok/s denominators",
    ),
    _spec("sampling", "EXO_DEFAULT_TEMPERATURE", "server-wide default temperature"),
    _spec("sampling", "EXO_DEFAULT_TOP_P", "server-wide default top-p"),
    _spec("sampling", "EXO_DEFAULT_TOP_K", "server-wide default top-k"),
    _spec("sampling", "EXO_DEFAULT_MIN_P", "server-wide default min-p"),
)

_REGISTRY_BY_NAME: Final[Mapping[str, EnvVarSpec]] = {
    spec.name: spec for spec in FINGERPRINT_ENV_REGISTRY
}


def registered_env_names() -> tuple[str, ...]:
    """Every environment variable the harness considers measurement-relevant."""
    return tuple(sorted(_REGISTRY_BY_NAME))


def capture_registered_env(
    environ: Mapping[str, str] | None = None,
) -> dict[str, str | None]:
    """Snapshot every registered variable, recording ``None`` when unset.

    Unset is recorded explicitly rather than omitted: "the knob was absent" and
    "the knob was never looked at" must not be the same record.
    """
    source = os.environ if environ is None else environ
    return {name: source.get(name) for name in registered_env_names()}


@final
class LinkTopology(BaseModel):
    """Physical interconnect shape at measurement time."""

    model_config = ConfigDict(frozen=True, strict=True)

    thunderbolt_link_count: int = Field(ge=0)
    link_descriptors: tuple[str, ...]
    source: str

    def summary(self) -> str:
        return f"{self.thunderbolt_link_count} TB links via {self.source}"


def _run(command: Sequence[str]) -> str:
    completed = subprocess.run(  # noqa: S603
        list(command), capture_output=True, text=True, check=True, timeout=60
    )
    return completed.stdout


def _walk_link_names(node: object, found: list[str]) -> None:
    if isinstance(node, dict):
        mapping = cast(Mapping[str, object], node)
        name = mapping.get("_name")
        receptacle = mapping.get("receptacle_1_tag")
        if isinstance(name, str) and ("link" in name.lower() or receptacle is not None):
            found.append(name)
        for value in mapping.values():
            _walk_link_names(value, found)
    elif isinstance(node, list):
        for item in cast(Sequence[object], node):
            _walk_link_names(item, found)


def probe_link_topology(runner: CommandRunner | None = None) -> LinkTopology:
    """Enumerate Thunderbolt link descriptors on this host.

    ``runner`` is injectable so tests can supply canned ``system_profiler``
    output instead of shelling out. On failure the topology is recorded as
    ``source="unavailable"`` with zero links; callers that need topology to be
    known should reject that value rather than silently proceeding.
    """
    command_runner = _run if runner is None else runner
    try:
        raw = command_runner(["system_profiler", "-json", "SPThunderboltDataType"])
        parsed: object = cast(object, json.loads(raw))
    except (OSError, subprocess.SubprocessError, ValueError):
        return LinkTopology(
            thunderbolt_link_count=0, link_descriptors=(), source="unavailable"
        )
    names: list[str] = []
    _walk_link_names(parsed, names)
    return LinkTopology(
        thunderbolt_link_count=len(names),
        link_descriptors=tuple(names),
        source="system_profiler",
    )


@final
class Fingerprint(BaseModel):
    """Complete "which build, which config, which hardware" envelope."""

    model_config = ConfigDict(frozen=True, strict=True)

    exo_commit: str = Field(min_length=7)
    mlx_commit: str = Field(min_length=7)
    exo_dirty: bool
    mlx_dirty: bool
    registered_env: Mapping[str, str | None]
    link_topology: LinkTopology
    hostname: str

    def env_diff(self, other: Fingerprint) -> dict[str, tuple[str | None, str | None]]:
        """Registered env vars that differ - the A/B "what actually changed"."""
        keys = set(self.registered_env) | set(other.registered_env)
        return {
            key: (self.registered_env.get(key), other.registered_env.get(key))
            for key in sorted(keys)
            if self.registered_env.get(key) != other.registered_env.get(key)
        }


def _git(repo: str, args: Sequence[str], runner: CommandRunner) -> str:
    return runner(["git", "-C", repo, *args]).strip()


def capture_fingerprint(
    exo_repo: str,
    mlx_repo: str,
    *,
    runner: CommandRunner | None = None,
    environ: Mapping[str, str] | None = None,
    link_topology: LinkTopology | None = None,
    hostname: str | None = None,
) -> Fingerprint:
    """Capture the full fingerprint for the current build and host."""
    command_runner = _run if runner is None else runner
    topology = (
        probe_link_topology(command_runner) if link_topology is None else link_topology
    )
    return Fingerprint(
        exo_commit=_git(exo_repo, ["rev-parse", "HEAD"], command_runner),
        mlx_commit=_git(mlx_repo, ["rev-parse", "HEAD"], command_runner),
        exo_dirty=bool(_git(exo_repo, ["status", "--porcelain"], command_runner)),
        mlx_dirty=bool(_git(mlx_repo, ["status", "--porcelain"], command_runner)),
        registered_env=capture_registered_env(environ),
        link_topology=topology,
        hostname=os.uname().nodename if hostname is None else hostname,
    )
