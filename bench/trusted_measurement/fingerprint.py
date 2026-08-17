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
