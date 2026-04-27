from enum import Enum

from pydantic import model_validator

from exo.shared.models.model_cards import ModelTask
from exo.shared.types.common import Host, Id, NodeId
from exo.shared.types.worker.runners import RunnerId, ShardAssignments, ShardMetadata
from exo.utils.pydantic_ext import FrozenModel, TaggedModel


class InstanceId(Id):
    pass


class InstanceMeta(str, Enum):
    MlxRing = "MlxRing"
    MlxJaccl = "MlxJaccl"


class BaseInstance(TaggedModel):
    instance_id: InstanceId
    shard_assignments: ShardAssignments

    # Per-instance KV cache caps. None = unbounded (default behavior unchanged).
    # max_kv_tokens: per-active-request token cap (wraps KVCache → RotatingKVCache)
    # max_prefix_sessions: max number of cached past-prompt KV entries
    # max_prefix_bytes: max total bytes across all prefix cache entries
    max_kv_tokens: int | None = None
    max_prefix_sessions: int | None = None
    max_prefix_bytes: int | None = None

    # Per-instance prefill chunk size override. Resolution order at runtime:
    #   1. instance.prefill_step_size (if not None)  ← this field
    #   2. EXO_PREFILL_STEP_SIZE env var (if set)
    #   3. Hardcoded default 4096
    # DSv4 sparse-index attention cubic-blowup forces ≤1024 here.
    prefill_step_size: int | None = None

    # Per-instance KV cache quantization override. Resolution order at runtime:
    #   1. instance.kv_cache_bits (if not None)    ← this field
    #   2. EXO_KV_CACHE_BITS env var (if set)
    #   3. No quantization
    # Sentinel `0` means "explicitly disabled" (wins even when env sets a value).
    # Positive N means "quantize KV to N bits". None means "fall through".
    kv_cache_bits: int | None = None

    # Per-instance sampling defaults. Used when a request omits the field.
    # Resolution order: request → instance → card → cluster env → hardcoded fallback.
    default_temperature: float | None = None
    default_top_p: float | None = None
    default_top_k: int | None = None
    default_min_p: float | None = None
    default_presence_penalty: float | None = None
    default_repetition_penalty: float | None = None
    default_frequency_penalty: float | None = None

    def shard(self, runner_id: RunnerId) -> ShardMetadata | None:
        return self.shard_assignments.runner_to_shard.get(runner_id, None)


class MlxRingInstance(BaseInstance):
    hosts_by_node: dict[NodeId, list[Host]]
    ephemeral_port: int


class MlxJacclInstance(BaseInstance):
    jaccl_devices: list[list[str | None]]
    jaccl_coordinators: dict[NodeId, str]


# TODO: Single node instance
Instance = MlxRingInstance | MlxJacclInstance


class BoundInstance(FrozenModel):
    instance: Instance
    bound_runner_id: RunnerId
    bound_node_id: NodeId

    @property
    def bound_shard(self) -> ShardMetadata:
        shard = self.instance.shard(self.bound_runner_id)
        assert shard is not None
        return shard

    @property
    def is_image_model(self) -> bool:
        return (
            ModelTask.TextToImage in self.bound_shard.model_card.tasks
            or ModelTask.ImageToImage in self.bound_shard.model_card.tasks
        )

    @model_validator(mode="after")
    def validate_shard_exists(self) -> "BoundInstance":
        assert (
            self.bound_runner_id in self.instance.shard_assignments.runner_to_shard
        ), (
            "Bound Instance must be constructed with a runner_id that is in the instances assigned shards"
        )
        return self
