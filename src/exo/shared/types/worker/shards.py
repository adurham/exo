from enum import Enum
from typing import TypeAlias, final

from pydantic import Field

from exo.shared.models.model_cards import ModelCard
from exo.utils.pydantic_ext import TaggedModel


class Sharding(str, Enum):
    Tensor = "Tensor"
    Pipeline = "Pipeline"
    Hybrid = "Hybrid"


class BaseShardMetadata(TaggedModel):
    """
    Defines a specific shard of the model that is ready to be run on a device.
    Replaces previous `Shard` object.
    """

    model_card: ModelCard
    device_rank: int
    world_size: int

    # Error handling; equivalent to monkey-patch, but we can't monkey-patch runner.py
    # This is kinda annoying because it allocates memory in the ShardMetadata object. Can be rethought after Shanghai.
    immediate_exception: bool = False
    should_timeout: float | None = None

    start_layer: int = Field(ge=0)
    end_layer: int = Field(ge=0)
    n_layers: int = Field(ge=0)

    @property
    def is_first_layer(self) -> bool:
        return self.start_layer == 0

    @property
    def is_last_layer(self) -> bool:
        return self.end_layer == self.n_layers

    def __hash__(self) -> int:
        return hash(
            (
                self.model_card.model_id,
                self.start_layer,
                self.end_layer,
                self.n_layers,
                self.device_rank,
                self.world_size,
            )
        )


@final
class PipelineShardMetadata(BaseShardMetadata):
    """
    Pipeline parallelism shard meta.

    Layers are represented as a half-open interval [start_layer, end_layer),
    where start_layer is inclusive and end_layer is exclusive.
    """


@final
class CfgShardMetadata(BaseShardMetadata):
    """Shard metadata for CFG-parallel image generation models."""

    cfg_rank: int  # 0 = positive branch, 1 = negative branch
    cfg_world_size: int = 2

    # Pipeline-relative coordinates (computed at placement time)
    pipeline_rank: int  # rank within the pipeline group (0, 1, 2, ...)
    pipeline_world_size: int  # number of nodes per pipeline group


@final
class TensorShardMetadata(BaseShardMetadata):
    pass


@final
class HybridShardMetadata(BaseShardMetadata):
    """Hybrid tensor + pipeline parallel shard metadata.

    TP nodes share the same [start_layer, end_layer) range and split each
    layer's compute via all-reduce.  PP nodes own a disjoint layer range and
    communicate via send/recv.
    """

    tp_size: int  # number of nodes in the TP sub-group (e.g. 2)
    tp_rank: int  # rank within TP sub-group (-1 if not in TP group)
    pp_rank: int  # rank within the 2-stage pipeline
    pp_size: int  # total pipeline stages (e.g. 2)
    pipeline_send_to: int | None = None  # global rank to send output to
    pipeline_recv_from: int | None = None  # global rank to receive input from


ShardMetadata: TypeAlias = (
    PipelineShardMetadata | CfgShardMetadata | TensorShardMetadata | HybridShardMetadata
)
