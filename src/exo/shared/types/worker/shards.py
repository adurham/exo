"""Shard metadata types for model parallelism.

This module defines types for describing how models are sharded across
devices using different parallelism strategies (pipeline, tensor).
"""

from enum import Enum

from pydantic import Field

from exo.shared.types.models import ModelMetadata
from exo.utils.pydantic_ext import TaggedModel


class Sharding(str, Enum):
    """Sharding strategy for model parallelism.

    Values:
        Tensor: Tensor parallelism (splits layers across devices).
        Pipeline: Pipeline parallelism (splits layers sequentially).
    """

    Tensor = "Tensor"
    Pipeline = "Pipeline"


class BaseShardMetadata(TaggedModel):
    """Base metadata for a model shard assigned to a device.

    Defines which layers of a model are assigned to a specific device
    in a distributed setup. Each shard runs on a single device with
    a specific rank in the world.

    Attributes:
        model_meta: Metadata for the model being sharded.
        device_rank: Rank of this device in the distributed setup (0-based).
        world_size: Total number of devices in the distributed setup.
        immediate_exception: If True, raise exception immediately (testing/debugging).
        should_timeout: Optional timeout in seconds (testing/debugging).
        start_layer: First layer index assigned to this shard (inclusive).
        end_layer: Last layer index assigned to this shard (exclusive).
        n_layers: Total number of layers in the model.
    """

    model_meta: ModelMetadata
    device_rank: int
    world_size: int
    immediate_exception: bool = False
    should_timeout: float | None = None
    start_layer: int = Field(ge=0)
    end_layer: int = Field(ge=0)
    n_layers: int = Field(ge=0)

    @property
    def is_first_layer(self) -> bool:
        """Check if this shard contains the first layer of the model.

        Returns:
            True if start_layer is 0.
        """
        return self.start_layer == 0

    @property
    def is_last_layer(self) -> bool:
        """Check if this shard contains the last layer of the model.

        Returns:
            True if end_layer equals n_layers.
        """
        return self.end_layer == self.n_layers

    def __hash__(self) -> int:
        """Hash shard metadata for use in sets/dicts.

        Returns:
            Hash based on model ID, layer ranges, and rank.
        """
        return hash(
            (
                self.model_meta.model_id,
                self.start_layer,
                self.end_layer,
                self.n_layers,
                self.device_rank,
                self.world_size,
            )
        )


class PipelineShardMetadata(BaseShardMetadata):
    """Pipeline parallelism shard metadata.

    For pipeline parallelism, layers are split sequentially across devices.
    Layers are represented as a half-open interval [start_layer, end_layer),
    where start_layer is inclusive and end_layer is exclusive.

    Each device processes a contiguous range of layers, passing activations
    to the next device in the pipeline.
    """


class TensorShardMetadata(BaseShardMetadata):
    """Tensor parallelism shard metadata.

    For tensor parallelism, layers are split across devices by splitting
    the weight tensors. All devices process all layers, but each device
    holds a portion of each layer's weights.
    """

    pass


ShardMetadata = PipelineShardMetadata | TensorShardMetadata
"""Discriminated union of shard metadata types."""
