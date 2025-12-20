"""Instance type definitions for model deployment.

This module defines types for model instances, which represent deployed
models distributed across nodes in the cluster.
"""

from enum import Enum

from pydantic import model_validator

from exo.shared.types.common import Host, Id, NodeId
from exo.shared.types.worker.runners import RunnerId, ShardAssignments, ShardMetadata
from exo.utils.pydantic_ext import CamelCaseModel, TaggedModel


class InstanceId(Id):
    """Identifier for model instances in the cluster."""

    pass


class InstanceMeta(str, Enum):
    """Instance metadata type (backend implementation).

    Values:
        MlxRing: MLX with ring-based communication (standard).
        MlxJaccl: MLX with InfiniBand Verbs (IBV) communication (experimental).
    """

    MlxRing = "MlxRing"
    MlxJaccl = "MlxJaccl"


class BaseInstance(TaggedModel):
    """Base class for model instances.

    Attributes:
        instance_id: Unique identifier for this instance.
        shard_assignments: Assignment of shards to runners and nodes.
    """

    instance_id: InstanceId
    shard_assignments: ShardAssignments

    def shard(self, runner_id: RunnerId) -> ShardMetadata | None:
        """Get the shard assigned to a runner.

        Args:
            runner_id: Runner identifier.

        Returns:
            ShardMetadata if runner has an assigned shard, None otherwise.
        """
        return self.shard_assignments.runner_to_shard.get(runner_id, None)


class MlxRingInstance(BaseInstance):
    """MLX instance using ring-based communication.

    Attributes:
        hosts: List of host addresses (IP:port) for ring communication.
    """

    hosts: list[Host]


class MlxJacclInstance(BaseInstance):
    """MLX instance using InfiniBand Verbs (IBV) communication.

    Attributes:
        ibv_devices: Matrix of IBV device names per node (experimental).
        ibv_coordinators: Mapping of node IDs to coordinator addresses.
    """

    ibv_devices: list[list[str | None]]
    ibv_coordinators: dict[NodeId, str]


Instance = MlxRingInstance | MlxJacclInstance
"""Discriminated union of all instance types."""


class BoundInstance(CamelCaseModel):
    """Instance bound to a specific runner on a specific node.

    Used to pass instance configuration to a runner process with the
    specific shard and node context.

    Attributes:
        instance: The instance configuration.
        bound_runner_id: Runner ID this instance is bound to.
        bound_node_id: Node ID where the runner is executing.
    """

    instance: Instance
    bound_runner_id: RunnerId
    bound_node_id: NodeId

    @property
    def bound_shard(self) -> ShardMetadata:
        """Get the shard assigned to the bound runner.

        Returns:
            ShardMetadata for the bound runner.

        Raises:
            AssertionError: If runner has no assigned shard.
        """
        shard = self.instance.shard(self.bound_runner_id)
        assert shard is not None
        return shard

    @model_validator(mode="after")
    def validate_shard_exists(self) -> "BoundInstance":
        """Validate that the bound runner has an assigned shard.

        Returns:
            Self after validation.

        Raises:
            AssertionError: If bound_runner_id is not in shard assignments.
        """
        assert (
            self.bound_runner_id in self.instance.shard_assignments.runner_to_shard
        ), (
            "Bound Instance must be constructed with a runner_id that is in the instances assigned shards"
        )
        return self
