"""Runner type definitions for model execution.

This module defines types for runners, which are processes that execute
model inference. Includes runner status states and shard assignments.
"""

from collections.abc import Mapping

from pydantic import model_validator

from exo.shared.types.common import Id, NodeId
from exo.shared.types.models import ModelId
from exo.shared.types.worker.shards import ShardMetadata
from exo.utils.pydantic_ext import CamelCaseModel, TaggedModel


class RunnerId(Id):
    """Identifier for runners (model execution processes)."""

    pass


class RunnerError(Exception):
    """Exception raised by runner processes."""

    pass


class BaseRunnerStatus(TaggedModel):
    """Base class for runner status states.

    Runners transition through various states during their lifecycle.
    """

    def is_running(self) -> bool:
        """Check if runner is in running state.

        Returns:
            True if status is RunnerRunning.
        """
        return isinstance(self, RunnerRunning)


class RunnerWaitingForModel(BaseRunnerStatus):
    """Runner is waiting for model download to complete."""

    pass


class RunnerLoading(BaseRunnerStatus):
    """Runner is loading model weights into memory."""

    pass


class RunnerLoaded(BaseRunnerStatus):
    """Runner has loaded model weights but is not yet ready."""

    pass


class RunnerWarmingUp(BaseRunnerStatus):
    """Runner is warming up (compiling, initializing KV cache, etc.)."""

    pass


class RunnerReady(BaseRunnerStatus):
    """Runner is ready to accept inference tasks."""

    pass


class RunnerRunning(BaseRunnerStatus):
    """Runner is currently executing an inference task."""

    pass


class RunnerShutdown(BaseRunnerStatus):
    """Runner has been shutdown gracefully."""

    pass


class RunnerFailed(BaseRunnerStatus):
    """Runner has failed with an error.

    Attributes:
        error_message: Human-readable error message describing the failure.
    """

    error_message: str | None = None


RunnerStatus = (
    RunnerWaitingForModel
    | RunnerLoading
    | RunnerLoaded
    | RunnerWarmingUp
    | RunnerReady
    | RunnerRunning
    | RunnerShutdown
    | RunnerFailed
)
"""Discriminated union of all runner status types."""


class ShardAssignments(CamelCaseModel):
    """Assignment of model shards to runners and nodes.

    Maps a model's shards to specific runners (processes) and nodes (devices),
    defining the distribution of work across the cluster.

    Attributes:
        model_id: Identifier for the model being assigned.
        runner_to_shard: Mapping from runner IDs to their assigned shards.
        node_to_runner: Mapping from node IDs to their assigned runner IDs.
    """

    model_id: ModelId
    runner_to_shard: Mapping[RunnerId, ShardMetadata]
    node_to_runner: Mapping[NodeId, RunnerId]

    @model_validator(mode="after")
    def validate_runners_exist(self) -> "ShardAssignments":
        """Validate that all runners in node_to_runner have shards.

        Returns:
            Self after validation.

        Raises:
            ValueError: If a runner in node_to_runner lacks a shard assignment.
        """
        for runner_id in self.node_to_runner.values():
            if runner_id not in self.runner_to_shard:
                raise ValueError(
                    f"Runner {runner_id} in node_to_runner does not exist in runner_to_shard"
                )
        return self
