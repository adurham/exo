"""Task type definitions for the worker task system.

This module defines all task types that workers execute. Tasks represent
operations that need to be performed, such as downloading models, loading
models, running inference, etc. All tasks inherit from BaseTask and are
part of the Task discriminated union.
"""

from enum import Enum

from pydantic import Field

from exo.shared.types.api import ChatCompletionTaskParams
from exo.shared.types.common import CommandId, Id
from exo.shared.types.worker.instances import BoundInstance, InstanceId
from exo.shared.types.worker.runners import RunnerId
from exo.shared.types.worker.shards import ShardMetadata
from exo.utils.pydantic_ext import TaggedModel


class TaskId(Id):
    """Identifier for tasks in the task system."""

    pass


class TaskStatus(str, Enum):
    """Status of a task's execution lifecycle."""

    Pending = "Pending"
    """Task is queued but not yet started."""

    Running = "Running"
    """Task is currently executing."""

    Complete = "Complete"
    """Task finished successfully."""

    TimedOut = "TimedOut"
    """Task exceeded its timeout and was cancelled."""

    Failed = "Failed"
    """Task encountered an error and failed."""


class BaseTask(TaggedModel):
    """Base class for all tasks in the system.

    All tasks inherit from this class and include common fields for tracking.

    Attributes:
        task_id: Unique identifier for this task instance.
        task_status: Current status of the task execution.
        instance_id: Identifier for the instance this task is associated with.
    """

    task_id: TaskId = Field(default_factory=TaskId)
    task_status: TaskStatus = Field(default=TaskStatus.Pending)
    instance_id: InstanceId


class CreateRunner(BaseTask):
    """Task to create a new runner for model execution.

    Emitted by Worker to spawn a new runner process that will handle
    model loading and inference.

    Attributes:
        bound_instance: Instance configuration to bind the runner to.
    """

    bound_instance: BoundInstance


class DownloadModel(BaseTask):
    """Task to download a model shard.

    Emitted by Worker to download a specific shard of a model from
    Hugging Face or another source.

    Attributes:
        shard_metadata: Metadata identifying which shard to download.
    """

    shard_metadata: ShardMetadata


class LoadModel(BaseTask):
    """Task to load a model into memory.

    Emitted by Worker after download completes to load the model
    weights into the runner's memory.
    """

    pass


class StartWarmup(BaseTask):
    """Task to start model warmup/preparation.

    Emitted by Worker to prepare the model for inference (e.g., compile,
    warm up KV cache, etc.).
    """

    pass


class ChatCompletion(BaseTask):
    """Task to generate a chat completion.

    Emitted by Master to request text generation from a model instance.

    Attributes:
        command_id: Command ID that triggered this task.
        task_params: Parameters for the chat completion (messages, temperature, etc.).
        error_type: Type of error if task failed. None if successful.
        error_message: Human-readable error message if task failed. None if successful.
    """

    command_id: CommandId
    task_params: ChatCompletionTaskParams

    error_type: str | None = Field(default=None)
    error_message: str | None = Field(default=None)


class Shutdown(BaseTask):
    """Task to shutdown a runner.

    Emitted by Worker to gracefully shutdown a runner process.

    Attributes:
        runner_id: Identifier for the runner to shutdown.
    """

    runner_id: RunnerId


Task = (
    CreateRunner | DownloadModel | LoadModel | StartWarmup | ChatCompletion | Shutdown
)
"""Discriminated union of all task types in the system.

Used for type checking and pattern matching over tasks.
"""
