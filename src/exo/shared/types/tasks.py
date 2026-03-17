from enum import Enum

from pydantic import Field

from exo.shared.types.api import (
    ImageEditsTaskParams,
    ImageGenerationTaskParams,
)
from exo.shared.types.common import CommandId, Id
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.worker.instances import BoundInstance, InstanceId
from exo.shared.types.worker.runners import RunnerId
from exo.shared.types.worker.shards import ShardMetadata
from exo.utils.pydantic_ext import TaggedModel


class TaskId(Id):
    pass


CANCEL_ALL_TASKS = TaskId("CANCEL_ALL_TASKS")


class TaskStatus(str, Enum):
    Pending = "Pending"
    Running = "Running"
    Complete = "Complete"
    TimedOut = "TimedOut"
    Failed = "Failed"
    Cancelled = "Cancelled"


class BaseTask(TaggedModel):
    task_id: TaskId = Field(default_factory=TaskId)
    task_status: TaskStatus = Field(default=TaskStatus.Pending)
    instance_id: InstanceId


class CreateRunner(BaseTask):  # emitted by Worker
    bound_instance: BoundInstance


class DownloadModel(BaseTask):  # emitted by Worker
    shard_metadata: ShardMetadata
    repo_url: str | None = Field(default=None)


class LoadModel(BaseTask):  # emitted by Worker
    pass


class ConnectToGroup(BaseTask):  # emitted by Worker
    pass


class StartWarmup(BaseTask):  # emitted by Worker
    pass


class TextGeneration(BaseTask):  # emitted by Master
    command_id: CommandId
    task_params: TextGenerationTaskParams

    error_type: str | None = Field(default=None)
    error_message: str | None = Field(default=None)


class CancelTask(BaseTask):
    cancelled_task_id: TaskId
    runner_id: RunnerId


class ImageGeneration(BaseTask):  # emitted by Master
    command_id: CommandId
    task_params: ImageGenerationTaskParams

    error_type: str | None = Field(default=None)
    error_message: str | None = Field(default=None)


class ImageEdits(BaseTask):  # emitted by Master
    command_id: CommandId
    task_params: ImageEditsTaskParams

    error_type: str | None = Field(default=None)
    error_message: str | None = Field(default=None)


class DraftGeneration(BaseTask):  # emitted by Master
    """Stateful draft token generation for speculative decoding.

    The runner maintains a persistent KV cache across DraftGeneration tasks.
    Each call feeds token_id through the model and generates num_tokens predictions.
    trim > 0 trims rejected tokens from the cache before generating.
    """
    command_id: CommandId
    token_id: int = 0
    num_tokens: int = 10
    trim: int = 0
    action: str = "draft"  # "draft", "prefill", "reset"
    prefill_token_ids: list[int] = Field(default_factory=list)


class Shutdown(BaseTask):  # emitted by Worker
    runner_id: RunnerId


Task = (
    CreateRunner
    | DownloadModel
    | ConnectToGroup
    | LoadModel
    | StartWarmup
    | TextGeneration
    | CancelTask
    | ImageGeneration
    | ImageEdits
    | DraftGeneration
    | Shutdown
)
