"""Command type definitions for the command system.

This module defines all command types that can be sent to the cluster.
Commands are requests that trigger state changes through event generation.
All commands inherit from BaseCommand and are part of the Command discriminated union.
"""

from pydantic import Field

from exo.shared.types.api import ChatCompletionTaskParams
from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.models import ModelMetadata
from exo.shared.types.worker.instances import Instance, InstanceId, InstanceMeta
from exo.shared.types.worker.shards import Sharding
from exo.utils.pydantic_ext import CamelCaseModel, TaggedModel


class BaseCommand(TaggedModel):
    """Base class for all commands in the system.

    All commands inherit from this class and include a command_id for tracking.

    Attributes:
        command_id: Unique identifier for this command instance.
    """

    command_id: CommandId = Field(default_factory=CommandId)


class TestCommand(BaseCommand):
    """Test command used for testing purposes.

    Has no effect when processed (no-op command).
    """

    __test__ = False


class ChatCompletion(BaseCommand):
    """Command to generate a chat completion response.

    Triggers inference on a model instance to generate text completion.

    Attributes:
        request_params: Parameters for the chat completion including messages,
            model, temperature, etc.
    """

    request_params: ChatCompletionTaskParams


class PlaceInstance(BaseCommand):
    """Command to place and create a model instance.

    Requests placement of a model instance across nodes with specified sharding
    strategy. The master will determine optimal placement and create the instance.

    Attributes:
        model_meta: Metadata for the model to place.
        sharding: Sharding strategy (e.g., tensor parallelism configuration).
        instance_meta: Metadata for the instance (ports, etc.).
        min_nodes: Minimum number of nodes required for placement.
    """

    model_meta: ModelMetadata
    sharding: Sharding
    instance_meta: InstanceMeta
    min_nodes: int


class CreateInstance(BaseCommand):
    """Command to create a model instance with explicit configuration.

    Creates an instance with the provided configuration without placement
    calculation. Used for manually configured instances.

    Attributes:
        instance: The instance configuration to create.
    """

    instance: Instance


class DeleteInstance(BaseCommand):
    """Command to delete a model instance.

    Attributes:
        instance_id: Identifier for the instance to delete.
    """

    instance_id: InstanceId


class TaskFinished(BaseCommand):
    """Command indicating a task has finished.

    Sent by workers to notify the master that a task completed. Used to
    track command completion.

    Attributes:
        finished_command_id: Command ID of the command that finished.
    """

    finished_command_id: CommandId


class RequestEventLog(BaseCommand):
    """Command to request events since a given index.

    Used by workers to catch up on missed events (NACK mechanism).
    Master responds by sending all events from since_idx onwards.

    Attributes:
        since_idx: Index to request events from (inclusive).
    """

    since_idx: int


Command = (
    TestCommand
    | RequestEventLog
    | ChatCompletion
    | PlaceInstance
    | CreateInstance
    | DeleteInstance
    | TaskFinished
)
"""Discriminated union of all command types in the system.

Used for type checking and pattern matching over commands.
"""


class ForwarderCommand(CamelCaseModel):
    """Command wrapper for network transmission.

    Wraps a command with origin information for forwarding across the network.

    Attributes:
        origin: Node ID of the node that originated this command.
        command: The actual command being forwarded.
    """

    origin: NodeId
    command: Command
