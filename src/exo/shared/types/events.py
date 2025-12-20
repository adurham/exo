"""Event type definitions for the event sourcing system.

This module defines all event types used in EXO's event sourcing architecture.
Events are immutable facts that describe state changes. All events inherit from
BaseEvent and are part of the Event discriminated union.
"""

from datetime import datetime

from pydantic import Field

from exo.shared.topology import Connection, NodePerformanceProfile
from exo.shared.types.chunks import GenerationChunk
from exo.shared.types.common import CommandId, Id, NodeId, SessionId
from exo.shared.types.profiling import MemoryPerformanceProfile
from exo.shared.types.tasks import Task, TaskId, TaskStatus
from exo.shared.types.worker.downloads import DownloadProgress
from exo.shared.types.worker.instances import Instance, InstanceId
from exo.shared.types.worker.runners import RunnerId, RunnerStatus
from exo.utils.pydantic_ext import CamelCaseModel, TaggedModel


class EventId(Id):
    """Identifier for events in the event sourcing system.

    Provides unique identification for each event instance.
    """

    pass


class BaseEvent(TaggedModel):
    """Base class for all events in the system.

    All events inherit from this class and include an event_id for unique
    identification. Events are immutable facts about state changes.

    Attributes:
        event_id: Unique identifier for this event instance.
        _master_time_stamp: Internal timestamp set by master when indexing
            the event. For debugging only - do not rely on this field.
    """

    event_id: EventId = Field(default_factory=EventId)
    _master_time_stamp: None | datetime = None


class TestEvent(BaseEvent):
    """Test event used for testing purposes.

    This event has no effect when applied to state (returns state unchanged).
    """

    __test__ = False


class TaskCreated(BaseEvent):
    """Event indicating a new task was created.

    Attributes:
        task_id: Unique identifier for the task.
        task: The task instance that was created.
    """

    task_id: TaskId
    task: Task


class TaskAcknowledged(BaseEvent):
    """Event indicating a task was acknowledged.

    Currently unused/no-op event. Tasks are typically tracked through
    TaskStatusUpdated events instead.

    Attributes:
        task_id: Identifier for the acknowledged task.
    """

    task_id: TaskId


class TaskDeleted(BaseEvent):
    """Event indicating a task was deleted.

    Attributes:
        task_id: Identifier for the deleted task.
    """

    task_id: TaskId


class TaskStatusUpdated(BaseEvent):
    """Event indicating a task's status changed.

    Attributes:
        task_id: Identifier for the task.
        task_status: New status of the task.
    """

    task_id: TaskId
    task_status: TaskStatus


class TaskFailed(BaseEvent):
    """Event indicating a task failed with an error.

    Attributes:
        task_id: Identifier for the failed task.
        error_type: Type/class of the error that occurred.
        error_message: Human-readable error message.
    """

    task_id: TaskId
    error_type: str
    error_message: str


class InstanceCreated(BaseEvent):
    """Event indicating a new model instance was created.

    Attributes:
        instance: The instance that was created.
    """

    instance: Instance

    def __eq__(self, other: object) -> bool:
        """Compare two InstanceCreated events for equality.

        Args:
            other: Object to compare with.

        Returns:
            True if other is InstanceCreated with same instance and event_id.
        """
        if isinstance(other, InstanceCreated):
            return self.instance == other.instance and self.event_id == other.event_id

        return False


class InstanceDeleted(BaseEvent):
    """Event indicating a model instance was deleted.

    Attributes:
        instance_id: Identifier for the deleted instance.
    """

    instance_id: InstanceId


class RunnerStatusUpdated(BaseEvent):
    """Event indicating a runner's status changed.

    Attributes:
        runner_id: Identifier for the runner.
        runner_status: New status of the runner.
    """

    runner_id: RunnerId
    runner_status: RunnerStatus


class RunnerDeleted(BaseEvent):
    """Event indicating a runner was deleted.

    Attributes:
        runner_id: Identifier for the deleted runner.
    """

    runner_id: RunnerId


class NodeCreated(BaseEvent):
    """Event indicating a new node joined the cluster.

    Attributes:
        node_id: Identifier for the new node.
    """

    node_id: NodeId


class NodeTimedOut(BaseEvent):
    """Event indicating a node timed out and should be removed.

    Published when a node hasn't sent heartbeat messages within the timeout
    period.

    Attributes:
        node_id: Identifier for the timed-out node.
    """

    node_id: NodeId


class NodePerformanceMeasured(BaseEvent):
    """Event containing performance measurements for a node.

    Attributes:
        node_id: Identifier for the node.
        when: Timestamp when the measurement was taken (as ISO string).
            This is overridden by the master when indexing, not the local
            device time.
        node_profile: Performance profile containing CPU, GPU, and other metrics.
    """

    node_id: NodeId
    when: str
    node_profile: NodePerformanceProfile


class NodeMemoryMeasured(BaseEvent):
    """Event containing memory measurements for a node.

    Attributes:
        node_id: Identifier for the node.
        when: Timestamp when the measurement was taken (as ISO string).
            This is overridden by the master when indexing, not the local
            device time.
        memory: Memory profile containing RAM availability and usage.
    """

    node_id: NodeId
    when: str
    memory: MemoryPerformanceProfile


class NodeDownloadProgress(BaseEvent):
    """Event indicating progress on a model download for a node.

    Attributes:
        download_progress: Download progress information including shard
            metadata and status.
    """

    download_progress: DownloadProgress


class ChunkGenerated(BaseEvent):
    """Event containing a generated token chunk from inference.

    Used for streaming responses from chat completion tasks.

    Attributes:
        command_id: Identifier for the command that generated this chunk.
        chunk: The generated chunk containing token information.
    """

    command_id: CommandId
    chunk: GenerationChunk


class TopologyEdgeCreated(BaseEvent):
    """Event indicating a new connection was established between nodes.

    Attributes:
        edge: Connection information between two nodes.
    """

    edge: Connection


class TopologyEdgeDeleted(BaseEvent):
    """Event indicating a connection between nodes was lost.

    Attributes:
        edge: Connection information for the lost connection.
    """

    edge: Connection


Event = (
    TestEvent
    | TaskCreated
    | TaskStatusUpdated
    | TaskFailed
    | TaskDeleted
    | TaskAcknowledged
    | InstanceCreated
    | InstanceDeleted
    | RunnerStatusUpdated
    | RunnerDeleted
    | NodeCreated
    | NodeTimedOut
    | NodePerformanceMeasured
    | NodeMemoryMeasured
    | NodeDownloadProgress
    | ChunkGenerated
    | TopologyEdgeCreated
    | TopologyEdgeDeleted
)
"""Discriminated union of all event types in the system.

Used for type checking and pattern matching over events.
"""


class IndexedEvent(CamelCaseModel):
    """An event indexed by the master with a globally unique sequence number.

    Events are indexed by the master to ensure ordering and enable replay.
    The index is monotonically increasing and unique across all events in
    a session.

    Attributes:
        idx: Globally unique index assigned by the master. Must be >= 0.
        event: The event that was indexed.
    """

    idx: int = Field(ge=0)
    event: Event


class ForwarderEvent(CamelCaseModel):
    """Event wrapper for network transmission.

    Wraps an event with origin information for forwarding across the network.
    The forwarder uses this to serialize and send events between nodes.

    Attributes:
        origin_idx: Sequence index from the originating node's local event
            stream. Used for ordering events from the same source.
        origin: Node ID of the node that originated this event.
        session: Session ID identifying the cluster session this event belongs to.
        event: The actual event being forwarded.
    """

    origin_idx: int = Field(ge=0)
    origin: NodeId
    session: SessionId
    event: Event
