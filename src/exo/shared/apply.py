"""Event sourcing state application logic.

This module implements the event sourcing pattern for EXO's cluster state management.
Events are immutable facts that describe state changes. State is derived by applying
events in order using pure functions.

Each event type has a corresponding apply function that takes the current state and
an event, returning a new state with the event's changes applied. This ensures
state transitions are deterministic and replayable.
"""

import copy
from collections.abc import Mapping, Sequence
from datetime import datetime

from loguru import logger

from exo.shared.types.common import NodeId
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    IndexedEvent,
    InstanceCreated,
    InstanceDeleted,
    NodeCreated,
    NodeDownloadProgress,
    NodeMemoryMeasured,
    NodePerformanceMeasured,
    NodeTimedOut,
    RunnerDeleted,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskCreated,
    TaskDeleted,
    TaskFailed,
    TaskStatusUpdated,
    TestEvent,
    TopologyEdgeCreated,
    TopologyEdgeDeleted,
)
from exo.shared.types.profiling import NodePerformanceProfile, SystemPerformanceProfile
from exo.shared.types.state import State
from exo.shared.types.tasks import Task, TaskId, TaskStatus
from exo.shared.types.topology import NodeInfo
from exo.shared.types.worker.downloads import DownloadProgress
from exo.shared.types.worker.instances import Instance, InstanceId
from exo.shared.types.worker.runners import RunnerId, RunnerStatus


def event_apply(event: Event, state: State) -> State:
    """Apply an event to state using event sourcing pattern.

    Routes events to their specific apply functions. Some events (TestEvent,
    ChunkGenerated, TaskAcknowledged) are no-ops that return state unchanged.

    Args:
        event: The event to apply to the state.
        state: Current cluster state.

    Returns:
        New state with the event's changes applied. The original state is not
        modified (immutable update pattern).

    Note:
        This function is pure: same event and state always produce the same
        result. No side effects.
    """
    match event:
        case (
            TestEvent() | ChunkGenerated() | TaskAcknowledged()
        ):  # TaskAcknowledged should never be sent by a worker but i dont mind if it just gets ignored
            return state
        case InstanceCreated():
            return apply_instance_created(event, state)
        case InstanceDeleted():
            return apply_instance_deleted(event, state)
        case NodeCreated():
            return apply_topology_node_created(event, state)
        case NodeTimedOut():
            return apply_node_timed_out(event, state)
        case NodePerformanceMeasured():
            return apply_node_performance_measured(event, state)
        case NodeDownloadProgress():
            return apply_node_download_progress(event, state)
        case NodeMemoryMeasured():
            return apply_node_memory_measured(event, state)
        case RunnerDeleted():
            return apply_runner_deleted(event, state)
        case RunnerStatusUpdated():
            return apply_runner_status_updated(event, state)
        case TaskCreated():
            return apply_task_created(event, state)
        case TaskDeleted():
            return apply_task_deleted(event, state)
        case TaskFailed():
            return apply_task_failed(event, state)
        case TaskStatusUpdated():
            return apply_task_status_updated(event, state)
        case TopologyEdgeCreated():
            return apply_topology_edge_created(event, state)
        case TopologyEdgeDeleted():
            return apply_topology_edge_deleted(event, state)


def apply(state: State, event: IndexedEvent) -> State:
    """Apply an indexed event to state with ordering validation.

    Validates that events are applied in strict sequential order (no gaps, no
    duplicates). Updates the state's last_event_applied_idx to track progress.

    Args:
        state: Current cluster state.
        event: Indexed event to apply (contains event and its index).

    Returns:
        New state with the event applied and last_event_applied_idx updated.

    Raises:
        AssertionError: If event index does not match expected next index
            (indicates out-of-order or duplicate event application).
    """
    if state.last_event_applied_idx != event.idx - 1:
        logger.warning(
            f"Expected event {state.last_event_applied_idx + 1} but received {event.idx}"
        )
    assert state.last_event_applied_idx == event.idx - 1
    new_state: State = event_apply(event.event, state)
    return new_state.model_copy(update={"last_event_applied_idx": event.idx})


def apply_node_download_progress(event: NodeDownloadProgress, state: State) -> State:
    """Update or add download progress for a node.

    Updates the download progress for a specific shard on a node. If the shard
    already has progress tracked, replaces it. Otherwise, adds a new entry.

    Args:
        event: NodeDownloadProgress event containing the download status.
        state: Current cluster state.

    Returns:
        New state with updated downloads mapping.
    """
    dp = event.download_progress
    node_id = dp.node_id

    current = list(state.downloads.get(node_id, ()))

    replaced = False
    for i, existing_dp in enumerate(current):
        if existing_dp.shard_metadata == dp.shard_metadata:
            current[i] = dp
            replaced = True
            break

    if not replaced:
        current.append(dp)

    new_downloads: Mapping[NodeId, Sequence[DownloadProgress]] = {
        **state.downloads,
        node_id: current,
    }
    return state.model_copy(update={"downloads": new_downloads})


def apply_task_created(event: TaskCreated, state: State) -> State:
    """Add a new task to state.

    Args:
        event: TaskCreated event containing the task to add.
        state: Current cluster state.

    Returns:
        New state with the task added to the tasks mapping.
    """
    new_tasks: Mapping[TaskId, Task] = {**state.tasks, event.task_id: event.task}
    return state.model_copy(update={"tasks": new_tasks})


def apply_task_deleted(event: TaskDeleted, state: State) -> State:
    """Remove a task from state.

    Args:
        event: TaskDeleted event containing the task ID to remove.
        state: Current cluster state.

    Returns:
        New state with the task removed from the tasks mapping.
    """
    new_tasks: Mapping[TaskId, Task] = {
        tid: task for tid, task in state.tasks.items() if tid != event.task_id
    }
    return state.model_copy(update={"tasks": new_tasks})


def apply_task_status_updated(event: TaskStatusUpdated, state: State) -> State:
    """Update a task's status.

    Updates the status of an existing task and clears error fields if the status
    is not Failed. If the task doesn't exist, returns state unchanged.

    Args:
        event: TaskStatusUpdated event with task ID and new status.
        state: Current cluster state.

    Returns:
        New state with the task's status updated, or unchanged state if task
        doesn't exist.
    """
    if event.task_id not in state.tasks:
        return state

    update: dict[str, TaskStatus | None] = {
        "task_status": event.task_status,
    }
    if event.task_status != TaskStatus.Failed:
        update["error_type"] = None
        update["error_message"] = None

    updated_task = state.tasks[event.task_id].model_copy(update=update)
    new_tasks: Mapping[TaskId, Task] = {**state.tasks, event.task_id: updated_task}
    return state.model_copy(update={"tasks": new_tasks})


def apply_task_failed(event: TaskFailed, state: State) -> State:
    """Mark a task as failed with error details.

    Updates a task's error_type and error_message fields. If the task doesn't
    exist, returns state unchanged.

    Args:
        event: TaskFailed event with task ID and error information.
        state: Current cluster state.

    Returns:
        New state with the task's error fields updated, or unchanged state if
        task doesn't exist.
    """
    if event.task_id not in state.tasks:
        return state

    updated_task = state.tasks[event.task_id].model_copy(
        update={"error_type": event.error_type, "error_message": event.error_message}
    )
    new_tasks: Mapping[TaskId, Task] = {**state.tasks, event.task_id: updated_task}
    return state.model_copy(update={"tasks": new_tasks})


def apply_instance_created(event: InstanceCreated, state: State) -> State:
    """Add a new model instance to state.

    Args:
        event: InstanceCreated event containing the instance to add.
        state: Current cluster state.

    Returns:
        New state with the instance added to the instances mapping.
    """
    instance = event.instance
    new_instances: Mapping[InstanceId, Instance] = {
        **state.instances,
        instance.instance_id: instance,
    }
    return state.model_copy(update={"instances": new_instances})


def apply_instance_deleted(event: InstanceDeleted, state: State) -> State:
    """Remove a model instance from state.

    Args:
        event: InstanceDeleted event containing the instance ID to remove.
        state: Current cluster state.

    Returns:
        New state with the instance removed from the instances mapping.
    """
    new_instances: Mapping[InstanceId, Instance] = {
        iid: inst for iid, inst in state.instances.items() if iid != event.instance_id
    }
    return state.model_copy(update={"instances": new_instances})


def apply_runner_status_updated(event: RunnerStatusUpdated, state: State) -> State:
    """Update a runner's status.

    Args:
        event: RunnerStatusUpdated event with runner ID and new status.
        state: Current cluster state.

    Returns:
        New state with the runner's status updated.
    """
    new_runners: Mapping[RunnerId, RunnerStatus] = {
        **state.runners,
        event.runner_id: event.runner_status,
    }
    return state.model_copy(update={"runners": new_runners})


def apply_runner_deleted(event: RunnerDeleted, state: State) -> State:
    """Remove a runner from state.

    Args:
        event: RunnerDeleted event containing the runner ID to remove.
        state: Current cluster state.

    Returns:
        New state with the runner removed from the runners mapping.

    Raises:
        AssertionError: If runner doesn't exist in state (indicates event
            ordering issue - runner deleted before being created).
    """
    assert event.runner_id in state.runners, (
        "RunnerDeleted before any RunnerStatusUpdated events"
    )
    new_runners: Mapping[RunnerId, RunnerStatus] = {
        rid: rs for rid, rs in state.runners.items() if rid != event.runner_id
    }
    return state.model_copy(update={"runners": new_runners})


def apply_node_timed_out(event: NodeTimedOut, state: State) -> State:
    """Remove a node from state due to timeout.

    Removes the node from topology, node profiles, and last_seen tracking.
    Used when a node hasn't sent heartbeat messages within the timeout period.

    Args:
        event: NodeTimedOut event containing the node ID to remove.
        state: Current cluster state.

    Returns:
        New state with the node removed from all relevant mappings.
    """
    topology = copy.copy(state.topology)
    state.topology.remove_node(event.node_id)
    node_profiles = {
        key: value for key, value in state.node_profiles.items() if key != event.node_id
    }
    last_seen = {
        key: value for key, value in state.last_seen.items() if key != event.node_id
    }
    return state.model_copy(
        update={
            "topology": topology,
            "node_profiles": node_profiles,
            "last_seen": last_seen,
        }
    )


def apply_node_performance_measured(
    event: NodePerformanceMeasured, state: State
) -> State:
    """Update node performance profile and topology.

    Updates the performance profile for a node and adds the node to topology
    if it doesn't exist. Also updates last_seen timestamp.

    Args:
        event: NodePerformanceMeasured event with node ID and performance data.
        state: Current cluster state.

    Returns:
        New state with updated node profiles, topology, and last_seen.
    """
    new_profiles: Mapping[NodeId, NodePerformanceProfile] = {
        **state.node_profiles,
        event.node_id: event.node_profile,
    }
    last_seen: Mapping[NodeId, datetime] = {
        **state.last_seen,
        event.node_id: datetime.fromisoformat(event.when),
    }
    state = state.model_copy(update={"node_profiles": new_profiles})
    topology = copy.copy(state.topology)
    # TODO: NodeCreated
    if not topology.contains_node(event.node_id):
        topology.add_node(NodeInfo(node_id=event.node_id))
    topology.update_node_profile(event.node_id, event.node_profile)
    return state.model_copy(
        update={
            "node_profiles": new_profiles,
            "topology": topology,
            "last_seen": last_seen,
        }
    )


def apply_node_memory_measured(event: NodeMemoryMeasured, state: State) -> State:
    """Update node memory profile.

    Updates the memory information in a node's performance profile. If the node
    doesn't have a profile yet, creates one with default values. Also updates
    topology and last_seen.

    Args:
        event: NodeMemoryMeasured event with node ID and memory profile.
        state: Current cluster state.

    Returns:
        New state with updated node profiles, topology, and last_seen.
    """
    existing = state.node_profiles.get(event.node_id)
    topology = copy.copy(state.topology)

    if existing is None:
        created = NodePerformanceProfile(
            model_id="unknown",
            chip_id="unknown",
            friendly_name="Unknown",
            memory=event.memory,
            network_interfaces=[],
            system=SystemPerformanceProfile(
                # TODO: flops_fp16=0.0,
                gpu_usage=0.0,
                temp=0.0,
                sys_power=0.0,
                pcpu_usage=0.0,
                ecpu_usage=0.0,
                ane_power=0.0,
            ),
        )
        created_profiles: Mapping[NodeId, NodePerformanceProfile] = {
            **state.node_profiles,
            event.node_id: created,
        }
        last_seen: Mapping[NodeId, datetime] = {
            **state.last_seen,
            event.node_id: datetime.fromisoformat(event.when),
        }
        if not topology.contains_node(event.node_id):
            topology.add_node(NodeInfo(node_id=event.node_id))
            # TODO: NodeCreated
        topology.update_node_profile(event.node_id, created)
        return state.model_copy(
            update={
                "node_profiles": created_profiles,
                "topology": topology,
                "last_seen": last_seen,
            }
        )

    updated = existing.model_copy(update={"memory": event.memory})
    updated_profiles: Mapping[NodeId, NodePerformanceProfile] = {
        **state.node_profiles,
        event.node_id: updated,
    }
    # TODO: NodeCreated
    if not topology.contains_node(event.node_id):
        topology.add_node(NodeInfo(node_id=event.node_id))
    topology.update_node_profile(event.node_id, updated)
    return state.model_copy(
        update={"node_profiles": updated_profiles, "topology": topology}
    )


def apply_topology_node_created(event: NodeCreated, state: State) -> State:
    """Add a node to the topology.

    Args:
        event: NodeCreated event containing the node ID.
        state: Current cluster state.

    Returns:
        New state with the node added to topology.
    """
    topology = copy.copy(state.topology)
    topology.add_node(NodeInfo(node_id=event.node_id))
    return state.model_copy(update={"topology": topology})


def apply_topology_edge_created(event: TopologyEdgeCreated, state: State) -> State:
    """Add a connection edge to the topology.

    Args:
        event: TopologyEdgeCreated event containing the connection to add.
        state: Current cluster state.

    Returns:
        New state with the connection added to topology.
    """
    topology = copy.copy(state.topology)
    topology.add_connection(event.edge)
    return state.model_copy(update={"topology": topology})


def apply_topology_edge_deleted(event: TopologyEdgeDeleted, state: State) -> State:
    """Remove a connection edge from the topology.

    If the connection doesn't exist, returns state unchanged.

    Args:
        event: TopologyEdgeDeleted event containing the connection to remove.
        state: Current cluster state.

    Returns:
        New state with the connection removed from topology, or unchanged state
        if connection doesn't exist.
    """
    topology = copy.copy(state.topology)
    if not topology.contains_connection(event.edge):
        return state
    topology.remove_connection(event.edge)
    return state.model_copy(update={"topology": topology})
