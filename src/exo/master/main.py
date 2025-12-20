"""Master component for coordinating cluster state.

The Master is responsible for managing cluster state through event sourcing,
processing commands, and coordinating instance placement. Only one master
exists per cluster session (determined by election).
"""

from datetime import datetime, timedelta, timezone

import anyio
from anyio.abc import TaskGroup
from loguru import logger

from exo.master.placement import (
    add_instance_to_placements,
    delete_instance,
    get_transition_events,
    place_instance,
)
from exo.shared.apply import apply
from exo.shared.types.commands import (
    ChatCompletion,
    CreateInstance,
    DeleteInstance,
    ForwarderCommand,
    PlaceInstance,
    RequestEventLog,
    TaskFinished,
    TestCommand,
)
from exo.shared.types.common import CommandId, NodeId, SessionId
from exo.shared.types.events import (
    Event,
    ForwarderEvent,
    IndexedEvent,
    InstanceDeleted,
    NodeTimedOut,
    TaskCreated,
    TaskDeleted,
)
from exo.shared.types.state import State
from exo.shared.types.tasks import (
    ChatCompletion as ChatCompletionTask,
)
from exo.shared.types.tasks import (
    TaskId,
    TaskStatus,
)
from exo.shared.types.worker.instances import InstanceId
from exo.utils.channels import Receiver, Sender, channel
from exo.utils.event_buffer import MultiSourceBuffer


class Master:
    """Master component that coordinates cluster state and commands.

    The Master manages cluster state through event sourcing, processes commands
    from the API, coordinates instance placement, and maintains the authoritative
    event log. Events from workers are collected, ordered, indexed, and then
    broadcast back to all nodes.

    Attributes:
        state: Current cluster state (derived from events).
        node_id: Node ID of this master node.
        session_id: Session ID for the current cluster session.
        command_task_mapping: Mapping from command IDs to task IDs for tracking.
        command_receiver: Channel to receive commands.
        local_event_receiver: Channel to receive events from workers.
        global_event_sender: Channel to send indexed events to all nodes.
        event_sender: Channel for internal event routing.
        _event_log: List of all events (for event log requests).
        _multi_buffer: Buffer for ordering events from multiple sources.
        _loopback_event_receiver: Receiver for loopback events (master's own events).
        _loopback_event_sender: Sender for loopback events.
        _tg: Task group for managing concurrent operations.
    """

    def __init__(
        self,
        node_id: NodeId,
        session_id: SessionId,
        *,
        command_receiver: Receiver[ForwarderCommand],
        local_event_receiver: Receiver[ForwarderEvent],
        global_event_sender: Sender[ForwarderEvent],
    ):
        """Initialize the Master component.

        Args:
            node_id: Node ID of this master node.
            session_id: Session ID for the current cluster session.
            command_receiver: Channel to receive commands from API/router.
            local_event_receiver: Channel to receive events from workers.
            global_event_sender: Channel to send indexed events to all nodes.
        """
        self.state = State()
        self._tg: TaskGroup = anyio.create_task_group()
        self.node_id = node_id
        self.session_id = session_id
        self.command_task_mapping: dict[CommandId, TaskId] = {}
        self.command_receiver = command_receiver
        self.local_event_receiver = local_event_receiver
        self.global_event_sender = global_event_sender
        send, recv = channel[Event]()
        self.event_sender: Sender[Event] = send
        self._loopback_event_receiver: Receiver[Event] = recv
        self._loopback_event_sender: Sender[ForwarderEvent] = (
            local_event_receiver.clone_sender()
        )
        self._multi_buffer = MultiSourceBuffer[NodeId, Event]()
        self._event_log: list[Event] = []

    async def run(self) -> None:
        """Run the master's main event loop.

        Starts background tasks for:
        - Event processing (collecting and indexing events from workers)
        - Command processing (handling commands from API)
        - Loopback processing (routing master's own events)
        - Planning (cleanup of broken instances and timed-out nodes)

        The method blocks until shutdown is requested via shutdown().
        """
        logger.info("Starting Master")

        async with self._tg as tg:
            tg.start_soon(self._event_processor)
            tg.start_soon(self._command_processor)
            tg.start_soon(self._loopback_processor)
            tg.start_soon(self._plan)
        self.global_event_sender.close()
        self.local_event_receiver.close()
        self.command_receiver.close()
        self._loopback_event_sender.close()
        self._loopback_event_receiver.close()

    async def shutdown(self) -> None:
        """Shutdown the master gracefully.

        Cancels the task group, stopping all background operations.
        """
        logger.info("Stopping Master")
        self._tg.cancel_scope.cancel()

    async def _command_processor(self) -> None:
        """Process commands from the API/router.

        Handles different command types:
        - ChatCompletion: Creates a chat completion task on an available instance
        - PlaceInstance: Calculates placement and creates instance
        - CreateInstance: Creates instance with explicit configuration
        - DeleteInstance: Removes an instance
        - TaskFinished: Cleans up completed tasks
        - RequestEventLog: Sends event log to requesting node

        Generates events for state changes and sends them via event_sender.
        """
        with self.command_receiver as commands:
            async for forwarder_command in commands:
                try:
                    logger.info(f"Executing command: {forwarder_command.command}")
                    generated_events: list[Event] = []
                    command = forwarder_command.command
                    match command:
                        case TestCommand():
                            pass
                        case ChatCompletion():
                            instance_task_counts: dict[InstanceId, int] = {}
                            for instance in self.state.instances.values():
                                if (
                                    instance.shard_assignments.model_id
                                    == command.request_params.model
                                ):
                                    task_count = sum(
                                        1
                                        for task in self.state.tasks.values()
                                        if task.instance_id == instance.instance_id
                                    )
                                    instance_task_counts[instance.instance_id] = (
                                        task_count
                                    )

                            if not instance_task_counts:
                                raise ValueError(
                                    f"No instance found for model {command.request_params.model}"
                                )

                            available_instance_ids = sorted(
                                instance_task_counts.keys(),
                                key=lambda instance_id: instance_task_counts[
                                    instance_id
                                ],
                            )

                            task_id = TaskId()
                            generated_events.append(
                                TaskCreated(
                                    task_id=task_id,
                                    task=ChatCompletionTask(
                                        task_id=task_id,
                                        command_id=command.command_id,
                                        instance_id=available_instance_ids[0],
                                        task_status=TaskStatus.Pending,
                                        task_params=command.request_params,
                                    ),
                                )
                            )

                            self.command_task_mapping[command.command_id] = task_id
                        case DeleteInstance():
                            placement = delete_instance(command, self.state.instances)
                            transition_events = get_transition_events(
                                self.state.instances, placement
                            )
                            generated_events.extend(transition_events)
                        case PlaceInstance():
                            placement = place_instance(
                                command,
                                self.state.topology,
                                self.state.instances,
                            )
                            transition_events = get_transition_events(
                                self.state.instances, placement
                            )
                            generated_events.extend(transition_events)
                        case CreateInstance():
                            placement = add_instance_to_placements(
                                command,
                                self.state.topology,
                                self.state.instances,
                            )
                            transition_events = get_transition_events(
                                self.state.instances, placement
                            )
                            generated_events.extend(transition_events)
                        case TaskFinished():
                            generated_events.append(
                                TaskDeleted(
                                    task_id=self.command_task_mapping[
                                        command.finished_command_id
                                    ]
                                )
                            )
                            if command.finished_command_id in self.command_task_mapping:
                                del self.command_task_mapping[
                                    command.finished_command_id
                                ]
                        case RequestEventLog():
                            # We should just be able to send everything, since other buffers will ignore old messages
                            for i in range(command.since_idx, len(self._event_log)):
                                await self._send_event(
                                    IndexedEvent(idx=i, event=self._event_log[i])
                                )
                    for event in generated_events:
                        await self.event_sender.send(event)
                except ValueError as e:
                    logger.opt(exception=e).warning("Error in command processor")

    async def _plan(self) -> None:
        """Periodic planning task for cleanup operations.

        Runs every 10 seconds to:
        1. Remove instances whose nodes are no longer connected
        2. Timeout nodes that haven't sent heartbeats in 30+ seconds

        Note:
            This represents a design choice - these operations could
            potentially be commands instead of a periodic loop.
        """
        while True:
            connected_node_ids = set(
                [x.node_id for x in self.state.topology.list_nodes()]
            )
            for instance_id, instance in self.state.instances.items():
                for node_id in instance.shard_assignments.node_to_runner:
                    if node_id not in connected_node_ids:
                        await self.event_sender.send(
                            InstanceDeleted(instance_id=instance_id)
                        )
                        break

            for node_id, time in self.state.last_seen.items():
                now = datetime.now(tz=timezone.utc)
                if now - time > timedelta(seconds=30):
                    logger.info(f"Manually removing node {node_id} due to inactivity")
                    await self.event_sender.send(NodeTimedOut(node_id=node_id))

            await anyio.sleep(10)

    async def _event_processor(self) -> None:
        """Process and index events from workers.

        Collects events from workers, orders them using MultiSourceBuffer,
        applies them to state, indexes them, and broadcasts them to all nodes.
        Only processes events from the current session.

        Events are assigned a global index and timestamped before being
        added to the event log and broadcast.
        """
        with self.local_event_receiver as local_events:
            async for local_event in local_events:
                if local_event.session != self.session_id:
                    continue
                self._multi_buffer.ingest(
                    local_event.origin_idx,
                    local_event.event,
                    local_event.origin,
                )
                for event in self._multi_buffer.drain():
                    logger.debug(f"Master indexing event: {str(event)[:100]}")
                    indexed = IndexedEvent(event=event, idx=len(self._event_log))
                    self.state = apply(self.state, indexed)

                    event._master_time_stamp = datetime.now(tz=timezone.utc)  # pyright: ignore[reportPrivateUsage]

                    self._event_log.append(event)
                    await self._send_event(indexed)

    async def _loopback_processor(self) -> None:
        """Process loopback events from the master itself.

        Routes events generated by the master (from command processing) back
        through the event forwarding system so they are indexed like worker events.
        This ensures master-generated events go through the same ordering and
        indexing pipeline.

        Note:
            This design is a workaround - ideally master events would go through
            the same path, but this is cleaner than previous approaches.
        """
        local_index = 0
        with self._loopback_event_receiver as events:
            async for event in events:
                await self._loopback_event_sender.send(
                    ForwarderEvent(
                        origin=NodeId(f"master_{self.node_id}"),
                        origin_idx=local_index,
                        session=self.session_id,
                        event=event,
                    )
                )
                local_index += 1

    async def _send_event(self, event: IndexedEvent) -> None:
        """Send an indexed event to all nodes via global event sender.

        Wraps the indexed event in a ForwarderEvent for network transmission.

        Args:
            event: Indexed event to broadcast.

        Note:
            This function is re-entrant - may be called concurrently.
        """
        await self.global_event_sender.send(
            ForwarderEvent(
                origin=self.node_id,
                origin_idx=event.idx,
                session=self.session_id,
                event=event.event,
            )
        )
