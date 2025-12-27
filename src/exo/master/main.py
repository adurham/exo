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
from exo.shared.static_config import create_static_topology
from exo.shared.topology import Topology
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
    def __init__(
        self,
        node_id: NodeId,
        session_id: SessionId,
        *,
        command_receiver: Receiver[ForwarderCommand] | None = None,
        # Receiving indexed events from the forwarder to be applied to state
        # Ideally these would be WorkerForwarderEvents but type system says no :(
        local_event_receiver: Receiver[ForwarderEvent] | None = None,
        # Send events to the forwarder to be indexed (usually from command processing)
        # Ideally these would be MasterForwarderEvents but type system says no :(
        global_event_sender: Sender[ForwarderEvent] | None = None,
        initial_topology: Topology | None = None,
    ):
        # Initialize state with static topology if provided, otherwise empty topology
        if initial_topology is not None:
            self.state = State(topology=initial_topology)
        else:
            # For backward compatibility, use static topology if available
            try:
                static_topology = create_static_topology()
                self.state = State(topology=static_topology)
                logger.info("Initialized Master state with static topology")
            except Exception as e:
                logger.warning(f"Failed to create static topology, using empty: {e}")
                self.state = State()
        # TaskGroup will be created in run() when in async context
        self._tg: TaskGroup | None = None
        self.node_id = node_id
        self.session_id = session_id
        self.command_task_mapping: dict[CommandId, TaskId] = {}
        self.command_receiver = command_receiver
        self.local_event_receiver = local_event_receiver
        self.global_event_sender = global_event_sender
        send, recv = channel[Event]()
        self.event_sender: Sender[Event] = send
        self._loopback_event_receiver: Receiver[Event] = recv
        # Only create loopback sender if local_event_receiver is provided
        self._loopback_event_sender: Sender[ForwarderEvent] | None = (
            local_event_receiver.clone_sender() if local_event_receiver else None
        )
        self._multi_buffer = MultiSourceBuffer[NodeId, Event]()
        # TODO: not have this
        self._event_log: list[Event] = []
        # Worker client pool for pushing state updates (static setup)
        self._worker_clients = None

    async def run(self):
        logger.info("Starting Master")
        
        # Initialize worker client pool for static setup (skip in tests)
        import os
        if not os.environ.get("EXO_TESTS"):
            logger.info("Initializing worker client pool for static cluster setup...")
            try:
                from exo.master.worker_client import WorkerClientPool
                from exo.shared.static_config import get_static_config
                self._worker_clients = WorkerClientPool()
                config = get_static_config()
                logger.info(f"Found {len(config.workers)} workers in static config")
                for worker in config.workers:
                    worker_url = f"http://{worker.tailscale_ip}:8080"  # Workers listen on port 8080
                    logger.info(f"Adding worker {worker.node_id} at {worker_url}")
                    await self._worker_clients.add_worker(worker.node_id, worker_url)
                logger.info(f"✓ Initialized worker client pool with {len(config.workers)} workers")
                # Push initial state to all workers to sync any existing instances/runners
                logger.info("Pushing initial state to all workers...")
                await self._worker_clients.push_state_to_all(self.state)
                logger.info("✓ Initial state push complete")
            except Exception as e:
                logger.error(f"✗ Failed to initialize worker client pool: {e}", exc_info=True)
                self._worker_clients = None
        else:
            logger.info("Skipping worker client pool initialization (EXO_TESTS is set)")
        
        # Create task group in async context
        self._tg = anyio.create_task_group()
        async with self._tg as tg:
            if self.local_event_receiver:
                tg.start_soon(self._event_processor)
            if self.command_receiver:
                tg.start_soon(self._command_processor)
            if self._loopback_event_sender:
                tg.start_soon(self._loopback_processor)
            tg.start_soon(self._plan)
        if self.global_event_sender:
            self.global_event_sender.close()
        if self.local_event_receiver:
            self.local_event_receiver.close()
        if self.command_receiver:
            self.command_receiver.close()
        if self._loopback_event_sender:
            self._loopback_event_sender.close()
        self._loopback_event_receiver.close()
        # Close worker clients
        if self._worker_clients:
            await self._worker_clients.close_all()

    async def shutdown(self):
        logger.info("Stopping Master")
        if self._tg and hasattr(self._tg, 'cancel_scope') and self._tg.cancel_scope:
            self._tg.cancel_scope.cancel()

    async def _command_processor(self) -> None:
        if self.command_receiver is None:
            return
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
                    # Apply events directly to state (for static setup without Router/loopback)
                    for event in generated_events:
                        indexed = IndexedEvent(event=event, idx=len(self._event_log))
                        self.state = apply(self.state, indexed)
                        event._master_time_stamp = datetime.now(tz=timezone.utc)  # pyright: ignore[reportPrivateUsage]
                        self._event_log.append(event)
                        await self.event_sender.send(event)
                    
                    # Push state update to all workers after applying events
                    if self._worker_clients and generated_events:
                        await self._worker_clients.push_state_to_all(self.state)
                except ValueError as e:
                    logger.opt(exception=e).warning("Error in command processor")

    # These plan loops are the cracks showing in our event sourcing architecture - more things could be commands
    async def _plan(self) -> None:
        while True:
            # kill broken instances
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

            # time out dead nodes
            for node_id, time in self.state.last_seen.items():
                now = datetime.now(tz=timezone.utc)
                if now - time > timedelta(seconds=30):
                    logger.info(f"Manually removing node {node_id} due to inactivity")
                    await self.event_sender.send(NodeTimedOut(node_id=node_id))

            await anyio.sleep(10)

    async def _event_processor(self) -> None:
        if self.local_event_receiver is None:
            return
        with self.local_event_receiver as local_events:
            async for local_event in local_events:
                # Discard all events not from our session
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
                
                # Push state update to all workers after applying event
                if self._worker_clients:
                    await self._worker_clients.push_state_to_all(self.state)

    async def _loopback_processor(self) -> None:
        # this would ideally not be necessary.
        # this is WAY less hacky than how I was working around this before
        if self._loopback_event_sender is None:
            return
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

    # This function is re-entrant, take care!
    async def _send_event(self, event: IndexedEvent):
        # Convenience method since this line is ugly
        if self.global_event_sender is None:
            return
        await self.global_event_sender.send(
            ForwarderEvent(
                origin=self.node_id,
                origin_idx=event.idx,
                session=self.session_id,
                event=event.event,
            )
        )
