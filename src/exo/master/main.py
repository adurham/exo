import time
from collections import deque
from datetime import datetime, timedelta, timezone
from itertools import islice

import anyio
from loguru import logger

from exo.master.placement import (
    add_instance_to_placements,
    cancel_unnecessary_downloads,
    delete_instance,
    get_transition_events,
    place_instance,
)
from exo.master.placement_utils import (
    find_ip_prioritised,
    jit_enabled,
    jit_idle_unload_seconds,
    jit_instances_to_reap,
)
from exo.shared.apply import apply
from exo.shared.constants import EXO_TRACING_ENABLED
from exo.shared.models.model_cards import ModelId
from exo.shared.types.chunks import ErrorChunk, ImageChunk
from exo.shared.types.commands import (
    AddCustomModelCard,
    CreateInstance,
    DeleteCustomModelCard,
    DeleteInstance,
    DeleteInstanceLink,
    ForwarderCommand,
    ForwarderDownloadCommand,
    ImageEdits,
    ImageGeneration,
    PlaceInstance,
    RequestEventLog,
    SendInputChunk,
    SetInstanceLink,
    TaskCancelled,
    TaskFinished,
    TestCommand,
    TextGeneration,
)
from exo.shared.types.common import CommandId, NodeId, SessionId, SystemId
from exo.shared.types.events import (
    ChunkGenerated,
    CustomModelCardAdded,
    CustomModelCardDeleted,
    Event,
    GlobalForwarderEvent,
    IndexedEvent,
    InputChunkReceived,
    InstanceDeleted,
    InstanceLinkCreated,
    InstanceLinkDeleted,
    LocalForwarderEvent,
    NodeGatheredInfo,
    NodeTimedOut,
    StateSnapshot,
    TaskCreated,
    TaskDeleted,
    TaskStatusUpdated,
    TraceEventData,
    TracesCollected,
    TracesMerged,
)
from exo.shared.types.instance_link import InstanceLink
from exo.shared.types.state import State
from exo.shared.types.tasks import (
    ImageEdits as ImageEditsTask,
)
from exo.shared.types.tasks import (
    ImageGeneration as ImageGenerationTask,
)
from exo.shared.types.tasks import (
    TaskId,
    TaskStatus,
)
from exo.shared.types.tasks import (
    TextGeneration as TextGenerationTask,
)
from exo.shared.types.worker.instances import InstanceId
from exo.utils.channels import Receiver, Sender
from exo.utils.event_buffer import MultiSourceBuffer
from exo.utils.log_format import truncate_for_log
from exo.utils.task_group import TaskGroup

# Every node reports its info about once a second. A node the master hasn't heard from for
# this long is treated as gone (crashed, frozen, asleep or cut off), and its instances are
# deleted so that requests on them end with an error instead of waiting. On four busy nodes
# (800 events/s) a healthy node's reports were never more than 6.3 s apart.
# adurham fork: kept at the previously deployed 30 s (upstream #2361 uses 15 s). Our
# nodes run very large models whose load/prefill can starve a worker's event loop
# for longer than upstream's measured 6.3 s; a false timeout deletes a live instance.
NODE_SILENCE_TIMEOUT = timedelta(seconds=30)
# How often the master looks for silent nodes and for instances that lost a node
PLAN_INTERVAL = 1.0
# How long to wait for a removal to be applied before sending it again
REMOVAL_RESEND_INTERVAL = 5.0
# A check this late means the master itself was stalled (frozen, asleep or overloaded) and
# couldn't hear anyone. Nodes then get time to reconnect (discovery holds off for a while
# after a stall) and a full NODE_SILENCE_TIMEOUT to report again before any is removed.
MASTER_STALL = 5.0
RECONNECT_ALLOWANCE = timedelta(seconds=15)
# Recent events kept for nodes that missed a few. A node further behind than this
# is sent a snapshot of the current state instead of the whole history.
REPLAYABLE_EVENTS = 10_000
# ...holding at most this much image data. Image chunks are the only large events, and a
# few requests with large images would otherwise make the window hold gigabytes.
REPLAYABLE_IMAGE_BYTES = 256 * 1024 * 1024
# Events replayed per RequestEventLog; the requester asks again for the rest.
REPLAY_BATCH_SIZE = 1000


def _image_bytes(event: Event) -> int:
    match event:
        case InputChunkReceived(chunk=chunk):
            return len(chunk.data)
        case ChunkGenerated(chunk=ImageChunk() as chunk):
            return len(chunk.data)
        case _:
            return 0


def task_for_command(state: State, command_id: CommandId) -> TaskId | None:
    """The generation task created for a command, as state has it. Unlike the master's own map
    from commands to tasks, state carries over to the next master."""
    for task_id, task in state.tasks.items():
        if (
            isinstance(task, (TextGenerationTask, ImageGenerationTask, ImageEditsTask))
            and task.command_id == command_id
        ):
            return task_id
    return None


def _prefill_endpoint_for(state: State, decode_instance_id: InstanceId) -> str | None:
    decode = state.instances.get(decode_instance_id)
    if decode is None:
        return None
    decode_node = next(iter(decode.shard_assignments.node_to_runner.keys()), None)
    if decode_node is None:
        return None

    sources: set[InstanceId] = set()
    for link in state.instance_links.values():
        if decode_instance_id in link.decode_instances:
            sources.update(link.prefill_instances)
    sources.discard(decode_instance_id)

    in_flight = {TaskStatus.Pending, TaskStatus.Running}
    task_counts: dict[InstanceId, int] = {
        src_id: sum(
            1
            for task in state.tasks.values()
            if task.instance_id == src_id and task.task_status in in_flight
        )
        for src_id in sources
    }
    for src_id in sorted(sources, key=lambda sid: task_counts[sid]):
        instance = state.instances.get(src_id)
        if instance is None:
            continue
        for node_id, runner_id in instance.shard_assignments.node_to_runner.items():
            port = state.prefill_server_ports.get(runner_id)
            if port is None:
                continue
            ip = find_ip_prioritised(
                decode_node, node_id, state.topology, state.node_network, ring=True
            )
            if ip is None:
                continue
            return f"{ip}:{port}"
    return None


def _without_requests(state: State) -> State:
    """The state without its requests' tasks. Requests don't outlast their master: every
    node's API ends the ones in flight when the master changes, since the old master was
    routing their output, so their tasks would only linger."""
    return state.model_copy(
        update={
            "tasks": {
                task_id: task
                for task_id, task in state.tasks.items()
                if not isinstance(
                    task, (TextGenerationTask, ImageGenerationTask, ImageEditsTask)
                )
            }
        }
    )


# How many processed command ids the master remembers, to act on a repeated command only once
PROCESSED_COMMANDS_KEPT = 100_000


class NoInstanceForModelError(Exception):
    """A request for a model with no running instance, such as one deleted after the API checked
    for it. The request is ended with this error: dropping it would leave it open forever."""

    def __init__(self, model: str):
        super().__init__(f"No instance found for model {model}")
        self.model = ModelId(model)


class Master:
    def __init__(
        self,
        node_id: NodeId,
        session_id: SessionId,
        *,
        command_receiver: Receiver[ForwarderCommand],
        local_event_receiver: Receiver[LocalForwarderEvent],
        global_event_sender: Sender[GlobalForwarderEvent],
        snapshot_sender: Sender[StateSnapshot],
        download_command_sender: Sender[ForwarderDownloadCommand],
        initial_state: State | None = None,
    ):
        self.node_id = node_id
        self.session_id = session_id
        # A master elected in a running cluster carries on from the state its node already
        # has, so model instances survive the change of master. Its events are numbered on
        # from that state, and other nodes catch up from a snapshot of it.
        self.state = (
            _without_requests(initial_state) if initial_state is not None else State()
        )
        self._tg: TaskGroup = TaskGroup()
        self.command_task_mapping: dict[CommandId, TaskId] = {}
        self.command_receiver = command_receiver
        self.local_event_receiver = local_event_receiver
        self.global_event_sender = global_event_sender
        self.snapshot_sender = snapshot_sender
        self.download_command_sender = download_command_sender
        self._system_id = SystemId()
        self._multi_buffer = MultiSourceBuffer[SystemId, Event]()
        self._recent_events: deque[Event] = deque()
        self._recent_image_bytes = 0
        self._replayable_events = REPLAYABLE_EVENTS
        self._replayable_image_bytes = REPLAYABLE_IMAGE_BYTES
        self._pending_traces: dict[TaskId, dict[int, list[TraceEventData]]] = {}
        # When a removal was last sent, for nodes and instances not yet removed from the state
        self._removing_nodes: dict[NodeId, float] = {}
        self._deleting_instances: dict[InstanceId, float] = {}
        # Since when the master has been able to hear nodes, and when it last checked on them
        self._listening_since = time.monotonic()
        self._last_check: float | None = None
        self._processed_commands: set[CommandId] = set()
        self._processed_order: deque[CommandId] = deque()
        self._expected_ranks: dict[TaskId, set[int]] = {}
        # JIT idle reaper: per-instance last-use wall-clock (anyio.current_time).
        # Bumped on every generation dispatch to that instance; the reaper unloads
        # JIT-placed instances idle beyond EXO_JIT_IDLE_UNLOAD_SECONDS.
        self._jit_instance_last_use: dict[InstanceId, float] = {}
        # Serializes event indexing/state mutation between command_processor
        # and event_processor. Without this, the command processor's routing
        # decisions would race against state updates and concurrent requests
        # would all see stale task counts and pile onto a single instance.
        self._state_lock = anyio.Lock()

    async def run(self):
        logger.info("Starting Master")

        try:
            async with self._tg as tg:
                tg.start_soon(self._event_processor)
                tg.start_soon(self._command_processor)
                tg.start_soon(self._plan)
                tg.start_soon(self._jit_idle_reaper)
        finally:
            self.global_event_sender.close()
            self.snapshot_sender.close()
            self.local_event_receiver.close()
            self.command_receiver.close()

    async def shutdown(self):
        logger.info("Stopping Master")
        self._tg.cancel_tasks()

    def _sort_by_load_then_node(
        self, instance_task_counts: dict[InstanceId, int]
    ) -> list[InstanceId]:
        """Sort candidate instances by (task_count, node_task_count).

        Primary key: fewest in-flight tasks on the instance (greedy LB).
        Secondary key: fewest in-flight tasks on the instance's physical
        node.  This breaks ties so that concurrent requests spread across
        nodes rather than piling onto whichever node has the
        lexicographically-first instance IDs.
        """
        in_flight = {TaskStatus.Pending, TaskStatus.Running}
        node_task_counts: dict[NodeId, int] = {}
        for task in self.state.tasks.values():
            if task.task_status not in in_flight:
                continue
            inst = self.state.instances.get(task.instance_id)
            if inst is not None:
                for nid in inst.shard_assignments.node_to_runner:
                    node_task_counts[nid] = node_task_counts.get(nid, 0) + 1

        def _key(instance_id: InstanceId) -> tuple[int, int]:
            inst = self.state.instances.get(instance_id)
            node_load = 0
            if inst is not None:
                for nid in inst.shard_assignments.node_to_runner:
                    node_load += node_task_counts.get(nid, 0)
            return (instance_task_counts[instance_id], node_load)

        return sorted(instance_task_counts.keys(), key=_key)

    def _first_time(self, command_id: CommandId) -> bool:
        """Whether this command hasn't been processed yet. The API sends a chat request again
        if it doesn't see it accepted, so the same command can arrive more than once. A new
        master doesn't know what the old one processed, but a request it accepted is a task."""
        if command_id in self._processed_commands or self._has_task_for(command_id):
            return False
        self._processed_commands.add(command_id)
        self._processed_order.append(command_id)
        if len(self._processed_order) > PROCESSED_COMMANDS_KEPT:
            self._processed_commands.discard(self._processed_order.popleft())
        return True

    def _has_task_for(self, command_id: CommandId) -> bool:
        return any(
            isinstance(task, (TextGenerationTask, ImageGenerationTask, ImageEditsTask))
            and task.command_id == command_id
            for task in self.state.tasks.values()
        )

    async def _command_processor(self) -> None:
        with self.command_receiver as commands:
            async for forwarder_command in commands:
                if not self._first_time(forwarder_command.command.command_id):
                    logger.debug(
                        f"Ignoring command {forwarder_command.command.command_id}: already processed"
                    )
                    continue
                try:
                    logger.info(
                        f"Executing command: {truncate_for_log(forwarder_command.command)}"
                    )

                    generated_events: list[Event] = []
                    command = forwarder_command.command
                    instance_task_counts: dict[InstanceId, int] = {}
                    match command:
                        case TestCommand():
                            pass
                        case TextGeneration():
                            in_flight = {TaskStatus.Pending, TaskStatus.Running}
                            prefill_only: set[InstanceId] = set()
                            for link in self.state.instance_links.values():
                                prefill_only.update(link.prefill_instances)
                            for link in self.state.instance_links.values():
                                prefill_only.difference_update(link.decode_instances)

                            for instance in self.state.instances.values():
                                if (
                                    instance.shard_assignments.model_id
                                    == command.task_params.model
                                    and instance.instance_id not in prefill_only
                                ):
                                    task_count = sum(
                                        1
                                        for task in self.state.tasks.values()
                                        if task.instance_id == instance.instance_id
                                        and task.task_status in in_flight
                                    )
                                    instance_task_counts[instance.instance_id] = (
                                        task_count
                                    )

                            if not instance_task_counts:
                                raise NoInstanceForModelError(command.task_params.model)

                            available_instance_ids = self._sort_by_load_then_node(
                                instance_task_counts
                            )

                            decode_instance_id = available_instance_ids[0]
                            self._jit_instance_last_use[decode_instance_id] = (
                                anyio.current_time()
                            )
                            task_id = TaskId()
                            params = command.task_params.model_copy(
                                update={
                                    "prefill_endpoint": _prefill_endpoint_for(
                                        self.state, decode_instance_id
                                    ),
                                }
                            )
                            generated_events.append(
                                TaskCreated(
                                    task_id=task_id,
                                    task=TextGenerationTask(
                                        task_id=task_id,
                                        command_id=command.command_id,
                                        instance_id=decode_instance_id,
                                        task_status=TaskStatus.Pending,
                                        task_params=params,
                                    ),
                                )
                            )
                            self.command_task_mapping[command.command_id] = task_id
                        case ImageGeneration():
                            in_flight = {TaskStatus.Pending, TaskStatus.Running}
                            for instance in self.state.instances.values():
                                if (
                                    instance.shard_assignments.model_id
                                    == command.task_params.model
                                ):
                                    in_flight = {TaskStatus.Pending, TaskStatus.Running}
                                    task_count = sum(
                                        1
                                        for task in self.state.tasks.values()
                                        if task.instance_id == instance.instance_id
                                        and task.task_status in in_flight
                                    )
                                    instance_task_counts[instance.instance_id] = (
                                        task_count
                                    )

                            if not instance_task_counts:
                                raise NoInstanceForModelError(command.task_params.model)

                            available_instance_ids = self._sort_by_load_then_node(
                                instance_task_counts
                            )

                            task_id = TaskId()
                            selected_instance_id = available_instance_ids[0]
                            generated_events.append(
                                TaskCreated(
                                    task_id=task_id,
                                    task=ImageGenerationTask(
                                        task_id=task_id,
                                        command_id=command.command_id,
                                        instance_id=selected_instance_id,
                                        task_status=TaskStatus.Pending,
                                        task_params=command.task_params,
                                    ),
                                )
                            )

                            self.command_task_mapping[command.command_id] = task_id

                            if EXO_TRACING_ENABLED:
                                selected_instance = self.state.instances.get(
                                    selected_instance_id
                                )
                                if selected_instance:
                                    ranks = set(
                                        shard.device_rank
                                        for shard in selected_instance.shard_assignments.runner_to_shard.values()
                                    )
                                    self._expected_ranks[task_id] = ranks
                        case ImageEdits():
                            in_flight = {TaskStatus.Pending, TaskStatus.Running}
                            for instance in self.state.instances.values():
                                if (
                                    instance.shard_assignments.model_id
                                    == command.task_params.model
                                ):
                                    in_flight = {TaskStatus.Pending, TaskStatus.Running}
                                    task_count = sum(
                                        1
                                        for task in self.state.tasks.values()
                                        if task.instance_id == instance.instance_id
                                        and task.task_status in in_flight
                                    )
                                    instance_task_counts[instance.instance_id] = (
                                        task_count
                                    )

                            if not instance_task_counts:
                                raise NoInstanceForModelError(command.task_params.model)

                            available_instance_ids = self._sort_by_load_then_node(
                                instance_task_counts
                            )

                            task_id = TaskId()
                            selected_instance_id = available_instance_ids[0]
                            generated_events.append(
                                TaskCreated(
                                    task_id=task_id,
                                    task=ImageEditsTask(
                                        task_id=task_id,
                                        command_id=command.command_id,
                                        instance_id=selected_instance_id,
                                        task_status=TaskStatus.Pending,
                                        task_params=command.task_params,
                                    ),
                                )
                            )

                            self.command_task_mapping[command.command_id] = task_id

                            if EXO_TRACING_ENABLED:
                                selected_instance = self.state.instances.get(
                                    selected_instance_id
                                )
                                if selected_instance:
                                    ranks = set(
                                        shard.device_rank
                                        for shard in selected_instance.shard_assignments.runner_to_shard.values()
                                    )
                                    self._expected_ranks[task_id] = ranks
                        case DeleteInstance():
                            placement = delete_instance(command, self.state.instances)
                            transition_events = get_transition_events(
                                self.state.instances, placement, self.state.tasks
                            )
                            for cmd in cancel_unnecessary_downloads(
                                placement, self.state.downloads
                            ):
                                await self.download_command_sender.send(
                                    ForwarderDownloadCommand(
                                        origin=self._system_id, command=cmd
                                    )
                                )
                            generated_events.extend(transition_events)
                        case PlaceInstance():
                            placement = place_instance(
                                command,
                                self.state.topology,
                                self.state.instances,
                                self.state.node_memory,
                                self.state.node_network,
                                self.state.node_backends,
                                download_status=self.state.downloads,
                                node_rdma_ctl=self.state.node_rdma_ctl,
                                runners=self.state.runners,
                            )
                            transition_events = get_transition_events(
                                self.state.instances, placement, self.state.tasks
                            )
                            generated_events.extend(transition_events)
                        case CreateInstance():
                            placement = add_instance_to_placements(
                                command,
                                self.state.topology,
                                self.state.instances,
                            )
                            transition_events = get_transition_events(
                                self.state.instances, placement, self.state.tasks
                            )
                            generated_events.extend(transition_events)
                        case SendInputChunk(chunk=chunk):
                            generated_events.append(
                                InputChunkReceived(
                                    command_id=chunk.command_id,
                                    chunk=chunk,
                                )
                            )
                        case TaskCancelled():
                            if (
                                task_id := self._task_for_command(
                                    command.cancelled_command_id
                                )
                            ) is not None:
                                generated_events.append(
                                    TaskStatusUpdated(
                                        task_status=TaskStatus.Cancelled,
                                        task_id=task_id,
                                    )
                                )
                            else:
                                logger.warning(
                                    f"Nonexistent command {command.cancelled_command_id} cancelled"
                                )
                        case TaskFinished():
                            # The API resends a finish while the task is still in its state,
                            # so a finish for a task already deleted is expected
                            if (
                                task_id := self._task_for_command(
                                    command.finished_command_id
                                )
                            ) is not None:
                                self.command_task_mapping.pop(
                                    command.finished_command_id, None
                                )
                                generated_events.append(TaskDeleted(task_id=task_id))

                        case AddCustomModelCard():
                            generated_events.append(
                                CustomModelCardAdded(model_card=command.model_card)
                            )
                        case DeleteCustomModelCard():
                            generated_events.append(
                                CustomModelCardDeleted(model_id=command.model_id)
                            )
                        case SetInstanceLink():
                            link = InstanceLink(
                                link_id=command.link_id,
                                prefill_instances=list(
                                    dict.fromkeys(command.prefill_instances)
                                ),
                                decode_instances=list(
                                    dict.fromkeys(command.decode_instances)
                                ),
                            )
                            generated_events.append(InstanceLinkCreated(link=link))
                        case DeleteInstanceLink():
                            generated_events.append(
                                InstanceLinkDeleted(link_id=command.link_id)
                            )
                        case RequestEventLog():
                            await self._serve_event_log_request(
                                command.since_idx,
                                forwarder_command.origin,
                                snapshot=command.snapshot,
                            )
                    for event in generated_events:
                        await self._index_apply_broadcast(event)
                except NoInstanceForModelError as error:
                    command_id = forwarder_command.command.command_id
                    logger.warning(f"Ending request {command_id}: {error}")
                    await self._index_apply_broadcast(
                        ChunkGenerated(
                            command_id=command_id,
                            chunk=ErrorChunk(
                                model=error.model, error_message=str(error)
                            ),
                        )
                    )
                except ValueError as e:
                    logger.opt(exception=e).warning("Error in command processor")

    def _task_for_command(self, command_id: CommandId) -> TaskId | None:
        """The task this cluster created for a request. The mapping holds only what this master
        created; a request started under a previous master is found in the state, so it can
        still be finished or cancelled after the master changes."""
        if (task_id := self.command_task_mapping.get(command_id)) is not None:
            return task_id
        return next(
            (
                task_id
                for task_id, task in self.state.tasks.items()
                if isinstance(
                    task, (TextGenerationTask, ImageGenerationTask, ImageEditsTask)
                )
                and task.command_id == command_id
            ),
            None,
        )

    # These plan loops are the cracks showing in our event sourcing architecture - more things could be commands
    async def _plan(self) -> None:
        while True:
            await self._remove_silent_nodes_and_broken_instances()
            await anyio.sleep(PLAN_INTERVAL)

    async def _remove_silent_nodes_and_broken_instances(self) -> None:
        check = time.monotonic()
        if self._last_check is not None and check - self._last_check > MASTER_STALL:
            logger.warning(
                f"Master was unresponsive for {check - self._last_check:.0f}s; "
                "giving nodes time to report before removing any"
            )
            self._listening_since = check + RECONNECT_ALLOWANCE.total_seconds()
        self._last_check = check
        # A node is silent only if the master was listening and didn't hear from it
        listened = timedelta(seconds=check - self._listening_since)
        now = datetime.now(tz=timezone.utc)
        silent = {
            node_id: now - seen
            for node_id, seen in self.state.last_seen.items()
            if min(now - seen, listened) > NODE_SILENCE_TIMEOUT
        }
        for node_id, silence in silent.items():
            if self._resend_due(self._removing_nodes, node_id):
                logger.info(
                    f"Manually removing node {node_id} due to inactivity "
                    f"({silence.total_seconds():.0f}s without hearing from it)"
                )
                await self._index_apply_broadcast(NodeTimedOut(node_id=node_id))

        # An instance that lost a node can't serve anything, so delete it straight away,
        # without waiting for the node's removal to be applied
        connected = set(self.state.topology.list_nodes()) - silent.keys()
        for instance_id, instance in self.state.instances.items():
            if any(
                node_id not in connected
                for node_id in instance.shard_assignments.node_to_runner
            ) and self._resend_due(self._deleting_instances, instance_id):
                await self._index_apply_broadcast(
                    InstanceDeleted(instance_id=instance_id)
                )

        # Forget what has been applied
        self._removing_nodes = {
            k: v for k, v in self._removing_nodes.items() if k in self.state.last_seen
        }
        self._deleting_instances = {
            k: v
            for k, v in self._deleting_instances.items()
            if k in self.state.instances
        }

    @staticmethod
    def _resend_due[K](sent: dict[K, float], key: K) -> bool:
        """Whether to send a removal for key: not sent yet, or sent a while ago and still
        not applied."""
        now = time.monotonic()
        if key in sent and now - sent[key] < REMOVAL_RESEND_INTERVAL:
            return False
        sent[key] = now
        return True

    async def _jit_idle_reaper(self) -> None:
        """Unload JIT-placed instances idle beyond EXO_JIT_IDLE_UNLOAD_SECONDS.

        Safety invariants (see JIT handoff doc, Step 3):
        - Only ``instance.jit`` instances are ever unloaded; the interactive /
          explicitly-placed model (jit=False) is immune ("pinned").
        - Never unloads an instance with in-flight work: it skips any instance
          with a Pending/Running task, which closes the check→delete race (a
          request arriving between the idle check and the delete creates a
          Pending task that this guard sees on the next pass, and the delete is
          only issued when both the timer AND the in-flight count agree).
        - Dormant unless EXO_JIT_ENABLED (the feature kill-switch).
        """
        in_flight = {TaskStatus.Pending, TaskStatus.Running}
        while True:
            await anyio.sleep(5)
            if not jit_enabled():
                continue

            idle_window = jit_idle_unload_seconds()
            now = anyio.current_time()

            # In-flight task count per instance (event-sourced, authoritative).
            inflight_by_instance: dict[InstanceId, int] = {}
            for task in self.state.tasks.values():
                if task.task_status in in_flight:
                    inflight_by_instance[task.instance_id] = (
                        inflight_by_instance.get(task.instance_id, 0) + 1
                    )

            live_instance_ids = set(self.state.instances.keys())
            # Drop bookkeeping for instances that no longer exist.
            for stale_id in set(self._jit_instance_last_use) - live_instance_ids:
                del self._jit_instance_last_use[stale_id]

            # Seed a last-use clock for any JIT instance we haven't observed yet
            # (placed but never used) and bump it for ones with active work, so
            # the idle window is measured from first observation / last activity.
            for instance_id, instance in self.state.instances.items():
                if not instance.jit:
                    continue
                if (
                    inflight_by_instance.get(instance_id, 0) > 0
                    or instance_id not in self._jit_instance_last_use
                ):
                    self._jit_instance_last_use[instance_id] = now

            to_reap = jit_instances_to_reap(
                self.state.instances,
                inflight_by_instance,
                self._jit_instance_last_use,
                now,
                idle_window,
            )
            for instance_id in to_reap:
                instance = self.state.instances.get(instance_id)
                if instance is None:
                    continue
                # Re-check in-flight immediately before deleting. A request can
                # arrive between the reap decision above and here (the loop body
                # awaits), creating a Pending task — skip the unload if so. The
                # delete is computed + broadcast with NO await in between, so no
                # request can slip in after this guard. This plus the per-pass
                # decision is the in-flight race guard.
                if any(
                    task.instance_id == instance_id and task.task_status in in_flight
                    for task in self.state.tasks.values()
                ):
                    self._jit_instance_last_use[instance_id] = now
                    continue
                model_id = instance.shard_assignments.model_id
                last_use = self._jit_instance_last_use.get(instance_id, now)
                logger.info(
                    f"JIT idle reaper unloading instance {instance_id} "
                    f"(model {model_id}, idle {now - last_use:.0f}s "
                    f">= {idle_window:.0f}s)"
                )
                placement = delete_instance(
                    DeleteInstance(instance_id=instance_id), self.state.instances
                )
                transition_events = get_transition_events(
                    self.state.instances, placement, self.state.tasks
                )
                download_cancels = cancel_unnecessary_downloads(
                    placement, self.state.downloads
                )
                for event in transition_events:
                    await self._index_apply_broadcast(event)
                for cmd in download_cancels:
                    await self.download_command_sender.send(
                        ForwarderDownloadCommand(origin=self._system_id, command=cmd)
                    )
                self._jit_instance_last_use.pop(instance_id, None)

    async def _event_processor(self) -> None:
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
                    if isinstance(event, TracesCollected):
                        await self._handle_traces_collected(event)
                        continue

                    await self._index_apply_broadcast(event)

    def _remember(self, event: Event) -> None:
        self._recent_events.append(event)
        self._recent_image_bytes += _image_bytes(event)
        while (
            len(self._recent_events) > self._replayable_events
            or self._recent_image_bytes > self._replayable_image_bytes
        ):
            self._recent_image_bytes -= _image_bytes(self._recent_events.popleft())

    async def _serve_event_log_request(
        self, since_idx: int, requester: SystemId, snapshot: bool = False
    ):
        next_idx = self.state.last_event_applied_idx + 1
        oldest_idx = next_idx - len(self._recent_events)
        if snapshot or since_idx < oldest_idx:
            logger.info(
                f"Sending a state snapshot at event {next_idx - 1} to a node "
                f"that asked for events from {since_idx}"
            )
            await self.snapshot_sender.send(
                StateSnapshot(
                    session=self.session_id, requester=requester, state=self.state
                )
            )
            return
        # Copy before sending: new events evict old ones from the deque while we await.
        end = min(since_idx + REPLAY_BATCH_SIZE, next_idx)
        events = list(
            islice(self._recent_events, since_idx - oldest_idx, end - oldest_idx)
        )
        # Other nodes ignore events they already have
        for idx, event in enumerate(events, start=since_idx):
            await self._send_indexed_event(IndexedEvent(idx=idx, event=event))

    # This function is re-entrant, take care!
    async def _send_indexed_event(self, event: IndexedEvent) -> None:
        # Convenience method since this line is ugly
        await self.global_event_sender.send(
            GlobalForwarderEvent(
                origin=self.node_id,
                origin_idx=event.idx,
                session=self.session_id,
                event=event.event,
            )
        )

    async def _index_apply_broadcast(self, event: Event) -> IndexedEvent:
        """Index, apply, persist, and broadcast a single event atomically.

        This is the master's only path for committing events into state. It
        is called from both command_processor (for events the master itself
        generates) and event_processor (for events received from workers).
        The lock serializes the read-modify-write of the event index
        (state.last_event_applied_idx) and `state`, which is what allows the command processor to make routing
        decisions based on its own just-emitted events without waiting for a
        network round-trip.
        """
        async with self._state_lock:
            event = event.model_copy(
                update={"_master_time_stamp": datetime.now(tz=timezone.utc)}
            )
            if isinstance(event, NodeGatheredInfo):
                event = event.model_copy(
                    update={"when": str(datetime.now(tz=timezone.utc))}
                )

            # Numbered on from state (not a per-master log) so a master elected
            # with initial_state continues the cluster's event numbering (#2369).
            indexed = IndexedEvent(
                event=event, idx=self.state.last_event_applied_idx + 1
            )
            logger.debug(f"Master indexing event: {str(event)[:100]}")
            self.state = apply(self.state, indexed)
            # Bounded in-memory replay window replaces the DiskEventLog (#2322)
            self._remember(event)
            await self._send_indexed_event(indexed)
            return indexed

    async def _handle_traces_collected(self, event: TracesCollected) -> None:
        task_id = event.task_id
        if task_id not in self._pending_traces:
            self._pending_traces[task_id] = {}
        self._pending_traces[task_id][event.rank] = event.traces

        if (
            task_id in self._expected_ranks
            and set(self._pending_traces[task_id].keys())
            >= self._expected_ranks[task_id]
        ):
            await self._merge_and_save_traces(task_id)

    async def _merge_and_save_traces(self, task_id: TaskId) -> None:
        all_trace_data: list[TraceEventData] = []
        for trace_data in self._pending_traces[task_id].values():
            all_trace_data.extend(trace_data)

        await self._index_apply_broadcast(
            TracesMerged(task_id=task_id, traces=all_trace_data)
        )

        del self._pending_traces[task_id]
        if task_id in self._expected_ranks:
            del self._expected_ranks[task_id]
