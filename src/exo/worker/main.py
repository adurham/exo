import hashlib
import os
import time
from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Final

import anyio
from anyio import fail_after, to_thread
from loguru import logger

from exo.api.types import ImageEditsTaskParams
from exo.download.download_utils import is_read_only_model_dir, resolve_existing_model
from exo.routing.event_router import (
    EventRouterBrokenResourceError,
    EventRouterClosedResourceError,
    NodeEventReceiver,
)
from exo.shared.apply import apply
from exo.shared.constants import EXO_MAX_INSTANCE_RETRIES
from exo.shared.models.model_cards import ModelCard, ModelId, card_cache
from exo.shared.types.chunks import ErrorChunk, InputImageChunk
from exo.shared.types.commands import (
    AddCustomModelCard,
    DeleteInstance,
    ForwarderCommand,
    ForwarderDownloadCommand,
    StartDownload,
)
from exo.shared.types.common import CommandId, NodeId, SessionId, SystemId
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    InputChunkReceived,
    InstanceDeleted,
    NodeDownloadProgress,
    NodeGatheredInfo,
    RunnerStatusUpdated,
    StateSnapshot,
    TaskCreated,
    TaskDeleted,
    TaskStatusUpdated,
    TopologyEdgeCreated,
    TopologyEdgeDeleted,
)
from exo.shared.types.multiaddr import Multiaddr
from exo.shared.types.state import State
from exo.shared.types.tasks import (
    CancelTask,
    CreateRunner,
    DownloadModel,
    ImageEdits,
    ImageGeneration,
    LoadModel,
    Shutdown,
    Task,
    TaskId,
    TaskStatus,
    TextGeneration,
)
from exo.shared.types.text_generation import Base64Image, Base64ImageHash
from exo.shared.types.topology import Connection, SocketConnection
from exo.shared.types.worker.downloads import DownloadCompleted
from exo.shared.types.worker.instances import InstanceId
from exo.shared.types.worker.runners import RunnerId, RunnerReady
from exo.utils.channels import Receiver, Sender, channel
from exo.utils.info_gatherer.info_gatherer import GatheredInfo, InfoGatherer
from exo.utils.info_gatherer.net_profile import check_reachable
from exo.utils.keyed_backoff import KeyedBackoff
from exo.utils.task_group import TaskGroup
from exo.worker.phase_marks import (
    MARKS_ENABLED,
    WakeKind,
    mark_plan_step_observed,
    mark_state_applied,
)
from exo.worker.plan import plan
from exo.worker.runner.supervisor import RunnerSupervisor

# The historical `plan_step` loop-top poll interval. Under the default (gate
# OFF) path this is the plain sleep it always was; under the event-wake path it
# becomes the FALLBACK timeout on the wait, so a missed signal degrades to
# exactly this polling behaviour rather than hanging.
_PLAN_TICK_SECONDS: Final[float] = 0.1

# I16 (round 12): event-triggered wake for `plan_step`, replacing the
# unconditional 100ms tick at the top of its loop. Read ONCE at import into a
# module-level `Final[bool]` -- never `os.environ` in the loop body -- mirroring
# `exo.api.phase_marks` / `exo.worker.engines.mlx.phase_marks`. Default OFF:
# unset or "0" leaves the loop-top wait byte-for-byte the historical
# `await anyio.sleep(0.1)`.
_PLAN_EVENT_WAKE_ENABLED: Final[bool] = os.environ.get(
    "EXO_WORKER_PLAN_EVENT_WAKE", ""
) not in (
    "",
    "0",
    "false",
    "False",
)


def _may_dispatch_non_blocking(task: Task) -> bool:
    """Whether `plan_step` may hand this task to `start_soon` instead of awaiting it.

    True ONLY for the generation tasks, whose `task_id` is STABLE (they are
    created by the master and delivered through cluster state, not built fresh
    by `plan()`). A repeat send of one of those is deduped by the runner's
    `seen` set and the supervisor's `pending` / `in_progress` guards, so it is
    safe for the planner to re-evaluate `plan()` while one is in flight -- which
    is exactly what the c>=2 batched-prefill rendezvous needs (see the caller).

    False for every LIFECYCLE task (`ConnectToGroup`, `StartWarmup`, ...).
    Those are rebuilt by `plan()` with a NEW `task_id` on each call, so nothing
    task_id-keyed can suppress a duplicate; only `plan()`'s own state
    precondition can, and that depends on a status the runner publishes
    asynchronously. They MUST be awaited so the dispatch and the guard status
    cannot interleave (I16 regression, 2026-09-15).

    Written as an explicit runtime predicate rather than an OR-pattern match
    arm because basedpyright's pattern narrowing overflows the stack on an
    OR-pattern in this match (confirmed against this file, v1.29.0).
    """
    return isinstance(task, (TextGeneration, ImageGeneration, ImageEdits))

# How often the custom model cards saved on this node are synced with the cluster state,
# and how often the ones the cluster doesn't know yet are announced to the master.
CUSTOM_CARD_SYNC_INTERVAL = 1.0
CUSTOM_CARD_ANNOUNCE_INTERVAL = 10.0
# How long the worker waits for an instance it asked to delete to go, before asking again
DELETION_REQUEST_RETRY = 10.0


class Worker:
    def __init__(
        self,
        node_id: NodeId,
        *,
        event_receiver: NodeEventReceiver,
        event_sender: Sender[Event],
        # This is for requesting updates. It doesn't need to be a general command sender right now,
        # but I think it's the correct way to be thinking about commands
        command_sender: Sender[ForwarderCommand],
        download_command_sender: Sender[ForwarderDownloadCommand],
        api_port: int,
    ):
        self.node_id: NodeId = node_id
        self.event_receiver = event_receiver
        self.event_sender = event_sender
        self.command_sender = command_sender
        self.download_command_sender = download_command_sender
        self.api_port = api_port

        self.state: State = State()
        self.runners: dict[RunnerId, RunnerSupervisor] = {}
        self._tg: TaskGroup = TaskGroup()

        self._system_id = SystemId()

        # Images a request sends ahead of itself, kept per request until it runs here or
        # turns out not to need this node
        self.input_chunk_buffer: dict[CommandId, dict[int, InputImageChunk]] = {}
        self.input_chunk_counts: dict[CommandId, int] = {}
        self.input_images: dict[CommandId, dict[Base64ImageHash, Base64Image]] = {}
        self._tasks_with_inputs: dict[TaskId, CommandId] = {}
        self._inputs_lost: set[TaskId] = set()

        self._download_backoff: KeyedBackoff[ModelId] = KeyedBackoff(base=0.5, cap=10.0)
        self._instance_backoff: KeyedBackoff[InstanceId] = KeyedBackoff(
            base=0.5, cap=10.0
        )
        # When this worker last asked for an instance it gave up on to be deleted
        self._deletion_requested: dict[InstanceId, float] = {}
        self._stopped: anyio.Event = anyio.Event()

        # The latest information of each kind gathered about this node, so that a new
        # master can be told all of it: some of it is only gathered once
        self._gathered: dict[type[GatheredInfo], GatheredInfo] = {}
        # After a change of master, the session whose state this worker waits for before
        # acting on the cluster state again
        self._awaiting_state_of: SessionId | None = None
        # The session of the last state snapshot applied
        self._snapshot_session: SessionId | None = None

        # I16: signalled by `_event_applier` after EVERY state apply, awaited at
        # the top of `plan_step`. `anyio.Event` has no `clear()`, so the wake
        # path REPLACES this object with a fresh Event rather than resetting it
        # (see `_signal_state_applied` / `_wait_for_state_change`).
        self._state_applied: anyio.Event = anyio.Event()

        # Round-13 Gate-A phase marks: the `IndexedEvent.idx` most recently
        # applied by `_event_applier`, used as the pairing key between the
        # `state_applied` mark (emitted at apply time) and the
        # `plan_step_observed` mark (emitted when `plan_step` wakes and acts
        # on it). Written unconditionally (cheap int assignment) so the field
        # always reflects reality regardless of the marks gate; only the
        # `mark_*` calls themselves are gated on EXO_PHASE_MARKS.
        self._last_applied_event_idx: int = -1

    def _signal_state_applied(self) -> None:
        """Wake `plan_step`. MUST be called AFTER the state mutation it reports.

        `anyio.Event` has no `clear()`, so this swaps in a FRESH Event before
        setting the old one: every waiter already parked on the old object is
        released, and the next capture sees a new, unset object.

        No-op under the gate so the default path allocates nothing.
        """
        if not _PLAN_EVENT_WAKE_ENABLED:
            return
        previously_waited_on = self._state_applied
        self._state_applied = anyio.Event()
        previously_waited_on.set()

    async def _wait_for_state_change(self, waiting_on: anyio.Event) -> WakeKind:
        """The `plan_step` loop-top wait.

        Gate OFF (default): exactly the historical plain
        ``await anyio.sleep(0.1)``; `waiting_on` is not consulted at all.
        Returns ``"timeout"`` unconditionally in this branch -- true by
        construction, since the only way this branch returns is the sleep
        elapsing.

        Gate ON: park on the event the caller captured BEFORE its last
        `plan()` state read, bounded by the same 0.1s as a FALLBACK timeout.
        A missed signal therefore degrades to exactly today's 100ms polling
        rather than hanging.

        `cancelled_caught` ALONE cannot classify the wake: if the event's
        `set()` and the `move_on_after` deadline land in the same scheduling
        window, the fallback's cancellation can still be delivered to this
        task even though `waiting_on` WAS set and the wakeup WAS, in effect,
        event-driven -- `cancelled_caught` would read True regardless. So
        `waiting_on.is_set()` is ALSO read, immediately after the wait
        returns, and the two booleans together give an honest 3-way
        classification (see `WakeKind` in `phase_marks.py` for the full
        truth table): a true timeout only when the event was never set.
        """
        if not _PLAN_EVENT_WAKE_ENABLED:
            await anyio.sleep(_PLAN_TICK_SECONDS)
            return "timeout"

        with anyio.move_on_after(_PLAN_TICK_SECONDS) as scope:
            await waiting_on.wait()
        event_was_set = waiting_on.is_set()
        if not event_was_set:
            return "timeout"
        return "event_raced_timeout" if scope.cancelled_caught else "event"

    async def run(self):
        logger.info("Starting Worker")

        info_send, info_recv = channel[GatheredInfo]()
        info_gatherer: InfoGatherer = InfoGatherer(info_send)

        try:
            async with self._tg as tg:
                tg.start_soon(info_gatherer.run)
                tg.start_soon(self._forward_info, info_recv)
                tg.start_soon(self.plan_step)
                tg.start_soon(self._event_applier)
                tg.start_soon(self._poll_connection_updates)
                tg.start_soon(self._reconcile_custom_cards)
        except* (EventRouterBrokenResourceError, EventRouterClosedResourceError):
            # Event router has been closed (try-star syntax handles error groups)
            pass
        finally:
            # Actual shutdown code - waits for all tasks to complete before executing.
            logger.info("Stopping Worker")
            self.event_sender.close()
            self.command_sender.close()
            self.download_command_sender.close()
            for runner in self.runners.values():
                runner.shutdown()
            self._stopped.set()

    async def _forward_info(self, recv: Receiver[GatheredInfo]):
        with recv as info_stream:
            async for info in info_stream:
                self._gathered[type(info)] = info
                await self._send_info(info)

    async def _send_info(self, info: GatheredInfo) -> None:
        await self.event_sender.send(
            NodeGatheredInfo(
                node_id=self.node_id,
                when=str(datetime.now(tz=timezone.utc)),
                info=info,
            )
        )

    async def _event_applier(self):
        with self.event_receiver as events:
            async for update in events:
                if isinstance(update, StateSnapshot):
                    self.state = update.state
                    self._snapshot_session = update.session
                    if update.session == self._awaiting_state_of:
                        self._awaiting_state_of = None
                        # Every request from before the change of master has ended
                        self._drop_all_inputs()
                    # A snapshot replaces the whole state: wake plan_step too (I16)
                    self._signal_state_applied()
                    continue
                # 2. for each event, apply it to the state
                self.state = apply(self.state, event=update)

                # Round-13 Gate-A mark 1 (state-update-applied): must be
                # emitted right after the mutation above, while the
                # `IndexedEvent.idx` (the pairing key) is still in hand.
                # `_last_applied_event_idx` is also updated here,
                # unconditionally, for `plan_step` to read at wake time
                # regardless of whether marks are enabled.
                self._last_applied_event_idx = update.idx
                mark_state_applied(update.idx)

                event = update.event

                if isinstance(event, InstanceDeleted):
                    self._instance_backoff.reset(event.instance_id)
                    self._deletion_requested.pop(event.instance_id, None)

                # Our runner for an instance has started: the attempts before it no longer
                # count towards giving up on the instance, which is for one that can't start
                if isinstance(event, RunnerStatusUpdated) and isinstance(
                    event.runner_status, RunnerReady
                ):
                    for instance in self.state.instances.values():
                        node_to_runner = instance.shard_assignments.node_to_runner
                        if node_to_runner.get(self.node_id) == event.runner_id:
                            self._instance_backoff.reset(instance.instance_id)

                # Buffer input image chunks for image editing
                if isinstance(event, InputChunkReceived):
                    cmd_id = event.command_id
                    if cmd_id not in self.input_chunk_buffer:
                        self.input_chunk_buffer[cmd_id] = {}
                        self.input_chunk_counts[cmd_id] = event.chunk.total_chunks

                    self.input_chunk_buffer[cmd_id][event.chunk.chunk_index] = (
                        event.chunk
                    )
                    if (
                        len(self.input_chunk_buffer[cmd_id])
                        == self.input_chunk_counts[cmd_id]
                    ):
                        per_image: defaultdict[int, list[InputImageChunk]] = (
                            defaultdict(list)
                        )
                        for chunk in self.input_chunk_buffer[cmd_id].values():
                            per_image[chunk.image_index].append(chunk)
                        for chunks_for_image in per_image.values():
                            sorted_chunks = sorted(
                                chunks_for_image, key=lambda c: c.chunk_index
                            )
                            img = Base64Image("".join(c.data for c in sorted_chunks))
                            self.input_images.setdefault(cmd_id, {})[
                                Base64ImageHash(
                                    hashlib.sha256(img.encode("ascii")).hexdigest()
                                )
                            ] = img

                if isinstance(event, TaskCreated) and isinstance(
                    event.task, (TextGeneration, ImageEdits)
                ):
                    self._keep_inputs_if_needed_here(event.task)

                if isinstance(event, TaskDeleted):
                    self._inputs_lost.discard(event.task_id)
                    if (
                        cmd_id := self._tasks_with_inputs.pop(event.task_id, None)
                    ) is not None:
                        self._drop_inputs(cmd_id)

                # I16: the SINGLE per-event state-apply signal site. Placed at the
                # END of the loop body so EVERY mutation this event makes --
                # `self.state` above, plus the input-image buffers -- has already
                # landed before the wake fires (mutate first, THEN set). Same task
                # group and therefore the same event loop and thread as `plan_step`
                # (both are `tg.start_soon`-ed in `run`), so a bare `set()` is
                # correct here; no cross-thread scheduling hop is required.
                self._signal_state_applied()

    def _keep_inputs_if_needed_here(self, task: TextGeneration | ImageEdits) -> None:
        """Every node receives every request's images; only the nodes that run it keep them."""
        instance = self.state.instances.get(task.instance_id)
        if (
            instance is None
            or self.node_id not in instance.shard_assignments.node_to_runner
        ):
            self._drop_inputs(task.command_id)
        elif task.command_id in self.input_chunk_buffer:
            self._tasks_with_inputs[task.task_id] = task.command_id

    def _drop_all_inputs(self) -> None:
        self.input_chunk_buffer.clear()
        self.input_chunk_counts.clear()
        self.input_images.clear()
        self._tasks_with_inputs.clear()
        self._inputs_lost.clear()

    def _drop_inputs(self, command_id: CommandId) -> None:
        self.input_chunk_buffer.pop(command_id, None)
        self.input_chunk_counts.pop(command_id, None)
        self.input_images.pop(command_id, None)

    async def _report_lost_inputs(self, task: TextGeneration | ImageEdits) -> None:
        """Fail a request whose images never reached this node.

        A request's images always arrive before the request itself, so if they are missing
        once it is here, they are not coming: this node caught up from a state snapshot part
        way through them. Waiting would hold the request, and the model's instance, forever.
        """
        if task.task_id in self._inputs_lost:
            return
        self._inputs_lost.add(task.task_id)
        self._drop_inputs(task.command_id)
        logger.warning(f"Images for request {task.command_id} never reached this node")
        await self.event_sender.send(
            ChunkGenerated(
                command_id=task.command_id,
                chunk=ErrorChunk(
                    model=ModelId(task.task_params.model),
                    error_message=(
                        "The request's images didn't reach every device running the "
                        "model. Please send it again."
                    ),
                ),
            )
        )
        await self.event_sender.send(
            TaskStatusUpdated(task_id=task.task_id, task_status=TaskStatus.Failed)
        )

    async def _reconcile_custom_cards(self) -> None:
        """Keep the custom model cards saved on this node in sync with the cluster.

        The cluster state starts empty whenever a new master takes over (for example after
        the whole cluster restarts), so the cards saved on the nodes are the only lasting
        copy. Cards this node has saved are announced until the cluster knows them, and a
        saved card is only deleted after it was in the state and then removed from it: a
        state that is empty or still catching up never deletes anything.
        """
        unannounced = {card.model_id: card for card in await card_cache.list_custom()}
        synced: Mapping[ModelId, ModelCard] = {}
        next_announcement = 0.0
        while True:
            await anyio.sleep(CUSTOM_CARD_SYNC_INTERVAL)
            target = self.state.custom_model_cards
            for model_id, card in target.items():
                unannounced.pop(model_id, None)
                if synced.get(model_id) != card:
                    await card_cache.save(card)
            for model_id in synced.keys() - target.keys():
                await card_cache.pop(model_id)
            synced = target

            if unannounced and anyio.current_time() >= next_announcement:
                next_announcement = anyio.current_time() + CUSTOM_CARD_ANNOUNCE_INTERVAL
                for card in unannounced.values():
                    await self.command_sender.send(
                        ForwarderCommand(
                            origin=self._system_id,
                            command=AddCustomModelCard(model_card=card),
                        )
                    )

    async def plan_step(self):
        # Captured BEFORE the first `plan()` state read, and re-captured at the
        # same point on every iteration. A signal that lands after the capture
        # but before/during `plan()` sets THIS object, so the next wait returns
        # immediately instead of parking for the full fallback -- the
        # lost-wakeup window is closed by the capture ordering, not by luck.
        waiting_on: anyio.Event = self._state_applied
        while True:
            wake_kind = await self._wait_for_state_change(waiting_on)
            waiting_on = self._state_applied
            # Read right after the wait returns, before `plan()` runs, so it
            # reflects the state this `plan()` call is actually about to
            # read (a concurrent apply landing mid-`plan()` would otherwise
            # attribute the wrong event_idx to this dispatch).
            observed_event_idx = self._last_applied_event_idx
            # Round-13 Gate-A mark 2 (plan_step-observed) timestamp: captured
            # HERE, immediately after the wait returns and BEFORE `plan()`
            # runs, so the recorded `t=` represents THE WAKE itself, not the
            # wake plus `plan()`'s runtime. `plan()` (src/exo/worker/plan.py)
            # is a plain synchronous function -- no `await`, no I/O, no
            # subprocess/sleep calls anywhere in it, just in-memory dict/tuple
            # lookups and Python-level branching over already-fetched state --
            # so capturing the timestamp after plan() would still be safe in
            # practice, but capturing it here removes the question entirely
            # and costs nothing.
            #
            # Gated on `MARKS_ENABLED` -- the same module-level constant
            # `phase_marks.mark_plan_step_observed` itself checks -- so that
            # with EXO_PHASE_MARKS unset this line is an inline boolean
            # check and nothing else, preserving the OFF-path invariant: NOT
            # a wasted `time.perf_counter()` call that is computed only to
            # be discarded by that function's own `if not _MARKS_ENABLED:
            # return`.
            wake_observed_at = time.perf_counter() if MARKS_ENABLED else 0.0
            if self._awaiting_state_of is not None:
                continue
            task: Task | None = plan(
                self.node_id,
                self.runners,
                self.state.downloads,
                self.state.instances,
                self.state.runners,
                self.state.tasks,
                self._instance_backoff,
                self._download_backoff,
                self.state.node_network,
            )

            # Round-13 Gate-A mark 2 (plan_step-observed): emitted on EVERY
            # wake of this loop, regardless of whether `plan()` produced a
            # task, so amendment A4 ("each wake pairs to the earliest
            # unpaired state_applied since the PRIOR WAKE") has a real,
            # complete record of every wake to pair against -- not just the
            # wakes that happened to dispatch something. `task=None` is
            # itself a valid, informative pairing anchor: it means the
            # planner woke, ran `plan()`, and correctly found nothing to do.
            # `wake_kind` lets the analyzer count timeout-driven wakes on
            # this (request) path explicitly, per Gate A's PASS condition.
            mark_plan_step_observed(
                observed_event_idx, wake_observed_at, wake_kind, task
            )

            if task is None:
                continue

            if isinstance(task, CreateRunner):
                iid = task.instance_id
                if self._instance_backoff.attempts(iid) >= EXO_MAX_INSTANCE_RETRIES:
                    # Plan comes back to this instance every 0.1 s until it is deleted: ask
                    # once, and again only if it is still here after a while (the request
                    # may have been lost, e.g. while the master changed)
                    requested_at = self._deletion_requested.get(iid)
                    if (
                        requested_at is None
                        or time.monotonic() - requested_at >= DELETION_REQUEST_RETRY
                    ):
                        logger.warning(
                            f"Instance {iid} exceeded {EXO_MAX_INSTANCE_RETRIES} retries, requesting deletion"
                        )
                        self._deletion_requested[iid] = time.monotonic()
                        await self.command_sender.send(
                            ForwarderCommand(
                                origin=self._system_id,
                                command=DeleteInstance(instance_id=iid),
                            )
                        )
                    continue

            logger.info(f"Worker plan: {task.__class__.__name__}")
            assert task.task_status
            # Generation tasks come from the cluster state, so they exist already. This
            # node's view of the state lags the master's: announcing such a task again
            # can recreate one the master has just cancelled or deleted.
            if task.task_id not in self.state.tasks:
                await self.event_sender.send(
                    TaskCreated(task_id=task.task_id, task=task)
                )

            # lets not kill the worker if a runner is unresponsive
            match task:
                case CreateRunner():
                    await self._create_supervisor(task)
                    self._instance_backoff.record_attempt(task.instance_id)
                    await self.event_sender.send(
                        TaskStatusUpdated(
                            task_id=task.task_id, task_status=TaskStatus.Complete
                        )
                    )
                case DownloadModel(shard_metadata=shard, repo_url=repo_url):
                    model_id = shard.model_card.model_id
                    self._download_backoff.record_attempt(model_id)

                    found_path = await to_thread.run_sync(
                        resolve_existing_model, model_id, shard.model_card
                    )
                    if found_path is not None:
                        logger.info(f"Model {model_id} found at {found_path}")
                        await self.event_sender.send(
                            NodeDownloadProgress(
                                download_progress=DownloadCompleted(
                                    node_id=self.node_id,
                                    shard_metadata=shard,
                                    model_directory=str(found_path),
                                    total=shard.model_card.storage_size,
                                    read_only=is_read_only_model_dir(found_path),
                                )
                            )
                        )
                        await self.event_sender.send(
                            TaskStatusUpdated(
                                task_id=task.task_id,
                                task_status=TaskStatus.Complete,
                            )
                        )
                    else:
                        if repo_url:
                            logger.info(
                                f"P2P download available for {model_id} from {repo_url}"
                            )
                        await self.download_command_sender.send(
                            ForwarderDownloadCommand(
                                origin=self._system_id,
                                command=StartDownload(
                                    target_node_id=self.node_id,
                                    shard_metadata=shard,
                                    repo_url=repo_url,
                                ),
                            )
                        )
                        await self.event_sender.send(
                            TaskStatusUpdated(
                                task_id=task.task_id,
                                task_status=TaskStatus.Running,
                            )
                        )
                case Shutdown(runner_id=runner_id):
                    runner = self.runners.pop(runner_id)
                    try:
                        with fail_after(3):
                            await runner.start_task(task)
                    except TimeoutError:
                        await self.event_sender.send(
                            TaskStatusUpdated(
                                task_id=task.task_id, task_status=TaskStatus.TimedOut
                            )
                        )
                    finally:
                        runner.shutdown()
                case CancelTask(
                    cancelled_task_id=cancelled_task_id, runner_id=runner_id
                ):
                    await self.runners[runner_id].cancel_task(cancelled_task_id)
                    await self.event_sender.send(
                        TaskStatusUpdated(
                            task_id=task.task_id, task_status=TaskStatus.Complete
                        )
                    )
                case ImageEdits() if (
                    task.task_params.total_input_chunks > 0
                    and len(self.input_chunk_buffer.get(task.command_id, {}))
                    < task.task_params.total_input_chunks
                ):
                    await self._report_lost_inputs(task)
                case ImageEdits() if task.task_params.total_input_chunks > 0:
                    # Assemble image from chunks and inject into task
                    cmd_id = task.command_id
                    chunks = self.input_chunk_buffer.get(cmd_id, {})
                    assembled = "".join(chunks[i].data for i in range(len(chunks)))
                    logger.info(
                        f"Assembled input image from {len(chunks)} chunks, "
                        f"total size: {len(assembled)} bytes"
                    )
                    # Create modified task with assembled image data
                    modified_task = ImageEdits(
                        task_id=task.task_id,
                        command_id=task.command_id,
                        instance_id=task.instance_id,
                        task_status=task.task_status,
                        task_params=ImageEditsTaskParams(
                            image_data=assembled,
                            total_input_chunks=task.task_params.total_input_chunks,
                            prompt=task.task_params.prompt,
                            model=task.task_params.model,
                            n=task.task_params.n,
                            quality=task.task_params.quality,
                            output_format=task.task_params.output_format,
                            response_format=task.task_params.response_format,
                            size=task.task_params.size,
                            image_strength=task.task_params.image_strength,
                            bench=task.task_params.bench,
                            stream=task.task_params.stream,
                            partial_images=task.task_params.partial_images,
                            advanced_params=task.task_params.advanced_params,
                        ),
                    )
                    self._drop_inputs(cmd_id)
                    await self._start_runner_task(modified_task)

                case TextGeneration() if task.task_params.image_hashes and not all(
                    h in self.input_images.get(task.command_id, {})
                    for h in task.task_params.image_hashes.values()
                ):
                    await self._report_lost_inputs(task)
                case TextGeneration() if task.task_params.image_hashes:
                    cmd_id = task.command_id
                    images = self.input_images[cmd_id]
                    resolved_images = [
                        images[h]
                        for _, h in sorted(task.task_params.image_hashes.items())
                    ]
                    modified_task = task.model_copy(
                        update={
                            "task_params": task.task_params.model_copy(
                                update={"images": resolved_images}
                            )
                        }
                    )
                    self._drop_inputs(cmd_id)
                    await self._start_runner_task(modified_task)
                case LoadModel(instance_id=instance_id):
                    if (instance := self.state.instances.get(instance_id)) is not None:
                        model_id = instance.shard_assignments.model_id
                        self._download_backoff.reset(model_id)

                    await self._start_runner_task(task)
                case task:
                    # GENERATION tasks are dispatched NON-BLOCKING so
                    # concurrent requests (c≥2) can reach the runner's
                    # work_queue within the batched-prefill rendezvous window.
                    # The runner's handle_generation_tasks drains its queue for
                    # EXO_BATCHED_PREFILL_RENDEZVOUS_MS (default 200ms) to
                    # batch arriving tasks into one prefill_batched call.
                    # Blocking here (await) serialized c=2+ prefill: the 2nd
                    # task couldn't be sent until the 1st completed, so the
                    # rendezvous never saw it. Non-blocking dispatch + the
                    # in_progress guard in plan() prevents re-dispatch.
                    #
                    # That is safe ONLY for generation tasks, because they
                    # carry a STABLE task_id (they come from cluster state, not
                    # from `plan()`), so the runner's own `seen` set and this
                    # supervisor's `pending`/`in_progress` guards dedupe a
                    # repeat send.
                    #
                    # LIFECYCLE tasks (ConnectToGroup, StartWarmup) MUST be
                    # dispatched BLOCKING -- never via `start_soon`.
                    #
                    # Unlike generation tasks, lifecycle tasks are constructed
                    # fresh by `plan()` on every call with a NEW `task_id`, so
                    # no task_id-keyed guard can suppress a re-dispatch of the
                    # same logical task. What stops a duplicate is `plan()`'s
                    # own state precondition (e.g. `_init_distributed_backend`
                    # requires `isinstance(runner.status, RunnerIdle)`,
                    # `_ready_to_warmup` requires RunnerLoaded), and that
                    # status is only published asynchronously by the runner.
                    #
                    # `supervisor.start_task` returns after the runner
                    # ACKNOWLEDGES, and every `handle_first_task` arm publishes
                    # its new `RunnerStatusUpdated` BEFORE calling
                    # `acknowledge_task` (Connecting before the ConnectToGroup
                    # ack; WarmingUp before the StartWarmup ack). Blocking here
                    # therefore orders the two: `plan_step` cannot re-enter
                    # `plan()` until the guard status is already visible, so a
                    # second lifecycle task for the same instance is
                    # structurally impossible.
                    #
                    # Regression (2026-09-15, EXO_WORKER_PLAN_EVENT_WAKE=1):
                    # lifecycle tasks previously took the non-blocking branch
                    # too, so a state-apply wake could re-enter `plan()` inside
                    # that window. `plan_step`'s own `TaskCreated` echo came
                    # back through the master ~5ms later and re-planned while
                    # the runner was still RunnerIdle, producing two
                    # `ConnectToGroup` tasks ~5ms apart for the SAME instance
                    # with DIFFERENT task_ids. The runner processed the first
                    # (`RunnerIdle` -> connect -> `RunnerConnected`), then hit
                    # the second with `RunnerConnected` already set, fell
                    # through to the `case _:` arm and died with
                    # `ValueError: Received ConnectToGroup outside of state
                    # machine in self.current_status=RunnerConnected()` --
                    # SIGKILL-free process death, RunnerFailed, teardown and
                    # re-placement, i.e. the observed crash-loop.
                    if _may_dispatch_non_blocking(task):
                        self._tg.start_soon(self._start_runner_task, task)
                    else:
                        await self._start_runner_task(task)

    def follow_new_master(self, session_id: SessionId) -> None:
        """Carry on under a new master, keeping this node's runners.

        Until the new master's state arrives, the worker doesn't act on the state it has:
        that is the old master's, and still holds the requests that ended with the change.
        Call this as the node switches to the new master's session, before its state can
        arrive.
        """
        if self._snapshot_session != session_id:
            self._awaiting_state_of = session_id

    async def announce_to_master(self) -> None:
        """Tell a new master everything this worker knows about its node. The master's state
        may lack it: the node may have just joined the master's cluster, and the last
        updates it sent the old master may never have arrived."""
        for info in list(self._gathered.values()):
            await self._send_info(info)
        for runner in list(self.runners.values()):
            await self.event_sender.send(
                RunnerStatusUpdated(
                    runner_id=runner.bound_instance.bound_runner_id,
                    runner_status=runner.status,
                )
            )

    async def shutdown(self):
        self._tg.cancel_tasks()
        await self._stopped.wait()

    async def _start_runner_task(self, task: Task):
        if (instance := self.state.instances.get(task.instance_id)) is not None:
            await self.runners[
                instance.shard_assignments.node_to_runner[self.node_id]
            ].start_task(task)

    async def _create_supervisor(self, task: CreateRunner) -> RunnerSupervisor:
        """Creates and stores a new AssignedRunner with initial downloading status."""
        this_runner_id = task.bound_instance.bound_runner_id

        def _sibling_loading() -> bool:
            # True while any OTHER runner on this node is bringing up a model
            # (connect/load/warmup). A co-host JIT load saturates the GPU and
            # can starve a mid-generation runner silent for minutes; the hang
            # watchdog defers instead of SIGKILLing it (observed 2026-07-09
            # 16:15:08: DSv4 runner killed at 298s silent during a Qwen load).
            from exo.shared.types.tasks import LoadModel as _LoadModel
            from exo.shared.types.worker.runners import (
                RunnerConnecting as _Conn,
            )
            from exo.shared.types.worker.runners import (
                RunnerLoading as _Load,
            )
            from exo.shared.types.worker.runners import (
                RunnerWarmingUp as _Warm,
            )

            for rid, sup in self.runners.items():
                if rid == this_runner_id:
                    continue
                if isinstance(sup.status, (_Conn, _Load, _Warm)):
                    return True
                if any(isinstance(t, _LoadModel) for t in sup.in_progress.values()):
                    return True
            return False

        runner = await RunnerSupervisor.create(
            bound_instance=task.bound_instance,
            event_sender=self.event_sender.clone(),
            sibling_loading=_sibling_loading,
        )
        self.runners[task.bound_instance.bound_runner_id] = runner
        self._tg.start_soon(runner.run)
        return runner

    async def _poll_connection_updates(self):
        while True:
            edges = set(
                conn.edge for conn in self.state.topology.out_edges(self.node_id)
            )
            conns: defaultdict[NodeId, set[str]] = defaultdict(set)
            async for ip, nid in check_reachable(
                self.state.topology,
                self.node_id,
                self.state.node_network,
                api_port=self.api_port,
            ):
                if ip in conns[nid]:
                    continue
                conns[nid].add(ip)
                edge = SocketConnection(
                    # nonsense multiaddr
                    sink_multiaddr=Multiaddr(address=f"/ip4/{ip}/tcp/{self.api_port}")
                    if "." in ip
                    # nonsense multiaddr
                    else Multiaddr(address=f"/ip6/{ip}/tcp/{self.api_port}"),
                )
                if edge not in edges:
                    logger.debug(f"ping discovered {edge=}")
                    await self.event_sender.send(
                        TopologyEdgeCreated(
                            conn=Connection(source=self.node_id, sink=nid, edge=edge)
                        )
                    )

            for conn in self.state.topology.out_edges(self.node_id):
                if not isinstance(conn.edge, SocketConnection):
                    continue
                # ignore mDNS discovered connections
                if conn.edge.sink_multiaddr.port != self.api_port:
                    continue
                if (
                    conn.sink not in conns
                    or conn.edge.sink_multiaddr.ip_address not in conns[conn.sink]
                ):
                    logger.debug(f"ping failed to discover {conn=}")
                    await self.event_sender.send(TopologyEdgeDeleted(conn=conn))

            await anyio.sleep(10)
