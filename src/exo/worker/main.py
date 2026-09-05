import hashlib
import os
import time
from collections import defaultdict
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
)
from exo.shared.apply import apply
from exo.shared.constants import EXO_MAX_INSTANCE_RETRIES
from exo.shared.models.model_cards import ModelId, card_cache
from exo.shared.types.chunks import InputImageChunk
from exo.shared.types.commands import (
    DeleteInstance,
    ForwarderCommand,
    ForwarderDownloadCommand,
    StartDownload,
)
from exo.shared.types.common import CommandId, NodeId, SystemId
from exo.shared.types.events import (
    Event,
    IndexedEvent,
    InputChunkReceived,
    InstanceDeleted,
    NodeDownloadProgress,
    NodeGatheredInfo,
    TaskCreated,
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
    LoadModel,
    Shutdown,
    Task,
    TaskStatus,
    TextGeneration,
)
from exo.shared.types.text_generation import Base64Image, Base64ImageHash
from exo.shared.types.topology import Connection, SocketConnection
from exo.shared.types.worker.downloads import DownloadCompleted
from exo.shared.types.worker.instances import InstanceId
from exo.shared.types.worker.runners import RunnerId
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


class Worker:
    def __init__(
        self,
        node_id: NodeId,
        *,
        event_receiver: Receiver[IndexedEvent],
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

        # Buffer for input image chunks (for image editing)
        self.input_chunk_buffer: dict[CommandId, dict[int, InputImageChunk]] = {}
        self.input_chunk_counts: dict[CommandId, int] = {}
        self.image_cache: dict[Base64ImageHash, Base64Image] = {}

        self._download_backoff: KeyedBackoff[ModelId] = KeyedBackoff(base=0.5, cap=10.0)
        self._instance_backoff: KeyedBackoff[InstanceId] = KeyedBackoff(
            base=0.5, cap=10.0
        )
        self._stopped: anyio.Event = anyio.Event()

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
                await self.event_sender.send(
                    NodeGatheredInfo(
                        node_id=self.node_id,
                        when=str(datetime.now(tz=timezone.utc)),
                        info=info,
                    )
                )

    async def _event_applier(self):
        with self.event_receiver as events:
            async for event in events:
                # 2. for each event, apply it to the state
                self.state = apply(self.state, event=event)

                # Round-13 Gate-A mark 1 (state-update-applied): must be
                # emitted right after the mutation above and BEFORE `event`
                # is rebound to `event.event` below, since `event.idx` (the
                # pairing key) only exists on the `IndexedEvent`, not on the
                # inner `Event`. `_last_applied_event_idx` is also updated
                # here, unconditionally, for `plan_step` to read at wake
                # time regardless of whether marks are enabled.
                self._last_applied_event_idx = event.idx
                mark_state_applied(event.idx)

                event = event.event

                if isinstance(event, InstanceDeleted):
                    self._instance_backoff.reset(event.instance_id)

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
                            self.image_cache[
                                Base64ImageHash(
                                    hashlib.sha256(img.encode("ascii")).hexdigest()
                                )
                            ] = img

                # I16: the SINGLE state-apply signal site. Placed at the END of
                # the loop body so EVERY mutation this event makes -- `self.state`
                # above, plus `input_chunk_buffer` / `image_cache`, all of which
                # `plan()` reads -- has already landed before the wake fires
                # (mutate first, THEN set). Same task group and therefore the
                # same event loop and thread as `plan_step` (both are
                # `tg.start_soon`-ed in `run`), so a bare `set()` is correct
                # here; no cross-thread scheduling hop is required.
                self._signal_state_applied()

    async def _reconcile_custom_cards(self) -> None:
        while True:
            await anyio.sleep(1)
            target = dict(self.state.custom_model_cards)
            for model_id, card in target.items():
                if card_cache.get(model_id) == card:
                    continue
                await card_cache.save(card)

            for card in await card_cache.list_all():
                if card.model_id not in target:
                    await card_cache.pop(card.model_id)

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
            task: Task | None = plan(
                self.node_id,
                self.runners,
                self.state.downloads,
                self.state.instances,
                self.state.runners,
                self.state.tasks,
                self.input_chunk_buffer,
                self.image_cache,
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
                    logger.warning(
                        f"Instance {iid} exceeded {EXO_MAX_INSTANCE_RETRIES} retries, requesting deletion"
                    )
                    await self.command_sender.send(
                        ForwarderCommand(
                            origin=self._system_id,
                            command=DeleteInstance(instance_id=iid),
                        )
                    )
                    continue

            logger.info(f"Worker plan: {task.__class__.__name__}")
            assert task.task_status
            await self.event_sender.send(TaskCreated(task_id=task.task_id, task=task))

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
                    # Cleanup buffers
                    if cmd_id in self.input_chunk_buffer:
                        del self.input_chunk_buffer[cmd_id]
                    if cmd_id in self.input_chunk_counts:
                        del self.input_chunk_counts[cmd_id]
                    await self._start_runner_task(modified_task)

                case TextGeneration() if task.task_params.image_hashes:
                    cmd_id = task.command_id
                    resolved_images = [
                        self.image_cache[h]
                        for _, h in sorted(task.task_params.image_hashes.items())
                    ]
                    modified_task = task.model_copy(
                        update={
                            "task_params": task.task_params.model_copy(
                                update={"images": resolved_images}
                            )
                        }
                    )
                    if cmd_id in self.input_chunk_buffer:
                        del self.input_chunk_buffer[cmd_id]
                    if cmd_id in self.input_chunk_counts:
                        del self.input_chunk_counts[cmd_id]
                    await self._start_runner_task(modified_task)
                case LoadModel(instance_id=instance_id):
                    if (instance := self.state.instances.get(instance_id)) is not None:
                        model_id = instance.shard_assignments.model_id
                        self._download_backoff.reset(model_id)

                    await self._start_runner_task(task)
                case task:
                    # Dispatch generation tasks non-blocking so concurrent
                    # requests (c≥2) can reach the runner's work_queue within
                    # the batched-prefill rendezvous window. The runner's
                    # handle_generation_tasks drains its queue for
                    # EXO_BATCHED_PREFILL_RENDEZVOUS_MS (default 200ms) to
                    # batch arriving tasks into one prefill_batched call.
                    # Blocking here (await) serialized c=2+ prefill: the 2nd
                    # task couldn't be sent until the 1st completed, so the
                    # rendezvous never saw it. Non-blocking dispatch + the
                    # in_progress guard in plan() prevents re-dispatch.
                    self._tg.start_soon(self._start_runner_task, task)

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
