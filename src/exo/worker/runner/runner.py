import gc
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import BinaryIO

from anyio import ClosedResourceError, EndOfStream

from exo.shared.constants import (
    ENABLE_DISAGGREGATION,
    EXO_BATCHED_PREFILL_RENDEZVOUS_MS,
    EXO_DSV4_BATCHED_PREFILL,
)
from exo.shared.types.chunks import Chunk
from exo.shared.types.common import CommandId
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
)
from exo.shared.types.tasks import (
    ConnectToGroup,
    GenerationTask,
    ImageEdits,
    ImageGeneration,
    LoadModel,
    Shutdown,
    StartWarmup,
    Task,
    TaskId,
    TaskStatus,
    TextGeneration,
)
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runner_response import (
    CancelledResponse,
    FinishedResponse,
)
from exo.shared.types.worker.runners import (
    RunnerConnected,
    RunnerConnecting,
    RunnerIdle,
    RunnerLoaded,
    RunnerLoading,
    RunnerReady,
    RunnerRunning,
    RunnerShutdown,
    RunnerShuttingDown,
    RunnerStatus,
    RunnerWarmingUp,
)
from exo.utils.channels import MpReceiver, MpSender
from exo.utils.log_format import truncate_for_log
from exo.utils.ports import random_ephemeral_port
from exo.worker.disaggregated.server import (
    PrefillRequest,
    PrefillServer,
)
from exo.worker.engines.base import Builder, Engine
from exo.worker.runner.bootstrap import logger

PREFILL_PICKUP_TIMEOUT_SECONDS = 3
PREFILL_FINISH_TIMEOUT_SECONDS = 300

# Reclaim MLX's caching allocator pool back to the OS when the runner goes idle
# (all generation tasks complete). MLX's allocator otherwise holds freed GPU
# buffers indefinitely for reuse; macOS keeps those pages resident (counted as
# "used" by the exo /metrics gauge and dashboard), so an idle runner reports far
# more memory than its actual model footprint — the freed prefill/decode working
# set never returns to the OS until something forces it. Measured 2026-06-20:
# idle DSv4-Flash reported ~110GB/node where the model itself is only ~78GB; the
# ~30GB delta was unreclaimed inactive/compressed standby. One clear_cache at the
# idle transition (NOT in the hot decode loop — so zero steady-state tok/s cost)
# makes the reported number reflect reality and returns the pages for the next
# session's prefill. On by default; set EXO_RECLAIM_ON_IDLE=0 to disable.
_RECLAIM_ON_IDLE = os.environ.get("EXO_RECLAIM_ON_IDLE", "1") != "0"


@dataclass
class PrefillTask:
    request: PrefillRequest
    wfile: BinaryIO
    started: threading.Event
    done: threading.Event


class _TaskStreamClosed:
    pass


WorkItem = Task | PrefillTask | _TaskStreamClosed


class ExitCode(str, Enum):
    AllTasksComplete = "AllTasksComplete"
    Shutdown = "Shutdown"


class Runner:
    def __init__(
        self,
        bound_instance: BoundInstance,
        builder: Builder,
        event_sender: MpSender[Event],
        task_receiver: MpReceiver[Task],
    ):
        self.event_sender = event_sender
        self.task_receiver = task_receiver
        self.bound_instance = bound_instance

        self.instance, self.runner_id, self.shard_metadata = (
            self.bound_instance.instance,
            self.bound_instance.bound_runner_id,
            self.bound_instance.bound_shard,
        )
        self.model_id = self.shard_metadata.model_card.model_id
        self.device_rank = self.shard_metadata.device_rank

        logger.info("hello from the runner")
        if getattr(self.shard_metadata, "immediate_exception", False):
            raise Exception("Fake exception - runner failed to spin up.")
        if timeout := getattr(self.shard_metadata, "should_timeout", 0):
            time.sleep(timeout)

        self.setup_start_time = time.time()

        self.generator: Builder | Engine = builder

        self.seen: set[TaskId] = set()
        self.active_tasks: dict[
            TaskId,
            GenerationTask,
        ] = {}

        self._prefill_server: PrefillServer | None = None
        self._prefill_server_port: int | None = None
        self._work_queue: queue.Queue[WorkItem] = queue.Queue()
        self._task_reader_thread: threading.Thread | None = None

        self._step_beat: float = time.monotonic()
        self._stall_sampler_thread: threading.Thread | None = None
        self._start_stall_sampler()

        logger.info("runner created")
        self.update_status(RunnerIdle())

    def _start_stall_sampler(self) -> None:
        """Diagnostic sampler for step-loop stalls (task #25 admission wedge).

        When ``EXO_STALL_SAMPLER_SECONDS`` is set (> 0), a daemon thread
        watches ``self._step_beat`` — updated every time ``generator.step()``
        returns. If tasks are active and step() hasn't returned for that many
        seconds, it appends every thread's Python stack to
        ``/tmp/exo_stall_rank{rank}_pid{pid}.log`` (throttled to one dump per
        5 s). Repeated dumps sample a silent in-generator loop line-by-line;
        a rank parked in a collective shows the exact mx.eval/all_reduce
        frame. Reads other threads' frames from THIS thread, so it works even
        while the main thread is stuck inside a C call — the case py-spy
        would need root for on macOS. Zero overhead when the env var is
        unset.
        """
        interval = float(os.environ.get("EXO_STALL_SAMPLER_SECONDS", "0") or "0")
        if interval <= 0 or self._stall_sampler_thread is not None:
            return

        def loop() -> None:
            import traceback

            path = f"/tmp/exo_stall_rank{self.device_rank}_pid{os.getpid()}.log"
            last_dump = 0.0
            while True:
                time.sleep(1.0)
                if not self.active_tasks:
                    continue
                now = time.monotonic()
                stalled = now - self._step_beat
                if stalled < interval or now - last_dump < 5.0:
                    continue
                last_dump = now
                try:
                    with open(path, "a") as handle:
                        handle.write(
                            f"\n===== stall={stalled:.1f}s wall={time.time():.3f} "
                            f"active={list(self.active_tasks.keys())} =====\n"
                        )
                        for thread_id, frame in sys._current_frames().items():  # noqa: SLF001
                            handle.write(f"--- thread {thread_id} ---\n")
                            handle.write("".join(traceback.format_stack(frame)))
                except Exception:  # noqa: BLE001
                    pass

        self._stall_sampler_thread = threading.Thread(
            target=loop, name="stall-sampler", daemon=True
        )
        self._stall_sampler_thread.start()

    def _start_prefill_server(self) -> int | None:
        if not ENABLE_DISAGGREGATION:
            return None
        if self.device_rank != 0:
            return None
        if self._prefill_server_port is not None:
            return self._prefill_server_port

        def resolve(request: PrefillRequest, wfile: BinaryIO) -> bool:
            req = PrefillTask(
                request=request,
                wfile=wfile,
                started=threading.Event(),
                done=threading.Event(),
            )
            self._work_queue.put(req)
            if not req.started.wait(timeout=PREFILL_PICKUP_TIMEOUT_SECONDS):
                logger.warning(
                    f"Prefill request {request.request_id} not picked up within "
                    f"{PREFILL_PICKUP_TIMEOUT_SECONDS}s — runner busy"
                )
                return False
            if not req.done.wait(timeout=PREFILL_FINISH_TIMEOUT_SECONDS):
                logger.warning(
                    f"Prefill request {request.request_id} did not finish within "
                    f"{PREFILL_FINISH_TIMEOUT_SECONDS}s"
                )
            return True

        port = random_ephemeral_port()
        self._prefill_server = PrefillServer(resolve=resolve, host="0.0.0.0", port=port)
        self._prefill_server_port = port
        return self._prefill_server_port

    def _start_task_reader(self) -> None:
        if self._task_reader_thread is not None:
            return

        def loop() -> None:
            try:
                with self.task_receiver:
                    for task in self.task_receiver:
                        self._work_queue.put(task)
            except (EndOfStream, ClosedResourceError):
                pass
            finally:
                self._work_queue.put(_TaskStreamClosed())

        self._task_reader_thread = threading.Thread(target=loop, name="task-reader")
        self._task_reader_thread.start()

    def _serve_prefill(self, req: PrefillTask) -> None:
        req.started.set()
        try:
            assert isinstance(self.generator, Engine)
            self.generator.serve_prefill(req.request, req.wfile)
        except Exception:
            logger.opt(exception=True).warning(
                f"Failed to serve prefill request {req.request.request_id}"
            )
        finally:
            req.done.set()

    def update_status(self, status: RunnerStatus):
        self.current_status = status
        self.event_sender.send(
            RunnerStatusUpdated(
                runner_id=self.runner_id, runner_status=self.current_status
            )
        )

    def send_task_status(self, task_id: TaskId, task_status: TaskStatus):
        self.event_sender.send(
            TaskStatusUpdated(task_id=task_id, task_status=task_status)
        )

    def acknowledge_task(self, task: Task):
        self.event_sender.send(TaskAcknowledged(task_id=task.task_id))

    def main(self):
        self._start_task_reader()
        try:
            while True:
                item = self._work_queue.get()
                if isinstance(item, _TaskStreamClosed):
                    break
                if isinstance(item, PrefillTask):
                    self._serve_prefill(item)
                    continue
                if item.task_id in self.seen:
                    logger.warning("repeat task - potential error")
                    continue
                self.seen.add(item.task_id)
                self.handle_first_task(item)
                if isinstance(self.current_status, RunnerShutdown):
                    break
        finally:
            if self._prefill_server is not None:
                self._prefill_server.stop()
                self._prefill_server = None
            self.task_receiver.close()
            if self._task_reader_thread is not None:
                self._task_reader_thread.join(timeout=5)
                self._task_reader_thread = None

    def handle_first_task(self, task: Task):
        self.send_task_status(task.task_id, TaskStatus.Running)

        match task:
            case ConnectToGroup() if isinstance(self.current_status, RunnerIdle):
                assert isinstance(self.generator, Builder)
                logger.info("runner connecting")
                self.update_status(RunnerConnecting())
                self.acknowledge_task(task)

                self.generator.connect(self.bound_instance)

                self.send_task_status(task.task_id, TaskStatus.Complete)
                self.update_status(RunnerConnected())
                logger.info("runner connected")

            # we load the model if it's connected with a group, or idle without a group. we should never tell a model to connect if it doesn't need to
            case LoadModel() if isinstance(self.generator, Builder) and (
                isinstance(self.current_status, (RunnerConnected, RunnerIdle))
            ):
                total_layers = (
                    self.shard_metadata.end_layer - self.shard_metadata.start_layer
                )
                logger.info("runner loading")

                self.update_status(
                    RunnerLoading(layers_loaded=0, total_layers=total_layers)
                )
                self.acknowledge_task(task)

                for load_progress in self.generator.load(self.bound_instance):
                    self.update_status(
                        RunnerLoading(
                            layers_loaded=load_progress.layers_loaded,
                            total_layers=load_progress.total,
                        )
                    )

                self.generator = self.generator.build()

                self.send_task_status(task.task_id, TaskStatus.Complete)
                self.update_status(RunnerLoaded())
                logger.info("runner loaded")

            case StartWarmup() if isinstance(self.current_status, RunnerLoaded):
                assert isinstance(self.generator, Engine)
                logger.info("runner warming up")

                self.update_status(RunnerWarmingUp())
                self.acknowledge_task(task)

                self._warmup_with_reconnect()

                logger.info(
                    f"runner initialized in {time.time() - self.setup_start_time} seconds"
                )

                self._start_prefill_server()
                self.send_task_status(task.task_id, TaskStatus.Complete)
                self.update_status(
                    RunnerReady(prefill_server_port=self._prefill_server_port)
                )
                logger.info("runner ready")

            case TextGeneration() | ImageEdits() | ImageGeneration() if isinstance(
                self.current_status, RunnerReady
            ):
                return_code = self.handle_generation_tasks(starting_task=task)
                if return_code == ExitCode.Shutdown:
                    return

            case Shutdown():
                self.shutdown(task)
                return

            case _:
                raise ValueError(
                    f"Received {task.__class__.__name__} outside of state machine in {self.current_status=}"
                )

    def _warmup_with_reconnect(self) -> None:
        """Run engine warmup, retrying through jaccl transport faults.

        The first reliable collectives after QP creation intermittently hit a
        unidirectional dead UC path (observed 2026-07-06, 2 of 3 warmups:
        reliable_all_reduce drain deadline with rank 0 all_recv=0 while
        rank 1 all_recv=7 — one direction drops everything for the whole
        15 s window). A warmup crash previously took BOTH runners down and
        left the instance as a permanent WARMING UP zombie. Treat the same
        fault class the step() loop already recovers from as retryable:
        group.reconnect() (both ranks fault → both reach the reconnect
        coordinator barrier, which re-syncs them) and re-run warmup.
        """
        assert isinstance(self.generator, Engine)
        attempts = int(os.environ.get("EXO_WARMUP_RECONNECT_ATTEMPTS", "2"))
        for attempt in range(attempts + 1):
            try:
                self.generator.warmup()
                return
            except Exception as warmup_err:  # noqa: BLE001
                group = getattr(self.generator, "group", None)
                message = str(warmup_err)
                recoverable = (
                    group is not None
                    and attempt < attempts
                    and any(
                        marker in message
                        for marker in (
                            "STALLED",
                            "[Event::wait] Timed out",
                            "drain_acks",
                            "all_reduce",
                            "[jaccl]",
                        )
                    )
                )
                if not recoverable:
                    raise
                logger.warning(
                    f"jaccl transport fault during warmup "
                    f"(attempt {attempt + 1}/{attempts + 1}): {message}. "
                    "Reconnecting group and retrying warmup."
                )
                assert group is not None
                try:
                    group.reconnect()  # pyright: ignore[reportAny]
                except Exception as reconnect_err:
                    logger.error(
                        f"jaccl reconnect during warmup failed "
                        f"({reconnect_err!r}); propagating original fault."
                    )
                    raise warmup_err from reconnect_err

    def shutdown(self, task: Task):
        logger.info("runner shutting down")
        self.update_status(RunnerShuttingDown())
        self.acknowledge_task(task)
        self.generator.close()
        import gc

        gc.collect()
        self.send_task_status(task.task_id, TaskStatus.Complete)
        self.update_status(RunnerShutdown())

    def submit_generation(self, task: GenerationTask):
        assert isinstance(self.generator, Engine)
        self.active_tasks[task.task_id] = task
        self.generator.submit(task)

    def handle_generation_tasks(self, starting_task: GenerationTask):
        assert isinstance(self.current_status, RunnerReady)
        assert isinstance(self.generator, Engine)

        logger.info(f"received chat request: {truncate_for_log(starting_task)}")
        self.update_status(RunnerRunning())
        logger.info("runner running")
        self.acknowledge_task(starting_task)
        self.seen.add(starting_task.task_id)

        self.submit_generation(starting_task)

        # Rendezvous window for batched prefill: when ``EXO_DSV4_BATCHED_PREFILL``
        # is on, drain the work queue briefly BEFORE the first step() call so
        # any concurrent c=2+ requests can land in the engine's queue at the
        # same step() iteration as ``starting_task``. Without this, the first
        # task's prefill blocks the runner thread before any subsequent task
        # can even reach the engine, and the batched-prefill gate at
        # ``BatchGenerator.step()`` never sees ``len(queue) >= 2``.
        # Latency cost: ``EXO_BATCHED_PREFILL_RENDEZVOUS_MS`` added to c=1
        # first-token times when batched prefill is on.
        if EXO_DSV4_BATCHED_PREFILL and EXO_BATCHED_PREFILL_RENDEZVOUS_MS > 0:
            rendezvous_deadline = (
                time.monotonic() + EXO_BATCHED_PREFILL_RENDEZVOUS_MS / 1000.0
            )
            from exo.shared.constants import EXO_MAX_CONCURRENT_REQUESTS

            extras_seen = 0
            while time.monotonic() < rendezvous_deadline and (
                len(self.active_tasks) < EXO_MAX_CONCURRENT_REQUESTS
            ):
                remaining = rendezvous_deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = self._work_queue.get(timeout=remaining)
                except queue.Empty:
                    break
                # Stash non-generation items (PrefillTask, _TaskStreamClosed,
                # other commands) back into the queue so the existing main
                # loop handles them — only batch GenerationTasks at this
                # rendezvous point.
                if isinstance(item, GenerationTask):
                    if item.task_id in self.seen:
                        continue
                    self.seen.add(item.task_id)
                    self.acknowledge_task(item)
                    self.submit_generation(item)
                    extras_seen += 1
                else:
                    # Re-enqueue and exit rendezvous early so the loop can
                    # handle this item promptly.
                    self._work_queue.put(item)
                    break
            if extras_seen > 0:
                logger.info(
                    f"Rendezvous batched {extras_seen + 1} concurrent tasks "
                    f"(window={EXO_BATCHED_PREFILL_RENDEZVOUS_MS}ms)"
                )

        # Track A probe: per-cycle attribution (gated MLX_GPU_TIME=1).
        # Helps separate generator.step() wall from send_chunk + queue
        # overhead between cycles.
        import os as _os
        _runner_probe = bool(_os.environ.get("MLX_GPU_TIME"))
        _probe_log_every = int(_os.environ.get("MLX_GPU_TIME_LOG_EVERY", "32"))
        _probe_cycle_count = 0
        _probe_sum_step_ns = 0
        _probe_sum_send_ns = 0
        _probe_sum_total_ns = 0
        _probe_last_total_start: float = 0.0
        if _runner_probe:
            _probe_last_total_start = time.perf_counter()

        self._step_beat = time.monotonic()
        while self.active_tasks:
            if _runner_probe:
                _t_step_start = time.perf_counter()
            try:
                results = self.generator.step()
                self._step_beat = time.monotonic()
            except Exception as step_err:
                # In-place recovery of a c>=2 jaccl transport wedge. Both ranks
                # surface the fault (StallWatch on the rank owning the lost
                # completion; Event::wait total-timeout on the peer), so both
                # reach group.reconnect(), whose coordinator barrier re-syncs
                # them. We reset the QPs, drop the in-flight batch (clients
                # retry those requests), and resume serving with the 72GB model
                # RESIDENT — instead of exiting -> full instance re-place (~90s
                # reload). Only jaccl transport faults are recoverable this way;
                # anything else (or a reconnect that itself fails, e.g. the
                # driver rejects RESET) propagates and the instance re-places.
                grp = getattr(self.generator, "group", None)
                _msg = str(step_err)
                _recoverable = grp is not None and any(
                    marker in _msg
                    for marker in (
                        "STALLED",
                        "[Event::wait] Timed out",
                        "drain_acks",
                        "all_reduce",
                        "[jaccl]",
                    )
                )
                if not _recoverable:
                    raise
                import os as _os_diag

                if _os_diag.environ.get("MLX_DIAG_HOLD_WEDGE"):
                    # DIAGNOSTIC ONLY: hold the wedge open (do NOT reconnect or
                    # re-place) so the PEER stays parked in its real stuck
                    # location for clean sampling — instead of thrashing into a
                    # model reload that contaminates every capture. Pair with a
                    # large EXO_RUNNER_HANG_TIMEOUT_SECONDS.
                    import time as _time_diag

                    logger.warning(
                        f"[DIAG] jaccl fault ({_msg}); HOLDING wedge open for "
                        "300s (MLX_DIAG_HOLD_WEDGE) — sample the peer NOW."
                    )
                    _time_diag.sleep(300)
                    raise step_err
                logger.warning(
                    f"jaccl transport fault in generator.step(): {_msg}. "
                    "Attempting in-place reconnect (both ranks) to avoid a re-place."
                )
                try:
                    grp.reconnect()
                except Exception as rc_err:
                    logger.error(
                        f"jaccl reconnect failed ({rc_err!r}); propagating for re-place."
                    )
                    raise step_err from rc_err
                dropped = self.generator.reset_after_reconnect()
                for task_id in list(self.active_tasks.keys()):
                    self.send_task_status(task_id, TaskStatus.Failed)
                    self.active_tasks.pop(task_id, None)
                logger.warning(
                    f"jaccl reconnect complete; dropped {len(dropped)} in-flight "
                    "sequence(s), resumed serving with model resident."
                )
                break
            if _runner_probe:
                _t_step_end = time.perf_counter()

            finished: list[TaskId] = []
            _n_results_this_cycle = 0
            for task_id, result in results:
                _n_results_this_cycle += 1
                match result:
                    case CancelledResponse():
                        finished.append(task_id)
                    case FinishedResponse():
                        self.send_task_status(task_id, TaskStatus.Complete)
                        finished.append(task_id)
                    case other:
                        self.send_chunk(other, self.active_tasks[task_id].command_id)

            for task_id in finished:
                self.active_tasks.pop(task_id, None)

            # Liveness heartbeat. Under sustained c>=2, when a request is
            # admitted into a running batch the batched generator can decode
            # for many seconds (batch-size transition) WITHOUT yielding any
            # response — step() returns [] every cycle even though the GPU is
            # busy and the runner is healthy. With no event emitted, the
            # supervisor's hang watchdog (_last_event_monotonic) SIGKILLs a
            # working runner (false positive). A returned-but-empty step()
            # proves liveness (a genuine hang stuck INSIDE step() never returns,
            # so it is still caught). Re-emit the current status, throttled, to
            # reset the watchdog clock. Cheap: only fires during such stalls.
            if self.active_tasks:
                _hb_now = time.monotonic()
                if _n_results_this_cycle > 0:
                    self._last_hb_monotonic = _hb_now
                elif _hb_now - getattr(self, "_last_hb_monotonic", 0.0) >= 15.0:
                    self.update_status(self.current_status)
                    self._last_hb_monotonic = _hb_now

            if _runner_probe:
                _t_loop_end = time.perf_counter()
                # total = step + send_chunks + finished_pop + queue check
                _step_ns = int((_t_step_end - _t_step_start) * 1e9)
                _send_ns = int((_t_loop_end - _t_step_end) * 1e9)
                _total_ns = int((_t_loop_end - _probe_last_total_start) * 1e9)
                _probe_last_total_start = _t_loop_end
                _probe_cycle_count += 1
                _probe_sum_step_ns += _step_ns
                _probe_sum_send_ns += _send_ns
                _probe_sum_total_ns += _total_ns
                if _probe_cycle_count % _probe_log_every == 0:
                    n = _probe_cycle_count
                    sys.stderr.write(
                        f"[RUNNER_LOOP pid={_os.getpid()}] "
                        f"cycles={n} ntasks={_n_results_this_cycle} "
                        f"avg_step_ms={_probe_sum_step_ns/n/1e6:.2f} "
                        f"avg_send_ms={_probe_sum_send_ns/n/1e6:.2f} "
                        f"avg_total_ms={_probe_sum_total_ns/n/1e6:.2f}\n"
                    )
                    sys.stderr.flush()

            try:
                item = self._work_queue.get_nowait()
            except queue.Empty:
                continue
            if isinstance(item, _TaskStreamClosed):
                return ExitCode.Shutdown
            if isinstance(item, PrefillTask):
                self._serve_prefill(item)
                continue
            if item.task_id in self.seen:
                logger.warning("repeat task - potential error")
                continue
            self.seen.add(item.task_id)
            match item:
                case TextGeneration() | ImageGeneration() | ImageEdits():
                    self.acknowledge_task(item)
                    self.submit_generation(item)
                case Shutdown():
                    self.shutdown(item)
                    return ExitCode.Shutdown
                case _:
                    raise ValueError(
                        f"Received {item.__class__.__name__} outside of state machine in {self.current_status=}"
                    )

        # All active tasks drained → runner is going idle. Reclaim the MLX
        # caching allocator pool now so the freed prefill/decode working set
        # returns to the OS instead of sitting as resident standby that inflates
        # the reported memory (see _RECLAIM_ON_IDLE). This runs ONLY on the
        # idle transition, never inside the per-token decode loop, so there is
        # no steady-state throughput cost. gc.collect() first breaks the MLX
        # array-graph ref cycles so clear_cache can actually release the buffers.
        if _RECLAIM_ON_IDLE:
            try:
                import mlx.core as mx

                gc.collect()
                mx.clear_cache()
                logger.info("runner idle: reclaimed MLX allocator pool")
            except Exception:
                logger.debug("idle reclaim failed", exc_info=True)

        self.update_status(RunnerReady(prefill_server_port=self._prefill_server_port))
        logger.info("runner ready")

        return ExitCode.AllTasksComplete

    def send_chunk(
        self,
        chunk: Chunk,
        command_id: CommandId,
    ):
        assert isinstance(self.generator, Engine)
        # CRITICAL: only rank 0 emits ChunkGenerated. Both ranks run the
        # full TP forward and reach this method on every accepted token; if
        # both send, the API receives every token twice and the user sees
        # 'FALCONFALCON-MERCURY-MERCURY-7749-7749' instead of
        # 'FALCON-MERCURY-7749'. Upstream removed this guard
        # (origin/main@2026-05-25); we keep it because our TP-on-2-machines
        # topology requires deduplication at the emission point.
        if self.device_rank != 0:
            return
        self.event_sender.send(ChunkGenerated(command_id=command_id, chunk=chunk))
