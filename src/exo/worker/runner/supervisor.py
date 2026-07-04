import codecs
import contextlib
import os
import signal
import time
from dataclasses import dataclass, field
from os import PathLike
from typing import Callable, Self

import anyio
from anyio import (
    AsyncFile,
    BrokenResourceError,
    CancelScope,
    ClosedResourceError,
)
from loguru import logger

from exo.shared.constants import EXO_RUNNER_STDERR_LOG, EXO_RUNNER_STDOUT_LOG
from exo.shared.types.chunks import ErrorChunk
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
)
from exo.shared.types.tasks import (
    CANCEL_ALL_TASKS,
    ImageEdits,
    ImageGeneration,
    Task,
    TaskId,
    TaskStatus,
    TextGeneration,
)
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runners import (
    RunnerConnecting,
    RunnerFailed,
    RunnerIdle,
    RunnerLoading,
    RunnerReady,
    RunnerRunning,
    RunnerShuttingDown,
    RunnerStatus,
    RunnerWarmingUp,
)
from exo.shared.types.worker.shards import ShardMetadata
from exo.utils.async_process import AsyncProcess
from exo.utils.channels import MpReceiver, MpSender, Receiver, Sender, mp_channel
from exo.utils.fs import ensure_parent_directory_exists
from exo.utils.log_format import truncate_for_log
from exo.utils.task_group import TaskGroup
from exo.worker.runner.bootstrap import RunnerTerminationError, entrypoint
from exo.worker.runner.diagnostics import (
    RunnerDiagnosticCollector,
    RunnerUnknown,
)

PREFILL_TIMEOUT_SECONDS = 60
DECODE_TIMEOUT_SECONDS = 5

# Hang watchdog: if a runner has in-progress work but emits NO event (prefill
# progress, generated token, status update) for this long, it is presumed hung
# — the c>=2 degen-kill / GPU-timeout wedge spins a runner at 100% CPU inside a
# native jaccl collective, where SIGTERM does not reach the interpreter. The
# watchdog SIGKILLs it so is_alive() flips false -> RunnerFailed -> the master
# tears the instance down and re-places it (self-heal), instead of the cluster
# hanging until a manual `kill -9` + relaunch. Generous vs real work: decode
# emits a token sub-second, prefill emits per-chunk progress; only a true hang
# stays silent this long. 0 disables. Override: EXO_RUNNER_HANG_TIMEOUT_SECONDS.
def _env_seconds(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


HANG_TIMEOUT_SECONDS = _env_seconds("EXO_RUNNER_HANG_TIMEOUT_SECONDS", 45.0)

# Companion watchdog for the PRE-serving phase. _check_hang only fires once a
# task is in_progress, so a runner wedged during connect/load — e.g. the jaccl
# re-place after a peer's mid-GPU-op SIGKILL leaves RDMA state that never
# completes QP->RTR — sits in RunnerConnecting forever with no in_progress task
# and no event, and nothing ever re-places it (the "needs a reboot" mode). If a
# runner emits no event for this long while STILL not serving (not Ready/Running),
# SIGKILL it so is_alive() flips -> RunnerFailed -> master re-places (self-heal).
# Healthy connect/load streams status + progress events well inside this window;
# only a truly stuck bring-up stays silent. 0 disables. Override:
# EXO_RUNNER_CONNECT_TIMEOUT_SECONDS.
CONNECT_TIMEOUT_SECONDS = _env_seconds("EXO_RUNNER_CONNECT_TIMEOUT_SECONDS", 180.0)


@dataclass(eq=False)
class RunnerStdioHandler:
    _stdout_rx: Receiver[bytes]
    _stderr_rx: Receiver[bytes]
    _stdout_log: AsyncFile[str]
    _stderr_log: AsyncFile[str]
    diagnostics: RunnerDiagnosticCollector = field(
        default_factory=RunnerDiagnosticCollector
    )

    _tg: TaskGroup = field(default_factory=TaskGroup, init=False)

    @classmethod
    async def create(
        cls,
        *,
        stdout_rx: Receiver[bytes],
        stderr_rx: Receiver[bytes],
        stdout_log_path: PathLike[str] = EXO_RUNNER_STDOUT_LOG,
        stderr_log_path: PathLike[str] = EXO_RUNNER_STDERR_LOG,
    ) -> Self:
        # these are append only logs used to gather data for log template mining
        #
        # TODO: in the future use [Drain3](https://github.com/logpai/Drain3)
        #       to mine these logs
        ensure_parent_directory_exists(stdout_log_path)
        ensure_parent_directory_exists(stderr_log_path)
        stdout_log = await anyio.open_file(stdout_log_path, "a")
        stderr_log = await anyio.open_file(stderr_log_path, "a")

        # instantiate and return
        self = cls(
            _stdout_rx=stdout_rx,
            _stderr_rx=stderr_rx,
            _stdout_log=stdout_log,
            _stderr_log=stderr_log,
        )
        return self

    async def run(self):
        try:
            async with self._tg as tg:
                tg.start_soon(  # pyright: ignore[reportUnknownArgumentType]
                    self._handle_runner_output,
                    self._stdout_rx,
                    self._stdout_log,
                    lambda line: logger.info(f"Runner stdout: {line}"),  # pyright: ignore[reportUnknownLambdaType]
                    lambda _: None,  # pyright: ignore[reportUnknownLambdaType]
                )
                tg.start_soon(  # pyright: ignore[reportUnknownArgumentType]
                    self._handle_runner_output,
                    self._stderr_rx,
                    self._stderr_log,
                    lambda line: logger.warning(f"Runner stderr: {line}"),  # pyright: ignore[reportUnknownLambdaType]
                    self.diagnostics.record_line,
                )
        finally:
            with CancelScope(shield=True):
                await self._stdout_log.aclose()
                await self._stderr_log.aclose()

    async def _handle_runner_output(
        self,
        rx: Receiver[bytes],
        logfile: AsyncFile[str],
        log_line: Callable[[str], None],
        record_diagnostic_line: Callable[[str], None],
    ):
        # The diagnostic collector is deliberately line-level for now. It records
        # bounded stderr context and known failure anchors; the supervisor
        # correlates those hints with the runner exit status before surfacing an
        # error.

        # not using TextReceiveStream because it doesn't do final=True handling on errors
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        pending_line = ""

        async def handle_line(line: str):
            # preserve whitespace for later log-mining
            line = line.removesuffix("\r")
            if not line:
                return

            # Send to logger & error recovery task
            log_line(line)
            record_diagnostic_line(line)

        async def handle_text(text: str):
            nonlocal pending_line

            if not text:
                return

            await logfile.write(text)
            await logfile.flush()

            # newline buffering
            pending_line += text
            lines = pending_line.split("\n")
            pending_line = lines.pop()

            for line in lines:
                await handle_line(line)

        try:
            with rx:
                async for chunk in rx:
                    await handle_text(decoder.decode(chunk, final=False))
        except (ClosedResourceError, BrokenResourceError):
            logger.warning("Runner stdio stream closed before clean EOF")
        finally:
            with CancelScope(shield=True):
                await handle_text(decoder.decode(b"", final=True))
                await logfile.flush()

                if pending_line:
                    await handle_line(pending_line)
                    pending_line = ""


@dataclass(eq=False)
class RunnerSupervisor:
    shard_metadata: ShardMetadata
    bound_instance: BoundInstance
    runner_process: AsyncProcess
    _runner_stdio_handler: RunnerStdioHandler
    initialize_timeout: float
    _ev_recv: MpReceiver[Event | RunnerTerminationError]
    _task_sender: MpSender[Task]
    _event_sender: Sender[Event]
    _cancel_sender: MpSender[TaskId]
    _tg: TaskGroup = field(default_factory=TaskGroup, init=False)
    status: RunnerStatus = field(default_factory=RunnerIdle, init=False)
    pending: dict[TaskId, anyio.Event] = field(default_factory=dict, init=False)
    in_progress: dict[TaskId, Task] = field(default_factory=dict, init=False)
    completed: set[TaskId] = field(default_factory=set, init=False)
    cancelled: set[TaskId] = field(default_factory=set, init=False)
    _cancel_watch_runner: anyio.CancelScope = field(
        default_factory=anyio.CancelScope, init=False
    )
    # Monotonic timestamp of the last event received from the runner. Drives the
    # hang watchdog (see HANG_TIMEOUT_SECONDS). Bumped on every event.
    _last_event_monotonic: float = field(default_factory=time.monotonic, init=False)
    _hang_killed: bool = field(default=False, init=False)

    @classmethod
    async def create(
        cls,
        *,
        bound_instance: BoundInstance,
        event_sender: Sender[Event],
        initialize_timeout: float = 400,
    ) -> Self:
        ev_send, ev_recv = mp_channel[Event | RunnerTerminationError]()
        task_sender, task_recv = mp_channel[Task]()
        cancel_sender, cancel_recv = mp_channel[TaskId]()

        runner_process = AsyncProcess(
            target=entrypoint,
            args=(
                bound_instance,
                ev_send,
                task_recv,
                cancel_recv,
                logger,
            ),
            daemon=True,
        )
        runner_stdio_handler = await RunnerStdioHandler.create(
            stdout_rx=runner_process.stdout, stderr_rx=runner_process.stderr
        )

        shard_metadata = bound_instance.bound_shard

        self = cls(
            bound_instance=bound_instance,
            shard_metadata=shard_metadata,
            runner_process=runner_process,
            _runner_stdio_handler=runner_stdio_handler,
            initialize_timeout=initialize_timeout,
            _ev_recv=ev_recv,
            _task_sender=task_sender,
            _cancel_sender=cancel_sender,
            _event_sender=event_sender,
        )

        return self

    async def run(self):
        try:
            async with self._tg as tg:
                # start the process itself & handle its stdout/stderr
                await tg.start(self.runner_process.run)
                tg.start_soon(self._runner_stdio_handler.run)

                tg.start_soon(self._watch_runner)
                tg.start_soon(self._forward_events)
        finally:
            logger.info("Runner supervisor shutting down")
            if not self._cancel_watch_runner.cancel_called:
                self._cancel_watch_runner.cancel()
            with contextlib.suppress(ClosedResourceError):
                self._ev_recv.close()
            with contextlib.suppress(ClosedResourceError):
                self._task_sender.close()
            with contextlib.suppress(ClosedResourceError):
                self._event_sender.close()
            with contextlib.suppress(ClosedResourceError):
                self._cancel_sender.send(CANCEL_ALL_TASKS)
            with contextlib.suppress(ClosedResourceError):
                self._cancel_sender.close()

            with anyio.CancelScope(shield=True):
                await self.runner_process.stop()
                logger.info(
                    f"Runner process successfully terminated: {self.runner_process.exitcode}"
                )

    def shutdown(self):
        self._tg.cancel_tasks()

    async def start_task(self, task: Task):
        if task.task_id in self.pending:
            logger.warning(
                f"Skipping invalid task {task.task_id} as it has already been submitted"
            )
            return
        if task.task_id in self.completed:
            logger.warning(
                f"Skipping invalid task {task.task_id} as it has already been completed"
            )
            return
        logger.info(f"Starting task {truncate_for_log(task)}")
        event = anyio.Event()
        self.pending[task.task_id] = event
        self.in_progress[task.task_id] = task
        try:
            await self._task_sender.send_async(task)
        except ClosedResourceError:
            self.in_progress.pop(task.task_id, None)
            logger.warning(f"Task {task} dropped, runner closed communication.")
            return
        await event.wait()

    async def cancel_task(self, task_id: TaskId):
        if task_id in self.completed:
            logger.info(f"Unable to cancel {task_id} as it has been completed")
            self.cancelled.add(task_id)
            return
        self.cancelled.add(task_id)
        with anyio.move_on_after(0.5) as scope:
            try:
                await self._cancel_sender.send_async(task_id)
            except ClosedResourceError:
                # typically occurs when trying to shut down a failed instance
                logger.warning(
                    f"Cancelling task {task_id} failed, runner closed communication"
                )
        if scope.cancel_called:
            logger.error("RunnerSupervisor cancel pipe blocked")
            await self._check_runner(TimeoutError("cancel pipe blocked"))

    async def _forward_events(self):
        try:
            with self._ev_recv as events:
                async for event in events:
                    # Any event = the runner made progress; reset the hang clock.
                    self._last_event_monotonic = time.monotonic()
                    if isinstance(event, RunnerTerminationError):
                        # try to get exception if possible
                        await self._check_runner(event)
                        break
                    if isinstance(event, RunnerStatusUpdated):
                        self.status = event.runner_status
                    if isinstance(event, TaskAcknowledged):
                        self.pending.pop(event.task_id).set()
                        continue
                    if (
                        isinstance(event, TaskStatusUpdated)
                        and event.task_status == TaskStatus.Complete
                    ):
                        # If a task has just been completed, we should be working on it.
                        assert isinstance(
                            self.status,
                            (
                                RunnerRunning,
                                RunnerWarmingUp,
                                RunnerLoading,
                                RunnerConnecting,
                                RunnerShuttingDown,
                            ),
                        )
                        self.in_progress.pop(event.task_id, None)
                        self.completed.add(event.task_id)
                    await self._event_sender.send(event)
        except (ClosedResourceError, BrokenResourceError):
            # this is the happy path shutdown - we don't need to spam log with it
            await self._check_runner()
        finally:
            for tid in self.pending:
                self.pending[tid].set()

    async def _watch_runner(self) -> None:
        with self._cancel_watch_runner:
            while True:
                await anyio.sleep(5)
                if not self.runner_process.is_alive():
                    await self._check_runner(RuntimeError("Runner found to be dead"))
                    return
                self._check_hang()
                self._check_stuck_init()

    def _check_hang(self) -> None:
        """SIGKILL a runner that has in-progress work but has gone silent.

        Detects the c>=2 degen-kill / GPU-timeout wedge: the runner spins at
        100% CPU inside a native jaccl collective, so it is_alive() (this loop's
        other check never fires) yet emits no events. SIGTERM does not reach the
        interpreter mid-spin, so we go straight to SIGKILL; the next _watch_runner
        tick then sees is_alive()==False and raises RunnerFailed, which the
        master turns into instance teardown + re-placement (self-heal).

        Gated on in_progress being non-empty so model load / idle windows (which
        legitimately emit no events) never trip it. Fires once per runner.
        """
        if (
            HANG_TIMEOUT_SECONDS <= 0
            or self._hang_killed
            or not self.in_progress
            or not self.runner_process.is_alive()
        ):
            return
        silent_for = time.monotonic() - self._last_event_monotonic
        if silent_for < HANG_TIMEOUT_SECONDS:
            return
        self._hang_killed = True
        logger.critical(
            f"Runner {self.bound_instance.bound_runner_id} hung: "
            f"{len(self.in_progress)} task(s) in progress, no event for "
            f"{silent_for:.0f}s (>{HANG_TIMEOUT_SECONDS:.0f}s). SIGKILLing to "
            f"force RunnerFailed + re-placement."
        )
        with contextlib.suppress(Exception):
            os.kill(self.runner_process.pid, signal.SIGKILL)

    def _check_stuck_init(self) -> None:
        """SIGKILL a runner wedged during bring-up (never reached serving).

        Companion to _check_hang for the PRE-serving phase. _check_hang is
        gated on in_progress being non-empty, so a runner that wedges while
        still connecting/loading — most importantly the jaccl re-place that
        gets stuck in RunnerConnecting (QP->RTR never completes) after a peer's
        mid-GPU-op SIGKILL corrupts RDMA state — has no in_progress task, emits
        no event, and would otherwise hang forever (the "needs a reboot" mode).

        If the runner has emitted no event for CONNECT_TIMEOUT_SECONDS while
        still NOT serving (not Ready/Running), SIGKILL it: is_alive() flips
        false -> RunnerFailed -> master re-places (self-heal). Any status/
        progress event bumps _last_event_monotonic, so a healthy bring-up that
        keeps making progress never trips this. Fires once per runner.
        """
        if (
            CONNECT_TIMEOUT_SECONDS <= 0
            or self._hang_killed
            or not self.runner_process.is_alive()
            or isinstance(self.status, (RunnerReady, RunnerRunning))
        ):
            return
        silent_for = time.monotonic() - self._last_event_monotonic
        if silent_for < CONNECT_TIMEOUT_SECONDS:
            return
        self._hang_killed = True
        logger.critical(
            f"Runner {self.bound_instance.bound_runner_id} stuck in bring-up: "
            f"status={type(self.status).__name__}, no event for "
            f"{silent_for:.0f}s (>{CONNECT_TIMEOUT_SECONDS:.0f}s). SIGKILLing to "
            f"force RunnerFailed + re-placement."
        )
        with contextlib.suppress(Exception):
            os.kill(self.runner_process.pid, signal.SIGKILL)

    async def _check_runner(
        self, e: RunnerTerminationError | Exception | None = None
    ) -> None:
        if not self._cancel_watch_runner.cancel_called:
            self._cancel_watch_runner.cancel()
        logger.info("Checking runner's status")
        if self.runner_process.is_alive():
            logger.info("Runner was found to be alive, stopping process")
            with anyio.CancelScope(shield=True):
                await self.runner_process.stop()
        rc = self.runner_process.exitcode
        logger.info(f"Runner exited with exit code {rc}")

        # A clean (rc==0) exit normally means any transient errors were
        # recoverable and no diagnostics are needed.
        #
        # BUT rc==0 is also what we observe when *we* SIGTERM a runner that is
        # still mid-generation (e.g. the runner aborted a forward pass and the
        # event channel closed, so _forward_events called _check_runner while
        # generation tasks were still in flight). In that case the exit code is
        # 0 only because the process was stopped cleanly on the way down — the
        # task was abandoned. Returning here leaves the instance marked healthy:
        # the master keeps it "ready" in the UI and hot-loops re-dispatching to
        # a dead runner ("Skipping invalid task ... already submitted") until
        # someone manually deletes the instance. Fall through to surface a
        # RunnerFailed so the master tears the instance down instead.
        abandoned_generation = [
            t
            for t in self.in_progress.values()
            if isinstance(t, (TextGeneration, ImageGeneration, ImageEdits))
        ]
        if rc == 0 and not abandoned_generation:
            return

        if isinstance(rc, int) and rc < 0:
            sig = -rc
            try:
                if (description := signal.strsignal(sig)) is not None:
                    cause = f"signal={sig} ({description})"
                else:
                    cause = f"signal={sig}"
            except Exception:
                cause = f"signal={sig}"
        else:
            cause: str = f"exitcode={rc}"

        if e is not None:
            # Record how runner shut down, try exception, resort to RunnerTerminationError fallback
            if isinstance(e, Exception):
                logger.opt(exception=e).error(f"Runner terminated with {cause}")
            else:
                cause = f"{cause}\nRunner error: {e}"
                logger.error(f"Runner terminated with {cause}")
        else:
            logger.error(f"Runner terminated with {cause}")

        # Surface the raw runner stderr tail into the MAIN process log. The
        # real crash cause (segfault traceback, ImportError, pydantic error)
        # otherwise lives only in runner_log/stderr.log and is easy to miss
        # mid-incident — especially when nothing classified into a known
        # diagnostic (the failure then looks like a silent clean exit upstream).
        stderr_tail = self._runner_stdio_handler.diagnostics.stderr_tail()
        if stderr_tail:
            logger.error(
                "Runner stderr tail ({} lines) for {}:\n{}".format(
                    len(stderr_tail),
                    self.bound_instance.bound_runner_id,
                    "\n".join(stderr_tail),
                )
            )

        diagnostics = [
            d
            for d in self._runner_stdio_handler.diagnostics.diagnostics()
            if not isinstance(d, RunnerUnknown)
        ]
        for task in self.in_progress.values():
            if isinstance(task, (TextGeneration, ImageGeneration, ImageEdits)):
                with anyio.CancelScope(shield=True):
                    await self._event_sender.send(
                        ChunkGenerated(
                            command_id=task.command_id,
                            chunk=ErrorChunk(
                                model=self.shard_metadata.model_card.model_id,
                                diagnostics=diagnostics,
                                error_message=(
                                    "Runner shutdown before completing command "
                                    f"({cause})"
                                ),
                            ),
                        )
                    )

        try:
            self.status = RunnerFailed(
                error_message=f"Terminated ({cause})", diagnostics=diagnostics
            )
            with anyio.CancelScope(shield=True):
                await self._event_sender.send(
                    RunnerStatusUpdated(
                        runner_id=self.bound_instance.bound_runner_id,
                        runner_status=self.status,
                    )
                )
        except (ClosedResourceError, BrokenResourceError):
            logger.warning(
                "Event sender already closed, unable to report runner failure"
            )
        self.shutdown()
