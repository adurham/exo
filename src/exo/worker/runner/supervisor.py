import codecs
import contextlib
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from os import PathLike
from typing import Callable, Optional, Self

import anyio
from anyio import (
    AsyncFile,
    BrokenResourceError,
    CancelScope,
    ClosedResourceError,
    to_thread,
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

# LIVENESS PROBE (2026-09-21). Root cause of two independently-reproduced
# false-positive hang-kills this session (a from-scratch cache-busting
# prefill's first chunk, and fresh model loading — both 100% reproducible):
# the event-silence heuristic above cannot distinguish "genuinely wedged"
# from "real GPU compute that happens to not have emitted its own progress
# event yet". Both false-positive cases were confirmed via THIS SAME
# `sample` diagnostic (already fired post-mortem, right before the kill) to
# be inside live `mlx::core::eval()` / real Metal dispatch calls with an
# actively large-to-growing physical memory footprint at the moment of the
# wrongful kill — not a spin-loop, not a deadlock.
#
# This turns that same diagnostic into a PRE-kill liveness check: once
# HANG_TIMEOUT_SECONDS of silence has elapsed, take a cheap 1s `sample`
# and record the process's physical footprint. If a SECOND probe
# (HANG_PROBE_INTERVAL_SECONDS later) shows the footprint grew by at least
# HANG_PROBE_GROWTH_THRESHOLD_GB, that is real forward progress (weight
# materialization, KV-cache construction, etc.) — extend and re-check later
# rather than kill. If the footprint has genuinely plateaued, or `sample`
# itself fails, fall through to the original kill path unchanged. Capped by
# HANG_PROBE_MAX_EXTENSIONS so a runner that is truly wedged AFTER some
# initial real growth (e.g. finishes loading, then wedges on a subsequent
# collective) is still caught, not given infinite reprieve.
#
# Deliberately NOT a larger HANG_TIMEOUT_SECONDS default: a blind timeout
# bump would also delay detection of the genuine wedge this watchdog exists
# to catch (a runner spinning at 100% CPU inside a hung native collective)
# for its entire increased duration. This mechanism only extends when there
# is verified evidence of real work — a wedged process's footprint is
# static by definition, so it is caught at the FIRST probe interval past
# the original timeout, same as before this change for the case the
# watchdog was actually designed for.
HANG_PROBE_INTERVAL_SECONDS = _env_seconds(
    "EXO_RUNNER_HANG_PROBE_INTERVAL_SECONDS", 20.0
)
HANG_PROBE_GROWTH_THRESHOLD_GB = _env_seconds("EXO_RUNNER_HANG_PROBE_GROWTH_GB", 0.25)
HANG_PROBE_MAX_EXTENSIONS = int(
    _env_seconds("EXO_RUNNER_HANG_PROBE_MAX_EXTENSIONS", 20.0)
)


def _sample_physical_footprint_gb(pid: int, duration_s: int = 1) -> float | None:
    """Read `pid`'s current physical memory footprint (GB) via a short,
    read-only `sample` invocation — the SAME tool already used (at a longer
    duration) for the post-mortem hang diagnostic below, just run earlier
    and non-destructively as live evidence of whether the process is still
    doing real work. `sample` is a statistical profiler (microstackshots via
    task_for_pid) — it does not SIGSTOP or otherwise pause the target the
    way an xctrace/Instruments attach does, so this is safe to call
    repeatedly on a live, busy process.

    Returns None on any failure (missing binary, timeout, unparseable
    output) — callers must treat None as "no evidence either way" and fall
    back to the original kill decision, never as "confirmed hung".
    """
    try:
        result = subprocess.run(
            ["/usr/bin/sample", str(pid), str(duration_s)],
            capture_output=True,
            timeout=duration_s + 10,
            check=False,
            text=True,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped.startswith("Physical footprint:"):
            continue
        value_str = stripped.split(":", 1)[1].strip()
        try:
            if value_str.endswith("G"):
                return float(value_str[:-1])
            if value_str.endswith("M"):
                return float(value_str[:-1]) / 1024.0
            if value_str.endswith("K"):
                return float(value_str[:-1]) / (1024.0 * 1024.0)
            return float(value_str) / (1024.0**3)
        except ValueError:
            return None
    return None


# Symbols that identify a task BLOCKED IN NATIVE NETWORK/COLLECTIVE SETUP.
# A process parked in one of these is waiting on the peer, not spinning and
# not deadlocked in our code -- and crucially it consumes ~zero memory while
# blocked, so the footprint-growth probe alone misreads it as a hang.
#
# CONFIRMED CASE (2026-09-23): a 565K deep-context request had its runner
# SIGKILLed at 69s of silence. /tmp/exo_hang_32526.txt shows the main thread
# at 387 MB footprint (vs ~87 GB steady state, i.e. NOT yet loaded) in:
#   mlx::core::distributed::init
#     -> mlx::core::distributed::jaccl::init
#       -> jaccl::init(jaccl::Config const&, bool)
#         -> jaccl::Config::get_side_channel() const
#           -> jaccl::TCPAllGather::TCPAllGather(int, int, char const*)
# That is RDMA/JACCL side-channel bring-up blocking on a TCP all-gather with
# the peer. Genuine progress: it completes once the peer arrives. The kill
# destroyed a healthy request and returned an empty response.
#
# This is a THIRD manifestation of the watchdog false-positive class (the
# skill reference documents cold cache-busting prefill and model loading).
# The distinguishing evidence is the STACK, which the footprint probe never
# looked at.
_NATIVE_BLOCKED_SYMBOLS = (
    "get_side_channel",
    "TCPAllGather",
    "jaccl::init",
    "distributed::init",
)


def _run_sample(pid: int, duration_s: int) -> str | None:
    """Run `/usr/bin/sample <pid> <duration_s>` and return its stdout as text.

    `sample` is a statistical profiler (microstackshots via task_for_pid); it
    does NOT SIGSTOP the target, so it is safe on a live busy process. Returns
    None on any failure (missing binary, timeout, empty output) so callers
    treat it as "no evidence".
    """
    try:
        result = subprocess.run(
            ["/usr/bin/sample", str(pid), str(duration_s)],
            capture_output=True,
            timeout=duration_s + 10,
            check=False,
            text=True,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    # The bytes guard is NOT dead code despite text=True: this repo's own test
    # helpers build CompletedProcess(stdout=b"...") -- bytes. Keep it
    # fail-safe (a first draft crashed on exactly that).
    raw_stdout = result.stdout
    if not isinstance(raw_stdout, str):  # pyright: ignore[reportUnnecessaryIsInstance]
        if not isinstance(raw_stdout, bytes):  # pyright: ignore[reportUnnecessaryIsInstance]
            return None
        try:
            raw_stdout = raw_stdout.decode("utf-8", errors="replace")
        except Exception:
            return None
    if not raw_stdout:
        return None
    return raw_stdout


def _main_thread_blob(raw: str) -> str:
    """The main-thread section of a `sample` dump (whole dump if no thread
    header is found), so a parked background thread cannot mask or replace
    the signal."""
    main: list[str] = []
    in_main = False
    for line in raw.splitlines():
        if "main-thread" in line or "Thread_" in line:
            in_main = "main-thread" in line
        if in_main:
            main.append(line)
    return "\n".join(main) if main else raw


def _native_setup_hits(raw: str) -> list[str]:
    """Distinct native-setup symbols present in `raw`'s main-thread blob."""
    blob = _main_thread_blob(raw)
    return [s for s in _NATIVE_BLOCKED_SYMBOLS if s in blob]


# Retained as the documented, directly-tested entry point for the narrow
# native-setup check (see tests/unittests/test_runner/
# test_runner_supervisor_hang_probe.py). The stack-class guard reuses the same
# primitives; this wrapper stays for that public contract and its test suite.
def _sample_is_blocked_in_native_setup(  # pyright: ignore[reportUnusedFunction]
    pid: int, duration_s: int = 2
) -> bool:
    """True if `pid`'s main thread is parked in native network/collective
    SETUP (JACCL side-channel / TCP all-gather), i.e. waiting on a peer.

    This is the discriminator the footprint probe lacks. Such a task is NOT
    doing compute, so its footprint is flat by definition, and it is NOT
    wedged either -- it proceeds as soon as the peer responds.

    Returns False on any failure, matching the footprint probe's contract:
    absence of evidence is never treated as evidence of a hang.
    """
    raw = _run_sample(pid, duration_s)
    if raw is None:
        return False
    hits = _native_setup_hits(raw)
    if len(hits) >= 2:
        # Require >=2 distinct setup symbols so an incidental single mention
        # elsewhere in the dump cannot arm this.
        logger.warning(
            f"Runner pid {pid} main thread appears BLOCKED IN NATIVE SETUP "
            f"(symbols={hits}) -- awaiting peer, not hung; extending instead "
            f"of killing."
        )
        return True
    return False


# ── Stack-class hang guard (2026-10-05, design doc "Fix B") ───────────────
#
# The growth probe above cannot discriminate at the hardware ceiling: a
# healthy deep-context prefill pinned at ~physical size shows a "flat"
# footprint and was SIGKILLed as hung (2026-10-04 soak-2 postmortem). "Flat"
# is not evidence of a wedge. This layer classifies the paused runner's stack
# AND its CPU burn before a flat-footprint tick is allowed to kill:
#
#   growth                                   -> extend (unchanged)
#   flat + SPIN                              -> kill fast (jaccl 100%-CPU wedge)
#   flat + blocked + gpu + at-ceiling        -> extend (bounded)
#   flat + blocked + native-setup (narrow)   -> extend (unchanged path)
#   flat + blocked + unknown                 -> kill
#   classifier/sample failure                -> extend once + alert, then kill
#
# SHADOW is the default: the classifier runs on plateau ticks and logs the
# verdict + every input field, but the kill decision is left EXACTLY as it is
# today. Arm only after a soak's shadow logs validate the classifier.
#
# The SPIN metric separates the two incident classes on one axis: healthy
# at-ceiling prefill is BLOCKED (main thread in Scheduler::wait_for_one,
# workers in __psynch_cvwait) with a near-zero CPU-time delta, while the wedge
# burns ~100% of a core. All extensions draw from the ONE shared monotonic
# budget (HANG_PROBE_MAX_EXTENSIONS); reclassification never resets it, and a
# real event still resets everything (existing behavior).
_stack_mode_raw = os.environ.get("EXO_RUNNER_HANG_STACK_MODE", "shadow").strip().lower()
if _stack_mode_raw not in ("off", "shadow", "arm"):
    logger.warning(
        f"EXO_RUNNER_HANG_STACK_MODE={_stack_mode_raw!r} is not one of "
        "off|shadow|arm — defaulting to 'shadow'."
    )
HANG_STACK_MODE = (
    _stack_mode_raw if _stack_mode_raw in ("off", "shadow", "arm") else "shadow"
)
HANG_SPIN_FRACTION = _env_seconds("EXO_RUNNER_HANG_SPIN_FRACTION", 0.5)
HANG_CEILING_MARGIN_GB = _env_seconds("EXO_RUNNER_HANG_CEILING_MARGIN_GB", 2.0)

# Symbols/images that identify a runner BLOCKED IN LIVE GPU COMPUTE (mlx eval
# + Metal scheduler frames; AGXMetal*/IOGPU are the Metal driver images,
# libmlx the mlx dylib). A process parked here is doing real work that simply
# has not emitted its own progress event yet — NOT a wedge. This must NOT
# overlap the narrow native-setup signature above: the jaccl collective wedge
# stays on the fast-kill path precisely because it matches neither set.
_GPU_STACK_SYMBOLS = (
    "mlx::core::eval",
    "mlx::core::eval_impl",
    "Scheduler::wait_for_one",
    "MetalAllocator",
    "metal::allocator",
)
_GPU_STACK_IMAGES = ("AGXMetal", "IOGPU", "libmlx")


def _parse_cputime_seconds(text: str) -> float | None:
    """Parse a `ps -o cputime=` value into seconds.

    macOS formats are ``mm:ss[.ss]``, ``hh:mm:ss[.ss]`` and
    ``dd-hh:mm:ss[.ss]``. Returns None on anything unexpected so the caller
    falls back to "no evidence" rather than a wrong number.
    """
    if not text:
        return None
    days = 0
    if "-" in text:
        day_str, _, text = text.partition("-")
        try:
            days = int(day_str)
        except ValueError:
            return None
    try:
        nums = [float(part) for part in text.split(":")]
    except ValueError:
        return None
    if len(nums) == 3:
        hours, minutes, seconds = nums
        return days * 86400.0 + hours * 3600.0 + minutes * 60.0 + seconds
    if len(nums) == 2:
        minutes, seconds = nums
        return days * 86400.0 + minutes * 60.0 + seconds
    if len(nums) == 1:
        return days * 86400.0 + nums[0]
    return None


def _read_ps_cputime_seconds(pid: int) -> float | None:
    """Cumulative CPU time (seconds) for `pid` via `ps -o cputime=`.

    Drives the SPIN metric: the delta between two probe ticks divided by the
    wall interval estimates the fraction of a core the runner is burning. A
    wedged jaccl collective spins at ~1.0; a healthily-blocked runner at the
    memory ceiling sits near 0.

    Returns None on any failure (missing binary, non-zero exit, unparseable
    output) — callers treat None as "no spin evidence" and must NOT
    spin-classify on it.
    """
    try:
        proc = subprocess.run(
            ["/bin/ps", "-o", "cputime=", "-p", str(pid)],
            capture_output=True,
            timeout=2,
            check=False,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    if proc.returncode != 0:
        return None
    text = proc.stdout.decode("utf-8", errors="replace").strip()
    return _parse_cputime_seconds(text)


@dataclass
class _MemsizeCache:
    probe_done: bool = False
    bytes_total: int | None = None


_MEMSIZE_CACHE = _MemsizeCache()


def _host_memsize_bytes() -> int | None:
    """Total physical memory in bytes via `sysctl -n hw.memsize`, cached.

    Returns None on failure, which callers treat as "not at ceiling" (never as
    "at ceiling") — a missing ceiling reading must not arm an extension.
    """
    if _MEMSIZE_CACHE.probe_done:
        return _MEMSIZE_CACHE.bytes_total
    _MEMSIZE_CACHE.probe_done = True
    try:
        proc = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "hw.memsize"],
            capture_output=True,
            timeout=2,
            check=False,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    try:
        value = int(proc.stdout.decode("utf-8", errors="replace").strip())
    except ValueError:
        return None
    if value <= 0:
        return None
    _MEMSIZE_CACHE.bytes_total = value
    return value


def _sample_stack_class(pid: int, duration_s: int = 2) -> str | None:
    """Classify `pid`'s stack from a short `sample`.

    Returns:
      "native-setup" — main thread parked in the narrow JACCL/TCP setup chain
                       (reuses ``_NATIVE_BLOCKED_SYMBOLS`` UNCHANGED).
      "gpu"          — mlx eval / Metal scheduler / Metal-driver frames
                       (libmlx, AGXMetal*, IOGPU).
      "unknown"      — sampled successfully but matched no known signature.
      None           — the sample itself failed (no evidence). Callers must
                       treat this as a classifier FAILURE, not as "unknown".
    """
    raw = _run_sample(pid, duration_s)
    if raw is None:
        return None
    # Native-setup is a MAIN-THREAD signature (same blob rule as the existing
    # classifier), so an incidental mention elsewhere cannot arm it.
    if len(_native_setup_hits(raw)) >= 2:
        return "native-setup"
    # GPU frames are matched across the WHOLE dump (symbols and image names):
    # a working runner's workers also touch libmlx/AGXMetal, and the driver
    # image names are the robust, version-independent signal.
    if any(s in raw for s in _GPU_STACK_SYMBOLS) or any(
        i in raw for i in _GPU_STACK_IMAGES
    ):
        return "gpu"
    return "unknown"


def _decide_plateau_verdict(
    *,
    growth_gb: float,
    spin: bool | None,
    stack_class: str | None,
    at_ceiling: bool,
    sample_failed_once: bool,
) -> str:
    """Pure decision for a plateau probe tick.

    Returns one of "extend", "kill", or "failure-extend" (a classifier/sample
    failure earns exactly ONE extension; a repeat failure kills). The caller
    owns the shared extension budget and the failure-once latch.
    """
    if growth_gb >= HANG_PROBE_GROWTH_THRESHOLD_GB:
        return "extend"
    # Flat footprint from here on.
    if spin is True:
        # 100%-CPU wedge class — kill fast, regardless of stack or ceiling.
        return "kill"
    if stack_class is None:
        # Classifier/sample failure: extend once + alert, then kill on repeat.
        return "kill" if sample_failed_once else "failure-extend"
    if stack_class == "native-setup":
        return "extend"
    if stack_class == "gpu" and at_ceiling:
        return "extend"
    # gpu-not-at-ceiling and unknown both kill.
    return "kill"


@dataclass(frozen=True)
class _PlateauEvidence:
    """Sampled inputs for one plateau-tick verdict (logged in every mode)."""

    stack_class: str | None
    spin: bool | None
    cpu_delta_seconds: float | None
    wall_interval_seconds: float | None
    at_ceiling: bool


def _process_is_stopped_or_traced(pid: int) -> bool:
    """Return True if `pid` reports a macOS process state containing 'T'.

    `ps -o state=` prints per-process state flags. On macOS/BSD 'T' means the
    process is STOPPED (SIGSTOP) or being TRACED (ptrace/xctrace attach). A
    profiling attach — `xcrun xctrace record --attach <pid> --time-limit ...`
    for a Metal System Trace — puts the target in exactly this state, which
    suppresses all runner event emission and would otherwise trip _check_hang
    and get the runner SIGKILLed mid-capture (2026-08-22 incident, P2 capture
    killer). Best-effort and time-boxed: on any subprocess error we return
    False so the caller falls back to the existing kill path.

    We shell out to `ps` rather than pull in a new dependency; psutil is
    already available but this keeps the check parseable and easy to unit-test
    via a mocked subprocess call.
    """
    try:
        proc = subprocess.run(
            ["/bin/ps", "-o", "state=", "-p", str(pid)],
            capture_output=True,
            timeout=2,
            check=False,
        )
    except (subprocess.SubprocessError, OSError):
        return False
    if proc.returncode != 0:
        return False
    state = proc.stdout.decode("utf-8", errors="replace").strip()
    # BSD state field: first char is the primary state; on macOS a 'T' anywhere
    # in the flag string means stopped/traced. Empty output = process gone.
    return "T" in state


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
    # Liveness-probe state (see HANG_PROBE_* above): the footprint reading
    # from the PREVIOUS probe (None until the first probe has run for this
    # silence episode) and how many extensions have been granted so far.
    # Reset to None/0 the moment any real event arrives (see _forward_events)
    # so a fresh silence episode always starts its own probe sequence rather
    # than inheriting stale readings from an earlier one.
    _hang_probe_last_footprint_gb: float | None = field(default=None, init=False)
    _hang_probe_extensions_used: int = field(default=0, init=False)
    # The kill is deferred until THIS monotonic timestamp once an extension is
    # granted -- checked on EVERY tick (not just at probe-interval boundaries),
    # which is the fix for the 2026-09-21 bug where an extension only
    # protected the exact tick it was granted on: the interval gate that
    # decided WHEN to probe next was also (wrongly) gating the kill decision
    # itself, so a granted extension had no effect on the ticks in between and
    # the kill fired anyway on the very next tick. 0.0 means "no extension
    # currently in effect" (initial state and post-reset state).
    _hang_probe_deadline_monotonic: float = field(default=0.0, init=False)
    # Stack-class guard state (design doc "Fix B"). The CPU-time reading from
    # the previous probe tick, used for the SPIN delta; the once-latch for a
    # classifier/sample failure (one extension, then kill on repeat); and the
    # wall-clock of the previous probe tick that anchors the CPU delta. All
    # three reset with the rest of the probe state on any real event.
    _hang_probe_last_cputime_seconds: float | None = field(default=None, init=False)
    _hang_probe_wall_last_monotonic: float = field(default=0.0, init=False)
    _hang_probe_sample_failed_once: bool = field(default=False, init=False)
    # Worker-injected predicate: True while a SIBLING runner on this node is
    # loading a model. A co-host JIT load saturates the GPU/memory bus and can
    # starve a mid-generation runner of progress for minutes (observed
    # 2026-07-09 16:15:08: DSv4 runner SIGKILLed at 298s silent while Qwen
    # weights loaded for a consult) — that is starvation, not a wedge, so the
    # hang watchdog must hold fire for the duration of the sibling load.
    sibling_loading: Optional[Callable[[], bool]] = None

    @classmethod
    async def create(
        cls,
        *,
        bound_instance: BoundInstance,
        event_sender: Sender[Event],
        initialize_timeout: float = 400,
        sibling_loading: Optional[Callable[[], bool]] = None,
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
            sibling_loading=sibling_loading,
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
                    # Any event = the runner made progress; reset the hang clock
                    # AND the liveness-probe state, so a fresh silence episode
                    # after real progress always starts its own probe sequence
                    # from scratch rather than inheriting a stale footprint
                    # reading or extension count from a previous episode.
                    self._last_event_monotonic = time.monotonic()
                    self._hang_probe_last_footprint_gb = None
                    self._hang_probe_extensions_used = 0
                    self._hang_probe_deadline_monotonic = 0.0
                    self._hang_probe_last_cputime_seconds = None
                    self._hang_probe_wall_last_monotonic = 0.0
                    self._hang_probe_sample_failed_once = False
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
                await self._check_hang()
                self._check_stuck_init()

    async def _sample_plateau_evidence(
        self, pid: int, now: float, footprint_gb: float
    ) -> _PlateauEvidence:
        """Collect the classifier inputs for one plateau tick.

        Only called in shadow/arm modes (off mode skips sampling entirely):
          - stack class via ``_sample_stack_class`` (None = sample failed),
          - SPIN = CPU-time delta between the previous probe tick and this one
            divided by the wall interval, vs HANG_SPIN_FRACTION. ps failure or
            a missing baseline yields None (no spin evidence) and must never
            produce a True verdict,
          - at-ceiling = footprint within HANG_CEILING_MARGIN_GB of hw.memsize
            (memsize failure => False, i.e. never arm on a missing reading).
        """
        cpu_now = await to_thread.run_sync(_read_ps_cputime_seconds, pid)
        wall_interval: float | None = None
        cpu_delta: float | None = None
        if (
            cpu_now is not None
            and self._hang_probe_last_cputime_seconds is not None
            and now > self._hang_probe_wall_last_monotonic
        ):
            wall_interval = now - self._hang_probe_wall_last_monotonic
            cpu_delta = cpu_now - self._hang_probe_last_cputime_seconds
        if cpu_now is not None:
            self._hang_probe_last_cputime_seconds = cpu_now
        # Advance the wall anchor on every probe so the next delta uses a real
        # interval even if this tick had no usable CPU reading.
        self._hang_probe_wall_last_monotonic = now

        spin: bool | None = None
        if (
            cpu_delta is not None
            and wall_interval is not None
            and wall_interval > 0
            and cpu_delta >= 0
        ):
            spin = (cpu_delta / wall_interval) >= HANG_SPIN_FRACTION

        stack_class = await to_thread.run_sync(_sample_stack_class, pid, 2)

        memsize = await to_thread.run_sync(_host_memsize_bytes)
        at_ceiling = (
            memsize is not None
            and footprint_gb >= (memsize / (1024.0**3)) - HANG_CEILING_MARGIN_GB
        )
        return _PlateauEvidence(
            stack_class=stack_class,
            spin=spin,
            cpu_delta_seconds=cpu_delta,
            wall_interval_seconds=wall_interval,
            at_ceiling=at_ceiling,
        )

    async def _check_hang(self) -> None:
        """SIGKILL a runner that has in-progress work but has gone silent AND
        shows no verified real-memory growth (see the liveness probe below).

        Detects the c>=2 degen-kill / GPU-timeout wedge: the runner spins at
        100% CPU inside a native jaccl collective, so it is_alive() (this loop's
        other check never fires) yet emits no events. SIGTERM does not reach the
        interpreter mid-spin, so we go straight to SIGKILL; the next _watch_runner
        tick then sees is_alive()==False and raises RunnerFailed, which the
        master turns into instance teardown + re-placement (self-heal).

        Gated on in_progress being non-empty so model load / idle windows (which
        legitimately emit no events) never trip it. Fires once per runner.

        NOTE (2026-08-20): long-but-legitimate native recovery must signal
        progress rather than have this timeout raised. The runner's
        _warmup_with_reconnect() re-emits its status at reconnect START and
        reconnect COMPLETE precisely so a multi-cycle reconnect_fresh recovery
        (16-20 s per cycle, 60-100+ s across the retry budget) keeps bumping
        _last_event_monotonic on VERIFIED progress. Any future path that can
        legitimately block in native code past this window should do the same;
        do not widen HANG_TIMEOUT_SECONDS to cover it, since that also delays
        detection of genuinely wedged runners.

        LIVENESS PROBE (2026-09-21): two independently-reproduced false
        positives this session (a from-scratch cache-busting prefill's first
        chunk; fresh model loading) proved the runner can be inside real,
        progressing GPU compute for well over HANG_TIMEOUT_SECONDS with a
        correctly-wired-but-not-yet-fired progress event. Once the silence
        threshold is reached, this now probes the process's physical memory
        footprint before killing: if it has grown by
        HANG_PROBE_GROWTH_THRESHOLD_GB since the last probe, that is
        externally-verified real progress -- extend (bounded by
        HANG_PROBE_MAX_EXTENSIONS) rather than kill. A genuinely wedged
        process's footprint is static by construction, so it is still caught
        at the first probe interval past the original timeout -- this does
        NOT weaken detection of the wedge this watchdog was built for, it
        only stops it from firing on real, still-progressing work.
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
        if self.sibling_loading is not None and self.sibling_loading():
            # A sibling runner is mid-model-load on this node: silence here is
            # GPU starvation, not a wedge. Reset the clock so the runner gets
            # a full HANG_TIMEOUT window after the load finishes; a genuinely
            # wedged runner is still killed one timeout past load completion.
            logger.warning(
                f"Runner {self.bound_instance.bound_runner_id} silent for "
                f"{silent_for:.0f}s but a sibling runner is loading a model — "
                "deferring hang watchdog until the load completes."
            )
            self._last_event_monotonic = time.monotonic()
            return
        if _process_is_stopped_or_traced(self.runner_process.pid):
            # A profiling attach — `xcrun xctrace record --attach <pid>` for a
            # Metal System Trace — SIGSTOPs / ptraces the target, suppressing
            # all event emission. Killing here destroys the in-flight capture
            # (2026-08-22, P2 killer). Defer: reset the clock so the runner
            # gets a full HANG_TIMEOUT window after the trace ends; a genuinely
            # dead-but-not-stopped runner still falls through to the kill path
            # below on subsequent ticks.
            logger.warning(
                f"Runner {self.bound_instance.bound_runner_id} appears "
                "stopped/traced (state T); deferring hang kill."
            )
            self._last_event_monotonic = time.monotonic()
            return

        now = time.monotonic()
        if HANG_PROBE_MAX_EXTENSIONS > 0:
            # BUG FIX (2026-09-21, same-day as the probe's introduction): a
            # granted extension MUST hold off the kill for every tick until
            # its deadline, not just the tick it was granted on. The original
            # shipped version used a single "next probe time" for BOTH "when
            # should I sample memory again" AND "am I allowed to kill yet",
            # which are different questions -- an extension set the next
            # PROBE time correctly but left the KILL falling through
            # unconditionally on every tick that wasn't itself a probe tick.
            # Caught live: a real 106K-token cache-busting prefill got a
            # baseline probe + a logged "extending 20s" at t=46s silent, then
            # was SIGKILLed anyway at t=54s -- 8s later, nowhere near the
            # promised 20s. This check is now unconditional on every tick:
            # if a deadline is active and not yet reached, defer, full stop,
            # no probe needed this tick.
            if now < self._hang_probe_deadline_monotonic:
                return
            footprint_gb = await to_thread.run_sync(
                _sample_physical_footprint_gb, self.runner_process.pid, 1
            )
            if footprint_gb is None:
                logger.warning(
                    f"Runner {self.bound_instance.bound_runner_id} silent for "
                    f"{silent_for:.0f}s; liveness probe could not read memory "
                    "footprint (sample failed/unavailable) — proceeding to kill "
                    "with no growth evidence either way."
                )
            elif self._hang_probe_last_footprint_gb is None:
                # First probe of this silence episode: nothing to compare
                # against yet. Record it and extend once so the NEXT probe
                # has a baseline to diff against -- a real hang is still
                # caught one probe interval later than before this change,
                # a real load/prefill gets a chance to show its growth.
                self._hang_probe_last_footprint_gb = footprint_gb
                self._hang_probe_extensions_used += 1
                self._hang_probe_deadline_monotonic = now + HANG_PROBE_INTERVAL_SECONDS
                # Anchor the SPIN metric here too, so the very next plateau
                # tick can compute a CPU delta over a real probe interval
                # (matching the incident timeline: baseline t=46, plateau
                # t=66). Skipped entirely in off mode (no sampling overhead).
                if HANG_STACK_MODE != "off":
                    cpu0 = await to_thread.run_sync(
                        _read_ps_cputime_seconds, self.runner_process.pid
                    )
                    if cpu0 is not None:
                        self._hang_probe_last_cputime_seconds = cpu0
                    self._hang_probe_wall_last_monotonic = now
                logger.warning(
                    f"Runner {self.bound_instance.bound_runner_id} silent for "
                    f"{silent_for:.0f}s; liveness probe baseline footprint="
                    f"{footprint_gb:.2f}GB, extending "
                    f"{HANG_PROBE_INTERVAL_SECONDS:.0f}s for a growth check "
                    f"({self._hang_probe_extensions_used}/{HANG_PROBE_MAX_EXTENSIONS})."
                )
                return
            else:
                growth_gb = footprint_gb - self._hang_probe_last_footprint_gb
                self._hang_probe_last_footprint_gb = footprint_gb
                # GROWTH: real forward progress (unchanged path, all modes).
                if (
                    growth_gb >= HANG_PROBE_GROWTH_THRESHOLD_GB
                    and self._hang_probe_extensions_used < HANG_PROBE_MAX_EXTENSIONS
                ):
                    self._hang_probe_extensions_used += 1
                    self._hang_probe_deadline_monotonic = (
                        now + HANG_PROBE_INTERVAL_SECONDS
                    )
                    logger.warning(
                        f"Runner {self.bound_instance.bound_runner_id} silent for "
                        f"{silent_for:.0f}s but memory footprint grew "
                        f"{growth_gb:+.2f}GB since the last probe (now "
                        f"{footprint_gb:.2f}GB) — real progress, not a hang. "
                        f"Extending {HANG_PROBE_INTERVAL_SECONDS:.0f}s "
                        f"({self._hang_probe_extensions_used}/{HANG_PROBE_MAX_EXTENSIONS})."
                    )
                    return

                # FLAT footprint, or growth with a spent budget: consult the
                # stack-class guard. A spent budget kills first (pre-existing
                # behavior — a slow leak or an endless load must not stall the
                # watchdog forever), in every mode.
                budget_exhausted = (
                    self._hang_probe_extensions_used >= HANG_PROBE_MAX_EXTENSIONS
                )
                if budget_exhausted:
                    logger.warning(
                        f"Runner {self.bound_instance.bound_runner_id} silent for "
                        f"{silent_for:.0f}s; extension budget exhausted "
                        f"({self._hang_probe_extensions_used}"
                        f"/{HANG_PROBE_MAX_EXTENSIONS}) — proceeding to kill."
                    )
                    self._hang_killed = True
                    self._do_hang_kill(silent_for)
                    return
                if HANG_STACK_MODE == "off":
                    # Guard disabled: ZERO classifier/sample overhead. A flat
                    # footprint kills (the pre-guard behavior). NOTE: because
                    # no stack is inspected, `off` also does not extend on the
                    # pre-existing native-setup signature — `off` is the most
                    # kill-aggressive mode. Shadow (the default) DOES preserve
                    # that extension, so the shipped default is unchanged.
                    logger.warning(
                        f"Runner {self.bound_instance.bound_runner_id} silent for "
                        f"{silent_for:.0f}s; liveness probe shows footprint "
                        f"plateaued (growth={growth_gb:+.2f}GB, extensions used="
                        f"{self._hang_probe_extensions_used}/{HANG_PROBE_MAX_EXTENSIONS}) "
                        "— proceeding to kill."
                    )
                    self._hang_killed = True
                    self._do_hang_kill(silent_for)
                    return

                evidence = await self._sample_plateau_evidence(
                    self.runner_process.pid, now, footprint_gb
                )
                verdict = _decide_plateau_verdict(
                    growth_gb=growth_gb,
                    spin=evidence.spin,
                    stack_class=evidence.stack_class,
                    at_ceiling=evidence.at_ceiling,
                    sample_failed_once=self._hang_probe_sample_failed_once,
                )
                logger.warning(
                    f"[HANG_STACK] mode={HANG_STACK_MODE} runner "
                    f"{self.bound_instance.bound_runner_id} silent for "
                    f"{silent_for:.0f}s verdict={verdict} "
                    f"stack_class={evidence.stack_class} spin={evidence.spin} "
                    f"footprint_gb={footprint_gb:.2f} growth_gb={growth_gb:+.2f} "
                    f"cpu_delta_s={evidence.cpu_delta_seconds} "
                    f"wall_s={evidence.wall_interval_seconds} "
                    f"spin_fraction={HANG_SPIN_FRACTION} "
                    f"at_ceiling={evidence.at_ceiling} "
                    f"extensions_used={self._hang_probe_extensions_used}"
                    f"/{HANG_PROBE_MAX_EXTENSIONS}"
                )
                if HANG_STACK_MODE == "shadow":
                    # Kill path UNCHANGED. Today's flat-footprint behavior is
                    # exactly "extend iff the stack shows native setup, else
                    # kill" (the classifier's native-setup class is the same
                    # main-thread signature as the incumbent check). Every NEW
                    # verdict above — spin, gpu+ceiling, unknown — is logged
                    # only and does not alter this decision.
                    if evidence.stack_class == "native-setup":
                        self._hang_probe_extensions_used += 1
                        self._hang_probe_deadline_monotonic = (
                            now + HANG_PROBE_INTERVAL_SECONDS
                        )
                        logger.warning(
                            f"Runner {self.bound_instance.bound_runner_id} silent for "
                            f"{silent_for:.0f}s, footprint flat ({growth_gb:+.2f}GB) "
                            "BUT stack shows native network/collective setup — "
                            "waiting on peer, not hung. Extending "
                            f"{HANG_PROBE_INTERVAL_SECONDS:.0f}s "
                            f"({self._hang_probe_extensions_used}/"
                            f"{HANG_PROBE_MAX_EXTENSIONS})."
                        )
                        return
                    self._hang_killed = True
                    self._do_hang_kill(silent_for)
                    return
                # ARM: enforce the decision table, drawing every extend from
                # the SAME monotonic budget (never reset on reclassification;
                # the budget-exhausted check above already killed when spent).
                if verdict == "kill":
                    self._hang_killed = True
                    self._do_hang_kill(silent_for)
                    return
                if verdict == "failure-extend":
                    # ONE failure extension; the latch makes a repeat failure
                    # return "kill" above, so this can never loop unboundedly.
                    self._hang_probe_sample_failed_once = True
                self._hang_probe_extensions_used += 1
                self._hang_probe_deadline_monotonic = now + HANG_PROBE_INTERVAL_SECONDS
                logger.warning(
                    f"Runner {self.bound_instance.bound_runner_id} silent for "
                    f"{silent_for:.0f}s; stack-class guard verdict={verdict} "
                    f"(stack_class={evidence.stack_class}, spin={evidence.spin}, "
                    f"at_ceiling={evidence.at_ceiling}) — extending "
                    f"{HANG_PROBE_INTERVAL_SECONDS:.0f}s "
                    f"({self._hang_probe_extensions_used}/{HANG_PROBE_MAX_EXTENSIONS})."
                )
                return

        # Fall-through kill, matching the pre-change behavior for the cases
        # that reach here: the footprint sample failed (no evidence either
        # way), or HANG_PROBE_MAX_EXTENSIONS is disabled (<=0, so the probe
        # block above was skipped entirely).
        self._hang_killed = True
        self._do_hang_kill(silent_for)

    def _do_hang_kill(self, silent_for: float) -> None:
        """SIGKILL the runner and capture a post-mortem thread dump first.

        Extracted unchanged from the tail of the original ``_check_hang`` so
        both the legacy and the stack-class kill paths share exactly one kill.
        """
        logger.critical(
            f"Runner {self.bound_instance.bound_runner_id} hung: "
            f"{len(self.in_progress)} task(s) in progress, no event for "
            f"{silent_for:.0f}s (>{HANG_TIMEOUT_SECONDS:.0f}s). SIGKILLing to "
            f"force RunnerFailed + re-placement."
        )
        # DIAGNOSTIC (2026-08-16, design doc Section 116): capture WHERE the
        # runner is stuck before killing it. Without this the SIGKILL destroys
        # the only evidence -- we learn a runner hung but never which call it
        # hung in, which is exactly the state the TP two-link bring-up hang was
        # left in. `sample` attaches to the live process and writes a call
        # graph for every thread; it is read-only and safe to run on a wedged
        # process. Best-effort and time-boxed: never let diagnostics delay or
        # prevent the kill.
        with contextlib.suppress(Exception):
            dump_path = f"/tmp/exo_hang_{self.runner_process.pid}.txt"
            _ = subprocess.run(
                ["/usr/bin/sample", str(self.runner_process.pid), "3", "-f", dump_path],
                capture_output=True,
                timeout=20,
                check=False,
            )
            logger.critical(f"[HANG_STACK] wrote thread dump to {dump_path}")
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
        # `e` non-None means the runner told us it died (RunnerTerminationError
        # from bootstrap's crash handler) or we detected it dead ourselves —
        # never classify that as a clean exit, even at rc==0. Bootstrap catches
        # a critical exception, sends the termination error, and exits 0; the
        # rc==0 early-return here used to swallow exactly that for NON-generation
        # work: a warmup crash (jaccl reliable deadline) emitted no RunnerFailed,
        # so the master kept the instance WARMING UP forever with zero runner
        # processes alive — the 2026-07-06 zombie, twice.
        if rc == 0 and e is None and not abandoned_generation:
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
        # Snapshot: the awaits below yield to the event loop, which can
        # mutate in_progress (task completion/cleanup) mid-iteration —
        # observed 2026-07-10 as "RuntimeError: dictionary changed size
        # during iteration" here, escalating a contained runner crash into
        # killing the whole exo-main process (no self-heal).
        for task in list(self.in_progress.values()):
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
