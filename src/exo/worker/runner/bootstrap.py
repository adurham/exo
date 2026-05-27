import ctypes
import os
import resource
import sys
import traceback
from dataclasses import dataclass
from typing import Self, cast

import loguru

from exo.shared.types.events import Event, RunnerStatusUpdated
from exo.shared.types.tasks import Task, TaskId
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runners import RunnerFailed
from exo.utils.channels import ClosedResourceError, MpReceiver, MpSender
from exo.worker.engines.base import Builder

logger: "loguru.Logger" = loguru.logger


@dataclass(frozen=True)
class RunnerTerminationError:
    """Records the cause of a runner subprocess termination.

    Sent back from the runner to the parent supervisor via the event
    channel just before the runner exits. The supervisor uses this to
    decide whether to restart the runner or fail the instance.
    """

    exception_type: str
    exception_message: str
    exception_repr: str
    traceback: str

    @classmethod
    def from_exception(cls, e: Exception) -> Self:
        return cls(
            exception_type=type(e).__qualname__,
            exception_message=str(e),
            exception_repr=repr(e),
            traceback="".join(
                traceback.TracebackException.from_exception(e).format(chain=True)
            ),
        )

    def __str__(self) -> str:
        return f"{self.exception_type}: {self.exception_message}\n{self.traceback}"


# QoS classes from <sys/qos.h>. Apple's documented values for
# pthread_set_qos_class_self_np().
_QOS_CLASSES = {
    "user_interactive": 0x21,
    "user_initiated": 0x19,
    "default": 0x15,
    "utility": 0x11,
    "background": 0x09,
}


def _set_self_qos_class_darwin(qos_class_name: str) -> None:
    """Pin the calling thread's QoS class on macOS via pthread_set_qos_class_self_np.

    On a multi-runner Mac Studio, sibling Python processes can drift to
    different QoS classes under contention; macOS may then deprioritize
    one of them, producing a stratified throughput pattern. Pinning each
    runner to user_initiated at startup keeps them on equal footing.

    No-op on non-Darwin platforms or if the API is unavailable.
    """
    if sys.platform != "darwin":
        return
    qos_value = _QOS_CLASSES.get(qos_class_name.lower())
    if qos_value is None:
        logger.warning(
            f"Unknown EXO_RUNNER_QOS={qos_class_name!r}; "
            f"valid values: {sorted(_QOS_CLASSES.keys())}. Skipping QoS pin."
        )
        return
    try:
        libsystem = ctypes.CDLL("/usr/lib/libSystem.dylib", use_errno=True)
        # int pthread_set_qos_class_self_np(qos_class_t qos_class, int relative_priority)
        fn = libsystem.pthread_set_qos_class_self_np
        fn.argtypes = [ctypes.c_uint, ctypes.c_int]
        fn.restype = ctypes.c_int
        rc = int(fn(qos_value, 0))  # pyright: ignore[reportAny]
        if rc != 0:
            err = ctypes.get_errno()
            logger.warning(
                f"pthread_set_qos_class_self_np({qos_class_name}) failed: rc={rc} errno={err}"
            )
            return
        logger.info(f"Runner QoS class pinned to {qos_class_name}")
    except OSError as e:
        logger.warning(f"Could not load libSystem to set QoS class: {e}")
    except AttributeError:
        logger.warning(
            "pthread_set_qos_class_self_np unavailable on this libSystem; skipping QoS pin"
        )


def _maybe_register_profiler_hook() -> None:
    """Install an mlx-lm profiler hook based on ``EXO_PROFILER`` / ``EXO_PROFILER_LEVEL``.

    ``EXO_PROFILER`` is a comma-separated list of hook variants:
      * ``spans`` — per-span wall-time accumulator (replaces the old
        ``EXO_MINIMAX_TRACE``); dumps on SIGUSR1 / atexit.
      * ``layer_memory`` — per-layer Metal memory snapshots (replaces the
        old ``EXO_PROFILE_LAYERS``); ``EXO_PROFILER_LEVEL=2`` adds pre-layer
        snapshots in addition to the post-layer ones.

    Unset / empty ⇒ no hook registered, all calls short-circuit to no-ops.
    """
    raw = os.environ.get("EXO_PROFILER", "").strip()
    if not raw:
        return

    requested = {p.strip().lower() for p in raw.split(",") if p.strip()}
    valid = {"spans", "layer_memory"}
    unknown = requested - valid
    if unknown:
        logger.warning(
            f"Unknown EXO_PROFILER variants {sorted(unknown)}; valid: "
            f"{sorted(valid)}. Skipping unknown."
        )
    requested &= valid
    if not requested:
        return

    try:
        from mlx_lm.profiler import (
            CompositeHook,
            MemorySnapshotHook,
            SpanProfilerHook,
            install_signal_dump,
            register,
        )
    except ImportError as e:
        logger.warning(f"Profiler hook unavailable in mlx-lm: {e}")
        return

    hooks: list[object] = []
    span_hook: SpanProfilerHook | None = None
    if "spans" in requested:
        span_hook = SpanProfilerHook()
        hooks.append(span_hook)
    if "layer_memory" in requested:
        level = int(os.environ.get("EXO_PROFILER_LEVEL", "1"))
        hooks.append(MemorySnapshotHook(level=level))

    if not hooks:
        return

    if len(hooks) == 1:
        register(hooks[0])  # pyright: ignore[reportArgumentType]
    else:
        register(CompositeHook(*hooks))  # pyright: ignore[reportArgumentType]

    if span_hook is not None:
        install_signal_dump(span_hook)
    logger.info(f"mlx-lm profiler hook registered: {sorted(requested)}")


def entrypoint(
    bound_instance: BoundInstance,
    event_sender: MpSender[Event | RunnerTerminationError],
    task_receiver: MpReceiver[Task],
    cancel_receiver: MpReceiver[TaskId],
    _logger: "loguru.Logger",
) -> None:
    global logger
    logger = _logger

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(max(soft, 2048), hard), hard))

    # Enable core dumps when EXO_RUNNER_COREDUMP=1. Defaults to off; gated
    # because production cores can pin tens of GB of address space to disk
    # on a SEGV. Pair with `sudo chmod 1777 /cores/` (or sysctl
    # kern.corefile=<writable-path>) so the unprivileged runner can land
    # the file.
    if os.environ.get("EXO_RUNNER_COREDUMP") == "1":
        try:
            resource.setrlimit(
                resource.RLIMIT_CORE,
                (resource.RLIM_INFINITY, resource.RLIM_INFINITY),
            )
            logger.info("Core dumps enabled (RLIMIT_CORE=unlimited)")
        except (ValueError, OSError) as e:
            logger.warning(f"Failed to raise RLIMIT_CORE: {e}")

    # Pin QoS so siblings on the same machine don't drift apart under contention.
    # Defaults to user_initiated on Darwin; set EXO_RUNNER_QOS=off to disable.
    qos_choice = os.environ.get("EXO_RUNNER_QOS", "user_initiated")
    if qos_choice.lower() != "off":
        _set_self_qos_class_darwin(qos_choice)

    fast_synch_override = os.environ.get("EXO_FAST_SYNCH")
    if fast_synch_override == "false":
        os.environ["MLX_METAL_FAST_SYNCH"] = "0"
    else:
        os.environ["MLX_METAL_FAST_SYNCH"] = "1"

    logger.info(f"Fast synch flag: {os.environ['MLX_METAL_FAST_SYNCH']}")

    _maybe_register_profiler_hook()

    # Downcast the union-typed sender to MpSender[Event] for the internal
    # Builder/Runner machinery that only knows Event. MpSender is variant-
    # safe in practice; the cast is just to appease the type checker.
    event_sender_downcast: MpSender[Event] = cast(MpSender[Event], event_sender)

    # Import main after setting global logger - this lets us just import logger from this module
    try:
        from exo.worker.runner.runner import Runner

        builder: Builder

        if bound_instance.is_image_model:
            from exo.worker.engines.image.builder import MfluxBuilder

            builder = MfluxBuilder(
                event_sender_downcast, cancel_receiver, bound_instance.bound_shard
            )
        else:
            from exo.worker.engines.mlx.patches import apply_mlx_patches

            apply_mlx_patches()

            from exo.worker.engines.mlx.builder import MlxBuilder

            # evil sharing of the event sender
            builder = MlxBuilder(
                model_id=bound_instance.bound_shard.model_card.model_id,
                event_sender=event_sender_downcast,
                cancel_receiver=cancel_receiver,
            )

        runner = Runner(bound_instance, builder, event_sender_downcast, task_receiver)
        runner.main()

    except ClosedResourceError:
        logger.warning("Runner communication closed unexpectedly")
    except Exception as e:
        # Use plain traceback.format_exc() instead of logger.opt(exception=e),
        # which invokes loguru's better_exceptions diagnose pass. That pass
        # introspects every frame's locals; with Hermes-sized payloads
        # (large prompts + tool definitions in the local scope) it deadlocks
        # for minutes inside _format_value, freezing the runner mid-crash.
        # While the runner hangs in the formatter, the peer rank busy-polls
        # the RDMA collective forever — wedging the entire cluster on what
        # should have been a clean runner-restart cycle.
        logger.warning(
            f"Runner {bound_instance.bound_runner_id} crashed with critical exception {e}\n"
            f"{traceback.format_exc()}"
        )
        # Notify the supervisor with our RunnerStatusUpdated event AND the
        # upstream-style RunnerTerminationError. Supervisor handles both
        # (see exo.worker.runner.supervisor:_send_post_init_failure).
        event_sender.send(
            RunnerStatusUpdated(
                runner_id=bound_instance.bound_runner_id,
                runner_status=RunnerFailed(error_message=str(e)),
            )
        )
        event_sender.send(RunnerTerminationError.from_exception(e))
    finally:
        try:
            event_sender.close()
            task_receiver.close()
        finally:
            event_sender.join()
            task_receiver.join()
            logger.info("bye from the runner")
