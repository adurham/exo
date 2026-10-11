import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Iterable
from typing import BinaryIO

from exo.shared.types.chunks import Chunk
from exo.shared.types.tasks import CANCEL_ALL_TASKS, GenerationTask, TaskId
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runner_response import (
    CancelledResponse,
    FinishedResponse,
    ModelLoadingResponse,
)
from exo.worker.disaggregated.server import PrefillRequest


class Engine(ABC):
    _cancelled_tasks: set[TaskId]

    # Liveness hook installed by the Runner after build(). Non-zero ranks emit
    # no client-visible events (the c>=2 chunk dedup guard), so during a long
    # prefill they would otherwise stay silent past the supervisor's hang
    # watchdog and get SIGKILLed while healthy — same starvation class as the
    # decode-side rank heartbeat in Runner.handle_generation_tasks.
    heartbeat: Callable[[], None] | None = None
    _last_prefill_heartbeat_monotonic: float = 0.0

    def prefill_heartbeat(self) -> None:
        if self.heartbeat is None:
            return
        now = time.monotonic()
        if now - self._last_prefill_heartbeat_monotonic >= 15.0:
            self._last_prefill_heartbeat_monotonic = now
            self.heartbeat()

    def should_cancel(self, task_id: TaskId) -> bool:
        return (
            task_id in self._cancelled_tasks
            or CANCEL_ALL_TASKS in self._cancelled_tasks
        )

    def idle_tick_seconds(self) -> float | None:
        """Interval at which the runner should call :meth:`idle_tick` while it
        has no generation task, or ``None`` (the default) to keep the runner's
        blocking idle wait exactly as before."""
        return None

    def idle_tick(self, request_pending: Callable[[], bool] | None = None) -> object:
        """Opportunistic between-requests work on the RUNNER (engine) thread.

        Called only when the runner is Ready and its work queue stayed empty for
        :meth:`idle_tick_seconds`. ``request_pending()`` turns True as soon as a
        new work item arrives; an implementation must poll it and yield quickly.
        Default: no-op.
        """
        del request_pending
        return None

    @abstractmethod
    def warmup(self) -> None: ...

    @abstractmethod
    def submit(
        self,
        task: GenerationTask,
    ) -> None: ...

    @abstractmethod
    def step(
        self,
    ) -> Iterable[tuple[TaskId, Chunk | CancelledResponse | FinishedResponse]]: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def serve_prefill(self, request: PrefillRequest, wfile: BinaryIO) -> None: ...


class Builder(ABC):
    @abstractmethod
    def connect(self, bound_instance: BoundInstance) -> None: ...

    @abstractmethod
    def load(
        self,
        bound_instance: BoundInstance,
    ) -> Generator[ModelLoadingResponse]: ...

    @abstractmethod
    def build(self) -> Engine: ...

    @abstractmethod
    def close(self) -> None: ...
