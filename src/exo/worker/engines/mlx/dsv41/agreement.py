"""Rank-agreement helpers shared by the DSv4.1 engine.

Same shape as ``SequentialGenerator.agree_on_tasks``/``agree_on_cancellations``
(reuse, not reimplementation): every rank must decide to start/stop the SAME
request, because a rank that starts a forward its peer skipped will block
forever in the model's ``all_sum`` inside the MoE layers.

Two differences from the generic path, both deliberate:

* ``mx_any``/``mx_all_gather_tasks`` run over the group the caller passes, and
  the DSv4.1 engine passes the COORD subgroup (``get_coord_group``) so control
  traffic never shares the model group's call-id counter -- exactly the hygiene
  ``utils_mlx`` documents for every non-model collective. This mirrors
  ``SequentialGenerator``; ``mx.distributed.all_gather(group=None)`` would use
  the *default* group, which is precisely the mistake that comment warns about.
* nothing here touches model state, so ``reset()`` after a reconnect only drops
  bookkeeping and the engine can rebuild that bookkeeping without rebuilding a
  cache.
"""

from __future__ import annotations

from collections import deque

import mlx.core as mx

from exo.shared.types.tasks import CANCEL_ALL_TASKS, TaskId, TextGeneration
from exo.worker.engines.mlx.utils_mlx import mx_all_gather_tasks, mx_any
from exo.worker.runner.bootstrap import logger


class RankAgreement:
    """One rank's pending/queued/cancelled task bookkeeping."""

    def __init__(self, group: mx.distributed.Group | None) -> None:
        self.group = group
        self.maybe_queue: list[TextGeneration] = []
        self.maybe_cancel: list[TextGeneration] = []
        self.queue: deque[TextGeneration] = deque()
        self.all_tasks: dict[TaskId, TextGeneration] = {}
        self.cancelled: set[TaskId] = set()

    def submit(self, task: TextGeneration) -> None:
        self.cancelled.discard(CANCEL_ALL_TASKS)
        self.all_tasks[task.task_id] = task
        self.maybe_queue.append(task)

    def agree_on_tasks(self) -> None:
        if not mx_any(len(self.maybe_queue) > 0, self.group):
            return
        agreed, different = mx_all_gather_tasks(self.maybe_queue, self.group)
        # Extend from ``agreed`` (sorted by task_id on all ranks): preserves the
        # cross-rank ordering invariant, exactly as SequentialGenerator does.
        self.queue.extend(agreed)
        self.maybe_queue = list(different)

    def agree_on_cancellations(self, cancel_ids: list[TaskId]) -> None:
        self._collect(cancel_ids)
        if mx_any(CANCEL_ALL_TASKS in self.cancelled, self.group):
            self.cancelled.add(CANCEL_ALL_TASKS)
        agreed, different = mx_all_gather_tasks(self.maybe_cancel, self.group)
        self.cancelled.update(task.task_id for task in agreed)
        self.maybe_cancel = list(different)

    def agree_on_cancellations_fast(self, cancel_ids: list[TaskId]) -> None:
        """Cancellation check cheap enough to run between prefill chunks."""
        has_cancel_all = self._collect(cancel_ids)
        has_anything = has_cancel_all or len(self.maybe_cancel) > 0
        if not mx_any(has_anything, self.group):
            return  # fast path: no rank has cancellations
        if mx_any(CANCEL_ALL_TASKS in self.cancelled, self.group):
            self.cancelled.add(CANCEL_ALL_TASKS)
        agreed, different = mx_all_gather_tasks(self.maybe_cancel, self.group)
        self.cancelled.update(task.task_id for task in agreed)
        self.maybe_cancel = list(different)

    def _collect(self, cancel_ids: list[TaskId]) -> bool:
        """Fold newly received cancel ids into maybe_cancel; True if cancel-all."""
        has_cancel_all = CANCEL_ALL_TASKS in self.cancelled
        for task_id in cancel_ids:
            if task_id == CANCEL_ALL_TASKS:
                self.cancelled.add(CANCEL_ALL_TASKS)
                has_cancel_all = True
                continue
            task = self.all_tasks.get(task_id)
            if task is not None:
                self.maybe_cancel.append(task)
        return has_cancel_all

    def should_cancel(self, task_id: TaskId) -> bool:
        return task_id in self.cancelled or CANCEL_ALL_TASKS in self.cancelled

    def forget(self, task_id: TaskId) -> None:
        self.all_tasks.pop(task_id, None)
        self.cancelled.discard(task_id)
        self.queue = deque(t for t in self.queue if t.task_id != task_id)

    def reset(self) -> None:
        """Drop all in-flight bookkeeping (the post-reconnect recovery path)."""
        dropped = len(self.queue)
        self.queue.clear()
        self.maybe_queue.clear()
        self.maybe_cancel.clear()
        self.all_tasks.clear()
        if dropped:
            logger.warning(
                f"[DSV41] reset after reconnect: dropped {dropped} queued "
                "request(s); clients retry."
            )

    def __repr__(self) -> str:
        return (
            f"RankAgreement(queue={len(self.queue)}, "
            f"maybe={len(self.maybe_queue)}, cancelled={len(self.cancelled)})"
        )
