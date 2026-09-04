"""Round-11 c=1 phase-boundary instrumentation for the RUNNER-process request
path (task_received -> stop_detected).

Mirrors the existing ``request_trace`` singleton in
``exo.worker.engines.mlx.trace``: ONE active recorder per runner process.
That is the established precedent in this exact file/module family for
lightweight request-lifecycle tracing, and it matches this instrumentation's
scope -- a c=1 (strictly-one-request-at-a-time) study, where a single active
recorder cannot be corrupted by concurrent submissions.

``perf_counter()`` is mach_absolute_time-based: comparable across processes
on ONE host, NEVER across nodes. Every value recorded here is an INTERNAL
DELTA (milliseconds) between two consecutive marks in THIS PROCESS -- never
an absolute counter value -- so a cross-node clock comparison is impossible
by construction (not just discouraged by a comment).

The one cross-boundary quantity the round-11 study wants (dispatch + IPC
gap) is DERIVED in the analysis script, never computed here, by subtracting
two same-clock intervals:

    dispatch_and_ipc_gap = (
        (API: first_chunk_received_ms cumulative from handler_entered)
        - (RUNNER: cumulative ms from task_received to first_token_emitted)
    )

Both terms are same-process cumulative sums of same-process deltas; neither
side ever subtracts a RUNNER perf_counter reading from an API perf_counter
reading.

Env-gated by EXO_PHASE_MARKS, read ONCE at import into a module-level
``Final[bool]``. When unset, every public function below is exactly
``if not _MARKS_ENABLED: return`` and does nothing else -- no further
os.environ read, no perf_counter() call, no dict write, no allocation.
"""

import os
import time
from typing import Final

_MARKS_ENABLED: Final[bool] = os.environ.get("EXO_PHASE_MARKS", "") not in (
    "",
    "0",
    "false",
    "False",
)


class _RunnerPhaseMarks:
    """Single active mark sequence, one per runner process.

    Not safe for genuinely concurrent in-flight generations (see module
    docstring for why that is an accepted, documented scope limit for this
    c=1 round rather than an oversight).
    """

    __slots__ = ("_marks", "_last_t", "_active")

    def __init__(self) -> None:
        self._marks: dict[str, float] = {}
        self._last_t: float = 0.0
        self._active: bool = False

    def begin(self) -> None:
        """Start a new mark sequence. Call at task_received (b1)."""
        if not _MARKS_ENABLED:
            return
        self._marks = {}
        self._last_t = time.perf_counter()
        self._active = True

    def mark(self, name: str) -> None:
        """Record the delta (ms) since the previous mark/begin() under ``name``.

        A no-op if disabled or if begin() was never called (e.g. bench mode,
        or a request that started before this module was configured).
        """
        if not _MARKS_ENABLED or not self._active:
            return
        now = time.perf_counter()
        self._marks[name] = (now - self._last_t) * 1000.0
        self._last_t = now

    def snapshot_and_clear(self) -> dict[str, float] | None:
        """Return marks recorded since begin() and end the sequence.

        Returns None when disabled or no active sequence -- callers must
        treat "no marks" as a valid, expected value (e.g. bench requests
        never call begin()).
        """
        if not _MARKS_ENABLED or not self._active:
            return None
        result = self._marks
        self._marks = {}
        self._active = False
        return result


runner_phase_marks: Final = _RunnerPhaseMarks()
