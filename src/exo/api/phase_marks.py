"""Round-11 c=1 phase-boundary instrumentation for the API-process request
path (handler_entered -> stream_closed).

Same design rules as the sibling runner-side module
(``exo.worker.engines.mlx.phase_marks``): every recorded value is an
INTERNAL DELTA (milliseconds) between two consecutive marks in THIS
PROCESS, using ``time.perf_counter()`` -- comparable across processes on
ONE host, never across nodes, so a cross-node clock comparison is
impossible by construction.

Two-phase API because the FIRST two marks (a1 handler_entered, a2
messages_serialized) happen BEFORE a ``CommandId`` exists -- the command is
only constructed after the request is converted to
``TextGenerationTaskParams``. So:

1. ``new_recorder()`` starts a local, unkeyed recorder at the top of the
   chat-completions handler (a1).
2. ``register(command_id, recorder)`` adopts it into the module-level,
   ``CommandId``-keyed store once the command exists (a3, right after
   publish), so ``mark(command_id, ...)`` can be called from the separate
   SSE-generator coroutine for a4-a7.

The API process legitimately serves multiple concurrent commands, so the
module-level store is keyed by ``CommandId`` rather than being a single
singleton (unlike the runner side, which is c=1-scoped to one active
generation).

Env-gated by EXO_PHASE_MARKS, read ONCE at import into a module-level
``Final[bool]``. When unset, every public function is exactly
``if not _MARKS_ENABLED: return`` (or ``return None``) -- no further
os.environ read, no perf_counter() call, no dict mutation.
"""

import os
import time
from typing import Final

from exo.shared.types.common import CommandId

_MARKS_ENABLED: Final[bool] = os.environ.get("EXO_PHASE_MARKS", "") not in (
    "",
    "0",
    "false",
    "False",
)

# command_id -> (marks so far, timestamp of the previous mark, timestamp of
# handler_entered / new_recorder() -- kept INDEPENDENT of the delta chain so
# the analysis script's closure check has something non-tautological to
# compare the summed deltas against)
_active: dict[CommandId, tuple[dict[str, float], float, float]] = {}

# The dict key under which the independently-measured total span
# (handler_entered -> most recent mark) is attached at snapshot time. Never
# fed back into the delta chain itself.
TOTAL_SPAN_KEY = "_handler_to_last_mark_span_ms"


class ApiPhaseRecorder:
    """Local (unkeyed) mark sequence, used before a CommandId exists."""

    __slots__ = ("marks", "last_t", "start_t")

    def __init__(self) -> None:
        self.marks: dict[str, float] = {}
        now = time.perf_counter()
        self.last_t: float = now
        self.start_t: float = now

    def mark(self, name: str) -> None:
        if not _MARKS_ENABLED:
            return
        now = time.perf_counter()
        self.marks[name] = (now - self.last_t) * 1000.0
        self.last_t = now


def new_recorder() -> ApiPhaseRecorder | None:
    """Start a1 (handler_entered). Returns None when disabled -- callers
    must guard every subsequent ``.mark()`` call on this being non-None,
    which is itself the G3 no-op guard (no perf_counter() call, no
    attribute mutation, when disabled)."""
    if not _MARKS_ENABLED:
        return None
    return ApiPhaseRecorder()


def register(command_id: CommandId, recorder: ApiPhaseRecorder | None) -> None:
    """Adopt a local recorder into the CommandId-keyed store (a3, right
    after the command is published) so the separate SSE-generator
    coroutine can keep marking via ``mark(command_id, ...)``."""
    if not _MARKS_ENABLED or recorder is None:
        return
    _active[command_id] = (recorder.marks, recorder.last_t, recorder.start_t)


def mark(command_id: CommandId, name: str) -> None:
    """Record the delta (ms) since the previous mark/register() under
    ``name``. A no-op if disabled or if register() was never called for
    this command_id (e.g. the non-streaming /bench path)."""
    if not _MARKS_ENABLED:
        return
    entry = _active.get(command_id)
    if entry is None:
        return
    marks, last_t, start_t = entry
    now = time.perf_counter()
    marks[name] = (now - last_t) * 1000.0
    _active[command_id] = (marks, now, start_t)


def snapshot_and_clear(command_id: CommandId) -> dict[str, float] | None:
    """Return marks recorded since register() and drop this command's
    entry. Returns None when disabled or no active sequence for this
    command_id -- callers must treat "no marks" as a valid, expected
    value.

    The returned dict includes one extra key, ``TOTAL_SPAN_KEY``, computed
    as (timestamp of the most recent mark) - (timestamp of new_recorder()),
    i.e. an INDEPENDENT single-subtraction measurement of the total elapsed
    time -- not a resummation of the individual deltas. This is what makes
    the analysis script's closure check (sum of deltas vs this independent
    total) a real check rather than a tautology.
    """
    if not _MARKS_ENABLED:
        return None
    entry = _active.pop(command_id, None)
    if entry is None:
        return None
    marks, last_t, start_t = entry
    marks[TOTAL_SPAN_KEY] = (last_t - start_t) * 1000.0
    return marks
