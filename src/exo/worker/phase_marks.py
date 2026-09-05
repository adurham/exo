"""Round-13 Gate-A phase-boundary instrumentation for the WORKER process
`plan_step` loop: `state-update-applied` -> `plan_step-observed`.

Distinct from the sibling modules for the other two processes in this
campaign's instrumentation family:
- `exo.api.phase_marks` (API process, per-command keyed recorder, a1-a7)
- `exo.worker.engines.mlx.phase_marks` (RUNNER process, one-active-recorder,
  b1-b11)

This module is a plain STREAM of individually-identified mark events, not
an accumulate-then-snapshot recorder, because Gate A is not one request's
bounded lifecycle -- it is a continuous, indefinite series of
(state-apply, plan-observe) pairs over the life of the worker process, one
pair per dispatched plan-step iteration. Every mark line is self-contained
and carries the shared pairing key needed to join a `state_applied` mark to
its corresponding `plan_step_observed` mark after the fact, in the
analysis script -- no in-process buffering or pairing is attempted here.

Pairing key: `event_idx`, the master-assigned, globally monotonic
`IndexedEvent.idx` (see `exo.shared.types.events.IndexedEvent`) that is
already available at the exact point `Worker._event_applier` applies a
state mutation. This is an EXISTING identity, not an invented counter --
every worker event carries one, and it is unique and increasing across the
whole run, which is exactly what per-dispatch pairing needs.

Uses `time.perf_counter()`, exactly like the other two phase_marks
modules, for internal consistency -- though Gate A itself never needs to
compare this worker process's clock against another process or node; both
marks here are same-process, same-clock by construction (that is the
entire point of Gate A per the round-13 pre-registration).

Env-gated by EXO_PHASE_MARKS, read ONCE at import into a module-level
`Final[bool]`, mirroring the sibling modules byte-for-byte. When unset,
`mark_state_applied` / `mark_plan_step_observed` are exactly
``if not _MARKS_ENABLED: return`` -- no further os.environ read, no
perf_counter() call, no logger call, no `mx.eval` anywhere near this file.
"""

import os
import time
from typing import Final, Literal

from loguru import logger

from exo.shared.types.tasks import Task

_MARKS_ENABLED: Final[bool] = os.environ.get("EXO_PHASE_MARKS", "") not in (
    "",
    "0",
    "false",
    "False",
)

# Distinguishes how the `plan_step` loop-top wait returned. `cancelled_caught`
# alone is not a safe classifier: if the state-apply `set()` and the
# `move_on_after` deadline land in the same scheduling window, the task can
# still receive the cancellation (so `cancelled_caught` is True) even though
# the event WAS set and the wakeup WAS, in effect, delivered. Recording
# `waiting_on.is_set()` at the moment the wait returns, alongside
# `cancelled_caught`, disambiguates the three reachable states:
#
#   is_set()=True,  cancelled_caught=False -> "event": clean event-driven wake,
#       `wait()` returned normally before the fallback ever fired.
#   is_set()=True,  cancelled_caught=True  -> "event_raced_timeout": the event
#       WAS set, but the fallback cancellation was delivered anyway in the same
#       window (a photo finish). The wakeup was event-caused, not a polling
#       artifact -- Gate A's zero-timeout-wake count must NOT charge this
#       against the fix.
#   is_set()=False, cancelled_caught=True  -> "timeout": a true timeout: no
#       state-apply signal had landed by the time the fallback fired.
#
# (is_set()=False, cancelled_caught=False is unreachable: `wait()` only
# returns normally, uncancelled, once the event it is parked on is set.)
#
# Gate A's PASS condition requires counting timeout-driven wakes on the
# request path explicitly -- only "timeout" above is a true timeout-driven
# dispatch; "event_raced_timeout" is a genuine event-driven wake that merely
# lost a race with the fallback's cancellation delivery.
WakeKind = Literal["event", "event_raced_timeout", "timeout"]

# Stable, greppable prefix for the raw mark stream. One line per mark, no
# other machine-parseable structure is assumed of the log sink.
_MARK_PREFIX: Final[str] = "PHASE_MARK"


def mark_state_applied(event_idx: int) -> None:
    """Gate A mark 1: the state mutation from `event_idx` just landed in
    `Worker.state`.

    Call AFTER `self.state = apply(...)` for this event -- this is the
    literal "moment the state update was APPLIED" the round-13
    pre-registration defines Gate A's left edge as.
    """
    if not _MARKS_ENABLED:
        return
    logger.info(
        f"{_MARK_PREFIX} state_applied event_idx={event_idx} "
        f"t={time.perf_counter():.6f}"
    )


def mark_plan_step_observed(
    event_idx: int, wake_kind: WakeKind, task: Task
) -> None:
    """Gate A mark 2: `plan_step` woke, ran `plan()`, and produced a
    non-None task -- the literal "moment plan_step OBSERVED it" the
    round-13 pre-registration defines Gate A's right edge as.

    `event_idx` is the pairing key: the `IndexedEvent.idx` most recently
    recorded by `mark_state_applied` as of the instant this wake occurred
    (captured by the caller immediately after the loop-top wait returns,
    before `plan()` runs, to minimize the race against a concurrent new
    state apply).

    `task` is the dispatched task itself (already non-None at the call
    site, after the `task is None` guard); its concrete class name is
    recorded VERBATIM as the `task=` field. This function is DUMB by
    design: it does not classify the task as backoff-gated vs
    request-path -- that policy lives in the round-13 analyzer, where it
    can be audited against the PREDICTION.md pre-registration (amendment
    A2), not hardcoded into worker instrumentation. Accepting the task
    object (not a pre-formatted string) keeps the attribute access behind
    the `_MARKS_ENABLED` gate below, so it costs nothing when marks are
    disabled.
    """
    if not _MARKS_ENABLED:
        return
    logger.info(
        f"{_MARK_PREFIX} plan_step_observed event_idx={event_idx} "
        f"t={time.perf_counter():.6f} wake_kind={wake_kind} "
        f"task={task.__class__.__name__}"
    )
