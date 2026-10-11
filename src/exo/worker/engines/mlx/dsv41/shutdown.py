"""Plan §G3 — park-on-shutdown (the graceful-restart fast path).

WHAT THIS IS. When the deploy script stops a runner it sends SIGTERM, waits
``GRACE`` seconds (``scripts/exo_graceful_shutdown.sh``, default 15 s, then
SIGKILL: ``kill -TERM`` -> wait -> ``kill -9``). Today a runner on SIGTERM just
raises ``SystemExit(0)`` (``bootstrap.entrypoint``) and the in-memory KV of
every resident conversation is lost — the next start cold-prefills. §G3 turns
that window into a best-effort checkpoint: stop admitting, drain the in-flight
turn briefly, abort it if it does not finish, force-checkpoint the remaining
dirty *between-turns* conversations under a wall-clock deadline, then let the
existing clean exit (RDMA teardown) run.

THE BUDGET MATH (audit A8). The launcher's real budget is the shutdown script's
``GRACE`` = ``EXO_SHUTDOWN_GRACE_SECONDS`` (default 15.0 s). There is **no
launchd / no ExitTimeOut** for exo (it runs under ``screen -dmS``), and the
kill -> rsync ordering is already correct (R3 is struck). So we require, and
assert in the log, ``DRAIN_S + PARK_S + EXIT_MARGIN_S <= GRACE`` and the park
phase is capped in wall-clock terms at ``signal_time + GRACE - EXIT_MARGIN_S``
independently of how long the drain actually took — so the process always
exits (and tears down RDMA) before the script escalates to SIGKILL, even if the
drain finished early. Defaults: DRAIN 10.0 + PARK 4.0 + MARGIN 1.0 = 15.0.

THE SEQUENCE (plan §G3 steps 1-5), all on the runner/engine MAIN thread — the
one thread that owns every MLX call, so there is never concurrent MLX:

1. **stop admitting** — the runner stops pulling new work; the engine sets
   ``_shutting_down`` so ``step()`` will not start a queued turn.
2. **drain** — the runner keeps stepping the in-flight turn normally until it
   completes (then it commits exactly as it would have) or ``DRAIN_S`` expires.
3. **abort** — if it did not finish, the in-flight conversation is DROPPED from
   residency (``Dsv41Sessions.abort_inflight``); nothing is parked mid-turn, so
   the last §G1 generation stays the on-disk restore point and only the aborted
   turn's delta is lost.
4. **force checkpoint** — remaining dirty *between-turns* conversations,
   most-recent-first, through the EXISTING §G1/§G4 coordinated path
   (``Dsv41Sessions.maybe_checkpoint`` -> ``park.checkpoint_conversation`` ->
   ``coord.run_checkpoint``), under ``PARK_S`` / the wall-clock cap.
5. **exit** — ``main()`` returns and the pre-existing ``entrypoint`` finally
   releases the MLX pool and exits the interpreter normally (RDMA QP teardown).

FAILURE MODES (COLD-safe by construction). A SIGKILL before completion leaves a
``*.tmp`` generation dir and the previous generation intact, swept at the next
startup (§G5). A half-written turn is impossible: step 3 never parks in-flight
state, and the checkpoint writer refuses (``_check_unchanged``) if a
conversation moved. Repeat/duplicate signals are idempotent — ``ShutdownSignal``
records the first and ignores the rest.

MULTI-RANK (TP2). Both ranks receive SIGTERM from the same deploy script. The
force checkpoint reuses the G4 ALL-OR-NOTHING coordinated path verbatim, so it
is rank-invariant and degrades to COLD on ANY disagreement; a peer that is
absent/desynced makes the exchange a no-go, and a coord deadline past GRACE is
absorbed by SIGKILL (temp swept, previous generation kept). See OPEN-QUESTION 1.

DEFAULT OFF. Everything here is gated on ``EXO_DSV41_SHUTDOWN_PARK=1``; with it
unset ``shutdown_park_enabled()`` is False, ``entrypoint`` keeps its exact
pre-G3 SIGTERM handler, the runner keeps its plain blocking idle wait, and no
code in this module runs.
"""

from __future__ import annotations

import os
import signal
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from exo.worker.runner.bootstrap import logger

#: Master gate. Unset/falsey -> the whole G3 fast path is inert.
SHUTDOWN_PARK_ENV = "EXO_DSV41_SHUTDOWN_PARK"
#: Seconds to let the in-flight turn finish before aborting it (plan step 2).
SHUTDOWN_DRAIN_S_ENV = "EXO_DSV41_SHUTDOWN_DRAIN_S"
#: Nominal seconds for the force-checkpoint loop (plan step 4).
SHUTDOWN_PARK_S_ENV = "EXO_DSV41_SHUTDOWN_PARK_S"
#: Safety margin held back from GRACE so the process exits before SIGKILL.
SHUTDOWN_EXIT_MARGIN_S_ENV = "EXO_DSV41_SHUTDOWN_EXIT_MARGIN_S"
#: The launcher's graceful-shutdown budget. MUST match the deploy script's
#: ``GRACE`` (``scripts/exo_graceful_shutdown.sh``, default 15 s). Audit A8:
#: this — not a launchd ExitTimeOut (there is none) — is the real budget.
GRACE_ENV = "EXO_SHUTDOWN_GRACE_SECONDS"

DEFAULT_DRAIN_S = 10.0
DEFAULT_PARK_S = 4.0
DEFAULT_EXIT_MARGIN_S = 1.0
DEFAULT_GRACE_S = 15.0

#: Idle-wait poll cadence while a shutdown is armed. Small enough that the
#: flag is observed well within GRACE even when §G1 idle ticks are off; only
#: used when the fast path is enabled, so the default wait is untouched.
SHUTDOWN_POLL_S = 0.25


def shutdown_park_enabled() -> bool:
    """Whether §G3 park-on-shutdown is armed (``EXO_DSV41_SHUTDOWN_PARK``)."""
    raw = os.environ.get(SHUTDOWN_PARK_ENV)
    if raw is None:
        return False
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return float(default)
    try:
        return float(raw)
    except ValueError:
        logger.warning(f"[DSV41] ignoring non-float {name}={raw!r}; using {default}")
        return float(default)


@dataclass(frozen=True)
class ShutdownBudget:
    """The §G3 wall-clock budget, derived from env and checked against GRACE."""

    drain_s: float
    park_s: float
    exit_margin_s: float
    grace_s: float

    @property
    def fits(self) -> bool:
        """``DRAIN + PARK + MARGIN <= GRACE`` — the plan's budget invariant."""
        return (
            self.drain_s + self.park_s + self.exit_margin_s
            <= self.grace_s + 1e-9
        )

    def park_deadline(self, *, signal_time: float, park_start: float) -> float:
        """Wall-clock instant the force-checkpoint loop must stop by.

        ``min(nominal phase budget, hard GRACE cap)``. The hard cap
        (``signal_time + GRACE - MARGIN``) is what actually protects the
        SIGKILL boundary when the drain overran its own budget.
        """
        return min(
            park_start + self.park_s,
            signal_time + self.grace_s - self.exit_margin_s,
        )


def shutdown_budget() -> ShutdownBudget:
    """Read the §G3 budget from env (all three knobs overridable)."""
    b = ShutdownBudget(
        drain_s=max(0.0, _float_env(SHUTDOWN_DRAIN_S_ENV, DEFAULT_DRAIN_S)),
        park_s=max(0.0, _float_env(SHUTDOWN_PARK_S_ENV, DEFAULT_PARK_S)),
        exit_margin_s=max(
            0.0, _float_env(SHUTDOWN_EXIT_MARGIN_S_ENV, DEFAULT_EXIT_MARGIN_S)
        ),
        grace_s=max(0.0, _float_env(GRACE_ENV, DEFAULT_GRACE_S)),
    )
    if not b.fits:
        # Loud but non-fatal: the hard cap in park_deadline() still keeps the
        # process inside GRACE, but the nominal phases no longer add up.
        logger.warning(
            f"[DSV41] §G3 shutdown budget does not fit GRACE: drain={b.drain_s} "
            f"+ park={b.park_s} + margin={b.exit_margin_s} > grace={b.grace_s}. "
            "The park phase is hard-capped at signal+GRACE-margin regardless."
        )
    return b


class ShutdownSignal:
    """Idempotent SIGTERM/SIGINT bookkeeping for the §G3 fast path.

    The handler runs on the runner process's MAIN thread — the same thread that
    runs the synchronous ``runner.main()`` loop and owns every MLX call — so no
    cross-thread marshalling is needed: it records the request and the runner
    loop observes the flag (its idle wait is bounded while the fast path is
    armed). This is the "minimal safe equivalent" of ``loop.add_signal_handler``
    the plan names; there is no asyncio loop in this process.

    Idempotency: the FIRST signal stamps ``requested``/``signal_time``; every
    later signal is a no-op (the sequence runs exactly once).

    Clean-exit guarantee: with the fast path off, OR with the request arriving
    before ``runner.main()`` has started polling (during builder setup/load),
    the handler raises ``SystemExit(0)`` — the exact pre-G3 behaviour, so the
    ``entrypoint`` finally still releases MLX and the interpreter exits normally
    (JACCL/RDMA QP static destructors run). Never a bare ``os._exit``.
    """

    def __init__(self, *, park: bool) -> None:
        self.park = bool(park)
        self.requested = False
        self.signal_time: float | None = None
        #: Total signals seen (for logging / tests); only the first acts.
        self.count = 0
        #: Flipped True by ``Runner.main`` once the polling loop is live. Before
        #: that, no loop will observe the flag, so a signal must unwind directly.
        self.runner_active = False
        #: Idle-wait poll cadence the runner uses while this signal is armed.
        self.poll_s = SHUTDOWN_POLL_S

    def on_signal(self, signum: int | signal.Signals, frame: Any) -> None:
        """The installed handler. Idempotent; never does MLX work."""
        name = getattr(signum, "name", None) or f"signal {int(signum)}"
        if not self.park:
            # Pre-G3 disposition, byte-for-byte: every SIGTERM unwinds cleanly.
            logger.info(
                f"Runner received {name} — exiting cleanly to tear down RDMA."
            )
            raise SystemExit(0)
        self.count += 1
        if self.requested:
            # Duplicate signal while a shutdown is already in flight: no-op.
            logger.info(
                f"[DSV41] §G3 duplicate shutdown signal #{self.count}; ignoring "
                "(sequence already armed)."
            )
            return
        self.requested = True
        self.signal_time = time.monotonic()
        if not self.runner_active:
            logger.info(
                f"Runner received {name} — exiting cleanly to tear down RDMA."
            )
            raise SystemExit(0)
        logger.info(
            f"[DSV41] §G3 {name} received — arming park-on-shutdown "
            "(drain then force-checkpoint)."
        )

    def install(self) -> None:
        """Install SIGTERM (and SIGINT when park) handlers; never fatal."""
        sigs: list[signal.Signals] = [signal.SIGTERM]
        if self.park:
            sigs.append(signal.SIGINT)
        for sig in sigs:
            try:
                signal.signal(sig, self.on_signal)
            except (ValueError, OSError):
                # Not the main thread, or signals unavailable — non-fatal;
                # the default disposition stays.
                logger.warning(
                    f"Could not install §G3 shutdown handler for {sig!r} in "
                    "runner entrypoint."
                )


@dataclass
class ShutdownOutcome:
    """What the engine-thread shutdown sequence did."""

    requested: bool = False
    aborted: list[Any] = field(default_factory=list)
    parked: list[Any] = field(default_factory=list)
    park_deadline: float = 0.0
    deadline_hit: bool = False

    @property
    def parked_count(self) -> int:
        return len(self.parked)


def run_shutdown_park(
    sessions: Any,
    *,
    signal_time: float,
    budget: ShutdownBudget | None = None,
    now: Callable[[], float] = time.monotonic,
    precheck: Callable[[], bool] | None = None,
    request_pending: Callable[[], bool] | None = None,
) -> ShutdownOutcome:
    """§G3 steps (3)-(4) on the engine thread: abort, then force-checkpoint.

    The caller (``Runner``) has already done steps (1) admission-stop and
    (2) drain, so by the time this runs any conversation still ``_inflight`` is
    the one the drain could not finish. Nothing here parks in-flight state, and
    every checkpoint goes through §G1's keep-resident, cancellable, G4-
    coordinated path — no new collective protocol.

    ``precheck`` (the engine supplies one) mirrors the §G1 idle tick: it pairs an
    ``agree_on_tasks`` with a peer that may be inside ``step()`` and returns
    False to stand down (COLD-safe) rather than mispair a checkpoint collective.
    """
    b = budget if budget is not None else shutdown_budget()
    park_start = now()
    deadline = b.park_deadline(signal_time=signal_time, park_start=park_start)
    if now() >= deadline:
        # No time left for a coordinated checkpoint; skip straight to exit.
        aborted = sessions.abort_inflight()
        return ShutdownOutcome(
            requested=True, aborted=aborted, park_deadline=deadline,
            deadline_hit=True,
        )
    aborted = sessions.abort_inflight()
    parked, deadline_hit = sessions.force_checkpoint_all(
        deadline=deadline, now=now, precheck=precheck,
        request_pending=request_pending,
    )
    return ShutdownOutcome(
        requested=True,
        aborted=aborted,
        parked=parked,
        park_deadline=deadline,
        deadline_hit=deadline_hit,
    )
