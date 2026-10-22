"""Per-round timing of the TP ``all_sum`` waits (``EXO_DSV41_ALLSUM_WAIT=1``).

WHY. The first decode after a large prefill runs ~4x slower than a warm resend
of the same context, and a controlled NOP bisect over every candidate component
(all_sum included) shows no single one accounts for it. ``all_sum`` is a
BARRIER: a stall on either rank appears on both ranks' logs within a few ms, so
per-round wall time cannot localise it. Timing the host-side completion of each
collective can: the rank that arrives at the barrier early WAITS (large ms), the
rank that arrives late waits ~0. Per round, per rank.

MEASUREMENT WINDOW. Only DECODE rounds are instrumented: ``_one_round`` opens the
window (``begin_round``) and ``emit_round`` closes it, so the prefill -- where
collectives are far more frequent -- runs exactly as it does uninstrumented and
its 20-30 minute duration is not inflated. Inside a round every collective is
host-synced and timed.

MECHANICS. The model reaches the collective through the module attribute --
``mlx_lm.models.deepseek_v41.collective.all_sum`` (``from . import collective
as _coll`` then ``_coll.all_sum(...)``) -- so patching that attribute is enough.
By default the result is left unevaluated: JACCL runs the collective on a CPU
stream and any GPU work consuming it waits on an event inside a command buffer.
With this instrument ON and a round open the wrapper ``mx.eval``s the result on
the host and times THAT wait, then accumulates (count, total ms, max ms) until
the round ends and ``_one_round`` emits one line:

    [ALLSUMWAIT] n=<round> rank=<r> calls=<c> sum_ms=<..> max_ms=<..> ms=<round ms> committed=<n>

Forcing the sync costs throughput (same class of cost as ``EXO_PROFILER=spans``,
which ``mx.eval``s every span), so round times under this instrument are NOT
comparable to un-instrumented numbers. What is read from it is the SPLIT between
the two ranks and the correlation of the wait with slow rounds.

DEFAULT OFF. With the env unset nothing is installed, the module attribute keeps
its original function, and the engine's behaviour is byte-identical.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable

import mlx.core as mx

#: Env gate. Unset/empty/"0" => nothing is installed (default OFF).
ENV = "EXO_DSV41_ALLSUM_WAIT"

#: The model-side collective signature: ``all_sum(x, group=None) -> mx.array``.
_Collective = Callable[..., mx.array]

_calls = 0
_sum_ms = 0.0
_max_ms = 0.0
_round_seq = 0
_installed = False
_active = False
_target: Any = None
_original: _Collective | None = None


def enabled() -> bool:
    """True when the instrument is switched on by the environment."""
    return os.environ.get(ENV, "") not in ("", "0")


def install(module: Any | None = None) -> bool:
    """Patch the model's ``all_sum`` to time its host-side wait.

    Returns True when the patch is in place (freshly installed or already
    installed), False when the env gate is off. ``module`` exists so a test can
    supply a stand-in for ``mlx_lm.models.deepseek_v41.collective``; production
    callers pass nothing. Idempotent: a second call is a no-op.
    """
    global _installed, _original, _target
    if _installed:
        return True
    if not enabled():
        return False
    resolved: Any = module
    if resolved is None:
        from mlx_lm.models.deepseek_v41 import collective as _coll

        resolved = _coll
    original = resolved.all_sum
    resolved.all_sum = _timed(original)
    _original = original
    _target = resolved
    _installed = True
    return True


def _timed(original: _Collective) -> _Collective:
    """Wrap one collective so its completion is timed on the host.

    Outside an open round this is a pass-through: the prefill's collectives stay
    asynchronous exactly as they are uninstrumented.
    """

    def all_sum(x: mx.array, group: Any = None) -> mx.array:
        started = time.perf_counter()
        y = original(x, group=group)
        if not _active:
            return y
        mx.eval(y)
        _add((time.perf_counter() - started) * 1e3)
        return y

    return all_sum


def _add(ms: float) -> None:
    global _calls, _sum_ms, _max_ms
    _calls += 1
    _sum_ms += ms
    if ms > _max_ms:
        _max_ms = ms


def reset() -> None:
    """Clear the accumulator (start of a round)."""
    global _calls, _sum_ms, _max_ms
    _calls = 0
    _sum_ms = 0.0
    _max_ms = 0.0


def stats() -> tuple[int, float, float]:
    """``(calls, sum_ms, max_ms)`` since the last :func:`reset`."""
    return (_calls, _sum_ms, _max_ms)


def reset_rounds() -> None:
    """Reset the per-turn round counter and close any stale round window."""
    global _round_seq, _active
    _round_seq = 0
    _active = False
    reset()


def begin_round() -> None:
    """Open a decode round's measurement window (no-op unless installed)."""
    global _active
    reset()
    _active = _installed


def rank_of(engine: Any) -> int:
    """The engine's device rank, or -1 for a stand-in that has none (tests)."""
    rank = getattr(engine, "device_rank", None)
    return rank if isinstance(rank, int) else -1


def emit_round(rank: int, round_ms: float, committed: int) -> str | None:
    """Close one round: emit its wait line and start a fresh accumulator.

    Returns None when the instrument is not installed, so the caller's logging
    path stays off (byte-identical default).
    """
    global _round_seq, _active
    if not _installed:
        return None
    _active = False
    _round_seq += 1
    calls, sum_ms, max_ms = stats()
    reset()
    return (
        f"[ALLSUMWAIT] n={_round_seq} rank={rank} calls={calls} "
        f"sum_ms={sum_ms:.1f} max_ms={max_ms:.1f} "
        f"ms={round_ms:.1f} committed={committed}"
    )


def uninstall() -> None:
    """Restore the original collective (tests / teardown)."""
    global _installed, _original, _target
    if _installed and _target is not None and _original is not None:
        _target.all_sum = _original
    _installed = False
    _target = None
    _original = None
    reset()
    reset_rounds()
