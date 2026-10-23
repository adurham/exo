"""Bound decode pipeline depth: host-sync every Kth TP collective.

WHY. ``docs/dsv41-1017-vs-sept-ab-2026-10-20.md`` section 8r.9 tried to test the
"cheap cousin" of the 40% recovery that host-syncing EVERY collective
(``EXO_DSV41_ALLSUM_WAIT=1``) produces on the slow 350K turn: fence every layer
instead of every 4. It toggled ``EXO_DSV4_FENCE_EVERY_N_LAYERS`` -- which is read
only by ``mlx_lm.models.deepseek_v4``, while this model is
``mlx_lm.models.deepseek_v41``, and exo's engine never reads it either (verified:
0 hits in both). That run therefore compared two no-op configurations, so its
negative is void ("no improvement AND no cost at 1K" is what a no-op looks like)
and the question is still open.

This module is the real, neutral form of that lever: no timing, no instrumentation,
just one ``mx.eval`` on every Kth collective INSIDE an open decode round. The
prefill stays asynchronous (the window is closed there), so its 20-30 minutes are
untouched -- unlike ``DSV41_SYNC_COLLECTIVES``, which syncs every forward.

WHY A CADENCE AND NOT "EVERY LAYER". Syncing every collective costs ~37% at 1K
(8r). If the slow state's cost is paid per cross-stream wait, the gain falls off
~1/K with the cost. If it is a pipeline-depth excursion that is paid once per
excursion, the gain saturates as soon as the queue is bounded at all and K>1 is
almost free. That difference is exactly what decides whether this is shippable
above a depth threshold or not shippable at all.

RUNTIME SWEEP. Each cold-350K data point costs a ~28-minute prefill, so K is also
read from a JSON levers file (``EXO_DSV41_LEVERS_FILE``, key ``sync_every_n``) at
every round start: one boot can sweep K without a relaunch. The file wins over
the env when it carries the key; ``0`` disables the lever mid-boot (the wrapper
stays installed and passes through).

DEFAULT OFF. Unset/0 => nothing is installed and the model keeps its own
collective, byte-identical to a build without this module.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx

#: Env gate: K collectives per host sync (unset/0 => off).
ENV = "DSV41_SYNC_EVERY_N"

#: Optional JSON file read at every round start, e.g. {"sync_every_n": 4}.
ENV_FILE = "EXO_DSV41_LEVERS_FILE"

#: Key this lever reads from the levers file.
KEY = "sync_every_n"

_installed = False
_target: Any = None
_original: Callable[..., mx.array] | None = None
_active = False
_k = 0
_calls = 0
_synced = 0
_round_seq = 0


def _env_k() -> int:
    """``DSV41_SYNC_EVERY_N`` -> cadence; unset/junk/<=0 mean off."""
    raw = os.environ.get(ENV, "").strip()
    if not raw:
        return 0
    try:
        k = int(raw)
    except ValueError:
        return 0
    return k if k > 0 else 0


def _file_k() -> int | None:
    """The levers file's cadence, or None when absent/unreadable/not an int."""
    path = os.environ.get(ENV_FILE, "").strip()
    if not path:
        return None
    try:
        data = json.loads(Path(path).read_text())
    except Exception:
        return None
    if not isinstance(data, dict) or KEY not in data:
        return None
    try:
        k = int(data[KEY])
    except (TypeError, ValueError):
        return None
    return k if k >= 0 else None


def current_k() -> int:
    """The cadence in force (0 = off). The levers file wins over the env."""
    from_file = _file_k()
    return _env_k() if from_file is None else from_file


def enabled() -> bool:
    """True when the lever is switched on by the env or the levers file."""
    return current_k() > 0


def install(module: Any | None = None) -> bool:
    """Patch the model's ``all_sum`` with the cadence wrapper.

    Returns True when the wrapper is in place, False when the lever is off.
    Idempotent. ``module`` exists so a test can supply a stand-in for
    ``mlx_lm.models.deepseek_v41.collective``; production callers pass nothing.
    """
    global _installed, _target, _original, _k
    if _installed:
        return True
    k = current_k()
    if k <= 0:
        return False
    resolved: Any = module
    if resolved is None:
        from mlx_lm.models.deepseek_v41 import collective as _coll

        resolved = _coll
    original = resolved.all_sum
    resolved.all_sum = _wrapped(original)
    _original = original
    _target = resolved
    _k = k
    _installed = True
    return True


def _wrapped(original: Callable[..., mx.array]) -> Callable[..., mx.array]:
    """Wrap one collective so every Kth call is host-synced inside a round."""

    def all_sum(x: mx.array, group: Any = None) -> mx.array:
        global _calls, _synced
        y = original(x, group=group)
        if not _active or _k <= 0:
            return y
        _calls += 1
        if _calls % _k:
            return y
        mx.eval(y)
        _synced += 1
        return y

    return all_sum


def reset_rounds() -> None:
    """Reset the round counter and close any stale round window."""
    global _round_seq, _active, _calls, _synced
    _round_seq = 0
    _active = False
    _calls = 0
    _synced = 0


def begin_round() -> None:
    """Open a decode round: re-read the cadence and count collectives."""
    global _active, _k, _calls, _synced
    _calls = 0
    _synced = 0
    _k = current_k() if _installed else 0
    _active = _installed and _k > 0


def emit_round(rank: int, round_ms: float, committed: int) -> str | None:
    """Close one round and emit its line; None when the lever is not installed."""
    global _round_seq, _active
    if not _installed:
        return None
    _active = False
    _round_seq += 1
    return (
        f"[SYNCCAD] n={_round_seq} rank={rank} k={_k} calls={_calls} "
        f"synced={_synced} ms={round_ms:.1f} committed={committed}"
    )


def rank_of(engine: Any) -> int:
    """The engine's device rank, or -1 for a stand-in that has none (tests)."""
    rank = getattr(engine, "device_rank", None)
    return rank if isinstance(rank, int) else -1


def uninstall() -> None:
    """Restore the original collective (tests / teardown)."""
    global _installed, _original, _target, _k
    if _installed and _target is not None and _original is not None:
        _target.all_sum = _original
    _installed = False
    _target = None
    _original = None
    _k = 0
    reset_rounds()
