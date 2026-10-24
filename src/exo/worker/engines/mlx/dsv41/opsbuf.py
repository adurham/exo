"""Per-round command-buffer batching threshold (``max_ops_per_buffer``).

WHY. ``docs/dsv41-1017-vs-sept-ab-2026-10-20.md`` 8r.11-8r.13: the DSv4.1
slow-decode state at deep context is governed by how much work is batched into
one Metal command buffer. One boot per value is not enough to rank the values --
cross-boot variance (4.66-5.04 at the same setting, and one boot at 3.11) is
larger than some of the gaps being compared -- so the threshold has to be
flippable INSIDE one cold session, where the slow state is shared.

The mlx side now has ``mx.metal.set_max_ops_per_buffer`` (fork commit
cfa28fbbd); this module is the engine-side surface: it re-reads the levers file
at every decode round and applies the value before the round runs, so one boot
can A/B 200/50/8/1 against the SAME cache and the SAME thermal state.

DEFAULT OFF. Without the key (or without the mlx binding) nothing is called and
the process keeps the value it booted with.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

ENV_FILE = "EXO_DSV41_LEVERS_FILE"
KEY = "max_ops_per_buffer"
ENV = "DSV41_MAX_OPS_PER_BUFFER"

_applied: int | None = None
_value: int | None = None
_round_seq = 0
_cache: dict | None = None
_cache_round = -1


def _file_data() -> dict | None:
    global _cache, _cache_round
    if _cache_round == _round_seq and _cache is not None:
        return _cache
    path = os.environ.get(ENV_FILE, "").strip()
    if not path:
        return None
    try:
        data = json.loads(Path(path).read_text())
    except Exception:
        data = None
    _cache = data if isinstance(data, dict) else None
    _cache_round = _round_seq
    return _cache


def requested() -> int | None:
    """The value the levers file asks for, else the env fallback, else None."""
    data = _file_data()
    if data is not None and KEY in data:
        try:
            v = int(data[KEY])
        except (TypeError, ValueError):
            v = None
        if v is not None and v > 0:
            return v
    raw = os.environ.get(ENV, "").strip()
    if raw:
        try:
            v = int(raw)
        except ValueError:
            return None
        return v if v > 0 else None
    return None


def applied() -> int | None:
    """The value this process last set (None until the first apply)."""
    return _applied


def _setter() -> Any:
    try:
        import mlx.core as mx

        return getattr(getattr(mx, "metal", None), "set_max_ops_per_buffer", None)
    except Exception:
        return None


def begin_round() -> str | None:
    """Apply this round's value; return a log line when it changed."""
    global _applied, _value, _round_seq
    _round_seq += 1
    want = requested()
    _value = want
    if want is None or want == _applied:
        return None
    setter = _setter()
    if setter is None:
        return "[OPSBUF] requested but this mlx has no set_max_ops_per_buffer"
    setter(want)
    _applied = want
    return f"[OPSBUF] n={_round_seq} max_ops_per_buffer={want}"


def reset_rounds() -> None:
    """Reset the round counter and the per-round file cache (tests/turn start)."""
    global _round_seq, _cache_round
    _round_seq = 0
    _cache_round = -1


# ---------------------------------------------------------------------------
# Cold-turn policy (Fable's 8r.18 proposal).
#
# The lever is regime-dependent in opposite directions: T=1 is worth +80% on a
# cold deep turn and costs ~50% at 1K. Context length is the WRONG key -- a
# delta-grown 350K session is fast and would eat the 1K-style penalty -- while
# the slow state is created by a long cold prefill. So the key is "this turn had
# to prefill from scratch AND the prompt is deep":
#
#     cold (nothing cached) AND prompt_len >= EXO_DSV41_OPS_COLD_MIN -> T_COLD
#     otherwise                                                     -> T_WARM
#
# Default OFF (EXO_DSV41_OPS_POLICY=1 enables): the mechanism is not understood
# well enough to change default behaviour without a review, and a per-session
# setting is only sound while a decode step is single-tenant. The levers file
# still wins over the policy so a diagnostic run can force any value.
ENV_POLICY = "EXO_DSV41_OPS_POLICY"
ENV_COLD_MIN = "EXO_DSV41_OPS_COLD_MIN"
ENV_T_COLD = "EXO_DSV41_OPS_T_COLD"
ENV_T_WARM = "EXO_DSV41_OPS_T_WARM"


def policy_enabled() -> bool:
    return os.environ.get(ENV_POLICY, "0").strip() not in ("", "0")


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


#: Conversations whose cache was created cold at depth. The slow state is a
#: property of the SESSION (it persists for its life), so once a conversation has
#: been cold-filled at depth it keeps the cold threshold for every later turn --
#: keying only on "this turn was cold" would hand the resends straight back to the
#: slow state.
_cold_deep: set[str] = set()


def apply_for_turn(
    cached_tokens: int, prompt_len: int, key: str | None = None
) -> str | None:
    """Pick the threshold for this request; returns a log line when it changes.

    No-op unless the policy is enabled and the levers file is not overriding.
    """
    global _applied
    if not policy_enabled():
        return None
    data = _file_data()
    if data is not None and KEY in data:
        return None  # a diagnostic run owns the value
    cold_min = _int_env(ENV_COLD_MIN, 200_000)
    if cached_tokens <= 0 and prompt_len >= cold_min and key is not None:
        _cold_deep.add(key)
    sticky = key is not None and key in _cold_deep
    cold_now = cached_tokens <= 0 and prompt_len >= cold_min
    target = _int_env(ENV_T_COLD, 1) if (cold_now or sticky) else _int_env(ENV_T_WARM, 50)
    setter = _setter()
    if setter is None:
        return "[OPSBUF] policy requested but this mlx has no set_max_ops_per_buffer"
    if target == _applied:
        return None
    setter(target)
    _applied = target
    return (
        f"[OPSBUF] policy cached={cached_tokens} prompt={prompt_len} "
        f"cold_min={cold_min} cold_deep={'yes' if sticky else 'no'} "
        f"-> max_ops_per_buffer={target}"
    )
