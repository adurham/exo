"""Repetition-loop (decode degeneration) guards for the DSv4.1 serve path.

WHAT THIS IS. The two degeneration detectors the GENERIC MLX path owns live in
``exo.worker.engines.mlx.generator.batch_generate``:

* ``_detect_token_loop`` -- a tight EXACT-token cycle at the tail of the recent
  ids (default window 160, max period 24, min repeats 6);
* ``_detect_long_period_loop`` -- the SAME periodicity scan run over hashes of
  fixed 24-token blocks, which reaches word-for-word-repeated multi-SENTENCE
  cycles (period up to 12 blocks = 288 raw tokens, min repeats 3, exact match
  only, re-verified on the raw blocks before terminating).

The custom DSv4.1 engine deliberately bypasses ``mlx_generate`` /
``ExoBatchGenerator`` (see ``dsv41/engine.py``'s module docstring), so it
inherited NEITHER guard: its only stop conditions were EOS, stop sequences and
the ``MAX_TOKENS`` cap. A greedy decode collapse is an ABSORBING state -- the
model re-emits the same cycle forever -- so a detector that kills the turn is
the only deterministic guarantee of termination. This module is that port.

REUSE, NOT REIMPLEMENTATION. Both detectors and every tunable env knob are
imported VERBATIM from ``batch_generate`` (private names included, on purpose):
one source of truth for the two call sites, so a threshold change can never
apply to only one engine. What is new here is only the per-request DRIVER. The
generic path feeds the detectors one token at a time from its own streaming
loop; a DSv4.1 decode round commits a whole BATCH of tokens at once (DSpark
draft + verify), so the driver here presents the guard a strictly sequential
COMMITTED-token stream -- every committed token, in order, never a draft, never
only round boundaries -- which is what keeps the cycle alignment identical to
the generic path's.

PURITY / RANK AGREEMENT. ``DegenerationGuard.feed`` is a pure function of the
committed token ids (plus the import-time env knobs). Both TP ranks commit
byte-identical tokens -- the DSpark round only ever emits tokens the TARGET
model itself confirmed -- so both ranks compute an identical verdict and
terminate together; the termination decision is rank-invariant BY CONSTRUCTION,
with no extra collective. The one way the ranks could disagree is a KNOB that
differs between the two nodes, so

* the effective values are logged once per rank at engine construction
  (:func:`guard_env_report`), and
* the guard-fire WARNING line carries the env digest (:func:`guard_env_digest`),
  so the two ranks' lines for the SAME request can be diffed directly in the
  runner logs -- which is the cross-rank knob check, at zero cost and at exactly
  the moment it matters.

A dedicated rank-agreement collective was considered and deliberately NOT added:
the engine has no startup-synchronised program point (``warmup`` is only reached
on the reconnect path, and the request-path collectives pair across the
task/cancel channels in a way that is not trivially symmetric), so a new
collective there would add a whole new hang class for a diagnostic. Documented
as a residual in the buildnote instead.

FINISH-REASON POLICY. The default action is ``"error"`` -- the SAME knob and the
SAME default as the generic path, so the two engines cannot silently disagree
about what a detected loop means. What the CLIENT then does is source-traced in
``docs/dsv41-degeneration-guards-2026-10-13.md`` and is NOT the x3-api-retry
story an earlier revision of this note told: a guard error reaches the wire as an
SSE ``data: {"error": ...}`` (never a ``finish_reason="error"`` delta), and after
PARTIAL delivery Hermes does NOT retry -- it returns a partial ``length`` stub and
auto-continues (bounded, truncation-surfaced). ``"error"`` stays the default for
signal integrity and parity with the generic path: ``"stop"`` would surface the
already-degenerate text to the agent as a *successful* turn (and a loop inside a
tool-call argument as a truncated, malformed tool call on a turn the client
believes completed). ``"stop"`` and ``"warn"`` remain available through the same
knob; ``"warn"`` is the recommended FIRST staging step (measure the real fire rate
before arming the kill).
"""
# pyright: reportPrivateUsage=false
#   (deliberate: the guard reuses batch_generate's detectors and knobs verbatim so the
#   two engines keep ONE source of truth; the names are private there by convention.)

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from exo.worker.engines.mlx.generator.batch_generate import (
    _DEGENERATION_ERROR_TEXT,
    _LOOP_DETECT_ENABLED,
    _LOOP_DETECT_LONG_BLOCK,
    _LOOP_DETECT_LONG_ENABLED,
    _LOOP_DETECT_LONG_MAX_PERIOD,
    _LOOP_DETECT_LONG_MIN_REPEATS,
    _LOOP_DETECT_LONG_WINDOW_BLOCKS,
    _LOOP_DETECT_MAX_PERIOD,
    _LOOP_DETECT_MIN_REPEATS,
    _LOOP_DETECT_WINDOW,
    _detect_long_period_loop,
    _detect_token_loop,
)
from exo.worker.runner.bootstrap import logger

__all__ = [
    "DEFAULT_LOOP_ACTION",
    "DEGENERATION_ERROR_TEXT",
    "DUMP_TOKEN_LIMIT",
    "REPETITION_LOOP_ERROR_CODE",
    "DegenerationGuard",
    "LoopAction",
    "LoopHit",
    "assert_guard_env_agrees_across_ranks",
    "guard_env_digest",
    "guard_env_fingerprint",
    "guard_env_report",
    "guard_env_values",
    "loop_action",
    "write_degeneration_dump",
]

#: The knob shared with the generic MLX path (``batch_generate``). Reused as-is:
#: a single var must mean the same thing on both engines.
LOOP_ACTION_ENV = "EXO_LOOP_DETECT_ACTION"

#: dsv41's default action. Deliberately the SAME as the generic path's default
#: ("error") -- see the module docstring's FINISH-REASON POLICY section.
DEFAULT_LOOP_ACTION: "LoopAction" = "error"

#: What a detected loop does to the turn:
#: ``"error"`` -- fail the turn cleanly (finish_reason="error" -> ErrorChunk);
#: ``"stop"``  -- terminate but surface the (already streamed) partial as a
#:                normal completion;
#: ``"warn"``  -- log only, never terminate (the pre-port behaviour).
LoopAction = Literal["error", "stop", "warn"]

#: The diagnostic that REPLACES the degenerate partial under the "error" action
#: (re-exported so ``engine.py`` has one import seam for the whole guard).
DEGENERATION_ERROR_TEXT = _DEGENERATION_ERROR_TEXT

#: Structured OpenAI-style error code carried on a guard clean-fail ``ErrorChunk``
#: (``finish_reason="error"``). A client can classify a degeneration kill WITHOUT
#: string-matching the human message; the message itself is unchanged. Emitted
#: through ``_final_response(error_code=...)`` -> ``map_responses_to_chunks`` ->
#: ``ErrorChunk.error_code``, the same plumbing the context-length refusal uses.
REPETITION_LOOP_ERROR_CODE = "repetition_loop"

#: Committed token ids dumped to the log dir when a turn is cut short by the
#: guard or by a client cancel. Bounded on purpose: the dump exists so the next
#: incident is diagnosable from disk (the incident that motivated this port
#: streamed ~3m41s of repetition and its text was never captured), and 2048 ids
#: comfortably covers every detector's window while staying a few KB.
DUMP_TOKEN_LIMIT = 2048

#: Override for where a dump lands. Read at CALL time, so a test can point it at
#: a tmp dir; production leaves it unset and the dump goes to exo's own log dir
#: (resolved lazily -- importing ``exo.shared.constants`` probes the dashboard
#: directory at import time, which must not become a new import-time requirement
#: of the engine module).
DUMP_DIR_ENV = "EXO_DSV41_DEGENERATION_DUMP_DIR"


def _dump_dir_default() -> Path:
    """exo's own log directory, resolved on demand (never at import time)."""
    from exo.shared.constants import EXO_LOG_DIR

    return Path(EXO_LOG_DIR)


#: Filename-safe characters only, so a task id can never escape the log dir.
_UNSAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")


def loop_action() -> "LoopAction":
    """The effective guard action from ``EXO_LOOP_DETECT_ACTION``.

    Resolved at call time (not import time) so a test can drive it with
    ``monkeypatch.setenv``. An unrecognized value falls back to the default
    rather than silently behaving like "warn": a typo must not disarm the guard.
    """
    raw = os.environ.get(LOOP_ACTION_ENV)
    if raw is None:
        return DEFAULT_LOOP_ACTION
    value = raw.strip().lower()
    if value in ("error", "stop", "warn"):
        return cast("LoopAction", value)
    logger.warning(
        f"[DSV41] ignoring unknown {LOOP_ACTION_ENV}={raw!r}; "
        f"using {DEFAULT_LOOP_ACTION!r}"
    )
    return DEFAULT_LOOP_ACTION


def guard_env_values() -> tuple[tuple[str, str], ...]:
    """The EFFECTIVE guard settings, as an ordered, diffable ``(name, value)`` tuple.

    Reports what the process actually resolved -- including the import-time
    defaults -- not what the operator believed they exported. This is the
    deployed-config-is-checkable convention the prefill controls already use: the
    launcher builds the runner environment from an explicit allow-list, so an
    unlisted variable is silently absent and BOTH ranks would quietly run the
    default.
    """
    return (
        ("EXO_LOOP_DETECT", "1" if _LOOP_DETECT_ENABLED else "0"),
        ("EXO_LOOP_DETECT_WINDOW", str(_LOOP_DETECT_WINDOW)),
        ("EXO_LOOP_DETECT_MAX_PERIOD", str(_LOOP_DETECT_MAX_PERIOD)),
        ("EXO_LOOP_DETECT_MIN_REPEATS", str(_LOOP_DETECT_MIN_REPEATS)),
        (LOOP_ACTION_ENV, loop_action()),
        ("EXO_LOOP_DETECT_LONG", "1" if _LOOP_DETECT_LONG_ENABLED else "0"),
        ("EXO_LOOP_DETECT_LONG_BLOCK", str(_LOOP_DETECT_LONG_BLOCK)),
        ("EXO_LOOP_DETECT_LONG_MAX_PERIOD", str(_LOOP_DETECT_LONG_MAX_PERIOD)),
        ("EXO_LOOP_DETECT_LONG_MIN_REPEATS", str(_LOOP_DETECT_LONG_MIN_REPEATS)),
    )


def guard_env_digest() -> str:
    """Short stable digest of the effective guard env (identical across ranks iff
    their knobs are). Carried on every guard-fire log line so the two ranks'
    lines for the same request can be compared without any extra exchange."""
    canon = "\n".join(f"{name}={value}" for name, value in guard_env_values())
    return hashlib.blake2b(canon.encode("utf-8"), digest_size=4).hexdigest()


def guard_env_report(rank: int, world: int) -> str:
    """One startup log line naming this rank's effective guard settings."""
    values = " ".join(f"{name}={value}" for name, value in guard_env_values())
    return (
        f"[DSV41] loop-guard env (rank {rank}/{world}, digest={guard_env_digest()}): "
        f"{values}"
    )


def guard_env_fingerprint() -> int:
    """A non-negative int fingerprint of the effective guard env, for the
    cross-rank equality reduce.

    Masked to 31 bits so the value marshals cleanly through the int32 the
    collective sends; the full (32-bit) hex digest stays the human-facing one in
    the log lines, so a fingerprint collision would still leave the digests
    diffable by eye.
    """
    return int(guard_env_digest(), 16) & 0x7FFFFFFF


def assert_guard_env_agrees_across_ranks(group: Any) -> None:
    """Fail the BOOT when the TP ranks' guard knobs differ.

    A knob that differs between ranks is a hang class, not a cosmetic drift: the
    guard's verdict is a pure function of the committed ids (identical on both
    ranks), so with, say, a different ``EXO_LOOP_DETECT_MIN_REPEATS`` one rank
    terminates a degenerate turn while its peer keeps decoding -- and the peer
    then blocks forever in the model's own TP ``all_sum``. The previous feature
    only LOGGED the effective env per rank, so a mismatch survived startup and
    only showed up as a hung decode.

    The check reduces each rank's :func:`guard_env_fingerprint` with BOTH a min
    and a max over ``group`` (the coord subgroup) and raises on disagreement.
    It is placed at the engine-construction point, where ``get_coord_group`` is
    ALREADY called by both ranks in matching order (its docstring's own
    requirement: "same code path on both ranks at the same point"), and reuses
    the existing ``mx_min_int`` collective helper rather than inventing a new
    one. ``group is None`` (single rank, PP mode -- see ``get_coord_group`` -- or
    the unit tests) is a no-op: there is nothing to compare against.
    """
    if group is None:
        return
    from exo.worker.engines.mlx.utils_mlx import mx_min_int

    fingerprint = guard_env_fingerprint()
    lo = mx_min_int(fingerprint, group)
    hi = -mx_min_int(-fingerprint, group)
    if lo != hi:
        raise RuntimeError(
            f"[DSV41] loop-guard env MISMATCH across ranks: min={lo} max={hi} "
            f"(this rank fingerprint={fingerprint}, digest={guard_env_digest()}). "
            "A guard knob differs between the TP ranks, so one rank would "
            "terminate a degenerate turn while its peer keeps decoding -- a "
            "silent hang. Align the runner environments (the launcher must pass "
            "every EXO_LOOP_DETECT* var to BOTH ranks) and restart."
        )


@dataclass(frozen=True)
class LoopHit:
    """One CONFIRMED repetition cycle, over the committed-token stream.

    ``detector`` says which mechanism fired (they cover disjoint failure
    signatures and can both be live on one request). ``period``/``repeats`` are
    in the detector's own units -- raw tokens for ``"token"``, 24-token blocks
    for ``"long_period"`` -- and ``cycle_token_ids`` is always the RAW token ids
    of the repeating tail, which is what gets decoded for the log line.
    """

    detector: Literal["token", "long_period"]
    period: int
    repeats: int
    cycle_token_ids: list[int]


@dataclass
class DegenerationGuard:
    """Per-request degeneration guard over the COMMITTED token stream.

    Construct one per request and call :meth:`feed` with every committed token,
    in order, as it is committed (never a draft token: a draft the target model
    did not confirm never reaches the client and must not shape the verdict).

    The two detectors mirror ``batch_generate``'s call site exactly, including
    its warn-once semantics: once the tight detector has fired, it stops
    scanning, and the long-period detector keeps its own independent
    once-only latch (so with the non-terminating ``"warn"`` action both can
    report, and with a terminating action the caller stops feeding after the
    first hit anyway).

    All thresholds default to the shared module-level knobs; they are fields so
    a test can pin them without touching the process env.
    """

    enabled: bool = _LOOP_DETECT_ENABLED
    window: int = _LOOP_DETECT_WINDOW
    max_period: int = _LOOP_DETECT_MAX_PERIOD
    min_repeats: int = _LOOP_DETECT_MIN_REPEATS
    long_enabled: bool = _LOOP_DETECT_LONG_ENABLED
    long_block: int = _LOOP_DETECT_LONG_BLOCK
    long_max_period: int = _LOOP_DETECT_LONG_MAX_PERIOD
    long_min_repeats: int = _LOOP_DETECT_LONG_MIN_REPEATS
    long_window_blocks: int = _LOOP_DETECT_LONG_WINDOW_BLOCKS
    #: Decodes a list of committed ids to text. When set, the tight detector
    #: exempts a run of a single SEPARATOR glyph (whitespace/punctuation only: a
    #: markdown rule, an ``=====`` banner, a dotted ellipsis, a code-fence or
    #: ASCII-art row, a table separator) -- formatting, not degeneration. It is
    #: deliberately optional so a bare guard (the unit tests, and any caller with
    #: no tokenizer) keeps the pre-fix behaviour.
    token_text: Callable[[list[int]], str] | None = None
    #: Bounded sliding window of recent committed ids (the tight detector's input).
    recent_token_ids: list[int] = field(default_factory=list)
    #: Partial trailing 24-token block (the long detector's input).
    pending_block_tokens: list[int] = field(default_factory=list)
    #: Completed block hashes + their raw blocks (for the exact re-verify).
    block_hashes: list[int] = field(default_factory=list)
    block_tokens: list[list[int]] = field(default_factory=list)
    _token_warned: bool = field(default=False, init=False)
    _long_warned: bool = field(default=False, init=False)

    @property
    def token_warned(self) -> bool:
        """Whether the tight detector has already reported this request."""
        return self._token_warned

    @property
    def long_warned(self) -> bool:
        """Whether the long-period detector has already reported this request."""
        return self._long_warned

    def feed(self, token_id: int) -> LoopHit | None:
        """Feed ONE committed token; returns a hit on the token that confirms one.

        ``None`` for the (overwhelmingly common) no-loop case and for every
        token after a detector has already reported.
        """
        if not self.enabled:
            return None
        hit = self._feed_tight(int(token_id))
        if hit is not None:
            return hit
        return self._feed_long(int(token_id))

    def _feed_tight(self, token_id: int) -> LoopHit | None:
        """The exact-token cycle detector, over the bounded recent-id window."""
        if self._token_warned:
            return None
        ids = self.recent_token_ids
        ids.append(token_id)
        if len(ids) > self.window:
            del ids[: len(ids) - self.window]
        loop = _detect_token_loop(
            ids, self.max_period, self.min_repeats, self.token_text
        )
        if loop is None:
            return None
        self._token_warned = True
        period, repeats = loop
        return LoopHit("token", period, repeats, list(ids[len(ids) - period :]))

    def _feed_long(self, token_id: int) -> LoopHit | None:
        """The block-hash cycle detector, over completed 24-token blocks."""
        if not self.long_enabled or self._long_warned:
            return None
        pending = self.pending_block_tokens
        pending.append(token_id)
        if len(pending) < self.long_block:
            return None
        block = pending[: self.long_block]
        self.pending_block_tokens = pending[self.long_block :]
        self.block_hashes.append(hash(tuple(block)))
        self.block_tokens.append(block)
        if len(self.block_hashes) > self.long_window_blocks:
            overflow = len(self.block_hashes) - self.long_window_blocks
            del self.block_hashes[:overflow]
            del self.block_tokens[:overflow]
        loop = _detect_long_period_loop(
            self.block_hashes, self.long_max_period, self.long_min_repeats
        )
        if loop is None:
            return None
        blocks = self.block_tokens
        period, _hash_repeats = loop
        tail = blocks[len(blocks) - period :]
        # Re-verify the RAW blocks: the hash match is only a candidate (see
        # _detect_long_period_loop's docstring). Cheap -- detection is rare.
        verified = 1
        pos = len(blocks) - period
        while pos - period >= 0 and blocks[pos - period : pos] == tail:
            verified += 1
            pos -= period
        if verified < self.long_min_repeats:
            return None
        self._long_warned = True
        return LoopHit(
            "long_period",
            period,
            verified,
            [t for block_tokens in tail for t in block_tokens],
        )


def write_degeneration_dump(
    *,
    token_ids: Sequence[int],
    reason: str,
    task_id: str,
    generated: int,
    extra: Mapping[str, Any] | None = None,
    log_dir: Path | str | None = None,
) -> str | None:
    """Write the last ``DUMP_TOKEN_LIMIT`` committed ids to the exo log dir.

    Best-effort and bounded: this exists so the NEXT incident's actual loop text
    is on disk (the one that motivated this port was never captured), so a
    failure to write must never disturb serving -- it logs and returns ``None``.
    Returns the written path on success.
    """
    try:
        override = os.environ.get(DUMP_DIR_ENV)
        if log_dir is not None:
            directory = Path(log_dir)
        elif override:
            directory = Path(override)
        else:
            directory = _dump_dir_default()
        directory.mkdir(parents=True, exist_ok=True)
        tail = [int(t) for t in list(token_ids)[-DUMP_TOKEN_LIMIT:]]
        slug = _UNSAFE_FILENAME.sub("-", str(reason))[:32] or "dump"
        stamp = time.strftime("%Y%m%d-%H%M%S")
        name = f"dsv41-degeneration-{slug}-{stamp}-{os.getpid()}.json"
        path = directory / name
        payload: dict[str, Any] = {
            "reason": str(reason),
            "task_id": str(task_id),
            "generated": int(generated),
            "returned_tokens": len(tail),
            "guard_env_digest": guard_env_digest(),
            "guard_env": dict(guard_env_values()),
            "token_ids": tail,
        }
        if extra:
            payload.update(dict(extra))
        path.write_text(json.dumps(payload), encoding="utf-8")
        return str(path)
    except Exception as exc:  # noqa: BLE001 - a diagnostic must never break serving
        logger.warning(
            f"[DSV41] degeneration dump failed ({type(exc).__name__}: {exc})"
        )
        return None
