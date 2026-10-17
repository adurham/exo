"""No-repeat n-gram ban for the DSv4.1 serve path (a greedy loop ESCAPE).

WHAT THIS IS. The GENERIC MLX path (``generator/batch_generate.py``) keeps a
sampler and its loop defenses -- ``_detect_token_loop`` /
``_detect_long_period_loop`` (knobs ``EXO_LOOP_DETECT*``) -- are DETECTORS: they
notice a cycle that has ALREADY formed and let the caller warn/stop/error the
turn. This engine has neither half. It is GREEDY ONLY (``engine``'s
``_resolve_sampler`` refuses ``temperature > 0``; there is no sampler and no
repetition-penalty stage), and it argmaxes a whole DSpark draft + verify BATCH at
once (``rounds._one_round``). So an argmax attractor like the observed
``"Let me write."`` family is not a "degeneration" the turn can be warned out of
-- it is an ABSORBING STATE of the decode map: when the model's next-token
distribution puts that phrase on top, greedy decode reproduces it forever, and
re-running the SAME prompt (which is exactly what a retrying client does) lands
in the SAME state. A detector can only kill such a turn after the fact.

WHY A BAN, NOT A PENALTY. Cloud-grade stacks escape such states through
SAMPLING: temperature, top-p, and a repetition penalty perturb the distribution
enough that the loop is no longer the argmax. Greedy has no such exit. The
cheapest DETERMINISTIC escape that keeps this engine greedy is a NO-REPEAT
N-GRAM BAN: immediately BEFORE the argmax, mask (-inf) any token id that would
complete an n-gram that ALREADY OCCURRED in the sequence's recent history. When
a loop tries to re-enter a cycle it has just run, the token that would close the
repeated n-gram is banned and the argmax lands on the best non-repeating
continuation instead -- the model keeps writing, it just cannot repeat itself
verbatim. No sampling, no randomness, and no change to the greedy contract: the
draft/verify machinery is untouched, the target still argmaxes its OWN rows, it
simply does so over banned logits.

PURITY / RANK AGREEMENT. The ban is a PURE FUNCTION of the produced-token
history (:func:`banned_next_ids`): no model state, no RNG, no wall clock. Both TP
ranks commit byte-identical tokens under greedy, so both build the identical
history and therefore the identical banned set -- the decision needs NO
collective, and adding one would be a hazard, not a help (a rank that decided
differently from its peer would wedge the peer inside the model's own ``all_sum``
-- the exact failure ``dsv41/reasoning_budget.py`` documents). The masking itself
is row-local: it reads and writes only each verify row's own logits.

DEFAULT OFF. With ``EXO_DSV41_NO_REPEAT_NGRAM`` unset (or <= 2) every decode code
path is byte-identical to before this module existed: :func:`banned_next_ids`
returns an EMPTY ban, the knob is "off", and ``rounds._one_round`` keeps its
original ``argmax=True`` forward. ``n=5`` is the suggested value when enabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "MIN_NGRAM",
    "NEG_INF",
    "NoRepeatNgram",
    "argmax_after_ban",
    "banned_next_ids",
    "mask_rows",
]

#: n < MIN_NGRAM is a NO-OP, by contract. A 1-gram ban would ban every token that
#: has ever appeared (including the only sane continuation), and a 2-gram ban
#: fires on ordinary prose; either turns the escape into a different pathology.
#: 3 is the smallest length that targets a genuinely repeated PHRASE.
MIN_NGRAM = 3

#: The masked sentinel. ``float("-inf")`` (rather than a large negative finite
#: value) so a banned id can never win an argmax against a finite logit at any
#: scale, AND so a fully-banned row is unambiguously detectable (its max is -inf).
NEG_INF = float("-inf")


def _maybe_mlx() -> Any | None:
    """``mlx.core`` when importable, else None.

    Keeping this lazy is what lets the pure half of this module (the id-list
    scan and the ban policy) be unit-tested on a box with no MLX at all, while
    the array half runs against real mx arrays in production and numpy
    stand-ins in the tests.
    """
    try:  # pragma: no cover - exercised implicitly by whichever backend is present
        import mlx.core as mx

        return mx
    except Exception:  # noqa: BLE001 - no-MLX build still runs the pure tests
        return None


def banned_next_ids(history: list[int], n: int, window: int) -> set[int]:
    """Ids that would EXTEND an n-gram already present in ``history``.

    Pure over an id list. ``history`` is the produced-token sequence so far
    (oldest first, the just-fed token LAST); ``n`` is the n-gram length; ``window``
    bounds how far back an earlier occurrence may sit, counted in produced tokens
    (the knob's default 2048). A candidate id ``t`` is banned iff the n-gram
    ``history[-(n-1):] + [t]`` already occurred with its last token inside that
    window -- i.e. iff ``t`` is the token that FOLLOWED an earlier occurrence of
    the current ``(n-1)``-token suffix.

    This is the standard no-repeat-n-gram rule (``transformers``'
    ``NoRepeatNGramLogitsProcessor``), reduced to the one thing the greedy path
    needs: the SET of ids to mask. ``n < MIN_NGRAM`` (0/1/2/unset) is the
    documented no-op, so a mis-set knob cannot make the engine degenerate.
    """
    if n < MIN_NGRAM or n - 1 > len(history):
        return set()
    suffix = tuple(int(t) for t in history[-(n - 1):])
    lo = max(0, len(history) - int(window))
    # An earlier occurrence of the suffix must still be FOLLOWED by a token, so
    # its start index is bounded by ``len(history) - n`` (exclusive ``hi``).
    hi = len(history) - n + 1
    banned: set[int] = set()
    for i in range(lo, hi):
        if tuple(int(t) for t in history[i:i + n - 1]) == suffix:
            banned.add(int(history[i + n - 1]))
    return banned


def _clone(logits: Any) -> Any:
    """A fresh, non-aliasing array of the same values.

    mx arrays have no ``.copy()`` (a constructor call is the copy), numpy has
    one; the tests pass whichever the backend provides.
    """
    mx = _maybe_mlx()
    if mx is not None and isinstance(logits, mx.array):
        return mx.array(logits)
    return logits.copy()


def mask_rows(logits: Any, banned_per_row: list[set[int]]) -> Any:
    """Copy of ``logits`` with ``-inf`` at every banned id, one row at a time.

    ``logits`` is ``[rows, vocab]`` (mx or numpy). ``banned_per_row`` is one id
    iterable per row. Rows are independent and the result never aliases the input,
    so two TP ranks applying the same ban to the same row get the same output
    without communicating.
    """
    masked = _clone(logits)
    for row, banned in enumerate(banned_per_row):
        ids = [int(i) for i in banned]
        if ids:
            masked[row, ids] = NEG_INF
    return masked


def _argmax_last(x: Any):
    """Per-row argmax, spelled the way the backend wants it."""
    mx = _maybe_mlx()
    if mx is not None and isinstance(x, mx.array):
        return mx.argmax(x, axis=-1)
    return x.argmax(axis=-1)


def argmax_after_ban(
    logits: Any, banned_per_row: list[set[int]]
) -> tuple[list[int], Any]:
    """Per-row argmax with the ban applied; full-ban rows fall back.

    Returns ``(ids, masked)``. If masking would leave EVERY candidate of a row
    banned (the masked row is all ``-inf``), that row falls back to the UNMASKED
    argmax: the engine must never emit an invalid token, and an all-banned row is
    the one input that could otherwise select one. A full ban is pathological (the
    window would have to contain the current suffix once per vocab entry), but it
    is cheap to make impossible here rather than to argue it away.
    """
    masked = mask_rows(logits, banned_per_row)
    picked = _argmax_last(masked)
    plain = _argmax_last(logits)
    ids: list[int] = []
    for row in range(len(banned_per_row)):
        fully_banned = float(masked[row].max()) == NEG_INF
        ids.append(int(plain[row]) if fully_banned else int(picked[row]))
    return ids, masked


@dataclass(frozen=True)
class NoRepeatNgram:
    """Resolved knob state: n-gram length ``n`` and history ``window``.

    ``enabled`` is the whole "DEFAULT OFF" contract in one place: ``n <
    MIN_NGRAM`` (0/1/2, or an unset knob resolved to 0) disables the ban and every
    method here becomes a no-op, so the decode site can pass this object
    unconditionally.
    """

    n: int = 0
    window: int = 0

    @property
    def enabled(self) -> bool:
        return int(self.n) >= MIN_NGRAM

    def banned(self, history: list[int]) -> set[int]:
        """The banned set for the next token after ``history`` (empty if off)."""
        if not self.enabled:
            return set()
        return banned_next_ids(history, self.n, self.window)

    def banned_per_row(self, history: list[int], draft: list[int]) -> list[set[int]]:
        """One ban per verify-batch row.

        Row 0 predicts the token after ``history``'s last id. Row ``i`` predicts the
        token after the ``i``-th fed draft token, so its context is the produced
        history with the (accepted-in-place) draft prefix ``draft[:i]`` appended.
        Both the history and the draft ids are byte-identical across TP ranks (a
        greedy target and a deterministic DSpark head), so both ranks build the
        same list -- and it is only ever consulted for rows 0..accepted, the rows
        the round can actually commit (see ``rounds._one_round``).
        """
        if not self.enabled:
            return [set() for _ in range(len(draft) + 1)]
        rows = [self.banned(history)]
        context = [int(t) for t in history]
        for token in draft:
            context.append(int(token))
            rows.append(self.banned(context))
        return rows
