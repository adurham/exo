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

CONTROL-TOKEN EXEMPTION. Repetition of special/control tokens -- EOS, the
thinking start/end delimiters, tool-call delimiters and other template tokens --
is NORMAL and MUST NOT be banned; an engine that masked its own delimiters could
not close a reasoning block or a tool call. The engine resolves a small EXEMPT
set (the tokenizer's EOS ids plus the think start/end ids it already resolves for
the reasoning-budget limiter) and may be EXTENDED with the comma-separated
``EXO_DSV41_NO_REPEAT_NGRAM_EXEMPT_IDS``; exemption is subtracted from the ban
AFTER the n-gram rule, so an exempt id is never in a ban set.

DELIBERATE NON-GOALS (this pass). Two follow-ups a review raised are intentionally
OUT of scope here, listed so they are not mistaken for oversights:
  * ADAPTIVE ARMING of the ban after a repetition detector trips -- the knob stays
    a static, operator-set value; arming it from a live detector would add a
    rank-visible control decision this module deliberately avoids;
  * banning DRAFT PROPOSALS too -- only the TARGET's verify rows are masked; the
    DSpark head is still free to propose a repeat (the target rejects it), so the
    draft head's own argmax is untouched.

DEFAULT OFF. With ``EXO_DSV41_NO_REPEAT_NGRAM`` unset (or <= 2) every decode code
path is byte-identical to before this module existed: :func:`banned_next_ids`
returns an EMPTY ban, the knob is "off", and ``rounds._one_round`` keeps its
original ``argmax=True`` forward. :data:`SUGGESTED_N` (6) is the suggested value
when enabled.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

__all__ = [
    "MIN_NGRAM",
    "NEG_INF",
    "SUGGESTED_N",
    "BanTurnStats",
    "contiguous_repeat_banned_ids",
    "NoRepeatNgram",
    "argmax_after_ban",
    "argmax_after_ban_detailed",
    "banned_next_ids",
    "format_ban_line",
    "mask_rows",
]

#: n < MIN_NGRAM is a NO-OP, by contract. A 1-gram ban would ban every token that
#: has ever appeared (including the only sane continuation), and a 2-gram ban
#: fires on ordinary prose; either turns the escape into a different pathology.
#: 3 is the smallest length that targets a genuinely repeated PHRASE.
MIN_NGRAM = 3

#: The suggested n-gram length when the knob is enabled (see the knobs in
#: ``engine.py``). Documented here, next to the rule, so comments/docs and the
#: tests all read the suggestion from ONE place.
SUGGESTED_N = 6

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


def banned_next_ids(
    history: list[int], n: int, window: int, exempt: frozenset[int] = frozenset()
) -> set[int]:
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

    ``exempt`` is the CONTROL-TOKEN exemption (EOS, think delimiters, template
    tokens): it is subtracted AFTER the rule, so an exempt id can never appear in
    the returned set no matter how strongly the n-gram rule would forbid it.
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
    if exempt:
        banned -= {int(i) for i in exempt}
    return banned


def _has_letter(text: str) -> bool:
    return any(ch.isalpha() for ch in text)


def contiguous_repeat_penalties(
    history: list[int],
    max_period: int = 0,
    min_blocks: int = 3,
    exempt: frozenset[int] = frozenset(),
    word_only: bool = False,
    decode: Any | None = None,
    soft_multiplier: float = 0.0,
    soft_base: float = 1.75,
    soft_allowed: int = 4,
) -> dict[int, float]:
    """DRY-style graded penalty for a contiguous repeat: ``{id: delta_logit}``.

    The first qualifying repeat (exactly ``min_blocks`` back-to-back blocks) gets
    a SOFT penalty -- ``soft_multiplier * soft_base ** max(0, period -
    soft_allowed)`` -- subtracted from the candidate's logit, the shape
    llama.cpp/vLLM's DRY sampler uses: enough to nudge a model off a coincidence
    (the next octet of an IPv4 tail, a table separator) without forbidding a
    token the model strongly expects. Once one MORE block repeats (the attractor
    is real, ``k >= min_blocks + 1``) the candidate is banned outright
    (``NEG_INF``). ``soft_multiplier <= 0`` makes every tier hard (the old
    behaviour). ``exempt`` control tokens and the ``word_only`` structure filter
    are applied last, as in the hard-only rule.
    """
    if int(max_period) < 1 or int(min_blocks) < 2:
        return {}
    need = int(max_period) * (int(min_blocks) + 1)
    h = [int(t) for t in history[-need:]] if need > 0 else []
    limit = min(int(max_period), len(h) // int(min_blocks))
    out: dict[int, float] = {}
    for p in range(1, limit + 1):
        last = h[-p:]
        ok = True
        for b in range(2, int(min_blocks) + 1):
            if h[-b * p:-(b - 1) * p] != last:
                ok = False
                break
        if not ok:
            continue
        at_least = int(min_blocks) + 1
        deeper = len(h) >= p * at_least and h[-at_least * p:-int(min_blocks) * p] == last
        if deeper or float(soft_multiplier) <= 0.0:
            score = NEG_INF
        else:
            score = float(soft_multiplier) * (
                float(soft_base) ** max(0, p - int(soft_allowed))
            )
        cand = int(last[0])
        if cand in out and out[cand] <= score:
            continue
        out[cand] = score
    if exempt:
        out = {i: s for i, s in out.items() if i not in exempt}
    if word_only and decode is not None:
        out = {i: s for i, s in out.items() if _has_letter(decode(i))}
    return out


def contiguous_repeat_banned_ids(
    history: list[int],
    max_period: int = 0,
    min_blocks: int = 3,
    exempt: frozenset[int] = frozenset(),
    word_only: bool = False,
    decode: Any | None = None,
) -> set[int]:
    """Ids that would EXTEND a CONTIGUOUS periodic repeat (hard-only view).

    Kept for callers/tests that want the binary rule; production uses
    :func:`contiguous_repeat_penalties` (graded soft/hard tiers).
    """
    return set(
        contiguous_repeat_penalties(
            history,
            max_period=max_period,
            min_blocks=min_blocks,
            exempt=exempt,
            word_only=word_only,
            decode=decode,
            soft_multiplier=0.0,
        ).keys()
    )

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

    ``logits`` is ``[rows, vocab]`` (mx or numpy). ``banned_per_row`` is one
    mapping ``{id: delta}`` per row (``NEG_INF`` = hard ban; a finite value is the
    DRY-style graded penalty, ADDED to the logit), or a plain id set for a hard
    ban. Rows are independent and the result never aliases the input, so two TP
    ranks applying the same penalty to the same row get the same output without
    communicating.
    """
    masked = _clone(logits)
    for row, banned in enumerate(banned_per_row):
        if isinstance(banned, dict):
            for i, delta in banned.items():
                masked[row, int(i)] = masked[row, int(i)] + float(delta)
        else:
            ids = [int(i) for i in banned]
            if ids:
                masked[row, ids] = NEG_INF
    return masked


def _argmax_last(x: Any):
    """Per-row argmax, spelled the way the backend wants it.

    PRODUCTION IS MX: on the real engine ``x`` is always an ``mlx.core`` array, so
    this takes the ``mx.argmax`` branch on BOTH TP ranks (identical op, identical
    result, no collective). The duck-typed ``x.argmax(axis=-1)`` branch exists only
    so the CPU tests can pass a numpy stand-in; it is a test affordance, never a
    production path.
    """
    mx = _maybe_mlx()
    if mx is not None and isinstance(x, mx.array):
        return mx.argmax(x, axis=-1)
    return x.argmax(axis=-1)


def argmax_after_ban_detailed(
    logits: Any, banned_per_row: list[set[int]]
) -> tuple[list[int], Any, list[bool], list[int]]:
    """Masked argmax PLUS the per-row ban telemetry (see :func:`argmax_after_ban`).

    Returns ``(ids, masked, changed, masked_counts)``.

    * ``ids``/``masked`` are exactly what :func:`argmax_after_ban` returns;
    * ``changed[row]`` is True iff the ban MOVED the argmax off the id the raw
      logits would have picked (a row whose every candidate was banned fell back
      to the raw argmax and is therefore NOT "changed");
    * ``masked_counts[row]`` is how many candidate ids the ban removed from that
      row.

    The RAW argmax (``plain``) is computed HERE, inside the ban helper, and is used
    only for the all-banned fallback and this diagnostic count: a caller on the
    enabled path never computes a raw argmax of its own, so the committed token can
    never be a raw/sharded argmax.
    """
    masked = mask_rows(logits, banned_per_row)
    picked = _argmax_last(masked)
    plain = _argmax_last(logits)
    ids: list[int] = []
    changed: list[bool] = []
    masked_counts: list[int] = []
    for row in range(len(banned_per_row)):
        fully_banned = float(masked[row].max()) == NEG_INF
        ids.append(int(plain[row]) if fully_banned else int(picked[row]))
        changed.append((not fully_banned) and int(picked[row]) != int(plain[row]))
        masked_counts.append(len(banned_per_row[row]))
    return ids, masked, changed, masked_counts


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
    ids, masked, _changed, _masked_counts = argmax_after_ban_detailed(
        logits, banned_per_row
    )
    return ids, masked


@dataclass
class BanTurnStats:
    """Turn-cumulative no-repeat n-gram telemetry (ONE INFO line per turn).

    ``rows_changed`` counts COMMITTED verify rows where the mask moved the argmax
    off the id the raw logits would have picked; ``masked_total`` is the number of
    candidate ids masked across those rows. Both are diagnostics only -- neither
    feeds the decode decision (the committed tokens are byte-identical with the
    counters on or off).
    """

    rows_changed: int = 0
    masked_total: int = 0


def format_ban_line(ban: "NoRepeatNgram", stats: BanTurnStats) -> str:
    """The ONE per-turn INFO line: stable field order and spelling."""
    return (
        f"[DSV41] no-repeat-ngram n={int(ban.n)} window={int(ban.window)} "
        f"contig_max_period={int(ban.contig_max_period)} "
        f"contig_min_blocks={int(ban.contig_min_blocks)} "
        f"contig_word_only={int(bool(ban.contig_word_only))} "
        f"soft_multiplier={float(ban.soft_multiplier):g} "
        f"rows_changed={int(stats.rows_changed)} "
        f"masked_total={int(stats.masked_total)}"
    )


@dataclass(frozen=True)
class NoRepeatNgram:
    """Resolved knob state: n-gram length ``n``, history ``window``, ``exempt``.

    ``enabled`` is the whole "DEFAULT OFF" contract in one place: ``n <
    MIN_NGRAM`` (0/1/2, or an unset knob resolved to 0) disables the ban and every
    method here becomes a no-op, so the decode site can pass this object
    unconditionally.

    ``exempt`` is the CONTROL-TOKEN set (EOS, think start/end, template tokens)
    that is NEVER banned; it is installed by the engine at resolve time when the
    knob is enabled and is empty otherwise.
    """

    n: int = 0
    window: int = 0
    exempt: frozenset[int] = field(default_factory=frozenset)
    #: CONTIGUOUS-PERIOD escape (the shipped rule): mask only while the model
    #: repeats a block back-to-back. 0 = off; ``contig_min_blocks`` identical
    #: blocks of period p <= ``contig_max_period`` must end the history.
    contig_max_period: int = 0
    contig_min_blocks: int = 3
    #: Only ban candidates whose decoded text carries a letter (structure --
    #: digits/punctuation -- is never a semantic loop). Needs ``decode``.
    contig_word_only: bool = False
    #: id -> text, supplied by the engine from the tokenizer (word_only filter).
    decode: Any | None = None
    #: DRY-style graded penalty for the FIRST qualifying repeat; <= 0 = hard.
    soft_multiplier: float = 0.0
    soft_base: float = 1.75
    soft_allowed: int = 4

    @property
    def legacy_enabled(self) -> bool:
        return int(self.n) >= MIN_NGRAM

    @property
    def contig_enabled(self) -> bool:
        return int(self.contig_max_period) >= 1 and int(self.contig_min_blocks) >= 2

    @property
    def enabled(self) -> bool:
        return self.legacy_enabled or self.contig_enabled

    def with_exempt(self, exempt: "frozenset[int] | set[int] | None") -> "NoRepeatNgram":
        """A copy with the control-token ``exempt`` set installed (frozen)."""
        return replace(self, exempt=frozenset(int(i) for i in (exempt or ())))

    def banned(self, history: list[int]) -> set[int]:
        """The banned set for the next token after ``history`` (empty if off).

        Both rules can be armed at once; the shipped configuration uses the
        contiguous-period rule alone (the global rule is kept for A/B work and
        remains default-off).
        """
        if not self.enabled:
            return set()
        out: set[int] = set()
        if self.legacy_enabled:
            out |= banned_next_ids(history, self.n, self.window, self.exempt)
        if self.contig_enabled:
            out |= set(
                contiguous_repeat_penalties(
                    history,
                    self.contig_max_period,
                    self.contig_min_blocks,
                    self.exempt,
                    word_only=bool(self.contig_word_only),
                    decode=self.decode,
                    soft_multiplier=float(self.soft_multiplier),
                    soft_base=float(self.soft_base),
                    soft_allowed=int(self.soft_allowed),
                ).keys()
            )
        return out

    def penalties(self, history: list[int]) -> dict[int, float]:
        """``{id: delta_logit}`` -- the graded rule; ``NEG_INF`` = hard ban."""
        if not self.enabled:
            return {}
        out: dict[int, float] = {}
        if self.legacy_enabled:
            for i in banned_next_ids(history, self.n, self.window, self.exempt):
                out[int(i)] = NEG_INF
        if self.contig_enabled:
            for i, s in contiguous_repeat_penalties(
                history,
                self.contig_max_period,
                self.contig_min_blocks,
                self.exempt,
                word_only=bool(self.contig_word_only),
                decode=self.decode,
                soft_multiplier=float(self.soft_multiplier),
                soft_base=float(self.soft_base),
                soft_allowed=int(self.soft_allowed),
            ).items():
                if int(i) not in out or s < out[int(i)]:
                    out[int(i)] = float(s)
        return out

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
        return [{i for i in row} for row in self.penalties_per_row(history, draft)]

    def penalties_per_row(
        self, history: list[int], draft: list[int]
    ) -> list[dict[int, float]]:
        """One graded penalty map per verify-batch row (see ``banned_per_row``)."""
        if not self.enabled:
            return [{} for _ in range(len(draft) + 1)]
        rows = [self.penalties(history)]
        context = [int(t) for t in history]
        for token in draft:
            context.append(int(token))
            rows.append(self.penalties(context))
        return rows
