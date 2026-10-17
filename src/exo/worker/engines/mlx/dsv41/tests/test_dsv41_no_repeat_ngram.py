"""CPU-only tests for the DSv4.1 no-repeat n-gram ban (the greedy loop ESCAPE).

Three things are pinned here:

* the PURE ban (``no_repeat.banned_next_ids`` + ``no_repeat.argmax_after_ban``): a
  candidate that would COMPLETE an already-seen n-gram is masked when the ban is
  enabled; unrelated candidates are untouched; ``n`` in {0, 1, 2} (and the unset
  knob) is an exact no-op; the history WINDOW rolls off; a row whose every
  candidate is banned falls back to its UNMASKED argmax; the CONTROL-TOKEN
  exemption (EOS + think start/end + ``_EXEMPT_IDS``) is subtracted AFTER the rule.
* the ENGINEERED ABSORBING LOOP: a synthetic greedy model keyed on a REAL
  ``"Let me write."`` token sequence (the golden 2026-10-17 degeneration dumps
  decoded under ``/Users/adam.durham/.hermes/cache/scratch/detector/``) loops
  forever with the ban OFF and ESCAPES with the ban ON (n=5) -- and the escape is
  exactly the ban firing (the banned set contains the would-be repeat token, and
  the argmax lands on a different candidate).
* DETERMINISM: the ban is a pure function of the produced-token history, so the
  same history yields the same mask and the same argmax -- which is what lets the
  two TP ``ranks`` agree with no collective (calling the helper twice stands in
  for the two ranks).

It also pins the VERIFY/ACCEPT path (a ``gamma > 0`` round with a head): the
target choice is the MASKED argmax, so a draft that agrees with the RAW target is
still REJECTED once the ban fires; the ban path insists on the FULL vocab row on
every rank (a shard fails loud) and computes its argmax with mx; the log-prob of a
committed row is read for the MASKED committed id; the per-turn telemetry counts
only changed committed rows and prints ONE stable INFO line, only when enabled;
and the SUGGESTED ``n`` is 6.

No GPU: the mock arrays are tiny (the ban scan is over id lists, the argmax over
one synthetic row), and ``conftest`` pins MLX to the CPU for the whole suite.
"""

# This is a WHITE-BOX test: it calls the decode site's private ``_one_round`` and
# the ban module's private array helpers, and it imports the liveness suite's
# logger recorder. None of that is a real defect, so the file-level rules are
# disabled rather than ignored line-by-line (same convention as
# ``test_dsv41_decode_liveness``).
# pyright: reportPrivateUsage=false, reportAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from exo.worker.engines.mlx.dsv41 import engine as eng_mod
from exo.worker.engines.mlx.dsv41 import no_repeat as nr
from exo.worker.engines.mlx.dsv41 import rounds as rounds_mod
from exo.worker.engines.mlx.dsv41.rounds import _one_round

from .test_dsv41_prefill_liveness import _capture

# --------------------------------------------------------------------------
# the real loop family, straight out of the golden incident data
# --------------------------------------------------------------------------
#: The 2026-10-17 degeneration dumps (on-node, last 2048 committed ids each),
#: decoded under this scratch dir. ``_GOLDEN_DUMP`` is the 074328 kill stream; the
#: test below ties ``SEED_IDS`` back to it whenever the golden data is present.
_DETECTOR_DIR = Path("/Users/adam.durham/.hermes/cache/scratch/detector")
_GOLDEN_DUMP = (
    _DETECTOR_DIR
    / "dumps"
    / "dsv41-degeneration-aborted-SystemExit-20261017-074328-92152.json"
)
_TOKENIZER_DIR = "/Users/adam.durham/.exo/models/deepseek-ai--DeepSeek-V4-Flash"

#: ``token_ids[60:100]`` of ``_GOLDEN_DUMP`` -- the REAL stream tail where the
#: "Let me write." family cycles (decodes to "...prefill degradation\n\nLet me
#: write reasonable... \n\nLet me write.\n\nLet me be honest and concrete.\n\n
#: Writing now...."). It is the seed the engineered attractor is keyed on.
SEED_IDS: list[int] = [
    2503, 113567, 5990, 201, 15, 852, 27472, 24262, 271,
    5718, 678, 5085, 13732, 2526, 4454, 538, 270, 11147, 9318, 940,
    3459, 7316, 339, 5718, 678, 5085, 339, 5718, 678, 366,
    14001, 305, 15360, 339, 46837, 1928, 16, 128822, 271, 30,
]

#: The real tokenization of ``"Let me write."`` under the checkpoint's
#: tokenizer (pinned by ``test_the_loop_block_is_the_real_tokenization``): the
#: four ids the absorbing attractor cycles.
BLOCK: list[int] = [5718, 678, 5085, 16]

#: A synthetic "healthy" continuation. Id 0 is deliberately absent from both
#: ``SEED_IDS`` and ``BLOCK``, so it can never sit inside an already-seen n-gram
#: and can never itself be the token the ban fires on -- the escape the tests
#: assert is therefore unambiguously the ban working, not a second-order effect.
HEALTHY = 0

#: The checkpoint's vocab size: big enough to hold every id the fixture uses.
VOCAB = 128_000

#: Rounds the simulation runs. The first re-entry of the cycle can only be caught
#: once the cycle has actually repeated (see the escaping test), so this must clear
#: two full passes of ``BLOCK`` past the 40-token seed.
STEPS = 32


# --------------------------------------------------------------------------
# (b) the engineered absorbing loop
# --------------------------------------------------------------------------


def _attractor_logits(history: list[int]) -> np.ndarray:
    """A synthetic greedy model ABSORBED by a real ``"Let me write."`` cycle.

    Keyed on the produced history: it puts logit 10.0 on the next token of the
    cycled block (indexed by the history LENGTH, so it is a pure function of the
    sequence and not of any hidden state) and 5.0 on the healthy alternative.
    Everything else is 0.0, so the argmax is the attractor unless the ban removes
    it -- exactly the absorbing-state shape the incident showed.
    """
    row = np.zeros(VOCAB, dtype=np.float32)
    row[BLOCK[len(history) % len(BLOCK)]] = 10.0
    row[HEALTHY] = 5.0
    return row.reshape(1, -1)


def _simulate(
    seed: list[int], *, n: int, window: int = 2048, steps: int = STEPS
) -> tuple[list[int], list[set[int]]]:
    """Greedy-decode the attractor for ``steps`` rounds, with the ban at ``n``.

    Returns the emitted ids and, per round, the banned set that was in force -- so
    a test can check BOTH what the argmax picked and why it picked it.
    """
    history = list(seed)
    emitted: list[int] = []
    bans: list[set[int]] = []
    for _ in range(steps):
        logits = _attractor_logits(history)
        banned = nr.banned_next_ids(history, n, window)
        ids, _masked = nr.argmax_after_ban(logits, [banned])
        token = int(ids[0])
        bans.append(banned)
        emitted.append(token)
        history.append(token)
    return emitted, bans


def test_with_the_ban_off_the_engineered_attractor_loops_forever() -> None:
    """Control: greedy argmax on the attractor never leaves the cycle.

    Every emitted id is a member of the real ``"Let me write."`` block and the run
    is exactly the block repeated -- the "stays looping" half of the fixture.
    """
    emitted, bans = _simulate(SEED_IDS, n=0)
    assert len(emitted) == STEPS
    assert set(emitted) <= set(BLOCK), emitted
    assert emitted[:16] == BLOCK * 4, emitted[:16]
    assert all(b == set() for b in bans), "the ban is OFF: nothing may be banned"


def test_with_the_ban_on_the_same_attractor_escapes() -> None:
    """The ban turns the absorbing state into an escape, at the exact round.

    With n=5, the token that would RE-ENTER the cycle is the one that closes a
    repeated n-gram, so it is masked and the argmax lands on the healthy
    alternative. The two runs diverge at one round, and that round is the ban
    firing -- not a coin flip.
    """
    off, _off_bans = _simulate(SEED_IDS, n=0)
    on, on_bans = _simulate(SEED_IDS, n=5)

    assert HEALTHY in on, "the ban must let the run leave the attractor"
    assert on != off, "the ban changes what the greedy argmax emits"

    first = next(i for i, (a, b) in enumerate(zip(off, on)) if a != b)
    # The would-be repeat token is what the ban masks...
    assert off[first] == BLOCK[0]
    assert off[first] in on_bans[first], on_bans[first]
    # ...and the argmax then picks a different candidate.
    assert on[first] != off[first]
    assert on[first] == HEALTHY
    # Two full passes of the block were needed before the cycle was "already
    # seen": the FIRST pass is legitimate work, the second one is the loop.
    assert first == len(BLOCK) * 2, first


def test_the_escape_needs_the_history_window_to_still_cover_the_repeat() -> None:
    """The window knob is load-bearing: roll it past the earlier occurrence and
    the same attractor stays absorbed (nothing about the ban is implicit)."""
    wide, _ = _simulate(SEED_IDS, n=5, window=2048)
    tight, _ = _simulate(SEED_IDS, n=5, window=4)
    assert HEALTHY in wide
    assert set(tight) <= set(BLOCK), tight
    assert tight[:16] == BLOCK * 4


# --------------------------------------------------------------------------
# (b) the fixture is the REAL golden data, not an invention
# --------------------------------------------------------------------------


@pytest.mark.skipif(not _GOLDEN_DUMP.exists(), reason="golden incident dump absent")
def test_the_embedded_seed_is_the_golden_dump_slice() -> None:
    ids = json.loads(_GOLDEN_DUMP.read_text())["token_ids"]
    assert ids[60:100] == SEED_IDS


def _real_tokenizer() -> Any:
    if not Path(_TOKENIZER_DIR).exists():
        pytest.skip(f"checkpoint tokenizer absent: {_TOKENIZER_DIR}")
    try:
        from transformers import AutoTokenizer
    except Exception:  # noqa: BLE001 - transformers is a dev-only extra here
        pytest.skip("transformers not importable")
        return None
    return AutoTokenizer.from_pretrained(_TOKENIZER_DIR)


def test_the_loop_block_is_the_real_tokenization() -> None:
    tokenizer = _real_tokenizer()
    assert tokenizer.encode("Let me write.", add_special_tokens=False) == BLOCK


# --------------------------------------------------------------------------
# (a) the pure ban: mask the repeat, leave everything else alone
# --------------------------------------------------------------------------


def test_a_candidate_completing_a_seen_n_gram_is_masked() -> None:
    """The whole point: a token that would close a repeated n-gram gets -inf, and
    the argmax moves off it."""
    history = list(SEED_IDS) + list(BLOCK) + list(BLOCK)
    banned = nr.banned_next_ids(history, 5, 2048)
    assert BLOCK[0] in banned, banned

    logits = _attractor_logits(history)
    ids, masked = nr.argmax_after_ban(logits, [banned])
    assert float(masked[0, BLOCK[0]]) == nr.NEG_INF
    assert ids[0] == HEALTHY


def test_unrelated_candidates_are_untouched() -> None:
    logits = np.zeros((1, 8), dtype=np.float32)
    logits[0, 3] = 9.0  # the would-be repeat
    logits[0, 5] = 8.0  # an unrelated runner-up
    ids, masked = nr.argmax_after_ban(logits, [{3}])
    assert float(masked[0, 3]) == nr.NEG_INF
    assert float(masked[0, 5]) == np.float32(8.0), "unrelated ids must not move"
    assert ids == [5], "the argmax takes the best UNBANNED candidate"


@pytest.mark.parametrize("n", [0, 1, 2])
def test_a_small_or_absent_n_is_an_exact_noop(n: int) -> None:
    """n in {0,1,2} is DISABLED by contract: the ban set is empty, the policy is
    not enabled, and the argmax is the plain unmasked one."""
    history = list(SEED_IDS) + list(BLOCK) * 3
    assert nr.banned_next_ids(history, n, 2048) == set()
    assert nr.NoRepeatNgram(n=n, window=2048).enabled is False

    logits = _attractor_logits(history)
    ids, _masked = nr.argmax_after_ban(logits, [set()])
    assert ids[0] == BLOCK[len(history) % len(BLOCK)]


def test_the_knob_defaults_to_off() -> None:
    ban = nr.NoRepeatNgram()
    assert ban.n == 0 and ban.window == 0
    assert ban.enabled is False
    assert ban.banned(list(SEED_IDS) + list(BLOCK) * 3) == set()
    assert ban.banned_per_row([1, 2, 3], [4, 5]) == [set(), set(), set()]


@pytest.mark.parametrize("n", [3, 4, 5])
def test_an_enabled_n_masks_the_repeating_token(n: int) -> None:
    """Above MIN_NGRAM the ban is live and the repeated-n-gram rule is exact.

    Only n up to len(BLOCK)+1 is exercised here: beyond that the (n-1)-token
    suffix spans more than one repetition of the block, which the two-block
    history does not contain twice yet (that is the two-pass effect the
    absorption test pins).
    """
    # history ends with the block twice: the (n-1)-suffix is inside the second
    # block, and its earlier occurrence (the first block) closes on BLOCK[0].
    history = list(SEED_IDS) + list(BLOCK) + list(BLOCK)
    assert nr.banned_next_ids(history, n, 2048) == {BLOCK[0]}
    assert nr.NoRepeatNgram(n=n, window=2048).enabled is True


def test_the_history_window_rolls_off() -> None:
    """An earlier occurrence stops counting once it falls outside the window."""
    # The 2-gram (5, 6) sits at index 0 (followed by 99) and at the tail. A window
    # that still covers index 0 bans 99; one that has rolled past it bans nothing.
    history = [5, 6, 99, 11, 12, 13, 5, 6]
    assert nr.banned_next_ids(history, 3, 2048) == {99}
    assert nr.banned_next_ids(history, 3, 8) == {99}, "window covers the occurrence"
    assert nr.banned_next_ids(history, 3, 7) == set(), "window has rolled off it"
    assert nr.banned_next_ids(history, 3, 4) == set()


def test_a_fully_banned_row_falls_back_to_the_unmasked_argmax() -> None:
    """Never emit an invalid token: if every candidate is banned, use the plain
    argmax for that row (and only that row)."""
    logits = np.array([[1.0, 2.0, 3.0], [1.0, 5.0, 2.0]], dtype=np.float32)
    ids, masked = nr.argmax_after_ban(logits, [{0, 1, 2}, {0, 2}])
    assert float(masked[0].max()) == nr.NEG_INF
    assert ids[0] == 2, "row 0 is fully banned -> unmasked argmax"
    assert ids[1] == 1, "row 1 is partially banned -> masked argmax"
    assert float(masked[1, 0]) == nr.NEG_INF
    assert float(masked[1, 2]) == nr.NEG_INF


def test_mask_rows_never_aliases_the_input() -> None:
    """The masked copy is what the model's own logits must NOT be mutated into --
    a second row (or the other TP rank) may still need the raw values."""
    logits = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    masked = nr.mask_rows(logits, [{1}])
    assert float(logits[0, 1]) == 2.0
    assert float(masked[0, 1]) == nr.NEG_INF


def test_banned_per_row_extends_each_rows_context_with_the_draft() -> None:
    """Row 0 predicts after the produced history; row i after the i-th draft."""
    ban = nr.NoRepeatNgram(n=3, window=2048)
    history = [5, 6]
    draft = [5, 6, 7]
    rows = ban.banned_per_row(history, draft)
    assert len(rows) == len(draft) + 1
    assert rows[0] == nr.banned_next_ids(history, 3, 2048)
    assert rows[2] == nr.banned_next_ids(history + draft[:2], 3, 2048)
    assert rows[3] == nr.banned_next_ids(history + draft, 3, 2048)


# --------------------------------------------------------------------------
# (c) determinism: the same history yields the same mask on both "ranks"
# --------------------------------------------------------------------------


def test_the_ban_is_deterministic() -> None:
    history = list(SEED_IDS) + list(BLOCK) * 3
    rank0 = nr.banned_next_ids(history, 5, 2048)
    rank1 = nr.banned_next_ids(history, 5, 2048)
    assert rank0 == rank1

    logits = _attractor_logits(history)
    ids0, masked0 = nr.argmax_after_ban(logits, [rank0])
    ids1, masked1 = nr.argmax_after_ban(logits, [rank1])
    assert ids0 == ids1
    np.testing.assert_array_equal(np.asarray(masked0), np.asarray(masked1))


# --------------------------------------------------------------------------
# the decode site: the ban is applied to the greedy argmax, and OFF is identical
# --------------------------------------------------------------------------


def _greedy_round(
    ban: nr.NoRepeatNgram | None,
    history: list[int],
    stats: nr.BanTurnStats | None = None,
) -> list[int]:
    """Drive the real ``_one_round`` down its plain-greedy path with a fake model.

    ``head=None`` picks the single-row greedy forward (one argmax over a whole
    vocab row); the fake model ignores the ids and returns the attractor row, so
    the round's committed token is exactly the argmax the decoder took. The cache
    is a bare namespace: ``_ensure_capacity_for_round`` no-ops without an
    ``ensure_capacity`` attribute, which is the documented double behaviour.
    ``stats`` is forwarded so a test can watch the turn telemetry.
    """
    import mlx.core as mx

    logits = _attractor_logits(history)

    class _Model:
        def __call__(self, ids: Any, cache: Any, **kwargs: Any) -> Any:
            del ids, cache, kwargs
            return mx.array(logits)

    committed, _ms, _accepted, _gamma = _one_round(
        SimpleNamespace(_draft_windows={}),  # type: ignore[arg-type]  # a fake engine
        model=_Model(),
        cache=SimpleNamespace(),
        token=history[-1],
        head=None,
        policy=None,
        ban=ban,
        history=history,
        ban_stats=stats,
    )
    return [int(t) for t in committed]


def test_one_round_applies_the_ban_before_the_greedy_argmax() -> None:
    history = list(SEED_IDS) + list(BLOCK) + list(BLOCK)
    off = _greedy_round(None, history)
    on = _greedy_round(nr.NoRepeatNgram(n=5, window=2048), history)
    assert off == [BLOCK[0]], "unbanned: the greedy argmax re-enters the loop"
    assert on == [HEALTHY], "banned: the round commits the escape token instead"


def test_one_round_is_byte_identical_when_the_ban_is_off() -> None:
    """The DEFAULT contract: ban None and an explicitly-disabled policy produce
    the same committed token as each other -- and as the pre-ban expression."""
    import mlx.core as mx

    history = list(SEED_IDS) + list(BLOCK) + list(BLOCK)
    assert _greedy_round(None, history) == _greedy_round(
        nr.NoRepeatNgram(n=0, window=2048), history
    )
    logits = mx.array(_attractor_logits(history))
    assert _greedy_round(None, history) == [
        int(mx.argmax(logits.reshape(-1), axis=-1).item())
    ]


# --------------------------------------------------------------------------
# the engine knob: read at resolve time, DEFAULT OFF, forwarded by the launcher
# --------------------------------------------------------------------------


def test_the_knob_names_and_defaults_are_the_documented_ones() -> None:
    assert eng_mod.NO_REPEAT_NGRAM_ENV == "EXO_DSV41_NO_REPEAT_NGRAM"
    assert eng_mod.NO_REPEAT_NGRAM_DEFAULT == 0
    assert eng_mod.NO_REPEAT_NGRAM_WINDOW_ENV == "EXO_DSV41_NO_REPEAT_NGRAM_WINDOW"
    assert eng_mod.NO_REPEAT_NGRAM_WINDOW_DEFAULT == 2048
    assert nr.MIN_NGRAM == 3


def test_the_resolved_knob_is_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(eng_mod.NO_REPEAT_NGRAM_ENV, raising=False)
    monkeypatch.delenv(eng_mod.NO_REPEAT_NGRAM_WINDOW_ENV, raising=False)
    ban = eng_mod.resolve_no_repeat_ngram()
    assert ban.n == 0
    assert ban.window == eng_mod.NO_REPEAT_NGRAM_WINDOW_DEFAULT
    assert ban.enabled is False


def test_the_resolved_knob_reads_both_env_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(eng_mod.NO_REPEAT_NGRAM_ENV, "5")
    monkeypatch.setenv(eng_mod.NO_REPEAT_NGRAM_WINDOW_ENV, "64")
    ban = eng_mod.resolve_no_repeat_ngram()
    assert ban.n == 5 and ban.window == 64
    assert ban.enabled is True
    # A value below MIN_NGRAM is accepted (it is the documented no-op) but inert.
    monkeypatch.setenv(eng_mod.NO_REPEAT_NGRAM_ENV, "2")
    assert eng_mod.resolve_no_repeat_ngram().enabled is False


def test_a_malformed_knob_warns_and_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A launcher typo degrades to the disabled default, loudly, never crashes --
    and never turns the ban ON by accident."""
    monkeypatch.setenv(eng_mod.NO_REPEAT_NGRAM_ENV, "five")
    with _capture(eng_mod) as cap:
        ban = eng_mod.resolve_no_repeat_ngram()
    assert ban.n == eng_mod.NO_REPEAT_NGRAM_DEFAULT
    assert ban.enabled is False
    assert any("non-integer" in m for m in cap.of("WARNING")), cap.messages


# --------------------------------------------------------------------------
# (d) the VERIFY/ACCEPT path with a head: masked argmax, full-vocab row, mx
# --------------------------------------------------------------------------


class _FixedPolicy:
    """A gamma policy double: always returns the same width."""

    def __init__(self, gamma: int) -> None:
        self._gamma = int(gamma)

    def next(self) -> int:
        return self._gamma


class _VerifyHead:
    """A DSpark head double: drafts a FIXED script, ignores the context."""

    def __init__(self, script: list[int]) -> None:
        self._script = [int(t) for t in script]
        self.appended: list[int] = []

    def make_cache(self, bsz: int = 1) -> Any:
        del bsz
        return SimpleNamespace()

    def append_ctx(self, hidden: Any, caches: Any) -> None:
        del caches
        self.appended.append(int(hidden.shape[1]))

    def draft(
        self, anchor: Any, embed: Any, head: Any, caches: Any, *, width: int
    ) -> Any:
        del anchor, embed, head, caches
        return mx.array([self._script[: int(width)]], dtype=mx.int32), None


class _VerifyModel:
    """A body double returning preset RAW logits rows, advancing like the body.

    ``argmax=True`` mirrors the real head (``mx.argmax`` ids); ``argmax=False``
    returns the full-vocab rows the ban needs. ``shard=True`` returns a NARROWER
    row (half the vocab) so a test can prove the ban path fails loud.
    """

    def __init__(self, rows: list[np.ndarray], *, shard: bool = False) -> None:
        self.args = SimpleNamespace(dspark_target_layer_ids=(0,), vocab_size=VOCAB)
        self.embed: Any = None
        self.head: Any = None
        self._rows = rows
        self._shard = shard
        self.argmax_calls: list[bool] = []

    def __call__(
        self,
        ids: Any,
        cache: Any,
        last_logit_only: bool = False,
        return_taps: bool = False,
        argmax: bool = False,
        logprobs: int = 0,
    ) -> Any:
        del last_logit_only, logprobs
        n = int(ids.shape[1])
        start = int(cache.offset)
        cache.offset = start + n
        self.argmax_calls.append(bool(argmax))
        rows = [
            (r[: VOCAB // 2] if self._shard else r)
            for r in self._rows[start:start + n]
        ]
        logits = mx.array(np.stack(rows).reshape(1, n, -1).astype(np.float32))
        taps = {0: mx.zeros((1, n, 4))}
        if return_taps:
            if argmax:
                return mx.argmax(logits, axis=-1).astype(mx.int32), taps
            return logits, taps
        if argmax:
            return mx.argmax(logits, axis=-1).astype(mx.int32)
        return logits


class _RingLayer:
    """The one cache layer the spec rollback helpers touch (no compressor)."""

    def __init__(self, window: int) -> None:
        self.window = window
        self.win_kv = mx.zeros((1, window))
        self.comp_state = None
        self.comp_kv = None
        self.index_k = None
        self.ratio = 0

    def window_write_slots(self, start: int, rows: int) -> Any:
        return mx.arange(int(start), int(start) + int(rows)) % self.window


class _RingCache:
    """A minimal ring cache: enough for ``spec.snap``/``stashes``/``rollback``."""

    def __init__(self, window: int = 8) -> None:
        self.offset = 0
        self.layers = [_RingLayer(window)]


def _verify_round(
    ban: nr.NoRepeatNgram | None,
    history: list[int],
    *,
    gamma: int = 1,
    logprobs: int = 0,
    lp_out: list[Any] | None = None,
    stats: nr.BanTurnStats | None = None,
    shard: bool = False,
) -> tuple[list[int], int, int]:
    """One REAL ``_one_round`` down the DSpark verify path, with a fake head.

    The draft is exactly ``BLOCK[0]`` (the loop token) and row 0's RAW argmax is
    ALSO ``BLOCK[0]``, so with the ban OFF the draft is fully accepted (acceptance
    1.0) and the loop token is committed VIA acceptance. Returns
    ``(committed, accepted, gamma)``.
    """
    draft = [BLOCK[0]] * int(gamma)
    rows = [_attractor_logits(history)[0]]
    for k in range(1, int(gamma) + 1):
        rows.append(_attractor_logits(history + draft[:k])[0])
    committed, _ms, accepted, g = _one_round(
        SimpleNamespace(_draft_windows={}),  # type: ignore[arg-type]  # a fake engine
        model=_VerifyModel(rows, shard=shard),
        cache=_RingCache(),
        token=history[-1],
        head=_VerifyHead(draft),
        policy=_FixedPolicy(gamma),
        draft_state=SimpleNamespace(),
        logprobs=logprobs,
        lp_out=lp_out,
        ban=ban,
        history=history,
        ban_stats=stats,
    )
    return [int(t) for t in committed], int(accepted), int(g)


@pytest.mark.parametrize("n, blocks", [(5, 2), (nr.SUGGESTED_N, 3)])
def test_a_verify_round_accepts_the_loop_draft_when_off_and_rejects_it_when_on(
    n: int, blocks: int
) -> None:
    """The absorbing-loop shape at gamma>0, exactly as the review demands.

    Ran at the SUGGESTED ``n`` as well as at 5, so the config operators are told to
    use is actually exercised. The draft AGREES with the RAW target (acceptance is
    1.0 with the ban OFF and the loop token is committed VIA acceptance). With the
    ban ON the MASKED argmax differs, the row rejects, and the committed token
    escapes -- a raw or sharded argmax in the accept comparison would have left the
    ban silent.
    """
    history = list(SEED_IDS) + list(BLOCK) * blocks
    draft = [BLOCK[0]]

    off, off_accepted, gamma = _verify_round(None, history)
    assert gamma == 1
    assert off_accepted == gamma, "OFF: the draft agrees with the raw target (1.0)"
    assert off[0] == draft[0] == BLOCK[0], "OFF: loop token committed VIA acceptance"

    ban = nr.NoRepeatNgram(n=n, window=2048)
    on, on_accepted, _ = _verify_round(ban, history)
    # ...the MASKED argmax differs from the raw target, so the row REJECTS...
    bans = ban.banned_per_row(history, draft)
    assert BLOCK[0] in bans[0], bans[0]
    masked_ids, _masked = nr.argmax_after_ban(
        mx.array(_attractor_logits(history)), [bans[0]]
    )
    assert masked_ids[0] == HEALTHY
    assert masked_ids[0] != draft[0]
    assert on_accepted == 0, "ON: the row must reject"
    # ...and the committed token ESCAPES the loop.
    assert on == [HEALTHY]
    assert on != off


def test_the_ban_path_reports_the_logprob_of_the_masked_committed_id() -> None:
    """(d): the log-prob read must be for the MASKED committed id, not the raw
    argmax the ban moved off."""
    history = list(SEED_IDS) + list(BLOCK) + list(BLOCK)
    lp_out: list[Any] = []
    committed, accepted, _ = _verify_round(
        nr.NoRepeatNgram(n=5, window=2048), history, logprobs=3, lp_out=lp_out
    )
    assert committed == [HEALTHY] and accepted == 0
    assert len(lp_out) == 1, lp_out
    sel, ids, _vals = lp_out[0]
    assert ids[0] == HEALTHY, "the read must be for the MASKED committed id"
    assert ids[0] != BLOCK[0], "it must NOT be the raw argmax the ban moved off"
    # The masked row read must yield a FINITE log-prob (not -inf/NaN): that is
    # the failure mode if the read ran before the all-banned fallback decision.
    assert math.isfinite(float(sel)), sel


def test_the_ban_path_refuses_a_sharded_verify_row() -> None:
    """The ban needs the FULL vocab row on every rank: a shard fails loud."""
    history = list(SEED_IDS) + list(BLOCK) + list(BLOCK)
    with pytest.raises(RuntimeError, match="FULL vocab row"):
        _verify_round(nr.NoRepeatNgram(n=5, window=2048), history, shard=True)


def test_the_ban_path_row_shape_must_be_the_full_vocab() -> None:
    """``row.shape[-1] == vocab`` on the ban path; a shard raises, loudly."""
    model = SimpleNamespace(args=SimpleNamespace(vocab_size=VOCAB))
    full = mx.zeros((2, VOCAB))
    rounds_mod._assert_full_vocab_row(full, model)  # full vocab: no raise
    shard = mx.zeros((2, VOCAB // 2))
    with pytest.raises(RuntimeError, match="FULL vocab row"):
        rounds_mod._assert_full_vocab_row(shard, model)


def test_the_production_argmax_uses_mx_on_both_ranks() -> None:
    """Production argmax is mx on BOTH ranks; duck-typing is a test affordance."""
    assert nr._maybe_mlx() is mx
    history = list(SEED_IDS) + list(BLOCK) + list(BLOCK)
    logits = mx.array(_attractor_logits(history))
    banned = nr.banned_next_ids(history, 5, 2048)
    masked = nr.mask_rows(logits, [banned])
    for _rank in range(2):  # the two TP ranks
        picked = nr._argmax_last(masked)
        assert isinstance(picked, mx.array), "the argmax must be mx, not duck-typed"
    ids0, _m0 = nr.argmax_after_ban(logits, [banned])
    ids1, _m1 = nr.argmax_after_ban(logits, [banned])
    assert ids0 == ids1 == [HEALTHY]


# --------------------------------------------------------------------------
# (e) the control-token exemption: EOS + think delimiters are never banned
# --------------------------------------------------------------------------


class _ExemptTokenizer:
    """A tokenizer double exposing only what the exemption resolver reads."""

    def __init__(
        self,
        eos: tuple[int, ...] = (1, 2),
        think_start: int | None = 128821,
        think_end: int | None = 128822,
    ) -> None:
        self.eos_token_ids = list(eos)
        self.think_start_id = think_start
        self.think_end_id = think_end


def test_control_token_ids_are_never_banned() -> None:
    """EOS is subtracted from a ban the n-gram rule WOULD impose on it."""
    history = [7, 8, 1, 9, 10, 11, 7, 8]  # the raw 3-gram rule forbids id 1 (EOS)
    assert nr.banned_next_ids(history, 3, 2048) == {1}
    exempt = frozenset({1, 128821, 128822})
    assert nr.banned_next_ids(history, 3, 2048, exempt) == set()
    ban = nr.NoRepeatNgram(n=3, window=2048).with_exempt(exempt)
    assert ban.banned(history) == set()
    assert ban.banned_per_row(history, []) == [set()]


def test_a_think_delimiter_is_never_banned() -> None:
    history = [7, 8, 128822, 9, 10, 11, 7, 8]  # would close on the think-END id
    assert nr.banned_next_ids(history, 3, 2048) == {128822}
    ban = nr.NoRepeatNgram(n=3, window=2048).with_exempt({128822})
    assert 128822 not in ban.banned(history)


def test_the_default_exempt_set_is_the_eos_and_think_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(eng_mod.NO_REPEAT_NGRAM_EXEMPT_ENV, raising=False)
    tokenizer = _ExemptTokenizer()
    assert eng_mod.resolve_no_repeat_ngram_exempt_ids(tokenizer) == frozenset(
        {1, 2, 128821, 128822}
    )
    # ...and with no tokenizer only the (absent) env contributes.
    assert eng_mod.resolve_no_repeat_ngram_exempt_ids(None) == frozenset()


def test_the_exempt_env_extends_the_default_set(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = _ExemptTokenizer()
    monkeypatch.setenv(eng_mod.NO_REPEAT_NGRAM_EXEMPT_ENV, "4242, 4343")
    got = eng_mod.resolve_no_repeat_ngram_exempt_ids(tokenizer)
    assert {1, 2, 128821, 128822, 4242, 4343} <= got

    monkeypatch.setenv(eng_mod.NO_REPEAT_NGRAM_EXEMPT_ENV, "4242, oops")
    with _capture(eng_mod) as cap:
        got = eng_mod.resolve_no_repeat_ngram_exempt_ids(tokenizer)
    assert 4242 in got
    assert any("non-integer" in m for m in cap.of("WARNING")), cap.messages


def test_the_resolved_ban_installs_the_exempt_set_only_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = _ExemptTokenizer()
    monkeypatch.setenv(eng_mod.NO_REPEAT_NGRAM_ENV, "6")
    assert eng_mod.resolve_no_repeat_ngram(tokenizer).exempt == frozenset(
        {1, 2, 128821, 128822}
    )
    monkeypatch.setenv(eng_mod.NO_REPEAT_NGRAM_ENV, "0")
    assert eng_mod.resolve_no_repeat_ngram(tokenizer).exempt == frozenset()


# --------------------------------------------------------------------------
# (f) telemetry: only enabled, only on a real argmax change, ONE stable line
# --------------------------------------------------------------------------


def test_ban_counters_track_changed_rows_and_masked_ids() -> None:
    logits = np.zeros((2, 8), dtype=np.float32)
    logits[0, 3] = 9.0
    logits[0, 5] = 8.0
    logits[1, 0] = 4.0
    ids, _masked, changed, counts = nr.argmax_after_ban_detailed(
        mx.array(logits), [{3}, {1}]
    )
    assert ids == [5, 0]
    assert changed == [True, False], "only row 0's argmax actually moved"
    assert counts == [1, 1]


def test_the_telemetry_counts_only_changed_committed_rows() -> None:
    stats = nr.BanTurnStats()
    history = list(SEED_IDS) + list(BLOCK) + list(BLOCK)
    committed, accepted, _ = _verify_round(
        nr.NoRepeatNgram(n=5, window=2048), history, stats=stats
    )
    assert accepted == 0 and committed == [HEALTHY]
    assert stats.rows_changed == 1, "the ban moved the committed row's argmax"
    assert stats.masked_total == 1, "row 0 masks exactly the loop token"


def test_the_telemetry_is_zero_when_the_knob_is_off() -> None:
    stats = nr.BanTurnStats()
    history = list(SEED_IDS) + list(BLOCK) + list(BLOCK)
    _greedy_round(nr.NoRepeatNgram(n=0, window=2048), history, stats=stats)
    assert stats.rows_changed == 0 and stats.masked_total == 0


def test_the_telemetry_counts_a_single_row_greedy_ban_too() -> None:
    stats = nr.BanTurnStats()
    history = list(SEED_IDS) + list(BLOCK) + list(BLOCK)
    assert _greedy_round(nr.NoRepeatNgram(n=5, window=2048), history, stats=stats) == [
        HEALTHY
    ]
    assert stats.rows_changed == 1 and stats.masked_total == 1


def test_the_telemetry_line_format_is_stable() -> None:
    ban = nr.NoRepeatNgram(n=6, window=2048)
    line = nr.format_ban_line(ban, nr.BanTurnStats(rows_changed=3, masked_total=7))
    assert line == (
        "[DSV41] no-repeat-ngram n=6 window=2048 rows_changed=3 masked_total=7"
    )


def test_rounds_emits_one_telemetry_line_only_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ONE INFO line at turn end when ON; NOTHING when OFF (byte-identical)."""
    engine = SimpleNamespace(
        loaded=SimpleNamespace(head=None, tokenizer=None), speculative=False
    )
    monkeypatch.setenv(eng_mod.NO_REPEAT_NGRAM_ENV, "6")
    monkeypatch.setenv(eng_mod.NO_REPEAT_NGRAM_WINDOW_ENV, "2048")
    with _capture(eng_mod) as cap:
        assert list(
            eng_mod.Dsv41Engine._rounds(engine, None, 0, 0)  # type: ignore[arg-type]
        ) == []
    assert cap.of("INFO") == [
        "[DSV41] no-repeat-ngram n=6 window=2048 rows_changed=0 masked_total=0"
    ]

    monkeypatch.delenv(eng_mod.NO_REPEAT_NGRAM_ENV, raising=False)
    with _capture(eng_mod) as off_cap:
        list(eng_mod.Dsv41Engine._rounds(engine, None, 0, 0))  # type: ignore[arg-type]
    assert off_cap.of("INFO") == [], "OFF: the ban never logs"


# --------------------------------------------------------------------------
# (g) the suggested value: n=6
# --------------------------------------------------------------------------


def test_the_suggested_n_is_six() -> None:
    assert nr.SUGGESTED_N == 6
    assert nr.SUGGESTED_N > nr.MIN_NGRAM
