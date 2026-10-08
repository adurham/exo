# pyright: reportPrivateUsage=false, reportAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportUnusedFunction=false, reportMissingModuleSource=false, reportOptionalMemberAccess=false, reportMissingTypeArgument=false
"""Scoped tests for the opt-in per-round PHASE TIMER (``round_prof`` 1/2).

These pin the timer that fills the ``round_prof`` hook the plumbing change
wired in: the JSONL line per round, the two modes (host-wallclock vs
eval-fenced), and -- load bearing -- that PROF=0/unset stays byte-identical
and that a broken/undwritable dump path can NEVER kill a round.

Mocks (a fixed script + a scripted draft head) mirror ``test_dsv41_engine.py``
so the round's accept/rollback contract is exercised through the REAL
``mlx_lm.models.deepseek_v41.spec`` helpers, and the timer is exercised at its
real call sites in ``rounds._one_round``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import pytest

from exo.worker.engines.mlx.dsv41 import rounds as rounds_mod
from exo.worker.engines.mlx.dsv41.rounds import _one_round, _spec_policy

# A round's JSONL line must carry every bracket the design names, plus the
# round identifier and the rank tag.
_REQUIRED_KEYS = (
    "round_idx",
    "gamma",
    "n_accepted",
    "draft_build_ms",
    "verify_block_ms",
    "tail_bookkeep_ms",
    "round_total_ms",
    "emit_ms",
    "rank",
)


# --------------------------------------------------------------------- mocks


class _Args:
    dspark_target_layer_ids: tuple[int, ...] = (0,)
    max_seq_len: int = 4096


class _CompState:
    """A layer's compressor carry; ``rollback``'s force target in PROF=2."""

    def __init__(self) -> None:
        self.kv_state = mx.zeros((1, 2, 4))
        self.score_state = mx.zeros((1, 2, 4))
        self.chunk_kv: Any = None
        self.chunk_score: Any = None


class _Layer:
    def __init__(self) -> None:
        self.ratio = 2
        self.comp_state = _CompState()


class _Win:
    """A draft stage's window ring; ``append_ctx``'s force target in PROF=2."""

    def __init__(self) -> None:
        self.win_kv = mx.zeros((1, 8, 4))
        self.n_ctx = 0


class _MockCache:
    def __init__(self, max_seq_len: int = 64, *, with_layers: bool = False) -> None:
        self.max_seq_len = max_seq_len
        self.offset = 0
        self.capacity = max_seq_len
        # Non-empty so PROF=2's rollback eval has a destination to force.
        self.layers: list[Any] = [_Layer()] if with_layers else []

    def ensure_capacity(self, required_tokens: int) -> None:
        if int(required_tokens) > self.max_seq_len:
            raise ValueError(f"capacity {self.max_seq_len} < {required_tokens}")


class _MockModel:
    """Scripted body: answers from a pre-built table, one row per position."""

    def __init__(self, script: list[int]) -> None:
        self.script = list(script)
        self.prompt_tokens = 0
        self.embed = object()
        self.head = object()
        self.args = _Args()
        self.calls: list[tuple[int, int]] = []
        #: Every forward that took the ``argmax=True`` (verify) path.
        self.argmax_calls: list[list[int]] = []
        #: Offset the round started from; the draft answers from here.
        self.round_entry_offset = 0

    def make_cache(self, bsz: int = 1, max_seq_len: int | None = None, **_: Any):
        return _MockCache(max_seq_len or 64)

    def __call__(
        self,
        input_ids: mx.array,
        cache: _MockCache,
        last_logit_only: bool = False,
        return_taps: bool = False,
        argmax: bool = False,
        logprobs: int = 0,
    ):
        del last_logit_only
        rows, fed = int(input_ids.shape[0]), int(input_ids.shape[-1])
        cache.offset += fed
        self.calls.append((rows, fed))
        if argmax:
            ids = self._ids_for(input_ids, cache.offset)
            self.argmax_calls.append(ids)
            out = mx.array([ids], dtype=mx.int32)
            if logprobs:
                k = int(logprobs)
                lp = {
                    "selected": mx.zeros((1, fed)),
                    "top_ids": mx.array([[t] * k for t in ids], dtype=mx.int32),
                    "top_logprobs": mx.zeros((1, fed, k)),
                }
                return (out, self._taps(fed), lp) if return_taps else (out, lp)
            return (out, self._taps(fed)) if return_taps else out
        logits = _one_hot(self.script[self._index(cache.offset)])
        return (logits, self._taps(1)) if return_taps else logits

    def _index(self, offset: int) -> int:
        return max(0, min(offset - self.prompt_tokens, len(self.script) - 1))

    def _ids_for(self, input_ids: mx.array, offset: int) -> list[int]:
        rows = int(input_ids.shape[-1])
        first = offset - rows
        return [self.script[self._index(first + row + 1)] for row in range(rows)]

    def _taps(self, rows: int) -> dict[int, mx.array]:
        return {0: mx.zeros((1, rows, 4), dtype=mx.float32)}


class _MockHead:
    """Drafts the script's next tokens (never lies unless asked)."""

    def __init__(self, model: _MockModel) -> None:
        self.model = model
        self.appended: list[int] = []

    def make_cache(self, bsz: int = 1) -> list[_Win]:
        return [_Win() for _ in range(3)]

    def append_ctx(self, taps: mx.array, dsc: Any) -> None:
        self.appended.append(int(taps.shape[1]))
        for w in dsc:
            w.n_ctx += int(taps.shape[1])

    def draft(
        self, anchor: mx.array, embed: Any, head_lin: Any, dsc: Any, *, width: int
    ):
        del anchor, embed, head_lin, dsc
        start = self.model._index(self.model.round_entry_offset)
        ids = list(self.model.script[start : start + width])
        return mx.array([ids], dtype=mx.int32), mx.ones((1, len(ids)))


class _Engine:
    """Bare engine the round helpers touch (``_draft_windows`` + a rank).

    ``device_rank`` is what ``_round_prof_hook_for`` reads to fill the
    ``<rank>`` placeholder in the dump path.
    """

    def __init__(
        self, draft_windows: dict[int, Any] | None = None, device_rank: int = 0
    ) -> None:
        self._draft_windows = draft_windows if draft_windows is not None else {}
        self.device_rank = device_rank
        #: Attached lazily by ``rounds._round_prof_hook_for`` on the first
        #: profiled round; declared here so the test can reach it typed.
        self._round_prof: Any = None


def _one_hot(token: int) -> mx.array:
    row = [0.0] * (token + 1)
    row[token] = 1.0
    return mx.array([row])


# ---------------------------------------------------------------- fixtures


def _spec_setup(*, with_layers: bool = True):
    """A fresh (model, head, cache, engine) primed for a speculative round."""
    model = _MockModel([34, 35, 36, 34, 35, 1])
    head = _MockHead(model)
    cache = _MockCache(64, with_layers=with_layers)
    engine = _Engine(draft_windows={0: head.make_cache()})
    model.round_entry_offset = 1  # the round feeds the anchor as its first row
    return model, head, cache, engine


def _run(
    engine: _Engine, model: _MockModel, cache: _MockCache, head: Any, *, round_prof: int
):
    return _one_round(
        engine,  # type: ignore[arg-type]
        model=model,
        cache=cache,
        token=model.script[0],
        head=head,
        policy=_spec_policy(3),
        round_prof=round_prof,
    )


# -------------------------------------- (1) PROF unset vs =1 identical out


def test_prof_unset_and_one_produce_identical_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The timer must not change a single committed token.

    PROF=0 (off) and PROF=1 (host-wallclock) run the SAME round on a fixed
    script and must agree on the committed tokens, the acceptance count, the
    gamma, the cache position AND the exact forwards fed -- the timer only
    reads the clock at sync points the round already has.
    """
    monkeypatch.setenv("EXO_DSV41_ROUND_PROF_PATH", str(tmp_path / "off.jsonl"))

    m0, h0, c0, e0 = _spec_setup()
    out_off = _run(e0, m0, c0, h0, round_prof=0)

    m1, h1, c1, e1 = _spec_setup()
    out_on = _run(e1, m1, c1, h1, round_prof=1)

    assert out_off[0] == out_on[0], "committed tokens diverged"
    assert out_off[2:] == out_on[2:], "accepted/gamma diverged"
    assert c0.offset == c1.offset, "cache position diverged"
    assert m0.calls == m1.calls, "forward shapes diverged"
    assert m0.argmax_calls == m1.argmax_calls


def test_prof_one_writes_jsonl_with_required_keys_and_sane_ms(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """PROF=1 emits one parseable JSON line per round with every bracket."""
    path = tmp_path / "prof.jsonl"
    monkeypatch.setenv("EXO_DSV41_ROUND_PROF_PATH", str(path))

    model, head, cache, engine = _spec_setup()
    out = _run(engine, model, cache, head, round_prof=1)
    # ``_rounds`` (the engine) is what flushes; emulate that one call here so
    # the line is completed exactly as production does.
    prof = engine._round_prof  # noqa: SLF001 - the test owns this engine
    prof.flush_pending(round_total_ms=12.5, emit_ms=1.0)

    lines = path.read_text().splitlines()
    assert len(lines) == 1, lines
    row = json.loads(lines[0])
    for key in _REQUIRED_KEYS:
        assert key in row, f"missing {key}"
    assert row["round_idx"] == 1
    assert row["gamma"] == 3
    assert row["n_accepted"] == out[2]
    assert row["rank"] == 0
    for key in (
        "draft_build_ms",
        "verify_block_ms",
        "tail_bookkeep_ms",
        "round_total_ms",
        "emit_ms",
    ):
        value = row[key]
        assert isinstance(value, (int, float))
        assert value >= 0.0 and value == value and value != float("inf"), (key, value)
    assert row["round_total_ms"] == 12.5
    assert row["emit_ms"] == 1.0


# ---------------------------------------- (3) PROF=2 adds evals, same tokens


def test_prof_two_adds_evals_but_returns_identical_tokens(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The eval-fenced mode forces each bracket, and still commits the same.

    PROF=2 must add at least one ``mx.eval`` per round (draft + rollback +
    append) on top of the verify's own, and must return the SAME tokens as
    PROF=1 -- forcing a bracket cannot change a greedy result.
    """
    monkeypatch.setenv("EXO_DSV41_ROUND_PROF_PATH", str(tmp_path / "p2.jsonl"))

    real_eval = mx.eval
    counts = {"n": 0}

    def counted(*args: Any) -> Any:
        counts["n"] += 1
        return real_eval(*args)

    monkeypatch.setattr(mx, "eval", counted)

    m1, h1, c1, e1 = _spec_setup()
    counts["n"] = 0
    out1 = _run(e1, m1, c1, h1, round_prof=1)
    evals_p1 = counts["n"]

    m2, h2, c2, e2 = _spec_setup()
    counts["n"] = 0
    out2 = _run(e2, m2, c2, h2, round_prof=2)
    evals_p2 = counts["n"]

    assert out1[0] == out2[0], "PROF=2 changed the committed tokens"
    assert out1[2:] == out2[2:]
    assert c1.offset == c2.offset
    assert evals_p2 > evals_p1, (evals_p1, evals_p2)


# --------------------------------------- (4) a broken path never kills a round


def test_broken_prof_path_does_not_raise_and_self_disables(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An undwritable dump path must cost one log line, never a round.

    The path is a DIRECTORY (``open(..., "a")`` fails with ``IsADirectoryError``
    on macOS), the worst realistic case: it exists, so no parent-create path
    hides the failure.
    """
    bad = tmp_path / "notafile"
    bad.mkdir()
    monkeypatch.setenv("EXO_DSV41_ROUND_PROF_PATH", str(bad))

    model, head, cache, engine = _spec_setup()
    out = _run(engine, model, cache, head, round_prof=1)  # must not raise

    # Same tokens as the off path: a failed timer is a pure no-op.
    m0, h0, c0, _e0 = _spec_setup()
    assert out[0] == _run(_Engine({0: h0.make_cache()}), m0, c0, h0, round_prof=0)[0]

    prof = engine._round_prof  # noqa: SLF001
    prof.flush_pending(round_total_ms=1.0, emit_ms=1.0)  # must not raise either
    assert prof._disabled is True  # noqa: SLF001


def test_missing_directory_prof_path_does_not_raise(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A path under a non-existent directory is handled the same way."""
    monkeypatch.setenv(
        "EXO_DSV41_ROUND_PROF_PATH", str(tmp_path / "no" / "such" / "dir" / "p.jsonl")
    )
    model, head, cache, engine = _spec_setup()
    out = _run(engine, model, cache, head, round_prof=1)
    assert out[0]  # a token was committed; nothing raised
    assert engine._round_prof._disabled is True  # noqa: SLF001


# ------------------------------------------------ (bonus) greedy round path


def test_greedy_round_without_a_head_is_timed_and_written(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The no-head (greedy) round still stages a row under PROF=1."""
    path = tmp_path / "greedy.jsonl"
    monkeypatch.setenv("EXO_DSV41_ROUND_PROF_PATH", str(path))

    model = _MockModel([34, 35, 36, 1])
    cache = _MockCache(64)
    engine = _Engine()
    committed, _ms, accepted, gamma = _one_round(
        engine,  # type: ignore[arg-type]
        model=model,
        cache=cache,
        token=model.script[0],
        head=None,
        policy=None,
        round_prof=1,
    )
    engine._round_prof.flush_pending(round_total_ms=5.0, emit_ms=0.5)  # noqa: SLF001

    assert committed == [model.script[1]]
    assert (accepted, gamma) == (1, 1)
    row = json.loads(path.read_text().splitlines()[0])
    assert row["gamma"] == 1 and row["n_accepted"] == 1
    assert row["draft_build_ms"] == 0.0 and row["tail_bookkeep_ms"] == 0.0
    assert row["verify_block_ms"] > 0.0


# ------------------------------------------------------------- path / rank


def test_path_default_and_rank_substitution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unset path -> the documented default; ``<rank>`` fills per worker."""
    monkeypatch.delenv("EXO_DSV41_ROUND_PROF_PATH", raising=False)
    assert rounds_mod._round_prof_path_from_env() == (
        "/tmp/dsv41_round_prof.<rank>.jsonl"
    )
    monkeypatch.setenv("EXO_DSV41_ROUND_PROF_PATH", "/tmp/foo.<rank>.jsonl")
    prof = rounds_mod._RoundProf(1, rank=2)
    assert prof.path == "/tmp/foo.2.jsonl"
    prof._fh.close()  # noqa: SLF001


def test_off_path_builds_no_timer_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PROF=0 must not attach a hook (the byte-identical guarantee)."""
    engine = _Engine()
    assert rounds_mod._round_prof_hook_for(engine, 0) is None  # type: ignore[arg-type]
    assert engine._round_prof is None  # nothing was attached
    # A second call for the same mode reuses the one hook (one open per worker).
    first = rounds_mod._round_prof_hook_for(engine, 1)  # type: ignore[arg-type]
    assert first is not None
    assert rounds_mod._round_prof_hook_for(engine, 1) is first  # type: ignore[arg-type]
    first._fh.close()  # noqa: SLF001
