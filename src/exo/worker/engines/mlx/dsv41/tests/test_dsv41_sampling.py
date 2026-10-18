"""Sampling math for dsv41: resolution, transforms, rejection sampling, determinism."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest

from exo.worker.engines.mlx.dsv41 import sampling as sm


def _params(**kw):
    base = {"temperature": None, "top_p": None, "top_k": None}
    base.update(kw)
    return SimpleNamespace(**base)


def test_resolution_request_then_default_then_clamp():
    assert sm.resolve_sampling(_params(temperature=0.7)).temperature == 0.7
    assert sm.resolve_sampling(_params(), default_temperature=0.3).temperature == 0.3
    assert sm.resolve_sampling(_params(temperature=-2.0)).temperature == 0.0
    assert sm.resolve_sampling(_params()).greedy
    assert sm.resolve_sampling(_params(temperature=0.0)).greedy
    assert not sm.resolve_sampling(_params(temperature=0.01)).greedy


def test_resolution_drops_invalid_knobs():
    s = sm.resolve_sampling(_params(temperature=0.5, top_p=1.5, top_k=0))
    assert s.top_p is None and s.top_k is None
    s2 = sm.resolve_sampling(_params(temperature=0.5), default_top_p=0.9, default_top_k=40, default_min_p=0.05)
    assert (s2.top_p, s2.top_k, s2.min_p) == (0.9, 40, 0.05)


def test_top_k_keeps_only_k_candidates():
    logits = mx.array([[0.0, 1.0, 2.0, 3.0]])
    out = sm.apply_sampling(logits, sm.Sampling(temperature=1.0, top_k=2))
    kept = [i for i in range(4) if float(out[0, i]) > -1e30]
    assert kept == [2, 3]


def test_top_p_drops_the_tail():
    logits = mx.array([[10.0, 0.0, 0.0, 0.0]])  # ~1.0 on id 0
    out = sm.apply_sampling(logits, sm.Sampling(temperature=1.0, top_p=0.5))
    assert float(out[0, 0]) > -1e30
    assert [i for i in range(1, 4) if float(out[0, i]) > -1e30] == []


def test_temperature_scales_the_logits():
    logits = mx.array([[0.0, 2.0]])
    out = sm.apply_sampling(logits, sm.Sampling(temperature=0.5))
    assert float(out[0, 1]) == pytest.approx(4.0)


def test_sample_one_is_greedy_guarded():
    with pytest.raises(ValueError):
        sm.sample_one(mx.array([[0.0, 1.0]]), sm.Sampling(temperature=0.0))


def test_sample_one_follows_the_distribution():
    # A 10x-odds candidate must dominate; over many seeds it must win most draws.
    logits = mx.array([[0.0, 2.302585]])  # p(1) = 0.9
    wins = sum(
        sm.sample_one(logits, sm.Sampling(temperature=1.0, seed=s)) == 1
        for s in range(200)
    )
    assert wins > 140  # ~180 expected; this bounds flakiness without hiding bias


def test_sample_prefix_accepts_everything_when_p_is_one():
    # Rows are one-hot on the draft ids -> p(draft)=1 -> every draft accepted,
    # and the committed list is the draft plus a bonus token drawn from row gamma.
    logits = mx.array([
        [0.0, -30.0],   # predicts draft[0] = 0
        [0.0, -30.0],   # predicts draft[1] = 0
        [-30.0, 0.0],   # bonus row -> id 1
    ])
    committed, accepted = sm.sample_prefix(
        logits, [0, 0], sampling=sm.Sampling(temperature=1.0, seed=7)
    )
    assert accepted == 2
    assert committed[:2] == [0, 0]
    assert committed[2] == 1


def test_sample_prefix_rejects_and_samples_the_residual():
    # The draft is an implausible token: rejection, and the committed token must
    # come from p with that token's mass removed (so it can never be the draft).
    logits = mx.array([[-25.0, 0.0, -30.0], [-30.0, -30.0, 0.0]])
    committed, accepted = sm.sample_prefix(
        logits, [0], sampling=sm.Sampling(temperature=1.0, seed=11)
    )
    assert accepted == 0
    assert len(committed) == 1
    assert committed[0] != 0


def test_sample_prefix_is_deterministic_for_a_seed():
    logits = mx.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    a, _ = sm.sample_prefix(logits, [0], sampling=sm.Sampling(temperature=1.0, seed=42))
    b, _ = sm.sample_prefix(logits, [0], sampling=sm.Sampling(temperature=1.0, seed=42))
    assert a == b


def test_sample_prefix_varies_across_seeds():
    logits = mx.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    drawn = {
        sm.sample_prefix(logits, [1], sampling=sm.Sampling(temperature=1.0, seed=s))[0][0]
        for s in range(40)
    }
    assert len(drawn) >= 2, "a flat distribution must not collapse to one token"


def test_hard_ban_is_respected_in_sampling():
    logits = mx.array([[0.0, 5.0, 0.0]])
    committed, accepted = sm.sample_prefix(
        logits, [1], sampling=sm.Sampling(temperature=1.0, seed=3),
        banned_per_row=[{1: float("-inf")}],
    )
    assert accepted == 0
    assert committed[0] != 1


def test_fully_banned_row_falls_back_instead_of_nan():
    logits = mx.array([[0.0, 1.0]])
    committed, _ = sm.sample_prefix(
        logits, [0], sampling=sm.Sampling(temperature=1.0, seed=5),
        banned_per_row=[{0: float("-inf"), 1: float("-inf")}],
    )
    assert committed[0] in (0, 1)


def test_sampling_marginal_is_exact_chi_square():
    """The draw must be an EXACT sampler of the transformed distribution.

    A poorly-ranked draft (p ~ 0.07) is rejected almost every time, so nearly every
    draw comes from the residual p (draft mass removed, renormalized) while a few
    accept it -- the MIXTURE must reproduce p itself. 120k draws, chi-square on the
    full marginal: this catches residual-renormalisation / off-by-one bugs that
    per-seed determinism tests cannot see.
    """
    import math

    pv = [0.40, 0.25, 0.15, 0.10, 0.07, 0.03]
    rows = mx.array([[math.log(v) for v in pv]], dtype=mx.float32)
    draft = [4]                                    # p=0.07: mostly rejected
    n = 120_000
    counts = [0] * 6
    for seed in range(n):
        committed, _accepted = sm.sample_prefix(
            rows, draft, sampling=sm.Sampling(temperature=1.0, seed=seed)
        )
        counts[committed[0]] += 1
    exp = [v * n for v in pv]
    chi2 = sum((o - e) ** 2 / e for o, e in zip(counts, exp))
    assert chi2 < 25.0, f"chi2={chi2:.1f} obs={counts} exp={[round(e) for e in exp]}"


def test_step_advances_the_key():
    """A key re-derived from the seed alone would replay the same uniform every
    round; folding the step in must change the stream."""
    rows = mx.array([[0.0, 0.0, 0.0, 0.0]])
    def draw(step: int) -> int:
        return sm.sample_prefix(rows, [0], sampling=sm.Sampling(temperature=1.0, seed=1), step=step)[0][0]
    assert draw(0) == draw(0)        # same seed + step -> same draw
    assert draw(0) != draw(1)        # the step advances the key
    draws = {draw(st) for st in range(40)}
    assert len(draws) >= 3           # and the stream really varies
