"""Round-level cost model + adaptive gamma policy (``dsv41.gamma``) and its
opt-in wiring into ``rounds._one_round`` (``DSV41_GAMMA_ADAPT=1``).

Fixtures are REAL per-round acceptance traces from the D1 gamma sweep
(2x M4 Max TP=2, served env, 256 tokens, fixed gamma 4, rank 0, round 0 = the
draft-priming round dropped; the 3 reps are byte-identical): each digit is the
accepted-draft count of one round, i.e. the target/draft agreement run length
capped at 4. Replaying a policy over them answers "what would it have chosen".
"""

from __future__ import annotations

from typing import Any

import pytest

from exo.worker.engines.mlx.dsv41 import gamma as G  # noqa: N812
from exo.worker.engines.mlx.dsv41.rounds import _one_round, _spec_policy
from exo.worker.engines.mlx.dsv41.tests.test_dsv41_engine import (
    MockCache,
    MockHead,
    ScriptedModel,
    _EngineStub,
)

D1_G4_TRACES = {
    "docA4k": "1101012430402130400401400102223123200030121002022000012422214141302342423343421121014100002214444",
    "docA7k": "0024321010102440000401444111122434212444440442340041201021102020233110201404444424444",
    "docB4k": "01000113143223402000100111010200441111342444444444444314431443144333100013022111001311",
    "docB7k": "01310040110240410011002222243123101300042210110200343302000011114400301220110044442410223420400114144",
}
#: The stale table + defaults the legacy ``spec.GammaPolicy`` prices rounds with.
LEGACY_VERIFY_MS = {1: 58.5, 2: 74.9, 3: 87.9, 4: 97.7, 5: 111.7}


def _legacy_round_ms(g: int) -> float:
    return 8.5 + 0.9 * g + LEGACY_VERIFY_MS[g + 1] + 4.0


def _replay(policy: Any, runs: list[int], cost: G.RoundCost) -> tuple[list[int], float]:
    """Drive ``policy`` over agreement run lengths; (gammas chosen, tok/s)."""
    gammas, ms, tokens = [], 0.0, 0
    for run in runs:
        g = policy.next()
        acc = min(run, g)
        policy.update(g, acc)
        gammas.append(g)
        ms += cost.round_ms(g)
        tokens += acc + 1
    return gammas, 1000.0 * tokens / ms


def _fixed(g: int, runs: list[int], cost: G.RoundCost) -> float:
    return 1000.0 * sum(min(r, g) + 1 for r in runs) / (cost.round_ms(g) * len(runs))


def _runs(name: str) -> list[int]:
    return [int(c) for c in D1_G4_TRACES[name]]


# ------------------------------------------------------------------ cost model


def test_round_cost_reproduces_d1_measured_rounds() -> None:
    """The model is within 1.5 ms of every measured D1 round (g1..g4)."""
    cost = G.RoundCost()
    for g in (1, 2, 3, 4):
        assert cost.round_ms(g) == pytest.approx(G.D1_ROUND_MS[g], abs=1.5), g


def test_legacy_table_is_what_is_being_replaced() -> None:
    """Pin WHY: the legacy pricing is >= 10 ms off every measured round."""
    for g in (1, 2, 3, 4):
        assert abs(_legacy_round_ms(g) - G.D1_ROUND_MS[g]) >= 10.0, g


def test_round_cost_is_monotone_and_body_overridable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cost = G.RoundCost()
    assert [cost.round_ms(g) for g in (1, 2, 3, 4)] == sorted(
        cost.round_ms(g) for g in (1, 2, 3, 4)
    )
    monkeypatch.setenv(G.BODY_MS_ENV, "40,50,60,70,80")
    c2 = G.RoundCost.from_env()
    assert c2.body_ms == {1: 40.0, 2: 50.0, 3: 60.0, 4: 70.0, 5: 80.0}
    assert c2.round_ms(3) == pytest.approx(70.0 + 5.0 + 4.9 + 3.75 + 1.5 + 0.6)


# ----------------------------------------------------------------- estimator


@pytest.mark.parametrize("name", sorted(D1_G4_TRACES))
def test_estimator_reproduces_observed_accepted_per_round(name: str) -> None:
    """With the prior switched off and no forgetting, E[accepted] at the traced
    gamma equals the observed mean accepted/round (D1 §8: 1.653/2.023/1.942/1.539)."""
    runs = _runs(name)
    est = G.AcceptanceEstimator(max_pos=4, prior_n=1e-9, half_life=0)
    for r in runs:
        est.observe(4, r)
    observed = sum(runs) / len(runs)
    assert est.expected_accepted(4) == pytest.approx(observed, abs=1e-6)


def test_estimator_forgetting_returns_untried_positions_to_the_prior() -> None:
    est = G.AcceptanceEstimator(max_pos=4, prior_q=0.7, prior_n=8.0, half_life=32.0)
    for _ in range(3):
        est.observe(3, 0)  # three early misses at position 1
    hit = est.q(1)
    assert hit < 0.6
    for _ in range(200):
        est.observe(1, 1)  # position 1 keeps landing, 2/3 never tried
    assert est.q(1) > 0.9
    assert est.q(2) == pytest.approx(0.7, abs=0.01)  # decayed back to the prior


# -------------------------------------------------------------------- policy


@pytest.mark.parametrize("name", sorted(D1_G4_TRACES))
def test_policy_is_deterministic(name: str) -> None:
    cost = G.RoundCost()
    a, ra = _replay(G.RoundCostGammaPolicy(start=3, cost=cost), _runs(name), cost)
    b, rb = _replay(G.RoundCostGammaPolicy(start=3, cost=cost), _runs(name), cost)
    assert a == b and ra == rb


@pytest.mark.parametrize("name", sorted(D1_G4_TRACES))
def test_policy_matches_best_fixed_gamma_on_d1_traces(name: str) -> None:
    """Within 1 % of the best fixed gamma (hindsight) on every D1 prompt."""
    cost = G.RoundCost()
    runs = _runs(name)
    _gs, rate = _replay(G.RoundCostGammaPolicy(start=3, cost=cost), runs, cost)
    best = max(_fixed(g, runs, cost) for g in (1, 2, 3, 4))
    assert rate >= 0.99 * best


def test_policy_does_not_collapse_where_legacy_did() -> None:
    """D1 §7/§8: legacy+update fell to gamma 1-2 on docA7k (-9.7 %). The new
    policy keeps gamma 3 there; the legacy policy (same trace) does not."""
    from mlx_lm.models.deepseek_v41.spec import GammaPolicy

    cost = G.RoundCost()
    runs = _runs("docA7k")
    new, _ = _replay(G.RoundCostGammaPolicy(start=3, cost=cost), runs, cost)
    old, _ = _replay(GammaPolicy(start=3), runs, cost)
    assert new.count(3) / len(new) >= 0.95
    assert old.count(3) / len(old) <= 0.10


@pytest.mark.parametrize(("q", "expect"), [(0.3, {1}), (0.95, {3, 4})])
def test_policy_follows_acceptance_regime(q: float, expect: set[int]) -> None:
    """Deterministic stream with constant acceptance: low q -> gamma 1,
    high q -> long drafts. (Agreement run length L with P(L>=k)=q^k, built
    deterministically by stratified quantiles.)"""
    runs = []
    n = 200
    for i in range(n):
        u = (i * 0.6180339887) % 1.0  # low-discrepancy, no RNG
        run = 0
        while run < 4 and u < q ** (run + 1):
            run += 1
        runs.append(run)
    cost = G.RoundCost()
    gs, _ = _replay(G.RoundCostGammaPolicy(start=3, cost=cost), runs, cost)
    tail = gs[50:]
    assert max(set(tail), key=tail.count) in expect


def test_policy_holds_start_through_warmup() -> None:
    pol = G.RoundCostGammaPolicy(start=3, warmup=4)
    probe = G.RoundCostGammaPolicy(start=3, warmup=0)
    for _ in range(4):
        assert pol.next() == 3  # pinned while rounds < warmup ...
        pol.update(3, 0)
        probe.update(3, 0)
    for _ in range(8):  # ... and free to move once past it
        pol.update(3, 0)
        probe.update(3, 0)
    assert probe.next() == 1 and pol.next() == 1


# ---------------------------------------------------------------- wiring


def _spec_round(policy: Any) -> tuple[int, int]:
    model = ScriptedModel([34, 35, 36, 34, 35, 1], prompt_tokens=0)
    head = MockHead(model, lie_at=1)
    engine = _EngineStub(_draft_windows={0: object()})
    model.round_entry_offset = 1
    _c, _ms, accepted, gamma = _one_round(
        engine,  # type: ignore[arg-type]
        model=model,
        cache=MockCache(64),
        token=model.script[0],
        head=head,
        policy=policy,
    )
    return accepted, gamma


def test_default_is_the_legacy_policy_and_is_not_updated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flag unset: served behaviour unchanged (legacy policy, never fed)."""
    from mlx_lm.models.deepseek_v41.spec import GammaPolicy

    monkeypatch.delenv(G.GAMMA_ADAPT_ENV, raising=False)
    pol = _spec_policy(3)
    assert type(pol) is GammaPolicy
    _spec_round(pol)
    assert pol.rounds == 0 and pol.next() == 3


def test_flag_on_round_feeds_the_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(G.GAMMA_ADAPT_ENV, "1")
    pol = _spec_policy(3)
    assert isinstance(pol, G.RoundCostGammaPolicy)
    accepted, gamma = _spec_round(pol)
    assert (accepted, gamma) == (1, 3)
    assert pol.rounds == 1
    assert pol.est.tried[1:4] == [1.0, 1.0, 0.0]
    assert pol.est.acc[1:4] == [1.0, 0.0, 0.0]


def test_flag_on_policy_moves_off_start_after_warmup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end through _one_round: repeated position-1 misses drive gamma down."""
    monkeypatch.setenv(G.GAMMA_ADAPT_ENV, "1")
    pol = _spec_policy(3)
    gammas = []
    for _ in range(12):
        model = ScriptedModel([34, 35, 36, 34, 35, 1], prompt_tokens=0)
        head = MockHead(model, lie_at=0)  # every first draft wrong
        model.round_entry_offset = 1
        _c, _ms, _acc, g = _one_round(
            _EngineStub(_draft_windows={0: object()}),  # type: ignore[arg-type]
            model=model,
            cache=MockCache(64),
            token=model.script[0],
            head=head,
            policy=pol,
        )
        gammas.append(g)
    assert gammas[:4] == [3, 3, 3, 3]
    assert gammas[-1] == 1
