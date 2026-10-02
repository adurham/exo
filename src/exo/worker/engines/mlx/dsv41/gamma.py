"""Round-level cost model and adaptive draft-length policy for the DSv4.1 spec round.

OFF BY DEFAULT. ``rounds._spec_policy`` returns this policy only when
``DSV41_GAMMA_ADAPT=1``; otherwise the engine keeps the legacy
``spec.GammaPolicy`` which, on the served path, is never updated and therefore
stays pinned at its start gamma (D1 report §0.1/§7).

Why a new cost model. ``spec.VERIFY_MS`` (``spec.py:77``) is a verify-forward
table from an older build: {R1 58.5, R2 74.9, R3 87.9, R4 97.7, R5 111.7} plus a
flat ``draft_ms=(8.5, 0.9)`` and ``overhead_ms=4``. Against the served round
measured in D1 §6.4 (2x M4 Max TP=2, FAST_SYNCH=1, 4K/7K context) it overprices
the cheap rounds and underprices drafting, so an adaptive policy fed with it
collapses to gamma 1 on a prompt that accepts 1.68 drafts/round at gamma 3
(D1 §8, docA7k -9.7 %). This module rebuilds the round from its measured parts:

    round_ms(g) = body_ms[g + 1]              verify forward, 40 layers, R = g+1 rows
                + tail_ms                     sharded LM head + argmax all_sum + drain
                + draft_fixed + draft_step*g  DSpark draft (+ previous round's lazy
                                              rollback/append), forced at ids->numpy
                + host_fixed + host_step*g    cancel agreement, draft graph build,
                                              engram hash, accept/compare, glue

Constants (D1 §6.4, docA4k rank 0, medians of 38 rounds; docA7k within 1 ms):

    body R1..R5          45.8 / 59.9 / 75.0 / 84.2 / 100.3 ms
    draft GPU g1..g4     6.05 / 7.65 / 9.03 / 9.66 ms  -> 4.9 + 1.25 g (LSQ)
    tail g1..g4          4.62 / 3.89 / 5.89 / 5.62 ms  -> 5.0 (mean; no trend)
    host                 2.3 ms at g3 (0.39 cancel + 0.76 build + 0.77 accept +
                         0.16 engram + 0.19 glue); build/accept grow with g
                         (0.85 at g1 .. 1.44 at g4) -> 1.5 + 0.2 g
    measured round       g1 71.8 / g2 88.2 / g3 101.0 / g4 117.1 ms
    model                g1 72.8 / g2 89.3 / g3 100.0 / g4 117.5 ms  (|err| <= 1.1 ms)

Acceptance estimator. Same chain-rule form as ``spec.GammaPolicy`` --
q_k = P(draft k accepted | drafts 1..k-1 accepted), expected committed tokens
E(g) = 1 + sum_{k<=g} prod_{j<=k} q_j -- with three changes, each aimed at the
D1 §7 collapse (A7k: 4 early misses at positions 2/3, policy drops to gamma 1,
and position 2 is then never tried again, so its estimate can never recover):

* a Beta prior of mean ``prior_q`` and weight ``prior_n`` instead of Beta(1,1),
  so two or three early misses do not move q_2 from 0.7 to 0.25;
* exponential forgetting (``half_life`` rounds) of the observed counts. A
  position that is no longer tried decays back to the prior, which in turn makes
  a longer gamma attractive again -- deterministic self-exploration with no
  random probes;
* hysteresis: switch away from the current gamma only when another gamma's
  expected tokens/ms beats it by more than ``switch_margin`` (relative).

Everything is host-only arithmetic on Python floats: no MLX ops, no syncs, and
the choice is a pure function of the (gamma, accepted) history.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

#: Env flag that routes the engine to :class:`RoundCostGammaPolicy` and wires
#: ``update()`` into the round. Anything other than ``"1"`` keeps the legacy path.
GAMMA_ADAPT_ENV = "DSV41_GAMMA_ADAPT"
#: Optional override of the verify-body table, comma-separated ms for R=1..5.
BODY_MS_ENV = "DSV41_GAMMA_BODY_MS"


def gamma_adapt_enabled() -> bool:
    return os.environ.get(GAMMA_ADAPT_ENV, "0") == "1"


#: D1 §6.4 verify body (40 layers) by verify rows R; docA4k rank 0.
D1_BODY_MS: dict[int, float] = {1: 45.8, 2: 59.9, 3: 75.0, 4: 84.2, 5: 100.3}
#: D1 §6.4 measured whole round (incl. cancel) by gamma; g0 = plain greedy step.
D1_ROUND_MS: dict[int, float] = {0: 49.8, 1: 71.8, 2: 88.2, 3: 101.0, 4: 117.1}


def _body_from_env() -> dict[int, float] | None:
    raw = os.environ.get(BODY_MS_ENV, "").strip()
    if not raw:
        return None
    vals = [float(v) for v in raw.split(",")]
    return {r + 1: v for r, v in enumerate(vals)}


@dataclass(frozen=True)
class RoundCost:
    """Wall ms of one spec round as a function of the draft length ``g``."""

    body_ms: dict[int, float] = field(default_factory=lambda: dict(D1_BODY_MS))
    tail_ms: float = 5.0
    draft_fixed_ms: float = 4.9
    draft_step_ms: float = 1.25
    host_fixed_ms: float = 1.5
    host_step_ms: float = 0.2

    def round_ms(self, g: int) -> float:
        if g < 1:
            raise ValueError("round_ms models a speculative round (g >= 1)")
        r = g + 1
        if r not in self.body_ms:
            raise KeyError(f"no verify-body cost for R={r} rows")
        return (
            self.body_ms[r]
            + self.tail_ms
            + self.draft_fixed_ms
            + self.draft_step_ms * g
            + self.host_fixed_ms
            + self.host_step_ms * g
        )

    @classmethod
    def from_env(cls) -> RoundCost:
        body = _body_from_env()
        return cls(body_ms=body) if body else cls()


class AcceptanceEstimator:
    """Per-position conditional acceptance with a Beta prior and forgetting."""

    def __init__(
        self,
        *,
        max_pos: int = 8,
        prior_q: float = 0.7,
        prior_n: float = 8.0,
        half_life: float = 32.0,
    ) -> None:
        self.prior_q, self.prior_n = prior_q, prior_n
        self.decay = 0.5 ** (1.0 / half_life) if half_life > 0 else 1.0
        self.tried = [0.0] * (max_pos + 1)
        self.acc = [0.0] * (max_pos + 1)

    def observe(self, gamma: int, n_acc: int) -> None:
        d = self.decay
        for k in range(1, len(self.tried)):
            self.tried[k] *= d
            self.acc[k] *= d
        for k in range(1, gamma + 1):
            self.tried[k] += 1.0
            if n_acc >= k:
                self.acc[k] += 1.0
            else:
                break

    def q(self, k: int) -> float:
        a, n = self.acc[k], self.tried[k]
        return (a + self.prior_q * self.prior_n) / (n + self.prior_n)

    def expected_tokens(self, g: int) -> float:
        """E[committed tokens] for a round of draft length ``g`` (accepted + 1)."""
        e, p = 1.0, 1.0
        for k in range(1, g + 1):
            p *= self.q(k)
            e += p
        return e

    def expected_accepted(self, g: int) -> float:
        return self.expected_tokens(g) - 1.0


class RoundCostGammaPolicy:
    """Pick the draft length that maximises expected committed tokens per ms.

    Interface-compatible with ``spec.GammaPolicy`` (``next()`` / ``update()``);
    ``observes_rounds = True`` tells ``rounds._one_round`` to feed it every
    speculative round's (gamma, accepted).
    """

    observes_rounds = True

    def __init__(
        self,
        gammas: tuple[int, ...] = (1, 2, 3, 4),
        start: int = 3,
        *,
        cost: RoundCost | None = None,
        warmup: int = 4,
        switch_margin: float = 0.05,
        prior_q: float = 0.7,
        prior_n: float = 8.0,
        half_life: float = 32.0,
    ) -> None:
        if start not in gammas:
            raise ValueError(f"start gamma {start} not in {gammas}")
        self.gammas, self.g = tuple(gammas), start
        self.cost = cost or RoundCost.from_env()
        self.warmup, self.margin = warmup, switch_margin
        self.est = AcceptanceEstimator(
            max_pos=max(gammas), prior_q=prior_q, prior_n=prior_n, half_life=half_life
        )
        self.rounds = 0

    def update(self, gamma: int, n_acc: int) -> None:
        self.rounds += 1
        self.est.observe(gamma, n_acc)

    def rate(self, g: int) -> float:
        """Expected committed tokens per second at draft length ``g``."""
        return 1000.0 * self.est.expected_tokens(g) / self.cost.round_ms(g)

    def next(self) -> int:
        if self.rounds < self.warmup:
            return self.g
        cur = self.rate(self.g)
        best, best_rate = self.g, cur
        for g in self.gammas:  # ascending; ties keep the earlier/current
            r = self.rate(g)
            if r > best_rate:
                best, best_rate = g, r
        if best != self.g and best_rate <= cur * (1.0 + self.margin):
            best = self.g
        self.g = best
        return best
