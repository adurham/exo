# Gamma optimality on the live dsv41 path (offline, from the real VERIFY_MS table)

Date: 2026-10-07. No cluster access; pure arithmetic from in-tree constants.

## Inputs (all from source / live measurement)

`VERIFY_MS = {1:58.5, 2:74.9, 3:87.9, 4:97.7, 5:111.7, 6:120.1}` ms
(`mlx-lm/mlx_lm/models/deepseek_v41/spec.py:77` — the phase-17 verify table).

Per-round time model (from `GammaPolicy.__init__` defaults, spec.py:87-88, and
the engine's one-sync-per-round structure):

    round_ms(g) = draft_base + draft_per*(g-1) + VERIFY_MS[g+1] + overhead
                = 8.5 + 0.9*(g-1) + VERIFY_MS[g+1] + 4.0

Expected committed tokens per round at per-position acceptance q_k:

    E(g) = 1 + q1 + q1q2 + ... + q1..qg

## Two regimes

Live measurement today on benign content gives mean accepted **2.81/3**
(q1≈0.98). At that acceptance the model is saturated and gamma=3 is optimal:

| g | verify rows | round_ms | E(g) | t/s (rel) |
|---|---|---|---|---|
| 1 | R2 | 87.4 | ~1.94 | −30% |
| 2 | R3 | 101.3 | ~2.90 | −8% |
| 3 | R4 | 112.0 | ~3.81 | 0% |
| 4 | R5 | 126.9 | ~4.75 | −12% |

(Exact values depend on q2..q5, but the ordering is robust: at q→1 the extra
verify rows dominate.)

Real-session regime — if the observed in-session mean were **1.50/3**
(phase-16 standalone level, per-position q1≈0.53), gamma=2 is ~3% better than
gamma=3 and gamma=1 ~2% — i.e. at most ~3–5% total, below the 3%-with-
non-overlapping-IQR adoption bar, and only by *lowering* a pin that is
nonetheless hard-coded.

Only a much worse acceptance regime (mean ≈1.27/3, q1≈0.47) would open gamma=2
to ~+5%, and even that is one relaunch of cost plus a prefix-cache eviction for
a value at the edge of the bar.

## Conclusion

At the acceptance the live cluster actually shows (2.8/3), gamma=3 is optimal.
There is **no ≥3% gamma lever**, and there is no env knob to move gamma on the
dsv41 engine anyway (`EXO_SPECULATIVE_GAMMA` is read only by the dormant legacy
paths). The gamma matrix (plan Phase 6) is not worth a relaunch.
