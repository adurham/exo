# DSv4.1 P1 indexer query-band — deploy-readiness (2026-10-11)

**State:** built and tested on `p20/dsv41-indexer-qsplit` (mlx-lm
`f728bb681`). **NOT deployed, NOT A/B'd, NOT quality-gated.** Deployment is
USER-GATED and LATER. This note is what a launch would need, written so the
decision is a checklist rather than a re-derivation.

## What ships when it is armed

`DSV41_INDEXER_QSPLIT=1` (default OFF; the launcher now forwards it). It engages
only on prefill-sized chunks (`n > 16`); decode (m=1) / verify (m=4) stay
replicated, value-identical. The indexer's query side is split 50/50 across the 2
TP ranks; the key side stays full; a small `cidx` `all_sum` reconstructs the full
selection. `sparse_attn` and the candidate-mask consumer are unchanged.

## Preconditions before arming

1. **STEP-0 indexer-share gate.** The entire P1 payoff is gated on the reconciled
   @100K indexer share (design of record §5: reconcile ≳20 % → GO; collapse toward
   the phase-22 3.6 % → STOP). If STEP-0 FAILs, **do not arm**.
2. **Live A/B on identical shapes.** Flag ON vs flag OFF, same build, only the env
   flipped. Prove the shapes matched by diffing the two runs' `[DSV41] turn reuse:`
   and `prefill controls: (rows=…)` lines — a delta refed at a different depth is
   schedule noise, not the knob. Measure on a **delta** shape (the win lives where
   the indexer share is high) and on a deep rung, not just fresh.
3. **Live quality battery** (this path changes *which rank* computes the indexer;
   the banded top-k is exact-in-theory, so the gate is the standard battery on the
   ON arm: planted needles, low-margin free-prose, glued-fragment / U+FFFD /
   repetition detectors, tool-call sanity, parked-restore continuation).
4. **Verify the flag is live on the runners**, not just exported:
   `ssh <node> 'ps eww -p <pid>' | grep DSV41_INDEXER_QSPLIT` — and read the
   one-line stderr banner `[DSV41] indexer qsplit: DSV41_INDEXER_QSPLIT=1` in the
   runner log. The runner env is an allow-list; an unforwarded knob silently runs
   the default.

## Expected gain (modeled, unmeasured)

`+6.9 % @25K → +15.4 % @100K → +16.9 % @130K` per-chunk wall `[MODELED]` (§4 of the
design of record), i.e. the indexer half-share removed → `1/(1−X)` speedup. These
are **ceilings from a synthetic share fit**; the realized number after the A/B is
the only one to quote. The added wire is small: `cidx all_gather`-equivalent
`[b,n,512]` int32 ≈ 4.19 MB/layer × 8 source layers ≈ 34 MB/chunk, negligible
beside the existing per-chunk sync.

## Risks / what to watch

- **Top-k near-ties.** `M=n/2` vs `M=n` can pick a different Metal matmul tiling,
  perturbing fp scores so a near-tied top-k flips. The design documents this and
  downgrades "bit-exact" to **top-k overlap**; measure the overlap and the
  final-logit diff, do not assert exactness.
- **New collective reliability.** The `all_sum` follows the v4-default ARQ-reliable
  zero-pad path (chosen over a bare `all_gather` precisely for JACCL reliability).
  The rank-skew Metal-watchdog class applies as to every `all_sum`; the existing
  `DSV41_SYNC_COLLECTIVES` / warm-guard covers the first post-build forward.
- **Memory.** P1 adds none (the padding is transient, `[b,n,512]` int32).
- **Regression surface.** P1 must not move decode or the shipped M2 hierarchical
  pass; the flag-OFF path is byte-identical (regression control: dsv41 suite
  unchanged).

## Rollback

`DSV41_INDEXER_QSPLIT` unset/`0` (the default) — no code path change needed. The
branch stacks on `deploy/next22-rankfix`; abandoning it costs nothing to the live
cluster (nothing was launched).
