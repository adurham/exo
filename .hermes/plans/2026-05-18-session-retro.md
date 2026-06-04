# 2026-05-18 Session Retrospective — DSv4 Verify-Forward Plan

Plan: `2026-05-18_1505-dsv4-verify-forward-toward-35tps.md`
Goal: 30 t/s → 35 t/s at γ=2 100K c=1, quality-correct (`EXO_DSV4_INDEX_TOPK=512`).
Outcome: **baseline 30.06 t/s holds; +0 t/s net; both attempted levers regressed.**

## Result Summary

| Task | Description                                          | Outcome           | t/s      |
|------|------------------------------------------------------|-------------------|----------|
| 1    | Quality-correct baseline (10/10 iters)               | PASS              | 30.062 σ=0.059 |
| 2    | Lever 1 take 2 — L<=8 fused-SDPA L-into-batch fold  | REGRESSED → revert | 28.9     |
| 3    | Lever 2 — fuse post_attn+ffn_pre, inline hc_expand   | CATASTROPHIC → revert | 7.2-10.5 |
| 5    | Re-profile (FENCE=43) — verify breakdown            | unchanged from plan | — |
| 5b   | FENCE=8 probe (user-suggested alt)                   | NOT A WIN          | 29.5-29.6 |

Baseline anchor: `baseline-2026-05-18-mtp-g2-topk512-30.06` (pushed).
Cluster restored to baseline; production code unchanged on origin/main.

## Tag List

```
baseline-2026-05-18-mtp-g2-topk512-30.06   ← TODAY'S production anchor
baseline-2026-05-18-mtp-g2-32.35           ← prior, topk=160 (quality unvalidated)
champion-2026-05-17-mtp-g2-fenced-31.5     ← May 17, may not be reproducible
```

## What Worked

- Task 0 deployment-verification protocol (strings-grep, inference probe,
  small-prefill bench) caught the patch-actually-deployed question every time.
- Per-bench discipline (abort after 2 confirmatory failures) saved time —
  each revert decision was made on 2-3 data points, not 10-iter completions.
- The plan's branch logic ("if lever fails, revert and capture failure mode")
  was followed faithfully; no fabricated "we'll iterate on it" rationalizations.

## What Didn't Work, And Why (Best Hypothesis)

### Lever 1 Take 2 (mlx-lm 49a05d95 → reverted as b4d9e410)

Approach: gate the L_q==1 fused-SDPA fast path to also cover small L_q
(verify pass, L_q=3 at γ=2), folding L into the batch axis so each (b,l)
row becomes an independent single-query attention call from
`mx.fast.scaled_dot_product_attention`. Helper `_to_4d_mask_fold` lifted
the mask shapes safely. Prefill stayed on the inner kernel (gated on L<=8).

Result: 30.06 → 28.9 t/s (~4% slower) in 2 confirmatory iters.

Hypothesis: the existing `_sparse_pooled_attention_inner` is already
`@partial(mx.compile, shapeless=True)` with tight dispatch on its 10-op
chain. Adding a `mx.fast.scaled_dot_product_attention` call at small
B*L=3 incurs per-call Metal kernel launch overhead that beats any
intrinsic kernel-fusion benefit. The microbench that showed 1.24× wall
on L=1 was an isolated comparison that didn't account for in-context
dispatch costs.

Also flagged by the code-analyzer pre-attempt: the `local_kv` broadcast
materializes a `(B*L, 1, sw, D)` copy at reshape time (B*L=3, sw=4096,
D=128 → ~3MB per layer × 60 sparse layers = ~180MB extra memory traffic
per verify cycle). Not validated as the dominant cost but likely a
contributing factor.

### Lever 2 (mlx-lm 294e155c → reverted as ac267339)

Approach: fuse `_raw_post_attn` + `_raw_ffn_pre` (the two boundaries
between attention and FFN that don't cross-rank) into a single
`_raw_attn_to_ffn` compile. To avoid the nested-compile pitfall (pitfall
#16), inline the body of `_hc_expand_op` as `_hc_expand_inline`.

Result: 30.06 → 10.5 t/s (first attempt warmup), then 7.2 t/s (second
3-iter run warmup, wall=705s = 2× baseline). Catastrophic regression
across BOTH iters of both attempts. Prefill rate was actually slightly
FASTER (256 tok/s vs 232) — the slowdown is purely in decode and the
MTP cache prefill setup.

Hypothesis: `HyperConnection.__call__` calls `_hc_kernel` which uses
`mx.fast.metal_kernel` (a JIT custom Metal kernel). Capturing
`mx.fast.metal_kernel` inside an `mx.compile` boundary appears to
trigger pathological recompile or cache-contention behavior. The
original code kept `ffn_hc` in its own (already-compiled) chunk; my
fused version pulled it INTO another compile, breaking some
mlx-internal invariant. The MTPModule (NOT compiled, NOT touched by my
patch) also slowed down systemically — suggesting shared state pollution
(allocator? Metal kernel cache?). **Not root-caused.** Needs an
isolated single-layer microbench to characterize.

### FENCE=8 (no code change)

Approach: per the script default and the May-17 champion claim, try
`EXO_DSV4_FENCE_EVERY_N_LAYERS=8` (instead of the plan-prescribed 43).

Result: 29.5-29.6 t/s in 3 confirmatory iters. ~1.5% slower than
FENCE=43 baseline 30.06. Profile data at N=200 cycles showed mean total
61.59ms (FENCE=8) vs 62.65ms (FENCE=43) — only 1.7% better, but the
end-to-end agg_tps suggests the savings are eaten by overhead elsewhere.

The first 80-cycle profile snapshot looked promising (verify mean
54.23ms FENCE=8 vs 57.20ms FENCE=43, a 5% gap) but converged toward
parity by N=200. Either the early window was unrepresentative or
FENCE=8 trades mean for tail differently than I hypothesized. Not
worth more investigation right now — the empirical agg_tps is the
honest metric.

## What Got Profiled (FENCE=43, plan-reproducible)

```
draft      4.54 ms (7.2%)   min=4.40, max=5.07,  n=120
verify    57.10 ms (91.1%)  min=51.74, max=70.40 ← 35% spread
accept     0.81 ms (1.3%)   min=0.64, max=2.57
rollback   0.20 ms (0.3%)   min=0.14, max=0.35
total     62.65 ms          min=57.17, max=78.24

Acceptance: mean=1.04/2, hist 0:30, 1:36, 2:34 → α2 ≈ 0.52
```

Verify is 91% of the cycle — **unchanged from the plan's 91% measurement.
The plan's premise (verify is the bottleneck) is current.** What didn't
work was attacking it from the compile-boundary direction.

## New Findings Worth Future Investigation

1. **Verify variance is 35%** (52-70ms) — wider than expected for a
   nominally deterministic operation. Possibly the same chained-collective
   tail mechanism that caused the May-17 γ=2 iter-1 stall (now mitigated
   per-step in `draft_tokens`). If verify also has a chained-tail, the
   per-step-fence pattern that fixed draft may apply to verify too.
   **Action**: add MLX_SIGNAL_PROBE / JACCL_POLL_INSTRUMENT to a verify
   path and dump per-layer collective latency; look for tails clustered
   at the 50% mean (which would indicate buffer drain timing pattern,
   not a routing hazard).

2. **The compile-boundary collapse pattern is fragile.** Two attempts to
   fuse compile chunks (lever 1 take 2 changed sparse-attn dispatch
   subtly; lever 2 fused post_attn+ffn_pre) both regressed despite being
   "obvious" wins on paper. This area should NOT be revisited without a
   single-layer microbench that proves the in-context savings BEFORE
   touching the cluster code. Specifically:
   - Microbench `_raw_attn_pre` standalone vs `_raw_attn_pre` →
     `_raw_attn_to_ffn` (fused) at γ=2 verify shape, with the EXACT
     same upstream/downstream inputs the production layer sees.
   - Need to learn whether `mx.fast.metal_kernel`-inside-`mx.compile`
     has a known cost.

3. **The L_q==1 fast path may itself be a regression source.** It was
   landed May 13 with a microbench showing 1.24× — but at small-L
   in-context the inner kernel may also win. Worth A/B-testing by
   disabling the fast path via env (the file already has
   `EXO_DSV4_TOPK_FUSED` toggle); see commit `a16a5f2`. If decode wins
   without the fast path, the 1.24× microbench was misleading and we
   should revert the fast path entirely.

## Cluster State At End Of Session

- Cluster running on baseline code (mlx-lm `ac267339`, exo `9be275d5`)
- env: GAMMA=2, FENCE=43, TOPK=512, MTP=1, NO profile env
- Inference probe passes: "What is 2+2?" → "4"
- Production-ready for normal workloads

## Open Questions / Followups

- Where is the 9ms/cycle gap between profile-sum (62.65ms) and
  agg-tps-implied wall (68.9ms)? Tracing finalize() / span() overhead
  could give the next 1-2 t/s without algorithmic changes.
- Can the verify path's per-step latency be flattened (variance reduction)
  even if the mean is unchanged? At 30 t/s and 35% variance, P99 latency
  matters for tail-rate workloads.
- The May-17 champion claim (31.5 t/s at FENCE=8, MTP-G2) appears not
  reproducible at current HEAD. Worth bisecting between
  `champion-2026-05-17-mtp-g2-fenced-31.5` and today's `9be275d5` to
  find the +1.5 t/s regression source.
