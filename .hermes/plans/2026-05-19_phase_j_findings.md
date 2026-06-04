# Phase J Findings: Concurrency c=2 at 100K is catastrophic

**Date:** 2026-05-19 ~14:50 CDT
**Plan:** `.hermes/plans/2026-05-19_to_35tps.md` Phase J + MTP investigation

## Concurrency c=2 result

3-iter bench at TOPK=512 FENCE=43 gamma=2 MTP-on c=2:
- iter 0 warmup: **4.5 t/s aggregate** (2.3 per stream), wall 794s
- Bench killed at iter 1 due to extreme slowness; cluster wedged
- Required hard restart to recover

**c=2 at 100K context is unusable.** The "2.7x scaling" mentioned in
the DSv4MTPBatchGenerator docstring (line 341) is a relative claim vs
naive c=2 baseline, not vs c=1 single-stream. At 100K context, c=2
runs ~7x SLOWER per stream than c=1. The min-acceptance drag plus the
verify batching cost dominate.

**Conclusion: c>1 is not a path to 35 t/s.** Per-stream c=1 single-
threaded is what we're stuck with for long context.

## MTP / Larger Gamma Investigation

### What's already true

- gamma=2 at TOPK=512: alpha_2 = 1.04/2 = 0.52 → 2.04 tokens/cycle → 30 t/s
- gamma=3 at TOPK=512 (tested earlier today): alpha_3 = 1.12/3 = 0.37
  → 2.12 tokens/cycle → 26.1 t/s. Per-cycle cost grew faster than
  acceptance.
- The MTP head was trained for ONE-step prediction
  (num_nextn_predict_layers=1). Chained gamma>1 prediction compounds
  errors because the model was never trained for multi-step.

### Why raising gamma alone doesn't help

```
For gamma=N to beat gamma=2 in t/s:
  tokens_per_cycle(N) / cycle_cost(N) > tokens_per_cycle(2) / cycle_cost(2)
  
With:
  tokens_per_cycle(N) ≈ 1 + sum(alpha_i for i in 1..N)
  cycle_cost(N) ≈ 4.5*N (draft) + verify_with_L_q=(N+1) + accept_overhead
  verify_with_L_q=(N+1) ≈ 55ms (verify cost grows slowly with L_q at L_q<32)

For our measured alphas:
  gamma=2: (1 + 1.04) / 62.65ms = 32.6 t/s theoretical
  gamma=3: (1 + 1.12) / 72.17ms = 29.4 t/s — confirmed measurement
  
To hit gamma=3 break-even, alpha_3 would need to be:
  (1 + alpha_3) / 72ms = 32.6/1000  →  alpha_3 > 1.34 → 0.45 acceptance rate
  
We measure 0.37. Gap = 0.08. Not closeable without a better head.
```

### Real MTP-side levers (not "raise gamma")

**Lever M1: predict_from_hidden chain**
Current `draft_tokens` uses `predict(h, tok_arr, return_hidden=True)` which
embeds the predicted token. There's an alternative method
`predict_from_hidden(prev_hidden)` defined in mtp_module.py:536 but NEVER
CALLED. It skips the embed_tokens roundtrip and feeds the previous
normed hidden as if it were the embedding.

This is unproven; could either help (less info loss) or hurt (the MTP
wasn't trained for that input). Worth testing as a 2-3 hour
experiment: modify draft_tokens loop + bench gamma=2 + measure alpha_2.

**Lever M2: Token-tree drafting (eagle-style)**
Instead of chained gamma drafts, generate a TREE of drafts at each
step. At step i, take top-K predictions (K=2 or 3) instead of
argmax. Verify all 2^gamma branches in one batched forward. Accept
the longest matching branch.

Math at K=2, gamma=3:
- 8 draft sequences per cycle, verify input L_q=8*4=32
- verify_cost(L_q=32) ≈ ~60ms (slightly more than L_q=3=57ms)
- Effective alpha_branched can reach 0.6-0.7 (compounds across
  branches)
- tokens_per_cycle ≈ 1 + 0.7 + 0.49 + 0.34 = 2.5 tokens
- t/s = 2.5 / 0.065s = ~38 t/s

This IS plausible to hit 35+. But requires:
- mlx-lm code: modify draft_tokens to emit top-K + structured indices
- exo code: modify _speculative_next to handle tree verify + branch
  selection
- mlx-side: verify input must be shaped correctly for tree attn masks
- Quality probe must pass at 100K

Effort estimate: 2-3 days of focused work.

**Lever M3: Better MTP head training** — off limits this session.

**Lever M4: Re-evaluate gamma=2 alpha sensitivity to MTP fence**
The May-17 per-step `mx.eval(tok_arr)` fence was added to break
chained-collective stalls. It may also degrade alpha slightly by
forcing earlier graph materialization that biases the rng. Worth
verifying alpha is the same with/without (gamma=1 control).

## Recommended next move

Honest options at this point:

A. **MTP M1 experiment** (predict_from_hidden chain) — 2-3 hours.
   May or may not improve alpha. If improves enough, also test
   higher gamma with this chain.

B. **MTP M2 token-tree** — 2-3 days. The realistic structural path
   to 35+ t/s.

C. **Accept 30 t/s ceiling** and pick this up tomorrow with proper
   scoping for M2.

## Cluster state

Restored to production baseline (TOPK=512, FENCE=43, GAMMA=2, MTP=1,
no probes, no NOP targets). Inference probe passes.
