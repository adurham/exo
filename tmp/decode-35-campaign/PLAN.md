# Decode throughput campaign — 28 → 35-40 tok/s (started 2026-09-22)

## Baseline (healthy post-reboot cluster, verified)

`bench/long_decode_probe.py 4000 --max-tokens 900-1000`, needle verified:

| metric | value |
|---|---|
| decode_tps | **27.8 - 28.2** |
| accepted drafts/cycle | **1.39** |
| tokens/cycle | **0.60** |
| sec/cycle | **21.2 ms** |
| gamma | 3 |
| quality | needle_hit=true on every run |

Identity: tok/s = tokens_per_cycle / sec_per_cycle
  = 0.60 / 0.0212 = 28.3 ✓

## What 35-40 tok/s requires

Either (a) acceptance up, or (b) cycles cheaper. Concretely:

- **35 tok/s** = +24% over baseline
  - acceptance-only: tokens/cycle 0.60 → 0.74 (+24%)
  - OR cycle-cost-only: 21.2 ms → 17.1 ms (−19%)
- **40 tok/s** = +42%
  - acceptance-only: tokens/cycle 0.60 → 0.85
  - OR cycle-cost-only: 21.2 ms → 14.9 ms (−30%)

## Pre-registered bands (BEFORE measuring, per standing rule)

- **PROMOTE**: ≥ +8% over baseline AND needle_hit=true AND finish_reason clean
- **REJECT**: ≤ +2% (within boot-to-boot spread)
- **INCONCLUSIVE**: between → needs a second paired run
- Boot-to-boot decode spread on this cluster historically ~1.3-6 t/s; a
  single-arm result inside ±2 t/s of baseline is NOT evidence.

## Levers, ranked by (a) not-yet-measured, (b) decode-specific, (c) quality-safe

### L1. `EXO_DSV4_MTP_EAGLE_T` sweep — PRIMARY (never swept)
Shipped as commit `ace0259a` ("soft-emb logit temperature for acceptance")
but never A/B'd; docs reference only a plan file that isn't in the repo.
- Mechanism: divides draft logits by T before top-K softmax → T<1 sharpens
  the mixture onto top-1, matching the hard embed the MTP head was trained on.
- Code path: `mtp_module.py:799,846` — K>1 branch only; K=8 is live.
- **Quality-safe**: draft-only. Downstream RMSNorm normalizes magnitude
  ("effect is purely directional"); verify/accept side untouched, so at
  temp=0 accepted tokens remain the target's own argmax. Only the
  acceptance RATE can move. This is the rejection-sampling property.
- Sweep: T ∈ {1.0 (baseline), 0.7, 0.5, 0.3}

### L2. `EXO_DSV4_MTP_DRAFT_LMHEAD_BITS=4` — conditional-by-context
MEASURED 2026-07-06, rolled back: **+4% at 4K, −8.7% at 586K** (sign flips
with context). Doc itself proposes the refinement: enable only below ~64K.
- Since our probe depth is 4K, this is squarely in its positive regime.
- But it's a per-boot env var, not context-adaptive — enabling it globally
  would help short-context and hurt long-context.
- Decision: measure at 4K to confirm the +4%; only worth shipping if made
  context-conditional (needs a code change + its own validation).

### L3. Cycle-cost side (21.2 ms/cycle) — largely closed
- I3 kernel bandwidth: 83.9% of peak (closed)
- I1 TP all_sum: 2.6% of per-layer budget (closed)
- I5 gamma re-tune: γ=4 is −4.5 t/s; cycle 61→74ms monotonic in γ (closed, HOLD γ=3)
- verify-batch: G0 gate FAILED (shape mismatch), reverted — do NOT reopen
- P14: the "33-41% MoE efficiency" figure was a measurement artifact;
  corrected to ~59-64%

## Known traps (from the record — do not re-learn these)

1. **NEVER quote t/s from generations under ~400 tokens** (startup-dominated).
2. **Forced `mx.eval` at phase boundaries serializes async pipelining** —
   the source of most retracted numbers in this repo (MTP-PROF's own code
   comment says its numbers are "upper bounds on real production walls").
3. **Lazy eval**: if you don't eval the copy, you measured nothing.
   (This is how the bogus "778 GB/s" happened.)
4. **Quality gate is mandatory**: needle_hit must be true. A speedup that
   breaks output is a regression.
5. Boot-to-boot spread ~1.3-6 t/s → paired/interleaved comparisons only.

## Status

- [x] Baseline established (28.2 tok/s, needle verified)
- [ ] L1 EAGLE_T sweep (T=1.0 / 0.7 / 0.5 / 0.3)
- [ ] L2 DRAFT_LMHEAD_BITS=4 at 4K
- [ ] Decide ship/reject against bands
