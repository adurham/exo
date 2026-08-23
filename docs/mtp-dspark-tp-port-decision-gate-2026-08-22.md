# T6: MTP/DSpark TP-port decision gate — real ceiling estimate is 1.29x (below the 1.5x threshold), AND the PP+DSpark comparison baseline is 5 weeks stale — MTP port NOT recommended as next priority — 2026-08-22 (session 4)

## Why this check

Per the Fable-provided plan, T6 is a decision gate, not an
implementation task: "after T2/T3, if the realistic single-token
ceiling (roofline adjusted for measured achieved bandwidth/clock, minus
irreducible collective cost) is <~1.5x current throughput, MTP becomes
the highest-EV remaining item and gets scoped. Until then, untouched."

## Part 1: realistic kernel-fix-only ceiling (T2/T3 synthesis)

Using T3's real, measured `switch_mlp` finding (27.7% of peak bandwidth,
38.9% of decode wall time) and assuming an optimistic-but-plausible fix
brought it to 65% of peak (roughly matching the healthier attention
kernels' 83-85% efficiency scaled down for MoE's structurally harder
sparse-gather access pattern, per T3's own candidate-cause discussion —
NOT itself validated as achievable, a ceiling estimate only):

```
current decode wall: 34.25 ms/token (29.2 tok/s, T1's post-fix baseline)
switch_mlp share: 13.32 ms (38.9%)
IF switch_mlp reached 65% of peak (optimistic, unvalidated): 5.68 ms
potential savings: 7.64 ms/token
realistic ceiling (kernel-fix-only): 26.60 ms/token = 37.6 tok/s
ratio to current: 1.29x
```

**1.29x is BELOW the plan's 1.5x threshold** — read literally, this
would flag MTP as the highest-EV remaining item. But this estimate is
narrow (single-kernel-fix-only) and doesn't account for other possible
gains (T10's un-decomposed prefill remainder doesn't apply to decode;
no equivalent decode-side "un-decomposed remainder" investigation has
been done this session beyond T1-T5). Treat 1.29x as a lower-bound
estimate of the non-MTP ceiling, not a tight one.

## Part 2: the MTP/DSpark comparison itself is 5 weeks stale — a real, previously-unflagged problem

Before treating "MTP could reach 27-33 tok/s" as the alternative to
compare against, checked the actual source of that number
(`docs/fork-notes.md`, dated 2026-07-31, citing a "2026-07-23 sweep"):
**PP+DSpark speculative decode measured 27-33 tok/s single-request**,
compared against a TP baseline of "~15-20 tok/s" at the time.

**This TP baseline is now known to be wrong/stale**: it predates BOTH
the MoE gate+up fusion (+3.01%, 2026-08-21) AND — far more
significantly — the async-fence fix (+58-67%, 2026-08-22) that is this
entire session's headline finding. TP's REAL current single-request
throughput is **29.2-31.1 tok/s** (T1), not 15-20 tok/s.

**Redone honestly**: PP+DSpark's 27-33 tok/s figure, if it reproduced
completely unchanged today, would represent only a **-8% to +13%**
difference vs current TP — a dramatically smaller (and partially
NEGATIVE) advantage than the 5-week-old doc's framing implied (+35% to
+120% vs the stale TP number). **This is a genuinely important,
previously-unflagged finding**: the case for "PP+DSpark is clearly
faster, worth porting" rests entirely on a comparison baseline that is
now obsolete. Nobody has re-measured PP+DSpark itself since 2026-07-23,
so it's equally possible PP+DSpark has its own undiscovered wins
sitting in it (the same class of bug the async-fence fix was) as it is
that TP has now caught up or surpassed it.

## Decision (per the plan's stated gate, applied honestly)

**MTP/DSpark TP-port is NOT recommended as the next priority**, for two
independent reasons, either of which alone would be sufficient:

1. The narrow kernel-fix-only ceiling estimate (1.29x) is close to but
   below the 1.5x gate threshold, but is itself an incomplete estimate
   (single-kernel-only, doesn't yet incorporate any decode-side
   equivalent of T10's un-decomposed-remainder investigation).
2. **The comparison baseline motivating an MTP port at all (the
   "PP+DSpark reaches 27-33 tok/s" figure) is 5 weeks stale and
   directly contradicted by intervening work in this very session** —
   it was measured against a TP baseline that the async-fence fix has
   since raised by 58-67%. Porting MTP to TP on the strength of a
   stale comparison would be a real risk of chasing a gap that may no
   longer exist, before the comparison itself is re-validated.

**Recommended next step, if MTP/DSpark work is pursued at all**: before
any porting effort, re-run the PP+DSpark 2026-07-23 sweep AS-IS (no
porting, just re-measurement) against the current cluster state, to
get an honest apples-to-apples number. This is cheap (no new
engineering, just a bench re-run) and would immediately resolve whether
the "MTP is worth porting to TP" premise still holds at all, before
committing to the multi-day TP-native speculative-decode engineering
effort implied by a real port.

## What this does NOT establish

Does not implement or scope the actual TP-native MTP/DSpark port work
(explicitly out of scope for a gate-only task, per the plan). Does not
re-run the PP+DSpark sweep itself this session (flagged as the correct
next step if this line is pursued, not executed here — would require
switching `DSV4_SHARDING=Pipeline` and accepting PP's concurrency
limitations for the duration of the test, a real cluster-config change
warranting its own explicit go-ahead). Does not invalidate T3's real
switch_mlp finding or T7's prefill roofline work — those stand
independently of this MTP-specific question.

## Standing recommendation update

Given both the modest (1.29x, itself likely optimistic) kernel-fix
ceiling AND the newly-flagged staleness of the MTP comparison baseline,
**T10 (decompose prefill's 28.8% non-GEMM remainder, real quantified
1.40x headroom, comparable magnitude to the async-fence win) remains
the single highest-expected-value item on the list** — it targets a
concrete, already-measured gap using a methodology (per-op wall-time
attribution) that has a proven track record this session (it's exactly
how the async-fence bug itself was found). MTP/DSpark should not be
prioritized ahead of it without first doing the cheap re-validation
sweep flagged above.
