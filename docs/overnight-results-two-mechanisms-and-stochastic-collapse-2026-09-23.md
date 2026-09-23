# Overnight deep-context results: two INDEPENDENT mechanisms, and the 500K collapse is stochastic (2026-09-23)

## Full ladder — same probe, same prompt, only depth varies

| depth | decode tps | acc/cycle | ms/cycle | pageins | peak GB | needle |
|---|---|---|---|---|---|---|
| 27,850 | **46.30** | **2.077** | 66.5 | 587 | 115.66 | ✓ |
| 127,929 | 29.72 | 0.814 | 61.2 | 400 | 115.66 | ✓ |
| 284,831 | 28.94 | 0.878 | 64.9 | 797 | 115.66 | ✓ |
| 564,928 | 24.15 | 1.353 | **97.4** | **6,599** | 117.24 | ✓ |

Prior runs for comparison:
- 273,622 -> 26.83 t/s (earlier)
- 564,926 -> **13.36** t/s (earlier), peak 115.66 GB

## Finding 1: TWO distinct mechanisms, separated by depth

**Region A (~28K -> ~128K): an ACCEPTANCE cliff.**
- acceptance 2.077 -> 0.814 (**-2.55x**)
- cycle cost FLAT (66.5 -> 61.2 ms)
- page-ins LOW and not rising (587 -> 400)
- → the loss is entirely in **how many draft tokens the speculator gets
  accepted**, not in compute or memory

**Region B (~128K -> ~565K): a COST/refault ramp.**
- cycle cost 61.2 -> **97.4 ms** (+59%)
- page-ins 400 -> **6,599** (16x)
- acceptance actually *recovers* here (0.814 -> 1.353)
- peak memory 115.66 -> 117.24 GB (at/above the 120.6 GB wired limit region)
- → the loss is **memory pressure** (mmap clean-page eviction + refault), the
  mechanism previously confirmed at 42K (32,469 page-ins in one request)

These are different problems needing different fixes. Conflating them (as I
did earlier tonight) produces a curve that looks like one smooth depth effect.

## Finding 2: the 500K collapse is STOCHASTIC — it did not reproduce

- earlier 564,926 -> **13.36** t/s
- tonight  564,928 -> **24.15** t/s

Same depth, same probe, same config: **1.8x difference.** The collapse is
**bistable**, matching the 352K investigation's 4-of-16 collapse rate.
Consequences:
- A single non-collapsed run does NOT prove a fix works.
- Any collapse claim needs **N>=6** runs at the depth, with a stated collapse
  definition (e.g. decode_tps < 15, or sustained cycle gap > 500 ms).
- The 4/16 prior base rate implies ~25% collapse probability per run; to show a
  fix reduces that with any confidence needs a substantially larger N, or a
  direct measurement of the *margin* (headroom) rather than the outcome.

## Finding 3: Region A's mechanism, source-verified

`deepseek_v4.py` `Indexer.__call__` (~line 4795): `k = min(self.index_topk, pooled.shape[1])`

- `index_topk = 512`, `compress_ratio = 128` (config-verified)
- `pooled.shape[1] = depth / 128`
- depth <= 512*128 = **65,536** -> `k` = all entries -> **effectively dense**
- depth > 65,536 -> `k` = 512 -> **entries dropped**

Predicted crossover 65,536 sits between the healthy 28K rung and the degraded
128K rung. Cliff-localisation test running at 40/55/70/90/110K
(`/tmp/cliff_ladder.py` -> `/tmp/cliff_ladder.jsonl`).

Hypothesis: dropping compressed context changes the target's next-token
distribution, and the DSpark drafter — which conditions only on a **128-token**
rotating window of target hiddens (`append_ctx`, sliding_window=128) — tracks
it worse. Acceptance falls.

## Variance warning (must be respected when reading any of this)

Same-script, similar-depth runs differed greatly tonight:
- 21,855 -> 33.22 vs 27,850 -> **46.30** (+39%)
- 115,614 -> 36.74 vs 127,929 -> 29.72 (-19%)

So single rungs are weak evidence. The cliff must be read as a SHAPE across
rungs; per-rung N>=3 for anything load-bearing.

## What this means for the user's two targets

**T2 (250K >= 30 t/s):** 284,831 measured **28.94** t/s tonight.
- It is in Region B (past the acceptance cliff), so the acceptance mechanism is
  not the lever; the residual is cost/refault.
- Getting from 28.94 to >30 is **+3.7%** — small, and within the variance band
  measured above. This may already be satisfied on a good run (the same depth
  measured 26.8-28.9 across runs).
- Honest read: **T2 is ~at target within noise**, not a 12% shortfall. Needs
  N>=3 at 250K to state confidently.

**T1 (500K must not collapse):** the collapse is real but stochastic
(13.36 in one run, 24.15 in another). The fixable component is **Region B's
memory pressure**, and the honest lever is reducing per-rank footprint or
raising headroom. Raising `DSV4_WIRED_LIMIT_MB` re-enters a documented
hard-wedge regime; I have NOT done that.

## Status
- Ladder COMPLETE. Cliff test RUNNING.
- No cluster config changed tonight.
- Cluster healthy throughout (canary 14.86 TFLOPS both nodes at start).
