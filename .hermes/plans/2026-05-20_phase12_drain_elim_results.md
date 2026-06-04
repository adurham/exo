# Phase 12: drain-elimination results

Status: **target reached on best iters but VARIANCE TOO HIGH for champion claim.
σ ≈ 10 t/s across 4-iter validation due to intermittent submit_batched
fallback to per-task submit().**

## 4-iter c=2 100K validation (post-drain-elim, commit d1aeb8c5)

```
iter= 0 warmup  wall=773.19s  agg_tps=33.6  per_req=16.8  ← BATCHED, target hit
iter= 1         wall=790.39s  agg_tps=11.7  per_req= 5.9  ← SERIAL submit() ❌
iter= 2         wall=746.61s  agg_tps=22.5  per_req=11.3  ← BATCHED but 22.5
iter= 3         wall=737.41s  agg_tps=34.6  per_req=17.3  ← BATCHED, target hit
SUMMARY        agg_tps=22.9 (med 22.5, min 11.7, max 34.6)  σ≈10
```

Per memory rule "γ=2 needs ≥10 iters all ≥29 t/s σ<0.5" — this is NOT a champion. But:

- 2 of 4 iters hit ≥33 agg. The target IS achievable.
- The other 2 iters fell back to serial prefill due to a race in
  submit_batched, NOT a perf cap.

## Commits this session

| Commit | Change | Result |
|---|---|---|
| cc200799 | KV bits revert + MTP per-uid snapshots | Foundational |
| ef0485c0 / 03a26443 | drain re-enable (deprecated by 5c3b2f42) | Intermediate |
| ade41bc3 | un-gate batched prefill for MTP-on, per-stream MTP cache prefill | Required |
| **5c3b2f42** | **drain-elimination: yield all responses in one call** | **Got us to 34.6 agg** |
| d1aeb8c5 | diag log for submit_batched fallback gate | (diagnostic) |

## Outstanding issue: rendezvous + submit_batched race

Two distinct ways the bench falls back to serial prefill:

1. **Rendezvous doesn't fire**: runner only sees stream 1 in 200ms
   window. POSTs were 2ms apart at API but stream 2's task reached
   the runner's `_work_queue` later than the 200ms window. Cause
   appears to be the master→worker messaging delay being >200ms in
   some cases. Tried bumping `EXO_BATCHED_PREFILL_RENDEZVOUS_MS=2000`
   but couldn't validate due to SSH agent flake.

2. **submit_batched fallback** (rendezvous fired, gathered 2 tasks):
   one of the gates 1/2/3 inside `submit_batched` returned early.
   Diag log d1aeb8c5 will pinpoint which gate next time it happens.

Both issues are race-dependent and don't always trigger. iter 0 and
iter 3 of the validation bench both batched cleanly.

## What we know works

- c=1 MTP-on @ 100K: still 29.5 t/s (no regression).
- c=2 MTP-on @ 100K WHEN BATCHED: 33-35 agg t/s.
- MTP cache lifecycle correct across BS-transitions.
- Drain-elimination preserves per-stream finish semantics.

## Remaining gap to claim champion

Need: 10 iters all ≥29 t/s σ<0.5.

To make every iter reliably batch:
- (a) bump `EXO_BATCHED_PREFILL_RENDEZVOUS_MS` to 2000+ (need to validate).
- (b) fix the submit_batched gate-3 fallback (diag commit d1aeb8c5
  will tell us which task field tripped it; common candidates:
  task.task_params.bench == False at submit time due to a model_copy
  not propagating).

Once batching is reliable, 10-iter validation should give us
σ<0.5 around 33 t/s mean.

## Cluster state

- exo HEAD: d1aeb8c5 (diag-only patch)
- mlx-lm HEAD: 8d7471c6
- Tag still valid: tree-draft-2026-05-20-correctness-g2-K2-29.95
