# Phase 10: c>=2 MTP cache fix — status update

Status: **MTP cache lifecycle bug FIXED in commit cc200799. KV-bits
default reverted from 4 → 0 (bf16). But c=2 100K bench still at 5.7
agg t/s. Verify cost at B=2 long-context dominates and isn't addressed
by either fix.**

## What was fixed (committed cc200799)

1. **DSV4_KV_CACHE_BITS default**: 4 → 0 (bf16). The script's comment
   block already said "leaving it bf16 until we confirm the quant path
   round-trips cleanly" but the code defaulted to 4. Per skill canonical
   constraint, prod is bf16.

2. **MTP cache lifecycle for c>=2**:
   - Added `DSv4MTPPredictor.snapshot_for_uid(uid)` to stash the
     just-prefilled single-stream cache.
   - Added `activate_for_uids(uids)` which extracts each remaining uid's
     state from the live cache + merges in join-time snapshots for
     newcomers via `BatchRotatingKVCache.merge`.
   - Added `drop_uid(uid)` for stream finish.
   - Wired in: `batch_generate.submit()` snapshots after prefill;
     `_speculative_next._next()` calls `activate_for_uids` before
     dispatch; `_filter_finished_uid` calls `drop_uid`.

   Verified the fix WORKS: at B=2 the draft phase went from
   **147ms (max 7827!) to 4.81ms**, a 30x improvement. cycle wall
   halved (325ms → 173ms).

## What's still broken

Bench c=2 at 100K context still shows **5.7 agg t/s** (vs ~25 expected
if it scaled like MTP-OFF c=2's 1.85x wall). The bottleneck is now
the VERIFY forward at B=2 long context (167ms vs 51ms at B=1), not the
MTP draft.

Small prompt (500 words) c=2 hits 20 agg t/s = roughly same as c=1
20.2 t/s. So c=2 throws away the parallelism at small prompt too —
just less catastrophically. The verify-cost-at-B=2 issue is general.

## Possible verify-cost regression sources at B=2 long context

1. **4D mask path for SDPA at B>1**: PerStreamBatchRotatingKVCache
   emits a 4D `(B,1,L_q,kv)` mask vs RotatingKVCache's 2D `(L_q,kv)`
   mask. The 4D mask forces SDPA out of its fused causal kernel into
   the slower explicit-mask path. Mitigation: detect uniform
   right_padding=0 (all streams same offset) and emit 2D mask.

2. **BatchPoolingCache compute overhead** at B>1 with long context.
   The pool cache is shared across streams; ratio=4 layers have
   ~25K pool entries at 100K context × 2 streams = 50K entries
   sloshing through SDPA per cycle.

3. **all_sum collective volume doubles** at B=2 because each rank
   sends 2-stream activations. With 43 layers × fence-every-43 the
   final layer's all_sum is 2x the per-token cost.

## Decision options

This is now Phase-11 territory — a structural perf investigation for
B=2 long-context verify cost. Not a 1-day fix.

**Option A: Lock in the cc200799 fix as a correctness milestone**
(MTP+c=2 no longer broken to 5.8 from a cache bug; it's just bandwidth-
bound). Defer the verify-cost-B=2 work. Production c=2 with MTP-on
will still be slower than c=2 MTP-off (5.7 vs 31.4) because verify
dominates with MTP. Not at 35 yet.

**Option B: Investigate the verify-cost-B=2 regression directly**.
- Bench c=2 MTP-OFF (just decode at L_q=1 BS=2) at 100K to get the
  verify-only B=2 floor.
- If MTP-OFF c=2 verify scales well (which we know it does, 1.85x),
  the regression IS MTP-specific (L_q=3 BS=2 vs L_q=1 BS=2).
- Look at SparseCompressedAttention + Indexer at BS>1 long context.
  Maybe their batch dispatch goes serial.

**Option C: Skip MTP for c>=2; use MTP-OFF c=2 ≈ 31.4 + some other
lever to add 3-4 t/s**.
- MTP-OFF c=2 31.4 + Plan A Eagle soft-emb (which works at MTP-OFF? no,
  needs MTP) → not viable.
- MTP-OFF c=2 31.4 + nothing → 31.4 t/s. Close but not 35.

**Option D: Reduce verify cost via index_topk** (Plan C.1 from Phase
8 plan). At 100K, TOPK=512 → ~25K sparse indices to score. TOPK=384
would cut sparse-attention cost by ~25%, which could lift MTP c=2.
Risk: quality regression at 100K (per memory pitfall, TOPK=160 was
needle-broken; TOPK=384 might be safe).

## Recommendation

If 35 t/s c=2 100K is the hard target, **Option B** (investigate B=2
verify) is the most direct path. Effort: 2-3 days, structural diving
into mlx-lm SDPA + sparse attention at BS>1.

Alternative path: lock cc200799 as correctness fix, ship 31.4 t/s
agg c=2 MTP-OFF as the new prod, declare 35 t/s requires either Plan
A (Eagle) — back to single-stream MTP — OR fixing BS>1 SDPA path.

## Open question

Should we pursue **Option B** (B=2 verify cost investigation, 2-3
days, uncertain outcome) or **Option C/D** (~30 t/s aggregate ceiling
at c=2, ship correctness fix, document gap)?
