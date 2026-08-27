# DSpark draft-epilogue fusion (TP path) — design + implementation — 2026-08-27

**STATUS: code-only implementation (cluster untouched — 352.6K protocol running).
Env-gated `EXO_DSV4_DRAFT_EPILOGUE=1`, default OFF. Shipped to main; cluster
A/B validation deferred until the 352.6K protocol completes.**

## The optimization

Move the DSpark draft forward off the per-cycle critical path by computing
the **next** cycle's draft in the **current** cycle's epilogue (after the
DSpark ctx feed and the bonus token are known) rather than at the **start**
of the next cycle. The next cycle then consumes the pre-computed draft
without paying the ~10.8 ms `_dspark.draft()` block-forward + Markov-loop
cost on its critical path — that work overlapped with the prior cycle's
accept/rollback/bookkeeping tail instead of serializing before the next
verify.

This mirrors the PP path's existing implementation
(`pp_speculation.py` ~line 2952: after `append_ctx`, call
`_dspark.draft(bonus_token, ...)` and stash the result for the next cycle
to consume). The TP path (`dsv4_mtp.py::_speculative_next`) previously
computed the draft at cycle start (line ~3881); this change adds the
epilogue computation and a consume-or-compute path.

## Scoping findings (what the next-cycle draft depends on)

The DSpark draft head (`DeepseekV4DSparkModule.draft`, `deepseek_v4.py:6524`)
takes three inputs and produces three outputs:

**Inputs:**
1. `anchor_tokens` (B,) — the committed token `y` from the last cycle
   (= the previous cycle's `bonus_val`).
2. `embed_tokens`, `lm_head` — model weights (static).
3. `caches` — the DSpark ctx-KV caches (3 `RotatingKVCache`, one per stage),
   populated by the **previous** cycle's `append_ctx`.

**Outputs:** `draft_tokens` (B, block_size), `corrected` (B, block_size, V),
`confidence` (B, block_size).

**Process:** one parallel block forward over `[anchor, noise×(bs-1)]`
through 3 stages → `base_logits` (B, bs, V) for all block_size positions at
once; then a **sequential Markov loop**: `prev = anchor`, for k in 0..bs:
`step_logits_k = base_logits[:,k,:] + markov_w2(markov_w1(prev))`, sample
`nxt_k`, `prev = nxt_k`. The block forward writes block KV into the stage
caches; the caller trims by `width` afterwards so only the ctx-KV persists.

### Markov sequentiality verdict

The Markov loop IS sequential **within** a single `draft()` call (position
k's logits depend on position k-1's sampled token). But **no state
persists across cycles** except `_dspark_caches` (the ctx-KV). The Markov
state is fully recomputed each `draft()` call from the anchor token. So the
sequentiality does NOT block epilogue fusion — the entire next draft (all
gamma positions) is computable in the epilogue from `(bonus_val,
_dspark_caches)`.

### What the next-cycle draft depends on

1. **The anchor token `y`** = the previous cycle's `bonus_val`. This is
   known after the accept step and finalized (cross-rank broadcast) before
   the epilogue. ✅ available in the epilogue.
2. **The DSpark ctx-KV caches** (`_dspark_caches`) — populated by
   `append_ctx` (the existing ctx-feed step at the epilogue). ✅ available
   immediately after `append_ctx` runs.
3. **The block forward + Markov loop** — self-contained within one
   `draft()` call; no external state. ✅ computable in the epilogue.

The draft does **NOT** depend on:
- `_mtp_pre_norm` (that's the MTP-chain path only; DSpark uses the anchor
  token + ctx-KV, not a hidden state).
- The target's prompt-cache / KV caches (rollback trims those; they don't
  feed the DSpark draft).
- The bonus staging (`gen_batch._next_tokens` is set AFTER the epilogue;
  the epilogue reads `bonus_val` directly).

### Conclusion: full fusion is sound

The entire next draft (all gamma positions) is computable in the epilogue.
No partial fusion is needed.

## The one ordering hazard: tie-reverify

`EXO_DSV4_MTP_TIE_REVERIFY` (default OFF, **RETIRED FROM PROD**
2026-07-10) can mutate `bonus_val` AFTER the epilogue via a clean
re-forward (step 6, line ~4986). A pre-computed draft anchored on the
pre-reverify `bonus_val` would then be wrong.

**Guard:** when both `EXO_DSV4_DRAFT_EPILOGUE=1` and tie-reverify are ON,
the epilogue caches the draft with the pre-reverify `bonus_val`, and if
tie-reverify fires (`_tie_reverify == 1`) the cached draft is invalidated
(`pop`'d) so the next cycle recomputes from scratch. In production
(tie-reverify OFF) this branch is dead code. The consumer also validates
`anchor_tok == y_val` as a belt-and-suspenders check.

## What shipped

All in `src/exo/worker/engines/mlx/speculative/dsv4_mtp.py`, single-uid
DSpark branch of `_speculative_next` only (the MTP-chain fallback and the
BS>1 batched path are untouched):

1. **Env flag** `_DRAFT_EPILOGUE` (line ~404): `EXO_DSV4_DRAFT_EPILOGUE=1`,
   default OFF, read once at import.
2. **Instance state** `_dspark_next_draft` dict (line ~1625): per-uid cache
   of `(draft_ids, draft_probs, corrected, conf, gamma, anchor_tok)`.
3. **Consume path** (line ~3958): at cycle start, if a cached draft exists
   for this uid with matching gamma and `anchor_tok == y_val`, unpack it
   and skip the `_dspark.draft()` call. The `_dspark_sample` closure was
   moved before the `if _dspark is not None:` block so both the inline
   draft and the epilogue draft can pass it to `draft()`.
4. **Epilogue draft** (line ~4639, right after `append_ctx`): compute
   `_dspark.draft(bonus_val, ...)` for the next cycle, `mx.eval` the
   tokens, trim the block KV, stash the result keyed by uid.
5. **Tie-reverify invalidation** (line ~5180): if tie-reverify fires,
   `pop` the cached draft (correctness over speed).
6. **Cleanup** (line ~1846): `_filter_finished_uid` drops the cached draft
   so it doesn't leak into a recycled uid's first cycle.

### Fallback / zero-behavior-change guarantees

- First cycle (no cached draft): inline `draft()` at cycle start —
  identical to the pre-change path.
- `EXO_DSV4_DRAFT_EPILOGUE` unset (default): `_DRAFT_EPILOGUE` is `False`,
  the consume path's `_cached` is always `None`, the epilogue block is
  skipped — bit-identical to the pre-change code.
- Gamma mismatch / anchor mismatch / tie-reverify: the cache is dropped
  and the next cycle recomputes inline.

## Expected win

Per-step @100K (current): draft 10.8 ms, verify 60.6 ms, accept+rollback
~2.3 ms, bookkeeping ~8 ms, cycle 73.7 ms. With full fusion the draft
forward (10.8 ms) moves off the critical path — it overlaps with the
prior cycle's accept+rollback+bookkeeping tail (~10.3 ms), so the net
per-cycle saving is ~10.8 ms minus the overlap, i.e. the draft becomes
~free when the epilogue work fits in the tail. Theoretical cycle time:
73.7 → ~62.9 ms, C_s 2.14 → ~2.51, throughput 36.6 → ~42.5 tok/s
(+16%). The real win depends on how much of the 10.8 ms draft overlaps
with the ~10.3 ms tail on the actual hardware (Metal command-buffer
overlap, RDMA collective timing); the cluster A/B will measure the
realized fraction.

## Cluster validation (deferred — 352.6K protocol running)

After the 352.6K protocol completes:

1. **Byte-identity gate**: with `EXO_DSV4_DRAFT_EPILOGUE=1` + shadow mode
   (`EXO_DSV4_SPEC_SHADOW=1`), output must be byte-identical to the
   shadow-OFF path (the consume path produces the same draft tokens as
   the inline path would, since `_dspark.draft()` is deterministic given
   the same anchor + ctx-KV). Run the ldiff / REFCHECK harness.
2. **A/B throughput**: 24-run paired comparison @100K (and @352.6K once
   that point exists) of `EXO_DSV4_DRAFT_EPILOGUE=1` vs `=0`, measuring
   per-cycle `draft_ms` (should drop to ~0 on consume cycles),
   `cycle_ms`, `C_s`, acceptance (should be unchanged — the draft is
   identical, just computed earlier), and aggregate tok/s.
3. **Overlap measurement**: use the phase timer (`EXO_DSV4_MTP_PROFILE=1`)
   to confirm the epilogue draft's wall time overlaps with the
   accept/rollback/bookkeeping tail (the `draft` phase record on a
   consume cycle should read ~0; the epilogue cost should show up in a
   new `epilogue_draft` phase or be attributable to the `total` minus
   the sum of the other phases).
4. **No-regression on first cycle / uid recycle**: confirm the fallback
   (inline draft on the first cycle of each stream, and after any
   invalidation) produces identical acceptance to the baseline.

## Gates passed (laptop, code-only)

- `ruff check`: 20 issues before, 20 after — identical error codes, zero
  new issues in edited line ranges.
- `basedpyright`: 734 errors (735 before removing one unnecessary
  `# type: ignore` I initially added). All errors in edited ranges are
  `reportAny` / `reportUnknownMemberType` — pre-existing classes
  unavoidable when calling the untyped `_dspark.draft()` /
  `self.model.model.embed_tokens` pattern the existing code already
  uses everywhere. Zero new error types.
- `pytest` (scoped): `test_pp_spec_gen_by_uid.py` 8/8 pass. The one
  failure in `test_pp_speculation_cache_snapshot.py` is **pre-existing**
  (confirmed by running it against clean HEAD `dsv4_mtp.py` — same
  failure; a stale test vs the mlx-lm submodule's 5-tuple
  `PoolingCache.state` shape), unrelated to this change.
- Module import: clean, `_DRAFT_EPILOGUE` defaults `False`.

## NOT verified (cluster-only)

- Real per-cycle timing (the 10.8 → ~0 ms draft drop on consume cycles).
- The actual overlap fraction of the epilogue draft with the tail.
- Acceptance / byte-identity under live TP=2 load.
- Behavior at 352.6K context (the running protocol's regime).