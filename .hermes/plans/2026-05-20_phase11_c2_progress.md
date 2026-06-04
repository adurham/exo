# Phase 11: c=2 100K MTP-on path to 35 agg t/s — progress + remaining gap

Status: **2.8x improvement landed (6.0 → 16.7 agg t/s), 47% of the 35 target.
Remaining gap is from c=2 cycle wall = 5x c=1 cycle wall instead of the
~1.5-2x expected from BS scaling.**

## Commits landed this session

| Commit | Change |
|--------|--------|
| cc200799 | KV bits revert (4→0) + per-uid MTP cache snapshots (snapshot_for_uid / activate_for_uids / drop_uid) |
| ef0485c0 | re-enable drain at TP c>1 + extend instead of clobber-assign in 3 buffer-write sites |
| 03a26443 | drain unconditional (broadcast already guarantees rank symmetry) |
| ade41bc3 | un-gate batched prefill for MTP-on, add per-stream MTP cache prefill in submit_batched |

## Numbers

| Config | per_req t/s | agg t/s | acceptance | comment |
|--------|-------------|---------|------------|---------|
| c=1 100K MTP-on (production) | 29 | 29 | 0.9/2 | baseline |
| c=2 100K MTP-on (pre-fix) | 2.9 | 6.0 | 0.15/2 | broken — all γ drafts clobbered + drain skipped + serial prefill |
| **c=2 100K MTP-on (post-fix, fresh cluster)** | **8.4** | **16.7** | **1.8/2** | drain re-enabled, cache extended, batched prefill |
| c=2 100K MTP-off (skill memory) | 16.2 | 32.4 | N/A | scales fine without MTP |
| Target | ~17.5 | 35 | ~1.8/2 | needs ~2x lift from current |

## Bugs found and fixed

### Bug 1: `DSV4_KV_CACHE_BITS=4` default (script)

start_cluster.sh:257 defaulted to 4-bit KV even though the comment block above said "leaving it bf16 until quant path validated". 4-bit forces SDPA out of the fused causal path at c>1 (BS=2 verify 168ms at 4-bit vs 43ms at bf16). Reverted to 0.

### Bug 2: MTP cache lifecycle assumed single-stream

batch_generate.submit() called `mtp.reset_cache()` UNCONDITIONALLY on every request, clobbering the shared cache when stream 2 arrived while stream 1 was still running. Stream 1's drafts then ran with stream 2's prefill state → 0% acceptance.

Fix: per-uid snapshots via `snapshot_for_uid(uid)` after prefill. `activate_for_uids(uids)` rebuilds the active batched cache at every BS-transition via `BatchRotatingKVCache.merge` of single-stream snapshots (or extracts from the live cache for incumbent uids). `drop_uid(uid)` on stream finish.

### Bug 3: drain disabled at TP c>1 + buffer assignment-clobber

The TP-c>1 drain skip was a pre-broadcast defensive measure. With the unconditional `broadcast_from_canonical(n_accepted_per + bonus_vals)` at line 1213 of `_speculative_next_batch` already in place, yielded tokens are bit-identical across ranks even at temp>0 — drain is safe.

Also the buffer write `self._token_buffer[uid] = deque(rest)` CLOBBERED any buffered tokens still pending from a prior cycle. Switched to `setdefault(uid, deque()).extend(rest)` in 3 sites (BS=1 spec, BS>1 spec, tree spec).

### Bug 4: batched prefill skipped for MTP

submit_batched fell back to per-task submit() when MTP was active, so c=2 streams prefilled SEQUENTIALLY (~6 min each = 12 min total). With both streams using the bench=True path's batched prefill via prefill_batched, total prefill is ~12 min for B=2 (instead of 12 for B=1 × 2 streams = 24 min). MTP cache prefill per stream uses slices of the batched final-chunk pre_norm.

## Remaining gap: c=2 cycle wall is 5x c=1

```
c=1 100K MTP-on: cycle ~65ms emitting 1.9 tokens (29 t/s per stream)
c=2 100K MTP-on: cycle ~333ms emitting 5.6 tokens across 2 streams (16.7 agg t/s)
ratio: 5.1x
```

Expected ratio at B=2 should be ~1.5-2x (per the MTP-off c=2 scaling pattern of 1.85x c=1 wall). At MTP-on we see 5.1x — the BS=2 forward itself is fine (verify=170ms vs ~62ms at c=1, ~2.7x as expected), but the c=2 path runs ~2x more `_next()` calls per emitted token.

### Why? Each `_next()` call yields 1 token from buffer drain (one uid at a time). For c=2 with γ=2 acceptance≈1.8, per spec cycle ~5.6 tokens land in buffers; draining them takes ~5 separate `_next()` calls at ~50ms each.

PROF only captures spec cycles (176ms wall). Drain cycles take ~50ms each due to per-`_next()` overhead — likely the `mx_any(local_has_work, coord_group)` collective at line 1694 of batch_generate.py firing on every server step, plus `on_generation_token` callbacks per response.

## Open levers to close the gap

1. **Drain MULTIPLE tokens per `_next()` call**: yield all buffered tokens for ALL uids in one call. Returns N×γ responses instead of 1. Cuts drain `_next()` count from γ×N to 1 per spec cycle. Expected lift: ~2x → 33 agg t/s.

2. **Yield all tokens directly from spec cycle**: instead of buffering γ tokens per uid for later, return N×(γ+1) responses immediately from `_speculative_next_batch`. Eliminates drain entirely. Same effect as #1 but cleaner code.

3. **Skip the `mx_any(has_work)` collective at drain cycles**: it's only meaningful when generation_batch is empty / changing. Add a fast-path that skips collectives when nothing has changed since last cycle.

4. **Increase `check_for_cancel_every`**: default 50 means agree_on_tasks fires every 50 tokens. At 256 tokens × 2 streams that's ~10 calls. Cheap but not free. Could bump to 200.

Recommended next step: **option 2** — refactor `_yield_buffered` to be eliminated; have `_speculative_next_batch` return all N×(γ+1) responses in one call. mlx-lm `BatchGenerator.next_generated()` returns a list, so multi-response returns are supported by the API.

## Cluster state at session end

- Last commit: ade41bc3
- mlx-lm HEAD: 8d7471c6
- Tag: tree-draft-2026-05-20-correctness-g2-K2-29.95 (still valid)
- Forensics: this file + 2026-05-20_phase10_mtp_c2_fix.md
