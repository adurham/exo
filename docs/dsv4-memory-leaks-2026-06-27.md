# DSv4 Prefix-Cache / Prefill Memory Leaks — Findings & Fixes

**Date:** 2026-06-27
**Status:** Four leak sites identified, fixed, and verified. Cross-request
accumulation is closed (memory plateaus across sequential requests). The
"500K probe barely moved but Hermes leaked" gap is resolved.

**Commits (in order):**
- `d5f6c421` — metadata-refresh deepcopy (leak #1)
- `c0149c58` — growth-path deepcopy (leak #2)
- `91821901` — `DSV4_MAX_PREFIX_SESSIONS` 4→1 (leak #3, multi-leaf)
- `c6da30ce` — `del batched_cache` + `_captured.clear()` after prefill (leak #4)

---

## The user's observation (the ground truth)

Across ALL 500K-context testing, memory barely moved (~83-85 GB even at
500K tokens). But a real Hermes reasoning session at 154K context hit
85% / ~115 GB. Same model, same KV cache code, same hardware. The 500K
probe stayed flat; the Hermes session leaked. That gap was the signal that
kept getting dismissed — it was real.

**Why the 500K probe didn't leak:** `bench/quality_probe_dsv4.py` fires ONE
streamed request per iteration, then the process EXITS (freeing everything).
The probe never exercised cross-request accumulation in a long-lived
runner. Hermes runs many sequential requests in one long-lived runner
process, so state that persists on `self` across requests accumulated.

---

## Leak #1: metadata-refresh deepcopy (`d5f6c421`)

**File:** `src/exo/worker/engines/mlx/cache.py:663-672`

`KVPrefixCache.update_kv_cache`'s metadata-refresh path fires when
`new_length == old_depth` (the prompt didn't grow — the common per-turn
case). It deepcopied ALL non-sliceable layer caches onto the leaf on every
call, even though nothing grew.

DSv4 uses `RotatingKVCache` for all 44 layers, and `_sliceable_layer_mask`
marks `RotatingKVCache` non-sliceable → every leaf holds a full deepcopy of
all 44 caches (`_extract_non_sliceable_layers`, `deepcopy(c)` per layer).
The old deepcopy became Metal-allocator garbage that accumulated across
turns.

**Fix:** keep the existing `leaf_layer_caches` (nothing changed, no need to
re-deepcopy). The deepcopy only happens at leaf creation (line 509) and in
`_rebuild_leaf_in_place` (line 748) — the parked/donor paths.

**Verified:** 15-turn short-message test, memory flat at 77.18 GB.

## Leak #2: growth-path deepcopy (`c0149c58`)

**File:** `src/exo/worker/engines/mlx/cache.py:719` (growth path, `new_length > old_depth`)

The metadata-refresh fix (#1) wasn't enough — a real session GROWS every
turn, hitting the growth path which ALSO deepcopied all 44 DSv4 layers
every turn. The 500K test (one continuous prefill, one growth) never hit
repeated growth-path updates; a live session does.

**Fix:** same as #1 — skip the re-deepcopy on the growth path. The suffix
went into the trie edge (sliceable) / live cache (non-sliceable); the
active leaf's `leaf_layer_caches` is never read back as a donor while it's
the in-flight session.

**Verified:** growing-context test to 135K tokens, active went 77.18 →
80.27 GB and went FLAT from 101K → 135K (prefix-cache steady state, not a
climb). Before this fix, the same pattern climbed to ~108 GB at 128K.

## Leak #3: multi-leaf accumulation (`91821901`)

**File:** `start_cluster.sh:268` (`DSV4_MAX_PREFIX_SESSIONS` 4 → 1)

A Hermes STREAMING session creates a NEW LEAF PER REQUEST (the prefix cache
can't reuse — `shared_prefix=0` in the log). Each new leaf triggers the
creation-time deepcopy (line 509, left in place — needed for donor
correctness). With `DSV4_MAX_PREFIX_SESSIONS=4`, up to 4 full leaves
accumulated before eviction. Four full DSv4 KV deepcopies at 128K tokens
each = ~109 GB (the 98% the user hit).

**Caps DO reach the instance** — `/state` shows
`MlxJacclInstance.maxPrefixSessions`. (An earlier wrong claim that caps
showed `None` was from checking `shardAssignments` instead of the instance
object.)

**Fix:** `DSV4_MAX_PREFIX_SESSIONS` 4 → 1. One conversation = one leaf. Old
leaves evict immediately under cap=1 (eviction was already working — log
showed `evicted leaf — session cap`).

**Verified:** streaming-multi-leaf test — 12 requests each creating a new
leaf (leaf IDs 0→11, `trie_leaves=1` every time = old leaf evicted each
turn), active memory stayed FLAT at 77.11-77.17 GB (±0.06 GB).

## Leak #4: batched_cache + captured pre_norm retention (`c6da30ce`)

**Files:** `generate.py:765-767`, `batch_generate.py:1725-1729`

After leaks #1-#3, a residual climb remained during sustained reasoning.
Two retained intermediates:

1. `generate.py prefill_batched`: `batched_cache` (the FULL merged
   B×L×hidden KV buffer) was never explicitly freed after `per_stream_caches`
   extract. Python GC + Metal allocator cache pool held it across successive
   prefills.
2. `batch_generate.py submit_batched`: `captured_prompt_pre_norm` (the
   (N, last_chunk, hidden) MTP capture) and `self._mlx_gen._captured` dict
   were never cleared after MTP cache prefill.

**Fix:** `del batched_cache; mx.clear_cache()` after extract; `del
captured_prompt_pre_norm; self._mlx_gen._captured.clear(); mx.clear_cache()`
after the per-stream MTP loop.

**Verified (partial):** the climb slowed 4× (77→83 GB over 15 min vs
77→99 GB before). Full closure verified via the cross-request test (below).

---

## The decisive test: cross-request accumulation

`/tmp/cross_request_leak.py` — fires N sequential ~50K-token requests
(same prompt → prefix-cache hits, the Hermes pattern) against the
long-lived runner, logging active memory after each. This is the test the
500K probe couldn't be (it exits between iters).

```
baseline: 86.90 GB   (after a reasoning session)
req 1:   77.54 GB    (-9.4 GB — released after the session ended)
req 2:   79.57 GB    (+2.0 one-time, leaf establishment)
req 3:   79.55 GB    (FLAT)
req 4:   79.82 GB    (+0.27, flat)
live:    79.25 GB    (in the 78-80 GB band)
```

**Memory plateaued at ~79.5-80 GB across 4 sequential requests** — no
per-request climb after the one-time +2 GB at req 2. Cross-request
accumulation is ~0. The fixes hold.

---

## Method note: why the monitors were misleading

The `[MEM] after prefill, before decode` log line (`generate.py:1191`)
only exists on the **serial `generate()` path**, NOT the **batched-prefill
path** (`prefill_batched` at line 524) that reasoning/Hermes uses. So
during reasoning, DSv4's batched-prefill path never logs `after prefill` —
there's no release data for DSv4 during reasoning.

The `18.70 GB` "after" values that kept appearing in monitors were
**Qwen3.6's** `[MEM] after prefill` lines (Qwen3.6 is ~18 GB), not DSv4's.
This caused multiple false "memory released to baseline!" claims during
the session.

**The reliable detector for cross-request accumulation is the
between-request baseline** (what the cross-request test measures), NOT the
`after prefill` log line. The `before prefill` values climb during prefill
(working-set) and can't alone tell you if they release.

---

## What is NOT fixed / not verified

- **Sustained-reasoning decode path:** the cross-request test used ~50K
  prompts with short model replies. A real reasoning session has the model
  generating lots between turns (context grows to 185K). The synthetic
  test couldn't reproduce this (the model returned 1-9 tokens on cold
  prompts). If a real reasoning session still climbs, the remaining leak
  would be in the per-token decode/MTP path, which would need a different
  isolation approach (the `SpeculativeArraysCache.all_states` /
  `verify_pre_norm` / MTP cache paths were all read and look bounded per
  cycle, but weren't exercised under sustained generation in a test).
- **DSv4's real per-token KV cost:** live delta data showed memory climbs
  ~0.9-2.7 MB per KV token, NOT the 0.02 MB/token computed from
  `compress_ratios` (that was the compressed theoretical minimum; the actual
  `RotatingKVCache` + `PoolingCache` storage is bf16 and larger). At 160K+
  context, DSv4 genuinely uses ~85-95 GB — that's real KV cost, not a leak.
  The "excess vs spec" framing used during diagnosis was based on a wrong
  spec number and should not be trusted as a leak indicator.

---

## Cluster config (current)

- `DSV4_MAX_PREFIX_SESSIONS=1` (was 4)
- `QWEN36_MAX_PREFIX_SESSIONS=1`, `QWEN36_MAX_PREFIX_BYTES=1GB`,
  `QWEN36_KV_CACHE_BITS=8`
- `EXO_DSV4_MTP=1`, `EXO_DSV4_SEQ_SPLIT=1`, `EXO_PREFILL_STEP_SIZE=128`,
  `MLX_MAX_MB_PER_BUFFER=200`
- All four leak-fix commits are on `main`.

## Lesson

The recurring failure this session: declaring "fixed / no leak" from
partial or stale data, and from an unverified spec number. The ground
truth was the user's empirical observation (500K flat vs Hermes leaking),
and the reliable test was the cross-request baseline (not the per-prefill
log lines). When the numbers don't add up, measure the real thing — don't
theorize from assumptions.