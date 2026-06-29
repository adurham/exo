# DSv4-Flash Multi-Turn Memory Leak — Resolution Post-Mortem

**Date:** 2026-06-29
**Status:** RESOLVED — root-caused, fixed, unit-tested, and confirmed on hardware.
**Fix commit:** `947d7e50b` (`fix(dsv4): stop multi-turn PoolingCache snapshot leak in KVPrefixCache`)
**Files:** `src/exo/worker/engines/mlx/cache.py`,
`src/exo/worker/tests/unittests/test_mlx/test_kv_prefix_cache.py`

---

## 1. Symptom (ground truth, user-confirmed)

A single growing Hermes conversation on DSv4-Flash climbed in DSv4-runner active
memory and never released:

- Measured session (139 msgs, 68 API calls, ctx 16K→105K): DSv4 runner active
  memory went **77.13 GB → 106.61 GB = +29.5 GB**, footprint peak 108 GB.
- **~0.2–0.4 GB leaked per TURN**, monotonic, never freed.
- Survived **idle** (idle-reclaim ran, `cache=0.00GB`) AND **session end** — at
  idle post-session the runner stayed pinned at 106.65 GB active. A hard
  pinned-reference leak, not reclaimable Metal pool.

Ruled out by measurement before the fix: not Qwen (separate process), not KV
cache cost (~1 GB at 31K, on-spec), not the reclaimable Metal allocator pool,
not the prefix-cache trie token storage.

## 2. The leaking objects

gc heap-census instrumentation (`EXO_DSV4_HEAPCENSUS=1`,
`_heap_census_mx_arrays()` in `generator/generate.py`) showed:

- Big arrays (≥16 MB) FLAT at count 400 — the static MoE expert weights. Not
  leaking.
- The leak = small bf16 tensors of shape `(1, P, 512)` and `(1, P, 128)`
  (P = pooled length, ~7936 at 31K ctx) growing by **exactly +21 per turn**.
- +21/turn == the number of DSv4 sparse-attention layers (compress_ratio=128;
  config has 43 layers alternating [4,128] ⇒ 21 of compress_ratio 128).
- Total `mx.array` count climbed ~+272/turn; bf16 pooled GB ~+0.2/turn.

These tensors are `PoolingCache` arrays (DSv4 sparse-attention pooled KV). The
trace-to-root referrer chain decodes exactly:

```
(1, P, 512) bf16
  -> tuple            (= CacheList.caches, which is a tuple)
  -> CacheList
  -> list             (= CacheSnapshot.states, a list[per-layer state])
  -> list -> list ... (= the growing per-leaf snapshot lists)
```

## 3. Root cause

### Why DSv4 hit the snapshot path at all

`PoolingCache.is_trimmable()` returns `self.pooled is None`. Once a layer has
pooled entries (any real context length), the whole DSv4
`CacheList(RotatingKVCache, PoolingCache, PoolingCache)` is **non-trimmable** ⇒
`has_non_kv_caches()` is True for DSv4 ⇒ the snapshot/restore machinery in
`KVPrefixCache` is live. (The code comments implied this path was SSM-only —
it is not.)

### Two never-pruned `CacheSnapshot` accumulators

Both in `src/exo/worker/engines/mlx/cache.py`. Each pinned one full 21-layer
PoolingCache set per turn ⇒ the measured +21/turn.

**(1) `update_kv_cache` snapshot merge — the primary leak.**

```python
merged = [s for s in leaf.leaf_snapshots if s.token_count <= restore_pos]
merged.extend(snapshots)
```

In a continuing conversation `restore_pos` climbs monotonically every turn, so
the `<= restore_pos` filter NEVER drops a prior snapshot. `leaf.leaf_snapshots`
grew +1 full set/turn forever. The leaf lives in the persistent trie, so this
survives session end.

**(2) `_build_edge_node` / `_split_edge` node snapshots — the secondary leak.**

Every continuing turn attaches a new suffix edge, and each edge stored a full
per-layer `CacheSnapshot` via `_snapshot_at(...)`. These node snapshots are
**write-only**: the restore path (`_resolve_restore_position` →
`_materialize_cache_to_depth`) reads ONLY `donor_leaf.leaf_snapshots` +
`leaf.leaf_layer_caches`, never a node's `.snapshot`. The sole reader was the
`edge_nbytes()` byte-accounting diagnostic.

## 4. Fix

**(1)** Bound the merged leaf snapshots to the deepest `_LEAF_SNAPSHOT_RETENTION`
(default 4, env `EXO_LEAF_SNAPSHOT_RETENTION`) by `token_count`:

```python
if len(merged) > _LEAF_SNAPSHOT_RETENTION:
    merged.sort(key=lambda s: s.token_count)
    merged = merged[-_LEAF_SNAPSHOT_RETENTION:]
```

Correctness: the next turn's continuation hit always lands at/near the leaf's
full length and resolves to a retained snapshot. Only a partial hit shallower
than ALL retained snapshots degrades to a (correct, just slower) full
re-prefill. Mirrors the producer-side `_SNAPSHOT_RETENTION=2` cap in
`generator/generate.py`.

**(2)** Stop populating node `.snapshot` in `_build_edge_node` and `_split_edge`
(pass `snapshot=None`). Removed the now-dead `_snapshot_at` helper.

Both are structural fixes, not symptom mitigations.

## 5. Verification

### Unit test (in-suite, no model required)

`test_kv_prefix_cache.py::TestMultiTurnSnapshotLeak` builds non-trimmable
PoolingCache `CacheList`s and drives 7 continuing turns. Asserts (a)
`leaf_snapshots` bounded by retention, and (b) live `PoolingCache` object count
reaches a flat steady state.

- Before fix: `leaf_snapshots` 1→8, live PoolingCache 63→231 (+21/turn).
- After fix: `leaf_snapshots` plateaus at 4, live PoolingCache flat at 168.

Full suite: 24 passed. `ruff` clean. `basedpyright` error count unchanged from
baseline (159 pre-existing; file has loose mlx typing).

### On-hardware confirmation (the decisive test)

Relaunched cluster with `EXO_DSV4_HEAPCENSUS=1`, drove 15 turns at 31K context
(1 big prime + 14 short follow-ups — the cache-hit case where the leak was
starkest). Census trend after the prime:

```
turn  active GB   total GB   array count
 1    77.77       78.40      4450
 2    77.78       78.61      4722   (+272)
 3    77.99       78.83      4994   (+272)
 4    78.39       79.04      5266   (+272)   <- retention buffer (=4) fills
 5    78.39       79.04      5267   (+1)     <- PLATEAU
 6-15 78.39-78.60 79.04      5268..5276 (+1/turn, ZERO GB growth)
```

- `total GB` **pinned at 79.04 GB** for 11 consecutive turns. Pre-fix this
  climbed monotonically to +29 GB over a session.
- Array count grows +272/turn for exactly 4 turns then drops to +1/turn —
  precisely the retention buffer filling then reaching steady state. The
  residual +1/turn adds **zero GB** (benign bookkeeping, not the
  21-PoolingCache-set/turn leak).
- `big(>=16MB)=400` flat throughout. `[MEMPROF]` cache flat at 0.42 GB/turn.

## 6. Prior failed fix attempts (all wrong-site)

1. `tool_loop_guardrails.hard_stop_enabled` — fixed a separate Hermes looping
   bug, unrelated to this leak.
2. commit `4bc391be` — `_copy_pooling_cache` replacing `deepcopy(inner)` in
   `_copy_cache_list`. The snapshot-COPY path was not the accumulation site.
   (Harmless, left in.)
3. commit `2988ae9b` — `contextlib.closing()` around `stream_generate` in
   `prefill()`. Correct hygiene, but not the leak. (Left in.)

All three guessed at the snapshot/generator machinery; none addressed the
unbounded RETENTION of snapshots across turns, which was the actual mechanism.

## 7. Diagnostic instrumentation (kept, gated, zero-cost when off)

All gated by `EXO_DSV4_HEAPCENSUS=1`, off by default:
- `d6c757b2` / `74b0b057` — `[MEMPROF]` cache-byte attribution + `[MEM] after
  prefill`.
- `d9fd8f74` — heap census group-by-shape (found the +21/turn bf16 class).
- `d120d138` / `a3155d41` — referrer trace / trace-to-root.

To reproduce the census measurement:
1. Launch with `EXO_DSV4_HEAPCENSUS=1 DSV4_KV_CACHE_BITS=0 ./start_cluster.sh`.
2. Drive `/tmp/leak_drive_census.py` (1 big ~31K turn + short follow-ups;
   needs `/tmp/medprompt.txt` ~40K-token primer).
3. Read: `ssh macstudio-m4-1 "grep -aE 'live mx.arrays: total|\[MEM\] after
   prefill' ~/.exo/exo_log/exo.log | tail -34"`. Flat `total GB` = fixed.

Note: runner logs live at `~/.exo/exo_log/exo.log` (and
`~/.exo/exo_log/runner_log/{stdout,stderr}.log`), NOT `~/repos/exo/exo.log`.

## 8. Deployment

- Committed `947d7e50b` to `adurham/exo` main, pushed to origin.
- Both Studios deploy via the launcher's `git fetch origin && git reset --hard
  origin/main` (src-only change ⇒ editable install, no mlx rebuild, ~60-90s
  launch). Confirmed both nodes at `947d7e50`.
- Production relaunch WITHOUT the census flag; DeepSeek-V4-Flash + Qwen3.6 both
  READY 2/2; quality-probed clean ("Paris", finish_reason=stop, no BOS spam).
