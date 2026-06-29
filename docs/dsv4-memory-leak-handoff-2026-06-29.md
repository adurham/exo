# DSv4-Flash Multi-Turn Memory Leak — Investigation Handoff (2026-06-29)

**Status: RESOLVED (2026-06-29).** Root cause pinned to TWO write-only /
never-pruned `CacheSnapshot` accumulators in the `KVPrefixCache` trie, both
holding a full per-sparse-layer `PoolingCache` set per turn. Fixed in
`src/exo/worker/engines/mlx/cache.py`; regression-guarded by
`test_kv_prefix_cache.py::TestMultiTurnSnapshotLeak`. See "ROOT CAUSE & FIX"
below. The original investigation notes are kept beneath it for provenance.

---

## ROOT CAUSE & FIX (2026-06-29)

The leaking `(1,P,512)`/`(1,P,128)` bf16 tensors are `PoolingCache` arrays held
inside `CacheSnapshot.states`. The referrer chain decodes exactly:
`PoolingCache -> tuple` (= `CacheList.caches`, a tuple) `-> CacheList -> list`
(= `CacheSnapshot.states`) `-> list -> list ...` (= the growing snapshot lists).

Key fact that made DSv4 hit the snapshot path at all: `PoolingCache.is_trimmable()`
returns `self.pooled is None`, so once a layer has pooled entries the whole DSv4
`CacheList(RotatingKVCache, PoolingCache, PoolingCache)` is NON-trimmable →
`has_non_kv_caches()` is True for DSv4 → the snapshot/restore machinery is live
(not just for SSM models as the code comments implied).

TWO accumulators, both in `cache.py`, each leaking one full 21-layer
(compress_ratio=128) PoolingCache set per turn (= the measured +21/turn):

1. **`update_kv_cache` snapshot merge (the primary leak).**
   `merged = [s for s in leaf.leaf_snapshots if s.token_count <= restore_pos]`
   then `merged.extend(snapshots)`. In a continuing conversation `restore_pos`
   climbs monotonically every turn, so the `<= restore_pos` filter NEVER drops a
   prior snapshot → `leaf.leaf_snapshots` grew +1 set/turn forever.
   **Fix:** bound `merged` to the deepest `_LEAF_SNAPSHOT_RETENTION` (=4, env
   `EXO_LEAF_SNAPSHOT_RETENTION`) by `token_count`. The next turn's continuation
   hit always lands at/near the leaf's full length and resolves to a retained
   snapshot; only a partial hit shallower than ALL retained snapshots degrades
   to a (correct, just slower) full re-prefill.

2. **`_build_edge_node` / `_split_edge` node snapshots (the secondary leak).**
   Every continuing turn attaches a new suffix edge, and each edge stored a full
   per-layer `CacheSnapshot`. These node snapshots are **write-only**: the
   restore path (`_resolve_restore_position` -> `_materialize_cache_to_depth`)
   reads ONLY `donor_leaf.leaf_snapshots` + `leaf.leaf_layer_caches`, never a
   node's `.snapshot`. The single remaining reader was the `edge_nbytes()`
   byte-accounting diagnostic. **Fix:** stop populating node `.snapshot`
   (pass `snapshot=None`); removed the now-dead `_snapshot_at` helper.

**Verification (no cluster needed — pure prefix-cache logic):** a standalone
repro built non-trimmable PoolingCache CacheLists and drove 7 continuing turns.
Before: `leaf_snapshots` 1→8 and live `PoolingCache` 63→231 (+21/turn). After:
`leaf_snapshots` plateaus at 4, live `PoolingCache` flat at 168 from turn 3 on.
Promoted to `TestMultiTurnSnapshotLeak`. Full `test_kv_prefix_cache.py` suite:
24 passed. `ruff` clean; `basedpyright` error count unchanged from baseline (159
pre-existing, file has loose mlx typing).

**NOT YET CONFIRMED ON HARDWARE.** The fix is proven on the prefix-cache logic
in isolation. A live multi-turn drive on the cluster with `EXO_DSV4_HEAPCENSUS=1`
(see "How to reproduce" below) should now show flat `(1,P,512/128)` counts and a
flat `[MEM] after prefill` active GB across short follow-up turns. Run that
before declaring it fully closed end-to-end.

---

## ORIGINAL INVESTIGATION NOTES (pre-fix, kept for provenance)

**Status:** Leak CONFIRMED and tightly characterized. Root-cause reference NOT
yet pinned. Three fix attempts failed (all wrong-site guesses). Diagnostic
instrumentation is committed on `main` and gated behind `EXO_DSV4_HEAPCENSUS=1`.
This doc is written so the fix can be completed from solid ground.

---

## The symptom (ground truth, user-confirmed)

A single growing Hermes conversation on DSv4-Flash climbs in DSv4-runner active
memory and never releases:

- Measured session (139 msgs, 68 API calls, ctx 16K→105K): DSv4 runner active
  memory went **77.13 GB → 106.61 GB = +29.5 GB**, footprint peak 108 GB.
- **~0.2–0.4 GB leaked per TURN**, monotonic, never freed.
- Survives **idle** (idle-reclaim ran, `cache=0.00GB`) AND **session end** — at
  idle post-session the runner stayed pinned at 106.65 GB active / 109 GB
  footprint. This is a hard pinned-reference leak, not reclaimable Metal pool.

## What it is NOT (ruled out with measurement)

- **NOT Qwen.** `get_active_memory()` is per-process; the 90–106 GB is the DSv4
  runner alone. Qwen3.6 is a separate ~18–21 GB process on the node.
- **NOT KV cache cost.** `[MEMPROF]` cache-walker shows the per-request KV +
  pooled cache is ~1 GB at 31K ctx — on-spec (kv-cache-architecture.md says
  ~3 GB at 100K). KV is fine.
- **NOT the reclaimable Metal allocator pool.** `cache=0.00GB` at the leak
  point; idle-reclaim already ran and couldn't free it.
- **NOT the prefix-cache trie size.** `trie_bytes` logs ~256 KB; `maxPrefixSessions=1`
  so only 1 leaf; `_total_bytes()` (which DOES count edge snapshots) stays tiny.

## What it IS (measured via gc heap census)

Instrumentation `_heap_census_mx_arrays()` in
`src/exo/worker/engines/mlx/generator/generate.py` (gated `EXO_DSV4_HEAPCENSUS=1`,
fires at end of `prefill()`):

- **Big arrays (>=16MB) count is FLAT at 400** across turns — those are the
  static MoE expert weights (`(256,1024,512)`/`(256,4096,128)` uint32, 512 MB
  each). NOT leaking.
- **The leak is small bf16 tensors of shape `(1, P, 512)` and `(1, P, 128)`**
  (P = pooled length, ~7936 for ~31K ctx) **growing by exactly +21 per turn.**
- **+21/turn == the number of DSv4 sparse-attention layers** (compress_ratio=128
  layers; config has 43 layers alternating [4,128]). So **one pooled-KV tensor
  set leaks per sparse layer per turn.**
- These are **`PoolingCache` objects** (the DSv4 sparse-attention pooled KV).
- Total `mx.array` count climbs ~+272/turn; bf16 pooled GB climbs ~0.2/turn,
  matching the macro symptom.

### Referrer trace (trace-to-root, commit a3155d41)
```
(1, 7936, 512) bf16 -> PoolingCache -> tuple -> CacheList -> list -> list -> list -> list -> list -> list -> list
```
The chain reaches `CacheList` then dead-ends in a **wall of ≥7 nested anonymous
`list`s** within the 10-hop budget — never reaching a frame/module/named owner.
This deep list-of-lists nesting is the signature of **a list that accumulates a
list (of CacheLists) per turn** — i.e. something doing `persistent.append(list(...))`
each turn, where the holder is itself list-nested (not an object attribute, or
the trace would have named the class).

## Fix attempts that FAILED (do not retry these — proven wrong-site)

1. **`tool_loop_guardrails.hard_stop_enabled`** — unrelated (that fixed the
   separate Hermes looping bug; real and kept, but not this leak).
2. **commit `4bc391be`** — `_copy_pooling_cache` replacing `deepcopy(inner)` in
   `cache.py::_copy_cache_list`. Hypothesis: deepcopy of PoolingCache in the
   snapshot/trie path retained copies. **Leak unchanged.** The snapshot/trie
   copy path is NOT the accumulation site. (Change left in — harmless, avoids a
   deepcopy regardless.)
3. **commit `2988ae9b`** — wrap `stream_generate` in `contextlib.closing()` in
   `prefill()` (was `for _ in stream_generate(...): break` without closing the
   generator). Hypothesis: suspended generator frame pinned per-layer caches.
   **Leak unchanged.** (Change left in — closing a generator you break out of
   is correct hygiene regardless, just not THE leak.)

## The ONE diagnostic that will pin it (next step)

The trace dead-ends in anonymous nested lists because `gc.get_referrers` order
is nondeterministic and the census only names types, not identity/length. The
definitive instrument:

- For one leaked `(1,P,512)` bf16 array, walk referrers and for **every `list`
  in the chain, log `id(list)` and `len(list)`**.
- Re-run the 6-turn drive. **The list whose `len()` increases by ~21 (or whose
  count of such lists grows) each turn is the accumulator.** Its stable `id`
  across turns + growing len names it unambiguously.
- Alternatively: snapshot `gc.get_objects()` counts of `CacheList` instances
  per turn — if `CacheList` instance count grows ~+21/turn, find what list holds
  the growing set (search `gc.get_referrers` of the CacheLists for the one
  persistent container).

Candidate accumulators to check FIRST once the holder list is identified
(these are `append(list(cache))`-style sites seen in grep, not yet confirmed):
- `batch_generate.py:1538  cache_list.append(list(cache))` — local in
  submit_batched; check if `cache_list` (or what consumes it) escapes to `self`.
- `pp_speculation.py:76  snap.append(list(c.cache))` — PP speculation snapshot
  list; check lifetime.
- `_active_tasks` / MTP `self._cache` / `self._mlx_gen._captured` retention on
  the long-lived generator (`ExoBatchGenerator` / `DSv4MTPBatchGenerator`).
- Any module-level list in `deepseek_v4.py` (e.g. a profiling/route-history
  accumulator left enabled).

## How to reproduce / measure (fast loop)

1. Nodes must be on the instrumentation commit and `EXO_DSV4_HEAPCENSUS=1` must
   be passed to `start_cluster.sh` (passthrough already wired in start_cluster.sh).
2. Drive: `/tmp/leak_drive_census.py` — 1 big (~31K) turn + 5 short follow-ups
   (each re-prefills ~18 tok = cache-hit case where the leak is starkest).
   Takes ~4 min.
3. Read: `ssh macstudio-m4-1 "grep -aE 'x \(1, [0-9]+, (512|128)\) bfloat16' ~/exo.log"`
   — counts grow +21/turn if leak present, flat if fixed.
4. Macro check: `[MEM] after prefill (...)` active GB climbing across the short
   turns = leak; flat = fixed.

## CRITICAL operational gotcha (cost ~6 user reboots this session)

Every `start_cluster.sh` run risks wedging the inter-node Thunderbolt/RDMA link
("No device connected", placement fails "no RDMA-connected cycles") — this is
the QP-leak failure-mode E in the `exo-cluster-operations` skill: the RDMA
preflight probe leaks queue-pairs on SIGTERM, and accumulated orphaned RoCE
state wedges the Apple TB stack. **Only a full reboot of BOTH studios clears it.**
The L2 ping (`ssh macstudio-m4-1 "ping -c2 192.168.200.2"`) is the reliable
link-health signal — `system_profiler ... "No device connected"` text can be
stale/misleading. **Minimize restarts**: batch instrumentation changes; do the
diagnostic-len() pass ONCE, identify the holder, then deploy the real fix ONCE.
A worthwhile side-fix: make the RDMA preflight in start_cluster.sh clean up its
QPs (destroy_qp/dealloc_pd) on exit so restarts stop wedging the link.

## Instrumentation commits on main (all gated, additive, safe to keep)
- `d6c757b2` / `74b0b057` — `[MEMPROF]` cache-byte attribution + `[MEM] after
  prefill` in serial `prefill()`.
- `d9fd8f74` — heap census group-by-shape (found the +21/turn bf16 class).
- `d120d138` / `a3155d41` — referrer trace / trace-to-root.
- (Leave gated by `EXO_DSV4_HEAPCENSUS`; off by default, zero cost normally.)

## Warm memory
Fact id 789 has the condensed version of this for cross-session recall.
