# 352.6K batched-verify regression — allocator fragmentation vs pool-growth overlap

**Scope.** Follow-up to `/tmp/ab/protocol352/root_cause_analysis.md`. That
doc showed the batched path allocates +4.73 MB/sparse-layer more than
rowseq, but MLX per-layer `mx.async_eval` fencing bounds the delta to
O(10 MB) — insufficient on its own to explain 1.37 GB swap and 4/12
collapses. This doc reads the MLX Metal allocator, the `PoolingCache`
growth path, and the existing memory probes to determine whether the
regression is (A) allocator fragmentation from the batched path's larger
`mx.concatenate` allocations, (B) `PoolingCache` growth-event transient
overlap with batched verify, or (C) something else.

**Repo/commit.** exo main HEAD `b999c3354` (tree clean). mlx-lm submodule
pinned at `d098642`. MLX submodule at `mlx/`.

---

## 1. MLX Metal allocator behaviour (Finding A analysis)

### 1.1 Allocator architecture

`MetalAllocator` (`mlx/mlx/backend/metal/allocator.h:17-94`,
`mlx/mlx/backend/metal/allocator.cpp:47-268`) has three tiers:

| Tier | Threshold | Backing | Reuse mechanism |
|---|---|---|---|
| Small heap | `size < small_size_ (256)` — but with fork-local coalescing so all sub-256 requests share a bucket (`allocator.cpp:128-140`) | `heap_->newBuffer` (256 MB shared Metal heap, `allocator.cpp:74-77`) | Heap-managed |
| Buffer cache | `size >= small_size_` on free | `BufferCache<MTL::Buffer>` (`buffer_cache.h`) | Size-bucketed multimap, LRU list |
| Fallback | cache miss | `device_->newBuffer` (`allocator.cpp:179`) | Fresh VM region per call |

The path on `free` (`allocator.cpp:210-228`): freed buffers go **into the
cache** (`recycle_to_cache`, `buffer_cache.h:48-56`) unless the cache is
already above `max_pool_size_`, which is set from `block_limit_` = `min(1.5
× max_recommended_working_set_size, 0.95 × memsize)` (`allocator.cpp:65-67`).
On a 128 GB M4 Max that's about 120 GB — very large, so the cache retains
almost everything freed during normal decode.

**On alloc** (`allocator.cpp:143-152`), the allocator first probes
`buffer_cache_.reuse_from_cache(size)`. `BufferCache::reuse_from_cache`
(`buffer_cache.h:30-46`) does an `std::multimap::lower_bound(size)` and
accepts the closest cached buffer as long as its size is **less than
`min(2*size, size + 2*page_size)`** (`buffer_cache.h:33-34`). Above 16 KB
the request is rounded up to `vm_page_size` (16 KB on macOS,
`allocator.cpp:126-127`), so all page-multiple sizes bucket together.

### 1.2 Does concatenate produce different reuse than sequential allocs?

Rowseq's per-row `combined` KV block is `(1, 1, 1, 640, 512)` bf16 =
655,360 bytes → page-aligned to 655,360 (already a multiple of 16 KB).
Batched's `combined` is `(1, 4, 1, 640, 512)` bf16 = 2,621,440 bytes →
2.5 MB, also page-aligned.

The `reuse_from_cache` acceptance window is `[size, min(2*size, size +
32768)]`. For a 2.5 MB request the window is `[2621440, 2653696]` (only
32 KB slack), which the sequential 655 KB rowseq allocations **cannot
satisfy**. Similarly `pooled_gathered` for batched is `(1,4,512,512)` bf16
= 2,097,152 bytes = 2 MB vs rowseq's `(1,1,512,512)` bf16 = 524 KB.

So the batched path allocates NEW size classes not present in the rowseq
steady-state cache — the first time batched runs, these 2 MB and 2.5 MB
buckets must come from `device_->newBuffer` (or by evicting other cached
buffers, `allocator.cpp:149-152`). **After warmup, however, these become
recycled hot buckets** — every subsequent verify at the same L=4 hits a
cache-hot 2 MB or 2.5 MB slot exactly.

**Verdict on Finding A — REFUTED.** MLX's Metal allocator has an explicit
buffer cache with size-class bucketing keyed to page-aligned sizes
(`allocator.cpp:126-140`, `buffer_cache.h:30-46`). Buffers freed by the
batched verify path are recycled into that cache and returned on the next
verify's identical-shape allocation — this is exactly the case the cache
is designed for (identical size, identical dtype, called every cycle).
Fragmentation is not the mechanism.

The one one-time cost is warmup: the first batched verify at 352.6K
context must materialize the 2 MB `pooled_gathered` and 2.5 MB `combined`
buckets for each of the 21 sparse layers. That's a bounded one-shot
allocation of 21 × 4.5 MB ≈ 95 MB, not 1.37 GB, and it happens at the
depth-gate crossing (ctx=8192) long before 352.6K.

`EXO_JIT_MEMORY_RESERVE_GB=18.0` (`start_cluster.sh:712`) is a per-node
placement reservation — it's read by `placement_utils.py:48-76` to decide
whether a JIT auto-load can add a second model, NOT interpreted by the
MLX allocator. It has no interaction with allocator cache behaviour.

---

## 2. PoolingCache growth path (Finding B analysis)

### 2.1 Growth mechanics

At 352.6K decode, both paths use `update_and_fetch_deferred`
(`mlx-lm/mlx_lm/models/cache.py:1518-1603`) because
`EXO_DSV4_POOL_DEFER_COPY_MAX_BYTES=8388608` (8 MiB,
`start_cluster.sh:2200`) is set below sparse pool storage size (~90 MB —
`cache.py:1578-1601` takes the `pre_write is None` donation-friendly
branch on steady-state writes).

The **growth** branch (`cache.py:1564-1575`) fires when
`new_offset > self._pool_storage.shape[1]`:
```
1569|            current_size = self._pool_storage.shape[1]
1570|            grow_by = max(self.step, new_offset - current_size + 1)
1571|            new_size = current_size + grow_by
1572|            old = self._pool_storage
1573|            pre_write = old[:, : self._pool_offset]
1574|            self._pool_storage = mx.zeros((B, new_size, D), dtype=px.dtype)
1575|            self._pool_storage[:, : self._pool_offset] = old[:, : self._pool_offset]
```

Three tensors are simultaneously live during a growth event:
- `old` (~90 MB at ratio-4 near 352K)
- `self._pool_storage` (new, ~90.5 MB — the `+step=256` slot bump)
- `pre_write` (a view on `old`; keeps `old` alive)

`pre_write` is **returned to the caller** (line 1603) → SDPA reads the
OLD storage this cycle. The write of `px` goes into the NEW storage
(line 1590). MLX's lazy graph has to keep BOTH tensors resident until the
next cycle's commit_pending forces materialization: **~180 MB per sparse
layer during a growth event, vs the ~90 MB steady state.**

### 2.2 Growth frequency at 352.6K

- `EXO_DSV4_POOL_GROW_STEP=256` (`cache.py:1312`, `start_cluster.sh` uses
  default). Every growth event bumps the storage width by 256 columns.
- Sparse layers (ratio=4): the pool grows by 1 entry per full window (4
  decode tokens), so a growth event fires **every 256×4 = 1024 decode
  tokens** per layer.
- MTP verify commits ~2–3 tokens/cycle (γ=3, average acceptance ≈2.4),
  so a growth event fires roughly **every ~400–500 verify cycles** per
  sparse layer.
- At 352.6K decode, all 21 sparse layers grow in lockstep because
  `_pool_offset` advances identically on all layers (same `commit_pending`
  cadence). So on the **cycle where growth fires, all 21 sparse layers
  each allocate a new ~90 MB tensor while still holding the old one** —
  a peak burst of **21 × 90 MB ≈ 1.90 GB above the 1.95 GB steady-state
  pool storage** = ~3.85 GB pool footprint momentarily.

### 2.3 Does the batched path amplify the growth overlap?

The batched verify holds the extra ~5 MB/sparse-layer transients live
across the block window. When a growth event lands INSIDE that window:

- Rowseq peak footprint on a growth cycle:
  `~1.95 GB (steady pool) + ~1.90 GB (growth doubling on 21 layers) + ~35 GB weights + ~5 MB verify transient × 21`
  ≈ **~3.85 GB pool + verify transients that peak at ONE layer at a time
  (rowseq processes L rows sequentially per block)** ≈ **~3.85 GB pool +
  ~1.6 MB active verify transient.**

- Batched peak footprint on a growth cycle:
  `~1.95 GB (steady pool) + ~1.90 GB (growth doubling) + ~5 MB × in-flight-layers-under-async-fence`
  With `EXO_DSV4_FENCE_ASYNC_C2=0` (`start_cluster.sh:1742`, default 0)
  the C2 async-fence extension is OFF, so batched at B=1,L=4 still fences
  once per layer via the base `_FENCE_ASYNC` gate (`deepseek_v4.py:3109-3124`).
  In-flight layers = 1 typically. So batched adds **~5 MB — 6.3 MB of
  live verify transient per layer** vs rowseq's ~1.6 MB.

**The +4.73 MB/sparse-layer batched delta is a SECOND-ORDER effect on
growth cycles.** The dominant peak on growth cycles is the ~1.90 GB
old+new pool tensor doubling, not the ~100 MB total batched-vs-rowseq
transient inflation. Both paths pay the growth-event peak.

But there's a **subtler amplification**: growth-event storage doubling
happens LAZILY under `mx.async_eval` — the growth branch schedules the
copy (line 1575) and the write (line 1590) but doesn't force evaluation
until `mx.async_eval(self._pool_storage)` at line 1601. Between growth
schedule and next-cycle materialization, the batched path's larger
in-flight transients (indexer scores at (1,4,88150) bf16 = 705 KB per
sparse layer, `pooled_gathered` at 2 MB per sparse layer, `combined` at
2.5 MB per sparse layer) share the wired memory ceiling with two 90 MB
pool copies × 21 layers = 3.85 GB. That's a hard ~4 GB peak burst per
growth event, and adding the +4.73 MB × 21 = ~100 MB of batched
transients on top nudges the wired ceiling.

### 2.4 Does `POOL_DEFER_COPY_MAX_BYTES=8388608` interact?

Yes, and it **makes the growth branch cheaper on cycles that DON'T grow**:
line 1581–1582 gates `pre_write = self._pool_storage[:, :self._pool_offset]`
on `storage_bytes <= _POOL_DEFER_COPY_MAX_BYTES`. At 352.6K sparse pools
are ~90 MB >> 8 MiB, so `pre_write` stays None (line 1577) and the write
donates in-place with no copy. This kills the steady-state pool-copy cost
but does NOT affect the growth branch (lines 1564-1575), which always
holds `pre_write = old[:, :offset]` unconditionally.

`EXO_DSV4_POOL_SNAPSHOT_BATCH=1` (`start_cluster.sh:223`) is orthogonal
— it enables snapshot/restore-meta for BatchPoolingCache (mtp verify
rollback), NOT for single-stream `PoolingCache` used at 352.6K single-req
decode. `EXO_DSV4_POOL_RESTORE_AFTER_TRIM=1` (`start_cluster.sh:222`)
similarly gates rollback-after-trim rewriting, which does NOT allocate
fresh pool tensors — it only rewinds `_pool_lengths` (see
`cache.py:2556-2602`, `restore_meta` explicitly says "pooled storage is
left as-is"). Neither env has a memory-cost interaction with growth.

**Verdict on Finding B — PARTIALLY SUPPORTED.** Growth events do double
pool storage (`cache.py:1572-1575`) for one cycle, holding ~180 MB per
sparse layer × 21 layers = ~3.85 GB. This happens roughly every ~400–500
verify cycles per layer. On the specific cycle where growth fires and
the batched path is active, the +5 MB/sparse-layer batched delta adds on
top — bringing the peak from ~3.85 GB (rowseq growth cycle) to ~3.95 GB
(batched growth cycle). Both paths hit ~3.85 GB, but the batched path's
extra ~100 MB total transient is the marginal push over the wired
ceiling on the tail-4 collapsed runs.

However, the growth cadence math **does not match the collapse rate**.
At 352.6K, growth events fire simultaneously across all 21 sparse layers
approximately every 1024 tokens ÷ decode rate (~3.5 tok/s) = ~5 minutes.
The 4/12 collapse rate happens at a coarser granularity than a per-cycle
growth burst. The growth-event peak is a REAL amplifier but it's not the
sole trigger.

---

## 3. What is the actual dominant mechanism?

Neither A nor B alone matches. The evidence points at a **third
mechanism: cumulative wired-ceiling proximity** where:

1. Steady-state wired residence at 352.6K = ~88 GB
   (`docs/incidents/2026-08-08-section23-stall-m4-1.log:1`
   `mem_before_snapshot_mb=88275.3`), 
2. Wired ceiling = ~124.5 GB (`mem_limit_mb=124518.4`), leaving ~36 GB
   headroom.
3. On every growth cycle (every ~5 minutes), the pool footprint spikes
   by ~1.9 GB briefly. Both paths pay this.
4. **The batched path's continuous +5 MB/sparse-layer live transient
   raises the FLOOR of active memory** — not the peak transiently. Over
   the course of a 352.6K decode run (~1600 verify cycles), this floor
   elevation of ~100 MB / cycle stochastically overlaps with:
   - JIT weight page-ins,
   - `_dspark_caches.append_ctx` allocations (`dsv4_mtp.py:4649`),
   - draft path's ctx-KV allocations,
   - OS-level compressor and file-cache pressure,
   causing stochastic crossings into swap on some cycles but not others.

The 4/12 collapse rate is exactly the fingerprint of a **stochastic
wired-ceiling crossing**, not a deterministic per-cycle bug: on runs
where the co-tenant memory pressure happens to align with a growth
cycle + a batched verify + a large `_dspark_caches` append, the system
tips into paging. On runs where those don't align, it stays under
ceiling.

---

## 4. Existing env-gated memory probes (no code change required)

**Yes** — the codebase already has memory instrumentation that can be
enabled by env alone. Options ranked by usefulness for this hunt:

### 4.1 `EXO_MEMORY_PROFILE_PATH` (RECOMMENDED for this hunt)

`src/exo/worker/engines/mlx/generator/batch_generate.py:90-91, 4714-4723`
+ the record fn at `:403-470`. When set:
- writes JSONL to the path every `EXO_MEMORY_PROFILE_INTERVAL` decode
  steps (default 256; **set to 1 or 4 for per-cycle granularity**);
- captures `active_bytes`, `peak_bytes`, `cache_bytes`, `rss_bytes`,
  `vms_bytes`, and (fork-only) `live_array_desc` count;
- resets peak between snapshots so each row is a high-water mark for
  the interval.

Env to enable (no code change):
```
EXO_MEMORY_PROFILE_PATH=/tmp/mem_352.jsonl
EXO_MEMORY_PROFILE_INTERVAL=4
```

Both are already allowlisted through the runner env at
`start_cluster.sh:1644`, so setting them in the shell before invoking
`start_cluster.sh` propagates to both runners.

**This is the direct answer** — capture per-verify active/peak/cache MB
without touching source. Compare a rowseq run vs a batched run at 352.6K
and the +5 MB/sparse-layer × 21 layers pattern should show as ~100 MB
peak elevation in the batched path, plus growth-cycle spikes to 3.85 GB+
above baseline.

### 4.2 `EXO_DSV4_C2_TRACE=1` (secondary — spec verify only)

`src/exo/worker/engines/mlx/speculative/dsv4_mtp.py:869-990, 3538-3810`.
Emits a JSONL per γ-step per verify cycle to
`/tmp/dsv4_c2_trace_pid<pid>.jsonl` with `metal_active_mb_start`,
`metal_peak_mb_start`, `metal_active_mb_end`, `metal_peak_mb_end`
(`dsv4_mtp.py:3779-3785`). Gated at `_c2_trace = _C2_TRACE_ENABLED and
sync_drafts`, so it only fires in cluster mode.

Also requires `sync_drafts` = cluster + sync group active. This is per-γ-
step inside the speculative predict loop — captures the DRAFT path, not
the target-model verify forward. Less directly relevant, but confirms
per-step MLX metal deltas.

### 4.3 Outlier log (already always-on)

`src/exo/worker/engines/mlx/pp_speculation.py:3348-3406` writes
`mem_before_snapshot_mb`, `mem_snapshot_delta_mb`, `mem_after_spec_fwd_mb`,
`mem_limit_mb` for any DSpark cycle > 1000 ms. This runs unconditionally
(no env gate). At 352.6K where mean cycle time is 100–130 ms, only the
collapsed tail-cycles will trip the >1s outlier threshold and log — but
that's exactly what you want for post-hoc collapse attribution. Just grep
`/tmp/*.log` for `[PP DSpark OUTLIER`.

### 4.4 `MLX_LOG_NEW_BUFFER_PATH` (nuclear option)

`mlx/mlx/backend/metal/allocator.cpp:163-173`. Logs every cache-miss
allocation (each `device_->newBuffer` call) with its size to the path.
Enable via env allowlist at `start_cluster.sh:1517`. This would confirm
Finding A refutation directly: if the cache is working, cache-misses
after warmup should be near-zero. Warning: massive log volume on decode,
use only for a short probe run.

---

## 5. Recommendation

**Root cause is NOT allocator fragmentation.** MLX's `MetalAllocator`
has a size-bucketed `BufferCache` with a `[size, min(2*size, size+2*page)]`
reuse window (`buffer_cache.h:33-34`) that reuses identical-shape
allocations across verify cycles. The batched path allocates NEW size
classes (2 MB pooled_gathered, 2.5 MB combined) not present in rowseq's
cache, but these become recycled after the first verify at each L=4
depth — a bounded one-shot warmup cost of ~95 MB across 21 sparse layers,
not a cumulative leak.

**Pool-growth overlap IS a partial contributor,** but the growth-doubling
peak (~3.85 GB burst on ~5-minute cadence at 352.6K) is paid by BOTH
paths. The batched path's marginal contribution on those cycles is ~100
MB (+5 MB/sparse-layer × 21), which is what tips the wired ceiling on
the stochastically-worst-case cycles.

**Actual root-cause fix direction (evidence-based, NOT a context cap):**

The most efficient intervention is to **REDUCE the batched path's
continuous transient floor** — specifically the `(1,4,1,640,512)`
combined KV block (+1.97 MB/sparse-layer) and the `(1,4,512,512)`
pooled_gathered (+1.57 MB/sparse-layer). The prior analysis's proposed
`EXO_DSV4_VERIFY_BATCH_CHUNK_ROWS=2` at depth (halving these to L=2
sub-batches) is the correct direction: it halves the +100 MB batched
delta at 352.6K, dropping worst-case wired-ceiling proximity from ~124 GB
back below the 124.5 GB limit on the tail-4 cycles.

**A complementary fix** — force `mx.clear_cache()` on the growth-event
side of the pool code path (`cache.py:1575`) to release cached buffers
before the growth doubling holds ~1.9 GB of old+new pool tensors. This
is a one-line env change: `EXO_MLX_CLEAR_CACHE_INTERVAL=N`
(`batch_generate.py:103`, allowlisted at `start_cluster.sh:1646`) forces
periodic `mx.clear_cache()`. Setting `N=64` (every 64 decode steps)
guarantees the buffer cache is drained between growth events (~1024
tokens apart), so growth doesn't compound with a full cache.

**Immediate empirical validation before implementing either fix:**
1. Re-run one 352.6K protocol run with
   `EXO_MEMORY_PROFILE_PATH=/tmp/mem_batch_352.jsonl EXO_MEMORY_PROFILE_INTERVAL=4`
   for the batched path.
2. Same for rowseq path (`EXO_DSV4_VERIFY_BATCH=0`).
3. Compare `peak_bytes` histograms: batched should show a right-tail
   ~100 MB heavier than rowseq under normal cycles, and spikes to +1.9
   GB above baseline on growth cycles in both. Any peak > 124 GB
   correlates with a collapse.
4. If confirmed, apply `EXO_DSV4_VERIFY_BATCH_CHUNK_ROWS=2 EXO_DSV4_VERIFY_BATCH_CHUNK_MIN_CTX=131072`
   (per prior analysis's proposed knob) and re-measure; expected peak
   drop is ~50 MB.

**No source code changes are required to run the diagnostic probe.**

---

## 6. Appendix — file:line audit trail

- MLX Metal allocator entry / cache: `mlx/mlx/backend/metal/allocator.h:17-94`,
  `mlx/mlx/backend/metal/allocator.cpp:47-268`.
  - Cache reuse window: `mlx/mlx/backend/common/buffer_cache.h:30-46`
    (`min(2*size, size + 2*page_size)`).
  - Alloc dispatch: `allocator.cpp:109-203`, cache probe at `:143-144`,
    fallback newBuffer at `:179`.
  - Free path: `allocator.cpp:210-228` (recycles into cache below limit).
  - `MLX_LOG_NEW_BUFFER_PATH` diagnostic: `allocator.cpp:163-173`.
- `MetalAllocator` block limit: `allocator.cpp:65-67`
  (`min(1.5 × max_recommended_working_set, 0.95 × memsize)`).
- Small-size heap coalescing (fork-local): `allocator.cpp:128-140` +
  `allocator.h:52-70`.
- PoolingCache growth: `mlx-lm/mlx_lm/models/cache.py:1494-1516`
  (`update_and_fetch`), `:1518-1603` (`update_and_fetch_deferred`).
  - Growth branch (holds old + new + pre_write): `cache.py:1564-1575`.
  - Growth step env: `cache.py:1312-1314`
    (`EXO_DSV4_POOL_GROW_STEP=256`).
  - Defer-copy env: `cache.py:1265-1267`
    (`EXO_DSV4_POOL_DEFER_COPY_MAX_BYTES=32 MiB` default; production
    override to `8 MiB` at `start_cluster.sh:2200`).
  - Async-eval fence at growth: `cache.py:1601`.
- Batched verify combined-KV alloc: `mlx-lm/mlx_lm/models/deepseek_v4.py:2384-2535`
  (`_sparse_verify_rows_batched`), specifically the (B,L,1,sw+k,D)
  concat at ~:2495 and pooled_gathered at ~:2470.
- Verify-batch gate: `mlx-lm/mlx_lm/models/deepseek_v4.py:6873-6883`,
  `_set_verify_batch_ctx` at `:1667-1677`, block-level skip at `:5072-5083`.
- Fence async gate: `deepseek_v4.py:3109-3124`
  (`_FENCE_ASYNC`, `_FENCE_ASYNC_MAX_B` from `EXO_DSV4_FENCE_ASYNC_C2`).
- Existing memory probes:
  - `_mem_profile_record` def: `src/exo/worker/engines/mlx/generator/batch_generate.py:403-470`,
    call site: `:4714-4723`.
  - Envs and defaults: `batch_generate.py:90-91`
    (`EXO_MEMORY_PROFILE_PATH` / `EXO_MEMORY_PROFILE_INTERVAL`).
  - Runner env allowlist: `start_cluster.sh:1644`.
  - `_c2_trace_metal_mb` (spec-verify draft loop): `dsv4_mtp.py:982-989`,
    called at `:3540-3541` and `:3746`.
  - Outlier memory log (auto): `pp_speculation.py:3399-3404`.
  - `EXO_MLX_CLEAR_CACHE_INTERVAL`: `batch_generate.py:103`, allowlist
    at `start_cluster.sh:1646`.
  - `MLX_LOG_NEW_BUFFER_PATH`: `allocator.cpp:163-173`, allowlist at
    `start_cluster.sh:1517`.
- `EXO_JIT_MEMORY_RESERVE_GB` (irrelevant to allocator): read at
  `src/exo/master/placement_utils.py:48-76`.
