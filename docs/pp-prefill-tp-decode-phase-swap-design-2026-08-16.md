> # ⚠️ SUPERSEDED — THIS DESIGN IS DEAD. DO NOT IMPLEMENT.
>
> **Decision date: 2026-08-16.** The PP-prefill -> TP-decode phase swap
> described below was evaluated and **rejected**. The authoritative
> decision record is
> `docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md` **Section 107**
> (why the swap was dropped) and **Section 108** (why expert-locality
> placement, the follow-on idea, is also impossible).
>
> Why it was rejected, in short:
> - Full-request math gives only **~11.1%** on a cold 500K request
>   (TP-only 1584s vs PP+swap+TP 1409s), and **ZERO** on follow-up turns
>   that hit the prefix cache while still paying the ~37.4s swap.
> - It costs **all cross-phase concurrency** (Requirement 1) and
>   cancellation, which TP provides natively.
> - It needs a cross-rank cache gather that **has never existed**
>   (`serve_prefill` emits only the local rank's layer-half) plus either
>   partial-teardown lifecycle surgery or a new push-before-exit
>   protocol leg.
> - **Neither topology clears Requirement 4's 400+ tok/s at depth**
>   (PP 364 / TP 319 at 500K), so the swap converts a miss into a
>   smaller miss.
>
> **The shipped architecture is TP for BOTH prefill and decode.**
>
> Kept as a historical record of what was considered and why it does not
> work. The one piece of this effort that WAS retained and shipped is the
> DSv4 `CacheList`/`PoolingCache` wire codec (commit `e3b6a0bed`) -- a
> real fix, and the enabling primitive should a prefill/decode split ever
> make sense on a larger cluster where both layouts can be co-resident on
> different node pairs.

# PP-prefill → TP-decode phase swap: design (2026-08-16)

**Status: design, not yet implemented. No code in `src/**`, `mlx/`, or
`mlx-lm/` was touched to produce this document.**

**Directive (settled, not up for debate): prefill runs through PP,
decode runs through TP.** This document is the HOW. It does not
re-litigate whether TP is the right choice for decode — see Section 100
of `docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md` for why that
question is closed, and Section 7 of the same doc for why *simultaneous*
dual-layout residency is closed too (~125.4 GB of 128 GB, confirmed
twice independently). This design accepts both facts and builds within
them: the two layouts are never resident together; the transition
between them is a **sequential swap** with a strict free-then-load
ordering.

---

## 0. The shape of the problem, restated precisely

| | PP layout, rank0 | TP layout, rank0 |
|---|---|---|
| Layers held | 0–21 (22 of 43) | 0–42 (all 43) |
| Experts per held layer | ALL | HALF |
| Size | ~79.5 GiB | ~77.7 GiB |
| Overlap with the other layout | half the experts of layers 0–21 (already resident, correct sharding) | same overlap, viewed from the other side |
| Delta to DROP | half-experts of layers 0–21 (~39.8 GiB — the OTHER half, once TP's convention is fixed, see §1.1) | — |
| Delta to ACQUIRE | — | all-43-layers-half-experts for layers 22–42 (~37.9 GiB) — layers 0–21 already have half already resident, just possibly the wrong half depending on shard convention |

Net weight delta per rank is dominated by:
- **Layers 22–42**: entirely new to rank0 (rank0 never held these under
  PP — rank1 did). This is the acquire-heavy part, ~37.9 GiB worth of
  half-expert weight (rank0's half).
- **Layers 0–21**: rank0 already has ALL experts. TP only needs HALF.
  This is a pure drop of the local half rank0's TP shard convention
  does NOT keep — up to ~39.8 GiB freed, no network/disk needed for
  this part AT ALL if (and only if) TP's expert-half assignment for
  layers 0–21 is chosen to match a subset of what's already resident
  (see §1.1 — this is a real design choice, not automatic).

Peak-memory constraint: never exceed ~118 GB resident (128 GB minus
~8 GB MLX runtime minus a few GB margin) at any point during the swap.
Simultaneous dual residency of full PP layer 0–21 + full TP layer 0–21
would burn ~79.5 + ~38.9 GiB just on that one span — instant OOM. The
swap must be **strictly ordered: free before load, in slices small
enough that peak never spans both layouts for the same layer range.**

---

## 1. THE SWAP MECHANISM

### 1.1 Free choice that removes half the "acquire" cost for free

`DeepseekV4ShardingStrategy.shard_model` (auto_parallel.py:1031) assigns
each rank's expert half via `all_to_sharded_linear_in_place` /
`sharded_to_all_linear_in_place`, which shard along the expert axis by
rank index deterministically (`self.group`'s rank, `world_size=2` — even
experts to rank0, odd to rank1, or whatever the fixed convention inside
`ShardedLinear` is — call it convention `E_r`). This convention is fixed
per rank, not per phase. **Design choice: reuse the exact same
convention for the swap.** Concretely: when rank0 (PP owner of layers
0–21, ALL experts for those layers) transitions to TP, it should keep
**exactly the expert half `E_0`** for layers 0–21 that `DeepseekV4ShardingStrategy`
would assign it anyway, and free the complement `E_1` half. This turns
"acquire TP layers 0–21" into "free half of already-resident layers
0–21" — zero disk/network I/O for that slice, only in-place buffer
release. This is free, but it does depend on `E_0`/`E_1` being a
**stable, rank-pinned convention** already true today (both ranks always
load the same expert-half assignment on every launch — confirmed by
reading `shard_model`'s `self.group` rank use, which is deterministic
per `mx.distributed.Group`). If that convention ever became
data-dependent or randomized, this optimization silently breaks and
every swap pays full acquire cost for layers 0–21 too — call this out
explicitly as an invariant the implementation must assert, not assume.

### 1.2 The ordering: layer-interleaved, strict free-then-load

Two delta classes, handled differently:

**Class A — layers 0–21 (already fully resident under PP):**
For each layer `i` in `0..21`, in order:
1. Evaluate current weight arrays for layer `i` are fully materialized
   (`mx.eval`) — no lazy graph holding a reference that would keep the
   about-to-be-freed half alive.
2. Slice out and drop the reference to expert-half `E_1`'s weight
   arrays for layer `i` (`gate_proj`/`up_proj`/`down_proj` for
   `switch_mlp`, per §1.1). Python refcount to zero + `mx.clear_cache()`
   (already the pattern `shard_model` uses after every layer, see
   auto_parallel.py:1031 loop's `mx.eval(layer); mx.clear_cache()`).
3. Do NOT touch expert-half `E_0` or the attention block at all — TP
   replicates attention exactly like PP already stores it once per
   rank (§DeepseekV4ShardingStrategy docstring: "Replicates attention on
   every rank; shards only the MoE block"), so attention weights for
   layers 0–21 need **zero movement**, they're already correctly laid
   out for TP.

This alone frees ~39.8 GiB before any acquire begins — do this pass
FIRST, entirely, across all 22 layers, before touching Class B. That
gets the rank from ~79.5 GiB down to roughly ~39.7 GiB + runtime before
any new bytes are pulled in, which is the headroom that makes Class B
safe.

**Class B — layers 22–42 (new to rank0 under TP, half-experts):**
For each layer `i` in `22..42`, in order:
1. Load (disk or peer-stream, §1.3) rank0's expert-half `E_0` weights
   plus the attention block for layer `i`.
2. `mx.eval(layer)` to force materialization, `mx.clear_cache()`
   immediately (mirrors the existing per-layer pattern in both
   `shard_model` and `pipeline_auto_parallel`, auto_parallel.py:509–545).
3. Advance to layer `i+1` only after step 2 completes — this is what
   keeps peak memory bounded to "N layers freed + 1 layer's worth of
   new weight in flight," not "all 22 layers' worth."

Interleaving A and B further (free layer i's half, then load a slice of
B, alternating) is possible but not necessary given the headroom math
below — doing Class A fully first is simpler to reason about and to
recover from (see §3), and the peak is already comfortable:

```
Start of swap (steady-state PP resident):     ~79.5 GiB (rank0)
After Class A complete (all 22 halves freed): ~39.7 GiB
Class B peak (39.7 + one extra layer ~1.8 GiB in flight): ~41.5 GiB
End of swap (steady-state TP resident):       ~77.7 GiB
```

Peak during the ENTIRE swap is ~79.5 GiB (the starting point, before
anything is freed) — the swap only ever *reduces* peak from there
until Class B pushes it back up to ~77.7 GiB at the end. **The
transient peak never approaches the ~118 GB budget; it stays under the
PP starting residency the whole time**, because Class A is intentionally
sequenced to complete before Class B starts. This is the single most
important property of the design: doing "free everything first, then
load everything" (rather than any interleaving that acquires before
freeing) makes the transient literally safer than steady state, not a
new risk.

The one thing that breaks this guarantee: **if MLX's allocator does not
actually release freed buffer memory back to the OS/GPU pool
immediately on `mx.clear_cache()`**. This is flagged explicitly in
Section 100 of the referenced doc as "Needs strict free-then-load
ordering in an allocator we do not fully control" — that caveat stands
here unchanged. `mx.clear_cache()` clears MLX's *internal* cache of
freed-but-retained buffers; it does not guarantee immediate OS-level
`munmap`/`free` if some other array still references overlapping
storage, and Metal's unified-memory allocator's actual release timing
under macOS is not something this fork controls or has instrumented at
the byte level. **This is a genuine blocker candidate, not merely
tedious**: before committing to this design, Phase 1 (§5) must measure
actual RSS immediately after each Class A layer's release, not just
trust `mx.clear_cache()`'s return. If release lags by even a few layers'
worth, the "peak never exceeds start" property degrades and the safety
margin has to come from elsewhere (e.g. forcing all Class B loads to be
smaller per-step, or eating a real wait between clear and next load).

### 1.3 Delta transfer: disk re-read vs peer-stream over TB5

Two candidate sources for Class B's ~37.9 GiB (rank0's half) / ~37.9 GiB
(rank1's half, symmetric) of new weight:

**(a) Re-read from local disk.**
Both nodes hold the full model on local disk (standard exo model
placement — every node has its own copy, no cross-node dependency to
load). Rank0 re-reads its TP-shard slice of layers 22–42 directly from
its own local SSD.

```
Cost = bytes / disk_throughput
     = 37.9 GiB / 5–7 GB/s
     = 5.4 – 7.8 s
```

Pro: no dependency on rank1, no coordination protocol, no jaccl/TB5
involvement — simplest failure mode (a local disk read either succeeds
or the process errors cleanly, no partner to desync from).
Con: re-reads bytes rank1 already has resident in RAM right now
(layers 22–42 are rank1's PP-native layers, fully loaded) — wasteful
but not disqualifying at this size.

**(b) Stream from peer over TB5/jaccl.**
Rank1 already holds layers 22–42 fully resident (PP layout, all
experts, its own layer range). Rank0's TP-shard half of those layers is
a strict subset of bytes rank1 already has in GPU-resident memory. Rank1
sends the appropriate expert-half slice per layer over the jaccl RDMA
link; rank0 receives directly into place, no disk I/O for rank0 at all.

```
Cost = bytes / TB5_effective_bandwidth
     ≈ 37.9 GiB / (measured ~450–546 GB/s memory bandwidth quoted,
       but that is LOCAL unified-memory bandwidth, not the TB5 link
       bandwidth — TB5's raw link is ~80 Gbps ≈ 10 GB/s per direction,
       and jaccl/RDMA overhead further reduces effective throughput
       below raw link rate)
     ≈ 37.9 GiB / ~6–8 GB/s realistic jaccl-over-TB5 throughput
     ≈ 4.7 – 6.3 s
```

**Important correction to avoid a modeling error**: the ~450–546 GB/s
figure in the hard facts is *local unified memory bandwidth within one
Mac Studio*, not the TB5 interconnect bandwidth between the two nodes.
TB5's link rate is on the order of 10 GB/s bidirectional (80 Gbps),
and real jaccl RDMA throughput observed on this cluster for the
existing PP pipeline hand-off traffic is the correct number to use here
— this design doc does not have a directly-measured jaccl bulk-transfer
number and treats ~6–8 GB/s as a *plausible estimate bounded by TB5's
link ceiling*, not a validated figure. **This must be measured in Phase
1 before the cost model is trusted** (see §5) — if jaccl's realistic
sustained throughput for large sequential transfers turns out
meaningfully lower (e.g. due to RDMA setup overhead per message, or
jaccl's transport being tuned for the small KV/activation messages the
pipeline hand-off uses today rather than bulk multi-GB weight streams),
option (a) may simply win outright and (b) is not worth building.

**Recommendation: build (a) first.** It's simpler, has no
cross-rank coordination during the swap (each rank's Class A/B pass is
fully local and independent — the two ranks can even swap on different
schedules without a synchronization barrier, since neither needs
anything from the other during the *weight* half of the swap; only KV
translation in §2 needs cross-rank coordination), and the cost (5.4–7.8s)
is already noise against a multi-minute 500K prefill. (b) is a strictly
harder engineering lift (new bulk-transfer path through jaccl, which
today is tuned for small pipeline messages, not multi-GB weight
streams) for a plausible but unverified ~1–3s improvement. Do not build
(b) unless (a) is measured and found to be a real bottleneck relative
to the KV translation and other swap-adjacent costs.

### 1.4 Total weight-swap wall time estimate

```
Class A (free 22 layers' worth of half-experts): sub-second — pure
    Python refcount + mx.clear_cache(), no I/O. Call it ~0.5–1s for
    22 layers' worth of per-layer eval+clear overhead (each existing
    per-layer eval+clear in shard_model/pipeline_auto_parallel already
    runs at model-load speed, which is much faster than model-load's
    own disk-read-bound steady state since no new bytes are read).
Class B (disk re-read of 37.9 GiB per rank):     5.4 – 7.8 s
-----------------------------------------------------------------
Total per-rank weight-swap wall time:            ~6 – 9 s
Both ranks run this independently and in parallel (no cross-rank
dependency for weights) — so wall time for the CLUSTER is ~6–9s, not
2x that.
```

Against a ~23-minute 500K-context PP prefill (Section 100's own number),
this is well under 1% overhead — confirms Section 100's own conclusion
that the transition cost itself is not the risk; the transient-peak-
memory correctness of the free-then-load ordering is.

---

## 2. THE KV TRANSLATION

### 2.1 What's actually different between PP's and TP's KV cache

- **PP**: KV cache is split by LAYER across ranks. Rank0 holds the
  `CacheList` entries for layers 0–21; rank1 holds layers 22–42. Neither
  rank has the other's layers' cache at all — by design, since each
  rank only ever computes attention for its own layer range.
- **TP**: attention is REPLICATED on every rank (`DeepseekV4ShardingStrategy`
  docstring: "Replicates attention on every rank; shards only the MoE
  block"). This means **every rank needs the FULL KV cache for ALL 43
  layers** — TP's forward pass computes attention independently and
  identically on both ranks (only the MoE block is actually sharded and
  reduced), so if a rank is missing a layer's KV cache it cannot compute
  that layer's attention at all.

At 500K context, 11.7 KB/token × 500,000 tokens ≈ 5.85 GB total KV
across all 43 layers. Under PP that's split ~2.9 GB/rank (roughly, by
layer count — not perfectly even since layer-parameter size varies
slightly, but close). Under TP each rank needs the **full** 5.85 GB.

### 2.2 The exchange

Each rank already holds its own layer-half of the cache (rank0: layers
0–21's ~2.9 GB; rank1: layers 22–42's ~2.9 GB). The missing half for
each rank is exactly what the OTHER rank already holds. This is a
**pairwise all-to-all of exactly one message each way**, not a
multi-round protocol:

```
rank0 → rank1: rank0's layers 0-21 KV cache   (~2.9 GB)
rank1 → rank0: rank1's layers 22-42 KV cache  (~2.9 GB)
```

Both transfers can run concurrently over jaccl's bidirectional TB5 link
(this is exactly the shape of traffic jaccl already handles for the PP
pipeline's activation hand-off between ranks, just larger in volume and
one-shot rather than per-token — so this reuses an existing, validated
transport path, unlike the bulk weight-stream option in §1.3(b) which
would need a new one).

```
Cost = 2.9 GB / jaccl_effective_bandwidth
     ≈ 2.9 GB / ~6-8 GB/s (same caveat as §1.3 — unmeasured, treat as
       estimate)
     ≈ 0.4 – 0.5 s per direction, concurrent, so ~0.4-0.5s wall time
```

This matches Section 100's own estimate ("plus ~1s of KV translation
over TB5") — this design's number is slightly more optimistic but in
the same ballpark; call it 0.5–1s and move on, since even the pessimistic
end is noise against the multi-second weight swap.

### 2.3 Cache object surgery: what moves, what's rebuilt, what's reused

exo's per-layer cache is `CacheList(RotatingKVCache, PoolingCache,
PoolingCache)` (per the task's framing — confirmed in
`mlx-lm/mlx_lm/models/cache.py`: `CacheList` at line 1155 is a thin
container wrapping a tuple of sub-caches, exposing `.state` /
`.meta_state` as parallel lists over `self.caches`).

- **`RotatingKVCache`** (cache.py:583): holds raw `keys`/`values`
  arrays plus bookkeeping (`offset`, `_idx`, `keep`, `max_size`). This
  is the part that must actually be TRANSMITTED — it's the bulk of the
  bytes (raw K/V tensors). **Must be moved, not rebuilt**: `keys` and
  `values` are `mx.array`s; the receiving rank needs the actual tensor
  data, which only exists on the sending rank. No cheap reconstruction
  is possible — this is real data transfer, not metadata sync.
- **`PoolingCache`** ×2 (cache.py:1270, one for MLA's compressed-KV pool,
  one for a second pooled stream per the task's framing — DSv4's
  indexer pooling and MLA's compression are both represented this way):
  holds `pooled` (step-allocated growing buffer), `buf_kv`/`buf_gate`
  (remainder buffer not yet forming a full window), plus scalar
  bookkeeping (`ratio`, `remainder`, `_pool_offset`,
  `_pending_offset_bump`). The `pooled`/`buf_kv`/`buf_gate` arrays are
  real data and must move exactly like `RotatingKVCache`'s
  `keys`/`values`. The scalar bookkeeping (`ratio`, `_pool_offset`, etc.)
  is small and can ride along in the same message trivially (it is not
  bytes-dominant, no separate optimization needed).
- **What can be reused in place, with no movement or rebuild**: nothing
  about a rank's OWN existing layer-half cache needs to move for itself
  — rank0 keeps its own layers 0–21 cache objects exactly as they are,
  in place, and only RECEIVES rank1's layers 22–42 objects to
  materialize alongside them. There is no cache "conversion" or format
  change between PP and TP for a given layer's cache — `RotatingKVCache`/
  `PoolingCache` are the SAME classes under both sharding modes (TP
  doesn't split KV differently per-layer, it just needs more layers'
  worth present). This is a meaningful simplification versus what
  Section 7 of the earlier doc worried about ("a real, nontrivial
  cache-format translation on the hot path") — the format is IDENTICAL;
  only the SET of layers each rank holds changes. The "translation" is
  really just "receive the other rank's cache objects and insert them
  into your own `CacheList`/per-layer cache dict at the right layer
  indices," not any bit-level reformatting.
- **Serialization surface**: `CacheList.state` / `.meta_state` (cache.py
  ~1170–1195) already exist as the mechanism for extracting a cache's
  full array + metadata state as a plain nested structure — this is
  designed for exactly this kind of external move (it's presumably used
  today for cache save/restore or similar). Reuse `.state`/`.meta_state`
  getters on the sending rank's per-layer cache objects for layers it's
  handing off, ship the resulting arrays via jaccl, and use the
  `.state`/`.meta_state` SETTERS on the receiving rank to reconstruct.
  This means the "cache object surgery" is close to zero new
  serialization code — the existing state accessors already do the
  array/metadata separation needed; new code is only the "collect one
  rank's layer-range into a batch of these, ship over jaccl, reassemble
  into the receiving rank's per-layer cache list at the right indices."

### 2.4 A subtlety worth flagging: RotatingKVCache's ordering state

`RotatingKVCache` is not a flat append-only buffer — `_idx`, `offset`,
and `keep` encode a rotating/circular write position (see `_temporal_order`,
cache.py ~605), used to bound cache size at long context by evicting
old entries in a ring rather than growing unboundedly. If PP's per-rank
cache for a layer has entries in rotated (non-temporal) physical order,
a naive raw-bytes copy to the peer would hand over a cache whose logical
token order does not match a linear `[0..offset)` walk. **This is not
disqualifying** — `_temporal_order()` already exists precisely to
re-linearize a rotated cache into true temporal order, and it's the
kind of operation that should run once as part of packaging the
handoff (on the SENDING rank, before serialization) rather than being
reinvented. Flagging explicitly: whoever implements this must call
`_temporal_order()` (or the equivalent already-fixed logic under
`.state`, if `.state` already does this — not confirmed from the
signatures alone and should be checked in the implementation phase) on
each `RotatingKVCache` before shipping it, or the receiving rank's
cache will silently point at the wrong tokens. This is TEDIOUS, not a
blocker — the fix is calling an existing method correctly, not new
design.

### 2.5 Total KV translation cost

```
~0.5 – 1s wall time (bidirectional, concurrent, jaccl/TB5)
+ negligible CPU time for state/meta_state (de)serialization and
  RotatingKVCache re-linearization (small, no measured number yet —
  should be sub-100ms for arrays already resident in GPU memory, but
  unverified, measure in Phase 1)
```

---

## 3. FAILURE / RECOVERY

### 3.1 The commit point

Define the swap as having exactly ONE commit point per rank, chosen to
be the LATEST point at which the rank can still cleanly abort back to
its starting layout without having discarded anything unrecoverable:

**Commit point = "Class B weight load for the LAST layer (layer 42)
completes `mx.eval()` successfully AND the KV exchange (§2.2) for both
directions has been ACKed by the peer."**

Before this point: the rank still has (a) enough of the old PP layout's
weights physically absent already (Class A ran first, see §1.2) that it
CANNOT trivially "roll back" to PP either — this is important and means
the transaction is not symmetric. Once Class A has freed the PP-only
expert halves for layers 0–21, going back to PP requires re-loading
those halves, which is itself a mini version of the same swap in
reverse. **This is the real design tension**: making the transient safe
(free-before-load) means there is no clean "instantaneous" rollback
point — recovery after a crash is always "finish going forward" or
"reload from scratch," never "cleanly undo."

### 3.2 Detecting a half-swapped boot

A runner that crashes mid-swap and restarts (or a fresh runner process
that inherits a crashed one's on-disk/in-memory state — relevant only
if state is persisted somewhere; if the runner process dies, MLX's
in-memory weight arrays die with it, so a plain process restart from
a runner supervisor actually can't observe half-swapped WEIGHT state at
all — it just cold-starts and reloads coherently either PP or TP from
scratch, whichever the runner is TOLD to load) needs a way to know, at
boot, "was a swap in progress." Two distinct crash scenarios:

1. **The whole runner process dies mid-swap.** No half-swapped IN-MEMORY
   state survives a process death — MLX arrays are process-local. The
   restarted process simply loads whichever layout the orchestrator
   (exo's Master, via the same placement/task machinery that already
   decides PP vs TP per model instance) tells it to load, from a clean
   process start. **This is actually the easy case and needs no new
   detection machinery** — it degrades to "runner crashed, restart it,"
   which exo's existing worker/master supervision already handles for
   any runner crash today.
2. **The crash is a cross-rank DESYNC, not a process death** — e.g.
   rank0 completes its swap and enters TP-decode mode, but rank1's swap
   hangs or fails (jaccl link drop mid-KV-transfer, disk I/O error on
   Class B, etc.), and rank0 is now waiting on a peer that will never
   respond, OR worse, rank0 proceeds into a TP forward pass assuming
   rank1 is also in TP layout when rank1 is still mid-Class-B. **This
   is the real failure mode to design for** — a cluster-level split
   layout, not a single-process crash.

Detection for case 2 requires an explicit **swap-phase barrier / cross-
rank state machine**, structurally similar to how the existing pipeline
hand-off already coordinates rank0/rank1 via typed messages (the
`PipelineFirstLayer`/`PipelineLastLayer`/queue-sends machinery in
auto_parallel.py is exactly this kind of coordination primitive, just
for per-token pipeline flow rather than a one-time phase transition).
Concretely:
- Define an explicit `SwapPhase` enum: `PP_STEADY`, `SWAP_CLASS_A`,
  `SWAP_CLASS_B`, `SWAP_KV_EXCHANGE`, `TP_STEADY`.
- Each rank publishes its own `SwapPhase` to the peer (and/or to the
  Master, reusing exo's existing event-sourced `GLOBAL_EVENTS`/
  `LOCAL_EVENTS` pub/sub topics already documented in AGENTS.md — this
  is exactly the kind of state transition the Master's event log is
  built to track) BEFORE beginning each phase transition, not after.
- Neither rank may enter `TP_STEADY` (i.e. start accepting/serving TP
  decode requests) until it has observed the PEER also reach
  `SWAP_KV_EXCHANGE`-complete. This is a two-phase-commit-flavored
  barrier, not full 2PC — there's no need for a coordinator vote, just
  a mutual "I'm ready, are you ready" handshake before either side
  starts trusting the other's cache/weights are in TP shape.
- If a rank observes the peer's `SwapPhase` has not advanced within a
  timeout, or observes a peer disconnect (jaccl link-down, which the
  existing PP transport already must detect for its own per-token
  send/recv liveness), it aborts the *whole cluster's* swap: both ranks
  are instructed (Master-driven, via the same command topic used for
  model placement decisions) to reload cleanly from disk into a KNOWN
  layout (almost certainly PP, since that's the layout more likely to
  still be intact for whichever rank hadn't started Class A yet) rather
  than trying to resume a partial swap. **This "reload from scratch on
  any desync" policy is a deliberate simplification** — a smarter
  design could try to resume Class B from wherever it left off, but
  given the swap only costs ~6-9s total (§1.4), a full reload-on-failure
  policy is far cheaper to implement correctly and far cheaper in wall
  time than debugging resumable partial-swap state, and matches this
  fork's stated priority (per the referenced doc's Section 8, item 5)
  of correctness over speed for exactly this class of cross-rank state
  bug.

### 3.3 Why "recover without a full model reload" is optimistic scope

The task description asks how a runner recovers "without a full model
reload." **Being honest: for case 2 (cross-rank desync), reload-from-
scratch is the recommended answer, not something to design around.**
A full model reload for one rank costs on the order of the swap itself
plus base model-load time (the ~85.68 GB/node floor implies the base
load itself is not free — likely tens of seconds from local NVMe, not
measured precisely here). This is more expensive than the swap's own
~6-9s, but it is FAR less code and FAR less risk than building a
resumable partial-swap protocol for what should be a rare event (a
mid-swap crash/desync, not a steady-state occurrence). Recommend
treating "no full reload" as a Phase-2-or-later optimization (§5), not
a Phase-1 requirement — ship the safe-but-slower reload-on-failure path
first, and only build incremental-resume if telemetry shows swap
failures are common enough to matter.

---

## 4. WHERE IT PLUGS IN

### 4.1 The two sharding strategies (already exist, no changes needed to them for Phase 1)

- **TP**: `DeepseekV4ShardingStrategy.shard_model`,
  `src/exo/worker/engines/mlx/auto_parallel.py:1031`. This is the
  function to call to materialize the TP layout for layers 22–42
  (Class B, §1.2) and to compute which expert-half convention
  (`E_0`/`E_1`, §1.1) applies for layers 0–21.
- **PP**: `pipeline_auto_parallel`, `src/exo/worker/engines/mlx/auto_parallel.py:509`,
  which wraps `layers[0]` in `PipelineFirstLayer` and `layers[-1]` in
  `PipelineLastLayer` (auto_parallel.py:536-537) — this is the state the
  runner starts in and must return to on full reload (§3.3).
- The swap driver is new code that does NOT belong inside either
  strategy class — it's a phase TRANSITION orchestrator that calls into
  both, one class at a time, per the Class A/B ordering in §1.2. Natural
  home: a new module, e.g.
  `src/exo/worker/engines/mlx/phase_swap.py`, with a top-level
  `swap_pp_prefill_to_tp_decode(model, pp_group, tp_group, model_shard_meta) -> Generator[SwapProgress, None, nn.Module]`
  mirroring the existing `Generator[ModelLoadingResponse, None, nn.Module]`
  shape both `shard_model` and `pipeline_auto_parallel` already use (so
  progress reporting plugs into whatever consumes `ModelLoadingResponse`
  today, e.g. dashboard load-progress UI, with minimal new wiring).

### 4.2 The phase boundary — where prefill ends and decode begins

Concretely, in `src/exo/worker/engines/mlx/generator/generate.py`:
- `set_pipeline_prefill(model, is_prefill)` at line 369 is the EXISTING
  per-request phase flag setter — it flips `PipelineFirstLayer`/
  `PipelineLastLayer.is_prefill` and, for the metaframe path,
  `MetaFramedPipelineLastLayer.is_prefill` plus
  `set_forward_step_phase(ForwardPhase.PREFILL_FINAL | ForwardPhase.DECODE)`.
  **This is the exact call site that currently marks "prefill is done,
  decode begins" for a single PP-only request** — it is called once
  prefill's last chunk completes and before the first decode step's
  forward pass. This is the natural hook point for the swap trigger:
  the moment code is ABOUT to call `set_pipeline_prefill(model, False)`
  for the last time on a request that's about to enter its own decode
  loop is precisely the phase boundary this design needs to intercept.
- `prefill()` (generate.py:709) and `pipeline_parallel_prefill()`
  (generate.py:671) are the PP-mode prefill entry points — the swap
  must be sequenced to start only AFTER these return (i.e. after the
  full prompt has been consumed and the KV cache for the request is
  complete under PP), never mid-prefill.
- On the batch-generation side, `src/exo/worker/engines/mlx/generator/batch_generate.py`'s
  `ExoBatchGenerator` (line 686) and its internal `_step_batched_decode`
  (referenced around line 605, though the file also shows
  `_batched_decode_active`/`_batched_decode_rank0_glue`/
  `_batched_decode_rank1_glue` fields around lines 780-792) is where the
  PP-specific batched-decode machinery lives today — under this design,
  for a genuinely SINGLE model instance transitioning PP→TP once per
  session (not per-request), the swap is a MODEL-INSTANCE-level event,
  not a per-request one. The natural boundary is: **the FIRST request's
  prefill in a session completes under PP; the swap runs once; every
  decode step for that request AND all subsequent requests in the same
  session (while prefix-cache-sharing the same KV) runs under TP.**
  This matches Section 100's own framing ("with prefix caching only the
  FIRST turn pays it" — the swap cost, like the PP-prefill win itself,
  is a once-per-session cost, not once-per-request).

### 4.3 Concurrency implication that must be designed around, not ignored

TP's real advantage is concurrency (Section 100: "TP's genuine,
unarguable advantages are concurrency and cancellation, not
single-request latency"). If a SECOND request's prefill arrives while
the model instance is already in TP-decode mode (post-swap, serving
request 1's decode), that second request's prefill would, per the
directive, want to run under PP too — but the instance has already
swapped OUT of PP layout. **This design does not solve that case** —
it is explicitly a single-session, single-swap-per-instance design.
Multi-request concurrent PP-prefill-while-TP-decode-is-active would
require EITHER a second model instance (doubling memory, defeating the
whole point) OR a more complex per-request phase-aware scheduler that
this doc does not attempt to design. **Flag this as an open scope
question for whoever picks up implementation**: is the swap meant to
happen once at session start and stay in TP for the whole session
(simple, matches this design), or does every new request's prefill need
its own swap-back-to-PP-then-forward-to-TP cycle (which would make the
~6-9s swap cost NOT noise anymore if it happens on every turn, not just
turn one)? This document assumes the former (once per session, matches
Section 100's "once per session" framing) — the latter is a
substantially different, harder design not covered here.

---

## 5. PHASED PLAN

### Phase 1 — prove the swap is physically possible, no live request involved

**Goal: measure real transition time and real peak memory, with zero
correctness risk (no request in flight, no output to validate).**

- Load a runner in PP layout (existing `pipeline_auto_parallel` path,
  unmodified).
- Run the Class A / Class B / KV-exchange sequence from §1–2 with a
  SYNTHETIC KV cache (e.g. fabricate a `CacheList` with some chosen
  context depth's worth of dummy `RotatingKVCache`/`PoolingCache`
  content, no real request behind it) rather than a live request's real
  cache.
- Measure, via the same RSS/memory_pressure sampling this fork already
  uses elsewhere (per the "memory_pressure says 97% free" style
  measurement referenced in the hard facts): peak resident memory
  DURING the swap (validate the §1.2 claim that peak never exceeds PP's
  own starting residency), and wall-clock time for Class A, Class B,
  and KV exchange separately.
- Explicitly validate the §1.1 expert-half-convention assumption (dump
  which expert indices `DeepseekV4ShardingStrategy` assigns rank0 vs
  rank1 today, confirm it's the SAME every run).
- Explicitly measure real jaccl bulk-transfer throughput (the §1.3/§2.2
  "~6-8 GB/s, unmeasured" placeholder) — this is the single most
  consequential unmeasured number in this whole design; Phase 1 must
  close it before Phase 2 makes any go/no-go call on disk-vs-peer for
  weight streaming.
- **Independently testable, lands value even if Phase 2 never ships**:
  this phase alone answers "is the swap even physically possible within
  the memory budget," which is the load-bearing assumption for the
  entire user directive. If Phase 1 finds the transient peak DOES
  exceed budget (e.g. because `mx.clear_cache()` doesn't release as
  cleanly as assumed), that's the single most important thing to learn
  before writing one more line of swap-orchestration code — it would
  mean returning to interleaved Class A/B ordering (freeing and loading
  layer-by-layer in lockstep, smaller working set per step, more
  complex code) rather than the simpler "free all, then load all"
  sequencing this doc proposes.

### Phase 2 — swap with a real single-request, PP-prefill-then-TP-decode, correctness validated

- Take a real prompt, run PP prefill to completion (`prefill()`/
  `pipeline_parallel_prefill()`, unmodified), trigger the swap at the
  natural boundary identified in §4.2, then run TP decode for the same
  request using the migrated KV cache.
- Validate correctness the same way this fork validates other batching/
  sharding changes elsewhere in this doc's tree of sibling designs
  (byte-for-byte at temp=0 against a known-good SERIAL baseline — e.g.
  the same prompt run PP-prefill → PP-decode with no swap at all,
  compare tokens).
- This is the phase that actually tests the KV translation (§2) for
  real — Phase 1's synthetic cache only tests the mechanics of moving
  bytes, not that the moved bytes still produce correct attention
  output once used by real TP forward passes.
- **Independently testable, lands value even if Phase 3 never ships**:
  a single-request swap that's provably correct is itself a complete,
  shippable feature for the common case (one request per session,
  prefix caching across turns as already noted).

### Phase 3 — failure/recovery (§3) hardening

- Implement the `SwapPhase` barrier and cross-rank handshake (§3.2).
- Inject synthetic failures (kill one rank mid-Class-B, drop the jaccl
  link mid-KV-exchange) and confirm the reload-from-scratch fallback
  (§3.3) actually recovers the cluster to a clean, single, agreed-upon
  layout rather than leaving it split.
- **Independently testable**: this phase's value is entirely
  operational robustness — Phase 2 already proves the happy path works;
  this proves the unhappy path degrades safely rather than catastrophically.

### Phase 4 — measure the actual requirement: 30 tok/s decode at 500K, post-swap

- Only after Phases 1–3 are solid: run the full 500K-context PP-prefill
  → swap → TP-decode-with-MTP pipeline end to end and measure decode
  throughput against the 30 tok/s bar.
- This is explicitly gated behind Section 100's own "fix the 34x bug
  first" sequencing note and its call for the un-run 2×2 (`{PP,TP} x
  {short,500K}`, with/without MTP) — this design's Phase 4 is exactly
  the "TP decode at 500K, with MTP, after a real PP prefill and real
  swap" cell of that matrix, which per Section 100 IS the requirement
  itself and has never been measured under any config.

---

## 6. What is genuinely hard vs merely tedious — summary

**Genuinely hard / real open risk (not just work):**
1. Whether `mx.clear_cache()` + Python refcount-to-zero actually
   releases GPU-resident buffer memory back to a usable pool WITHIN the
   timeframe needed to keep the transient peak bounded (§1.2's central
   safety claim depends on this and is currently an assumption, not a
   measurement). If this doesn't hold, the whole "free-then-load keeps
   peak below PP's own starting point" argument weakens and a more
   defensive (and complex) interleaved ordering is needed instead.
2. Real jaccl sustained bulk-transfer throughput for multi-GB payloads
   is unmeasured — every cost-model number in §1.3(b) and §2.2 that
   depends on it is a plausible estimate, not a validated figure. This
   doesn't block the design (option (a), disk re-read, doesn't depend
   on it) but it DOES block trusting the KV-exchange cost (§2.2), which
   has no viable non-jaccl alternative (the peer literally has the only
   copy of the missing KV half).
3. The multi-request concurrency scope question (§4.3) is unresolved by
   this design and deliberately out of scope — flagged so it isn't
   silently assumed solved.

**Merely tedious (real work, not a blocker):**
- `RotatingKVCache` temporal re-linearization before shipping (§2.4) —
  known fix, known existing method, just needs to be called at the
  right point.
- The `SwapPhase` barrier / cross-rank handshake (§3.2) — standard
  distributed-coordination code, no unknown unknowns, similar in kind
  to work this fork has already done for PP's per-token pipeline
  hand-off.
- Wiring the new `phase_swap.py` module into the existing
  `ModelLoadingResponse`-generator progress-reporting convention (§4.1)
  — mechanical, matches an existing pattern.
- Reload-from-scratch on failure (§3.3) — more expensive at runtime
  than a hypothetical resumable-partial-swap design, but far less risky
  to implement; explicitly recommended over the harder alternative.

**Not attempted / explicitly out of scope for this document:**
- Multi-request concurrent PP-prefill-while-TP-decode-active (§4.3).
- A resumable (non-reload) recovery path for cross-rank desync (§3.3) —
  flagged as a possible future optimization, not designed here.
