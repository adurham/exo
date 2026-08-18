# DSv4-Flash: Metal Event::wait stalls — root-cause direction & next step (2026-08-18)

## Purpose

Follow-up to `dsv4-220k-prefill-rdma-wait-breakdown-2026-08-18.md`. That
doc identified 8 discrete Metal `Event::wait` stalls (3.0s–11.7s each,
71.6s total ≈ 46.6% of the 220K-token prefill's comms wall time)
correlated 1:1 with `rounds=1` on 16MB MoE `all_sum` calls. It left
open: does the retransmit CAUSE the stall (Story B) or is the retransmit
a SYMPTOM of a GPU-side stall (Story A)?

This doc executes the three zero-cost checks from an external senior
review before any new cluster repro, and documents what they resolved
and what they did not.

## Zero-cost checks executed

All against artifacts already on disk: `mlx/mlx/backend/metal/event.cpp`,
`mlx/mlx/distributed/jaccl/lib/jaccl/{jaccl.cpp,lib/jaccl/mesh_impl.h}`,
and `~/exo_investigation_2026-08-18/jaccl_v2_raw.log` (14,919 paired
ENTER/EXIT events). No cluster touched.

### Check 1 — Who signals the stuck event? (code read)

The `EventImpl` waited on in `event.cpp` is a `MTL::SharedEvent`.
Reading `Event::signal(Stream stream)` in the same file (lines ~180-233)
and `Event::wait(Stream stream)` (lines ~161-176): on GPU streams the
signal is encoded onto the Metal command encoder
(`encoder.signal_event(...)`) and fires only when the GPU command
buffer runs to that point; on CPU streams the wait is enqueued on the
CPU scheduler thread.

The MoE `all_sum` in `jaccl.cpp:105-116` calls
`cpu::get_command_encoder(stream).dispatch(...)` on the JACCL group's
pinned CPU communication stream. That CPU dispatch is what invokes
`group_->all_sum(...)` → `reliable_all_reduce_v2()` in `mesh_impl.h`.
The `input` and `output` arrays are `set_input_array`/`set_output_array`
on the CPU encoder; MLX inserts a cross-stream event wait between the
GPU stream that PRODUCES the input tensor and this CPU comms stream.

**Conclusion:** the event being waited on is signaled by GPU command
buffer completion of the compute that produces the `all_sum`'s input
tensor. This is exactly the Story A architecture. A GPU/driver stall
on THIS rank delays the collective start, which is then reflected as
long ENTER→EXIT latency (and the peer, also stalled on its own
mirror-side signal, sees the same). It is not a transport-side event
signaled on collective completion.

### Check 2 — Is rounds=1 exclusive to the 10 outliers?

**Distribution across all 14,919 rank-0 calls:**

| rounds | count | median dur | max dur |
|--------|-------|------------|---------|
| 0      | 5,803 | 0.0 ms     | 0.27 s  |
| 1      | 9,116 | 8.0 ms     | 11.70 s |

**All 5,803 rounds=0 calls are `small=1` (small path). All 9,116
rounds=1 calls are `small=0` (large path). All 8,988 16MB calls are
rounds=1.**

Reading `reliable_all_reduce_v2` in `mesh_impl.h:970-1008`: the large
path performs a mandatory `coordinator_->reliable_barrier(...)` on
successful `data_done` and unconditionally does `round++`. On a clean
call `round` therefore leaves the function equal to 1, not 0.

**Conclusion:** `rounds=1` is the ARCHITECTURAL BASELINE for the large
path, not a retransmit signal. The 10 outliers are simply the 10
slowest of the 9,116 large-path calls; their `rounds=1` value carries
no forensic signal at all about retransmits. **This decisively falsifies
Story B** — no evidence a retransmit round ever fired on the outliers;
the `rounds` counter in the previous doc was misinterpreted.

### Check 3 — Do the 8 stalls land on chunk boundaries?

The prefill drives 2048-token chunks (`EXO_PREFILL_STEP_SIZE=2048`) with
`EXO_PREFILL_CLEAR_CACHE_INTERVAL=1` (cache cleared every chunk). If
stalls were `mx.clear_cache()`-induced, they should land at the FIRST
16MB `all_sum` of a chunk — visible as a large inter-ENTER gap (the
gap containing the CPU-side clear_cache + first-forward-pass cost)
immediately BEFORE the outlier's ENTER.

Inter-ENTER gap on rank 0 immediately before each outlier:

```
cid=3114  gap_before=0.154s
cid=3115  gap_before=1.408s     (fallout of 3114's stall)
cid=4076  gap_before=0.046s
cid=4125  gap_before=0.062s
cid=4920  gap_before=0.045s
cid=7641  gap_before=0.075s
cid=7642  gap_before=11.698s    (fallout of 7641's stall)
cid=10000 gap_before=0.136s
cid=10017 gap_before=0.078s
cid=11176 gap_before=0.078s
```

The largest gaps in the whole run (60.99s before cid=3106, 29.17s
before cid=4, 17.93s before cid=2380) do NOT sit adjacent to outliers.
Every genuine outlier onset (dropping the 3115/7642 stall-fallout
pairs) has a sub-200ms inter-ENTER gap — the stalls happen MID-CHUNK,
not at a boundary.

**Conclusion:** `mx.clear_cache()` at chunk boundaries is not the
trigger. The 8 stalls are intra-chunk events.

## Root-cause direction (established)

The event is GPU-signaled (Check 1). The retransmit counter is
architectural noise, not a stall driver (Check 2). Chunk boundaries
do not align with stalls (Check 3). **Story A holds: the Metal command
buffer whose completion would signal the awaited event fails to run to
that signal promptly, for 3–12s at a time, on ~8 mid-chunk events per
220K-token prefill.** The 8 events are on both ranks within 1–2 ms of
each other, which is trivially explained by both ranks entering the
collective in TP lockstep and both firing the log at the same
hardcoded 3.0s elapsed threshold in `event.cpp:126-129` — the
lockstep is not independent corroboration.

**What is NOT yet established:** WHERE in the command-buffer lifecycle
the stall sits (not committed / committed-not-scheduled /
scheduled-not-completed). Candidates the code invites, in the order
they should be tested:

1. `EXO_DSV4_FENCE_EVERY_N_LAYERS=4` + `EXO_DSV4_FENCE_ASYNC=1` —
   async-fence pattern producing intermittent scheduling backpressure
   or completion-handler pile-up.
2. Metal driver allocator hiccup (a large evict/refault under memory
   pressure at 220K context, independent of clear_cache-per-chunk).
3. Interaction between the CPU comms encoder's cross-stream event
   wait ordering and the GPU compute stream's in-flight command buffers
   (i.e., a stream-serialization edge case, not a straight driver
   stall).

## Recommended next step (single-shot)

Per the external review's plan, the one call that discriminates all
three candidates is the already-built `EXO_CMDBUF_RING_DIAG=1` dump.
`event.cpp:143-159` already invokes `metal::dump_recent_command_buffers`
at exactly the moment `elapsed_us >= 3'000'000` — the same threshold
that produced our 8 log lines. Both env vars are plain launcher vars
already allow-listed in `start_cluster.sh`; no code change needed.

Concrete relaunch envelope (keep the standing `MLX_JACCL_DATA_RECV_POOL=0`
fix — cluster will fail to load without it):

```
MLX_JACCL_DATA_RECV_POOL=0 \
EXO_CMDBUF_RING_DIAG=1 \
JACCL_TRACE_PROGRESS=1 \
EXO_PROFILER=spans \
./start_cluster.sh
```

Then repeat the same 220K-token needle-in-haystack prefill used to
produce the two prior 2026-08-18 docs. Interpretation of the ring-diag
output (from `event.cpp:143` comment):

| Ring-diag says              | Story                                               |
|----------------------------|-----------------------------------------------------|
| buffer NOT COMMITTED       | CPU-side / scheduling bug (fence-async, ordering).  |
| COMMITTED, NOT SCHEDULED   | Driver / allocator stall (a candidate #2).          |
| SCHEDULED, NOT COMPLETED   | GPU execution stall (candidate #1 most likely).     |

Budget: one repro, expect one more iteration after it to convert
"where stuck" into "why stuck" (the review's expectation-setting
note). Do NOT attempt a fix before this dump exists — the code
already narrows to three families, and picking the wrong one costs
another 15-minute cluster launch cycle plus a re-verified 220K prefill.

## What was NOT done and why

- No live cluster relaunch. The three zero-cost checks materially
  moved the diagnostic (Story B falsified, chunk-boundary cause ruled
  out) before spending any cluster time.
- No code change. Every candidate root cause identified above is a
  Metal/driver-level or scheduler-ordering question the source
  reading cannot answer alone; a mitigation (retry, backoff,
  cache-invalidation) would be rejected under the standing
  root-cause-only rule.
- No modification of the two existing 2026-08-18 docs. Their headline
  numbers stand — this doc supersedes only the `rounds=1`
  interpretation embedded in the second doc's "Root cause of the
  outliers" section (rounds=1 is universal on the large path, not a
  retransmit indicator).

## Files & data preserved

- `~/exo_investigation_2026-08-18/jaccl_v2_raw.log` — the paired
  ENTER/EXIT log source used by both this doc and the prior one.
- `~/exo_investigation_2026-08-18/{analyze_jaccl2.py,check_eventwait.py}`
  — disposable analysis scripts. Extended in this doc's Check 2/3
  with rounds-vs-small and inter-ENTER gap analysis (results
  reproduced inline above; scripts not re-committed).
