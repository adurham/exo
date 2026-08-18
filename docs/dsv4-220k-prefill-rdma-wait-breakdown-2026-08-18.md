# DSv4-Flash: RDMA/GPU-sync wait breakdown within the 220K prefill (2026-08-18)

## Question being answered

Follow-up to `dsv4-220k-prefill-span-profile-2026-08-18.md`: of the
18.0% of prefill wall time attributed to cross-node collectives
(`attn.all_gather` 8.5% + `moe.all_sum` 9.5%), how much is genuine
RDMA/sync wait versus other overhead?

## Method

Relaunched cluster with `JACCL_TRACE_PROGRESS=1` added on top of the
standing `MLX_JACCL_DATA_RECV_POOL=0` fix (both plain launcher env
vars, no source changes). Fired a fresh, uniquely-seeded 220K-token
prompt (same methodology as the span-profile doc — cache-busted,
verified `cached_tokens: 0`, verified correct needle-in-haystack
answer).

`JACCL_TRACE_PROGRESS=1` does NOT emit a ready-made per-call
`elapsed_us` field for the TP collective path used here (that field
only exists in the raw PP `send()`/`recv()` p2p functions in
`mesh_impl.h`, not in `reliable_all_reduce_v2` — the function this
workload actually calls, selected via `MLX_JACCL_RELIABLE_OPTIMISTIC=1`
+ populated `pool_connections_`). Instead it emits `[jaccl-v2] ENTER`
and `[jaccl-v2] EXIT` log lines with millisecond-resolution timestamps
per `call_id`. Real per-call duration was computed by pairing
ENTER→EXIT timestamps (`/tmp/analyze_jaccl2.py`, not committed —
disposable analysis script). 14,919 unique collective calls were
captured with zero unmatched ENTER-without-EXIT (no dropped/hung
calls at the end of the run).

## Headline numbers

- **Total collective (ENTER→EXIT) time summed across the whole
  220K-token prefill: 153.8s of a 803.9s wall-clock segment = 19.1%.**
  This closely corroborates the independent `EXO_PROFILER=spans`
  measurement from the companion doc (18.0% comms: all_gather 8.5% +
  all_sum 9.5%) — two different instrumentation layers (MLX-model-code
  spans vs. jaccl-transport-level call timing) agree to within ~1
  percentage point. High confidence in both numbers as a result.
- 14,919 total collective calls. The overwhelming majority are cheap
  and unremarkable: median 5.0ms, p90 9.0ms — normal RDMA round-trip
  cost for the message sizes involved.
- **10 severe outlier calls, ranging 1.27s to 11.70s each, sum to
  71.6 seconds — 46.6% of the ENTIRE 153.8s comms total, concentrated
  in 0.067% of the calls.** All 10 outliers are EXACTLY
  `total_bytes=16777216` (16MB, matching a full 2048-token-chunk ×
  4096-hidden bf16 MoE `all_sum` payload) and all have `rounds=1`
  (the ARQ protocol's retransmit-round counter — meaning the fast
  path did not complete cleanly and a retransmit round was needed).

## Root cause of the outliers (found, not inferred)

Every one of the 10 outlier windows contains, on BOTH ranks, at
matching millisecond timestamps (verified — e.g. 15:58:33.174 rank0
vs 15:58:33.173 rank1), the line:

```
[Event::wait] slow wait: elapsed=3.0s signaled=0 target=1 (polling; self-abort at 20000ms)
```

This is an **MLX Metal GPU shared-event wait stalling**, not a
network/RDMA-hardware-level stall — both ranks are parked waiting on
a Metal event signal that fails to arrive promptly, in lockstep with
each other. 8 total `Event::wait slow` occurrences were found across
the whole run (both nodes logged all 8, matching to the millisecond),
and each one falls inside the ENTER→EXIT window of one of the 10
outlier calls (two outlier pairs — 3114/3115 and 7641/7642 — share a
single underlying stall event whose fallout spans two consecutive
collective calls).

The very next collective call after each stall (same 16MB payload
size) completes in single-digit milliseconds — so this is NOT "large
payloads are inherently slow." It is an intermittent stall, not a
structural cost of the message size.

## Interpretation

This refines, not contradicts, the companion span-profile doc's
compute-bound verdict. The aggregate 58%/42% attn/ffn split against
~18-19% comms is real. But within that comms slice, **roughly half
(9% of total prefill wall time) is not smoothly-distributed
transfer/wait cost — it is 8 discrete, pathological GPU-event-signal
stalls**, each costing multiple seconds, landing on full-size MoE
all_sum calls.

This is a materially different kind of lever than "shave compute" or
"optimize the collective's algorithm":
- It is isolated (8 events in ~4730 layer-forward passes ≈ 1 per ~590
  layers, or roughly 1 per full prefill request given ~4730 total
  MoE all_sum calls this run).
- It has a specific, reproducible-looking trigger signature (always
  the full 2048-chunk-size 16MB payload, always `rounds=1`,
  Metal-event-wait specifically — not TCP/RDMA-transport-level).
- If root-caused and fixed, recovering even half of the 71.6s lost
  here would be a ~4-5% throughput win on this specific 220K request
  — a real, surgical improvement, distinct from and additive to any
  gain from the seq-split A/B or other levers noted in the companion
  doc.

## What this does NOT establish

- Root cause of WHY the Metal event fails to signal promptly on these
  specific calls is not yet investigated. Candidates worth checking
  next: whether these 8 stalls correlate with `EXO_PREFILL_STEP_SIZE`
  chunk boundaries, `EXO_PREFILL_CLEAR_CACHE_INTERVAL=1` cache-clear
  timing, GC pauses (`EXO_GC_COLLECT_INTERVAL=0` is set — GC should be
  disabled, ruling that out unless the setting isn't taking effect),
  or genuine Metal command-queue contention from concurrent Python
  work on the same core. Not established in this session.
- Not established whether this stall rate (8 per 220K-token, ~110-chunk
  request) scales linearly, sub-linearly, or unpredictably with
  context length or chunk count — only one run at one context size was
  measured.
- Does not revive or relate to the dead PP-prefill/TP-decode hybrid
  design (see warm memory fact ~1472) — this is pure TP-path
  transport-layer forensics.
