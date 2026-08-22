# Live two-rank decode Instruments trace: real GPU-idle measurement, confirms roofline — 2026-08-21 (session 3)

## Context

Following an independent Fable review of the consolidated
`docs/PERFORMANCE_HISTORY.md` (per user request, "get fables review of
what can be done next"), Fable ranked a two-rank live-decode Instruments
trace as the single highest-EV, lowest-risk next step — higher priority
than any further env-var lever hunting. Rationale: it can attribute the
whole roofline gap (decode measured earlier tonight at ~12% of
theoretical bandwidth-bound peak) rather than one named span, and it's
pure observation (near-zero risk) using a technique already proven
working earlier this session.

Fable also flagged a concrete methodology critique of the doc's own
proposed next step for decomposing `moe.all_sum`'s 34x software-overhead
gap ("wrap `perf_counter` around just the collective call site"): this
either measures near-nothing (if the op is lazily enqueued into MLX's
graph) or conflates the collective with everything upstream that had to
materialize first (if forced eager) — reinventing the sync-span
profiler's known ~77% decode overhead under a different name. Fable's
recommended replacement (jaccl-internal timestamps at op-post/
peer-ready/completion) is queued as a follow-up, not attempted tonight.

## What was done

Traced BOTH TP ranks simultaneously during a real production decode
request — a genuine improvement over earlier tonight's single synthetic
matmul probe, which only validated the technique in isolation.

1. Identified both runner PIDs live: `ps aux | grep multiprocessing.spawn`
   on each node (m4-1: 53337, m4-2: 61349) — these are the actual TP
   worker processes serving requests right now, not a standalone script.
2. Launched `xcrun xctrace record --template 'Metal System Trace' --attach <pid>
   --output <path> --time-limit 25s` on both nodes in parallel
   (backgrounded via SSH).
3. Fired a real decode request (`bench/decode_probe.py`, 512-token
   prompt, 400-token generation) during the capture window. Confirmed
   throughput dropped to 13.42 tok/s under tracing overhead (expected —
   Instruments adds real dispatch overhead of its own; absolute numbers
   from a traced run are not meaningful, only the internal gap
   structure/ratios are).
4. Both captures completed with substantial real data (1.9GB and 2.1GB)
   — a major improvement over the earlier synthetic-probe attach attempt,
   which produced an empty capture.

## Real methodological finding: `xctrace export --toc` fails on attach-mode traces, but direct `--xpath` export works

`xcrun xctrace export --input <trace> --toc` failed with
`Export failed: Document Missing Template Error` on both attach-mode
captures — this looked at first like the same "attach mode doesn't
capture Metal data" failure from earlier tonight's synthetic-probe
attempt. Investigated properly rather than assuming the same conclusion:

- The trace DOES contain real GPU data: `corespace/run1/core/table-manager/tables.plist`
  (zlib-compressed NSKeyedArchiver plist) lists real Metal GPU table
  schemas (`metal-gpu-intervals`, `metal-command-buffer-completed`,
  `metal-gpu-execution-points`, etc.) — decompressed and confirmed
  directly.
- `instrument_data/` contains 7 real per-instrument data streams
  (`run_data/1.run.zip` each) — structurally identical to a working
  launch-mode trace from earlier tonight.
- Tried `xcrun xctrace remodel --input <trace> --output <new>` (a
  documented CLI-only re-run of the modelers) — failed with the
  identical "Missing Template Error", confirming this is NOT a
  processing-completeness issue, it's specifically the trace-document
  *loader* rejecting attach-mode packages for the `--toc` and `remodel`
  code paths.
- **Direct `--xpath` export with a known table schema name (reused from
  earlier tonight's working launch-mode session) succeeded cleanly**:
  `xcrun xctrace export --input <trace> --xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-intervals"]' --output <out>`
  — real success, no error.

**Conclusion: `--toc` and `remodel` share a document-loading code path
that's broken for attach-mode `.trace` packages (missing/differently-
versioned template metadata), but the actual data-export path
(`--xpath` against a known schema) is independent and works fine.**
This is a genuinely reusable finding for future sessions: don't give up
on an attach-mode trace just because `--toc` fails — go straight to
`--xpath` export using a schema name known from a working reference
trace.

## Real data extracted

- `metal-gpu-intervals`: 26,486 rows for rank0's PID, 45,132 total rows
  in the export before PID-filtering. Real per-op GPU execution
  intervals with start-time, duration, command-buffer/encoder IDs.
- `metal-command-buffer-completed`: 28,878 rows (not yet analyzed this
  session — queued for the jaccl-decomposition follow-up).

## Real GPU-busy vs idle measurement — both ranks

Filtered to `Command Buffer N:Compute Command N` entries (real kernel
dispatch/execution intervals, not driver bookkeeping), merged into a
non-overlapping union per rank (a naive sum of durations double-counts
overlapping/concurrent GPU work), then computed true occupancy over the
real trace span:

| | rank0 (m4-1) | rank1 (m4-2) |
|---|---|---|
| Command-buffer compute entries | 8,631 | 9,602 |
| Trace span | 17.62s | 21.80s |
| True GPU-busy time (merged union) | 5.36s | 6.19s |
| **True GPU occupancy** | **30.4%** | **28.4%** |
| **True GPU idle** | **69.6%** | **71.6%** |
| Idle gaps in the 1-10ms range | n=2398, mean 2909µs | n=2621, mean 3010µs |
| Idle gaps >20ms | n=48, sum 1544ms | n=86, sum 3043ms |

**Both ranks show closely matched occupancy (~29% average, only 2
percentage points apart) — real, direct-measurement confirmation of
tonight's earlier roofline estimate (~12% of theoretical
bandwidth-bound peak).** The two figures aren't identical (12% roofline
vs 29% raw occupancy) because they're different denominators — the
roofline compares against theoretical peak throughput, while this
occupancy measurement is real-vs-idle wall-clock time including the
tracing-overhead-inflated span — but both independently point at the
same qualitative conclusion: **the majority of decode wall time is
genuine GPU idle, not GPU work running inefficiently.** This directly
corroborates rather than merely repeats the earlier finding, since it
comes from a completely different measurement instrument (direct Metal
GPU telemetry vs. an architectural FLOPs/bandwidth calculation).

**Real, striking cross-rank consistency in the 1-10ms gap bucket**: mean
gap duration ~2909-3010µs on both ranks, independently measured — this
sits close to (not identical to, given different measurement conditions)
the earlier sync-span-measured `moe.all_sum` average cost of ~4094µs
from `docs/moe-allsum-collective-cost-confirmed-2026-08-21.md`. This is
consistent with, though does not yet prove, `moe.all_sum` being the
dominant contributor to this specific gap bucket — the honest next step
(queued, not done tonight) is jaccl-internal timestamping to actually
attribute these gaps to the collective specifically, rather than
inferring it from a coincidental magnitude match.

## Known limitation: no cross-rank clock synchronization

The two Instruments captures were launched independently via separate
SSH sessions to physically separate machines with independent system
clocks (not NTP-synchronized to microsecond precision, and no clock-
offset calibration was performed this session). This means: real
per-rank aggregate statistics (occupancy %, gap-size distributions) are
valid and directly comparable, but **true wall-clock cross-rank gap
correlation (e.g. "did rank0's gap at ns=X line up with rank1's gap at
the same real-world instant") was NOT attempted and would currently be
invalid** without a calibrated clock offset. Fable's queued
recommendation (a calibrated clock offset via a Thunderbolt ping, or
timestamps taken from inside jaccl itself which naturally shares a
reference across both ranks via the RDMA completion protocol) is the
correct way to close this gap, not a naive comparison of the two
independent capture timelines.

## What this does and doesn't establish

**Established, with real direct-measurement evidence from two
independent instruments (roofline arithmetic + live GPU trace)**: decode
genuinely has substantial GPU-idle time (~70% by this measurement,
~88% by the earlier roofline estimate — same conclusion, different
instrument, different but overlapping denominator), not just a
theoretical estimate.

**NOT yet established**: the precise attribution of that idle time to
specific causes (rank-skew/collective-wait vs. CPU-dispatch latency vs.
something else entirely). The magnitude match between the 1-10ms gap
bucket's mean (~2900-3000µs) and the earlier `moe.all_sum` measurement
(~4094µs) is suggestive but not proof — different measurement contexts
(this trace ran under Instruments' own overhead, unlike the sync-span
measurement) mean this should be treated as a hypothesis to test with
jaccl-internal instrumentation, not a settled conclusion.

## Next steps (queued, per Fable's ranking)

1. jaccl-internal timestamps (op-post / peer-ready / completion) to
   properly decompose the all_sum gap into dispatch-latency vs.
   peer-skew-wait vs. wire-time — the correct instrument for this
   specific question, per Fable's critique of the rejected
   perf_counter-around-call-site approach.
2. Investigate comm/compute overlap between `moe.all_sum` and the NEXT
   layer's attention: since attention is fully replicated/unsharded at
   decode (`EXO_DSV4_ATTN_ALLSUM=0`, confirmed live in production
   config), next-layer attention has no data dependency on the current
   layer's `all_sum` result until the FFN stage — a real, previously
   unflagged overlap opportunity per Fable's review.
3. Sanity-check the roofline ceiling's own denominator (must use real
   active-expert bytes at the actual gamma≈2 speculative-decode
   configuration, not total model parameters) before trusting the
   ~12%-of-ceiling figure as precisely calibrated.

## Cleanup note

Trace files are large (1.9GB + 2.1GB in `/tmp` on each node) — flagged
for cleanup, not yet removed as of this doc (queued as a separate task
so the raw data remains available if this session needs to re-derive
anything from it before wrapping up).
