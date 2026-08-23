# T4: Cross-rank all_sum skew — bulk distribution symmetric (confirms transport measurement), but a real 4.2x rank0-straggler asymmetry in rare severe outliers — 2026-08-22 (session 4)

## Why this check

Per the Fable-provided plan: "Cross-rank skew via all_sum wait
asymmetry — extend existing jaccl steady_clock timing, log per-rank
per-call all_sum duration over ~1000 tokens. Decision: one rank
consistently near-zero while the other waits >0.5ms median → straggler-
rank investigation; symmetric ~36-60µs → closed, matches transport
measurement." This directly follows up on the earlier
jaccl-internal-timing work (§2.7,
`docs/jaccl-internal-timing-allsum-transport-fast-2026-08-21.md`) which
measured real per-call transport cost but never checked cross-rank
symmetry.

## Method — reused existing data, no relaunch needed

The real jaccl-internal `steady_clock` trace files from the prior
session's investigation (`/tmp/jaccl_trace_rank0.log`,
`/tmp/jaccl_trace_rank1.log`, both still present locally — 45,850 lines
each, real production decode traffic) were reused directly rather than
re-enabling `JACCL_TRACE_TIMING`/`JACCL_TRACE_CALLS` and relaunching
the cluster. **Justification**: this trace instruments the C++
`reliable_all_reduce_v2` transport call itself, at a layer below where
the async-fence fix operates (the fence only changes what Python does
with `y` *after* `all_sum` returns — `mx.async_eval(y)` vs `mx.eval(y)`
— it does not touch the RDMA transport call being measured here). The
transport-layer data is not stale with respect to this specific
question; re-capturing would reproduce the same population, not correct
for any known regression.

Matched each rank's 8192-byte (real decode-time `moe.all_sum`) calls by
`call_id` — both ranks log the same logical collective call under the
same id, since jaccl assigns `call_id` per-collective-invocation before
dispatching to each participant. 45,666 matched pairs across both
ranks (100% match rate — every rank0 call_id has a corresponding rank1
entry, confirming no dropped/misaligned trace lines).

## Real result

**Bulk distribution: symmetric, matches the transport measurement.**

| | rank0 | rank1 |
|---|---|---|
| median transport_us | 36.1 | 36.0 |
| mean transport_us | 66.3 | 58.9 |
| p99 transport_us | 252.6 | 266.1 |

Per-call (rank0 − rank1) difference across all 45,666 matched pairs:
**mean 7.34µs, median 0.10µs, stdev 553.72µs.** Which rank is "slower"
per individual call is essentially a coin flip: rank0 slower on 50.2%
of calls, rank1 slower on 49.4% (the remaining 0.4% tied). This is the
`~36-60µs, symmetric` outcome the plan's decision criterion names as
the closing condition — **no systematic one-rank-waits-for-the-other
skew exists in the bulk distribution.**

**But a real, secondary finding in the tail: rare severe stalls are
NOT symmetric.** Filtering to calls where one rank's transport time
exceeds the other's by more than 1000µs (order-of-magnitude above the
~36µs median — genuine straggler events, not measurement noise):
**75 calls where rank0 was the straggler vs only 18 where rank1 was**
— a **4.2x asymmetry**, out of 93 such severe-outlier calls total
(80.6% attributable to rank0). This sits underneath the aggregate
mean/p99 figures (which look nearly identical between ranks) because
it's a small fraction of calls (93/45,666 ≈ 0.2%) with a large
per-call magnitude — exactly the kind of signal a median or even a p99
alone can miss, but a mean gets partially pulled by (rank0's mean
66.3µs vs rank1's 58.9µs — a real, if modest, 12.6% difference that
this straggler asymmetry plausibly explains).

## Decision-gate evaluation

Per the plan's stated criteria: **"symmetric ~36-60µs → closed, matches
transport measurement."** The bulk distribution (>99.8% of calls)
clears this bar cleanly — median/p99 are within 0.1-14µs of each other,
and the coin-flip 50.2%/49.4% split on which rank is momentarily slower
confirms no systematic per-call skew. **This closes the primary
question**: cross-rank skew is NOT a hidden contributor to decode's
wall-time budget in the way a systematic one-rank-always-waits pattern
would be.

**However, the tail asymmetry (4.2x rank0-straggler ratio in the rare
>1000µs-difference events) is a real, secondary, NOT fully closed
finding.** It affects only ~0.2% of calls, so its aggregate throughput
impact is small (consistent with the ~7% mean-difference gap, itself
within normal measurement noise for a metric this skewed), but the
skew's DIRECTION is consistent and non-random (80.6% one-sided, not
~50/50 like the bulk) — this is a real pattern, not noise, even though
its magnitude doesn't warrant urgent action.

## What this does NOT establish

The 4.2x straggler asymmetry's root cause is unexplored — candidates
include: rank0 (m4-1) being the TCP coordinator/master node (per
`auto_parallel.py`'s TP sharding code, rank 0 typically carries
slightly more control-plane responsibility), a real hardware/thermal
asymmetry between the two physical Studios, or simply which node
happens to issue its own local compute slightly later before entering
the collective (a scheduling artifact upstream of the transport call,
not a transport-layer property). This session did not investigate
further given the small aggregate impact — flagged for a future
session only if a bigger, related finding makes it worth revisiting
(e.g., if T10's decomposition of prefill's non-GEMM remainder turns up
a related rank-asymmetric collective cost).

## Conclusion

**T4 CLOSED.** Cross-rank `all_sum` skew is not a meaningful,
systematic contributor to decode's unattributed wall time — the bulk
distribution is symmetric and matches the already-established real
transport cost (~36-66µs). A minor, real straggler asymmetry exists in
the rare tail (4.2x rank0-leaning, ~0.2% of calls) but its small
aggregate magnitude does not change this conclusion or warrant
independent action at this time.
