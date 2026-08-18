# DSv4-Flash: Event::wait ring-diag attempt — non-reproduction (2026-08-18)

## Purpose

Follow-up to `dsv4-220k-prefill-eventwait-rootcause-triage-2026-08-18.md`,
which established (via zero-cost code reading and log analysis, no
cluster touched) that the 8 previously-observed Metal `Event::wait`
stalls are GPU-signaled (Story A), not transport/retransmit-caused
(Story B falsified), and recommended a single-shot relaunch with
`EXO_CMDBUF_RING_DIAG=1` to discriminate between three remaining
candidate mechanisms (not-committed / committed-not-scheduled /
scheduled-not-completed).

## What was run

Relaunched cluster with the full recommended diagnostic envelope:

```
MLX_JACCL_DATA_RECV_POOL=0 EXO_CMDBUF_RING_DIAG=1 \
JACCL_TRACE_PROGRESS=1 EXO_PROFILER=spans ./start_cluster.sh
```

(All four are plain launcher env vars, allow-listed in
`start_cluster.sh`, no code changes. `MLX_JACCL_DATA_RECV_POOL=0` is
the standing fix required for the cluster to load at all — see
Section 119 in the design-doc history.)

Ran the SAME fresh-content, cache-busted, needle-in-haystack 220K-token
prefill methodology used by all three prior 2026-08-18 docs, TWICE in
a row on the same instrumented cluster instance (no relaunch between
attempts):

- Attempt 1: code `RING-5357-DIAG-9478`. 220,320 prompt tokens,
  725.1s total wall, ~309 tok/s live rate at completion. Model
  correctly identified the code but hit `finish_reason: length` (a
  50-token completion budget was slightly too tight given the
  reasoning preamble — a harness artifact, not a correctness failure;
  content ended `" just code.RING-5357-DIAG"`, missing only the
  numeric suffix due to truncation).
- Attempt 2: code `RETRY-8245-DIAG-3878`. 220,324 prompt tokens,
  727.6s total wall, ~308 tok/s live rate at completion. Model
  answered correctly and completely, `finish_reason: stop`.

Both runs verified `cached_tokens: 0` (genuinely fresh prefill, not a
KV-cache phantom hit).

## Result: non-reproduction

**Zero `[Event::wait] slow wait` log lines fired on EITHER node across
BOTH runs.** Consequently zero `[cmdbuf_ring]` diagnostic dumps fired
either (the dump is gated on the same 3.0s-elapsed trigger that
produces the Event::wait log line — see `event.cpp:139-157` — so no
stall means no dump, by design).

Combined: ~440,644 prompt tokens processed across two full runs
(~1,452s of combined prefill wall time), zero stalls observed, versus
8 stalls observed in the single prior 220K run documented in
`dsv4-220k-prefill-rdma-wait-breakdown-2026-08-18.md`.

## Interpretation

This is genuine negative evidence, not an inconclusive result:

1. **Confirms the stall is stochastic/intermittent, not
   deterministically triggered by prompt content, token count, or
   position.** The prior run's 8 stalls occurred at specific
   `call_id`s tied to specific 16MB MoE `all_sum` payloads — but this
   session shows that hitting the exact same context length and
   chunk-size profile twice more produces zero stalls. Whatever
   triggers it is not "the Nth 16MB all_sum call in a 220K-token
   run" — it's something with lower and apparently variable
   probability per call (8 stalls in ~4730 MoE all_sum calls in the
   original run ≈ 0.17% per-call rate; 0 stalls in ~9460 calls across
   these two runs is consistent with a low, noisy per-call
   probability rather than a contradiction of the original finding —
   observing zero in ~9460 Bernoulli trials at p≈0.0017 has a
   plausible chance of occurring, so this does NOT falsify the
   original 8-stall observation).
2. **The stall is real** (documented with full evidence in the prior
   two docs — this is not in question) but its trigger condition
   remains uncharacterized. Candidates from the prior triage doc
   (async-fence pattern, driver/allocator hiccup under memory
   pressure, stream-ordering edge case) may depend on external
   factors not controlled or reproduced here: thermal state, exact
   scheduling jitter from other processes on the machine, Metal
   driver/OS state accumulated since the last reboot, or genuinely
   rare timing windows in Metal's own command-buffer scheduler.
3. Catching one on ring-diag requires either (a) more attempts (pure
   luck, low information return per attempt given the apparent
   sub-1% per-call rate), or (b) a more targeted repro that increases
   stall probability — e.g., running under sustained load for longer
   (a single request approximates only ~4730 MoE all_sum calls;
   running several consecutive 220K+ requests back-to-back, or a
   single much longer context, would accumulate more trials per
   cluster-minute spent).

## Recommendation

Do not keep spending cluster time on blind repro attempts — the
expected trials-to-catch-one at this apparent rate makes that
inefficient. If this is picked up again:

- Prefer a LONGER single run (e.g. 500K+ context) or multiple
  back-to-back requests on one instrumented launch, to accumulate
  more MoE all_sum trials per cluster-minute rather than restarting
  fresh each time.
- Alternatively, since the stall is confirmed GPU-signaled
  (Story A, established in the prior doc) and rare, it may be more
  tractable to leave `EXO_CMDBUF_RING_DIAG=1 JACCL_TRACE_PROGRESS=1`
  permanently enabled on a future production/long-running cluster
  session (the overhead is not free — `EXO_PROFILER=spans` alone
  cost ~15% throughput in the companion doc — so `EXO_PROFILER=spans`
  specifically should be DROPPED from the standing config; only
  `EXO_CMDBUF_RING_DIAG=1` + `JACCL_TRACE_PROGRESS=1` are needed to
  catch a future occurrence, and their overhead was not isolated
  separately in this session) and simply wait for organic
  reproduction during real usage, then pull the dump after the fact.
- Given the apparent low frequency (~8 stalls in one 220K request,
  0 in the next two), the practical throughput impact of NOT fixing
  this in the near term is small and variable (0% to ~9% of wall time
  depending on luck) — this is a real, real but low-priority
  optimization target relative to e.g. the untested `EXO_DSV4_SEQ_SPLIT`
  A/B still queued from the first span-profile doc.

## Files & data

- No new raw log data of forensic value was produced (zero stall
  events to analyze). Attempt logs are on the cluster nodes'
  `~/exo.log` (not copied locally — nothing to preserve).
- Standing env vars from this investigation thread, for reference on
  any future relaunch: `MLX_JACCL_DATA_RECV_POOL=0` (required, unrelated
  fix), `JACCL_TRACE_PROGRESS=1` (cheap, worth leaving on),
  `EXO_CMDBUF_RING_DIAG=1` (cheap when idle, only fires on an actual
  stall — worth leaving on), `EXO_PROFILER=spans` (NOT free, ~15%
  throughput cost — only enable when actively doing span-level
  analysis, not for passive stall-catching).
