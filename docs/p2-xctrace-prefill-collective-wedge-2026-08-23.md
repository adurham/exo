# P2: `xctrace` Metal System Trace capture against a live TP=2 prefill wedges the collective — reproduced 3/3, NOT root-caused, tracing during deep prefill is now a HAZARD — 2026-08-23

> **OPERATIONAL WARNING (read this even if you read nothing else).**
> Do **not** attach `xcrun xctrace record --template "Metal System Trace"`
> to a live DSv4-Flash TP=2 runner during a long/deep prefill. It has
> wedged the cross-rank collective and cost both runners **3 times out of
> 3 attempts**, including once with the "safe" simultaneous dual-rank
> design. This is a **separate and additional** failure mode from the
> watchdog-vs-SIGSTOP false-positive fixed in `fc954293` — that fix is
> good and is not implicated here. Decode-window and idle-window captures
> remain fine (many prior successes, §2.7/T2). Until root-caused, profile
> prefill via idle-window or synthetic-load captures, or via
> `mx.metal.start_capture()` — not by attaching to a live production
> prefill.

## Why this document

P2's goal was a real per-rank GPU-idle measurement during a ~300K-token
prefill, via `xctrace` attach on both TP ranks. The measurement partly
succeeded (numbers below), but the attempt surfaced a more important
finding: **the capture itself destabilizes the prefill collective.** That
hazard, not the idle number, is the substantive result and is what this
doc is primarily about.

Prior context: `fc954293` ("tracing-aware runner hang watchdog") fixed a
*different* problem — `xctrace --attach` SIGSTOPs/ptraces the target, the
runner stops emitting events, and the 45s hang watchdog SIGKILLed it
mid-capture. That fix defers the kill while the runner is in state `T`.
The failures below are **not** that failure.

## The three incidents

All three during a chunked ~300K-token prefill (~2048 tok/chunk, ~380
tok/s steady, per-chunk progress logged). Commit `fc954293` deployed on
both nodes throughout.

| # | time | capture design | client-abort confound | capture duration | depth reached | outcome |
|---|---|---|---|---|---|---|
| 3 | 03:14 | **simultaneous dual-rank** (both ranks, attached within 0.30s) | **none** | 120.91s | 77,824 / 300,014 (25.9%) | both runners SIGKILLed |
| 1 | 10:55 | rank0-only | yes (an 834K over-long prefill was client-aborted 132s earlier) | 15.65s | 61,440 / 834,898 | both runners SIGKILLed |
| 2 | 11:09 | rank0-only | **none** (deliberately designed to remove it) | 15.86s | 69,632 / 299,591 (23.2%) | both runners SIGKILLed |

### Incident 3 — the decisive one (dual-rank, no confound)

```
03:14:42.359  dual capture START, both ranks within 0.30s
              ... prefill HEALTHY at 378-385 tok/s for ~93s DURING capture ...
03:16:15.864  last normal chunk 73728  (-27.4s vs capture end)
03:16:25.724  chunk 75776 after a 9.9s gap   <- first slip, still DURING capture
03:16:43.273  capture ENDS (rank1 +0.30s)
03:16:49.780  rank1 (m4-2) "[Event::wait] slow wait: signaled=0 target=1"
03:16:51.988  chunk 77824 after a 26.3s gap
03:17:17.693  rank0 (m4-1) enters the same stalled wait
03:17:46.737  rank0 SIGKILL "no event for 46s"
03:17:49.029  rank1 SIGKILL "no event for 51s"
```

This incident alone kills three otherwise-plausible explanations:
- **Not attach-skew.** Both ranks were attached within 0.30s of each other.
- **Not SIGSTOP event-suppression.** The runner kept emitting per-chunk
  progress at full speed for ~93s *during* the capture.
- **Not the client-abort confound.** There was no abort anywhere near it.

### Incidents 1 and 2 — the timing arithmetic

```
                          capture end     rank1 stalls     rank0 stalls     SIGKILL
incident 1 (rank0-only)   10:55:45.125    +2.07s           +2.61s           "no event for 49s"
incident 2 (rank0-only)   11:09:10.891    +2.98s           +3.43s           "no event for 50s"
```

The captures were 15.65s and 15.86s. The watchdog threshold is 45s.
**SIGSTOP suppression during a 15.9s capture can produce at most ~15.9s
of silence and can never reach 45s.** The silence that actually triggered
each kill was 49–50s and *began 2–3s AFTER the tracer had already
detached*. Corroborating this from the other direction: the count of the
new `"deferring hang kill"` log line across all incidents is **0** — the
runner was **not** in state `T` when the watchdog sampled it, so
`fc954293`'s defer branch never even engaged. Different precondition,
different failure.

### The cross-rank signature

`[Event::wait] slow wait: elapsed=3.0s signaled=0 target=1` is a
cross-rank event wait that never got signaled. In incidents 1 and 2
**rank1 was never attached at all** — yet rank1 entered the stalled wait
*first* in all three incidents (by 0.55s, 0.46s, and 27.9s). Both nodes
are NTP-synced (checked against `time.apple.com`: +0.011s ±0.022s on both,
so these deltas are real and not clock skew).

**Correct interpretation (stated carefully):** this shows *propagation*,
not that the tracer reaches across machines. The most economical reading
is that the perturbation is **local to the traced rank**, which then fails
to deliver a signal the untraced peer is blocked on — the untraced rank
merely notices first because it is the one sitting in the wait. Rank1
being "first" is not evidence that rank1 is affected.

## Control: an untraced prefill at the same depth is fine

Run fresh this session at the current commit, same script, same 300K
target, **no tracing at any point**:

| metric | value |
|---|---|
| depth reached | **135,168 / 299,591 and still climbing** at time of writing |
| throughput | 366.7 tok/s, steady |
| chunk cadence | ~5.5s, no outliers |
| `slow wait` / `hung:` events | **0** |

This clears **1.7x past the deepest traced failure** (77,824) and passes
straight through the 59–78K band where all three traced runs died. A
second, independent control exists in the archived logs: on 2026-08-22
20:32:14→21:07:05 an untraced 300K prefill ran to **300014/300014 =
100% completion**, 366 progress lines, median inter-chunk gap 6.0s, max
19.0s. (That control ran on `406074fa`, the parent of `fc954293`;
`fc954293` touches only `thunderbolt.py`, `info_gatherer.py` and
`supervisor.py` — nothing on the prefill or collective path — so the two
are comparable for this purpose.)

Together these make "deep prefill just wedges on its own around 60–78K"
a poor explanation for the three traced failures.

## The measurement that did succeed

Union-of-intervals occupancy on the Metal compute channel (`0x123459`),
own-process rows, computed by merging overlapping GPU execution
intervals (never a naive sum, which would exceed wall time on concurrent
command buffers). Analyzer: `/tmp/p2_gpu.py`.

| trace | rank | window | GPU busy | GPU idle |
|---|---|---|---|---|
| incident-2 `p2_r0_w1` | rank0 | 13.45s | **87.40%** | 12.60% |
| incident-1 `p2_val` | rank0 | 13.24s | **88.32%** | 11.68% |
| incident-3 `p2v2_rank0` | rank0 | 118.50s | **76.12%** | 23.88% |
| incident-3 `p2v2_rank1` | rank1 | 118.43s | **76.63%** | 23.37% |

Two independent short-window rank0 measurements agree closely (87.4% /
88.3%). The dual-rank 120s windows are lower (76.1% / 76.6%) and are
**contaminated** — they include the degradation and wedge onset described
above, so they are a floor, not a clean steady-state figure.

The clean reading is therefore: **prefill GPU occupancy is ~87–88% on
rank0**, i.e. roughly 12% genuine idle. This is consistent with the
post-async-fence-fix decode figures (78.6–78.9%, T2) and firmly refutes
any remaining "prefill is mostly idle" framing. Both ranks are near-
symmetric (76.12% vs 76.63% over the matched 120s window — a 0.5-point
difference), so there is **no meaningful cross-rank occupancy asymmetry**
in prefill.

## What this does and does not establish

**Established:**
- Attaching a Metal System Trace to a live TP=2 prefill collective
  preceded a cross-rank stall in **3 of 3 attempts**, across two
  different capture designs and two different capture durations.
- The stall is **not** explained by the SIGSTOP/watchdog interaction that
  `fc954293` fixed (arithmetic above; defer branch never fired).
- Untraced prefill at the same and greater depth is healthy (two controls).
- Prefill rank0 GPU occupancy is ~87–88%, ranks near-symmetric.

**NOT established (do not over-read):**
- **Attach vs detach is unresolved, and the evidence actually leans
  detach.** In incidents 1 and 2 the runner was healthy for the entire
  capture and stalled 2–3s *after* the tracer detached. Incident 3's
  first slip came during the capture, but 93s in and near its end. The
  honest phrasing is "the capture *lifecycle* — most plausibly
  detach/buffer-flush — perturbs Metal event signaling", **not**
  "attaching triggers a deadlock".
- **Permanence is unverified.** The watchdog SIGKILLs at 45–51s, so we
  never observed whether the collective would eventually self-recover.
  Note incident 3's 26.3s-gap chunk *did* eventually complete, which is
  evidence against a hard permanent deadlock. Call it "a stall exceeding
  the 45s watchdog window", not "a deadlock".
- **The mechanism is unknown.** No lost-signal root cause was proven.
- **Depth interaction is untested.** All three wedges were at 60–78K
  tokens; every prior *decode* capture was fine. Early-prefill tracing
  was never attempted, so the hazard is scoped to "tracing during
  long/deep TP prefill" with a possible depth interaction.

**Leading unproven hypothesis, recorded for whoever picks this up:**
unified-memory pressure. Deep prefill holds a large KV cache near the
wired-memory ceiling, and a system-wide Metal trace's buffers are not
small. That would neatly explain both the prefill-specificity and the
decode-safety, and it is checkable from `memory_pressure`/`[MEM]` logs in
the incident windows **without** running another capture.

## Decision

**Documented as an unresolved hazard; no further capture attempted.**

Rationale: the evidence is 3/3 with the main confound (client abort)
already eliminated in incident 2 and absent in incident 3. A 4th capture
that wedges moves belief almost not at all; a 4th that doesn't wedge
muddies attribution without resolving mechanism — and either way it
risks another cluster-down for a marginal data point, against the
standing rule not to destabilize the cluster. The higher-value next
experiments are cheaper and safer: (a) the `memory_pressure` check above,
purely from existing logs; (b) tracing a **single-node** long prefill —
if it survives, that localizes the hazard to the collective path, and the
worst case costs one node rather than the cluster.

## Cluster state

Verified healthy **before** this work (decode probe 30.14 tok/s) and the
untraced control ran clean throughout (135K+ tokens, 366.7 tok/s, zero
stalls). No lingering `xctrace` processes on either node. Trace artifacts
left in `/tmp` on both Studios (`p2_smoke`, `p2_val`, `p2_r0_w1`,
`p2v2_rank0`, `p2v2_rank1`) — ~1.3 GB each for the 120s ones; safe to
delete when disk is needed.

Outcome class: **(c) genuine finding with real evidence, reproduced,
NOT root-caused — recorded as an operational hazard with a concrete
avoid-this rule.**
