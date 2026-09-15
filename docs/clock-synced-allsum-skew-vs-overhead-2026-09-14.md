# Clock-synced two-rank measurement of the all_sum sync-span-vs-transport gap: INCONCLUSIVE — instrumentation precision ceiling reached before a determination, root cause and correct next step identified (2026-09-14)

## Question

`docs/PERFORMANCE_HISTORY.md` §13 flags, as one of three genuinely open
threads: does the ~34x gap between `moe.all_sum`'s real jaccl-internal
transport time (median 36µs, mean 58-66µs,
`docs/jaccl-internal-timing-allsum-transport-fast-2026-08-21.md`) and the
much larger command-level sync-span average (~4094µs,
`docs/moe-allsum-collective-cost-confirmed-2026-08-21.md`) represent
**genuine cross-rank arrival skew** (rank0 idle because it is waiting on
rank1, or vice versa) or **shared local overhead** (MLX eval-fence /
CPU-GPU dispatch / Python-level scheduling cost that independently
inflates both ranks' apparent stall, with no real cross-rank wait
involved)? Every prior measurement of this gap used two independent,
non-synchronized system clocks on the two physical machines, which
cannot distinguish these two hypotheses — a genuine wait requires a
clock-synced (or shared-anchor) two-rank capture, which had never been
attempted before this session.

## Method attempted, and why it did not reach the target precision

**Step 1 — clock offset via direct RDMA-interconnect UDP exchange
(this part worked well).** A Cristian's-algorithm-style 4-timestamp UDP
round-trip exchange, run directly over the dedicated Thunderbolt/RDMA
interconnect subnet (192.168.201.1 ↔ 192.168.201.2 — the same physical
link `moe.all_sum` itself uses, not the management LAN) rather than over
SSH/LAN, per a design review's specific correction. Results across
several independent 300-500-sample runs: **min RTT 61-151µs, best-decile
offset stdev 1.5-2.4µs, error bound (min_RTT/2) ≈ 30-75µs.** A
repeated-measurement check also found a real, small, consistent clock
drift of ~1.0-1.5µs/s between the two machines' crystal oscillators
(expected and negligible over a capture window of seconds). **This
sub-100µs clock-sync precision is genuinely more than sufficient** to
resolve a 1-4ms-scale phenomenon, if it could be applied to the actual
measurement instrument.

**Step 2 — attempted to anchor `xctrace`/Instruments Metal-trace
captures to that clock offset (this is where the plan failed).** The
plan was to launch simultaneous `xctrace record --template 'Metal System
Trace' --attach <pid>` captures on both ranks' real production runner
PIDs (read-only, no relaunch, reusing the exact technique already
proven safe in `docs/live-decode-two-rank-instruments-trace-2026-08-21.md`),
timed via the offset-corrected UDP clock sync so both captures'
process-launch instants would land within the ~30-75µs error bound of
each other, then use the captured Metal GPU busy/idle intervals to check
whether rank0's idle gaps in the previously-flagged 1-10ms bucket
coincide with rank1 being busy (→ genuine wait) or also idle (→ shared
overhead).

**This failed for a reason specific to the instrument, not the clock
sync**: measured directly, `xctrace`'s own internal
launch-to-actual-recording-start latency varied by **~300ms** between
two invocations fired within ~75µs of each other (1.40s on rank0 vs
1.71s on rank1, both measured via each host's own local
`subprocess.Popen()`-to-stdout-line timestamps). This single-instrument
launch jitter is **100-1000x larger** than the 1-4ms phenomenon under
investigation, and dwarfs the ~30-75µs clock-sync precision achieved in
step 1 by roughly the same factor. Confirmed via an independent method
(see below) that this is a real, load-bearing precision ceiling, not a
one-off fluke.

**Step 3 — attempted an anchor-free fallback (cross-correlating the two
ranks' GPU busy/idle envelope signals), which surfaced a real, generally
useful methodology lesson.** Both ranks' `metal-gpu-intervals` exports
were parsed (a genuinely nontrivial XML format: values are
deduplicated via document-wide `id`/`ref` back-references, sometimes
defined only inside a nested sub-element of an EARLIER row, requiring a
global cross-row resolution table, not a per-row one — see
`extract_intervals4.py` logic in the session transcript for the working
parser) and merged into non-overlapping busy-fraction time series. A
full-signal FFT cross-correlation of the two ~10.5s busy/idle envelopes
found a very sharp, high-confidence-LOOKING peak (230x the noise floor).
**This peak was WRONG** — a periodicity-aliasing artifact of the
quasi-periodic per-token/per-layer decode loop, not the true cross-trace
offset. Caught by a sub-window stability check (splitting the signal
into five independent 2-second windows and finding each window's
best-fit lag separately): the five windows disagreed with each other by
tens of milliseconds, which a true fixed clock offset cannot do. A
second attempt using a genuinely APERIODIC landmark (the one-time,
monotonic step from idle-background chatter to sustained full-decode
load at the start of the real decode workload fired during the capture,
located via a 300ms-smoothed half-max-crossing on each rank's own
timeline independently) gave **~470ms**, consistent in order of
magnitude with the step-2 launch-jitter estimate — confirming the ~300ms
xctrace-launch-jitter ceiling is the real, load-bearing limitation, not
an artifact of one particular anchoring attempt.

**Conclusion of the measurement attempt: INCONCLUSIVE, precision-limited,
not a false-confidence answer either way.** The tools available without
root access on this hardware (`xctrace`/Instruments works read-only;
`dtrace` and `powermetrics` both refuse without sudo, which this session
did not have interactively; `py-spy` is not installed; `sample` produces
an aggregated call-tree, not an easily cross-machine-correlatable
per-instant timestamp stream) cannot resolve cross-machine timing below
xctrace's own ~300ms launch-jitter floor. Since that floor is 100-1000x
coarser than the 1-4ms effect under study, **no confident skew-vs-shared-overhead
determination can be drawn from this attempt** — reporting a verdict
either way from data this imprecise would be false-confidence, which
this write-up deliberately declines to do.

**Important scoping note, since it changes what "inconclusive" means
here**: this precision ceiling is a property of xctrace/Instruments as
the measurement instrument, not of the RDMA network or the clock-sync
methodology. The direct-RDMA-interconnect UDP clock sync itself achieved
~30-75µs precision — genuinely sufficient for this question. The
blocker is entirely that no currently-available *root-free* instrument
was found that can both (a) capture cross-rank-comparable timestamps at
that precision and (b) attach to the live production runner processes
read-only. This is a narrower, more specific, and more actionable
finding than "clock sync is impossible on this hardware."

## The correct next step (not executed this session — would require a production relaunch, out of scope for a read-only diagnostic per this session's brief)

jaccl's C++ transport layer already has a working, proven
`steady_clock`-based per-call timer
(`JACCL_TRACE_CALLS`/`JACCL_TRACE_TIMING`, used successfully in
`docs/jaccl-internal-timing-allsum-transport-fast-2026-08-21.md` and
`docs/p01a-allsum-arrival-skew-at-depth-2026-08-29.md`) — but it is
LOCAL/monotonic-only per machine (records a *duration*, `transport_us`,
never an absolute timestamp; confirmed by reading
`mesh.cpp:trace_call`/`trace_duration` directly this session), so its
existing output cannot be correlated cross-machine at all, by
construction, regardless of clock sync. The correct fix — flagged here
for whoever picks this thread up next, since it is the one path this
session did NOT find a precision ceiling on — is to run the SAME
UDP-based direct-RDMA-interconnect clock-sync protocol proven in step 1
above (~30-75µs error bound, no code changes needed, already a working
standalone script) **at the same time as** a `JACCL_TRACE_CALLS=1
JACCL_TRACE_TIMING=1` production relaunch, and apply the measured offset
to bring rank1's `steady_clock`-timestamped per-call trace file into
rank0's clock frame. This directly measures the actual quantity in
question (RDMA collective call timing) rather than a GPU-busy/idle proxy
signal, sidesteps xctrace and its ~300ms launch jitter entirely, and
would reach the needed sub-millisecond precision using only tools
already proven safe and working in this exact codebase. **This requires
a production relaunch** (to set the two trace env vars), which this
session's brief scoped as read-only/measurement-only and did not
authorize — flagged as the concrete, ready-to-execute next step for a
session with that authorization, not attempted here.

## Implication for the parallel switch_mlp/kernel-trace investigation

Per this session's brief, noting the implication without further
investigation: since this session did NOT establish that the sync-span
gap is genuine cross-rank skew, it also did NOT reopen "cross-rank
timing" as a confirmed lever. The gap's true nature remains exactly as
open as it was before this session (not closed toward either hypothesis)
— so the standing attribution split from §13/§2.7 (transport is fast;
the 34x gap is "MLX's `mx.eval` fence, dispatch coordination, or
Python-level scheduling" as an undifferentiated bucket, never further
decomposed) is UNCHANGED by this session. Local-dispatch/scheduling-cost
attribution work (e.g. a `switch_mlp` kernel trace) does not need to
wait on or coordinate with this thread; nothing here confirms or rules
out overlap between the two questions.

## Production impact

Zero. All measurement was read-only (`xctrace --attach` to live
production PIDs, `ping`, a UDP clock-sync exchange over the RDMA subnet,
one ordinary `/bench/chat/completions` decode request identical in kind
to routine load). No relaunch occurred. Both nodes' top-level runner
processes (pid 43738 on m4-1, pid 10072 on m4-2) were confirmed
unchanged (same PID, monotonically increasing uptime) before, during,
and after this session's testing. `/ollama/api/ps` and a real
`/v1/chat/completions` smoke-test completion were confirmed healthy
before and after. All temporary trace files (~250MB/node) and scripts
were cleaned up from `/tmp` on both nodes at the end of the session.

## Files referenced (all pre-existing, read-only reads this session)

- `docs/PERFORMANCE_HISTORY.md` §13 (open-threads tracker), §2.7
  (all_sum/collective investigation history)
- `docs/jaccl-internal-timing-allsum-transport-fast-2026-08-21.md`
- `docs/cross-rank-allsum-skew-2026-08-22.md` (T4 CLOSED — bulk skew
  symmetric, small tail asymmetry)
- `docs/p01a-allsum-arrival-skew-at-depth-2026-08-29.md` (Phase 1(a)
  CLOSED — in-collective arrival skew grows only +0.079ms/tok, ruled out
  as the depth residual's owner)
- `docs/live-decode-two-rank-instruments-trace-2026-08-21.md` (prior
  session's independent-clock two-rank trace, explicitly flagged its own
  "known limitation: no cross-rank clock synchronization" — the gap this
  session set out to close)
- `docs/offline-collective-microbenchmark-2026-08-21.md`
- `mlx/mlx/distributed/jaccl/lib/jaccl/mesh.cpp`,
  `mesh.h`, `mesh_impl.h` (jaccl transport + tracing implementation,
  read to confirm the existing trace format's exact fields)

---

## FOLLOW-UP (2026-09-14→15): the correct next step above was executed — RESOLVED via three converging tests. The 34x gap is shared local overhead, NOT genuine cross-rank wait (in any of the forms tested).

This follow-up executed exactly the concrete next step the prior session
identified but did not have authorization to run: a production relaunch
with `JACCL_TRACE_CALLS=1 JACCL_TRACE_TIMING=1`, combined with the same
proven UDP clock-sync protocol, applied to jaccl's own per-call trace
files instead of xctrace. This sidesteps xctrace's ~300ms launch-jitter
ceiling entirely (no xctrace involvement at all) and reaches a
confident, evidence-based determination.

### Step 0 — rebuilt and re-validated the UDP clock-sync protocol

The prior session's clock-sync scripts were disposable `/tmp` scratch
and had been cleaned up, so this session rebuilt them from the doc's
"Method attempted" protocol description (4-timestamp Cristian's
algorithm, plain UDP sockets, run directly over the RDMA-interconnect
subnet 192.168.201.1↔192.168.201.2, not the management LAN). Two
independent 500-sample validation runs before relying on it for
anything real:

- Run 1: min RTT 91.1µs, error bound 45.5µs, best-decile offset stdev 1.26µs
- Run 2: min RTT 94.2µs, error bound 47.1µs, best-decile offset stdev 1.98µs

Both runs land inside the prior session's ~30-75µs error-bound range —
confirms the protocol reproduces, it isn't a one-off. The same exchange
run live during the actual capture (see below) measured offset
-926.7µs (m4-1 relative to m4-2) with error bound 37.9µs — the tightest
of the three runs.

### Step 1 — confirmed the trace-file format gap the prior session flagged, and closed it correctly

Reading `mesh.cpp` directly (this session, independently) confirmed the
prior session's finding: `JACCL_TRACE_CALLS`/`JACCL_TRACE_TIMING` trace
lines contain `call_id`, `op`, `msg_bytes`, and `transport_us` (a
*duration*) — genuinely no absolute timestamp field exists to read
out of the file. Bridging cross-machine therefore requires stamping
each line with a LOCAL wall-clock read at the moment it becomes
available, then applying the UDP-measured offset — not just applying
the offset to something already in the file. A second opinion (external
consult) flagged the load-bearing risk in doing this: a naive polling
watcher process competing for CPU with the live GPU-bound production
runner could itself introduce an unvalidated jitter ceiling, structurally
the same failure class that sank the xctrace attempt (an instrument
whose own overhead exceeds the phenomenon under study). Two changes
were made in response, both followed through on, not just noted:

1. Used macOS `kqueue`/`EVFILT_VNODE`/`NOTE_WRITE` (a kernel write-event
   wakeup) instead of a polling loop, to minimize watcher-introduced
   scheduling jitter.
2. Validated the watcher's own precision **under real production GPU
   load**, not on an idle system — a synthetic writer emitting
   known-timestamped lines at production-like ~500/s ran concurrently
   with a real `/v1/chat/completions` decode request (12.6s, live GPU
   traffic on the exact node being measured). Result: 3000/3000 lines
   captured, 98.0% of watcher-observed arrivals within 50µs of true
   write time, p99 = 67.9µs, only one ~3ms outlier (a one-time kqueue
   cold-start cost on the very first event, not a per-event ceiling).

Combined system precision (clock-sync error bound + watcher jitter,
both empirically measured, not assumed): tens of µs typical, worst-case
~100-200µs — 10-40x finer than the 1-4ms phenomenon under study, and
crucially validated under the SAME kind of load conditions as the real
capture, closing the exact gap that sank the xctrace attempt.

### Step 2 — one production relaunch, trace env vars set, data captured

Relaunched production once via `JACCL_TRACE_CALLS=1 JACCL_TRACE_TIMING=1
./start_cluster.sh` (env-var passthrough already existed in
`start_cluster.sh`, no script changes needed). Verified via `ps eww` on
the fresh PIDs (m4-1 pid 12331→process group with jaccl worker pid
12558; m4-2 similarly) that both trace env vars landed correctly, and
via `/ollama/api/ps` + a live completion that the model reloaded
healthy. Started a kqueue capture watcher on each node targeting the
freshly-created `/tmp/jaccl_trace_rank_{0,1}_color0_pid*.log`, ran the
UDP clock-sync exchange, then fired one real
`/v1/chat/completions` decode request (400 tokens, 15.0s) — all inside
the same ~90s watcher capture window, exactly per the doc's "at the same
time as" plan. Both watchers captured 11,697 lines each (11,696 real
collective calls, 1 header line) — the two per-rank call sequences are
in exact lockstep (`call_id` 1-11696 on both ranks, **zero `msg_bytes`
mismatches across all 11,696 common call_ids** — confirms both ranks
executed the identical, correctly-ordered collective-call sequence, a
necessary precondition for the cross-rank correlation below to mean
anything).

### Step 3 — analysis: applied the offset, tested the actual question

For each call, computed `entry_ts = observed_ts(watcher) - transport_us`
(recovering the call's start instant from its known duration and the
watcher's stamped completion instant), converted rank1's timestamps into
rank0's clock frame via the measured offset, then computed, per rank,
`gap[N] = entry_ts[N] - finish_ts[N-1]` (this rank's idle time between
consecutive collective calls) and the cross-rank `entry_skew[N] =
entry_ts0[N] - entry_ts1[N]` (how far apart in real wall-clock time the
two ranks actually started the SAME call).

**Sanity check on the offset-application itself before trusting any
result from it** (the same discipline that caught the xctrace jitter
problem was applied here too): computed entry-skew three ways —
(a) no clock sync at all (naive, what every prior attempt was stuck
with): median skew 925.8µs, matching the raw ~927µs inter-machine clock
disagreement — this is *why* no prior unsynced attempt could resolve
the question; (b) offset applied with the WRONG sign on purpose: median
skew 1852.5µs ≈ 2× the true offset, exactly as expected if sign matters;
(c) offset applied correctly: median skew **-1.0µs**. The three-way
comparison is a load-bearing internal check, not just a single number
taken on faith — a correctly-applied offset collapsing ~927µs of raw
clock disagreement down to ~1µs, while the wrong sign roughly *doubles*
it, is strong evidence the correction is doing genuine work, not
producing a coincidentally-near-zero number regardless of input.

**Phase segmentation (correction made during analysis, not before
reporting)**: the raw 9,320-call "real" region includes 44 calls at the
very start (call_id 2376-2505, `msg_bytes` 196608/204800) that are
prefill-chunk `all_sum`s, not per-token decode `all_sum`s — a much
bigger payload from prompt processing, a different phenomenon than the
34x decode-gap question this investigation is about. Left in, they drag
the skew stdev up (146.6µs) and are 41 of the 50 largest gap0 events
found when eyeballing outliers, which would have wrongly suggested a
much noisier/less clean result than the decode phenomenon actually is.
Segmented by `msg_bytes` (204800/196608 = prefill; everything else =
decode) and reran on the phase-clean **9,277 decode-only calls**; all
numbers below are this phase-clean set.

**The core result, over 9,277 phase-clean steady-state decode `all_sum`
calls:**

- Per-call cross-rank entry-timestamp skew: median **-1.0µs**, mean
  -1.9µs, stdev 128.9µs, **p1..p99 = [-120.9, +106.1]µs** — the full
  middle 98% of calls sits within a band consistent with the
  clock-sync's own ~38-47µs error bound plus watcher jitter. Both ranks
  start each collective call at essentially the same real wall-clock
  instant, essentially always.
- Per-call idle-gap correlation: **Pearson r = 0.9975** between rank0's
  gap and rank1's gap at the SAME call_id.
- **Lag test (added after a second-opinion review raised a real
  alternative-hypothesis concern: could rank0's call N actually be
  waiting on rank1's call N-1 or N+1, e.g. from pipelining/prefetch
  offset, rather than N itself — which same-call correlation alone
  can't distinguish from "no real dependency at all, both idle for
  unrelated local reasons"?)**: computed the SAME Pearson correlation
  at lag ±1 and ±2 calls. Result: **r = 0.9975 at lag 0, r ≈ 0.001 at
  lag +1, r ≈ 0.001 at lag -1** (full range tested: lag 0 alone stands
  out sharply; ±1 and ±2 are statistically indistinguishable from zero).
  This is a clean, sharp distinguishing result, not a marginal one — it
  rules out a lagged/pipelined dependency as the mechanism, and confirms
  the correlation is specifically a same-instant phenomenon.
- **Second alternative-hypothesis test (same review): could the
  near-zero entry-skew be explained by a SYMMETRIC mutual wait baked
  into `all_sum`'s own completion/ACK semantics** (both ranks
  genuinely blocking on each other inside the collective, which would
  also produce near-zero entry-skew and high gap-correlation, and so
  isn't distinguished by those two tests alone)**, rather than shared
  overhead OUTSIDE the collective entirely?** Tested directly using the
  trace's own `transport_us` field, which measures wall time strictly
  INSIDE jaccl's transport call (dispatch + wait-for-peer + wire, per
  the source comment in `mesh.cpp`) — if the multi-ms gaps were real
  waiting absorbed into the collective's own rendezvous protocol, the
  calls immediately following a large gap should show elevated
  `transport_us`. They do not: of the 4,985 calls whose PRECEDING idle
  gap fell in the flagged 1-10ms bucket, their OWN `transport_us` has
  median 39.1µs, p90 64.2µs — only **0.10% (5/4,985)** exceed 500µs.
  `Pearson r(gap0[N], transport_us[N]) = 0.21`, i.e. weak. **The
  multi-millisecond delay is not happening inside jaccl's collective
  call at all — jaccl's own internal transport/ACK timing stays at its
  normal ~36-40µs baseline even immediately after a multi-ms gap.**
  This rules out "wait absorbed into the collective's own barrier/ACK
  semantics" as the mechanism, leaving "cost outside the collective
  call, before jaccl.all_sum is even invoked, and shared symmetrically
  by both ranks" as the only hypothesis consistent with all three tests
  (same-call-only correlation, near-zero skew, and normal transport_us
  even during flagged gaps).
- **The direct test of the originally-flagged question**: of the 4,985
  calls (53.7% of the decode-only sample) where rank0's idle gap fell in
  the previously-flagged 1-10ms bucket: **97.5% had rank1 ALSO idle in
  that same exact 1-10ms bucket at the same call; the remaining 2.5% had
  rank1 idle just outside the bucket's arbitrary edge** (gap1 in
  [100µs,1ms) or ≥10ms — inspected individually, e.g. 616-980µs, clearly
  idle, not busy, just below the round-number 1ms cutoff) — **0% had
  rank1 genuinely busy (gap<100µs) at any threshold checked.** (An
  earlier pass restricted to the small n=215 exact-8192-byte subset,
  before the phase-clean segmentation above, found a noisier-looking
  42.9%-also-idle figure at n=7 in the flagged sub-bucket; inspecting
  those 7 individually showed this was a small-sample boundary artifact
  of the same 1ms cutoff, not a genuine third state — corrected here
  rather than left in an earlier, weaker form.)

**Determination: the ~34x sync-span-vs-transport gap is SHARED LOCAL
OVERHEAD, not genuine cross-rank arrival skew.** Three independent
tests, each targeting a different alternative explanation, converge on
the same answer:

1. **Rules out one-sided arrival skew** (rank0 blocked because rank1
   hasn't arrived): a genuine one-sided wait would show rank0's idle
   gap correlating with rank1 being BUSY at that moment (rank1's gap
   near zero while rank0's is large) — checked directly, found in 0% of
   4,985 flagged-bucket calls.
2. **Rules out a lagged/pipelined dependency** (rank0's call N actually
   waiting on rank1's call N±1, which same-call correlation alone
   can't distinguish from coincidence): the gap-correlation lag test
   shows r=0.9975 at lag 0 collapsing to r≈0.001 at lag ±1 — a sharp,
   same-instant-only signature inconsistent with any offset dependency.
3. **Rules out a symmetric mutual wait absorbed into `all_sum`'s own
   barrier/ACK semantics** (both ranks genuinely blocking on each other
   INSIDE the collective, which would also produce near-zero entry-skew
   and high correlation, so isn't distinguished by tests 1-2 alone):
   the calls immediately following a flagged multi-ms gap have their
   OWN jaccl-internal `transport_us` at a completely normal ~39µs
   median (only 0.10% exceed 500µs) — if the wait were happening inside
   jaccl's transport/ACK protocol, `transport_us` itself would be
   elevated on exactly these calls, and it is not.

With all three structurally distinct alternative mechanisms ruled out
by direct tests (not just an absence of evidence for them), what
remains consistent with every measurement is: whatever produces the
multi-millisecond idle gaps happens BEFORE `jaccl.all_sum` is even
invoked, on BOTH ranks, by nearly identical amounts, with no real
cross-rank dependency of any of the three forms tested. This resolves
the question §13/§2.7 had open since 2026-08-21/22: the standing "MLX
eval-fence / dispatch coordination / Python-level scheduling"
attribution for the 34x gap is now the CONFIRMED explanation, not
merely "not yet disconfirmed." This does not further decompose WHICH of
eval-fence vs. dispatch vs. scheduling dominates — that remains a real,
separate open question (see PERFORMANCE_HISTORY.md's still-open
`switch_mlp` kernel-trace item) — but "is it cross-rank skew, in any of
the forms a careful review could name" is now closed.

**Confidence level, stated plainly, per a second-opinion review's
correction during this session**: an earlier draft of this section
used "DECISIVE, CONFIRMED" based only on tests 1's result (near-zero
skew + high correlation). A second-opinion consult correctly pointed
out that signature alone doesn't distinguish shared overhead from a
symmetric mutual-wait mechanism intrinsic to the collective, and that a
lagged dependency wasn't yet tested either — both genuinely could have
produced the same near-zero-skew/high-correlation numbers. Tests 2 and
3 above were added specifically to close those two gaps, using data
already in hand (no new capture needed). With all three tests
converged, the finding is reported as **decisive** on its merits, not
as a reflexive strong claim — but the record here also shows why:
this determination survived a real attempt to break it, not just a
first look that happened to come back clean.

### Production impact

One relaunch to enable tracing (`JACCL_TRACE_CALLS=1
JACCL_TRACE_TIMING=1`), one relaunch to restore baseline (trace vars
unset — per `start_cluster.sh`'s own comment, these are diagnostic-only
and "don't leave on permanently, fflush per call slows decode").
Confirmed before/after each relaunch: same-commit nodes (29dd274f7),
`/ollama/api/ps` shows the model loaded, a real `/v1/chat/completions`
completion succeeds, fresh PIDs with the expected env vars present
(trace-on relaunch) or correctly absent (restore relaunch). All other
standing production env vars (`EXO_DSV4_DSPARK_TP_SHARD`,
`EXO_DSV4_DSPARK_NATIVE`, `EXO_MLX_CLEAR_CACHE_INTERVAL`, etc.)
confirmed unchanged across both relaunches. Trace files (~1MB/node this
session — far smaller than the prior session's xctrace captures) and
all `/tmp` scratch scripts cleaned up from both nodes after data
collection. Both nodes' runner processes healthy at session end
(fresh post-restore PIDs, real completion verified).

### Files added this follow-up

- This section (appended to the existing 2026-09-14 doc, per this
  repo's established convention of appending dated follow-ups rather
  than rewriting prior findings).
- Raw capture data and analysis summary retained at
  `/home/hermes/clocksync-allsum-20260915-results/` on the hermes-gw-01
  gateway box (not committed to the repo — raw JSONL captures, ~1MB each
  rank, plus `ANALYSIS_SUMMARY.json` with the full stats reported above,
  including the phase-clean decode-only re-analysis) for anyone who
  wants to re-derive or extend the analysis.
