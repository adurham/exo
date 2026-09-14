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
