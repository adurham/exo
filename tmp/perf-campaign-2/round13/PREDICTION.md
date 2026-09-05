# CAMPAIGN 2 / ROUND 13 — PRE-REGISTRATION
Written 2026-09-04, BEFORE any measurement, while cluster access is still DENIED.
Derived mechanically from round 12's outcome; **contains no data-dependent choices.**

## What R13 is

**R13 is round 12's deferred Boot 1. Nothing else.** No new lever, no new hypothesis,
no branch selection. R12's T+75min checkpoint clause fixed R13's content in advance:
*"do NOT attempt the fix arm this round, defer the live boot to R13."*

**There is no pending decision for a supervisor to make.** R13 is blocked on one thing only:
read access to the cluster. The moment that access exists, R13 executes the steps below in
order, with no further authorization needed — every gate and band was pre-registered in
`round12/PREDICTION.md` and is inherited here VERBATIM.

## Lever: I16 (unchanged) — worker/main.py plan_step 100 ms poll tick

Fix already written, committed, pushed: `84bdcd756`. Env-gated `EXO_WORKER_PLAN_EVENT_WAKE`,
**DEFAULT OFF**. Allow-list line already present at `start_cluster.sh:1618`.

**Definition of SHIPPED (binding — a commit on main is NOT a ship):**
> I16 is shipped only when the flag is ON in the production launch path AND Gate A has PASSED
> on real hardware. Until both hold, the 100 ms tick is LIVE IN PRODUCTION and the fix is
> dormant code. **A future session must not flip the default to ON without a Gate-A pass.**

## Execution order (do not reorder; step 0 gates everything after it)

**Step 0 — access + health, as SEPARATE commands.** R12 batched curl+ssh into one command and
the whole batch was denied at once, which destroyed the ability to tell *which* access was
refused. Issue these as three separate read-only commands so a denial is granular:
  1. `curl -s -o /dev/null -w "%{http_code}" http://192.168.86.201:52415/v1/models`  → expect 200
  2. `ssh adam.durham@192.168.86.201 '...'` (read-only: hostname, git SHA, PIDs, load, tunables)
  3. `ssh adam.durham@192.168.86.202 '...'` (same)

**Step 0a — health check on REAL PIDs (charter guardrail, inherited).** API 200; both nodes
READY; `EXO_BATCHED_PREFILL_RENDEZVOUS_MS=0`; `EXO_SPECULATIVE_GAMMA=3`;
`MLX_STEEL_BATCH_INVARIANT=1`; `EXO_PHASE_MARKS` **ABSENT**; `EXO_WORKER_PLAN_EVENT_WAKE`
**ABSENT**. R12 could not perform this check. **The cluster's health is an INHERITED claim from
R11, not a fresh measurement** — treat it as unverified until this step passes.

**Step 0b — post-reboot environment validation** (inherited verbatim from round12/PREDICTION.md;
the control host rebooted since R10 and this was never run): tunables (`iogpu.wired_limit_mb`),
interconnect is TB5 RDMA and not a fallback NIC, `git rev-parse HEAD` identical on both nodes
and matching what was pushed, no background mds/softwareupdate/backup load, discard first 3
requests as cold, baseline prefill/decode TPS within ±3% of 29.1 t/s @2K on the calibrated ruler.
**If ANY check fails: STOP. Do not spend the workload on an invalid boot.**

**Step 0c — local pre-boot item carried from R12 (no cluster needed, do it while waiting).**
A randomized-interleaving lost-wakeup stress test for the flag-ON path: many random
applier/planner orderings, asserting zero `move_on_after` timeout-driven wakes and zero missed
wakes. Gate A requires "ZERO timeout-driven wakes on the request path"; proving that
statistically offline de-risks the boot. R12's unit tests cover the *specific* interleaving
(assertions A–E, 37 passed) but not a randomized sweep.

**Step 1 — Boot 1 (relaunch #1 of 2).** Fix AND marks together, exactly as pre-registered:
```bash
cd ~/repos/exo
EXO_WORKER_PLAN_EVENT_WAKE=1 EXO_PHASE_MARKS=1 ./start_cluster.sh
```
**MANDATORY GATE before spending the boot** (R4/Ask-A lesson — a var that does not reach the
runner PIDs silently zeroes the entire run). Verify **BOTH** vars on the REAL runner PIDs:
```bash
ssh adam.durham@192.168.86.201 'for p in $(pgrep -f "python.*exo"); do ps eww $p | tr " " "\n" | grep -E "EXO_PHASE_MARKS|EXO_WORKER_PLAN_EVENT_WAKE"; done'
```
Expect both `=1`. **If either is absent: STOP, do not run the workload.**

**Step 2 — workload.** Use the study's EXISTING capture path. **Never build a new harness — R5
lost a whole round to one.** Per round11/REPORT.md §3: passive capture proxy, then
`replay_c1.py` (≥20 requests, 90–150K depth, mostly cache hits, several ending in `tool_calls`),
then `analyze_marks.py`. Pin `/usr/bin/python3` — Homebrew python3 lacks httpx (R10 lesson).

**Step 3 — Gates, applied VERBATIM from round12/PREDICTION.md. No renegotiation after seeing
numbers.**
- **Gate A (ships the fix):** intra-worker delta, same clock, state-update-applied →
  plan_step-observed. Baseline fingerprint by construction: ~uniform, median 35–65 ms, p95
  85–110 ms. **Post-fix PASS: median ≤10 ms, p99 ≤20 ms, and ZERO timeout-driven wakes on the
  request path** (count them explicitly — any timeout-driven dispatch means the event-signal
  path is incomplete, i.e. a partial fix; do not ship as complete).
- **Gate B (informational only, does NOT gate the ship):** |unattributed gap| ≤10 ms.
  A Gate-B miss does not block a Gate-A ship; a Gate-A ship does not imply Gate-B closure.
  Report separately.
- **Safety gates (all inherited, all still binding):** byte-identity at 2K / 90K / 150K+,
  temp=0, fix-on vs fix-off; zero worker errors/retries introduced; worker idle CPU ≤ baseline
  +1%; plan_step wake rate bounded under a c=4 burst; the lost-wakeup implementation
  requirements (already satisfied in code and unit-tested).
- **Actionability floor 75 ms**; requests returning no marks must be **0**.
- **Never add `mx.eval()` at a mark**; no cross-node clock arithmetic.

**Step 4 — Boot 2 (relaunch #2 of 2), restore.** **Restore target is the NEW SHA with the flag
OFF** (not the old SHA) — the fix is inert by default, so this leaves production behaviourally
identical while keeping the committed code. Boot 2's baseline-TPS match is therefore the
*empirical* test of the "production-identical" claim; record it as such.
- If Gate A **PASSED**: the ship decision is to flip the launcher default ON. Treat that as a
  separate, explicit change with its own byte-identity evidence — do not smuggle it into the
  restore boot.
- If Gate A **FAILED / inconclusive / any safety gate tripped**: restore with the flag OFF,
  ship nothing, record **I16 CLOSED-NEGATIVE**, and proceed to the branch table. Do not spend a
  third boot chasing it (round12/PREDICTION.md is explicit on this).

**Step 5 — branch table.** Once Boot 1's marks exist, `round12/PREDICTION.md`'s R13 branch table
applies (it now governs **R14**), selected MECHANICALLY: branches H / T / W / U, tie-break =
largest measured median, ties to H; or CLOSE-OUT if no branch triggers and Gate B closes clean.

### Branch T is AMENDED by round 12's seam-harness finding (binding if T is ever taken)
R12 proved, off-cluster, that **template position-invariance FAILS**: `render(msgs[:4])` is not a
byte-prefix of `render(msgs[:5])` (diverges at char 403) because the vendored DSv4 encoder
re-sorts tool results on every `encode_messages()` call. **A prefix cache keyed on message-list
position is therefore provably UNSAFE for multi-tool-result conversations** — 44 of 55 real
requests end in `tool_calls`. Tokenizer-level seam safety (which HOLDS, normalizer inert) is
necessary but **not sufficient**. If Branch T triggers, its design must either carry a
reorder-detection invalidation trigger or be scoped out of multi-tool-call conversations. The
naive position-keyed design is off the table before it is funded.

## What would make R13 invalid
- Any gate or band altered after seeing a number.
- Shipping on Gate A alone without the safety gates.
- Treating the committed-but-OFF fix as "already shipped."
- Building a new measurement harness instead of using the existing capture path.
- Quoting "basedpyright baseline 425" — that figure is WRONG for this tree. Real baseline via
  `git worktree`: 4909 (`src`) / 13155 (repo-wide). The delta is what gates a change, and it is 0.

---

# DATED AMENDMENT — 2026-09-04, written BEFORE any measurement exists

**Nothing above this line is edited. The numeric bands are UNCHANGED.** This amendment records
apparatus corrections found while validating the plan pre-boot. It exists because the plan above
was found to be latently unrunnable as written; recording that honestly is preferable to silently
fixing it or to discovering it after spending a scarce relaunch.

## A1. The apparatus this plan depends on DID NOT EXIST when the plan was written

Step 1 pre-registers Boot 1 as `EXO_WORKER_PLAN_EVENT_WAKE=1 EXO_PHASE_MARKS=1`, assuming
`EXO_PHASE_MARKS` yields Gate A's `state-update-applied -> plan_step-observed` pair. **It does not.**
At the time this plan was written, EXO_PHASE_MARKS instrumentation existed only in:
- `src/exo/api/phase_marks.py` — API process (a1-a7). Different process, different clock;
  **disqualified by Gate A's own "same clock" requirement.**
- `src/exo/worker/engines/mlx/phase_marks.py` — runner subprocess (b1-b11), request lifecycle only.
- `src/exo/worker/main.py` (where `plan_step` lives): **ZERO marks.**

Boot 1 as written would have consumed a relaunch and produced **no Gate-A data whatsoever**.
Apparatus added this round: `src/exo/worker/phase_marks.py`, under the SAME `EXO_PHASE_MARKS` gate
(no new env var), pairing on the pre-existing `IndexedEvent.idx`, no `mx.eval()` near any mark,
gate-OFF path behaviourally identical.

**General lesson for every future pre-registration: include a "the measurement apparatus exists at
HEAD" check. Pre-registering a gate does not conjure the instrument that reads it.**

## A2. Gate A's "ZERO timeout-driven wakes" is SCOPED (it would otherwise fail a CORRECT fix)

`KeyedBackoff.should_proceed` (`src/exo/utils/keyed_backoff.py:20`) is `now - last >= delay` —
**purely clock-driven, with no state precondition.** `plan()` gates `CreateRunner` on it
(`src/exo/worker/plan.py:131`) and the download path likewise (`plan.py:204`). So `plan()` can
legitimately return a non-None task with **no new state applied**, driven only by a backoff timer
expiring — work the old unconditional `sleep(0.1)` was quietly serving. Under the unscoped wording,
that correct behaviour would be charged against the fix.

> **SCOPING (binding):** "ZERO timeout-driven wakes on the request path" excludes dispatches gated
> by `KeyedBackoff.should_proceed` — i.e. `CreateRunner`/`DownloadModel` retries whose eligibility
> is time-based by design. A timeout-driven wake dispatching a backoff-gated retry is expected and
> does not count against the zero. **All other timeout-driven dispatches count exactly as before.**

## A3. Wake classification is 3-way, because `cancelled_caught` alone is not honest

Confirmed against installed anyio 4.11.0 (`_backends/_asyncio.py:492-509`): cancellation delivery
and the event's `set()` are independent scheduling events. If both land in the same window,
`cancelled_caught` reads True even though the event WAS set and the wakeup WAS delivered. Under a
strict "zero", one photo-finish would fail the fix for a scheduling coincidence.

`WakeKind` is therefore `Literal["event", "event_raced_timeout", "timeout"]`, derived from
`cancelled_caught` AND `waiting_on.is_set()`. **`event_raced_timeout` is an EVENT-driven wake and
does NOT count against the zero.** A true timeout is only when the event was never set.

## A4. Pairing semantics for the Gate-A delta (fixed now, before any data exists)

Several `state_applied` marks landing between two planner wakes is **correct coalescing, not a lost
wake.** Each `plan_step_observed` pairs to the **EARLIEST unpaired `state_applied`** since the prior
wake. Justification: Gate A exists to detect the removal of a polling delay, so worst-case
observation latency is the honest statistic; pairing to the latest would launder
coalesced-but-late observations into a falsely good number. Group by "state_applied marks not yet
claimed by a prior plan_step_observed" — `event_idx` is monotonic but not contiguous per window,
so do not pair by counting.

**Note:** `tmp/perf-campaign-2/round11/analyze_marks.py` parses a DIFFERENT stream (API-side JSON
from replay captures) and does **not** understand these worker log-line marks. A parser for this
stream **still needs to be written.**

## A5. Why Boot 1 was NOT taken this round (the plan is corrected, NOT spent)

Two capabilities required by this plan do not exist in the sanctioned access path, so **no boot
could have produced a Gate-A number**:
1. **Marks are unreadable.** They emit into `~/exo.log` on the worker node; the only allowlisted
   cluster path (`cluster-diag.sh`) has **no log-read subcommand**.
2. **The Step-2 workload cannot be driven.** It needs >=20 POSTs at 90-150K context; the script's
   only network call is a fixed GET to `/v1/models`.

Verified rather than assumed: the control host is NOT a cluster node (no local `~/exo.log`, no local
exo PIDs) and `start_cluster.sh` mirrors no logs back to it. Indirect workarounds (piggybacking
data onto the `/v1/models` response, encoding state into process titles, config-file IP indirection)
were considered and **rejected as routing around the restriction**; none were attempted.

**Relaunch budget remains 2 authorized, 0 used.** Steps 0-5 above stand as written, subject to
A2/A3/A4, and become runnable the moment a `marks` read capability exists.

---

# SECOND DATED AMENDMENT — 2026-09-04 (later same day), written BEFORE any measurement exists

**Nothing above this line is edited. The numeric bands are STILL UNCHANGED**
(median <=10 ms, p99 <=20 ms, request-path timeout-driven wakes == 0; Gate B |gap| <=10 ms;
actionability floor 75 ms). This amendment records (a) the three blockers named in A5 being
resolved, and (b) one further apparatus gap found and closed while building the reader — again
found pre-boot, again recorded rather than silently fixed.

## B1. All three A5 blockers are RESOLVED (two by the supervisor, one verified empirically here)

1. **Marks are now readable.** `cluster-diag.sh marks <m4-1|m4-2> [N]` (commit `7c441e110`) runs
   `tail -n N ~/exo.log | grep PHASE_MARK` on the node. Read-only by construction: fixed log path,
   fixed grep pattern, only the numeric tail count is caller-controlled. **A5 blocker 1 CLOSED.**
2. **`start_cluster.sh`'s silent-hang risk is gone.** Its pre-deploy push-check had a bare
   `read -p` with no TTY guard, which would have blocked forever in a background context. It now
   tests `[ -t 0 ]` first and fails loudly and immediately instead (`start_cluster.sh:1141`).
   **A5 blocker 3 CLOSED.**
3. **The workload CAN be driven — verified empirically this round, not assumed.** The existing
   round-11 capture path was launched and confirmed live: `passive_capture_proxy.py` binds
   127.0.0.1:52416 and forwards to the cluster; `curl http://127.0.0.1:52416/v1/models` returned
   **HTTP 200 in 0.098 s**. No new harness was built (R5's lesson). **A5 blocker 2 CLOSED.**

## B2. Amendment A2 was pre-registered but NOT MECHANICALLY DECIDABLE. Apparatus added.

A2 (binding, above) scopes Gate A's "ZERO timeout-driven wakes" to EXCLUDE dispatches gated by
`KeyedBackoff.should_proceed` — `CreateRunner`/`DownloadModel` retries whose eligibility is
time-based by design (`keyed_backoff.py:20` is `now - last >= delay`, no state precondition).

**But the mark did not carry the information needed to apply that exclusion.**
`mark_plan_step_observed(event_idx, wake_kind)` emitted only those two fields, so a backoff-gated
retry and a request-path dispatch were **indistinguishable in the data**. Any true timeout wake in
the run would have made Gate A's third sub-condition **UNDETERMINED — and would have partly wasted
one of only two authorized relaunches.**

Closed pre-boot. The dispatched `task` is already in scope at the mark site (`worker/main.py:344`,
after `plan()` at :322 and after the `task is None` guard at :335) — the disambiguator was present
and simply unrecorded. The mark now appends `task=<ClassName>` **verbatim**:

```
PHASE_MARK plan_step_observed event_idx=<int> t=<%.6f> wake_kind=<...> task=<ClassName>
```

**Deliberate design split: the emitter is DUMB, the policy lives in the ANALYZER.**
`phase_marks.py` records the class name and classifies nothing. The backoff-gated set
(`{CreateRunner, DownloadModel}`) lives only in `parse_worker_marks.py`, where it is auditable
against this pre-registration instead of being hardcoded into worker instrumentation. An
unanticipated task type therefore appears in the data **as itself** rather than being silently
bucketed into either side of the gate.

**OFF-path identity preserved (the binding constraint).** The call site passes the task *object*;
`__class__.__name__` and all string formatting happen **after** `if not _MARKS_ENABLED: return`,
which remains the first statement. Verified, not asserted: with `EXO_PHASE_MARKS` unset,
`_MARKS_ENABLED` imports as `False`. No `mx.eval()` was added (hard gate G1). Mark position,
control flow, and ordering relative to `plan()` are untouched. `state_applied`'s format is
unchanged. basedpyright delta **0**, ruff delta **0** on both changed files.

## B3. The A4 reader now exists (it did not when A4 was written)

A4 noted `round11/analyze_marks.py` parses a DIFFERENT stream (API-side JSON) and that a parser
for the worker log-line stream "still needs to be written." It is now written:
`tmp/perf-campaign-2/round13/parse_worker_marks.py` (stdlib-only, `/usr/bin/python3`), implementing
A2/A3/A4 exactly: earliest-unpaired pairing (never pair by counting — `event_idx` is monotonic but
NOT contiguous), 3-way `WakeKind` with `event_raced_timeout` counted as EVENT-driven, per-node
analysis only (no cross-node clock arithmetic, gate G2), medians and RANGES never bare means.

**Fail-loud properties, deliberate:** zero parseable marks exits non-zero; a Gate verdict is never
printed from an empty set; and old-format lines lacking `task=` raise the UNDETERMINED path rather
than silently defaulting to a clean pass. 25 synthetic unit tests pass under `/usr/bin/python3`
(PM re-ran them independently), covering coalescing, non-contiguous indices, both timeout
classifications, the backoff-gated exclusions, and empty/garbage input.

**Relaunch budget still 2 authorized, 0 used at the time of writing.** Steps 0-5 are now runnable
as written. No gate, band, or threshold has been altered by this amendment.

---

# THIRD DATED AMENDMENT - 2026-09-04 (later still), written BEFORE any measurement exists

**Nothing above this line is edited. The numeric bands are STILL UNCHANGED.** This amendment
records a THIRD latent apparatus defect, found in an adversarial pre-mortem run immediately before
Boot 1 and fixed pre-boot. Like the first two, it would have consumed a scarce relaunch and
produced an unusable number.

## C1. A4's pairing was UNDEFINED, because most planner wakes emitted NOTHING

Verified in source, not theorized:
- `mark_state_applied(event.idx)` sits at `worker/main.py:248`, inside `_event_applier`'s
  `async for event in events:` loop. It fires **unconditionally for EVERY IndexedEvent** - runner
  status updates, task status updates, per-token chunk events, gathered-info events, everything.
- `mark_plan_step_observed(...)` sat at `worker/main.py:344`, emitted **only AFTER** the
  `if task is None: continue` guard - i.e. only on the small minority of wakes where `plan()`
  actually produced a task.

**Consequence: the overwhelming majority of planner wakes left no record at all.** The planner
woke (correctly, on the event), ran `plan()`, got `None`, and continued silently. A4 says each wake
pairs to the earliest unpaired `state_applied` **since the prior wake** - but with no record of
task-less wakes, "prior wake" could only resolve to "prior `plan_step_observed`". The earliest
unpaired `state_applied` after that point would be a token or status event from **seconds** earlier.

**The measured median would have landed in the hundreds of ms or seconds, and Gate A would have
FAILED for a purely instrumental reason** - while the pre-registration correctly forbids
renegotiating the bands after seeing data. A correct fix would have been recorded as refuted.

**FIX (instrumentation correction; no band moves).** The mark is now emitted on **EVERY** wake of
the `plan_step` loop, before and regardless of the `task is None` guard, with `task=None` as a
first-class value meaning "the planner woke, ran `plan()`, and correctly found nothing to do."
Every `state_applied` now has a genuine successor wake, so A4's "since the prior wake" is
well-defined and coalescing becomes directly observable instead of inferred.

## C2. The wake timestamp is captured BEFORE `plan()`, not after

`t=` must represent **the wake**, not the wake plus `plan()`'s runtime; stamping inside the mark
function (which runs after `plan()` returns) would bias every delta upward by `plan()`'s cost.
`time.perf_counter()` is therefore captured at the top of the loop iteration, immediately after the
wait returns and before `plan()`, and passed in as `wake_observed_at`. Evidence recorded for the
record: `plan()` (`worker/plan.py`) is a plain synchronous function - no `await`, no I/O, no
subprocess or sleep - so stamping after it would likely have been harmless in practice; capturing
before it removes the question at zero cost rather than relying on that argument.

## C3. Mark-before-signal ordering: AUDITED, already correct

If the applier signalled the planner's event **before** emitting `state_applied`, the planner could
log `plan_step_observed` first, yielding negative deltas and orphaned marks. Checked rather than
assumed: `mark_state_applied` is at `worker/main.py:248`; `self._signal_state_applied()` is at
`:293`, the sole signal site, at the end of the same loop body. **Mark precedes signal. No change
needed.**

## C4. A2 refinement, decided NOW, before any data exists

Backoff-gated dispatches (`CreateRunner`/`DownloadModel`) are excluded from the Gate-A **delta
distribution** (median/p95/p99), not merely from the timeout count. Rationale: their eligibility is
purely clock-driven, so such a dispatch pairs to a stale `state_applied` and would pollute p99 with
a value that measures a backoff timer rather than wake latency. The excluded population is reported
separately with its own count and stats, so the exclusion is auditable rather than asserted. This
tightens A2 in the direction it already pointed; it does not move a threshold.

## C5. Apparatus self-check, replacing the un-measured baseline arm

The pre-registration gates on absolute thresholds against a flag-OFF baseline asserted *by
construction* (a 100 ms poll implies ~uniform arrival phase, median ~50 ms). Boot 2 restores
production with marks OFF and yields no baseline data, so no measured OFF arm will exist.

**Internal validity check, pre-registered here:** within the fix-ON arm, wakes with
`wake_kind=event` must show **sub-millisecond** deltas - that is what an event-driven wake means.
If `wake_kind=event` pairs to ~40 ms deltas, **the APPARATUS is wrong, not the fix**, and the run
must NOT be read as a Gate-A result. The analyzer prints this as an explicit PASS/SUSPECT line.
This is a falsifier for the instrument, deliberately fixed before data exists.

**Cold-start handling corrected:** marks carry no request id, so discarding "the first 3 requests"
positionally is unsound. Cold/startup marks are identified by TASK TYPE
(`CreateRunner`/`LoadModel`/`DownloadModel`/`StartWarmup`) and reported separately.

## C6. The OFF-path invariant was briefly weakened, and is restored

R11 established a load-bearing invariant: with `EXO_PHASE_MARKS` unset, the production path
executes **one boolean check of a module constant and nothing else**. That invariant is the safety
argument for leaving this instrumentation permanently in the shipping tree. C2's timestamp capture
initially ran unconditionally, quietly violating it (a `perf_counter()` call per wake, forever, in
production, immediately discarded by the early-return).

The wall-clock cost was immaterial (~30-80 ns at ~10 wakes/sec). **It was fixed anyway, because a
quietly-weakened invariant is worse than a slow one.** The capture is now gated inline on the same
module-level constant (`MARKS_ENABLED`, a public alias of `_MARKS_ENABLED` - no second source of
truth, no new `os.environ` read). Deliberately NOT wrapped in a helper function: a Python call
would cost more than the `perf_counter()` it avoids. Verified on both arms: `MARKS_ENABLED` is
`False` when unset and `True` under `EXO_PHASE_MARKS=1`.

**Gates on all of C1-C6:** 40 analyzer unit tests pass (up from 25; the new ones prove a wake pairs
to the correct nearby `state_applied` and NOT to a stale seconds-old one - the exact defect above).
`pytest src/exo/worker/tests`: 291 passed, 0 failed. basedpyright delta **0**; repo-wide count
unchanged at 4909. ruff clean. `nix fmt` unavailable on this host - skipped, not faked.

**Relaunch budget: 2 authorized, 0 used. No gate, band, or threshold has been altered.**
