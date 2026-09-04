# CAMPAIGN 2 / ROUND 12 — PRE-REGISTRATION
Committed BEFORE any boot, BEFORE any number is computed. Fable-reviewed 2026-09-04.

## Lever ID: I16 — worker/main.py:195 plan_step 100ms poll tick

## Difference from prior work (GREP-THE-RECORD-FIRST requirement)
R10 fixed an analogous poll sleep in the runner's rendezvous path (EXO_BATCHED_PREFILL_RENDEZVOUS_MS
200->0, shipped -224ms/turn). I16 is a DIFFERENT sleep, DIFFERENT process (worker/main.py vs
runner.py), DIFFERENT trigger (plan_step's outer while-loop vs prefill rendezvous). Never
previously measured on hardware — R11 found it by code read only. No prior round closed this.

## Boot budget: 2 relaunches authorized. Fable flagged the original plan needed 3
(baseline-instrumented, fix+instrumented, restore-production). Resolved as fable's option (a):
Boot 1 = fix applied AND EXO_PHASE_MARKS=1 together. No separate baseline-only boot.
Rationale: worker/main.py:195's sleep(0.1) is a known constant; we don't need a live baseline
boot to prove a `await anyio.sleep(0.1)` blocks up to 100ms — that's true by inspection. What
needs live measurement is (i) whether removing it actually shows up in the measured
dispatch_and_ipc_gap, and (ii) whether the fix is safe (no lost wakeups, no error/retry storm).
Boot 1 = fix + marks. Boot 2 = restore production (marks off, fix reverted OR kept per ship
decision below).

If Boot 1 shows the fix is unsafe (errors, retries, timeout-driven wakes on the request path) or
inconclusive on timing: REVERT the fix on Boot 2, ship nothing, record as I16 CLOSED-NEGATIVE,
hand to R13 branch table below. Do not spend a 3rd boot chasing it this round.

## Gate A vs Gate B (fable correction — do not conflate)
- **Gate A (ships the fix):** intra-worker delta, same clock, state-update-applied ->
  plan_step-observed (NOT the composite dispatch_and_ipc_gap). Baseline fingerprint by
  construction (known sleep(0.1), random phase): ~uniform, expected median 35-65ms, p95
  85-110ms, hard ceiling ~100ms+loop-body. Post-fix PASS: median <=10ms, p99<=20ms, ZERO
  timeout-driven wakes on the request path (count them explicitly — any timeout-driven dispatch
  means the event-signal path is incomplete, i.e. partial fix, do not ship as complete).
- **Gate B (decomposition closes, informational only):** |unattributed gap| <= 10ms per R11's
  original closure check. A Gate-B miss does NOT block a Gate-A ship. A Gate-A ship does NOT
  imply Gate-B closure — report them separately.

## Correctness / safety gates (pre-registered, non-negotiable)
1. Byte-identity: c=1 output tokens identical, fix-on vs fix-off, same prompt, temp=0, at 3
   context sizes (2K, 90K, 150K+). This is a pure scheduling change — it must not alter output.
2. Zero worker errors/retries/mis-dispatches introduced (compare error-log line count fix-on vs
   fix-off over the same request set).
3. Worker idle CPU <= baseline + 1% (no spin-wait regression).
4. plan_step wake rate under a c=4 burst does not run away (bounded, no unbounded wake storm).
5. Lost-wakeup safety: `anyio.Event` has no `clear()` — implementation MUST create a fresh Event
   after each wake and grab the reference BEFORE checking state (state-setter mutates state,
   THEN sets event) or a wakeup can be lost between check and wait. If state is applied from a
   different thread/task than plan_step's loop, cross-context signaling
   (call_soon_threadsafe-equivalent under anyio) must be used, not a bare set() from another
   thread. Unit test the lost-wakeup window before the live boot.
6. Sleep(0.1) MUST remain as a fallback timeout on the wait — never a bare await-forever on the
   event. Signal from the single state-apply point, not per-task-type heuristics (fewer missed
   paths).

## Post-reboot environment validation (fable flag — the control host rebooted since R10)
Before trusting ANY number from Boot 1: verify non-persistent tunables match prior rounds
(iogpu.wired_limit_mb etc per exo-cluster-operations skill), interconnect link type/path
unchanged (TB5 RDMA, not a fallback NIC), `git rev-parse HEAD` identical on both nodes and
matches what was pushed, no background mds/softwareupdate/backup load, discard first 3 requests
as cold, and confirm baseline prefill/decode TPS within +/-3% of last recorded production numbers
(29.1 t/s @2K per calibrated ruler) before treating the boot as valid. If any check fails: STOP,
do not spend the workload on an invalid boot.

## Round-time checkpoint (fable flag — R11 lost time to an external interrupt)
If Boot 1 + baseline verification isn't complete by T+75min: commit the fix + unit tests as-is,
do NOT attempt the fix arm this round, defer the live boot to R13. Commit+push all round-12
artifacts BEFORE attempting Boot 2, regardless of how the round goes.

## Cheap parallel work (no extra boots, do during any idle/wait time)
- Compare worker-side vs client-side inter-token interval from Boot 1 capture: constant offset
  is fine, a RATE difference flags a return-path bottleneck masked by worker-side "physics
  floor" framing.
- Write (do not ship) the tokenization-cache seam-rule harness per R11 REPORT.md section 4,
  including fable's adversarial seam corpus additions: unicode combining chars, emoji, digit/
  whitespace runs at the seam, plus a template position-invariance test (render(msgs[:i]) must
  be a byte-prefix of render(msgs[:j]) at any candidate safe seam). This feeds R13 branch T
  without spending a boot.

## R13 BRANCH TABLE — pre-registered NOW so the loop does not stall on a supervisor round-trip
Selected mechanically from Boot 1 data, whether or not I16 ships:
- Branch H (any OTHER hop gap — API->master, master->worker, return path — shows the same
  uniform-tick fingerprint, median >=20ms): apply the same event-wake pattern, same Gate A/B
  structure as I16. Highest prior given two ticks found already (R10, R11).
- Branch T (render+tokenize phase >=75ms at 150K context, from Task-2 marks): tokenization
  prefix cache using the seam-rule harness above. Bands: phase reduced to <= max(20ms, 10% of
  baseline); token-identity 100% on >=200 real requests + the adversarial corpus; cached_tokens
  hit rate unchanged; byte-identity PASS mandatory. Measure render and tokenize as SEPARATE
  phases — if render dominates over tokenize, the fix is incremental template rendering, a
  different design than a token cache.
- Branch W (trie/prefix-walk phase >=75ms): vectorized longest-prefix-match replacing any
  remaining per-token Python loop; match length must be identical on all requests.
- Branch U (|unattributed| >10ms on Gate B with no single phase crossing the 75ms actionability
  floor): extend instrumentation granularity next round, no fix attempted.
- No branch triggers AND Gate B closes clean: pre-register this as CAMPAIGN 2 CLOSE-OUT — the
  fixed-cost residual is many-small-things below the actionability floor, do not invent a lever
  to keep the loop busy.
- Tie-break if multiple branches trigger: largest measured median wins; ties go to H.

Note: `_save_prefix_cache` running pre-first-token inside submit() (R11 section 2e) is a HEAD
cost on the critical path, not a refutation — bracket submit()'s internal phases (commit /
restore / compute) separately in Boot 1 so this is resolved without a dedicated round.
