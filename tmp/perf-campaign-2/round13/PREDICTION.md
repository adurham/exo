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
