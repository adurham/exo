# OVERNIGHT LOOP — STATE

**Mode:** AUTONOMOUS (authorized 2026-09-02, runs until user says STOP). **Do NOT pause for a
go/no-go between rounds** (user, 09-04: "You don't need to stop and wait for me"). Record →
consult → dispatch → report. Stop only for a genuine blocker or an explicit STOP.
**Charter:** `/Users/adam.durham/.hermes/cache/OVERNIGHT-LOOP-CHARTER.md` (read first on context loss;
includes the mandatory GREP-THE-RECORD-FIRST step).
**Supervisor model:** claude-opus-5

---

## WHERE THINGS STAND

**Campaign 1 (2026-08-30 → 09-03): CLOSED.** The 20-vs-34 tok/s "gap" was a measurement
convention (94.5% TTFT). Shipped: pad-strip (server cached_tokens 0→351), canonical serializer +
golden bytes, BatchPoolingCache fix, profiler unit metadata, CI re-enabled + guards live.

**Hardening (09-03): DONE.** CI had been `disabled_manually` for 8 months; now runs pytest green
(1107 passed). Alignment guard, kernel guard, golden-token nondeterminism root-caused (OOB embedding
read into recycled Metal pages). tmp/ 16GB→34MB + rsync exclude.

**Campaign 2 (09-03 → 09-04, 8 rounds): THROUGHPUT PHASE CLOSED. ZERO SHIPPED.**
Every lever closed with source/GPU evidence. Decode is at a physics floor for this model on this
hardware; prefill at c=1 is exhausted and confirmed on the production path. Full ledger below.
Five supervisor errors in briefs were caught by PMs and are in the record.

---

## CAMPAIGN 2 LEDGER (per-item, with the round that closed it)

| # | lever | verdict | round |
|---|---|---|---|
| I1 | TP all_sum latency | CLOSED — jaccl 2.6% of budget; MLX has NO GPU collective (`AllReduce::eval_gpu` throws, jaccl stream is CPU); GPU-resident collective INFEASIBLE (no public GPU→DMA coherence API; jaccl host-bounce by construction). Stack-level floor. | R1-R3 |
| I2 | c=2 tax knobs | CLOSED — 3 knobs dead code; FENCE cadence live but moot with async fence armed; `MLX_STEEL_BATCH_INVARIANT` is CORRECTNESS-load-bearing at c=1 (byte-identity fails both regimes); RENDEZVOUS=0 safe but unresolvable on the TTFT instrument → folds into TTFT pivot | R1/R4/R7 |
| I3 | kernel bandwidth | CLOSED — 83.9% of peak (chained-graph). The "53%" was the retracted 08-22 serial-sync artifact reproduced | R1 |
| I4 | Fix B re-open | CLOSED — round-4 test was invalid (sub-chunk) but conclusion held on a valid re-test | R1 |
| I5 | γ re-tune | CLOSED — all reachable γ measured on a calibrated ruler; γ=4 is −4.5 t/s vs 1.3 boot spread; cycle 61→74ms monotonic in γ. HOLD γ=3 | R5/R6 |
| I6 | per-row expert reads | CLOSED — 1.42× shared-vs-distinct, ~4-8ms of 56ms | R1 |
| I7 | lm_head vocab-sharding | CANCELLED — ~0.45 t/s, below measurement floor | R7 review |
| I8 | per-draft cost | RESOLVED — ~3-6ms per extra draft row; acceptance doesn't cover it | R6 |
| I9 | GPU P-state | CLOSED — decode clock 0.2% ABOVE prefill | R8 |
| I10 | Fix A trie persistence | CANCELLED by user — "I don't normally re-launch in the middle of sessions." The 49% figure measured CAMPAIGN relaunches, not normal usage; ~0 value in steady state | user, 09-04 |
| I11 | expert precision | **PREMISE FALSE** — experts are ALREADY mxfp4 4-bit at load (`deepseek_v4.py:952`). "6-bit" was a supervisor error in 3 briefs. 3-bit = +4%, inside boot variance. No change | R8 |
| I12 | serial-vs-batched prefill parity | CLOSED — all six optimizations reachable from the serial driver by construction | R8 |
| I13 | idle-gap warmup | CANCELLED — only matters if warmup is in the shipping loop | R7 review |
| I14 | within-boot drift | CANCELLED — measurement artifact | R7 review |
| I15 | kernel-launch count | BLOCKED — probe vars boot-gated; rides free on the next boot | R8 |

**Key facts established this campaign** (each source- or GPU-verified): async fence IS armed
(≥98.5%, the 08-22 +58-67% fix holds under MTP-on); verify GPU is 117% busy on real compute —
no idle gap; removing the entire collective moves wall ≤8.4%; the 06-26 "fence is load-bearing
for bit-equiv" story was an algebra bug; `FENCE_EVERY_N_LAYERS` HAS a live reader
(deepseek_v4.py:2959 — R4's "dead" claim was wrong); `EXO_PROFILER=spans` silently re-blocks the
fence; `ab_probe_tier1.py` is NOT a pass/fail gate.

**Supervisor errors caught by PMs (all in the record):** R1 "never attacked" framing → reproduced
a retracted artifact; R3 SHA mis-parse of `git submodule status` (hardening); R4 "FENCE is dead"
(wrong); R7 cited a nonexistent gate; R8 "experts are 6-bit" (wrong, 3 briefs). Process fix:
charter now mandates grep-the-record before any brief.

**Measurement reality:** ~6 t/s between-boot decode variance (P13-P15), 1.3 t/s within bracketed
boots this week; prefill boot-stable to 0.02%. Server `stats.generation_tps` via
`bench/long_decode_probe.py` (calibrated 29.1@2K, 32.8@89K) is the only trusted decode ruler.
Never build a new harness — R5 lost a round to one.

---

## NEXT (loop BLOCKED on a NAMED APPARATUS GAP — see R13 below)

### >>> RESUME POINTER (read this first on context loss) <<<

**CLUSTER ACCESS IS RESTORED. The R12 "DO NOT TOUCH THE CLUSTER" block is SUPERSEDED and has
been deleted.** It described a permission denial that the supervisor has since fixed. Do not
reinstate it; do not ask the user to re-authorize read-only cluster contact.

**The ONLY sanctioned path to the nodes is `/Users/adam.durham/repos/exo/cluster-diag.sh`**
(chmod 555, read-only, committed 062c4117e, allowlisted by exact path in Hermes'
`command_allowlist`). Subcommands: `health|env|sha|ps|gpu <m4-1|m4-2>` (`env` takes a VAR from a
fixed 10-name list). **Raw ssh/curl to the cluster is STILL hard-denied and must never be
attempted** — including indirectly (no config-file IP indirection, no piggybacking data onto
`/v1/models`, no encoding state into process titles). If a round needs a capability the script
lacks, that is a REPORTABLE BLOCKER: name it precisely so the supervisor can extend the script
via a reviewable diff.

**CLUSTER IS HEALTHY — FRESHLY VERIFIED 2026-09-04 (R13) ON REAL PIDs, not inherited.**
API 200; HEAD `096a00a58` identical on both nodes; m4-1 PIDs 16063/16064/16065/16075 (byte-identical
to R11's set, confirming R12 never touched it), m4-2 PIDs 29674/29675/29676/29685; RV=0 on all four;
γ=3; steel-BI=1; gemv-BI=1; `EXO_PHASE_MARKS` ABSENT; `EXO_WORKER_PLAN_EVENT_WAKE` ABSENT.
Production config, nothing left behind. **Relaunch budget: 2 authorized, 0 used.**

### R13 (2026-09-04): PRE-REGISTRATION WAS LATENTLY INVALID. CAUGHT PRE-BOOT. CORRECTED, NOT SPENT.

**R13's PREDICTION.md assumed `EXO_PHASE_MARKS=1` produces Gate A's
`state-update-applied -> plan_step-observed` pair. IT DOES NOT.** Marks existed only in the API
process (wrong process/clock) and the runner subprocess (wrong scope). `worker/main.py`, where
`plan_step` lives, had ZERO marks. **Boot 1 as pre-registered would have burned a relaunch and
produced no Gate-A data at all.** The apparatus was built this round (`src/exo/worker/phase_marks.py`,
same EXO_PHASE_MARKS gate, `IndexedEvent.idx` as pairing key, no `mx.eval` near a mark).
Three further latent Gate-A flaws were found and adjudicated pre-measurement — see the R13 section
of `docs/PERFORMANCE_HISTORY.md`, including a **dated Gate-A amendment (apparatus/scoping only;
the median<=10ms / p99<=20ms bands are UNCHANGED)**.

**THE BOOT WAS DELIBERATELY NOT TAKEN.** Two required capabilities do not exist in the sanctioned
path, so a boot could not have yielded a Gate-A number no matter how it went:
1. **Marks are unreadable** — they land in `~/exo.log` on the worker node; `cluster-diag.sh` has
   **no log-read subcommand**.
2. **The workload cannot be driven** — Step 2 needs >=20 POSTs at 90-150K context; the script's only
   network call is a fixed GET to `/v1/models`.
Verified, not assumed: the control host (`adams-macbook-pro-m4`) is NOT a cluster node and holds no
local `exo.log`; `start_cluster.sh` mirrors no logs back (its only `scp` is push-direction, line 2896).

### WHAT THE SUPERVISOR MUST PROVIDE TO UNBLOCK (the actual next action)

1. **`cluster-diag.sh marks <node> [n]`** — read back only `PHASE_MARK`-prefixed lines from
   `~/exo.log`. Stays read-only by construction. **Without this, Gate A is unobtainable, forever.**
2. **A workload driver** for >=20 POSTs at 90-150K context. **Cannot** be a read-only extension —
   this is a genuine write capability and is the supervisor's security call (either a
   `replay <node> <profile>` wrapper around the existing `replay_c1.py`, or a supervisor-run workload).
3. **`start_cluster.sh` allowlist entry + non-interactive mode.** It is NOT allowlisted and was NOT
   attempted this round. **LANDMINE:** `start_cluster.sh:1141` has an interactive
   `read -p "Continue anyway? (y/N)"` that fires when local HEAD is not an ancestor of `origin/main`
   — **it will hang forever in a background context.** Push before launching. There is no dry-run,
   single-node, or config-check mode; every invocation does a real rsync + kill + relaunch.

**DEPLOY MECHANISM (durable fact, established from source this round):** `start_cluster.sh:1337-1344`
**rsyncs the control host's WORKING TREE** to both nodes; it does NOT `git pull` on the nodes.
**Uncommitted changes DO ship, and the nodes' `git rev-parse HEAD` does NOT reliably indicate the
code they run** — verify file CONTENT. Allow-listed env vars are prefixed directly onto the remote
`.venv/bin/python -m exo -v` (line 2849), i.e. they reach the process containing `plan_step`; an
unset var is absent, never `0`, never stale.

**Once capability #1 (and #2 or a supervisor-run workload) exists, R13's PREDICTION.md is runnable
as written** — health/sha/ps/env validation → Boot 1 with `EXO_WORKER_PLAN_EVENT_WAKE=1
EXO_PHASE_MARKS=1` (verify BOTH vars on real runner PIDs before spending the workload) → Gate A /
Gate B under the dated amendment → Boot 2 restore at the NEW SHA with the flag OFF → R14 branch table.
**No supervisor DECISION is pending — only the capability.**

### DEFINITION OF "SHIPPED" (a commit on main is NOT a ship)

**I16 is shipped only when the flag is ON in production AND Gate A has PASSED on hardware.**
Neither holds. **The 100 ms tick is STILL LIVE IN PRODUCTION.** The fix (`84bdcd756`) is
committed, pushed, and **env-gated DEFAULT OFF** — dormant code. **Do NOT flip the default to
ON without a Gate-A pass.**

### CLUSTER HEALTH — FRESHLY VERIFIED (R13, 2026-09-04)

Superseding R12's inherited claim: health was re-verified on REAL PIDs in R13 via `cluster-diag.sh`.
API 200, HEAD identical on both nodes (`096a00a58`), m4-1 PIDs 16063/16064/16065/16075 — **the exact
same PID set R11 recorded**, which independently confirms R12's by-construction claim that it never
touched the cluster. RV=0, γ=3, steel-BI=1, gemv-BI=1, `EXO_PHASE_MARKS` absent,
`EXO_WORKER_PLAN_EVENT_WAKE` absent. No probe/diag leftovers. **Production config, healthy.**

**R12 status:** lost-wakeup safety gate PASSED in unit tests (37 passed, PM re-ran
independently). OFF-path identity **empirically tested**, not asserted: with the gate unset the
loop makes exactly `sleep(0.1)` calls, zero `move_on_after`, and the signal is a no-op —
so `main` is production-identical. No performance number exists; the 100 ms figure is still
code-read-derived and R11's "do not ship on a code read" gate is UNMET. Relaunch budget:
2 authorized, **0 used**.

**Free win banked (zero cluster time):** the seam harness RAN against the LOCAL on-disk
checkpoint (`/dev/disk3s5`, no network mounts — verified, no cluster access involved).
Tokenizer-level seam rule HOLDS, normalizer inert — but **template position-invariance FAILS**:
`render(msgs[:4])` is not a byte-prefix of `render(msgs[:5])` (diverges at char 403) because the
vendored DSv4 encoder re-sorts tool results on every call. **A prefix cache keyed on
message-list position is provably UNSAFE for multi-tool-result conversations** (44/55 real
requests end in tool_calls). This pre-constrains Branch T's design before it is funded; it does
not close it. Branch T is amended accordingly in round13/PREDICTION.md.

**Correction to propagate:** the "basedpyright baseline = 425" figure in R11's REPORT is WRONG
for this tree. Real baseline (git worktree): 4909 for `src`, 13155 repo-wide. Delta gates the
change and is 0. **Do not quote 425 again.**

---

## (historical) NEXT as of R11

**User's standing bar (09-04): "every possible performance enhancement we can get without impacting
quality matters."** Sub-second wins earn a bounded round if quality is provably untouched.

**R10 SHIPPED** (RENDEZVOUS_MS 200→0, −224 ms/turn, 096a00a58). **R11 PARTIAL** (wrapped for a
control-host reboot; cluster never relaunched). **RESUME HERE →** R11 Task 2: one instrumented
boot + closure check per tmp/perf-campaign-2/round11/REPORT.md §3, then R12 on the top-ranked
phase (leading candidate: the 100 ms plan_step poll tick, worker/main.py:195). Prior in-flight note follows for history:
**(was) IN FLIGHT: R10 (deleg_0d2cf7c9)** — close RENDEZVOUS_MS 200→0 on the RESIDUAL instrument R9
established (raw TTFT carries an arm-independent compute term that flips sign; the residual is
stable at −183 ms, in band). Pre-register residual as governing, recompute from R9 JSONs, one
confirmatory pair n=25, byte-identity with self-controls, ship or hold. Rides along: R7's missing
steel-BI 89K same-arm self-control.

**AFTER R10:** consult for next direction under the standing bar. Candidates not yet exhausted
under "small but quality-free": the R7 steel-BI re-test IF the self-control voids its 89K leg
(its <8192 leg still stands, so likely still HOLD); anything the consult surfaces. Fix A is DEAD
(relaunches are not part of normal sessions). I15 CLOSED (no launch-count probe exists on the
decode path). Everything else in the ledger is closed.

**Cluster (as of R11, 2026-09-04 — NOT re-verified since):** was healthy on shipped production
config (γ=3, BI=1, RV=200→0 per R10, mxfp4 experts, async fence armed), both nodes READY. No
probe/diag env leftover. **R12 could NOT re-verify this** — cluster access was denied at the
tool-permission layer. R12 provably did not change it (`start_cluster.sh` never ran, so no deploy
reached the nodes), but "healthy" is now an INHERITED claim, not a fresh measurement. Re-verify on
real PIDs before trusting any number.
