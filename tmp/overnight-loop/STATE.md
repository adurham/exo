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

## NEXT (loop BLOCKED — RELAUNCH BUDGET EXHAUSTED, 2 of 2 USED — see R13-continued below)

### >>> RESUME POINTER (read this first on context loss) <<<

**CLUSTER IS HEALTHY. Verified 2026-09-04 on REAL PIDs after the restore boot, not inherited.**
API 200; runners **READY 2/2**; a real completion was confirmed against the placed checkpoint.
`EXO_PHASE_MARKS` **ABSENT**, `EXO_WORKER_PLAN_EVENT_WAKE` **ABSENT**, RV=0, γ=3, steel-BI=1.
Production config, nothing left behind.

**The ONLY sanctioned path to the nodes is `/Users/adam.durham/repos/exo/cluster-diag.sh`**
(read-only, allowlisted by exact path). Subcommands: `health|env|sha|ps|gpu|marks <m4-1|m4-2>`.
`marks <node> [N]` was ADDED and is now PROVEN on hardware — it reads back `PHASE_MARK` lines from
`~/exo.log`. **Raw ssh/curl to the cluster remains hard-denied**; if a round needs a capability the
script lacks, name it as a REPORTABLE BLOCKER rather than routing around it.

### >>> THE ONE THING BLOCKING GATE A: RELAUNCH BUDGET <<<

**2 authorized, 2 USED. Both spent 2026-09-04.** Gate A needs **ONE more instrumented boot plus its
restore**. Nothing else is missing — every other piece is built, verified on hardware, and
pre-registered. **This is a supervisor decision, not a capability gap.**

### R13 (2026-09-04): APPARATUS PROVEN END-TO-END. GATE A **NOT MEASURED**. NOTHING SHIPPED.

**Full record: `docs/PERFORMANCE_HISTORY.md`, R13-continued section. Read it before any new brief.**

**Three defects were caught PRE-BOOT** (the third by an adversarial pre-mortem run minutes before
Boot 1), each of which would have burned a relaunch and produced an unusable number:
1. **A2 was not mechanically decidable** — the mark lacked the field distinguishing a backoff-gated
   retry from a request-path dispatch. Fixed: `task=<ClassName>` recorded verbatim.
2. **A4's pairing was UNDEFINED** — `mark_state_applied` fires for EVERY event, but the wake mark
   sat AFTER the `task is None` guard, so most wakes left no record and each wake would have paired
   to a `state_applied` from seconds earlier. **Gate A would have failed for an instrumental reason,
   and the bands are correctly not renegotiable after seeing data.** Fixed: emit on EVERY wake with
   `task=None` as a first-class value.
3. **The OFF-path invariant was weakened** (unconditional `perf_counter()` per wake) **and restored**
   — immaterial in ns, but it is the safety argument for keeping this instrumentation in the
   shipping tree. Now gated inline on `MARKS_ENABLED`; verified on both arms.
Audited and already correct: mark-before-signal ordering (`:248` precedes `:293`).

**APPARATUS SELF-CHECK PASSED ON HARDWARE — the round's durable win.** `EXO_PHASE_MARKS=1` reached
the real runner PIDs on both nodes; marks emitted, were read back via `cluster-diag.sh marks`, and
parsed correctly on the first attempt. On the fix-ON arm, `wake_kind=event` pairs showed
**median 0.436 ms (n=312)** against a pre-registered SUSPECT threshold of ≥1.0 ms.

> **THIS IS NOT GATE A AND MUST NEVER BE CITED AS ONE.** It is boot/idle traffic. The p95/p99
> (~16–18 s) are pure startup artifacts. Gate A requires ≥20 request-path completions at 90–150K
> depth; **ZERO occurred.** It is apparatus validation plus weak mechanism evidence — it does **NOT**
> authorize a ship.

**WHY THE WORKLOAD PRODUCED NOTHING (root-caused, not guessed):** `replay_c1.py:40` requested
`deepseek-ai/DeepSeek-V4-Flash` but `start_cluster.sh:379` places
`deepseek-ai/DeepSeek-V4-Flash-0731`. **Both ids appear in `/v1/models`, so the API ACCEPTS the wrong
one**, then tries to JIT-load a second ~152 GB checkpoint, fails on memory, and 503s after 120 s.
**Listing ≠ serveability.** Boot 1 separately never converged (0/2 runners Ready) due to a
memory-reclaim race at boot; the restore boot hit the same transient message and recovered — **not a
code defect, instrumentation not implicated.**

**HARDENED so it cannot recur:** `replay_c1.py` now defaults to `-0731` (comment cites
`start_cluster.sh:379` as source of truth), accepts `--model`, **runs a PRE-FLIGHT probe that proves
the model is SERVEABLE and exits non-zero BEFORE spending the workload**, aborts if the first 3
requests all error, and is genuinely Python-3.9 compatible (PEP-604 annotations were crashing it
instantly under the pinned `/usr/bin/python3`).

### DEFINITION OF "SHIPPED" (a commit on main is NOT a ship)

**I16 is shipped only when the flag is ON in production AND Gate A has PASSED on hardware.**
Neither holds. **The 100 ms tick is STILL LIVE IN PRODUCTION.** The fix is committed, pushed, and
**env-gated DEFAULT OFF** — dormant code. **Do NOT flip the default without a Gate-A pass.**

### EXACT RESUME STEPS (everything below is built and verified — only the boot is missing)

1. `EXO_WORKER_PLAN_EVENT_WAKE=1 EXO_PHASE_MARKS=1 ./start_cluster.sh` (push first; HEAD must equal
   origin/main or it now fails LOUDLY rather than hanging).
2. Verify BOTH vars on real runner PIDs: `./cluster-diag.sh env <node> EXO_PHASE_MARKS`. **If either
   is absent: STOP, do not spend the workload.**
3. Confirm runners reached **READY 2/2** — Boot 1 did not, and that alone invalidates the run.
4. `/usr/bin/python3 tmp/perf-campaign-2/round11/replay_c1.py --requests 40` (≥40, not 20: with ~17
   samples p99 == max and one GC pause fails the gate). The preflight now blocks a bad-model run.
5. `./cluster-diag.sh marks <node> 20000 > marks.txt` for BOTH nodes, then
   `/usr/bin/python3 tmp/perf-campaign-2/round13/parse_worker_marks.py marks.txt` per node
   (never pool across nodes — no cross-node clock arithmetic).
6. Apply Gate A / Gate B **VERBATIM** from `round13/PREDICTION.md` (bands UNCHANGED through three
   amendments: median ≤10 ms, p99 ≤20 ms, request-path timeout-driven wakes == 0).
7. Restore boot with both flags OFF; re-verify health on real PIDs.

**Bands are pre-registered and NOT renegotiable after seeing data.**

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
