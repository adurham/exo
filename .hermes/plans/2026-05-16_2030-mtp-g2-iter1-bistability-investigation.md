# γ=2 MTP Iter-1+ Bistability — Investigation and Fix Plan

> For Hermes: Execute task-by-task. Each task is one decision gate;
> proceed to next task only if exit criteria are met. STOP and present
> data to user at every gate; do NOT auto-continue across gates.

**Goal:** Find and fix the residual γ=2 MTP-on bistable stall on iters 1+
on the 2-node exo cluster, so that 5-iter c=1 100K benches run stably at
>=30 t/s across ALL iters (not just iter 0).

**Tech Stack:** mlx fork (adurham/mlx@main at 4d21baa2 — has the eager-commit
patch already), exo (adurham/exo@main at dbbd03ce), DSv4-Flash-8bit on 2x
Mac Studio M4 Max + TB5 RDMA.

**State Today (2026-05-16 ~20:30):**
- Eager-commit patch deployed on cluster, env var
  MLX_EAGER_COMMIT_BEFORE_CPU_COLLECTIVE=1 confirmed in libmlx symbols
  and runner ps -E.
- 3-iter γ=2 c=1 100K bench at HEAD: 31.7 / 31.9 / 32.0 -> CLEAN (lucky).
- 5-iter γ=2 c=1 100K bench at HEAD: 31.9 / 11.0 / 6.1 / 11.1 / (killed
  before iter 4) -> iter 0 HIGH, iters 1+ LOW. Reproduces the documented
  bistability from references/mtp-poll-stall-diagnosis.md.
- Conclusion: eager-commit eliminates the iter-0 cold-start LOW lock
  (independently valuable, prevents 3.5 t/s artifact) but does NOT
  resolve the persistent iter-1+ stall mechanism.

**Architecture of the investigation:** Each task is a hypothesis,
experiment, and decision gate. We do not write patches until we have
data that tells us which patch to write.

---

## Task 1: Baseline reproducibility check (no new patches)

**Hypothesis:** The iter-1+ stall is reproducible (not session-bound,
not noise). A second 5-iter run on the SAME cluster state will show
the same pattern — iter 0 HIGH, some/all of iters 1+ LOW.

**Why we need this:** Pitfall #41 (3-iter clean is necessary but not
sufficient). The single 5-iter run with iters 1-3 stalled is N=1. If
it's a real bug we need N>=2; if it's transient it won't reproduce.

**Files touched:** None. Bench-only.

**Cluster pre-state required:**
- Current cluster from 18:32 launch (MLX_EAGER_COMMIT_BEFORE_CPU_COLLECTIVE=1,
  γ=2, NO_BROADCAST=1, QoS=user_interactive, trace+profiler off).
- No restart between killing the prior bench and Task 1.

**Steps:**
1. SSH to m4-2, truncate bench log, launch 5-iter γ=2 c=1 100K bench
   identical to prior:
   --concurrency 1 --iterations 5 --warmup 0 --max-tokens 128
   --prompt-words 75000 --timeout 1800
   --json-out /tmp/bench_task1.json --label task1_repro
2. Poll every 4-6 min per the honesty discipline (pgrep BEFORE each
   sleep). Total wall ~32 min for 5 iters.
3. Pull JSON via scp to laptop.

**Exit criteria and branching:**
- STALL REPRODUCES (>=1 of iters 1-4 has gen_tps <25): Proceed to Task 2.
- CLEAN RUN (all 5 iters gen_tps >=30): the prior 5-iter LOW-mode run was
  lucky-bad. Note in memory, declare eager-commit a real fix, run a
  3rd 5-iter to confirm. Tag champion. STOP plan.

**Cost:** ~32 min cluster, ~5 min orchestrator wall.

---

## Task 2: Diagnostic restart — collect per-call data on stalled iters

**Hypothesis:** The stall happens at specific call_id(s) and the existing
JACCL_POLL_INSTRUMENT + MLX_SIGNAL_PROBE diagnostics in the binary can
tell us exactly which sub-mechanism fires.

**Why we need this:** The diagnosis doc has decision-matrix interpretations
of these diagnostics. Running them tells us which of three remaining
hypotheses is true:
1. in_poll_us / total_wall_us > 90% AND max_single_poll_us > 1000 ->
   librdma syscall is blocking inside kernel. Apple driver issue.
2. in_poll_us / total_wall_us > 90% AND max_single_poll_us < 100 ->
   CQE genuinely arrives late from peer. Peer-side or wire-side latency.
3. in_poll_us / total_wall_us < 50% -> thread descheduled despite
   busy-poll. Already ruled out by Mach RT but never know.

**Files touched:** None (just env-var changes on restart).

**User action required (SSH agent chain pitfall #42):**

User pastes in their shell:

    cd ~/repos/exo
    EXO_TRACING_ENABLED=false EXO_PROFILER_LEVEL=0 \
    EXO_DSV4_MTP=1 EXO_SPECULATIVE=1 EXO_SPECULATIVE_GAMMA=2 \
    EXO_DSV4_TOPK_FUSED=1 EXO_DSV4_MTP_NO_BROADCAST=1 \
    MLX_STREAM_QOS=user_interactive \
    MLX_EAGER_COMMIT_BEFORE_CPU_COLLECTIVE=1 \
    JACCL_POLL_INSTRUMENT=1 JACCL_POLL_INSTRUMENT_THRESHOLD_US=5000 \
    MLX_SIGNAL_PROBE=1 \
    ./start_cluster.sh

**Steps:**
1. User pastes command, cluster relaunches with diagnostics ON.
2. Hermes verifies env on runners (libmlx symbol check + ps -E grep on
   the 3 new vars).
3. Run 3-iter γ=2 c=1 100K bench (3 not 5 — we already have N=1 evidence
   of stall, just need diagnostic capture). ~20 min wall.
4. WHILE BENCH RUNS: prepare log-mining commands in scratch. Do not
   wait for it.
5. After bench finishes, mine ~/exo.log on both nodes:
   - Per-call all_reduce instrumentation: grep jaccl-instr
   - GPU signal encode-to-complete gaps: grep SIGNAL_PROBE_DONE, awk gap_us
   - Cross-reference stalled call_ids vs gap_us at those times
6. Capture wall, in_poll_us, iters, iters_with_cqes, max_single_poll_us
   for stalled calls. Compute decision-matrix ratios.

**Exit criteria and branching:**
- Branch A (driver-side, hypothesis 1): STOP plan. Not fixable without
  replacing Apple's RDMA stack. Document, fall back to γ=1 champion.
- Branch B (peer-CQE-late, hypothesis 2): Proceed to Task 3.
- Branch C (thread descheduled, hypothesis 3): Unexpected given Mach RT
  data. Re-investigate scheduler.
- No instrumentation hits during stalled iter: Different mechanism.
  Open Task 4 — instrument the SEND-side path.

**Cost:** ~25 min cluster (restart + bench), ~10 min orchestrator wall.

---

## Task 3: (Conditional on Task 2 = Branch B) Test inter-iter queue-depth hypothesis

**Hypothesis:** Eager-commit at the input-event boundary works for the
FIRST collective in an iter, but speculative MTP draft chains generate
new GPU work BETWEEN collectives within an iter. That work fills the
queue back up. Iter 0 fast because fresh; iter 1+ inherit pollution
from preceding iter tail.

**Files touched:** None for probes 3.1-3.3.

**Steps (executed in order, STOP at first one that triggers a clean run):**

Probe 3.1 — does EXO_MAX_ACTIVE_TASKS=1 stack with eager-commit?
Per diagnosis doc EXO_MAX_ACTIVE_TASKS=1 alone yields stable 16 t/s.
Combined with eager-commit:
- If 16 flat: eager-commit does not help once queue is already drained.
- If 32 flat: eager-commit is the lever, queue-depth was a red herring.
  STOP plan, declare champion.

Probe 3.2 — does iter 0 stay HIGH in a 10-iter run?
Bench 10 iters. ~64 min wall — skip if 3.1 already answered.

Probe 3.3 — γ=1 + eager-commit baseline.
Launch with EXO_SPECULATIVE_GAMMA=1. 5-iter bench. Compare to today's γ=2.

**Exit criteria and branching:**
- Any probe shows clean 5-iter at >=30 t/s: identify lever, declare
  champion, document, STOP plan.
- All probes confirm γ=2 has inherent inter-iter pollution: proceed
  to Task 4 — write a real mlx patch.

**Cost:** ~75 min cluster wall worst case (3.1 + 3.3 sequentially).

---

## Task 4: (Conditional on Task 3 failing all probes) Write the real fix

**Hypothesis:** [Whatever Task 3 narrowed it to.] Probably some variant
of "force GPU stream completion at iter boundaries, not just submit".

**Files touched:** Almost certainly mlx-lm (mtp_module.py or dsv4_mtp.py)
to inject the sync at iter boundaries Python-side. Possibly mlx C++ if
the sync primitive needs to be more efficient than mx.synchronize().

**This task is DELIBERATELY UNDERSPECIFIED.** Do not write it now —
write it AFTER Tasks 1-3 give us data. Premature specification is how
we ended up with eager-commit overclaim.

**Process when we reach this task:**
1. Re-read references/mtp-poll-stall-diagnosis.md "Candidate real fixes".
2. Pick the candidate whose mechanism matches Task 2/3 findings.
3. Write a tight brief for claude-code with named files + named test
   (5-iter γ=2 c=1 100K, mean >=30 t/s sigma<0.5).
4. Verify per orchestration rules.

**Exit criteria:** 5-iter bench, all iters >=30 t/s, sigma<0.5, repeated
2x on fresh restarts. Then merge, tag, memory update.

---

## Task 5: Documentation and memory update (do regardless of outcome)

1. Update references/mtp-poll-stall-diagnosis.md with new section noting
   eager-commit fixes iter-0 cold-start ONLY.
2. Update hot memory champion line.
3. If Task 4 succeeded: tag new champion in git, add to memory.
4. If eager-commit needs to be reverted out of mlx@main: do it explicitly
   and note why so we do not re-attempt the same patch.
5. Evaluate new skill pitfalls. Candidate pitfall #44:
   "3-iter clean for γ=2 is meaningless — need 5-iter minimum, and
   the 'lucky 3-iter' pattern WILL trigger premature champion claims."

---

## Stop conditions for the entire plan

Stop and consult user if:
1. Cluster fails to launch or returns errors at restart.
2. Any task produces NEW symptoms not predicted (SIGKILL, quality
   regression, runner crash mid-bench).
3. We have used > 4h of cluster time on the investigation without
   landing a fix. At some point γ=2 is not worth the engineering hours.
4. User says stop. (Every gate transition is a natural stop point.)

---

## What this plan deliberately does NOT include

- Premature patch writing. Tasks 2-3 must run BEFORE Task 4 is specified.
- Heroic re-instrumentation of mlx. The existing infrastructure should
  be enough.
- Generalization to γ=4 or other gamma values. Once γ=2 is fixed.
- Memory or kernel-level mitigations (e.g. usleep backoff) — diagnosis
  doc has already enumerated these as not-a-fix.

---

## Ready signal

To execute this plan:
1. User reviews this doc.
2. User confirms (or amends) the approach.
3. Hermes starts Task 1. (No auto-start.)
