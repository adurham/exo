# γ=2 MTP Bistability — Next Steps After Eager-Commit Revert

> For Hermes: Read fully before starting any work. Decision points are
> explicit; do not auto-cascade across them.

**Date:** 2026-05-16 ~22:55 CDT
**Predecessor plan:** ./2026-05-16_2030-mtp-g2-iter1-bistability-investigation.md

**State after revert:**
- adurham/mlx@main rolled back to `facbed9a` (pre-eager-commit, has
  MLX_SIGNAL_PROBE diagnostic only).
- adurham/exo@main pinned to that mlx (commit 957eb637).
- Cluster restarted with γ=1 production config to verify champion is
  back to 30.5 t/s stable.

**Key learnings from today's investigation:**

1. Eager-commit (MLX_EAGER_COMMIT_BEFORE_CPU_COLLECTIVE=1) was tested
   thoroughly and FAILED:
   - Fixed iter-0 cold-start LOW lock on aged clusters
   - Did NOT fix γ=2 iter-1+ bistability
   - INTRODUCED bistability on γ=1 where there was none
   - Net production impact: NEGATIVE → reverted

2. Diagnostic data from Task 2 (JACCL_POLL_INSTRUMENT + MLX_SIGNAL_PROBE)
   confirmed:
   - JACCL poll-loop side is healthy (60% in_poll, max_single_poll < 30µs)
   - Mean GPU signal gap is fine (204 µs)
   - But: 0.78% of GPU signals have gap_us > 10ms, with max 75-107ms tail
   - This IS the stall mechanism. GPU command buffer execution latency
     on the producing side of the AllReduce.

3. Probe 3.1 (γ=2 + EAGER_COMMIT=1 + MAX_ACTIVE_TASKS=1) was stable at
   26.6 t/s σ=0.16 — a new operating point but worse than γ=1 30.5.

4. The diagnosis doc's already-tested mitigations (RT class, usleep
   backoff, Event::signal patches) all failed for the same root cause.

**The honest position:** γ=1 + EAGER_COMMIT_OFF + default everything =
30.5 t/s stable is the production answer. γ=2 has more all_reduce
calls per cycle AND larger payloads on verify, making it inherently
more sensitive to the GPU-buffer-execution-latency mechanism. We
have not found a way to make γ=2 better than γ=1 on this hardware.

---

## Decision Point 1: What to do next (user choice)

### Option A: Accept γ=1 30.5 t/s, end the investigation
- Pros: Cheapest. Production already runs on it.
- Cons: We leave +10% headroom (γ=2 ceiling at 32 t/s) on the table.
- Action: skip to Task 5 of the predecessor plan (docs + memory update).
- Cluster cost: 0 min.

### Option B: Pursue the diagnosis doc's untested candidates
The doc explicitly enumerated four candidate "real fixes" that were
NEVER attempted:
1. `mx.synchronize()` on input array stream before AllReduce::eval_cpu
   in `mlx/backend/cpu/distributed.cpp` — directly. Not via the
   eager-commit dependency-tracker layer that we just proved breaks
   γ=1. This is a single function change.
2. Restructure MTP cycle to coalesce verify+draft Metal work in
   `mlx_lm/models/deepseek_v4.py::dsv4_mtp.py` (multi-day, named in doc).
3. Change `mlx_distributed_send` to post on Python-thread instead
   of post-fence in mlx/distributed/jaccl/mesh.cpp (doc lists this
   as the SEND-side analogue of eager-commit, also untested).
4. Coalesce per-layer all_sums into a single batched all_sum across
   layers (multi-week, the "real" fix per the doc's "fix option scoping").

Candidates 1, 3 are 1-day experiments. 2, 4 are multi-day/week.

### Option C: Investigate γ=1 stability more carefully
Before option B, validate the γ=1 30.5 number with a 5+ iter
post-revert bench. If γ=1 + default + EAGER_COMMIT_OFF is also
bistable on some iters, the picture changes — maybe ALL configs
are bistable and we've been benchmarking only the lucky runs all
along.

(Note: this is the verification bench currently running, PID 61138.)

---

## Recommendation

**Start with Option C** (let the verification bench finish, ~32 min
total wall, ETA ~23:30 CDT) — it costs nothing additional and produces
the data we need to choose between A and B.

Then:
- If γ=1 is bulletproof (5/5 ≥30 t/s σ<0.5): go to A or B based on
  appetite for more engineering.
- If γ=1 also shows bistability: the whole picture changes. Open
  a new investigation into "all configurations have stochastic stalls
  but we've been benchmarking only short clean runs."

---

## Decision Point 2 (after verification bench finishes): A or B?

If you want to ship now and end the day: **A**.

If you want to take one more swing at γ=2: **B candidate 1**
(direct mx.synchronize in distributed.cpp). Single function change.
Either kills γ=2 stalls or doesn't. ~1 day of work split across
claude-code (the patch), Hermes (deploy + bench), user (paste & 
review). The risk profile is bounded: if it breaks γ=1 like
eager-commit did, we revert in 10 minutes.

---

## DO NOT

- Re-attempt eager-commit at the transforms.cpp dependency-tracker
  layer. The data is in: it breaks γ=1.
- Re-attempt Event::signal patches. SIGKILLs documented twice.
- Re-attempt usleep backoff. User explicitly rejected as mitigation.
- Re-attempt Mach RT class. Falsified by data.

---

## Task 5 (docs + memory regardless of A/B/C choice)

1. Update memory champion line: explicitly call out that eager-commit
   was tried and reverted, do not reattempt without new mechanism.
2. Update references/mtp-poll-stall-diagnosis.md with the
   2026-05-16 eager-commit results section (negative result).
3. Add skill pitfall: "3-iter clean for γ=2 is statistically
   meaningless — bistability shows up at iter 2-4 frequency.
   Use 5-iter minimum; treat 3-iter as smoke test only."
4. Save the Task 2 JACCL_POLL_INSTRUMENT + MLX_SIGNAL_PROBE data
   as `references/mtp-stall-2026-05-16-signal-probe-data.md` so we
   don't have to re-capture it on the next investigation.

---

## Ready signal

After γ=1 verification bench completes (or if user wants to skip ahead):
1. Hermes reports the 5-iter γ=1 result.
2. User picks A, B (candidate 1), or stops the investigation.
3. Hermes executes Task 5 docs always, plus B-candidate-1 if chosen.
