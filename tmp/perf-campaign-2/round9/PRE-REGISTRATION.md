# CAMPAIGN 2, ROUND 9 — PRE-REGISTRATION

**Committed BEFORE any round-9 cluster measurement was taken.** Task 1 (code read) is complete
and its numeric prediction is fixed here. Task 2 has not been run.

---

## 1. Task 1 result — the code-derived prediction (fixed, not revisable)

Full analysis with line cites: `TASK1-CODE-ANALYSIS.md`. PM-verified independently (the grep for
every consumer, the `runner.py:566-625` block, and the R7 JSON decomposition were all re-run by
the PM, not taken on the worker's word).

**Findings, with cites:**

- **Exactly one functional consumer.** Repo-wide grep for `EXO_BATCHED_PREFILL_RENDEZVOUS_MS`
  returns: definition `constants.py:138-140`; import `runner.py:16`; gate `runner.py:580`;
  deadline arm `runner.py:582`; log string only `runner.py:624`; launcher `start_cluster.sh:136`
  + propagation `:1697`; a non-consuming fingerprint-registry entry
  `bench/trusted_measurement/fingerprint.py:95`; a comment at `worker/main.py:377`.
  **No second read site. The window is not paid twice per request.**

- **Paid once, bounded by construction.** The deadline is computed ONCE
  (`runner.py:581-583`) and the drain loop (`runner.py:594-620`) recomputes `remaining` against
  that fixed deadline every tick. Total blocked time is therefore ≤ W by construction regardless
  of how many times the loop iterates. At c=1 the first `self._work_queue.get(timeout=remaining)`
  raises `queue.Empty` at the deadline and `break`s — one tick, full window, no more.

- **`> 0` is a clean skip, not a sentinel** (`runner.py:580`) — confirms R7 §1.

- **Not serialized across ranks.** Each rank arms its own deadline from its own non-blocking
  dispatch (`worker/main.py:383`); the ranks first meet at the `agree_on_tasks` all_gather
  barrier (`utils_mlx.py:2284-2338`). Joint start = `max(t0,t1) + W`, i.e. **W, not 2W**.

- **Nothing serialized behind it; no downstream quantum is missed.** The only 100 ms poll tick in
  the region (`worker/main.py:195`, planner) is strictly UPSTREAM of the window. Runner→client is
  push, not poll, end to end. The engine-side batched gate (`batch_generator.py:757`) has no
  sleep/timeout at all.

- **Decode/stream adds no term that scales with prefill start time** (anchors captured after
  prefill: `generate.py:827`, `batch_generate.py:2776/2782`).

### PREDICTED DELTA — pre-registered, fixed now

> **The code predicts the RV=200 → RV=0 short-prompt TTFT delta is exactly −200 ms.**

There is no code path that produces 400 or 480. **The R7 −480 ms is therefore NOT explained by
the rendezvous alone**, and this round pre-commits to that position before measuring.

### What the other ~280 ms of R7's 480 most likely was (PM-verified decomposition)

The PM re-ran R7's own result JSONs (`round7/results/{Z,P2}_short_r*.json`) and split TTFT using
the **server-side** `prompt_tps`, which is timed *inside* `prefill()` and therefore structurally
excludes the window:

| bucket | Z (RV=0) | P2 (RV=200) | delta |
|---|---|---|---|
| TTFT (client wall) | 1510 | 1990 | **+480** |
| in-`prefill()` compute `(prompt_tokens−1)/prompt_tps` | 1027 | 1296 | **+269** |
| **residual** (the only bucket the window can live in) | **441** | **726** | **+285** |

Server `prompt_tps` medians: Z **221.1** vs P2 **176.0**, on effectively identical prompt sizes
(220–234 tokens both arms, all 20 reps `prefix_cache_hit=none`).

**A 25% difference in in-`prefill()` compute throughput cannot be caused by a scheduling sleep that
completes before `prefill()` is entered.** That ~270 ms is a boot / warm-state artifact — exactly
the confound R7 §8.2 named when it demanded a paired-boot design. The residual bucket moved
**+285 ms**, consistent with the 200 ms prediction plus ordinary spread (Z's own residual spans
412–577 ms across its 10 reps).

**Status: SPECULATION as to the specific cause** of the prefill-compute difference (candidates:
idle `mx.clear_cache()` at `runner.py:848-856`; `_save_prefix_cache`/`_evict_if_needed`). It is
NOT speculation that the difference exists and that the window cannot produce it.

---

## 2. Task 2 — measurement design (fixed before execution)

**Four boots, alternating, in this order:** RV=200 (A) → RV=0 (first) → RV=200 (B) → RV=0 (second).

Per boot, in this order:
1. Launch; wait for READY.
2. **Wait ≥ 5 minutes idle after READY before the first rep.** (Kills the warmup confound.)
3. Verify RV on the **real runner PIDs** via `ps eww` on **both** nodes. A boot whose PIDs do not
   show the intended value is void and must be relaunched.
4. 5× 2K-prompt reps (matched ordering, exactly as R7 did — these exist to match warm state, they
   are not the decision statistic).
5. n ≥ 10 short (~20-token) reps.

**Instrument:** the R7 A2 short-prompt harness, reused unmodified —
`bench/long_decode_probe.py 20 --max-tokens 16`, fresh uuid salt per rep. **No new harness.**

**TTFT** = client wall time to first streamed token. `stats.generation_tps` is irrelevant here;
this is latency. `prompt_tokens` printed per rep. Every rep must read `prefix_cache_hit = none`.

**Reported per boot:** short-prompt **median + full range**. Ranges, never bare means.

**Boot-variance bar for this instrument** = the spread between the two RV=200 boots (A vs B).

### Pre-registered decision band — applied VERBATIM, no substitution

**SHIP** if **all three** hold:
1. Both RV=0 boots' ranges lie **entirely below** both RV=200 boots' ranges; **AND**
2. the median gap **exceeds** the RV=200 A-vs-B spread; **AND**
3. the gap is consistent with the Task 1 code-derived prediction **within ±25%** — i.e. the
   measured median gap falls in **[150 ms, 250 ms]** (200 ± 25%).

**HOLD + report as a finding** if the gap is real but does NOT match the prediction. In that case
the round must **name** what else is in the path rather than shipping an unexplained number.

**CLOSE** (effect inside boot variance) if the ranges overlap.

### Secondary diagnostic — declared now, DOES NOT govern the decision

The residual bucket `TTFT − (prompt_tokens−1)/prompt_tps × 1000` will also be reported per boot,
because Task 1 shows it is the only bucket the window can occupy. It is **reported for
explanation only**. The ship/hold call is made on the raw short-prompt TTFT band above, exactly as
written. This is stated in advance specifically so the residual cannot be swapped in later as a
more favourable statistic.

### Independent veto (must also pass)
Re-run the R7 clean-logs veto on **both** RV=0 boots: mixed short + ~89K requests, **zero errors,
zero rank-disagreement / task-set-mismatch evidence** in both nodes' runner logs and the launch log.

---

## 3. Task 3 — quality gate (hard, non-negotiable)

temp=0 **byte-identity** of 3 prompts (short, 2K, ~89K), RV=0 vs RV=200 — must be **IDENTICAL**.
Deterministic prompts via `--run-id` (no uuid salt). **Any divergence disqualifies the change
regardless of the TTFT number**, because it would mean the window affects task-set agreement after
all.

---

## 4. Ride-alongs (free on these boots; must not cost the round)

- **I15** — set the boot-gated launch-count probe vars on **one** RV=0 boot; report launches/step.
  Pre-registered: **>500 → scope a COMPILE_LAYER rebuild (do NOT build)**; **<200 → close I15**.
  (200–500 → report, no action pre-registered.)
- **I12** — on one ~89K request, confirm at runtime that the serial driver ran AND the
  tiled-SDPA / exact-topk log markers fired. R8's audit was static reachability only.

---

## 5. Constraints acknowledged

Local commits only, **no pushes**. Never `git add -A`, never `git stash`. Relaunches are
pre-authorized. Cluster must end **HEALTHY on the final config**, PM-verified on real PIDs + API +
a coherent temp=0 completion. Time-box 4h.
