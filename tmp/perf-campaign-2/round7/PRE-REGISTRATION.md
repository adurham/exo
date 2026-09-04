# CAMPAIGN 2, ROUND 7 — PRE-REGISTRATION (written BEFORE any measurement)

**Date:** 2026-09-04 | **PM:** round-7 delegation
**Scope:** I2, the two LIVE c=2-tax knobs. Task A `EXO_BATCHED_PREFILL_RENDEZVOUS_MS` 200→0;
Task B `MLX_STEEL_BATCH_INVARIANT` 1→0; Task C stale-comment housekeeping.

This file is committed before a single measurement is taken. Every band, stopping rule, and
declared deviation below is fixed as of this commit.

---

## 0. Starting state — VERIFIED on the real runner PID before pre-registering

`ssh 192.168.86.201 "ps eww 50187"` (the real runner, not the SCREEN/login wrappers):

```
EXO_DSV4_BATCHED_PREFILL=1          EXO_BATCHED_PREFILL_RENDEZVOUS_MS=200
MLX_STEEL_BATCH_INVARIANT=1         MLX_GEMV_BATCH_INVARIANT=1
EXO_DSV4_VERIFY_BATCH=1             EXO_DSV4_VERIFY_BATCH_MIN_CTX=8192
EXO_DSV4_VERIFY_ROWSEQ_VEC=1        EXO_SPECULATIVE_GAMMA=3   EXO_DSV4_MTP=1
```

`/v1/models` → HTTP 200. This is the round-6 end-state boot (γ=3), unchanged.

**The 200ms sleep is genuinely being paid:** `EXO_DSV4_BATCHED_PREFILL=1` is the outer gate on
the rendezvous block (`runner.py:580`), and it is 1 in production
(`start_cluster.sh:129`, propagated to the runner env at `start_cluster.sh:1696-1697`).
Had that gate been 0, Task A would have been unmeasurable and this round would have said so.

---

## 1. Instruments — FIXED, no new harness

| quantity | instrument | why |
|---|---|---|
| **decode throughput** | server `stats.generation_tps` via `bench/long_decode_probe.py` | round-6 ruler, `perf_counter`-timed inside the generator; client rate is cross-check only |
| **TTFT** | `prefill_s` from the SAME probe = `t_first - t_start`, client wall clock (`long_decode_probe.py:145`) | the rendezvous is a deterministic `queue.get(timeout=...)` sleep; client wall is the correct instrument, and this field already exists |
| **byte identity** | `tmp/perf-campaign-2/round5/identity_gate.py` (`--capture` / `--compare`) | already built and smoke-tested 5/5 in round 5 |
| **tier-1 quality** | `bench/ab_probe_tier1.py` | unchanged |

**Declared deviation (the only permitted code edit this round):** `long_decode_probe.py` gains an
optional `--run-id` argument that makes `build_prompt`'s uuid salt deterministic. Byte-identity at
~89K is impossible otherwise — the probe currently salts every prompt with a fresh uuid4, so two
runs are never comparable. This adds **no timing arithmetic** and changes no metric; when
`--run-id` is omitted the behavior is bit-for-bit the current behavior. Any arm measured for
throughput still uses fresh salts (cache-cold), exactly as in round 6.

---

## 2. TASK A — `EXO_BATCHED_PREFILL_RENDEZVOUS_MS` 200 → 0

### 2.1 Read-through gate (must pass before any boot)
The four questions of the brief must be answered from source with `path:line` cites, and
question (d) — can window=0 make rank 0 and rank 1 see different task sets at c=1 — must come
back **clean**. If (d) is not clean, Task A is abandoned with no boot spent.

### 2.2 Measurement
Cache-cold ~2K prompt (`long_decode_probe.py 2000`, fresh uuid salt per rep, `prefix_cache_hit`
must read `none` on every rep), **n=5** per arm.
- Arm P (production): `RENDEZVOUS_MS=200` — measured on the current boot, no relaunch.
- Arm Z: `RENDEZVOUS_MS=0` — one relaunch.

### 2.3 Pre-registered band (verbatim from the brief)
> TTFT delta in **[−250, −150] ms** with clean logs → **SHIP**. Delta outside that band →
> something else is in the path; report, do not ship.

**Decision statistic:** `median(prefill_s | Z) − median(prefill_s | P)`, in ms.
**Secondary diagnostic, declared now so it cannot be invented later:** the n=5 ranges of the two
arms are also reported. If the medians land in band but the ranges **overlap**, the delta is not
cleanly resolved by this design; the brief's band still governs the ship decision, but the overlap
is reported as an explicit caveat rather than buried.

### 2.4 Clean-logs requirement
20 sequential requests on the Z boot (mix of short and ~89K), requiring: zero errors, zero rank
disagreement / task-set mismatch in the runner logs on **both** nodes, all responses coherent.
Any rank-disagreement evidence → **HOLD regardless of the TTFT number.**

---

## 3. TASK B — `MLX_STEEL_BATCH_INVARIANT` 1 → 0

### 3.1 Read-through gate (must pass before any boot)
(i) which kernels the flag changes and whether any of them is on the c=1 decode path at M=1
(draft) / M=4 (verify at γ=3); (ii) whether `EXO_DSV4_VERIFY_ROWSEQ_VEC` is LIVE at ~89K under
`EXO_DSV4_VERIFY_BATCH=1`. Both with cites.

### 3.2 Arms — bracketed A/0/A on the round-6 ruler
`long_decode_probe.py 79000 --max-tokens 1200`, 1 discarded warmup + **n=3** measured, on each of:

| arm | boot | steel-BI | rendezvous |
|---|---|---|---|
| **A** (opening bracket) | boot 2 | 1 | 0 |
| **0** (candidate) | boot 3 | 0 | 0 |
| **B** (closing bracket) | boot 4 | 1 | 0 |

Both `=1` boots are mandatory; **their spread is the bar.** Rendezvous is held at 0 across all
three so the bracket differs in steel-BI only. (The rendezvous window is a pre-prefill sleep and
cannot touch `generation_tps`, which is timed inside the generator — but holding it constant costs
nothing and removes the question.)

Per-rep validity, identical to round 6: `decode_sample_trustworthy == true`,
`prompt_tokens >= 85000`, `finish_reason == "length"`, `prefix_cache_hit == "none"`.

### 3.3 Pre-registered throughput band (verbatim from the brief)
> Range entirely above both `=1` ranges by more than the A-vs-B spread AND all quality gates
> clean → **SHIP**. Overlap → **hold**. Quality failure → **hold**.

Formally, with `bi1_union_max = max(A ∪ B)` and `bi1_spread = |median(A) − median(B)|`:
SHIP requires `min(arm 0) > bi1_union_max` **and** `min(arm 0) − bi1_union_max > bi1_spread`.

### 3.4 THE QUALITY GATE IS THE POINT OF THIS KNOB — and it runs FIRST
This knob exists for batch-invariance / bit-exactness. Speed is the secondary question.

**Pre-registered ordering (a stopping rule, not a preference): on boot 3 the byte-identity gate
runs BEFORE the three decode reps.** If identity fails, Task B stops immediately: the decode reps
on that boot are not run, the closing-bracket boot 4 is **not** spent, and the finding is recorded
as "the knob is load-bearing for correctness at c=1." Measuring the speed of a configuration that
has already failed its correctness gate would burn a boot to decorate a decision already made.

Gate contents (temp=0, vs the steel-BI=1 reference captured on boot 2):
1. byte-identity on a **<8192** prompt,
2. byte-identity on a **~89K** prompt,
3. byte-identity on a **tool-call** prompt,
4. needle exact-match @89K,
5. `bench/ab_probe_tier1.py` **7/7**.

### 3.5 The prediction this round is actually testing (recorded now, scored later)
The read-through says `rowseq_vec`'s docstring requires `MLX_STEEL_BATCH_INVARIANT=1` for per-row
bitexactness, and that `ROWSEQ_VEC` is **dead at ≥8192 ctx** (short-circuited by
`not (_VERIFY_BATCH and _VERIFY_BATCH_CTX["active"])`) but **live below 8192**. Production runs
`EXO_DSV4_VERIFY_ROWSEQ_VEC=1`.

**Therefore I predict, before running it: byte-identity FAILS on the <8192 prompt and PASSES at
~89K.** If that is what happens, the honest reading is that at the production configuration the
knob is correctness-load-bearing at c=1 and must stay =1 — a HOLD, not a speed result.

**Pre-registered handling of that exact outcome (so it cannot be re-litigated after the fact):**
a <8192 identity failure with a 89K pass is a **HOLD at the production config**. This round will
**not** expand scope to test `steel-BI=0 + ROWSEQ_VEC=0` — that is a different configuration with
its own quality surface and belongs to a future round. It will be recorded as the named follow-up.

### 3.6 Scoring
Whether prediction 3.5 is right or wrong is recorded plainly in the report, in the round-6 style.

---

## 4. TASK C — housekeeping
Comment-only corrections to `start_cluster.sh` for FUSED_MOE / COMPILE_FFN / COMPILE_LAYER (wiring
removed 2026-06-18), `EXO_DSV4_MTP_C2_MAX_CTX` (gate removed 2026-06-24), `FENCE_EVERY_N_LAYERS`
(round 4 claimed zero readers), and a steel-BI framing note. Knobs are NOT deleted. Executable
behavior must be byte-identical; verified by `bash -n` and by checking every changed line is a
comment.

**Pre-registered honesty rule:** each correction is written only if the PM re-verifies the claim
against live source first. If a claim from a prior round does not hold, the comment is left
untouched and the prior round's error is reported instead of propagated.

---

## 5. Constraints (restated, binding)
1. Round-6 ruler only for decode (`stats.generation_tps`); client wall-clock for TTFT only.
2. Bit-equivalence is a hard gate on Task B. **Ranges, never bare means.**
3. **No pushes.** Local commits only. Never `git add -A`, never `git stash`.
4. Cluster HEALTHY at the end on whatever config ships, verified on real runner PIDs (`ps eww`)
   + API + a coherent temp=0 completion. Round 6 left the completion check unclosed because it was
   blocked; this round closes it.
5. Time-box 4h. **Task A alone is a valid round** — if budget runs out, Task A ships and Task B is
   reported as unmeasured rather than half-measured.

## 6. Boot budget
4 boots total: the current production boot (arm P + identity reference work), then 3 relaunches
(Z/A, 0, B). Authorized: Task A 1–2, Task B 3.
