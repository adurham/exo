# ROUND-Q1C-MEASURE.md — post-dense measurement pass + experts GATE 0

Author: Phase-20 PM (delegation). Date: 2026-10-10 (CDT). Worktree `/private/tmp/phase20-campaign`.
Branch `deploy/phase20-campaign`.

This round is **GATE 0 for the experts requant campaign**: it MEASURES the current production round
composition on the SHIPPED dense build so the experts decision rests on real numbers, not the stale
pre-dense ones. **NO encode/requant spend. NO new code shipped.** Measurement + one decision only.

---

## 0. DECLARED BUDGET (fixed BEFORE spending, per the brief)

- **GPU-hour cap: ~3–4 GPU-h, idle-guarded.** Every chunk idle-guarded (`phase20_guard.py idle`) and
  canary-checked before/after; canary after any boot.
- **Relaunches: ≤2 TOTAL** (1 measurement boot + 1 restore) — **only if the mechanism needs them.**
  A **third** needs a documented reason.
- **Preference: OFFLINE (no relaunch).** Per the Q5 GO memo, Path 1 (per-op-class GPU time via
  `MLX_GPU_TIME=1` → `mx.metal.gpu_time_ns()`) runs **without touching the live server** — the Q4/Q1
  offline microbench pattern already loads the real layer-20 modules in the node venv. If that suffices
  for the expert op-class split, **0 boots are spent** and the ≤2 relaunch reserve is held unspent
  (declared as such at the end). A measurement boot is taken ONLY if the offline split cannot resolve
  the kernel-vs-overhead question.
- **No encode spend. No ship.** Production stays `deploy/next19-dense @ 99e2966ee` + mlx-lm `689e4ea`,
  env `DSV41_DENSE=affine6` + `DSV41_DENSE_POLICY=layers.*.ffn.shared_experts.*=q8g64`, untouched.
- **Hard stop at the GATE 0 verdict even if data is partial** (write what is known). Genuine blocker
  (hw fault / cluster wedged / owner traffic saturating for hours) → STOP and report.

### Entry state (verify at entry)
Production LIVE: exo `deploy/next19-dense @ 99e2966ee` + mlx-lm `689e4ea`; baked runner env
`DSV41_DENSE=affine6` + `DSV41_DENSE_POLICY=layers.*.ffn.shared_experts.*=q8g64`; tag
`known-good-dense-q68-20261009-231842`. Perf: benign 20K 81.3 ms/45.2 t/s; agentic 91K 86.03 ms/34.3 t/s.
Rollback refs: prev prod `fb4f9290b`/`16830e1`; in-place `DSV41_DENSE=exl3`.

---

## 1. GATE 0 — pre-registered falsifier (THE decision this round exists for)

From the measured **N1** op-class split of expert decode time:

- **CLOSE the experts lead** if EITHER
  - **(a)** expert **kernel** time is **under ~25 ms** of the **~86 ms** agentic round, OR

    (i.e. the expert GEMM kernels are not a big-enough slice of the round to be worth a requant campaign,
    even at the Q1-priced native-qN kernel speedup), 
  - **(b)** **non-kernel overhead dominates** the expert time: `gather/scatter/routing + all-sum/RDMA +
    unattributed  >>  kernel`. (Requant changes the tensor FORMAT → the kernel; it does NOT change routing,
    index gather/scatter, or the TP all-sum. If those dominate, a format change cannot move the block.)

- **PROCEED with the experts requant campaign** ONLY if BOTH
  - expert **kernel** time is **~25 ms+**, AND
  - the expert block is **kernel-dominated** (kernel ≳ non-kernel overhead).

PROCEED only *opens* the campaign (its own offline microbench + quality pre-screen + live A/B is a LATER
round — **nothing of that happens now**). The verdict is written explicitly as **PROCEED / CLOSE with the
numbers**.

**Why this is the right gate (mechanism).** Q1 priced the DENSE slice: native qN g64 is **2.47×** the fused
EXL3 trellis kernel at m=4 (`ROUND-PRICING.md` §Q1) — the win is **ALU/issue-bound** (the k=7 SWAR trellis
decode), *not* bandwidth (q4 streams 62 MB vs EXL3's 66.7 MB yet is 2.5× faster). The experts arm asks the
same question of the **routed experts** (384/layer, EXL3 2.9 bpw trellis): a native-qN re-encode buys the
same kernel speedup **iff** the expert block is kernel-dominated. Q4 already showed the *whole* expert
block at the verify shape is **1.077 ms/layer × 40 = 43.08 ms** standalone (vs the ~37 ms unattributed,
1.16×). Gate 0 must now split that block into **kernel vs overhead** — the number Q4 did not produce.

**Falsifier (fires → CLOSE):** any of — expert kernel ms < 25; OR `(gather/scatter/routing + all-sum/RDMA +
unattributed) > kernel`; OR the offline harness cannot separate the fused kernel's internal gather/scatter
(→ the "kernel" category is not a clean requant target → CLOSE by (b)).

---

## 2. THE ROUND (deliverables)

- **N1 — Path1 op-class GPU timing on current prod (core).** Per-op-class GPU time for (a) agentic 91K
  decode round, (b) benign 20K decode round, (c) one prefill chunk. Split expert time into
  `expert qmm kernel / gather-scatter-routing / all-sum+R DMA / other`; record the **UNATTRIBUTED remainder
  explicitly**. Best-effort per-expert m-distribution of the gather ops (from routing info if cheap).
- **N2 — Prefill ladder re-measure on the dense build** (free TTFT data): delta-ladder at 20K/50K/110K,
  compare vs the recorded EXL3-dense ceiling **272.6 / 249.6 / 242.4 rows/s**.
- **N3 — Engram check (offline, no GPU):** is the `engram` tensor group (8 % of dense bytes per the census
  attn 74 / shared 18 / engram 8) covered by `DSV41_DENSE=affine6` or does it remain EXL3? FLAG ONLY if it
  remains EXL3 (joins a future experts re-encode list as a ~8 % free extra; **do not encode now**).
- **N4 — Acceptance baseline:** ship-smoke acceptance (agentic **1.9395** / benign **2.6723** median
  mean-accepted); recorded as the baseline for future comparison; no new run unless a cheap spot-check.
- **GATE 0 VERDICT: PROCEED / CLOSE with numbers.** → §5 below.

*(Sections N1–N4 + verdict + end-state + budget accounting are filled in as the measurements land;
this pre-registration (budget §0, gate §1, falsifier) is committed BEFORE any spend.)*

---

## 3. N1 — op-class GPU timing (FILLED)

_(pending measurement)_

## 4. N2 — prefill ladder on the dense build (FILLED)

_(pending measurement)_

## 5. N3 — engram coverage (FILLED)

_(pending measurement)_

## 6. N4 — acceptance baseline (FILLED)

Ship-smoke acceptance (from `~/.hermes/cache/scratch/next19-ship-logs/smoke/`):
- **agentic**: mean-accepted median **1.9395** (all 1.952 / 1.927); ms/round median **86.03** (85.96 / 86.09);
  decode_tps median **34.299** at depth 91 043.
- **benign**: mean-accepted median **2.6723**; ms/round median **81.30**; decode_tps **45.2** at depth 20 000.
→ baseline recorded for future comparison; no new run.

## 7. GATE 0 VERDICT (FILLED)

_(pending measurement)_

## 8. END STATE / BUDGET ACCOUNTING (FILLED)

_(pending measurement)_
