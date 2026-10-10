# ROUND-Q1D-MICROBENCH.md — experts requant, Round 1: OFFLINE microbench (GATE 1)

Author: Phase-20 PM (delegation). Date: 2026-10-10 (CDT). Worktree `/private/tmp/phase20-campaign`.
Branch `deploy/phase20-campaign`. Round 1 of the experts-requant campaign (the offline microbench).

**Offline-first. NO live spend, NO encode spend, NO boots.** The whole round runs as an offline
single-node bench on `studio2` (the same pattern as the gate-0 N1 Path1 run). **0 boots** planned.
Production is not touched; the one live thing is the pre/post idle-guard + canary (read-only).

---

## 0. DECLARED BUDGET (fixed BEFORE spending)

- **Boots: 0.** No relaunch, no restart, no POST. The bench loads layer-20 expert modules in the
  node venv exactly like N1; it never touches the exo server. (The ≤2 relaunch reserve held by
  ROUND‑Q1C stays **unspent** and is NOT drawn here.)
- **Live spend: 0.** No generation requests. Owner traffic is live — every step that touches the
  cluster is idle-guarded (`phase20_guard.py idle`) and canary-checked before starting; the timed
  bench itself is offline (no engine touch), so it cannot perturb owner traffic beyond ordinary
  GPU sharing, which the guard bounds.
- **GPU-hour cap: ~2–4 GPU-h offline** on studio2 (idle-guarded). Declared plan: one offline bench
  process, ~4 arms × (R=4 primary + R=1 secondary + one prefill no-regress) over warmup+medians,
  plus a routing-distribution study and a memory-fit pass. Expected wall ≈ 30–75 min of actual
  node compute; cap 4 GPU-h hard, stop and write what is known if hit.
- **Encode spend: 0** (no re-encode; the arms are built in-memory from the EXL3 reconstruction).
- **Sampled-shard fetch:** 2–4 sampled layers only (~few GB), throttled, from the ORIGINAL
  (pre-EXL3) checkpoint IF it exists and IF auth is present — else NOTE and SKIP. Never the full
  >200 GB checkpoint.
- **Hard stop at the GATE 1 verdict even if data is partial** (write what is known). Genuine
  blocker (hw fault / cluster wedged / owner traffic saturating for hours) → STOP and report.

### Entry state (verified ~00:41 CDT 2026-10-10)
Production LIVE: exo `deploy/next19-dense @ 99e2966ee` + mlx-lm `689e4ea` (node `~/repos/exo` HEAD
`99e2966ee`, submodule `689e4ea`); runner env verbatim contains `DSV41_DENSE=affine6` +
`DSV41_DENSE_POLICY=layers.*.ffn.shared_experts.*=q8g64`; tag `known-good-dense-q68-20261009-231842`.
Cluster **idle** (`phase20_guard.py idle` rc=0, `state_active_tasks=0`). Canary **healthy**:
studio1 14.85 / studio2 14.86 t/s. Node `Adams-Mac-Studio-M4-2`, mlx `0.32.3.dev20260918+603f16eb7`.
Node free disk 206 GiB. Rollback refs: prev prod `fb4f9290b`/`16830e1`; in-place `DSV41_DENSE=exl3`.

---

## 1. GATE 1 — pre-registered (the decision this round exists for)

**GATE 1 = does a native-quantized expert path beat EXL3 by enough, measured as WALL time, to be
worth the quality work?**

- **PASS** ⇔ **any arm saves ≥30 % WALL time on the ×40-layer expert stack vs a SAME-SESSION EXL3
  baseline**, AND its per-arm correctness check is clean, AND its memory fit is arithmetically OK.
- **FAIL** ⇔ no arm clears it → **CLOSE the experts lead honestly with the numbers** (a valid,
  fully-acceptable outcome).

**Primary metric.** The ×40 expert-**module WALL** ms at the production verify shape (R=4, topk6):
`baseline_wall_40 = EXL3_whole_ms_per_layer × 40`, **re-measured in THIS session** (the N1 number
0.9398 ms/layer is reference only — thermal/clock drift means it must NOT be reused across
sessions). An arm PASSES the speed limb iff `1 − arm_wall_40 / baseline_wall_40 ≥ 0.30`.

**Anchor.** Gate 0 measured the expert **kernel** at 34.57 ms ×40 (0.8643/layer, 92 % of the
37.59 ms whole). 30 % of the kernel-only 34.57 ms ≈ **10.4 ms/round** (the number the brief cites);
30 % of the whole-module 37.59 ms = 11.28 ms/round. The gate is applied to the **whole-module
WALL** (the stricter, honest frame); the **kernel-only** ratio is reported alongside for
attribution. Passing the whole-module gate implies passing the kernel gate, so the strict frame is
used and both are tabulated.

**Why wall, not GPU-ms.** Net decode throughput gain from saving `s` ms of the 86.03 ms agentic
round, at acceptance-loss fraction `a`, is `(1−a)·86/(86−s)`. At the worst allowed acceptance loss
(3 %) a ≥8 % net gain needs `s ≥ 8.8 ms` ≈ 25.4 % of the 34.57 ms kernel — i.e. the 30 % gate is
the **zero-margin** line: below it the requant is not worth the quality risk. GPU-ms is recorded
for attribution only (dispatch/launch overhead is excluded from `gpu_time_ns()`).

## 2. FALSIFIER (pre-registered failure modes, so the round cannot be won by a broken arm)

- **Speed limb fails** on every arm after the q4g64 anchor → the native-quant transfer is not there.
  Per the brief: **if q4g64 fails the 30 % gate, CLOSE the lead honestly, no further arms.**
- **Correctness limb fails**: an arm that is fast but numerically wrong (per-arm output vs the
  dequantized-weight reference beyond tolerance) is **disqualified**, not a winner.
- **Shapes falsified**: if the real routing distribution is NOT "6 experts × 4 rows" (the N1
  [4,4,4,4,4,4] histogram), the bench shapes are **corrected to the measured unique-expert
  histogram** before the verdict; a verdict taken on stale shapes is void. Corroborated by a
  unique-expert sweep so the result cannot hinge on one distribution.
- **Memory-fit falsified**: if no passing arm fits per-rank resident size incl. KV at the max
  agentic context, the arm is not deployable → the verdict is bounded accordingly.

---

## 2b. PRE-REGISTERED REFINEMENTS (A1–A6, added BEFORE any arm result is seen)

Added after a design review, still ahead of the bench, so the bar cannot be retro-fitted.

- **A1 — Dense-precedent sign (resolved, pro-native).** From the raw ROUND-PRICING Q1 artifact
  (`raw/pricing/q1/p20_pricing_q1_dense_layer20.json`): native q4 g64 is **2.47× the EXL3 rate**
  (fair drop-in, Hadamards kept) / **3.38×** raw at m=4 (`rate_ratio_kmean` 2.467 / 3.381;
  `prod_m4_whole 0.945 ms` vs `nativehad_m4_B4 0.483 ms` / `native_m4_B4 0.374 ms`). I.e. on the
  dense slice native g64 is **~2.5× FASTER** than EXL3 (the trellis decode is ALU/issue-bound), not
  slower. ⇒ the experts premise is **plausible and pro-native**; a 30 % cut is not structurally
  doomed. (Q1's experts arm was PARKED, not measured — this round measures it.)
- **A2 — Cache-busting (a load-bearing fairness fix).** Single-layer repeated timing lets the
  active expert set sit in SLC between reps, which flatters native (bandwidth-bound) and not EXL3
  (ALU-bound) → false PASS risk. **Requirement:** each timed rep uses a *freshly drawn* hidden
  state → fresh gate routing (record the union of experts touched across reps); run every arm in
  BOTH a **warm** (back-to-back) and a **flushed** mode (evict SLC between reps by `mx.eval` of a
  ≥1 GiB unrelated scratch read, outside the timed region). The gating number is the **flushed**
  (cold) one.
- **A3 — Achieved-GB/s readout per arm** (bytes streamed ÷ wall). An anomalously high GB/s flags a
  warm cache or a construction bug. If a native arm lands under ~40 % of peak streaming GB/s,
  suspect the construction before concluding "EXL3 is faster".
- **A4 — Pre-registered unique-expert gating point = the CONSERVATIVE 24.** Native `gather_qmm`
  batches rows sharing an expert ⇒ native (unlike EXL3, which Q4 showed is slot-bound) IS
  unique-expert-sensitive, so the sweep is load-bearing. Report the full sweep {6,12,18,24}; the
  **headline gate number is taken at 24 unique** (conservative), with the realistic-histogram value
  (from D1) reported alongside. No re-picking after seeing results.
- **A5 — Module-boundary comparison + fair native build.** Both sides measured from hidden x +
  indices to the combined output y (gather, activation, down, weighted-sum included). The native
  drop-in reconstructs the **true W** (undo Hadamard/suh/svh) and runs plain `gather_qmm` (matches
  production `AffineProj`); Hadamards are a **Round-2 quality lever only**, not added to the speed
  arm. Use the fused gate+up `gather_qmm` from `mlx_lm/models/switch_layers.py` and `mx.compile`
  the activation so the native arm is not built unfairly slow.
- **A6 — Two correctness checks, in order.** (a) reconstructed-fp16-W path vs the EXL3 fused-kernel
  output — must be **near-exact** (validates reconstruction: missing svh/suh/transpose/sign bugs);
  only then (b) each native arm's output vs EXL3 output (relative error / cosine), plus vs the
  same-arm dequantized reference. A fast-but-wrong arm is disqualified, never a winner.

**Memory-fit is a real limb, not a formality (both nodes have only 128 GiB RAM).** Per-rank resident
(EXL3 ≈ 105.5 GB measured on-cluster, ROUND-Q1B) + KV at max agentic ctx + workspaces must fit;
native qN (≈4.25/5.25/6.25 bpw vs EXL3 ~2.9) grows the expert bytes ~1.5×, so q4–q6 may **not** all
fit. Computed exactly in D3.

---

## 3. THE ROUND (deliverables — filled in below as results land)

- **D0 — pre-registration** (this section + §0/§1/§2). ✅
- **D1 — routing distribution**: per-layer UNIQUE-EXPERT histogram from the agentic routing (real
  trace if one exists, else real-gate-on-correlated-rows, distribution-insensitive per Q4); the
  corrected bench shapes. → §4
- **D2 — arm-by-arm table**: wall ms, GPU ms, correctness, implied ms/round, implied net t/s at
  3 %/1 % acceptance loss, for EXL3 (same-session) + q4g64 → q5g64 → mixed(gu q4/dn q6) → q6g64 →
  q4g32(if needed), at R=4 (primary) + R=1 (secondary) + one prefill no-regress check. → §5
- **D3 — memory-fit table**: per-rank resident bytes per passing arm incl. KV at max agentic
  context. → §6
- **D4 — sampled-shard fetch status** (original pre-EXL3 checkpoint, 2–4 layers, throttled). → §7
- **D5 — GATE 1 VERDICT: PASS (which arm) / FAIL (close).** → §8
- **D6 — end state + budget accounting.** → §9

Raw scripts + JSON under `docs/benchmarks/phase20-throughput/raw/pricing/q1/q1d/`, committed+pushed.
`PERFORMANCE_HISTORY.md` on main gets a same-turn entry for the verdict.
