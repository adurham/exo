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

## 3. THE ROUND (deliverables — all landed)

- **D0 — pre-registration** (§0/§1/§2/§2b). ✅
- **D1 — routing distribution** → §4
- **D2 — arm-by-arm table** → §5
- **D3 — memory-fit table** → §6
- **D4 — sampled-shard fetch status** → §7
- **D5 — GATE 1 VERDICT** → §8
- **D6 — end state + budget accounting** → §9

Scripts: `bench/p20_q1d_native_experts.py` (the bench), `bench/p20_q1d_routing_hist.py` (routing).
Raw under `docs/benchmarks/phase20-throughput/raw/pricing/q1/q1d/`. Committed+pushed.
`PERFORMANCE_HISTORY.md` on main gets a same-turn entry (§8 verdict).

---

## 4. D1 — Routing distribution (the real histogram; the "6×4" claim is FALSE)

**A real captured trace exists and was used — no live capture, no boot.** The phase-10 full 40-layer
routing trace of the real EXL3 checkpoint, `docs/benchmarks/phase10-planb-gate-2026-09-28/raw/
p30_exl3_trace_clamp.json` (sha256 `b4f1b142…`, 531 tokens × 40 layers = 21 240 real gate decisions,
clamped `silu_clamp` = production routing). Sanity: the numpy gate used here is verified **identical**
to the production `mlx_lm/models/deepseek_v41/moe.py::Gate` (set-match + exact for layers 0/20/39).

| R | unique experts (mean / med) | p5 / p95 | min / max | m mean | m-dist (1/2/3/4) |
|---|---:|---:|---:|---:|---|
| 1 | **6.00** / 6 | 6 / 6 | 6 / 6 | 1.00 | 1.00 / – / – / – |
| **4 (verify)** | **16.25** / 16 | **12 / 21** | 7 / 24 | **1.48** | **0.703 / 0.172 / 0.071 / 0.054** |
| 8 | 26.19 / 26 | 17 / 36 | 10 / 46 | 1.83 | … |

R=4 unique-expert histogram (count : #groups of 21 120): `7:2 8:26 9:84 10:269 11:644 12:1121 13:1740
14:2231 15:2535 16:2675 17:2644 18:2285 19:1778 20:1368 21:933 22:503 23:226 24:56`. Per-layer R=4 mean
unique: **13.70 (L6) … 16.05 … 20.84 (L0)** — *no* layer is anywhere near 6.

- **The N1 "[4,4,4,4,4,4] = 6 experts × 4 rows" claim is FALSE on real routing: 0 / 21 120 groups
  (0 %); zero groups had unique == 6.** It is a degenerate near-zero-perturbation artifact of the
  synthetic model: `[4,4,4,4,4,4]` occurs in 83.8 % of groups at corr=0.01, **55.9 % at corr=0.03
  (exactly q1c's setting)**, 14.8 % at 0.1, 0.16 % at 0.3, 0 % for corr ≥ 0.5. Real adjacent-token
  hidden states match the real 16-unique count only at the equivalent corr ≈ 1.0. (Q4's "14 unique"
  is also invalid — its gate used `sigmoid(x@Wᵀ+b)`, not the real `sqrt(softplus(x@Wᵀ/temp))+bias`.)
- **The 6×4 assumption understated the verify gather width by ≈2.7×.** Fable's correction #1 is
  **confirmed**: the real distribution is ~16 unique experts at m ≈ 1.48 (70 % m=1 / 17 % m=2 /
  7 % m=3 / 5 % m=4), working band 12–21 — not 6.

**Corrected shapes (baked into §5).** Realistic verify = **16 unique experts, m ≈ 1.48**; conservative
upper bound = **24 unique** (uniform routing = the trace max). Per A4, the headline gate number is
taken at the **conservative 24**, with the full sweep {6,12,18,24} reported. (Cost note: for EXL3 the
R=4 module is slot-bound, so the shape change moves the *gather width*, not the EXL3 GEMM ms; for
native `gather_qmm` — which batches rows sharing an expert — the unique count does move the ms, hence
the sweep is load-bearing.)

---

## 5. D2 — Arm-by-arm table (R=4 verify, topk6, u24 FLUSHED = the GATE condition)

Same-session, cache-busted (fresh routing per rep; 1.5 GiB SLC flush between reps, outside the timed
region), median of 14 reps, 3 seeds. Node `Adams-Mac-Studio-M4-2`, layer 20, per-rank H=1152, D=5120,
E=384, k_trellis=2. Script `bench/p20_q1d_native_experts.py`.

| arm | wall ms/layer | GPU ms/layer | ×40 wall ms | % saved vs EXL3 | GB/s | cosine (A6b) |
|---|---:|---:|---:|---:|---:|---:|
| **EXL3 baseline @u24** | **1.2243** | 0.9063 | **48.97** | — | — | ref |
| **native q4g64** (ANCHOR, fused gu, sorted) | **0.8250** | **0.5116** | **33.00** | **+32.6 % PASS** | 289.6 | 0.9881 |
| native q4g32 | 0.8265 | 0.5590 | 33.06 | +32.5 % PASS | 321.1 | 0.9905 |
| native mixed (gate_up q4g64 + down q6g64) | 0.8928 | 0.5751 | 35.71 | +27.1 % | 307.2 | 0.9919 |
| native q5g64 | 0.9098 | 0.6116 | 36.39 | +25.7 % | 320.9 | 0.9972 |
| native q6g64 | 0.9824 | 0.7152 | 39.30 | +19.8 % | 351.2 | 0.9993 |
| EXL3 baseline @natural (flush) | 1.3959 | 0.8867 | 55.84 | — | — | ref |

- **Only the two q4 arms clear the 30 % wall gate**; q5g64 (25.7 %), mixed (27.1 %) and q6g64 (19.8 %)
  miss. **The speed lever is the 4-bit packed stream, not precision headroom** — q4g32 buys nothing
  over q4g64 (more scale bytes).
- **Unique-expert sweep (q4g64, flushed wall ms/layer)** {6,12,18,24} = **0.5860 / 0.6604 / 0.7434 /
  0.8250** (GPU 0.3257/0.3394/0.4256/0.5116; GB/s 101.9/180.9/241.0/289.6) — monotonic; the 24-unique
  point is the slowest, i.e. the conservative headline. At the *realistic* 16 unique the native wall
  interpolates to ≈0.71 ms/layer ⇒ the win is **larger** at real routing, so the conservative gate is safe.
- **Fused gate+up beats unfused** (native R4-natural flush 0.7803 fused vs 0.8134 unfused); **sorted
  beats unsorted** (0.7792 vs 0.9181) → sorted chosen.
- **R=1 secondary** (fallback/draft): EXL3 0.5468 wall. **Prefill S=2048 no-regress CHECK** (not a
  gate): native q4g64 85.2 ms wall / 35.6 GPU vs EXL3 99.3 / 56.9 → **no regress**.
- **Kernel-only GPU ratio native/EXL3 = 0.564 (native 44 % faster on GPU).** The wall win (32.6 %) is
  smaller than the GPU win (44 %) because both arms carry ~0.31 ms/layer of Python dispatch the
  format change cannot remove (EXL3 wall/GPU = 1.35×, native = 1.61×).

**Corroboration of the baseline**: the same-session EXL3 GPU time (0.9063 ms/layer u24) matches N1's
recorded 0.9398, so the load-bearing wall baseline (1.2243) is credible.

**Correctness (A6).**
- **A6a (reconstruction — must be near-exact):** reconstructed-true-fp16-W path vs the EXL3 fused
  kernel on a 24-expert subset, 3 seeds → `cosine_min = 0.99999955`, `rel_l2_max = 9.5e-4`,
  `max_abs_err = 5.86e-3` → **near-exact, PASS.** This validates `[out,in]` orientation, suh/svh sign
  undo, transpose and `silu_clamp`; a transposed or sign-stripped projection would fail loudly here.
- **A6b (arm vs EXL3, float64):** q4g64 0.9881 / 0.155; q4g32 0.9905 / 0.138; mixed 0.9919 / 0.128;
  q5g64 0.9972 / 0.074; q6g64 0.9993 / 0.037. All legitimate (fidelity to an already-2.9 bpw-lossy
  tensor, not to the original model). **No fast-but-wrong arm.**

---

## 6. D3 — Memory-fit table (per-rank resident, all 40 layers, vs 128 GiB)

Exact arithmetic from the checkpoint headers (per-rank routed-expert weight count
N_rank = 271 790 899 200; group-scale overhead exact: q4g64 = 4.25 bpw, q4g32 = 4.50, q5g64 = 5.25,
q6g64 = 6.25). Node = **128 GiB = 137.44 GB**; production wired limit `iogpu.wired_limit_mb=115000`
(= 120.59 GB). Non-expert rest per rank = 10.71 GB; KV@91K = 0.30 GB.

| arm | experts GB | total GB/rank | total GiB | % of 128 GiB | verdict |
|---|---:|---:|---:|---:|---|
| **EXL3 2.9 bpw trellis** | **98.25** | **109.26** | 101.76 | 79.5 % | **FITS** |
| q4g32 (4.50 bpw) | 152.88 | 163.89 | 152.63 | 119.2 % | DOES-NOT-FIT |
| **q4g64 (4.25 bpw)** | **144.39** | **155.40** | 144.72 | **113.1 %** | **DOES-NOT-FIT** |
| mixed gu q4 / dn q6 | 167.04 | 178.04 | 165.82 | 129.5 % | DOES-NOT-FIT |
| q5g64 (5.25 bpw) | 178.36 | 189.37 | 176.36 | 137.8 % | DOES-NOT-FIT |
| q6g64 (6.25 bpw) | 212.34 | 223.34 | 208.00 | 162.5 % | DOES-NOT-FIT |

- **Every native arm is memory-FALSIFIED.** Even the cheapest (q4g64) is **+46 GB/rank over the EXL3
  it replaces** → 14 % over the 128 GiB node and over the 120.59 GB wired limit. Only the current
  **EXL3** representation fits (109.26 GB, ~28 GB headroom, vs the measured live anchor ≈105.5 GB).
- The routed-expert budget that fits is 137.44 − 10.71 − 0.30 = **126.4 GB ⇒ ≤ 3.72 bpw — below q4.**
  Native affine has no ≥q4 arm that fits; a sub-4-bit native format (q3-class) is the explicitly
  dropped worst double-quant error class. A native arm becomes deployable only with the
  streaming/tiering mitigation stack (memory-correction doc §3–4), **out of Round-1 scope.**
- **KV is negligible**: per rank 3200 B/token (bf16-resident comp_kv+index_k on the 4 kv_source
  layers, window ring constant) → 0.297 GB @91K, 3.36 GB @1M. Not the constraint; the experts are.
- Cross-check: header per-rank weights total 108.96 GB vs the memory-correction doc's 108.76 GB
  (0.2 %) and the live measured ~105.5 GB. Consistent.

---

## 7. D4 — Original pre-EXL3 checkpoint: sampled-shard fetch (STATUS: DONE, 3 layers)

- `dealignai/DeepSeek-V4.1-Flash-UNCENSORED` (the guessed bf16 name) → **absent** (HF API "not found").
- **Discovered original: `dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8`** — exists, **public, not gated**,
  510.3 GB, 48 shards; `quant_method=fp8`, `expert_dtype=fp4`, block [32,32] ue8m0; same arch (40 layers,
  384 experts, D=5120, H=2304). 1 shard per layer (shards 3–42 = layers 0–39).
- **Fetched only the sampled layers {0, 20, 39} → shards model-0000{3,23,42} = 22.18 GB**, throttled
  (`curl --limit-rate 20M`, resumable, nohup), on studio1. **sha256 bit-exact vs the HF LFS pointers**
  (00003 `e1281f85…`, 00023 `680947670…`, 00042 `e1a4d5d3…`). Read-only header re-check confirms each
  shard carries its layer's attn + experts 0..383. **Retained for Round 2 at
  `studio1:/tmp/q1d_mem/orig/`** (+ config.json, model.safetensors.index.json).

---

## 8. D5 — GATE 1 VERDICT

**GATE 1 = FAIL — the experts requant lead CLOSES (on the memory/deployability limb, not the speed limb).**

Pre-registered PASS required **all three** conjuncts: ≥30 % WALL saved ∧ correctness clean ∧ memory
fit OK. The measurement:

| limb | result | verdict |
|---|---|---|
| **Speed (≥30 % wall, ×40 expert stack, same-session)** | q4g64 **+32.61 %** (48.97 → 33.00 ms) | **PASS** (thin, +2.6 pp) |
| (speed, GPU-time attribution) | native/EXL3 kernel GPU ratio **0.564** (native 44 % faster) | PASS |
| **Correctness (A6a + A6b)** | recon near-exact; all arms legitimate | **PASS** |
| **Memory fit (per-rank, incl. KV@91K, vs 128 GiB)** | q4g64 155.4 GB/rank = **113 % of 128 GiB**; every native arm fails | **FAIL** |

No arm clears the full pre-registered bar → **the lead CLOSES**, and Round 2 (quality pre-screen) is
**not** authorized. **This is a memory-ceiling close, not a speed close.**

- **What is TRUE and worth keeping:** the native-quant drop-in *is* genuinely faster than the fused
  EXL3 trellis expert kernel at the production verify shape (q4g64: +32.6 % wall / +43.6 % GPU), and
  it is numerically legitimate. The dense-slice precedent (native q4 ~2.5× EXL3, A1) transfers to the
  expert path. The speed lever is the 4-bit packed stream; q5/q6 gain nothing but bytes.
- **Why it still closes:** the ×40 native experts do not fit the 128 GiB nodes (q4g64 is already
  +46 GB/rank over EXL3, 14 % over RAM and over the wired limit), so no passing arm is deployable
  as-is, and the quality work would be spent on an unshippable artifact. The fitting budget implies
  ≤3.72 bpw — below q4, i.e. the dropped q3-class. A native arm ships only with the streaming/tiering
  mitigation stack (out of scope).
- **Read with the alternative reading in mind (documented for override).** The brief's other note
  says the memory-fit "feeds gate 2" (the live A/B — which cannot co-reside two encodings). If the
  owner intends memory to gate **only** gate 2, then the speed limb **PASSES** and Round 2 on q4g64
  would be authorized, with memory as the known gate-2 blocker. The numbers above support either
  read; the pre-registered conjunctive condition makes the default **FAIL/close**.
- **Reopen condition (explicit).** Reopen the experts lead IF (a) the owner adopts the
  streaming/tiering memory stack, OR (b) a native format ≤3.7 bpw is judged quality-acceptable.
  Absent either, the requant offers a real but unshippable speedup.

---

## 9. D6 — END STATE / BUDGET ACCOUNTING

- **Production UNCHANGED and LIVE:** exo `deploy/next19-dense @ 99e2966ee` + mlx-lm `689e4ea`; runner
  env exactly `DSV41_DENSE=affine6` + `DSV41_DENSE_POLICY=layers.*.ffn.shared_experts.*=q8g64`; READY
  2/2. Nothing shipped, nothing re-encoded. Rollback refs untouched (prev prod `fb4f9290b`/`16830e1`;
  in-place `DSV41_DENSE=exl3`).
- **Canary healthy** (studio1 14.85 / studio2 14.86 t/s) after the round; cluster idle. **0 boots, 0
  relaunches** (the ≤2 reserve from ROUND-Q1C stays unspent). **0 encode spend, 0 live generation spend.**
- **Budget spent:** bench 33 s of actual node GPU (after a memory-lean rebuild; the first
  27 GiB-fp16 attempt GPU-timed-out on the production-live node and was abandoned — see §5 method);
  routing-hist CPU/numpy on studio1; mem-fit CPU; the sampled-shard fetch ~15 min throttled at
  20 MB/s (22.18 GB). All **well inside the ~2–4 GPU-h cap**. No strays: node `/tmp/q1d_*` cleaned;
  no writes under `~/repos/exo` on either node.
- **Anomaly noted (honest):** the bench's first arm build (full fp16 reconstruction + copy, ~27 GiB)
  hit `[Event::wait] no signal` under the production-live node (~98 GiB wired, ~22 GiB free); rebuilt
  memory-lean (~7 GiB peak, per-expert quantize == whole-tensor) and it ran clean. The EXL3 "natural"
  (6-unique synthetic) wall (1.3959/layer) reads ~14 % above the u24 wall (1.2243) although EXL3 is
  slot-bound — treated as measurement/order drift; the load-bearing gate comparison is u24-vs-u24
  same condition, so it does not move the verdict.
- **Artifacts:** `bench/p20_q1d_native_experts.py`, `bench/p20_q1d_routing_hist.py`;
  `raw/pricing/q1/q1d/{q1d_native_experts.json,.stdout.txt, q1d_routing_hist.json,.stdout.txt,
  q1d-FINDINGS.md, q1d-routing-hist.md, q1d-memfit-and-fetch.md, q1d_memfit.py,
  fetch_q1d_orig_shards.sh}`. Retained original shards: `studio1:/tmp/q1d_mem/orig/`.
