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

### Entry state (verified at entry ~00:16 CDT)
Production LIVE: exo `deploy/next19-dense @ 99e2966ee` + mlx-lm `689e4ea`; baked runner env
`DSV41_DENSE=affine6` + `DSV41_DENSE_POLICY=layers.*.ffn.shared_experts.*=q8g64` (verified verbatim in both
nodes' `SCREEN exorun` process command lines); tag `known-good-dense-q68-20261009-231842`. Perf: benign 20K
81.3 ms/45.2 t/s; agentic 91K 86.03 ms/34.3 t/s. Cluster **idle** (`phase20_guard.py idle` → ok, no reasons).
Rollback refs: prev prod `fb4f9290b`/`16830e1`; in-place `DSV41_DENSE=exl3`.

---

## 1. GATE 0 — pre-registered falsifier (THE decision this round exists for)

From the measured **N1** op-class split of expert decode time:

- **CLOSE the experts lead** if EITHER
  - **(a)** expert **kernel** time is **under ~25 ms** of the **~86 ms** agentic round, OR
  - **(b)** **non-kernel overhead dominates** the expert time: `gather/scatter/routing + all-sum/RDMA +
    unattributed  >>  kernel`. (Requant changes the tensor FORMAT → the kernel; it does NOT change routing,
    index gather/scatter, or the TP all-sum. If those dominate, a format change cannot move the block.)
- **PROCEED with the experts requant campaign** ONLY if BOTH
  - expert **kernel** time is **~25 ms+**, AND
  - the expert block is **kernel-dominated** (kernel ≳ non-kernel overhead).

PROCEED only *opens* the campaign (its own offline microbench + quality pre-screen + live A/B is a LATER
round — **nothing of that happens now**). The verdict is written explicitly as **PROCEED / CLOSE with the
numbers** (§7).

**Mechanism.** Q1 priced the DENSE slice: native qN g64 is **2.47×** the fused EXL3 trellis kernel at m=4
(`ROUND-PRICING.md` §Q1) — the win is **ALU/issue-bound** (the k=7 SWAR trellis decode), *not* bandwidth
(q4 streams 62 MB vs EXL3's 66.7 MB yet is 2.5× faster). The experts arm asks the same question of the
**routed experts** (384/layer, EXL3 2.9 bpw trellis). Q4 already showed the *whole* expert block at the
verify shape is **1.077 ms/layer wall × 40 = 43.08 ms** standalone (≈ the ~37 ms unattributed, 1.16×).
Gate 0 must now split that block into **kernel vs overhead** — the number Q4 did not produce.

---

## 2. THE ROUND (deliverables)

- **N1 — Path1 op-class GPU timing on current prod (core).** ✅ §3
- **N2 — Prefill ladder re-measure on the dense build.** ✅ §4
- **N3 — Engram coverage check (offline).** ✅ §5
- **N4 — Acceptance baseline.** ✅ §6
- **GATE 0 VERDICT: PROCEED / CLOSE with numbers.** ✅ §7

---

## 3. N1 — Path1 per-op-class GPU timing of the MoE expert block (OFFLINE, studio2)

**Method.** Offline single-node bench (`bench/p20_q1c_opclass.py`), **no engine touch, 0 boots**. Loaded the
real layer-20 `EXL3SwitchGLU` via `load_experts(ckpt, 20, n_experts=384, rank=0, world=2,
activation="silu_clamp")` (per-rank **H=1152**, D=5120, E=384) — the exact production TP=2 rank slice. Real
per-op **GPU** ms via the Q5 mechanism: `MLX_GPU_TIME=1` set before `import mlx`, then per sub-stage
`reset_gpu_time()` → build one stage → `mx.eval` → `mx.synchronize()` → `mx.metal.gpu_time_ns()`. Median over
15 reps (decode) / 11 reps (prefill), 3 warmup, 3 correlated-draft seeds. Router = the real
`deepseek_v41/moe.py` `Gate`. **Two arms** (`EXL3_MOE_FUSED=1` = production, `=0` = fully-split). node
`Adams-Mac-Studio-M4-2`, mlx `0.32.3.dev20260918+603f16eb7`, layer 20, 40 MoE layers.

### 3.1 The split — fused arm (PRODUCTION path), per sub-stage GPU ms (medians)

| stage | R1 draft (`_decode_fused2`) | **R4 verify (`_decode_fused2`)** | share (R4) |
|---|---:|---:|---:|
| prep / gather + per-row Hadamard | 0.0091 | 0.0089 | 0.9 % |
| **gate_up KERNEL** | **0.2847** | **0.5405** | 57.5 % |
| **down KERNEL** (act inline) | **0.2241** | **0.3238** | 34.5 % |
| **KERNEL total** | **0.5088** | **0.8643** | **92.0 %** |
| **UNATTRIBUTED** (whole − Σ stages) | **+0.0171** | **+0.0666** | **7.1 %** |
| whole `module(x, idx)` | 0.5350 | 0.9398 | — |
| router (separate module, outside the expert block) | 0.0412 | 0.0481 | — |

**Corroboration — unfused arm (`EXL3_MOE_FUSED=0`), R1 `_decode` (the clean full decomposition):**
of 0.3888 ms, gate_up (`mapped_gemv`) 0.2341 + down (`mapped_gemv`) 0.1155 = **0.3496 ms kernel (89.9 %)**;
**all** non-kernel ops (gather+Hadamard prep, finish, activation, ×2 projections) = **0.0392 ms (10.1 %)**.

The fused production kernel hides the activation / per-row Hadamard / `svh` gather **inside** the two GEMM
launches (only 3 dispatches separable), so the unfused arm is run precisely to expose the non-kernel share —
and it agrees: ~90 % kernel, ~10 % non-kernel.

### 3.2 ×40-layer round totals (whole expert module, GPU ms)

| shape | path | per-layer | ×40 layers |
|---|---|---:|---:|
| R1 draft (`topk6`) | `_decode_fused2` | 0.5350 | **21.40** (kernel 20.35) |
| **R4 verify (`topk6`) — PRODUCTION shape** | `_decode_fused2` | 0.9398 | **37.59** (**kernel 34.57**) |
| S=512 prefill | `_prefill` | 13.75 | 550.1 |
| S=2048 prefill (engine prefill step) | `_prefill` | 52.90 | 2115.8 |
| router R4 ×40 | — | 0.0481 | 1.92 |

**Per-round expert block = the single R4 verify pass**: `37.59 ms` GPU (**kernel `34.57 ms`**) across all 40
MoE layers. *(N.B.: the raw artifact also carries a `γ·draft + 1·verify` extrapolation `3×R1 + R4 ≈ 101.8 ms`;
that is **not** the per-round figure — it would exceed the 86 ms round, and DSv4's draft is the dedicated
DSpark/MTP head (`EXO_DSV4_MTP=1 EXO_DSV4_DSPARK=1`), not the 40-layer model. The full-model expert block runs
once per round, at the verify shape. Disregard the `101.8` line; it is a decomposition artifact.)*

**Cross-check vs Q4:** Q4 (wall clock) got the same block at 1.077 ms/layer → 43.08 ms; N1 (GPU time) gets
0.9398 → 37.59 ms. The 0.14 ms/layer gap is the Python-dispatch + launch overhead that `gpu_time_ns()`
excludes — the same gpu/wall ≈ 0.85–0.9 signature Q5 measured on low-occupancy ops. Both ≈ the ~37 ms
unattributed residual Q4 attributed to the MoE expert GEMM. **Coherent.**

### 3.3 Prefill split (S=2048) — with a caveat

`_prefill` (segmented `seg_mm`): gate_up kernel 39.05 + down kernel 12.72 = **51.78 ms kernel** of a 52.90 ms
whole = **97.9 %**. Non-kernel stages (gu prep/gather+Hadamard 8.09, gu finish 1.25, act 0.49, dn prep 0.50,
dn finish 2.14, scatter 2.65, sort 0.09, segtable 0.03) **over-sum** the whole (Σ = 67.01, **unattributed
−14.12**) — in one `eval` MLX pipelines adjacent dispatches inside a command buffer, so isolated per-stage
spans can exceed the whole-call span. **The prefill split's non-kernel line is therefore unreliable**; only the
kernel/whole ratio (≳92 %, here 97.9 %) is trustworthy. The decode shapes — the ones the gate is about —
close to ≈0 unattributed (+0.017 / +0.067), validating the bracket there.

### 3.4 Best-effort per-expert m-distribution (verify shape, R4 topk6)

The 24 (row, slot) pairs of a correlated verify batch land on exactly **6 distinct experts, 4 rows each**
(`hist = [4,4,4,4,4,4]`). Unique-expert count: R1 6.0, R4 6.0, S2048 7.33 mean. ⇒ the expert GEMM operates at
**tiny m (4 rows/expert)** — launch/ALU-bound, gather touches the minimum unique-expert set (corroborates
Q4's "slot-bound, routing-distribution-independent").

### 3.5 N1 finding

**Expert decode is KERNEL/ALU/issue-bound, not gather-scatter-routing-bound.** At the production verify
shape the two expert GEMM kernels are **92 %** of the whole expert-module call (90 % in the fully-split
unfused arm); gather/prep/finish/activation ≈ 1–10 %; the unattributed remainder is 7.1 %; and the router
(0.048 ms) is comparable to the *entire* non-kernel share of the expert module.

---

## 4. N2 — delta-prefill ladder on the dense build (LIVE, idle-guarded)

Re-measured with `bench/phase20_delta_ladder.py` against the live cluster (idle-guarded chunks, canary per
chunk, 0 reboots). New raw under `raw/pricing/q1/q1c/ladder/`. 2048-row delta at each ctx; median of 3 reps.

| ctx | **new rows/s (dense build)** | old EXL3-dense ceiling | Δ |
|---|---:|---:|---:|
| 20K | **269.18** (266.0 / 270.5 / 269.2) | 272.60 | **−1.3 %** |
| 50K | **253.73** (253.7 / 253.3 / 254.0) | 249.62 | **+1.6 %** |
| 110K | **251.52** (238.4 / 251.5 / 251.7) | 242.46 | **+3.7 %** |

**Verdict: FLAT.** All three rungs land within ~±4 % of the recorded EXL3-dense ceiling (two above, one
below; the 110K rung reads marginally higher, mostly from one low outlier rep at 238.4 — the other two
reps are 251.5/251.7). ctx-depth benign: **6.6 %** 20K→110K (old 11.1 %; ≤15 %). **The dense format change did
not move the prefill ceiling — the prior "no lever on prefill" close STANDS** (footnote: re-confirmed on the
shipped dense build; 110K nominally +3.7 %, within variance, not a lever).

---

## 5. N3 — engram coverage under `DSV41_DENSE=affine6` (OFFLINE code-trace, no GPU)

**VERDICT: COVERED.** The served engram dense group (`layers.{1,14}.engram.wkv`, the census's ~8 % bucket) is
governed by `DSV41_DENSE=affine6`.

- Call path (`mlx_lm/models/deepseek_v41/exl3_build.py`): `:688` puts every non-expert `….trellis` key —
  including `layers.{1,14}.engram.wkv` — into the `dense` set; `:713-724` (else branch `:724`) routes it through
  `_dense()`; `:615-632` `_resolve_dense_mode("layers.1.engram.wkv", policy, "affine6")` → the shared-experts
  selector does **not** match → falls to **base `affine6`** → `AffineProj` (6-bit, group 64).
- Served config has `engram_layer_ids = [1, 14]` (not the `()` default) — engram layers are live.
- A **separate** object, the native fp8 `engram.embed` hash table (`:730-733`, `LazyEngramTable`), is streamed
  row-on-demand from the *native release* — it is neither EXL3 nor affine and is **not** a "dense byte"; out of
  scope, not a quant candidate.
- **Flag condition (remains EXL3) does NOT fire ⇒ NO ~8 % free extra** for a future experts re-encode. No
  action. (Full evidence: `raw/pricing/q1/q1c/n3-engram.md`.)

---

## 6. N4 — Acceptance baseline (recorded, no new run)

Ship-smoke acceptance (from `~/.hermes/cache/scratch/next19-ship-logs/smoke/`):

| arm | depth | mean-accepted (median) | ms/round (median) | decode t/s (median) |
|---|---:|---:|---:|---:|
| **agentic** | 91 043 | **1.9395** (1.952 / 1.927) | **86.03** (85.96 / 86.09) | **34.299** |
| **benign** | 20 000 | **2.6723** (2.6455 / 2.6991) | **81.30** (81.48 / 81.12) | **45.159** |

→ recorded as the baseline for future comparison; no new run.

---

## 7. GATE 0 VERDICT: **PROCEED** (with recorded caveats)

Measured split at the **production verify shape (R=4 topk6)**, ×40 MoE layers:

| quantity | value | gate test |
|---|---:|---|
| expert **kernel** (gate_up+down GEMM) ×40 | **34.57 ms** | **≥ ~25 ms** ✅ (40 % of the 86.03 ms agentic round) |
| expert module whole ×40 (GPU) | 37.59 ms | — |
| non-kernel (prep/gather+Hadamard) | 0.0089 ms/layer — 0.9 % | **<< kernel** ✅ |
| unattributed | 0.0666 ms/layer — 7.1 % | **<< kernel** ✅ |
| router (separate module) | 0.0481 ms/layer | small |

- **(a) does NOT fire** — expert kernel 34.57 ms **is** ≥ ~25 ms (even the lowest per-seed kernel, 34.5, and
  the high seed-0, 40.6, both clear it).
- **(b) does NOT fire** — the expert block is **kernel-dominated** (92 % kernel; ~90 % in the independent
  fully-split unfused arm). gather/scatter/routing ≈ 1–10 %; unattributed 7.1 %; Q4 independently bounds
  exposed non-compute (comm) at **≤ 5.25 ms**, so the unmeasurable **all-sum/RDMA** term is also **<< kernel**.

**⇒ GATE 0 = PROCEED.** The experts requant campaign is opened (its own offline microbench + quality
pre-screen + live A/B is a LATER round).

**Recorded caveats (do NOT invert the verdict; they bound the campaign's *upside*, not its *existence*):**
1. **x40 from layer 20** assumes layer-uniformity — sound (Q4 verified all 40 layers carry MoE); layer 20
   carries k=2 trellis (per loop-2, layers 18-22 are 2-bit) so deeper-layer kernels could differ slightly.
2. **Single-node ⇒ exposure is an UPPER bound.** The standalone 37.59 ms is the expert block's standalone GPU
   cost; the live round may overlap it with other work, so the *realized* win is ≤ that. The campaign's live
   A/B measures it.
3. **The realized native-qN speedup on the *expert* path is UNPROVEN.** Q1's 2.47× was the dense
   `quantized_matmul` path; the expert path is a fused gather/segmented kernel. Pricing that transfer is the
   campaign's **first** task — it is **not** assumed here, and if it fails the campaign closes cheaply.
4. **All-sum/RDMA is not measurable offline** (JACCL carries no GPU timestamp — Q5 §2). Bounded only by Q4's
   ≤ 5.25 ms exposed non-compute. Gate (b) is therefore not *directly* falsified for the comm term, but the
   ≤ 5.25 ms bound is far below kernel.

---

## 8. END STATE / BUDGET ACCOUNTING

- **Production UNCHANGED and LIVE:** `deploy/next19-dense @ 99e2966ee` + mlx-lm `689e4ea`; runner env exactly
  `DSV41_DENSE=affine6` + `DSV41_DENSE_POLICY=layers.*.ffn.shared_experts.*=q8g64`; READY 2/2 (1 runner per
  node, no strays); canary **healthy 14.86 / 14.85** (after the round). Nothing shipped, nothing re-encoded.
- **Budget spent: 0 boots of ≤2** (N1 ran fully offline on studio2; N2 was live but boot-free). The ≤2
  relaunch reserve is **held unspent** — the offline Path1 split resolved the kernel-vs-overhead question, so
  no measurement boot was needed (as the budget §0 pre-declared). Live GPU work: 3 idle-guarded delta-ladder
  chunks (~28 min of cluster requests) + 2 canaries; well inside the 3–4 GPU-h cap. No encode spend.
- **Rollback refs untouched:** prev prod `fb4f9290b`/`16830e1`; in-place `DSV41_DENSE=exl3`.
- **Anomaly noted:** the Q5 `phase20_gpu_busy` whole-round GPU-residency raw (`raw/gpu_busy.*.json`) is stale
  (pre-FIX-3 monotonic epochs; `window_valid=false`) — NOT used here. N1's per-op GPU time is the load-bearing
  instrument this round.
- **Artifacts:** `bench/p20_q1c_opclass.py`; `raw/pricing/q1/q1c/{q1c_opclass.json,q1c_opclass.stdout.txt,
  n1-opclass.md,n3-engram.md}`; `raw/pricing/q1/q1c/ladder/{delta_ladder.chunk1|chunk2a|chunk2b.jsonl,
  delta_ladder.summary.md,.csv}`.
