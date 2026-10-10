# ROUND-Q1E-Q3ARM.md — experts requant, Round 2: the q3-class arm (GATE 1 RE-DECISION)

Author: Phase-20 PM (delegation). Date: 2026-10-10 (CDT). Worktree `/private/tmp/phase20-campaign`.
Branch `deploy/phase20-campaign`. Round 2 of the experts-requant campaign. **OUTCOME: CLOSE.**

**Offline-first. 0 boots, 0 live spend, 0 encode spend.** Phase A desk checks were ~0 GPU (node
observations + a one-layer packing probe); **Phase B (the q3 arm) did NOT run** — the pre-registered
memory rule failed at the desk-check stage; Phase C trace mining ran CPU-only. Production untouched;
idle-guard + canary (read-only) only. The retained sampled originals (`studio1:/tmp/q1d_mem/orig`)
are intact (sha256 re-verified) and **retained** (the lead is closed but the source is cheap insurance).

---

## 0. DECLARED BUDGET (fixed BEFORE spending)

- **Boots: 0** (none spent). **GPU-hour cap: ~1–2 GPU-h offline** declared; **spent ≈ 0** (Phase B did
  not run; Phase A's A3 probe is a single `mx.quantize` of a 1152×5120 tensor). **Live spend: 0.**
  **Encode spend: 0.** Hard stop honored at the GATE-1 RE-DECISION.

### Entry state (verified ~01:35 CDT 2026-10-10)
Production LIVE: exo `deploy/next19-dense @ 99e2966ee` + mlx-lm `689e4ea`; runner env
`DSV41_DENSE=affine6` + `DSV41_DENSE_POLICY=layers.*.ffn.shared_experts.*=q8g64`. Cluster idle
(`phase20_guard.py idle` rc=0, `state_active_tasks=0`), canary healthy (studio1/studio2 14.85 t/s).
Rollback: prev prod `fb4f9290b`/`16830e1`; in-place `DSV41_DENSE=exl3`.

---

## A1. DESK CHECK — live wired limit (the load-bearing constant) — CONFIRMED

| node | `sysctl iogpu.wired_limit_mb` | `hw.memsize` | wired limit |
|---|---:|---:|---:|
| studio1 (`macstudio-m4-1`) | **120000** | 137 438 953 472 B | 125.829 GB = 117.19 GiB |
| studio2 (`macstudio-m4-2`) | **120000** | 137 438 953 472 B | 125.829 GB = 117.19 GiB |

**Setter (mechanism):** `start_cluster.sh:1474` — `_want_wl="${DSV4_WIRED_LIMIT_MB:-120000}"`, applied per
node via `sudo -n sysctl iogpu.wired_limit_mb=$_want_wl` (line 1475, NOPASSWD sudoers rule), read back at
line 1478 with a WARN on mismatch. ⇒ the live 120000 is the **launcher default**, overridable with
`DSV4_WIRED_LIMIT_MB`. (An unrelated low-memory code path at line 1452 sets 32000; not active. The
`bench/section17_memory_headroom_check.py` comment saying "115GB/node" is STALE — it is the source of the
Round-1 doc's prose error.) The runner log confirms the applied value: `Wired limit set to 117.19 GiB`.

**W_live = 120000 MiB = 125.829 GB = 117.19 GiB.** All Round-2 arithmetic uses this value.
Physical RAM = 128 GiB = 137.44 GB. `footprint` reports **decimal** GB (verified: its "105 GB"
IOAccelerator ≈ MLX active 104.7e9), so no GiB/GB unit slip in the peak readings below.

---

## C0. THE BUDGET-ARITHMETIC CORRECTION (recorded; permissive error; Round-1 verdict UNCHANGED)

The Round-1 doc (`ROUND-Q1D-MICROBENCH.md` §6) computed its memory-fit limb against the **full node RAM
(128 GiB = 137.44 GB)** rather than the production wired limit, and used a stale `115000` MB wired figure
in a cross-reference. Verified: the doc's `≤3.72 bpw` = (137.44 − 11.1 − 0.30) / 33.95 exactly.

**Effect on the Round-1 verdict: NONE — it is a *permissive* error.** The doc's ceiling was too generous,
so a correct computation is *stricter*, and every measured native arm still fails (and fails harder
against the live wired ceiling). **The Round-1 FAIL stands.** This correction matters only because it
re-opens exactly one question the doc dropped: whether a **q3-class** arm (never run in Round 1) fits.
That is what this round measured — and the answer, with the *live* ceiling, is still **no** (see §R).

A **second, compounding permissive error** was found and verified this round: the doc's per-arm bpw
figures **omit the bias term**. MLX affine `mx.quantize` emits **both** fp16 scales *and* fp16 biases
(2 × 16 bits per group), so the true effective bpw is `bits + 2·16/group_size`, not `bits + 16/group_size`.
The doc's own per-layer arm footprint (3.82 GB at q4g64) is consistent with the correct 4.5 bpw, so its
*footprint* was right while its *stated* 4.25 was the error. Independently reproduced (this round, studio2):

| bits/gs | emitted (fp16 in) | eff bpw | doc said |
|---|---|---:|---:|
| 3 / 128 | q uint32 + fp16 scales + fp16 biases | **3.25** | (candidate) |
| 3 / 64 | q uint32 + fp16 scales + fp16 biases | **3.50** | (candidate) |
| 4 / 64 | q uint32 + fp16 scales + fp16 biases | **4.50** | 4.25 |
| 4 / 32 | q uint32 + fp16 scales + fp16 biases | 5.00 | 4.50 |

(The rival "3.125 dense scale-only" packing was **refuted**; a `float32` *input* would inflate scales to
fp32 → 3.50/5.00/6.00, but production and the Round-1 harness both quantize **fp16** weights.)

### Correct fit line (per rank, GB), used in the Round-2 rule
Exact from the checkpoint header: per-rank routed-expert weight count `N_rank = 271 790 899 200`;
`GB_per_bpw = N_rank/8/1e9 = 33.974`. Non-expert "rest" weights (live) = 6.51 GB; KV@91K = 0.30 GB.

```
total_GB/rank = rest + KV + 33.974 * bpw        [bpw now including the bias term]
```

Cross-check vs the doc's §6 arms (recomputed at the CORRECT bpw): q4g32 5.0 → 180.9 GB, q4g64 4.5 →
163.9 GB, q5g64 5.5 → 197.9 GB, q6g64 6.5 → 231.9 GB — every native arm is even further over the node
than the doc showed. Confirms the Round-1 close.

---

## A2. DESK CHECK — incumbent (EXL3) non-weight peak → P

Sources used (and what failed):
- **VictoriaMetrics** `http://172.16.0.42:8428` — series `exo_peak_memory_bytes` ("Peak memory reported
  at completion of the most recent request", set from `mx.get_peak_memory()`). Exact query:
  `max_over_time(exo_peak_memory_bytes{instance="macstudio-m4-1",instance_id="7aa2dbd3-…"}[30d])` =
  **124.833 GB**, reached right after the 102 411-token request that finished 2026-10-10 00:41:06 CDT
  (≥ the 91K target, so KV is included).
- **Node runner log** (current process: studio1 pid 49282, launched 2026-10-09 22:43; studio2 pid 59746):
  `[DSV41] body loaded … active=104.7 GB` — `active = mx.get_active_memory()/1e9` (per-rank resident
  weights, both ranks). Log also confirms `Wired limit set to 117.19 GiB`.
- **Idle-guarded resident observation**: `footprint -p 49330` (studio1) → current **106.0 GB**,
  lifetime **peak 114.0 GB** (IOAccelerator 105 GB + malloc 1.4 GB at the time of reading).
- **Unavailable:** exo `/state` exposes no per-process footprint field.

Derivation (`non_weight_peak = peak_total − 104.7 GB`, where 104.7 is the live per-rank weight total):

| peak reading | peak_total | non-weight peak | **P = +2 GB** | note |
|---|---:|---:|---:|---|
| **resident** (`footprint -p`, OS phys_footprint_peak) | 114.0 GB | 9.3 GB | **11.3 GB** | "peak" as a resident quantity; corroborated by campaign doctrine (deep prefill ≈ +10 % over the 105 GB steady) |
| **allocator** (`mx.get_peak_memory`, the Metal/wired domain) | 124.833 GB | 20.13 GB | **22.13 GB** | allocator high-water; exceeds the resident peak by ~10.8 GB of allocated-but-not-resident transient |

Reconciliation with the doc: the doc's "non-weight" was 10.71 GB (rest **weights**) + 0.30 GB (KV). The
measured resident non-weight peak (9.3) is the KV **plus the operational prefill transient**; the rest
weights are *separate* and are themselves inside the 104.7 GB weight total.

> **Dimension note (the hinge of the rule).** `B_arm` (§A3) counts the **routed experts only**. So the
> rule's `P` must carry **everything else that is resident**: the 6.51 GB rest weights **+** the
> transient **+** margin. The dimensionally-correct demand is therefore
> `rest(6.51) + transient` — **not** the bare `peak − all_weights` figure. §R applies it correctly.

---

## A3. DESK CHECK — actual 3-bit packing → B_arm

Measured on studio2 (idle-guarded; mlx `0.32.3.dev20260918`), fp16 input (the production/harness path),
verified independently by the PM:

| arm | eff bpw | **B_arm = 33.974 × bpw** (GB/rank, all 40 MoE layers) |
|---|---:|---:|
| **q3g128** (candidate) | **3.25** | **110.42** |
| q3g64 (reference, NON-SHIPPABLE) | 3.50 | 118.91 |
| q4g64 (Round-1 anchor, cross-check) | 4.50 | 152.86 |

- Packing discriminated: **3.125 REFUTED, 3.25 MATCHES, 3.325 (10-vals/uint32) REFUTED.** Effective bpw =
  `bits + 2·16/gs` (scale **and** bias). Emitted arrays: q `uint32`, scales `float16`, biases `float16`.
- Real-weight validation: reconstructed `layers.20.ffn.experts.0.w1` (fp16 `[5120,2304]`) reproduced the
  same bpw as the production-shaped tensors (nbytes depend only on shape+dtype+bits+group_size).
- **Sha256 of the retained sampled originals — ALL 3 MATCH** (`model-00003` `e1281f85…d4c9`,
  `00023` `68094767…4524`, `00042` `e1a4d5d3…c2ef`). Manifest committed at
  `raw/pricing/q1/q1e/q1e_orig_sha256_manifest.txt`.

Raw: `raw/pricing/q1/q1e/{a2_nwpeak.json,a3_packing.json,q1e_packing.py,q1e-a2-note.md}`.

---

## R. PRE-REGISTERED DECISION RULE — EVALUATED

> **Run the q3 arm IFF `B_arm + P ≤ W_live` (= 125.829 GB); else CLOSE.** `P` = (measured non-weight
> demand: rest weights + transient) + 2 GB margin, since `B_arm` is routed-experts-only.

Headroom available for everything that is not a routed expert: `W_live − B_arm`.

| arm | B_arm GB | headroom = 125.829 − B_arm | P (resident, +2) | P (allocator, +2) | verdict |
|---|---:|---:|---:|---:|---|
| **q3g128** | **110.42** | **15.41** | **17.81 ✗** | **28.64 ✗** | **FAIL** |
| q3g64 (ref) | 118.91 | 6.92 | 17.81 ✗ | 28.64 ✗ | FAIL |

Where the demand is `rest(6.51) + transient` :

- **resident transient 9.3 GB** → demand 6.51 + 9.3 = **15.81 GB** vs headroom 15.41 → **over by 0.4 GB**
  (0.3 %); with the mandated +2 GB margin → 17.81 → **over by 2.4 GB**.
- **allocator transient 20.13 GB** → demand 26.64 → **over by 11.2 GB**; +2 → 13.2 GB.
- Equivalent incremental form (independent of weight-splitting): `peak_incumbent + Δweights` =
  114.0 + (116.93 − 104.7) = **126.23 GB > 125.829** (resident); 124.833 + 12.23 = **137.06** (allocator).

**The rule FAILS under BOTH defensible peak readings.** The only readings that pass are
dimensionally wrong: (a) pairing routed-only `B_arm` with a `P` that drops the 6.51 GB rest weights, or
(b) the Round-1 doc's transient-free model (`11.01 + 33.974·bpw` → 121.5 ≤ 125.83) — the very model this
round exists to correct. A second-opinion review confirmed the dimensional reasoning and the CLOSE.

---

## B. PHASE B — NOT RUN (the pre-registered rule failed)

No arm was measured. The pre-registered bars stand unspent for any future round:
- Speed (≥30 % WALL on the ×40 extrapolation vs same-session EXL3, at **both R=4 and R=1**), correctness
  (q3g128 per-layer rel-MSE ≤ 1.10 × EXL3's vs the bf16 originals on EVERY sampled layer, AND cos ≥
  EXL3 cos − 0.002), scope guard (existing `gather_qmm` 3-bit path only, else CLOSE).
- Budget reserved (1–2 GPU-h) is **unspent**.

---

## C. PHASE C — TRACE MINING (ran; free, local CPU, from the 531×40 real clamp trace)

Source: `docs/benchmarks/phase10-planb-gate-2026-09-28/raw/p30_exl3_trace_clamp.json` (531 tokens × 40
layers = 21 240 real gate decisions; top-6). PM re-derived the headline numbers independently — exact
match. Raw: `raw/pricing/q1/q1e/{c_trace_mining.json,c_trace_mining.py,c-trace-mining.md}`.

**1 — Per-layer unique-expert UNION + frequency curve.** Per-layer distinct experts over the full 531
tokens: **min 227 (L18) / median 276 / max 331 (L0)**; **overall union = 384/384** (no dead experts — all
384 are touched somewhere). Coverage of the 3186 per-layer picks by the top-k experts (per-layer median):

| k | 8 | 16 | **24** | 32 | 48 | 96 | 144 | 184 | 245 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| coverage | 23.9 % | 34.9 % | **43.6 %** | 50.5 % | 61.2 % | 80.0 % | 90.1 % | 95.1 % | 99.0 % |

⇒ 50 % of picks need ≈32 experts; 80 % need ≈96; the tail is long (no small dead set).

**2 — Hot-set concentration + STABILITY (the headline).** Routing is **concentrated but CHURNY** — neither
flat nor stable. Top-24 experts carry **43.6 %** of picks (≈7× the 6.25 % uniform share), but the top-24
set is **not stable across the prompt**: Jaccard of the top-24 sets between contiguous halves = **0.263**
(min 0.14, max 0.50); across thirds = 0.19; a later half's picks land in the earlier half's top-24 only
~34 % of the time. (A parity control gives 0.68 vs the 0.032 random baseline — so the drift is a real
positional/context effect, not noise.) ⇒ a **static** hot-q4/cold-EXL3 split is **marginal**; a
**windowed/dynamic** re-selection is the viable form (if that path is ever revisited).

**3 — Per-rank touched-expert IMBALANCE.** Production DSv4.1 TP is **intermediate-WIDTH sharding**
(each rank holds all 384 experts at half width — verified in `auto_parallel.py` + `section108…`), **not**
expert-id EP, so the real imbalance is **≡ 1.0**. As an EP **counterfactual** (rank0 = experts 0–191,
rank1 = 192–383), R=4 max/mean = p95 **1.50** / max **2.00**, with ≥2× in **0.02 %** of batches
(1 / 5280) ⇒ **no traffic-aware placement warranted**.

**4 — Batch-to-batch overlap (R=4).** Batch union mean **16.32** unique (median 16, max 24; cross-checks
the Round-1 16.25); consecutive-batch Jaccard **median 0.241**, trend **flat** (no warm-up). R=1 is
always exactly **6** unique (topk=6 distinct).

---

## G. GATE 1 RE-DECISION — **FAIL → the experts lead CLOSES**

**Verdict: CLOSE.** A native **q3g128** expert arm does **not** fit the live memory budget:
`B_arm (110.42) + P (≥15.81)` = **≥126.2 GB/rank > W_live 125.83 GB**, and with the pre-registered
+2 GB margin ≥128.2 GB. The miss is razor-thin on the most favorable (resident) reading — **0.4 GB
(0.3 %)** — but it is a miss, and the allocator (wired-domain) reading misses by 11 GB. q3g64 is far
worse. Per the pre-registered rule, the arm does **not** run and there is no speed/correctness headroom
worth spending on an unshippable shape.

- **What is TRUE and worth keeping:** the packing question is now settled with hard numbers — the q3
  candidate is **3.25 bpw**, not the hoped-for 3.125 (the bias term is real), and the incumbent EXL3
  already sits **within ~12 GB of the wired ceiling** at depth. Against the *physical* 137.44 GB the arm
  would fit (126.2 ≤ 137.44); it is the **live wired limit (125.83 GB), a raisable launcher guardrail
  (`DSV4_WIRED_LIMIT_MB`), that blocks it.**
- **Reopen condition (explicit, and a genuinely NEW hypothesis — not this arm):** (a) a deliberate
  wired-limit raise (e.g. 128000 MiB) with its own risk analysis, or (b) shrinking the demand — smaller
  prefill chunks / quantized shared experts (rest) or a measured-smaller arm transient, or (c) a native
  format ≤ ~3.24 bpw judged quality-acceptable. Each is a separate pre-registered round. Running the arm
  after a rule failure because the gap "looks small" is exactly the forking path pre-registration blocks.

---

## E. END STATE / BUDGET ACCOUNTING

- **Production UNCHANGED and LIVE:** exo `deploy/next19-dense @ 99e2966ee` + mlx-lm `689e4ea`;
  `DSV41_DENSE=affine6` + shared-experts@q8g64; READY 2/2. Nothing shipped, nothing re-encoded.
- **Canary healthy** (studio1/studio2 14.85 t/s) after the round; cluster idle; **0 boots, 0 relaunches**
  (the ROUND-Q1C ≤2 reserve stays unspent). **0 encode spend, 0 live generation spend.**
- **Budget spent:** Phase A ≈ 0 GPU (a single `mx.quantize` of a 1152×5120 tensor on studio2; read-only
  observations + VM/state queries); **Phase B not run (1–2 GPU-h reserved, unspent)**; Phase C local CPU
  numpy (~seconds). Node scratch removed (`/tmp/q1e_packing.*` on studio2; none on studio1). No writes
  under `~/repos/exo` on either node.
- **Retained:** `studio1:/tmp/q1d_mem/orig` (layers {0,20,39}, 22.18 GB, sha256 re-verified intact) —
  kept as cheap insurance; cheap to delete if the lead is formally retired.
- **Artifacts:** this doc + `raw/pricing/q1/q1e/{a2_nwpeak.json,a3_packing.json,q1e_packing.py,
  q1e-a2-note.md, c_trace_mining.json,c_trace_mining.py,c_trace_mining.stdout.txt,c-trace-mining.md,
  q1e_orig_sha256_manifest.txt}` on `deploy/phase20-campaign`. `PERFORMANCE_HISTORY.md` (main) carries a
  same-turn entry.
