# PHASE 20 — Phase-4 continuation campaign (P1 lever-2 guard → P2 MoE → P3 prefill → P5 roofline)

Author: Phase-4 PM. Opened 2026-10-08 ~16:30 CDT. **This doc is the resume anchor — commit+push after EVERY step.**

## 0. Entry state (verified by the Phase-4 PM, 2026-10-08 ~16:20 CDT)

- **PRODUCTION NOW**: `deploy/next13 @ 576e9d279` (exo) + mlx-lm `3bf8316` on both nodes.
  Both nodes `git rev-parse HEAD` == `576e9d27994b5bd3c2317f7dffbdf384580db0a4`, branch `deploy/next13`.
  Installed-venv lever-1 guard present (`grep -c "and m > _FENCE_MIN_ROWS"` = 1 both nodes).
  `/state`: 1 instance, 2 runners `RunnerRunning`. Idle.
- Tagged: `known-good-decode-next17-20261008-160746` (both forks). Baseline doc:
  `docs/known-good-decode-baseline-20261008.md` on main.
- **Proven numbers (Phase-3B, same-session symmetric arms)**: benign 20K g3 800tok /
  agentic 91K real-session-replay g3 800tok. production-before 145.75 / 157.10 ms;
  next17-defaults 118.56 / 130.07 ms; next17+`DSV41_INDEXER_HIER=0` **101.06 ms ≈30.2 t/s agentic**.
  Split: lever-1 code −27.0 ms, lever-2 HIER −29.0 ms, additive (sum 56.04 == total).

## 1. The frozen plan (Fable consult)

Each phase: impl → prove → A/B → battery → ship → doc. Run in order.

| phase | deliverable | gate | budget |
|---|---|---|---|
| **P1** | **next18** = lever-2 code guard `indexer.py:531 if _HIER and n > _FENCE_MIN_ROWS:` + shared `_gates.py` | full identity proof suite (below) — **ABORT on any divergence (ties = divergence)**; target ≈101 ms ≈30 t/s agentic | 1 promotion |
| **P2** | **next19** = `DSV41_MOE_ALLSUM_BF16` decode A/B (env kill-switch), ship only if it wins | ≥3 ms/round net agentic above noise, `mean_accepted` not < ~2.0, battery clean | 1 promotion (if it wins) |
| **P3** | **next20** = loop-2 consumer-index-skip PREFILL behind an env knob | ≥8% of deep-context prefill wall (half the audit max); decode byte-for-byte unchanged; battery clean | 1 promotion (if it wins) |
| **P5** | roofline close-out (bench-only) | (i) reconcile round accounting 91.5% vs 98.5%; (ii) bytes-roofline ONE verify round vs MEASURED read BW → near-floor vs headroom verdict | bench-only |
| — | adaptive gamma — **PARKED** (feature-blocked; `GammaPolicy.update()` never called, candidate set hardcoded {1,2,3,4}; γ5 benign-only) | one paragraph, zero relaunches | 0 |

## 2. Budget / tripwires (declared UP FRONT)

- Total: **≤3 production relaunches** (P1/P2/P3 promotions as they win) **+ 1 rollback reserve
  + 2–3 bench-only sessions**.
- Per-phase hard cap: if any single phase needs **>2 unplanned relaunches, HALT that phase**,
  doc it, move to the next. Never exceed the total.
- **Every relaunch declared in this doc BEFORE spending** (the prior rounds' pattern).
- If a phase's proof fails → **do NOT ship**; write it up as a speed-vs-output tradeoff; the
  prior production build stays; proceed to the next phase. An abort is a valid result.

### Relaunch ledger (declare-before-spend)

| # | phase | deploy | purpose | status |
|---|---|---|---|---|
| R1 | P1 | `deploy/next18-identity` (detach + mlx-lm next18) | A/B next18-at-defaults vs prod; battery; guard-engagement (+benign-HIER-off new measure) | PLANNED |
| R2 | P2 | next19 (only if MoE wins) | MoE decode A/B ship | CONTINGENT |
| R3 | P3 | next20 (only if prefill wins) | prefill ship | CONTINGENT |
| RESERVE | — | restore best SHIPPED build | rollback, only if a promotion degrades | HELD |
| B1-B3 | P2/P3/P5 | bench-only (no relaunch; measurement sessions on the live boot) | MoE arm captures, prefill captures, roofline | PLANNED |

## 3. P1 — lever-2 code guard (next18)

### 3a. The edit (spec §7 of PHASE3B-SHIP-VALIDATION.md)

- `mlx_lm/models/deepseek_v41/indexer.py` line ~531: `if _HIER:` → `if _HIER and n > _FENCE_MIN_ROWS:`
  where `n = x.shape[1]` (query-row count; same quantity lever-1 guards on as `m`).
- Hoist the threshold to a NEW `mlx_lm/models/deepseek_v41/_gates.py` read by BOTH
  `indexer.py` and `sparse_attention.py` (retires the twin-default foot-gun).
- `DSV41_INDEXER_HIER=0` and `DSV41_SPARSE_FENCE_MIN_ROWS=0` must STILL force old behavior for A/B
  (HIER=0 → always fallback; FENCE_MIN_ROWS=0 → always HIER).
- One edit covers both candidate-source (layers 2/8/14/20) and consumer (24-36) roles — do NOT split
  the predicate (same `n` within a forward).

### 3b. Required proof suite (ABORT on any divergence; ties count as divergence)

1. **Elementwise index-identity**: hier topk ≡ tiled/untiled fallback on the returned `[b,n,k]` int32
   tensor (incl. `-1` mask pattern + position order) over synthetic grid: n ∈ {1,2,3,4} ∪ {16,17}
   (guard boundary), nb ∈ {64,512,4096,16384}, many seeds. Also k±1, fully-masked (-1) blocks,
   deliberate bf16 near-tie mis-ranks (block maxima differing within the mantissa), true-top-k column
   placed exactly at rank 16 (overfetch boundary).
2. **REAL-TENSOR diff**: capture indexer inputs (and outputs) from a real 6-8 rep 91K agentic replay at
   decode(m=1) and verify(m=4) steps, at source AND consumer layers; run hier vs fallback on those exact
   tensors; require zero diff. (Agentic traffic is tie-rich — synthetic-only is NOT sufficient.)
3. **Greedy token-identity diff** vs the shipped next17 build on K≥4 real prompts (greedy-only).
4. Per-forward producer/consumer path-agreement assertion + a grep proving no other module reads
   `DSV41_INDEXER_HIER` directly (single read point).
5. Same-build A1/A2 determinism replicate; battery R8a clean on the new build; `mean_accepted` unchanged.
6. Guard-engagement: next18-at-defaults reproduces ≈101.06 ms ±2% agentic same-session (equivalence AND
   engagement); also measure BENIGN on the new build (never measured for HIER-off: expect ≈95 ms).

### 3c. P1 status — **ABORTED (proof suite DIVERGED; not shipped; 0 relaunches spent)**

- [x] Worktrees: `/private/tmp/next18-lever2` (branch `deploy/next18-lever2` from `3bf8316`),
      `/private/tmp/next18-exo` (branch `deploy/next18-identity` from `576e9d279`).
- [x] Implementation + unit tests — `indexer.py:546` `if _HIER and n > _FENCE_MIN_ROWS:`;
      shared `deepseek_v41/_gates.py`; `sparse_attention.py` imports the same symbol.
- [x] Proof suite 1 (`tests/test_dsv41_indexer_smallm_hier.py`, 2045 cells) — **FAIL → ABORT**
- [x] Independent PM reproduction of the divergence (below)
- [x] Committed + pushed: `deploy/next18-lever2` @ `938b811` (proof `d4531e2`; capture harness
      `bench/next18_capture.py`). **Production unchanged (`576e9d279`); no relaunch spent.**

#### 3d. Proof result (the load-bearing finding)

`tests/test_dsv41_indexer_smallm_hier.py` → **2045 cases, 1106 equal, 939 DIVERGENT**, all at
`n <= 16` (decode m=1, verify m=4). `n=17` (HIER on both sides) is identically 0-diff. Deterministic
across reruns; path-boundary assertions PASS (n=1,4,15,16 → fallback; 17,18,32 → HIER); RED/GREEN
sabotage controls PASS (comparator detects a 1-slot diff; overfetch=0 on real keys loses 7 slots).

Minimal reproducing cell: `role=plain, n=1, nb=4096, k=511` → 261/511 slots differ.

**Root cause (PM, independently reproduced on the laptop GPU):**
```
fallback(fp32 row) vs hier        : 0 diffs / 165,924 slots (12 seeds × 5 shapes)
fallback(bf16 row) vs hier        : 264 diffs (the production config)
fallback(bf16 row) vs fallback(fp32): 264 diffs
```
The hierarchical path exact-rescores in **fp32**; the fallback `_tiled_scores_buffer` /
untiled reference path **stores the row in `_ROW_DTYPE` = bf16** (`DSV41_INDEXER_ROW_BF16`
default 1). At the k-th boundary the bf16 row creates **ties** the fp32 row does not have, and the
two `argpartition`s break them differently. So the guard would change decode/verify output vs the
shipped next17 build — **not** a near-tie margin issue: it is a real semantic gap between the two
branches of the same `Indexer.__call__`, confirmed *against* §7's stated worry. Setting
`DSV41_INDEXER_ROW_BF16=0` removes the *continuous-data* divergence (0/165k) but the engineered
bf16 near-tie / overfetch-boundary cells **still diverge** → `overfetch=16` is genuinely a
heuristic, so the frozen abort rule (ties count as divergence) fires regardless.

**Verdict:** lever-2 code guard **does not ship**. next17 remains production. The lever-2 *value*
(29 ms/round) is real but only reachable via an env flip (`DSV41_INDEXER_HIER=0`) that trades
output for speed — a genuine speed-vs-output tradeoff, reported honestly, **not** papered over.
Constructive path for a future round (documented, not attempted here): make the small-n fallback
rank on an **fp32 row** (cheap at n=1/[1,4]) so both branches share one precision, and re-run this
suite to green *including* the engineered tie cells — i.e. bound the overfetch residual or emit a
runtime certificate — before anyone ships the guard.

**Budget: P1 spent 0 relaunches** (proof was off-line). Proceeding to P2.


## 4. P2 — MoE_ALLSUM_BF16 decode A/B (next19 only if it wins)

### 4a. Prereg + design

`DSV41_MOE_ALLSUM_BF16` (`moe.py:51`, default "1") rounds each rank's fp32 MoE-tail
partial to bf16 before the `all_sum` (halved payload), upcasting the result back to fp32.
**It is ALREADY default-ON in production** (promoted 2026-10-07, PERFORMANCE_HISTORY). So P2 is a
pure **env-flip A/B on the live production boot**, kill-switch off vs on:
`DSV41_MOE_ALLSUM_BF16=0` (exact fp32, the comparison arm) vs default 1 (bf16, live).

- Gate (this phase): the bf16 arm wins by **≥3 ms/round net agentic** above noise, `mean_accepted`
  not below ~2.0, battery clean — else it does not ship (env stays at its shipped default ON).
- Note: the ON arm needs **no relaunch** (live). The OFF arm needs **1 relaunch** (env-flip), spent
  only if the ON-vs-historical contrast warrants the confirmation.
- Same-build symmetric protocol: `p3b_driver.py` benign 20K g3 ×4 reps + agentic 91K g3 ×6 reps
  (unique-salt cold prefill each rep), no PROF, idle-guarded per chunk.

### 4b. P2 ON-arm (live next17 boot, `576e9d279`, DSV41_MOE_ALLSUM_BF16 default 1) — DONE

| arm | reps | ms/round median | decode t/s | mean_accepted |
|---|---|---|---|---|
| benign 20K g3, 800tok | 4 | **118.53** (118.17,118.53,118.57) | **32.36** | 2.9314 |
| agentic 91K g3, 800tok | 6 | **129.90** (129.89,130.08,129.90,129.76,130.16) | **23.62** | 2.0611 |

**Same-session check:** the benign ON-arm (118.53 ms) reproduces the §P3B next17-defaults benign
(118.56) to **0.03 ms**, and agentic (129.90) to the §P3B agentic (130.07) to **0.17 ms** — the live
boot is bit-for-bit in the §P3B next17-defaults regime. So the ON arm **is** the §P3B
next17-defaults point (MOE_ALLSUM_BF16 does not move the ms/round from that number → its own effect
is inside the ±noise of these two sessions).

### 4c. P2 OFF-arm relaunch — DECLARED before spending (relaunch R2)

`p2_off_deploy.sh`: idle-guard → `git checkout --detach 576e9d279` + mlx-lm `3bf8316` → **export
`DSV41_MOE_ALLSUM_BF16=0`** (exact fp32 arm) → `EXO_TARGET_BRANCH=deploy/next17-levers
./start_cluster.sh` → post-boot matmul canary + **MEASURED read-bandwidth canary** (both nodes).
Then `p2_off_measure.sh`: benign 4 + agentic 6, same harness. This is the same-build, same-session
kill-switch arm for the ≥3 ms gate. Log: `p4/p2_off_deploy.log`. **Spent 2026-10-08 ~17:36 CDT.**
Deploy verified: "Nodes synchronized on commit 576e9d279" → READY (2/2), post-boot matmul canary
**14.85/14.86** healthy, `DSV41_MOE_ALLSUM_BF16=0` confirmed in BOTH runners' env.

### 4d. P2 OFF-arm result + A/B verdict — DONE

| arm | MOE_ALLSUM_BF16 | reps | ms/round median | decode t/s | mean_accepted |
|---|---|---|---|---|---|
| benign 20K | **1 (bf16, shipped)** | 4 | **118.53** (118.17,118.53,118.57) | **32.36** | 2.9314 |
| benign 20K | **0 (fp32, this boot)** | 4 | **121.21** (122.04,121.21,120.92) | **31.06** | 2.7783 |
| agentic 91K | **1 (bf16, shipped)** | 6 | **129.90** (129.89,130.08,129.90,129.76,130.16) | **23.62** | 2.0611 |
| agentic 91K | **0 (fp32, this boot)** | 6 | **133.22** (132.36,132.98,133.22,133.27,133.75) | **23.63** | 2.1575 |

**A/B (bf16-ON minus fp32-OFF):**
```
agentic : 133.22 − 129.90 = 3.32 ms/round  →  PASS (≥3 ms bar, ranges DISJOINT)
benign  : 121.21 − 118.53 = 2.68 ms/round  →  below the 3 ms bar
mean_accepted: agentic ON 2.0611 vs OFF 2.1575 (−0.096, still > 2.0) ; benign 2.9314 vs 2.7783
```
- **Gate verdict: PASS on agentic** (3.32 ms ≥ 3 ms; the ON range max 130.16 < OFF range min 132.36,
  so it is robust to rep noise). mean_accepted stays >2.0 (2.06) → the acceptance leg passes; note
  bf16 is marginally *lower* acceptance than fp32 (2.06 vs 2.16) — a mild quality direction, already
  accepted at the 2026-10-07 promotion (battery PASSED on the bf16 arm then).
- **Caveat (honest):** this is a **cross-boot** A/B (the env is set at launch, so no same-boot
  kill-switch is possible). The ON boot reproduced the §P3B next17-defaults point to 0.03–0.17 ms,
  and both boots show the same benign:agentic ratio, so the ~3 ms is far more likely the lever than
  boot drift — but it is NOT a same-boot measurement and is labelled as such.
- **Consequence: no next19 build is needed.** `DSV41_MOE_ALLSUM_BF16=1` is **already the shipped
  default** and this A/B *validates* it clears the Phase-4 ≥3 ms bar. Production stays as-is.
- **Budget: P2 spent 1 relaunch (R2, the OFF arm).**

## 4e. ADAPTIVE GAMMA — **PARKED** (feature-blocked, one paragraph per the brief)

The adaptive-gamma lever is **feature-blocked** and gets **zero relaunches** this campaign: the
`GammaPolicy.update()` hook is **never called** anywhere in the code path, and the candidate γ set
is **hardcoded to {1,2,3,4}** — so γ5 (a benign-only regime) can never be selected adaptively. Until
the update path is wired and the candidate set is opened, adaptive gamma cannot be measured as a
lever; it is parked, not attempted. No build, no budget.


## 5. P3 — loop-2 consumer-index-skip, PREFILL (next20) — **ALREADY SHIPPED; nothing to do**

### 5a. Finding: the lever is in production already

The Phase-4 brief describes P3 as "loop-2 consumer-index-skip, PREFILL … implement behind an env
knob; gate ≥8% of deep-context prefill wall". The audit it refers to is
`bench/dsv41_loop2/consumer_index_sizing.md` — "consumer-layer coarse-pass waste … up to ~15% of
wall, provably exact". That lever is **`coarse_block_scores_candidates` +
`DSV41_INDEXER_CONSUMER_SKIP`**, i.e. the consumer index layers (24/28/32/36) scoring only candidate
blocks in the coarse pass instead of all `nb` columns.

It was **implemented and shipped in `5a986da`** (loop-2, 2026-10-07; PERFORMANCE_HISTORY), and the
ancestry check confirms it is live:
```
git merge-base --is-ancestor 5a986da 6cc9c1e   -> IS ancestor (next13 pin)
git merge-base --is-ancestor 5a986da 3bf8316   -> IS ancestor (next17 / live pin)
installed prod module: indexer.py:138 _HIER_CONSUMER_SKIP = env("DSV41_INDEXER_CONSUMER_SKIP","1")==1
                       indexer.py:574 consumer_skip=_HIER_CONSUMER_SKIP
```
So the consumer-skip is **already the default in the live production build**, in its prefill role
(consumers run the coarse pass at prefill m=2048 where the O(offset) O(nb) sweep dominates). The
measured prefill wins are already recorded (soak13: r500 +29.0%, r750 +57.3%, r1m +65.5%; 350K build
+17.0%) — all far above the 8% bar.

### 5b. Verdict

**P3 = no-op / already-shipped.** There is no new work: the "loop-2 consumer-index-skip" the brief
asks to ship prefill-side is `5a986da`, already the default on `576e9d279`. No implementation, no
relaunch, no budget spent. (If the intent were the *variant* in the memo §4 — a single fp32 consumer
pass, not bit-equivalent — that is a different, output-changing lever and would need its own
identity/quality gate; it is **not** the exact lever the brief describes and is not attempted here.)

## 6. P5 — roofline close-out (bench-only, last)

Full report: `PHASE4-P5-ROOFLINE.md` (same dir). Summary of the verdicts:

- **(A) round accounting reconciled.** `verify_block 92.4 / round_total 93.8 = 98.5%` (server-internal)
  vs `92.4 / 101.06 = 91.4%` (client). The 8.66 ms splits MECE into **1.40 ms in-round server residual**
  (measured: draft 0.55 + tail 0.09 + 0.76 host) + **7.26 ms client↔server per-round boundary** (bounded,
  not measured — no PROF-capable build deployed). New finding: the raw OFF PROF is **bimodal** — the
  frozen 92.4 is the *benign* component (agentic component 97.81/99.27) — so the brief's "8.7 ms" is
  dominated by a benign-vs-agentic mismatch, not in-round overhead; the 98.5% ratio is arm-robust.
- **(B) bytes-roofline = NOT near-floor; ~79 ms headroom.** One verify round at 91K m=4 = **6.65 GB/rank**
  (routed-expert 4.46 + dense EXL3 2.04 + KV 0.15 + collectives 0.003). FLOPs floor 3.8 ms ≪ bytes floor
  → memory-bound. **Read bandwidth MEASURED at 497 GB/s** (pure-read/GEMV canary, both nodes; the old 450
  was a triad figure — read+write ≈304 here) → floor **13.4 ms**; verify **92.4 ms = 6.9× the floor**.
  Largest slice = **dense/shared EXL3 (≈34 ms, 8.5× its floor, decode-ALU/issue-bound)** → the lever;
  experts second (~1.5-3×). Localizing differential (design, not run): force top-k routing down; KV
  cannot move ms/round (0.2% of bytes). Read-bw canary retained: `read_bw_canary.py`.

## 7. RESTORE + end state

```
RESTORED deploy/next13 @ 576e9d279 (exo) + mlx-lm 3bf8316, gates UNSET, canary healthy
```
- Final restore (relaunch R3) 2026-10-08 18:48–18:58 CDT (`p4/restore_final.sh`). "Nodes synchronized on
  commit 576e9d279" → READY (2/2). **Both nodes** `git rev-parse HEAD` = `576e9d27994b…`; lever-1 guard
  present in the installed module (`grep -c` = 1 both). **All lever gates UNSET** on both runners
  (`DSV41_MOE_ALLSUM_BF16`, `DSV41_INDEXER_HIER`, `DSV41_SPARSE_COLSPLIT`, `DSV41_SPARSE_FENCE_MIN_ROWS`
  all absent → build defaults). Post-boot matmul canary **14.86/14.85** healthy.
- **Parity smoke** (benign 20K g3, 4 reps, restored boot): **118.69 ms/round / 31.73 t/s** median —
  matches the pre-Phase-4 production benign (118.56 ms) to **0.13 ms**. Production behaviour is
  bit-for-bit what it was before the campaign.
- No new tag: nothing shipped this round; the existing `known-good-decode-next17-20261008-160746`
  (exo `576e9d279` + mlx-lm `3bf8316`, both forks) remains the production tag.
- PERFORMANCE_HISTORY.md on main: entry committed + pushed (`f9f6c36ed`).

## 8. Resume pointer

Last commit on this doc says where we are. If resuming: read §3c / §4 checkboxes, `git log` this
branch, re-verify live state against §0, continue from the first unchecked box.
