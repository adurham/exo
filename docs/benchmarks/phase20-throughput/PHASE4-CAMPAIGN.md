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

See its own section, appended below.


## 7. Resume pointer

Last commit on this doc says where we are. If resuming: read §3c / §4 checkboxes, `git log` this
branch, re-verify live state against §0, continue from the first unchecked box.
