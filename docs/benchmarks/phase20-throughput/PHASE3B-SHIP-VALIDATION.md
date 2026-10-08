# PHASE 20 — Phase-3B SHIP-VALIDATION (staged decode fix vs production)

Author: Phase-3B PM. Round opened 2026-10-08 ~11:30 CDT. **LIVE — in progress.**

Scope (from the brief): measure what `deploy/next17-levers` (mlx-lm `3bf8316`: the
small-m row-count guard on the C1 column-boundary derivation) actually delivers versus
production; split the lever-1 vs lever-2 shares of the known ~24 ms benign / ~52 ms
agentic **both-gates** win; run the R8a byte-identity battery on the real build; spec
(do NOT implement) the matching lever-2 code fix if it is worth >=3 ms/round; then RESTORE
production and report the RESTORED line.

## 0. What is being validated (source-verified this round)

- **PRODUCTION** = `deploy/next13` @ `f4bb14746`, mlx-lm pin `6cc9c1e`, gates unset.
  LIVE at round open: both nodes `git rev-parse HEAD` == `f4bb14746` (branch deploy/next13),
  `/state` live, canary 14.82–14.88 TFLOPS, idle. Production's `sparse_attention.py`
  contains `_FENCE_MIN_ROWS` but its C1 block reads `if kv2 is not None and _COLSPLIT:`
  (line 413) — **no row-count guard**. Installed `mlx_lm` in the node venv also has NO guard
  (`HAS_GUARD False` both nodes).
- **STAGED FIX** = `deploy/next17-levers` @ `576e9d279` (pushed to origin), one-line superproject
  diff vs production: mlx-lm gitlink `6cc9c1e` -> `3bf8316`. Real change in the fork:
  `sparse_attention.py` C1 block becomes
  `if kv2 is not None and _COLSPLIT and m > _FENCE_MIN_ROWS:` (`_FENCE_MIN_ROWS` default 16);
  small m (decode m=1, 4-row verify) skips the two host round-trips in `_column_boundary` /
  `_leading_window_columns` and takes the value-identical `_gather_split` fallback; large-m
  prefill keeps C1 unchanged. `attention.py` gets comments only. +315 lines of new tests.
- **LEVER 2** (untouched, still runs on decode): `DSV41_INDEXER_HIER` default `1`
  (`indexer.py:122`), entry `hierarchical_topk_prod` (`indexer.py:531/568`) gated only on
  the env var, NOT on row count.

## 1. Relaunch budget — DECLARED UP FRONT (3, per the fresh-round brief)

| # | deploy | purpose | status |
|---|---|---|---|
| #1 | `EXO_TARGET_BRANCH=deploy/next17-levers` (detach `576e9d279` + mlx-lm `3bf8316`, gates default) | measure next17-at-defaults | **SPENT** 2026-10-08 13:00:25 (exit=0, READY 2/2; prior PM, log `/tmp/p3b/deploy_next17.log`) |
| #2 | `EXO_TARGET_BRANCH=deploy/next17-levers` + `DSV41_INDEXER_HIER=0` | lever-2 split (OPTIONAL) | PLANNED |
| #3 | `EXO_TARGET_BRANCH=deploy/next13 ./start_cluster.sh` | RESTORE production f4bb14746 | MANDATORY |

If #1 fails READY/canary/verify, that failure consumes #3 (restore) and the round ends.
Hard cap 3. #2 is spent only if the lever-2 split is judged worth a relaunch; otherwise it
is explicitly skipped and #3 proceeds directly.

## 2. Protocol / gates (pre-registered BEFORE the judged runs)

- **Baseline first, on PRODUCTION as-is** (f4bb14746, gates default), then next17.
- Harness: the shipped `phase19_round_measure.py` (benign) + `phase19_agentic_measure.py`
  (agentic) from `/private/tmp/next16-instr/bench`, driven by `p3b_driver.py` under the
  `phase20_guard.py` ChunkGuard (own-request registration; idle-gate before every chunk;
  abort-on-arrival / wall-cap watcher; chunks <=15 min).
- `round_prof` is **not** passed: neither production nor next17 carries the per-request
  round_prof field (it exists only on `next16-instr`, and next17 is based on production, not
  on next16). Both arms are therefore measured **symmetrically** by client end-to-end
  ms/round + decode t/s + mean_accepted. Consequence: no per-bracket attribution is available
  on either arm; the C1 mechanism is instead pinned directly on the deployed code (§4).
- reps: benign >=5 (800 tok @20K), agentic >=5 (800 tok @91K real session replay, unique salt
  per rep => genuine cold prefill every rep, ~394 s/rep -> agentic chunks of 2 reps).
- **Gates:** next17-at-defaults beats production-at-defaults by **>=5 ms/round on agentic**
  (>=~10% of the 52 ms both-gates win) AND no benign regression AND R8a battery clean
  (needles+tools byte-identical to the frozen `g3` baseline; prose detectors clean).
- **Falsifier:** if next17 does NOT beat production, STOP and report — the staged fix is then
  wrong or inert, which is itself the finding.

## 3. Baseline (production f4bb14746, gates default) — DONE

Run 2026-10-08 11:34–12:52 CDT on the live production boot (f4bb14746, mlx-lm 6cc9c1e,
gates unset; both nodes verified). Symmetric protocol (no round_prof available on prod).
Raw artifacts: `raw/p3b/prod_benign.json`, `raw/p3b/prod_agentic.json` (+ `.recs.jsonl`),
env snapshots `raw/p3b/prod-env-studio{1,2}.txt` (103 vars, lever gates ABSENT).

| workload | reps | ms/round median (all) | decode t/s (all) | mean_accepted |
|---|---|---|---|---|
| benign 20K g3, 800 tok | 8 | **145.75** (145.08, 150.20, 145.80, 145.92, 145.42, 145.75, 145.21) | **25.55** (24.47–26.93) | 2.769 |
| agentic 91K g3, 800 tok | 6 | **157.10** (157.05, 157.18, 156.97, 157.10, 157.39) | **19.46** (18.50–20.39) | 2.050 |

Notes: rep 0 of each arm has no delta (first generation_stats frame) so it reports no
ms/round; n=8 → 7 benign ms/round values, n=6 → 5 agentic. Agentic reps are genuinely cold
(prefix_cache_hit none, ttft ~358–360 s each; ~394 s wall/rep) — the harness's unique salt
forces a fresh prefill each rep, so the 15-min chunk cap admits 2 reps/chunk. Baseline
matches the Phase-3 production control (benign 144.5 ms / agentic 156.6 ms) within noise →
the cluster is in the same thermal/state regime as the Phase-3 round.

## 3b. Post-deploy canary (relaunch #1 boot, 12:58) — DONE

The previous PM died 2s after `start_cluster.sh` exited, so the 12:58 boot never got a
post-boot canary. Run 13:04 CDT by the resuming PM:

- `phase20_guard.py canary` → **healthy**: studio1 median **14.86** TFLOPS (14.81/14.86/14.87),
  studio2 median **14.87** (14.81/14.87/14.87). No reboot needed (budget spend: none).
- `phase20_guard.py idle` → ok, both nodes `state=ready`, no running request, no post routes
  (clean post-deploy). `/state` serves on both nodes; `git rev-parse HEAD` == `576e9d279` both.
- Guard verified present in the installed module on both nodes
  (`and m > _FENCE_MIN_ROWS`, `site-packages/mlx_lm/models/deepseek_v41/sparse_attention.py`).

## 4. next17-at-defaults — DONE (both arms)

Run 2026-10-08 13:03:53–14:15 CDT on the live relaunch-#1 boot (`576e9d279`, mlx-lm `3bf8316`,
gates default; post-deploy canary healthy §3b). Symmetric protocol to the §3 baseline
(same flags, same harness, no round_prof; unique salt per rep ⇒ cold prefill each rep).
Raw artifacts: `/tmp/p3b/next17_benign.json` (+`.jsonl`), `/tmp/p3b/next17_agentic.json`.

| workload | reps | ms/round median | decode t/s | mean_accepted |
|---|---|---|---|---|
| benign 20K g3, 800 tok | 8 | **118.56** (118.50,118.83,118.75,118.56,118.39,118.86,118.23) | **31.47** (30.42–33.08) | **2.769** |
| agentic 91K g3, 800 tok | 6 | **130.07** (130.20,130.07,130.05,129.81,130.27) | **23.72** (23.07–25.11) | **2.077** |

### 3-gate check (pre-registered §2 vs production §3)

| gate | production | next17 | Δ | verdict |
|---|---|---|---|---|
| **agentic ≥5 ms/round faster** | 157.10 | 130.07 | **−27.03 ms (−17.2%)** | **PASS** (5.4× the bar) |
| **no benign regression** | 145.75 | 118.56 | **−27.19 ms (−18.7%)** | **PASS** (large improvement) |
| **mean_accepted unconfounded** | 2.7689 / 2.0496 | 2.7689 / 2.0769 | identical / +0.027 | **clean** |

`mean_accepted` is **identical to 4 decimals on benign** (2.7689) and marginally *higher* on
agentic (2.0769 vs 2.0496) — so the ms/round win is a pure per-round latency reduction, not a
spec-decode/acceptance artefact. The staged fix (`deploy/next17-levers`) **beats production on both
arms**; the falsifier (next17 inert) is **not** triggered. Decode t/s: benign +23%, agentic +22%.

### Residual vs the M3 lever measurement (why the split is worth spending)

M3 (`PHASE3-M3.md`, next16-instr, **both** levers OFF) measured benign **94.9** / agentic **101.3**
ms — i.e. the *full* both-levers win. next17-at-defaults (lever-1 code shipped, **lever-2 still
ON**) sits at 118.56 / 130.07, ≈27 ms *above* the both-off build and ≈ M3's same-build levers-ON
baseline (119.16 benign). So the ~27 ms next17 recovered ≈ **lever-1's share**, and the residual
next17-defaults − M3-OFF ≈ **24 ms benign / 29 ms agentic** is (by subtraction) lever-2's share.
That is far larger than M3 §3's *assumed* "small, context-flat" lever-2 — a load-bearing
contradiction, and the reason the optional lever-2 split (§5) is judged **worth relaunch #2**.

## 5. Lever-2 split — PENDING / maybe skipped

## 6. R8a battery on next17 — PENDING

## 7. Lever-2 code-fix spec — PENDING (only if worth >=3 ms/round)

## 8. RESTORED line — PENDING
