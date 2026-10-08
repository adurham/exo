# PHASE 20 — Phase-3B SHIP-VALIDATION (staged decode fix vs production)

Author: Phase-3B PM (round 2, resumed after the round-1 PM died post-deploy). Round opened
2026-10-08 ~11:30 CDT, resumed 13:03 CDT, closed 15:35 CDT. **COMPLETE — production restored.**

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

## 5. R8a battery on next17 — DONE (CLEAN)

`battery_next17.sh` → `battery.py --label p3bnext17 --depth 40000 all` on the **live relaunch-#1
boot** (next17-levers @ `576e9d279`, gates default), 2026-10-08 14:15–14:30 CDT, idle-gated,
exit=0. Artifacts: `raw/p3b/battery_p3bnext17/`, `raw/p3b/battery_next17.stdout`.

| metric | frozen g3 baseline | **p3bnext17 (this build)** | verdict |
|---|---|---|---|
| needles | 6/6 | **6/6** (exact, paraphrase, negation, distractor, control, multihop) | identical ✅ |
| tools | 10/10 | **10/10** | identical ✅ |
| free prose | 0 DIRTY / 0 REVIEW | **0 DIRTY / 0 REVIEW** (20 probes) | clean ✅ |
| advisory `same_script_glue` | **4** | **4** | identical ✅ |
| `phase_verdict` | CLEAN | **CLEAN** | ✅ |
| parked-restore recall | `None` (not run in g3) | **`recall_teal=True`** (turn2 9.0 s) | improved |

`compare.py results/g3 results/p3bnext17` → **VERDICT: REVIEW**, with exactly **one** REVIEW item:
*"parked-restore recall improved in B"* — i.e. the candidate is **better**, not worse (g3 simply did
not run the park phase). No FAIL items: no new detector hits, no needle/tool regression, no lost
recall. Per PREREG D4 this is a **byte-identity PASS on the deterministic subset** (needles+tools)
with prose detectors clean and the advisory count identical. **R8a gate: PASS.**

**Ship gate (§2) — all three legs PASS:** agentic −27.03 ms/round (≥5 ms bar, 5.4×), no benign
regression (−27.19 ms), R8a clean. The staged fix beat production and its output is
quality-identical. **Recommendation: SHIP `deploy/next17-levers` as the production line** (§8).

## 6. Lever-2 split — DONE (relaunch #2)

| arm (agentic 91K g3, 800 tok) | reps | ms/round median | decode t/s | mean_accepted |
|---|---|---|---|---|
| production (HIER=1, no levers) | 6 | 157.10 | 19.46 | 2.050 |
| next17-defaults (HIER=1, lever-1 code) | 6 | 130.07 | 23.72 | 2.077 |
| **next17 + `DSV41_INDEXER_HIER=0`** (lever-1 code, HIER off) | 4 | **101.06** (101.06,101.07,100.67) | **30.24** (29.62–31.23) | 2.027 |

### Split (same-build, clean)

```
total win vs production  = 157.10 − 101.06 = 56.04 ms/round   (35.7%)
  lever 1 (code guard)   = 157.10 − 130.07 = 27.03 ms/round  (48% of total)
  lever 2 (indexer HIER) = 130.07 − 101.06 = 29.01 ms/round  (52% of total)
  sum of parts           = 56.04 ms  ==  total ✓ (cleanly additive, no interaction term)
```

**Lever-2 is NOT small** — it is ~29 ms/round, **5–8× M3 §3's assumed "small, context-flat"
contribution**, and it is marginally *the larger* lever. M3's assumption was wrong; this is the
load-bearing correction of this round. The same-build HIER=0 point (101.06) reproduces M3's
**cross-build** both-off number (101.3) to 0.24 ms — so the cross-build confound in M3 §2b was real
but small, and the ~29 ms lever-2 share is robust. `mean_accepted` held (2.03 vs next17-def 2.08 vs
prod 2.05) → the split is a latency effect, not an acceptance artefact.

**Consequence:** the lever-2 code fix (§7) is worth a relaunch-sized investment — 29 ms/round at
91K, and larger at deeper context (the same ctx-scaling that made lever-1 bigger on agentic).

## 7. Lever-2 code-fix spec — **SPEC ONLY, NOT IMPLEMENTED** (lever-2 = 29.0 ms/round ≥ 3 ms bar)

Warranted: lever-2's measured share (29.0 ms/round at 91K agentic) is 9.7× the 3 ms shipping bar.

**Where (exact call site):** `mlx_lm/models/deepseek_v41/indexer.py`, `Indexer.__call__`, the branch at
**line 531 `if _HIER:`**. Today it is gated *only* on the import-time env
`_HIER = os.environ.get("DSV41_INDEXER_HIER","1")=="1"` (line 122) — **no row-count guard**, which is
the whole lever: the hierarchical path's coarse-pass + streamed-exact machinery is pure overhead at
decode (`n=1`) and verify (`n=4`), where the fallback's score row is a single/small row.

**The fix (mirror the lever-1 guard exactly):**
```python
# indexer.py __call__: n := x.shape[1] (query-row count; same quantity as m in sparse_attention)
if _HIER and n > _FENCE_MIN_ROWS:      # was:  if _HIER:
    ... hierarchical_topk_prod(...) ...        # lines 531-577, unchanged body
# else: falls through to the existing tiled path (line 579) / untiled reference (line 603)
```
- Reuse the **same** `_FENCE_MIN_ROWS` symbol already proven for lever-1
  (`sparse_attention.py:119` = `int(os.environ.get("DSV41_SPARSE_FENCE_MIN_ROWS","16"))`). Import it,
  or hoist to a shared `deepseek_v41/_gates.py`, so **both levers share one threshold and one
  default** — this retires the "two disagreeing module defaults" foot-gun that M3 §7.4 flagged.
- One edit covers **both** roles: the candidate-source and candidate-consumer branches are the *same*
  `if _HIER:` block (lines 531–577). **Critical:** source layers (2/8/14/20) and consumer layers
  (24..36) all see the *same* `n` within one forward, so the single `n > _FENCE_MIN_ROWS` predicate
  keeps the published `shared.candidates` mask (`model.py:80`, per-forward) produced and consumed on
  the same path. **Do not** gate source and consumer on different predicates.
- Env overrides keep working for A/B: `DSV41_INDEXER_HIER=0` still forces HIER off unconditionally.

**Required value-identity proof BEFORE shipping (load-bearing).** Lever-1 was safe because
`_gather_split` is *stated byte-identical* to the C1 derivation. Lever-2 is **not** obviously
identity-preserving: HIER ranks blocks by **bf16 coarse maxima** and fp32-exact-scores only the
surviving top-`(k+overfetch)` blocks, while the fallback exact-scores **all** columns. They select the
same top-k **iff no true-top-k column lies in a block the coarse pass dropped** — the `overfetch=16`
margin is a heuristic, **not** a proof. So the fix ships only with:
1. **Direct index-diff test** `test_dsv41_indexer_smallm_hier.py`: over a grid of `n∈{1,2,3,4}` ×
   `nb∈{64,512,4096,16384}` × many seeds, assert `hier_topk` ≡ `tiled_topk` **elementwise on the
   returned `[b,n,k]` int32 index tensor** (including the `-1` mask pattern and position order) — the
   value-identity proof, per PREREG G-P3.3.
2. **Same-build A1-vs-A2 determinism replicate** (PREREG D4) so real GPU drift cannot masquerade as
   identity.
3. **R8a battery** byte-identical on the deterministic subset (needles+tools) + detectors clean, on
   the *new* build (the §5 battery is HIER=1 and does **not** cover this change).
4. **Abort rule:** if step 1 shows *any* index divergence at small m, **do NOT ship the guard** —
   report lever-2 as a genuine speed-vs-output tradeoff and re-validate quality end-to-end. Never
   paper over a divergence with the overfetch margin.

**Guard / rollback:** pure-additive (one extra conjunct on an existing `if`); blast radius = 1 import
+ 1 condition. `DSV41_SPARSE_FENCE_MIN_ROWS=0` restores historical always-HIER; `DSV41_INDEXER_HIER=0`
forces off. Both remain.

**Expected value:** ≈29 ms/round at 91K agentic (measured §5), larger at deeper ctx per lever-1's
ctx-scaling. Combined with the shipped lever-1, the both-defaults build should reach ≈101 ms/round
agentic ≈ **30 t/s at gamma 3**.

## 8. RESTORED line, recommendation, and round close

### RESTORED

```
RESTORED f4bb14746 READY 2/2 canary 14.85/14.73 TFLOPS parity decode=25.70 (benign g3, 144.82 ms/round) prefill=282.4 rows/s
```

**Relaunch #3** SPENT 2026-10-08 15:22–15:27 CDT (`restore_next13.sh`, idle_ok=True; log
`/tmp/p3b/restore_next13.log`). Verified **independently by the resuming PM** (not from the script's
own echo):

- Both nodes **and** the laptop repo: `git rev-parse HEAD` = `f4bb14746c68deea005f41f590e27e6b182b6384`,
  branch **`deploy/next13`**, mlx-lm pin = **`6cc9c1e8709e228fca99ac152cd5e681ddcce65d`**. Launcher
  "Nodes synchronized on commit f4bb14746" → **READY (2/2)** @ 15:27:33, exit=0.
- Post-boot canary: studio1 14.83/14.87/14.85, studio2 14.72/14.73/14.74 → **healthy**. `/state` and
  `idle` (ok) both healthy.
- **Guard absent** from the installed `sparse_attention.py` on both nodes (`grep -c` = 0) → the
  lever-1 code is gone, production behaviour restored.
- **Env parity:** live runner var-name set == the pre-campaign `raw/p3b/prod-env-studio{1,2}.txt`
  snapshots (103/103 identical) **plus** a benign extra `LOG_LEVEL=INFO` (present on all boots,
  including the ones that produced the baseline). **Both lever gates (`DSV41_INDEXER_HIER`,
  `DSV41_SPARSE_COLSPLIT`) are ABSENT** in the live env → production defaults.
- `docs/benchmarks/phase19-latency/` moved back on the laptop; the deploy-state conflict is resolved.

**Parity smoke (PREREG §3) — both gates PASS on the restored boot** (2026-10-08 15:30–15:54 CDT,
idle-gated; `raw/p3b/restore_smoke.log`, `/tmp/p3b/restore_benign.json`):

| smoke | gate | restored boot | verdict |
|---|---|---|---|
| benign 20K g3, 4 reps | ≥ 24.0 t/s | **25.70 t/s** median (25.04–26.65), **144.82 ms/round** | **PASS** |
| fresh ~98K prefill | ≥ 260 rows/s | **282.4 rows/s** (ttft 348.8 s, 98 523 tok) | **PASS** |

The restored boot's benign round time (**144.82 ms**) matches the §3 production baseline
(**145.75 ms**) to 0.9 ms — independent, on-cluster confirmation that the lever-1 code is gone and
production behaviour is bit-for-bit what it was before the campaign.

### Recommendation — DEFAULT POLICY: **SHIP `deploy/next17-levers` as the production line**

Pre-registered ship gate (§2) is met on every leg (§4): agentic **−27.03 ms/round** (bar was ≥5 ms,
5.4× cleared), no benign regression (**−27.19 ms**), R8a **clean** (needles 6/6, tools 10/10, prose 0
dirty, advisory count identical to `g3`). `mean_accepted` identical/higher → not an acceptance
artefact. The fix is the shippable *code* form of lever 1 (no env flip, no foot-gun).

**Second, larger finding — spend a relaunch on lever-2.** The split (§6) shows the indexer
hierarchy (`DSV41_INDEXER_HIER`) is **29.0 ms/round** at 91K agentic — **52%** of the 56 ms total,
**the larger lever**, and **5–8× the "small" value M3 §3 assumed**. A matching code guard (§7,
`indexer.py:531` → `if _HIER and n > _FENCE_MIN_ROWS:`) would make the both-levers-off behaviour the
default **without any env flip**, targeting ≈101 ms/round ≈ **30 t/s at gamma 3** agentic (from
production's 157 ms / 19.5 t/s). **It is specced, not implemented** (as instructed); shipping it
requires the value-identity proof in §7 first, because lever-2's coarse-then-exact path is *not*
obviously bit-identical to the full-width fallback.

### Round close

- Relaunch budget: **3/3 spent** (#1 next17-defaults, #2 next17+HIER=0, #3 production restore); cap
  respected, no overrun, no degraded-canary reboot needed.
- All artifacts under `raw/p3b/` (baseline §3, next17 §4, battery §5, HIER=0 §6, env snapshots).
- Cluster returned to production `f4bb14746`, both nodes serving, gates unset, idle.
