# PHASE 5 — R1c SHIP ROUND: RESULTS (FINAL — quality-battery-governed, owner ruling)

Session: R1c (Phase-5 final ship round), 2026-10-08/09 CDT. Operator: R1c subagent.
Worktree: `/private/tmp/phase20-campaign` (branch `deploy/phase20-campaign`).
Status: **SHIPPED-IN-PLACE.**

## 0. TL;DR / verdict

**OWNER RULING (verbatim, 2026-10-08):** *"as long as we pass our quality tests then I'm fine with it."*

Under that ruling the **R8a quality battery is the GOVERNING SHIP GATE** for lever-2. The strict
greedy token-identity gate is **SUPERSEDED by explicit owner decision**; the deterministic output
change at token 63/300 vs production (proved in R1b) is **ACCEPTED** by the owner. R1b already
proved the perf win cleanly (agentic 91K **130.38 → 99.30 ms**, −31.08 disjoint; benign 118.86 →
94.93 ms).

| item | result |
|---|---|
| production artifact | exo `deploy/next18-identity @ fb4f9290b` (mlx-lm gitlink → `16830e1`), pushed to `origin` (adurham/exo) |
| deploy | exo `fb4f9290b` + mlx-lm WT `16830e1`, ALL lever/capture env UNSET, READY (2/2), canary healthy |
| installed-module verify | BOTH nodes: lever-2 guard + `_L2_FULL` in the installed indexer.py; lever-1 guard in installed sparse_attention.py; runner env has **zero** `DSV41_*` keys |
| **battery (G3, governing)** | **CLEAN** — needles **6/6**, tools **10/10**, prose **0 DIRTY / 0 REVIEW**, park **recall_teal=True**; `compare.py` vs frozen `g3` → **no fails** (1 benign REVIEW: park recall improved in B) |
| ship decision | **SHIPPED-IN-PLACE** — build left LIVE as production; tag on BOTH forks |
| budget | **1/2 spent** (ship deploy); contingency restore HELD/unspent |
| open item | token-63 divergence root cause NOT fully explained — see §8 |

## 1. Budget (declared UP FRONT; ≤2 this ship round)

| # | purpose | deploy | status |
|---|---|---|---|
| 1 | SHIP deploy (all gates unset) | exo `fb4f9290b` + mlx-lm `16830e1` | **SPENT** |
| 2 | contingency RESTORE (only on DIRTY) | exo `576e9d279` + mlx-lm `3bf8316` | **HELD** (not spent — battery CLEAN) |

The R1c ship round spent **1 of its declared 2** relaunches (ship deploy only; no DIRTY, so the
contingency restore was not needed). (Historical context: R1/R1b/R1b-restore are separate, earlier,
already-closed rounds — see `PHASE5-R1-RESULTS.md`, `PHASE5-R1B-RESULTS.md`.)

## 2. Production artifact (step 1)

Made lever-2 a proper installable default-ON build so the default install carries it (next17
gitlink pattern):

- **mlx-lm** — `deploy/next18-lever2 @ 16830e1` (already pushed to `origin` = adurham/mlx-lm).
  Contains L2-full (`cd68bf4`) + adversarial suite/attribution (`17bbd98`) + the off-thread capture
  harness fix (`16830e1`). Verified `git -C /private/tmp/next18-lever2 log --oneline -1 16830e1`
  → `16830e1 next18 capture: flush OFF the request thread + runtime timing-arm gate (R1b)`.
- **exo** — `deploy/next18-identity`: bumped the mlx-lm gitlink `3bf8316` → `16830e1`
  (commit **`fb4f9290b`**), pushed to `origin` (adurham/exo).
  `git ls-tree fb4f9290b mlx-lm` → `160000 commit 16830e1739775eec89272853c90ec44de1156c21  mlx-lm`.
  The branch is `576e9d279` + the capture-launcher forwarding patch (`34a79dba1`, `ff676b3ca`).
- All lever gates **default ON** (`DSV41_INDEXER_L2_FULL` module default `1`; the lever-2 guard
  predicate `n > _FENCE_MIN_ROWS`, `_FENCE_MIN_ROWS` default 16); capture **OFF** (env unset).

No upstream repo was touched (adurham forks only).

## 3. Deploy (relaunch #1/2 — SPENT)

- Pre-flight: idle guard `ok=true`; both nodes `HEAD=576e9d279 MLX=3bf8316`; canary **14.87 / 14.86**
  healthy.
- `git checkout --detach fb4f9290b` + `git -C mlx-lm checkout --detach 16830e1`; working-tree guard
  check (lever-2 guard = 1, `_L2_FULL` = 3, lever-1 guard = 1); all `DSV41_*`/`EXO_*` gate vars
  unset; `EXO_TARGET_BRANCH=deploy/next18-identity ./start_cluster.sh`.
- `Nodes synchronized on commit fb4f9290b.` → HEALTHY → **READY (2/2)**.
- Post-boot canary: **studio1 14.84 / studio2 14.84** healthy (exit 0).

## 4. Installed-module + runner-env verify (BOTH nodes)

| check | studio1 | studio2 |
|---|---|---|
| exo / mlx-lm-wt | `fb4f9290b` / `16830e1` | `fb4f9290b` / `16830e1` |
| installed indexer.py lever-2 guard (`if _HIER and n > _FENCE_MIN_ROWS`) | 1 | 1 |
| installed indexer.py `_L2_FULL` occurrences | 3 | 3 |
| installed sparse_attention.py lever-1 guard (`and m > _FENCE_MIN_ROWS`) | 1 | 1 |
| runner `DSV41_*` keys | **(none)** | **(none)** |

The installed module resolved to the site-packages path on each node
(`…/site-packages/mlx_lm/models/deepseek_v41/indexer.py`), not the gitlink — i.e. the code that is
actually RUNNING was verified. Pure default config: no `DSV41_INDEXER_L2_FULL` /
`DSV41_INDEXER_HIER` / `DSV41_INDEXER_SMALLN_ROW_BF16` / `DSV41_SPARSE_FENCE_MIN_ROWS` /
`DSV41_NEXT18_CAPTURE*` keys present in the runner env.

## 5. Battery (G3 — the governing gate), live build, standard depth

Harness `/private/tmp/next16-instr/bench/dsv41_quality_battery/battery.py` run per its runbook;
`--label r1cship --depth 40000 all` (same depth as the frozen `g3` baseline); capture hook OFF
(runner env has no `DSV41_NEXT18_CAPTURE`).

```
[build]  status=200  est_tokens=40075   (cached 38912)
[needles] 6/6 pass   (exact, paraphrase, negation, distractor, control, multihop)
[prose]   20 prompts, 0 DIRTY, 0 REVIEW   (same_script_glue advisory: 5)
[tools]   10/10 pass
[park]    turn2 recall_teal=True
PHASE VERDICT: CLEAN  (needles 6/6, tools 10/10, prose DIRTY 0, REVIEW 0)
```

`compare.py results/g3 results/r1cship` → **verdict REVIEW, fails = []**, one REVIEW note
("parked-restore recall improved in B"). The frozen `g3` baseline carries no park record
(`park_recall: null`), so the candidate's `recall_teal=True` reads as an improvement, not a
regression. Free-prose texts differ between arms at temp 0 — per the battery README this is a
**printed note, not a gate**. **No hard failure.**

**Gate assessment:** per the pre-registered G3 (`summary.json phase_verdict == "CLEAN"` → needles
6/6, tools 10/10, prose 0 DIRTY/0 REVIEW, park PASS): **all four categories PASS → G3 PASS.**

## 6. Decision (step 4)

**CLEAN ⇒ SHIP-IN-PLACE.** The build **`exo deploy/next18-identity @ fb4f9290b` + `mlx-lm 16830e1`
(gates unset)** is left LIVE as production. Tagged on BOTH forks:

```
known-good-decode-next18-20261009-001052
  exo     -> fb4f9290b4e0b9caa2052f4fd66453d8b6c2df6c   (adurham/exo)
  mlx-lm  -> 16830e1739775eec89272853c90ec44de1156c21   (adurham/mlx-lm)
```

## 7. Parity smoke (live shipped build)

Benign 20K ×3 + agentic 91K ×2, fixed salt `r1fix-a`, same harness as R1/R1b. Idle-guarded;
0 aborts.

| arm | ms/round median | decode t/s | expectation (R1b) |
|---|---|---|---|
| benign 20K | **94.94 ms** [95.15, 94.74] | **39.13** | ~94.9 ms / ~40 t/s — **matches** |
| agentic 91K | **99.46 ms** [99.46] | **31.03** | ~99.3 ms / ~31 t/s — **matches** |

Within noise of the R1b lever numbers ⇒ the shipped build reproduces the lever's perf win at the
production shape. `mean_accepted` reproduces the R1b/lever values (benign 2.8239 median, agentic
2.0651).

## 8. OPEN ITEM (kept honest — do not paper over)

The **token-63 divergence root cause is NOT fully explained.** R1b's 91K capture replay showed only
**64/128** records replayable with **0 diffs on those 64** (`L2full_vs_hier_total_ndiff: 0`), yet the
live greedy trajectory diverges deterministically at token 63/300. The offline harness does not
faithfully reproduce the live path, so the replay's "0-diff" is a partial net. The 64 non-replayable
records are the missing surface. **Recorded as an explicit open item for the next round: harden the
capture/replay harness so it models the live path (and re-derive whether HIER (production) ≠
fp32-full-row (lever) is fully accounted for).** Under the owner ruling this open item does NOT
block the ship (the quality battery governs), but it must not be lost.

## 9. Final cluster state

- **LIVE (SHIPPED):** exo `deploy/next18-identity @ fb4f9290b` + mlx-lm `16830e1`, BOTH nodes;
  all lever/capture env **UNSET** (runner env shows no `DSV41_*` keys); canary **14.85 / 14.85**
  healthy; **2 runners RunnerReady**; no stray bench processes on either node.
- **Relaunch:** R1c = **1/2 spent** (ship deploy). Contingency restore not needed.
- **Code pushed:** exo `origin/deploy/next18-identity @ fb4f9290b`; mlx-lm
  `origin/deploy/next18-lever2 @ 16830e1`. Tags pushed to both forks.

## 10. Evidence IDs

| ID | artifact |
|---|---|
| E-R1C-ARTIFACT | exo `fb4f9290b` (gitlink `16830e1`), pushed `origin/deploy/next18-identity` |
| E-R1C-PREFLIGHT | idle ok; `576e9d279`/`3bf8316` both; canary 14.87/14.86 |
| E-R1C-DEPLOY | `/tmp/p5r1c/deploy_r1c.log` — synced `fb4f9290b`, READY (2/2), canary 14.84/14.84 |
| E-R1C-INSTALL | installed-module + runner-env both nodes (§4) |
| E-R1C-BATTERY | `/private/tmp/next16-instr/bench/dsv41_quality_battery/results/r1cship/` (`summary.json` CLEAN, `compare.json` vs `g3`) |
| E-R1C-SMOKE | `/tmp/p5r1c/ship_smoke_{benign,agentic}.json` — 94.94 ms / 99.46 ms |
| E-R1C-TAG | `known-good-decode-next18-20261009-001052` on adurham/exo + adurham/mlx-lm |
| E-R1C-OPEN | §8 token-63 open item |
