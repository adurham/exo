# PHASE 5 — P1 AMENDMENT EVIDENCE PACKAGE
## Amendment of the frozen lever-2 index-identity gate (Phase-4 §3b)

Author: Phase-5 P1 subagent (depth-2). 2026-10-08. **ALL WORK OFFLINE** — no cluster contact,
no relaunch, no deploy, no API POST. Production remains `deploy/next13 @ 576e9d279` + mlx-lm
`3bf8316`, gates unset, **untouched**. 0 relaunches spent.

**What this document is.** The frozen gate (Phase-4 `PHASE4-CAMPAIGN.md` §3b) requires elementwise
**index-identity** between the hierarchical and the fallback paths, *"ABORT on any divergence (ties
count as divergence)"*. It FAILS 251/2045 on the L2-full design. The P1 memo
(`PHASE5-P1-LEVER2.md`) diagnosed that failure as a **synthetic-fixture artifact** (the suite's
`index_n_heads=2` exact-zero tie class), not a design defect. This package is the **evidence** a
plan owner needs to decide whether to amend the gate. **It is a USER / gate-owner decision. The
amendment below is PROPOSED, pre-registered, and NOT APPLIED by us.**

**Bottom line.** Three independent lines of evidence say the residual is a head-count artifact:
(1) the zero-column census follows `2^-H` exactly and the divergence vanishes at `H ≥ 8` (§2);
(2) against a full-width **fp32 truth row**, L2-full/fallback loses **0 value slots** — it is HIER
that loses 48 slots in one cell (§3); (3) the incumbent **also** fails the frozen gate at the
fixture's `H=2` (939/2045), so index-identity does not hold for production *today* at `H=2`
either (§6). Proposed amendment text + pre-registered failure branch: §9, §10.

---

## 0. Provenance / how every number was produced

| item | path |
|---|---|
| mlx-lm worktree (L2-full code) | `/private/tmp/next18-lever2` @ `cd68bf4`, branch `deploy/next18-lever2` |
| frozen suite | `tests/test_dsv41_indexer_smallm_hier.py` (2045 cells; `_args()` line 95 sets `index_n_heads=2`; prints `DSV41_SMALLM_HIER`) |
| probes | `bench/next18_{prodgate,headcount,classify_all,losscell,g0_probe,replay_capture,gap_hist}.py` |
| raw logs (scratch, not committed) | `/Users/adam.durham/.hermes/cache/scratch/p5/` (see §12) |
| this package's own capture | `/Users/adam.durham/.hermes/cache/scratch/p5/amend/replay_capture.log` |

Every table row below cites its raw-log source and/or the probe command (§11).

---

## 1. The frozen gate as written (quoted, for the record)

> `PHASE4-CAMPAIGN.md:63` — **"### 3b. Required proof suite (ABORT on any divergence; ties count as
> divergence)"** … `:65` "Elementwise index-identity: hier topk ≡ tiled/untiled fallback on the
> returned `[b,n,k]` int32 tensor (incl. `-1` mask pattern + position order) over synthetic grid:
> n ∈ {1,2,3,4} ∪ {16,17} …, many seeds."

The rule is binary and admits no granularity: **any** differing index, `-1` slot, or slot order is a
FAIL. Under it, the L2-full design is a **determinate FAIL** (251/2045), and so is the shipped
incumbent (939/2045 at the same fixture). That is the state the gate owner is being asked to rule on.

---

## 2. Correction first: production head count is **32**, not 64

The P1 memo §6 labels production as `H=64`. **That is a mislabel.** Production is
`index_n_heads = 32`, verified two independent ways:

* mlx-lm default: `mlx_lm/models/deepseek_v41/config.py:82` → `index_n_heads: int = 32`
  (and `:188` `_get(c, "index_n_heads", default=32)`).
* the real node `config.json` (PM-verified, `PHASE5-CAMPAIGN.md:123-125`).

The suite fixture sets `index_n_heads=2` (`tests/test_dsv41_indexer_smallm_hier.py:95`).
**The conclusion is unchanged by the correction:** the head-count sweep reports **0 index diffs at
H = 8, H = 32, AND H = 64** (258,048 slots each — `prodgate_full.log:12-14`). Since the design is
clean at *every* H ≥ 8, which value production actually uses (32 vs 64) does not move the result —
but the package states the **true** value = **32** so the record is right.

---

## 3. TABLE 1 — Head-count class table (the load-bearing evidence)

Zero-column fraction is the fraction of *visible* columns scoring **exactly 0.0** (a column is 0 iff
every head's pre-ReLU dot ≤ 0, `P ≈ 2^-H`). Diffs are `L2-full`/`bf16-fallback` vs the shipped HIER
path, on identical inputs. **Source: `prodgate_full.log:2-14`, `headcount.log:5-9`.**

| H (index_n_heads) | zero-col fraction (measured) | `2^-H` | L2-full vs HIER diffs | bf16-fallback vs HIER diffs | slots measured | source |
|---|---|---|---|---|---|---|
| 2 (fixture) | **0.2455** | 0.25 | **1187** | **1187** | 193,536 | `headcount.log:5` |
| 4 | **0.0594** | 0.0625 | **528** | **528** | 193,536 | `headcount.log:6` |
| 8 | **0.00415** | 0.00391 | **0** | **0** | 193,536 (sweep) / 258,048 (stability) | `headcount.log:7`, `prodgate_full.log:12` |
| 16 | 0.0 | 1.5e-5 | **0** | **0** | 193,536 | `headcount.log:8` |
| **32 (PRODUCTION)** | 0.0 | 2.3e-10 | **0** | **0** | 193,536 (sweep) / 258,048 (stability) | `headcount.log:9`, `prodgate_full.log:13` |
| 64 | 0.0 | 5.4e-20 | **0** | **0** | 258,048 (stability) | `prodgate_full.log:14` |

Census (`prodgate_full.log:2-7`): zero-col fractions 0.2455 / 0.0594 / 0.00415 at H=2/4/8 vs the law's
0.25 / 0.0625 / 0.00391 — the `2^-H` law is confirmed, and the zero-count is **0 at H ≥ 16**.
The head-count sweep (`headcount.log`) spans H ∈ {2,4,8,16,32}; the production-H **stability** sweep
(`prodgate_full.log:8-14`, 8 seeds × n∈{1,4,16} × nb∈{512,4096,16384}) supplies H = 64.

**Reading.** The divergence is a monotone function of the exact-zero tie class, which decays as
`2^-H` and is *gone* (0/258,048) at every H ≥ 8 — i.e. at production. At the fixture's H=2, ~24.5 % of
columns are exact zeros; no two finite-precision implementations break that tie class identically.

> **Granularity note (not a discrepancy).** The H=2 sweep reports **1187 *slot*** diffs over 54 cells
> / 193,536 slots (`headcount.log:5`); the 2045-cell *suite* reports **939 *cell*** divergences at its
> H=2 (`suite_bf16_1.log`). Same phenomenon — a cell is pass/fail over its `n·k` slots — on the same
> fixture family with different seeds/shapes. No cherry-pick; the sweep's job is the H-trend.

---

## 4. TABLE 2 — Value-loss matrix vs full-width fp32 truth

Sound metric: each arm's returned **value multiset**, per query row, sorted descending, compared
rank-wise against the full-width fp32 score row `R` (the oracle). A "loss" = a finite truth rank where
the arm's value sits strictly more than `tol` (1e-6) below truth. **Source: `classify_all_fixed.log`
(+ `losscell.log`, `g0_probe.log`).**

| arm | value-loss slots vs fp32 truth | where | source |
|---|---|---|---|
| **fallback = L2-full** | **0** | — (never loses a slot) | `classify_all_fixed.log:96` |
| bf16-row fallback (shipped) | **0** | — | `losscell.log:3` |
| **hierarchical (HIER)** | **48** | **exactly 1 cell**: `plain n=16 nb=16384 k=513 seed=385721` | `classify_all_fixed.log:6,97` |
| max `fallback − hier` per rank | **9.73e-6** | same cell (pure fp32 non-associativity) | `classify_all_fixed.log:6,96` |

Head-line (`classify_all_fixed.log:96`): `fallback_loss_slots_vs_truth: 0`, `hier_loss_slots_vs_truth:
48`, `max_fb_minus_hier: 9.73e-06`. **It is HIER that deviates from truth in the worst cell, not the
fallback** (`losscell.log:2-6`: fallback fp32-row loss 0 / worst 0.0; HIER loss 16 / worst 1e-5;
`hier_vs_fallback_fp32row` slots below = 16, worst 9.73e-6). All 250 other residual cells are **exact
fp32 value ties** (verdict `VALUE-TIE`) — same value multiset, different tie-break index order.

**Per-stratum (role × n) attribution of the value loss.** Only one stratum contains a value loss
(`plain/n=16`, 1 cell); every other stratum is value-tie only:

| stratum (role / n) | cells | fallback value-loss slots | HIER value-loss slots | note |
|---|---|---|---|---|
| plain / n=2 | 1 | 0 | 0 | values tie (index tie-break only) |
| plain / n=3 | 7 | 0 | 0 | values tie |
| plain / n=4 | 54 | 0 | 0 | values tie |
| **plain / n=16** | 61 | **0** | **48** (1 cell: nb=16384,k=513,seed=385721) | fp32 assoc-order; HIER deviates |
| consumer / n=4 | 64 | 0 | 0 | value tie |
| consumer / n=16 | 64 | 0 | 0 | value tie |

(Stratum cell counts from `l2full_run1.log` `by_role_n`; per-cell verdicts from `classify_all_fixed.log`.)

**Independent corroboration — G0 probe** (`g0_probe.log:50`), HIER vs a full-width fp32 truth row over
**48 cells / 141,312 slots**: `hier_VALUE_LOSS_vs_fp32truth: 0`, `hier_idiff_vs_fp32truth: 52` (one
cell, exact ties). So the shipped HIER architecture itself is value-exact against truth in that
regime — a value-identity gate is satisfiable.

> **Metric note.** The worst cell's **index** diff is 46 slots, while its **value** loss vs truth is
> **48** slots — these are two distinct quantities (index-set disagreement vs value-multiset
> shortfall), both recorded in `classify_all_fixed.log:6`.

---

## 5. TABLE 3 — Determinism replicates at production H

Identical inputs, repeated calls, at **H = 64** (production class). **Source:
`prodgate_full.log:15-21`; whole-grid replicate `determinism.log:5`.**

| shape (n, nb) | L2-full self-diff | HIER self-diff | L2-full vs HIER cross-diff |
|---|---|---|---|
| n=1, nb=4096 | 0 | 0 | 0 |
| n=1, nb=16384 | 0 | 0 | 0 |
| n=4, nb=4096 | 0 | 0 | 0 |
| n=4, nb=16384 | 0 | 0 | 0 |
| n=16, nb=4096 | 0 | 0 | 0 |
| n=16, nb=16384 | 0 | 0 | 0 |

**All 6 shapes: 0 / 0 / 0.** Full-grid replicate (`determinism.log:5`, 72 cells):
`hier_self_diff_cells 0, fp32row_self_diff_cells 0, bf16row_self_diff_cells 0,
fp32row_vs_hier_diff_cells 0, bf16row_vs_hier_diff_cells 0` (0 total slots both ways).

---

## 6. TABLE 4 — The 2045-cell suite arms (fixture H=2) + incumbent parity

The frozen suite (`tests/test_dsv41_indexer_smallm_hier.py`, 2045 cells) run fresh-process, twice.
**Source: `l2full_run1.log`, `l2full_run2.log`, `l2full_bf16off.log`, `suite_bf16_0.log`,
`suite_bf16_1.log`.**

| arm | equal | divergent | mask_mismatch | note |
|---|---|---|---|---|
| **L2-full (default)** | 1794 | **251** | 15 | run1 ≡ run2 (byte-identical verdicts) |
| L2-full + `DSV41_INDEXER_ROW_BF16=0` | 1794 | 251 | 15 | identical (fp32 already the small-n dtype) |
| **`DSV41_INDEXER_L2_FULL=0`** (ablation) | 1106 | **939** | 14 | **reproduces the shipped abort exactly** |
| **`DSV41_INDEXER_SMALLN_ROW_BF16=1`** (ablation) | 1106 | **939** | 14 | **reproduces the shipped abort exactly** |
| shipped build (baseline, `suite_bf16_1.log`) | 1106 | **939** | 14 | the incumbent |

Two independent ablations each reproduce the shipped **939/2045** — the mechanism is confirmed;
`n=17` (HIER on both sides) is 0-diff in every arm (path-boundary cells pass).

Residual strata for the 251 (`l2full_run1.log` `by_role_n`, non-zero rows): `plain/n=2` (1),
`plain/n=3` (7), `plain/n=4` (54), `plain/n=16` (61), `consumer/n=4` (64), `consumer/n=16` (64).
Every residual is at **n ≤ 16** — i.e. inside the fixture's H=2 tie regime.

### 6b. Incumbent parity note (one paragraph)

Index-identity does **not** hold for production today either, even at the fixture's parameters. The
shipped next17 build has two internal paths for the same `Indexer.__call__` — the bf16-row fallback
and the hierarchical exact pass — and at the suite's `H=2` they **diverge 939/2045** (73 % reduction
under L2-full → 251; `suite_bf16_1.log` vs `l2full_run1.log`). At `H=2` the only configuration that
satisfies index-identity is **bit-identical compute on both branches**, since the ~24.5 % exact-zero
tie class is broken differently by any two independent `argpartition`/sort orderings. In other words,
the gate as written is not measuring "does L2-full change behaviour vs shipped" — it is measuring
"does L2-full match a sibling path that already disagrees with itself on the same inputs". A gate that
the current production build fails at its own fixture is not a shipping criterion; it is a flag that
the fixture parameters are unrepresentative.

---

## 7. Real-tensor replay (91K precondition — what exists today)

A 4-call functional capture is on disk: `/private/tmp/next18_functional.npz` (`nb=480`,
`n ∈ {1,4}`, source + consumer roles). Replayed through L2-full / bf16-row / HIER
(`next18_replay_capture.py`; this package's run → `amend/replay_capture.log`):

| call | role | n | nb | k | L2-full vs HIER | bf16-row vs HIER | slots | replayable |
|---|---|---|---|---|---|---|---|---|
| i=0 | source | 1 | 480 | 64 | **0** | **0** | 64 | yes |
| i=1 | consumer | 1 | 480 | 64 | — | — | — | needs `shared.candidates` (not captured) |
| i=2 | source | 4 | 480 | 64 | **0** | **0** | 256 | yes |
| i=3 | consumer | 4 | 480 | 64 | — | — | — | needs `shared.candidates` (not captured) |

`REPLAY_CAPTURE {"records": 4, "replayable": 2, "L2full_vs_hier_total_ndiff": 0,
"bf16row_vs_hier_total_ndiff": 0, "slots": 320}` (`amend/replay_capture.log`).

The **4-call / 91K / 165K-token capture does not exist on disk**
(`find /private/tmp -name '*.npz'` → only the 480-column one + `t.npz`); it **cannot be fabricated
offline** and **must not** trigger a relaunch to obtain. It rides the Phase-3 **R1** validation boot
for free. The 480-column replay is suggestive (0 diffs on the two replayable calls) but not
sufficient: the sweep pins the H-driven tie mechanism but does not enumerate every tie source
(padding, repeated content, masked regions) that only the 91K replay at production H can. **This is
exactly why A2 below is a SHIP PRECONDITION, not a claim.**

---

## 8. Supporting evidence (design-level, unchanged by the correction)

* **L2-full mechanism.** At `n ≤ _FENCE_MIN_ROWS` (16) the fallback stores its `[b,n,nb]` score row in
  fp32 (matching the hierarchical exact-rescore's precision); both branches then run the *same*
  global `topk_from_row` on a bitwise-identical row → trivially exact. Guarded by
  `DSV41_INDEXER_L2_FULL` (default ON); `DSV41_INDEXER_SMALLN_ROW_BF16=1` is the mechanism control.
* **Non-destructive ablation.** `L2_FULL=0` restores the shipped 939 verbatim (§6) — a reversible
  kill switch, env-retirable at ship.
* **Per-call cost.** `next18_l2full_perf.py`: max Δ = **+0.063 ms/call** (fp32 vs bf16 row at
  `index_n_heads=32, index_head_dim=128`), typical < 0.005 ms → ≤ 0.5 ms/round vs the lever-2 win of
  29.0 ms/round (bar 15 ms). Design-level; not required for the gate decision, reproduced in the memo.
* **L2-guard not shipped.** `next18_gap_hist.py` (`gap_hist.log:56`): δ = 2⁻⁸·max|row|; measured
  trigger rate **252/252 = 100 %** → it degenerates to L2-full for extra machinery; not shipped.

---

## 9. PROPOSED AMENDMENT TEXT (A1 / A2 / A3) — for the gate owner

> **Status: PROPOSED. Pre-registered. NOT applied by us.** This is a USER / gate-owner sign-off.
> Adopting it changes the frozen gate (`PHASE4-CAMPAIGN.md` §3b); declining it leaves lever-2
> BLOCKED. We take no unilateral action.

**Amended gate (replaces the frozen "ABORT on any divergence (ties count as divergence)" rule for the
lever-2 identity gate):**

> **A1 — Suite re-instantiation at production head count.** The 2045-cell index-identity suite
> (`tests/test_dsv41_indexer_smallm_hier.py`) must be **re-instantiated at the production head count**
> `index_n_heads = 32` (equivalently `≥ 8`) and must report **0 divergent cells** under the existing
> elementwise rule (ties still count as divergence). *This is the gate's primary criterion.*
>
> **A2 — Real-tensor replay at production H is a SHIP PRECONDITION.** Before any SHIP label, a
> **≥ 91K-token real agentic capture** replayed at production H through L2-full vs the shipped HIER
> path must show **0 index diffs** (source AND consumer layers). This rides the Phase-3 R1 validation
> boot (0 dedicated relaunches). *A "SUITE-CLEAN, REPLAY-PENDING" state is NOT shippable.*
>
> **A3 — H=2 fixture cells retained as an exact-tie STRESS COHORT, gated on VALUE-identity.** The
> existing `index_n_heads=2` cells are **kept** but re-scored on **value-identity**: the design must
> lose **0 value slots vs the full-width fp32 truth** (the metric in §4), not achieve index-identity.
> *(Measured: fallback/L2-full 0 value-loss slots; the only >tol rank deviation is HIER's 48 slots in
> one cell — the stress cohort must be interpreted on value, since index-identity is unsatisfiable
> there by ANY finite-precision compute, incumbent included — §6b.)*

**Rationale.** The amendment **corrects unrepresentative fixture parameters** and **preserves the
gate's original semantics** — "no behavior change vs shipped". It does not weaken the gate: at
production H it demands **stricter-than-current** behaviour (0 diffs where the design is measured to
be 0), adds the **real-tensor precondition** the frozen gate always intended (§3b item 2), and keeps
the H=2 cohort under a *sound* metric (value-identity) instead of an unsatisfiable one (index-identity).
The original gate's "no behavior change" claim was never true for the incumbent at H=2 (§6b); the
amendment restores it at the parameters production actually runs.

---

## 10. Pre-registered FAILURE BRANCH (declare before spending)

> **Declared now, before any relaunch/measurement spend.** If the **A2** 91K real-tensor replay at
> production H shows **ANY index diff**, then **L2-full IS behaviour-changing at production
> parameters** (the H-sweep's 0-diff result would have missed a non-H tie source). In that case the
> amendment's value-identity argument must be **made on real production data** — i.e. show the
> fallback/L2-full loses **0 value slots vs a production-truth oracle** and that any deviation is
> strictly ≤ the incumbent HIER's own deviation — **or the lever is BLOCKED** and the speed-vs-output
> tradeoff (§Phase-4 §3d) stands as the honest closing statement.

**Expected outcome.** Per §3 (0 diffs at H = 8/32/64 over 258,048 slots) and §7 (0/0 on the 320-slot
real replay), the expected result is **clean** → the lever proceeds to R2 with the amended gate.

---

## 11. Reproduce (exact commands)

All offline. Local GPU python = `/Users/adam.durham/repos/exo/.venv/bin/python`. Worktree root for
every probe line below = `/private/tmp/next18-lever2`. **Never run the full test suite on this Mac;
single-file pytest only.**

```bash
cd /private/tmp/next18-lever2

# (a) the frozen suite, L2-full defaults  → expect: 2045 cases, 1794 equal, 251 divergent
PYTHONPATH=$PWD /Users/adam.durham/repos/exo/.venv/bin/python -m pytest \
  tests/test_dsv41_indexer_smallm_hier.py -q -s

# (b) ablations (each must print 939 divergent — reproduces the shipped abort)
DSV41_INDEXER_L2_FULL=0         PYTHONPATH=$PWD /Users/adam.durham/repos/exo/.venv/bin/python -m pytest tests/test_dsv41_indexer_smallm_hier.py -q -s
DSV41_INDEXER_SMALLN_ROW_BF16=1 PYTHONPATH=$PWD /Users/adam.durham/repos/exo/.venv/bin/python -m pytest tests/test_dsv41_indexer_smallm_hier.py -q -s

# (c) head-count class table (census 2^-H + H-sweep) and production-H stability + determinism
PYTHONPATH=$PWD /Users/adam.durham/repos/exo/.venv/bin/python bench/next18_headcount.py   # -> headcount.log
PYTHONPATH=$PWD /Users/adam.durham/repos/exo/.venv/bin/python bench/next18_prodgate.py    # -> prodgate_full.log

# (d) value-loss matrix vs fp32 truth over all residual cells
PYTHONPATH=$PWD /Users/adam.durham/repos/exo/.venv/bin/python bench/next18_classify_all.py \
  /Users/adam.durham/.hermes/cache/scratch/p5/l2full_run1.log     # -> classify_all_fixed.log

# (e) the worst-cell / G0 corroboration
PYTHONPATH=$PWD /Users/adam.durham/repos/exo/.venv/bin/python bench/next18_losscell.py    # -> losscell.log
PYTHONPATH=$PWD /Users/adam.durham/repos/exo/.venv/bin/python bench/next18_g0_probe.py    # -> g0_probe.log

# (f) real-tensor replay (480-col capture on disk; 91K capture pending on R1)
PYTHONPATH=$PWD /Users/adam.durham/repos/exo/.venv/bin/python bench/next18_replay_capture.py \
  /private/tmp/next18_functional.npz

# (g) the rejected L2-guard trigger rate
PYTHONPATH=$PWD /Users/adam.durham/repos/exo/.venv/bin/python bench/next18_gap_hist.py   # -> gap_hist.log
```

---

## 12. Sources (raw logs)

All under `/Users/adam.durham/.hermes/cache/scratch/p5/` unless noted:

| file | what it backs |
|---|---|
| `prodgate_full.log` (=`prodgate.log`) | Table 1 census `:2-7` + stability `:12-14` + determinism `:16-21` |
| `headcount.log` | Table 1 H-sweep (2/4/8/16/32; 193,536 slots) `:5-9` |
| `classify_all_fixed.log` | Table 2 value-loss matrix `:1,6,96,97` |
| `losscell.log` | Table 2 worst-cell / arm-vs-truth `:2-6` |
| `g0_probe.log` | Table 2 G0 corroboration `:50` |
| `determinism.log` | Table 3 full-grid replicate `:5` |
| `l2full_run{1,2}.log`, `l2full_bf16off.log` | Table 4 L2-full arms |
| `suite_bf16_{0,1}.log` | Table 4 shipped / L2-full-with-ROW_BF16=0 arms |
| `gap_hist.log` | §8 L2-guard trigger rate `:56` |
| `amend/replay_capture.log` (this package) | §7 replay (0/0, 320 slots) |

**Discarded / superseded artifacts (do not cite as evidence):** `classify_all_l2full.log`,
`classify_all_l2full_v2.log`, `classify_l2full.log` — earlier classifier iterations. `*_v2.log` has a
NaN-handling bug (missing `np.isfinite` guard) that false-positived **129** consumer cells as
`VALUE-LOSS` on `NaN`, plus mis-counted `max_fb_minus_hier` as 0; the authoritative output is
`classify_all_fixed.log`.

---

## 13. Discrepancies found (vs the established facts)

1. **Production H label.** The P1 memo §6 says production `index_n_heads=64`; the true value is
   **32** (`config.py:82`, node `config.json`). Already flagged by the PM; corrected throughout this
   package. **Conclusion unchanged** (0 diffs at H=8/32/64).
2. **H=64 is not covered by the `headcount.log` sweep** — that sweep spans H ∈ {2,4,8,16,32}
   (`headcount.log:5-9`, 193,536 slots). The H=64 datum comes **only** from the `prodgate_full.log`
   stability line (`:14`, 258,048 slots). Immaterial (both 0), but the memo's single-table "H-sweep"
   row for 64 actually mixes two sources.
3. **Two metrics conflated in memo §5** for the worst cell: the **index** diff is 46 slots and the
   **value** loss vs truth is 48 slots (`classify_all_fixed.log:6`). Both are correct; they are
   different quantities. Stated explicitly in §4.
4. **`losscell.log` (16 slots) vs `classify_all_fixed.log` (48 slots).** These are different probes
   on different cells (losscell's reproduction cell vs the full residual sweep); not a conflict, but
   they should not be quoted interchangeably. Table 2 uses the full-sweep value (48).
5. **Superseded classifier artifacts** (`*_v2.log` etc.) remain on disk in the same dir as the
   authoritative log; a naive `grep` over the directory would read the buggy 129-cell value-loss
   count. Flagged in §12 so triage can prune them.

No discrepancy changes any conclusion: the residual is a fixture-artifact (H=2 exact-zero tie class),
the design loses 0 value slots vs fp32 truth, and the incumbent fails the gate at H=2 too.
