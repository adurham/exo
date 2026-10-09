# PHASE 5 — P1: lever-2 ship-design (L2-full) vs the frozen index-identity gate

Author: Phase-5 P1 (depth-2). 2026-10-08 ~20:00 CDT. **ALL WORK OFFLINE** (source + local-GPU
pytest + real-tensor replay); **0 relaunches spent**. Production remains
`deploy/next13 @ 576e9d279` + mlx-lm `3bf8316`, gates unset — untouched by this phase.

---

## 0. Outcome (read this first)

**Verdict: `BLOCKED pending gate-owner amendment`.**

* **The frozen gate, as written, FAILS: 251 / 2045 divergent cells** (from 939 in the shipped
  build — a 73 % reduction), **every residual cell at `n ≤ 16`**. The frozen rule is
  *"ABORT on any divergence (ties count as divergence)"*, so a FAIL is a FAIL; no label other
  than *blocked* is honest without an owner-signed gate amendment on record.
* **The blocking residual is a fixture artifact, not a design defect — and it is unfixable in
  scope.** The suite's synthetic Indexer is built with `index_n_heads=2`; production
  DeepSeek-V4.1-Flash runs `index_n_heads=64`. With `H=2`, ~24.5 % of columns score *exactly*
  `0.0` (a column is 0 iff every head's pre-ReLU dot ≤ 0, `P ≈ 2^-H`), creating a giant exact
  tie class at the k-boundary that **no two implementations break identically**. At production
  `H ≥ 8` the tie class vanishes and **L2-full ≡ HIER with 0 index diffs** (measured, §6).
* **The incumbent also fails the gate at fixture parameters**: the shipped build's own two
  paths (bf16 fallback vs HIER) are **939/2045 divergent**. The identity the gate demands does
  not hold for production today at `H=2`; only bit-identical compute satisfies it there.
* **Recommendation to the gate owner**: amend the gate to evaluate at production head count
  (`H≥8`) — this preserves the gate's original *"no behavior change"* semantics and merely
  corrects unrepresentative fixture parameters — and make the real 91K-tensor replay a **ship
  precondition**. A pre-registered amendment + failure branch is given in §10. **This is a
  user decision; the gate is not unilaterally redefined here.**

**Status of L2-full**: design-complete, implemented, and — **subject to the amendment and the
91K replay** — **SHIP-DESIGN-READY**. Retention projection ≈ +27–29 ms/round agentic (bar 15 ms).

---

## 1. G0 — source premises (the design dies here if it dies)

Each with `file:line` in the mlx-lm worktree (`/private/tmp/next18-lever2`, branch
`deploy/next18-lever2`).

**(1) The coarse pass is precision-reduction ONLY — no subsampling/striding. CONFIRMED.**
`indexer_hierarchical.py:243` `for c0 in range(0, nb, st):` streams **every** column of `[0,nb)`
in `st`-wide strips; `:247` `s = _score_shared_columns(qc, keys, wc)` scores all of them
(bf16); `:260` reduces to `s.reshape(..., block).max(-1)` — a *true* block maximum. The module
docstring states it explicitly (`:104–108`): the `n*(nb/block + B*block)` "downsampled coarse"
work model is **deliberately not** taken because it "is exactly the thing that loses the
exactness guarantee". **Consequence: a full-width (un-pruned) small-n path is trivially exact
— the guarded design would have been sound.** (The consumer-side variant
`coarse_block_scores_candidates` `:308` also scores only blocks holding a candidate and is
docstring-asserted elementwise-identical to score-then-mask, `:324`.)

**(2) The hierarchical path is exact GIVEN its candidates; it does NOT rely on the overfetch
heuristic for *values*. CONFIRMED empirically.** `indexer_hierarchical.py:45–65` carries the
*proof* — the top-`k` blocks by block-max **always** include every block containing a column
`> v*` (the k-th value); this is a proof for **any** distribution, not a heuristic; `overfetch`
(`HIER_OVERFETCH=16`, `:147`) is only insurance for bf16 *index* reordering. Measured against a
full-width **fp32 truth row** (`bench/next18_g0_probe.py`), the shipped
`H.hierarchical_topk_prod(..., overfetch=16)` path lost **0 values / 141,312 slots** (48 cells,
`n∈{1,2,4,16}`, `nb∈{512,4096,16384}`); the only index diffs (52, one cell) are exact ties.
**Consequence: a value-identity gate is satisfiable by the shipped architecture; the "NO design
passes the existing gate" branch of the task does not fire.**

**(3) Both paths share the SAME top-k op. CONFIRMED.**
* fallback: `indexer.py:309` (untiled) / `:377 _tiled_scores_buffer` → `:309`
  `mx.argpartition(-s, k-1)` then `mx.sort` (position order) via `topk_from_row` (`:303`).
* hierarchical: `indexer_hierarchical.py:502` `mx.argpartition(-v, kk-1)` in
  `exact_rescore_streaming` (`:457`), then `mx.argsort` (`:512`) — the same
  `argpartition`-then-position-sort shape.
Same *op*, but the *comparison order* differs (fp32 re-score vs the fallback's stored row), so a
precision split at the boundary can still break to different sides — which is precisely the
abort root cause, and precisely what L2-full removes (§3).

**G0 result: all three premises hold → the design is worth building. It did not die here.**

---

## 2. Cheap-first: stratified re-analysis of the EXISTING suite (before building)

The shipped suite (`tests/test_dsv41_indexer_smallm_hier.py`, 2045 cells) re-run with the
**existing** `DSV41_INDEXER_ROW_BF16=0` (fp32 row, no code change), stratified by `(role,n)`:

| config | equal | divergent | residual strata |
|---|---|---|---|
| shipped (`ROW_BF16=1`) | 1106 | **939** | `plain/{1,2,3,4,16}`, `source/{1,4,16}`, `consumer/{1,4,16}`; `n=17` = 0 everywhere |
| `ROW_BF16=0` | 1794 | **251** | `plain/{2,3,4,16}` (61+54+7+1), `consumer/{4,16}` (128); **`plain/n=1` → 0**, `source/*` → 0 |

Two readings: (a) `n=1` and the `source` role go **fully clean** at `ROW_BF16=0` — the env lever
*does* concentrate the residual; (b) but the residual does **not** fully concentrate into one
removable stratum, so no narrowing of the gate by `(role,n)` is available "today at zero new
machinery". **→ build L2-full.**

---

## 3. The design: L2-full (preferred per the pre-registered selection rule)

**One-line intent.** At `n ≤ _FENCE_MIN_ROWS` (16), the fallback stores its `[b,n,nb]` score row
in **fp32** instead of `_ROW_DTYPE`=bf16, so the fallback and the hierarchical exact re-score
(already fp32, `indexer_hierarchical.py:493`) rank on a **bitwise-identical fp32 row**; both then
run the same global `topk_from_row`. No coarse pass, no overfetch, full width — **trivially
exact by construction** (`G0(1)`).

**Implementation** (`indexer.py`, worktree `deploy/next18-lever2`):

```
:139  _L2_FULL          = env("DSV41_INDEXER_L2_FULL", "1") == "1"        # default ON
:140  _SMALLN_ROW_BF16  = env("DSV41_INDEXER_SMALLN_ROW_BF16", "0") == "1" # mechanism control
:143  def _row_dtype(n):  return _ROW_DTYPE if not (_L2_FULL and n<=16) else (bf16 if _SMALLN_ROW_BF16 else fp32)
:257  tile_width():  row_size = _row_dtype(n).size          # row-size-consistent tiling
:280  tiled():       bsz*n*nb*(n_heads*4 + _row_dtype(n).size) <= _TILE_BUDGET
:377  _tiled_scores_buffer():  row_dtype = _row_dtype(n) ; :380 row = mx.zeros(..., dtype=row_dtype)
:675  untiled reference:  scores = mx.sum(scores,2).astype(_row_dtype(n))
```

* **Scope**: engages only when `_L2_FULL` and `n ≤ _FENCE_MIN_ROWS`; `n>16` (prefill) keeps the
  bf16 row untouched — the bounded-row hierarchical path is still the prefill win.
  `DSV41_SPARSE_FENCE_MIN_ROWS=0` (guard off) never engages it → historical configs reproduce.
* **Kill switch / A-B**: `DSV41_INDEXER_L2_FULL=0` reverts; `DSV41_INDEXER_SMALLN_ROW_BF16=1`
  re-creates the bf16 row at small n (diagnostic).
* **Blast radius**: additive; one helper + 4 call sites; no change to the hierarchical module,
  the guard predicate (`:591 if _HIER and n > _FENCE_MIN_ROWS:`), or the large-n path.
* **Exo side**: **no exo-side change needed** (this is mlx-lm only); `deploy/next18-identity`
  stays diff-empty vs base `576e9d279`.

**L2-guard was NOT built** — the selection rule prefers L2-full unless it is *materially
slower* than L2-guard, and the per-call cost shows it is not (§8, §9).

---

## 4. Suite results + ablations (G1, fresh process, run twice)

`tests/test_dsv41_indexer_smallm_hier.py` (2045 cells), fresh process:

| arm | equal | divergent | mask_mismatch | notes |
|---|---|---|---|---|
| **L2-full (default env)** | **1794** | **251** | 15 | run 1 == run 2 (byte-identical verdicts) |
| L2-full + `ROW_BF16=0` | 1794 | 251 | 15 | identical (fp32 is already the small-n dtype) |
| **`L2_FULL=0`** (ablation) | 1106 | **939** | 14 | **reproduces the shipped abort exactly** |
| **`SMALLN_ROW_BF16=1`** (ablation) | 1106 | **939** | 14 | **reproduces the shipped abort exactly** |

**Mechanism confirmed by two independent ablations**: turning L2-full off, and forcing the
small-n row back to bf16, *each* reproduce the shipped 939/2045 exactly. `n=17` (HIER on both
sides) is 0-diff in every arm; path-boundary cells (fallback at `n≤16`, HIER at `n≥17`) PASS;
the RED/GREEN sabotage controls PASS (overfetch=0 loses 7 slots on real keys; coarse corruption
detected).

---

## 5. Classification of the 251 residual cells (value vs index)

Because the two arms return index sets **sorted ascending by index**, a slot-wise diff is
meaningless when the sets differ; the sound metric is the per-row **value multiset** against the
full-width fp32 row `R` (score oracle). `bench/next18_classify_all.py` over all 251 cells:

| metric | result |
|---|---|
| residual cells | 251 (262 raw records incl. the Summary/case rows) |
| **fallback (=L2-full) value-loss slots vs fp32 truth** | **0** |
| hierarchical value-loss slots vs fp32 truth | **48** (all in one cell) |
| max `fallback − hier` per rank | **9.73e-6** (one cell; float non-associativity) |
| value-ties (all other cells) | 100 % — differences are **exact fp32 score ties** |

**Interpretation.** L2-full **never** picks a column worse than the fp32 truth. The sole
near-miss cell (`plain, n=16, nb=16384, k=513, seed=385721`, 46 slot diffs) is the fallback and
the hierarchical exact pass summing the same fp32 products in **different association order**;
the 9.73e-6 deviation is pure fp32 rounding, and it is the **hierarchical** arm that deviates from
truth there, not the fallback (`bench/next18_losscell.py`: fallback loss 0 / worst 0.0; hier loss
16 / worst 9.7e-6). Every other residual is an **exact value tie** broken differently by the two
`argpartition` calls — the tie class the module docstring already documents (`:62–65`).

---

## 6. The load-bearing finding: the residual is a head-count artifact

The fixture sets `index_n_heads=2`. A column scores **exactly** `0.0` iff every head's pre-ReLU
dot ≤ 0 (`P ≈ 2^-H`). Censused on the fp32 row (visible columns only, `bench/next18_prodgate.py`):

| H | zero-column fraction | `2^-H` | L2-full vs HIER index diffs (193,536 slots) |
|---|---|---|---|
| 2 | 0.2455 | 0.25 | **1187** |
| 4 | 0.0594 | 0.0625 | **528** |
| 8 | 0.00415 | 0.00391 | **0** |
| 16 | 0.0 | 1.5e-5 | **0** |
| **32** | 0.0 | 2.3e-10 | **0** |
| **64 (production)** | 0.0 | 5.4e-20 | **0** |

The zero-column census **confirms the `2^-H` law** (0.2455/0.0594/0.00415 vs 0.25/0.0625/0.00391).
Broadened production-H stability proof (`bench/next18_prodgate.py`, 8 seeds × `n∈{1,4,16}` ×
`nb∈{512,4096,16384}`, **258,048 slots each**): `H=8` → **0**; `H=32` → **0**; `H=64` → **0**
index diffs (L2-full vs HIER). The shipped bf16-row fallback gives the **same** counts as
L2-full in the same sweep (so L2-full is never worse than the incumbent, and equal at `H≥8`).
Determinism replicate at `H=64` (PREREG D4): L2-full self-diff 0, HIER self-diff 0, cross 0.

> **Auditor note (numeric reconciliation).** The suite (2045-cell test) reports bf16-row-vs-HIER
> as 939 *cell* divergences at its `H=2`, while the sweep reports 1187 *slot* diffs at `H=2`.
> These are the **same phenomenon at different granularity** (cell = pass/fail over a cell's
> `n·k` slots) on the **same fixture family but different seeds/shapes** — not a cherry-pick.
> The sweep's job is the H-trend, and its `H=2`/`H=4` counts match L2-full's exactly.

---

## 7. Real-tensor replay + the 91K precondition

* **On disk**: `/private/tmp/next18_functional.npz` — a 4-call functional capture (`nb=480`,
  `n∈{1,4}`, source + consumer roles). Replay (`bench/next18_replay_capture.py`): the **2
  replayable (source) calls** give **L2-full vs HIER = 0 diffs / 0 diffs** (bf16-row also 0);
  320 slots. The 2 consumer calls require `shared.candidates` (not stored) → flagged, not
  replayed.
* **The 91K / 165K-token capture does not exist on disk** (`find /private/tmp -name '*.npz'` →
  only `next18_functional.npz`, `t.npz`). It **cannot be fabricated offline** (needs a live
  boot) and **must not** trigger a relaunch to obtain.
* **→ Ship precondition (pre-registered, §10): the 91K replay at production H must be clean
  before any ship label.** It **rides the Phase-3 R1 validation session for free** (the ledger
  already plans the capture on the validation boot). The functional `nb=480` capture is
  suggestive but not sufficient — the sweep pins the H-driven tie mechanism but does not
  enumerate other tie sources (padding, repeated content, masked regions) that only the 91K
  replay at production H can.

---

## 8. Per-call cost + retention projection (G1b)

`bench/next18_l2full_perf.py` — the fp32 row vs the bf16 row at production indexer shapes
(`index_n_heads=32, index_head_dim=128`):

| n | nb | bf16 row ms | fp32 row ms | Δ ms |
|---|---|---|---|---|
| 1 | 16384 | 0.686 | 0.685 | −0.001 |
| 4 | 16384 | 0.368 | 0.366 | −0.002 |
| 4 | 65536 | 0.929 | 0.992 | **+0.063** (max) |
| 16 | 16384 | 0.762 | 0.762 | −0.001 |
| 16 | 65536 | 3.316 | 3.373 | +0.057 |

**Max Δ = +0.063 ms/call**, typical < 0.005 ms. With **≈1 small-n body pass/round** (P0-units:
a single m=4 body pass; the draft is a separate DSparkHead with no Indexer) plus a handful of
indexer calls/round at small n, the per-round cost is **≤ 0.5 ms**.

**Retention (G1b):** lever-2 win = **29.0 ms/round agentic** (same-build, PHASE3B §6). Net
retention = `29.0 − ≤0.5 ≈ +28.5 ms/round` ≫ the **15 ms** G1b bar. Next18 at L2-full defaults
should reproduce **≈101 ms/round agentic ≈ 30 t/s** (from production 157 ms / 19.5 t/s).
**G1b: PASS** (offline projection; the on-cluster confirmation is an R1 item).

**R1 / Phase-0 fold-in:** the missing **`DSV41_INDEXER_HIER=0` BENIGN** point (never measured in
any session) is an R1 item — expected **≈95 ms** (by the benign:agentic ratio) and it **rides
R1 for free**, 0 relaunches.

---

## 9. L2-guard: ε derivation + trigger rate (reported, not shipped)

L2-guard (fp32 rows + coarse pass + per-call escape) was **not built**, because the selection
rule only requires it if L2-full is materially slower — and L2-full's small-n cost is
negligible (§8). Numbers the guard *would* have run on, as evidence (`bench/next18_gap_hist.py`):

* **Bound** `δ = 2⁻⁸ · max|row|` (the bf16 storage rounding of the fp32 score row — **derived,
  not tuned**; 2⁻⁸ = half a bf16 ulp relative to a power-of-two-scaled row).
* **Guard** = fall back to full width when the k-th-vs-(k+1)-th coarse gap ≤ `2δ`.
* **Measured trigger rate = 252 / 252 = 100 %** at `n∈{1,4,16}`, `nb∈{512,4096,16384}`; δ ranged
  9.7e-4 … 8.5e-3. **The guard always escapes → it degenerates to L2-full exactly, for extra
  machinery and a premise risk.** Selection rule → **L2-full** (no ε doc, no premise risk).

---

## 10. Pre-registered amended gate + failure branch (for the gate owner)

**Frozen gate as written: FAIL** (251/2045; all residual in the `H=2` exact-tie class; the
incumbent fails it too, 939/2045). This is a **determinate** outcome, not a pending choice.

**Proposed amendment (requires owner sign-off; not applied here)** — evaluate the identity gate
at the **production head count** `index_n_heads ≥ 8`:

* **(A1)** the 2045-cell suite re-instantiated at `H = 64` (or `≥ 32`) must be **0 divergent**;
* **(A2)** the 91K real-tensor replay at production H must be **0 index diffs**;
* **(A3)** the `H=2` fixture cells are retained as an **exact-tie stress cohort**, gated on
  **value-identity** (0 value-loss slots vs fp32 truth), not index-identity.

**Pre-registered failure branch (declare before spending):** *if the 91K replay shows ANY index
diff at production H, L2-full is behavior-changing; value-identity must then be argued on real
production data, or the lever is BLOCKED.* (Given §6, the expected outcome is clean.)

**Ship plan under the amendment (Phase-3, R1→R2→R3, ≤3 relaunches, declare-before-spend):**
* **R1** — `deploy/next18-identity` (+ mlx-lm `next18`): canary → same-session fixed-replay A/B
  vs clean baseline → **91K capture replay** (precondition A2) → battery both arms → missing
  `HIER=0` BENIGN point.
* **R2** — ship L2-full **default-on** (env retired), fresh boot, canary + battery + parity
  smoke within noise, `known-good-*` tag.
* **R3** — reserve only.

---

## 11. Budget / reallocation

**P1 spent 0 relaunches** (all proof offline). Relaunch ledger unchanged: R1/R2/R3 untouched,
0/3 spent this round. On the amendment path, **P1 folds into the Phase-3 R1 session** (its 91K
replay and BENIGN point are already R1 items) — P1 itself consumes no dedicated relaunch.
**P2 (dense EXL3) is already complete** (see `PHASE5-P2-DENSE.md`); **reallocate any P1 slack to
P3's R1 preparation.**

---

## 12. Reproduce (exact)

```bash
# the frozen suite, L2-full defaults (expect: 2045 cases, 1794 equal, 251 divergent)
cd /private/tmp/next18-lever2 && PYTHONPATH=$PWD \
  /Users/adam.durham/repos/exo/.venv/bin/python -m pytest tests/test_dsv41_indexer_smallm_hier.py -q -s

# ablations (each must print 939 divergent)
DSV41_INDEXER_L2_FULL=0          PYTHONPATH=$PWD /Users/adam.durham/repos/exo/.venv/bin/python -m pytest tests/test_dsv41_indexer_smallm_hier.py -q -s
DSV41_INDEXER_SMALLN_ROW_BF16=1  PYTHONPATH=$PWD /Users/adam.durham/repos/exo/.venv/bin/python -m pytest tests/test_dsv41_indexer_smallm_hier.py -q -s

# G0 premise 2 (hier loses 0 values vs fp32 truth) and the head-count proof
PYTHONPATH=$PWD /Users/adam.durham/repos/exo/.venv/bin/python bench/next18_g0_probe.py
PYTHONPATH=$PWD /Users/adam.durham/repos/exo/.venv/bin/python bench/next18_prodgate.py
PYTHONPATH=$PWD /Users/adam.durham/repos/exo/.venv/bin/python bench/next18_headcount.py
PYTHONPATH=$PWD /Users/adam.durham/repos/exo/.venv/bin/python bench/next18_classify_all.py \
  /Users/adam.durham/.hermes/cache/scratch/p5/l2full_run1.log
PYTHONPATH=$PWD /Users/adam.durham/repos/exo/.venv/bin/python bench/next18_replay_capture.py /private/tmp/next18_functional.npz
PYTHONPATH=$PWD /Users/adam.durham/repos/exo/.venv/bin/python bench/next18_l2full_perf.py
PYTHONPATH=$PWD /Users/adam.durham/repos/exo/.venv/bin/python bench/next18_gap_hist.py
```

Never run the full test suite on this Mac; single-file only.

---

## 13. Artifacts

**Code (mlx-lm worktree `/private/tmp/next18-lever2`, branch `deploy/next18-lever2`,
origin=adurham/mlx-lm):** `mlx_lm/models/deepseek_v41/indexer.py` (L2-full), plus probes under
`bench/`: `next18_g0_probe.py`, `next18_diffdiag.py`, `next18_repro.py`, `next18_classify.py`,
`next18_classify_all.py`, `next18_losscell.py`, `next18_l2full_perf.py`, `next18_gap_hist.py`,
`next18_determinism.py`, `next18_headcount.py`, `next18_prodgate.py`, `next18_replay_capture.py`.
**exo worktree** `/private/tmp/next18-exo` (`deploy/next18-identity`): **diff-empty vs base —
no exo-side change needed.**

**Raw logs** (scratch, not committed): `/Users/adam.durham/.hermes/cache/scratch/p5/`
(`suite_bf16_{0,1}.log`, `l2full_run{1,2}.log`, `l2full_bf16off.log`, `classify_all_fixed.log`,
`g0_probe.log`, `headcount.log`, `prodgate.log`, `losscell.log`, `gap_hist.log`,
`determinism.log`).

**Production**: unchanged — `deploy/next13 @ 576e9d279` + mlx-lm `3bf8316`, gates unset.
**RESTORE target intact. 0 relaunches spent.**

---

## 14. RATIFIED AMENDMENT (plan owner / Fable adjudication, 2026-10-08 ~21:00 CDT) — recorded verbatim, evidence-ID'd, then RE-FROZEN

**Correction applied (requirement 1).** §6's table row labelled "64 (production)" is WRONG:
production `index_n_heads` = **32**, not 64. Verified two ways — `mlx_lm/models/deepseek_v41/config.py:82`
(`index_n_heads: int = 32`) and the SERVED checkpoint's real `config.json` on the node
(`ssh studio1 'grep index_n_heads ~/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw/config.json'`
→ `"index_n_heads": 32`). The conclusion is unchanged: the H-sweep gives **0 diffs at H=8, 32 AND 64**
(258,048 slots each), so H=32 (production) is in the clean regime. The amendment's gate is evaluated at
the SERVED checkpoint's ACTUAL H, **re-verified from `config.json` at EVERY promotion**.

**Ratified amendment (verbatim requirements):**
1. The identity gate is evaluated at the served checkpoint's actual head count `H`, re-verified from
   `config.json` at EVERY promotion. Production H = **32** (corrected from the memo's H=64 claim).
   Threshold evidence: H≥8 → 0 diffs; actual 32.
2. **"Ties count as divergence" RETAINED IN FULL at production H.** The amendment narrows the
   *parameter axis*, not the strictness.
3. The small-H cohort (H∈{2,4}) is accepted via **PER-DIFF-SLOT ATTRIBUTION**: a cell-level artifact
   asserting every diff in all 251 residual cells sits in an exact-zero column, and that the 262
   loss-checked cells are a **superset** of the 251 divergent cells. "Mechanism fits" becomes a
   per-cell artifact, not a correlation.
4. **ONE-AMENDMENT RULE:** any second amendment requires new mechanistic evidence, never a failed check.
5. **Identity story phrasing (exact):** *"bitwise-identical at production H; strictly more precise than
   shipped at small H (0 vs 48 value-loss slots vs truth)"*. Never claim "identical everywhere".
   Census arithmetic note: at H=32 the expected zero-column count ≈ 7.6e-6 (2.3e-10 × 32768) —
   essentially zero; the earlier "0.25" figure was H=2's.

**Added required legs (in R1):**
- **A.** END-TO-END GREEDY TOKEN-IDENTITY DIFF vs current production on fixed replay (greedy-only =
  exact), WITH a **production-vs-production determinism CONTROL arm** (TP2 reduction order can break
  bitwise reproducibility — prove "0 token diffs" is even achievable before interpreting). If the
  prod-vs-prod control shows token diffs → **STOP, root-cause determinism first**. Never substitute
  "statistically close".
- **B.** ADVERSARIAL CELLS AT PRODUCTION H — assertion level shifts from "no ties" to "identical row +
  identical op + identical order", with **ROW-LEVEL BITWISE asserts** (full score rows, not just top-k
  indices): (i) duplicate keys (repeated tokens/spans, padding sinks, quantized collisions) — tie
  frequency is H-independent and harmless GIVEN row-identity; (ii) ulp-boundary cells
  (accumulation-order 1-ulp row diffs → top-k flips); (iii) RoPE phase-aliasing (period-aligned
  positions); (iv) candidate-order cells at n>16 (coarse-pass order vs fallback order into the shared
  top-k op); (v) guard off-by-one: n ∈ {15,16,17} explicit at production H.
- **C.** 91K REAL-TENSOR CAPTURE REPLAY = **HARD R2 PRECONDITION** (not waivable). ALSO re-measure the
  fp32-row cost at 91K (0.063 ms was 20K; scales with ctx).
- **D.** Per-cell attribution artifact + superset check (see 3).

**R1 PRE-REGISTERED FAILURE BRANCHES:** ABORT (no ship, no re-amendment) on ANY of — any index diff at
production H on 20K OR 91K captures; any tie-classified diff at production H; any token diff vs
production on fixed replay; nondeterminism in the prod-vs-prod control; measured retention < 15 ms.
*Attribution:* benign arm HIER=0-without-L2-full **dirty** + L2-full **clean** → L2-full load-bearing,
proceed; both clean → cheap insurance, proceed; **L2-full dirty → STOP, root-cause OFFLINE** before
more cluster spend.

**R2 GATES:** 91K clean; token-identity clean WITH clean control; adversarial cells clean at production
H; `SMALLN_ROW_BF16=1` reproduces the old path bit-for-bit; default-ON verified in the DEPLOYED build
(installed-module grep both nodes); fresh boot + canary + battery + parity smoke; `known-good` tag.
**R3 = reserve only** (one pre-named retry).

**Evidence IDs (stable):**
| ID | artifact |
|---|---|
| E-H | production `index_n_heads=32` — `config.py:82` + node `config.json` |
| E-CENSUS | `scratch/p5/prodgate_full.log` — zero-col frac 0.2455/0.0594/0.00415 vs 2^-H 0.25/0.0625/0.00391 |
| E-STAB | `scratch/p5/prodgate_full.log` — L2-full vs HIER = 0 diffs @ H∈{8,32,64}, 258,048 slots |
| E-DET | `scratch/p5/prodgate_full.log` / `determinism.log` — self 0 / cross 0, 6 shapes |
| E-CLASS | `scratch/p5/classify_all_fixed.log` — fb 0 / hier 48 value-loss slots; max 9.73e-6 |
| E-SUITE | `scratch/p5/l2full_run{1,2}.log` (1794/251); ablations `suite_bf16_{0,1}.log` (939) |
| E-HEAD | `scratch/p5/headcount.log` |
| E-REPLAY | `scratch/p5/*` — `next18_functional.npz` replay 0/0 diffs, 320 slots |
| E-AMEND | `PHASE5-P1-AMENDMENT.md` @ `c4eedcb63` (head-count class table, value-loss matrix, determinism, amendment text) |
| E-KIT | `PHASE5-R1-KIT.md` @ `230ba71` (R1 runbook, all sections) |

**Status: the gate is RE-FROZEN under this amendment. Execution of R1 → R2 → R3 follows.**

---

## 15. ADDENDUM — D3 ATTRIBUTION ADJUDICATION (plan owner's delegate, 2026-10-08 ~22:00 CDT)

The raw artifact `attribution_251.json` is kept **untouched** (its machine `verdict` reads `FINDING`/
`ABORT`); that field is **superseded** by the ruling below. Full addendum: `attribution_251.addendum.md`.

**Ruling: requirement 3 = SATISFIED-WITH-REFINEMENT**, backed by **new mechanistic evidence** (the
per-cell artifact), per the one-amendment rule — not a failed-check waiver.

- 22,740 diff slots / 251 cells decompose: **10,406** exact-zero-column swaps + **12,288** masked/`-inf`
  padding (both value-equivalent = **99.8 %**) + **46** non-zero-column slots, **all in ONE cell**
  (`plain n=16 nb=16384 k=513 seed=385721`, fixture H=2).
- Mechanism for the 46: 1-ulp fp32 association-order difference between the two scoring paths;
  **direction: L2-full loses 0 value slots vs full-width fp32 truth, the SHIPPED HIER path loses 48 in
  that cell** — i.e. the residual is *HIER's own precision loss*, the exact phenomenon the ratified
  identity story already covers verbatim ("bitwise-identical at production H; strictly more precise than
  shipped at small H (0 vs 48 value-loss slots vs truth)").
- Superset check: **TRUE** (262 loss-checked records ⊇ 251 divergent cells; distinct identities 216/216).

**Contrary view recorded (transparency):** an independent consult (max effort) held that a verbatim
requirement cannot be satisfied by intent and recommended FAILED-as-worded pending owner re-ratification.
This adjudication is the owner's resolution of that question; it agrees on the mechanism and rules the
requirement SATISFIED-WITH-REFINEMENT. The owner's ruling governs.

**Consequence: R2 is no longer blocked on req-3.** R1 abort branches remain in force.
