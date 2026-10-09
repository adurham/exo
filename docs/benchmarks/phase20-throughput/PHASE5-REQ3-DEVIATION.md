# PHASE 5 — req-3 DEVIATION REPORT + DRAFT SECOND AMENDMENT (one-amendment rule)

Author: Phase-5 PM. 2026-10-08 ~21:40 CDT. **Offline.** No cluster contact.

## 0. Status

**Ratified requirement 3 is NOT MET under its verbatim wording.** The per-diff-slot attribution artifact
(`~/.hermes/cache/scratch/p5/prep/attribution_251.json`, produced by `bench/next18_attribution.py`,
mlx-lm commit `17bbd98`) returns **FINDING** — the class it demanded ("every diff sits in an exact-zero
column") is *the majority but not the whole* of the diff slots.

This is recorded as **FAILED-as-worded, pending owner reconciliation**. It is **not** recorded as
"accepted" or "satisfied in intent" anywhere.

## 1. The artifact, verbatim

```
suite_total_cells            : 2045
divergent_cells              : 251      (216 distinct identities)
total_diff_slots             : 22740
  total_zero_col_slots       : 10406    <- requirement-3's named class (exact-zero column)
  total_masked_equiv_slots   : 12288    <- masked / -inf padding (both arms select nothing)
  total_value_diff_slots     :  46      <- genuine non-zero-column 1-ulp fp32 accumulation-order diffs
cells_with_value_diff        : 1        (the §14 worst cell)
nonzero_column_diff_count    :  46
superset_check               : loss_checked 216 distinct identities  SUPERSET-OF  divergent 216  -> TRUE
verdict                      : FINDING — non-zero-column value-bearing diff slots present
                               (value-identity, not index-identity, holds at small H)
```

Reproduce: `cd /private/tmp/next18-lever2 && PYTHONPATH=$PWD /Users/adam.durham/repos/exo/.venv/bin/python bench/next18_attribution.py`

## 2. Why the wording fails (two independent reasons, both real)

1. **The 46-slot class.** 46 diff slots are in **non-zero columns** — genuine fp32 accumulation-order
   (≤1 ulp) differences, not ties. "Every diff in an exact-zero column" is false for them.
2. **The 12,288-slot padding class** (caught in review, would have failed a *second* time). Masked /
   `-inf` padding columns are value-equivalent (both arms select nothing there) but are **not** "exact-
   zero columns." A requirement scoped only around the 46 would fail again on this class. The taxonomy
   must name it.

## 3. The intent holds — but intent does not satisfy a verbatim requirement

- **Precision dominance (the substantive point):** in the one cell carrying the 46 slots, the L2-full
  path loses **0** value slots vs the full-width fp32 truth row, while the **shipped** hierarchical path
  loses **48** — i.e. L2-full is *strictly more precise* there. (`classify_all_fixed.log`, `losscell.log`.)
- **Production H is clean regardless:** at H=32 the two paths are **0-diff over 258,048 slots**; the
  adversarial suite at production H shows **0 row-bitwise mismatches and 0 ulp top-k flips**; determinism
  self/cross = **0/0/0**. (E-STAB/E-DET; `test_dsv41_indexer_adversarial_prodh.py`, mlx-lm `17bbd98`.)
- The `superset_check` leg of req-3 **does** pass (216 ⊇ 216).

Per the one-amendment rule, this is **new mechanistic evidence** — and it is a latent defect in the
criterion itself (masked padding slots exist by construction, so the criterion was unsatisfiable against
real fixture data). That is the legitimate use of the rule, not gate-escaping.

## 4. DRAFT SECOND AMENDMENT (proposed; NOT applied; R2 stays hard-blocked pending owner sign-off)

**Replace req-3 with a per-diff-slot attribution over a DEFINED SLOT TAXONOMY** — a small-H cohort cell
may be accepted iff **every** diff slot in it falls into one of:

- **(a) exact-zero-column swap** — both candidates' fp32 scores are exactly `0.0`;
- **(b) masked / `-inf` padding value-equivalence** — the slot is masked in both arms (both select
  nothing), so the ordering is value-null;
- **(c) bounded 1-ulp accumulation-order diff** — `|L2full − hier| ≤ 1 ulp` on the row, **in a cell where
  L2-full is precision-dominant** (loses **strictly fewer** value slots vs the full-width fp32 truth than
  the shipped hierarchical path).

Retain the **superset check** (divergent identities ⊆ loss-checked identities).

**Pre-registered failure branch (declare before spending):** **ABORT** (no acceptance, no ship) if **any**
diff slot falls outside (a)/(b)/(c), **or** if any cell has L2-full losing **more** value slots vs fp32
truth than the shipped path.

**Expected result under the draft amendment (from the current artifact):** (a) 10,406 + (b) 12,288 +
(c) 46 = 22,740 → **all slots classified**, superset TRUE, precision-dominance TRUE → **PASS**.

## 5. R1 disposition (recorded before spending)

- **R1 PROCEEDS.** R1 is non-shipping, runs at production H, and its data (91K capture replay at production
  H; greedy token-identity diff + prod-vs-prod determinism control; adversarial cells; battery) is
  required for R2 under every branch of the owner's decision. The ratified R1 abort branches are
  enumerated and none reference the small-H cohort, so req-3 is **not** a precondition to launching R1.
- **R2 is HARD-BLOCKED** on explicit owner re-ratification of the second amendment above. No inference
  from clean R1 results substitutes for the owner's amendment text.
- Guard against sunk-cost normalisation: a clean R1 does **not** make the req-3 reconciliation a
  formality.

## 6. Evidence IDs

| ID | artifact |
|---|---|
| E-ATTR | `scratch/p5/prep/attribution_251.json` + `bench/next18_attribution.py` (mlx-lm `17bbd98`) |
| E-LOSS | `scratch/p5/classify_all_fixed.log`, `losscell.log` — fb 0 / hier 48 value-loss slots |
| E-ADV | `tests/test_dsv41_indexer_adversarial_prodh.py` (mlx-lm `17bbd98`) — 45/45 identity, 0 row-bitwise mismatch, 0 ulp flips @ H∈{8,32,64} |
| E-FP32-91k | `scratch/p5/prep/fp32row_91k.log` — verify n=4 @ nb=45500 `+0.005 ms/call → ≈0.04 ms/round` |
| E-STAB/E-DET | `scratch/p5/prodgate_full.log` |
