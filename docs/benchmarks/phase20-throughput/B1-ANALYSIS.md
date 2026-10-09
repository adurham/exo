# B1 — token-63 indexer A/B probe: root-cause of the first divergence

Source: `token63.m4-1.npz`, `token63.m4-2.npz` (24 indexer call records c0000..c0023).
Arms: OFF = `_L2_FULL=False` -> bf16 stored score row (pre-ship); ON = `_L2_FULL=True` -> fp32 row (lever).
Both arms run the REAL `__call__` on IDENTICAL inputs, same as-found `shared.candidates`; ONLY the row dtype flips.
`_HIER=True`, `_FENCE_MIN_ROWS=16`, `_SMALLN_ROW_BF16=False` held constant.

Cross-capture A/B array identity (m4-1 vs m4-2): **True** (all idx/row arrays byte-equal).

## (a) Per-record table

`wrel = start_pos - gen0` (generation token index region). `set` = npz `ab_set_ndiff` (see NOTE).
`order` = npz `ab_order_ndiff` = # rank slots whose column differs. `relem` = `ab_row_elem_ndiff`.
`swaps` = our positional column mismatches; `tie/gen` = tie-break vs genuine splits; `sym` = true membership symmetric diff.

| node | wrel | layer | n | nb | k | ratio | path | rowdt | usesC | set | order | relem | swaps | tie | gen | sym | class |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| c0000 | 53 | 2 | 4 | 10067 | 512 | 2 | untiled | fp32 | False | 945 | 1574 | 40264 | 1574 | 1574 | 0 | 50 | TIE-BREAK |
| c0001 | 53 | 8 | 4 | 10067 | 512 | 2 | untiled | fp32 | False | 872 | 1322 | 40263 | 1322 | 1322 | 0 | 20 | TIE-BREAK |
| c0002 | 53 | 14 | 4 | 10067 | 512 | 2 | untiled | fp32 | False | 1153 | 1339 | 40262 | 1339 | 1339 | 0 | 12 | TIE-BREAK |
| c0003 | 53 | 20 | 4 | 20134 | 512 | 1 | untiled | fp32 | False | 711 | 1196 | 80526 | 1196 | 1196 | 0 | 30 | TIE-BREAK |
| c0004 | 53 | 24 | 4 | 20134 | 512 | 1 | untiled | fp32 | True | 856 | 1493 | 65520 | 1493 | 1493 | 0 | 26 | TIE-BREAK |
| c0005 | 53 | 28 | 4 | 20134 | 512 | 1 | untiled | fp32 | True | 803 | 1201 | 65518 | 1201 | 1201 | 0 | 22 | TIE-BREAK |
| c0006 | 53 | 32 | 4 | 20134 | 512 | 1 | untiled | fp32 | True | 759 | 1119 | 65520 | 1119 | 1119 | 0 | 28 | TIE-BREAK |
| c0007 | 53 | 36 | 4 | 20134 | 512 | 1 | untiled | fp32 | True | 374 | 1133 | 65520 | 1133 | 1133 | 0 | 18 | TIE-BREAK |
| c0008 | 54 | 2 | 4 | 10067 | 512 | 2 | untiled | fp32 | False | 502 | 1568 | 40265 | 1568 | 1568 | 0 | 32 | TIE-BREAK |
| c0009 | 54 | 8 | 4 | 10067 | 512 | 2 | untiled | fp32 | False | 523 | 1352 | 40265 | 1352 | 1352 | 0 | 14 | TIE-BREAK |
| c0010 | 54 | 14 | 4 | 10067 | 512 | 2 | untiled | fp32 | False | 545 | 1303 | 40264 | 1303 | 1303 | 0 | 22 | TIE-BREAK |
| c0011 | 54 | 20 | 4 | 20135 | 512 | 1 | untiled | fp32 | False | 617 | 1164 | 80533 | 1164 | 1164 | 0 | 26 | TIE-BREAK |
| c0012 | 54 | 24 | 4 | 20135 | 512 | 1 | untiled | fp32 | True | 828 | 1515 | 65522 | 1515 | 1515 | 0 | 30 | TIE-BREAK |
| c0013 | 54 | 28 | 4 | 20135 | 512 | 1 | untiled | fp32 | True | 1091 | 1182 | 65525 | 1182 | 1182 | 0 | 32 | TIE-BREAK |
| c0014 | 54 | 32 | 4 | 20135 | 512 | 1 | untiled | fp32 | True | 473 | 1081 | 65525 | 1081 | 1081 | 0 | 14 | TIE-BREAK |
| c0015 | 54 | 36 | 4 | 20135 | 512 | 1 | untiled | fp32 | True | 659 | 1095 | 65525 | 1095 | 1095 | 0 | 16 | TIE-BREAK |
| c0016 | 56 | 2 | 4 | 10068 | 512 | 2 | untiled | fp32 | False | 991 | 1585 | 40270 | 1585 | 1585 | 0 | 38 | TIE-BREAK |
| c0017 | 56 | 8 | 4 | 10068 | 512 | 2 | untiled | fp32 | False | 785 | 1363 | 40269 | 1363 | 1363 | 0 | 22 | TIE-BREAK |
| c0018 | 56 | 14 | 4 | 10068 | 512 | 2 | untiled | fp32 | False | 397 | 1338 | 40270 | 1338 | 1338 | 0 | 10 | TIE-BREAK |
| c0019 | 56 | 20 | 4 | 20137 | 512 | 1 | untiled | fp32 | False | 623 | 1190 | 80541 | 1190 | 1190 | 0 | 12 | TIE-BREAK |
| c0020 | 56 | 24 | 4 | 20137 | 512 | 1 | untiled | fp32 | True | 878 | 1507 | 65523 | 1507 | 1507 | 0 | 32 | TIE-BREAK |
| c0021 | 56 | 28 | 4 | 20137 | 512 | 1 | untiled | fp32 | True | 397 | 1187 | 65526 | 1187 | 1187 | 0 | 18 | TIE-BREAK |
| c0022 | 56 | 32 | 4 | 20137 | 512 | 1 | untiled | fp32 | True | 643 | 1116 | 65524 | 1116 | 1116 | 0 | 12 | TIE-BREAK |
| c0023 | 56 | 36 | 4 | 20137 | 512 | 1 | untiled | fp32 | True | 452 | 1088 | 65523 | 1088 | 1088 | 0 | 4 | TIE-BREAK |

**NOTE on `ab_set_ndiff`:** in `token63_probe.py:767-775` it is `(set_off != set_on).sum()` where `set_*` are *positional* top-k (element-wise over identical shapes), i.e. it duplicates `ab_order_ndiff`'s positional semantics, NOT a set difference. It is therefore NOT a membership metric — the true membership symmetric difference is the `sym` column (tiny). Do not read `ab_set_ndiff` as 'columns that swapped in/out of the set'.

## (b) First-divergence verdict

- **node**: `c0000`  **window_rel**: `53`  **layer_id**: `2`
- **op**: indexer top-k column selection (ranked order) via dtype-dependent tie-break at the stored score row.
- n=4, nb=10067, k=512, ratio=2, path=untiled (untiled), row_dtype(fp32).
- order_ndiff=1574 rank slots differ, all pure ties (gen_swap=0); membership symdiff=50.
- First swapped rank position b=0: pos 26 (0-based), columns 10026 <-> 10051,
  both bf16 values = -0.5703125 (OFF arm) → gap **0.0** -> exact tie.
- **margin**: bf16 boundary gap between k-th and (k+1)-th score, per query row = [0.0, 0.0, 0.0, 0.0] (K=512) -> the entire top-512 selection boundary is tied (0.0).
  (All 24 records: max bf16 gap among ALL swapped columns = **0.0**.)

## (c) Tie-break vs genuine re-rank

- Divergent records: **24/24**.
- Tie-break records: **24/24** (gen_swap=0 in every record).
- Genuine re-rank records: **0/24**.
- Total swapped rank-slots across all records x 4 query-rows: 31011; tie-break 31011, genuine 0.

Proof of pure tie-break: bf16-rounding the fp32 arm's full stored row (`rn`) reproduces the bf16 arm's row (`ro`) **exactly** — 96/96 query-rows match bit-for-bit (worst finite discrepancy 0.0). The two arms therefore carry the SAME bf16-rounded score row; only the fp32 row preserves sub-bf16 distinctions. Ties are pervasive: in c0000 the bf16 row has ~9,850/10,067 finite values collapsed to ~200 unique magnitudes (~98% tied).

## (d) Consumer-layer coverage

- Records with `uses_candidates=True`: **12** (c0004, c0005, c0006, c0007, c0012, c0013, c0014, c0015, c0020, c0021, c0022, c0023).
- Of those, diverging: **12/12** — ALL of them.
- This is the part the offline R1b capture could not resolve: consumer-layer (uses_candidates) indexer calls DO exhibit the same tie-break divergence.

## (e) lg0000 logits record

- ONE step only (`meta_nlogits=1`), window_rel=60, start_pos=20137.
- top_ids (rows=4 query slots): [[20, 7835, 19, 16176, 1602], [201, 7835, 15090, 4588, 271], [7835, 21, 1613, 3108, 15090], [12747, 50249, 2581, 43, 5718]]
- margin12 (top1-top2 logprob): [9.234375, 9.203125, 1.875, 4.71875]
- margin13 (top1-top3 logprob): [13.65625, 9.703125, 11.703125, 6.265625]
- selected logprobs: [-9.918212890625e-05, -0.0001678466796875, -0.1426849365234375, -0.01425933837890625]  (top_ids_sha=1c31dbd7c588e71b)
- Large margins (~9.2, 1.9, 4.7 logprob) at this single captured step — no near-tie flipping in the logits, but nlogits=1 gives no sequential evidence.

## (f) Limitations (explicit)

- **nlogits = 1**: the logits capture is a single step at window_rel=60 — no logits were recorded at the first divergent token (wrel=53). The logits margin cannot be attributed to this divergence.
- **nrounds = 0**: no round-shape hook records were captured (the `spec.generate` wrapper was not on the hot path for this run). No round data exists — not invented.
- **Window coverage**: call records cover window_rel in {53, 54, 56} (layers 2,8,14,20,24,28,32,36 at each); logits cover wrel=60. `meta_window=[55,63]` is the configured band; the recorded calls land slightly below it because `window_rel` is measured at the forward's `start_pos` and a step's tokens straddle the band edge. No records exist between the first divergence (wrel=53) and the production divergence token (63) — the sequential gap is UNOBSERVED.
- **`ab_set_ndiff` semantics** (see (a) NOTE): it is a positional comparator mislabeled as a set comparator; this analysis used the true membership symmetric difference (`sym`) and the per-slot tie test instead.
- All 24 records are `path=untiled`, `ratio∈{1,2}`, n=4. No tiled/hier-path or n>16 records in this window.

## Verdict

The first divergence is a **pure dtype tie-break** at window_rel=53, layer 2, node c0000: the bf16 arm's top-512 boundary (k-th vs (k+1)-th score) is EXACTLY tied (margin 0.0), and the two arms emit different-but-score-equal columns. No genuine re-rank anywhere in the window, including all 12 consumer-layer records. The divergence is a coin-flip among tied scores, deterministic within a build.
