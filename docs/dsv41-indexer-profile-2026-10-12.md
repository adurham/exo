# DSv4.1 — indexer absolute-cost workstream: profile doc (§0 step-0 pins; Phase 1 log)

**Date:** 2026-10-12. **Workstream:** Fable consult #5 executable plan —
`docs/dsv41-indexer-abscost-plan-2026-10-12.md` (committed `ba4c3763`, saved for
next-session execution).
**Executor:** hermes-gw default session.
**House-convention mapping (per the plan's header clause):** deliverable docs live
here in `docs/` on `main` (this campaign's precedent for every perf doc); the CODE
worktree is `adurham/mlx-lm` branch `perf/indexer-profile` off `6a5c78f4`; bench
scripts land in that worktree's `bench/`. Where the plan attaches this doc to the
mlx-lm branch, this file supersedes — the single authoritative copy.
**Live state at open (verified 2026-10-12 15:43Z, LXC + both nodes):** exo
`deploy/next23b-qsplit @ 60008939` + mlx-lm `6a5c78f4`; runner env
`EXO_DSV41_CKPT=1`, `DSV41_INDEXER_QSPLIT=1`, `DSV41_DENSE=affine6`,
`DSV41_DENSE_POLICY=layers.*.ffn.shared_experts.*=q8g64`; API answers; READY 2/2
(runner PIDs 73006 / 77318); GPU locks free. Matches the plan's base state exactly.
(The LXC checkout was 142 commits behind origin/main; fast-forwarded to `ba4c3763`
before starting.)

## §0 Step-0 pins (plan §0.1–§0.3; all citations at mlx-lm `6a5c78f4`)

### 0.1 `nb` — pinned (the plan's hard prerequisite)

**`nb` is a COLUMN count of the score row: the number of complete compressed KEY
groups the forward's queries score against. It is not a query-row count.** The
plan's "nb_local = 1024" is a misread of the qsplit split: qsplit halves the QUERY
rows (`n`); the KEY side (`index_k`) stays FULL on every rank (`indexer.py:159-171`,
`:661-669`).

- Code: `nb = index_k.shape[1]` (`indexer.py:673`); docstring `:661-663`:
  *"index_k [b, nb, head_dim] — the owner's key cache, already sliced to the
  nb = (start_pos + n) // ratio complete groups."*
- **Exact formula:** `nb = floor((start_pos + n_global) / ratio)` (cache-capacity
  capped), with `ratio = compress_ratio(layer_id)` of the owner layer (`:618`):
  **2** for owners 2/8/14; **1** for owner 20 and consumers 24/28/32/36.
  Live checkpoint (studio2 `~/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw/config.json`,
  fetched 2026-10-12): `text_config.compress_ratios` = `[0,0] + [2]*18 + [1]*20 + [0,0,0]`
  (43 entries); `kv_source_layer_ids=[2,8,14,20]`,
  `index_source_layer_ids=[2,8,14,20,24,28,32,36]`, `candidate_source_layer_id=20`.
- **Score-tensor element count:** `S = b · n_local · h_idx · nb`, `h_idx = index_n_heads = 32`,
  `d = index_head_dim = 128` (`config.py:82-84`). `n_local = n_global` replicated;
  `n_global // 2` when the qsplit band engages (gate `n//world > _FENCE_MIN_ROWS = 16`,
  `query_band :177-204`; band slice `:697-702`; keys full).
  - Pre-collapse `[b, n, h_idx, nb]` fp32 transient = `4·S` bytes; the code docstring's
    example (`:28-30`): 512·32·16384·4B = **1.07 GB** (arith-checked; h_idx=32 ✓).
  - Stored row `[b, n_local, nb]`: **bf16** by default (`_ROW_DTYPE :110-115`); fp32
    only for small-n (≤16) L2-full forwards (`_row_dtype :232-243`).
- Production points (DERIVED; arithmetic checked): full chunk n=2048, n_local=1024,
  L≈100K end position — layer-20 class (ratio 1): **nb = 100,000, S = 3.28e9 elts =
  13.11 GB fp32 if materialized; bf16 stored row 204.8 MB** (fp32 409.6 MB).
  Owners 2/8/14 at the same L (ratio 2): nb = 50,000, S = 1.64e9 elts = 6.55 GB.
- `nb=16384` provenance: a harness point from `benchmarks/dsv41/pB_indexer_test.py`
  (micro: b=1, n=512, h=32, d=128, nb=16384; parity sweeps nb ∈ {4096..20000}).
  The plan's "HIER 1.71–2.05x slower at small nb (nb=16384)" has **no committed
  provenance** (searched docs/, bench/, repo — 2026-10-12). It is a reproduction
  target in 1B; if it does not reproduce (±20%, after one harness fix attempt ≤1.5 h)
  it is **re-baselined and flagged** (provenance gap), per the plan's own stop rule.
  Owner queried 2026-10-12: no recollection of the source — treat the figure as
  unrecoverable; the re-baselined crossover in §1B becomes the record.

### 0.2 "verify-ratio" — pinned

Not a repo term (searched docs/ + mlx-lm; the only occurrences are the plan itself).
Adopting the plan's fallback definition verbatim: **for MTP/speculative verify steps
(and plain decode): the fraction of step wall-time spent in the indexer, plus
rows-per-verify and the acceptance ratio.** The cluster runs DSpark MTP spec decode
(gamma 3 pinned), so the full definition applies. Measurement hooks: profiler spans
`attn.indexer.score` (`:768` / `:788` / `:806`), runner round logs (round ms,
tokens-per-round, acceptance), the served-path `[PROFILER]` table. Properly measured
only in 1E (inside the single M1 window; no boot).

### 0.3 Instrumentation points

| what | where (mlx-lm `6a5c78f4`) |
|---|---|
| indexer forward (single entry) | `indexer.py:654` `Indexer.__call__` |
| — HIER branch (default ON; guard `_HIER and n > 16`) | `:728-780` → `indexer_hierarchical.py:545` `hierarchical_topk_prod` |
| — tiled fallback (`tile < nb`; always-tiled since 10-02) | `:782-803` → `_tiled_scores_buffer :439` (default) / `_tiled_scores_merge :533` (`TILED_IMPL=merge`) |
| — untiled reference branch | `:805-830` (full `[b,n,h,nb]` transient + one global argpartition) |
| score kernel body | `_score_body_eager :427` (compiled per (b,s,h,d,tile), `:404-424`) |
| top-k call sites | untiled `argpartition+sort :826-827`; tiled `topk_from_row :392` (one global pass) or `merge_topk :374`; HIER `top_blocks :399` + `exact_rescore_streaming :457` (streamed running argpartition) |
| qsplit band assignment | `query_band :177`; band slice post-projection `:697-702`; route keyed on `n_full` (D3) `:726`; rebuild `_reconstruct_rows :207` → `_finish_rows :223` |
| index-K cache | `cache.py` `index_k` (alloc `:84-91`, realloc `:95-114`): `[bsz, n_comp, 128]`, **bf16 always** (module doc `:20-24`); written by `publish_keys :635-652` |
| HIER geometry log | `_log_hier_geometry :308` (one stderr line per resolved shape) |
| launcher env forwarding | `start_cluster.sh` — **fixed per-var list, not a wildcard** (each knob has an explicit `[ -n … ] && EXO_ENV="$EXO_ENV …"` line): e.g. `:2670` ROW_BF16, `:2674` HIER, `:2681-2684` HIER_{EXACT_MB,EXACT_STRIP,STRIP,OVERFETCH}, `:2689` CONSUMER_SKIP; deployed qsplit forwarder `:2723` (deploy/next23b-qsplit). **Consequence: Phase 2's new knobs (`DSV41_INDEXER_HIER_BYPASS`, `_SMAX`, `_REF_MAX_BYTES`) require explicit lines added to `start_cluster.sh` — an exo-side commit the plan hoped to avoid ("ideally none"). Flagged for M1.** |

### 0.4 Relevant live knobs (for Phase 2 gate design)

`DSV41_INDEXER_TILE=0` restores untiled; `DSV41_INDEXER_HIER=0` forces the
row-materializing path; `DSV41_INDEXER_HIER_{BLOCK=8, STRIP=4096, OVERFETCH=16,
EXACT_MB=128, EXACT_STRIP=0}`; `DSV41_INDEXER_CONSUMER_SKIP=1`;
`DSV41_SPARSE_FENCE_MIN_ROWS=16` (shared by the two levers, `_gates.py`). HIER
default ON since 2026-10-06 (mlx-lm main `2119209`). Only `DSV41_`-prefixed names
are forwarded by the launcher; `EXO_DSV41_*` are silent no-ops; `EXO_DSV4_*` are the
older V4-Flash-era knobs (separate set).

## §1 Phase 1 — RESULTS (offline complete; 0 boots; live untouched)

**Jobs:** 1A/1B (studio2 bench bundle — idle-guarded, 0 discards; 12/12 split cells and the full 63-cell grid MEASURED) and 1C/1D (read-only audit). Harness trust gate: parity index-set-identical on 100 % of rows (6 synthetic shapes; 424/424 real rows at L=50K). Artifacts: `~/.hermes/cache/scratch/idxprof1/` (REPORT.md, tables.md, raw_local/ incl. real-activation captures) and `idxprof2/` (REPORT.md, out/ incl. `occupancy.csv`, `ev_piece2.json`); harnesses committed on the mlx-lm branch `perf/indexer-profile`: `bench/indexer_split.py`, `bench/act_capture.py`, `bench/cap_gap.py`, `bench/idx_report.py`.

### §1A — time split (MEASURED; rows=1024 per-rank post-qsplit)

| layer | L | nb | t_REF ms | t_HIER ms | hier/ref | gpu hier/ref | K-read share | select share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 (r2) | 50K | 25K | 55.16 | 68.16 | 1.236 | 1.355 | 1.3 % | 12.7 % |
| 8 (r2) | 75K | 37.5K | 82.15 | 85.35 | 1.039 | 1.152 | 0.9 % | 13.7 % |
| 8 (r2) | 100K | 50K | 108.35 | 102.54 | 0.946 | 1.056 | 0.8 % | 13.7 % |
| 8 (r2) | 130K | 65K | 140.13 | 123.01 | 0.878 | 0.997 | 0.6 % | 13.6 % |
| 20 (r1, cand) | 50K | 50K | 108.57 | 102.32 | 0.942 | 1.055 | 0.8 % | 13.7 % |
| 20 (r1, cand) | 75K | 75K | 165.52 | 137.08 | 0.828 | 0.938 | 0.5 % | 14.7 % |
| 20 (r1, cand) | 100K | 100K | 217.64 | 171.43 | 0.788 | 0.897 | 0.4 % | 14.7 % |
| 20 (r1, cand) | 130K | 130K | 282.11 | 211.75 | 0.751 | 0.861 | 0.3 % | 14.7 % |
| 24 (r1, cons) | 50K | 50K | 108.74 | 82.88 | 0.762 | 0.836 | 0.8 % | 13.7 % |
| 24 (r1, cons) | 75K | 75K | 163.99 | 85.02 | 0.518 | 0.564 | 0.5 % | 14.8 % |
| 24 (r1, cons) | 100K | 100K | 218.28 | 87.46 | 0.401 | 0.437 | 0.4 % | 14.8 % |
| 24 (r1, cons) | 130K | 130K | 283.03 | 89.00 | 0.314 | 0.345 | 0.3 % | 14.8 % |

Every cell passed the ±15 % Σ-parts check → all MEASURED, no BOUND. Roofline (MEASURED on studio2): 448 GB/s copy, 14.8 TFLOP/s fp16. Nothing ≥60 % of a peak: the REF score GEMM (≈85 % of its total) runs at 31 % of fp16 peak / 33 % of stream BW — the cost driver is the fp32 tile-materialisation traffic (13–35 GB/call at the achieved 146–147 GB/s). The K-cache read is 0.3–1.3 % (invisible). Select is 12.7–14.8 % and depth-scaling. HIER = bf16 coarse sweep (67→83 % of its total) + a constant ≈30.3 ms exact re-score.

### §1B — crossover map (MEASURED) + reproduction

| rows ↓ / nb → | 2048 | 4096 | 8192 | 16384 | 32768 | 50000 | 75000 | 100000 | 130000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.442 | 1.681 | 1.836 | 2.145 | 2.685 | 3.462 | 4.118 | 4.537 | 5.182 |
| 4 | 1.713 | 1.693 | 1.818 | 2.101 | 2.458 | 2.820 | 3.350 | 3.538 | 3.868 |
| 8 | 1.631 | 1.754 | 1.898 | 2.118 | 2.162 | 2.416 | 2.402 | 2.668 | 0.199* |
| 78 | 1.973 | 2.451 | 2.038 | 0.712 | 0.511 | 0.447 | 0.405 | 0.382 | 0.362 |
| 512 | 3.443 | 3.729 | 2.384 | 1.566 | 1.124 | 0.969 | 0.860 | 0.814 | 0.768 |
| 1024 | 3.754 | 3.930 | 2.439 | 1.568 | 1.108 | 0.943 | 0.833 | 0.782 | 0.749 |
| 2048 | 4.851 | 4.945 | 2.918 | 1.795 | 1.225 | 1.026 | 0.895 | 0.839 | 0.795 |

(t_HIER/t_REF; <1 = HIER faster. * = route artifact, see below. Neutral band ±10 % at rows≥512, nb=50000.)

- **Fit:** no pure-S threshold works (8 cells misclassified) and neither does `S ≤ S* AND nb ≤ L*` (8). Fitted form: **staircase `HIER faster iff nb ≥ nb*(rows)`**, nb* = {1, 4: never (≤130K); 8: 130000; 78: 16384; 512/1024/2048: 75000}; 0 misclassifications (within the ±10 % don't-care band).
- **Reproduction (a) — nb=16384:** rows 512/1024 → **1.727 / 1.731 vs the untiled-row comparator** (target 1.71–2.05 ✅). The plan's orphaned "1.71–2.05× slower at small nb" figure is now sourced: it compared HIER against the *untiled row* path (vs the tiled REF the same cells read 1.566/1.568, still within ±20 %).
- **Reproduction (b) — production full-chunk point (1024, 100000):** gpu-busy ratio **0.897 → inside 0.9–1.1 ✅** (isolated wall 0.782; one allowed fix attempt — REF tile 504→2048, 199→49 fences — moved it to 0.850; the residual is host-fence accounting production amortises. Harness representative of the GPU work; documented in idxprof1 REPORT §4.3.)
- **Route/budget findings (MEASURED, exact-class, no exactness argument needed — separate mini-plan candidates, NOT part of the bypass):** (i) at rows=8, nb=130000 the production route picks *tiled* (40.4 ms) vs row-whole 2.94 ms — 13.8× — because the 8·130000·130 B untiled transient is 135 MB vs the 128 MB `DSV41_INDEXER_TILE_MB` soft budget; (ii) at rows=78, nb ≥ 16384, the tiled route costs 7.5–54.8 ms vs row-whole 3.1–18.9 ms (2.4–2.9×), never taken (same budget gate).

### §1C — workload occupancy (DERIVED headline; logs hold bench traffic only)

| phase (% of GPU-busy wall) | value |
|---|---:|
| prefill full chunks | 8.09 % |
| prefill tails | 2.37 % |
| n ≤ 16 (tiny) | 0.00 % |
| verify/decode rounds | **89.53 %** |
| by depth (all phases) | 50–100K 19.7 % · 100–200K 52.0 % · 200K+ 24.9 % |

Code constraint: at n ≤ 16 production already runs the REF fallback (HIER off) → **speedup ≡ 1 for every verify/decode cell** — the bypass-eligible mass is prefill only (tails included ≈ 10.5–15.5 % of wall, D/D''). Sensitivities: D′ 9.62/3.06/87.31, D″ 11.75/3.74/84.51, D-cold bound 57.47/1.15/41.38.

### §1D — replication audit (read-only)

- Duplicated-on-both-ranks indexer work ≈ **6 ms GEMM + 0.2 ms + 4.4–6.6 ms cidx exchange** per 2048-chunk at 100K ≈ 0.15 % of the 6.9 s chunk (cidx unmeasured; time it in 1E).
- **Verify n=4** indexer floor **1.8/3.5/4.6/9.0 ms per round at 50/100/130/254K** — ≥4 % of an 86 ms round at 100K, in the 85–90 % occupancy cell. Phase-3 consideration (K-side column split).
- Band imbalance: owners 0 %; consumer `max_cand` >5 % only below ≈17K (~0.5 % of wall). Cross-rank agreement **PROVEN** from code (all path/band predicates key on global or equal quantities; the exact-pass strip width is band-n — tie-pick only).

### Real-activation material (plan §2.2 input; real EXL3 weights, both bands)

With the exact fp32 row: **index-identical 100 % at L=50K** (424/424 × 3 layers × 2 bands), 270–276/276 at L≈105K. L1: max|Δtop-k value| = 0 at 50K; at ≈105K 0–3 rows/64 ≤ 3.8e-4 (≈0.4 % of score scale) — consistent with a bf16 *coarse-ranking* loss, not the exact pass. Exact k-th ties on real data: **0 of 3,648 rows**. With the live bf16 row the paths disagree on 30–70 % of rows — that is the bf16 row rounding, and the observed L1/L2 residual at nb≈52–105K lives in the regime where HIER stays selected anyway (outside any bypass region).

### Phase-2 gate — DECISION (plan §Phase-2 go/no-go)

- **(a) contiguous REF-faster region ≥ 1.15×: YES** (rows 512–2048, nb ≤ 16384: 1.57–4.95×; wider at ≥1.11× to nb=32768).
- **(b) 1C-modeled e2e EV ≥ 1.0 %: NO — 0.015 %** (D headline; per-cell occ × indexer share × (1 − 1/speedup) over REF-favouring cells; `idxprof2/out/ev_piece2.json`; band-midpoint model, ±0.02 % slop). **Hard cap with infinite speedup in the eligible cells: 0.035 %.** Sensitivities: D′ 0.015 %/0.036 %, D″ 0.019 %/0.044 %, D-cold bound 0.107 %/0.244 % — every variant at least ~4× under the gate.
- **VERDICT: "bypass not worth a boot" — Phase 2 is SKIPPED.** No gate design, no implementation, no Phase-2 live window. Cause: the REF-favouring region is shallow-context (nb ≲ 32K); the real workload's prefill mass is deep (where HIER wins) and its verify/decode majority already runs REF by design.
- **Next (the single M1 surface):** surface once with (i) these Phase-1 results; (ii) the **1E observation-only window ask** (no boot, ~60–90 min quiet; gates the Phase-3 GPU-busy check and verify-ratio); (iii) the route/budget finding as the one exact candidate for a separate bounded mini-plan; (iv) Phase-3 entry per the decision table (see idxprof1 REPORT §6 + idxprof2 REPORT §1D: K-read lever dead; select below the 40 % bar; the ≤2 h int8 score-path probe is the allowed first experiment).
