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

## §1 Phase 1 log

- 2026-10-12 (~15:50Z) — §0 pinned (this doc's first committed revision). Phase 1
  offline dispatched as two jobs: **1A/1B** (studio2 bench bundle — idle-guarded,
  no boots) and **1C/1D** (read-only occupancy + replication audit). Live cluster
  untouched. 1E waits for the single M1 window.
- (pending) §1A time-split tables; §1B crossover map; §1C occupancy; §1D audit —
  appended here as they land.
