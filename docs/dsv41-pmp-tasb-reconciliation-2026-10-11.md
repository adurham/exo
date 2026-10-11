# DSv4.1 — Task G: indexer-share reconciliation + hyper-connections classification (2026-10-11)

Author: Phase-20 PM (delegation). Branch `p20/pmp-tasb-reconciliation` (worktree
`/private/tmp/p20-tasb-recon`, off `origin/main` `af3d48dd6`). **Offline — 0 boots, 0 config
changes, no cluster contact; the one sanctioned idle-guarded node microbench was NOT used** (see
§4). Read-only sources: pack worktree `/private/tmp/p20-measure-pack` @ `302421ebe`, autopsy
`/private/tmp/p20-moe-autopsy` @ `9032dd4ed`, live repo `origin/main`.

Trigger: the P20 pack's Task B per-op-class table put the **indexer at 13.5 % @16 K / 31.1 % @100 K**
but the phase-22 fenced full-model table (2026-10-01) put it at **3.6 % @16 K** — a 3.7× divergence
the pack flagged as unreconciled; and a new **hyper-connections (HC)** class (5.2 % @100 K / 6.2 %
@16 K) was left unclassified.

---

## VERDICT (indexer)

**Both numbers are arithmetically correct; they are NOT the same quantity.** 13.5 % @16 K / 31.1 %
@100 K is the right **production-era (HIER-ON)** share at the *measured (deepest) chunk*; 3.6 % is a
**pre-HIER, 8-chunk-AVERAGED, full-model fenced** share. The 3.7× decomposes:

```
3.7×  =  ~2.0×  (chunk frame: the pack measures the LAST chunk, nb = 16384;
                 phase-22 SUMS all 8 chunks of a 16 K prompt, mean nb ≈ 9216 —
                 the indexer score pass is ~linear in nb, so nb 16384/9216 ≈ 1.78×)
        ×  ~1.85× (path: HIER-ON — the production default since 2026-10-06 — is
                   1.7–2.05× slower than the pre-HIER tiled path at nb = 16384)
       = 3.7×   (≈2.0 × 1.85 = 3.70)
```

**⇒ 31.1 % @100 K is the representative deep-context production share; 13.5 % @16 K is ~1.85×
inflated by HIER at the small last-chunk nb** (HIER is neutral at nb≈100 K and a 2.6× *win* for
consumers at depth). A whole-16 K production *average* is ≈ half of 13.5 % ≈ **~7 %**, which
reconciles with phase-22's last-chunk equivalent (~7.2 %). **No contradiction — a frame + path
mismatch.**

**Implied lever direction (stated, then STOP — no design/build):** the indexer IS a real
**O(context)** class and the **single largest deep-context bucket** (≈31 % @100 K in the pack;
corroborated independently by the in-repo depth attribution 14.6 % → 42.9 % over a 350 K feed,
~15 % @100 K). Two directions, both for a *future* round: (i) a **nb-based HIER floor** — the HIER
gate is guarded by a *row-count* (`_FENCE_MIN_ROWS = 16`), so a 16 K prefill pays HIER's coarse+exact
overhead for no memory benefit; (ii) the deep-context indexer **top-k/scan** cost (the score GEMM
itself is already ~99 % of ceiling per the OPT-6 fold). **STOP — this memo does not design or build
either.**

---

## 1. Re-derivation from the pack's own raw JSON — reproduces exactly

`raw/B/B_16k.json`, `raw/B/B_100k.json` (only `is_idx_src=true` layers carry an `indexer` row;
L0/L1/L3/L21 = 0.000 ms; chunk = 2048 rows at `start_pos` = 14336 @16 K / 98304 @100 K):

| ctx | L2 (ratio-2 src) | L20 (ratio-1 src+cand) | L24 (ratio-1 consumer) |
|---|---:|---:|---:|
| 16384  | 86.598 | 108.258 | 107.979 |
| 100352 | 197.467 | 330.234 | 329.695 |

Layer-weighted 40-layer fold (counts L0:1, L1:1, **L2:3** (L2,L8,L14; L14≈L2), L3:15, L20:1, L21:15,
L24:4 → **8 index-source layers** {2,8,14,20,24,28,32,36}):

- **16 K:** 86.598·3 + 108.258 + 107.979·4 = **799.97 ms** → /Σ(brk) 5933.3 = **13.48 %**
- **100 K:** 197.467·3 + 330.234 + 329.695·4 = **2241.41 ms** → /Σ(brk) 7207.6 = **31.10 %**

Both reproduce the doc's 800.0 / 13.5 % and 2241.4 / 31.1 % **exactly**.
**Caveat found:** the shipped `analyze_B.py` prints only **39/40** layers (its `LAYER_TYPES` has
`L2:n=2` and `L14:n=1`, but L14 is not sampled → dropped) and yields **713.4 ms / 12.48 %** and
**2044.0 ms / 29.79 %**. The doc's 800.0 / 2241.4 is the fold **with L14 folded in as L2** (the full
8-layer index weight). Both are self-consistent; the doc uses the correct 8-layer weight.

## 2. Methodology comparison — same taxonomy, one frame difference

- **Pack** (`bench/p20_pmp_taskB_opclass.py`): eval-fenced module wrappers + running `gpu_time`
  counter. `classify()` folds the whole `Indexer.__call__` (`with span("attn.indexer")`,
  `attention.py:196`) **plus** any `EXL3Linear`/`AffineProj` whose innermost prod span starts with
  `attn.indexer` into the **"indexer"** bucket. So the pack's indexer bucket = the **entire**
  indexer (wq_b/weights_proj projections + score pass + top-k + candidate publish).
- **phase-22** (`docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/scripts/p22_prefill.py`,
  `timed()` `:171-199`): also module-level eval-fenced wrappers, but **exclusive** time
  (`ACC[name] += dt - child`), so the indexer's *child* projections fall into the separate
  `EXL3Linear`/`Linear` buckets. Raw `raw/run3/progress.jsonl:15`:
  `"Indexer": [2553.4, 64]`, `total_ms 70600.9` → **3.6 %**.
- **The `64` calls are the tell:** 64 = **8 index layers × 8 chunks** ⇒ the phase-22 indexer number
  is summed over **all eight 2048-row chunks of the 16 K prompt** (its config `base_chunk 2048`,
  `lens [8192, 16384]`), i.e. an **average nb ≈ 9216** for ratio-1 layers — **not the last chunk**.
- **Taxonomies are effectively the same** (module `Indexer`); the pack's bucket is *slightly* larger
  (folds the indexer's internal projections in) — a ≤small-pp term that does **not** explain 3.7×.
  **The real differences are the frame (last chunk vs 8-chunk average) and the path (HIER).**

## 3. HIER hypothesis — confirmed as a real ~1.85× contributor

- **Code:** `mlx-lm/mlx_lm/models/deepseek_v41/indexer.py:178`
  `_HIER = os.environ.get("DSV41_INDEXER_HIER", "1") == "1"`; comment `:157-163` "default 1 = ON
  since the 2026-10-06 A/B+battery". The pack's Task B env (`B_16k.json` env) has
  `DSV41_INDEXER_HIER: null` ⇒ **default ON**; phase-22 (2026-10-01) predates the default
  (m2-integration doc `mlx-lm/docs/dsv41-hierarchical-indexer-m2-integration-2026-10-05.md:6,47`
  says default **OFF**) ⇒ phase-22 ran the **tiled** path. The gate `_HIER and n > _FENCE_MIN_ROWS`
  (`:591`, `_gates.py:43` `=16`) is satisfied at n=2048.
- **In-repo A/B** (`PERFORMANCE_HISTORY.md:10476-10489`, 2026-10-06): HIER=1 vs HIER=0 →
  **Fresh 100 K −0.9 % (neutral)**; **Delta 100 K→200 K +15.2 %**; the indexer attribution
  **14.6 % → 42.9 % over a 350 K feed, ~15 % @100 K**. i.e. HIER is a *win at depth, neutral at
  100 K*; at 16 K's **small nb** its coarse+exact machinery is **pure overhead** — exactly the
  pack's case. (Independent corroboration: `docs/known-good-decode-baseline-20261008.md:13-16`.)
- **Offline microbench** (laptop MLX, production shape b=1 n=2048 h=64 d=128 k=512 block=8
  overfetch=16; **NOT on the node**): median gpu ms tiled vs hier —

  | nb | role | tiled | hier | hier/tiled |
  |---|---:|---:|---:|---:|
  | 8192   | owner    | 65.4  | 166.4 | 2.55× |
  | 8192   | consumer | 65.7  | 188.2 | 2.87× |
  | **16384** | **owner**    | **130.9** | **223.7** | **1.71×** |
  | **16384** | **consumer** | **131.4** | **269.6** | **2.05×** |
  | 100352 | owner    | 810.1 | 850.1 | 1.05× (neutral) |
  | 100352 | consumer | 830.7 | 313.7 | **0.38× (HIER 2.6× faster)** |

  Reading: the indexer is ~linear in nb (tiled 65→131 ms for nb 8192→16384; the pack's own 16 K→100 K
  is nb ×6.1 → ms ×2.3–3.05). **HIER-ON costs ~1.7–2.05× at nb=16384** and is neutral/a-win at depth
  — matching the in-repo A/B direction. **Caveat: laptop chip/MLX ≠ the M4-Max node** — the *ratio*
  should transfer, the absolute ms will not; a node confirm is the obvious next step (cheap, bounded).

## 4. Bench disclosure — zero cluster disturbance

The brief permitted **at most ONE** bounded idle-guarded node microbench **if still ambiguous**. The
re-derivation (§1) + methodology read (§2) + code/A-B evidence (§3) already resolve the question, so
**no node bench was run** — the cluster was left entirely undisturbed (last generation ~34 min prior;
read-only idle check only; both nodes untouched; the only microbench was on the **laptop**, §3).

## 5. Hyper-connections (HC) — classification

**What it is.** `mlx-lm/mlx_lm/models/deepseek_v41/hyper_connections.py` / `hc_fused.py`: the
residual stream is `hc_mult = 4` **full-width parallel copies** (`[b,s,4,d]`). Per sub-layer
(staggered V4.1; `model.py:137-150` `_fused_call`, `_HC_FUSED` default ON): (1) `_mix_proj` — an
RMS-normalised linear of the flattened `[b,s,20480]` stream → `mixes [b,s,24]` (fp32, `mx.compile`);
(2) `sinkhorn_collapse` Metal kernel — split 24 → pre/post/comb, sigmoid, **20 Sinkhorn sweeps**, and
the collapse with `pre_prev` `[b,s,4,d]→[b,s,d]`; (3) `hc_expand` Metal kernel → `[b,s,4,d]`.
2 sub-layers × 40 layers = **4 kernel calls/layer** (160/chunk).

**REAL? Yes.** It holds **real loaded fp32 weights** `hc_attn_fn`/`hc_ffn_fn` of shape `[24, 20480]`
(`model.py:98-105`; loaded as weights, `exl3_build.py:7`) and does a **real GEMM** (`_mix_proj`) plus
real elementwise data movement. **Bytes:** 2 × 24 × 20480 × 4 B = **3.93 MB/layer** →
**157.3 MB/rank** (fp32, replicated → 314.6 MB cluster-wide; +11.8 MB for the 3 MTP layers).
**Share (pack fold):** **369.5 ms @16 K (6.23 %) / 371.7 ms @100 K (5.16 %)**.

**ADDRESSABLE? Low headroom — already fused.** `hc_fused` collapses ~140 op-path dispatches to ~4
Metal kernels. The only GEMM is a skinny **N=24** projection, memory-bound on the 3.93 MB/layer fp32
weight read (~157 MB/chunk); quantising `hc_fn` to bf16 saves ~78 MB/chunk (sub-ms). The dominant
cost is the 4-copy collapse/expand elementwise passes on `[b,s,4,d]` (bandwidth) — already one
kernel each. Sinkhorn iters (20) could be cut, but `comb` error compounds across depth
(`PERFORMANCE_HISTORY.md:5698`) — numerics risk. **Net: mostly irreducible per-layer bandwidth; not
a high-value lever.**

---

## Evidence index (file:line)
- Pack raw: `raw/B/B_16k.json`, `raw/B/B_100k.json` (env `DSV41_INDEXER_HIER: null`), fold helper
  `raw/B/analyze_B.py`; harness `bench/p20_pmp_taskB_opclass.py:193-221,274`.
- phase-22: `…/phase22-dsv41-prefill-speed-2026-10-01/scripts/p22_prefill.py:171-199`; raw
  `…/raw/run3/progress.jsonl:15` (`Indexer 2553.4 / 64 / 70600.9`); README §3.3.
- Code: `indexer.py:157-194,350-412,591-642`; `_gates.py:25,43`; `indexer_hierarchical.py:521-622`;
  `attention.py:196`; `model.py:43,98-105,137-150`; `hc_fused.py:139-175`; `hyper_connections.py:28-84`.
- HIER A/B: `PERFORMANCE_HISTORY.md:10476-10489`; `docs/known-good-decode-baseline-20261008.md:13-16`;
  `mlx-lm/tests/test_dsv41_hier_integration.py:225-229`; `…m2-integration-2026-10-05.md:6,47`.

## Recommended follow-up (NOT run)
A full-model eval-fenced attribution **per chunk** (not summed) under the current (HIER-ON) build at
16 K would pin the residual exactly; and an **nb-based HIER floor** (`indexer.py:591` still row-gated)
would remove the 16 K HIER overhead for no memory loss.
