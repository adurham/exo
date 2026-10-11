# DSv4.1 MoE prefill autopsy — SECONDARY findings (read-only code read + doc arithmetic)

Author: secondary (mid-tier delegation). No node, no bench, no GPU. All claims cite file:line.
Parent report is the PM's — this note is input only.

## 1. Expert placement — TP-split (intermediate-width), NOT expert-locality

- `mlx-lm/mlx_lm/models/exl3/loader.py:286-299` — `load_experts` calls `_slice_intermediate`
  when `world>1`; `_slice_intermediate` (`:174-208`) cuts the gate/up **out-tiles**
  `t0:t1 = rank*hpt:(rank+1)*hpt` (`hpt=gu_tiles//world`) and the down **K-tiles**
  `k0:k1 = rank*kpt:(rank+1)*kpt` (`kpt=hid_tiles//world`). gu_suh/dn_svh replicated.
  → the intermediate axis is split by rank; each rank holds **half the width of EVERY expert**.
- `mlx_lm/models/deepseek_v41/exl3_build.py:508` — `blk.ffn.experts = Exl3Experts(load_experts(ck, layer_id, rank=rank, world=world))`;
  **all** experts are built on every rank (no expert subsetting). `:509-510` `if world>1: blk.ffn.group = group`.
- `mlx_lm/models/deepseek_v41/moe.py:180-189` — one `_all_sum_tail` per layer after the routed sum.
- `src/exo/worker/engines/mlx/dsv41/load.py:9-11` docstring: experts "take one rank's
  intermediate-width slice and are all_sum'ed". `tp_geometry` (`:68-96`) allows world∈{1,2} only.
- Confirmed geometry: per-rank H = 1152 = `moe_intermediate_size` 2304 / world 2
  (`bench/dsv41_loop2/moe_bench_report.md:63-67`; `docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/README.md:137`).

Effect on M-mix: **none.** avgM = tokens·topk/E = 2048·6/384 = **32** at chunk 2048, E=384
(`moe_bench_report.md:156`). Placement halves the per-rank GEMM N (gate/up) and K (down), i.e.
per-rank FLOPs halve (M=32: 2.26→1.13 GFLOP/expert-slot) — it does **not** split the routed M.
Drain added: one full `[tokens, 5120]` all_sum / layer / chunk = 125.8 MB bf16 ×40 = **5.03 GB/chunk**
(`moe.py:56-63`, `_MOE_ALLSUM_BF16` default bf16). EP/experts-locality would instead halve per-rank M
but require all-to-all dispatch+combine with real imbalance (routing churny; top-24 Jaccard 0.26,
`docs/PERFORMANCE_HISTORY.md:11953-11958`).

## 2. Dual-microbatch drain comm-hiding — ABSENT

- `moe.py:56-63` `_all_sum_tail` = plain `_coll.all_sum` (`deepseek_v41/collective.py:73-77`); no
  overlap path. `moe.py:183-189`: collective issued inline once per layer, result consumed by the
  shared-expert add immediately after → serialized on the same GPU-stream FIFO; `collective.py:1-17`
  documents the Metal-command-buffer wait on the JACCL CPU-stream collective.
- `prefill.py:60-63` "bounded async queue" = host-side queue bound (oldest synced when exceeded),
  NOT drain/compute overlap. `session_cache.py:154-156` `chunked_prefill` accepts `async_depth` and
  **ignores** it. exo dsv41 engine has no micro-batch split (`agreement.py:6`, `engine.py:787` only).
- The V4 overlap probe (`PERFORMANCE_HISTORY.md:344-350`: same-stream `overlap_ratio=0.995`;
  2nd stream near-perfect) shows overlap is *possible* — it is simply **not implemented**.
  → present/absent = **ABSENT**; a genuine, unpriced TP lever.

## 3. Reprice the 15.14 TF ceiling calibration

- Calibration (`moe_bench_report.md:29-51`): dense **bf16-in / fp32-accum** `mx.matmul`, LARGE shapes
  up to 4096×8192×8192 = 15.14 TF; **no dequant** (pure MMA); ~450 GB/s triad (449/435/472).
- Crossover M (compute↔BW) = 15.14·b/(BW·16): at 450 GB/s → b=2:**4.2**, b=3:**6.3**, b=4:**8.4**;
  at 550 GB/s → b=2:3.4, b=3:**5.2**, b=4:**6.9**.
- Production per-expert M ≈ **32-48** ≫ crossover → **compute-bound**. The BW term is huge
  (M=32,b=3 → 76.8 TF) so it never binds; **15.14 TF is the correct denominator.**
- ⇒ the achieved 51-60% (uniform 8.88 / skew 7.55 TF, `moe_bench_report.md:143-146`) is a **real
  ~40-49% below-ceiling gap** (ALU/occupancy/issue), NOT a shape-roofline artifact. Fable's Loop-3
  "if mean M~16 the gap is the shape roofline" hypothesis is **falsified by the geometry**.
- Caveats to record: (1) calibration had no fused-kernel dequant/Hadamard/scale ALU → the *achievable*
  ceiling for the fused kernel sits somewhat below 15.14 (that extra ALU is the kernel-loss suspect,
  not a calibration defect); (2) dtype matches the kernel's MMA precision; (3) **82-TF lesson**
  (`moe_bench_report.md:192-202`): 82 ≫ M4 Max ~16 TF silicon → physically impossible, retired;
  max anywhere = 9.58 TF < 15.14, no >30 TF alarm — no recurrence.

## 4. Wall model closure with shipped spans

Best current DSv4.1 full-model prefill decomposition = **phase22 README.md:116-129** (16K, fenced,
**sums to 100%**):

| bucket | share |
|---|---|
| EXL3SwitchGLU (routed MoE) 30.7 + MoE 0.6 + Gate 0.5 | **≈31.8%** (Loop-2 "MoE block ~⅓", `PH:11369`) |
| all_sum (collectives) | 16.9% (fenced-inflated) |
| EXL3Linear (dense projections) | 16.6% |
| sparse_attn (sdpa) | 15.6% (Loop-2 memo quotes 21%, `PH:11372`) |
| LazyEngramTable | 6.0% |
| RMSNorm | 4.1% |
| Indexer | **3.6%** |
| Exl3Member / remainder | 1.1% / 1.7% |

The parent's **~40% unaccounted** = the non-MoE/non-sdpa buckets: dense EXL3 16.6% + collectives
5-17% + engram 6.0% + norms/glue/remainder ~7.5%.
**The indexer DOES have a fresh number**: 3.6% @16K (`phase22 README:124`), consistent with 4.0% @220K
(`tmp/prefill-round1-20260902/findings/item6-indexer-and-gates.md:58`) → well under the 15% flag,
**not** a deep-side wall lever (consumer waste already eliminated, `PH:11369`).
Unpriced/caveats: (a) the phase22 table is **fenced** (per-span eval → extra host syncs, all_sum
inflated; `phase22 README:131-135`); sync-span shares are attribution, not real per-op cost
(`PH:583-587`); (b) sdpa 21% (Loop-2 memo) vs 15.6% (phase22) — reconcile before quoting; (c) the
diffuse residue (residuals, HC expand, norms) and the exact unfenced collective share (real transport
2.9-5.3%, `PH:445,555-583`) remain unpriced. Net: closes to 100% under the fenced convention; the
honest unfenced absolute is 68.5 s/16K (`phase22 README:135`).
