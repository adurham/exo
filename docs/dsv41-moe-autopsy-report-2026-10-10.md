# DSv4.1 MoE expert-GEMM offline autopsy — Loop-3 prescribed audit

Author: Phase-20 PM (delegation). Date: 2026-10-10 (CDT). Worktree `/private/tmp/p20-moe-autopsy`,
branch `p20/dsv41-moe-autopsy` off `origin/main` `94daf8c18`. Offline single-node bench on `studio2`
only. **0 boots, 0 serving requests, production untouched** (verified: node site-packages and
src-tree `gemv_metal.py` both stayed at md5 `649d2d6a0a68c8d6eab2a8a5394b7ab9` before and after;
instrumented mlx_lm lived only in a scratch dir `/tmp/p20moe_pkg/`).

Prescription: `docs/PERFORMANCE_HISTORY.md:11394-11406` (Loop-3 consult, Fable). Gate pre-registered
at `docs/PERFORMANCE_HISTORY.md:11408-11410` BEFORE the bench ran.

---

## GATE VERDICT: **GO** — the MoE kernel headroom is REAL and unharvested (not a shape-roofline artifact)

Pre-written rule (verbatim): *aggregate ≥85% of mix-roofline → KILL the Metal project, pivot;
segments ≥15% below shape-roofline (esp. top-decile-time) → GO.*

| measured aggregate | value | vs the **85% KILL bar** |
|---|---|---|
| real production M-mix (E=384, chunk 2048×top6, real routing trace) | **6.63 TF = 44.3%** of mix-roofline 14.96 | **40.7 pts short → not KILL** |
| uniform E=192 (control) | 8.85 TF = **58.5%** of 15.15 | **26.5 pts short → not KILL** |
| skew Zipf-0.9 (control) | 7.55 TF = **49.9%** of 15.10 | **35.1 pts short → not KILL** |

The **15% GO bar is cleared by ~3×**: every segment class sits **40.8–55.7% below** its shape-roofline.
The highest-time class (the 70 full M=64 tiles = 4480 of 12288 rows, and the uniform control they
equal) is at **58.5% → 41.5% below** shape-roofline. Nothing anywhere approaches a roofline.
Achieved effective bandwidth is **7.3–10.7% of the 431–438 GB/s peak** → **not bandwidth-bound**.

**This falsifies Loop-3 hypothesis #1.** The 8.88 TF coincidence was checked *for bf16 weights at
M=16*; the experts are **3-bit EXL3** (`k=3` on layer 0). For b-bit weights FLOPs/byte = 16·M/b, so
the bandwidth-roofline binds only below the crossover **M* ≈ 6.6** (b=3, 431 GB/s; ≈5.2 at 550 GB/s).
Production per-expert M ≈ 31–64 is 5–10× above M* → **firmly compute-bound** → the shape-roofline *is*
the ~15.1 TF dense ceiling → the 44–58% is genuine headroom, not physics. The Metal kernel project is
**GO**.

---

## Step-by-step evidence

### Step 1 — segmentation of a production chunk (n=2048). `file:line`
- Instrumentation (one log line, env-gated, default OFF → production path byte-identical):
  `mlx-lm/mlx_lm/models/exl3/gemv_metal.py:2011` (`if _MM_SEG_LOG: _mm_seg_log_capture(...)`,
  inside `inner_mm_seg_mlx` `:1963`); helper `_mm_seg_log_capture` `:1901`; flag `:1585-1586`.
  Patch: `bench/p20_moe_instrumentation.patch`. Submodule working-tree commit `58961c8`
  (local submodule branch only — not pushed; the portable artifact is the patch).
- **Real production chunk** (phase-10 trace `p30_exl3_trace_clamp.json`, layer 0, 2048 tok × top-6 =
  12288 pairs/rank, E=384, k=3): **3 launches** (gu-gate, gu-up, dn), **each `nb_real=397`,
  `nb_max=577`, `M_sum=12288`, `M_max=64`, M_mean=31, 330 experts touched, `blocks_full/row=2.068`**.
  M-hist spans 1..64 with a long ragged tail (69 blocks at M≤8; 70 full M=64 blocks). Shapes
  gu N=1152 K=5120 (tpe=72), dn N=5120 K=1152 (tpe=320). `full _prefill = 65.59 ms = 6.63 TF`
  (seg-only 3-launch sum 57.06 ms = 7.62 TF; glue ≈8.5 ms).
- The single Metal dispatch (`grid=((out_e//64)*GEM_THREADS, nb_max, 1)`) exposes **no per-segment GPU
  cycles**; the log records the host-side per-segment M/N/K decomposition and the launch wall is
  measured host-side around the drained `mx.eval` (stated in the log line + RUN-NOTES).
- Raw: `docs/benchmarks/phase20-throughput/raw/p20-moe-autopsy/{p20_moe_autopsy_stage1.json, stage1.log}`.

### Step 2 — per-segment roofline + achieved effective BW (the TRUE ceiling for the mix)
- In-session ceilings (recalibrated, reproduces Task A within noise): dense bf16 `mx.matmul`
  **15.14 TF** (4096×8192×8192); triad BW **431–438 GB/s**.
- Per-segment shape-roofline `ceil(M)=min(15.15, BW·16·M/3/1000)`, `M*≈6.6`: only M≤6 blocks are
  BW-bound; every M≥7 bucket is compute-bound at 15.146. FLOP-weighted **mix-roofline = 14.96 TF**
  (real), 15.15 (uniform), 15.10 (skew).
- Measured vs roofline: **real 6.63 TF = 44.3%**, uniform 8.85 = 58.5%, skew 7.55 = 49.9%.
  Achieved effective BW (weight bytes ÷ wall): **46.2 GB/s real / 31.5 uniform / 44.2 skew =
  7.3–10.7% of peak** → not BW-bound.
- Raw: `.../p20_moe_autopsy_analysis.json` (script `bench/p20_moe_roofline.py`).

### Step 3 — dequant arm with conversion cost EXCLUDED (pre-converted fp16 weights)
- E_sub=32, M=64, fp16 W cache 1132 MB built ONCE outside the timed region; pre-vs-in-loop
  rel_max_err = **0.0** (bit-exact). Median, drained, interleaved:
  **segmented 7.95 TF (52.5%) · pre-converted-dense 6.93 TF (45.8%) · in-loop-dense 4.39 TF (29.0%)**.
  Ratios: **pre/inloop = 1.58×**, **pre/seg = 0.87×** (matches the prior killed in-loop result 0.47–0.55×).
- **Verdict: in-loop dequant is NOT the dominant bottleneck.** Even with conversion fully excluded the
  dense per-expert path does **not** reach ceiling (46%) and is **slower** than the current segmented
  kernel (0.87×). The fix direction "persistent hot-expert fp16 weight cache" would **NOT** beat the
  current kernel → **noted, not built**.
- Raw: `.../p20_moe_autopsy_stage3.json`.

### Step 4 — uniform vs skewed (8.88 → 7.55, −15%): recoverable
- E=192, N=12288, 2 interleaved rounds: uniform 8.86 TF (192 blocks, waste 1.00) → skew 7.55 TF
  (326 blocks, waste 1.70): **−14.7%**. **skew_balanced** (same skew weights, every expert rounded to
  a multiple of 64 → all-full blocks, 192 blocks, waste 1.00): **9.04 TF = +19.6%**, i.e. **102% of
  uniform**. The entire −15% is partial-tail-block / block-packing waste (326→192 blocks) — a block-grid
  load imbalance — and it **is recoverable**.
- Paper-only spec (NOT implemented): `.../PARTITION-SPEC-paper-only.md` — M-major/work-stealing
  segmented GEMM: grid `(out_e/64, ceil(N/64))`, every threadgroup fills a full 64-row M-tile from a
  static M-major schedule / atomic work-queue over `blk_tab`, per-8-row-fragment expert index, device-
  side x-row gather across expert boundaries, numerics bit-identical. This is the block's **candidate
  shippable** (~1.5–2% e2e, battery-gated later).
- Raw: `.../p20_moe_autopsy_stage4.json`.

### Secondary (cheap) — code-reads only (`docs/dsv41-moe-autopsy-secondary.md`)
- **Expert placement = TP-SPLIT (intermediate-width), NOT expert-locality.** `mlx-lm/models/exl3/loader.py:174-208`
  (`_slice_intermediate`) + `:286-299` (world>1 slice); all 384 experts resident on **both** ranks
  (`deepseek_v41/exl3_build.py:508-510`). Placement does **not** change the per-rank M-mix; it halves
  per-rank N (gu) and K (dn) and adds one `[tokens,5120]` bf16 `all_sum`/layer = 125.8 MB ×40 = 5.03 GB/chunk.
  The "expert-locality" angle is moot as a kernel lever (EP would add all-to-all dispatch + real imbalance).
- **Dual-microbatch drain comm-hiding = ABSENT.** `deepseek_v41/moe.py:56-63` `_all_sum_tail` is a plain
  `_coll.all_sum` (`collective.py:73-77`); issued inline once per layer and consumed immediately
  (`moe.py:180-189`); `async_depth` is accepted then ignored (`session_cache.py:145-157`). The V4 overlap
  probe (`PERFORMANCE_HISTORY.md:344-350`, same-stream 0.995) shows overlap is *possible* but not implemented.
  A genuine, still-unpriced TP lever (code-read only).
- **Ceiling reprice:** 15.14 TF = dense bf16-in/fp32-accum pure MMA, **no dequant**, large shapes.
  Crossover M ≈ 5–7 → production M (32–48) is compute-bound → 15.14 is the fair denominator; the
  achieved 44–58% is a real gap. 82-TF lesson holds: max measured anywhere **15.15 TF**, no >30 TF recurrence.
  Caveat: the fused kernel's decode ALU legitimately lowers the *achievable* ceiling below 15.14 → quote
  the gap as "below-ceiling", not "physics-impossible".
- **Wall model closes to ~100% under the fenced convention** (`docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/README.md:116-129`):
  EXL3SwitchGLU 30.7% + all_sum 16.9% + EXL3Linear 16.6% + sparse_attn 15.6% + LazyEngram 6.0% +
  RMSNorm 4.1% + **Indexer 3.6%** + rest. **The indexer DOES have a fresh number (3.6%, well under the
  15% flag)** — not a deep-side wall lever. Caveat: the table is fenced (inflates all_sum); the honest
  unfenced absolute is 68.5 s/16K.

---

## Ranked consequences

1. **GO — continue the Metal kernel project.** The ~40–56% below-ceiling gap is real, not a shape
   artifact (3-bit weights ⇒ compute-bound above M≈6.6). Fable's KILL hypothesis is falsified.
2. **Lever #1 (highest, block-attributable): block-quantization / pad waste.** Production packs 12288
   rows into **397** BM=64 blocks (`blocks_full/row=2.068`) vs the ~192 an ideal packing needs; the M-hist
   has 69 wasted near-empty blocks. Step 4 isolates the −15% as exactly this and shows it recoverable
   (skew→balanced +19.6%, 9.04 TF ≥ uniform). → Ship the **M-major/work-stealing partition patch**
   (paper spec done; ~1.5–2% e2e; battery-gated later).
3. **Lever #2: fused decode→MMA pipeline occupancy/issue** — even at waste=1.0 (uniform) the kernel is
   only 58.5% of dense. This is the "Metal project" target (kernel engineering).
4. **Killed directions (do not pursue):** in-loop dequant **and** a persistent hot-expert fp16 cache —
   the pre-converted dense path is 0.87× the current segmented kernel (slower). Dequant is not the bottleneck.
5. **Still unpriced (code-read, not benchmarked):** dual-microbatch drain comm-hiding (absent — a real TP
   lever); TP-split placement (present, M-mix unchanged — not a kernel lever); an unfenced per-op wall capture.

## What was NOT redone (all previously closed/shipped)
Task-A Loop-2 MoE microbench (reused 8.88/7.55 as reference; re-measured in-session to 8.85/7.55 to
confirm comparability), consumer-index-skip, chunk-4096, dense requant, Q1d/Q1e native-experts requant.
No quantizer, no re-encode, no boots.

## Raw artifacts (under this worktree)
`docs/benchmarks/phase20-throughput/raw/p20-moe-autopsy/{p20_moe_autopsy_stage1.json, _stage3.json,
_stage4.json, _analysis.json, stage1.log, stage34.log, RUN-NOTES.md, PARTITION-SPEC-paper-only.md}`;
scripts `bench/p20_moe_autopsy.py`, `bench/p20_moe_roofline.py`, `bench/p20_moe_instrumentation.patch`,
`bench/phase20_guard.py`; code-reads `docs/dsv41-moe-autopsy-secondary.md`.
