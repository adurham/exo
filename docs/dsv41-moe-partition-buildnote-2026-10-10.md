# dsv41 MoE expert-GEMM partition patch (lever #1): buildnote, 2026-10-10

**Bottom line.** `EXL3_MM_SEG=v20` (env-gated; the default stays `v19d`) is **bit-exact** against
v19d and v19c (**153/153 cases with 0 differing bits on studio2**, including the real checkpoint
weights and full `EXL3SwitchGLU._prefill`). On the real production chunk it recovers only
**+2.9%** of MoE GEMM throughput: 6.59 → 6.78 TF, which is 44.1% → 45.4% of the mix-roofline.
That works out to roughly **+0.9% end-to-end prefill**. The autopsy priced this lever at
~1.5–2% and projected ~8.9 TF. This patch gets nowhere near that. The reasons are below, and
they come from the hardware, not from leaving work unfinished.

## 1. Bit-exact verdict: 0-diff

Harness: `bench/p20_moe_partition_bitexact.py` (shared builders in `bench/p20_moe_partition_common.py`).
For each case it runs the same inputs through `inner_mm_seg_mlx` with the version gate set to
v19d (production), v19c (stock), and v20. It then compares the **fp16 bit patterns as uint16**,
so the comparison is stricter than `array_equal` (it also catches ±0 and NaN-payload
differences).

| set | weights | codebook | cases | bit diffs |
|---|---|---|---|---|
| real trace (layer 0, 2048×top6 = 12288 pairs, E=384), uniform E=192/N=12288, skew E=192/N=12288, and 10 adversarial mixes (listed below) | synthetic random trellis | MUL1 (cb=2, the production codebook) | 65 | 0 |
| real trace | synthetic | DEFAULT (cb=0), MCG (cb=1) | 10 | 0 |
| all 13 mixes | **real dsv41 checkpoint, layer-0 experts** (rank 0 of world 2, k=3, MUL1) | 2 | 65 | 0 |
| all 13 mixes, full `_prefill` (gate+up+act+down+finish), v20 vs v19d | real checkpoint | 2 | 13 | 0 |
| **total (studio2, mlx 0.32.3.dev20260918+603f16eb7)** | | | **153** | **0** |

- Each mix runs gu gate (`tn_base=0`), gu up (`tn_base=E·72`) and dn (K=1152, N=5120, tpe=320).
- Adversarial mixes:
  - tail-only experts with 1–7 rows
  - all experts len=1
  - all experts full (64 rows)
  - lengths straddling 8/32/64/128 boundaries (1, 7, 8, 9, …, 63, 64, 65, …, 193)
  - max-length 64 mixed with tiny experts
  - long runs (1000/517 rows) plus a sparse tail
  - two random ragged mixes
  - a single row on the last expert
  - a single row
- All reference outputs are finite and non-trivial (min |max| = 5.96).
- A host-side schedule checker confirms, for every mix: each live row is covered exactly once,
  by its own expert, at the v19d 32-row-window origin; no half exceeds 32 rows; dead schedule
  columns are zero.
- The laptop run (M4 Max, mlx 0.32.0.dev20260804) also gave **75/75 with 0 diffs** on the final
  harness (cb=2 on all 13 mixes, plus cb=0/cb=1 on the real trace).
- Also verified: the v19c/v19d/v19e kernel **sources** are byte-identical to `6cc9c1e` for
  k=2..4 and cb=0..2.
- Raw: `docs/benchmarks/phase20-throughput/raw/p20-moe-partition/p20_moe_partition_bitexact_studio2.json`
  and `bitexact_studio2.log` (plus the `_laptop` equivalents). `bitexact_studio2_attempt1_cbassert.log`
  is a first launch that aborted on a wrong harness assertion: I had assumed cb=0, but the
  checkpoint is MUL1. The harness was fixed to cover all three codebooks, with MUL1 everywhere.

### Why v20 is exact

- Packing works at **32-row sgr-half granularity**, never at 8-row-fragment granularity.
  1. Every simdgroup's A window starts at exactly the sorted row v19d uses (a v19d block start,
     or block start + 32).
  2. It multiplies B fragments decoded from its own expert's tile.
  3. It runs the **identical guarded mma body** (`_MMA_GUARDED`, with `wblk` → `wbuf`), with
     k2 in ascending order and literal row indices.
- So every live output element gets the same `simdgroup_multiply_accumulate` sequence on the
  same operands.
- Decode writes the same `dq_val` into a different threadgroup buffer. The decode expression is
  identical, so the half values are identical.

## 2. What changed (mlx-lm `mlx_lm/models/exl3/gemv_metal.py`, commit `168685d`)

- `:1579`: `_MM_SEG_VERSION` default is **still `v19d`**. v19c/v19d/v19e are untouched.
- `:1910`: `_MM_SEG_V20 = ("v20",)`, plus the design comment.
- `:1914` `_seg_pair_sched_fn(nb_max)`: an `mx.compile`'d, device-side schedule built from
  `blk_tab`/`nb_real`, with no host sync.
  - Blocks with more than 32 rows keep their own threadgroup: half 0 = rows 0–31, half 1 =
    rows 32–63, same expert.
  - Blocks with ≤32 rows are paired two per threadgroup, in expert order.
  - Output is `sched` (6, nb_max) = `[e0, row0_0, len0, e1, row0_1, len1]`, and
    `nts = n_long + ceil(n_short/2)` (the live threadgroup count).
  - **No `metal_kernel` is inside this compile boundary** (house rule 1).
- `:1957` `_seg_pair_sched`: memoized by identity on the last `(blk_tab, nb_real)`, so the
  gate/up/down launches of one `_prefill` share one schedule build.
- `:1968` `_mm_seg_source_v20`: 128 threads (unchanged), `wblk[2×32×72]` (two tiles, `:2036`).
  - `ndec = 1` when both halves are the same expert or half 1 is empty; otherwise 2 (two
    serialized tile decodes per k-stage, each done by all 128 threads).
  - Same barrier count per k-stage as v19d.
  - The store is masked to each half's own `len`.
- `:2204` selects the kernel source; `:2228` swaps in the schedule. The grid is unchanged:
  `((out_e/64)·128, nb_max, 1)`. Threadgroups past `nts` return immediately.
- Branch `p20/dsv41-moe-partition-instr` (local submodule branch, not pushed, same convention
  as the autopsy):
  - `cae541c`: the autopsy `EXL3_MM_SEG_LOG` instrumentation, re-applied from
    `bench/p20_moe_instrumentation.patch`.
  - `168685d`: v20.
- Portable patch: `bench/p20_moe_partition.patch` = `git diff 6cc9c1e 168685d`. It **applies
  cleanly to a clean `6cc9c1e`** (checked in a scratch worktree; md5 of the result is identical)
  and contains both commits (the instrumentation and v20).
- The exo tree's gitlink stays at `6cc9c1e`, i.e. **nothing is wired into production**.

## 3. Measurement (studio2, real checkpoint, same shapes as the autopsy)

Setup:
- `bench/p20_moe_partition_bench.py` reuses `bench/p20_moe_autopsy.py`: ceilings, builders,
  `time_launches`, and the TF definition `6·D·H·pairs / full _prefill wall`.
- v19d and v20 are **interleaved in one process**: 2 rounds with the order alternated, 7
  drained reps each plus drained warmups, medians.
- In-session dense ceiling: 15.14 TF; BW: 438 GB/s.
- Mix-roofline = the autopsy mix ceiling rescaled to the in-session dense ceiling. The
  M-histogram is unchanged by v20, so this ratio does not change.

| shape | autopsy baseline | v19d (in-session) | **v20** | v20 vs v19d | `_prefill` ms v19d → v20 |
|---|---|---|---|---|---|
| **real** E=384, 12288 pairs | 6.63 TF / 44.3% | 6.59 TF / 44.1% | **6.78 TF / 45.4%** | **+2.9%** | 65.99 → 64.10 |
| uniform E=192 | 8.85 / 58.5% | 8.90 / 58.8% | **9.03 / 59.6%** | +1.5% | 48.87 → 48.14 |
| skew E=192 | 7.55 / 49.9% | 7.56 / 50.1% | **7.74 / 51.3%** | +2.4% | 57.54 → 56.21 |

- Per-launch on the real chunk: gate 19.24 → 18.71, up 19.30 → 18.68, dn 18.58 → 18.04 ms.
  The 3-launch segment sum is 57.31 → 55.43 ms (−3.3%).
- Repeatability: the two rounds agree within 0.15 ms on every row.
- In-session v19d reproduces the autopsy within 0.6%, so the comparison is valid.
- Guards: idle rc=0 and canary healthy (14.85–14.87 t/s on both nodes) before and after both
  the bit-exact and the timed region; `state_active_tasks=0` throughout.
- The node's site-packages and src-tree `gemv_metal.py` stayed at `649d2d6a…` before and after.
  Only `/tmp/p20moe_partition{,_pkg}` were written.
- Raw: `p20_moe_partition_bench.json`, `bench_studio2.log`, `guard_*.json`, `node_md5_after.txt`.

## 4. Honest e2e estimate

- MoE (EXL3SwitchGLU) is ~30.7% of the fenced chunk wall (~1/3).
- −1.89 ms on a 65.99 ms MoE layer is −2.9% of MoE, so **≈0.85–0.95% e2e prefill**.
  This is an upper bound, because the microbench excludes all_sum/attention overlap effects.
- **Below the autopsy's 1.5–2% pricing.** Not worth a deploy on its own. It is a
  numerics-neutral, env-gated building block.

## 5. Reality vs paper: why the +19.6% control did not transfer

1. **The decode floor is per expert, not N/64.**
   - The skew_balanced control removed tail blocks *and* made every block single-expert full.
   - The real chunk has 330 live experts with a mean of 37 rows. The minimum number of
     weight-tile decodes per column tile is ≥ the number of (expert, 64-row) segments
     (343 at 128-row sharing, 397 at 64).
   - v20 cuts **threadgroups** 397 → 280 (each live sgr pair now has work). It does **not** cut
     **decodes**: a paired threadgroup decodes two tiles serially.
   - The saving is therefore only the idle-half/barrier/launch share. The recovery is bounded by
     the decode floor, exactly as warned.
2. **Fragment work dominates, not block overhead.** Synthetic-weight ablation
   (`bench/p20_moe_partition_ablate.py`, laptop, real mix, gu):
   - Removing the MMA collapses time to 0.5 ms out of 23 ms.
   - Removing the trellis loads saves ~23%.
   - The real mix issues 1687 live 8-row fragments per column tile, vs 1536 for uniform:
     +10% unavoidable ragged-tail work (rows rounded up to 8).
   - A 2-parameter fit across the three autopsy shapes gives ~0.022 ms per threadgroup and
     ~0.006 ms per fragment. Removing 117 threadgroups predicts ≈ −2.6 ms per launch; the
     measurement is −0.55 ms. Even the fixed per-threadgroup term mostly overlaps with other
     threadgroups' MMA when the GPU is full.
3. **No device-side x gather was needed** (confirmed). `xh_sorted` is expert-sorted, and every
   half's rows are a contiguous run of its own expert. v20 addresses x device-direct at the
   original sorted row. The paper spec's cross-expert x-row gather is unnecessary.
4. **8-row alignment is not needed at 32-row-half packing granularity.**
   - The fragment-straddle problem only arises if packing works at 8-row granularity across
     experts.
   - v20 never puts two experts in one sgr half. Fragments past `len` are dead and skipped by
     the existing `rlive` guard, which is exact because the store masks them.
5. **Variants tried and rejected** (all bit-exact by construction; rejected on timing):
   - **v20b**: shared-B per-k2 reuse in partial halves. Laptop real gu 6.49 → 6.39 TF, slower
     than v20. Removed.
   - **v21**: 256-thread / 128-row threadgroups pairing an expert's own consecutive blocks.
     Cuts decodes 397 → 343, but **regressed hard**: real gu 6.27 → 4.36 TF, uniform
     8.91 → 6.75. That points to occupancy/register pressure at 256 threads × 16
     accumulators/simdgroup. Source archived in
     `raw/p20-moe-partition/v21_256thread_NEGATIVE_source.py.txt`.
   - The fully general per-8-row-fragment, cross-expert, 8-slot packing (task design #4) needs
     up to 8 distinct expert tiles per threadgroup: 8 × 4.6 KB wblk, or 8 serialized decodes
     per k-stage. That multiplies the decode-and-barrier cost that items 1–2 show already
     dominates. **Not built**: on these numbers it cannot beat v20.

## 6. Tests

- `pytest src/exo/worker/engines/mlx/dsv41/tests` (worktree, mlx-lm first on PYTHONPATH):
  **178 passed** with the patched submodule (identical to the 178 baseline), and **178 passed**
  again with `EXL3_MM_SEG=v20` exported.
- The suite does not exercise the Metal seg kernel. The bit-exact harness above is the kernel
  proof.

## 7. Open questions / not done

- Not deployed, not wired as a default, no studio1 run, no TP2 pipelined e2e prefill
  measurement. House rule 2 means a real pipelined chunk run is required before any default
  flip. The +2.9% is an offline single-node `_prefill` number.
- Lever #2 (decode → MMA occupancy, 58.5% even at zero waste) is untouched, as instructed.
  The ablations suggest it, plus ragged-tail fragment waste, is where the remaining 55% gap
  lives.
- The schedule memo holds one reference to the last `blk_tab`. That is negligible memory, but
  the identity memo assumes the eager (non-compiled) `_prefill`, which is how dsv41 calls it
  today.
