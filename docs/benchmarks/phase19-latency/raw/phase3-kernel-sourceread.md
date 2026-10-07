# Phase-3 kernel go/no-go — SOURCE-READ of the EXL3 dense small-batch path

Worktree: `/private/tmp/levers-wt` (branch `deploy/next15-levers` @ `71a8c94c2`).
READ-ONLY. No benches, no cluster, no git commit.

Source of record: the `mlx-lm` submodule, pinned by the gitlink identical in
both `deploy/next13` and `deploy/next15-levers` — `mlx-lm @ 6cc9c1e8709e228fca99ac152cd5e681ddcce65d`,
clean (0 dirty files). All citations below are to that checkout:
`/Users/adam.durham/repos/exo/mlx-lm/mlx_lm/models/...` (the worktree's own
`mlx-lm/` is an **unpopulated** submodule dir, so the pinned submodule is the
correct source; the content is identical to what the worktree pins).

Question: is the EXL3 dense small-batch (M=1..8) GEMM at the memory-bandwidth
roofline (EXHAUSTED), or is there a concrete, named inefficiency a code change
could remove (GO)? And: does an M-tile padding boundary explain the R4->R5
verify cost anomaly (`VERIFY_MS` R4->R5 = +14.0 ms vs R3->R4 +9.8, R5->R6 +8.4;
`deepseek_v41/spec.py:77`)?

---

## 1. DISPATCH — `exl3_linear.py::EXL3Linear.__call__`

Full routing. `rows` = product of all dims except last (`exl3_linear.py:118-121`);
`x2d` is fp16 `(rows, in_features)`. Constants:

- `HUGE_WEIGHT_BYTES = 64*1024*1024` (`exl3_linear.py:50`)
- `FUSED_GEMM_ROW_LIMIT = int(os.environ.get("EXL3_FUSED_ROW_LIMIT", "64"))` (`exl3_linear.py:67`)
- `DECODE_FULL_MAX_BYTES = 1536*1024*1024` (`exl3_linear.py:70`)
- `_WCACHE = os.environ.get("EXL3_WCACHE", "0") == "1"` (`exl3_linear.py:48`) — **default 0**
- `_FUSE_HAD = os.environ.get("EXL3_FUSE_HAD", "0") == "1"` (`exl3_linear.py:40`) — default 0
- `self._huge = in_features*out_features*2 > HUGE_WEIGHT_BYTES` (`exl3_linear.py:83`)
  → the DSv4.1 dense *attn* linears are **not huge** (largest fp16 `wo_b`
  5120×8192×2 = 84 MB > 64 MB is the only one over; wq_a/wq_b/wkv/shared are
  under 64 MB; see §4). `_huge` only affects the `_WCACHE` and `release_source`
  branches, not the small-batch band.

Ordered routing in `__call__`:

| # | condition (file:line) | path |
|---|---|---|
| 1 | `rows == 1` (`exl3_linear.py:123`) | fused Metal **GEMV**. If `_FUSE_HAD` and `rt.suh` → `inner_gemv_had_mlx` (L129-135); else `xh = prepare_xh` then `inner_gemv_post_mlx` when `rt.svh` (v18 post-Hadamard epilogue, L137-138), else `inner_gemv_mlx` (v12 simd GEMV, L142). |
| 2 | `elif rows <= 16` (`exl3_linear.py:146`) | **This is the small-batch verify band.** `xh = rt.prepare_xh(x2d); y = rt.finish_y(inner_gemm_mlx(xh, ...))` (L152-153). |
| 3 | `elif _WCACHE and not self._huge` (`exl3_linear.py:154`) | cached fp16 `W = inner_weight_mlx(...)` + `prefill_matmul_mlx` (compiled pre-Had + matmul + post-Had). **Dead by default (`EXL3_WCACHE=0`).** |
| 4 | `elif rows <= FUSED_GEMM_ROW_LIMIT` (`exl3_linear.py:157`) | same fused trellis GEMM as branch 2 (`inner_gemm_mlx`). With the default limit 64 this covers rows 17..64 identically to branch 2; branch 2 makes it reachable *first* only when `FUSED_GEMM_ROW_LIMIT < 16`. |
| 5 | `elif in*out*2 <= DECODE_FULL_MAX_BYTES` (`exl3_linear.py:162`) | transient full decode `decode_full_mlx` + native matmul. |
| 6 | `else` (`exl3_linear.py:167`) | striped decode per `DEFAULT_STRIPE_COLS` (512) + matmul (lm_head-scale). |

**The docstring (L9-14) is misleading on two points.** (a) "huge layer, M ≤ 144
→ fused Metal GEMM (v5 grid-parallel batch)" — the code limit is
`FUSED_GEMM_ROW_LIMIT` **default 64** (`exl3_linear.py:67`), not 144; 144 is
stale. (b) "layer fits in memory → decode-once cached `W` + compiled matmul"
sounds like the default, but that branch is gated on `_WCACHE` (**default 0**,
L154) and is the *4th* check, not the first.

**Answers to the explicit checks:**

- **Which branch handles M=1:** branch 1, the fused Metal GEMV (`exl3_linear.py:123`).
- **Which handles 2 ≤ M ≤ 64:** branch 2 for `2 ≤ M ≤ 16`
  (`exl3_linear.py:146`, `inner_gemm_mlx`), and branch 4 for `17 ≤ M ≤ 64`
  (`exl3_linear.py:157`) — same kernel. **Note the limit that matters is 16,
  not 64**: `gemv_metal.py:1421` gates the barrier-free simd GEMM to
  `batch <= 8 (or devx ≤ 16)`, and `_M_TILE = 8` (`gemv_metal.py:1044`) tiles the
  fallback. `FUSED_GEMM_ROW_LIMIT=64` is a *safety* setting (see the comment at
  `exl3_linear.py:60-66`: reverting to 16 caused GPU timeouts at full depth),
  not a performance boundary.
- **Which handles M > 64:** branch 5 (transient full decode + matmul) up to the
  1.5 GB weight cap, else branch 6 (striped). Reached only by lm_head on
  full-sequence forwards (`exl3_linear.py:168-171`); the generate path never
  gets here.

---

## 2. DECODE REUSE / DEQUANT CACHING for the small-batch verify path

**There is NO persistent fp16 `W` cache in the default configuration. The M=1..8
path re-walks the trellis every call.** The trellis itself is uploaded to device
once per layer (`EXL3LayerRuntime.trellis`, `layer_state.py:125-171`), but the
*fp16 weights* are never materialized and nothing is memoized across calls.

Evidence:

- `_WCACHE = ... == "0"` default (`exl3_linear.py:48`). The comment
  (`exl3_linear.py:41-47`) is explicit: *"the trellis-direct prefill ladder …
  matches or beats the cached matmul, while the cache costs 2.56 GB resident …
  **Decode (rows=1) and verify (rows<=8) never read it.**"* So the cache is off
  *and* would be irrelevant to rows≤8 even if on.
- Branch 2 (`exl3_linear.py:146-153`) calls only `prepare_xh` (a compiled
  Hadamard, `layer_state.py:137-140`) and `inner_gemm_mlx(xh, rt.trellis, ...)`.
  It never touches `inner_weight_mlx`. The fp16 `W` (the
  `_inner_cache`, `layer_state.py:149,182-196`) is only read by branch 3
  (`_WCACHE`, `exl3_linear.py:155`) and by `prefill_matmul_mlx`, both off by
  default.
- `rt.trellis` in `EXL3LayerRuntime` *is* persistent (device-resident uint16,
  `layer_state.py:130,163`), re-broadcast to uint32 words in `_run_inner_gem`
  (`gemv_metal.py:1406`) — a zero-copy `.view`, no re-upload. So the *trellis
  bytes* are read from device memory once per call, which is the intended
  roofline traffic.

**Is the per-call decode cost proportional to M, or amortized over M?**
**Amortized once per MT rows, not per row.** The design re-decodes each trellis
tile **once per `mt`-row group** and applies it to `mt` rows:

- `mt = 1 if batch == 1 else _M_TILE (=8)` (`gemv_metal.py:1410`), and for the
  simd path `simd_mt`/`devx_groups` (`gemv_metal.py:1422-1423`):
  `simd_mt = {1:1, 2:2, ≤4:4, else 8}`. Both are "decode a tile, then fan it out
  to `MT` rows".
- v16 kernel docstring: *"x is staged transposed … decode stays the dominant
  cost, not the per-row fma."* (`gemv_metal.py:496-498`); the decode produces
  `dq_val` which is then FMA'd against `MT` rows' x values (`gemv_metal.py:558-562`).
- v20 kernel docstring: *"one contiguous load per weight serves all MT rows"*
  (`gemv_metal.py:1226-1227`); loop `for tk` decodes the 4 codewords once, then
  `slot_body` FMAs them into `nq = MT/vec` accumulators (`gemv_metal.py:1311-1320,1242-1258`).
- The trellis read is bounded by the in-tile loop, `for (uint tk …)`, once per
  `t = ...` tile — **its byte count is independent of M within a group**
  (`gemv_metal.py:616-635` v16, `1311-1320` v20).

**Crucial nuance for the R5 question (see §3):** `simd_mt` **steps 4 -> 8 at
batch = 5** (`gemv_metal.py:1422`). So the decode group changes size exactly at
M=5. But because the trellis bytes re-read **increases** when the group grows
(see §3/§6), M=5 is *more*, not less, expensive per group — the step cannot
create a *saving*; at most it modulates the marginal.

**Second small inefficiency worth naming:** the v20 path transposes + **pads**
`x` to `MT*groups` (`gemv_metal.py:1452-1459`), so for M in the 5..7 range it
carries 1..3 padded (zero) rows of `x` register work. That is **x-side work
only**, negligible next to the ~4 bpw trellis decode, and it does **not** change
the trellis byte count (the decode is per tile, not per row).

---

## 3. TILE HEIGHT / PADDING — THE KEY QUESTION

### 3a. The M-tile constant

`_M_TILE = 8` (`gemv_metal.py:1044`). This is the staged fallback's tile. The
fused (simd/v20) path uses a **step function** `simd_mt` instead
(`gemv_metal.py:1422`):

```python
mt        = 1 if batch == 1 else _M_TILE                                  # L1410
simd_mt   = 1 if batch == 1 else (2 if batch == 2 else (4 if batch <= 4 else 8))   # L1422
devx_groups = (batch + simd_mt - 1) // simd_mt if devx_ok else 1          # L1423
```

`simd_mt` is `{1:1, 2:2, 3:4, 4:4, **5:8**, 6:8, 7:8, 8:8}`.

### 3b. Is there an M_TILE such that M=5 pads up to 8 rows of work?

**No work padding. The kernel explicitly guards every per-row operation with
`mm < batch` (or `mm_base + mm < batch`), so padded rows produce no FMA and no
output store.** Citations:

- v16 reduction, `gemv_metal.py:641-656`: loop `for (uint mm = 0; mm < MT; mm++)`
  but the store is guarded — `if (tid < 16u && mm < batch) { … out[…] }`
  (**L648**). The register accumulator for `mm >= batch` is never read out.
- v16 staging, `gemv_metal.py:624`: `if (tk0 + t < tk_end && mm < batch) v = …`
  — padded `x` rows read as `0.0f`, not real work.
- v20 reduction, `gemv_metal.py:1323-1337`: same `for (mm < MT)` loop, store
  guarded by `if (tid < 16u && mm_base + mm < batch)` (**L1330**).
- v20 x padding at `gemv_metal.py:1457-1458` (`mx.pad(xt, …, (0, mt_total-batch))`)
  pads the *buffer*, and the kernel's guard skips those rows.

So the **FMA/store work scales with M (M=5 does 5 rows of work, not 8)**. There
is no "M=5 costs like M=8" on the compute/output side.

### 3c. Does anything step at M=5 that could cause a *cost* jump?

**Yes, one thing: the decode-amortization group size steps 4 -> 8 at M=5**
(`simd_mt`, `gemv_metal.py:1422`). Its effect: the **trellis is re-read once per
group**, so

- M ≤ 4: `devx_groups = 1`, trellis read **once** → 1x trellis bytes.
- M = 5: `devx_groups = ceil(5/8) = 1` → still **1x** trellis bytes.
  (M=5 does **not** re-read.)

So the step at M=5 does **not** add a trellis re-read either. For **M ≤ 8 the
trellis is read exactly once** (`devx_groups=1`, `gemv_metal.py:1423`). The
first trellis *re-read* (2x traffic) happens at **M = 9** (`devx_groups=2`), not
at M=5. This is stated in the code comment itself:
*"rows 9-16 take a second row group along grid.y … each group re-decodes
the trellis"* (`gemv_metal.py:1417-1419, 1449-1451`).

### 3d. Definitive answer to the R4->R5 question

**No padding boundary explains R4->R5.** From the source:

1. Fused FMA/store work is **M-proportional and guard-truncated** — no M-tile
   padding of work (`gemv_metal.py:648`, `1330`).
2. Trellis re-read count is **flat (1x) for all M ≤ 8** and doubles only at
   M ≥ 9 (`gemv_metal.py:1423`); nothing steps at M=5.
3. There is **no second kernel launch** at M=5: the dispatch is one
   `kernel(...)` call for both the simd/v20 path (`gemv_metal.py:1469-1483`) and
   the staged path (`gemv_metal.py:1490-1504`). Whether the v20 devx path or the
   staged fallback is taken is decided by `_USE_DEVX`/`_USE_SIMD_GEMV`/`k != 7`
   (`gemv_metal.py:1420-1421, 1343, 1050`) — a **static** property of the layer's
   `k`, not of M. All dense linears in the verify band share one kernel choice;
   **nothing in the dispatch key depends on `batch == 5` in a way that switches
   kernels.** (`simd_mt` changes the *compiled kernel specialization* at M=5,
   `gemv_metal.py:1460, 1468`, but both specializations do the same work per row
   and the same once-per-group trellis read.)
4. The bandwidth step, if any, is at **M=9 (2 groups) and M=17 (3 groups)**, per
   the "per-8-rows amortization limit" comment (`gemv_metal.py:1417-1419`).

**Conclusion:** the R4->R5 +14.0 ms is **not** attributable to source-visible
M-tile padding in the dense GEMM. Source shows **no non-monotonic mechanism at
M=5**. (Candidate non-dense explanations the source does not adjudicate: MoE
expert-uniqueness growth, the chunk-verify attention/cache path, or
measurement noise — see §6. The MoE side is `exl3_moe.py`, out of this probe's
scope but noted.) The `VERIFY_MS` table is a *measured* dict
(`spec.py:77`) fed to `GammaPolicy` (`spec.py:91`), i.e. empirical, not derived
from a tile formula.

---

## 4. IS THE SMALL-M PATH AT THE ROOFLINE? (analytic)

### Shapes

The real checkpoint geometry. Model card
`resources/inference_model_cards/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw.toml:15-23`:
`n_layers 40, hidden 5120, MoE 384 routed top-6 + 1 shared, head_dim 512,
vocab 129280, engine="dsv41"`, `storage_size` 210,713,013,432 B (with `mtp.*`);
the card explicitly says the tensor-parallelism is built into
`exl3_build.build_model` and the real per-tensor shapes are read from the
checkpoint. `deepseek_v41/config.py:32-49`: `vocab 129280, dim 5120, n_layers 40,
moe_inter_dim 2304, n_heads 64, head_dim 512, q_lora_rank 1280, o_lora_rank 1024,
o_groups 8, n_routed_experts 384, n_shared_experts 1, n_activated_experts 6`.
Dense attn/moe linears from `deepseek_v41/attention.py:90-97` and
`moe.py:131-133`. Because the checkpoint is **not present on this (down)
cluster**, these are the config-derived shapes (§4 note); the per-tensor
`(out, in)` are derivable exactly as: `wq_a 5120->1280`, `wq_b 1280->16384`
(=64·512/… at TP=2, per-rank), `wkv 5120->512`, `wo_a` grouped 8×(4096·? ->1024)
with in = `n_heads*head_dim/n_groups = 64·512/8 = 4096` per group,
`wo_b 8192->5120`, shared expert `w1/w3 5120->2304`, `w2 2304->5120`.
`DSV41_TP_SHARED`/`_SHARD_ATTN` shard attn + shared along 128-blocks at TP=2
(`exl3_build.py:354,515,529,535-545`), so **per-rank shapes are half of the
above along the sharded axis**.

### Roofline arithmetic

Bandwidth 450 GB/s (the campaign's measured M4 Max streaming figure); measured
compute peak 15 TFLOPS fp16 (`bench/dense_qmm_ceiling.py:46` prints
"M=8192: 13.6, Peak: 15.0"). For one layer's dense projections (per rank,
~141 M params — the sum of the sharded attn + shared-expert linears; grouped
`wo_a` counted as 8×(4096×1024)):

| linear (per-rank) | params | MB @2.9bpw | MB @4bpw | mem-roofline µs @450 | flops @M=4 | cmp-roofline µs @15T |
|---|---:|---:|---:|---:|---:|---:|
| wq_a 5120×1280 | 6.55 M | 2.4 | 3.3 | 5.5 | 52.4 M | 3.5 |
| wq_b 1280×16384 | 20.97 M | 7.6 | 10.5 | 17.5 | 167.8 M | 11.2 |
| wkv 5120×512 | 2.62 M | 0.9 | 1.3 | 2.2 | 21.0 M | 1.4 |
| wo_b 8192×5120 | 41.94 M | 15.2 | 21.0 | 35.0 | 335.5 M | 22.4 |
| wo_a (8 groups) | 33.55 M | 12.2 | 16.8 | 28.0 | 268.4 M | 17.9 |
| shared w1/w3 | 23.59 M | 8.6 | 11.8 | 19.7 | 188.7 M | 12.6 |
| shared w2 | 11.80 M | 4.3 | 5.9 | 9.8 | 94.4 M | 6.3 |
| **TOTAL/layer** | **141.0 M** | **50.9** | **70.3** | **118 µs** | — | — |

**×40 layers:**

- Trellis @2.9 bpw = **2.04 GB**; read-once roofline = **4.53 ms**, **flat
  across M**.
- Trellis @4 bpw (the code comments' internal 16 B / 4 bits-per-weight figure)
  = **2.81 GB**; read-once roofline = **6.25 ms**, flat across M.
- Compute roofline: **3.01 ms @M=4, 3.76 ms @M=5, 6.02 ms @M=8**.

**Arithmetic intensity** (FLOP per trellis-byte, @2.9 bpw):
`AI(M) = 2·M / (2.9/8) = 5.52·M`. Ridge point = 15e12/450e9 = **33.3 FLOP/B**.
So M=1..6 are **memory-bound** (M=6 → AI 33.1 ≈ ridge); M≥7 tip **compute-bound**
(M=8 → AI 44.2), and compute overtakes memory at M≈7 (6.25 ms flat memory vs 6.02
ms compute @M=8).

**Independent in-source corroboration of the roofline:** the kernel comments
quote measured decode throughput `201` Gw/s (v16 mt=8) → at 3 bits/weight that is
**75 GB/s effective → 17 % of 450 GB/s**; v20 mt=8 is `318` Gw/s → **119 GB/s →
26 %**. The v20 docstring says v16 is *"threadgroup-BANDWIDTH bound at mt=8"*
(`gemv_metal.py:1229-1232`) and that v20 reaching `309-318 Gw/s` was a **+58 %**
win, i.e. the old mt=8 rate was `201 Gw/s`. The source "slot model" ceiling is
`308 Gw/s` (`gemv_metal.py:1230`) — an **ALU/issue** bound for the decode, not
the 1200 Gw/s that 450 GB/s would allow at 3 bpw.

**Verdict on §4:** at M=4 the path *should* be memory-bound (AI 22 < ridge 33),
but the *achieved* rate (17-26 % of BW at mt=8) means it is currently
**issue/ALU-bound above the memory floor by ~3-4x**. The fused path is **NOT at
the memory-bandwidth roofline at M≈4-8**; it is decode-ALU bound. That is a real
headroom, and it is the thing a kernel change targets.

**How many dense projections are actually EXL3Linear?** All of
`wq_a, wq_b, wkv, wo_a (grouped), wo_b, compressor.wkv/wgate, indexer.wq_b, and
the shared expert w1/w2/w3` load as `EXL3Linear` under the default
`DSV41_DENSE="exl3"` (`exl3_build.py:461,468,536-547`). The **routed** experts are
`Exl3Experts`/`EXL3SwitchGLU` (a separate fused MoE path, `exl3_moe.py`), not this
dense path — so the lever below is attn + shared expert only, ~**141 M
params/layer/rank**, not the routed MoE.

---

## 5. EXISTING SMALL-M BENCHES

| bench | file | M-range | real EXL3 tensors? |
|---|---|---|---|
| `p51_dense.py` / `p52_dense_each.py` (phase 15) | `docs/benchmarks/phase15-dsv41-body-speedup-2026-09-28/scripts/` | **M=1 only** | **YES** — loads the real checkpoint via `Exl3Checkpoint` + `load_dense_layer` + `EXL3Linear` (`p51:14-15,18-26`); reconstructs fp16 `W` for comparison; reports `eff GB/s` per group (`p52:56-67`) |
| phase 2 EXL3 MoE bench | `docs/benchmarks/phase2-exl3-moe-kernels-2026-09-27/raw/scripts_p2_exl3_moe_bench.py` | R=1, 4, 512, 2048 (`:381`) | YES (real trellis, `EXL3SwitchGLU`) but **MoE**, not the dense small-batch GEMM |
| phase 15/17 `p48_bench.py` | `docs/benchmarks/{phase15,phase17}-.../scripts/p48_bench.py` | R=1..6 verify-forward via the **whole 8-layer harness** (`EXL3Linear.__call__` wrapped, `:157`) | YES but **end-to-end forward timing**, not an isolated dense-GEMM sweep |
| `bench/dense_qmm_ceiling.py` | bench/ | M=48..8192 | **NO** — `mx.quantize(..., bits=8)` affine, not EXL3 |
| `bench/p14d_smallm_ceiling.py` | bench/ | M=32..16384 | **NO** — synthesizes `mxfp4` |
| `bench/moe_vs_dense_qmm_isolation.py` | bench/ | MoE-scale M | **NO** — synth `mxfp4` |
| `bench/moe_isolation_and_qmm_gate.py`, `qmm_m192_gate.py`, `qmm_bandwidth_bench.py`, `qmm_mbatch_bench.py`, `qmm_tile_sweep.py` | bench/ | M=48..1536 / MoE | **NO** — all synth `mxfp4`/affine-8 |
| `bench/attn_production_class_bench.py` | bench/ | attention shapes | **NO** — affine `to_quantized` |
| `bench/lever1_smallm_headroom.py` | bench/ | dense qmm at M=1 ceiling | **NO** — synth `mxfp4` |
| EXL3 unit/parity tests | `mlx-lm/tests/parity/dsv41_parity.py`, `test_dsv41_*`, `mlx-lm/mlx_lm/models/exl3/ref/*` | correctness, not M-sweep | YES (real) but correctness only |

**Conclusion:** **there is NO existing script that sweeps the EXL3 dense
small-batch GEMM at M = 2..8 with real tensors.** The only real-EXL3 dense
timing is M=1 (`p51`/`p52`). The later microbench must be new — but it can
**reuse the p51/p52 loader scaffold** (`Exl3Checkpoint` + `load_dense_layer` +
`EXL3Linear`, plus the per-group dispatch-count / GB/s printout) and simply
sweep M ∈ {1,2,3,4,5,6,7,8} against the *same* linears, instead of the synth
`mxfp4` benches. `docs/benchmarks/phase2-.../scripts_p2_exl3_moe_bench.py` also
shows the model path (`~/.exo/models/dealignai--…-EXL3-2.9bpw`).

---

## 6. VERDICT (three-way)

### (b) POSITIVE / GO — with numbers

The small-M path is **not** at the memory-bandwidth roofline, and there are
concrete, named inefficiencies:

1. **Decode-ALU efficiency, not bandwidth, is the binding constraint.** At M=4
   the memory floor is ~4.5-6.3 ms/40L yet the achieved rate is 17-26 % of BW
   (`gemv_metal.py:1229-1232`, 201/318 Gw/s). The ALU "slot model" is 308 Gw/s
   vs the 1200 Gw/s a 450 GB/s read would allow. Removing that 3-4x gap is a
   *kernel* change (the v16→v20 work already bought +58 %).
2. **The mt-tax at mt=8.** The source is explicit that the **M≤8 band runs at
   `simd_mt=8` for M ∈ {5,6,7,8}** (`gemv_metal.py:1422`), and that mt=8 is the
   *worst* amortization point: `v20 mt=8 = 318 Gw/s vs mt=4 = 486 Gw/s` — i.e.
   **~53 % faster at mt=4** (`gemv_metal.py:1232-1233` verbatim: *"mt=4 456-486
   (+33 % … the mt-tax is gone)"*, vs mt=8 309-318). M=5,6,7 are **forced into
   mt=8 by `simd_mt`**, so a code change that let M=5..7 use mt=4 (two groups of
   4) *might* recover part of the mt-tax — but see the caveat below.
3. **Amortization re-read at M≥9**: `devx_groups=2` re-reads the trellis
   (`gemv_metal.py:1417-1419`). Only marginally relevant to the γ=3 (M=4) case.

**Estimated per-round saving at R4 (γ=3, all dense projections).** The dense
EXL3 projections cost **3.6 ms of the R1→R4 9.3 ms increase** at the 8-layer
harness scale — the phase-17 measurement, *"all dense EXL3 projections: R=1
6.75 → R=4 12.49 ms"* (`docs/benchmarks/phase17-dsv41-verify-cost-2026-09-28/README.md:20,28-30`).
Scaling the harness's dense cost by headroom: if the small-M dense GEMM were run
at the memory roofline it would shrink toward the ~0.2-0.4 ms/row-of-4 floor
(§4: ≈4.5-6.3 ms/40L for the *whole* model ≈ 0.9-1.3 ms per 8-layer harness at
R1); closing the mt-tax alone (up to +53 % on the affected linears) is worth
roughly **~0.5-1.5 ms/round at R4 across all dense projections**, and zeroing the
whole dense marginal (3.6 ms at harness scale → smaller at full 40-layer scale)
puts the optimistic bound at **~2 ms/round**. The campaign gate of **≥2 ms/round**
is at the **upper edge** of what a dense-GEMM kernel change can plausibly buy —
**plausible GO, but not a slam dunk**; the deciding quantity is the mt=4-vs-mt=8
gap *at M=4..8 on the real dense shapes*, which the source cannot settle
(because it depends on the real out-tile counts, i.e. the checkpoint's actual
shapes, which are on the down cluster).

**Caveat that keeps this honest.** For **M=4** the code already uses
`simd_mt=4` (`gemv_metal.py:1422`), i.e. the *good* mt=4 case; so the γ=3 (R4)
round is **not** obviously mt-taxed, and the mt=8 penalty hits M=5..8 — exactly
the band where the R4→R5 anomaly sits. If the anomaly is real, M=5 already runs
at mt=8, and mt=8 is the *slow* specialization only in the sense of per-weight
amortization; but §3 shows M=5 does **not** re-read the trellis, so this does not
by itself produce the +14 ms. A code change that splits M=5..7 into two mt=4
groups (halving the per-weight group but *doubling* the trellis re-reads) could
easily be a wash — that is precisely the trade the microbench must settle.

### (a) ROOFLINE / EXHAUSTED — rejected

Rejected: the achieved rate is 17-26 % of BW and the binding ceiling is 308 Gw/s
ALU, ~4x below the 1200 Gw/s a 450 GB/s read permits (§4). Also, the dequant is
**already amortized** (§2) and the padding is **already guarded** (§3) — the two
cheap wins are already taken; what remains is real kernel work.

### (c) INCONCLUSIVE — partially, and this is what the microbench must pin

The source cannot decide the *magnitude* because it cannot see the real per-tensor
`out_tiles` (governs `n_splits` and the mt-tax on the actual shapes) — those live
in the checkpoint on the down cluster. The microbench must measure, **with real
EXL3 tensors** from `~/.exo/models/dealignai--…-EXL3-2.9bpw` (reuse the p51/p52
scaffold):

1. Per-linear µs **and achieved GB/s** for M = 1..8, on the real dense linears
   (`wq_a, wq_b, wkv, wo_a, wo_b, shared w1/w2/w3`), split by M.
2. The **M=4 → M=5 delta** specifically (does any real linear step?), and the
   mt=4 vs mt=8 delta at each M (to size the mt-tax precisely).
3. Whether the dense marginal at R1→R4→R5 reproduces the 3.6 ms harness share at
   full 40-layer scale, and how much of the +14 ms R4→R5 is even in the dense
   path (vs MoE — `exl3_moe.py`, out of scope) or is noise.

### Does the source show a non-monotonic R4→R5 mechanism? — **NO**

No M-tile work-padding (`gemv_metal.py:648,1330`), no extra kernel launch
(one dispatch, `gemv_metal.py:1469-1483,1490-1504`), no bandwidth step at M=5
(trellis re-read stays 1x for M≤8, doubling only at M=9, `gemv_metal.py:1423`).
The only M=5 transition is the `simd_mt` specialization step 4→8
(`gemv_metal.py:1422`), which *increases* the amortization group and therefore
cannot create a per-token *saving*. The `VERIFY_MS` dict is empirical
(`spec.py:77`). **Source says: no R4→R5 anomaly mechanism in the dense path.**

---

*No repo file was modified; no bench was run; no git state was touched. This
document is the only artifact written (untracked, uncommitted).*
