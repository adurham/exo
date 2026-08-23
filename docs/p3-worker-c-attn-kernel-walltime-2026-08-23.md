# P3 Worker C — MEASURED attention-path kernel wall time per decode step vs context depth (2026-08-23)

**Question**: the live cluster costs **+6.80 ms/token** going from 100,026 →
352,599 real context tokens (35.79 → 42.59 ms/tok, Worker B1). Worker A's
code-derived byte-flow model predicts only **+1.19–1.64 ms/token** of that from
attention-path read bandwidth. Is the missing ~5.2 ms hiding in the
attention/indexer kernels themselves (latency/occupancy-bound, growing faster
than bytes), or is it somewhere else entirely?

**Answer, in one line**: the attention path's real kernel wall time grows by
**+2.56 to +3.34 ms/token** across that range (3 independent runs) — roughly
**2× Worker A's byte model, but still only ~38–49% of the live +6.80 ms**. The
kernels ARE bigger than bytes predict, and they are NOT the whole story. **~3.5
to 4.2 ms/token of the live depth cost is outside the attention block.**

**Machine**: every number in this doc was produced on
**`adams-mac-studio-m4-2.local` (rank1, M4 Max, the production silicon)**,
running the fork's own venv (`~/repos/exo/.venv`, MLX
`0.32.1.dev20260822+e40a416b2`) against the fork's `mlx-lm` at
`~/repos/exo/mlx-lm` (`1fea494`, branch `known-good-decode-fenceasync-20260822`).
Bench scripts were `scp`'d to `/tmp` and run from there. **Nothing under
`~/repos/exo` on either studio was created, edited, or deleted.** Cluster state
verified `RunnerReady` on BOTH runners before the first invocation and after the
last; no generation was in flight; peak GPU allocation of the bench never
exceeded **0.96 GB** (guard was 8 GB) against a 128 GB node holding the live
~85 GB model. A single 520-token smoke run on the MacBook is noted where it
appears and is not used for any conclusion.

---

## 0. Headline table

**43-layer attention-path total, per single-token decode step, B=1, production
shapes/dtypes/env** (async-pipelined fencing — see §2 for why the fencing
discipline is the single biggest methodological decision here):

| L (context tokens) | attn ms/token | Δ vs previous | ms per 100K ctx |
|---|---|---|---|
| 520 | **12.876** | — | — |
| 100,026 | **16.568** | +3.692 | +3.710 |
| 352,599 | **19.130** | +2.562 | +1.014 |
| 500,000 (synthetic) | **21.520** | +2.390 | +1.621 |

**Reproducibility** — three independent full sweeps on the same node:

| run | 100,026 | 352,599 | 500,000 | Δ 100K→352.6K |
|---|---|---|---|---|
| final (async fencing) | 16.568 | 19.130 | 21.520 | **+2.562** |
| earlier run A (held-view fencing) | 15.178 | 18.140 | 20.904 | **+2.962** |
| earlier run B (held-view fencing) | 15.039 | 18.383 | 21.031 | **+3.344** |
| | | | | mean **+2.956**, range 2.56–3.34 |

**Verdict vs the live anchor:**

| quantity | value |
|---|---|
| live measured Δ (B1, 100,026 → 352,599) | **+6.80 ms/token** |
| Worker A byte-flow PREDICTION | +1.19 to +1.64 ms/token |
| **this worker's MEASURED attention-kernel Δ** | **+2.56 to +3.34 ms/token** |
| fraction of the live Δ explained by attention kernels | **38–49%** (point est. **~43%**) |
| **unexplained residual, outside the attention block** | **~3.5–4.2 ms/token** |

The hypothesis stated in the task ("the kernels' wall time grows much faster
than their byte flow, i.e. the kernels ARE the depth cost") is **half right and
half wrong**. Kernel wall time does grow ~1.8× faster than the byte model
predicts. But it does not close the gap: a majority of the live depth cost is
still unaccounted for by the attention path.

---

## 1. Method

### 1.1 What is instantiated

`bench/p3_attn_depth_walltime_microbench.py` builds **one real instance of each
production attention class** straight out of the fork —
`dv4.v4_attention_factory(args, layer_idx)` — with random weights but exact
production config, then drives it with **real single-token decode calls**
(`attn(x, mask=None, cache=cache)`, B=1, L_q=1) against a synthetic pre-filled
cache at depth L.

Layer census, taken from `config.compress_ratios[:43]` of the real checkpoint
(`~/.exo/models/deepseek-ai--DeepSeek-V4-Flash-0731/config.json`, read via
read-only ssh; the raw list is `[0,0,4,128,4,128,...,4]`, 46 entries truncated
to 43 by `deepseek_v4.py:888`):

| ratio | count | class |
|---|---|---|
| 0 | 2 | `LocalAttention` |
| 4 | 21 | `SparseCompressedAttention` (has the indexer) |
| 128 | 20 | `CompressedAttention` |

Per-class median ms × count → 43-layer total. This is exact up to inter-layer
pipelining, which a per-class bench cannot capture (see §6).

**Why one layer per class and not 43**: memory. 43 blocks is ~5.3 GB of weights
(Worker A's A4 term) on a node already holding the live 85 GB model. One block
per class peaks at **0.96 GB** including the L=500K synthetic cache. The blocks
within a class are identical and independent, so the scaling is arithmetic, not
an approximation of the per-layer cost.

### 1.2 Production fidelity

Quantization replicates `make_quantization_config` (`deepseek_v4.py:899-931`)
exactly: `.attn.w*` and `.attn.indexer.wq*` → **mxfp8 g=32 b=8**; everything
else (`compressor.wkv`, `compressor.wgate`, `indexer.weights_proj`) → **affine
g=64 b=8**. Compute dtype **bf16**, KV cache **bf16 unquantized**
(`EXO_KV_CACHE_BITS=0`, `start_cluster.sh:151`).

Env set to `start_cluster.sh` defaults before importing mlx (several are read
once at module import): `EXO_DSV4_INDEX_TOPK=512`,
`EXO_DSV4_SPARSE_SDPA_TILE=128`, `EXO_DSV4_EXACT_TOPK=1`,
`EXO_DSV4_TOPK_FUSED=0`, `EXO_DSV4_SPARSE_FUSED_SDPA=0`,
`EXO_DSV4_ATTN_ALLSUM=0`, `EXO_DSV4_SINGLE_GATHER=1`,
`EXO_DSV4_POOL_DEFER_COPY_MAX_BYTES=8388608`. `EXO_DSV4_INDEXER_PBLOCK`,
`EXO_DSV4_QA_KV_FUSED`, `EXO_DSV4_FP32_ACT`, `EXO_PROFILER` explicitly popped.

Cache pre-fill mirrors `Model.make_cache` (`deepseek_v4.py:6956-6979`):
`RotatingKVCache(max_size=128)` put into steady-state rotation
(`_idx = max_size`, `offset = L`, zero-width values placeholder as DSv4 passes);
`PoolingCache(ratio)` filled to `P = L // ratio` entries with realistic
step-256 storage allocation, a realistic `remainder`, and (for ratio-4 layers)
the cross-call overlap carry populated.

**Steps run: 256 consecutive decode steps per class per depth** (warmup 32).
This matters: the Compressor only emits a pooled entry every `ratio` steps
(1-in-4 sparse, 1-in-128 compressed) and `PoolingCache` reallocs every 256
pooled entries, so a 256-step average amortizes both exactly as production does.
A short bench would systematically miss the pool-write cost.

**TP note**: exo replicates attention on both ranks and shards only the MoE
(`auto_parallel.py:1032-1034`, confirmed by Worker A), and seq-split requires
`L_q >= 16` (`_SEQ_SPLIT_MIN_L = 16`, `deepseek_v4.py:225`) so it is inactive at
single-token decode. A single-process microbench therefore measures the full
per-rank attention work with no correction needed.

---

## 2. The fencing decision — read this before trusting any absolute number

MLX is lazy. **How you fence dominates the answer**, by up to 2×. Three modes
were measured for every point:

| mode | what it does | r=4 layer @352.6K |
|---|---|---|
| `fenced` | `mx.eval + mx.synchronize` around EVERY step | 0.731 ms |
| `heldview` | 16 steps issued, ONE fence around the group | 0.546 ms |
| `async` | 16 steps issued, each `mx.async_eval`'d immediately | 0.540 ms |

`fenced` is unusable as an absolute: the measured **per-fence round-trip floor
on a 16-element op is 0.197 ms**, so a per-step fence adds ~0.20–0.30 ms to
every single sample — 8.5–13 ms/token once multiplied by 43 layers. The
fenced 43-layer totals (22.3 / 27.7 / 27.9 / 29.2 ms across the four depths) are
dominated by that artifact and are reported only as a serialized upper bound.

`async` is the production-faithful one. Production pipelines
(`EXO_DSV4_FENCE_EVERY_N_LAYERS=4` with the async fence,
`start_cluster.sh:437`/`:1626`) and `PoolingCache.update_and_fetch_deferred`
issues an explicit `mx.async_eval(self._pool_storage)` at `cache.py:1551` whose
own comment explains exactly why:

> *"Measured: without this, async-fence pipelining keeps stale views alive at
> eval time and the write silently degrades to a full O(P·D) copy (~0.85ms/flush
> at 500K shapes vs ~0 donated)."*

`heldview` reproduces precisely that failure — holding 16 step outputs alive
keeps 16 pool-storage views alive and defeats donation. **This is a real,
measured chain-length sensitivity**, and it is the reason the earlier runs A/B
(heldview) show slightly larger depth deltas than the final async run:

```
### chain=4                       ### chain=8                       ### chain=32
r=4 @100K  0.4346                 r=4 @100K  0.4084                 r=4 @100K  0.4043
r=4 @352K  0.4611                 r=4 @352K  0.4937                 r=4 @352K  0.5551
```

At 100K the compressor pool is 25,006×512 bf16 = **25.6 MB**; at 352.6K it is
**90.3 MB**. Both are far above `EXO_DSV4_POOL_DEFER_COPY_MAX_BYTES` (8 MiB), so
production takes the donation branch at both depths. The held-view mode defeats
that donation, and the cost of doing so scales with pool size — which is exactly
why chain length barely moves the 100K number (0.435→0.408→0.404 for chain
4/8/32) but strongly moves the 352.6K one (0.461→0.494→0.555). The async mode
removes the artifact at both depths.

**All headline numbers in §0 use async.** The held-view runs are kept in the
table because they bracket the answer from above and the conclusion is identical
either way.

---

## 3. Raw bench output (studio, final async run)

```
MLX 0.32.1.dev20260822+e40a416b2  device=Device(gpu, 0)
host=Adams-Mac-Studio-M4-2
env: {"EXO_DSV4_INDEX_TOPK": "512", "EXO_KV_CACHE_BITS": "0", "EXO_COMPUTE_DTYPE": "bf16", "EXO_DSV4_SPARSE_SDPA_TILE": "128", "EXO_DSV4_SEQ_SPLIT": "1", "EXO_DSV4_EXACT_TOPK": "1", "EXO_DSV4_TOPK_FUSED": "0", "EXO_DSV4_SPARSE_FUSED_SDPA": "0", "EXO_DSV4_ATTN_ALLSUM": "0", "EXO_DSV4_SINGLE_GATHER": "1", "EXO_DSV4_POOL_DEFER_COPY_MAX_BYTES": "8388608", "EXO_DSV4_PREFILL_ARGPARTITION": "1", "EXO_DSV4_ARGPARTITION_MIN_P": "8192"}
layer census: r=0 x2, r=4 x21, r=128 x20  (43 total)
steps/depth/class = 256 (warmup 32), chain=16
session streaming bandwidth (r+w): 404.7 GB/s
per-fence round-trip floor (mx.eval+synchronize on a 16-elem op): 0.1966 ms

========================================================================
=== DEPTH L = 520 ===
  r=0    ( 2 layers)  async(x16)  0.2736 ms | heldview(x16)  0.2310 | fenced  0.4857  P_comp=0 P_idx=0
        peak GPU mem so far: 0.81 GB
  r=4    (21 layers)  async(x16)  0.3123 ms | heldview(x16)  0.3347 | fenced  0.5345  P_comp=130 P_idx=130
        peak GPU mem so far: 0.89 GB
  r=128  (20 layers)  async(x16)  0.2885 ms | heldview(x16)  0.2454 | fenced  0.5058  P_comp=4 P_idx=0
        peak GPU mem so far: 0.89 GB
  --> 43-layer attention-path total (CHAINED, production-like):   12.876 ms/token
      (fenced/serialized upper bound:   22.312 ms/token)
      breakdown: r=0   0.547 | r=4   6.559 | r=128   5.771 ms
  components (chained) @P_idx=130 k=130 P_c=4:
    indexer.score   0.0278 ms  x21 =   0.583 ms
    indexer.topk    0.0014 ms  x21 =   0.030 ms
    kv gather       0.0240 ms  x21 =   0.504 ms
    sdpa sparse     0.0482 ms  x21 =   1.013 ms
    sdpa compress   0.0379 ms  x20 =   0.758 ms
  achieved GB/s on the depth-dependent reads: indexer-score     1.2  exact-topk     0.7  sdpa-compressed     3.6   (session peak 405)

========================================================================
=== DEPTH L = 100,026 ===
  r=0    ( 2 layers)  async(x16)  0.2749 ms | heldview(x16)  0.2278 | fenced  0.4857  P_comp=0 P_idx=0
        peak GPU mem so far: 0.89 GB
  r=4    (21 layers)  async(x16)  0.4550 ms | heldview(x16)  0.4240 | fenced  0.7224  P_comp=25,006 P_idx=25,006
        peak GPU mem so far: 0.89 GB
  r=128  (20 layers)  async(x16)  0.3232 ms | heldview(x16)  0.2833 | fenced  0.5791  P_comp=781 P_idx=0
        peak GPU mem so far: 0.89 GB
  --> 43-layer attention-path total (CHAINED, production-like):   16.568 ms/token
      (fenced/serialized upper bound:   27.724 ms/token)
      breakdown: r=0   0.550 | r=4   9.555 | r=128   6.463 ms
  components (chained) @P_idx=25,006 k=512 P_c=781:
    indexer.score   0.0280 ms  x21 =   0.589 ms
    indexer.topk    0.0154 ms  x21 =   0.323 ms
    kv gather       0.0254 ms  x21 =   0.533 ms
    sdpa sparse     0.0528 ms  x21 =   1.109 ms
    sdpa compress   0.0646 ms  x20 =   1.291 ms
  achieved GB/s on the depth-dependent reads: indexer-score   228.4  exact-topk    13.0  sdpa-compressed    14.4   (session peak 405)

========================================================================
=== DEPTH L = 352,599 ===
  r=0    ( 2 layers)  async(x16)  0.2715 ms | heldview(x16)  0.2269 | fenced  0.4866  P_comp=0 P_idx=0
        peak GPU mem so far: 0.89 GB
  r=4    (21 layers)  async(x16)  0.5403 ms | heldview(x16)  0.5455 | fenced  0.7309  P_comp=88,149 P_idx=88,149
        peak GPU mem so far: 0.89 GB
  r=128  (20 layers)  async(x16)  0.3620 ms | heldview(x16)  0.3228 | fenced  0.5776  P_comp=2,754 P_idx=0
        peak GPU mem so far: 0.89 GB
  --> 43-layer attention-path total (CHAINED, production-like):   19.130 ms/token
      (fenced/serialized upper bound:   27.875 ms/token)
      breakdown: r=0   0.543 | r=4  11.347 | r=128   7.240 ms
  components (chained) @P_idx=88,149 k=512 P_c=2,754:
    indexer.score   0.0473 ms  x21 =   0.993 ms
    indexer.topk    0.0196 ms  x21 =   0.412 ms
    kv gather       0.0202 ms  x21 =   0.423 ms
    sdpa sparse     0.0523 ms  x21 =   1.099 ms
    sdpa compress   0.0776 ms  x20 =   1.552 ms
  achieved GB/s on the depth-dependent reads: indexer-score   477.0  exact-topk    35.9  sdpa-compressed    38.0   (session peak 405)

========================================================================
=== DEPTH L = 500,000 ===
  r=0    ( 2 layers)  async(x16)  0.2721 ms | heldview(x16)  0.2300 | fenced  0.4822  P_comp=0 P_idx=0
        peak GPU mem so far: 0.89 GB
  r=4    (21 layers)  async(x16)  0.6464 ms | heldview(x16)  0.6619 | fenced  0.7971  P_comp=125,000 P_idx=125,000
        peak GPU mem so far: 0.96 GB
  r=128  (20 layers)  async(x16)  0.3701 ms | heldview(x16)  0.3324 | fenced  0.5773  P_comp=3,906 P_idx=0
        peak GPU mem so far: 0.96 GB
  --> 43-layer attention-path total (CHAINED, production-like):   21.520 ms/token
      (fenced/serialized upper bound:   29.249 ms/token)
      breakdown: r=0   0.544 | r=4  13.574 | r=128   7.402 ms
  components (chained) @P_idx=125,000 k=512 P_c=3,906:
    indexer.score   0.0573 ms  x21 =   1.204 ms
    indexer.topk    0.0212 ms  x21 =   0.446 ms
    kv gather       0.0443 ms  x21 =   0.931 ms
    sdpa sparse     0.0550 ms  x21 =   1.155 ms
    sdpa compress   0.0867 ms  x20 =   1.735 ms
  achieved GB/s on the depth-dependent reads: indexer-score   558.0  exact-topk    47.1  sdpa-compressed    47.6   (session peak 405)

========================================================================
=== SCALING ===
         L  attn ms/tok  delta vs prev  ms per 100K
       520       12.876              -            -
   100,026       16.568          3.692        3.710
   352,599       19.130          2.562        1.014
   500,000       21.520          2.390        1.621

linear fit over L>=50K: ms = 15.2182 + 1.2139 per 100K tokens
  L=  100,026  actual   16.568  fit   16.432  resid  +0.1357 ms ( +0.82%)
  L=  352,599  actual   19.130  fit   19.498  resid  -0.3681 ms ( -1.92%)
  L=  500,000  actual   21.520  fit   21.288  resid  +0.2325 ms ( +1.08%)
```

---

## 4. Per-component breakdown

### 4.1 Isolated kernels at exact decode shapes (async fencing, ms per call)

| component | L=520 | L=100,026 | L=352,599 | L=500,000 | ×layers | Δ 100K→352.6K (×layers) |
|---|---|---|---|---|---|---|
| `_indexer_score` GEMM | 0.0278 | 0.0280 | 0.0473 | 0.0573 | ×21 | **+0.405 ms** |
| `_exact_topk` (4-pass Metal) | 0.0014 | 0.0154 | 0.0196 | 0.0212 | ×21 | **+0.088 ms** |
| pooled-KV gather (OPT-10) | 0.0240 | 0.0254 | 0.0202 | 0.0443 | ×21 | −0.110 ms |
| core SDPA, sparse (128+512 rows) | 0.0482 | 0.0528 | 0.0523 | 0.0550 | ×21 | −0.010 ms |
| core SDPA, compressed (128+L/128) | 0.0379 | 0.0646 | 0.0776 | 0.0867 | ×20 | **+0.261 ms** |
| | | | | | **isolated sum** | **+0.64 ms** |
| | | | | | **whole-layer measured** | **+2.56 ms** |

Two things jump out.

1. **The depth-dependent kernels behave exactly as Worker A's byte model
   predicted.** The indexer score GEMM is genuinely linear in L (its pooled read
   is `P·128·2 B` and its measured time tracks that), the top-k grows slowly,
   the sparse gather and sparse SDPA are **flat in L** — confirming the sparse
   design's O(1) core attention empirically, not just from code. The compressed
   SDPA grows linearly with its L/128 pool. This is a clean empirical
   confirmation of Worker A §2.

2. **But the isolated kernels only account for ~25% of the whole-layer depth
   delta** (+0.64 of +2.56 ms). The remaining ~1.9 ms lives in per-layer work
   that a kernel-in-isolation bench cannot see — the compressor pool write, the
   `commit_pending`/storage-realloc bookkeeping, allocator pressure from the
   growing pool, and dispatch scheduling around a larger working set. Chasing it
   with a dedicated probe (§4.3) found it is dominated by the **pool write**.

### 4.2 Sub-span attribution inside a sparse layer (fork's own instrumentation)

Using `EXO_DSV4_SECTION_TIME=1`, which arms the `_ATTN_SUB_ACC` fenced blocks
already present in `SparseCompressedAttention.__call__` (`deepseek_v4.py:4505,
4532, 4561, 4589, 4758, 4783`). These are per-sub-span `mx.eval + mx.synchronize`
fences, so the TOTAL is a serialized upper bound; the useful signal is the
per-sub-span **change with L**.

```
         L  compressor   proj_qkv    qk_prep    indexer       sdpa   out_proj       wall
       520      0.1797     0.3040     0.2134     0.3082     0.2412     0.3733     1.6398
   100,026      0.1805     0.2960     0.2103     0.3530     0.3166     0.3751     1.7514
   352,599      0.1812     0.2960     0.2176     0.4196     0.3155     0.3729     1.8224
   500,000      0.1815     0.2983     0.2138     0.4702     0.2536     0.3720     1.8088

=== same deltas SCALED x21 sparse layers (ms/token) ===
                 segment  compressor   proj_qkv    qk_prep    indexer       sdpa   out_proj        sum
       520->100,026          +0.0166    -0.1681    -0.0639    +0.9409    +1.5836    +0.0371    +2.3462
   100,026->352,599          +0.0137    -0.0013    +0.1516    +1.3979    -0.0235    -0.0461    +1.4922
   352,599->500,000          +0.0071    +0.0500    -0.0790    +1.0623    -1.2983    -0.0180    -0.2759
```

**Only `indexer` grows monotonically with depth.** `proj_qkv`, `qk_prep`,
`out_proj`, `compressor` are flat to within noise (as they must be — none of
them touches L). `sdpa` is noisy and non-monotonic (the −1.30 at 500K is
noise-dominated; per-sub-span fences on a ~0.25 ms span have poor SNR).
Of the +1.49 ms/token sparse-layer growth over 100K→352.6K captured here,
**+1.40 ms is the indexer block** — 94%.

Note the `compressor` sub-span reads flat, which is *not* inconsistent with
§4.3: the SECTION_TIME fence sits after the compressor call, and the pool write
is `update_and_fetch_deferred` — a *deferred* lazy slice-assign whose cost lands
when the next step's graph evaluates, not inside this sub-span's fence.

### 4.3 Where the rest of the per-layer delta lives: the pool write

A dedicated probe isolating the `PoolingCache.update_and_fetch_deferred` write
at each pool width (per call, ms):

```
         L  indexer_module  idx_poolwrite  comp_poolwrite  combined_kv_cat
       520          0.0729         0.0207          0.0208           0.0145
   100,026          0.0870         0.0308          0.0867           0.0144
   352,599          0.1169         0.0771          0.3891           0.0143
   500,000          0.1447         0.1082          0.5509           0.0144

=== deltas x21 sparse layers (ms/token) ===
       520->100,026    indexer_module  +0.2960  idx_pool  +0.2126  comp_pool  +1.3851
   100,026->352,599    indexer_module  +0.6266  idx_pool  +0.9720  comp_pool  +6.3501
   352,599->500,000    indexer_module  +0.5845  idx_pool  +0.6533  comp_pool  +3.3977
```

The compressor pool write scales **linearly with pool size** — the signature of
an O(P·D) copy, i.e. donation NOT firing in this probe's isolation (the probe
holds a reference, which is exactly the failure mode documented at
`cache.py:1547-1553`). **These numbers are an UPPER BOUND, not a production
measurement**: in the whole-layer async bench the same write costs far less,
which is why the whole-layer delta is +2.56 ms and not the +7 ms this probe
would imply. Reported because it shows *what the mechanism is* and how large it
gets when donation fails — a real production risk, not a measured production
cost. The `combined_kv` concat is genuinely flat (0.0144 ms at every depth),
confirming Worker A's A3 term is depth-independent.

### 4.4 Achieved bandwidth on the depth-dependent reads

Session streaming bandwidth measured in the same process: **404.7 GB/s** (r+w).

| L | indexer-score GB/s | exact-topk GB/s | sdpa-compressed GB/s |
|---|---|---|---|
| 520 | 1.2 | 0.7 | 3.6 |
| 100,026 | **228.4** | 13.0 | 14.4 |
| 352,599 | **477.0** | 35.9 | 38.0 |
| 500,000 | **558.0** | 47.1 | 47.6 |

**This is the most informative single table in the doc.**

- **The indexer score GEMM is bandwidth-bound and running at or above the
  measured streaming ceiling** (477–558 GB/s vs a 404.7 GB/s streaming copy;
  it exceeds it because a GEMM reading a 22 MB pooled tensor gets L2 reuse a
  256 MB streaming copy does not). **There is no headroom here.** The kernel is
  doing the minimum possible work at the maximum possible rate. Worker A's
  bytes-based prediction for THIS term was right.
- **The top-k and compressed-SDPA kernels are latency/occupancy-bound**, running
  at 8–12% of streaming bandwidth. They have headroom in principle — but they
  are small in absolute terms (+0.088 and +0.261 ms over the range), so
  optimizing them cannot recover multiple ms.
- The **increasing** GB/s with depth for every component is the fingerprint of
  fixed launch overhead being amortized over more work. At small P these kernels
  are pure overhead (1.2 GB/s at L=520); at large P they approach their real
  efficiency. **This is the opposite of the hypothesised "kernels degrade at
  depth" mechanism** — they get *more* efficient, not less.

---

## 5. Scaling verdict — LINEAR, with the curvature coming from below 100K

Fit over the three deep points (L ≥ 50K), final async run:

```
ms = 15.2182 + 1.2139 per 100K tokens
  L=  100,026  actual   16.568  fit   16.432  resid  +0.1357 ms ( +0.82%)
  L=  352,599  actual   19.130  fit   19.498  resid  -0.3681 ms ( -1.92%)
  L=  500,000  actual   21.520  fit   21.288  resid  +0.2325 ms ( +1.08%)
```

Residuals within ±2%. Run A gives `13.6175 + 1.4034/100K` (resid +1.03/−2.35/
+1.29%), run B `13.4537 + 1.4792/100K` (resid +0.70/−1.56/+0.86%). **All three
fits are linear to within ±2.4%.** The consistent sign pattern (+,−,+) is a
faint hint of slight *concavity* (sub-linear), not super-linearity.

Marginal cost per 100K context tokens, final run:

| segment | ms per 100K |
|---|---|
| 520 → 100,026 | **+3.710** |
| 100,026 → 352,599 | **+1.014** |
| 352,599 → 500,000 | **+1.621** |

**The attention path does NOT reproduce B1's end-to-end curvature.** B1 found
+2.05 ms/100K over the first 100K and +2.69 ms/100K over the next 253K — costs
*accelerating* with depth. The attention kernels do the reverse: a large fixed
step from 520→100K (largely fixed kernel-launch and first-pool-allocation
effects, plus the sparse layers switching from the dense-concat branch to the
sparse branch above L≈2048) and then a much flatter, essentially linear regime
above 100K. The 352.6K→500K uptick (+1.62 vs +1.01) is within the fit's
residual band and within run-to-run spread (runs A/B give +1.88/+1.80 and
+1.32/+1.79 respectively — the ordering is not even stable across runs).

**Conclusion: attention-path kernel wall time is LINEAR in L above 100K. The
mildly super-linear end-to-end decay B1 measured is NOT coming from the
attention kernels.**

---

## 6. Comparison against the live budget, and what does NOT add up

| | 100,026 | 352,599 | Δ |
|---|---|---|---|
| live total per token (B1) | 35.79 ms | 42.59 ms | **+6.80** |
| this bench, attention path only | 16.57 ms | 19.13 ms | **+2.56** |
| attention as % of live total | 46.3% | 44.9% | — |
| **residual (non-attention)** | **19.22 ms** | **23.46 ms** | **+4.24** |

**Sanity check passes**: the attention-only sum is 45–46% of the live per-token
budget at both depths — comfortably under it, so nothing is double-counted or
miscounted. That is a plausible share for a 43-layer MoE model where each layer
also runs a 256-expert top-6 MoE block plus a TP all_sum.

**Prior-history check**: `docs/dsv4-attention-kernel-efficiency-2026-08-18.md`
warns microbenches can mismatch production 2–4×. This one does not appear to —
the absolute share is sane and the 520-token point (12.88 ms) sits close to
Worker A's constant-term prediction (18.5 ms at 297 GB/s; we measure 405 GB/s
streaming on the studio, and 5.298 GB / 405 GB/s = 13.1 ms, essentially exactly
what was measured). **The constant term validates the bench against an
independent code-derived model to within 2%.** That is a strong indication the
absolute scale is right.

### The honest gap

The +4.24 ms/token of non-attention depth cost is **outside anything this worker
measured**. Candidates, none tested here:

1. **MoE all_sum arrival skew.** Worker A proved the collective's payload cannot
   grow with L (fixed `(1,1,4096)`, 43×/token). But if one rank's attention runs
   ~1.3 ms/token slower at depth than the other's — and it will, since neither
   rank's attention is identical in scheduling — the collective waits.
   43 calls/token amplifies small per-layer skew. `docs/moe-all-sum-skew-vs-comms-2026-08-19.md`
   and `docs/cross-rank-allsum-skew-2026-08-22.md` already document this
   mechanism.
2. **Inter-layer pipelining loss.** This bench measures one layer at a time with
   a synchronize between classes. Production runs 43 layers back-to-back with a
   fence every 4. If the growing pool working set degrades cross-layer overlap
   at depth, this bench structurally cannot see it — and would understate the
   attention path's true production cost.
3. **Unified-memory / allocator pressure.** 2.43 GB of resident KV at 352.6K on
   top of ~83 GB of weights (Worker A §4) vs 0.69 GB at 100K. This bench runs a
   0.96 GB process on an idle-ish node; it cannot reproduce a 85 GB-resident
   allocator regime.
4. **Pool-write donation intermittently failing in production.** §4.3 shows the
   cost when it does: up to +6.35 ms/token over 100K→352.6K for the compressor
   pool alone. If donation fails on even a fraction of steps at depth, that alone
   could close the gap. **This is the single most testable follow-up.**

---

## 7. Honest limitations

- **Random weights.** Quantized mxfp8/affine tensors from random data have the
  right shapes, dtypes, and memory traffic but not production's value
  distribution. Irrelevant for GEMM/SDPA/gather timing; potentially relevant for
  the exact-topk kernel's branch behaviour (its 4-pass histogram/refine
  structure is data-dependent). Top-k is +0.088 ms of the delta, so any error
  here is negligible against the ~4 ms gap.
- **One layer per class, scaled arithmetically.** Cannot capture inter-layer
  pipelining, cross-layer cache effects, or scheduling interactions. See §6.2.
  This is the largest structural limitation and it biases the attention estimate
  **downward**.
- **No MoE, no collectives, no distributed anything.** By design (this is the
  attention-path question), but it means the +4.24 ms residual is bounded from
  below, not attributed.
- **Idle-node conditions.** The studio was hosting an idle-but-loaded runner
  (~85 GB resident) but no active generation. Production decode at depth runs
  under a very different memory-pressure and GPU-contention regime.
- **B1's live anchors are n=1 per depth** and its own doc flags the deep point
  as +9.2% above T1's. If the true live delta were T1's rather than B1's, the
  attention share would be different. Using B1 as instructed.
- **The 500,000 point is synthetic** — no live 500K anchor exists to compare it
  against. It is a genuine measurement of the kernels at that shape, not an
  extrapolation, but it has no live counterpart.
- **`heldview` vs `async` differ by 5–15%** at the layer level. Both are
  reported; the conclusion (attention explains ~40–50%, scaling is linear) is
  identical under either.
- **The §4.3 pool-write numbers are an upper bound**, explicitly labelled as
  such — the probe's own reference-holding defeats donation.
- Everything above 100K in §5's fit is 3 points. A 3-point linear fit with ±2%
  residuals is suggestive, not conclusive, about the absence of curvature.
- **Nothing was committed to git.**

---

## 8. Files

- `bench/p3_attn_depth_walltime_microbench.py` — new, additive. The main bench.
- `bench/p3_attn_subspan_attribution.py` — new, additive. §4.2 sub-span
  attribution using the fork's own `EXO_DSV4_SECTION_TIME` instrumentation.
- `/tmp/p3c_final.log`, `/tmp/p3c_studio_run1.log`, `/tmp/p3c_studio_run2.log`,
  `/tmp/p3c_chainsens.log`, `/tmp/p3c_subspan.log`, `/tmp/p3c_probe3.log` — raw
  stdout, local copies of the studio runs.
- `/tmp/p3c_final.json`, `/tmp/p3c_subspan.json` — machine-readable results.
- No existing bench script was modified. No file on either studio outside `/tmp`
  was touched.
