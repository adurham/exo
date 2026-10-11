# Task B — per-op-class GPU-time attribution (fragment)

Branch: `p20/prefill-measure-pack` (worktree `/private/tmp/p20-measure-pack`).
Harness: `bench/p20_pmp_taskB_opclass.py` (built locally; node-staged under `/tmp/p20_pmp_B/`).

## Method (as re-specified by design review)

REAL-WEIGHT LAYER-SLICE REPLAY of ONE decoder layer, composed by the PRODUCTION
builder `mlx_lm.models.deepseek_v41.exl3_build.build_block` (real per-rank TP2
`rank=0, world=2` weights + attention/shared-expert sharding), driven through the
real `Block.__call__` with a synthetic 2048-row chunk and a ~100K-token
prefix cache state. Two measurements:

- (a) UNBRACKETED: whole-layer **GPU-busy** (`mx.metal.gpu_time_ns`) + **wall**,
  one lazy production graph drained at the end (ground truth; production
  graph/compile boundaries intact).
- (b) BRACKETED per-op-class: `MLX_GPU_TIME=1` + eval-fenced **module wrappers**
  (each op-class module's OWN result is `mx.eval`'d and the `gpu_time_ns` delta
  since the previous fence is charged to it) — the phase-22 methodology. A
  running counter + nesting charges every GPU-busy ns exactly once; an outer
  wrapper gets only its non-inner glue.

`Σ(bracketed)` vs `(a)` = the bracketing distortion (lost overlap, broken
`mx.compile` fusion, extra command-buffer boundaries).

### Why NOT the `span()` hook
The model's own `span()` hook (profiler) can only `mx.synchronize()` at a
boundary — and MLX `synchronize` does **not** schedule an un-evaluated lazy
graph, so a synchronize-only bracket measures ~nothing (verified: only the
`sparse_attn` internal fence and the final drain got time; everything else read
0.000 ms). Eval-fencing the module result is required. (The production
`SpanProfilerHook` docstring claims sync-mode measures its own kernel time; it
does not, for this reason.)

### Taxonomy
Op classes are the production module boundaries, with the innermost production
`span()` name used to disambiguate call sites (e.g. `EXL3Linear` under
`attn.compressor` → compression, under `moe.shared_experts` → shared-expert
dense). MoE gate/up/down are split by wrapping the low-level
`gemv_metal.inner_mm_seg_mlx` (the segmented prefill kernel) and classifying by
trellis identity + `tn_base` (the production `_prefill` uses it for N=12288 ≫
`EXL3_MM_MAX_ROWS` at DSv4.1 dims, since the gather_mm fallback is not viable).
The `EXL3SwitchGLU` residual = routing glue (sort/gather/scatter).

### TP all_sum
Not attributable via `gpu_time` (MLX's filter DROPS the JACCL RDMA/CPU-only
collective buffers, timestamps ≤ 0). `collective.all_sum` is **stubbed to
identity** (so NO RDMA touches the live link — required) and its cost is bounded
two ways: analytic (payload/BW) + by-difference vs the live unfenced per-chunk
wall.

## Harness facts / staging
- Env: `MTL_DISABLE_TIMEOUT=1`, `MLX_GPU_TIME=1`, `EXL3_MM_SEG=v19d` (production).
- Node venv: `/Users/adam.durham/repos/exo/.venv/bin/python` (mlx
  `0.32.3.dev20260918+603f16eb`, has `gpu_time_ns`/`reset_gpu_time`).
- Checkpoint `~/.exo/models/dealignai--…EXL3-2.9bpw`; engram native release
  `~/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram` (layers 1/14 engram
  embed rows; the EXL3 ckpt has the engram wkv/q/k weights but NOT embed).
- Config: 40 layers; compress_ratios [0,0, 2×18, 1×20] (MTP 3 more);
  kv_source_layers [2,8,14,20]; index_source_layers [2,8,14,20,24,28,32,36];
  engram_layer_ids [1,14]; candidate_source_layer 20, candidate_block_size 8
  (== `DSV41_INDEXER_HIER_BLOCK` 8); moe_inter 2304, E=384, top-6, dim 5120.

## LOCAL SMOKE VALIDATION (mini random config, no checkpoint — plumbing only)
`env -u PYTHONPATH <venv>/bin/python bench/p20_pmp_taskB_opclass.py --mode smoke`
- layers 0/1/2 @ ctx 4096 and layers 1/2 @ ctx 100352 all ran clean.
- Σ(bracketed) closes to (a): **+0.5% / +0.9%** at ctx 100352 (the ±25% seen at
  ctx 4096 is tiny-config fixed-overhead noise, ~µs scale).
- GPU-busy vs wall split works: at 100K smoke, GPU frac ~69–72%.
- all_sum stub fires 0× in smoke (world=1); real mode world=2 ⇒ 2 calls/layer.

## NODE RUN (real EXL3) — see below
Node production `gemv_metal.py` md5 BEFORE: `649d2d6a0a68c8d6eab2a8a5394b7ab9`
(expected; captured read-only). AFTER: recorded post-run.

## Exact guarded command (run from the worktree; waits for /tmp/p20_pmp_gpu_release)
```
env -u PYTHONPATH /Users/adam.durham/repos/exo/.venv/bin/python bench/phase20_guard.py \
  watch --label p20_pmp_B_100k --max-wall 1500 -- \
  /usr/bin/ssh macstudio-m4-2 \
  "cd /tmp/p20_pmp_B && env -u PYTHONPATH MLX_GPU_TIME=1 MTL_DISABLE_TIMEOUT=1 \
     P20_PMP_TRACE=/tmp/p20_pmp_B/trace.json P20_PMP_LAYERS=0,2,3,20,21 \
     P20_PMP_CTX=100352 P20_PMP_REPS=5 P20_PMP_OUT=/tmp/p20_pmp_B/raw/B_100k.json \
     /Users/adam.durham/repos/exo/.venv/bin/python p20_pmp_taskB_opclass.py"
# then:
scp macstudio-m4-2:/tmp/p20_pmp_B/raw/B_100k.json \
    docs/dsv41-prefill-measure-pack-2026-10-11/raw/B/
```
(A second invocation with `P20_PMP_CTX=16384 --label p20_pmp_B_16k` for the 16K
method validation vs the phase-22 fenced table.)

## RESULT — node run completed (real EXL3 experts + production affine6 dense)

Release token `/tmp/p20_pmp_gpu_release` appeared 13:49; both runs executed under
`phase20_guard.py watch` (idle-guarded, exit 0, no aborts). Production
`gemv_metal.py` md5 BEFORE == AFTER == `649d2d6a0a68c8d6eab2a8a5394b7ab9` (no
production contact; node edits confined to `/tmp/p20_pmp_B/`; exo runner
untouched, still up).

Raw: `raw/B/B_16k.json`, `raw/B/B_100k.json` (+ `B_test.json`, `analyze_B.py`,
`run_remote.sh`). Sampled layers: **0, 1, 2, 3, 20, 21, 24** (1/1/2/15/1/15/4
weighted by layer count to the 40-layer chunk; L14≈L2, caveat).

### SHARE TABLE — one 2048-row prefill chunk, weighted 40-layer fold

**~100K ctx (deliverable):** GPU-busy-sum 7974 ms, WALL 7975 ms, Σ(bracketed)
7208 ms → **gap vs wall −9.6%**.

| op-class | ms | share |
|---|---:|---:|
| indexer | 2241.4 | **31.1%** |
| MoE expert (down) | 1376.8 | 19.1% |
| sdpa (sparse/compressed) | 979.7 | 13.6% |
| dense proj (attn o) | 519.9 | 7.2% |
| MoE expert (up) | 516.9 | 7.2% |
| **hyper-connections (HC)** | 371.7 | **5.2%** |
| dense proj (attn qkv+rope) | 308.7 | 4.3% |
| MoE experts total (routing glue: sort/gather/scatter) | 234.5 | 3.3% |
| dense proj (shared expert) | 233.7 | 3.2% |
| MoE expert (gate) | 212.3 | 3.0% |
| norms (RMSNorm) | 121.3 | 1.7% |
| dense proj (other) | 52.3 | 0.7% |
| MoE routing gate (score+topk) | 32.5 | 0.5% |
| CSA/HCA compression | 5.6 | 0.08% |
| engram (LazyEngramTable) | 0.5 | 0.01% |

**16K ctx (method validation):** GPU-busy-sum 6458 ms, WALL 6473 ms,
Σ(bracketed) 5933 ms → **gap vs wall −8.3%**.

| op-class | ms | share | phase-22 fenced (EXL3-dense era) |
|---|---:|---:|---:|
| MoE experts (gate+up+down+glue+total) | 2543.6 | 42.9% | EXL3SwitchGLU 30.7% |
| sdpa | 943.1 | 15.9% | 15.6% ✓ |
| indexer | 800.0 | 13.5% | 3.6% (✗ diverges) |
| dense (qkv+o+shared+other) | 1116.9 | 18.8% | EXL3Linear 16.6% ✓ish |
| HC | 369.5 | 6.2% | not listed |
| norms | 126.2 | 2.1% | 4.1% |
| routing glue (switch total) | 129.6 | 2.2% | Exl3Member 1.1% |
| CSA/HCA compression | 2.8 | 0.05% | not listed |

Reconciles well for the context-INDEPENDENT classes (sdpa 15.9 vs 15.6, dense
18.8 vs 16.6, MoE 42.9 vs 30.7). **The indexer does NOT reconcile** (13.5% vs
3.6% at 16K; 31.1% at 100K) — it scales with depth (nb = end_pos/ratio) and is
the single largest 100K bucket.

### CLOSURE / DISTORTION (item 4)
gap Σ(bracketed) vs (a): **16K −8.3%, 100K −9.6%** (vs busy-sum: −8.1% / −9.6%).
Both < the ~10% threshold ⇒ per-class shares reliable at ~10% resolution;
STATE the borderline status. Direction is NEGATIVE: the unbracketed busy-sum can
EXCEED wall (buffers overlap when unfenced: busy/wall 96.9–118.9% per layer),
while eval-fencing serializes, so Σ(bracketed) lands just under (a) WALL.

### GPU-BUSY vs WALL (item 5)
Own bucket = WALL − Σ(bracketed) = **dispatch / GPU-idle ≈ 8–10%** (16K 539 ms,
100K 768 ms per chunk). The (a) busy-SUM is NOT a clean busy measure: it exceeds
wall at 100K-equivalent layers when unfenced buffers overlap.

### TP all_sum (item 6) — bounded RANGE
- Census: **2 calls/forward, 41.9 MB/forward** per layer (attn `[1,2048,5120]` +
  MoE `[2048,5120]` bf16). gpu_time DROPS these buffers → 0 GPU measured.
- Analytic (@ existing TB5 numbers, raw ~10 GB/s / eff ~5 GB/s):
  MoE-only (production `EXO_DSV4_ATTN_ALLSUM=0`) 839 MB/chunk → **84–168 ms**;
  attn+MoE (as sliced) 1678 MB/chunk → **168–336 ms**.
- By difference (item 6's method): live unfenced 8380 ms/chunk @16K −
  reconstructed 6473 ms → **≈1907 ms** (upper; confounded with dispatch + seq-split).
- **RANGE ≈ 0.08–1.9 s/chunk (≈1%–24%)**; analytic is the RDMA-bound floor.

### LAYER HETEROGENEITY (item 8)
Sampled 7 distinct weight-set types (counts above). Per-layer closure is −0.3% to
−5.1% for compute layers; the engram layer (L1) is an outlier: wall 584 ms but
GPU-busy only 207 ms ⇒ **engram cost is host/SSD latency, ~0 GPU** (its 33.8%
"dense proj (other)" at 100K is the engram `wkv` GEMM, mis-bucketed because it is
un-spanned). L21 @16K showed an anomalous MoE-down (59 ms vs ~20) — flagged for
re-run.

### LARGEST UNEXPLAINED BUCKET
**hyper-connections (HC), ~5–6% of layer GPU** — a real production class NOT in
the requested taxonomy. (MoE routing glue 2–3% and "dense proj (other)" ~0.7%
are the only other unnamed residuals; Σ(bracketed) closes, so nothing else is
unexplained.)


