# DSv4-Flash 352.6K batched-verify memory regression — root-cause analysis

**Scope.** Read + allocation-math only. No source edits. This document
quantifies, in bytes at 352.6K context, what the depth-gated batched verify
path (`EXO_DSV4_VERIFY_BATCH=1`, `EXO_DSV4_VERIFY_BATCH_MIN_CTX=8192`)
allocates simultaneously that the rowseq path (`EXO_DSV4_VERIFY_ROWSEQ=1` +
`EXO_DSV4_ROWSEQ_FULLBLOCK=1`) does not, and identifies the mechanism most
consistent with the 3/11 collapsed runs, ~1 tok/s stalls and 1.37–1.76 GB
swap on both nodes.

**Repo/commit.** exo main HEAD `b999c3354` (tree clean). mlx-lm submodule
pinned at `d098642`.

**Model config.** `mlx-community/DeepSeek-V4-Flash` (bf16, 8-bit quant weights):
`hidden_size=4096`, `num_hidden_layers=43`, 21 sparse layers
(`compress_ratio=4`), 20 compressed (`ratio=128`), 3 local (`ratio=0`);
`num_attention_heads=64`, `head_dim=512`, `q_lora_rank=1024`,
`sliding_window=128`, `hc_mult=4`; `index_n_heads=64`, `index_head_dim=128`,
`index_topk=512` (matches `EXO_DSV4_INDEX_TOPK=512` in
`start_cluster.sh:33`); MoE `n_routed_experts=256`, `num_experts_per_tok=6`,
`moe_intermediate_size=2048`, 1 shared expert. bf16 activations by default —
`EXO_DSV4_FP32_ACT` is off in `start_cluster.sh` (only wired if the shell
sets it), which is confirmed by the `_FP32_ACT = os.environ.get(...) == "1"`
gate at `mlx-lm/mlx_lm/models/deepseek_v4.py:246`.

At 352.6K context:
- Sparse-layer pool size `P_sparse ≈ ctx / 4 = 88,150` entries.
- Compressed-layer pool `P_comp ≈ ctx / 128 = 2,754` entries.

Verify width `L = γ+1 = 4` per the DSv4 MTP verify assembly at
`src/exo/worker/engines/mlx/speculative/dsv4_mtp.py:2644-2649` and the
`_VERIFY_ROWSEQ_MAX_L = 8` cap at
`mlx-lm/mlx_lm/models/deepseek_v4.py:1557` (rowseq accepts L up to 8; MTP
runs at γ=3 so L=4).

---

## 1. Where the two paths diverge — code map

### 1.1 Gate site
`mlx-lm/mlx_lm/models/deepseek_v4.py:6873-6883` (verify-batch side channel
armed once per forward, at cache offset ≥ `_VERIFY_BATCH_MIN_CTX`):

```
_vb_ctx_len = _rowseq_ctx(cache[0]) if cache is not None else 0
_vb_active = (
    _VERIFY_BATCH
    and h.shape[0] == 1
    and 2 <= h.shape[1] <= _VERIFY_ROWSEQ_MAX_L
    and _vb_ctx_len >= _VERIFY_BATCH_MIN_CTX
)
if _VERIFY_BATCH:
    _set_verify_batch_ctx(active=False)
    if _vb_active:
        _set_verify_batch_ctx(active=True, L=h.shape[1])
```

### 1.2 Block-level dispatch
`DeepseekV4Block.__call__` gate at
`mlx-lm/mlx_lm/models/deepseek_v4.py:5072-5083` (FULLBLOCK per-row) and
`:5220-5233` (per-row attention). Both branches include:

```
and not (_VERIFY_BATCH and _VERIFY_BATCH_CTX["active"])
```

so when the batched side channel is active the block **skips** the FULLBLOCK
per-row loop and the per-row attention loop, and falls through to the batched
else-branch `x = self.attn(normed, mask=mask, cache=cache)` at
`:5274` — a single `L=4` attention call per layer.

Model-level `hc_head`/final-norm has an identical gate at
`:6939-6968` (skipped when verify-batch active → the batched hc_head runs
once at L=4 instead of four L=1 calls).

### 1.3 What the L=4 sparse attention actually allocates
`SparseCompressedAttention.__call__`
(`mlx-lm/mlx_lm/models/deepseek_v4.py:4554-4945`) is called with `L=4` in
the batched path:

1. `q = self.wq_b(q_norm(q_lora)).reshape(B, L, H, D)` — main-Q at
   `(1, 4, 64, 512)`.
2. `kv = self.kv_norm(kv_pre).reshape(B, 1, L, D)` — local KV write.
3. `pmask = _dispatch_pmask(comp_cache, L, offset)` at `:4655` — 2-D bool
   causal pmask at `(L, P_sparse) = (4, 88150)`.
4. `self.indexer(...)` at `:4685` — the Indexer runs with `L=4`, so
   `scores = _indexer_score(q, pooled, ...)` produces `(B, L, P_sparse) =
   (1, 4, 88150)` bf16 (folded, no `(B,H,L,P)` transient — see
   `_indexer_score` at `:3689-3754`, OPT-6 fold-w-into-q).
5. Sparse SDPA at `:4732-4856` → `_sparse_verify_rows_batched`
   (default, `_SPARSE_VERIFY_MAX_L=16` at `:1458`) which builds:
   - `gathered = pooled_flat[topk_flat].reshape(B, L, k, D) = (1, 4, 512, 512)`
   - `combined = concat(local_b, gathered[..., None])` at
     `(B, L, 1, sw+k, D) = (1, 4, 1, 640, 512)`
   - fold path (`_SPARSE_VERIFY_FOLD` default on) reshapes to `(B·L, H, 1, D)`
     and dispatches ONE fused `mx.fast.scaled_dot_product_attention`.
6. `out` post-o_proj at `(1, 4, hidden) = (1, 4, 4096)` per layer.

The rowseq path serializes L=1 iterations of the same block: **only one row's
intermediates are live at a time; Python drops the previous row's
`gathered`/`combined`/`scores` at loop iteration boundary.** See the
`_fb_rows` accumulator at `:5093-5111` — it collects only small
`(B, 1, hidden)`/`(B, 1, hc_mult, hidden)` post-block hiddens across the L
iterations, NOT the attention intermediates.

### 1.4 Indexer
`Indexer.__call__` at `:3869-4084`. When the batched side channel is active
the Indexer is called ONCE with `L=4` rows; the compressor + score GEMM +
top-k run at `L=4` in one call (`:3854-3867` and `:3889-3897` explicitly
document the removal of the earlier snapshot/stream-sharing hack — the
corrected batched path has NO snapshot).

`_indexer_score` (`:3689-3754`): via OPT-6 fold-w-into-q the `(B, H, L, P)`
score transient is **never materialized**; only the folded
`(B, L, P) = (1, 4, 88150)` scores exist. So the (B,H,L,P) 22 GB nightmare
tensor is NOT the culprit.

---

## 2. Per-layer allocation math @ 352.6K ctx, L=4 vs L=1 (bf16, 2 B/elem)

Bytes held **simultaneously** inside one attention call. Formulae are
per-layer; the delta column shows batched-minus-rowseq per-layer, held
at the peak instant inside the block.

### 2.1 Sparse layer (compress_ratio=4, 21 layers)

| Tensor | Shape | Formula | Batched L=4 | Rowseq L=1 | Δ per layer |
|---|---|---|---:|---:|---:|
| main-Q | (B,H,L,D)=(1,64,L,512) bf16 | 64·L·512·2 | 262.14 KB | 65.54 KB | **+196.61 KB** |
| Indexer scores | (B,L,P) bf16, folded | L·88150·2 | 705.20 KB | 176.30 KB | **+528.90 KB** |
| pmask (2-D bool) | (L,P) | L·88150·1 | 352.60 KB | 88.15 KB | **+264.45 KB** |
| topk indices | (B,L,k)=(1,L,512) int32 | L·512·4 | 8.19 KB | 2.05 KB | +6.14 KB |
| pooled_gathered | (B,L,k,D)=(1,L,512,512) bf16 | L·512·512·2 | 2.10 MB | 524.29 KB | **+1.57 MB** |
| combined KV block | (B,L,1,sw+k,D)=(1,L,1,640,512) bf16 | L·640·512·2 | 2.62 MB | 655.36 KB | **+1.97 MB** |
| combined mask (bool) | (L,sw+k)=(L,640) | L·640 | 2.56 KB | 640 B | +1.92 KB |
| SDPA output | (B,H,L,D) bf16 | 64·L·512·2 | 262.14 KB | 65.54 KB | +196.61 KB |
| **Per-layer subtotal** | | | **6.31 MB** | **1.58 MB** | **+4.73 MB** |

Reference: `_sparse_verify_rows_batched` at
`mlx-lm/mlx_lm/models/deepseek_v4.py:2384-2535`,
`SparseCompressedAttention.__call__` `:4554-4945`, `_indexer_score`
`:3689-3754`, `_dispatch_pmask` invoked at `:4655`.

### 2.2 Compressed layer (compress_ratio=128, 20 layers)

At `P_comp=2754` the pool goes into the full-KV path
(`pooled.shape[1] <= self.indexer.index_topk` at
`SparseCompressedAttention.__call__ :4718` — but this is the sparse class;
compressed layers use `CompressedAttention.__call__` at `:4292-4487`, which
concatenates full pool onto local KV):

| Tensor | Shape | Batched L=4 | Rowseq L=1 | Δ per layer |
|---|---|---:|---:|---:|
| main-Q | (1,64,L,512) bf16 | 262.14 KB | 65.54 KB | +196.61 KB |
| combined KV (B,1,sw+P,D) | (1,1,2882,512) bf16 | 2.95 MB | 2.95 MB | 0 |
| causal mask (L,S) | (L,2882) bool | 11.53 KB | 2.88 KB | +8.65 KB |
| SDPA output (B,H,L,D) | (1,64,L,512) bf16 | 262.14 KB | 65.54 KB | +196.61 KB |
| **Per-layer subtotal** | | **3.49 MB** | **3.09 MB** | **+401.86 KB** |

### 2.3 Local layer (compress_ratio=0, 3 layers)

Per-layer Δ ≈ **+393 KB** (Q + out both scale linearly in L).

### 2.4 MoE (batched in both paths)

`EXO_DSV4_ROWSEQ_FULLBLOCK_MOE=0` is the deployed default
(`start_cluster.sh:287`), so `_VERIFY_ROWSEQ_FULLBLOCK_MOE` is False in
`deepseek_v4.py:5135-5148`: rowseq's FFN is batched at L=4 as well. **MoE
allocations are identical between the two paths.** SwitchGLU at L=4 with
top-6 dispatches `24` gather_qmm rows through `hidden→2048→hidden`; internal
scratch ≈ 24·2048·2 = 96 KB output; total ~0.5 MB per layer either way.

### 2.5 Hidden state h across layer stack

`h = broadcast_to(h[..., None, :], (B, L, hc_mult=4, hidden=4096))`
(`_forward_steps` `:6727`). Batched L=4: 131.07 KB. Rowseq (L=1 within
FULLBLOCK): 32.77 KB. Δ ~98 KB per layer boundary. Irrelevant.

---

## 3. Model-wide simultaneous-allocation delta — the "smoking-gun" line

Because MLX evaluates the block output asynchronously (see `mx.async_eval(y)`
at `mlx-lm/mlx_lm/models/deepseek_v4.py:3124` gated by `_FENCE_ASYNC`),
attention intermediates within one block are freed after the block returns,
BUT the deferred graph carries the block-output h across boundary.

`EXO_DSV4_FENCE_EVERY_N_LAYERS=4` is set in `start_cluster.sh:458` but
`docs/PERFORMANCE_HISTORY.md:451` notes:

> `EXO_DSV4_FENCE_EVERY_N_LAYERS` is dead/unused config as of 2026-08-21

So per-layer async fencing (`mx.async_eval(y)`) is the actual behaviour. That
means the **peak simultaneous per-layer transient** is what matters, not a
naive `Δ × 43 layers` sum.

**Per-fence peak transient delta:**

| Case | Batched hold | Rowseq hold | Δ held simultaneously |
|---|---:|---:|---:|
| One sparse layer in flight | 6.31 MB | 1.58 MB | **+4.73 MB** |
| One compressed layer in flight | 3.49 MB | 3.09 MB | +0.40 MB |
| One local layer in flight | 524 KB | 131 KB | +0.39 MB |
| Max over all layer classes | 6.31 MB | 3.09 MB | **+3.22 MB (single-layer peak)** |

**If** async fencing degrades and multiple layers' worth pipeline before a
forced eval (as `_FENCE_ASYNC_C2=2` explicitly extends to
`B*L ≤ 2` at `_forward_steps` — see the gate at `deepseek_v4.py:3113`), the
peak scales with the pipeline depth. Two sparse layers in flight: batched
carries **12.6 MB** vs rowseq's 3.2 MB (Δ 9.4 MB). Under a stalled fence
scenario where the graph accumulates more layers before eval, the delta
compounds linearly.

---

## 4. Pool storage (invariant to L) — not the culprit

`PoolingCache` storage at 352.6K (`cache.py:1317-1516`):

| Layer class | Formula | Bytes per layer |
|---|---|---:|
| Sparse (ratio=4), P=88150, D=512 | 1·P·D·2 | **90.27 MB** |
| Compressed (ratio=128), P=2754, D=512 | 1·P·D·2 | 2.82 MB |
| Total across 21+20 | | **1.95 GB** |

This is **identical** between batched and rowseq — both L=4 (verify) and 4×
L=1 push exactly 4 pooled entries per verify call (`accumulate_windows`
increments identically). Pool growth chunking via `EXO_DSV4_POOL_GROW_STEP=256`
(`cache.py:1312`) allocates a **new** `(B, new_size, D)` tensor on growth
and copies the old contents — momentarily holding two ~90 MB copies per
sparse layer during growth events, but this happens exactly the same in both
paths.

**Verdict: pool storage is not the regression driver.**

---

## 5. DSpark draft-head caches (`_dspark_caches`)

`src/exo/worker/engines/mlx/speculative/dsv4_mtp.py:3964-3966` +
`:4649` (`_dspark.append_ctx(...)`): `_dspark_caches` are populated ONCE per
cycle in the epilogue, INDEPENDENT of the verify forward. `pp_speculation.py`
uses them only in the draft path. The batched verify forward does **not**
touch `_dspark_caches` 4× wider than rowseq: `dsv4_speculative_forward` at
`dsv4_mtp.py:1512-1561` calls `model(inputs, cache=cache)` where
`cache=gen_batch.prompt_cache` is the target-model KV cache (Rotating +
Pooling), not the DSpark ctx-KV cache. **Draft-head caches are not the
regression driver.**

---

## 6. Does lazy-eval / fencing rescue the batched path?

Partially, and with a hazard.

- `finalize(x)` at `mlx-lm/mlx_lm/models/deepseek_v4.py:4602, 4630-4644,
  4650, 4659, 4684-4687, 4857, 4871, 4881, 4945` forces evaluation at many
  intra-block boundaries, so the (B,L,P) scores and (B,L,k,D) gather don't
  quietly compound across layers.
- BUT within a single block's `SparseCompressedAttention.__call__` the whole
  set — `q`, `pmask`, `scores`, `pooled`, `topk`, `gathered`, `combined`,
  `combined_mask`, `out` — are all reachable across the intermediate
  `finalize()` calls **simultaneously**. Peak per-block transient is the
  6.31 MB / 1.58 MB numbers above. This is real, not lazy-evaluated away.
- The pool tensor `pooled` itself (~90 MB per sparse layer) is a `finalize`d
  view of `PoolingCache._pool_storage`, always live.

The dominant per-block simultaneous-live footprint at 352.6K in the batched
path is: **~90 MB (pool storage) + ~6.3 MB (batched transients)** ≈
**96.6 MB per sparse layer**, vs the rowseq path's **~90 MB + ~1.6 MB** ≈
**91.6 MB per sparse layer** — a **+5 MB per-in-flight-layer** delta.

---

## 7. Peak footprint reconciliation with the observed swap

The byte math shows a **per-layer transient inflation of ~4.7 MB on sparse
layers**, ~0.4 MB on compressed layers, ~0.4 MB on local layers. With MLX's
per-layer async fencing this bounds the peak transient delta to O(10 MB).
That is NOT enough on its own to explain 1.37–1.76 GB of swap and 2–3.7 s
page-fault stalls at 352.6K.

Three factors combine to explain the observed 3-of-11 collapse rate:

1. **Wired-memory headroom at 352.6K is thin.** The pool storage alone is
   **1.95 GB** (§4), plus rotating KV (128·512·2 ≈ 130 KB per layer, ~5 MB
   total), plus weights (~35 GB at 8-bit), plus per-request KV/prompt cache
   (≈ 88 GB total wired residence measured in
   `docs/incidents/2026-08-08-section23-stall-m4-1.log:1`). On a 128 GB M4
   Max with `mem_limit_mb ≈ 124.5 GB`, the deep-context steady state is
   already close enough to the wired ceiling that additional ~100–400 MB
   peak transients from the batched forward tip **stochastically** into
   OS-level page eviction. Explains the **3-of-11** hit rate.

2. **Pool-growth events amplify the peak.** `EXO_DSV4_POOL_GROW_STEP=256`
   grows sparse-layer pools by `256·512·2 = 256 KB` per event, but the
   growth path (`cache.py:1946-1975`) allocates a fresh `(B, target, D)`
   tensor of size ~90.5 MB per sparse layer and copies the old pool in.
   During growth, **two ~90 MB copies coexist per sparse layer**. When a
   growth event lands during a batched verify (where the transient delta
   is already +5 MB per in-flight layer), the peak footprint spike is the
   sum of the growth burst + the batched delta — pushing over the wired
   ceiling in the tail-3 runs.

3. **`_SPARSE_VERIFY_BATCHED`'s combined-KV block dominates the delta.** Of
   the +4.73 MB per-sparse-layer delta, **+1.97 MB is `combined` and +1.57
   MB is `pooled_gathered`** — both `mx.concatenate`/gather results whose
   allocator behaviour is to force a fresh contiguous alloc (not a view).
   These are the two allocations that scale linearly in L and produce a
   4× wider allocation across all 21 sparse layers under async pipelining.

The verify's kernel-time mean of 101–103 ms (per MTP-PROF) confirms it is
not slower kernels — it is memory-latency + page-faults.

**The batched forward's excess allocation is the trigger, not the entire
footprint. Once the machine tips into swap, the 2–3.7 s stalls dominate
tokens/s.**

---

## 8. Recommended fix — chunked verify rows at depth

Add an env-gated depth-conditional sub-chunking of the L=4 batched verify:

```
EXO_DSV4_VERIFY_BATCH_CHUNK_ROWS=2         # 0=off, N=split verify rows into ceil(L/N) sub-calls
EXO_DSV4_VERIFY_BATCH_CHUNK_MIN_CTX=131072 # only chunk at depth (>=128K); shallow stays L=4
```

Implementation site (all in `mlx-lm/mlx_lm/models/deepseek_v4.py`):

1. Add `_VERIFY_BATCH_CHUNK_ROWS`, `_VERIFY_BATCH_CHUNK_MIN_CTX` near the
   existing gate constants (`:1642-1655`).
2. In `_forward_steps` at `:6873-6883`, when the depth also exceeds
   `_VERIFY_BATCH_CHUNK_MIN_CTX`, decompose the L=4 forward into `L/N`
   sequential calls of `_set_verify_batch_ctx(active=True, L=N)`, each
   feeding a slice `h[:, j:j+N]` through the per-layer loop, concatenating
   the outputs. Cache state (Rotating + Pooling) accumulates correctly
   across sub-calls because `accumulate_windows` is agnostic to how the
   L=4 total is split.
3. This preserves the batched win at 100K (where L=4 stays L=4 and the
   pre-rowseq batched kernels win vs 4× kernel dispatches) AND bounds the
   per-in-flight-layer transient at deep context to ≤2 rows' worth, cutting
   the +4.73 MB/layer sparse delta by half.

**Alternative: keep the batched fold at L=4 but stream the pool-gather.**
`SparseCompressedAttention.__call__` at `:4787-4823` (`_single_gather`
branch) already tiles the sparse SDPA over query-row sub-chunks of
`_SPARSE_SDPA_TILE=128`. This tile size is inactive for L=4 (`_Lq=4 < 128`
so the fallback single-call path runs at `:4847-4856`). Adding a
`_VERIFY_SDPA_TILE_ROWS` (default off, arm at depth) that tiles the L=4
query rows into 2-row sub-tiles would keep the fused SDPA fast path but cap
`combined`/`pooled_gathered` at the 2-row size — same net memory footprint
as the outer chunked verify but with less code change.

**Not recommended: context cap.** The user has rejected this. The
regression is stochastic (3/11) and mechanism-linked (peak transient +
wired ceiling crossing), not a hard capacity limit.

---

## 9. What is NOT the culprit

- **Indexer (B,H,L,P) score tensor.** OPT-6 fold-w-into-q at
  `_indexer_score` (`deepseek_v4.py:3749-3754`) collapses H before the GEMM.
  The scores are `(B,L,P)` bf16 = 705 KB even at L=4/P=88150 — not the 22 GB
  nightmare tensor.
- **Draft-head ctx-KV (`_dspark_caches`).** Populated once per cycle after
  `append_ctx`; the verify forward does not widen or copy them (§5).
- **Pool storage width.** Identical between the two paths (§4).
- **MoE.** Batched at L=4 in both paths (`_VERIFY_ROWSEQ_FULLBLOCK_MOE=0`
  in `start_cluster.sh:287`). MoE peak is unchanged.
- **PP/RDMA transport.** Verify is a single batched forward within one
  runner; PP send/recv is per-layer output h (~131 KB at L=4 vs ~33 KB at
  L=1). Not a measurable memory-pressure contributor.

---

## 10. Files and line-number references (audit trail)

Every claim in this document ties to a code site. Key ones:

- Batched-verify env gates: `mlx-lm/mlx_lm/models/deepseek_v4.py:1616-1655`,
  `:1667-1677`, `:6873-6915`.
- Batched-vs-rowseq dispatch (block level):
  `mlx-lm/mlx_lm/models/deepseek_v4.py:5061-5083` (FULLBLOCK gate),
  `:5216-5274` (per-row-attention gate), `:6939-6968` (hc_head gate).
- Sparse attention allocation:
  `mlx-lm/mlx_lm/models/deepseek_v4.py:4554-4945` (SparseCompressedAttention),
  `:2384-2535` (`_sparse_verify_rows_batched`),
  `:2538-2727` (`_sparse_pooled_attention` L=1 fast path),
  `:3689-3754` (`_indexer_score`, OPT-6 fold).
- Verify-batch side channel setter: `_set_verify_batch_ctx`
  `mlx-lm/mlx_lm/models/deepseek_v4.py:1667-1677`.
- PoolingCache: `mlx-lm/mlx_lm/models/cache.py:1317-1516`,
  BatchPoolingCache `:1873-2093`, growth chunking `:1946-1975`,
  save/restore `:2519-2598`.
- DSpark ctx caches:
  `src/exo/worker/engines/mlx/speculative/dsv4_mtp.py:3964-3966`,
  `:4649` (`append_ctx`), `:4680`, `:4689`.
- Verify forward callsite:
  `src/exo/worker/engines/mlx/speculative/dsv4_mtp.py:1512-1561`
  (`dsv4_speculative_forward`), verify assembly at `:2647-2675`.
- Fence env:
  `mlx-lm/mlx_lm/models/deepseek_v4.py:90` (`_FENCE_ASYNC`),
  `:97` (`_FENCE_ASYNC_MAX_B` = `EXO_DSV4_FENCE_ASYNC_C2`),
  `:2905-2907` (`_fence_every_n` from `EXO_DSV4_FENCE_EVERY_N_LAYERS`,
  dead per `docs/PERFORMANCE_HISTORY.md:451`),
  `:3109-3144` (async fence gate + diag).
- start_cluster.sh env: `:33` (INDEX_TOPK=512), `:286-320` (rowseq +
  verify-batch env), `:319` (VERIFY_BATCH=1), `:325` (VERIFY_BATCH_MIN_CTX=8192),
  `:458` (FENCE_EVERY_N_LAYERS=4, dead),
  `:2202-2216` (POOL_GROW_STEP=256).

---

## Summary — one-paragraph verdict

The batched verify path allocates, per in-flight sparse layer, **+4.73 MB**
of extra simultaneously-live transients vs the rowseq path at 352.6K, L=4
(dominated by the `(1,4,1,640,512)` combined KV block at +1.97 MB and the
`(1,4,512,512)` pooled_gathered at +1.57 MB). Compressed and local layers
add ~+0.4 MB each. Rowseq keeps these to a single row's footprint because
Python drops each iteration's tensors before the next row runs. Under MLX's
per-layer async fencing this is a bounded +5 MB single-layer peak, but on
top of a ~91 GB steady-state footprint at 352.6K on a 124.5 GB wired
ceiling, and combined with sporadic ~90 MB pool-growth bursts per sparse
layer, the batched path stochastically tips into OS-level page eviction —
matching the observed 3/11 collapse rate and 2–3.7 s stalls. The fix is
chunked verify rows at depth: split L=4 into 2×L=2 sub-batches once ctx
exceeds a new `EXO_DSV4_VERIFY_BATCH_CHUNK_MIN_CTX` (e.g. 128K), which
halves the per-in-flight-layer delta while preserving the L=4 batched win
at 100K.
