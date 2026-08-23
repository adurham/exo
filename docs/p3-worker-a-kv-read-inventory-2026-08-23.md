# P3 Worker A — per-decode-step attention memory-read inventory vs context depth L, and TP all_sum L-dependence audit (2026-08-23)

**Scope**: CODE-DERIVED ONLY. Every number below traces to a file:line in
`~/repos/exo` (incl. the `mlx-lm` submodule, branch
`known-good-decode-fenceasync-20260822`, HEAD `1fea494`) or to a read-only
`ssh cat` of a config file on `adams-mac-studio-m4-1.local`. **Nothing here is a
measurement.** A parallel worker is measuring kernel wall time; where this doc
predicts ms/token it is labelled **PREDICTION** and must not be cited as
measured. No cluster state was modified (read-only `find`/`cat`/`python3 -c`
on the index JSON only).

---

## 0. Confirmed dimensions (raw evidence)

`ssh adams-mac-studio-m4-1.local 'cat ~/.exo/models/deepseek-ai--DeepSeek-V4-Flash-0731/config.json'`:

```
  "head_dim": 512,
  "hidden_size": 4096,
  "index_head_dim": 128,
  "index_n_heads": 64,
  "index_topk": 512,
  "max_position_embeddings": 1048576,
  "num_attention_heads": 64,
  "num_hidden_layers": 43,
  "num_key_value_heads": 1,
  "o_groups": 8,
  "o_lora_rank": 1024,
  "q_lora_rank": 1024,
  "qk_rope_head_dim": 64,
  "quantization_config": {
    "activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8",
    "scale_fmt": "ue8m0", "weight_block_size": [128, 128] },
  "sliding_window": 128,
  "compress_ratios": [0,0,4,128,4,128,...,4,0,0,0]
```

`compress_ratios` truncated to `num_hidden_layers=43` by
`mlx-lm/mlx_lm/models/deepseek_v4.py:888` (`self.compress_ratios =
list(self.compress_ratios[: self.num_hidden_layers])`). Layer-class census of
the first 43 entries:

| ratio | count | attention class | source |
|---|---|---|---|
| 0 | 2 (layers 0,1) | `LocalAttention` | `deepseek_v4.py:4846-4849` |
| 4 | 21 | `SparseCompressedAttention` | `deepseek_v4.py:4851` (fallthrough) |
| 128 | 20 | `CompressedAttention` | `deepseek_v4.py:4850` |

`v4_attention_factory` (`deepseek_v4.py:4844-4851`):
```python
    ratio = config.compress_ratios[layer_idx]
    if ratio == 0:   return LocalAttention(config, layer_idx)
    if ratio == 128: return CompressedAttention(config, layer_idx)
    return SparseCompressedAttention(config, layer_idx)
```

**index_topk is 512 in production**, not overridden downward:
`start_cluster.sh:33` `: "${EXO_DSV4_INDEX_TOPK:=512}"`, exported at
`start_cluster.sh:1681`. Read by `deepseek_v4.py:3757-3759`
(`self.index_topk = int(_topk_env) if _topk_env else config.index_topk`).

---

## 1. Depth-dependent tensors read per decode step, per layer

### (d) FIRST — the KV cache dtype ACTUALLY used at runtime: **bf16, unquantized**

Three independent code paths agree:

1. **Quantized KV is off in production.** `start_cluster.sh:151`
   `: "${EXO_KV_CACHE_BITS:=0}"`, with the comment at `start_cluster.sh:140-147`:
   *"Default 0 (bf16 KV, no quantization) — chosen 2026-05-09 for QUALITY
   safety… prod deployments must use bf16 KV"*. Exported at
   `start_cluster.sh:1584`. `EXO_TURBOQUANT` is unset (`start_cluster.sh:138`),
   so the TurboQuant branch (`src/exo/worker/engines/mlx/cache.py:2420`) is dead.
2. **`0` is a real "disabled" sentinel**, not `bits=0`:
   `src/exo/worker/engines/mlx/cache.py:2393` — `return KV_CACHE_BITS or None`,
   with the explicit guard comment at `cache.py:2388-2392`. So
   `make_kv_cache` skips the `QuantizedKVCache` replacement branch
   (`cache.py:2443`) entirely and returns `model.make_cache()` verbatim.
3. **Model-side casts pin bf16.** `deepseek_v4.py:4077-4078`, `:4240-4241`,
   `:4543-4544`: `if _FP32_ACT and kv.dtype == mx.float32: kv =
   kv.astype(mx.bfloat16)  # keep KV cache bf16 (batch-invariant)`. Pool side
   the same at `deepseek_v4.py:3238-3239`.

**Verdict: 2 bytes/element for every cache tensor in the attention path.**
`EXO_COMPUTE_DTYPE:=bf16` (`start_cluster.sh:153`) is consistent.

### Cache topology per layer — `deepseek_v4.py:6956-6979` (`Model.make_cache`)

```python
            ratio = layer.attn.compress_ratio
            if ratio == 0:
                caches.append(RotatingKVCache(max_size=self.args.sliding_window))
            elif isinstance(layer.attn, SparseCompressedAttention):
                caches.append(CacheList(
                    RotatingKVCache(max_size=self.args.sliding_window),
                    PoolingCache(ratio),   # compressor pool, dim=head_dim=512
                    PoolingCache(ratio),   # indexer pool,    dim=index_head_dim=128
                ))
            else:
                caches.append(CacheList(
                    RotatingKVCache(max_size=self.args.sliding_window),
                    PoolingCache(ratio)))
```

### (a) Core-attention MLA compressed-KV read — **does core attention read ALL L?**

**Answer: NO for the 21 sparse layers (constant, top-k=512 rows). YES for the
20 compressed layers (full pool, but the pool is L/128 entries so still small).**

- **`LocalAttention` (2 layers, ratio 0)** — reads only the rotating window.
  `RotatingKVCache(max_size=sliding_window=128)` caps physical length at 128
  (`mlx-lm/mlx_lm/models/cache.py:642-694`, `_update_in_place`: grow is clamped
  by `grow_room = self.max_size - prev`, then trim + rotate). SDPA at
  `deepseek_v4.py:4092-4098` passes `kv, kv` (K and V are the SAME tensor — MLA
  absorbed form, `num_key_value_heads=1`), so one 512-wide row per cached
  position, **128 positions max, constant in L**.

- **`CompressedAttention` (20 layers, ratio 128)** — reads its ENTIRE pool.
  `deepseek_v4.py:4245-4250`:
  ```python
                if pooled.shape[1] > 0:
                    pooled_mask = _dispatch_pmask(pool_cache, L, offset)
                    kv = mx.concatenate([kv, pooled[:, None]], axis=2)
  ```
  then one dense `scaled_dot_product_attention(q, kv, kv, ...)` at
  `deepseek_v4.py:4288-4312`. Pool length P = L/128
  (`PoolingCache.accumulate_windows` emits one pooled entry per `ratio`
  positions, `cache.py:1452-1466`). So the read is **L/128 × 512 × bf16 —
  linear in L, but with a 1/128 divisor.**

- **`SparseCompressedAttention` (21 layers, ratio 4)** — reads only a **top-k =
  512 selected subset**, NOT all L. `deepseek_v4.py:4785-4795` dispatches
  `_sparse_pooled_attention`; the L_q==1 decode fast path
  (`deepseek_v4.py:2527-2549`) gathers exactly `k_dim = topk.shape[2]` rows:
  ```python
        P_dim = pooled.shape[1]
        k_dim = topk.shape[2]
        with span("attn.gather"):
            pooled_flat = pooled.reshape(B * P_dim, D)
            offset = (mx.arange(B) * P_dim).reshape(B, 1, 1)
            topk_flat = (topk + offset).reshape(-1)
            pooled_kv = pooled_flat[topk_flat].reshape(B, L, k_dim, D)
        ...
        combined_kv = mx.concatenate([local_kv, pooled_kv], axis=2)
  ```
  with the OPT-10 comment at `deepseek_v4.py:2530-2536` stating explicitly:
  *"touches only k entries per query, O(B\*L\*k\*D) … and does NOT scale with
  P."* `k = min(self.index_topk, pooled.shape[1])` (`deepseek_v4.py:3885`) =
  **512** once P ≥ 512, i.e. once L ≥ 2048. **Constant in L.** Local window
  contribution is the same 128-row rotating cache. Combined KV width per sparse
  layer = 128 + 512 = **640 rows × 512 × bf16, depth-independent.**

  (The dispatch guard at `deepseek_v4.py:4448-4449` — `elif pooled.shape[1] <=
  self.indexer.index_topk:` takes the dense-concat branch — only fires below
  L≈2048; at 100K+ we are always on the sparse branch.)

### (b) Indexer / scorer reads — **YES, scans ALL L (as L/4 pooled entries), every step, every sparse layer**

`Indexer.__call__` (`deepseek_v4.py:3782-3982`):
```python
        pooled = self.compressor(x, pool_cache, offset)          # 3800
        ...
        q = self.wq_b(q_residual).reshape(B, L, self.n_heads, self.head_dim)  # 3818
        with span("indexer.score"):
            scores = _indexer_score(q, pooled, self.weights_proj(x),
                                    self.scale, self.n_heads**-0.5)          # 3833-3839
```
`_INDEXER_PBLOCK` defaults 0 (`deepseek_v4.py:327`) and is unset in
`start_cluster.sh:1958` (conditional export only), so the untiled `_indexer_score`
runs.

**Per-position bytes: 128 × 2 = 256 B per pooled indexer entry, and the pool has
L/4 entries.** Crucially the 64 index heads are FOLDED INTO THE QUERY before the
GEMM, so the pooled tensor is read ONCE, not 64 times — `deepseek_v4.py:3676-3680`:
```python
    w = (mx.sigmoid(weights_x) * (scale * n_heads_inv_sqrt))  # (B, L, H)
    q_blhd = q.transpose(0, 2, 1, 3)                          # (B, L, H, D)
    q_weighted = (w[..., None] * q_blhd).sum(axis=2)          # (B, L, D)
    return q_weighted @ pooled.swapaxes(-1, -2)               # (B, L, P)
```
with the OPT-6 note at `deepseek_v4.py:3667-3675` ("collapsing 64 heads to 1 …
the (B,H,L,P) transient is never materialized"). This is the single largest
depth-dependent term in the whole inventory.

The top-k selection then re-reads the `(1,1,P)` bf16 scores array. Production
takes the **exact fused top-k** path (`EXO_DSV4_EXACT_TOPK` default `"1"`,
`deepseek_v4.py:3426`; decode has `scores.shape[1]==1 <= 16`,
`deepseek_v4.py:3903-3910`). Its Metal kernel makes **4 strided full-P passes**
over `scores` — histogram pass (`deepseek_v4.py:3455`), refine pass
(`:3480`), count pass (`:3510`), emit pass (`:3545`), each
`for (... i < P; ...) { ushort key = dsv4_topk_key(scores[row + i]); ... }`.
(`EXO_DSV4_TOPK_FUSED` defaults `"0"` at `deepseek_v4.py:3893` and is only
conditionally exported at `start_cluster.sh:1680`, so the approximate kernel is
off.)

### (c) Auxiliary pooled/compressed caches

- **Compressor pool (all 41 non-local layers)** — `PoolingCache(ratio)`. Written
  once per `ratio` decode steps; the write is `update_and_fetch_deferred`
  (`deepseek_v4.py:3260`, decode-only branch guarded by `_prefill_L > 1` at
  `:3253`). **Storage grows in `step = 256`-entry chunks** (`cache.py:1290`),
  and each growth copies the old prefix (`cache.py:1517-1528`). Amortized that
  is a `2/256`-of-pool read+write per pooled entry — a genuine, if small,
  depth-linear term.
- **Indexer pool (21 sparse layers)** — a second `PoolingCache(ratio=4)` of width
  `index_head_dim=128`, same mechanics.
- **`_pool_storage` donation path**: at pools above
  `EXO_DSV4_POOL_DEFER_COPY_MAX_BYTES = 32 MiB` (`cache.py:1265-1267`), the
  deferred write donates the buffer instead of holding a pre-write view
  (`cache.py:1546-1556`), so the O(P·D) copy is avoided at large L. Good: the
  amortized realloc term is the only pool-copy cost left at depth.

---

## 2. Bytes-read inventory and the `bytes_per_rank(L)` formula

### TP=2 sharding note (this materially changes the "per-rank" answer)

**Attention is REPLICATED on both ranks; ONLY the MoE is sharded.** exo does
*not* call mlx-lm's `Model.shard()` (which would do `attn.n_heads //= N`,
`deepseek_v4.py:7170`) — grep for `.shard(` under `src/exo/` returns exactly one
unrelated hit (`src/exo/shared/types/worker/instances.py:88`). exo's TP path is
`tensor_auto_parallel` (`src/exo/worker/engines/mlx/utils_mlx.py:444`) →
`DeepseekV4ShardingStrategy.shard_model`
(`src/exo/worker/engines/mlx/auto_parallel.py:1049-1133`), whose docstring says
so plainly (`auto_parallel.py:1032-1034`):

> *"Sharding for DeepSeek V4 Flash / Pro — MoE-only. Replicates attention on
> every rank; shards only the MoE block."*

Inside that loop the ONLY attention mutations are the optional wq_a+wkv fusion
(`auto_parallel.py:1077-1085`, `EXO_DSV4_QA_KV_FUSED` default off) and setting
`layer.attn.sharding_group` (`auto_parallel.py:1099-1103`). No `n_heads //= N`,
no `wq_b`/`wo_a` split. **Therefore per-rank attention bytes == single-node
attention bytes; TP=2 does NOT halve any figure in this table.**
(`docs/dsv4-attention-kernel-efficiency-2026-08-18.md:38,52-55` assumes the
mlx-lm `shard()` head-halving — that assumption does not hold on exo's own TP
path. Flagged for the reviewer; it does not change any *depth-scaling* verdict,
only the absolute constant.)

### Component table

`D = head_dim = 512`, `SW = sliding_window = 128`, `K = index_topk = 512`,
`ID = index_head_dim = 128`, bf16 = 2 B. Layers: 2×r0, 21×r4, 20×r128.

**CONSTANT terms C (bytes/decode-step/rank):**

| id | component | expression | bytes | law |
|---|---|---|---|---|
| A1 | local rotating-KV read (all 43 layers) | `43·SW·D·2` | 5,636,096 | **constant** |
| A2 | sparse top-k gathered pool rows | `21·K·D·2` | 11,010,048 | **constant** |
| A3 | sparse `combined_kv` materialization (write + SDPA re-read) | `21·(SW+K)·D·2·2` | 27,525,120 | **constant** |
| A4 | attention weights (mxfp8 1.03125 B/p; affine-int8 1.0625 B/p) | see below | 5,253,382,144 | **constant** |
| | **C total** | | **5,297,553,408** (5.298 GB) | |

A4 breakdown per layer: `wq_a`(4096×1024) + `wq_b`(1024×32768) +
`wkv`(4096×512) + `wo_a`(8×4096×1024) + `wo_b`(8×1024×4096) at mxfp8; plus, on
non-local layers, `compressor.wkv/wgate`(4096×out_dim, out_dim=1024 for r=4 via
`Compressor.out_dim` `deepseek_v4.py:3094`, 512 for r=128) at affine-int8; plus,
on r=4 layers, `indexer.wq_b`(1024×8192) at mxfp8 and
`indexer.weights_proj`(4096×64) + `indexer.compressor.wkv/wgate`(4096×256) at
affine-int8. Quantization assignment from `make_quantization_config`
(`deepseek_v4.py:899-931`): `.attn.w*` and `.attn.indexer.wq` → mxfp8
(group_size 32, 8 bits → 1+1/32 B/param); default `{"group_size":64,"bits":8,
"mode":"affine"}` → 1+4/64 B/param. Tensor inventory cross-checked against the
real checkpoint index (`ssh … python3 -c 'json.load(model.safetensors.index.json)'`
for `layers.4.attn.*` [r=4] and `layers.5.attn.*` [r=128]) — the r=4 layer has
`indexer.wq_b`, `indexer.weights_proj`, `indexer.compressor.{wkv,wgate}`; the
r=128 layer has none of them. Matches.

**DEPTH-LINEAR terms m (bytes per context token):**

| id | component | expression | B/token | law |
|---|---|---|---|---|
| B1 | **indexer full-pool scan** (21 sparse layers) | `21·(1/4)·ID·2` | **1344.00** | **linear in L** |
| B2 | indexer scores write | `21·(1/4)·2` | 10.50 | linear |
| B3 | exact-topk kernel 4 strided passes over scores | `21·4·(1/4)·2` | 42.00 | linear |
| B4 | compressed-layer full pool read + concat write + SDPA re-read | `20·3·(1/128)·D·2` | 480.00 | linear |
| B5 | pool step-realloc amortized copy, r=4 compressor | `21·2·(1/4)·D·2/256` | 42.00 | linear (amortized; bursty) |
| B6 | pool step-realloc amortized copy, r=4 indexer | `21·2·(1/4)·ID·2/256` | 10.50 | linear (amortized) |
| B7 | pool step-realloc amortized copy, r=128 | `20·2·(1/128)·D·2/256` | 1.25 | linear (amortized) |
| | **m total** | | **1930.25** | |

### The formula

```
bytes_per_rank(L) = 5,297,553,408 + 1930.25 · L      [bytes, attention path only]
```

C is *per-token constant read traffic* (weights + fixed-width caches); it is
depth-independent by construction and therefore **cannot contribute to the
100K→500K decay**. It is included only so the depth term can be read against a
realistic denominator.

### Evaluated

| L | total | of which constant | of which depth-dependent |
|---|---|---|---|
| 100,000 | **5.4906 GB** | 5.2976 GB | 0.1930 GB |
| 352,595 | **5.9781 GB** | 5.2976 GB | 0.6806 GB |
| 500,000 | **6.2627 GB** | 5.2976 GB | 0.9651 GB |

Deltas: **100K→352.6K = +487.6 MB/token**; 352.6K→500K = +284.5 MB/token;
100K→500K = +772.1 MB/token.

### Scaling law per component — summary verdict

| component | law |
|---|---|
| Core attention, LocalAttention (2 layers) | **constant** (rotating window capped at 128) |
| Core attention, SparseCompressedAttention (21 layers) | **constant** (top-k=512 gathered rows; independent of L) |
| Core attention, CompressedAttention (20 layers) | **linear in L, slope L/128** (reads its whole pool) |
| **Indexer score GEMM (21 layers)** | **linear in L, slope L/4** — 70% of all depth-dependent bytes |
| Exact-topk kernel score passes | linear in L (4 passes × L/4 × 2 B × 21) |
| Pool realloc copies | linear in L, amortized (bursty every 256 pooled entries) |
| Attention weights | **constant** |
| MLA local KV cache | **constant** |

---

## 3. all_sum audit — TP collective call sites active during DECODE

Grep of every `mx.distributed.all_sum` / `all_gather` in the model file:

```
$ grep -n "mx.distributed.all_sum\|mx.distributed.all_gather" mlx_lm/models/deepseek_v4.py
507:_orig_all_sum = mx.distributed.all_sum                 (wrapper capture, not a call site)
508:_orig_all_gather = mx.distributed.all_gather           (wrapper capture)
536:    mx.distributed.all_sum = _all_sum_nop_aware        (monkey-patch install)
3007:                    y = mx.distributed.all_sum(y, group=self.sharding_group)
4117:                        mx.distributed.all_sum(out, group=self.sharding_group)
4360:                            mx.distributed.all_sum(_full, group=self.sharding_group)
4369:                        _g = mx.distributed.all_gather(out, group=_sg)
4818:                            mx.distributed.all_sum(_full, group=self.sharding_group)
4827:                        _g = mx.distributed.all_gather(out, group=_sg)
4837:                        mx.distributed.all_sum(out, group=self.sharding_group)
5416:        out = mx.distributed.all_sum(out, group=_sg)
6758:                h = finalize(mx.distributed.all_gather(h)[: h.shape[0]])
```

Per site, at **single-token decode (B=1, L=1)**:

| line | site | active in TP decode? | tensor shape | any dim ∝ L? |
|---|---|---|---|---|
| **3007** | `DeepseekV4MoE.__call__`, `span("moe.all_sum")` | **YES — 43×/token** | `y.shape == x.shape == (B, L_q, hidden) = (1, 1, 4096)` | **NO** |
| 4117 | `LocalAttention` tail all_sum | **NO** | — | — |
| 4360 / 4369 | `CompressedAttention` seq-split gather | **NO** (decode L=1) | — | — |
| 4818 / 4827 | `SparseCompressedAttention` seq-split gather | **NO** (decode L=1) | — | — |
| 4837 | `SparseCompressedAttention` tail all_sum | **NO** | — | — |
| 5416 | `_rowsdpa_sharding_allsum` (MTP/DSpark verify rows) | **NO** (MTP off; TP has no DSpark loop) | `(1, L_rows, hidden)` | NO |
| 6758 | `model.all_gather` | **NO** (`pipeline_size > 1` guard; TP has pipeline_size==1) | — | — |

**Why the attention-tail all_sums are dead:** `_ATTN_ALLSUM =
os.environ.get("EXO_DSV4_ATTN_ALLSUM", "1") == "1"`
(`deepseek_v4.py:1626`) — but production sets it to **0**:
`start_cluster.sh:1755` `: "${EXO_DSV4_ATTN_ALLSUM:=0}"`, exported at
`start_cluster.sh:1756`. Rationale in the surrounding comment
(`start_cluster.sh:1746-1754`): *"the probe proved it an EXACT 2.000000 doubling
of bitwise rank-identical replicas … With it off: … one fewer network round trip
per compressed/sparse layer."* Both tail sites are gated
`if … and _ATTN_ALLSUM:` (`deepseek_v4.py:4113`, `:4834`).

**Why the seq-split gathers are dead at decode:** the gate requires
`L >= _SEQ_SPLIT_MIN_L` with `_SEQ_SPLIT_MIN_L = 16`
(`deepseek_v4.py:225`), and single-token decode has L=1
(`deepseek_v4.py:4478-4482`, `:4257-4263`).

**Why the MoE all_sum is the surviving one, and why 43/token:** exo sets
`layer.ffn.sharding_group = self.group` on every one of the 43 layers
(`auto_parallel.py:1087`) and shards the expert weights along the intermediate
axis (`auto_parallel.py:1104-1109`). The correction comment at
`auto_parallel.py:1141-1152` is decisive:

> *"`all_to_sharded` slices axis `max(ndim-2, 0)` … so axis 1, the INTERMEDIATE
> WIDTH. Both ranks hold ALL 256 experts at HALF width; experts are never
> partitioned by identity. **Consequently the MoE all_sum reduces a fixed
> (B, L, hidden) tensor every layer regardless of which experts fired**, and
> there is no token gather/scatter to expert-owning ranks."*

At decode `y` is the post-combine MoE output, shape `(1, 1, 4096)` bf16 =
**8192 bytes per call, 43 calls/token, 352 KB/token total collective payload —
identical at 100K, 352.6K and 500K.** Note the wrapper at
`deepseek_v4.py:531-536` (+ `_collective_fp32_safe` at `:514-525`) can only
change dtype, never shape.

### ▶ VERDICT (3): **all_sum tensor shapes are INDEPENDENT of context depth L.**

The single decode-active collective is `deepseek_v4.py:3007` at
`(B=1, L_q=1, hidden=4096)`. No dimension of it, and no dimension of any other
collective reachable in TP decode, is a function of L. Whatever the prior
~2.19 ms/token / 43-calls figure represents, **it cannot grow with context depth
by payload size.** If measured all_sum time *does* grow with L, the cause must
be arrival-time skew (one rank's attention takes longer at depth, so the
collective waits) — not the collective itself. That distinction is testable and
is exactly the mechanism already documented in
`docs/moe-all-sum-skew-vs-comms-2026-08-19.md`.

---

## 4. Sanity check — resident KV bytes vs the 128 GB node budget

Resident (not per-step-read) cache bytes at depth, per node (caches are
replicated, matching the replicated attention):

| L | local rotating (43×128×512×2) | r=4 compressor pools (21×L/4×512×2) | r=4 indexer pools (21×L/4×128×2) | r=128 pools (20×L/128×512×2) | **total** | B/token |
|---|---|---|---|---|---|---|
| 100,000 | 5.6 MB | 0.538 GB | 0.134 GB | 0.016 GB | **0.694 GB** | 6,936 |
| 352,595 | 5.6 MB | 1.896 GB | 0.474 GB | 0.056 GB | **2.431 GB** | 6,896 |
| 500,000 | 5.6 MB | 2.688 GB | 0.672 GB | 0.080 GB | **3.446 GB** | 6,891 |

**Plausibility: yes, comfortably.** ~83 GB of weights + 2.43 GB of KV at 352.6K
= ~85.4 GB on a 128 GB node, well under the ~124 GB wired limit referenced at
`start_cluster.sh:146`. At 500K it is ~86.4 GB. The **~6.9 KB/token** figure is
the right sanity anchor: it is ~74× smaller than a naive dense-MLA estimate
would give (43 layers × 512 dims × 2 B = 44 KB/token if every layer cached every
token at full width), and the ratio is exactly what the compress_ratios predict
(21 layers at 1/4 + 20 at 1/128 + 2 windowed). **The per-step figure integrates
correctly**: the depth-linear read slope 1930.25 B/token vs 6,891 B/token
resident means each decode step re-reads ~28% of the resident cache — consistent
with "the indexer scans the whole 128-wide indexer pool (1/4·L·256 B = 1344 of
the 1930) while the 512-wide compressor pool is only touched by a 512-row
gather".

---

## 5. PREDICTION (not measurement): ms/token implied by the byte deltas

Achieved-bandwidth inputs, from this repo:
- `docs/dsv4-attention-kernel-efficiency-2026-08-18.md:28` — session-measured
  **streaming memory bandwidth (read+write): 297 GB/s** (on a laptop M4 Max,
  32-core GPU; the Studio nodes are 40-core, so this is conservative).
- `docs/decode-roofline-dispatch-bound-2026-08-21.md:29` and
  `docs/roofline-recalculated-post-fix-2026-08-22.md:19` — **546 GB/s** M4 Max
  public spec peak.
- `docs/roofline-sanity-check-inputs-confirmed-2026-08-22.md:40-45` confirms
  decode is **plain autoregressive** (1 forward pass/token; DSpark/MTP dormant
  under TP), so per-step == per-token.

| achieved BW | 100K | 352.6K | 500K | **Δ 100K→352.6K** | Δ 352.6K→500K |
|---|---|---|---|---|---|
| 297 GB/s (repo-measured streaming) | 18.49 ms | 20.13 ms | 21.09 ms | **+1.64 ms** | +0.96 ms |
| 327.6 GB/s (60% of 546) | 16.76 | 18.25 | 19.12 | **+1.49 ms** | +0.87 ms |
| 409.5 GB/s (75% of 546) | 13.41 | 14.60 | 15.29 | **+1.19 ms** | +0.69 ms |
| 546 GB/s (spec peak) | 10.06 | 10.95 | 11.47 | +0.89 ms | +0.52 ms |

**Headline prediction: going 100K → 352.6K adds ≈ 1.2–1.6 ms/token of
attention-path memory-read time, and 352.6K → 500K adds a further ≈ 0.7–1.0 ms.
Total 100K → 500K ≈ 1.9–2.6 ms/token.** (Point estimate used in the summary:
**+1.64 ms** at the repo-measured 297 GB/s, the most defensible single number
because it is a measurement from this repo rather than a spec sheet.)

### What this does and does not rule out — read this before citing it

Baseline decode is ~18.3 tok/s ≈ 54.6 ms/token
(`docs/decode-roofline-dispatch-bound-2026-08-21.md:32-33`). A ~1.6 ms
attention-read delta is **≈3% of one token's wall time**.

- **It DOES rule out**: "the attention path re-reads a growing KV cache and that
  bandwidth is the decay." The sparse design works — core attention is
  genuinely O(1) in depth on 21 of 43 layers, and the only real depth term is
  the indexer's L/4×128 scan, which is small in bytes.
- **It DOES NOT rule out** (explicitly, per the perf-hypothesis-discipline rule
  that fit is not causation): (i) the indexer scan and the 4-pass top-k kernel
  may be **latency/occupancy-bound rather than bandwidth-bound** at large P — a
  strided 4-pass reduction over a 88K-element array at 500K is a very different
  kernel-efficiency regime from a 22K-element one at 100K, and bytes do not
  capture that; (ii) **dispatch count** is depth-invariant here but the
  already-documented dispatch-bound regime
  (`docs/decode-roofline-dispatch-bound-2026-08-21.md:37-52`) means small
  per-kernel slowdowns land on top of a 48 ms overhead floor; (iii) **all_sum
  arrival skew** growing with depth (see §3 — payload cannot grow, wait time
  can); (iv) **unified-memory pressure** — 3.4 GB extra resident at 500K on top
  of ~83 GB of weights can shift allocator/residency behaviour in ways no
  bytes-read model predicts; (v) pool-realloc **bursts** (B5–B7 are amortized
  here but land as a single ~0.85 ms copy event at 500K shapes when donation
  fails — see the note at `cache.py:1547-1553`).

**Correct framing for the P3 synthesis: this worker rules out ONE mechanism
(attention-path read bandwidth) as the cause of the 500K decay, and quantifies
its true size at ~1.6 ms of the gap. It does not rule out the attention path.**
Worker B's measured kernel wall times at 100K vs 500K are the discriminator: if
`attn.indexer` wall grows by materially more than ~1.1 ms across that range
(1344/1930 of the delta), the excess is kernel-efficiency, not bytes.

---

## 6. Assumptions a reviewer should attack first

1. **B=1.** All figures are single-stream decode. At B>1 the pools are
   `BatchPoolingCache` and every depth-linear term scales with B.
2. **Steady state, L ≥ 2048.** Below that, `pooled.shape[1] <= index_topk` and
   the sparse layers take the dense-concat branch (`deepseek_v4.py:4448`).
3. **A4 (weights) is an upper bound on *useful* traffic** — it assumes every
   attention weight is read once per token with no cache reuse. Real reuse on a
   40-core GPU's L2 will reduce it. This does not touch the depth term at all.
4. **A3/B4 concat-materialization terms assume MLX actually materializes the
   `mx.concatenate` output** rather than fusing it into SDPA. If MLX elides
   these, C drops by 27.5 MB and m drops from 1930.25 to 1610.25 B/token
   (−17%), moving the +1.64 ms prediction to +1.37 ms. Verdict unchanged.
5. **Amortized realloc terms (B5–B7, 53.75 B/token combined, 2.8% of m)** are
   modelled as smooth; in reality they are bursts every 256 pooled entries.
6. **The one figure I could not close from code alone**: whether MLX's SDPA
   re-reads `combined_kv` per head-group. K and V are the same array
   (`deepseek_v4.py:4094-4095` etc.), which is why I counted one read — but the
   fused-kernel comment at `deepseek_v4.py:1703-1705` notes per-head key
   re-reads happen in the *hand-rolled* kernel. If Apple's SDPA re-reads across
   the 64 head-groups, A2/A3 inflate ~64× (constant terms only — the depth
   verdict is unaffected).
