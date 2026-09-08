# Phase 3b prerequisite — window / top-k geometry inventory (fork: `mlx-lm/mlx_lm/models/deepseek_v4.py`)

Written **before** any Phase 3b code was changed, per
`DSV4_VISION_PORT_PLAN_PHASE34.md` §3b "Required approach" step 1.

File surveyed: `mlx-lm/mlx_lm/models/deepseek_v4.py` @ `64cc7e6` (7,927 lines),
plus `mlx_lm/models/base.py` and `mlx_lm/models/cache.py`.

---

## 0. Headline structural finding

**The fork has no `get_window_topk_idxs` and no integer window-index geometry
at all.** Grep for `window_topk`, `topk_idxs`, `get_window` returns exactly
zero producers — the only hit is a prose comment at line 6911 that *refers* to
the reference's `topk_idxs` concept.

The reference expresses sliding-window attention as an explicit
`(bsz, seqlen, window)` int32 **index matrix** fed to DeepSeek's proprietary
`sparse_attn` kernel (`from kernel import sparse_attn`). The fork expresses the
*same geometry* two different ways, neither of them an index matrix:

| geometry component | reference | fork |
| --- | --- | --- |
| sliding window over local KV | `get_window_topk_idxs(win, bsz, seqlen, start_pos)` → int idx matrix | **boolean mask** `create_causal_mask(N, offset, window_size=128)` + `RotatingKVCache(max_size=128)` ring physically bounding the KV |
| compressed/pooled selection | `get_compress_topk_idxs(...)` or `Indexer(...)` → int idx matrix, concatenated onto the window idxs | `Indexer.__call__` → `topk` int32 `(B, L, k)` over the **pooled** axis only; consumed by `_sparse_pooled_attention` |
| the two concatenated | `torch.cat([topk_idxs, compress_topk_idxs], -1)` | never concatenated — kept as two separate objects (bool `mask` for local, int `topk` for pooled) all the way down |

Consequence for 3b: `get_window_topk_idxs_visible` **cannot be dropped in as a
function swap**, because there is nothing to swap it for. The correct fork-side
expression of image-span visibility is *extra `True` bits in the local-KV
boolean mask*. The pooled/`Indexer` side is untouched by visibility (the
reference only replaces the window half of the concatenation, never the
compress half — `model_vision.py:548-558`).

---

## 1. Producers of window geometry (the things `get_window_topk_idxs` maps to)

### 1.1 `mlx_lm/models/base.py:24` `create_causal_mask(N, offset, window_size, ...)`
The literal geometry kernel. `mask = (linds >= rinds) & (linds < rinds + window_size)`.
For `offset=0` this is bit-for-bit the same visible set as the reference's
`start_pos == 0` branch of `get_window_topk_idxs`
(`(base - win + 1).clamp(0) + arange(min(seqlen, win))`, masked `> base` → -1).
**This is the fork's `get_window_topk_idxs`.** Single producer, no env gates.

### 1.2 `mlx_lm/models/base.py:45` `create_attention_mask(h, cache, window_size, return_array)`
Thin dispatcher: delegates to `cache.make_mask(...)` when the cache has one
(always true for DSv4), else `create_causal_mask`.

### 1.3 `mlx_lm/models/cache.py:882` `RotatingKVCache.make_mask(N, window_size, return_array)`
The path DSv4 actually takes. For `N > 1` (prefill / verify):
`offset = min(self.max_size - 1, self.offset)` then
`create_causal_mask(N, offset, window_size=window_size)`.
The `min(max_size-1, ...)` clamp is the **ring coordinate frame**: mask column
`c` ↔ absolute position `R - offset + c` where `R` = cache offset before the
update. For `N == 1` (decode) it returns a 1-D ring-rotation mask or `None`.

### 1.4 `cache.py:2252` `PoolingCache.make_mask` / `cache.py:3361` `BatchRotatingKVCache.make_mask` / `cache.py:3782` `PerStreamBatchRotatingKVCache.make_mask`
Batch/pooled analogues. `PoolingCache.make_mask` is the **pooled** row-causal
mask (`pmask`), not window geometry.

### 1.5 `deepseek_v4.py:690` `_tree_pmask` / `:721` `_dispatch_pmask`
Speculative-tree-aware replacement for `PoolingCache.make_mask`. Pooled-side
only. Not window geometry.

### 1.6 The single model-level call site: `deepseek_v4.py:7267-7272`
```python
mask = create_attention_mask(
    h[:, :, 0, :], mask_cache, window_size=self.args.sliding_window,
    return_array=True,
)
```
inside `DeepseekV4Model._forward_steps`. **Every** transformer layer receives
this one `mask` object (`h = layer(h, mask, layer_cache, inputs)`, line 7392).
There is exactly ONE window-geometry production per forward pass.
Alternative branch above it: `_TREE_VERIFY_CTX["mask"]` (speculative tree
drafting) bypasses `create_attention_mask` entirely — decode-time only, never
prefill, so never coincident with image visibility.

---

## 2. Consumers of window geometry (every site that reads the local mask / assumes reach == `sliding_window`)

Ordered by how dangerous they are to Phase 3b.

| # | site | line(s) | assumes reach == `sliding_window`? | verdict for visibility |
| --- | --- | --- | --- | --- |
| C1 | `DeepseekV4Block.__call__` → `self.attn(..., mask=mask, ...)` | 5480+ | no, passes through | safe |
| C2 | `LocalAttention.__call__` → `_clamp_mask_to_kv(mask, kv.shape[2])` → SDPA | 4594-4670, clamp at 4637 | **no** — clamps mask to actual KV width, takes trailing columns | **safe**, widened mask flows straight into `mx.fast.scaled_dot_product_attention` |
| C3 | `CompressedAttention.__call__` → `_extend_mask(mask, pooled_mask, kv.shape[2])` → SDPA | 4737-4830, extend at 4799 | no — derives `local_len = N - pooled_width` from actual widths | **safe** |
| C4 | `SparseCompressedAttention.__call__` → `_extend_mask` / `_sparse_pooled_attention(mask=...)` | 5061-5330 | no | **safe** |
| C5 | `_extend_mask` | 1695-1727 | no — computes `local_len` from the caller's `N`; clamps `mask[..., -local_len:]` | safe **iff** the KV ring is wide enough to hold the extra visible keys (see §3) |
| C6 | `_clamp_mask_to_kv` | 1670-1692 | no | safe, same caveat |
| C7 | **`_query_tiled_ok` + the query-tiled SDPA block** (`EXO_DSV4_QUERY_TILED_SDPA=1`, default OFF) | gate 3904-3930, body 4827-4900 | **YES — hardcodes `_sw = self.config.sliding_window` and slices `kv[:, :, _key_lo : min(_local_len, _key_lo + _b - 1 + _sw)]`** | **UNSAFE.** It re-derives the visible key range from `sliding_window` instead of reading the mask, so any key made visible by an image span but further than `sw` back is silently dropped from the slice. Must fall back. |
| C8 | `_sparse_fused_sdpa` (`EXO_DSV4_SPARSE_FUSED_SDPA`) | 2506-2660, `_norm_mask` at 2543 | **YES — `_norm_mask(local_mask, sw)` requires the mask's trailing dim to equal `sw` exactly, and there is a hard `sw + k_sel > 768 → return None` register bound** | self-declining. Gated `L <= 16` (decode/verify) which is `start_pos > 0` where the reference never sets `visible` anyway. Falls back on its own contract check. Documented, no change needed. |
| C9 | `_sparse_verify_rows_batched` / `_cached_verify_mask` | 2694-2877 | partially (`sw` arg) | `L <= _SPARSE_VERIFY_MAX_L` (16) — decode/verify only. Never coincident with visibility. |
| C10 | seq-split row-band mask slicing (`_SEQ_SPLIT_ENABLED`) | 4820-4822, 5208-5209, 4340-4352 | no — slices mask **rows** only, keeps all columns | safe (visibility adds columns, not rows) |
| C11 | `_SPARSE_SDPA_TILE` query-row tiling + `EXO_DSV4_SINGLE_GATHER` | 5290-5330 | no — slices mask rows `[_s:_e]`, columns untouched | safe |
| C12 | `Indexer.__call__` → `topk` (`EXO_DSV4_INDEX_TOPK`, `EXO_DSV4_EXACT_TOPK*`, `EXO_DSV4_PREFILL_ARGPARTITION`) | 4310-4530 | n/a — **pooled** axis, not window | out of scope by construction; reference also leaves `compress_topk_idxs` untouched under visibility |
| C13 | `Compressor.__call__` / `PoolingCache.accumulate_windows` / overlap-carry | 3501-3698 | n/a — pooled axis | out of scope |
| C14 | `DSparkLocalAttention.draft_block` / `append_ctx` | 6890-6941 | passes `mask=None` | draft head, decode-time. Out of scope. |
| C15 | `DeepseekV4MTPModule.__call__` | 5858+ | passes caller mask | decode-time. Out of scope. |
| C16 | `_rowseq_vec_ring_mask` / `_local_rowseq_vec_loopreal` / `_compressed_rowseq_vec` / `_sparse_rowseq_vec` (`EXO_DSV4_VERIFY_ROWSEQ*`) | 6084-6877 | ring-slot mask, `sw`-shaped | `2 <= L <= _VERIFY_ROWSEQ_MAX_L`, decode/verify only. Out of scope. |

### Env flags in the geometry family (grep-confirmed, with defaults)
`EXO_DSV4_SPARSE_SDPA_TILE` (369, default 128) ·
`EXO_DSV4_EXACT_TOPK` (3840, default 1) ·
`EXO_DSV4_EXACT_TOPK_PREFILL` (3849, default 0) ·
`EXO_DSV4_QUERY_TILED_SDPA` (3868, default 0) ·
`EXO_DSV4_QUERY_TILED_B` (3889, default 64) ·
`EXO_DSV4_INDEX_TOPK` (4274, model-config override) ·
`EXO_DSV4_SINGLE_GATHER` (5296, default 1) ·
`EXO_DSV4_PREFILL_ARGPARTITION` (4492, default 0) ·
`EXO_DSV4_SPARSE_VERIFY_BATCHED` (1859, default 1) ·
`EXO_DSV4_VERIFY_ROWSEQ*` / `EXO_DSV4_VERIFY_BATCH*` (decode-side).

---

## 3. Why the mask formulation is *exactly* equivalent, and its one hard precondition

Reference visible set for query row `i` (derived from
`get_image_visible` + `get_window_topk_idxs_visible`):

```
left_i   = clamp(i - span_start, max=max_image_tokens-1)   if i inside a span else 0
right_i  = clamp(span_end - i,   max=max_image_tokens)     if i inside a span else 0
start_i  = max(0, i - max(window_size - 1, left_i))
visible_i = { j : start_i <= j <= i + right_i  and  j < start_i + width }
            width = min(seqlen, window_size + max_image_tokens)
```

For a non-image token `left=right=0` ⇒ `visible_i = [i-127, i]` — identical to
`create_causal_mask(..., window_size=128)` row `i`. So the visibility mask is a
strict **superset** of the existing causal-window mask, differing only on rows
inside an image span. That is what makes "OR the extra bits into the existing
mask" a faithful port rather than an approximation.

Note the set is **not causal** inside a span: `j` runs up to `i + right_i > i`.
Image-span tokens attend bidirectionally within their own span. This is
intentional in the reference and must be reproduced.

Width sufficiency check (the `j < start_i + width` clause): with
`window_size=128`, `max_image_tokens=384`, span length ≤ 384 —
worst case is `left_i ≤ 127` where the span-reach is `127 + right_i + 1 ≤ 512`,
exactly `window_size + max_image_tokens`. For `left_i > 127` the reach is
`left_i + right_i + 1 = span_len + 1 ≤ 385`. So `width` never truncates a
legal span, but the clause is reproduced anyway for bitwise parity.

**Hard precondition (the one real constraint):** the local KV must physically
still contain every key in `visible_i`. `RotatingKVCache(max_size=128)` retains
`max_size - 1 + S = 127 + S` rows after a prefill chunk of `S` tokens. An image
span reaching `≤ 384` positions back is therefore only retrievable when the
**entire span lies inside the current prefill chunk**. That is exactly the
invariant the reference itself asserts
(`assert (input_ids < self.vocab_size).all(), "image spans must be prefilled in
a single chunk"`) and exactly what Phase 4d is chartered to guarantee on the
exo side (`EXO_PREFILL_STEP_SIZE=2048` ≫ 386). Phase 3 asserts it; Phase 4
enforces it at the chunker.

Corollary: because visibility is prefill-only (`start_pos == 0` in the
reference's `Transformer.forward`) and prefill is `L ≫ 16`, **every** decode-
and verify-side optimized path (C8, C9, C16) is structurally out of reach and
needs no change — not by choice, by construction.

---

## 4. Implementation decision recorded before writing code

1. Port `get_image_visible` and `get_window_topk_idxs_visible` to MLX **verbatim**
   (`_get_image_visible`, `_get_window_topk_idxs_visible`) and prove exact
   integer equality against torch. These are the spec; they are also what a
   future index-based kernel would consume.
2. Add `_image_visible_mask(left, right, window_size, mask_offset, N, max_image_tokens)`
   producing the boolean mask the fork's SDPA path actually consumes, and prove
   it is *derivable from* the index matrix of (1) — i.e. mask[i,j] is True iff
   `j` appears in row `i` of `_get_window_topk_idxs_visible`. This makes the mask
   form provably the same geometry, not a re-derivation that could drift.
3. Gate the whole thing behind **`EXO_DSV4_IMAGE_VISIBILITY` (default OFF)**;
   additionally require `vision_n_layers > 0` and at least one `input_id >=
   vocab_size`. Three independent conditions, any of which being false leaves
   the mask object *identically* the one produced today.
4. **C7 (`EXO_DSV4_QUERY_TILED_SDPA`) gets an explicit, loud refusal**, not a
   silent wrong answer: `_query_tiled_ok` returns `False` whenever image
   visibility is active for the current forward, so the path falls back to the
   unchanged single fused SDPA call. Retrofitting it correctly means
   re-deriving each block's key slice from the mask's per-row visible span
   instead of from `sliding_window` — tractable, but it is a second
   optimization port with its own A/B burden and is not what Phase 3 is
   chartered to deliver. Recorded here as the next attack vector.
