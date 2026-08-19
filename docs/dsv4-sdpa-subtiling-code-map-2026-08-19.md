# DSv4-Flash attention forward: SDPA sub-tiling code map (2026-08-19)

Scoping pass for "Option A": at nominal `EXO_PREFILL_STEP_SIZE=4096`, keep the chunk loop /
KV-cache update / barriers / MoE dispatch unchanged, and split ONLY the per-rank query-row
dimension inside the attention forward into sequential SDPA sub-calls.

All file:line refs are `mlx-lm/mlx_lm/models/deepseek_v4.py` unless stated otherwise, at repo
HEAD `510182cd1`. Companion files: `mlx-lm/mlx_lm/models/base.py`,
`mlx-lm/mlx_lm/models/cache.py`.

Note there are THREE attention classes, dispatched per layer by
`v4_attention_factory` (**4571-4578**): `compress_ratio == 0` → `LocalAttention`,
`== 128` → `CompressedAttention`, else → `SparseCompressedAttention`.

---

## 1. The SDPA call sites

### 1a. `SparseCompressedAttention` — span `attn.sdpa`

Class **4126**; forward **4177-4568**. Span opens at **4327**. Three mutually exclusive
branches:

| branch | condition | SDPA site |
|---|---|---|
| local-only | `pooled.shape[1] == 0` (**4329**) | `scaled_dot_product_attention(q, kv, kv, ...)` **4330-4338** |
| dense compressed | `pooled.shape[1] <= index_topk` (**4341**) | `_extend_mask` **4343**, then SDPA **4344-4352** |
| sparse | else (**4355**) | `_sparse_pooled_attention(...)` at **4436-4446** (single-gather tiled), **4458-4467** (tiled), **4470-4479** (untiled) |

**Query shape at the SDPA call:** `q` is `(B, n_heads, L_q, head_dim)`, and `L_q` is
**already the per-rank SEQ_SPLIT band** — not full L. The band slice is applied way
upstream: `wq_b` itself is run on the banded q-lora at **4245-4251**
(`_q_res_band = q_residual[:, _seq_lo:_seq_hi, :]`, `_Lq = _seq_hi - _seq_lo`). So at
STEP_SIZE=4096, 2 ranks: `L = 4096`, `q.shape[2] = 2048`.

`kv` stays FULL width — `self.kv_norm(kv_pre).reshape(B, 1, L, self.head_dim)` at **4252**
uses full `L`, and the cache fetch at **4272** returns the whole window.

**Mask at the SDPA call:** a **materialized dense bool array** in prefill, never `"causal"`.
Origin: `DeepseekV4Model.__call__` **6356-6361** calls
`create_attention_mask(..., window_size=self.args.sliding_window, return_array=True)`.
`return_array=True` forces `create_causal_mask` (`base.py` **53-54**, definition
`base.py` **24-42**) which returns a 2-D `(N, offset+N)` bool array. The `"causal"` string
early-return in `RotatingKVCache.make_mask` (`cache.py` **888-891**) is bypassed by
`return_array`. Mask is then sliced to the band at **4324-4325**:
`mask = mask[..., _seq_lo:_seq_hi, :]` (explicitly guarded `not isinstance(mask, str)`).
In the dense-compressed branch `_extend_mask` (**1213-1244**) promotes it to 4-D
`(B,H,L,S)` and concatenates the pooled columns.

**Already-existing sub-tiling precedent (important):** the sparse branch ALREADY does
exactly the proposed transformation, over `_SPARSE_SDPA_TILE` (**269**, default 128,
`EXO_DSV4_SPARSE_SDPA_TILE`): loop at **4428-4447** / **4450-4468** slices
`q[:, :, _s:_e, :]`, `topk[:, _s:_e, :]`, `mask[..., _s:_e, :]`,
`sparse_mask[:, :, _s:_e, :]`, then `mx.concatenate(_parts, axis=2)` (**4447**/**4468**).
The comment at **4391-4403** states the invariant explicitly: *"Each sub-chunk is per-query-row
independent … slicing q/topk/masks by row and concatenating the outputs is bit-exact. No cache
mutation here (kv/pooled already built)."* So ~20 sparse layers are already sub-tiled at
tile=128 and are NOT the source of the 4096 regression.

### 1b. `CompressedAttention` — span `attn.sdpa.compressed`

Class **3877**; forward **3928-4123**. Span opens at **4015**. Two paths:

* verify L-split (`2 <= _Lq <= _CATTN_LSPLIT_MAX_L`, default 8, **1301**): per-row L=1 loop
  **4028-4040**, already row-slicing `q[:, :, _l:_l+1, :]` and `mask[..., _l:_l+1, :]`,
  concatenated on axis 2 (**4040**).
* prefill / general: **single fused call** at **4042-4052** — this is the un-tiled site and
  the actual target of Option A.

**Query shape:** `q` is `(B, n_heads, L_q, head_dim)`. Unlike the sparse class, `wq_b` runs at
FULL L here (**3963-3972**); the band slice is applied late, immediately before SDPA, at
**4005-4013**:
```python
_seq_lo = _sg.rank() * _band
q = q[:, :, _seq_lo:_seq_hi, :]
if mask is not None and not isinstance(mask, str):
    mask = mask[..., _seq_lo:_seq_hi, :]
```
So at 4096/2 ranks, `q.shape[2] == 2048` at line **4043**.

**Mask:** dense bool array, already `_extend_mask`-promoted to 4-D at **3990** (before the band
slice) so its `[-2]` axis is the query-row axis; band slice at **4013** is a pure row slice.
`kv` at that point is `concatenate([local_kv, pooled])` (**3989**) — full KV width.

`cache=local_cache` is passed to `scaled_dot_product_attention` (**4033**, **4047**) but is used
ONLY for the quantized-KV check (`base.py` **131-163**: `hasattr(cache,'bits')`,
`cache.group_size`, `cache.bits`). **No cache mutation happens inside the SDPA helper** — safe
to call it N times per chunk.

### 1c. `LocalAttention` (ratio 0 layers) — no SEQ_SPLIT

Forward **3798-3874**, SDPA at **3843-3854**. This class has **no SEQ_SPLIT path at all**
(only the `_ATTN_ALLSUM` all_sum at **3868-3874**), so `q.shape[2] == L` (full 4096). Mask is
clamped to KV width by `_clamp_mask_to_kv` (**1188-1210**) at **3841**. If sub-tiling is added
for cost reasons, this class is a separate (and larger-L) target.

---

## 2. Where the SEQ_SPLIT query-row slice is computed

Env gates: `_SEQ_SPLIT_ENABLED` **177** (`EXO_DSV4_SEQ_SPLIT`, default on),
`_SEQ_SPLIT_MIN_L` **178** (default 16), `_SEQ_SPLIT_GATHER_VIA_ALLSUM` **189**.
Design comment block: **164-191**.

* `SparseCompressedAttention`: computed **ONCE, up front**, **4201-4215**:
  ```python
  _seq = (_sg is not None and _SEQ_SPLIT_ENABLED and L >= _SEQ_SPLIT_MIN_L
          and L % _sg.size() == 0)
  _band = L // _sg.size(); _seq_lo = _sg.rank()*_band; _seq_hi = _seq_lo+_band
  _seq_band = (_seq_lo, _seq_hi)
  ```
  `_seq_band` is then threaded to: q-lora slice **4245**, rope offset **4263**, pmask slice
  **4279-4280**, indexer **4308-4309**, attention mask **4324-4325**.
* `CompressedAttention`: computed once at **3998-4013**, but LATE (after compressor, projections,
  rope, KV update, mask build).

**Answer: yes — the per-rank band is computed exactly once per chunk per layer, with a single
`(_seq_lo, _seq_hi)` pair, and everything downstream consumes it. It is a clean natural point to
sub-slice further:** the sub-tiles are simply `range(0, q.shape[2], tile)` over the already-banded
`q`, exactly as the sparse branch does at **4428**.

---

## 3. RoPE relative to SDPA

`DeepseekV4RoPE.__call__` **1084-1101** → `mx.fast.rope(..., offset=offset)`. `mx.fast.rope`
applies position `offset + row_index` per row (this is stated in the fork's own comments at
**4054-4056** and **4489-4492**).

Order per forward:
* forward RoPE ("rope_in") is applied to `q` and `kv` BEFORE the SDPA call:
  `CompressedAttention` **3973-3977**; `SparseCompressedAttention` **4261-4267**.
* inverse RoPE ("rope_out") is applied to the SDPA OUTPUT: **4053-4058** and **4488-4494**.

**Offset source:** it is the KV cache's own `offset`, read once at the top of the forward:
`offset = local_cache.offset` — **3938** (CompressedAttention), **4193**
(SparseCompressedAttention), **3806** (LocalAttention). It is read BEFORE
`update_and_fetch` (**3982**, **4272**), so it is the pre-write chunk-start absolute position.
`_rope_dispatch` **1165-1185** falls through to that unless the tree-verify side channel
supplies explicit per-token positions (`_TREE_VERIFY_CTX`, **517**, **1112-1163**).

**SEQ_SPLIT already shifts it:** q rope-in offset is `offset + _seq_lo` (**4263**, sparse) and
rope-out offset is `offset + _seq_lo` (**4056**, **4492**).

**Consequence for sub-tiling:** RoPE is NOT applied inside the SDPA call and NOT derived from
any per-call counter. Two options, both correct:
1. RoPE the full banded `q` once (as today) and sub-slice the ALREADY-ROPED `q` → no offset
   arithmetic needed at all, since row j of the roped tensor already carries position
   `offset + _seq_lo + j`. Likewise concatenate the sub-outputs on axis 2 BEFORE `rope_out`
   and leave **4056**/**4492** untouched. **This is the recommended shape.**
2. If instead you rope each sub-tile separately, the sub-tile's offset must be
   `offset + _seq_lo + sub_start` for both rope_in and rope_out.

The existing sparse tiling (**4428-4447**) uses option 1 — it tiles strictly between rope_in and
rope_out — which is why it needs no offset adjustment.

---

## 4. KV cache update ordering (the critical correctness question)

**Confirmed: the KV cache is updated with the FULL chunk's K/V strictly BEFORE any SDPA call,
and the SDPA helper never mutates the cache.**

* `CompressedAttention`: `kv, _ = local_cache.update_and_fetch(kv, _zero_values(B, L))` at
  **3982**, inside `with span("attn.kv_cache")` **3979-3983**. SDPA span opens at **4015**.
  Note `kv` passed to update is built at full `L` (**3970**), not the band.
* `SparseCompressedAttention`: `update_and_fetch` at **4272** (span **4269-4273**);
  SDPA span opens at **4327**.
* `LocalAttention`: **3832**, SDPA at **3843**.

Pooled/compressed state is likewise fully built first: `self.compressor(x, pool_cache, offset)`
at **3958** / **4225** (issued before the projections, deliberately — comment **3941-3950**), and
`pooled` is concatenated onto `kv` at **3989** before the band slice.

`scaled_dot_product_attention` (`base.py` **122-190**) takes `cache` only to read
`cache.bits`/`cache.group_size` — read-only. `_sparse_pooled_attention` (**2305**) takes no cache
at all.

**Therefore: repeated sequential SDPA calls over row-sub-slices of the same already-banded `q`
against the same full `kv`/`pooled` are safe — no cache-write reordering, no double-advance of
`cache.offset`, no interleaving hazard.** This is also asserted by the fork's own comment at
**4402-4403** ("No cache mutation here (kv/pooled already built)") for the existing tiling.

Caveat to preserve: do NOT move the sub-tile loop above **3982**/**4272**, and do not sub-tile
`kv_norm`/`update_and_fetch`/`compressor`/`indexer` — those must stay full-L for cross-rank
coherence (comments **4196-4200**, **3993-3997**, **3558-3563**).

---

## 5. Post-SDPA path up to all_gather — required output shape

`CompressedAttention` **4053-4114**; `SparseCompressedAttention` **4488-4559**. Identical shape
contract:

1. SDPA output `out` : `(B, n_heads, band_len, head_dim)`.
2. `rope_out` **4057** / **4493** — shape-preserving, offset `offset + _seq_lo`.
3. `o_proj` **4060-4067** / **4496-4504**:
   `_o_len = out.shape[2] if _seq else L` (**4062**, **4499**) — note it reads the shape, so a
   correctly concatenated `out` needs no change here;
   `_o_pre_a(out, B, self.o_groups, _o_len, self.head_dim)` → `wo_a` → `_o_pre_b` → `wo_b`,
   producing `(B, band_len, hidden_size)`.
4. Reconstruction **4069-4114** / **4514-4559**, two variants:
   * default `_SEQ_SPLIT_GATHER_VIA_ALLSUM` (**189**, default on): `mx.pad` to full L with
     `(_seq_lo, L - _seq_lo - _band_len)` on axis 1, then `all_sum` on the top-level group
     (**4090-4103** / **4535-4548**). `_band_len = out.shape[1]` — read from the tensor.
   * fallback: `all_gather` on the subgroup then `reshape(_N, _B, _band, _H).transpose(1,0,2,3)
     .reshape(_B, L, _H)` where `_band = L // _N` (**4105-4114** / **4550-4559**) — this one
     uses the COMPUTED `_band`, not the tensor shape.

**Requirement for an implementer:** concatenate the sub-call outputs on **axis 2** (the query-row
axis of the `(B,H,L_q,D)` SDPA output) *before* `rope_out`, i.e. the concatenated tensor must be
byte-identical in shape to today's `(B, n_heads, _seq_hi-_seq_lo, head_dim)`. Then everything
from **4053**/**4488** onward, including both all_gather variants, works unchanged. Sub-tile
boundaries must partition `[0, band_len)` contiguously and in increasing order — the fallback
gather path assumes exact `L // N` band length, so do not pad or reorder tiles.

---

## 6. Indexer state lifecycle

`Indexer` class **3515-3546**; `__call__` **3548-3748**. Called from
`SparseCompressedAttention` at **4307-4310** with `seq_band=_seq_band`.

Stateful pieces and where they live:
* `self.compressor(x, pool_cache, offset)` at **3564** — this is the ONLY state mutation, and it
  mutates `pool_cache` (`PoolingCache`, `cache.py` **1270**+, `_pool_storage`/`_pool_offset`,
  `accumulate_windows` `cache.py` **1342**+). It is deliberately run on **full `x`** before the
  band slice (comment **3558-3563**: "the compressor MUST see full x (it builds the pool and
  mutates pool_cache — coherence across ranks)").
* Everything after **3565** is **purely functional** off `pooled` + `q_residual` + `offset`:
  pmask **3570** (`_dispatch_pmask` **621-635** → `PoolingCache.make_mask` `cache.py`
  **1518-1540**), band slice **3572-3582**, `wq_b` + rope **3584-3586**,
  `indexer.score` span **3588-3605** (`_indexer_score` **3381**, `_indexer_score_tiled` **3466**),
  pmask apply **3606-3653**, `indexer.topk` span **3668**+.
* The only other per-instance mutable attributes are the opt-in diagnostic
  `self._prev_topk_set` / `self._topk_overlap_step` (**3544-3545**, gated by
  `EXO_DSV4_TOPK_OVERLAP_LOG`, default off) — diagnostic only, no correctness coupling.

**Answer: the indexer carries NO per-chunk state that a *post-indexer* attention-side sub-chunk
split would perturb.** By the time the SDPA branch runs (**4327**), `topk` is a plain
`(B, L_band, k)` int array and `pmask` a plain bool array; both are row-independent and are
already row-sliced per tile today at **4440** and **4435**. Sub-tiling the SDPA is invisible to
the indexer.

Corollary constraint: do **not** move the sub-tile loop to wrap `self.indexer(...)` — that would
call the compressor multiple times and double-advance the pool.

---

## 7. Causal mask construction and row-slicing correctness

Construction chain, in order:

1. `DeepseekV4Model.__call__` **6339-6363**: `mask_cache` = the layer-0 local cache;
   `create_attention_mask(h[:,:,0,:], mask_cache, window_size=self.args.sliding_window,
   return_array=True)`.
2. `base.py` **45-55** → `RotatingKVCache.make_mask` (`cache.py` **882-906**) →
   `create_causal_mask(N, offset, window_size)` (`base.py` **24-42**), returning a **2-D
   `(L, offset+L)` bool array**, `linds >= rinds` AND `linds < rinds + window_size`. Built ONCE
   per model forward and passed by reference into every layer (**6448**:
   `h = layer(h, mask, layer_cache, inputs)`).
3. Per-layer promotion/clamping: `_clamp_mask_to_kv` **1188-1210** (LocalAttention **3841**);
   `_extend_mask` **1213-1244** (CompressedAttention **3990**, sparse dense branch **4343**) —
   promotes 2-D → `(1,1,L,S)` at **1217-1218**, clamps trailing KV columns **1229-1233**,
   concatenates the pooled block **1242**.

**Is a row slice trivial? Yes.** After promotion the mask's `[-2]` axis is exactly the query-row
axis and rows are mutually independent (`linds` is per-row, `rinds` per-column; window and pool
columns are per-row functions of the same row index). The codebase already relies on this in five
places: **4013** (CompressedAttention band), **4035** (verify L-split), **4325** (sparse band),
**4432**/**4454** (sparse tile loop), **4435**/**4457** (`sparse_mask` tile slice).

**Side effects / dependencies to respect:**
* The mask array is **shared across all 43 layers** — slicing produces a view/new array and must
  never be written in place. Existing code only rebinds the local name; keep that discipline.
* Slicing must be done on the ALREADY-band-sliced mask (rows `[0, band_len)` relative to the
  band), matching the `q` you sub-slice. In `CompressedAttention` the band slice is at **4013**,
  so a sub-tile is `mask[..., _s:_e, :]` with `_s,_e` relative to the band — same convention as
  the sparse tile loop.
* Order matters in `CompressedAttention`: `_extend_mask` (**3990**) runs BEFORE the band slice
  (**4013**) and computes column widths from `kv.shape[2]` (full KV). Since sub-tiling touches
  only rows, `_extend_mask` must stay where it is — do not move it inside the tile loop.
* In the sparse class the dense-compressed branch calls `_extend_mask` at **4343** *inside* the
  SDPA span and *rebinds* `mask`; if that branch is ever sub-tiled, hoist the `_extend_mask` call
  above the loop (it is row-uniform) rather than calling it per tile.
* Guard for a string mask: every existing slice site checks `not isinstance(mask, str)`
  (**4012**, **4324**, **4431**, **4453**). In prefill it is always an array because of
  `return_array=True` at **6360**, but the guard is cheap and must be preserved for the
  decode/verify shapes that share this code.
* Watch the `CompressedAttention` verify L-split predicate at **4021-4026**: it tests
  `mask.shape[-2] == _Lq` where `_Lq = q.shape[2]`. If sub-tiling is inserted, `_Lq` and the
  sliced mask must stay consistent, or a prefill sub-tile of size ≤ 8 could accidentally fall
  into the verify path. Recommend gating any new sub-tiling on `_Lq > _CATTN_LSPLIT_MAX_L` and
  choosing tile sizes ≫ 8.

---

## Summary of what an implementation would touch

| Concern | Location | Change needed |
|---|---|---|
| Un-tiled SDPA (main target) | **4042-4052** (`CompressedAttention`, ~21 layers) | wrap in a row-tile loop over `q.shape[2]`, concat axis 2 |
| Already tiled | **4428-4468** (`SparseCompressedAttention`, ~20 layers) | none — copy the pattern |
| No SEQ_SPLIT at all | **3843-3854** (`LocalAttention`) | optional separate target; full-L `q` |
| Band computation | **3998-4013**, **4201-4215** | none |
| KV/pool/indexer | **3958**, **3982**, **4225**, **4272**, **4307** | none — must stay full-L, above the loop |
| rope_in / rope_out | **3973-3977**/**4053-4058**, **4261-4267**/**4488-4494** | none if tiling strictly between them |
| o_proj + all_gather | **4060-4114**, **4496-4559** | none if concat restores `(B,H,band,D)` |
| Suggested new env gate | mirror **269** (`EXO_DSV4_SPARSE_SDPA_TILE`) | e.g. `EXO_DSV4_CATTN_SDPA_TILE`, default 0 (off) |
