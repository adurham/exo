# Phase 9: c=2 + 35 t/s aggregate findings

Status: **partial — discovered c=2 MTP regression but full fix > 1 day**.

## Findings

### Configuration discovery: `DSV4_KV_CACHE_BITS=4` is the script default

`start_cluster.sh:257` sets `DSV4_KV_CACHE_BITS=4` for the DSv4 auto-place
overriding the global `EXO_KV_CACHE_BITS=0`. Production today runs 4-bit
KV by default, not bf16. Skill memory says bf16 is the canonical
constraint; the auto-place script silently flips it to 4-bit. Per
`references/feedback_kv_cache_quality_risk.md`, 4-bit KV has ~4% perf
gain on bandwidth pressure but quant noise hurts accuracy. Was this an
intentional change? Worth checking.

### c=2 with MTP-OFF (bf16 KV): 31.4 agg t/s, σ ~0.1, 3/3 clean

Benched 3-iter c=2 75K prompts. Wall 766–767s per iter.
per_req = 15.7 t/s. Below 35 target. Stable.

### c=2 with MTP-ON (γ=2): **STRUCTURALLY BROKEN** — agg = 3.5–5.8 t/s

3-iter c=2 75K bench (4-bit KV): warmup 5.8 agg, iter 1 5.8 agg.
Flipping to bf16 KV: 3.5 agg.

MTP-PROF data at B=2:

```
B=2 phase profile (bf16 KV):
  draft:   147.34ms  (with max=7827ms outliers)
  verify:  176.85ms  (min=161, max=657)
  total:   324.78ms  (max=8485ms!)
```

For comparison B=1 (same cluster, same config) was:
```
  draft:    4.40ms
  verify:  42.81ms
  total:   48.11ms
```

Draft alone at B=2 is **30x B=1**. With outliers up to 7.8 seconds per
draft. Verify at B=2 is **4x B=1**.

### Root cause: MTP cache lifecycle assumes single-stream

`batch_generate.py:1129` calls `self._mlx_gen.mtp.reset_cache()`
UNCONDITIONALLY on every new submit, with no `batch_size` argument.
When stream 2 arrives while stream 1 is still running, the reset
clobbers stream 1's MTP cache state. Stream 1's drafts then use stream
2's prompt-prefill state, producing garbage drafts → 0% acceptance →
no speculation lift.

Combined with: the B=2 verify cost is also ~4x B=1 (likely due to
PerStreamBatchRotatingKVCache forcing the 4-D mask path through SDPA,
plus the dequantize-then-attend slow path when KV is quantized).

### MTP+c=2 fix scope

1. **MTP cache lifecycle for concurrent streams** (~1 day):
   - `DSv4MTPPredictor.reset_cache(batch_size=N)` already exists; just
     plumb the right batch size.
   - But `mtp.reset_cache()` from submit_task on a SECOND submit
     would re-init the cache, losing stream 1's state. The fix is
     to NOT reset on subsequent submits when other streams are
     running — instead, ADD stream 2's prefill to an existing
     multi-stream cache.
   - Requires changing the MTP cache to be per-stream-extendable, or
     re-thinking the submit-time prefill to be lazy / queued.

2. **B=2 verify cost** (multi-day, structural):
   - `PerStreamBatchRotatingKVCache.make_mask` returns 4D mask vs
     RotatingKVCache's 2D mask. The 4D mask forces SDPA out of the
     fused causal kernel.
   - Mitigation: detect uniform right_padding=0 (all streams at same
     offset) and emit a 2D mask in that case. Add per-stream mask
     only when streams diverge.

### Projected ceilings

Even with both fixes:

```
Linear MTP γ=2 c=1:        30 t/s   (current)
Linear MTP γ=2 c=2 (FIXED): per_req ≈ 30/1.85 ≈ 16.2  →  agg ≈ 32.4
Linear MTP γ=2 c=4 (FIXED): per_req ≈ 30/2.7  ≈ 11.1  →  agg ≈ 44.4
```

**c=4 is the path to ≥35 t/s aggregate.** c=2 fix gets us to 32, still
short.

But c=4 with concurrent MTP requires the same fixes plus dealing with
4-stream KV cache memory (4 × 89K × 25 MB ≈ 9 GB). M4 Max has 128 GB
but with model + activations the budget is tight.

## Recommendation

To hit 35 agg t/s with this hardware/model:

**Path A: Fix MTP+c=2 + Eagle soft-emb at c=2** (~5-7 days total)
1. Fix MTP cache lifecycle for concurrent streams (~2 days).
2. Fix B=2 verify cost via 2D mask shortcut (~1 day).
3. Add Eagle soft-emb (~2 days from prior plan).
4. Bench. Target ~34 t/s agg c=2.

**Path B: Get c=4 MTP-OFF to work at 35 agg** (~3-5 days)
1. Fix MTP cache lifecycle for c=4 (same as Path A step 1).
2. Bench c=4 MTP-OFF at various prompt sizes.
3. If c=4 MTP-OFF agg > 35 at 75K, ship.

**Path C: Lower the 75K context requirement**
- At shorter contexts (e.g., 20K) c=2 may already hit 35 agg since
  verify cost scales with context length.
- If production tasks have variable context length, optimize for the
  typical use rather than 75K worst-case.

## Open questions for the user

1. Is `DSV4_KV_CACHE_BITS=4` (4-bit KV) an intentional production change
   or accidental? Skill says bf16 is canonical. If 4-bit was intentional,
   we should redo the 30.06 c=1 baseline measurement at 4-bit.
2. Is the 75K context a hard production requirement, or representative
   of an upper bound? If 50K is typical, c=2 fix might already hit 35.
3. Greenlight Path A (fix MTP+c=2 + Eagle, ~5-7 days)?
