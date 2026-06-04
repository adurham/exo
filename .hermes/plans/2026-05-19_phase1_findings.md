# Phase 1 Findings: Token-Tree Drafting Investigation

Baseline commits at start of investigation:
- exo: c7032932 (main)  -- "deps: bump mlx-lm to 6dcdd40a (remove mc_ping...)"
- mlx-lm: 6dcdd40 (main) -- "fix(dsv4): remove unconditional /tmp/mc_ping.txt write..."

Matches plan anchor (`baseline-2026-05-18-mtp-g2-topk512-30.06`). No code touched
yet -- this file is pure reading + small probes.

---

## 1.1 Speculative path map (corrected line numbers)

The plan had two file/line mix-ups -- correcting here for downstream phases.

### draft_tokens()  -- mtp_module.py:581 (NOT dsv4_mtp.py:581)

File: `src/exo/worker/engines/mlx/speculative/mtp_module.py`
Function: `draft_tokens(mtp_pred, hidden, first_token_arr, gamma, temp,
fast_lm_head=False, sync_group=None) -> (draft_ids, draft_probs)` @ L581.

What it does (lazy MLX graph, no `mx.eval` except the per-step fence):
- Inputs:
  - `mtp_pred`: instance of `MTPPredictor` (mtp_module.py) -- has `.predict()`,
    `.predict_hidden()`, `.predict_from_hidden()`, `.kv_cache` (a regular
    RotatingKVCache-like cache for the MTP head only).
  - `hidden`: pre-norm hidden of shape `(1, 1, D)` captured at the LAST verify
    position from the previous cycle (see `_speculative_next` step 7,
    dsv4_mtp.py:1469-1470).
  - `first_token_arr`: shape `(1, 1)`, dtype int -- the token just committed
    to the main model's cache (=`y` in `_speculative_next`).
  - `gamma`: int, draft depth. Production: 2.
  - `temp`: float. Production: 0 (greedy).
  - `sync_group`: coord subgroup for cross-rank broadcast of each draft tok.
- Loop body, iteration i in range(gamma):
  - `logits, h = mtp_pred.predict(h, tok_arr, return_hidden=True,
                                   draft_mode=fast_lm_head)`
    - logits shape: `(1, 1, vocab_size)` (predict() squeezes S=1 to `(1,
      vocab)` actually; see predict() L524-525). Returns h pre-norm
      `(1, 1, D)`.
  - if temp == 0: `tok_arr = argmax(logits).reshape(1,1)`; broadcast across
    ranks; append to draft_ids; draft_probs[i] = None.
  - else: `q = softmax(logits/temp)`, `tok_arr = random.categorical(...)`,
    broadcast tok; draft_probs[i] = q.
  - **Per-step fence:** `if i+1 < gamma: mx.eval(tok_arr)` -- the gamma>=2
    fix from 2026-05-17 to drain the chained-all_sum queue.
- Output: `draft_ids` is `list[mx.array]` length `gamma`, each `(1,)` scalar.

KV cache semantics: each call to `predict()` advances `mtp_pred.kv_cache.offset`
by 1 (via `update_and_fetch` in `_attn_mlp` L459). So after gamma draft
iterations the MTP cache has grown by gamma.

### _speculative_next()  -- dsv4_mtp.py:1255  (line number correct in plan)

File: `src/exo/worker/engines/mlx/speculative/dsv4_mtp.py`
Function: `DSv4MTPBatchGenerator._speculative_next(uid) -> [Response]` @ L1255.

The full verify+accept orchestration. Cycle outline (matches plan's "Current
linear speculation" diagram exactly, just heavier on cross-rank syncs):

1. **L1304-1308 -- Draft:**
   ```
   next_token_arr = y.reshape(1, 1)               # last committed token
   draft_ids, draft_probs = draft_tokens(
       self.mtp, pre_norm, next_token_arr, gamma, temp,
       sync_group=coord_group,
   )
   ```
   `pre_norm` is the MTP's seed hidden, captured from the previous cycle's
   verify forward at position `n_accepted` (or `gamma`).

2. **L1317-1322 -- Build verify input:**
   ```
   draft_concat  = concat([d.reshape(1,1) for d in draft_ids], axis=1)  # (1, gamma)
   verify_input  = concat([next_token_arr, draft_concat], axis=1)       # (1, gamma+1)
   ```
   Shape is **always (1, gamma+1)**. For gamma=2: (1, 3). For tree K=2 gamma=2
   we'll need (1, n_nodes) = (1, 7).

3. **L1323-1328 -- Verify forward:**
   ```
   verify_pre_norm, verify_logits = dsv4_speculative_forward(
       self.model, verify_input, gen_batch.prompt_cache, self._captured,
   )
   ```
   `dsv4_speculative_forward` (dsv4_mtp.py:288) is a 3-line shell:
   `logits = model(inputs, cache=cache); pre_norm = captured["pre_norm"]`.
   So the verify pass is just a regular `mlx-lm.Model.__call__` on the
   `(1, gamma+1)` input. The pre-final-norm hidden is captured as a side
   effect of the wrapped `DeepseekV4Model.norm` (`_setup_hidden_capture`,
   not read here).

4. **L1336 -- Target argmax (greedy path):**
   ```
   target_tokens = argmax(verify_logits[:, :gamma, :], axis=-1)  # (1, gamma)
   matches = equal(target_tokens, draft_concat).squeeze(0)        # (gamma,)
   all_next = argmax(verify_logits[0], axis=-1)                   # (gamma+1,)
   ```
   These shapes implicitly assume L_q = gamma+1. **For tree drafting they
   must be reshaped/reindexed by tree node, not by sequence position.**

5. **L1380-1394 -- Accept loop (greedy):**
   ```
   for i in range(gamma):
       if matches[i].item():
           n_accepted += 1
       else:
           break
   ```
   Walks left-to-right; first mismatch breaks.

6. **L1401-1415 -- Bonus token:**
   - If n_accepted == gamma: bonus = `all_next[gamma]`.
   - Else: bonus = `all_next[n_accepted]`.

7. **L1448-1465 -- Cache rollback:**
   `rollback = gamma - n_accepted` positions, applied to:
   - every `c` in `gen_batch.prompt_cache` (RotatingKVCache + PoolingCache
     wrapped in CacheList; `c.trim(rollback)` recurses).
   - `self.mtp._cache` (the MTP-head's own KV cache), same trim amount.

8. **L1467-1470 -- Stage next pre_norm:**
   `pos = gamma if n_accepted == gamma else n_accepted`
   `self._mtp_pre_norm[uid] = verify_pre_norm[:, pos : pos+1, :]`

### predict_from_hidden()  -- mtp_module.py:536

File: `src/exo/worker/engines/mlx/speculative/mtp_module.py`
Signature: `predict_from_hidden(self, prev_hidden) -> hidden`

NOTE: **NOT CURRENTLY USED.** Grep confirms zero callers in repo. It would
replace `embed_tokens(token) + pre_fc_norm_embedding` with `norm(prev_hidden)`,
skipping the lm_head -> argmax -> embed roundtrip when chaining MTP steps.

Plan Q1: Does it support batched K-way prediction? **NO.** It takes a single
`(1,1,D)` hidden and emits a single `(1,1,D)` hidden. K-way logic must
happen at the caller. Recommendation stands: **Phase 2 Option A** (call
`predict()` K times per parent, snapshot/restore the MTP cache between
branches). Plan estimate of 3 head invocations per cycle (1+2 for K=2 gamma=2)
is correct.

### model.__call__  -- mlx-lm/mlx_lm/models/deepseek_v4.py

File: `mlx-lm/mlx_lm/models/deepseek_v4.py`
- `Model.__call__(inputs, cache=None)` @ L2489 -- wraps embed+layers+lm_head.
- `DeepseekV4Model.__call__(inputs, cache=None)` @ L2371 -- the body. Mask
  construction is **centralized here**:
  ```
  L2393-2402:
    first_cache = cache[0]
    mask_cache = first_cache[0] if isinstance(first_cache, CacheList) else first_cache
    mask = create_attention_mask(
        h[:, :, 0, :],                       # only h[0] over the hc_mult dim
        mask_cache,
        window_size=self.args.sliding_window,
        return_array=True,
    )
  ```
  Then layers receive that mask:
  `for layer, layer_cache in zip(self.pipeline_layers, cache):
       h = layer(h, mask, layer_cache, inputs)`

  **This is the single chokepoint for tree mask plumbing.** We do NOT need to
  modify every attention class's signature. The plan section 3.3 suggested
  threading `tree_mask` through Model -> Block -> Attention; in practice we
  can either (a) override `mask_cache.make_mask` for tree forwards, or
  (b) compute the tree mask explicitly here and short-circuit
  `create_attention_mask`.

- `create_attention_mask` (base.py:45):
  - If cache has `make_mask` (always true for RotatingKVCache):
    `return mask_cache.make_mask(N, return_array=True, window_size=...)`
  - For N=1 (decode) returns None.
  - For N>1 (verify) returns a causal mask: `RotatingKVCache.make_mask`
    (cache.py:733-757) returns either `"causal"` (string) or an array from
    `create_causal_mask(N, offset, window_size)`.

- `RotatingKVCache.make_mask(N, window_size, return_array)`:
  - With our verify path (`return_array=True`), if `offset + N > window_size
    or return_array`, returns an array from `create_causal_mask`.
  - Shape: `(N, offset+N)` boolean (with the causal lower-triangular within
    the last-N block).

- `DeepseekV4Block.__call__(h, mask, cache, input_ids)` @ L2155 -- forwards
  mask straight to `self.attn(normed, mask=mask, cache=cache)`. No tree-aware
  reshaping needed here.

### Three attention variants  (all in deepseek_v4.py)

All three take `(x, mask, cache)` and pass `mask` to
`scaled_dot_product_attention(q, kv, kv, cache=..., mask=mask, sinks=...)`.

| Class                       | Lines       | compress_ratio | KV sources to attend                                              |
|-----------------------------|-------------|----------------|-------------------------------------------------------------------|
| LocalAttention              | 1626-1735   | 0              | RotatingKVCache only                                              |
| CompressedAttention         | 1738-1860   | >0, no indexer | local KV + pooled KV (concat along axis=2)                        |
| SparseCompressedAttention   | 1863-2058   | >0, w/ indexer | local KV + (optional) pooled KV with TOPK index over pool        |

All three apply RoPE via `self.rope(q, offset)` and `self.rope(kv, offset)`
where `offset = cache.offset` (a SCALAR int -- the cache's append position).

Production path at TOPK=512 is **SparseCompressedAttention** for most layers
(see `compress_ratios` in config). This is what tree mask must work with.

### DeepseekV4RoPE.__call__  -- deepseek_v4.py:654

```
def __call__(self, x, offset=0, inverse=False):
    head_dim = x.shape[-1]
    freqs = self._get_freqs(head_dim, inverse)
    return mx.fast.rope(
        x, head_dim, traditional=True, base=None, scale=1.0,
        offset=offset, freqs=freqs,
    )
```

**CRITICAL FINDING for Phase 3.4 (RoPE positions for tree).** The mlx
`mx.fast.rope` doc string says:

  > offset (int or array): The position offset to start at. If an array
  > is given it can be a scalar or vector of **B** offsets for each example
  > in the batch.

**So `mx.fast.rope` supports per-batch offset BUT NOT per-token offset.** It
applies positions `[offset, offset+1, ..., offset+T-1]` to a length-T sequence.

For tree drafting at L_q=7 with positions (L_kv, L_kv+1, L_kv+1, L_kv+2,
L_kv+2, L_kv+2, L_kv+2), `mx.fast.rope` CANNOT directly emit those positions.

Options ranked:

- **A. Reshape to make "L_q" the batch axis.** Pass `q` reshaped to
  `(7, n_heads, 1, head_dim)` and `offset = mx.array([L_kv, L_kv+1,
  L_kv+1, L_kv+2, L_kv+2, L_kv+2, L_kv+2])`. Per-batch offsets work.
  Then reshape back to `(1, n_heads, 7, head_dim)` for SDPA. KV side the
  same. **Cost:** extra reshape + a 7-way batched RoPE call.
  **Risk:** does mlx's RoPE preserve content when called per-row? Yes -- it's
  an element-wise rotation, no cross-row mixing. **This is the recommended
  path.**

- **B. Manual RoPE outside mx.fast.rope** -- compute cos/sin from positions
  vector explicitly. Slower (Python kernel composition), but a known-good
  fallback if A fails verification.

- **C. Multiple separate (1,1,D) RoPE calls, one per tree node.** Worst --
  L_q kernel launches per attention call * 43 layers = ~300 extra launches
  per verify. Reject.

**Option A is the plan.** No mlx-lm RoPE API change needed -- we just call
`self.rope(q.reshape(L_q, n_heads, 1, head_dim), offset=positions_array)`
and reshape back.

### Cache: where positions come from

Every attention layer reads `offset = cache.offset` (scalar int). For tree
verify we need to override THAT. Two strategies:

- **Strategy X (cache-level):** subclass RotatingKVCache and inject a
  per-call positions vector. Touches cache.py, broad blast radius.
- **Strategy Y (Attention-level):** plumb a `position_ids` kwarg through
  Model -> Block -> Attention; when present, use it instead of
  `cache.offset` for the RoPE call. Plumbs through 4 call sites and three
  attention classes.

Strategy Y is what the plan section 3.3 already calls for. Recommend
adding `tree_positions: Optional[mx.array]` and `tree_mask: Optional[mx.array]`
as two kwargs.

### Q4: shape assertions hardcoding L_q=gamma+1

Plan asks: grep for `assert` and `shape[1] ==`. Result:
- `dsv4_mtp.py` has 5 asserts (L1384/1390/1403/1406/1411), all
  `assert X is not None`. **NO L_q-dependent shape asserts.**
- BUT implicit assumptions exist:
  - `verify_logits[:, :gamma, :]` (L1336)
  - `verify_logits[0, gamma]` (L1366, L1408)
  - `verify_pre_norm[:, pos:pos+1, :]` with `pos in [0..gamma]` (L1470)
  - `logprobs_all[gamma]`, `logprobs_all[i]` with `i in range(gamma)`
    (L1408, L1476)

For tree drafting, ALL of these indexing patterns change from sequence
position to **tree node index**. The verify forward still emits a
`(1, n_nodes, vocab)` tensor; the accept walk must follow parent_idx.
Plan section 4.2 covers this; the code change is concentrated in
`_speculative_next` lines 1336-1480.

### batched_moe.py (Qwen3.5)

Read in full. **Conclusion: NOT a useful template for DSv4 tree verify.**

This file is the Qwen3.5 production MoE path with 4 fused custom Metal
dispatches. It doesn't implement token-tree -- it's just a batched MoE
forward that takes B (token) entries. The B dimension corresponds to
sequence length L_q, not tree nodes; mask handling, RoPE, and accept logic
all live elsewhere in the Qwen3.5 model/speculative code.

For tree drafting we don't reuse this. The DSv4 MoE (deepseek_v4.py:1106)
already handles `(B, L_q, D)` inputs and is L_q-agnostic. **No MoE changes
needed.**

---

## 1.2 EXECUTED -- Alpha distribution measurement (PLAN ABORT TRIGGERED)

Probe installed in commits ac727f58 + d1b09288 + f4922bf3 (probe-graph-leak
fix). Cluster restarted 3 times; final run with `EXO_DSV4_TREE_ALPHA_PROBE=1
EXO_DSV4_MTP_PROFILE=20 EXO_SPECULATIVE_GAMMA=2 EXO_DSV4_INDEX_TOPK=512`.

### Bug found and fixed during execution

First version of the probe stored `mx.array` refs in the queue and called
`.tolist()` later in `_speculative_next`. This held lazy refs into the
MTP forward graph across cycles and corrupted decode: MTP started emitting
all-zero logits (top5=[0,1,2,3,4]), main-model output went to gibberish
("ällenällenällen..."), t/s collapsed from 30 to 16. Verified by toggling
the probe env var: probe-on → broken, probe-off → 48.6 t/s coherent.

Fix (commit f4922bf3): call `.tolist()` inside `draft_tokens` immediately
after `mx.argsort`, queue only ever holds plain `list[int]`. Re-bench with
fixed probe: 33.05 / 33.04 / 33.12 t/s, coherent output ("We need to
summarize the computer science topics..."). Memory: warm-tier entry will
capture this so any future mlx-lm-probe authors avoid the same trap.

### Measured alpha distribution (n=360, 3x 100K c=1 g=2 iters, temp=0)

```
                P(top1)   P(top2)   P(top3)   top1→top2 lift
step 0           0.7833    0.8500    0.8833    1.0851x
step 1           0.4167    0.5500    0.5833    1.3200x
```

(All 3 iters identical due to deterministic temp=0; effective n is the
60-cycle bench prompt of single-stream decode at 100K.)

### Predicted tokens/cycle and t/s

```
Linear gamma=2 (current production):
  P(k=0)=0.217  P(k=1)=0.457  P(k=2)=0.326
  E[accepted]=1.110, +1 bonus = 2.110 tokens/cycle

Tree K=2 gamma=2 (proposed, assuming step-1 independence):
  P(k=0)=0.150  P(k=1)=0.383  P(k=2)=0.468
  E[accepted]=1.318, +1 bonus = 2.318 tokens/cycle
  Lift over linear: +9.85%

Predicted t/s after wall growth (L_q=3 → L_q=7):
  Best case   (0% wall growth):  33.0 t/s
  Midpoint    (+8% wall growth): 30.5 t/s
  Worst case  (+20% wall growth): 27.5 t/s
```

### Plan abort gate triggered

Plan section 1.2 line 150-151 says:
> If top-2 vs argmax gain is < 1.10x, the tree approach is NOT going to
> hit 35 t/s. Stop and reconsider.

**Measured step-0 lift is 1.085x, below the 1.10x abort threshold.**

The combined tokens/cycle lift is +9.85% (linear 2.11 → tree 2.32). After
the L_q=3→7 verify-wall growth, predicted t/s ceiling is 33.0 (best case,
0% wall growth). **Plan target of 35 t/s is not achievable** with K=2
gamma=2 token-tree drafting on the measured alpha distribution.

K=3 gamma=2 would lift to ~2.40 t/cycle (+13.7%), still ~34 t/s after
wall growth -- still below 35.

### Why the plan's estimate was optimistic

Plan section "Why tree drafting can lift alpha" estimated:
> if linear single-step match is ~0.72 (alpha_2 = 0.52 = 0.72^2), then top-2
> branched single-step match becomes ~0.85, tree alpha_2 ~ 0.85^2 = 0.72.
> That is a 38 percent lift in tokens/cycle.

Reality:
- linear single-step ISN'T uniform: step 0 = 0.78 (high), step 1 = 0.42 (LOW).
  Plan assumed 0.72 across both steps.
- The chained-MTP step-1 is the alpha-limiting step (P(top1)=0.42). Top-2
  lift there IS big (1.32x), confirming the plan's hypothesis qualitatively.
- BUT step 0 already had P(top1)=0.78, near a ceiling. Top-2 only adds
  another 6.7pp (to 0.85) -- a 1.085x lift, much smaller than the 1.18x
  the plan assumed.
- Compound: tree P(both accept) = 0.85 × 0.55 = 0.47 (vs plan's 0.72
  prediction). Only +9.85% tokens/cycle vs the 38% the plan predicted.

The plan's math was right; only the assumed P(top1)≈0.72 input was wrong.
Production alpha at 100K c=1 has bimodal step-confidence: step 0 is high
(~0.78), step 1 is low (~0.42).

---

## 1.2 DESIGN NOTES (kept for reference)

Goal: measure P(target == MTP_argmax) at draft step 0 and step 1, and
P(target in MTP_top2), to verify the >=1.15x top-K lift assumed in the
plan.

**Probe location: `_speculative_next` after L1305 (draft_tokens) and L1336
(target_tokens).** Simplest because:
- We have draft_ids[0], draft_ids[1] in hand (the MTP argmaxes).
- We have target_tokens (verify argmax) in hand.
- We need MTP top-K -- that requires modifying draft_tokens to emit top-K
  logits in addition to argmax.

**Probe design (NOT YET INSTALLED):**

```python
# In mtp_module.py:draft_tokens, gated by EXO_DSV4_TREE_ALPHA_PROBE=1:
if _ALPHA_PROBE and temp == 0:
    top5 = mx.argsort(-logits, axis=-1)[:, :5].reshape(-1)  # (5,)
    _probe_log.append({
        "step": i,
        "argmax": int(top5[0].item()),
        "top2": int(top5[1].item()),
        "top3": int(top5[2].item()),
        "top4": int(top5[3].item()),
        "top5": int(top5[4].item()),
    })

# In dsv4_mtp.py:_speculative_next after target_tokens computed, same gate:
if _ALPHA_PROBE:
    for i in range(gamma):
        tgt = int(target_tokens[0, i].item())
        rec = _probe_log[-(gamma - i)]
        rec["target"] = tgt
        rec["match_argmax"] = (tgt == rec["argmax"])
        rec["match_top2"]   = (tgt in (rec["argmax"], rec["top2"]))
        rec["match_top3"]   = (tgt in (rec["argmax"], rec["top2"], rec["top3"]))
        _probe_writer.write(json.dumps(rec) + "\n")
```

Write to `/tmp/dsv4_alpha_probe.jsonl`.

**Bench scope:** smallest prompt that still exercises real DSv4 attention
state -- the plan suggests 1K context (NOT 100K). One reason: the
**alpha distribution at 100K is what matters in production** (see plan
pitfall #5). The 1K probe is sanity, not verdict.

To get production-realistic alpha at 100K we'd need a probe run on the
cluster, which is expensive (~8 min/iter). Cheaper alternative for Phase 1:
run the existing baseline bench with the probe enabled for 1-2 iters at
100K, count probe records.

**Phase 1.2 deliverable for the next coding pass:** install the probe
(gated), run 1 iter at 100K, compute P(top1/top2/top3 match) at both
steps, and report. **NOT EXECUTED YET** -- this requires a cluster restart
(see exo-cluster-operations skill pitfalls 1, 2, 7). Defer to Phase 1
followup.

**Plan's analytic estimate stands:**
- Assumed linear single-step match = sqrt(alpha_2) = sqrt(0.52) = 0.721.
- Assumed top-2 single-step match ~ 0.85 (probable; from typical perplexity-2
  entropy distribution).
- Predicted tree alpha_2 ~ 0.85^2 = 0.72  -> 38% lift in tokens/cycle.
- Pass gate: top-2 vs argmax gain > 1.15x.

**No measurement yet -- decision deferred to a follow-up cluster probe pass.**

---

## 1.3 EXECUTED (partial) -- gamma=2 verify wall measured

Captured from the same probe-on run (commit f4922bf3,
`EXO_DSV4_MTP_PROFILE=20` log lines on m4-1):

```
At cycles=140 steady-state, B=1 gamma=2 (L_q=3):
  draft    = 4.93ms  (MTP forward × 2 steps = ~2.45ms each)
  verify   = 53.38ms (main model L_q=3 forward, 43 layers, 100K KV)
  accept   = 0.91ms
  rollback = 0.18ms
  total    = 58.99ms

  Tokens/cycle (observed): 33 t/s × 58.99ms / 1000 ≈ 1.95 -- close to
  the analytic 2.11 with PROFILE overhead inflating cycle wall slightly.
```

gamma=3 (L_q=4) wall NOT measured -- Phase 1.2 already abort-gates the
plan, so the gamma=3 datapoint isn't blocking the decision. If we ever
re-open tree drafting, the gamma=3 datapoint is one cluster restart away
(set `EXO_SPECULATIVE_GAMMA=3`, run one 100K iter, grep MTP-PROF lines).

Extrapolation to L_q=7 (K=2 gamma=2 tree):
- Per-token attention cost is O(L_kv) + O(L_q), bandwidth-dominated by
  L_kv=100K, so the 4-extra-tokens delta is dwarfed by KV reads. Predicted
  verify wall ~55-60ms (vs 53.4ms at L_q=3) -- +3 to +12%.
- MoE is L_q-independent (single gather_qmm batched over L_q dim).
- Indexer TOPK does scale with L_q (~linear in number of query rows). At
  L_q=7 vs L_q=3, +4 query rows, +130% nominally, but indexer cost is a
  small fraction of attention.

Total cycle wall prediction: 59-66ms (vs current 59ms). Roughly matches
the plan's "65ms" estimate.

---

## 1.3 DESIGN NOTES (kept for reference)

Per plan: run baseline with `EXO_SPECULATIVE_GAMMA=3` for ONE iter, measure
verify wall, compare to gamma=2's verify wall.

Currently the production champion uses gamma=2. The gamma=3 wall is what
we'd extrapolate to L_q=7 (tree verify at K=2 gamma=2).

**Status: NOT EXECUTED YET.** Requires cluster restart + 1 iter at 100K
with `EXO_SPECULATIVE_GAMMA=3` set; bench output already prints verify
wall via `EXO_DSV4_MTP_PROFILE=20`.

**Prediction (analytical):**
- Per plan and the 2026-05-18/19 findings:
  - MoE is L_q-independent (batched matmul, marginal cost per token small).
  - Attention is O(L_q * L_kv) but at L_kv=100K we're bandwidth-bound on
    KV reads regardless of L_q.
  - The L_q=3 -> L_q=4 cost ratio yesterday's data implied was ~1.05-1.10x
    (small, not the 1.33x naive scaling would predict).
- For L_q=3 -> L_q=7 we extrapolate to 1.15-1.30x verify wall.
- Current verify wall ~57 ms; predicted L_q=7 wall ~65-75 ms.

**To execute later:** cluster bench with `EXO_SPECULATIVE_GAMMA=3` and
profile=20, then capture the 'verify' line of the per-cycle profile log.

---

## Open Questions resolved (from plan section "Open Questions")

- **Q1: predict_from_hidden supports batched K-way?**
  NO. Single (1,1,D) in, single (1,1,D) out. Caller loops. Plan stays:
  draft_tokens_topk calls predict() K times per parent, with cache
  snapshot/restore between branches.

- **Q2: mlx fast.scaled_dot_product_attention accepts arbitrary masks?**
  YES. From mlx docs: "If the mask is an array it can be a boolean or
  additive mask. The mask can have at most 4 dimensions and must be
  broadcast-compatible with the shape `[B, N, T_q, T_kv]`". No manual
  SDPA fallback needed.

- **Q3: How does the current code handle 0-accept?**
  L1411-1412 (greedy): `bonus_val = int(all_next[0].item())`. The cycle
  emits 1 token (the bonus = verify argmax at root). Cache trims by
  gamma. For tree drafting at 0-accept the bonus is `target_argmax[0]` =
  verify's argmax at the root position. **Tree path inherits this
  behavior unchanged.**

- **Q4: L_q=gamma+1 shape asserts?**
  No explicit `assert shape[1] == gamma + 1`. BUT implicit assumptions
  pervade `_speculative_next` (slices like `verify_logits[:, :gamma, :]`,
  `verify_logits[0, gamma]`, `verify_pre_norm[:, pos:pos+1, :]`). All
  must be rewritten in terms of tree node indices in Phase 4.

- **Q5: MTP cache clone/snapshot API?**
  NO existing clone/snapshot method. The MTP head's `_attn_mlp` uses
  `self.kv_cache.update_and_fetch(keys, values)` (mtp_module.py:459)
  which is append-only. For tree drafting we need either:
  - a `cache.snapshot()` -> returns `(keys, values, offset)` tuple,
    `cache.restore(snap)` -> trims keys/values back to snap.offset.
  - or implement clone via `cache.trim(delta)` and the standard
    append flow (already supported -- see `_speculative_next`
    L1462-1465 calls `mtp_cache.trim(rollback)`).

  **Simpler:** use `trim()` between branches. After processing branch a
  to depth 2, `mtp_cache.trim(2)` rolls back to the snapshot, then branch
  b starts from the same state. **Zero new API. Phase 2.3 simplifies.**

---

## Implementation Risk Summary (next pass priorities)

Based on the read-through, the risks in priority order are:

1. **RoPE positions for tree nodes (Phase 3.4).** Resolved -- Option A
   (reshape L_q -> batch dim, pass per-batch offset vector to mx.fast.rope).
   But the THREE attention classes each have their own `self.rope(q, offset)`
   + `self.rope(kv, offset)` + `self.rope(out, offset, inverse=True)`
   calls -- that's 9 sites to update consistently. Inconsistency here means
   bad logits -> quality probe fails.

2. **Tree mask construction (Phase 3.1, 3.2).** Resolved -- mlx SDPA accepts
   arbitrary additive masks. The mask is built once in `DeepseekV4Model.__call__`
   (L2397) and broadcast to every layer. Easiest plumbing: override the
   mask there when `tree_mask` is provided.

3. **`_speculative_next` rewrite for tree (Phase 4.2).** L1336-1480 contain
   the linear-walk accept and indexing patterns that don't translate
   directly to tree. Plan section 4.2 algorithm is correct; the indexing
   shift from "sequence position i" to "tree node index" is the work.

4. **MTP cache fanout (Phase 2.3).** Resolved -- use existing
   `mtp_cache.trim(delta)` to roll back between branches. No new API.

5. **SparseCompressedAttention TOPK with tree.** The indexer
   (`self.indexer(x, q_residual, self.rope, idx_cache, offset)`, L1965)
   uses the cache offset for compressing pool indices. **Risk:** TOPK
   selection at a tree node depends on the path-history KV, which is
   exactly what the tree mask cuts off siblings from. Need to verify
   the indexer's TOPK scoring doesn't leak across sibling branches.

   Tentative mitigation: tree drafting only changes the LOCAL attention
   mask (RotatingKVCache); the indexer compresses against the prefill
   pool which is identical for all leaves, so TOPK selection is correct
   per-leaf as long as the q at each tree node uses the right q_residual.
   Since each tree node has its own q row, and the indexer scores
   per-q-row, this should work out of the box -- but Phase 5 microbench
   must verify with the mask correctness test (test_tree_mask.py).

---

## What's NOT been done in Phase 1 yet

Phase 1.1 (this file): DONE.

Phase 1.2 alpha probe: DESIGNED, NOT YET INSTALLED. Requires a cluster
restart to run.

Phase 1.3 verify L_q scaling: DESIGNED, NOT YET INSTALLED. Requires the
same cluster restart.

Both require:
- cluster restart with the (negligibly modified) probe code
- one bench iter at 100K (~8 min)
- log capture and analysis

These are cluster ops -- pre-flight per the exo-cluster-operations skill
pitfalls (#1-12 especially) is mandatory.

**Recommendation:** Before proceeding to Phase 2 (writing code), get the
Phase 1.2 alpha probe data. The plan's pass criterion is "top-2 vs argmax
gain > 1.15x." If the real cluster measurement says <1.10x, Phase 2-7
are wasted work and the plan needs revision (e.g. K=3 instead of K=2,
or beam-search-flavored draft instead of pure top-K).

---

## RECOMMENDATION: ABORT Phase 2-7 of the token-tree plan

Per the plan's own gate (section 1.2, line 150-151):
> If top-2 vs argmax gain is < 1.10x, the tree approach is NOT going to
> hit 35 t/s. Stop and reconsider.

**Measured step-0 lift = 1.085x, below the gate.** Combined tokens/cycle
lift is +9.85%, predicted t/s ceiling 33.0 (best case), well below the
35 t/s plan target. K=3 gamma=2 also fails (~34 t/s ceiling).

### Alternatives to explore (not in this plan)

Given that the chained-MTP step 1 is the alpha-limiting step
(P(top1)=0.42), levers that improve step-1 prediction quality directly
would have outsized impact. Some options to investigate in a follow-up:

1. **Train (or fine-tune) a 2nd MTP head specifically for step-1
   prediction.** Architecturally: the existing single MTP head was
   trained on next-token (=step-0) prediction. Chaining it for step 1
   re-uses the head with degraded inputs (its own argmax + post-norm
   hidden). A dedicated step-1 head could lift P(top1) from 0.42 to
   maybe 0.55-0.60 (analytic upper bound: matches single-step P(top1)).
   At step1 P(top1)=0.55, linear gamma=2 tokens/cycle = 1 + 0.78 + 0.78×0.55
   = 2.21 (vs current 2.11) ≈ +4.7%. Combined with tree K=2 at the new
   alpha: 1 + 0.85 + 0.85 × 0.65 ≈ 2.40 ≈ +13.8% over current → ~34 t/s
   ceiling. Still not 35.

2. **Eagle-style continuous step-1 hidden refinement.** Instead of feeding
   the MTP head's own argmax embedding as step-1 input, feed it the FULL
   step-0 logit distribution (or top-K embeddings averaged with their
   probs). Eagle paper claims this improves step-N draft alpha. Adds
   forward-pass complexity but no extra training.

3. **Reduce verify wall (the bigger lever).** Current cycle = 59ms with
   verify=53ms (~90% of cycle). Even a 10% verify reduction (=5ms saved,
   cycle = 54ms) lifts t/s from 30 → 33 with no alpha change.
   Levers: indexer microkernel fusion, sparse-attention TOPK reduction
   (with quality preservation), MoE residual sharding (memory bw), JACCL
   collective fusion. The 2026-05-18 / 2026-05-19 work already explored
   most of these and found small to negative.

4. **Concurrency increase (c=2 production).** At c=2 the verify wall
   amortizes over 2 streams: 33 → ~50 t/s aggregate (per stream falls
   to ~25 but throughput rises). Plan doesn't care about c=1 specifically;
   if 35 t/s aggregate is the metric, c=2 might already hit it. Check
   `EXO_DSV4_BATCHED_PREFILL=1` benches.

5. **Accept the 30 t/s ceiling.** Per memory facts, the 2026-05-18
   champion (`baseline-2026-05-18-mtp-g2-topk512-30.06`) is the
   quality-correct production baseline. The user's stated target was
   35 t/s; we've now done the work to know K=2 tree won't get there.
   Tree gives ~33 t/s (10% lift) -- if that's worth ~20h of risky
   implementation, the plan can still proceed knowing the cap.

### What to do RIGHT NOW

Bring this analysis to the user. Three reasonable next moves:
- **Pause** the tree-drafting effort, pursue alternatives 1-3 above.
- **Continue** the tree-drafting effort knowing the ceiling is ~33 t/s
  (still a real +10% win, just not the 35 t/s target).
- **Re-measure** with more / different prompts (the n=60 sample is small
  and from a single fixed prompt; alpha may vary).

The probe code itself is committed and can be re-enabled at any time:
just set `EXO_DSV4_TREE_ALPHA_PROBE=1` on cluster start.

---

## Phase 1 -> Phase 2 handoff checklist

For the next coding session to start Phase 2 cleanly:

- [x] All line numbers from the plan re-verified (and corrected where wrong).
- [x] Tensor shapes/dtypes at each cycle stage documented.
- [x] Mask construction site identified (single chokepoint at
      DeepseekV4Model.__call__:2397).
- [x] RoPE per-position strategy decided (Option A: L_q -> batch reshape).
- [x] MTP cache fanout strategy decided (use existing `trim()`).
- [x] All 5 plan open-questions answered.
- [x] Phase 1.2 alpha probe results captured: step0 lift 1.085x (BELOW
      1.10x abort gate), step1 lift 1.32x (above). Combined +9.85%
      tokens/cycle, predicted t/s ceiling 33.0 vs 35 target. ABORT.
- [x] Phase 1.3 verify wall captured at gamma=2 = 53.4ms. gamma=3 not
      measured (Phase 1.2 already abort-gates).

**Conclusion: Phase 1.2 abort gate triggered. Phase 2-7 should NOT proceed
without revising the plan to target ~33 t/s instead of 35 t/s, or pivoting
to one of the alternative levers in the "RECOMMENDATION" section above.**

(End of Phase 1 findings.)
