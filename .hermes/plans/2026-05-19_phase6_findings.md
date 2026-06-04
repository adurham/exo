# Phase 6 Findings: Token-Tree Drafting Cluster Deployment

Status: **PARTIAL - production-config quality degraded; needs sparse-path debug.**

Code is structurally complete, all 6 Phase 5.2 microbench tests pass at the
unit level (including the BatchRotatingKVCache 4-D mask path that crashed
the cluster on first deploy), but the deployed tree path produces wrong
logits at production config.

Commits this session (all on origin/main):
- ac727f58, d1b09288, f4922bf3: Phase 1.2 alpha probe + graph-leak fix
- 41ae6e82: Phase 2-5 (draft_tokens_topk, side-channel, _rope_dispatch,
  _speculative_next_tree, bench/test_tree_mask.py, start_cluster.sh forwarding)
- d0c92340: sliding-window kv-clamp fix (first deploy crash:
  `(1,1,7,69328) vs (1,64,7,134) broadcast`)
- 46c87b56: 4-D BatchRotatingKVCache mask squeeze (second deploy crash:
  `concatenate dim 4 vs 2`)
- 0d3d17f5: EXO_DSV4_TREE_DEBUG one-shot first-cycle log

mlx-lm fork on `main` at 7f80fffc (adds `_TREE_VERIFY_CTX`,
`_set_tree_verify_ctx`, `_rope_with_positions`, `_rope_dispatch`, and
10 RoPE call-site rewrites).

---

## What works

### Microbench (bench/test_tree_mask.py, runs locally without model load)

All 6 tests PASS:
- `test_tree_mask_structure`: K=2 g=2 ancestor mask correct.
- `test_rope_per_token_positions`: max abs diff 0.00e+00 vs scalar-loop rope.
- `test_same_depth_siblings_share_rope`: bit-equiv.
- `test_sdpa_with_tree_mask`: tree-mask SDPA at leaf node matches the
  equivalent linear-causal SDPA along the same path to **1.19e-07** (fp32
  noise level).
- `test_tree_mask_sliding_window_clamp`: at offset=69321 max_size=128 emits
  mask shape (7, 134), positions un-clamped at offset+depth.
- `test_tree_mask_4d_base_input`: BatchRotatingKVCache-style 4-D input
  correctly squeezes to 2-D, then splices in the tree sub-mask.

### Cluster doesn't crash

After the two iterative fixes (sliding-window clamp, 4-D mask squeeze),
the cluster runs the tree path to completion without throwing.
EXO_DSV4_TREE_DRAFT=1 is honored end-to-end (env var forwarded by
start_cluster.sh, gate triggers in `_speculative_next`, the side
channel is installed/cleared cleanly around the model forward).

### Cluster generates output at small prompt

```
Sanity test (~30-token prompt, max_tokens=64, EXO_DSV4_TREE_DRAFT=1 K=2):
  baseline (no tree):  47.8 t/s, "We need ... Key points: caches are
                                  small, fast memory close to CPU cores;
                                  hierarchy (L1, L2, L3..."  COHERENT
  tree-K2:             27.7 t/s, "to write a 5-sentence paragraph about
                                  to write a 5-sentence paragraph about
                                  need to write..."  COHERENT-BUT-LOOPY
  mtp_cycles=63, mtp_accepted_drafts=66 -> 1.05 drafts/cycle
```

Acceptance is near-linear (1+1.05 ≈ 2.05 tokens/cycle); plan predicted
~2.32 tokens/cycle for tree. Output is qualitatively wrong (loops on
prompt fragments).

### Cluster generates output at 100K prompt

```
Quality probe (FALCON-MERCURY needle at ~69K prompt, max_tokens=64):
  baseline: needle_found=True, response="FALCON-MERCURY-7749", clean.
  tree-K2:  needle_found=False, response=" secret".
            completion_tokens=64 (model emitted 64 tokens that
            de-tokenize to one word + EOS/PAD spam).
```

The tree path is functionally incorrect at production config (TOPK=512
sparse, sliding_window=128, 100K context).

---

## Why microbench didn't catch this

The microbench exercises:
- Tree mask construction at 2-D, 4-D, sliding-window-clamped shapes.
- `_rope_with_positions` per-token positions.
- Plain SDPA with the tree mask (matches linear-causal at the leaf).

The microbench does NOT exercise:
- **`_sparse_pooled_attention`** (TOPK=512 path used at 100K production).
  Takes a separate `pooled_mask` derived from `pool_cache.make_mask(L_q,
  offset)`. The pmask is causal-by-row-index, NOT by-depth -- so same-
  depth tree siblings get different pool-attend rows.
- **`PoolingCache.make_mask`** under tree-input. Pool_cache's machinery
  doesn't know about tree topology; it just uses the raw L_q query rows.
- **`Indexer.__call__`** with per-token-positioned Q. The pool keys it
  scores against were rotated at the COMPRESSOR's standard linear
  positions; mismatched rotations against tree-rotated Q.
- **`Compressor.__call__`** side-effect on pool_cache from tree-input
  embeddings (writes ~floor(L_q/compress_ratio) entries derived from
  tree tokens, NOT linear tokens -- subtle context contamination if
  the cycle ends without trimming the pool cache).
- **DeepseekV4ShardingStrategy** TP cross-rank effects with the side
  channel (we only test on a single laptop; cluster has 2x ranks).

Each of those is a likely site of the production-config wrong-logit bug.
The fact that the SMALL prompt (~26 tokens, no sliding-window, possibly
no sparse path) shows degradation but not collapse suggests at least
ONE of these paths is firing even at small scale -- probably the
Compressor / PoolingCache pair, which engages whenever the layer has
compress_ratio > 0 (= 42/43 DSv4 layers).

---

## Recommended next-session attack

1. **Validate without sparse**: at small prompt (1-4K tokens), bench
   tree-on vs tree-off, side-by-side same prompt with temp=0. If
   tree-on output differs from tree-off, the bug is in the
   pool_cache / compressor / indexer path even without sparse.

2. **Disable the indexer**: set
   `EXO_DSV4_NOP_SPARSE_LAYERS=0,1,2,...,42` (or use the file mechanism
   in /tmp/dsv4_nop_targets to "indexer" + "sparse_attn") to force
   the model to use ONLY local attention. If output normalises to
   coherent at this config, the bug is in the indexer / pool_cache
   path. (This also tanks quality of course, but the goal here is to
   bisect which subsystem the bug lives in.)

3. **Audit pool_cache.make_mask**: when we pass a tree input to a
   sparse-attention layer, what shape mask does it return? Is it
   causal-by-row in a way that conflicts with same-depth-siblings?
   The fix likely involves passing tree depth to the pool_cache's
   mask builder, or post-processing the pmask to be per-depth.

4. **Audit Compressor**: does it consume the tree input and write
   non-causal pool entries? If yes, the pool_cache state is corrupted
   for subsequent cycles. We may need to call `pool_cache.trim(...)`
   after the tree verify, similar to how we trim the prompt local
   caches.

5. **Add cluster-grade microbench**: run the model on the laptop with
   a tiny prompt (~256 tokens, single-node), tree-on vs tree-off,
   diff the logits. If they diverge, we have a controlled-environment
   repro that doesn't require burning 5-10 min per cluster restart.
   This is the right level of test that Phase 5 should have included.

---

## Tags / artifacts

- exo: `46c87b56` (HEAD with all Phase 2-5 + sliding-window fix +
  4D mask squeeze) -- broken at production config.
- exo: `0d3d17f5` -- adds EXO_DSV4_TREE_DEBUG diagnostic.
- mlx-lm: `7f80fffc` -- adds tree-verify side channel + rope dispatch.
- baseline-2026-05-18-mtp-g2-topk512-30.06 -- the production champion
  that still works at exo c7032932 / mlx-lm 6dcdd40a.

To pick up next session:
```bash
cd ~/repos/exo
git log --oneline 41ae6e82^..HEAD
# inspect commits 41ae6e82, d0c92340, 46c87b56, 0d3d17f5
.venv/bin/python bench/test_tree_mask.py  # verify microbench still passes
```

To bench again:
```bash
EXO_DSV4_TREE_DRAFT=1 EXO_DSV4_TREE_K=2 \
  EXO_SPECULATIVE=1 EXO_DSV4_MTP=1 EXO_SPECULATIVE_GAMMA=2 \
  EXO_DSV4_INDEX_TOPK=512 ./start_cluster.sh
# then python bench/quality_probe_dsv4.py --target-tokens 100000
```

To revert to known-good baseline:
```bash
cd ~/repos/exo
git checkout baseline-2026-05-18-mtp-g2-topk512-30.06
# (or just unset EXO_DSV4_TREE_DRAFT — default OFF preserves baseline)
```

(End of Phase 6 findings.)
