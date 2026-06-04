# Token-Tree Drafting Continuation (Phase 6 debug + Phase 7 sweep)

For Hermes: this picks up from 2026-05-19. Code is structurally complete
(Phases 1-5) and microbench-clean, but produces wrong logits at the
production config. Estimated total: 5-8h.

The bug LIKELY lives in `PoolingCache.make_mask` (row-causal pmask doesn't
know about tree topology), but Phase 6A bisects to confirm before any
code changes.

---

## State at session start

- exo HEAD: `0d3d17f5` (origin/main).
- mlx-lm HEAD: `7f80fffc` (adurham/mlx-lm main).
- Production baseline preserved: `EXO_DSV4_TREE_DRAFT` default OFF keeps
  `baseline-2026-05-18-mtp-g2-topk512-30.06` bit-exact.
- Phase 5.2 microbench: 6/6 pass.
- Phase 6 deploy: runs without crash; produces wrong logits at production:
  - small prompt (~26 tokens): 27.7 t/s coherent-but-loopy
    ("to write a 5-sentence paragraph about ..." repeating).
    Baseline = 47.8 t/s clean.
  - 100K needle probe: needle_found=False, response=" secret".
  - acceptance: 1.05 drafts/cycle (predicted 1.32).
- Prior session forensics: `.hermes/plans/2026-05-19_phase6_findings.md`.

---

## Pre-flight (10 min)

```bash
cd ~/repos/exo
# 1. Confirm versions.
git log -1 --oneline                                # expect 0d3d17f5 (or descendant)
cd mlx-lm && git log -1 --oneline && cd ..          # expect 7f80fffc

# 2. Microbench still passes.
.venv/bin/python bench/test_tree_mask.py            # expect 6/6 PASS

# 3. mlx-lm in venv is the patched version (uv sync may have refreshed).
grep -c "_set_tree_verify_ctx\|_rope_dispatch" \
  .venv/lib/python3.13/site-packages/mlx_lm/models/deepseek_v4.py
# expect >= 12 hits
```

If microbench fails or venv doesn't have the side channel, re-run
`uv lock --upgrade-package mlx-lm && uv sync`.

---

## Phase 6A: Bisect WHICH subsystem is wrong (1-2h)

Hypothesis ordered by likelihood:
1. **PoolingCache pmask**: row-causal-by-row-index, breaks for same-depth
   tree siblings (they get DIFFERENT pool-attend patterns despite sharing
   a RoPE position).
2. **Compressor side-effect**: writes tree-derived entries into
   pool_cache; not rolled back at cycle end like prompt_cache is.
3. **Indexer**: tree-rotated Q vs prefill-rotated pool keys. Probably
   correct (relative rotation matches) but verify.

### A.1 Baseline reference (no tree)

Restart cluster WITHOUT tree to capture the reference behavior:

```bash
EXO_SPECULATIVE=1 EXO_DSV4_MTP=1 EXO_SPECULATIVE_GAMMA=2 \
  EXO_DSV4_INDEX_TOPK=512 ./start_cluster.sh   # baseline
```

Wait READY, then:

```bash
.venv/bin/python /tmp/probe_sanity.py    # if missing, copy from prior session
# OR write a fresh one: 5-sentence paragraph prompt, temp=0, max_tokens=64
```

Save the output text. Expected: ~47 t/s, coherent paragraph about CPU caches.

### A.2 Tree-on + EXO_DSV4_TREE_DEBUG=1

Restart with tree + debug:

```bash
EXO_SPECULATIVE=1 EXO_DSV4_MTP=1 EXO_SPECULATIVE_GAMMA=2 \
  EXO_DSV4_TREE_DRAFT=1 EXO_DSV4_TREE_K=2 \
  EXO_DSV4_TREE_DEBUG=1 \
  EXO_DSV4_INDEX_TOPK=512 ./start_cluster.sh
```

Run sanity test. SSH to m4-1 + m4-2 and check for `[TREE-DEBUG] first cycle:`
line in `~/exo.log`. Confirm:

- n_nodes = 7
- parent_idx = [-1, 0, 0, 1, 1, 2, 2]
- depth = [0, 1, 1, 2, 2, 2, 2]
- mask.shape = (7, 134) [or whatever clamp matches sliding_window + L_q]
- positions[0] = offset (the prefill end), positions[1]==positions[2],
  positions[3:7] are all positions[1]+1

If ANY of these are wrong, fix `_speculative_next_tree` setup and re-run.
Don't proceed to A.3 until the debug line confirms the inputs to the
model forward look right.

### A.3 NOP-out subsystems to bisect

WITHOUT restarting (use the file-based NOP machinery; 1-sec TTL):

```bash
# Run sanity test between each step.

# Step 1: NOP indexer + sparse_attn (forces compressed-attention path).
for h in 192.168.86.201 192.168.86.202; do
  ssh -i ~/.ssh/exo_cluster -o IdentitiesOnly=yes adam.durham@$h \
    'echo "indexer,sparse_attn" > /tmp/dsv4_nop_targets'
done
sleep 2
.venv/bin/python /tmp/probe_sanity.py
# Output coherent? -> bug is in indexer/sparse_attn.
# Still loopy? -> bug is upstream (compressor or pool_cache).

# Step 2: ALSO NOP compressed_attn (forces local-only).
for h in 192.168.86.201 192.168.86.202; do
  ssh -i ~/.ssh/exo_cluster -o IdentitiesOnly=yes adam.durham@$h \
    'echo "compressed_attn,indexer,sparse_attn" > /tmp/dsv4_nop_targets'
done
sleep 2
.venv/bin/python /tmp/probe_sanity.py
# Output coherent (matches baseline-tree-off) -> bug is in
#   pool_cache/compressor (i.e., the compressed-attention or sparse
#   branch we just NOP'd).
# Still loopy -> bug is in LOCAL attention itself (means mask or RoPE
#   has a residual issue the microbench missed — re-audit Phase 5.2).

# Cleanup.
for h in 192.168.86.201 192.168.86.202; do
  ssh -i ~/.ssh/exo_cluster -o IdentitiesOnly=yes adam.durham@$h \
    'rm -f /tmp/dsv4_nop_targets'
done
```

Record the bisect result. Most likely outcome: bug is in pool_cache /
compressor side (i.e., step 2 produces coherent output but step 1
doesn't help much).

### A.4 OPTIONAL: tier-2 microbench

If bisect is ambiguous, write `bench/tier2_tree_logits_diff.py` that runs
ON the cluster (m4-1, single-node, no jaccl) and loads DSv4. Steps:
1. Load `mlx-community/DeepSeek-V4-Flash-8bit` on m4-1.
2. Prefill a 500-token prompt.
3. Run linear gamma=2 forward, capture verify_logits.
4. Run tree K=2 g=2 forward with the same prompt + same y_val,
   capture verify_logits.
5. Diff verify_logits[0, 0, :] (= bonus token logits at root) — should
   be bit-equiv since both forwards have identical input at root.
6. Diff verify_logits[0, accepted_path[-1], :] vs the linear equivalent
   for the matching path — should match to ~1e-5 fp16/bf16 noise.

If logits diverge, the diff TELLS you which layer goes wrong (run
through with `EXO_DSV4_FENCE_EVERY_N_LAYERS=1` for per-layer fences and
log per-layer hidden norms).

Time budget: 1h for tier-2 microbench scaffolding + run.

---

## Phase 6B: Fix the identified subsystem (2-4h)

### B.1 If pool_cache pmask is wrong (most likely path)

Read `mlx_lm/models/cache.py` `PoolingCache.make_mask` (cache.py:1222).
It's row-causal by `offset + row_index`:

```
pmask[i, j] = (pool position j's "kv-window end") < (offset + i)
```

For tree, want depth-based:
```
pmask[i, j] = (pool position j's "kv-window end") < (offset + depth[i])
```

Implementation strategy:
1. Extend `_TREE_VERIFY_CTX` (mlx-lm/mlx_lm/models/deepseek_v4.py:153) with a
   `depth` field (Python list of ints, length L_q).
2. `_speculative_next_tree` (dsv4_mtp.py) sets `depth` alongside mask + positions.
3. In SparseCompressedAttention.__call__ (deepseek_v4.py:~2049), CompressedAttention.__call__
   (deepseek_v4.py:~1924), AND Indexer.__call__ (deepseek_v4.py:1695):
   replace `pool_cache.make_mask(L, offset)` with:

```python
if _TREE_VERIFY_CTX.get("depth") is not None and pool_cache is not None:
    pmask = _tree_pmask(pool_cache, _TREE_VERIFY_CTX["depth"], offset)
elif pool_cache is not None:
    pmask = pool_cache.make_mask(L, offset)
else:
    pmask = None
```

`_tree_pmask` is a new helper:

```python
def _tree_pmask(pool_cache, depth: list[int], offset) -> mx.array:
    """Tree-aware pmask: each query row uses depth[i] for the causal cutoff
    instead of row_index. Same-depth siblings get IDENTICAL rows."""
    L_q = len(depth)
    # Build (L_q, pool_size) bool by stacking single-row make_masks at
    # the right (depth-adjusted) offset.
    rows = []
    for i in range(L_q):
        # row i uses depth[i] as if it were the "query position" in
        # pool_cache's row-causal logic.
        single = pool_cache.make_mask(1, int(offset) + int(depth[i]) - int(offset))
        # ^ pool_cache.make_mask wants offset to be the query-side
        # position; for tree we substitute offset + depth[i].
        rows.append(single.reshape(-1) if single is not None else None)
    if any(r is None for r in rows):
        return None
    return mx.stack(rows, axis=0)
```

(Exact signature depends on what `PoolingCache.make_mask` does; read it
first. If make_mask(L=1, offset=X) returns shape (1, pool_size), use
that directly. If it's already broadcast-friendly, the override may be
simpler.)

Add a new microbench test that builds a fake PoolingCache and verifies
same-depth siblings get identical pmask rows.

### B.2 If compressor side-effect is wrong

Compressor (deepseek_v4.py:1248-1356 ish) calls
`pool_cache.update_and_fetch(...)` per layer. During the tree verify,
it writes ~floor(L_q/compress_ratio) entries to each layer's pool_cache.
For compress_ratio = 16 and L_q = 7, that's 0 entries — usually safe.
BUT: at boundary cases where the verify spans a chunk boundary
(verify_input's first token completes a chunk that started in prefill),
the compressor writes ONE entry whose contents include sibling-branch
contamination.

Fix: snapshot every pool_cache's offset BEFORE the verify, restore
AFTER:

```python
# In _speculative_next_tree, BEFORE dsv4_speculative_forward:
pool_offsets_before: list[Optional[int]] = []
for c in gen_batch.prompt_cache:
    if isinstance(c, CacheList) and len(c.caches) > 1:
        # caches: [RotatingKVCache, PoolingCache, (optional) PoolingCache for idx]
        pool_offsets_before.append(c.caches[1].offset)
    else:
        pool_offsets_before.append(None)

# ... do the verify ...

# AFTER dsv4_speculative_forward (in addition to the existing
# prompt_cache trim by n_nodes):
for i, c in enumerate(gen_batch.prompt_cache):
    saved = pool_offsets_before[i]
    if saved is None or not isinstance(c, CacheList) or len(c.caches) <= 1:
        continue
    pool_cache = c.caches[1]
    delta = int(pool_cache.offset) - int(saved)
    if delta > 0:
        if hasattr(pool_cache, "trim"):
            pool_cache.trim(delta)
        elif hasattr(pool_cache, "offset"):
            pool_cache.offset -= delta
    # Also handle the indexer pool (caches[2]) if present.
    if len(c.caches) > 2:
        idx_cache = c.caches[2]
        # ... same trim logic ...
```

### B.3 If indexer is wrong

The indexer rotates Q with tree positions and scores against pool keys
that were rotated at prefill positions. Relative rotation matches IF
both Q and K use the same RoPE base. Audit:
- `Indexer.__call__` line 1686: `_rope_dispatch(position_rope, q, offset)`
  — Q gets tree positions. OK.
- The pool keys: `pooled = self.compressor(x, pool_cache, offset)` line
  1680. The compressor rotates new pool entries at the offset passed in;
  for verify with tree input, this would rotate at the SCALAR offset
  (= cache.offset, the prefill end). NOT tree depth.
- This is technically wrong: a depth-2 sibling Q at position offset+2
  scoring against a NEW pool entry rotated at position offset gives a
  different result than the linear case (Q at offset+2 vs key at
  offset+2 or whatever). BUT: the compressor only writes pool entries
  on chunk boundaries. At typical decode (1 to 7 new tokens added to
  prefill cache at a time), no new pool entries are written. So this
  rarely fires.

Audit only if A.4 bisect implicates the indexer.

---

## Phase 6C: Re-bench + validate (1h)

After fixing:

1. **Microbench**: 6/6 still pass + any new tests from B.1.
2. **Sanity test small prompt** (tree-on): output coherent, MATCHES the
   tree-off reference output (or close).
3. **Quality probe at 100K**:
   ```
   .venv/bin/python bench/quality_probe_dsv4.py \
     --base-url http://192.168.86.201:52415 \
     --target-tokens 100000 --max-tokens 64 \
     --label tree-K2-g2-fixed --out /tmp/quality_probe_fixed.json
   ```
   Pass criterion: `needle_found=True`, response contains
   `FALCON-MERCURY-7749`.

4. **Cluster bench** (only after quality passes):
   ```
   .venv/bin/python bench/concurrent_bench.py \
     --host 192.168.86.201 --port 52415 \
     --model mlx-community/DeepSeek-V4-Flash-8bit \
     --concurrency 1 --iterations 3 --warmup 1 \
     --max-tokens 128 --prompt-words 75000 \
     --timeout 3600 \
     --label tree-K2-g2-fixed \
     --json-out /tmp/bench-tree-K2-g2-fixed.json
   ```

GO/NO-GO gates:
- Quality FAIL: keep debugging, don't proceed.
- Quality PASS + t/s < baseline 30: investigate why; tree should have
  positive lift. Possibly a remaining mask/rope subtlety.
- Quality PASS + t/s in [30, 33]: success at the predicted ceiling.
- Quality PASS + t/s >= 33: above prediction. Suspicious; double-check
  acceptance histogram to make sure it's not just running fewer cycles
  (= effectively gamma=0). hist should show k=2 cases.

Acceptance gate per memory:
- gamma=2 production fix needs **>=10 iters all >= 29 t/s sigma < 0.5**
  before claiming a fix (memory says >=5 iters at FINAL production
  config). For this session, 3 iters is OK to declare "promising"; don't
  tag as champion until 10-iter validation.

---

## Phase 7: K/gamma sweep + tag (1h)

Only if 6C passes quality + decent t/s.

```bash
for K in 2 3; do
  EXO_SPECULATIVE=1 EXO_DSV4_MTP=1 EXO_SPECULATIVE_GAMMA=2 \
    EXO_DSV4_INDEX_TOPK=512 \
    EXO_DSV4_TREE_DRAFT=1 EXO_DSV4_TREE_K=$K \
    ./start_cluster.sh
  sleep 60
  .venv/bin/python bench/quality_probe_dsv4.py \
    --target-tokens 100000 --max-tokens 64 \
    --label tree-K$K --out /tmp/quality-K$K.json
  .venv/bin/python bench/concurrent_bench.py \
    --host 192.168.86.201 \
    --model mlx-community/DeepSeek-V4-Flash-8bit \
    --concurrency 1 --iterations 5 --warmup 1 \
    --max-tokens 128 --prompt-words 75000 --timeout 3600 \
    --label tree-K$K --json-out /tmp/bench-tree-K$K.json
done
```

Pick the K with highest mean t/s at sigma < 0.5 and quality passing.

Tag:
```bash
git tag tree-draft-2026-MM-DD-g2-K${best_K}-${mean_tps}
git push origin tree-draft-...
# Update warm memory with new baseline.
```

---

## Pitfalls to remember

1. **mlx-lm changes**: commit + push to adurham/mlx-lm + run
   `uv lock --upgrade-package mlx-lm` LOCALLY + commit uv.lock to exo +
   push exo. start_cluster.sh's `uv sync` on the studios then picks up
   the new commit. Verify with `grep -c "<new symbol>" .venv/lib/...`.

2. **Probe-graph-leak trap** (forensics: warm fact 165): NEVER store
   lazy mx.arrays in Python lists across spec cycles. Always materialise
   to `int` / `list[int]` at capture time. The MTP forward will produce
   all-zero logits otherwise.

3. **Cluster restart ~3 min, quality probe ~5 min, 3-iter bench ~25 min**
   at 100K. Each fix-and-redeploy is ~30 min minimum end-to-end.

4. **Don't bench without quality gate passing.** Memory has multiple
   examples of TOPK=160-style "wins" that were actually broken-needle
   output.

5. **Default-OFF discipline**: `EXO_DSV4_TREE_DRAFT` MUST stay default
   off. Production cluster restarts during/after this work should
   produce baseline-2026-05-18 perf bit-exactly when the env var is
   unset. Verify with one no-env baseline bench at the end.

6. **SSH-agent flake** (pitfall #12): after ~1h heavy cluster driving,
   use `ssh -i ~/.ssh/exo_cluster -o IdentitiesOnly=yes ...` to bypass
   the rate-limited 1Password agent.

7. **mlx-lm `origin` remote is now HTTPS** on the laptop (was SSH and
   failed yesterday under agent rate-limit). If push fails, check
   `git remote -v` says `https://github.com/adurham/mlx-lm.git`.

8. **Pool-cache audit shortcut**: if Phase 6A bisect points at the pool
   path, look at `EXO_DSV4_INDEXER_WINDOW` env var — at production it's
   unset (= 0 = unbounded), but historical configs used 8192. May
   interact with tree topology in surprising ways. Document & test both.

---

## Success criteria (in priority order)

1. **Quality**: needle probe returns `FALCON-MERCURY-7749` at 100K.
2. **No regression**: baseline (tree-off) t/s still ~30.06.
3. **Tree perf**: tree-on t/s >= 33 (3-iter mean, sigma < 0.5).
4. **10-iter validation**: before tagging champion, 10/10 clean at
   chosen K config. Per memory: gamma=2 production-fix requires this.
5. **Documented**: new Phase 6 findings + a skill update if the bug
   is interesting enough (it probably is — pool_cache + tree drafting
   is a generalisable mlx-lm gotcha).

If 1-3 pass but 4 fails (variance issue): still a real perf win, just
not a stable champion. Continue iterating or accept as gamma=2 best-effort
and document the variance.

If 1 fails after Phase 6B: time-box another 4h, then revert tree-draft
entirely and chase the alternative levers from the Phase 1 findings
(dedicated step-1 MTP head, Eagle-style hidden refinement, or just
accept the 30.06 ceiling).

(End of continuation plan.)
