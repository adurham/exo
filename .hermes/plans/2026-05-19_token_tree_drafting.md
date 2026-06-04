# Token-Tree Drafting for DSv4 -- Path to 35+ t/s at c=1 100K Quality-Correct

For Hermes (next session): This is a multi-day implementation plan,
NOT a tonight job. Estimated ~25 hours of focused engineering work
(~3 working days). Investigation-first: Phase 1 is pure code reading
and small-scale validation BEFORE any cluster work. Phase 5 microbench
is the gate -- no cluster bench until microbench shows the tree verify
produces sensible outputs.

Critical context: 2026-05-18 and 2026-05-19 sessions exhausted the
compile-boundary collapse levers (Lever 1+2 of the earlier plan --
both regressed). The all_sum collective is ~1.9 percent of verify
cost (not a target). The MoE NOP probe showed only 7.5ms wall savings
from removing the WHOLE MoE -- Apple gather_qmm is already pipelined
optimally. The only realistic structural path to 35+ t/s with the
existing single-head MTP is token-tree drafting.

Goal: mean >= 35 t/s, >= 10 iters, sigma < 0.3, 0 errors, at
gamma >= 2, 100K context, c=1, with EXO_DSV4_INDEX_TOPK=512
(model-correct quality, needle probe passes).

Baseline anchor: baseline-2026-05-18-mtp-g2-topk512-30.06
(30.062 t/s sigma=0.059, 10/10 clean, mlx-lm 6dcdd40a with mc_ping
removed and allsum probe code inactive, mlx main facbed9a, exo
c7032932).

Quality gate (mandatory): bench/quality_probe_dsv4.py must return
needle_found=True and the exact needle string FALCON-MERCURY-7749
after every config change before declaring any t/s number real.
TOPK=160 is confirmed BROKEN (BOS-only output at 100K) -- all
earlier champion tags at TOPK=160 are invalid.

Reference background: Eagle / Medusa / tree-attention papers on
speculative decoding. Math: instead of gamma chained linear drafts
with acceptance alpha_2 = 0.52 giving 2.04 tokens/cycle, generate
a TREE of 2^gamma candidates verified in one batched forward,
effective alpha_branched ~ 0.65 giving 2.5+ tokens/cycle.

---

## Architectural Overview

### Current linear speculation (production today, baseline-2026-05-18)

```
Cycle:
  1. draft_tokens(gamma=2): MTP head emits 2 sequential argmax tokens
     - draft_step_0: input=[last_committed], output=tok_d0
     - draft_step_1: input=[tok_d0], output=tok_d1
  2. verify_forward(input=[last_committed, tok_d0, tok_d1]): main model
     - shape (1, 3) into 43 layers, produces logits at each position
     - target_tokens = argmax(logits) at each position
  3. accept_loop: walk left-to-right
     - if target_tokens[0] == tok_d0: accept tok_d0, n=1
     - if target_tokens[1] == tok_d1: accept tok_d1, n=2
     - emit bonus_token = target_tokens[n] (free token from verify)
     - n_emitted = n + 1
  4. cache.trim(gamma - n) to discard unaccepted positions

Measured: alpha_2 = 0.52, tokens/cycle = 2.04, cycle = 62.65ms,
          t/s = 30.06.
```

### Proposed tree speculation

```
Cycle:
  1. draft_tree(K=2, gamma=2): single-head MTP emits top-K at each step
     - draft_step_0: input=[last_committed], output = top-2 tokens (a, b)
                     each spawns a branch
     - draft_step_1: TWO drafts run in parallel from a and b
       - from a: top-2 = (a_x, a_y)
       - from b: top-2 = (b_x, b_y)
     - Tree has 7 nodes: [root] [a, b] [a_x, a_y, b_x, b_y]
     - 4 leaf-paths: [a, a_x], [a, a_y], [b, b_x], [b, b_y]
  2. verify_forward(tree_input, tree_mask):
     - shape (1, 7) into 43 layers, but tree_mask ensures each leaf
       only sees its own ancestors (not sibling branches)
     - produces 7 logit positions
  3. tree_accept: find the longest leaf-path matching target argmax
     - target[0] tells us which of (a, b) to walk
     - target[branch] tells us which leaf in that branch
     - accept the deepest matching path
     - emit bonus_token = target_logits at the FINAL accepted node
  4. cache.trim_to_path(accepted_path): keep only ancestors of leaf

Target: alpha_branched ~ 0.65, tokens/cycle ~ 2.5-2.7, cycle ~ 65ms,
        t/s = 38-40.
```

### Why tree drafting can lift alpha

The single MTP head was trained for 1-step prediction. Its argmax is
the best single guess, but the head has substantial mass on the top-2
(typical perplexity-2 entropy: P(top-1) ~ 0.6, P(top-2) ~ 0.15). The
verifier sometimes disagrees with argmax at draft_step_0 -- when it
does, the linear scheme breaks. Top-2 catches the disagreement: if
verifier argmax happens to be the MTP top-2 choice, we still accept
that branch.

Naive estimate: if linear single-step match is ~0.72 (alpha_2 = 0.52
= 0.72^2), then top-2 branched single-step match becomes ~0.85, tree
alpha_2 ~ 0.85^2 = 0.72. That is a 38 percent lift in tokens/cycle.

Verify cost grows from L_q=3 to L_q=7 -- about 2x in input length,
but verify was not the bottleneck. MoE runs once per layer regardless
of L_q via batched matmul; the mask work is cheap. Estimated wall
growth: 57ms to ~65ms.

---

## Phase 1: Investigation (3h, NO code changes)

Goal: Confirm every assumption above by reading code. If any
assumption is wrong, this plan needs revision BEFORE coding starts.

### 1.1 Map the existing speculative path end-to-end

Files to read (all under /Users/adam.durham/repos/exo/):

- src/exo/worker/engines/mlx/speculative/dsv4_mtp.py
  - draft_tokens() ~ line 581 -- linear MTP draft loop
  - _speculative_next() ~ line 1255 -- verify + accept orchestration
- src/exo/worker/engines/mlx/speculative/mtp_module.py
  - predict_from_hidden() ~ line 536 -- exists but never called yet
- mlx-lm/mlx_lm/models/deepseek_v4.py
  - main __call__ with the verify pass
  - attention mask construction (search for causal_mask, mx.tril,
    additive_mask)
- src/exo/worker/engines/mlx/patches/qwen3_5_moe/batched_moe.py
  - PRIOR ART: Qwen3.5 token-tree / batched verify pattern; mine for
    mask shape, branch indexing, accept logic

Output: write phase1_findings.md listing for each file:
- exact line numbers of the functions named above
- the shape and dtype of tensors going in/out of each
- the attention mask currently produced (size, structure)
- any existing branch / tree infrastructure already present

### 1.2 Validate the alpha estimate

Add a temporary one-shot probe (NO behavior change) gated by env var
to draft_tokens(): at each draft step, log the top-5 logits values
to a file. Run a single 1K-context bench (NOT 100K -- this is for
distribution sanity, fast iteration). Compute:
- P(target == MTP_argmax) at step 0 and step 1
- P(target in MTP_top2) at step 0 and step 1
- P(target in MTP_top3)

If top-2 vs argmax gain is < 1.10x, the tree approach is NOT going to
hit 35 t/s. Stop and reconsider. If gain is > 1.15x, proceed.

Probe code goes in mlx_lm/models/deepseek_v4.py (mlx-lm side) or
dsv4_mtp.py (exo side, simpler). Pick whichever has logits already
realized. The probe writes to /tmp/dsv4_alpha_probe.jsonl and is
0-cost when env is unset.

### 1.3 Confirm verify scales with L_q

Run the existing baseline with EXO_SPECULATIVE_GAMMA=3 for ONE
iter (do not bench -- this is a single-shot wall measurement,
gamma=3 is below break-even per yesterday findings but it tells us
verify(L_q=4) cost). Compare to gamma=2 verify(L_q=3). Linear in
L_q? Sublinear? Superlinear? This tells us the cost of going to L_q=7.

Expected: verify wall is dominated by MoE work which is L_q-independent
(MoE batches over the L_q dimension at near-zero marginal cost since
each token still hits 6 experts). Attention is O(L_q * L_kv) which
grows linearly but at L_kv=100K is dominated by L_kv. So verify(7)
should be < 1.3x verify(3).

Output: append measurements to phase1_findings.md.

---

## Phase 2: Top-K Draft Extraction (4h)

File: src/exo/worker/engines/mlx/speculative/dsv4_mtp.py and possibly
mtp_module.py.

### 2.1 Add draft_tokens_topk(gamma, K) next to draft_tokens()

Do NOT modify draft_tokens() -- keep the linear path as fallback,
gated by env var.

Returns:
- tree_tokens: int32 array of shape (n_nodes,) -- token at each tree node
- parent_idx: int32 array of shape (n_nodes,) -- parent node index
  (root has parent_idx[0] = -1)
- depth: int32 array of shape (n_nodes,) -- depth of each node

For gamma=2, K=2: n_nodes = 1 (root) + K + K*K = 7.
For gamma=2, K=3: n_nodes = 1 + 3 + 9 = 13.
For gamma=3, K=2: n_nodes = 1 + 2 + 4 + 8 = 15.

### 2.2 Implementation strategy

Two options for generating the tree:

Option A (simpler): Run MTP head K times per parent, each on a
1-token input. K=2 gamma=2: 1 + 2 = 3 MTP head invocations total
(same as linear gamma=2 had 2 head invocations). Cheap, but launches
multiple kernels.

Option B (batched): Stack the K parents at depth d as a batched
input of shape (K^d, 1), single MTP forward emits (K^d, vocab),
take top-K to expand to depth d+1. Fewer kernel launches.

Recommend Option A for v1. It is < 5x more kernel work than the
linear draft (which is already 4.5ms / 62.65ms = 7 percent of cycle).
Doubling to 14 percent is acceptable for the first cut; if cycle time
blows up we can micro-optimize later via Option B.

### 2.3 MTP cache handling

Currently draft_tokens() updates the MTP head KV cache sequentially.
Tree drafting needs the cache to fan out: each branch maintains its
own KV state.

Easiest implementation: snapshot the MTP KV cache at the root, restore
it before each branch starts. mlx KV caches are mx.arrays so
cache.copy() is cheap (no data copy, just refcount bump, until
written-to).

Look at how mtp_module.py exposes the cache -- if there is a
clone/snapshot method already, use it; if not, add one.

Output: draft_tokens_topk() callable from _speculative_next when env
EXO_DSV4_TREE_DRAFT=1 is set. Default OFF.

---

## Phase 3: Tree-Attention Mask (6h, HIGHEST RISK)

File: mlx-lm/mlx_lm/models/deepseek_v4.py

This is the hardest piece. The verify pass must process all 7 tree
nodes in one batched forward, but the attention pattern must respect
the tree structure: each node attends to its ancestors (including
itself) but NOT to siblings.

### 3.1 Mask construction

For a tree with nodes 0..6 where:
- node 0 = root (last committed token, just for KV continuity)
- node 1, 2 = depth-1 branches (a, b)
- node 3, 4 = children of node 1 (a_x, a_y)
- node 5, 6 = children of node 2 (b_x, b_y)

The L_q x L_q sub-mask for the new tokens looks like:

```
         0    1    2    3    4    5    6
node 0:  *    .    .    .    .    .    .
node 1:  *    *    .    .    .    .    .
node 2:  *    .    *    .    .    .    .
node 3:  *    *    .    *    .    .    .
node 4:  *    *    .    .    *    .    .
node 5:  *    .    *    .    .    *    .
node 6:  *    .    *    .    .    .    *
```

Plus EVERY node attends to all L_kv prior context tokens (the 100K
prefill cache).

### 3.2 How to construct this in mlx-lm

The attention call is in Attention.__call__ inside
mlx-lm/mlx_lm/models/deepseek_v4.py. Search for
mx.fast.scaled_dot_product_attention or the path that builds mask.

For L_q=7 the mask is tiny (49 entries). It is fine to:
1. Pass tree_mask: mx.array | None as a kwarg through Model.__call__
   to DeepseekV4Layer.__call__ to Attention.__call__
2. When tree_mask is present, use it instead of the default causal mask
3. Always-attend-to-KV section: add a zeros column on the left of
   shape (L_q, L_kv) so the additive mask is (L_q, L_kv + L_q)

mlx scaled_dot_product_attention accepts an additive mask of shape
broadcastable to (B, n_heads, L_q, L_k). Use 0 for attend and
-inf (or -1e9) for do-not-attend.

### 3.3 Plumbing the mask through

Path from API surface to attention:
- Model.__call__(inputs, cache, ..., tree_mask=None)
- for each layer: layer(h, mask=mask_or_tree, cache=cache[i])
- Attention.__call__(x, mask=...) uses the mask

Add tree_mask parameter to:
- Model.__call__
- each DeepseekV4Layer.__call__
- Attention.__call__

Default None (preserve baseline behavior). When None, code path is
literally unchanged.

### 3.4 RoPE / positional encoding for tree nodes

CRITICAL gotcha: each tree node needs the correct POSITION for RoPE.
- node 0 (root): position L_kv (one past prefill)
- nodes 1, 2: position L_kv + 1 (both at depth 1, same position)
- nodes 3..6: position L_kv + 2 (all at depth 2)

This means same-depth siblings share a RoPE position. They are
distinguished only by the attention mask. The depth array from
Phase 2.1 maps directly to RoPE offset.

Implementation: pass position_ids: mx.array of shape (L_q,) to the
attention layer. mlx-lm deepseek_v4.py applies RoPE inside Attention;
check whether positions are derived implicitly from cache length or
passed explicitly. If implicit, we need to override.

This is the second-highest-risk piece after the mask itself. Get it
wrong and the model sees inputs at the wrong positions -- output will
be subtly bad (and quality probe will catch it, which is why the gate
is mandatory).

---

## Phase 4: Verify Orchestration (5h)

File: src/exo/worker/engines/mlx/speculative/dsv4_mtp.py

### 4.1 Wire tree path into _speculative_next

When EXO_DSV4_TREE_DRAFT=1:
1. Call draft_tokens_topk(gamma, K) to get tree_tokens, parent_idx, depth
2. Build tree_mask and position_ids from parent_idx and depth
3. Call model forward with tree_mask and position_ids
4. Extract per-node target argmax
5. Tree-walk accept (see 4.2)
6. Trim cache to accepted-path ancestors

### 4.2 Tree accept algorithm

```
def tree_accept(tree_tokens, parent_idx, target_argmax):
    # Find best leaf path.
    best_depth = 0
    best_leaf = 0  # root is the trivially accepted node
    # Walk every leaf upward, find the deepest match.
    leaves = [i for i in range(len(tree_tokens))
              if i not in parent_idx]  # nodes with no children
    for leaf in leaves:
        path = []
        cur = leaf
        while cur != -1 and cur != 0:
            path.append(cur)
            cur = parent_idx[cur]
        path.reverse()  # ancestors first, leaf last
        # Walk path: accept while target[parent] == tree_tokens[child]
        accepted = 0
        prev_node = 0  # root
        for node in path:
            if target_argmax[prev_node] == tree_tokens[node]:
                accepted += 1
                prev_node = node
            else:
                break
        if accepted > best_depth:
            best_depth = accepted
            best_leaf = prev_node
    # bonus token = target_argmax at the deepest accepted node
    bonus = target_argmax[best_leaf]
    return path_from(best_leaf), bonus
```

Edge case: best_depth=0 (no draft matched). Emit only bonus token
(verify argmax at the root) -- exactly like linear path with 0
accepts. Cycle still produces 1 token, no regression vs linear.

### 4.3 Cache trim semantics

After acceptance, the KV cache must contain ONLY the accepted-path
ancestors, not the sibling branches. The verify pass wrote ALL 7
tree nodes to the cache (positions L_kv .. L_kv+6).

Options:
- Option A: Use sparse cache indexing -- mlx does not support
  arbitrary index trim; cache is append-only. Would require deep
  changes to KV cache. NOT VIABLE.
- Option B: Drop the whole verify segment, re-run a tiny forward
  on just the accepted path. Cost: one extra mini-forward per cycle.
  At L_q ~ 2 the wall is maybe 1-2ms. Acceptable.
- Option C: Trim cache back to L_kv (root), then on next cycle
  the new draft step 0 input becomes the accepted path tokens. The
  next verify will re-include those tokens. Wasteful (re-attends to
  them) but simpler.

Recommend Option C for v1. It is how the linear path already behaves
implicitly (verify always re-attends the last committed position).
The extra cost is ~ accepted_path_len * L_kv attention work in next
verify, which is dwarfed by MoE.

If Option C costs too much in practice, switch to Option B in a
follow-up.

---

## Phase 5: Microbench Gate (3h, MANDATORY)

Before any cluster bench, validate on the local laptop with a
1B-context, 1-prompt smoke test:

### 5.1 Smoke test

bench/smoke_tree_draft.py (NEW file, write fresh):
- Load model on CPU or single-GPU (no exo, no jaccl, single-process)
- Run 10 prompts of varied length (256, 1K, 4K tokens of context)
- For each prompt:
  - Generate 64 tokens with linear gamma=2 (baseline)
  - Generate 64 tokens with tree K=2 gamma=2
  - Verify outputs are PLAUSIBLE (not gibberish, not BOS spam)
  - Measure tokens/cycle for each
- Report alpha_linear vs alpha_tree, t/s_linear vs t/s_tree

Pass criteria:
- Tree alpha > linear alpha by >= 15 percent (else top-K is not helping)
- Tree output is coherent (manual inspection of 1-2 outputs)
- No NaN, no -inf in logits at any layer

If smoke test FAILS, go back to Phase 1.2 and re-check the alpha
distribution math.

### 5.2 Mask correctness test

bench/test_tree_mask.py (NEW file):
- Build a 7-node tree
- Run verify forward with the tree mask
- ALSO run verify forward with the equivalent 4 separate linear paths
  (root + a + a_x, root + a + a_y, root + b + b_x, root + b + b_y)
- Compare logits at each tree node to the corresponding position in
  the linear paths -- they MUST match to within 1e-3 (fp8 noise)

If they do not match, the mask is wrong. Do not proceed.

---

## Phase 6: Cluster Bench + Quality Probe (2h)

Only after Phase 5 passes.

### 6.1 Deploy

```
# On laptop
cd /Users/adam.durham/repos/exo
git status  # confirm clean working tree
# Confirm mlx-lm changes are pushed if adurham/mlx-lm fork is used:
cd mlx-lm && git status && git log -3 && cd ..
# uv sync to pick up new mlx-lm
uv sync
# Confirm venv mlx-lm matches:
diff mlx-lm/mlx_lm/models/deepseek_v4.py \
     .venv/lib/python3.13/site-packages/mlx_lm/models/deepseek_v4.py
# If different, scp the venv copy to both cluster nodes:
for h in macstudio-m4-1 macstudio-m4-2; do
  scp -i ~/.ssh/exo_cluster \
    .venv/lib/python3.13/site-packages/mlx_lm/models/deepseek_v4.py \
    $h:~/repos/exo/.venv/lib/python3.13/site-packages/mlx_lm/models/deepseek_v4.py
done
```

### 6.2 Quality gate FIRST

```
# Start cluster with tree draft enabled
EXO_DSV4_TREE_DRAFT=1 ./start_cluster.sh
# Wait for ready
# Run quality probe
python bench/quality_probe_dsv4.py
# MUST return needle_found=True with FALCON-MERCURY-7749
```

If quality probe fails: STOP. The tree mask or RoPE is wrong.
Revert and debug.

### 6.3 Perf bench

```
python bench/concurrent_bench.py \
  --prompt-words 75000 \
  --max-tokens 128 \
  --concurrency 1 \
  --iters 10
# Check: mean >= 35, sigma < 0.3, 10/10 clean
```

If mean < 35 but > baseline (>30 but <35): the lift is real,
tune K and gamma in Phase 7.

If mean <= 30: regression. Compare alpha_tree from logs vs phase 5
microbench. Likely culprits:
- verify wall blew up (check EXO_DSV4_MTP_PROFILE=20 output)
- mask is mostly-but-not-fully correct -- some leaves degraded

### 6.4 Tag if win

If perf bench passes (mean >= 35, sigma < 0.3, 10/10 clean,
quality probe passes), tag:

```
git tag tree-draft-2026-MM-DD-g2-K2-XX.XX
git push --tags
```

Update memory with the new baseline number.

---

## Phase 7: Tune K and gamma (2h)

Once K=2 gamma=2 works, sweep:

```
for K in 2 3; do
  for gamma in 2 3; do
    EXO_DSV4_TREE_DRAFT=1 \
    EXO_DSV4_TREE_K=$K \
    EXO_SPECULATIVE_GAMMA=$gamma \
    ./start_cluster.sh
    sleep 60  # warmup
    python bench/quality_probe_dsv4.py  # gate
    python bench/concurrent_bench.py --prompt-words 75000 --max-tokens 128 --iters 5
    # Record mean, sigma, alpha
  done
done
```

Expected best point: K=2 gamma=2 (4 leaves) or K=2 gamma=3 (8 leaves).
K=3 gamma=2 (9 leaves) might also win if verify scales sublinearly in
L_q.

K=3 gamma=3 (27 leaves) is probably too much verify work.

Final answer: pick the config with highest mean at sigma < 0.3,
quality passes.

---

## Pitfalls / Do-Not-Do List

1. DO NOT skip Phase 5 microbench. Cluster bench is 8 minutes per
   iter at 100K. Microbench is 5 seconds per prompt at 4K. If the mask
   is wrong, you waste hours and corrupt the cluster.

2. DO NOT modify draft_tokens() in-place. Add draft_tokens_topk()
   alongside. The env var EXO_DSV4_TREE_DRAFT selects which to use.
   Baseline must remain bit-exact reproducible from the current code.

3. DO NOT change TOPK from 512. Quality is broken at TOPK<512 at
   100K. The whole point of this work is to lift t/s at quality-correct
   TOPK=512.

4. DO NOT skip the RoPE position fix. Same-depth siblings share a
   position. Forgetting this gives wrong-looking outputs that quality
   probe will catch -- but it is the FIRST thing to suspect on a
   quality-fail.

5. DO NOT trust an alpha lift in the microbench at small context.
   The alpha distribution at 4K context is different from 100K (less
   sparse attention, different MoE routing). Phase 6 must run at 100K
   prompt-words for the real number. The microbench is a sanity test,
   not a verdict.

6. DO NOT add a synthesizer or fusion step in the draft. Just top-K.
   Anything fancier (beam search, learned reweighting) is a research
   project, not a 25h plan.

7. DO NOT bench gamma>=4. alpha_3 was already below break-even
   yesterday. Tree drafting helps because it BRANCHES at low depth;
   it does not help with deeper chains.

8. DO NOT touch jaccl or all_reduce code. Confirmed yesterday it is
   1.9 percent of verify -- not the bottleneck. Any changes there
   delay this plan and do not help.

9. DO NOT delete files in mlx-lm/.venv/... -- venv mlx-lm is what
   the runner imports, NOT the submodule. Edits MUST land in BOTH:
   the source mlx-lm/mlx_lm/... AND the deployed
   .venv/lib/python3.13/site-packages/mlx_lm/... on each cluster
   node. See 6.1 deploy section. Verify with diff.

10. DO NOT bench without bench/quality_probe_dsv4.py PASSING first.
    Yesterday lesson: TOPK=160 looked like 32 t/s but produced
    BOS-only output at 100K -- invalid result. Quality gate is
    non-negotiable.

---

## Rollback Plan

If anything breaks the cluster or corrupts the baseline:

```
cd /Users/adam.durham/repos/exo
git stash  # or git reset --hard if changes uncommitted
git checkout baseline-2026-05-18-mtp-g2-topk512-30.06
cd mlx-lm
git checkout 6dcdd40a
cd ..
uv sync
# Redeploy mlx-lm to cluster nodes
for h in macstudio-m4-1 macstudio-m4-2; do
  scp -i ~/.ssh/exo_cluster \
    .venv/lib/python3.13/site-packages/mlx_lm/models/deepseek_v4.py \
    $h:~/repos/exo/.venv/lib/python3.13/site-packages/mlx_lm/models/deepseek_v4.py
done
./start_cluster.sh
python bench/quality_probe_dsv4.py  # must pass
python bench/concurrent_bench.py --prompt-words 75000 --max-tokens 128 --iters 3
# Expect ~30 t/s, needle pass -- back to baseline
```

---

## Success Criteria (in priority order)

1. Quality: bench/quality_probe_dsv4.py returns needle_found=True
   with FALCON-MERCURY-7749. Non-negotiable.

2. Stability: 10 iters at 100K c=1 gamma=2 TOPK=512 MTP=1, all 10
   clean, no NaN, no error, no hang.

3. Perf: mean t/s >= 35.0, sigma < 0.3.

4. Reproducibility: re-tagged commit ships in start_cluster.sh
   defaults; a fresh ./start_cluster.sh produces the >= 35 t/s number
   without manual env tweaks.

5. Code review: each phase PR (or commit cluster) has a 1-page
   summary including what was measured, what the math is, and what
   ENV vars gate the new path.

If only (1)(2)(3) pass but (4) is broken (e.g. EXO_DSV4_TREE_DRAFT=1
required to get >= 35), still a win but follow up with a small commit
to flip the default ON.

---

## Open Questions (resolve in Phase 1)

Q1: Does mtp_module.py predict_from_hidden() (line 536) already
support batched K-way prediction? If yes, draft_tokens_topk gets
simpler (just call it instead of looping argmax). Read it Phase 1.

Q2: Does mlx-lm scaled_dot_product_attention accept arbitrary
additive masks, or does it only support causal / boolean lower
triangular? Check the API. If only causal, we need to fall back to
manual (QK^T + mask) -> softmax -> @V, which is slower. Mitigation:
manual SDPA is still cheap at L_q=7.

Q3: How does the current code handle the case where draft accepts 0
tokens? The bonus_token path -- is it shared between linear and would
need adapting for tree? Trace it Phase 4.1.

Q4: Are there assertion failures in _speculative_next() that
hardcode L_q=gamma+1 or similar? Search for assert and shape[1] ==.
Grep dsv4_mtp.py for shape assertions before making structural changes.

Q5: Does the exo MTP cache (the one feeding back into draft step N+1)
support a clone/snapshot API? If not, designing one is part of
Phase 2.3. May add 2-3h to estimate.

---

## Timeline

Day 1 (8h):
- Phase 1: Investigation (3h)
- Phase 1.2: alpha probe + small bench to validate distribution (1h)
- Phase 2: draft_tokens_topk (4h)

Day 2 (10h):
- Phase 3: Tree-attention mask + RoPE positions (6h)
- Phase 4: Verify orchestration (4h)

Day 3 (7h):
- Phase 5: Microbench gate + mask correctness test (3h)
- Debug whatever Phase 5 turns up (2h budget)
- Phase 6: Cluster bench + quality probe (1h)
- Phase 7: K/gamma sweep + final tag (1h)

Total: ~25h. Add 30 percent buffer for unknowns -> 30-32h total.

If Phase 1 reveals the alpha lift is < 1.10x (Q-from-Phase-1.2), abort
plan -- the tree approach will not hit 35 t/s and we need to accept
the 30.06 t/s ceiling or find a non-MTP path (which does not
currently exist -- yesterday we exhausted MoE / jaccl / compile-fusion
levers).

---

## Notes for Hermes Operating This Plan

- This file is
  /Users/adam.durham/repos/exo/.hermes/plans/2026-05-19_token_tree_drafting.md.
- The previous-session summary is in the compaction note at the top of
  the chat. Re-read it before starting Phase 1.
- Memory facts that matter:
  - mlx-lm venv vs submodule trap (deploy both, verify with diff)
  - TOPK=512 only -- never lower for "perf"
  - Quality probe is non-negotiable
  - Root-cause not mitigation (no "soften the symptom" patches)
  - User wants action not analysis-paralysis; ask if Phase 1 reveals
    a blocker, otherwise execute
- Skill: exo-cluster-operations -- re-read pitfalls before each
  destructive command (cluster restart, file scp to nodes, mlx-lm
  redeploy)
- The user has the dashboard at adams-mac-studio-m4-1.local open;
  check it before declaring cluster "stuck" vs "loading."
- Use Discord ping for action-needed prompts, full payload in chat.
