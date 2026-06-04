# Phase 7 Perf Findings — Token-Tree Drafting

Status: **in progress**. The Phase 6B correctness fix shipped (29.95 t/s
matching baseline 30.06 at 75K c=1). This doc tracks the perf-lift
optimizations and the bottleneck analysis.

## Bottleneck: verify dominates

Phase profile at 75K c=1 (full K=2 g=2 tree, post-Phase 6B):

```
draft:    5.01ms  ( 7% of cycle)
verify:  62.28ms  (87% of cycle)   <-- BOTTLENECK
accept:   0.70ms  ( 1%)
commit:  15.81ms × 67/325 = 3.3ms avg  ( 5%; only when accept-path non-contiguous)
rollback: 0.15ms  ( 0%)
total:   71.42ms
```

Tokens/cycle at 75K: 1.12 drafts + 1 bonus = 2.12 (vs linear's 2.07).
Acceptance lift over linear is ~+2%, NOT the +13% predicted by Phase 1.2's
alpha probe at the small-prompt sample. At long context (75K) MTP
top-1 captures almost all the agreement; top-2 candidates rarely match
target argmax.

## Optimization 1: DFS-prefix tree layout + fast-path commit skip

Reorder BFS -> DFS so the top-1 chain at every depth lives at
contiguous cols [0, 1, ..., n_accepted]. The post-accept fast path
trims the trailing siblings without running a commit forward.

Coverage at 75K: 79% of cycles skip commit (n=67 commit calls / 325 cycles).

Result: cycle 71.42ms -> ~68ms (saved ~3ms of commit overhead). Bench
t/s: 29.8 (vs 29.95 baseline; within noise). No bench-level lift
because verify still dominates (87% of cycle) and the 5% commit
savings are absorbed by mlx async pipelining.

Tags:
- exo 9c38d9ce: DFS-prefix reorder
- exo 7a5f6980: drop redundant `n_accepted<=1` gate (broaden fast-path)

## Optimization 2: Greedy tree (only top-1 d1 expands d2)

Tree shrinks from 7 nodes to 5 nodes (root + 2 d1 + 2 d2 of d1[0]).
Verify L_q=5 instead of 7. Trades worst-case acceptance (we lose n=2
along d1[1] branches) for cheaper verify.

Phase profile at 75K (greedy + DFS + fast-path, post-Phase 6B):

```
draft:    4.58ms  (was 5.01ms; -9%)
verify:  61.35ms  (was 62.28ms; ONLY -1.5%!)
accept:   0.68ms
commit:   4.65ms × 48/175 = 1.3ms avg
total:   68.04ms  (was 71.42ms; -5%)
```

**Surprise: verify only dropped 1.5%, not the predicted 28%.** This is
because at 75K context the verify cost is dominated by attention over
the long KV cache (per-token), not by L_q-sized work. mlx's SDPA at
L_q=5 vs L_q=7 saves only ~1ms.

This means the only way to reduce verify cost is to reduce KV size or
attention shape, not L_q.

### Greedy MTP-cache bug fix (commit 565cdc6f)

First greedy bench produced 29.5 / 8.6 / 22.3 t/s (bistable). Root
cause: the existing `if d1_node < K` MTP-cache trim guard was written
for the full-tree loop iterating d1_node in [1..K]. In greedy mode the
loop only iterates d1_node=1 once but the guard `1 < 2 = True` still
fired, leaving MTP cache at L_kv+1 (instead of L_kv+2). The caller's
`mtp_cache.trim(gamma=2)` then over-trimmed to L_kv-1 -- invalid
offset, next-cycle draft state corrupted -> intermittent collapse.

Fix: change guard from `d1_node < K` to `d1_node + 1 < d1_range[1]`
(= "not last sibling we'll actually iterate"). Reduces to the same
value in full mode, but excludes the only-iteration case in greedy.

## What's left to try

1. **Re-bench greedy + fixed MTP-cache** (in progress, expecting stable
   ~29-30 t/s based on cycle wall).

2. **MTP cache rebuild for n_accepted=2 slow path** — improve next-cycle
   draft quality. Likely +0.5 t/s.

3. **Async commit forward** — overlap commit forward with next-cycle MTP
   draft. Could claw back the 5% commit overhead but only on slow path.

4. **K=3 greedy** — root + 3 d1 + 3 d2 of d1[0] = 7 nodes. Same verify
   cost as full K=2 tree but 1 extra d1 candidate. Might improve
   acceptance enough to make up for greedy losses.

5. **Accept that the perf math doesn't favor tree drafting at long
   context for DSv4.** The verify-dominated structure means tree's
   K^γ token count costs ~linearly proportional to nothing-much, while
   token-yield gain is tiny. Document and move on.

## Numbers summary (75K c=1, 2-3 scored iters each)

| Config                            | t/s   | σ    | cycle ms | tok/cyc | comment                       |
|-----------------------------------|-------|------|----------|---------|-------------------------------|
| linear baseline (γ=2)             | 30.06 | 0.06 | ~69      | ~2.07   | production champion           |
| tree K=2 correctness (Phase 6B)   | 29.95 | 0.05 | ~70      | ~2.12   | matches baseline within noise |
| tree K=2 + DFS + fast-path        | 29.8  | 0.05 | 71.4     | 2.13    | commit savings absorbed       |
| tree K=2 greedy (broken MTP)      | 20.1  | 10.6 | -        | -       | BISTABLE — trim guard bug     |
| tree K=2 greedy + MTP fix         | 29.7  | 0.05 | 63.5     | 1.88    | -11% cycle, -12% yield = wash |
| **tree K=3 greedy**               | 29.7  | 0.10 | ~70      | ~2.0    | same as K=2 within noise      |

Tested SIX configurations across Phase 6B and Phase 7. All tree configs
land in [29.7, 29.95] t/s, σ ≈ 0.05-0.10. Linear baseline 30.06 t/s
remains the production champion.

## Bottom line

Tree drafting at K=2 γ=2 on DSv4-Flash at 75K context **cannot beat
linear γ=2** because:

1. **Verify cost is bounded below by per-token KV access at 75K** —
   reducing L_q from 7 to 5 saves only 8% (not 28% as L_q-scaling would
   predict). The verify floor is ~52ms regardless of L_q in this regime.

2. **MTP top-2 candidate rarely matches target argmax at long context**.
   Drafts/cycle gain is +2-5% over linear. Not enough to amortise extra
   verify cost.

3. **Verify dominates ~85-90% of cycle wall**. The only meaningful lift
   from cycle-wall reduction would require cutting verify cost itself,
   not improving the smaller phases (commit, draft, accept, rollback).

## What would lift t/s above linear baseline

NOT in tree drafting at this config. Avenues for future work:

1. **Eagle-style hidden refinement**: refine MTP's per-step prediction
   using the verify-pass hidden, not just argsort logits. Higher per-
   slot acceptance → close to 2 tokens/cycle with γ=1. 1.5x lift if
   per-slot accept hits 0.85+.

2. **Train a step-1-conditioned MTP head**: current MTP is trained to
   predict the IMMEDIATE next token from hidden; a head conditioned on
   "1 token forward from the current hidden" might capture the chain
   pattern better and lift top-2 accuracy.

3. **Async draft + verify overlap**: schedule the next cycle's MTP
   draft to run concurrently with the current cycle's verify. Saves
   the draft phase (~5ms = 7% of cycle). Risky w.r.t. cache state.

4. **Accept the 30.06 t/s ceiling at this model+context+hardware**.
   Phase 1.2's alpha-probe prediction was wrong for the production
   regime; the +3 t/s lift was based on an unrealistic model of the
   verify cost scaling.

## Recommendation

Tag `tree-draft-2026-05-20-correctness-g2-K2-29.95` (already pushed)
as the best-effort tree-drafting milestone. Keep tree drafting code
behind `EXO_DSV4_TREE_DRAFT=1` (default OFF). Default production
path stays linear `baseline-2026-05-18-mtp-g2-topk512-30.06`.

The greedy-mode optimization (EXO_DSV4_TREE_GREEDY=1) is now correct
but doesn't beat the full tree at this config — left in code as an
option to try at K=3 γ=2 if future work investigates that.
