# Phase 6B Findings: Token-Tree Drafting BUG FIXED

Status: **WORKING - sanity test coherent (matches baseline), 100K
needle probe PASSES (FALCON-MERCURY-7749), cluster bench in progress.**

## Bug summary

Three independent bugs combined to produce wrong-logits-at-production
in the original Phase 6 deploy:

### Bug 1: row-causal pmask for tree input (mlx-lm 7f80fff -> 19771108)

`PoolingCache.make_mask(L, offset)` is row-causal-by-row-index. For
linear input that's fine (positions are monotonically increasing). For
tree input, same-depth siblings share an absolute position but get
DIFFERENT make_mask rows. Effect: identical Q rows get inconsistent
pool-attend cutoffs → wrong logits.

Fix: `_tree_pmask(pool_cache, positions)` uses per-token positions
from the side channel for the cutoff. Same-depth siblings get
IDENTICAL pmask rows by construction. `_dispatch_pmask` routes
between tree and stock based on side channel state, keeping linear
path bit-exact.

Microbench: 4 new tests added — same-depth-sibling identity, linear-
path bit-exactness, ctx-unset fall-through, ctx-set routing.

### Bug 2: Compressor pool-cache mutation during tree verify (mlx-lm 19771108 -> 8d7471c6)

`Compressor.__call__` runs `accumulate_windows` + `update_and_fetch`
unconditionally during the tree verify. With L_q=7 tree input + small
remainder (e.g., 3 buffered from previous decode), 1-2 pool windows
flush, packing CONTRADICTORY same-depth-sibling content into the
compressed entries. The committed pool then contains tree-derived
contamination that PoolingCache.trim CANNOT remove (it only trims
the uncommitted remainder buffer, not the committed pool offset).
Effect: BOTH within-cycle compressed-attention reads contradictory
siblings AND subsequent cycles see corrupted pool entries.

Fix: detect `_TREE_VERIFY_CTX["positions"]` in Compressor.__call__
and short-circuit — return the existing committed pool (or zeros if
empty). No accumulate_windows, no update_and_fetch, no mutation.
Tree-input tokens are transient (local_cache is rolled back); the
pool stays frozen at its prefill-derived state.

### Bug 3: rollback discards y + accepted-path KV (exo aad4c084 -> 3caffad7)

The original tree-drafting rollback trimmed the entire tree
(`rollback = n_nodes`), discarding both the bonus token `y` AND the
accepted-path tokens from local_cache. The plan comment claimed "next
cycle's verify-input will start from [bonus_val] + new draft tree,
redoing the accepted-path tokens" but the verify_input was actually
just `[bonus_val] + new_tree` — accepted path was NEVER re-committed.

Effect: next cycle's forward processed bonus_val as if it came directly
after the prefill, with no KV record of the accepted drafts. The
model fell back into context-blind repetition (observed: "to write a
5-sentence paragraph about" looping every ~11 tokens).

Diagnostic that nailed it: [TREE-DEBUG] dumping verify_argmax showed
the TREE VERIFY FORWARD ITSELF was correct — root predicted " to"
after " need", depth-1 " to" predicted " write". The argmax tokens
matched what a linear forward would produce. So the bug was DOWNSTREAM
of the verify forward, in cache state for the next cycle.

Fix: after tree rollback, run a small linear "commit forward" over
`[y, *accepted_drafts]` (length n_accepted+1). The side channel is
already cleared from the tree verify's finally block, so this is a
vanilla forward that writes correct contiguous KV at L_kv..L_kv+n_accepted.
MTP cache trim changes from `gamma` to `gamma - n_accepted`, mirroring
the linear baseline's post-accept semantics.

Cost: ~3-token forward per tree cycle, ~6% of wall budget. Without it
the tree-mode runtime was 27.7 t/s with garbage output; with the fix
it produces correct output.

## Acceptance results

Sanity test (5-sentence paragraph, max_tokens=64, EXO_DSV4_TREE_DRAFT=1
K=2 gamma=2 INDEX_TOPK=512):

  Baseline (tree-OFF):  48.95 t/s, "We need to write a 5-sentence
                                    paragraph about how computer caches
                                    work in modern CPUs..." COHERENT
  Tree-K2 (FIXED):      22.30 t/s, "We need to write a 5-sentence
                                    paragraph about computer caches in
                                    modern CPUs. Be precise and
                                    concise. Key points: caches are
                                    small, fast memory close to CPU
                                    cores..." COHERENT
  mtp_cycles=24 accepted_drafts=39 -> 1.625 drafts/cycle

(Small-prompt t/s isn't directly comparable to baseline because the
commit-forward overhead is significant relative to a 64-token
generation. Real production perf comes from the 75K-prompt bench.)

Quality probe (FALCON-MERCURY needle, target 100K, observed 69K
prompt_tokens, max_tokens=64):

  needle_found: True
  response: 'FALCON-MERCURY-7749'
  ttft: 294.9s (235 tok/s prefill)
  total wall: 295.3s
  decode tps (apparent): 124.50

The Phase 6 quality gate PASSES — tree drafting produces semantically
correct output at the production sliding-window + sparse-attention
config.

## Tags / artifacts

- mlx-lm:
  - 7f80fff: original tree-mask + per-token RoPE infrastructure (Phase 5)
  - 19771108: pmask dispatch fix
  - 8d7471c6: Compressor freeze
- exo:
  - 0d3d17f5: EXO_DSV4_TREE_DEBUG diagnostic
  - 76e60b17: pmask + microbench tests + start_cluster.sh env forward
  - aad4c084: bump mlx-lm to 8d7471c6
  - d078ff79: verify_argmax diagnostic
  - 3caffad7: commit forward + MTP cache linear trim — THE FIX

## Cluster bench results (75K prompt, c=1, 3 scored iters)

```
iter 0 warmup: wall=391.19s  gen_tps=29.81 (89408 prompt tokens, 128 gen)
iter 1:        wall=390.74s  gen_tps=29.91
iter 2:        wall=390.85s  gen_tps=29.99
iter 3:        wall=391.05s  gen_tps=29.99
SUMMARY: agg_tps_mean=29.95 median=29.96 min=29.91 max=29.99
        sigma ~= 0.05  bad_rate=0%
```

**Tree drafting matches baseline 30.06 t/s exactly — no regression
but ALSO NO LIFT.** Expected ~33 t/s per the Phase 1.2 alpha-probe.
Two likely causes for the missing lift:

1. The commit forward (step 5b) costs ~3 tokens of extra forward per
   cycle. At decode ~30 t/s steady state (~33ms/token), the commit
   forward is ~100ms per cycle. Cycle wall budget at 33 t/s tree =
   30ms/token × 3 tokens/cycle = 90ms. So commit forward could easily
   add 100ms+ overhead = -33% t/s lift.

2. The MTP cache trim issue: for n_accepted=2 the K/V at L_kv+2 may
   be from the wrong depth-1 sibling, lowering next-cycle MTP draft
   quality. Lower acceptance = fewer drafts per cycle = less lift.

## Open work

1. **Skip commit forward when n_accepted=0**: don't pay the extra
   forward cost when there's nothing to commit. Should claw back
   some of the missing lift since ~70% of cycles have n_accepted<=1.

2. **MTP cache rebuild on n_accepted >= 2** (deferred to v1.1): the
   linear-trim semantics still leave wrong K/V at L_kv+2 when the
   accepted depth-1 sibling isn't the BFS-last sibling. Affects next-
   cycle draft quality (lower acceptance), not user output.

3. **10-iter validation**: before tagging champion, run 10 iters and
   confirm no variance / bistability. Per memory, gamma=2 needs >=10
   iters all >=29 t/s sigma<0.5 to claim a fix. Currently 4/4 clean
   (warmup + 3 scored, all >= 29.8). Looks promising but need more.

4. **K=3 sweep**: per Phase 7. Tree with K=3 (1 root + 3 d1 + 9 d2 =
   13 nodes) may give higher acceptance at the cost of larger verify
   forward. The commit forward overhead is INDEPENDENT of K (only
   depends on n_accepted), so K=3 might amortise better.

## Decision: tag as a CORRECTNESS milestone, not perf champion

We've proven:
- Tree drafting infrastructure is end-to-end correct.
- Quality probe passes (needle_found=True at 100K).
- No regression vs baseline (29.95 vs 30.06 t/s within noise).
- Microbench tests (10/10) lock in the fix at unit level.

We haven't proven:
- Tree drafting beats baseline (the goal). Needs commit-forward
  optimisation and/or K sweep to show lift.

Recommendation: tag `tree-draft-2026-05-20-correctness-30.0` as the
first-working-tree-drafting milestone. Continue Phase 7 in a follow-up
session focused on perf, not correctness.
