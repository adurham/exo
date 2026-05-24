# 35 t/s plan execution — 2026-05-24 results

Plan source: `.hermes/plans/2026-05-23_35_tps_plan.md`

Single-session execution: started 2026-05-24 11:31 CDT, results captured
by 14:45. Cluster cold-launched fresh, all 4 commits already on
`origin/main` (HEAD `ee2f2abc`). All hard constraints from §"Hard
constraints" honored: KV bf16, INDEX_TOPK=512, model defaults as
shipped, no mitigations, quality validated at every step.

---

## W2 — fused-topk validation (10-iter c=1 100K γ=2)  → NEGATIVE, no ship

Per plan §W2: "if marginal: document the limit and move on."

| Metric                 | Baseline (no fused-topk) | Fused-topk (=1)       |
|------------------------|--------------------------|-----------------------|
| iters scored           | 10                       | 10                    |
| per_req_gen_tps mean   | 28.7998                  | 28.8378               |
| per_req_gen_tps σ      | 0.0995                   | 0.0710                |
| min                    | 28.59                    | 28.70                 |
| max                    | 28.90                    | 28.92                 |
| wall_mean (s)          | 396.36                   | 396.28                |
| tail_ratio_max         | 1.00                     | 1.00                  |
| bad_rate               | 0%                       | 0%                    |
| **Δ**                  | —                        | **+0.0380 (+0.13%)**  |
| **Welch t**            | —                        | **0.98** (p~0.34)     |
| **2σ_max separation**  | —                        | FAIL (need >0.199)    |

Bench: `bench/concurrent_bench.py --concurrency 1 --iterations 10
--max-tokens 256 --prompt-words 75000 --timeout 1800 --warmup 1`

Toggle mechanism: live `/tmp/dsv4_nop_targets` containing `topk_fused`
on both nodes (no restart needed; 1s TTL on
`_get_nop_targets()`). Fused path conditions
(`deepseek_v4.py:1808-1811`) satisfied at decode (L_q=1, pmask=None,
k=512 ≤ 1024).

Quality gates passed:
- Pre-bench c=1 100K probe (target_tokens=100000, max_tokens=128):
  needle ✓ FALCON-MERCURY-7749, BOS=0, bistability=0.
- Post-fused c=1 100K probe (cache-warm): needle ✓, BOS=0,
  bistability=0.

**Decision: do NOT set `EXO_DSV4_TOPK_FUSED:=1` as default in
start_cluster.sh.** Fused path is bit-equivalent and harmless to keep
available via env-var or file-toggle, but the win is below our
measurement floor (σ=0.10 at this config). Plan's own math predicted
this — 5× per-call microbench → ~3% expected lift → ~0.9 t/s — yet we
measure 0.04 t/s. The kernel is amortized into noise at γ=2 MTP-on
where the verify-forward all_sum/MoE waits dominate per-decode wall.

---

## W1 — c=2 100K spec-batched quality bug instrumentation plan  → READY TO APPLY (awaiting approval)

Output: `/tmp/w1_c2_instrumentation_plan.md` (500 lines, 22 KB).

Read-only stderr tracing, env-gated by `EXO_DSV4_C2_SPEC_TRACE=1`,
zero runtime cost when unset. Three hook clusters:

- Hook A: `_upgrade_cache_to_per_stream` pre/post swap, dumps per-cache
  `offset`, `_offset`, `_per_stream_max`.
- Hook B: `dsv4_speculative_forward` per-stream verify-logits peek
  (`argmax`, `softmax_peak`) BEFORE acceptance gate. **This is the
  highest-information signal** — distinguishes bad-attention from
  bad-acceptance-gate.
- Hook C: `_speculative_next_batch` per-iter dump (cycle, uids, draft,
  target, matches, n_accepted_per, bonus_vals,
  pre/post rollback offsets).

**Primary culprit hypothesis (high confidence, code-reading)**:
`PerStreamBatchRotatingKVCache.update_and_fetch`
(`mlx-lm/mlx_lm/models/cache.py:2391-2394`) returns a uniform-S KV
slice `keys[..., :_per_stream_max, :]` where `_per_stream_max` is the
MAX of all per-stream offsets, NOT per-stream. The per-stream
right-padding mask is then applied **additively in SDPA**, AFTER the
SparseCompressedAttention Indexer (`deepseek_v4.py:1808+`) has already
done top-K=512 score-based selection on the uniform-S tensor.

At 100K context with top-K=512 over 100K positions, the lagging
stream's "garbage tail" (positions that exist in the buffer but aren't
part of that stream's history) can score high enough to beat real
deep-document positions. The mask removes them in SDPA, but by then
the Indexer has already constrained attention to those wrong
positions. This explains the BOS-spam fingerprint and the c=1-OK /
c=2-short-OK / c=2-100K-broken pattern.

Secondary suspect: cache-state-swap timing (`_per_stream_max`
staleness immediately after `_upgrade_cache_to_per_stream`). Hook A
would catch this.

**Status**: patches written and reviewed, but not applied — touches
spec-critical code paths and the user has been burned before by
unapproved spec changes. **Requesting explicit go-ahead before
shipping a branch with `EXO_DSV4_C2_SPEC_TRACE=1`.**

If approved, the next steps are:
1. Apply the patch (3 hook clusters in `dsv4_mtp.py`).
2. Commit + push to origin/main (mandatory per skill pitfall #15
   before `start_cluster.sh` does `git reset --hard origin/main` on
   nodes).
3. Restart cluster with `EXO_DSV4_C2_SPEC_TRACE=1` in env override.
4. Run `bench/quality_probe_dsv4.py --concurrency 2 --iters 3
   --target-tokens 100000 --max-tokens 4096`.
5. Grep stderr for `[C2_SPEC_TRACE]` and look for Hook B's `stream b`
   row showing first-token argmax pinning to BOS or instruction-token
   IDs while stream a is healthy. That confirms primary hypothesis.

---

## W3 — Eagle implementation audit  → BLOCKED ON EMPIRICAL TOP-K DUMP

Output: `/tmp/w3_eagle_audit.md` (182 lines, 9 KB).

Findings:

1. **Top-1 dominance unknown.** K=1 fast path never computes
   `softmax(prev_logits)` — bypasses the distribution entirely
   (`mtp_module.py:703-706`). Need a one-time empirical dump to know
   whether p_1 ≈ 1.0 (Eagle structurally dead) or p_1 < 0.7 (Eagle
   should help; current acceptance flatness is a bug).

2. **Soft-emb math: matches spec WITH ONE DEVIATION.**
   `mtp_module.py:718` and `dsv4_mtp.py:1740-1742` **RENORMALIZE** the
   top-K probabilities to sum to 1 over the top-K subset:

   ```python
   _topk_probs = _topk_probs / _topk_probs.sum(axis=-1, keepdims=True)
   ```

   The spec literally says `softmax(top_K)` — raw masses summing to
   ≤1 (with residual falling into ignored tail). The code instead
   delivers a re-scaled distribution that sums to 1.

   Effect at p_1 ≈ 0.99: ~0% change. Effect at p_1 ≈ 0.4: every
   top-K weight scaled up by ~2×, soft_emb pushed toward a wider
   mixture than the head was ever trained to consume. **Candidate
   explanation for K>1 acceptance flatness if the MTP head is
   scale-sensitive.**

3. **Stale-logits ruled out.** Commit `40b6e9c0`'s fix is intact in
   both files. K=1 uses already-broadcast `tok_arr`. K>1 launders
   rank-local `prev_logits` through its own `topk_ids+topk_probs`
   broadcast from canonical rank. Cross-rank determinism preserved.

**Status**: 5-line dump patch written
(`mtp_module.py:740` insert, gated by `EXO_DSV4_MTP_DUMP_TOPK=1`).
**Not applied yet — also touches spec code.** Same approval gate as
W1.

If approved, the next steps are:
1. Apply 5-line patch + restart cluster with
   `EXO_DSV4_MTP_DUMP_TOPK=1`.
2. Send one short prompt ("capital of France?"), grep worker log for
   `[MTP_TOPK]` lines.
3. Inspect ~50 decode steps:
   - p_1 > 0.95 typical → **Eagle dead**, document in
     don't-re-attempt list, move on.
   - p_1 < 0.7 typical → **K>1 mixture should help**; experiment
     with renormalization-removed variant and re-bench K=4 vs K=0
     acceptance histograms.

---

## Summary against plan's stop-light criteria

Plan §"Stop-light": "After W1+W2+W3 (15-20 hours work), reassess".

In the half-day we actually spent:
- W2 is closed (negative result).
- W1 has a high-confidence root-cause hypothesis backed by code-read
  + a ready-to-apply diagnostic patch.
- W3 has falsified stale-logits, found a real deviation, and a 1-min
  empirical test that decides Eagle's fate.

The aggregate 35 t/s target is still out of reach. **c=1 ceiling at
this config measured today: 28.80 ± 0.10 t/s.** That's not the 29.7
the plan assumed — slightly lower. Possible reasons:
1. Fresh cold cluster, thermal not yet stabilized (unlikely;
   wall_mean=396s flat across 10 iters, no drift).
2. Plan's 29.7 was an older measurement; today's defaults are tuned
   identically and the cluster is regression-clean.
3. Real -3% drift somewhere since the 29.7 baseline was set.
   → Worth a short bisect against `champion-2026-05-17-mtp-g2-fenced-31.5`
   tag if user wants to chase.

**Aggregate-path attack via W1 (fix c=2 100K) remains the highest-EV
move.** Even if c=2 per-stream lands at 16 t/s (May 12 measurement) it
aggregates to ~32 t/s; closer to 17 t/s/stream (today's c=1 / 2-ish
back-of-envelope) aggregates to ~34 t/s and clears the target.

---

## What I would do next given approval

1. **Apply W3 dump patch (5 lines, cheapest)**, restart, send one
   short prompt, read p_1 distribution. ~10 min. Decides whether
   Eagle is dead or worth fixing the renormalization deviation.
2. **Apply W1 trace patch (≈40 lines, env-gated)**, restart with
   `EXO_DSV4_C2_SPEC_TRACE=1`, run c=2 100K probe (3 iters), grep
   Hook B output for stream-asymmetric argmax. ~30 min wall.
3. If Hook B confirms uniform-S-Indexer hypothesis, **structural fix
   is in `cache.py`**: either (a) compose per-stream KV slices and
   pad before SDPA (heavy), or (b) plumb a per-stream `valid_length`
   into `_indexer_score`/`_sparse_pooled_attention` so the
   top-K=512 selection excludes positions beyond each stream's
   `offsets_py[b]` (light, surgical, score-mask).

---

## What I will NOT do without further approval

- Apply ANY of the diagnostic patches (W1 hooks, W3 dump). They
  touch spec code paths; user policy is "verify intent first".
- Propose a structural fix to `cache.py` (option a/b above) without
  having the Hook B evidence in hand.
- Quote any tok/s number going forward without an accompanying
  c-aware quality probe pass (per plan §"What I will NOT do").
- Touch FENCE_EVERY_N_LAYERS, INDEX_TOPK, KV_CACHE_BITS, or the
  model defaults (hard-line constraints).
