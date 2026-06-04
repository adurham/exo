# Phase 14: Plans for (a) clean 10/10 validation and (b) push above 35 t/s

Status: planning. After Phase 13 hit 34.5 agg t/s on 8/10 iters with σ=0.04, two
follow-up tracks are open. This doc plans both with effort, risks, and gates so we
can sequence them.

Current baseline (verified 2026-05-21):
- c=1 100K MTP-on:  29-30 t/s (production champion)
- c=2 100K MTP-on:  34.5 agg t/s (8/10 iters σ=0.04, 2/10 outliers from bench artifact)
- Tag:               c2-100k-mtp-2026-05-21-34.50

---

## Plan A: clean 10/10 validation

Status: **2 hours**. Risk: low. Outcome: claim formal champion.

The 2 outliers in the 10-iter validation (16.68 and 22.46 vs the 34.5 norm) batched
correctly on the server (cluster log confirms Rendezvous + Batched prefill: B=2)
and MTP-PROF showed steady acceptance 1.83/2 throughout. The drop is a
BENCH-SIDE timing artifact in `/tmp/bench_c2_temp0_7iter.py`.

### Hypothesis

`per_req` is computed server-side as `state.completion_tokens / gen_time_delta`,
where `gen_time_delta = _mlx_gen_elapsed_seconds(now) - state.generation_time_at_start`.
For a c=2 batched bench:

- Both streams' `_active_tasks[uid]` entries are created in
  `_submit_batched_eligible` BEFORE the batched prefill returns (line 1434).
  Their `generation_time_at_start = perf_counter()` is captured BEFORE prefill
  completes.
- For normal iters, decode wall is ~14s and prefill ~735s — but
  `generation_time_at_start` was set during prefill, so `gen_time_delta` =
  ~14s (correct).

  Wait — actually `generation_time_at_start` is set AFTER prefill_batched returns
  (line 1434 in batch_generate.py), so it IS at "decode start". For normal iters
  this gives gen_time_delta ≈ 14s → per_req ≈ 17.3.

- Outlier iters: bench-side wall is 760-768s (vs normal 752s). Server reports
  per_req = 8.34 → gen_time_delta = 30.7s. So the server's
  `gen_time_delta` for outlier iters is ~16s LONGER than normal even though
  the model produces tokens at the same rate.

The most likely cause: one of the two streams' submit_batched return path
takes ~15s longer (post-batched-prefill, before decode starts). That delays
the second stream's `_active_tasks[uid]` registration, so by the time the
bench polls `now`, the server's `gen_time_delta` measures the EXTRA 15s.

### A.1 Diagnose

```bash
# Add a per-stream wall-time log to _submit_batched_eligible that emits
# the timestamps for each task's _active_tasks entry creation. Compare
# good vs outlier iters.
#
# 1. Edit batch_generate.py:1434 add:
#    logger.info(f"_active_tasks[{uid}]  generation_time_at_start={time.perf_counter():.3f}")
# 2. Run 10 iters of c=2 100K temp=0.
# 3. Look at the per-stream timestamps for outlier iters — confirm one
#    lags ~15s.
```

Time: 30 min code + 2hrs bench = 2.5hrs

### A.2 Fix

Two candidate fixes; pick the simplest that aligns the timing:

**Option 1: Capture `generation_time_at_start` once for the WHOLE batched
prefill** (not per-stream). This means both streams' `gen_time_delta`
measurements start from the SAME wall instant. Eliminates any per-stream
scheduling delay between iter end and `_active_tasks` registration.

```python
# In _submit_batched_eligible after `prefill_batched(...)` returns,
# capture ONE generation_start_time and assign it to all tasks:
common_start = time.perf_counter()
for i, ... in enumerate(tasks):
    self._active_tasks[uid] = _EngineTask(
        ...,
        generation_time_at_start=common_start,  # ← shared, not per-task
    )
```

**Option 2: Change agg_tps to be computed from total_tokens / total_wall**
in the bench script. Less sensitive to server-side per-stream timing.

Option 1 is the structural fix and matches how the server semantically should
report (prefill is BATCHED so decode-start IS the same wall for both streams).

### A.3 Validate

Re-run the 10-iter bench. If all 10 iters now report 34.5 ± 0.5 agg, push the
tag `c2-100k-mtp-2026-05-21-34.50-clean-10iter` and update memory.

### Acceptance gate

- 10 of 10 iters ≥ 33 agg
- σ < 0.5

### Effort

~3-4 hours total (30 min diag + 2hrs bench + 30 min fix + 1hr re-bench).

---

## Plan B: push c=2 100K above 35 t/s

Status: ~5 days. Risk: medium. Outcome: 36-40 agg.

### Where the headroom is

From measured numbers:
- per_req at c=2 = 17.3 t/s
- per_req at c=1 = 30 t/s
- c=2 wall = ~1.85x c=1 wall (matches MTP-OFF baseline scaling)
- acceptance at c=2 (temp=0 long context) = 1.83/2 = 92% per slot

Two levers:

1. **Lift c=1 per_req** (the ceiling driver) — c=2 scales ~linearly with c=1,
   so a +N% to c=1 multiplies into c=2.
2. **Higher tokens/cycle at the SAME wall** — i.e., increase γ or use a
   tree.

### B.1 Linear γ=3 at c=2 (1 day)

The mtp_module's `draft_tokens` is γ-agnostic (only the tree path has a
`gamma <= 2` assert). Try linear γ=3.

Expected:
- Step-2 P(top-1) likely ~0.30 (extrapolating from step 0=0.78, step 1=0.42).
- E[tokens/cycle] at γ=3 ≈ 1 + p₀ + p₀·p₁ + p₀·p₁·p₂
  ≈ 1 + 0.78 + 0.33 + 0.10 = 2.21 vs γ=2's 2.11. **+4.7% tokens.**
- Verify L_q = γ+1 = 4 (vs 3 at γ=2). Verify wall grows ~12% per skill's
  measured per-token cost.
- Net t/s: ~+4.7% tokens / +12% wall ≈ NET LOSS at the chained-MTP regime.

So γ=3 linear ALONE probably doesn't help. But...

### B.2 Eagle-style soft embedding at step 1 (2-3 days)

Step 1's P(top-1) = 0.42 is the chained-MTP alpha bottleneck. Eagle replaces
the hard `embed_tokens(argmax_id)` chain link with a probability-weighted
embedding mixture from step-0's full distribution. Captures the uncertainty
that the hard argmax throws away.

Per the Phase 8 plan I wrote earlier (in
`.hermes/plans/2026-05-20_phase8_beating_linear.md`), implementing this is
~2-3 days:

1. **mlx-lm change** (3 hours):
   ```python
   # In DeepseekV4MTPModule.__call__ (line 2468), accept a soft_emb
   # side-channel kwarg. When set, use it instead of embed_tokens(next_token):
   if soft_emb is not None:
       emb = soft_emb
   else:
       emb = embed_tokens(next_token)
   ```
   Side-channel via thread-local (same pattern as `_TREE_VERIFY_CTX` from Phase 5).

2. **exo change** (3 hours): in `draft_tokens` compute the soft embedding
   from step 0's logits:
   ```python
   K_EAGLE = int(os.environ.get("EXO_DSV4_MTP_EAGLE_K", "8"))
   probs = mx.softmax(logits_step0, axis=-1)
   topk_ids = mx.argsort(-logits_step0)[..., :K_EAGLE]
   topk_probs = mx.take_along_axis(probs, topk_ids, axis=-1)
   topk_probs = topk_probs / topk_probs.sum(axis=-1, keepdims=True)
   topk_embs = embed_tokens(topk_ids)  # (B, L, K, hidden)
   soft_emb = (topk_embs * topk_probs[..., None]).sum(axis=-2)
   ```
   For step-1 forward, set the side channel to `soft_emb`.

3. **microbench validation** (2 hours): build `bench/mtp_eagle_microbench.py`
   that measures step-1 P(top-1) hard vs soft on 200 fixed-prompt
   decode cycles. Gate: lift ≥ +3pp.

4. **cluster validation** (2 hours): if microbench passes, restart cluster
   with `EXO_DSV4_MTP_EAGLE_K=8`, run sanity probe + 3-iter c=2 100K
   bench.

Expected:
- Step-1 P(top-1) lift: +5pp (0.42 → 0.47).
- γ=2 tokens/cycle: 1 + 0.78 + 0.78×0.47 = 2.15 vs current 2.11 (+1.9%).
  Modest at γ=2. Larger at γ=3.
- Combined with B.1 (γ=3): 1 + 0.78 + 0.78×0.47 + 0.78×0.47×~0.35 ≈ 2.28
  tokens/cycle, +8% over γ=2 baseline.

Risk: the MTP head was trained on hard `embed_tokens(id)` inputs only;
feeding a distribution mixture is out-of-distribution. If the head is robust
this gives a free lift; if not it makes step-1 WORSE.

Pre-flight gate: the microbench MUST show ≥ +3pp before deploying.

### B.3 c=1 ceiling lift (the bigger lever, ~weeks)

c=1 is at 30 t/s. Every +1 t/s at c=1 propagates to +1.85 agg at c=2.
The structural bottleneck at c=1 is the per-cycle verify wall (~62ms at
L_q=3, 100K context). To lift that requires:

- Eagle (B.2) — described above.
- Better MTP head: dedicated step-N heads (Medusa-style) — see Phase 8
  Plan B. ~2 weeks. Skipped per user direction.
- Lower per-token attention cost. Structural, very hard.

### Recommended Plan B sequence

1. **B.1 + B.2 combined** (3 days total).
2. **Gate**: if microbench shows step-1 P(top-1) ≥ 0.47, ship.
3. **Bench**: c=2 100K with `EXO_DSV4_MTP_EAGLE_K=8` and γ=3.
4. **Expected outcome**: c=1 lifts from 30 → 31-32; c=2 lifts from 34.5 → 36-38.

### Acceptance gate for Plan B

- 10 iters c=2 100K agg ≥ 36 (~3% above current).
- σ < 0.5.
- No c=1 regression (sanity probe ≥ 29).

---

## Recommended execution order

**Run A first (3-4 hours).** It's cheap and gets us a clean 10/10 champion claim
that locks in the 34.5 floor. The fix is mechanical (move
`generation_time_at_start` to a single shared timestamp for batched-prefill
streams).

**Then B.1 + B.2 combined (3 days).** Eagle is the biggest single lever
for c=1 lift. Combined with γ=3 it could push c=2 to 36-38 agg. If the
microbench fails at step-1 P(top-1) gate, we abort B.2 and try B.1 alone
(modest but quick).

If B fails outright, the 34.5 milestone stands as the production champion.
