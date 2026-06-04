# Phase 8: Beating the 30 t/s Linear Baseline on DSv4 at 75K c=1

Status: **PLANNING ONLY — no code changes**. This document plans three
candidate efforts to lift the production DSv4-Flash-8bit c=1 75K t/s
ceiling above the current `baseline-2026-05-18-mtp-g2-topk512-30.06`.

Background: Phase 6B fixed token-tree drafting correctness (29.95 t/s);
Phase 7 tested six tree configurations (29.7–29.95 t/s) and confirmed
the tree approach cannot beat linear at this regime. The structural
bottleneck is verify cost at 87% of cycle wall, with the verify floor
set by per-token KV-attention over the 75K-prompt context (L_q
reduction from 7→5 saved only 1.5%). To lift t/s we have to attack
either (a) per-slot acceptance rate, or (b) verify wall directly.

## Grounding numbers (measured)

```
Linear γ=2 production:           30.06 t/s, σ=0.06, 3-iter at 75K c=1
  cycle ~69ms, tokens/cycle ~2.07
  verify dominates ~87% of cycle (~43ms at L_q=3)
  alpha (per Phase 1.2):
    step 0 P(top-1) = 0.78,  P(top-2) = 0.85   → top-2 lift  1.085x
    step 1 P(top-1) = 0.42,  P(top-2) = 0.55   → top-2 lift  1.32x
    compound:  P(accept k=2) = 0.78 × 0.42 = 0.327

Tree K=2 γ=2 best (Phase 6B + 7):  29.95 t/s, σ=0.05, 3-iter
  cycle ~71ms, tokens/cycle ~2.12 (only +2.4% over linear)
  verify ~62ms at L_q=7 (+44% wall growth ate the +13% draft win)
```

Per Phase 1.2 the chained-MTP step-1 (P(top-1)=0.42) is the alpha-
limiting step. Lifting it is the highest-leverage perf knob in the
draft side. The verify side has a ~52ms per-cycle floor that nothing
in tree-drafting design can break — that floor lives in the per-layer
KV attention shape.

---

## Plan A: Eagle-style hidden refinement for step-1 MTP

**Effort**: 2–3 days. **Risk**: medium. **Expected lift**: +5–8% t/s
(32 → 33 t/s). **Pre-req**: small ablation bench (~3 hours).

### The idea

Linear MTP step-1 takes the step-0 MTP head's argmax token id and
embeds it via `embed_tokens(arg_max_id)`, feeding `(prev_hidden,
embedded_id)` to the same MTP block. This throws away ALL the
distributional information the step-0 head produced — the head's full
top-K probability over the vocabulary. At long context, P(top-1) at
step-0 is only 0.78; the remaining 0.22 of mass holds information the
chain ignores.

Eagle-2 (Li et al, ICML 2024) replaces the token-embedding lookup
with a **probability-weighted embedding mixture**:

```
emb_step1 = sum_k=1..K  P(token_k)  *  embed_tokens(token_k)
```

…i.e., a soft, distribution-aware "next token" representation. The
chain is then `(prev_hidden, emb_step1) → MTP block → step-1 logits`.
Eagle reports +1.5x-2x throughput vs vanilla MTP at the same model
quality.

For DSv4 specifically: the MTP block at `mlx_lm/models/deepseek_v4.py:
2451` (`DeepseekV4MTPModule.__call__`) does the embed lookup at
line 2468: `emb = embed_tokens(next_token)`. The change is to replace
`next_token: mx.array (B, L)` with a **soft embedding**: `emb: mx.array
(B, L, hidden_size)` computed externally and passed in.

### Why it should lift step-1 P(top-1)

Step-1's job: given (h_prev, emb_y), predict the next token y_next.
At long context y is uncertain (top-1 only 0.78), but the model has
already computed THE FULL DISTRIBUTION over y. By weighting the
embedding by that distribution we propagate the uncertainty forward
instead of collapsing it. Empirically Eagle papers show this lifts
chained P(top-1) by ~10–15pp.

Eagle-2's exact recipe also includes a small "refiner" MLP that takes
(h_prev, soft_emb) and produces a refined hidden before the MTP block.
For DSv4 v1 we can SKIP the refiner (no extra weights) and just use
the soft embedding directly. If that doesn't lift acceptance enough,
v2 trains a refiner on a fixed dataset.

### Architecture

Three changes:

1. **mlx-lm `DeepseekV4MTPModule.__call__`**: accept either
   `next_token: mx.array(B, L)` int **OR** `soft_emb: mx.array(B, L, hidden_size)`.
   When `soft_emb` is provided, skip the `embed_tokens(next_token)`
   lookup and use `soft_emb` directly as the embedding input.
   Default `soft_emb=None` preserves backward compat.

2. **exo `mtp_module.draft_tokens`** (the linear γ-chain):
   - Step 0: as today. Get full logits + top-K argsort.
   - Compute `probs = mx.softmax(logits, axis=-1)` and the soft
     embedding for step 1.

   ```python
   K_EAGLE = int(os.environ.get("EXO_DSV4_MTP_EAGLE_K", "8"))
   topk_ids = mx.argpartition(-logits_step0, K_EAGLE, axis=-1)[..., :K_EAGLE]
   topk_probs = mx.take_along_axis(probs, topk_ids, axis=-1)
   topk_probs = topk_probs / topk_probs.sum(axis=-1, keepdims=True)  # renormalize
   topk_embs = embed_tokens(topk_ids)            # (B, L, K, hidden)
   soft_emb = (topk_embs * topk_probs[..., None]).sum(axis=-2)  # (B, L, hidden)
   ```

   - Step 1: pass `soft_emb` to MTP, NOT the argmax token.
   - Step 1's OUTPUT argmax is still used as the draft token for the
     verify pass.

3. **`draft_tokens_topk`** (tree path): same change to step-1
   expansion. Top-2 d1 candidates still come from step-0's top-K, but
   each d1's d2 children are computed from a soft-embedding chain
   (NOT from each d1's own argmax). This is slightly different from
   the plain tree — each d1 sibling's d2 inherits the SAME soft_emb,
   so their d2 candidates may overlap. v1 just takes them as-is; v2
   could anti-correlate by re-sampling from non-d1 mass.

### Acceptance gates

Pass criteria:
- Microbench (`bench/test_mtp_eagle.py`, new): step-1 P(top-1) with
  soft embedding vs hard argmax, on a 200-cycle fixed-prompt probe.
  PASS gate: `+5pp` lift (0.42 → 0.47+).
- Cluster bench: 3-iter at 75K c=1, EXO_DSV4_MTP_EAGLE_K=8.
  PASS gate: ≥31.5 t/s mean, σ<0.3, all iters ≥31.

If step-1 P(top-1) lift is < 3pp in the microbench, abort BEFORE the
cluster bench. The Eagle paper's lift comes from training; we're
betting that the inference-time soft-embedding alone captures the
bulk of the lift — that's the risk.

### Implementation phases

**A.1 Soft-embedding microbench** (2 hours):
- New script `bench/mtp_eagle_microbench.py` that loads the
  laptop-sized DSv4-Flash-8bit OR a smaller stand-in, runs both hard-
  argmax and soft-embedded step-1 over 200 prefill+decode steps from a
  fixed 4K prompt, measures step-1 P(top-1) and P(top-2) for each.
  - Hard reference: current `draft_tokens` at γ=2.
  - Soft variant: replicate just step-1 with the soft-embedding mixture.
- Output JSONL with `step1_p_top1_hard` vs `step1_p_top1_soft`.
- Decide: continue (lift ≥3pp) or abort.

**A.2 mlx-lm code change** (3 hours):
- Edit `DeepseekV4MTPModule.__call__` to accept `soft_emb` kwarg.
- Add `_set_soft_emb(emb)` and `_clear_soft_emb()` thread-local side
  channel, mirroring the tree-verify pattern from Phase 5. Avoid
  threading a new kwarg through the entire stack.
- Microbench locally to confirm no regression at `soft_emb=None`.
- Commit + push to adurham/mlx-lm.

**A.3 exo draft_tokens change** (2 hours):
- Edit `mtp_module.draft_tokens` and `draft_tokens_topk` to compute
  the soft embedding when `EXO_DSV4_MTP_EAGLE_K > 0`.
- Add `EXO_DSV4_MTP_EAGLE_K` env-var forwarding to start_cluster.sh.
- Microbench locally.

**A.4 Cluster deploy + bench** (2 hours):
- `uv lock --upgrade-package mlx-lm && uv sync`; commit exo + push.
- Restart cluster with `EXO_DSV4_MTP_EAGLE_K=8`.
- Quality probe at 100K (FALCON-MERCURY-7749 needle).
- 3-iter c=1 75K bench.
- If pass: 10-iter validation; tag champion.

### Pitfalls

1. **Soft-embedding numerically diverges from hard-argmax**. The MTP
   head was trained on `embed_tokens(token_id)` inputs only; feeding
   a distribution-weighted mixture is OUT OF DISTRIBUTION. Eagle's
   recipe trains the refiner on this; we're skipping training. The
   prediction quality might be WORSE than hard, not better, especially
   at high-confidence step 0 (where the mixture is close to embed of
   top-1 anyway).

2. **`embed_tokens` is shared between target model and MTP**. The
   embedding table is large (~vocab × hidden = 129K × 4096 = 0.5GB
   at bf16). Computing `embed_tokens(topk_ids)` with K=8 per draft
   step adds modest memory bandwidth at draft time. Not a wall
   killer.

3. **TP broadcast invariant**. Tree draft already broadcasts top-K
   IDs from rank 0 across TP ranks (`broadcast_from_canonical`). The
   soft embedding is a derived float vector — broadcast it the same
   way, OR ensure each rank computes it identically from the
   broadcasted top-K IDs + probs (cheaper, no extra collective).

4. **`mx.argpartition` vs `mx.argsort`** — argsort returns sorted
   indices; argpartition returns top-K unsorted. We need probs to
   match the IDs, so the cleanest is argsort + slice (as done in
   `draft_tokens_topk`). Use the same pattern.

### Expected outcome

- step-1 P(top-1) lift in microbench: +3 to +8pp (0.42 → 0.45–0.50).
- Linear γ=2 tokens/cycle: 2.11 → 2.20–2.30.
- Cluster bench: 31.5 to 32.5 t/s at c=1 75K. Σ likely ~0.15.

If we hit 32 t/s, that's a +6.5% lift over the 30.06 baseline. If
microbench shows the lift but cluster doesn't, the issue is
distributional drift compounding through verify — abort and write
findings.

---

## Plan B: Dedicated step-1 MTP head (Medusa-style)

**Effort**: 1–2 weeks (includes training). **Risk**: high (training
loop, hyperparameter search). **Expected lift**: +10–15% t/s
(33–34.5 t/s). **Pre-req**: training data, trainer infrastructure,
distillation pipeline.

### The idea

Current DSv4 has ONE MTP head trained for next-token prediction. The
chain at step 1 reuses that head with its own argmax as input — but
the head was NEVER trained on its own argmax. It's an inference
distribution mismatch.

Medusa (Cai et al, 2024) trains N parallel MTP heads — one per
prediction offset (heads 1, 2, …, N predict tokens at distance 1, 2,
…, N from the current hidden). At inference, each head consumes the
SAME `prev_hidden` (the target model's last hidden) and produces its
own logits over its own offset. No chaining → no distributional drift.

Adapted to DSv4: train a SECOND MTP head specifically for STEP-1
prediction (predicting the token at offset+2 from the target hidden
at offset+0). The first MTP head stays as-is. The result:

```
Cycle:
  draft_step0 = MTP_head_0(hidden, last_token).argmax  # P(top-1)≈0.78
  draft_step1 = MTP_head_1(hidden, last_token).argmax  # P(top-1)≈0.55 if trained well
  verify = main_model([y, draft_step0, draft_step1])
```

Step-1 acceptance lifts from 0.42 (chained) to ~0.55 (dedicated head).
Compound P(both accept) = 0.78 × 0.55 = 0.43 → tokens/cycle = 2.21
(vs current 2.11).

### Why it should help more than Plan A

Plan A (Eagle soft-embedding) is a runtime trick. Plan B trains a
NEW head specifically on the step-1 task. A head trained with
ground-truth supervision should outperform an inference-time
mixture by a large margin. Medusa papers report 2x+ throughput at
N=4 heads.

The downside is the training cost. We need:
- A representative training corpus.
- Distillation from the target model (or ground truth labels).
- ~1B tokens of training data minimum (rule of thumb for MTP heads).
- ~24 hours of GPU time on a Mac Studio (or cloud).

### Architecture

**B.1 Add a second MTP head slot**. The DSv4 architecture already
supports `num_nextn_predict_layers > 1` (line 309 in `dsv4_mtp.py`
checks `len(inner.mtp) <= mtp_idx`). Construct `mtp[1]` with the
same architecture as `mtp[0]`. Both heads share the target model's
`embed_tokens`, `final_norm`, and `lm_head`.

**B.2 Train it**. The training objective for head 1 is:

```
loss_1 = CrossEntropy(MTP_head_1(target_hidden_t, token_t), token_{t+2})
```

The target hidden is fixed (frozen DSv4 model); only `MTP_head_1`
weights are updated. Standard cross-entropy loss with label smoothing
0.1 and AdamW (lr=1e-4, weight decay=0.01).

Training data: ~1B tokens from a mix of:
- The model's pre-training corpus (if accessible).
- A high-quality general corpus (RedPajama, FineWeb-Edu sample).
- A code corpus (StarCoder data).

Run on the cluster overnight: 1B tokens × 2 epochs × ~10K tokens/s
training throughput = ~50 hours. Or rent an H100 for $2/hr for ~10
hours.

**B.3 Distillation alternative**: instead of training on labels, train
to MATCH the target model's outputs at offset+2:

```
loss_1 = KL(MTP_head_1(target_hidden_t, token_t) || target_model_logits_at_{t+2})
```

This is faster to converge (~100M tokens) and matches the target's
behavior directly. Standard Medusa training recipe.

### Acceptance gates

- B.1 (architecture): MTP head 1 instantiates without crashing, both
  heads forward independently.
- B.2 (training): loss curve converges; eval set P(top-1) at step 1
  ≥ 0.50 on holdout. Time-box: 5 days of training experimentation.
  Abort gate: if 5 days in P(top-1) ≤ 0.45, the training recipe
  isn't working; revert.
- B.3 (cluster bench): 3-iter at 75K c=1 with EXO_DSV4_MTP_DUAL_HEAD=1.
  PASS gate: ≥33 t/s mean, σ<0.3, all iters ≥32.5. 10-iter
  validation if step pass.

### Implementation phases

**B.1 Architecture** (1 day): add `mtp[1]` slot to DSv4 model config,
ensure shard works, weights load (initially random), smoke test
forward at non-zero shapes. Output: PR titled
"feat(dsv4): add MTP head 1 slot, untrained" gated behind
`EXO_DSV4_MTP_DUAL_HEAD=1`.

**B.2 Training infrastructure** (2–3 days):
- Pick a corpus (start with FineWeb-Edu 1B sample).
- Build a data loader that streams chunks of (target_hidden,
  token, token_at_offset+2) triples. Generate target hiddens by
  running the FROZEN target model in eval mode on the corpus.
- Standard LM training loop. Use mlx-lm's existing trainer scaffold
  if it exists; otherwise build minimal from scratch (the model is
  small — just one decoder block + norm + projections + HC head).

**B.3 Training run** (~24 hours wall):
- Run on 1 node (m4-1 only; the cluster's RDMA is for inference).
- Checkpoint every 50M tokens.
- Eval at end: P(top-1) at step 1 on holdout.

**B.4 Cluster integration** (1 day):
- Save trained `mtp[1]` weights to a separate file (so production can
  load it optionally).
- Plumb `EXO_DSV4_MTP_DUAL_HEAD=1` to instruct
  `mtp_module.draft_tokens` to call head 1 for step 1 instead of
  chaining head 0.
- 3-iter bench.

### Pitfalls

1. **Training corpus mismatch**. DSv4-Flash was likely trained on a
   specific blend; if our training corpus diverges, the new head
   will overfit to its corpus distribution and underperform at
   inference. Mitigation: use distillation (B.3) which only requires
   the target model's outputs, not a "right answer".

2. **The 8-bit quantization**. DSv4-Flash-8bit's hidden states are
   quantized. Training a new head on those quantized hiddens may
   capture quantization noise rather than signal. Use the BF16
   parent model for training; deploy the head at 8-bit (quantize
   the trained head's weights at the end).

3. **HyperConnection / HyperHead complexity**. The MTP module
   includes `attn_hc`, `ffn_hc`, `hc_head` per the DSv4 architecture
   (see `mlx_lm/models/deepseek_v4.py:2441-2445`). These are
   non-trivial to train from scratch. Initialize from `mtp[0]`
   weights and fine-tune ONLY, rather than train from random.

4. **Disk space + checkpoint shuffling**. Training checkpoints will
   be ~10GB each (single MTP module). Keep only 3 — best on eval
   plus most-recent two.

### Expected outcome

If training converges:
- step-1 P(top-1) lift: 0.42 → 0.55 (+13pp).
- Linear γ=2 tokens/cycle: 2.11 → 2.21.
- Cluster: ~33 t/s c=1 75K.

If we combine with tree drafting (K=2 γ=2): tokens/cycle ~2.45,
predicted t/s ~34 if verify wall stays ≤72ms.

The expensive part is the training. Most-honest read: this is real
ML eng work, not a 1-day hack. If 35 t/s is the target, this is the
plan that gets us closest.

---

## Plan C: Restructure DSv4 verify to amortise per-token attention

**Effort**: 1–2 weeks (requires understanding of DSv4 attention
internals). **Risk**: very high (touches MLX-level kernels, likely
runs into Metal-specific perf surprises). **Expected lift**: +10–25%
verify reduction → +5–15% t/s. **Pre-req**: detailed Metal-level
profiling.

### The idea

Phase 7 measured: verify at L_q=5 is 57ms; verify at L_q=7 is 62ms.
That's a 5ms / 2 = 2.5ms/token marginal cost — very low. But verify
at L_q=1 is ~30ms. So verify cost is `30 + 5.3 * L_q` ms approximately.

The 30ms FLOOR comes from per-token KV attention over 75K KV cache.
That's the L_q-invariant cost we'd like to cut. To beat linear (cycle
~69ms, ~32ms verify-amortized) we need to reduce that floor.

Three sub-plans:

### C.1 Sparse-attention TOPK reduction (cheapest)

**Effort**: 2–3 days. **Risk**: medium. **Expected lift**: +2–4%.

Current: `EXO_DSV4_INDEX_TOPK=512`. The sparse-attention path selects
top-K KV positions per query and attends only to those (line 2196
in `deepseek_v4.py`). At 75K context, attending to 512 of 75K is
already very sparse — but the indexer itself runs over the full
pool (~290 entries at compress_ratio=128), and the gather + SDPA
over 512 indices is non-trivial.

Lower TOPK = less attention work. Per memory pitfall #46 the 31.5
t/s "champion" at TOPK=160 was BROKEN (BOS-only at 100K). Need to
find the smallest TOPK that preserves needle quality.

**Procedure**:
1. Run quality probe (FALCON-MERCURY at 100K) at TOPK ∈ {256, 320,
   384, 448}. Lowest passing value defines the safe floor.
2. Bench c=1 75K at that TOPK. Measure t/s.
3. If lift is observable, validate with broader quality eval
   (lm-eval-harness MMLU subset or similar) to catch quality losses
   the simple needle doesn't.

Gates:
- Quality: needle_found=True at 100K AND MMLU-easy delta < 0.5%
  vs TOPK=512 baseline. If quality regresses, abort.
- Perf: t/s lift ≥ +1.5% (33 → 33.5 etc.) and σ<0.3.

This is low risk because TOPK is a runtime knob; no code changes.

### C.2 Layer skipping at verify time

**Effort**: 1 week. **Risk**: high (quality). **Expected lift**:
+10–20% verify reduction → +5–10% t/s.

The DSv4 architecture has 43 hidden layers. At verify time we run
all 43 every cycle. If the verify is to compare against a top-K
draft, the L_q=3 tokens (or 7 for trees) only need ENOUGH inference
depth to discriminate accepted-vs-rejected drafts. Lower layers
mostly do local syntactic patterns; the deep layers do semantic.

Idea: run a SHALLOW pass (e.g., 30 layers) at verify, sample the
acceptance decision. For tokens that pass shallow-accept, also run
the FULL pass on the bonus token only. The full pass is L_q=1,
running cheap at 30ms.

```
verify_shallow = model.forward_layers(0..29)([y, d1, d2])  # ~30ms
target_shallow = argmax(verify_shallow.logits)
accept_shallow = (target_shallow[i] == draft_i)
                                       # accept on shallow agreement
if shallow_n_accepted > 0:
    # The "bonus" token needs full 43-layer to be high quality.
    bonus_full = model.forward_layers(0..42)([y_accepted_tail])  # L_q=1
```

Expected cost:
- Cycle 1: shallow verify (L_q=3, 30 layers) ≈ 28ms.
- Cycle 2: full bonus forward (L_q=1, 43 layers) ≈ 30ms (parallel
  with next cycle's draft? maybe).

Net wall: 28ms instead of 43ms verify → savings ~15ms / cycle =
+22% on cycle wall → t/s up by ~+18%.

**MAJOR RISK**: shallow argmax may disagree with full argmax. Likely
~10–20% disagreement at long context. Disagreement means we accept
a token the FULL model wouldn't have generated — quality regression.

Mitigation: instead of accepting on shallow argmax alone, REQUIRE
match between shallow argmax AND draft. The shallow pass is just a
cheap "consensus check" — only accept when both agree. Acceptance
rate drops (shallow rejects ~25% of cycles vs full's 33%) but
quality is preserved (we never emit a token the model wouldn't
have).

Net: maybe a wash. This is the gnarliest of the three sub-plans
and most likely to regress quality. Not recommended as a starting
point.

### C.3 KV chunked attention (architectural)

**Effort**: 2+ weeks. **Risk**: very high. **Expected lift**: +5–10%
t/s.

The DSv4 attention reads the full 75K KV per token. The sparse
indexer picks 512 keys per query, but ALL 75K positions are still
materialized + dot-producted to build the index. If we re-architect
to keep the KV in CHUNKS and only un-pack the chunk a query needs,
we could reduce memory bandwidth.

This is a research project, not an engineering task. Skip in v1.

### Recommendation for Plan C

Pursue only **C.1 (TOPK reduction)**. It's a 2-day experiment with
low risk and potential +2% lift. Skip C.2 (layer skipping —
quality risk too high) and C.3 (research-grade).

If C.1 doesn't lift, the verify floor is genuinely structural at
this hardware/model/context point. Accept the 30 t/s ceiling and
move on.

---

## Comparison + recommended execution order

| Plan         | Effort   | Risk     | Expected t/s | Confidence |
|--------------|----------|----------|--------------|------------|
| A (Eagle)    | 2–3 days | Medium   | 31.5–32.5    | Medium     |
| B (2nd head) | 1–2 weeks| High     | 33–34.5      | High*      |
| C.1 (TOPK)   | 2 days   | Low      | 30.5–31      | Low-Med    |

*B is high-confidence on lift IF training converges; medium-confidence
that training will converge in the budget.

### Recommended execution order

1. **Start with C.1 (TOPK reduction)** — 2 days, low risk, mostly
   knob-twiddling + bench. If TOPK=384 holds quality and gives
   +1 t/s, lock it in regardless of next steps.

2. **Then Plan A (Eagle soft-embedding)** — 2–3 days, builds on
   existing tree-drafting infrastructure. If the soft-embedding
   microbench shows ≥3pp lift at step 1, push to cluster. If not,
   abort and skip to Plan B.

3. **Then Plan B (2nd MTP head, distillation)** — 1–2 weeks. Only
   pursue if A doesn't get us to 32 t/s. Highest-confidence lift if
   training converges, but expensive.

The intermediate result (A) at ~32 t/s is a meaningful win (+6.5%
over baseline) at much lower effort than B. If 32 t/s is acceptable,
stop there.

### What NOT to pursue (already ruled out)

- Tree drafting at any K, γ: 5 configs tested, all ≤29.95.
- Layer-skip verify (C.2): quality risk too high.
- KV chunked attention (C.3): research, not engineering.
- Async commit forward: saves ~3% only, but tree itself is dead.
- Concurrency c=2: orthogonal lever; if 35 t/s aggregate is the goal,
  c=2 amortises verify across 2 streams (per Phase 1.2 findings #4).

---

## Open questions for the user

1. **Is 35 t/s a hard target, or a stretch goal?** If a hard target,
   we need Plan B (training). If a stretch goal, Plans A+C.1 in
   sequence may suffice at ~32–33 t/s.

2. **Is training a 2nd MTP head feasible within infrastructure
   constraints?** Specifically: do we have training data
   accessible, and is mlx-lm's trainer in working shape?

3. **Is c=2 (aggregate throughput) an acceptable success metric, or
   must c=1 lift?** Production may not need c=1 — if 50 t/s
   aggregate at c=2 is fine, we can declare victory now.

If you greenlight Plan A first, I'll start with the soft-embedding
microbench (effort: 2 hours, no cluster restart needed).

---

## Appendix: alpha probe baseline (re-cited from Phase 1 findings)

```
                P(top1)   P(top2)   P(top3)   top1→top2 lift
step 0           0.7833    0.8500    0.8833    1.0851x
step 1           0.4167    0.5500    0.5833    1.3200x
```

Source: Phase 1.2 alpha probe, n=360 sample across 3 iters of 100K
c=1 g=2 (deterministic temp=0). Full data in
`.hermes/plans/2026-05-19_phase1_findings.md` lines 324–390.

Step-0 is near a ceiling; step-1 has runway. Plan A and Plan B both
target step-1. Plan C targets verify wall (orthogonal to alpha).
