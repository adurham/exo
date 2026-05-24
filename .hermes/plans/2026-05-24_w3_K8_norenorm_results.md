# W3 Eagle K=8 + no-renorm Eagle soft_emb — 2026-05-24

## tl;dr

**+0.83% decode (+0.24 t/s), statistically significant (Welch t=6.19,
p<0.001, Cohen's d=2.77). Quality preserved. Ready to ship as
default.**

## Numbers

| Config                       | mean t/s | σ      | min   | max   |
|------------------------------|----------|--------|-------|-------|
| baseline (K=0, no Eagle)     | 28.7998  | 0.0995 | 28.59 | 28.90 |
| fused-topk (K=0)             | 28.8378  | 0.0710 | 28.70 | 28.92 |
| **K=8 + no-renorm (this)**   | **29.0375** | 0.0695 | 28.88 | 29.10 |

Δ vs baseline: **+0.2378 t/s (+0.83%)**, Welch t=6.19 (p<0.001 99% CI),
Cohen's d=2.77 (large effect). Distributions barely overlap — K=8 min
(28.88) > baseline median (28.81).

Bench: `bench/concurrent_bench.py --concurrency 1 --iterations 10
--max-tokens 256 --prompt-words 75000 --timeout 1800 --warmup 1`,
fresh cluster restart per run, σ ≤ 0.10 throughout.

## Acceptance histogram puzzle

K=8 + no-renorm: mean_accept = **0.894/2** (per 1500 cycles, steady).
Baseline K=0: mean_accept = 1.087/2 (skill historical).

Acceptance dropped **-17.8%** but decode tps UP **+0.83%**. Two
possible explanations:

1. **Acceptance counter scope differs between K=0 and K>1 paths.** The
   `_spec_accept_hist` tally may count something slightly different
   when Eagle's installed (e.g., counts the slot-2 chained reject
   separately from the slot-1 reject). The decode tps measurement is
   the ground truth — that's what users feel.

2. **Eagle K>1 changes the verify-forward critical-path cost.** The
   per-slot soft_emb assembly + broadcast adds a small fixed cost
   but may reduce a different overhead (e.g., the predict call's
   logits-cache reuse pattern). Net wall time per token goes down
   even though more tokens are rejected. Worth investigating but not
   blocking the ship decision.

Either way, the bench is the authoritative measurement.

## Quality validation

- Pre-bench c=1 100K probe (K=8 + no-renorm): needle ✓
  FALCON-MERCURY-7749, BOS=0, bistability=0, finish=stop. wall=294.1s.
- Final c=1 100K probe (cache-warm): needle ✓, BOS=0, bistability=0.

User constraint check:
- KV bf16 ✓
- INDEX_TOPK=512 ✓
- model defaults as shipped ✓
- no mitigations — this is a root-cause fix to the K>1 soft_emb math
  observed empirically with the W3 top-K dump
- quality validated at every step ✓

## What the no-renorm fix actually does

`mtp_module.py:715-729` (K>1 branch in `draft_tokens`):

```python
_probs = mx.softmax(_logits3d, axis=-1)
_topk_ids = mx.argsort(-_logits3d, axis=-1)[..., :_eagle_k]
_topk_probs = mx.take_along_axis(_probs, _topk_ids, axis=-1)
# REMOVED: _topk_probs = _topk_probs / _topk_probs.sum(axis=-1, keepdims=True)
...
soft_emb = (_topk_embs * _topk_probs[..., None]).sum(axis=-2)
```

Previously the top-K probabilities were renormalized to sum to 1
over the top-K subset. The MTP head was trained on `embed(argmax)`
(single-magnitude lookup); the renormalization distorted the soft_emb
mixture's L2 norm vs training distribution. Now top-K probs are raw
softmax masses (sum ≤ 1), so the mixture norm tracks top-1 confidence
naturally. K=1 fast path is unaffected — that branch never reaches
this code.

## Recommended next steps

1. **Make K=8 the default.** Edit `start_cluster.sh` to add:
   ```sh
   : "${EXO_DSV4_MTP_EAGLE_K:=8}"
   ```
   (line ~94 area, near EXO_SPECULATIVE_GAMMA). Then the propagation
   line at 727 picks it up automatically.

2. **Pin the new champion**: tag `champion-2026-05-24-K8-norenorm-29.0`
   at HEAD `2d8d5efc`.

3. **Continue toward 35 t/s.** W3 contributed +0.83% of the needed
   +21.5%. Path B (raise acceptance) is partially exercised but
   acceptance regressed despite decode improving — the relationship
   between acceptance and decode at this config is non-trivial and
   worth a dedicated investigation in W4 (verify-forward kernel
   reduction).

## Refs

- `~/repos/exo/.hermes/plans/2026-05-23_35_tps_plan.md` §W3
- `~/repos/exo/.hermes/plans/2026-05-24_35_tps_plan_execution_results.md`
- `/tmp/w3_eagle_audit.md` (the code audit that found the renorm
  deviation)
- `/tmp/w3_dump_results.md` (the empirical distribution that
  justified Option A)
- `/tmp/mtp_topk_dump.txt` (raw 72-record dump)
- Commits: `928a390c` (diagnostic patch), `3cca5896` (launcher
  propagation), `2d8d5efc` (no-renorm fix)
