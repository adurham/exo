# Can we use a different MTP head for higher γ?

**Date:** 2026-05-19 ~15:15 CDT
**User question:** "can't we use a different MTP head then?"

## Short answer: No, not from existing weights.

## Long answer

### What we have

- **DSv4-Flash ships ONE MTP head.** The `model-00034-of-00034-mtp.safetensors`
  file contains exactly `mtp.0.*` keys (only head index 0). Verified by
  enumerating the safetensors keys. The model architecture has
  `num_nextn_predict_layers: 1` — there is no MTP head #1 in the
  weights.

- **DeepSeek V3 / V4-Flash papers confirm:** the MTP module is a single
  transformer layer trained as an auxiliary objective. The "k+1
  speedup" theoretical claim assumes chaining the same single head
  k times during inference, with acceptance degrading with each step.
  This is exactly the gamma=3 alpha_3=0.37 measurement we made earlier
  today.

- **The local `mtp-qwen35-397b-4bit` model is for Qwen3.5-397B**, a
  different architecture with different hidden dims. Cannot be used
  as a DSv4 draft.

### What can't be done (in this session)

- **Add a second MTP head with no training**: Not possible. The
  head needs to be trained against the target model's outputs. Random
  init would fail the quality probe immediately.

- **Train a second MTP head ourselves**: Possible in principle but:
  - The single MTP head is already 14B params (~10.6B unique + 3.4B
    shared) per the DeepWiki spec. For DSv4-Flash it's smaller (8-bit
    qsuant), but still significant.
  - Training requires GPU-cluster time + careful curriculum + data
    pipeline. Mac Studio M4 Max is inference hardware; training would
    take weeks of cluster time we don't have.
  - Quality verification is non-trivial — needle test isn't enough,
    need broader eval.

### What MIGHT work (cheaper experiments)

- **predict_from_hidden chain**: There's a method
  `mtp_module.py:536:predict_from_hidden(prev_hidden)` that's defined
  but never called. It skips the lm_head→argmax→embed roundtrip and
  feeds the previous hidden directly. Could be tried in `draft_tokens`
  but the model was NOT trained for this input pattern — it's a
  shortcut that may produce worse drafts.

- **Token-tree drafting (Eagle-style)**: Instead of chained γ drafts,
  generate a tree of drafts at each step using top-K from the existing
  MTP head. Verify all 2^γ branches in one batched forward. Accept the
  longest matching branch. The math allows effective_alpha to reach
  0.6-0.7 even with the existing single MTP head, because we get more
  chances at each step. This is the realistic path to 35+ t/s.

  Cost: 2-3 days of focused work (mlx-lm draft logic + exo accept
  logic + tree attention mask construction in the verify forward).

### Honest bottom line

There is no free MTP head improvement available. The only ways forward:
1. Token-tree drafting with the EXISTING head (2-3 days, real engineering)
2. Train a fresh multi-step-capable MTP head (weeks, requires training infra)
3. Wait for DeepSeek to ship V4-Flash with multi-head MTP (out of our control)
4. Accept 30.06 t/s as the per-stream ceiling on this hardware

## Cluster state

Production baseline restored. Inference probe passes.
