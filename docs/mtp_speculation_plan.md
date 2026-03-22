# MTP Speculation Plan (Qwen3.5-397B-A17B)

## Background

Qwen3.5 models are trained with **Multi-Token Prediction (MTP) heads** — a built-in decoder layer that predicts future tokens from the main model's hidden states. This is an alternative to the current separate 0.8B draft model for speculative decoding.

## MTP Architecture

From `config.json`:
- `mtp_num_hidden_layers: 1` — one extra decoder layer
- `mtp_use_dedicated_embeddings: false` — shares embeddings with main model

Components (1553 weight tensors):
- `mtp.fc` — linear projection combining hidden state + token embedding
- `mtp.pre_fc_norm_hidden` + `mtp.pre_fc_norm_embedding` — RMSNorm pair
- `mtp.layers.0` — one full decoder layer (attention + MoE with 512 experts, 10+1 activated)
- `mtp.norm` — final RMSNorm
- Shared `lm_head` for logits

## MTP Forward Pass

```
input = norm_hidden(last_hidden_state) + norm_embed(token_embedding)
input = fc(input)
output = decoder_layer(input)  # attention + MoE
output = norm(output)
logits = lm_head(output)  # shared with main model
```

Multi-step: chain by feeding MTP output as the new hidden state + predicted token's embedding.

## Performance Estimate

| | Draft Model (0.8B) | MTP |
|---|---|---|
| Weight reads/tok | 1.13ms (619MB) | ~0.15ms (~80MB: 11 experts + attention) |
| Dispatch overhead | 1.87ms (24 layers) | ~0.5ms (1 layer) |
| **Total/tok** | **3ms** | **~0.65ms** |
| Expected acceptance | 66-91% | >90% (full model representations) |
| Memory | +0.8GB | +8.5GB (full MoE layer) |

**4.6x faster drafting** with higher acceptance.

## Current Status

- Original model has MTP weights (files 91-94 of 94 safetensors)
- MLX community quantization **strips them** (`sanitize()` filters `"mtp."` keys)
- Need to quantize MTP weights separately

## Implementation Plan

Feature flag: `EXO_MTP_SPECULATION=1` (default: 0, uses existing draft model)

### Phase 1: Quantize MTP weights
- Download MTP safetensors from original model
- Quantize to 4-bit matching main model scheme
- Modify `sanitize()` to retain MTP weights when flag set

### Phase 2: MTP module
- Add MTP class to `qwen3_5.py` (fc + norms + 1 decoder layer)
- Implement forward pass
- Verify correctness against reference implementation

### Phase 3: Speculation integration
- `_speculate_mtp(hidden_state)` path alongside `_speculate_k(token_id)`
- MTP uses main model's last hidden state (already available)
- No separate cache management (MTP has its own small KV)

### Phase 4: PP integration
- MTP weights on rank 0 (or last rank which has lm_head)
- MTP runs during idle time, same as draft model
- Batch verify unchanged — rank 1 verifies K tokens regardless of draft source

## Key Insight

The MTP layer has 512 experts but only activates 10+1 per token. The actual weight reads are ~80MB (not the full 8.5GB), making it much faster than the 0.8B draft model despite being architecturally larger.

## PP Mode: NOT VIABLE (Tested March 22, 2026)

MTP on rank 0 (partial hidden state): **0% acceptance** — garbage predictions.
MTP on rank 1 (full hidden state): ~7ms wait penalty makes it **slower than draft model**.

| Approach | Step time | Acceptance | Effective tok/s |
|----------|-----------|-----------|----------------|
| Draft model (0.8B, rank 0) | 24ms | 80% | ~10.2ms/tok |
| MTP on rank 0 | 24ms | 0% | N/A (broken) |
| MTP on rank 1 | 31ms | ~95% est. | ~11.6ms/tok |

MTP only viable for: TP mode (all ranks have full hidden), single device, or PP with 3+ ranks.

## Risk: MTP layer is SDPA (not DeltaNet)

The MTP layer uses standard attention (`self_attn`), not DeltaNet. This means:
- **mx.compile MIGHT work** for MTP (no ArraysCache issues)
- MTP's KV cache is standard KVCache (simple offset rollback)
- Could potentially compile MTP forward, further reducing the ~0.5ms dispatch overhead
