# K>1 PP Batch SDPA Speculation (March 22, 2026)

## Overview

K>1 speculative decoding for PP (pipeline parallel) mode. Rank 0 drafts K tokens during idle time, batch-forwards them, and rank 1 verifies all K in a single forward pass — processing K SDPA queries against the KV cache simultaneously.

## Architecture

1. **Draft K tokens** (rank 0, idle time): 0.8B draft model generates d1..dK autoregressively
2. **Batch forward** (rank 0, idle time): forward [d1..dK] through rank 0's layers → K hidden states
3. **First-token check** (rank 0): compare incoming real token with d1
4. **Batch verify** (rank 1): process K hidden states in ONE forward pass → K logits → verify d2..dK
5. **Optimistic speculation**: rank 0 drafts NEXT K from d_K during rank 1's batch verify wait. On hit (~43% of steps), skips normal step — chains batch verify → batch verify.
6. **Rollback** (both ranks): on partial acceptance, restore cache + re-forward accepted tokens

## Protocol

- Rank 0 sends `(K+1,)` int32 header before hidden states
  - `header[0]=0` → normal step (1 hidden follows)
  - `header[0]=K` → batch verify (`header[1:]` = draft token IDs)
- All_gather: 2 values per rank for K>1 `[num_accepted, bonus_token]`
- K=1 accepts use the fast path (no header, 1-value all_gather)

## K Switching

Deterministic based on `</think>` token detection:
- Before `</think>`: K=1 (thinking tokens are unpredictable)
- After `</think>`: K=3 (output tokens are structured/predictable)
- Non-thinking models: K=3 always

## Performance (0.8B-8bit draft, K=3)

| Content Type | K=1 baseline | K>1 | Change |
|-------------|-------------|-----|--------|
| Factual | 45 tok/s | 54-64 tok/s | +20-42% |
| Code (after `</think>`) | 45 tok/s | 49 tok/s | +8% |
| Counting | 45 tok/s | 51-54 tok/s | +13-19% |
| Reasoning (thinking) | 45 tok/s | 45 tok/s | 0% (stays K=1) |
| Creative (thinking) | 45 tok/s | 44 tok/s | ~0% (stays K=1) |

## DeltaNet Architecture Ceiling

Qwen3.5-397B-A17B has 75% DeltaNet layers (45/60), 25% SDPA (15/60). Pattern: [D,D,D,S] repeating.

| Component | Per layer | K=1 total | K=3 total | Scales? |
|-----------|----------|-----------|-----------|---------|
| DeltaNet | 0.55ms | 6ms | 18ms (3x) | Linear (sequential recurrence) |
| SDPA | 2.0ms | 8ms | 8ms (1x) | Constant (batched queries) |
| **Total** | | **14ms** | **26ms** | |

DeltaNet accounts for 100% of the K>1 overhead. Three approaches to bypass it were investigated — all dead ends:

1. **SDPA-only verify + DeltaNet fixup**: 8ms verify + 26ms fixup = 34ms on full accept (worse than 26ms)
2. **Token-by-token DeltaNet + batched SDPA**: same compute, +2ms dispatch overhead (worse)
3. **Skip DeltaNet entirely**: corrupts recurrent state (breaks output quality)

**Conclusion**: 26ms batch forward already extracts the full available SDPA batch benefit. The remaining 18ms DeltaNet overhead is irreducible. Pure-attention models would see the full K× bandwidth improvement.

## Draft Model Comparison

| Model | Draft speed | K=1 thinking | K=3 output | Verdict |
|-------|-----------|-------------|-----------|---------|
| 0.8B-8bit | 3ms/tok | 45 tok/s | 62 tok/s | **Best for coding agents** |
| 2B-4bit | 5ms/tok | 37 tok/s | 64 tok/s | 22% slower on thinking |

Dual draft (0.8B for K=1 + 2B for K=3) not practical: both caches need every token processed to stay in sync, doubling draft overhead.

## PP=3 with MacBook

MacBook M4 Max 36GB (same bandwidth as Studios): PP=3 doesn't improve throughput. Same total compute regardless of split, extra pipeline stage adds ~1ms overhead. Only useful with 3+ identical nodes AND a much larger model.

## Key Implementation Details

### Optimistic speculation snapshot management
- `_orig_snap` saves pre-speculation snapshot before optimistic `_speculate_k` overwrites `_spec_snap`
- Finally block uses `_orig_snap` (if set) for cleanup, preventing ghost tokens in KV cache between requests
- `_orig_snap` cleared after rollback resolution (miss/partial), preserved on hit

### Draft cache
- Skip re-processing on partial accept (just restore snapshot, ~14ms saved)
- Draft cache diverges slightly but draft model is an approximation anyway

### Bugs found
1. **Lazy draft chain**: chaining K forwards without intermediate `mx.eval` → stale cache state → deadlock
2. **`_orig_snap` cleared before use**: TypeError crash on rank 0, rank 1 hung at recv
3. **Hidden state shape mismatch**: `(1,K,H)` left in `_cache_state` after batch verify → `recv_like` shape error

## Config

```bash
EXO_PP_DRAFT_K=3           # K tokens per batch verify
EXO_DRAFT_MODEL=mlx-community/Qwen3.5-0.8B-MLX-8bit
EXO_COMPILE_DECODE=0       # Required for speculation
```

## Files Modified

- `mlx-lm/mlx_lm/generate.py` — main implementation
- `src/exo/worker/engines/mlx/generator/generate.py` — `pp_think_end_token` passthrough
- `start_cluster.sh` — config
