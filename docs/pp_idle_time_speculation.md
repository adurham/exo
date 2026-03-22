# PP Idle-Time Speculation with Qwen3.5-0.8B Draft Model

*March 21, 2026 — Verified stable at 104K context*

## Overview

During PP decode on Qwen3.5-397B-A17B-4bit, rank 0 has ~14ms of idle GPU time
waiting for rank 1 via all_gather. This feature uses that time to run a small
draft model (Qwen3.5-0.8B-8bit, 980MB) + speculative main model forward. When
the draft matches, the pre-computed hidden state is sent immediately on the next
step — skipping rank 0's full compute cycle.

## Performance

| Metric | Value |
|--------|-------|
| Baseline (no speculation) | 27.4 ms/tok |
| With speculation | 22-24 ms/tok (**~20% speedup**) |
| Acceptance rate | 72-82% (varies with temperature) |
| Prefill throughput | 288-403 tok/s |
| Max verified context | 104K tokens (12-turn conversation) |
| Decode at 50K context | ~31 ms/tok (SDPA scaling) |

## Configuration

```bash
EXO_COMPILE_DECODE=0        # compile MUST be off on all ranks
EXO_DRAFT_MODEL=mlx-community/Qwen3.5-0.8B-MLX-8bit
EXO_DRAFT_TOKENS=3
EXO_PREFILL_STEP_SIZE=512   # 256 effective chunks in PP (avoids heartbeat timeout)
EXOMEMORY_THRESHOLD=0.98    # prevent needless KV cache eviction
```

## How It Works

1. PP detection is independent of `EXO_COMPILE_DECODE` — `_pp_step` (phase-
   separated decode) is created whenever PP layers are detected.
2. After rank 0 sends hidden to rank 1 (Phase 3), it runs:
   - Draft model forward (~3ms) → predicts next token
   - Snapshot main model cache
   - Speculative main model forward (~14ms) → pre-computes hidden for draft token
3. After all_gather, next step checks: does real token match draft?
   - **Accept (~80%)**: use pre-computed hidden, skip compute entirely
   - **Reject (~20%)**: restore cache from snapshot, run normal compute
4. **Finally block**: restores cache from snapshot if generation ends with pending
   speculation — prevents KV prefix cache divergence between PP ranks

## Critical Design Decisions

### 1. EXO_COMPILE_DECODE must be 0
mx.compile produces lazy output arrays that persist in the KV prefix cache.
On the next request, these stale compile artifacts cause "eval without primitive"
crashes. Compile on/off must be consistent across ALL ranks — mixed mode poisons
the cache. Cost: ~1ms/step per rank from losing graph-build optimization.

### 2. PP phase separation is independent of compile
`_pp_step` (recv → compute → send → speculate → all_gather) is created whenever
PP is detected, not only when `EXO_COMPILE_DECODE=1`. The `_compiled_decode_active`
flag on PipelineLastLayer controls whether it handles communication internally or
defers to `_pp_step`.

### 3. Never override the user's sampler/temperature
Forcing `temp=0.0` caused a deadlock on thinking-mode requests. The draft model
uses argmax internally; the main model uses the user's sampler.

### 4. Skip logprobs extraction on non-last PP ranks
Non-last ranks skip lm_head, producing logprobs with shape (1) instead of
(vocab_size). Guard: `if task.logprobs and out.logprobs.size > _n_logprobs`.

### 5. Draft model loads on PP device_rank, not JACCL group rank
JACCL assigns group ranks non-deterministically. Draft loading uses
`shard_metadata.device_rank == 0` with `group.rank() == 0` fallback for TP.
Auto-downloads via `huggingface_hub.snapshot_download` if not present.

### 6. KV prefix cache cleanup in finally block
The last decode step always speculates. The finally block restores from snapshot
so both ranks' caches match for KV prefix reuse.

### 7. Runner failure exits process
After a crash, GPU memory isn't released and RDMA state is dirty. `_kill_runner`
raises `SystemExit(1)` for clean restart.

### 8. Speculation overflows idle time by ~3ms
Idle budget: 14.3ms. Speculation: 17ms. Net positive because accept steps save
14ms while overflow costs only 3ms: `0.80 × 14 - 3 ≈ 8ms net`.

### 9. No post-prefill mx_barrier or mx.synchronize
Both deadlock at high context (160s+). Prefill pipeline sends already sync ranks.
Decode's `_pp_step` handles per-step sync. `generation_stream` IS the default
stream so `mx.synchronize` is redundant.

## Files Modified

- `mlx-lm/mlx_lm/models/cache.py` — `snapshot()`/`restore()` for KVCache, ArraysCache, CacheList
- `mlx-lm/mlx_lm/generate.py` — `_pp_step`, `_speculate()`, finally-block cleanup, logprobs guard
- `src/exo/worker/engines/mlx/auto_parallel.py` — `PipelineLastLayer._speculative_mode`
- `src/exo/worker/engines/mlx/generator/generate.py` — PP detection, draft routing, logprobs guard
- `src/exo/worker/engines/mlx/utils_mlx.py` — `device_rank` draft loading, auto-download
- `src/exo/worker/plan.py` — `SystemExit(1)` on runner failure
- `src/exo/master/api.py` — client disconnect cancellation
- `start_cluster.sh` — all configuration

## Lessons Learned

1. **Verify arithmetic before implementing.** 13ms + 3ms > 14ms — one line of math.
2. **mx.compile is all-or-nothing across ranks.** Mixed mode poisons KV prefix cache.
3. **Never override user-facing parameters.** Forced temp=0 deadlocked thinking mode.
4. **Non-last PP ranks produce garbage logits.** Guard logprobs extraction.
5. **JACCL group rank is non-deterministic.** Use PP device_rank.
6. **Test with webui parameters.** API tests miss logprobs, streaming, thinking — each triggered a different crash.
7. **`grep -i error ~/exo.log` first.** Caught exceptions logged as WARNING are invisible to crash-focused patterns.
8. **Preserve logs across restarts.** Use `>>` not `>`.
9. **PP phase separation doesn't require compile.** Decouple communication from compile wrapper.
