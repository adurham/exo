# K>1 PP Speculation Benchmarks (March 22, 2026)

## Setup

- **Cluster**: 2× Mac Studio M4 Max 128GB, PP mode (15 layers each)
- **Model**: Qwen3.5-397B-A17B-4bit
- **Draft model**: Qwen3.5-0.8B-MLX-8bit (3ms/tok)
- **Config**: `EXO_PP_DRAFT_K=3`, `EXO_COMPILE_DECODE=0`
- **K switching**: deterministic — K=1 during `<think>`, K=3 after `</think>`
- **K=1 baseline**: 45 tok/s (from March 21 K=1-only speculation)

## Synthetic Benchmarks

### Short output (100 tokens)

| Test | Wall time | Decode tok/s | eff_k | Accept |
|------|-----------|-------------|-------|--------|
| Factual (speed of light) | 3.7s | 53.0 | 1 | 91.0% |
| Code (fibonacci) | 4.3s | 44.3 | 1 | 66.0% |
| Reasoning (127×389) | 4.0s | 49.4 | 1 | 82.0% |

### Medium output (500 tokens)

| Test | Wall time | Decode tok/s | eff_k | Accept |
|------|-----------|-------------|-------|--------|
| Factual (US presidents) | 11.5s | 52.8 | 1 | 92.6% |
| Code (merge sort) | 13.5s | 43.5 | 3 | 69.9% |
| Reasoning (TCP congestion) | 12.6s | 47.4 | 1 | 76.0% |
| Creative (ocean poem) | 12.3s | 47.8 | 1 | 77.2% |

### Long output (1000 tokens)

| Test | Wall time | Decode tok/s | eff_k | Accept |
|------|-----------|-------------|-------|--------|
| Code (HTTP server) | 30.6s | 34.7 | 3 | 54.1% |
| Reasoning (CPU pipeline) | 24.1s | 45.5 | 1 | 73.5% |
| Creative (time traveler) | 24.9s | 43.5 | 1 | 66.8% |

### Temperature variation (500 tokens, same prompt)

| Temp | Wall time | Decode tok/s | eff_k | Accept |
|------|-----------|-------------|-------|--------|
| 0.0 | 13.5s | 43.9 | 3 | 69.2% |
| 0.7 | 12.6s | 47.9 | 3 | 71.7% |
| 1.0 | 11.7s | 51.6 | 3 | 73.9% |

Temperature does not degrade performance — argmax verification is temperature-independent.

## Practical Coding Benchmarks

| Test | Prompt tok | Output tok | Wall time | Decode tok/s | eff_k | Accept |
|------|-----------|-----------|-----------|-------------|-------|--------|
| Implement LRU cache | 34 | 800 | 20.2s | 43.9 | 1 | 67.9% |
| Debug code (merge intervals) | 110 | 500 | 13.8s | 44.1 | 3 | 76.0% |
| Explain async retry code | 92 | 500 | 12.5s | 49.3 | 1 | 84.4% |
| Multi-turn: add TTL to KVStore | 155 | 800 | 18.7s | 50.3 | 3 | 85.1% |
| Code review (RateLimiter) | 252 | 1000 | 26.1s | 43.6 | 1 | 67.7% |

## Analysis

### Performance by content phase

| Phase | eff_k | Typical tok/s | vs K=1 baseline |
|-------|-------|--------------|-----------------|
| Thinking (`<think>`) | 1 | 43-53 | 0% to +18% |
| Output after `</think>` — predictable | 3 | 50-53 | +11-18% |
| Output after `</think>` — code with low accept | 3 | 34-44 | -2% to -24% |

### Key findings

1. **No regressions on thinking content**: K=1 during `<think>` matches or exceeds the 45 tok/s baseline
2. **Multi-turn is the sweet spot**: 50.3 tok/s with 85.1% acceptance — KV cache reuse + predictable output structure
3. **Long code generation degrades**: 34.7 tok/s at 1000 tokens — thinking content dominates, K=3 output phase has 54% acceptance (code is variable)
4. **Temperature is irrelevant**: 44-52 tok/s across temp 0-1.0 on the same prompt
5. **Short outputs don't benefit**: 100 tokens is mostly thinking + prefill overhead. K=3 never activates.
6. **Factual thinking is fast**: 53 tok/s at K=1 — high acceptance (91%) even without K=3 promotion

### Bottleneck breakdown

| Component | Time | Notes |
|-----------|------|-------|
| Draft (0.8B, per token) | 3ms | 1.13ms weights + 1.87ms dispatch (irreducible) |
| Main model forward (rank 0, K=1) | 7ms | 15 layers |
| Main model forward (rank 1, K=1) | 14ms | 15 layers + lm_head |
| Main model batch forward (rank 1, K=3) | 26ms | 8ms SDPA + 18ms DeltaNet |
| RDMA send/recv | ~1ms | Per pipeline stage |
| All_gather | ~1ms | Per step |
