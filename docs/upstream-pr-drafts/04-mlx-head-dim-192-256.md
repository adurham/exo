**Repo:** ml-explore/mlx
**Branch:** adurham:pr/sdpa-head-dim-192-256
**Target:** ml-explore/mlx:main
**Commits:** 4 (`f749ddfa`)

---

# Title

`feat: head_dim=192 + 256 support in fused SDPA (with kL-aware routing)`

# Body

## Summary

Add support for `head_dim=192` and `head_dim=256` to the fused full-attention `steel_attention` Metal kernel and the SDPA vector kernel. Routes to the fused path automatically when the key sequence length is large enough that the unfused naive path would exceed Metal buffer limits.

## Motivation

Several model families use `head_dim ∈ {192, 256}`:
- Qwen3.5-122B-A10B (head_dim=256)
- DeepSeek-V3.1 family (head_dim=192 for indexer, 128 for attn)
- MiniMax M2.5 (head_dim=192)

Before this PR, `sdpa_full_supported_head_dim` only included {64, 80, 128}, so these models fell back to the unfused path that materializes the full score matrix as a single matmul. At 32K+ context this creates 8+ GB single allocations that crash Metal's buffer allocator.

For short sequences the unfused path is actually slightly faster than fused at large head_dim, so the routing decision is gated on key sequence length.

## Approach

1. **Vector kernel**: instantiate `sdpa_vector` and `sdpa_vector_aggregation` for head_dim 192 and 256 (256 was already supported; this adds 192)
2. **Fused kernel**: instantiate `steel_attention` for `bd=192` and `bd=256` (registered in `steel_attention.metal`). The kernel template handles arbitrary `BD` via template parameter — no kernel code changes needed
3. **Dispatch gate**: extend `sdpa_full_supported_head_dim` to accept 192/256 when `key_sequence_length > MLX_SDPA_FUSED_THRESHOLD` (default 16384)
4. **`MLX_SDPA_FUSED_THRESHOLD` env var** to override the routing threshold (set to 0 to always use fused)

## Changes

- `mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.metal`: instantiate `bd=192` and `bd=256` shapes
- `mlx/backend/metal/kernels/scaled_dot_product_attention.metal`: instantiate vector kernel for head_dim=192 (256 was already done)
- `mlx/backend/metal/scaled_dot_product_attention.cpp`:
  - Add `MLX_SDPA_FUSED_THRESHOLD` env var
  - Add `sdpa_full_large_hd_ok` to `use_fallback()` gate
  - Extend `sdpa_vector_supported_head_dim`

## Test Plan

- [x] `python -m pytest python/tests/test_fast_sdpa.py` — 15 pass, 1 skipped
- [x] Manual verification — head_dim=192 and 256 in vector mode (single-token decode) and full mode (T_q=64, kL=32768) with bf16 and fp16 dtypes — all pass
- [x] Build clean on Apple Silicon M4 Max

### Known limitation (pre-existing)

`head_dim=192 + float32 + T_q > 8 + kL > MLX_SDPA_FUSED_THRESHOLD` runs into the 32KB threadgroup memory limit on Apple GPUs (`Threadgroup memory size (40448) exceeds 32768`). Same as the existing `head_dim=256 + float32` skip in `test_sdpa_chunked.py`. bf16 and fp16 work fine — those are the practical inference dtypes anyway.

A follow-up PR could parameterize the tile shape (`bq`, `bk`) per dtype to fit float32 at high `bd` — out of scope for this PR.

## Notes

This is a fix for the multiple GitHub issues opened about Qwen3.5 and DeepSeek-V3 long-context crashes (e.g. #3312). The kernel template was always capable — it just wasn't instantiated.
