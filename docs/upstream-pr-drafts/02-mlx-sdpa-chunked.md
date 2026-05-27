**Repo:** ml-explore/mlx
**Branch:** adurham:pr/sdpa-chunked-dispatch
**Target:** ml-explore/mlx:main
**Commits:** 7 (`cf729e22`)
**Depends on:** PR #1 (sdpa-logsumexp-output) — chunked dispatch needs LSE output

---

# Title

`feat: chunked SDPA dispatch for long key sequences (>65K)`

# Body

## Summary

Add a chunked dispatch path for `mx.fast.scaled_dot_product_attention` that splits K/V along the sequence dimension, dispatches `steel_attention` per chunk with `output_logsumexp=True`, then merges partial outputs via a new `sdpa_chunked_reduce` Metal kernel.

This prevents GPU watchdog timeouts at >65K key tokens by bounding each kernel launch's working set.

## Motivation

On long-context inference (DeepSeek-V3.1/V3.2/V4 family with 100K+ context), a single fused SDPA dispatch with kL=100K+ tokens exceeds Metal's GPU watchdog limit. The kernel never returns — eval blocks indefinitely until macOS kills the process.

Empirically the threshold is around 65K kL on M2 Ultra and M4 Max. This PR splits the kernel launch at that boundary.

## Approach

1. **Chunk K/V** along the sequence dimension at a configurable threshold (`MLX_SDPA_CHUNK_THRESHOLD`, default 65536)
2. **Per-chunk dispatch** of `steel_attention` with `output_logsumexp=true` (uses PR #1)
3. **Merge** via new `sdpa_chunked_reduce` kernel: a numerically stable LSE-weighted combination of per-chunk outputs
4. **Preserve causal masking** by passing absolute K-offset per chunk; sinks applied to chunk 0 only
5. **NaN guard** in reduce kernel: skip zero-weight chunks where all keys are causally masked (otherwise 0 * NaN = NaN per IEEE 754)

## Configurability

- `MLX_SDPA_CHUNK_THRESHOLD` (default 65536) — kL above this triggers chunking
- `MLX_SDPA_CHUNK_SIZE` (default 32768) — chunk granularity

Set `MLX_SDPA_CHUNK_THRESHOLD=0` to disable chunking entirely (preserves pre-PR behavior).

## Changes

- `mlx/backend/metal/kernels/CMakeLists.txt`: register sdpa_chunked_reduce kernel
- `mlx/backend/metal/kernels/steel/attn/kernels/sdpa_chunked_reduce.h` (new): merge kernel template
- `mlx/backend/metal/kernels/steel/attn/kernels/sdpa_chunked_reduce.metal` (new): instantiations
- `mlx/backend/metal/scaled_dot_product_attention.cpp`:
  - Add `eval_chunked` path
  - Adapter for new CommandEncoder API (`metal::get_command_encoder(s)` + `compute_encoder.add_temporary(array)`)
  - Per-chunk dispatch with absolute causal offset
- `python/tests/test_sdpa_chunked.py`: 29 tests covering chunk boundaries, causal masking, sinks, multi-chunk correctness

## Test Plan

- [x] `python -m pytest python/tests/test_sdpa_chunked.py` — 27 pass, 2 skipped (pre-existing 32KB threadgroup limit for float32+D=256, not chunking-related)
- [x] `python -m pytest python/tests/test_sdpa_logsumexp.py` — 12/12 pass (regression guard for PR #1)
- [x] `python -m pytest python/tests/test_fast_sdpa.py` — 15 pass, 1 skipped
- [x] **Real-world end-to-end**: DeepSeek-V4-Flash-8bit at 100K context decode on 2-node M4 Max RDMA cluster — no GPU watchdog timeouts (was hard-blocked before this change), 30.7 t/s steady-state

## Notes

The cherry-pick history includes a small adapter commit (`cf729e22`) that updates the dispatch to the new CommandEncoder API (`metal::get_command_encoder(s)` + `compute_encoder.add_temporary(array)`). This was previously `Device::get_command_encoder(idx)` + `Device::add_temporary(array, idx)`, which was removed by your refactor in `5e2c4425` to make `CommandEncoder` thread-local.

Stacked on top of PR #1 (sdpa-logsumexp-output). Happy to combine if preferred.
