**Repo:** ml-explore/mlx
**Branch:** adurham:pr/metal-allocator-small-coalesce
**Target:** ml-explore/mlx:main
**Commits:** 1 (`804ad66c`)

---

# Title

`perf(metal): coalesce sub-small_size_ allocations to single cache bucket`

# Body

## Summary

In `MetalAllocator::malloc()`, round all sub-`small_size_` (256-byte) requests up to `small_size_` **before** the `buffer_cache_.reuse_from_cache(size)` lookup. This makes every small allocation share a single cache bucket instead of being keyed on its exact byte size — eliminating a cache-miss storm under workloads that allocate many small `mx.array` values per step.

**12-line change**, one block of code, no API change, opt-in behavior preserved.

## The bug

`MetalAllocator::buffer_cache_` is a `multimap<size_t, BufferHolder*>` keyed on **exact byte size**. When a workload allocates lots of small arrays with varying sizes (size-2 bf16 scalars, size-4 fp32 scalars, size-8 indices, ...), each distinct size gets its own cache bucket.

Under async-eval pipelining, the cache for any specific size tends to be **cold at the moment the next step's malloc runs** (the prior step's free for that size hasn't happened yet). So `reuse_from_cache()` returns `nullptr` for most small requests, falling through to:
```cpp
if (size < small_size_ && heap_) {
    buf = heap_->newBuffer(size, resource_options);  // small Metal heap, ~4K slots at 1MB default
}
if (!buf) {
    buf = device_->newBuffer(size, resource_options);  // ← consumes vm_page_size (16 KB) VM region per call
}
```

`device_->newBuffer()` allocates a fresh **16 KB IOAccelerator (graphics) VM region** even for a size=4 request. Because the underlying VM cost is per-region (not per-byte), the requested size only affects which cache bucket the buffer is filed under on free — not the underlying memory consumption.

The result on long-running workloads:
- Cache hit rate stays low for the most-frequent small sizes
- Every cache miss eats one fresh 16 KB VM region
- `vmmap` IOAccelerator region count grows monotonically per decoded token
- `mx.metal.get_active_memory()` stays flat (the leak is outside MLX's tracked active memory)
- Eventually macOS jetsam kills the process

## The fix

Round sub-`small_size_` requests up to `small_size_` **before** the cache lookup:

```cpp
// after upstream's vm_page_size align-up:
} else if (size < small_size_) {
    size = small_size_;
}
```

After warmup, every small request hits the single shared bucket → ~100% cache hit rate → zero fall-through to `device_->newBuffer`.

Cost: wastes `<small_size_` (256) bytes per scalar in MTLBuffer capacity (caller wrote 4 bytes; buffer is 256). Negligible at the scales where this matters.

## Reproducer

Workload: DeepSeek-V4-Flash-6bit decode on 2-node M4 Max RDMA tensor-parallel cluster. Measurement via `MLX_LOG_NEW_BUFFER_PATH=/tmp/log.txt` (an internal-fork debug knob that appends every cache-miss `newBuffer` size to a file) over 2K decoded tokens. Cache-miss histogram **before** this fix:

| Size  | Misses / step | Note |
|-------|---------------|------|
| 4     | 50            | fp32 scalars (offsets, indices, lengths) |
| 2     | 9             | bf16 scalars |
| 8192  | 11            | hidden_dim bf16 activations |
| ...   | ...           | total ~70 / step × 16 KB ≈ 1.1 MB / step |

vmmap IOAccelerator (graphics) region count grew **8,986 → 561,221** over 22K decoded tokens (~+25 regions/token, ~391 KB RSS/token).

After this fix: zero cache-miss allocations during steady-state decode.

## Verified impact

A/B on the same prompt, same workload, 10K decoded tokens, DSv4-Flash-6bit on 2× M4 Max:

| Config | Decode tok/s | RSS rate (KB/tok) |
|--------|--------------|--------------------|
| Upstream baseline (heap=1MB, small_size=256) | 32.1 | ~770 (after heap saturates ~10K) |
| Fork detour A: bump heap to 256MB | 29.3 | 200 |
| Fork detour B: heap=256MB + small_size=16384 | 28.5 | 195 |
| **This PR (heap=1MB + coalesce<256→256)** | **32.1** | **155** |

Result: **matches upstream decode rate** AND **eliminates the device_->newBuffer storm**.

`vmmap` region count at 5-10K decoded tokens:
- Without fix: 14K → 224K IOAccelerator (graphics) regions
- With fix: ~2,400 regions (near startup baseline)

Practical context ceiling on a single deep-think generation:
- Without fix: ~100K tokens (macOS jetsam kill)
- With fix: ~400K tokens (~4× lift)

## Production validation

This patch has been running on a 2× Mac Studio M4 Max RDMA cluster (DSv4 inference, 100K+ context, multi-hour decodes) for several weeks. Steady-state throughput matches upstream; no observed regressions in any workload.

## Why now

We originally worked around this by bumping `heap_size_` from 1 MB to 256 MB (giving us a much larger heap pool to absorb the small allocations before they reached `device_->newBuffer`). That kept us alive but cost ~9% of decode throughput at long context due to wired-memory pressure. The coalesce fix replaces that workaround completely — and is much smaller, cleaner, and zero-cost in throughput terms.

## Test plan

- [x] `python -m pytest python/tests/test_memory.py` — 3/3 pass
- [x] `python -m pytest python/tests/test_array.py python/tests/test_ops.py python/tests/test_random.py python/tests/test_linalg.py python/tests/test_quantized.py` — 660 passed, 9034 subtests passed, 25 skipped, 0 failures
- [x] `pip install -e .` build clean on Apple Silicon M4 Max
- [x] Multi-week production validation on 2-node RDMA cluster (DSv4 long-context decode)

## Notes

- Touches a single file (`mlx/backend/metal/allocator.cpp`): +12 lines, no deletions.
- No new env vars, no behavior change for callers.
- No effect on platforms without a heap allocator (`heap_` may be null on some configurations — the new branch runs before the heap check, but it just adjusts `size` and falls through normally).

## Co-author credit

Fork commit authored by "Exo Bot" (CI commit identity in our cluster repo).
