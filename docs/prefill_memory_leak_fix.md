# DeltaNet Prefill Memory Leak Fix

*March 21, 2026 — 104K context verified stable*

## The Problem

During prefill on Qwen3.5-397B-A17B-4bit in PP mode, active memory grew at
**540KB/tok** — 6× more than the 90KB/tok expected from KV cache data alone.
This limited cold prefill to ~24K tokens on 128GB before OOM.

## Root Cause

DeltaNet's `GatedDeltaNet.__call__` stores a **slice** of `conv_input` as
the conv1d cache state:

```python
conv_input = mx.concatenate([conv_state, qkv], axis=1)  # (1, T+2, conv_dim)
cache[0] = conv_input[:, -2:]  # small slice — BUT shares parent buffer
```

MLX slices share the parent array's `Data` buffer via `shared_ptr`. This
2-position slice keeps the ENTIRE `conv_input` alive (T+2 positions), which
keeps `qkv` alive → `proj` alive → chunk `inputs` alive. Each prefill chunk's
full computation graph persists via this chain.

## The Fix

### 1. `mx.contiguous` on cache entries in GatedDeltaNet (mlx-lm)
```python
cache[0] = mx.contiguous(conv_input[:, -(self.conv_kernel_size - 1):])
cache[1] = mx.contiguous(state)
```

### 2. Contiguous + eval between prefill chunks (exo generate.py)
```python
mx.eval(*[c.state for c in _prompt_cache])
for _c in _prompt_cache:
    if isinstance(_c, ArraysCache):
        _c.cache = [mx.contiguous(x) if x is not None else x for x in _c.cache]
        mx.eval(*[x for x in _c.cache if x is not None])
mx.clear_cache()
```

## Results

| Metric | Before | After |
|--------|--------|-------|
| Memory growth/tok | 540 KB | ~20 KB |
| Max cold prefill (128GB) | ~24K tokens | ~166K tokens |
| 15K prefill memory | 122.4 GB | 113.7 GB |
| 40K prefill | OOM (segfault) | Success (115.0 GB) |
| 104K multi-turn | Not possible | Success (118 GB, 21-31ms/tok) |

## Investigation Path (13 steps)

1. Measured 540KB/tok growth during prefill — assumed KV cache
2. Pre-allocated KV buffers to avoid concat — no effect
3. Replaced `mx.async_eval` with blocking `mx.eval` — no effect
4. Proved `cache_obj` (measured cache data) was CONSTANT — leak was NOT in cache objects
5. Proved 0.8B model on single node had NO leak — leak was PP-specific (wrong — it was cache-related on all configs)
6. Discovered `mx.depends` chain in PipelineLastLayer — removed it, no effect
7. Re-tested 0.8B model WITH cache: 10.5KB/tok overhead. WITHOUT cache: 0. Leak IS cache-related.
8. Tested pure KVCache concat simulation: 0 leak. Leak is model+cache interaction.
9. Isolated per-layer: **DeltaNet layers leak 2.3KB/tok each. SDPA layers: 0.**
10. Found the shared-buffer slice: `cache[0] = conv_input[:, -2:]`
11. Proved `mx.contiguous` breaks the reference: big array freed after del
12. Applied contiguous to single layer: **0 overhead**
13. Applied to full model in prefill loop: **0 overhead. 104K context succeeds.**

## Key Insight

MLX array slices share the parent's `Data` buffer via `shared_ptr<Data>`.
Even after `mx.eval` and `detach()` clear the computation graph, the `Data`
refcount keeps the parent buffer alive. `mx.contiguous` is the ONLY way to
break this — it allocates a new buffer and copies the data.

**Pattern to watch for:**
```python
# BAD: slice shares parent buffer — keeps entire parent alive
cache[0] = big_array[:, -2:]

# GOOD: contiguous copy — parent can be freed
cache[0] = mx.contiguous(big_array[:, -2:])
```

Any model that stores small slices of large intermediate arrays in its cache
will have this issue. It's not specific to DeltaNet — any recurrent or
convolutional layer with similar caching patterns needs `mx.contiguous`.

## Files Modified

- `mlx-lm/mlx_lm/models/qwen3_5.py` — `mx.contiguous` on `cache[0]` and `cache[1]`
- `src/exo/worker/engines/mlx/generator/generate.py` — contiguous + eval on
  ArraysCache entries between prefill chunks
