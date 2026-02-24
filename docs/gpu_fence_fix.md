# MLX Fast Synch GPU Fence Fix

**Date**: Feb 23, 2026
**Commits**: `3c19d389` (mlx), `78196d4c` / `94e33521` (exo)
**Status**: Verified stable — production-ready

---

## Problem

Enabling `MLX_METAL_FAST_SYNCH=1` causes non-deterministic GPU deadlocks. The `fence_wait` Metal kernel spins forever because the GPU never sees the CPU's write to the shared fence counter.

**Root Cause**: ARM64 `std::memory_order_seq_cst` compiles to `DMB ISH` (Data Memory Barrier, Inner Shareable), which only ensures ordering between CPU cores. The GPU and DMA engines operate in the Full System domain and may never see the write.

**Environment**: 3-node M4 Mac cluster (2× Mac Studio 128GB + MacBook Pro 36GB), Exo for distributed inference, JACCL (Thunderbolt RDMA).

---

## Fix

### CPU-side (`mlx/mlx/backend/metal/fence.cpp`)

Added `__builtin_arm_dsb(0xF)` (DSB SY — Data Synchronization Barrier, Full System) after the fence counter store in `Fence::update()`. This ensures the write reaches the point of coherence visible to GPU.

### GPU-side (`mlx/mlx/backend/metal/kernels/fence.metal`)

Two-tier wait strategy in `fence_wait`:

1. **Fast path** (~1M iterations): existing volatile reads + `atomic_thread_fence` with `thread_scope_system` — zero overhead when GPU cache is coherent
2. **Fallback**: `atomic_load_explicit` on a `device atomic_uint*` reinterpretation with `memory_order_relaxed`, forcing the GPU to re-fetch from the system-level coherence point

---

## Verification

### Build

Binary disassembly confirms `dsb sy` immediately after `stlr` (store-release):

```
0000000000002af0   stlr    w19, [x0]
0000000000002af4   dsb     sy
```

### Stress Test — TP-only (2× Mac Studio)

| Context | fence_wait | Decode Speed | Total Steps |
|---------|-----------|-------------|-------------|
| 26-71K | 0.3–1.0ms | 95-215ms/step | Thousands |

**372 fence transitions per step × thousands of steps = 370,000+ successful synchronizations. Zero deadlocks.**

### Stress Test — 3-node Hybrid TP+PP (all nodes, fresh reboot)

| Context | Cache Hit | Decode Speed | Status |
|---------|-----------|-------------|--------|
| 16K (cold) | 0% | 33ms/step (30 tok/s) | ✅ |
| 50K | 78% | 46ms/step (22 tok/s) | ✅ |
| 99K | 98% | 64ms/step (16 tok/s) | ✅ |
| **104K** | **99%** | **67ms/step (15 tok/s)** | ✅ |

- **5,795 total decode steps** over 16 minutes
- **2,127-step sustained generation** at 104K context
- **Zero deadlocks, zero SIGABRT, zero errors**

### Hardware at Peak (104K context)

| Node | GPU Power | Free RAM |
|------|-----------|----------|
| Mac Studio M4-1 (128GB) | 31W | ~5.8GB |
| Mac Studio M4-2 (128GB) | 56W | ~5.4GB |
| MacBook Pro M4 (36GB) | 0W (idle between PP stages) | ~4.2GB |

---

## Related PRs

- [mlx#3142](https://github.com/ml-explore/mlx/issues/3142) — Original bug report
- [mlx#3141](https://github.com/ml-explore/mlx/pull/3141) — Proposed upstream fix (DSB SY + atomic load)
- [mlx#3144](https://github.com/ml-explore/mlx/pull/3144) — Merged fix for cross-command-buffer fence (already in our submodule)
