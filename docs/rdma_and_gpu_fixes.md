# RDMA & GPU Fixes Reference

This document consolidates the cluster-level fixes and investigations for Thunderbolt RDMA and Metal GPU issues, ordered chronologically.

---

## 1. TP All-Reduce Investigation (Feb 22-23, 2026)

**Result:** All optimization paths exhausted — prefill is hardware-bound by Thunderbolt RDMA latency.

### Problem

During Hybrid TP+PP inference (3x M4 Macs, MiniMax-M2.5-5bit, 62 decoder layers), prefill of a 16K-token prompt takes ~50 seconds. Custom profiling (`EXO_EVAL_DEBUG`) showed 99.4% of time in `sched_wait` — GPU idle while CPU-dispatched RDMA all-reduce operations complete over Thunderbolt.

### Architecture Per Layer

Each decoder layer requires 2 large all-reduces (~95MB each over RDMA) plus 2 scalar all-reduces:
- After `o_proj` (ShardedToAll) — ~95MB
- After MoE block — ~95MB

Total: 2 large x 62 layers = **124 RDMA round-trips** per prefill, ~400ms each.

### Optimization Attempts (all failed)

1. **Eliminate o_proj all-reduce** — Mathematically impossible. RMSNorm between the two all-reduces is non-linear; partial sums produce wrong normalization.

2. **Chunked prefill** (~12 commits, reverted) — Smaller chunks = more round-trips with overhead. Made things worse.

3. **Increase RDMA buffer size** (~5 commits) — Thunderbolt driver hard limits: max ~8MB buffer, 32MB segfaults, too many MRs also segfaults.

4. **GPU-stream distributed ops** (deadlock) — RDMA requires CPU-side `ibv_post_send`/`ibv_post_recv` calls, can't run on GPU.

5. **All-gather + local GPU reduce** (deployed, no improvement) — CPU-side reduction was not the bottleneck; RDMA transfer time is.

6. **Async/overlapped all-reduce** (proven infeasible) — Each all-reduce result is a direct data dependency for the next operation. Zero overlap possible.

7. **Float8 compression** (ruled out) — Model already 5-bit quantized; additional lossy compression unacceptable.

### Root Cause

The 50-second prefill is the fundamental cost of TP across Thunderbolt RDMA:
- 124 sequential blocking RDMA ops (can't reduce — non-linear RMSNorm)
- Can't overlap with GPU compute (serial data dependency chain)
- Can't speed up transfers (driver buffer limits)

### What Would Help

| Change | Feasibility |
|--------|-------------|
| Apple increases RDMA buffer limits | Out of our control |
| Thunderbolt 5 with faster RDMA | Hardware upgrade |
| Pure Pipeline Parallelism (no TP) | Already works, lower TPS |

---

## 2. GPU Fence Fix — DSB SY for FAST_SYNCH (Feb 23, 2026)

**Commits:** `3c19d389` (mlx), `78196d4c` / `94e33521` (exo)
**Status:** Verified stable, production-ready.

### Problem

`MLX_METAL_FAST_SYNCH=1` causes non-deterministic GPU deadlocks. The `fence_wait` Metal kernel spins forever because the GPU never sees the CPU's write to the shared fence counter.

**Root cause:** ARM64 `std::memory_order_seq_cst` compiles to `DMB ISH` (Inner Shareable domain). GPU and DMA engines operate in the Full System domain and may never see the write.

### Fix

**CPU-side** (`mlx/backend/metal/fence.cpp`): Added `__builtin_arm_dsb(0xF)` (DSB SY — Full System barrier) after the fence counter store in `Fence::update()`.

**GPU-side** (`mlx/backend/metal/kernels/fence.metal`): Two-tier wait:
1. Fast path (~1M iterations): volatile reads + `atomic_thread_fence` with `thread_scope_system`
2. Fallback: `atomic_load_explicit` on reinterpreted `device atomic_uint*` with `memory_order_relaxed`

### Verification

Binary disassembly confirms `dsb sy` immediately after `stlr`:
```
0000000000002af0   stlr    w19, [x0]
0000000000002af4   dsb     sy
```

370,000+ successful fence synchronizations across stress tests. Zero deadlocks.

### Related PRs

- [mlx#3142](https://github.com/ml-explore/mlx/issues/3142) — Bug report
- [mlx#3141](https://github.com/ml-explore/mlx/pull/3141) — Proposed upstream fix
- [mlx#3144](https://github.com/ml-explore/mlx/pull/3144) — Merged cross-command-buffer fence fix

---

## 3. Thunderbolt RDMA Memory Region Limit (Feb 23-24, 2026)

**macOS:** 26.3 (Build 25D125)

### Finding

AppleThunderboltRDMA driver enforces a **hard limit of 100 Memory Regions (MRs)** per device. Verified across all 3 nodes with custom stress test (`rdma_stress.c`).

### Impact on JACCL

JACCL allocates: `MRs = BUFFER_SIZES(8) x NUM_BUFFERS x TP_peers x 2 (send+recv)`

| TP Peers | NUM_BUFFERS | MRs Used | Status |
|----------|-------------|----------|--------|
| 2 | 4 | 64 | Safe |
| 2 | 8 | 128 | Segfault |
| 3 | 4 | 96 | Tight |

### MR Pooling Bypass

The 100-MR limit can be bypassed by registering a single large memory pool and subdividing in software. Verified: single MRs up to 2GB, 256 logical 512KB buffers from 1 hardware MR slot.

### Why Likely a Software Limit

- 100 is decimal (hardware limits are powers of 2)
- Traditional RDMA NICs support millions of MRs
- Failure mode changed between macOS versions (segfault → NULL)

---

## 4. Metal GPU Timeout Fix — Dynamic Safe Sync (Feb 2026)

### Problem: `kIOGPUCommandBufferCallbackErrorTimeout`

At 100K+ tokens, macOS kills the process because GPU command buffers exceed the ~2s watchdog while the pipeline tail node waits for the heavy node to finish computing.

### Solution: Three Layers

**1. OS-level:** Environment variables to disable Metal command buffer timeout watchdog:
```bash
MTL_DISABLE_TIMEOUT=1 MTL_COMMAND_BUFFER_TIMEOUT=0 EXO_DISABLE_METAL_TIMEOUT=1
```

**2. MLX escape hatch:** `MLX_FORCE_DISTRIBUTED_GPU` env var. When `0`, forces fallback to CPU execution, freeing the GPU from timeout risk.

**3. Context-aware auto-adjustment:** In `generate.py`, if Metal timeout is not disabled, automatically switch to CPU sync above `EXO_SAFE_SYNC_LIMIT` (default 50K tokens).

### Unified Graph Fusion

Fused the two separate `mx.eval()` calls per token (model compute + token sync) into a single graph evaluation, saving ~1-2ms driver overhead per token.
