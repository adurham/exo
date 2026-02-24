# Thunderbolt RDMA Memory Region Limit

**Date**: Feb 23, 2026
**macOS**: 26.3 (Build 25D125)
**Test Tool**: [`rdma_stress.c`](../rdma_stress.c)

---

## Finding

The AppleThunderboltRDMA driver enforces a **hard limit of 100 Memory Regions (MRs)** per device. This was verified across all 3 nodes using a custom stress test.

### Test Results

| Node | RDMA Devices | Active Device | MR Limit | Failure Mode |
|------|-------------|---------------|----------|-------------|
| Mac Studio M4-1 (128GB) | 4 | `rdma_en3` | **100** | Graceful (NULL) |
| Mac Studio M4-2 (128GB) | 4 | `rdma_en3` | **100** | Graceful (NULL) |
| MacBook Pro M4 (36GB) | 3 | `rdma_en1` | **100** | Graceful (NULL) |

Buffer size per MR: 512KB. All registrations used `IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ`.

### Compilation

```bash
cc -o rdma_stress rdma_stress.c -lrdma -Wall -O2
```

Apple ships `libibverbs` headers in the macOS SDK (`/usr/include/infiniband/verbs.h`) and the library as `librdma` (`/usr/lib/librdma.tbd`).

---

## Impact on JACCL

JACCL (the RDMA transport in MLX) allocates MRs as:

```
MRs = BUFFER_SIZES × NUM_BUFFERS × size_ × 2  (send + recv)
```

Where `BUFFER_SIZES=8`, `NUM_BUFFERS=4`, and `size_` = number of TP peers.

| TP Peers | NUM_BUFFERS | MRs Used | Headroom (of 100) |
|----------|-------------|----------|-------------------|
| 2 | 4 | **64** | 36 ✅ |
| 2 | 6 | 96 | 4 ⚠️ |
| 2 | 8 | 128 | **-28 ❌ (segfault on older macOS)** |
| 3 | 4 | 96 | 4 ⚠️ |
| 3 | 5 | 120 | **-20 ❌** |

The current `NUM_BUFFERS=4` with a 2-node TP pair uses 64 MRs — safely within limits. The previous attempt at `NUM_BUFFERS=8` (128 MRs) caused driver segfaults on older macOS and would return NULL on macOS 26.3+.

---

## Why This Is Likely a Software Limit

1. **100 is a decimal round number** — hardware limits are powers of 2 (64, 128, 256)
2. **Traditional RDMA NICs support millions of MRs** — MR tables live in host memory, not on-chip
3. **Failure mode changed between macOS versions** — older macOS segfaulted at 128 MRs; 26.3 returns NULL at 100. Apple refactored this code and added bounds checking
4. **`__builtin_available(macOS 26.3)` in JACCL** — Apple actively develops this RDMA stack; 100 may be a conservative default

## Recommendation

Report to Apple via Feedback Assistant:

> **Title**: Increase per-device Memory Region limit for AppleThunderboltRDMA
>
> **Description**: The AppleThunderboltRDMA driver limits `ibv_reg_mr` to 100 Memory Regions per device. This constrains RDMA pipeline depth for distributed ML workloads (e.g., tensor-parallel inference with MLX). Increasing to 256+ or making it configurable via sysctl would enable higher throughput. Repro: compile and run the attached `rdma_stress.c` with `cc -o rdma_stress rdma_stress.c -lrdma`.

## Related

- [TP All-Reduce Investigation](tp_allreduce_investigation.md) — documents the `NUM_BUFFERS 8→4` reversion due to MR limits
- [GPU Fence Fix](gpu_fence_fix.md) — DSB SY fix enabling `MLX_METAL_FAST_SYNCH=1`
