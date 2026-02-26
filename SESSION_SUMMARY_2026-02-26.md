# Session Summary: Exo + MLX Long-Context Scaling
**Date:** February 26, 2026
**Objective:** Stress test the optimized 3-node M4 cluster (Safe Mode) to 200,000 tokens and analyze bottlenecks.

---

## 1. 100K+ Context Achievement
We successfully pushed the cluster past 100,000 tokens of context without a single out-of-memory crash or RDMA stutter, proving that our **Architecture-Aware Sharding** and **RDMA Prefill Revolution** from yesterday are rock solid.

**Metrics at 104K Context:**
- **Status:** Stable, continuous generation
- **Throughput:** ~6.8 TPS
- **Primary Bottleneck:** Memory Bandwidth Wall (546 GB/s)

## 2. The Memory Bandwidth Wall
As context size scales linearly, the time required to read the historical KV cache from memory during the decode phase also scales linearly. At 100K+ tokens, the GPU spends nearly all of its time waiting for the unified memory to feed it the massive KV cache, pulling the decode speed down from ~30 TPS to ~6.8 TPS.

## 3. The Flash Attention Trap (Why KV Quantization Failed)
To combat the memory bandwidth wall, we attempted to enable 8-bit KV Cache Quantization (`EXO_KV_BITS=8`) to halve the size of the cache and effectively double the TPS.

**The Result:** Immediate OS-level crash (`ENOMEM -12`).

**The Root Cause:**
- When KV cache quantization is disabled, `mlx-lm` correctly routes all attention calculations through Apple's highly optimized `mx.fast.scaled_dot_product_attention` Metal kernel, which utilizes **Flash Attention**. Flash Attention computes the attention scores in blocks directly on the GPU SRAM, never materializing the massive $N 	imes N$ attention matrix in main memory.
- When `EXO_KV_BITS=8` is set, `mlx-lm` detects the quantized cache and **bypasses the fast hardware kernel**, falling back to a naive Python-based `quantized_scaled_dot_product_attention` implementation.
- Without Flash Attention, the naive implementation attempts to materialize the full $O(N^2)$ attention matrix in RAM during the prefill phase. For a 10,000+ token prefill block, this intermediate matrix balloons to dozens of gigabytes instantly, exhausting the Mac Studio's memory and crashing the system.

**Conclusion:** We cannot use `EXO_KV_BITS` for long-context scaling until MLX supports quantized KV caches natively within its fast Flash Attention kernels.

## Next Steps
- Given the hard physical limit of the 546 GB/s memory bus during Pipeline Parallelism decode, the only architectural path forward to achieve high-speed 200K+ context is implementing **Context Parallelism (Sequence Parallelism / Ring Attention)**.
- Context Parallelism would split the *prompt* across the nodes instead of the *model*, pooling the memory bandwidth of the entire cluster (~1.4 TB/s) to process the attention scores simultaneously. This requires a major architectural rewrite of Exo's mesh and MLX's distributed operations.
