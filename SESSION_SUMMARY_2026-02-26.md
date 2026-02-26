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

## 4. Architectural Epiphany: Context Parallelism Abandoned
We briefly investigated implementing Context Parallelism (Ring Attention) as a new sharding strategy to break the 100K memory bandwidth wall. However, we proved mathematically that it is useless for our specific 3-node topology.

**The Math:**
- On two identical Mac Studios (Layers 0-59), CP splits the sequence length 50/50. Tensor Parallelism (TP) splits the attention heads 50/50. 
- In both architectures, the nodes read exactly 50% of the KV cache simultaneously, resulting in identical memory bandwidth pooling.
- Both CP and TP require the exact same network overhead: ~118 network syncs per generated token (`all_sum`).
- At 1-2ms of latency per sync (`FAST_SYNCH=off`), this network overhead completely dominates the compute time, capping the theoretical maximum speed at ~3.5 TPS.

**Conclusion:** Pure Pipeline Parallelism (which requires only 2 network syncs per token instead of 118) is the most efficient configuration for 100K+ context lengths on this cluster while running in Safe Mode, achieving ~7 TPS.

## Next Steps
- Maintain the current Pure Pipeline Parallelism architecture as the baseline for long-context runs.
- The next major frontier for boosting decode speeds is resolving the synchronization hangs that prevent us from running `FAST_SYNCH=on` natively, as that would drop network latency from ~2ms to ~0.05ms per sync.
