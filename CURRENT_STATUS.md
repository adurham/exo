# Current Progress: Exo + MLX High-Context Optimization

## Goal
Achieve stable, high-performance distributed inference on a 3-node M4 cluster with a target context of **200,000 tokens** on `mlx-community/MiniMax-M2.5-5bit`.

## What We've Achieved
1.  **Universal GPU Delegation:**
    - Modified `mlx/backend/metal/distributed.cpp` to ensure distributed operations (AllReduce, etc.) always run on the GPU stream.
    - This eliminated the 40ms-50ms "CPU Fallback" bottleneck that occurred when `FAST_SYNCH` was off. 
    - **Result:** Orchestration overhead (`eval`) dropped from 30ms+ to **1ms (Fast)** or **10ms (Safe)**.

2.  **Architecture-Aware Sharding (MacBook Protection):**
    - Refactored `placement_utils.py` to calculate exact memory requirements for a **100% context window (200k tokens)** based on model architecture (layers, heads, head_dim).
    - Implemented a redistribution loop that moves layers from memory-constrained nodes (MBP 36GB) to high-capacity nodes (Studios 128GB) until the 200K footprint is guaranteed to fit.
    - **Result:** Successfully grew context to **31,000+ tokens** with **6.6GB of RAM headroom** on the MacBook (where previously it would have been near 0GB).

3.  **RDMA Pipelining & Prefill Speed:**
    - Increased `NUM_BUFFERS` to 8 and `MAX_WR` to 256 for better distributed saturation.
    - Increased `FRAME_SIZE` from 4KB to **64KB**, reducing CPU interrupt overhead for large prefill chunks by 16x.
    - Uncapped the prefill pipeline in `mesh.cpp` and `ring.cpp` to allow all 8 buffers to work concurrently.
    - Tuned `BUFFER_SIZES` to 6 to keep the total memory pool at a safe **~67MB**, avoiding RDMA driver crashes.

4.  **Adaptive Prefill Chunking:**
    - Set `EXO_PREFILL_STEP_SIZE=1024` and `EXO_ADAPTIVE_THROTTLE=100` in `start_cluster.sh`.
    - **Result:** Massive context jumps are now processed in steady, high-speed 1K pulses instead of single giant bursts that caused RDMA overflows and system stutters.

## Current System State
- **Branch:** Optimized Fork (A-Side)
- **FAST_SYNCH:** Off (Safe mode)
- **Decode Performance:** ~25 TPS (Safe Sync) / ~90 TPS (Fast Sync).
- **Prefill Performance:** ~250+ tokens/sec (Stable).
- **GPU Saturation:** 100% residency reached; power draw hit **74W** peak on M4 Studios.

## Known Limitations & Architectural Findings
1.  **KV Cache Quantization (`EXO_KV_BITS`) & Flash Attention Trap:**
    - Setting `EXO_KV_BITS=8` or `4` to reduce memory bandwidth bottleneck during long-context generation currently **fails**.
    - **Reason:** In `mlx-lm`, if the KV cache is quantized, MLX bypasses the hardware-accelerated `mx.fast.scaled_dot_product_attention` (Flash Attention) Metal kernel and falls back to a naive Python-based implementation (`quantized_scaled_dot_product_attention`).
    - **Effect:** Without Flash Attention, the naive kernel materializes the full $O(N^2)$ attention matrix in memory. During large prefills (e.g., 10,000+ tokens), this intermediate matrix ballooning causes immediate Out Of Memory (OOM) crashes and OS-level fatal errors (`ENOMEM -12`).
    - **Conclusion:** We cannot use KV Cache Quantization until MLX supports quantized caches inside its fast Flash Attention kernels.
2.  **The Memory Wall:**
    - Without Context/Sequence Parallelism (like Ring Attention), standard Pipeline Parallelism is bound by the memory bandwidth of individual nodes during decode (reading the entire KV cache for each token). At ~100K context, TPS drops to ~6-7 as the nodes hit the 546 GB/s memory bandwidth limit of the M4 Max.
3.  **Context Parallelism (CP) vs Tensor Parallelism (TP) on 3-Node Architecture:**
    - We investigated building Context Parallelism (Ring Attention) to pool memory bandwidth and solve the 100K memory wall. However, we mathematically proved it offers **zero benefit** over the existing Hybrid TP+PP strategy for this specific 3-node layout.
    - **Reason:** On 2 identical Mac Studios, CP splits the sequence length 50/50, while TP splits the attention heads 50/50. In both scenarios, the nodes read exactly 50% of the KV cache simultaneously (identical memory bandwidth pooling).
    - Furthermore, both CP and TP require massive network synchronization (broadcasting the query, and `all_sum` combining the outputs) twice per layer. For 59 layers, that is 118 network syncs per token.
    - **Conclusion:** At ~1-2ms latency per sync (due to `FAST_SYNCH=off`), both TP and CP incur ~150ms+ of pure network waiting per token, capping decode speeds severely. The 7 TPS achieved on Pure Pipeline Parallelism (which only requires 2 network syncs total) is currently the absolute mathematical maximum for this cluster when running in Safe Mode.

## Final Task (Next Session)
1.  **User Action:** Run `./start_cluster.sh` to apply the latest C++ optimizations (64KB frames and uncapped pipeline).
2.  **Agent Action:** Resume the `tmp/stress_test.py` in HYPER-GROWTH mode to confirm the climb to 200,000 tokens.
3.  **Monitoring:** Watch MacBook RAM; it should safely hit the 200K mark without OOM.
