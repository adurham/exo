# Current Progress: Exo + MLX Inference Optimization

## Goal
Optimize distributed LLM inference on the 3-node M4 cluster (macstudio-m4-1, macstudio-m4-2, macbook-m4) running `mlx-community/MiniMax-M2.5-5bit`. We are currently compute-starved (50W / 50% utilization) and experiencing significant orchestration overhead during decode steps.

## What We've Discovered
1.  **Adaptive Throttle Was Lost:** The adaptive throttle (`MAX_ACTIVE_TASKS`) was introduced in a previous commit but its environment variables were lost in `start_cluster.sh`. We restored it using a single `EXO_ADAPTIVE_THROTTLE=1` variable that dynamically switches `MAX_ACTIVE_TASKS` between 30 (for prefill) and 100 (for decode). This successfully eliminated OOMs during massive prefill chunks while allowing decode to run faster.
2.  **GPU Command Buffer Timeouts:** Earlier crashes (`kIOGPUCommandBufferCallbackErrorTimeout`) were caused by the GPU spinning forever on stale memory during RDMA transfers. 
3.  **FAST_SYNCH Fix:** The user implemented a critical fix in the MLX submodule (`3c19d389`) adding a `DSB SY` barrier and atomic fallback to the Metal kernel, which solves the stale memory issue and makes `MLX_METAL_FAST_SYNCH=1` stable.
4.  **The 29ms Bottleneck:** Currently, a decode step takes 36ms total, but the actual GPU execution (`model=7ms`) and evaluation (`gpu_eval=0.5ms`) are tiny. The vast majority of the time (`eval=29ms`) is lost to orchestration overhead. This is because the MLX scheduler is forcing a full pipeline stall to perform RDMA transfers on the CPU stream.

## What We Just Did
To eliminate the 29ms overhead, we need the GPU to orchestrate the RDMA transfers without breaking the GPU stream. 
1.  **Conditionally Re-enabled `eval_gpu` Delegation:** In the `mlx` submodule (`mlx/backend/metal/distributed.cpp`), we updated `AllReduce`, `AllGather`, `Send`, `Recv`, and `ReduceScatter` to conditionally call `eval_cpu(inputs, outputs)` **only if** `metal_fast_synch()` is true.
2.  **Safety Fallback:** If `FAST_SYNCH` is off, it throws an error. This forces the MLX scheduler to fall back to the slower but safer `MTLSharedEvent` synchronization on the CPU stream, preventing the 60s timeout race condition.
3.  **Enabled FAST_SYNCH:** We updated `start_cluster.sh` to set `EXO_FAST_SYNCH=on` as the default.

All changes have been committed and pushed to both the `mlx` submodule and the `exo` repository.

## Next Steps (Upon Restart)
1.  **User Action:** The user needs to run `./start_cluster.sh` to pull the latest commits, rebuild the MLX submodule, and start the cluster.
2.  **User Action:** The user needs to create the inference instance.
3.  **Agent Action:** The agent needs to restart the continuous `tmp/stress_test.py` script.
4.  **Agent Action:** Monitor the remote logs (`tail -f /tmp/exo.log`) on `macstudio-m4-1` and `macstudio-m4-2`. We are looking for the `eval=` timing in the `[STEP X]` logs to drop significantly from the previous ~29ms overhead, confirming that GPU delegation is successfully keeping the pipeline saturated.
