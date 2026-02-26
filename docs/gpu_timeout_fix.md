# Metal GPU Timeout Fix (Dynamic Safe Sync)

## The Problem: `kIOGPUCommandBufferCallbackErrorTimeout`
When running massive contexts (100,000+ tokens) on macOS clusters, the system would abruptly crash during the first **Decode** step with the following error:
```
libc++abi: terminating due to uncaught exception of type std::runtime_error: [METAL] Command buffer execution failed: Caused GPU Timeout Error (00000002:kIOGPUCommandBufferCallbackErrorTimeout)
Fatal Python error: Aborted
```

### Why it happens:
1. **The Fast Path:** To achieve blazing 1ms orchestration times across the cluster without triggering kernel panics (`EXO_FAST_SYNCH=off`), we previously modified MLX to execute CPU network primitives *inside* the Metal GPU command buffer callback queue.
2. **The Pipeline Wait:** Because of this, the GPU of the Pipeline Tail node (Node 1) never yields to the CPU. It simply sits in a blocked state waiting for the heavy-lifting node (Node 2) to finish computing its 59 layers and send the token packet.
3. **The OS Watchdog:** As context size grows linearly, the time it takes Node 2's GPU to read the massive, unquantized KV cache for 59 layers eventually exceeds 2 to 5 seconds.
4. **The Crash:** macOS has a strict hardware/OS watchdog that assumes any GPU command buffer taking longer than ~2 seconds is "hung." When Node 1's GPU sits idle waiting for the network for >2s, macOS forcefully kills the entire Python process.

## The Solution: Dynamic Safe Sync
To fix this, we implemented a dual-layered approach that keeps the blazing fast 1ms latency for 90% of workloads, but provides a rock-solid safety net for extreme hyper-growth contexts.

### 1. OS-Level Mitigation
We added environment variables to `start_cluster.sh` to request the Metal driver to disable the command buffer timeout watchdog:
```bash
# Metal GPU Timeout mitigations
EXO_ENV="$EXO_ENV MTL_DISABLE_TIMEOUT=1 MTL_COMMAND_BUFFER_TIMEOUT=0"
```

### 2. The `MLX_FORCE_DISTRIBUTED_GPU` Escape Hatch (C++)
We modified `mlx/backend/metal/distributed.cpp` to introduce a dynamic environment variable `MLX_FORCE_DISTRIBUTED_GPU`. 
* When set to `1` (default), network ops run inside the GPU command buffer (1ms overhead).
* When set to `0`, MLX throws an internal error, forcing the primitive to fall back to standard CPU execution. This frees the GPU entirely so it cannot timeout while waiting.

### 3. Context-Aware Auto-Adjustment (Python)
In `src/exo/worker/engines/mlx/generator/generate.py`, we added a check directly into the `_step` generation loop:
```python
# Check current context size
_kv_len = prompt_cache[0].offset if prompt_cache else 0

# If context > 50,000 tokens, Node 2 will take >2 seconds to compute.
# We must move the network wait off the GPU to avoid triggering the Apple watchdog.
_massive_context = _kv_len > 50000

if _massive_context:
    os.environ["MLX_FORCE_DISTRIBUTED_GPU"] = "0" # Drop to CPU wait

try:
    sampled = mx.distributed.all_sum(contribution, group=hybrid_group)
    mx.eval(sampled)
finally:
    if _massive_context:
        os.environ["MLX_FORCE_DISTRIBUTED_GPU"] = "1" # Restore fast GPU path
```

**Result:** The cluster smoothly scales to 200,000 tokens. Short prompts sync in ~1ms. Massive prompts gracefully fall back to CPU syncing (adding ~20ms overhead) to ensure the system survives indefinitely.
