# Analysis: Minimal Static MLX RDMA Setup

## Your Requirements
- 3 static nodes (macstudio-m4, macbook-m4, work-macbook-m4)
- Static IPs over RDMA-enabled Thunderbolt
- Static model: Qwen3
- Static devices
- **ONLY RDMA-based MLX** - everything else is unneeded

## What MLX RDMA Actually Needs

### Core MLX Requirements
1. **IBV Devices Matrix** (3x3): `matrix[i][j]` = RDMA interface name on node i that connects to node j
   - Example: `[["rdma_en2", "rdma_en3", None], ["rdma_en4", None, "rdma_en5"], ...]`
   
2. **Coordinator Addresses**: IP:PORT for each node to connect to rank 0
   - Format: `{"node_id": "192.168.x.x:12345", ...}`
   
3. **Rank Assignment**: Each node needs rank (0, 1, or 2) and world_size (3)

4. **Environment Variables**:
   ```
   MLX_IBV_DEVICES=./devices.json  # Path to JSON matrix
   MLX_RANK=0                      # This node's rank
   MLX_WORLD_SIZE=3                # Total nodes
   MLX_JACCL_COORDINATOR=192.168.x.x:12345  # Coordinator address
   ```

5. **Model Sharding**: How Qwen3 layers are split across 3 nodes

### Current Codebase Complexity

**Infrastructure You DON'T Need:**
- ❌ Dynamic topology discovery (libp2p/router)
- ❌ Master election system
- ❌ Dynamic placement algorithms
- ❌ Multiple instance types (MlxRing vs MlxJaccl)
- ❌ Dashboard/web UI
- ❌ API server
- ❌ Complex event system
- ❌ Multi-model support
- ❌ Download system (if model is pre-downloaded)

**What You DO Need (minimal):**
- ✅ MLX distributed init code (`mlx_distributed_init`)
- ✅ Model loading/sharding (`shard_and_load`)
- ✅ Inference generation (`mlx_generate`)
- ✅ Static configuration for your 3 nodes

## Recommendation

### Option 1: Build Minimal Script (RECOMMENDED)
Create a simple ~200-300 line Python script that:
1. Hardcodes your 3-node configuration (IPs, RDMA interfaces, coordinator)
2. Sets MLX environment variables directly
3. Loads Qwen3 model with MLX
4. Runs distributed inference

**Pros:**
- Simple, easy to understand
- No unnecessary complexity
- Fast iteration
- Easy to debug

**Cons:**
- Lose any useful abstractions from current codebase
- Need to reimplement model sharding logic

### Option 2: Strip Down Current Fork
Remove/uncomment:
- Router/libp2p networking
- Election system
- Dynamic placement
- Dashboard/API (or keep minimal API)
- Multi-model support

**Pros:**
- Keep working model sharding/loading code
- Keep proven MLX integration

**Cons:**
- Still complex
- Hard to maintain
- Lot of dead code

## Essential Code Paths (if keeping fork)

The minimal code you'd need:

1. **MLX RDMA Init**: `src/exo/worker/engines/mlx/utils_mlx.py::mlx_distributed_init()`
   - Lines 131-320 handle MlxJacclInstance
   - Sets environment variables correctly

2. **Model Loading**: `src/exo/worker/engines/mlx/utils_mlx.py::initialize_mlx()`
   - Lines 323-366

3. **Inference**: `src/exo/worker/engines/mlx/generator/generate.py::mlx_generate()`

4. **Model Sharding**: `src/exo/worker/engines/mlx/auto_parallel.py`
   - Handles splitting model across nodes

## Your Static Configuration

Based on your hardcoded topology code, you likely need:

```python
# Static config for 3 nodes
STATIC_CONFIG = {
    "nodes": {
        "macstudio-m4": {
            "rank": 0,
            "ip": "192.168.202.1",  # Coordinator IP
            "coordinator_port": 12345,
            "rdma_interfaces": {
                "macbook-m4": "rdma_en2",      # Interface to macbook
                "work-macbook-m4": "rdma_en3"  # Interface to work-macbook
            }
        },
        "macbook-m4": {
            "rank": 1,
            "rdma_interfaces": {
                "macstudio-m4": "rdma_en4",
                "work-macbook-m4": "rdma_en5"
            }
        },
        "work-macbook-m4": {
            "rank": 2,
            "rdma_interfaces": {
                "macstudio-m4": "rdma_en6",
                "macbook-m4": "rdma_en7"
            }
        }
    },
    "model": "mlx-community/Qwen3-235B-A22B-Instruct-2507-4bit",
    "world_size": 3
}
```

## Next Steps

I can help you either:
1. **Create a minimal standalone script** (~300 lines) that does exactly what you need
2. **Strip down the current codebase** to remove unnecessary complexity
3. **Document the minimal code paths** so you can extract them yourself

What would you prefer?

