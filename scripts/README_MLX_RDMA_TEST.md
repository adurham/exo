# MLX RDMA Test Scripts

These scripts help diagnose MLX RDMA initialization issues by testing RDMA on each node.

## Test Scripts

### 1. `test_mlx_rdma_single_node.sh` - Single Node Test

Tests if MLX can detect RDMA on a single node without requiring coordination.

**Usage:**
```bash
./scripts/test_mlx_rdma_single_node.sh <hostname>
```

**Example:**
```bash
./scripts/test_mlx_rdma_single_node.sh macstudio-m4
```

**What it tests:**
- Checks if `ibv_devices` command is available
- Lists available RDMA devices
- Checks `mx.distributed.is_available()` - this tells us if MLX thinks distributed is available

**Expected output:**
- If `is_available()` returns `True`: MLX can detect RDMA (good sign)
- If `is_available()` returns `False`: MLX cannot detect any distributed backend (problem)

### 2. `test_mlx_rdma.py` - Full RDMA Initialization Test

Tests full MLX RDMA initialization with the same configuration as the main app.

**Usage:**
```bash
uv run python scripts/test_mlx_rdma.py \
    --rank <rank> \
    --world-size <world_size> \
    --coordinator-ip <ip> \
    --coordinator-port <port> \
    --devices-file <path>
```

**Example (rank 0):**
```bash
uv run python scripts/test_mlx_rdma.py \
    --rank 0 \
    --world-size 3 \
    --coordinator-ip 192.168.202.1 \
    --coordinator-port 52414 \
    --devices-file ./test_hosts_0.json
```

**What it tests:**
- Creates devices file with RDMA interface matrix
- Sets all environment variables (MLX_IBV_DEVICES, MLX_RANK, etc.)
- Checks `mx.distributed.is_available()`
- Attempts `mx.distributed.init(backend='any', strict=False)`
- Verifies group size and rank match expectations

**Expected output:**
- If successful: Group size matches world_size, rank matches expected rank
- If singleton group: MLX fell back to non-distributed mode (RDMA not working)
- If fails: MLX cannot initialize any backend

### 3. `test_mlx_rdma_all_nodes.sh` - Multi-Node Test

Tests all nodes simultaneously (required for RDMA to work).

**Usage:**
```bash
./scripts/test_mlx_rdma_all_nodes.sh
```

**What it does:**
- Runs `test_mlx_rdma.py` on all 3 nodes simultaneously
- All nodes initialize at the same time (required for RDMA)
- Reports results from each node

**Configuration:**
Edit the script to change:
- `RANK_0_IP`: IP address of rank 0 coordinator
- `COORDINATOR_PORT`: Port for coordinator
- `WORLD_SIZE`: Number of nodes
- `NODES`: Array of rank:hostname pairs

## Testing Strategy

### Step 1: Test Single Node Detection

First, test if MLX can detect RDMA on each node individually:

```bash
./scripts/test_mlx_rdma_single_node.sh macstudio-m4
./scripts/test_mlx_rdma_single_node.sh macbook-m4
./scripts/test_mlx_rdma_single_node.sh work-macbook-m4
```

**If `is_available()` returns `False` on all nodes:**
- MLX cannot detect RDMA
- Possible causes:
  - RDMA drivers not installed
  - MLX on macOS doesn't support RDMA
  - Network configuration issue

**If `is_available()` returns `True` on all nodes:**
- MLX can detect RDMA (good sign)
- Proceed to Step 2

### Step 2: Test Full Initialization

Test full RDMA initialization on all nodes simultaneously:

```bash
./scripts/test_mlx_rdma_all_nodes.sh
```

**If all nodes get singleton groups:**
- MLX cannot establish distributed communication
- Possible causes:
  - Nodes not initializing simultaneously
  - Coordinator not reachable
  - Network connectivity issues
  - MLX on macOS doesn't support RDMA via MLX_IBV_DEVICES

**If nodes get correct group size:**
- RDMA is working! The issue is in the main application code

## Interpreting Results

### `mx.distributed.is_available() = False`
- MLX cannot detect any distributed backend
- RDMA drivers may not be installed or configured
- MLX on macOS may not support RDMA

### `mx.distributed.is_available() = True` but `init()` fails
- MLX can detect RDMA but cannot initialize it
- Possible causes:
  - Coordinator not reachable
  - All nodes not initializing simultaneously
  - Network connectivity issues

### `init()` succeeds but `group.size() = 1`
- MLX initialized but fell back to singleton (non-distributed) mode
- RDMA is NOT being used
- Possible causes:
  - Nodes not initializing simultaneously
  - Coordinator not reachable
  - MLX on macOS doesn't support RDMA via MLX_IBV_DEVICES

### `init()` succeeds and `group.size() = world_size`
- RDMA is working correctly!
- The issue is in the main application code, not MLX configuration

## Troubleshooting

### If tests fail, check:

1. **RDMA devices are available:**
   ```bash
   ibv_devices
   ```
   Should show devices like `rdma_en2`, `rdma_en3`, etc.

2. **Network connectivity:**
   ```bash
   ping <coordinator_ip>
   ```
   All nodes should be able to reach the coordinator

3. **Thunderbolt interfaces:**
   ```bash
   networksetup -listallhardwareports
   ```
   Verify Thunderbolt interfaces are configured correctly

4. **MLX version:**
   ```bash
   uv run python -c "import mlx.core as mx; print(mx.__version__)"
   ```
   Check if MLX version supports RDMA on macOS

## Notes

- RDMA requires all nodes to initialize `mx.distributed.init()` at roughly the same time
- The coordinator (rank 0) must be reachable from all other nodes
- MLX on macOS may have limited RDMA support compared to Linux
- The test scripts use `strict=False` to see what happens without failing immediately

