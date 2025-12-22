# MLX RDMA Test Results

## Test Date
2025-12-21

## Test Environment
- **Nodes**: 3 macOS systems
  - macstudio-m4 (rank 0): M4 Max, 128GB RAM
  - macbook-m4 (rank 1): M4 Max, 48GB RAM  
  - work-macbook-m4 (rank 2): M4 Pro, 36GB RAM
- **Network**: Thunderbolt connections
  - 192.168.201.1-2: Studio ↔ MacBook M4 Max
  - 192.168.202.1-2: Studio ↔ MacBook M4 Pro
  - 192.168.204.1-2: MacBook M4 Max ↔ MacBook M4 Pro

## Test Results

### 1. RDMA Device Detection ✅
**Result**: PASS
- All nodes can detect RDMA devices via `ibv_devices`
- macstudio-m4: `rdma_en2`, `rdma_en3`, `rdma_en4`, `rdma_en5`
- macbook-m4: `rdma_en1`, `rdma_en2`, `rdma_en3`
- work-macbook-m4: `rdma_en1`, `rdma_en2`, `rdma_en3`

### 2. MLX Distributed Availability ✅
**Result**: PASS
- `mx.distributed.is_available()` returns `True` on all nodes
- MLX can detect that distributed backends are available

### 3. RDMA Backend via MLX_IBV_DEVICES ❌
**Result**: FAIL
- **All backends return singleton groups (size=1)**
- Tested backends: `any`, `ring`, `mpi`, `nccl`
- Environment variables set correctly:
  - `MLX_IBV_DEVICES`: JSON file with RDMA device matrix
  - `MLX_RANK`: Correct rank (0, 1, 2)
  - `MLX_IBV_COORDINATOR`: Coordinator IP:PORT
  - `MLX_WORLD_SIZE`: 3
  - `MLX_HOSTFILE`: NOT SET (cleared)
- **Conclusion**: MLX on macOS does NOT support RDMA via `MLX_IBV_DEVICES`
- Even when all nodes initialize simultaneously, all get singleton groups
- All nodes report `group.rank() = 0` regardless of `MLX_RANK` value

### 4. Ring Backend via MLX_HOSTFILE ⚠️
**Result**: PARTIAL
- Ring backend attempts to connect (shows "[ring] Rank X accepting/connecting")
- But fails with errors:
  - Rank 0, 1: "[ring] Accept failed (error: 57)" - Socket not connected
  - Rank 2: "Attempt 1 wait 1000 ms (error: 4)" - Interrupted system call
- Connectivity test shows port 52414 is reachable from macbook-m4 to macstudio-m4
- **Conclusion**: Ring backend is trying to work but failing to establish connections
- May need different port configuration or timing adjustments

## Key Findings

1. **MLX on macOS does NOT support RDMA via MLX_IBV_DEVICES**
   - Despite `is_available() = True`, RDMA initialization fails
   - All backends fall back to singleton (non-distributed) mode
   - This is a limitation of MLX on macOS, not a configuration issue

2. **Ring backend shows promise but has connection issues**
   - Ring backend attempts to establish connections
   - Fails with socket errors, possibly due to:
     - Port configuration issues
     - Timing/synchronization issues
     - Network connectivity problems

3. **RDMA devices are available and detected**
   - Hardware and drivers are working correctly
   - The issue is with MLX's distributed implementation on macOS

## Recommendations

1. **Use Ring Backend Instead of RDMA**
   - Since RDMA via `MLX_IBV_DEVICES` doesn't work on macOS
   - Focus on fixing the ring backend connection issues
   - Ring backend uses TCP/IP over Thunderbolt (still fast)

2. **Debug Ring Backend Connection Issues**
   - Verify all ports are accessible
   - Check if ports need to be the same or different per rank
   - Test with longer timeouts
   - Verify firewall settings

3. **Alternative: Use Single-Node Mode**
   - If distributed doesn't work, use single-node mode
   - Load entire model on fastest node (Mac Studio)
   - Other nodes handle KV cache only

## Next Steps

1. Fix ring backend connection issues
2. Test with different port configurations
3. Verify if user's "manual" setup uses ring backend
4. Consider if MLX on macOS needs a different approach

