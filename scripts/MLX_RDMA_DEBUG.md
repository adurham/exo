# MLX RDMA Debugging - Current Status

## Environment
- **macOS**: 26.3 (Tahoe)
- **MLX**: 0.29.3
- **Hardware**: M4 Max/M4 Pro with Thunderbolt 5
- **RDMA Devices**: Detected and available on all nodes

## Problem
MLX distributed initialization returns singleton groups (size=1) even when:
- All environment variables are set correctly
- RDMA devices are available
- `mx.distributed.is_available()` returns `True`
- All nodes initialize simultaneously

## What We've Tested

### ✅ Working
- RDMA device detection (`ibv_devices` shows devices)
- `mx.distributed.is_available()` returns `True`
- Environment variables are set correctly
- Devices file is created and readable

### ❌ Not Working
- `mx.distributed.init()` with `MLX_IBV_DEVICES` returns singleton groups
- All backends (`any`, `ring`, `mpi`, `nccl`) return size=1
- Even with simultaneous initialization across all nodes

## Current Configuration

```python
os.environ["MLX_IBV_DEVICES"] = "./hosts_{rank}.json"  # Relative path
os.environ["MLX_RANK"] = str(rank)
os.environ["MLX_IBV_COORDINATOR"] = "{ip}:{port}"
os.environ["MLX_WORLD_SIZE"] = str(world_size)
# MLX_HOSTFILE is NOT set

group = mx.distributed.init(backend="any", strict=True)
# Result: group.size() = 1 (singleton)
```

## Questions for Manual Setup

Since RDMA works manually, please share:

1. **What command/script do you run manually?**
   - Is it a Python script?
   - What's the exact sequence?

2. **What environment variables do you set?**
   - Are they the same as above?
   - Any additional variables?

3. **How do you initialize MLX?**
   - Same `mx.distributed.init()` call?
   - Different backend or parameters?

4. **Do all nodes initialize at the same time?**
   - How do you coordinate this?

5. **What's different from our automated setup?**
   - Different working directory?
   - Different process context?
   - Different user permissions?

## Possible Issues

1. **MLX version**: Maybe need a newer version or build from source?
2. **Initialization timing**: Maybe need specific coordination?
3. **Process context**: Maybe needs to be in a specific process/thread?
4. **macOS RDMA enablement**: Maybe needs additional system configuration?
5. **MLX build flags**: Maybe MLX needs to be built with specific flags for macOS RDMA?

## Next Steps

1. Get details on manual setup
2. Compare with automated setup
3. Identify missing configuration
4. Fix automated initialization

