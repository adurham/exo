# Pipeline Parallelism Debugging Checklist

## Current Issue
Model produces gibberish output when using pipeline parallelism across multiple nodes.

## Systematic Debugging Steps

### 1. Verify Rank Assignment ✅ (Now Logged)
- **Check**: MLX `group.rank()` must match our `device_rank`
- **Location**: `pipeline_auto_parallel()` now logs this
- **Expected**: `device_rank=0` → `group.rank()=0`, etc.
- **If mismatch**: Pipeline communication will be wrong

### 2. Verify Layer Assignment ✅ (Now Logged)
- **Check**: Each rank processes correct layer range
- **Location**: `pipeline_auto_parallel()` logs layer ranges
- **Expected**: Rank 0: layers [0, N), Rank 1: [N, M), Rank 2: [M, P)
- **If wrong**: Model computation will be incorrect

### 3. Verify Pipeline Communication (Now Logged)
- **Check**: `all_gather` operations and send/recv
- **Location**: `PipelineLastLayer.__call__()` now logs all operations
- **Expected**: 
  - Rank 0 sends to Rank 1
  - Rank 1 receives from Rank 0, sends to Rank 2
  - Rank 2 receives from Rank 1, produces final output
  - All ranks participate in `all_gather` to get final output
- **If wrong**: Data flow will be incorrect

### 4. Next Steps After Reviewing Logs

#### If Rank Mismatch Detected:
- Fix MLX group initialization to match device_rank
- Check `mlx_distributed_init()` in `utils_mlx.py`

#### If Layer Assignment Wrong:
- Check `get_shard_assignments_for_pipeline_parallel()` in `placement_utils.py`
- Verify `sorted_cycle` order matches device_rank assignment

#### If Communication Wrong:
- Verify `all_gather` concatenation order matches rank order
- Check if we need explicit synchronization before `all_gather`
- Consider if `all_gather` is the right operation (maybe need point-to-point broadcast)

#### If All Above Correct:
- Check tokenizer consistency across ranks
- Verify model weights loaded correctly
- Check if there's a shape mismatch in `all_gather` output extraction

## How to Use This

1. Run the model with the new logging
2. Check logs for:
   - "RANK MISMATCH!" errors
   - Pipeline setup information
   - all_gather operation details
   - Send/recv operations
3. Report findings so we can fix the root cause

