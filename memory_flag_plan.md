# Implementation Plan: --max-node-memory Flag

## Overview
Add a `--max-node-memory` CLI flag that limits the maximum memory a worker node advertises to the coordinator, preventing the coordinator from assigning model shards larger than the specified GB threshold.

## Architecture Understanding

### Current Flow:
1. **CLI**: `Args` class in `src/exo/main.py` parses command-line arguments
2. **Node Discovery**: `InfoGatherer` in `utils/info_gatherer/info_gatherer.py` gathers system info including `MemoryUsage`
3. **Memory Reporting**: Worker sends `NodeGatheredInfo` event with `MemoryUsage` to master
4. **Placement**: `place_instance()` in `master/placement.py` uses `node_memory` to filter cycles and assign shards
5. **Validation**: `_allocate_and_validate_layers()` in `placement_utils.py` validates per-node memory requirements

### Key Files:
- `src/exo/main.py` - CLI argument parsing (`Args` class)
- `src/exo/worker/main.py` - Worker initialization
- `src/exo/utils/info_gatherer/info_gatherer.py` - Memory gathering (`_monitor_memory_usage`)
- `src/exo/shared/apply.py` - Applies `NodeGatheredInfo` events to state
- `src/exo/shared/types/state.py` - State with `node_memory` mapping
- `src/exo/master/placement.py` - Placement logic using node memory
- `src/exo/master/placement_utils.py` - Shard assignment with memory validation

## Implementation Steps

### Step 1: Add CLI Flag
**File**: `src/exo/main.py`
- Add `max_node_memory_gb: int | None = None` to `Args` class
- Add argparse argument `--max-node-memory` that accepts an integer (GB)

### Step 2: Pass Flag to Worker
**File**: `src/exo/main.py`
- Pass `max_node_memory_gb` to `Worker` constructor
- Store in a location accessible to memory reporting

### Step 3: Modify Worker to Cap Memory
**File**: `src/exo/worker/main.py`
- Accept `max_node_memory_gb` parameter in `Worker.__init__`
- Store as instance variable

### Step 4: Cap Memory in Info Gatherer
**File**: `src/exo/utils/info_gatherer/info_gatherer.py`
- Modify `InfoGatherer` to accept `max_node_memory_gb` parameter
- In `_monitor_memory_usage()`, cap the reported memory to the max value

### Step 5: Propagate Flag Through Creation Chain
**File**: `src/exo/main.py`
- Pass `max_node_memory_gb` from `Node.create()` to `InfoGatherer`

### Step 6: Enforce in Placement Logic (Defense in Depth)
**File**: `src/exo/master/placement_utils.py`
- In `_allocate_and_validate_layers()`, add check that each node's shard doesn't exceed its max memory limit
- This ensures coordinator rejects placements that would exceed per-node limits

## Testing Strategy
- Create unit test in `src/exo/master/test_placement_max_memory.py` (or existing test file)
- Test that coordinator respects memory limit when distributing mock model
- Test with various memory configurations (limit higher/lower than model size)
- Verify error is raised when no valid placement exists