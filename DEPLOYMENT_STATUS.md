# Deployment Status Summary

**Date:** December 25, 2024  
**Branch:** `static-4node-master-worker-separation`  
**Model:** Qwen3-235B-A22B (4-bit) - single model only

## ✅ Fixed Issues

### 1. Entry Point Missing
- **Problem:** `ModuleNotFoundError: No module named 'exo.main'`
- **Solution:** Created `src/exo/main.py` that dispatches to `master_app` or `worker_app` based on node type
- **Status:** ✅ Fixed and deployed

### 2. Hostname Case Sensitivity
- **Problem:** Master/worker detection failed due to case mismatch (e.g., "Adams-MacBook-Pro-M1" vs "adams-macbook-pro-m1")
- **Solution:** Added case-insensitive hostname matching in `is_master_node()` and worker detection
- **Status:** ✅ Fixed and deployed

### 3. Worker Node Detection
- **Problem:** Worker node with hostname "MLHVYY0PWXMN" couldn't be matched
- **Solution:** Added Tailscale IP matching fallback when hostname doesn't match
- **Status:** ✅ Fixed and deployed

### 4. Model Cards Restriction
- **Problem:** 22 models available instead of 1 (Qwen3-235B)
- **Solution:** Commented out all models except `qwen3-235b-a22b-4bit` in `src/exo/shared/models/model_cards.py`
- **Status:** ✅ Fixed and deployed (API now shows 1 model)

### 5. Pipeline Parallelism Distribution
- **Problem:** Greedy allocation put all 94 layers on single node instead of distributing across 3 nodes
- **Solution:** Modified `placement_utils.py` to force pipeline parallelism for models >= 10GB, distributing layers proportionally
- **Status:** ✅ Fixed (layers now distributed: 62/19/13 across 3 nodes)

### 6. Multiple Instances Running
- **Problem:** Multiple instances could be created, causing confusion and resource conflicts
- **Solution:** Modified `place_instance` API to delete all existing instances before creating new one, with up to 30s wait for cleanup
- **Status:** ✅ Fixed and deployed

### 7. Deployment Script Branch Reference
- **Problem:** `deploy-all.sh` and `deploy.sh` were hardcoded to `origin/new_main`
- **Solution:** Updated to use `origin/static-4node-master-worker-separation`
- **Status:** ✅ Fixed

## ⚠️ Current Issues / Status

### 1. Inference Timeout
- **Problem:** Chat completion requests timeout (120-180s) without returning results
- **Observed:**
  - API creates streaming queue but receives no chunks
  - Workers are connected (sending `NodeMemoryMeasured` events)
  - Instance exists with distributed layers (62/19/13)
  - **0 runners created** - this appears to be the blocking issue
- **Likely Cause:** Workers not creating runners even though instance exists
- **Status:** 🔴 Investigation needed

### 2. No Runners Created
- **Problem:** State shows 0 runners despite instance existing
- **Expected Flow:**
  1. Instance created → `InstanceCreated` event
  2. Workers receive event → `_create_runner` should create runners
  3. Runners load model → `LoadModel` task
  4. Runners warm up → `StartWarmup` task
  5. Runners become ready → `RunnerReady` status
  6. Chat completions can process
- **Current:** Step 2 appears to be failing (no runners created)
- **Status:** 🔴 Blocking inference

### 3. Backend Testing (RDMA vs TCP/IP)
- **Action:** Temporarily forced TCP/IP (ring) backend for `MlxJaccl` instances
- **Result:** Still times out with TCP/IP, suggesting issue is not RDMA-specific
- **Note:** `MlxRing` instances already use TCP/IP by default
- **Status:** ✅ Confirmed not RDMA-specific issue

## 📊 Current System State

- **Master Node:** Running at `100.67.156.10:52415` (localhost)
- **Worker Nodes:** 3 nodes connected
  - `static-worker-0-adams-mac-studio-m4`
  - `static-worker-1-adams-macbook-pro-m4`
  - `static-worker-2-adams-work-macbook-pro-m4`
- **Instances:** 1 (MlxRingInstance with Pipeline sharding)
- **Runners:** 0 ⚠️
- **Tasks:** 0
- **Models Available:** 1 (qwen3-235b-a22b-4bit)

## 🔍 Next Steps for Investigation

1. **Check Worker Logs** for:
   - `InstanceCreated` events received
   - `_create_runner` execution
   - Model download status (if any)
   - Runner creation errors
   - Download progress logs

2. **Verify Worker State:**
   - Are workers receiving instance creation events?
   - Are workers executing `_create_runner` logic?
   - Are there any errors preventing runner creation?

3. **Add Download Status Logging:**
   - Ensure model download progress is logged (user requested)
   - Log when download starts/completes
   - Log download speed/progress

4. **Check Runner Creation Conditions:**
   - Verify `_create_runner` conditions are met
   - Check if instances are properly distributed to workers
   - Verify node-to-runner mappings

## 📝 Code Changes Summary

### Files Modified:
- `src/exo/main.py` - Created entry point dispatcher
- `src/exo/shared/static_config.py` - Case-insensitive hostname matching
- `src/exo/shared/models/model_cards.py` - Restricted to 1 model
- `src/exo/master/placement_utils.py` - Forced pipeline parallelism distribution
- `src/exo/master/api.py` - Instance deletion before creation
- `src/exo/worker/engines/mlx/utils_mlx.py` - TCP/IP backend fallback (temporary)
- `deploy-all.sh`, `deploy.sh` - Branch reference fixes

### Commits:
- `43306c1` - Fix: Use os.environ instead of sys.environ
- `7b75779` - Fix: Case-insensitive hostname matching for master node
- `60e2b1e` - Force pipeline parallelism for models >= 10GB
- `101932a` - Delete all existing instances before creating new one
- `e7ab209` - Wait for instance deletion to complete
- `0557476` - Temporarily force TCP/IP backend for MlxJaccl

## 🎯 Requirements

- ✅ Only 1 model available (Qwen3-235B)
- ✅ Only 1 instance running at a time
- ✅ Pipeline parallelism with distributed layers
- ❌ **5+ TPS inference performance** (blocked by no runners)
- ❌ Inference working (timeout issue)

