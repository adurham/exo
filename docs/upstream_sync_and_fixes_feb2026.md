# Exo Cluster Upstream Sync & Optimization Review (Feb 2026)

This document outlines the major synchronization efforts, architecture overhauls, and bug fixes applied to the local Exo cluster to align with the upstream repository while preserving custom Apple Silicon constraints and networking priorities.

## 1. Upstream Cherry-Pick Syncs (Phases 2-5)

We systematically cherry-picked commits from the upstream repository across four major categories to bring the local fork up to date without introducing merge conflicts or overwriting custom networking implementations.

### UI, UX, & Downloads

- Cherry-picked enhancements to the Svelte dashboard, including a better onboarding UX, macOS bug report formatting, ETA visibility on prefill progress, explicit offline mode toggles, and UI contrast fixes.
- Updated the download subsystem to report `downloaded_bytes`, properly detect completed model `.safetensors`, and cancel pending transfers gracefully upon coordinator shutdown.
- Restored the "Hybrid" Pipeline + Tensor Parallelism button to the dashboard and fixed model grouping layouts.

### Stability & Models

- Cherry-picked critical crash preventers, GossipSub `MessageTooLarge` error handling, and forced `spawn` start methods for multiprocess scaling.
- Integrated support for the Ollama API, updated the core MLX fork dependencies, fixed Qwen MoE tensor sharding logic, and added metadata for `Qwen3 Coder Next` variants.
- Optimized Claude prefix-cache hit rates by safely stripping volatile headers.

### Architecture & Systems

- Cherry-picked the integration of Rust into the coordinator event loop (moving routing and messaging to PyO3 bindings) to dramatically improve scaling overhead (`db73c4fd`, `639243aa`).
- Refactored runner implementations into separated process logic and ensured the cluster coordinator is strictly enforced as Rank 0.

### RDMA Networking

- Cherry-picked upstream's `FAST SYNCH` pipeline buffers and Ring-vs-RDMA prioritization logic.
- Safely integrated local `placement_utils.py` topology handling to preserve custom explicit direct-cable priority routing that upstream did not naturally support.

---

## 2. Resolving the Chunked-Prefill TP Deadlock

Following the Architecture sync, upstream introduced a feature called "Chunked Prefill" which utilized eager `mx.eval()` loops. This is beneficial for 16GB edge devices to prevent Out-Of-Memory exceptions, however, it artificially deadlocked Exo's high-performance Tensor Parallelism (TP+PP) by halting Python thread formulation for the entire cluster.

**Resolution:**
We reverted the eager evaluations back to Exo's "Deferred Graph" approach for Apple Silicon high-memory nodes. Network operations are safely queued for a single, simultaneous global evaluation at the end of `generate.py`. We wrapped upstream's feature behind an `EXO_CHUNKED_PREFILL=1` environment variable for users who explicitly require edge-device memory constraints.

---

## 3. Resolving the RDMA TPS Cold Start Regression

The `warmup_inference` functional scope was intentionally clearing the entire MLX Metal cache `mx.metal.clear_cache()` immediately before yielding to the user. This was a legacy precaution meant to workaround the `AppleThunderboltRDMA` driver's hard 100-Memory-Region boundary. However, since the system now utilizes a single massive MR pool allocation, this cache flush was actively destroying the pre-compiled matrices.

As a result, the very first inference following a cluster launch experienced extreme regressions (dropping from 29 TPS down to 22 TPS).

**Resolution:**
We completely removed the obsolete `mx.metal.clear_cache()` block from the `warmup_inference` loop. First-prompt execution across the Thunderbolt bridges now stably hits 29+ TPS immediately.

---

## 4. Fixing the Dashboard Hardware Stats (Macmon)

Following the Upstream Sync, the Svelte dashboard ceased displaying System Performance statistics (GPU Usage, Temp, Power) for the `macbook-m4` node, rendering flat `-` dashes despite Memory metrics populating correctly.

Thorough investigation into the node's background pipe outputs revealed two distinct compounding bugs residing in `src/exo/utils/info_gatherer/info_gatherer.py` and the Homebrew Rust compilation footprint:

1. **AnyIO TextReceiveStream Payload Corruption:**
   Commit `4d71c640` shifted the macmon stdout stream to sequentially await `TextReceiveStream.receive()` raw chunks. Because these chunks were arbitrary buffer reads and not validated newlines, Pydantic's `RawMacmonMetrics.model_validate_json()` was receiving partial strings like `{"temp":...` rather than complete JSON dicts.
   *Fix:* We embedded an explicit Python string-buffer algorithm to accumulate chunk bytes and dynamically `.split("\n")` them, guaranteeing `MacmonMetrics` only intercepts perfectly serialized strings.

2. **Homebrew vs. Cargo Binary Block-Buffering:**
   The MacBook `info_gatherer.py` loop was defaulting to `/opt/homebrew/bin/macmon`, which was natively compiled by Homebrew to utilize infinite block-buffering for the `println!` macro when piped outside of a TTY environment. This caused the AnyIO wrapper to hang silently reading 0 bytes. The Mac Studios avoided this because they compiled via `~/.cargo/bin/macmon`, which flushed cleanly.
   *Fix:* We patched `info_gatherer.py` to explicitly prioritize and execute the user's local `~/.cargo/bin/macmon` binary if it exists before invoking `shutil.which()`. This perfectly aligns the MacBook hardware stream with the Mac Studios, bypassing the Homebrew buffer trap and flawlessly restoring the live UI metrics on the dashboard.
