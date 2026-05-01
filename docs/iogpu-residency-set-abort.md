# IOGPU `MTLResidencySet::addAllocation` silent process abort on long sustained workloads

## Status

- **Reproduced**: Yes, deterministically on Apple M4 Max 128GB, macOS 26.4 (25E241)
- **Fork patch**: `adurham/mlx@f8bac642` — bypass `wired_set_->addAllocation` in `mlx/backend/metal/resident.cpp::ResidencySet::insert`
- **Filed upstream**: TBD (Apple Feedback + ml-explore/mlx issue)
- **Verified fix**: AIME-2025 Q1 (DSv4-Flash-8bit, 7,825-token decode, 2-node TP) goes from "always silent SIGABRT at ~12 min" → "completes clean with `[stop]` and correct answer"

## Symptom

Process dies with SIGABRT (`signal=6`, exit code -6) ~10–15 min into long-context inference. No log line precedes the death:

- No `libc++abi: terminating` message in stderr
- No `[METAL]` / MLX allocator throw
- No entry in macOS Console (`log show`) for the python process
- No diagnostic report in `~/Library/Logs/DiagnosticReports/` or `/Library/Logs/DiagnosticReports/`
- Python's `faulthandler` dumps the *receiving* thread's stack (typically `mx.async_eval` on the main Python thread) but that's just where the signal landed, not where `abort()` was called

The supervisor's only signal is `Runner exited with exit code -6 / Runner terminated with signal=6 (Abort trap: 6)`. Whatever called `abort()` did so silently from a thread Python's faulthandler doesn't enumerate.

## Root cause

Captured via `DYLD_INSERT_LIBRARIES` interposer hooking `abort()` with a native `backtrace_symbols_fd` dump. The backtrace at the moment of crash:

```
0   abort_tracer.dylib                  dump
1   abort_tracer.dylib                  my_abort
2   IOGPU                               -[IOGPUMetalResidencySet removeAllocation:] + 0
3   libmlx.dylib                        mlx::core::metal::ResidencySet::insert + 148
4   libmlx.dylib                        mlx::core::metal::MetalAllocator::malloc + 328
5   libmlx.dylib                        mlx::core::ArgPartition::eval_gpu + 84
6   libmlx.dylib                        mlx::core::gpu::eval + 204
7   libmlx.dylib                        mlx::core::eval_impl + 4668
8   libmlx.dylib                        mlx::core::async_eval + 112
9–18 (CPython interp + dyld start)
```

Reading the stack:

1. MLX's `ArgPartition::eval_gpu` (the `topk` op used heavily by DSv4-Flash's compressed sparse attention indexer) requests a fresh GPU buffer.
2. `MetalAllocator::malloc` allocates one and calls `ResidencySet::insert(buf)`.
3. `ResidencySet::insert` (in `mlx/backend/metal/resident.cpp:28`) calls `wired_set_->addAllocation(buf)` then `wired_set_->commit()` — both are Apple `MTLResidencySet` framework methods.
4. Apple's `addAllocation`/`commit` internally invoke the kernel-side `IOGPUMetalResidencySet::removeAllocation:` (presumably to evict another allocation to make room).
5. Inside `removeAllocation:`, on some undocumented condition, IOGPU calls `abort()` directly. No return value, no exception, no message.

This is reproducible *after* leaving headroom — we tried a 5% capacity-margin guard in `ResidencySet::insert` (commit `858dfd3b`), which did not help. Apple's abort condition is independent of size headroom.

The framework call hierarchy:

```
MLX userspace                Apple userspace                  Apple kernel
─────────────────            ───────────────                  ────────────
ResidencySet::insert
  └─ wired_set_->addAllocation ──> MTLResidencySet::add...
                                     └─ commit ─────────────> IOGPUMetalResidencySet::add
                                                                └─ ...evict path...
                                                                    └─ removeAllocation:
                                                                        └─ abort()  ← here
```

There is no userspace API contract documenting that `addAllocation`/`commit` can abort. From the caller's perspective these are infallible void methods.

## Trigger conditions

All three must hold:

1. **Workload allocates GPU buffers densely.** DSv4-Flash's per-decode-step pattern is roughly 21 dispatches per layer × 43 layers ≈ 900 GPU dispatches per output token, many requiring fresh buffer allocations through `MetalAllocator::malloc`. Models without `ArgPartition`/topk in the hot path (e.g. Huihui-35B-A3B, MiniMax-M2.7) do not surface the abort.
2. **Long sustained inference.** The trigger is probabilistic per allocation. Q1 of an AIME problem (~7K decode tokens, ~6 min) historically completed; Q2 at ~15K–20K decoded tokens and ~12 min sustained ran into the abort reliably.
3. **High wired-memory utilization.** Observed peak 112.3 GB / 124 GB `iogpu.wired_limit_mb` (90.5%) on the crashing node. At lower utilization the abort path appears dormant — the residency set's eviction logic only runs near capacity.

## Reproducer

Cluster: 2× Apple M4 Max Mac Studio (128GB unified memory each), Thunderbolt 5 RDMA, macOS 26.4 (25E241).

Model: `mlx-community/DeepSeek-V4-Flash-8bit` (~155GB, sharded TP across both Studios).

```bash
# In bench/
uv run python exo_eval.py \
  --host 192.168.86.201 \
  --model mlx-community/DeepSeek-V4-Flash-8bit \
  --reuse-instance --keep-instance \
  --tasks aime_2025 --limit 5 \
  --max-tokens 65536 \
  --enable-thinking 0
```

Without the fork patch, the runner SIGABRTs roughly 12 minutes into Q2 (~20K decode tokens) every run. The bench reports `Expecting value: line 1 column 1 (char 0)` (empty HTTP response from a dead runner), the supervisor logs `Runner terminated with signal=6 (Abort trap: 6)`, and `~/exo.log` contains a Python `faulthandler` dump showing the main thread at `mlx_lm/generate.py:1369` (`mx.async_eval`) but no preceding error message.

To capture the actual abort source, build the interposer:

```c
// abort_tracer.c — see /tmp/abort_tracer.c on the cluster nodes
#include <execinfo.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

typedef struct interpose_s { void* repl; void* orig; } interpose_t;
#define INTERPOSE(repl, orig)                                              \
  __attribute__((used)) static const interpose_t                           \
      _interpose_##orig __attribute__((section("__DATA,__interpose"))) = { \
          (void*)(repl), (void*)(orig)}

static void dump(const char* who) {
  fprintf(stderr, "\n=== ABORT_TRACER: %s called ===\n", who); fflush(stderr);
  void* frames[128]; int n = backtrace(frames, 128);
  backtrace_symbols_fd(frames, n, 2); fflush(stderr);
}

extern void abort(void) __attribute__((noreturn));
__attribute__((noreturn)) static void my_abort(void) {
  dump("abort"); signal(SIGABRT, SIG_DFL); raise(SIGABRT); _exit(134);
}
INTERPOSE(my_abort, abort);
```

Build and inject:

```bash
clang -arch arm64 -dynamiclib -o /tmp/abort_tracer.dylib /tmp/abort_tracer.c

# In the exo launch env (start_cluster.sh):
DYLD_INSERT_LIBRARIES=/tmp/abort_tracer.dylib python -m exo
```

Note: macOS strips `DYLD_INSERT_LIBRARIES` when exec-ing into hardened binaries. `caffeinate` is hardened, so wrapping `caffeinate -s python` strips the env. Run `caffeinate` as a sibling process instead.

When the abort fires, `~/exo.log` contains the `=== ABORT_TRACER: abort called ===` block with the IOGPU → ResidencySet → MetalAllocator → ArgPartition stack reproduced above.

## Fix (this fork)

`mlx/backend/metal/resident.cpp` — `adurham/mlx@f8bac642`:

```cpp
void ResidencySet::insert(MTL::Allocation* buf) {
  if (!wired_set_) { return; }
  // Skip wired_set_->addAllocation entirely. Apple's IOGPUMetalResidencySet
  // unconditionally calls abort() from inside addAllocation/commit on
  // certain (undocumented) conditions. Always route through unwired_set_.
  unwired_set_.insert(buf);
}
```

`unwired_set_` is a plain `std::unordered_set` of `MTL::Allocation*` — pure C++ MLX-internal bookkeeping that never touches the Apple residency-set API. Allocations still happen through `MetalAllocator::malloc` and `device_->newBuffer()`; we just don't pin them as "resident."

Tradeoff: GPU may page in lazily on first access for buffers macOS has paged out under pressure. In practice on M4 Max 128GB with `iogpu.wired_limit_mb=124000`, the OS rarely pages anything out — the cost is unmeasurable on our DSv4-Flash decode benchmark. Any future workload that genuinely needs the wired-residency optimization will need a different approach.

A history of attempted partial fixes that did not work:

| Attempt | Result |
|---------|--------|
| `EXO_MAX_ACTIVE_TASKS=5` (cap MLX scheduler queue depth) | Stops a *separate* WindowServer userspace_watchdog crash. Does not affect the IOGPU abort. |
| `AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1` (Apple env, suggested in mlx#3267) | Targets a different IOGPU watchdog (`Impacting Interactivity`). Not our error. |
| `caffeinate -s` wrapping exo | Stops display-related GPU competition. Not our error. |
| `EXO_DSV4_INDEXER_WINDOW=8192` (bound DSv4 indexer attention) | Real perf fix (decode 65ms → 35ms), but doesn't address the abort. |
| `mlx@858dfd3b` — 5% safety margin under `capacity_` in `ResidencySet::insert` | Same backtrace fires. Apple's abort condition isn't a simple size threshold. |
| `mlx@4ff54fed` — port of upstream PR #3318 deferred `check_error` | Doesn't fire — the abort isn't from a Metal command-buffer error escape. |

## Verification

Comparison across two runs of the same bench command (`AIME 2025, c=1, max_tokens=65536, enable_thinking=0`) on `mlx-community/DeepSeek-V4-Flash-8bit`:

| Metric | Before fix | After fix (`adurham/mlx@f8bac642`) |
|---|---|---|
| `signal=6` SIGABRTs in `~/exo.log` over 70 min | many | 0 |
| Q1 outcome | reaches `[stop]` once, often crashes mid-Q1 on retry | clean `[stop]` at 7,825 toks, correct answer (gold=70) |
| Crash window | ~12 min into sustained decode | None during decode |
| Decode rate | 35 ms/step | 30–40 ms/step (slightly *faster* — wiring overhead removed) |
| Peak RSS per node | 112 GB | 8–123 GB (no longer pinned, OS reclaims when possible) |

The bypass fix completely eliminates the silent IOGPU SIGABRT.

(A separate downstream issue — `Event::Event] Failed to create Metal shared event` on Q2 startup, suggesting a SharedEvent leak — surfaces *after* the IOGPU fix removes the original crash. That's a separate filing.)

## Why we think it's an Apple bug

1. The `MTLResidencySet::addAllocation` / `commit` API is documented as void/infallible. There's no documented error path.
2. The abort happens inside an Apple-shipped framework binary (`/System/Library/Frameworks/Metal.framework` + `IOGPU.kext`), called by an MLX userspace API call that follows Apple's documented usage pattern.
3. The abort is silent — `removeAllocation:` is being called from inside `addAllocation`, fails some internal invariant, and calls `abort()` rather than returning/throwing. macOS's own crash reporter doesn't fire for this path.
4. There is no userspace contract that lets the caller predict or avoid the condition. Even adding a 5% capacity safety margin doesn't help.

Related, not-yet-merged Apple/MLX issues:

- [mlx#3186 — Kernel panic in IOGPUMemory.cpp:550 on M4 Max with large context prefill](https://github.com/ml-explore/mlx/issues/3186) — same framework, kernel-panic variant. Filed `Apple Feedback ID: FB22091885`.
- [mlx#3267 — Metal GPU watchdog kills LoRA training when display is active](https://github.com/ml-explore/mlx/issues/3267) — `kIOGPUCommandBufferCallbackErrorImpactingInteractivity` watchdog, different abort path in the same family. Apple: "won't fix at the framework level, set `AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1`."
- [mlx#3346 — Repeated kernel panics in IOGPU.kext during MLX inference (M3 Ultra)](https://github.com/ml-explore/mlx/issues/3346) — same kext, same pattern, merged into #3186.
- [mlx#3224 / #3317 / #3390](https://github.com/ml-explore/mlx/issues/3224) — Metal completion handler `check_error` throw causing `std::terminate` → `abort()`. Different MLX-side path; we ported the proposed deferred-error fix from PR #3318 (`adurham/mlx@4ff54fed`) but that doesn't help the IOGPU residency-set abort.

This filing represents the same family of "Apple framework calls `abort()` from a code path the userspace API doesn't document." Apple has acknowledged the problem class (`AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1` exists as an undocumented escape hatch for a related path) but no fix has shipped for the residency-set variant.

## Test environment

- Hardware: 2× Apple Mac Studio M4 Max (16-core CPU / 40-core GPU / 128 GB unified, 546 GB/s memory bandwidth), Thunderbolt 5 RDMA between the two nodes.
- macOS: 26.4 (25E241), Darwin 25.4.0, kernel `xnu-12377.101.15~1/RELEASE_ARM64_T6041`.
- Python: cpython 3.13.11 (uv-managed).
- MLX: `adurham/mlx@bb17aea3..f8bac642` (fork off `ml-explore/mlx@cf568b1c` ≈ Mar 2026).
- mlx-lm: `adurham/mlx-lm@9f6a9d1` (fork off `ml-explore/mlx-lm@2a1dcf6`).
- exo: `adurham/exo@5159f0b4` (fork off `exo-explore/exo@main`).
- Workload: `mlx-community/DeepSeek-V4-Flash-8bit` (~155 GB, sharded TP).
