# Q5 — Per-kernel GPU-time instrument on MLX + Metal + M4 Max + JACCL: FEASIBILITY

**Date:** 2026-10-09 · **Tier:** mid-coder (delegation) · **Scope:** offline feasibility + minimal prototype
**Local mlx source:** `/Users/adam.durham/repos/mlx` @ `ac73d0c9eeb2240725fb203d3e6745c516a3dd8e`
**Node build:** `studio2` venv `~/repos/exo/.venv/bin/python`, mlx `0.32.3.dev20260918+603f16eb7`,
lib `.../mlx/lib/libmlx.dylib` (22.9 MB, Sep 18 17:17), `mlx/core.cpython-313-darwin.so` → `@rpath libmlx.dylib`
**Verdict: GO — but NOT via a flag flip. The existing mechanism is per-command-buffer GPU-busy
time, not per-kernel. It is turn-on-able today (`MLX_GPU_TIME=1`) and gives a real, cheap,
non-wall-clock instrument; closing the "unattributed 40 %" needs a small C++ patch or a
GPU-capture (Xcode) workflow, both bounded.**

---

## 1. TL;DR

| Question | Answer |
|---|---|
| Does MLX already measure per-command-buffer GPU time on Apple GPUs? | **YES.** Sum of `GPUEndTime() - GPUStartTime()` per completed `MTL::CommandBuffer`, accumulated into a global atomic. |
| Does it measure **per-kernel** GPU time? | **NO.** One command buffer can carry many dispatches (`buffer_ops_` up to 50); the counter returns the whole-buffer span. No per-kernel or per-dispatch timestamp exists in this tree. |
| What enables it? | `MLX_GPU_TIME=1` env var, set **before** `import mlx`. Exposed as `mx.metal.gpu_time_ns()` / `mx.metal.reset_gpu_time()`. |
| Does it work on THIS stack (M4 Max / custom EXL3+moe kernels / compiled)? | **YES, verified live.** M4 Max / Metal 4. Compiled and custom-kernel dispatches share the same `CommandEncoder` and are counted. Counter returns nonzero. |
| JACCL RDMA interaction? | None found. The filter *explicitly discards* RDMA/CPU-only collective stubs (timestamps ≤ 0 or delta > 1 s). This is the source of the "invalid timestamps" behaviour. |
| Granularity | Per committed command buffer; buffers commit every `max_ops_per_buffer_ = 50` ops on the `s` (Max) arch (`MLX_MAX_OPS_PER_BUFFER` overrides). Single-op eval ⇒ single-kernel buffer ⇒ ≈ per-kernel. |
| Overhead | ~0 added GPU work (Metal records timestamps automatically). Per-op attribution costs a forced commit+sync per op (~1.15× on hot matmul loops) — acceptable for a diagnostic, not free. |
| Gap to the campaign need (attribute 40 % of prefill chunk / decode round per kernel) | Per-op-class attribution is achievable today with existing primitives + a thin wrapper around the EXL3/MoE dispatch sites. True per-dispatch *name→time* attribution needs a C++ patch (per-buffer label→time map) or Xcode GPU capture. |

---

## 2. The existing mechanism (exact citations)

**Accumulation (the "invalid timestamps" filter that the profile review flagged):**
`mlx/backend/metal/eval.cpp:94-109` `accumulate_gpu_time_if_enabled(MTL::CommandBuffer*)`:
- `if (!metal::gpu_time_enabled()) return;` — `eval.cpp:95`
- `double start_s = cbuf->GPUStartTime(); end_s = cbuf->GPUEndTime();` — `eval.cpp:98-99`
  (Metal `CFTimeInterval`, seconds, **automatically recorded by Metal at completion**; comment `eval.cpp:81-83`)
- rejects `start_s <= 0 || end_s <= 0` — `eval.cpp:100-102` (CPU-only buffers, **jaccl RDMA collective stubs**, errored buffers → 0)
- rejects `delta <= 0 || delta > 1.0` s — `eval.cpp:104-106` (catches the start=0 pathological case)
- `accumulate_gpu_time_ns(delta_s * 1e9)` — `eval.cpp:107-108`

**Where it is called (completion handlers):**
- `eval.cpp:146-151` — commit-time handler: `notify_task_completion`, `check_error_deferred`, `accumulate_gpu_time_if_enabled`
- `eval.cpp:153-165` — non-commit branch deliberately does **not** accumulate (comment `eval.cpp:154-160`: N appends to one buffer would each see the SAME GPUStart/End → N× overcount). Canonical point is the single commit.
- `eval.cpp:168-178` `finalize(Stream)` — end_encoding + commit + handler. `finalize` is called by `mlx/transforms.cpp:285,324` (the eval/`mx.metal` finalize path).

**State + gate:**
- `mlx/backend/metal/device.cpp:65` `std::atomic<uint64_t> g_gpu_time_ns_{0};` (comment `device.cpp:55-64`)
- `mlx/backend/metal/device.cpp:896-904` `gpu_time_enabled()` — reads `getenv("MLX_GPU_TIME")`, `atoi != 0`, cached in a function-local static (first call wins → **must be set before mlx is imported**).
- `mlx/backend/metal/device.cpp:906-916` `gpu_time_ns()`, `reset_gpu_time()`, `accumulate_gpu_time_ns()`.

**Header contract:** `mlx/backend/metal/metal.h:32-43` — "Sum of (GPUEndTime - GPUStartTime) per command buffer … Gated on `MLX_GPU_TIME` (default off) … always returns 0 when unset."

**Python surface:** `python/src/metal.cpp:118-139` (nanobind) — `mx.metal.gpu_time_ns()`, `mx.metal.reset_gpu_time()`; docstrings at `metal.cpp:120-139` state the gate and the "call `mx.eval`/`mx.synchronize` before reading" contract.
**No `.pyi` stubs** exist in this tree; discovery is `hasattr`/`__doc__` (verified present on the node build).

**Why it is per-buffer, not per-kernel — the commit model:**
- `mlx/backend/metal/device.h:118` `int buffer_ops_{0};` and `:96-98` `buffer_ops()` accessor.
- `mlx/backend/metal/device.cpp:389-410` — every `dispatch_threadgroups`/`dispatch_threads` bumps `buffer_ops_++` (`device.cpp:393,404`).
- `mlx/backend/metal/device.cpp:484-487` `needs_commit()`: `buffer_ops_ > max_ops || (buffer_sizes_>>20) > max_mb`.
- `mlx/backend/metal/device.cpp:538-561` — arch switch: `'s'` (Max) → `max_ops_per_buffer_ = 50`, `max_mb = 50`; overridable via `MLX_MAX_OPS_PER_BUFFER` / `MLX_MAX_MB_PER_BUFFER` (`mlx/utils.h:176-186`, `get_var` `mlx/utils.cpp:289-295`).
- `mlx/backend/metal/device.cpp:489-494` `commit()` resets `buffer_ops_ = 0`.
- Encoders are per-stream: `mlx/backend/metal/device.h:236-238` `get_command_encoder(Stream)`, `new_stream` at `eval.cpp:52-57`, encoder created `device.cpp:512-519` (`computeCommandEncoder(MTL::DispatchTypeConcurrent)`).

⇒ **One `mx.eval` of a single op commits one buffer holding one dispatch ⇒ the counter reads that
op's GPU time.** One `mx.eval` of a fused graph of up to 50 ops commits one buffer ⇒ the counter
reads the *summed* GPU span of that buffer. This is exactly the granularity split the campaign cares about.

**Compiled + custom kernels (the EXL3/MoE path):**
- Compiled: `mlx/backend/metal/compiled.cpp:392-467` uses `metal::get_command_encoder(s)` and `dispatch_threads` — **same encoder/counter** as everything else.
- Custom (`mx.fast.metal_kernel`): `mlx/backend/metal/custom_kernel.cpp:327-425` likewise `get_command_encoder` + `dispatch_threads`.
⇒ Both are counted by the same `accumulate_gpu_time_if_enabled` at commit. **The compiled-prefill "does not enter spans" gap is covered by the counter** (the profiler-span gap is orthogonal; the GPU-time counter sits below it in C++).

**What does NOT exist in this tree (the gap):**
- No per-dispatch timestamp. `MTL::CommandBuffer` exposes only `GPUStartTime()/GPUEndTime()` (whole-buffer).
  `grep` for `sampleTimestamps`/`GPUStartTimestamp`/`MTLCounterSampleBuffer`/`insertDebugSignpost` in
  `mlx/`, `python/` → **empty** (source uses none). (The node `libmlx.dylib` does carry generic
  `MTL::Private::Selector s_ksampleTimestamps_gpuTimestamp_` / `s_knewCounterSampleBufferWithDescriptor_error_`
  selectors — these are vendored nanobind/MTL-private glue, **not** wired into the backend.)
- No per-kernel *name*→time map. Command-buffer labels exist (`mlx/backend/metal/utils.h:35-46`
  `debug_set_primitive_buffer_label`, called `eval.cpp:128`) but are **`#ifdef MLX_METAL_DEBUG`** only
  (`utils.h:38`), and in a fused buffer the label appends *every* op's name (`utils.h:40-44`) — it does
  not map to time. The shipping build was compiled **without** `MLX_METAL_DEBUG` (label-append symbol
  absent from `libmlx.dylib`; `nm | grep METAL_DEBUG` = 0).
- No `get_kernel_time`/profiler Python API beyond `gpu_time_ns`/`dispatch_count`/`capture`.
- **JACCL:** `mlx/backend/metal/distributed.cpp` is 38 lines and contains **no** GPU-time/timestamp hook.
  JACCL's own timing is host-side `mach_absolute_time` (`mlx/distributed/jaccl/lib/jaccl/mesh_impl.h:101-103,262-281,405,619-621`) — RDMA progress/watchdog, not GPU kernel time. So TP collective GPU cost is **only** visible as the RDMA-stub buffers the filter discards; the real all-reduce GPU work (if any) would appear in whatever buffer encodes it, but the *RDMA transport itself* has no GPU timestamp.

---

## 3. Prototype — RUN, verified on studio2 (M4 Max, node build)

Two standalone scripts (no engine, no cluster request, no deploy):

**`proto_gpu_time.py`** — `MLX_GPU_TIME=1 MLX_DISPATCH_COUNT=1 ~/repos/exo/.venv/bin/python`
```
mlx: 0.32.3.dev20260918+603f16eb7   (has gpu_time_ns / reset_gpu_time / dispatch_count)
WIRED_CHECK gpu_time_ns after one 1024^3 fp16 matmul = 343249 ns (343.2 us) -> EXPOSED

case                                       gpu_ms    wall_ms gpu/wall
matmul_1024x1024x1024_fp16                  0.447      0.611    0.732
matmul_4096x4096x4096_fp16                 13.285     13.482    0.985
elementwise_8M_fp16                         0.555      0.767    0.723
quantized_matmul_mxfp4_64x1024x4096         0.053      0.191    0.275
dispatch_count for one 1024^3 matmul = 12
```
Reading: real GPU time per isolated op, distinct per shape. `gpu/wall` at low occupancy
(0.28–0.73) is the **Python + sync/launch overhead** the wall clock hides; at saturation
(4096³ matmul) it rises to 0.985 ⇒ GPU-bound. This is exactly the "async-span" blind spot, now measurable.

**`proto_granularity.py`** (default and `MLX_MAX_OPS_PER_BUFFER=1`):
```
[A] single matmul   gpu=1.176ms wall=1.311ms dispatches=20   (20 because eval-drain reruns warmups*… see note)
[A] matmul+relu     gpu=1.187ms wall=1.379ms dispatches=40   -> 2nd op adds ~0 GPU ms (launch-latency hidden)
[A] matmul+exp      gpu=1.187ms wall=1.384ms dispatches=40   -> SAME: fused graph shares one buffer span
[C] mx.compile(matmul+relu) gpu=2.353ms->counted YES (1.191ms with MLX_MAX_OPS_PER_BUFFER=1)
[D] read-before-sync=0 ns   read-after-sync=2327250 ns       -> MUST synchronize (async undercount)
[B] 20-op batched wall=22.74ms (1.137ms/op); per-op-sync wall=1.305ms/op  -> 1.15x overhead
```
Reading:
- **[A] is the granularity proof**: adding a second op to the same buffer leaves the per-buffer GPU
  time ~unchanged unless the second op does material work — i.e. the counter reports the *buffer span*,
  not per-dispatch. (`MLX_MAX_OPS_PER_BUFFER=1` forces one-op buffers; the compiled case then reports
  1.19 ms instead of 2.35 ms — direct evidence that one buffer was holding multiple dispatches.)
- **[C] confirms the compiled path is counted** — the exact gap the span profiler had.
- **[D] confirms the async trap**: read `gpu_time_ns()` before `mx.synchronize()` → 0.
- **[B] quantifies the per-op-sync cost at ~1.15×** on a hot matmul — cheap enough for a diagnostic pass.

---

## 4. The gap to the campaign need, and the concrete path

**Need:** attribute the ~40 % unattributed prefill-chunk wall and decode round to specific kernels
(EXL3 MoE gather/segmented GEMM, dense projections, TP all-reduce, glue), as **GPU time**, not Python
queueing time, including compiled paths.

**What the existing mechanism gives for free:** per-op-class GPU time and the "GPU busy vs wall"
decomposition, for any op class you can wrap in `reset_gpu_time() / eval / synchronize / gpu_time_ns()`.
This is the SAME pattern already used in the repo (`bench/item2_opclass_breakdown.py:60-75`,
`bench/exl3_dense_smallm_probe_megaeval.py:242`), but those scripts note `gpu_time_ns` "is dead on
this build" — see §5 caveat.

**Chosen path (in order of cost):**

1. **Zero-code, today (hours):** a wrapper module in exo that brackets each EXL3/MoE/dense/all-reduce
   dispatch site with `reset_gpu_time() … synchronize() … gpu_time_ns()`, and sums `dispatch_count()`.
   Run under `MLX_GPU_TIME=1 MLX_DISPATCH_COUNT=1`. Output: **per-op-class GPU ms per layer/chunk** —
   directly closes the "unattributed 40 %" at *class* granularity. Effort: **~2–4 h** (mechanical,
   no mlx rebuild). Caveat: forces a sync per site (1.15× on hot loops; disable in production).

2. **Small C++ patch (1–2 rounds):** make the completion handler emit **per-commit-buffer records**
   `{buffer_label, GPUStartTime, GPUEndTime, buffer_ops, stream}` to a ring buffer / stderr, exposed as
   `mx.metal.gpu_time_records()`. Requires compiling with `-DMLX_METAL_DEBUG` (or unconditionally
   labelling at `eval.cpp:128`). With `MLX_MAX_OPS_PER_BUFFER` small, this gives near-per-kernel
   attribution **without** a per-op sync (keeps overlap). Effort: **~1 round** (edit `eval.cpp`,
   `utils.h`, `device.{h,cpp}`, `python/src/metal.cpp`; rebuild the node wheel — this is the "what would
   need building" item). File:line targets: label at `utils.h:35-46`/`eval.cpp:128`; emit at
   `eval.cpp:146-151`; ring buffer beside `device.cpp:65`.

3. **True per-kernel timestamps (senior, multi-round):** add an `MTL::CounterSampleBuffer`
   (stage-boundary / timestamp counter) or `sampleTimestamps(gpuTimestamp:)` at encode time
   (`CommandEncoder::get_command_encoder` `device.cpp:512-519`), reading GPU timestamps at each
   dispatch boundary. macOS/Metal 4 on M4 Max supports counter sample buffers. Effort: **hard, ~2–4
   rounds** — needs a rebuilt mlx with counter-heap plumbing and careful calibration. Only worth it if
   (1)+(2) prove insufficient.

4. **Xcode GPU capture (zero mlx code):** `mx.metal.start_capture("trace.gputrace")` / `stop_capture()`
   (`python/src/metal.cpp:78-94`, impl `mlx/backend/metal/metal.cpp:14-47`, uses `MTLCaptureManager`
   — present in the node `libmlx.dylib`). Capture a short bench window, open the `.gputrace` in Xcode →
   per-encoder/per-dispatch GPU time **by kernel name**. Effort: **~1 h** but requires a GUI Xcode
   session on the node (or the trace copied to a workstation). Best "see everything once" tool;
   not automatable into a numeric campaign artifact.

**Recommendation:** ship **path 1** this round (hours, no rebuild) to convert the 40 % into
per-op-class GPU ms; file **path 2** as the next round's instrument (near-per-kernel, no sync tax);
hold **path 3** in reserve. Path 4 as an occasional human-in-the-loop sanity check.

---

## 5. Caveats / UNKNOWNS

- **Version divergence (important).** Local source HEAD is `ac73d0c`; the node ships
  `0.32.3.dev20260918+603f16eb7`. The mechanism is **present and functional** in the node build
  (live-verified: nonzero counter, `gpu_time_ns`/`reset_gpu_time`/`dispatch_count`/`start_capture`
  all present; `MLX_GPU_TIME`, `MLX_DISPATCH_COUNT`, `MLX_SIGNAL_PROBE`, `MLX_EVENT_WAIT_*` strings
  and the `gpu_time_*` symbols present in `libmlx.dylib`). But the "dead on this build" remarks in
  `bench/dsv41_loop2/moe_bench_report.md:6` and `bench/exl3_dense_smallm_probe_megaeval.py:242` are
  **unexplained by the code** — most likely historical (env var set after import, or read before
  `synchronize`). **The prototype here reproduces a nonzero reading on that same node build**, so the
  mechanism is live *now*; the earlier "dead" note should be re-tested, not trusted.
- **Per-buffer ≠ per-kernel** is the central limitation — stated, measured (§3[A]).
- **Filter drops RDMA/CPU-only buffers** (`eval.cpp:100-106`): TP collective *transport* GPU cost is
  not attributed; only real GPU work encoded in a buffer is. JACCL RDMA has no GPU timestamp.
- **`gpu/wall` at low occupancy** mixes Python/launch overhead; it is not pure GPU idle.
- **UNKNOWN:** whether the node wheel's `MLX_MAX_OPS_PER_BUFFER` default matches `'s'`→50 (arch string
  from `MLX_METAL_GPU_ARCH` or device name). Not separately probed; the `=1` run demonstrates the
  override works. The dispatch counts in §3 are inflated by warmup/drain reruns in the harness — treat
  them as relative, not absolute.
- **UNKNOWN:** exact per-round cost of recompiling the node mlx wheel (path 2/3) — depends on the
  node's build toolchain, not inspected here (out of scope / offline read-only).

---

## 6. Files

- This memo: `raw/pricing/q5/q5-gpu-instrument.md`
- `raw/pricing/q5/proto_gpu_time.py` — per-op GPU-time prototype (source)
- `raw/pricing/q5/proto_granularity.py` — granularity/overhead probe (source)
- `raw/pricing/q5/q5_proto.json` — prototype #1 raw output (from studio2 `/tmp/q5_proto.json`)
