# A1 — MLX eval / synchronization semantics (source archaeology)

**Repo under examination:** `/Users/adam.durham/repos/exo/mlx` @ `e40a416b20851d118b061b3a57d8cab70f5756de` (the deployed submodule build). `~/repos/mlx` was NOT read.
All line numbers below are from that tree. Backend = Metal.

---

## Q1 — What does `mx.eval(y)` on the output of a distributed `all_sum` actually DRAIN?

### Direct answer

`mx.eval(y)` does **not** drain all streams and is **not** a device-level barrier. It walks only the unscheduled dependency subgraph of `y`, collects the streams of the ops actually in that tape (`open_streams`, transforms.cpp:242/248), commits only those GPU streams (`gpu::finalize`, transforms.cpp:329-331), and the **host** blocks on exactly one `MTLSharedEvent` — the one attached to the synchronizer array on the output stream (transforms.cpp:118-123, 368 → array.cpp:155 → event.cpp:196/57).

The reason it *behaves* like a barrier in DSv4 is a different mechanism entirely. Under jaccl, `communication_stream()` is a **CPU stream** (`new_stream(Device::cpu)`, jaccl.cpp:88), and the AllReduce has no GPU implementation at all (metal/distributed.cpp:17-19) — it is a `cpu::eval` that enqueues a lambda onto that CPU stream's worker thread (jaccl.cpp:110-115). The MoE `y` is produced on the GPU stream and consumed on the CPU comm stream, so `eval_impl` marks it `needs_fence` with `device_switch=true` (transforms.cpp:159-168) and inserts a `Fence` pair: `Fence::update` on the GPU producer stream (transforms.cpp:301-308) and `Fence::wait` on the CPU comm stream (transforms.cpp:268). With `MLX_METAL_FAST_SYNCH=1` — which exo forces unconditionally (bootstrap.py:300) — `Fence::wait` on a CPU stream is enqueued as a **task on the comm stream's worker thread that busy-spins `__dsb(0xF)` on a shared buffer** until the GPU signals it (fence.cpp:58-77). Correspondingly `Fence::update` encodes, into the GPU stream's command encoder, an `input_coherent` kernel over the *entire* `y` buffer (because `cross_device` is true), then `memoryBarrier(MTL::BarrierScopeBuffers)`, then the `fence_update` kernel (fence.cpp:131-154, barrier at device.cpp:571-573).

That `memoryBarrier` is the real cost. It forces **every dispatch already encoded into that GPU stream's current command encoder** to complete before `fence_update` runs. So the scope of the drain is: *one GPU stream's currently-accumulated command buffer* — not the device, but also not just the ops feeding `y`. And because the command buffer is not committed until `gpu::finalize` at the very end of `eval_impl` (transforms.cpp:330), the comm-stream worker thread is spinning, occupied, and unable to run the collective, for the entire GPU execution window. Only then does the RDMA `all_sum` start; only then is the synchronizer event signaled and the host released. The GPU→spin→collective→host chain is fully serialized, which is consistent with the measured "the wait is local, not on the peer."

`mx.async_eval(y)` **drains exactly the same amount of GPU/CPU work** — it runs the identical `eval_impl` body with `async=true` (transforms.cpp:348 vs 368); the fence insertion, the `input_coherent`/barrier/`fence_update` encode, the comm-thread spin, and the `gpu::finalize` commit are all identical. The *only* differences are (a) the host does not call `.wait()` on the synchronizer, and (b) in async mode an `Event` is created and attached per-stream to every tape array (transforms.cpp:250-261), which makes *subsequent* evals resolve cross-stream dependencies via `in.event().wait(stream)` (transforms.cpp:269-275) instead of `Fence`. For a GPU consumer stream that path is `encoder.wait_event` → `MTL::CommandBuffer::encodeWait` (device.cpp:654-659) — a genuine GPU-side wait with no host block.

### Evidence

1. `mlx/distributed/ops.cpp:34` — `all_sum` builds the `AllReduce` array on `detail::communication_stream(group, s)`, i.e. a stream chosen by the group impl, not the caller's.
2. `mlx/distributed/jaccl/jaccl.cpp:88` — `communication_stream_(new_stream(Device::cpu))`: the jaccl comm stream is a **CPU** stream, pinned once per group.
3. `mlx/distributed/jaccl/jaccl.cpp:90-94` — `communication_stream()` ignores the caller's `StreamOrDevice` and always returns that pinned CPU stream.
4. `mlx/backend/metal/distributed.cpp:17-19` — `AllReduce::eval_gpu` throws "has no GPU implementation": the collective can only ever run as a CPU-stream task.
5. `mlx/distributed/jaccl/jaccl.cpp:105-116` — `JACCLGroup::all_sum` grabs `cpu::get_command_encoder(stream)` and `encoder.dispatch(...)` the actual `group_->all_sum(...)`.
6. `mlx/backend/cpu/encoder.h:44-57` — `CommandEncoder::dispatch` → `scheduler::enqueue(stream_, task)`: work becomes a FIFO task on that stream's single worker thread.
7. `mlx/scheduler.h:29-41, 81-91` — one `StreamThread` per stream index, one `std::queue<std::function<void()>>`, one thread pulling tasks strictly one at a time. A blocking task on this queue blocks everything behind it.
8. `mlx/transforms.cpp:95-105` — `eval_impl(outputs, async)`; the "output stream" is taken from the first unscheduled output's primitive, i.e. for `mx.eval(y)` it is the **comm (CPU) stream**.
9. `mlx/transforms.cpp:118-123` — exactly one `Event` is created for the sync path, on that output stream, and attached to the synchronizer array. This is the only thing the host will block on.
10. `mlx/transforms.cpp:131-193` — the DFS walks only `synchronizer`'s input closure, and only descends into inputs whose `status() == unscheduled`. Already-evaluated arrays are not re-walked.
11. `mlx/transforms.cpp:159-168` — **the cross-stream dependency detector.** If `a.primitive().stream() != in.primitive().stream()`, record `needs_fence[in.id()] = {producer_stream_index, device_switch}` where `device_switch` is true when the devices differ (GPU producer → CPU comm consumer ⇒ true).
12. `mlx/transforms.cpp:242, 248` — `std::set<Stream> open_streams;` populated **only** from `arr.primitive().stream()` of arrays popped off the tape. Nothing enumerates all streams or all devices.
13. `mlx/transforms.cpp:263-277` — per-input dependency resolution: `needs_fence` hit ⇒ `fences[producer_idx].wait(consumer_stream, in)`; otherwise if the input carries a valid unsignaled `Event` on a different stream ⇒ `in.event().wait(stream)`.
14. `mlx/transforms.cpp:279-283` — dispatch: `gpu::eval(arr)` or `cpu::eval(arr)` by `arr.primitive().device()`.
15. `mlx/transforms.cpp:285-299` — the *only* place `eval_impl` can block mid-tape: if `scheduler::n_active_tasks() > MAX_ACTIVE_TASKS` (default 10, `EXO_MAX_ACTIVE_TASKS`, transforms.cpp:33-40) or memory-limit pressure, it `gpu::finalize`s **all open streams** and calls `scheduler::wait_for_one()`. Note this is all *open* streams, still not all streams globally.
16. `mlx/transforms.cpp:301-309, 314-318` — `maybe_update_fence`: after evaluating a producer array flagged in `needs_fence`, create/lookup `Fence` keyed by the **producer** stream index and call `it->second.update(stream, a, cross_device)`.
17. `mlx/transforms.cpp:325-332` — end of `eval_impl`: signal each open stream's event (only streams that *have* an event in the map) and `gpu::finalize(s)` for each open GPU stream. This is where the accumulated GPU command buffer is finally committed.
18. `mlx/transforms.cpp:351-371` — `eval()`: `eval_impl(..., false).wait()` then `scheduler::throw_if_stream_exception()`. The blocking is `array::wait()` on the synchronizer, nothing else.
19. `mlx/array.cpp:155-163` — `array::wait()` → `event().wait()` → host-side event wait.
20. `mlx/backend/metal/event.cpp:196-198, 57-169` — `Event::wait()` → `EventImpl::wait(value)`: a **host poll loop** on `mtl_event_->signaledValue()` (exo fork: spin then 50µs sleeps, with a 40s self-abort). This is the host block, on exactly one shared event.
21. `mlx/transforms.cpp:337-349` — `async_eval()` calls the same `eval_impl` with `async=true` and simply does not `.wait()`.
22. `mlx/transforms.cpp:250-261` — the async-only extra: an `Event` per tape stream, attached to every array and its siblings. This is the mechanism that carries the dependency to the *next* eval.
23. `mlx/backend/metal/fence.cpp:58-77` — `Fence::wait` when `stream.device == Device::cpu`: `scheduler::enqueue(stream, ...)` a lambda that **busy-spins** `while (f.cpu_value()->load(seq_cst) < count) { __dsb(0xF); }`. This occupies the comm stream's worker thread for the whole GPU window.
24. `mlx/backend/metal/fence.cpp:50-56` — slow-mode (`MLX_METAL_FAST_SYNCH=0`) fallback: `f.event->wait(stream)`, which for a CPU stream also enqueues a *polling* task on that same worker thread (event.cpp:200-210). Same serialization, different poll primitive.
25. `mlx/backend/metal/fence.cpp:131-141` — `Fence::update` with `cross_device=true` dispatches the `input_coherent` kernel over `x.data_size() * x.itemsize()` bytes — i.e. **proportional to the size of `y`**, on the GPU stream.
26. `mlx/backend/metal/fence.cpp:144` + `mlx/backend/metal/device.cpp:571-573` — `compute_encoder.barrier()` = `memoryBarrier(MTL::BarrierScopeBuffers)` on the concurrent compute encoder: everything previously encoded in that encoder must complete before `fence_update` runs. **This is the scope of the "drain".**
27. `mlx/backend/metal/fence.cpp:146-154` — the `fence_update` kernel writes `f.count` into the shared buffer the CPU thread is spinning on.
28. `mlx/backend/metal/eval.cpp:119-175` — `gpu::eval` appends to the *current* command buffer and only commits when `encoder.needs_commit()` (ops/MB thresholds).
29. `mlx/backend/metal/device.cpp:662-665` + `device.h:175-176` + `device.cpp:768-769, 780-781` — `needs_commit()` thresholds; on M-series "max" (`'s'`) the defaults are 50 ops / 50 MB, overridable via `MLX_MAX_OPS_PER_BUFFER` / `MLX_MAX_MB_PER_BUFFER`.
30. `mlx/backend/metal/eval.cpp:177-187` — `gpu::finalize(Stream)`: `end_encoding()` + `commit()`. **No host wait.** This is what `eval_impl:330` calls.
31. `mlx/backend/metal/eval.cpp:189-193` — `gpu::synchronize(Stream)` → `CommandEncoder::synchronize()`.
32. `mlx/backend/metal/device.cpp:716-727` — `CommandEncoder::synchronize()` = `end_encoding(); commit(); cbuf->waitUntilCompleted();` — **this is the only `waitUntilCompleted` in the eval/sync path, and `eval()` never reaches it.**
33. `mlx/scheduler.cpp:10-28` — `mlx::core::synchronize(Stream s)`: for a CPU stream, enqueue a promise and `f.wait()` (drain that one queue); for a GPU stream, `gpu::synchronize(s)` (the `waitUntilCompleted` above). **Single stream, explicitly named.**
34. `mlx/scheduler.cpp:34-36` — `synchronize()` with no args targets only `default_stream(default_device())`. It does **not** sweep all streams.
35. `mlx/scheduler.h:264-277, 307-311` — `take_any_stream_exception()` / `throw_if_stream_exception()` is the only code that iterates every stream thread, and it only reads a stored `exception_ptr`. It never waits.
36. `mlx/backend/metal/device.cpp:1085-1088` — `get_command_encoders()` is `static thread_local`. There is no global iterate-and-drain over encoders anywhere in `eval()`/`synchronize()`.
37. `~/repos/exo/src/exo/worker/runner/bootstrap.py:296-302` — exo sets `MLX_METAL_FAST_SYNCH=1` unless `EXO_FAST_SYNCH=false`, so the **fast (busy-spin) fence path is what production runs**.
38. `~/repos/exo/mlx-lm/mlx_lm/models/deepseek_v4.py:3076, 3098, 3130, 3150` — the call site: `all_sum` then `mx.eval(y)` (probe / default branches) or `mx.async_eval(y)` (the `EXO_DSV4_FENCE_ASYNC=1` armed branch).

### CONFIDENCE: **high**

For: the tape-scoped (not global) stream collection; the single-event host block; `Fence` being the cross-stream mechanism; the CPU-ness of the jaccl comm stream; the busy-spin; `async_eval` encoding identical work and differing only in host blocking + event attachment. All read directly, no inference.

### Could NOT determine from source

- **The actual measured share** of the ~1400µs attributable to (a) the `input_coherent` kernel, (b) the `memoryBarrier` drain of previously-encoded ops, and (c) command-buffer commit/scheduling latency. Source tells you these exist and are serialized; it cannot tell you the split. Requires the `MLX_SIGNAL_PROBE=1` / `MLX_GPU_TIME` instrumentation already present in this fork (event.cpp:222-258, eval.cpp:102-117).
- **How many ops are actually accumulated in the GPU stream's encoder at the fence point.** `needs_commit()` (50 ops / 50 MB on M4 Max) means the buffer may have been auto-committed mid-layer, which would shrink the `memoryBarrier` scope. Whether DSv4's per-layer op count crosses that threshold before the fence is an empirical question. `encoder.buffer_ops()` (device.h:100-102) is exposed for exactly this and is logged by `MLX_SIGNAL_PROBE`.
- **Whether `MTL::GPUFamilyMetal3` + macOS 15 actually hold on the deployed Studios**, which is the runtime precondition for fast mode (fence.cpp:14-18). Highly likely on M4 Max / current macOS, but I did not query the live machines (out of scope: no cluster access).
- **INFERENCE (labeled):** that the comm-stream worker thread being pinned in the spin is what serializes GPU-compute and collective. This follows necessarily from evidence 6, 7 and 23 (single FIFO worker thread + a blocking task on it), but I did not observe it in a live trace.

---

## Q4 — Is there any primitive that fences ONE stream without blocking the other?

### Direct answer

Yes — **`Event` is exactly that primitive, and it is already used automatically**, but only when both sides are GPU streams. `Event` wraps a real `MTL::SharedEvent` (metal/event.h:30, event.cpp:45). `Event::wait(stream)` on a **GPU** stream calls `encoder.wait_event(...)` → `MTL::CommandBuffer::encodeWait(mtl_event, value)` (device.cpp:654-659): a pure GPU-side wait, host never blocks. `Event::signal(stream)` on a GPU stream is `encodeSignalEvent` (device.cpp:646-651). On a **CPU** stream both degrade to enqueued host-thread work (event.cpp:200-210, 212-217). `Fence` is the higher-level wrapper `eval_impl` actually uses; in fast mode it is a shared `uint32_t` buffer written by a `fence_update` kernel and read either by a `fence_wait` kernel (GPU consumer — GPU-side, no host block, fence.cpp:80-97) or by a host busy-spin (CPU consumer, fence.cpp:58-77). In slow mode `Fence` just delegates to `Event` (fence.cpp:53-56, 104-108).

**MLX already inserts the cross-stream dependency edge automatically.** transforms.cpp:159-168 detects the stream mismatch during the graph walk and transforms.cpp:263-277 emits the `Fence::wait` / `Event::wait` before scheduling the consumer. This happens inside *any* `eval` that spans both streams — including the next layer's eval — with no user action. **Therefore the explicit `mx.eval(y)` at deepseek_v4.py:3098/3150 is redundant for correctness-of-ordering.** Its actual (and stated) purpose is different: the comment at deepseek_v4.py:3077-3092 says it is there to pin *when* the collective is evaluated so both ranks issue it at the same graph position, preserving cross-rank bit-equivalence — a determinism/lockstep concern, not a data-dependency concern.

**Neither `Fence` nor `Event` is exposed to Python.** Grep of `python/src/` finds no binding for either; the only hits under `python/` are the vendored `metal_cpp` headers and `SOURCES.txt` build manifests. Python's surface is `mx.eval`, `mx.async_eval` (transforms.cpp:1186, 1294) and `mx.synchronize` (stream.cpp:181-186). So mlx-lm can only reach these primitives *indirectly*, via which eval it calls and which streams the ops live on.

**The hard constraint:** a GPU-side-only fence is structurally impossible for this particular edge, because the consumer is jaccl's CPU comm stream (jaccl.cpp:88) and `AllReduce` has no `eval_gpu` (metal/distributed.cpp:17-19). A CPU thread must physically observe that the GPU finished before it can hand the buffer to RDMA. No amount of event plumbing removes that; only moving the collective onto a GPU stream (or overlapping it with unrelated GPU work) would.

### Evidence

1. `mlx/fence.h:9-26` — `Fence` doc: "*Calls to `wait` wait in the given stream until all previous calls to update are complete on their given stream*"; documents the slow/fast split and that fast mode needs `MLX_METAL_FAST_SYNCH=1` + Metal 3.2 / macOS 15+.
2. `mlx/fence.h:32-33` — API surface is exactly `update(Stream, const array&, bool cross_device)` and `wait(Stream, const array&)`.
3. `mlx/backend/metal/fence.cpp:11-27` — `FenceImpl` ctor: `use_fast` requires `supportsFamily(MTL::GPUFamilyMetal3)` **and** macOS 15/iOS 18 **and** `env::metal_fast_synch()`. Slow mode allocates an `Event`; fast mode allocates a 4-byte shared `MTL::Buffer`.
4. `mlx/utils.h:190-192` — `metal_fast_synch()` default is **0**; exo overrides it to 1 (bootstrap.py:300).
5. `mlx/backend/metal/fence.cpp:53-56` — **slow mode `Fence::wait` is just `Event::wait(stream)`.** So slow-mode `Fence` ≡ `Event`.
6. `mlx/backend/metal/fence.cpp:58-77` — **fast mode, CPU consumer: HOST-SIDE BLOCK.** Enqueued on the target stream's worker thread; `while (f.cpu_value()->load(std::memory_order_seq_cst) < count) { __dsb(0xF); }` — an unbounded busy-spin with a full-system barrier per iteration. No timeout, no yield.
7. `mlx/backend/metal/fence.cpp:80-97` — **fast mode, GPU consumer: GPU-SIDE WAIT.** `register_output_array(x)`, then dispatch the `fence_wait` kernel (1 thread) into that stream's encoder. The host does not block; the *GPU* spins in a 1-thread kernel until the value lands.
8. `mlx/backend/metal/fence.cpp:100-108` — `Fence::update` increments `count`; slow mode = `event->set_value(count); event->signal(stream)`.
9. `mlx/backend/metal/fence.cpp:110-125` — fast mode, CPU producer: enqueue a store + `__dsb(0xF)` on that stream's worker thread.
10. `mlx/backend/metal/fence.cpp:127-157` — fast mode, GPU producer: optional `input_coherent` over the whole array when `cross_device`, then `barrier()`, then `fence_update`. All encoded into the producer stream's encoder; **no host block on the update side**.
11. `mlx/event.h:12-56` — `Event` API: `wait()` (host), `wait(Stream)` (in-stream), `signal(Stream)`, `is_signaled()`, monotonic `value_`.
12. `mlx/backend/metal/event.h:8-31` + `event.cpp:43-50` — `EventImpl` holds `NS::SharedPtr<MTL::SharedEvent>` created by `d.mtl_device()->newSharedEvent()`. **Yes, it wraps `MTLSharedEvent`.**
13. `mlx/backend/metal/event.cpp:196-198` — `Event::wait()` (no stream) = `EventImpl::wait(value)` = **host poll**, exo-modified to poll `signaledValue()` in userspace rather than `waitUntilSignaledValue` (event.cpp:57-94 explains why: the kernel wait is uninterruptible and wedges peer ranks).
14. `mlx/backend/metal/event.cpp:200-210` — `Event::wait(Stream)`: CPU stream ⇒ enqueue a host poll on that stream's thread; **GPU stream ⇒ `encoder.wait_event(impl, value)`**.
15. `mlx/backend/metal/device.cpp:654-659` — `CommandEncoder::wait_event` = `end_encoding(); buffer_->encodeWait(event->mtl_event(), value);` — **one stream's command buffer waiting on an event another stream signals. Host does not block. This is the answer to "can one stream's command buffer encodeWait on another's event": yes.**
16. `mlx/backend/metal/device.cpp:646-651` — `CommandEncoder::signal_event` = `end_encoding(); buffer_->encodeSignalEvent(...)` — GPU-side signal.
17. `mlx/backend/metal/event.cpp:212-220` — `Event::signal(Stream)`: CPU ⇒ enqueued host `setSignaledValue`; GPU ⇒ `encoder.signal_event`.
18. `mlx/backend/metal/device.cpp:667-710` — `commit()` registers a completion handler that propagates errors from waited events onto the encoder and poisons/force-signals signaled events on failure, then `buffer_->commit()`. Commit is **non-blocking**; only `synchronize()` (device.cpp:716-727) blocks via `waitUntilCompleted`.
19. `mlx/transforms.cpp:159-168` — **the automatic inter-stream dependency edge.** Detected during the DFS, purely from `primitive().stream()` inequality. No user annotation.
20. `mlx/transforms.cpp:263-277` — where that edge is *materialized*: `Fence::wait` for same-eval cross-stream deps, `Event::wait(stream)` for deps that cross an `async_eval` boundary.
21. `mlx/transforms.cpp:301-308` — the matching `Fence::update` on the producer side, carrying the `cross_device` flag through.
22. `python/src/transforms.cpp:1186, 1294-1303` — Python exposes `eval` and `async_eval` only.
23. `python/src/stream.cpp:181-194` — Python exposes `mx.synchronize([stream])`, which maps to the single-stream `mlx::core::synchronize` (scheduler.cpp:10).
24. **Negative result:** `grep -rn "Fence|Event" python/src/` returns **zero** binding hits. The only `python/` matches are vendored `metal_cpp` headers (`python/mlx/include/metal_cpp/Metal/MTL*CommandEncoder.hpp`) and `python/mlx.egg-info/SOURCES.txt`. Neither `Fence` nor `Event` is constructible or callable from Python.
25. `mlx/distributed/jaccl/jaccl.cpp:88` + `mlx/backend/metal/distributed.cpp:17-19` — the structural constraint: comm stream is CPU, `AllReduce` has no GPU path. The GPU-side `fence_wait` kernel branch (fence.cpp:80-97) is therefore **unreachable for this edge**.
26. `~/repos/exo/mlx-lm/mlx_lm/models/deepseek_v4.py:3077-3092` — the call-site comment states the `mx.eval` exists for cross-rank determinism ("*a lazy graph can let two ranks dispatch the next MoE layer with subtly-different inputs*"), and records that removing it (OPT-7) cost 23% on B=2 prefill.

### CONFIDENCE: **high**

For: the exact host-block-vs-GPU-wait semantics of every `Fence`/`Event` branch; `Event` wrapping `MTLSharedEvent`; `encodeWait` cross-stream capability; the absence of Python bindings; MLX inserting the cross-stream edge automatically at transforms.cpp:159-168/263-277.

**Medium** on one derived claim, flagged explicitly:

- **INFERENCE:** that `mx.eval(y)` is *redundant for ordering correctness*. The code path is unambiguous — MLX will insert the same `Fence`/`Event` edge on the next eval whether or not you fence here. But "redundant" is only true for *data* ordering. The call site's own comment (evidence 26) claims a **cross-rank lockstep** purpose that MLX's automatic edge does not provide (MLX orders streams within one process; it says nothing about the two ranks reaching the collective at the same graph position). One of the three prior failed attempts broke bit-equivalence, which is consistent with that claim being real. Do **not** read this finding as "the eval can just be deleted."

### Could NOT determine from source

- **Whether the cross-rank determinism concern is actually load-bearing** at the current mlx-lm/jaccl versions. That is an empirical A/B, not a source question, and the historical record says one attempt at it already broke bit-equivalence.
- **Whether moving the collective onto a GPU stream is feasible.** It would require a GPU-resident `AllReduce::eval_gpu` (currently a hard throw, metal/distributed.cpp:17-19) plus jaccl exposing a GPU-visible completion. I found no such path in this tree. The nccl backend has its own `communication_stream` (nccl.cpp:320) but that is CUDA and irrelevant here.
- **Whether `EXO_MAX_ACTIVE_TASKS` throttling (transforms.cpp:285-299) is firing in the decode loop.** If `n_active_tasks() > 10`, `eval_impl` will `gpu::finalize` + `scheduler::wait_for_one()` *mid-tape*, adding host blocks unrelated to the fence itself. Source shows the branch exists; only a live counter tells you whether decode trips it. exo's benches set it to 5 in some arms (`bench/prefill_cliff_gclimit_repro_local_v2.py:112`) but I found no production default override — it stays at 10.
