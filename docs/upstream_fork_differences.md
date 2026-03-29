# Fork Changes That Could Break Things — And Why We Made Them

Generated 2026-03-28. Every change we made to MLX, MLX-LM, and EXO that
carries risk, organized by what could go wrong and why we did it anyway.

---

## Table of Contents

- [MLX Changes](#mlx-changes)
- [MLX-LM Changes](#mlx-lm-changes)
- [EXO Changes](#exo-changes)

---

# MLX Changes

## 1. Global Streams Instead of Per-Thread Streams

**What we changed:** Deleted `mlx/stream.cpp` (76 lines). Moved all stream
management into `Scheduler`. One global default stream per device type instead
of upstream's `thread_local` per-thread defaults.

**Why:** Our RDMA fence drain system needs to reach into other threads' stream
queues from the RDMA poll loop. With per-thread streams, the poll loop on
the communication thread can't access the GPU stream thread's fence queue.
Global streams make `drain_fences()` work.

**What could break:** Any MLX code that relies on per-thread default streams
will silently share streams across threads. If upstream MLX ever adds
multi-threaded evaluation (e.g., parallel compilation), our global streams
will serialize it. Also, `mx.set_default_stream()` in one thread affects ALL
threads — upstream isolates this per-thread.

---

## 2. Raw Metal Pointers Instead of Smart Pointers

**What we changed:** All Metal objects (`MTL::Device*`, `MTL::Heap*`,
`MTL::ResidencySet*`, `MTL::CommandQueue*`) use raw pointers with manual
`release()` instead of upstream's `NS::SharedPtr<T>`.

**Why:** We branched before upstream's PR #3282 added smart pointers. Not a
deliberate choice — we just haven't merged it.

**What could break:** Manual `release()` is error-prone. If any code path
skips a `release()` call (exception, early return), we leak Metal objects.
Upstream's smart pointers handle this automatically. We've wrapped some
allocation sites with `new_scoped_memory_pool()` to mitigate, but it's not
comprehensive.

---

## 3. Scheduler Cleanup on Non-Windows

**What we changed:** `~Scheduler()` synchronizes all streams and deletes
`StreamThread*` objects on non-Windows platforms. Upstream intentionally leaks
the scheduler on ALL platforms.

**Why:** RDMA resources (MR registrations, buffer pools, QPs) must be properly
cleaned up. Leaking them causes stale QPs on the NIC after process restart.

**What could break:** Static destruction order. If the scheduler destructs
after Metal objects it depends on (device, command queue), we crash during
shutdown. Upstream chose to leak specifically to avoid this. We've only seen
this issue on Windows, hence the platform guard.

---

## 4. FenceTask Drain System

**What we changed:** New `FenceTask` struct with `drainable` flag. Separate
`fence_q` deque in `StreamThread`. `drain_fences()` called from RDMA poll
loops. `Fence::wait` tasks marked `drainable=true`, `Fence::update` tasks
marked `drainable=false`.

**Why:** RDMA poll loops block the CPU stream thread. Without draining fence
tasks during polls, Metal fence signals starve and the GPU deadlocks.

**What could break:** If a fence task is incorrectly marked `drainable=true`
when it should be `false` (i.e., it writes fence values instead of just
checking), we corrupt GPU synchronization state — the GPU may proceed past a
fence that hasn't actually been signaled. If marked `drainable=false` when it
should be `true`, we're back to GPU deadlocks during long RDMA transfers.
The current classification (wait=drainable, update=not) is correct but
fragile — any new fence operation in upstream must be classified correctly.

---

## 5. Split-K Quantized Matmul Removed

**What we changed:** Deleted `qmm_splitk`, `fp_qmm_t_splitk` kernels and all
dispatch code. Removed `K_eff` parameter from `qmm_t_impl`.

**Why:** Our workload is batch=1 decode (GEMV). Split-K targets small-M batch
matmuls. No benefit for GEMV.

**What could break:** If we ever run batch>1 inference (multiple concurrent
requests), small-M matmuls during batched decode would be slower without
split-K. Upstream added this specifically for that case. Also, any upstream
code that calls `qmm_splitk` directly would fail to compile.

---

## 6. SliceUpdate Simplified (No Reduce Operations)

**What we changed:** `SliceUpdate` primitive has no `ReduceType` enum, no
`reduce_type_` member. GPU/CPU eval is copy-only.

**Why:** No model we run uses slice-update-with-reduce.

**What could break:** Any model or operation that uses `mx.slice_update` with
a reduce parameter would crash. This is a compile-time error (missing API)
so it would be caught immediately, not silently wrong.

---

## 7. RoPE Validation Removed

**What we changed:** Removed `dims > 0`, `dims % 2 == 0`, `dims <= input dim`
checks from `mlx/fast.cpp` `rope()`.

**Why:** Qwen3.5's DeltaNet layers use non-standard dims that fail these
validations.

**What could break:** Invalid dims would silently produce garbage output
instead of raising an error. If a model accidentally passes dims=0 or
negative dims, we'd get mysterious NaN/incorrect results instead of a clear
error message.

---

## 8. Module.update() Bounds Check Removed

**What we changed:** Removed `if i >= len(dst)` bounds check in
`python/mlx/nn/layers/base.py`.

**Why:** Our fused weight layouts produce lists with different lengths than the
original model definition. The bounds check rejected these legitimate loads.

**What could break:** If a weight file has more entries than the model expects
(e.g., corrupted checkpoint), we'd silently ignore the extra weights instead
of raising an error. Could mask real loading bugs.

---

## 9. CPU Sort NaN Handling for Half Types

**What we changed:** Uses `std::is_floating_point_v<T>` instead of upstream's
custom `is_floating_v<T>` trait.

**Why:** We branched before upstream added the custom trait.

**What could break:** `std::is_floating_point` returns `false` for
`float16_t` and `bfloat16_t`. This means NaN handling in sort operations is
BROKEN for half-precision types — NaN values won't be sorted to the end as
IEEE 754 expects. Any code that sorts float16/bfloat16 arrays containing NaN
will get incorrect ordering. **This is a real bug we should fix.**

---

## 10. Depthwise Conv 1D Kernel Name

**What we changed:** Kernel name concatenation order differs from upstream:
`"depthwise_conv_1d_" + large_suffix + type` vs upstream's
`"depthwise_conv_1d_" + type + large_suffix`.

**Why:** Unclear — likely a merge artifact.

**What could break:** If a model uses large depthwise conv 1D, the kernel
lookup would fail with "kernel not found". Whether our order or upstream's is
correct depends on how the kernels are registered in the Metal library. Need
to verify.

---

## 11. PIPELINE Depth 2→3

**What we changed:** All RDMA pipeline depths in mesh_impl.h and ring_impl.h
changed from 2 to 3.

**Why:** Deeper pipelining hides RDMA latency on Thunderbolt 5.

**What could break:** Requires `NUM_BUFFERS >= 3`. If someone sets
`MLX_JACCL_NUM_BUFFERS=2` via env var, the assertion fires and the process
aborts. Also, 3 in-flight buffers means more memory consumption for RDMA
buffers (50% more than PIPELINE=2). On memory-constrained nodes, this could
push total allocation past available unified memory.

---

## 12. All Sends Signaled

**What we changed:** `IBV_SEND_SIGNALED` always set on every send WR.

**Why:** Apple JACCL doesn't reap unsignaled WRs, causing send queue overflow.

**What could break:** Signaled sends generate more CQ entries, which means
more poll loop iterations. On very high-throughput transfers, CQ overflow is
possible if the poll loop can't keep up. We haven't seen this in practice
because our transfers are bounded by model layer sizes, but it's theoretically
possible with very small, very frequent sends.

---

## 13. Pre-Posted Recv Buffers

**What we changed:** `pre_post_recv()` posts recv buffers at array creation
time (before eval). Atomic counter tracks pre-posted count.

**Why:** UC RDMA silently drops sends if no recv is posted.

**What could break:** If a recv buffer is pre-posted but the corresponding
send never happens (e.g., sender crashes), the recv buffer is permanently
consumed — it will never complete and can't be reused. With limited buffer
pools, this could exhaust recv buffers over time. In practice, a sender crash
triggers process restart on all nodes, which resets everything.

---

## 14. Array Capture in RDMA Lambdas

**What we changed:** All dispatch closures capture input/output arrays by value.

**Why:** Prevents use-after-free when Python GCs arrays during async RDMA.

**What could break:** Capturing arrays by value keeps them alive longer than
necessary. For large arrays (e.g., 512MB hidden state tensors), this means
peak memory usage increases because both the original array AND the captured
reference exist simultaneously until the RDMA completes. Could trigger OOM
on very tight memory budgets.

---

## 15. send() Drains All Completions Before Returning

**What we changed:** Small sends prefill PIPELINE buffers then drain ALL CQ
entries before returning.

**Why:** Prevents stale CQ entries from corrupting subsequent all_gather.

**What could break:** Draining all completions is a blocking synchronization
point. If a completion is delayed (e.g., NIC busy), `send()` blocks the
entire stream thread until it arrives. This could add latency spikes to the
decode loop. We accept this tradeoff because the alternative (CQ corruption)
is worse.

---

## 16. NEON-Vectorized Reduce — bfloat16 Scalar Path

**What we changed:** bfloat16 reductions use scalar path instead of NEON
`vaddq_f16`.

**Why:** `vaddq_f16` is IEEE float16 only. Using it on bfloat16 (different
exponent/mantissa bit layout) produces numerically garbage results.

**What could break:** bfloat16 reductions are ~4x slower than float32 NEON
reductions. If a model uses bfloat16 throughout and relies heavily on
all_reduce (e.g., tensor parallel), the reduce step could become a
bottleneck. In practice, our PP mode doesn't use all_reduce on bfloat16
tensors heavily.

---

## 17. Fused Quantized SDPA Kernel

**What we changed:** New `sdpa_vector_quantized` Metal kernel with inline
dequantization.

**Why:** Eliminates one full KV cache read (dequantize + attention → fused
single-pass attention with inline dequant).

**What could break:** The inline `dequant4()` function extracts 4 uint8 from
packed uint32 with specific bit layout assumptions (8-bit affine, group_size
aligned). If the quantization format changes or a model uses a different
packing, the kernel silently produces wrong attention output. Also, the
comment in `base.py` notes "causes Metal GPU timeouts on M4 base" — the
kernel hangs on non-Max chips, hence the env var gate.

---

## 18. Two-Loop SDPA Kernel

**What we changed:** `sdpa_vector_2loop` uses threadgroup memory for scores.

**Why:** Eliminates online softmax serial dependency.

**What could break:** `MAX_POS = 8192` — if sequence length divided by block
count exceeds 8192 positions per block, the kernel writes out of bounds in
threadgroup memory. The dispatch code should prevent this, but if block count
calculation has a bug, it's a silent GPU memory corruption.

---

## 19. SharedBufferPool mlock()

**What we changed:** Pool memory is `mlock()`'d to wire pages.

**Why:** Prevents Metal GPU pressure from paging out RDMA buffers.

**What could break:** `mlock()` permanently reduces available physical memory.
On a 128GB machine, the pool might be 50-100MB (negligible), but if pool size
calculation has a bug and allocates too much, we could lock a significant
fraction of RAM, causing system-wide memory pressure and OOM kills of other
processes.

---

# MLX-LM Changes

## 20. DeltaNet State in bfloat16 Instead of float32

**What we changed:** DeltaNet/SSM recurrence state uses input dtype (bfloat16)
instead of float32. Removed `StT` template parameter from Metal kernel.

**Why:** float32 DeltaNet state doubled per-layer state size, changing RDMA
buffer sizes and triggering buffer overflows in the JACCL send path. The RDMA
abort was non-recoverable (process crash).

**What could break:** After 512+ recurrence steps, bfloat16 accumulation loses
precision. The recurrence `state = state * decay + new_value` accumulates
error over steps. Upstream added float32 specifically because they measured
precision degradation. Our `NaN clamp` in the generate loop catches total
blowups but NOT gradual precision loss. Long-context conversations (>4K
tokens of DeltaNet-processed positions) may have subtly degraded output
quality. **This is probably our highest-risk change for output quality.**

---

## 21. Generation Stream → Default Stream

**What we changed:** `mx.new_stream()` → `mx.default_stream()`.

**Why:** JACCL RDMA collectives deadlock on secondary Metal streams.

**What could break:** The default stream is shared with all other MLX
operations. If any code outside generate_step dispatches work on the default
stream concurrently (e.g., the API layer preprocessing the next request), it
will interleave with generation. Upstream created a separate stream
specifically to isolate generation from other work.

---

## 22. mx.contiguous() on DeltaNet Cache — Missing cache.advance(S)

**What we changed:** `cache[1] = mx.contiguous(state)` WITHOUT calling
`cache.advance(S)` afterward. Upstream has `cache[1] = state` followed by
`cache.advance(S)`.

**Why:** `mx.contiguous()` fixes the prefill memory leak (breaks shared-buffer
reference from small slice back to full computation graph).

**What could break:** `cache.advance(S)` updates the ArraysCache's internal
offset to track how many sequence positions have been processed. Without it,
the cache offset may be wrong, causing subsequent operations to read/write
at incorrect positions. **Upstream added `cache.advance(S)` in PR #1024
specifically to fix this.** Our snapshot/restore mechanism for speculation
may handle offsets differently, but this needs verification. If the offset
is wrong, we'd get incorrect attention output for DeltaNet layers after the
first prefill chunk. **This is a critical correctness risk.**

---

## 23. Late Binding in Cache Checkpoint Generator

**What we changed:** Fork uses inline `(c.extract(i) for c in prompt_cache)`
instead of upstream's `_lazy_extract_cache()` helper.

**Why:** We didn't pick up upstream's fix (PR #976).

**What could break:** In `BatchGenerator.prompt_checkpoint_callback`, the
generator expression captures `i` by reference (Python closure semantics).
By the time the generator is consumed, `i` may have advanced to a different
value. This means cache checkpoints could save the wrong layer's state.
**This is a correctness bug in batch generation with multiple requests.**

---

## 24. BatchRotatingKVCache Missing mx.depends

**What we changed:** Removed `self.keys = mx.depends(self.keys,
(self.left_padding, self.offset))` in two places.

**Why:** We branched before upstream's fix (PR #960).

**What could break:** `left_padding` and `offset` are updated via in-place
mutation. Without `mx.depends`, MLX's lazy evaluation may not materialize
these updates before they're needed. This means the rotating cache could
read from wrong positions. **Only affects models that use
`BatchRotatingKVCache`** (e.g., sliding window attention models).

---

## 25. Compiled Decode with mx.compile

**What we changed:** `EXO_COMPILE_DECODE=1` wraps the decode step in
`@mx.compile` with cache state as inputs/outputs.

**Why:** Eliminates ~9-11ms graph-building overhead per step.

**What could break:** `mx.compile` caches the Metal compute graph. If any
input shape changes between steps (e.g., dynamic KV cache growth, variable
batch size), the cached graph becomes invalid and produces wrong results.
We guard against this (disabled for KV quantization and logits processors),
but any new dynamic behavior in the decode path would silently produce
garbage. Also, compiled code produces lazy arrays — mixing with uncompiled
code that expects eager arrays causes "shape mismatch" or "array not
evaluated" errors. **This is why we disable compile when speculation is
active.**

---

## 26. _skip_lm_head on Non-Last PP Ranks

**What we changed:** `if getattr(self, "_skip_lm_head", False): return out`
in TextModel/Qwen3NextModel.

**Why:** Non-last ranks don't need logits. Saves ~500MB weight reads per step.

**What could break:** If ANY code path on a non-last rank needs logits (e.g.,
top-k sampling for draft speculation on rank 0), it gets raw hidden states
instead. The code must know to re-enable lm_head before using logits. Our
MTP module handles this correctly (calls `self.lm_head(out)` explicitly),
but any future code that calls `model(tokens)` expecting logits on rank 0
will get hidden states.

---

## 27. Weight Fusion in sanitize()

**What we changed:** Three fusion passes at load time: DeltaNet in_proj, MLP
gate_up, attention qkv. Concatenates quantized weight/scales/biases on
axis=0 (or axis=1 for expert weights).

**Why:** Fewer kernel dispatches per layer (~95us each on M4 Max).

**What could break:** If the concatenation axis is wrong for a particular
quantization format, the fused weight matrix is silently corrupt. All
outputs from that layer would be wrong. We've verified axis=0 for standard
`nn.Linear` and axis=1 for expert `SwitchLinear`, but a new quantization
scheme could use different layouts. Also, if upstream changes weight naming
conventions, the sanitize patterns (`"in_proj_qkv"`, `"gate_proj"`, etc.)
would stop matching and fusion would silently not happen — falling back to
unfused (slower but correct).

---

## 28. FusedSwitchGLU

**What we changed:** New `FusedSwitchGLU` in switch_layers.py with single
`gate_up_proj` SwitchLinear.

**Why:** Halves MoE dispatch cost for gate+up projections.

**What could break:** `gather_qmm` (the Metal kernel for MoE expert matmul)
must handle the doubled output dimension correctly. If `gather_qmm` has
assumptions about output size (e.g., alignment requirements), the fused
`2 * hidden_dims` output could violate them. We haven't seen issues, but
it's model-dependent.

---

## 29. Fused Quantized SDPA Fast Path in base.py

**What we changed:** Routes decode to
`mx.fast.quantized_scaled_dot_product_attention()` when conditions are met.

**Why:** Avoids materializing full attention matrix during decode.

**What could break:** The fast path checks `L <= 8` and `bits == 8` and
`D in (64, 96, 128, 256, 512)`. If a model has an unusual head dim not in
this list, it correctly falls through. But if a model has `bits=8` and a
supported dim but the quantization format doesn't match the kernel's
assumptions (e.g., symmetric instead of affine), the kernel produces wrong
attention output. The TODO comment notes "dead code with fp16 KV" — we
switched away from 8-bit KV, so this path rarely activates.

---

## 30. SuScaledRoPE/YarnRoPE Input Mutation

**What we changed:** Fork doesn't copy `x` before modifying `x[..., :self.dim]`.
Upstream added `x = x[...]` (copy) to prevent mutation.

**Why:** We didn't pick up upstream's fix (PR #1003).

**What could break:** If the input tensor is reused after the RoPE call, its
values have been modified in-place. In practice, MLX's graph evaluation
usually means tensors aren't reused, but if a model architecture branches
the same tensor through RoPE and another path, the other path sees corrupted
data.

---

## 31. Server LRUPromptCache — Different Trie Implementation

**What we changed:** Our own dict-based trie with `"cache"` sentinel keys and
`checkpoint: bool` API, vs upstream's `PromptTrie` with `"__value__"` keys
and `cache_type: str`.

**Why:** We built ours before upstream's refactor.

**What could break:** The two implementations may have different cache eviction
behavior. Our `CacheOrder` uses `_lru` + `_lru_checkpoints` deques;
upstream's uses per-type ordering (`assistant > user > system`). If the
eviction order matters for cache hit rates (it does — evicting recent user
messages before old system messages keeps the system prompt cached), our
implementation may have worse cache hit rates.

---

# EXO Changes

## 32. API Relocated to master/

**What we changed:** `src/exo/api/` → `src/exo/master/api.py` +
`src/exo/master/adapters/`. Types → `src/exo/shared/types/`.

**Why:** Workers need API types (e.g., `ChatCompletionRequest` for prompt
parsing). Upstream's layout had types in `exo.api.types` which created import
issues.

**What could break:** Any upstream PR that touches `src/exo/api/` will not
apply cleanly. Every import path changed. This is the single biggest source
of merge conflicts with upstream.

---

## 33. Bootstrap Peers Removed

**What we changed:** `--bootstrap-peers` and `--libp2p-port` removed from CLI,
Rust networking, and Python router.

**Why:** mDNS discovers all peers in <1s on the local Thunderbolt mesh.

**What could break:** If we ever need to connect to nodes outside the local
network (e.g., a cloud GPU for offload), we have no way to discover them.
We'd need to re-add bootstrap peers or use a different discovery mechanism.

---

## 34. Single-Dir Model Storage

**What we changed:** `EXO_MODELS_DIRS` (multi-dir with free-space checking) →
`EXO_MODELS_DIR` (single dir).

**Why:** Each Mac Studio has one 2TB SSD.

**What could break:** If a node runs out of disk space, there's no automatic
fallback to another directory. Upstream's multi-dir system would try the next
directory. We'd need to manually move models or expand storage.

---

## 35. Runner Crash = SystemExit(1)

**What we changed:** `RunnerFailed` raises `SystemExit(1)` instead of
attempting in-process recovery.

**Why:** GPU memory isn't released on runner failure. RDMA state becomes dirty.
Only clean process restart works.

**What could break:** Every runner failure kills the entire node process.
If the failure is transient (e.g., a single request with a bad prompt), the
node goes down for ~10 seconds during restart instead of just failing that
request. Also, any state held in the process (download progress, connection
state, pending requests) is lost.

---

## 36. In-Memory Event Log

**What we changed:** Disk-based msgpack+zstd event log → in-memory.

**Why:** We restart from scratch on failure (see #35), so persistence is
useless.

**What could break:** If the master crashes, all event history is lost. Any
pending downloads, runner status, etc. must be re-discovered on restart. This
is fine because we restart everything anyway, but if we ever want to do
graceful recovery, we'd need persistence back.

---

## 37. Custom Card Events Removed

**What we changed:** `CustomModelCardAdded`/`CustomModelCardDeleted` events
removed. Cards managed via filesystem.

**Why:** 3-node cluster with shared model directory doesn't need event-sourced
sync.

**What could break:** If two nodes simultaneously add/delete cards, filesystem
operations could race (partial writes, read-during-write). Event sourcing
provides serialization. In practice, card changes are rare manual operations,
not concurrent.

---

## 38. P2P File Server on Port 52416

**What we changed:** HTTP file server serving model files for peer downloads.

**Why:** Thunderbolt 5 is ~100x faster than HuggingFace downloads.

**What could break:** No authentication. Any machine on the Thunderbolt
network can download any model file. Path traversal protection exists, but
if it has a bug, arbitrary files under `EXO_MODELS_DIR` are exposed. Also,
the file server binds to `0.0.0.0`, not just the Thunderbolt interface —
if the machine is on a public network, model files are exposed.

---

## 39. _strip_volatile_headers in Claude Adapter

**What we changed:** Strips `x-anthropic-billing-header` from system prompts
before KV cache key computation.

**Why:** This header changes per-request, defeating prefix caching.

**What could break:** If the billing header ever contains information that
SHOULD affect generation (e.g., a feature flag that changes model behavior),
stripping it would cause incorrect cache hits — a request with one flag value
would reuse the cache from a request with a different flag value. Currently,
the billing header is purely metadata and doesn't affect generation.

---

## 40. Claude Errors as Text Content Blocks

**What we changed:** Errors emitted as `ClaudeTextBlock("Error: {msg}")` with
`stop_reason="end_turn"` instead of proper error events.

**Why:** Claude Code crashes on `_.input_tokens` when receiving error events.

**What could break:** The client sees a successful response containing error
text, not a proper error. Automated retry logic (which triggers on error
responses) won't fire. The user sees "Error: context window exceeded" as if
the model said it, which is confusing. Also, `input_tokens` and
`output_tokens` in the usage may be wrong for error responses.

---

## 41. Heartbeat Timeout Scaling

**What we changed:** Timeout = max(300s, prompt_length / 100). Shared-memory
heartbeat touched per prefill chunk and decode step.

**Why:** Fixed timeouts (60s prefill, 5s decode) killed legitimate long
prefills.

**What could break:** A stuck runner that makes no progress but is within the
timeout window won't be killed. For a 100K token prompt, the timeout is
1000s (~16 minutes). If the runner deadlocks during that prefill, the node
is unresponsive for 16 minutes before the supervisor kills it. The old 60s
timeout would have caught it sooner.

---

## 42. Deferred message_start in Claude SSE

**What we changed:** `message_start` SSE event deferred until first real token.

**Why:** Claude Code reads `input_tokens` from `message_start`. Emitting it
immediately gives `input_tokens=0`.

**What could break:** Claude Code may expect `message_start` immediately and
use it for UI feedback ("thinking..."). If there's a long prefill (30+
seconds), the client gets no SSE events at all during that time and may
time out or show a spinner with no feedback.

---

## 43. spawn Multiprocessing Start Method

**What we changed:** `set_start_method("spawn", force=True)` in `__main__.py`.

**Why:** `fork()` on macOS is unsafe with Metal/GPU state.

**What could break:** `spawn` is slower than `fork` (new process starts from
scratch instead of copying parent). Also, `spawn` requires all arguments to
child processes to be picklable. If any non-picklable object is passed to a
subprocess (e.g., a Metal buffer, a socket), it fails at spawn time with a
pickle error.

---

## 44. Qwen3.5 MTP Module Created Dynamically

**What we changed:** `self.mtp = None` in TextModel.__init__. Only allocated
when weights are available.

**Why:** Avoids 3.5GB of empty weights on nodes without MTP.

**What could break:** Code that checks `hasattr(model, "mtp")` would return
True (the attribute exists, it's just None). Code must check
`model.mtp is not None` instead. If any code path calls `model.mtp(...)`
without the None check, it crashes with `TypeError: 'NoneType' is not
callable`.

---

## 45. EXO_PP_LAYER_OFFSET

**What we changed:** Shifts layers from rank 0 to the last rank.

**Why:** Creates more idle time on rank 0 for speculation.

**What could break:** If the offset pushes too many layers to the last rank,
that rank becomes the bottleneck and decode speed drops. The offset must be
tuned per model — wrong values degrade throughput instead of improving it.
No validation prevents setting an offset larger than total layers.

---

## 46. Prefill Chunking with Forced Eval

**What we changed:** `mx.eval()` forced after each prefill chunk.

**Why:** Without it, MLX accumulates the entire computation graph lazily,
causing OOM before any computation runs.

**What could break:** Forced eval adds synchronization points. Each eval
waits for the GPU to finish the current chunk before starting the next.
This eliminates pipelining between chunks — upstream's lazy approach can
overlap CPU graph building with GPU execution. For very long prefills, our
approach is slower but doesn't OOM.

---

## 47. _truncate_prompt_tokens

**What we changed:** Truncates prompts exceeding max context by keeping 25%
start + 75% end.

**Why:** Prevents OOM on oversized prompts.

**What could break:** The truncation destroys context in the middle of the
prompt. If critical information is in the truncated region, the model's
output quality degrades. The 25/75 split is arbitrary — there's no guarantee
it preserves the most important context. Also, the truncation happens
silently with a log message — the user doesn't know their prompt was
modified.
