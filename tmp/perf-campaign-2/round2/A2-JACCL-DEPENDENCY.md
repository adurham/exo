# A2 — jaccl collective dependency: where the local wait is, and what the fence really protects

Read-only source archaeology. All line numbers are from the **deployed build**:
`/Users/adam.durham/repos/exo/mlx` @ `e40a416b2` (the exo submodule), plus
`/Users/adam.durham/repos/exo/mlx-lm` and `/Users/adam.durham/repos/exo/src`.
No files outside this document were modified. Nothing was run.

---

## Q2 — Does `communication_stream` run concurrently with the GPU compute stream on jaccl?

**Answer.** No — not in the sense that matters. `communication_stream` on the jaccl
backend is a **CPU** stream (`new_stream(Device::cpu)`), and there is no GPU
implementation of the collective at all (`AllReduce::eval_gpu` throws). The collective is a
plain synchronous C++ call executed FIFO on that CPU stream's single worker thread. Because
the collective's input `y` is produced on the GPU stream, MLX's `eval_impl` classifies the
edge as cross-stream and inserts a `Fence` wait *on the comm stream* before dispatching the
collective — and on a CPU stream `Fence::wait` degrades to a **host-side busy spin**
(`while (cpu_value < count) __dsb(0xF);`) enqueued immediately ahead of the jaccl lambda on
that same worker thread. That spin is the local drain. The producer→collective relationship
is a **TRUE DATA DEPENDENCY** (jaccl reads `y`'s bytes out of host memory with `data<char>()`,
so the GPU must have written them), **but the implemented wait is materially broader than
that dependency** in three independent ways, listed under "slack" below.

### Evidence

1. `mlx/distributed/ops.cpp:34` — `all_sum` computes `auto stream = detail::communication_stream(group, s);` and constructs the `AllReduce` primitive on it (`:36-40`).
2. `mlx/distributed/distributed.cpp:17-18` — `detail::communication_stream` forwards to `group.raw_group()->communication_stream(s)`.
3. `mlx/distributed/jaccl/jaccl.cpp:88` — **DECISIVE**: `communication_stream_(new_stream(Device::cpu))`. One CPU stream, created once at group construction. `:90-95` returns it unconditionally and *ignores* the caller-supplied stream (`(void)s;`). The ctor comment (`:65-87`) states the reason: pinning one stream forces one FIFO dispatch queue per rank so both ranks' post_send/post_recv interleavings match.
4. `mlx/backend/metal/distributed.cpp:17-19` — `AllReduce::eval_gpu` unconditionally `throw std::runtime_error("[AllReduce::eval_gpu] has no GPU implementation.")`. Same for `AllGather`/`Send`/`Recv`/`ReduceScatter` (`:21-36`). **There is no GPU path for any jaccl collective.** This file is on the path only as the throw-stub.
5. `mlx/backend/cpu/distributed.cpp:20-56` — the real `AllReduce::eval_cpu`. `donate_or_copy` (`:26-39`) donates the input buffer to the output when `in.is_donatable()` (`out.copy_shared_buffer(in)`, `:29`), then calls `distributed::detail::all_sum(group(), in, outputs[0], stream())`.
6. `mlx/distributed/jaccl/jaccl.cpp:105-116` — `JACCLGroup::all_sum` takes raw host pointers (`input.data<char>()`, `output.data<char>()`), grabs `cpu::get_command_encoder(stream)`, and `encoder.dispatch(lambda -> group_->all_sum(...))`.
7. `mlx/backend/cpu/encoder.h:43-57` — `dispatch` is `scheduler::enqueue(stream_, task)`: the lambda is queued onto that stream's worker thread and runs FIFO. Not concurrent with anything else already queued on that stream.
8. `mlx/transforms.cpp:159-168` — in `eval_impl`'s DFS, `if (a.primitive().stream() != in.primitive().stream())` registers `in` (i.e. `y`) in `needs_fence`, with `device_switch = true` because gpu != cpu. This fires on every jaccl collective by construction.
9. `mlx/transforms.cpp:263-268` — in the tape loop, **before** `cpu::eval(arr)` at `:282`, it runs `fences[gpu_stream_index].wait(comm_stream, y)`.
10. `mlx/backend/metal/fence.cpp:58-77` — **THE WAIT.** Because the waiting stream is a CPU stream: `scheduler::enqueue(stream, [...]{ while (f.cpu_value()->load(std::memory_order_seq_cst) < count) { __dsb(0xF); } });` — a **host-side busy spin with a full-system data barrier per iteration**, queued on the comm stream's worker thread *immediately ahead of* the collective lambda from evidence 6/7. It is a host thread blocking on a GPU-written memory location, not a GPU-waits-on-GPU edge and not lazy-graph materialization.
11. `src/exo/worker/runner/bootstrap.py:296-302` — the runner force-sets `MLX_METAL_FAST_SYNCH=1` unless `EXO_FAST_SYNCH=false`, so `FenceImpl::use_fast` is true (`mlx/backend/metal/fence.cpp:14-26`, `mlx/utils.h:190-193`) and evidence 10's spin is the **live** path. The `!use_fast` alternative (`fence.cpp:53-56`) is `Event::wait`, which on a CPU stream (`mlx/backend/metal/event.cpp:200-210` → `:111-167`) is a 2000-iteration spin followed by 50 µs sleep-polling — also a host block, just a cheaper-burning one.
12. `mlx/backend/metal/fence.cpp:143-154` — the matching signal side: `Fence::update` on the GPU stream encodes `compute_encoder.barrier()` (device-wide `memoryBarrier(BarrierScopeBuffers)`, `mlx/backend/metal/device.cpp:571-573`) and then a `fence_update` kernel, appended to the **current, still-uncommitted** command buffer. Called from `mlx/transforms.cpp:301-314` (`maybe_update_fence`) right after `gpu::eval(producer)`.
13. `mlx/backend/metal/eval.cpp:152-174` — `gpu::eval` only calls `encoder.commit()` when `encoder.needs_commit()`; otherwise it just appends a completion handler. `needs_commit()` = `buffer_ops_ > max_ops || (buffer_sizes_>>20) > max_mb` (`mlx/backend/metal/device.cpp:662-665`), with exo running `MLX_MAX_OPS_PER_BUFFER=200`, `MLX_MAX_MB_PER_BUFFER=200` (`start_cluster.sh:481-482`).
14. `mlx/transforms.cpp:325-332` — otherwise the buffer is only committed by `gpu::finalize(s)` at the **end of `eval_impl`**. So the `fence_update` kernel from evidence 12 can still be sitting un-submitted at the moment the comm thread starts spinning for it in evidence 10.
15. `mlx/distributed/jaccl/lib/jaccl/mesh.cpp:1902-1934` — `MeshGroup::all_sum` takes `collective_mutex_`, runs `all_reduce<T>` inline, and returns. Fully synchronous; nothing is left in flight. So the jaccl side contributes only its own transport time (independently measured at 36 µs median).
16. `mlx-lm/mlx_lm/models/deepseek_v4.py:3095-3151` — the call site: blocking `mx.eval(y)` (`:3150`) unless the async gate passes, in which case `mx.async_eval(y)` (`:3130`). `EXO_DSV4_FENCE_ASYNC=1` is the production default (`start_cluster.sh:1758`). **The async fence does not remove any of the above**: `async_eval` still runs the same `eval_impl` (`mlx/transforms.cpp:337-349`), so the comm-stream fence spin and the end-of-eval commit both still happen; only the *Python main thread's* block (`Event::wait`) is elided.

### True dependency vs. slack

**True and unavoidable:** the collective memcpy-reads `y` from host memory (evidence 6), so
the GPU must have finished writing `y`. Cross-layer pipelining is the only structural way
around it.

**Slack — the wait is broader than that dependency (all three are cheap to reason about, none is verified by measurement here):**

- **(a) Fence granularity is per-stream, not per-array.** `FenceImpl` holds a single monotonic
  `count` per stream (`fence.cpp:35,102`) and its update is preceded by a device-wide
  `memoryBarrier(BarrierScopeBuffers)` at the tail of the command buffer (`fence.cpp:144`,
  `device.cpp:571-573`). The spin therefore releases only when the whole in-order GPU queue
  up to that point has retired — not merely `y`'s producing kernels.
- **(b) The signal may not even be submitted yet.** Evidence 12→14: the `fence_update` kernel
  is appended to an uncommitted buffer, and (absent a `needs_commit()` trip) is only committed
  by `gpu::finalize` after the entire tape has been issued. The comm worker can begin spinning
  before the GPU has been handed any of the work it is spinning on. At the per-layer
  `mx.eval(y)` granularity this means: encode whole layer → commit → GPU runs whole layer →
  fence_update → spin releases → 36 µs of wire → done. That round trip, not the wire, is the
  shape of a ~1400 µs/layer local cost.
- **(c) The spin burns the very thread that must next run the collective.** `Fence::wait`'s
  lambda and the jaccl lambda are queued on the *same* CPU stream worker (evidence 7, 10), so
  the wait cannot be overlapped with the transport, with the peer's arrival, or with anything
  else; and `__dsb(0xF)` every iteration is a full-system barrier on a core that
  `MLX_STREAM_QOS` machinery elsewhere is already fighting to keep scheduled
  (`mlx/scheduler.h:51-77`).

**CONFIDENCE: HIGH** on the mechanism (CPU comm stream, no GPU collective, host-side spin at
`fence.cpp:58-77`, live because `MLX_METAL_FAST_SYNCH=1`) — every link is a direct code read
on the deployed build. **MEDIUM** on the slack items being the dominant share of the measured
~1400 µs, because that is an attribution claim I did not measure.

**Could not determine:**
- Whether a DSv4 layer's op count actually trips `needs_commit()` (>200 ops) mid-tape. If it
  does, slack (b) shrinks toward the tail of the layer; if it doesn't, the whole layer is
  submitted in one shot at `gpu::finalize`. Answerable with one `MLX_SIGNAL_PROBE=1`-style
  read of `encoder.buffer_ops()` (`device.h:100-102`) or an `EXO_CMDBUF_RING_DIAG=1` run.
- The actual wall-clock split between (a) waiting for GPU work that genuinely produces `y` and
  (a')-waiting for unrelated queued GPU work retiring behind the same fence counter.
- Whether the `Event::wait` path (`MLX_METAL_FAST_SYNCH=0`) has materially different cost —
  its 2000-spin + 50 µs sleep granularity (`event.cpp:99-102,166`) would quantize a sub-50 µs
  wait upward, which is a plausible but unverified separate penalty.

---

## Q3 — Why is the `mx.eval(y)` fence load-bearing for cross-rank bit-equivalence?

**Answer.** It is **not**, and the historical claim is wrong. **H4 — something else, and
specifically: the 2026-06-26 failure was a plain algebra bug, not a fence/ordering/determinism
problem.** exo shards `shared_experts.down_proj` **sharded-to-all**, and `shard_inplace` only
rewrites the parameter tree — it inserts no collective (its own docstring says the module must
"natively support" the communication). So `shared_out` is a *partial sum* on each rank, exactly
like `switch_mlp`'s output `y`. The single `all_sum` **after** `_moe_post_combine` is what
reduces **both** terms at once. The reverted patch (`mlx-lm@9bc2206`) moved `all_sum` *before*
the combine, so the shared-expert partial was never reduced on any of 43 layers — each rank
kept only its own half of the shared-expert contribution, and the two ranks legitimately
diverged. Near-zero-quality output is the expected outcome of that, with or without a fence.
I **rule out H1, H2 and H3 with code cites** below. The fence retains one *legitimate but
different* role — pinning cross-rank collective dispatch **order** — whose failure mode is a
loud `DESYNC` throw or a hang, not silently-wrong numbers.

### Evidence

1. `mlx-lm/mlx_lm/models/deepseek_v4.py:3056` / `:3072` / `:3076` — the working order is
   `shared_out = self.shared_experts(x)` → `y = _moe_post_combine(y, scores, shared_out)` →
   `y = mx.distributed.all_sum(y, ...)`. `_moe_post_combine` is
   `(y * scores[...,None]).sum(-2) + shared_out` (`:1144-1156`).
2. `mlx-lm@9bc2206` (diff read via `git show`) — the reverted patch reorders to
   `y_reduced = all_sum(y)` → `shared_out = self.shared_experts(x)` →
   `y = _moe_post_combine(y_reduced, scores, shared_out)` → `mx.eval(y)`. **`shared_out` is
   added after the only all_sum and is never reduced.**
3. `src/exo/worker/engines/mlx/auto_parallel.py:1127-1129` — `shared_experts.gate_proj` and
   `up_proj` are all-to-sharded; **`shared_experts.down_proj` is `sharded_to_all_...in_place`**.
   Same pattern for `switch_mlp` at `:1130-1132`.
4. `src/exo/worker/engines/mlx/auto_parallel.py:757-768` — `_sharded_to_all` returns axis `-1`
   (shard the input/contraction dim) and, for a bias, does `weight /= n` before returning
   `None`. **Dividing the bias by the group size is only correct if a later all_sum re-adds it
   n times** — direct proof that a sharded-to-all output is a partial sum awaiting reduction.
5. `mlx/python/mlx/nn/layers/distributed.py:118-155` — `shard_inplace` ends at
   `module.update(_shard(module.parameters(), sharding, group))`. Its docstring (`:133-136`)
   says explicitly: *"The module doesn't change so in order for distributed communication to
   happen the module needs to natively support it."* **No collective is inserted.** The one
   `all_sum` in `DeepseekV4MoE.__call__` is the entire reduction for the MoE block.
6. `mlx/python/mlx/nn/layers/distributed.py:15-27` — the other candidate collective,
   `sum_gradients(group)`, is an `mx.custom_function` that returns `x` unchanged in the forward
   pass; only its `.vjp` calls `all_sum`. So `x = sum_gradients(self.sharding_group)(x)` at
   `deepseek_v4.py:2959` reduces **nothing** at inference. It cannot be the missing reduce.
7. `mlx-lm/mlx_lm/models/deepseek_v4.py:1686-1687` — "DSv4 REPLICATES attention on every rank
   (MoE-only sharding)". So `x`, and therefore `inds`/`scores` from the unsharded gate, are
   already identical on both ranks; the only rank-divergent quantities entering the combine are
   the two sharded partials. Correct order ⇒ `all_sum(r_i·scores + s_i)` = `(r₀+r₁)·scores +
   (s₀+s₁)` ✔. Reverted order ⇒ each rank holds `(r₀+r₁)·scores + s_i` ✘ — wrong magnitude *and*
   rank-divergent.
8. **H2 (nondeterministic reduction order) — RULED OUT.** In every jaccl all-reduce path the
   peer's bytes are memcpy'd into a fixed-offset assembly buffer indexed by an on-wire sequence
   number, and `reduce_op` is applied **exactly once at the end** over the fully-assembled
   buffer: `mesh_impl.h:1137` (v2 optimistic), `:1423` (non-v2 reliable ARQ), `:1495`
   (`ack_all_reduce_small`). Assembly sites: `:1254-1275` (`consume_recv`, header-seq indexed),
   `:877-887` (v2, `hdr.seq`-indexed), `:760-773` (v2 lookahead stash, also seq-indexed).
   Arrival order, chunking, duplicate frames and retransmits cannot change the arithmetic —
   `got[seq]` guards make re-delivery idempotent. `reduction_ops.h:11-18` / `:76-86` is a plain
   elementwise `out[i] = out[i] + in[i]` with local operand fixed in `out_ptr`. A fence cannot
   be masking a bit-pattern nondeterminism that the code structurally cannot have.
9. **H3 (buffer reuse / in-flight aliasing) — RULED OUT.** `MeshGroup::all_sum`
   (`mesh.cpp:1902-1934`) holds `collective_mutex_` and returns only after the transport
   completes; `mesh_impl.h:1094-1098` explicitly caps posted recvs so *zero* recv WRs remain
   outstanding at completion. Nothing of this collective is still touching the buffer when the
   array's lifetime resumes, so the MLX allocator cannot hand it out under an in-flight transfer.
10. **H1 (next layer reads `y` before the collective lands) — RULED OUT as an intra-process
    concern.** MLX already inserts the dependency edge automatically and symmetrically: the same
    `needs_fence` machinery at `mlx/transforms.cpp:159-168` / `:263-277` that gates the
    collective on the GPU producer also gates any GPU consumer on the collective, via
    `Fence::update` on the CPU stream (`mlx/backend/metal/fence.cpp:110-125`) paired with
    `Fence::wait` on the GPU stream (`:80-98`, which additionally
    `compute_encoder.register_output_array(x)` so no dependent kernel can start early). A
    user-level `mx.eval` is not what creates that edge.
11. **The fence's one real, different job.** `mesh_impl.h:3573-3592`
    (`confirmed_coord_barrier`) has both ranks exchange their `call_id` and `throw` a logged
    `CONFIRMED BARRIER DESYNC` if they disagree; `call_id` is a per-process monotonic counter
    (`mesh.h:368-371`). So cross-rank collective **order** genuinely matters — and the
    `jaccl.cpp:65-87` ctor comment says the same thing about post_send/post_recv interleaving
    on a UC QP. But every failure in that class is loud (a throw) or a hang, **not** silently
    divergent numbers. Note also `mlx-lm/mlx_lm/models/deepseek_v4.py:3115-3130`: production
    runs `mx.async_eval(y)`, which does **not** block the main thread — if per-layer *blocking*
    were what held the ranks in numeric lockstep, the production default would already be
    broken. It isn't.
12. **In-place / donation, for completeness (relevant to H4's "the fence makes the in-place
    write safe" variant — also ruled out).** `mlx/backend/cpu/distributed.cpp:26-33`:
    when the input is row-contiguous and donatable, `out.copy_shared_buffer(in)`, so
    `in_ptr == out_ptr` and jaccl's `if (in_ptr != out_ptr) memcpy` (`mesh_impl.h:696-698`,
    `:1161-1163`, `:1475-1477`) is skipped and the reduce lands **in place** in the donated
    buffer. `is_donatable()` (`mlx/array.h:316-318`) requires refcount 1, i.e. no other live
    reader. Combined with evidence 9 (synchronous, nothing in flight), the in-place write needs
    no user-level fence to be safe.

**CONFIDENCE: HIGH** that H1/H2/H3 are out and that the 2026-06-26 garbage output is fully
explained by the unreduced `shared_experts` partial — evidence 3+4+5 is a direct, three-file
chain showing the sharded-to-all output *is* a partial sum with no collective of its own, and
evidence 2 shows the patch moved the only reduction ahead of where that partial is added.
**MEDIUM-HIGH** that the fence has *no* residual numeric role: I am inferring from
`async_eval` being the production default (evidence 11) rather than from a test.

**Could not determine:**
- Whether the OPT-7 regression (fence gated on `_fence_every_n`, −23% prefill) is purely the
  graph-accumulation cost its comment claims (`deepseek_v4.py:3088-3094`) or also involves
  cross-rank dispatch drift. That one is a *performance* revert, not a correctness one, and I
  did not trace it.
- Whether some path other than `DeepseekV4MoE.__call__` reduces `shared_out` under a
  configuration I did not read (e.g. an MTP-specific re-shard). I checked
  `auto_parallel.py:1180-1182`, which applies the identical sharded-to-all pattern to the MTP
  block, so the same reasoning holds there — but I did not exhaustively enumerate every
  sharding strategy class.
- Whether the fence contributes to keeping `call_id` streams aligned in some regime where
  `async_eval` is disarmed (prefill, B>1). Evidence 11 says the failure mode there would be a
  loud DESYNC, but I did not confirm the arming gates never disagree between ranks.

### The cheap experiment that would settle Q3 outright (NOT run)

Re-apply `mlx-lm@9bc2206`'s reorder **but add a second `all_sum` for the shared term** — i.e.
`y_red = all_sum(y)`; `sh_red = all_sum(self.shared_experts(x))`; `y = (y_red*scores).sum(-2) +
sh_red`; fence after. If output quality is restored, the 2026-06-26 failure was the algebra bug
(H4) and the fence's bit-equivalence justification is dead. If it still degenerates, something
ordering-related survives and H1 deserves a second look. One TP=2 needle run; no MLX rebuild
required (mlx-lm-only change).

---

## One-line corrections to the record

- `docs/dsv4-decode-stall-2026-06-26.md:88-89` says `all_sum` runs on a comm stream "separate
  from the GPU compute stream. The overlap *primitive* exists." The stream is separate, but it
  is a **CPU** stream with **no GPU collective implementation** (`jaccl.cpp:88`,
  `metal/distributed.cpp:17-19`) — the "overlap primitive" as imagined (a GPU-side async
  collective) does not exist, and the cross-stream edge is implemented as a host-thread busy
  spin (`metal/fence.cpp:58-77`).
- `docs/dsv4-decode-stall-2026-06-26.md:90-94` ("fence required for cross-rank bit-equiv;
  without it GPU stragglers let the two ranks dispatch the next MoE layer with subtly different
  inputs") is not supported by the code and its cited evidence has a simpler explanation — see
  Q3.
