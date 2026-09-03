# CAMPAIGN 2, ROUND 2 — MECHANISM: the five MLX-layer questions, answered

**Date:** 2026-09-03
**Scope:** Phase A (mlx source archaeology, zero cluster cost)
**mlx submodule:** `~/repos/exo/mlx` @ `e40a416b2` (the DEPLOYED build — *not* `~/repos/mlx`, a known trap)
**mlx-lm submodule:** `~/repos/exo/mlx-lm` @ `37260bbd6`

Every cite below was re-verified by the PM against the primary source, not accepted
from a worker summary. Cites marked **[PM-VERIFIED]** were read line-by-line a second
time because a decision turns on them.

---

## THE ONE-PARAGRAPH VERDICT

**The stall is (a) — a TRUE DEPENDENCY — and the fix that verdict (a) prescribes is
structurally blocked.** MLX has *no GPU collective at all*: `AllReduce::eval_gpu`
unconditionally throws, and jaccl's collective stream is a **CPU** stream. So every
per-layer `all_sum` requires a real GPU→CPU handoff — a CPU thread must observe that
the GPU finished producing `y` before it can hand the buffer to RDMA. That handoff is
implemented as a whole-payload `input_coherent` kernel plus a device-wide
`memoryBarrier` on the GPU side and a **host busy-spin** on the CPU side. This is a
genuine data dependency on the work that produces the collective's own input; it cannot
be overlapped with that work by any fencing trick. Verdict **(b) is ruled out** — MLX
already inserts the cross-stream edge automatically, and the GPU-side wait branch is
*structurally unreachable* for a GPU→CPU edge, so a stream-scoped event has nothing to
bite on. Verdict **(c) is ruled out as the dominant term** — the stream-boundary cost is
payload-proportional (~7 GB/s) and at decode's real 32 KiB payload predicts ~5–100 µs
against a measured ~1400 µs, i.e. **at most ~7%**. And the escape hatch that (a) implies —
overlapping layer N's collective with layer N+1's attention — is blocked because
`hc_expand` folds the MoE output into *all* hyper-connection streams, so layer N+1
depends fully on layer N's `all_sum` result, and no independent GPU work exists anywhere
in the forward to fill the gap. **The correct outcome is a structural close plus one
scoped, genuinely-untested root-cause fix (a GPU-resident collective), not a fourth
rearrangement.**

---

## Q1 — What does `mx.eval(y)` on an `all_sum` output actually drain?

**Answer: NOT a whole-device barrier, and NOT all streams — but the real mechanism is
worse than "the wrong stream got synchronized."**

`eval_impl` collects `open_streams` **only** from arrays actually on the tape
(`mlx/transforms.cpp:242,248`) — it does not enumerate all streams or all devices. The
host blocks on exactly one `MTLSharedEvent`. `waitUntilCompleted` exists at
`mlx/backend/metal/device.cpp:721` but **`eval()` never reaches it** — only
`mx.synchronize()` does.

The load-bearing part is what happens *because the edge is GPU→CPU*:

1. `mlx/distributed/jaccl/jaccl.cpp:88` — **[PM-VERIFIED]** the collective stream is
   constructed as `communication_stream_(new_stream(Device::cpu))`. A **CPU** stream.
   The in-source comment confirms this is deliberate ("NOT `default_stream()` (which is
   thread_local and produces different streams per caller thread)").
2. `mlx/backend/metal/distributed.cpp:17-19` — **[PM-VERIFIED]**
   `AllReduce::eval_gpu` is `throw std::runtime_error("[AllReduce::eval_gpu] has no GPU
   implementation.")`. **There is no GPU collective in this MLX at all.**
3. `mlx/transforms.cpp:159-168` — **[PM-VERIFIED]** the graph walk detects
   `a.primitive().stream() != in.primitive().stream()`, sets
   `device_switch = (…device != …device)`, and records it in `needs_fence`. GPU→CPU sets
   `device_switch=true`, which selects the `Fence` path.
4. `mlx/backend/metal/fence.cpp:128-158` — **[PM-VERIFIED]** the GPU producer side, when
   `cross_device`, dispatches an **`input_coherent` kernel over the entire payload**
   (`nthreads = x.data_size() * x.itemsize() / 4`), then `compute_encoder.barrier()`,
   then a `fence_update` kernel.
5. `mlx/backend/metal/fence.cpp:58-77` — **[PM-VERIFIED]** the CPU consumer side, in
   fast mode, is a **host busy-spin**: `while (f.cpu_value()->load(seq_cst) < count) {
   __dsb(0xF); }`, enqueued onto the CPU stream's worker thread. Production forces this
   path (`MLX_METAL_FAST_SYNCH=1`).

**Scope of the drain:** set by the device-wide `memoryBarrier`, *not* by `y`'s dependency
subgraph. The command buffer is not committed until `gpu::finalize` at end-of-eval, so the
comm thread can begin spinning before the GPU has even been handed the work.

**`async_eval` drains identically.** Same `eval_impl` body; the only difference is the
host does not block. `EXO_DSV4_FENCE_ASYNC=1` removes the *host* block — it removes
neither the GPU-side barrier nor the comm-thread spin.

---

## Q2 — Does `communication_stream` run concurrently with GPU compute on jaccl?

**Answer: NO. It is serialized by a real GPU→CPU data dependency.**

Because the collective runs on a **CPU** stream (Q1 cite 1) and there is **no GPU
implementation** (Q1 cite 2), the collective is a CPU worker thread performing an RDMA
transfer that must *read `y` out of host-visible memory*. It therefore cannot start
until the GPU has finished producing `y` and made it coherent. That is the wait, and
`fence.cpp:58-77` is the exact line where the local rank burns time.

**This is the ~1400 µs "local drain" round 1 measured.** It matches round 1's three
independent structural arguments: the wait is local (transport is 36 µs = 2.6%), the
waits are cross-rank symmetric, and a no-peer control reproduces the penalty with no
peer at all.

**Is it a true dependency or is there slack?** It is a **TRUE dependency** — jaccl
genuinely reads `y`. But the *wait as implemented is broader than the dependency*, in
three specific ways worth recording:
- the fence counter sits behind a **device-wide** `memoryBarrier` (`fence.cpp:144`), so
  it drains the whole GPU queue, not just `y`'s producers;
- the `fence_update` kernel sits in an **uncommitted** command buffer until
  `gpu::finalize`, so the comm thread can spin before the GPU has the work;
- the spin burns **the collective's own worker thread** — the thread that must next run
  the transport.

That slack is real, but per Q5 it is not big enough at decode payload to be the headline.

---

## Q3 — Why is the fence load-bearing for cross-rank bit-equivalence?

**Answer: it is NOT. The 2026-06-26 attempt-3 failure was an ALGEBRA bug, not a fence
or race phenomenon. This corrects the record.** (Hypothesis **H4**.)

The mechanism, verified at the primary source:

1. `src/exo/worker/engines/mlx/auto_parallel.py:1128` — **[PM-VERIFIED]**
   `self.sharded_to_all_linear_in_place(layer.ffn.shared_experts.down_proj)`.
   The shared experts' `down_proj` is sharded **sharded-to-all**.
2. `mlx/python/mlx/nn/layers/distributed.py` `shard_inplace` — **[PM-VERIFIED]** it only
   does `module.update(_shard(module.parameters(), sharding, group))`. It inserts **no
   collective**. Its own docstring says so: *"The module doesn't change so in order for
   distributed communication to happen the module needs to natively support it and for
   it to be enabled."*
3. `src/exo/worker/engines/mlx/auto_parallel.py:~758-762` — **[PM-VERIFIED]** the
   `_sharded_to_all` callable divides **biases** by the group size (`weight /= n`). That
   is only arithmetically correct if a later `all_sum` re-adds them `n` times. This is
   independent corroboration that a sharded-to-all output is a **partial sum awaiting
   reduction**.

Therefore `shared_out` is a **partial sum**, exactly like the routed `y`. The single
`all_sum` placed *after* `_moe_post_combine` is what reduces **both**. Confirmed at the
call site: `mlx-lm/mlx_lm/models/deepseek_v4.py:3072` computes
`y = finalize(_moe_post_combine(y, scores, shared_out))` and only then, at `:3076`, runs
`y = mx.distributed.all_sum(y, group=self.sharding_group)`.

**Attempt 3 (mlx-lm `9bc2206`) moved the `all_sum` BEFORE the combine.** That left the
shared-expert partial sum **never reduced, on all 43 layers**. Near-zero garbage output
is exactly what that predicts. It is a plain algebra error.

The competing hypotheses are ruled out with cites:
- **H1 (data race on `y`)** — OUT intra-process. MLX inserts the cross-stream edge
  itself (`transforms.cpp:159-168`, `:263-277`).
- **H2 (nondeterministic reduction order)** — OUT. Every jaccl path assembles peer bytes
  into a **sequence-indexed** buffer and applies `reduce_op` **exactly once at the end**
  (`jaccl/mesh_impl.h:1137`, `:1423`, `:1495`). Arrival order and retransmits
  structurally cannot change the bits.
- **H3 (buffer reuse / aliasing)** — OUT. `mesh.cpp:1902-1934` is synchronous under
  `collective_mutex_`; `mesh_impl.h:1094-1098` leaves zero recv work-requests
  outstanding.

**Confidence, stated honestly.** That attempt 3 failed for algebra reasons is **HIGH**
confidence — the sharded-to-all/`/= n` evidence is decisive and self-consistent. That the
fence has **zero** residual numeric role is only **MEDIUM-HIGH**: it is an argument from
absence, and the production comment at `deepseek_v4.py:3077-3092` asserts a *cross-rank
lockstep* purpose that MLX's intra-process edge does not provide. Two facts push against
the comment: the fence does have a *different* real job (pinning cross-rank collective
call order — but that fails **loudly** with a `DESYNC` throw at `mesh_impl.h:3573-3592`,
not as silent wrong numbers), and **production already runs `mx.async_eval`**
(`EXO_DSV4_FENCE_ASYNC=1`), so a per-layer *blocking* fence is already not what holds the
ranks in numeric lockstep.

**Falsifiable test (specified, deliberately NOT run — see REPORT.md).**

---

## Q4 — Is there a fence-one-stream-without-blocking-the-other primitive?

**Answer: the primitive exists, MLX already uses it automatically, and it CANNOT help
here.**

- **`Event` is the real one-way primitive.** It wraps `MTLSharedEvent`
  (`mlx/backend/metal/event.h:30`, `event.cpp:45`), and `Event::wait(gpu_stream)` reaches
  `buffer_->encodeWait(...)` (`device.cpp:654-659`) — a genuine **GPU-side** wait where
  the host never blocks. On a **CPU** stream it degrades to host work
  (`event.cpp:200-210`).
- **`Fence`** has both shapes: a GPU consumer gets a `fence_wait` kernel with no host
  block (`fence.cpp:80-97`); a **CPU consumer gets the host busy-spin**
  (`fence.cpp:58-77`).
- **Neither is exposed to Python.** Zero binding hits in `python/src/`; Python has only
  `mx.eval` / `mx.async_eval` / `mx.synchronize`.
- **MLX already inserts the cross-stream dependency edge automatically**
  (`transforms.cpp:159-168` detects, `:263-277` materializes). So `mx.eval(y)` is
  **redundant for data ordering**.

**The killer constraint:** for this edge the consumer is a **CPU** stream, so the
GPU-side `fence_wait` branch is **structurally unreachable**. A CPU thread must observe
GPU completion before it can hand the buffer to RDMA. **No amount of event plumbing
removes that** — only a GPU-resident collective, or overlap with genuinely unrelated GPU
work, would.

This is precisely why the hypothesis behind verdict (b) — "the 06-26 attempt failed
because it used `mx.eval` (device drain) where an event (stream edge) was needed" — is
**false**. The event already exists, MLX already inserts it, and it cannot span a
GPU→CPU handoff.

---

## Q5 — What is the 2.66x stream-boundary penalty?

**Answer: a payload-proportional coherency cost (~7 GB/s), NOT a fixed per-crossing
commit fee — and at decode's real payload it is small.**

- The 2.66x comes from `docs/phase0a-allsum-boundary-decomposition-2026-08-20.md:61-62`
  — **[PM-VERIFIED]** `layered_cpuop 63.232ms` vs `layered_gpuop 23.755ms`, a
  **whole-loop wall-clock ratio** on a plain `mx.abs` with **zero wire bytes and no
  peer**, at a **16.8 MB** payload.
- Its own payload sweep (same doc, lines 82-91) is **linear at ~7 GB/s**, with the doc
  stating explicitly that *"A fixed drain/refill latency would be flat across this
  sweep."* **[PM-VERIFIED]** — this **falsifies** the "86 crossings × a fixed commit
  cost" framing in the round-2 brief.
- **The mechanism is now identified**, and it is the same code Q1 found: the
  payload-proportional term is the **`input_coherent` kernel sweeping the whole buffer**
  (`fence.cpp:131-140`). Q5's independent measurement and Q1's source reading converge on
  the same line. That convergence is the strongest single result of this round.
- No MAX_OPS/MB threshold fires at a stream boundary; that cap (`device.cpp:757-781`,
  50 ops / 50 MB) only batches ops *within* one stream. `eval_impl` does force-commit
  every touched stream via `gpu::finalize()`, so a per-layer eval pattern does incur
  86 forced commits/forward — but the source gives **no constant** for a bare commit's
  cost, so that remains a structural argument only, deliberately unquantified.

**Extrapolated to the real decode payload of 32 KiB:** pure-linear = 4.68 µs;
floor+linear (using the ~97 µs fixed floor the sweep's sub-linear point implies, which
independently matches round 1's measured 96.8 µs ± 4.6 CPU→GPU dispatch latency) =
**~100 µs**. Against a measured ~1400 µs/layer that is **at most ~7%**.

**Verdict (c) is therefore real but NOT the dominant term at decode scale.** The 2.66x
was measured in a prefill-scale (MB) regime and does not transfer to a decode-scale
(KiB) one.

---

## The (a)/(b)/(c) verdict

| Verdict | Status | Deciding evidence |
|---|---|---|
| **(a) true dependency, pipelinable only across layers** | **THIS ONE — and its prescribed fix is blocked** | `jaccl.cpp:88` (CPU stream) + `metal/distributed.cpp:17-19` (`eval_gpu` throws) ⇒ mandatory GPU→CPU handoff. But `hc_expand` blocks cross-layer overlap (see below). |
| (b) over-broad fence a stream-scoped event would fix | **RULED OUT** | `Event` exists and MLX already inserts the edge automatically (`transforms.cpp:159-168`, `:263-277`); the GPU-side wait branch is unreachable for a GPU→CPU edge (`fence.cpp:58-77`). |
| (c) command-buffer churn at stream boundaries | **RULED OUT as dominant** (real, ≤~7%) | Cost is payload-linear ~7 GB/s, not fixed-per-crossing (phase0a:82-91); predicts ~5–100 µs at 32 KiB vs ~1400 µs measured. |

### Why (a)'s own fix — cross-layer pipelining — is also blocked

**Verdict P1: structurally impossible without changing the model's math.**

1. `mlx-lm/mlx_lm/models/deepseek_v4.py:3076` — the `all_sum`; `:3152` returns that `y`.
   The collective result **is** the MoE return value; no alternate path exists.
2. `deepseek_v4.py:5448,5451` — **[PM-VERIFIED]** `x = self.ffn(normed, input_ids)` then
   `out = finalize(hc_expand(x, residual, post, comb))`.
3. `mlx-lm/mlx_lm/models/hyper_connection.py:486-489` — **[PM-VERIFIED]**
   `_hc_expand_op` computes `post[..., None] * x[:, :, None, :]` plus
   `matmul(comb.swapaxes(-1,-2), residual)` — it mixes the MoE output into **all four**
   hyper-connection streams via a broadcast + dense matmul. **There is no clean residual
   stream that bypasses the FFN.** DSv4's structure is *stronger* than the standard
   transformer framing, not weaker.
4. `deepseek_v4.py:7046` — **[PM-VERIFIED]** `h = layer(h, mask, layer_cache, inputs)`:
   a straight-line rebind with no lookahead or branch.
5. Layer N+1's first op is an RMSNorm — a full-axis reduction, so every output element
   depends on every input element. No partial ordering survives.

**And there is no independent GPU work to overlap with instead** (an honest negative
result): expert dequantization has no separable node — it happens *inside* `gather_qmm`
on quantized weights; the MTP/DSpark draft head consumes hidden states from **after** all
43 layers; the attention indexer/compressor consume the current layer's own normed
hidden. The only loop-invariant find is a hash-layer `inds` int32 gather on 3 layers —
nowhere near enough to fill a collective.

**Two additional structural blockers**, independent of the above:
- `sharding_group` is set on Compressed/SparseCompressed attention
  (`auto_parallel.py:1122-1126`), so **41 of 43 layers carry TWO collectives**
  (`attn.all_gather` then `moe.all_sum`), not one.
- jaccl pins **one** CPU stream per group and `MeshGroup::all_sum` takes
  `collective_mutex_` — collectives cannot be simultaneously in flight, and any reorder
  must be rank-symmetric or it hits the documented FIFO-mismatch corruption class.

---

## What could NOT be determined from source

- The split of the ~1400 µs across `input_coherent` / `memoryBarrier` drain / commit
  latency. Needs the fork's existing `MLX_SIGNAL_PROBE=1` + `MLX_GPU_TIME`.
- Whether DSv4 crosses `needs_commit()` (50 ops / 50 MB, `device.cpp:768`) before the
  fence, which would shrink the barrier's scope.
- Whether `EXO_MAX_ACTIVE_TASKS` (default 10) trips the mid-tape `wait_for_one()` block
  at `transforms.cpp:285-299` during decode.
- Whether the two per-layer collectives ever partially overlap live (the mutex says no;
  unverified against a trace).
- Whether the cross-rank determinism concern in the `deepseek_v4.py:3077-3092` comment
  has *any* residual basis — this is the MEDIUM-confidence item in Q3.
