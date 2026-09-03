# CAMPAIGN 2, ROUND 3 — FEASIBILITY: is a GPU-resident collective possible on this stack?

**Date:** 2026-09-03
**Gating question (task line 15):** can TB5 RDMA move data from a Metal unified-memory
buffer without a host bounce, and can completion be signalled to the GPU — well enough
to credibly remove **≥40% of the ~1400 µs/layer** decode-stall (pre-registered bar,
`pm-campaign2-round3-task.md:60-62`)?

**Verdict up front: INFEASIBLE AT THE BAR. H-e is CLOSED; the decode-stall thread is
CLOSED AT THE STACK LEVEL.**

The transport is genuinely RDMA and the *completion-signal* half is mechanically cheap
(Q4), but the design's two load-bearing premises fail against source facts: there is no
cheaper coherence primitive to replace `input_coherent` (Q3 — the existing kernel already
uses private Metal internals, i.e. it *is* the minimal mechanism), and jaccl's buffers are
a host bounce *by construction* independent of the GPU→CPU fence (Q2 — new this round).
With cross-layer overlap structurally blocked (round 2, `hyper_connection.py:486-489`),
the addressable window is the CPU segment of the collective itself, which the record
bounds at ~50–150 µs of ~1400 µs — **~1–11%, far below ≥40%**. The blocking fact, the
architecture analysis, and the honest caveats are below.

---

## Q1 — What is jaccl's transport, exactly?

**Genuine libibverbs-style RDMA over Apple's Thunderbolt RDMA stack, loaded dynamically;
TCP is control-plane only.** Every `ibv_*` verb is resolved at runtime via
`dlopen("librdma.dylib")`/`dlsym` through the `IBVWrapper` class
(`mlx/distributed/jaccl/lib/jaccl/rdma.h:143-167`, verified: `get_device_list`,
`open_device`, `alloc_pd`, `create_qp`, `create_cq`, `query_port`, `query_gid`,
`modify_qp`, `reg_mr`, `dereg_mr` all present as wrapper function pointers). The hot
path verbs are real: `Connection::post_send`/`post_recv` wrap `ibv_post_send`/
`ibv_post_recv` and `Connection::poll` calls `ibv_poll_cq` directly
(`rdma.h:271-296, 314-315`). QPs are created **UC** (unreliable-connected) — the
recurring QP-isolation work in this fork's history is all on UC QPs. TCP
(`lib/jaccl/tcp.cpp`, `CoordGroup`, `SideChannel`) carries only connection setup,
rendezvous/barriers, and ack coordination — never tensor data.

**So the gating question's premise holds at the transport layer:** the DMA path is
ibverbs, exactly the class of engine that could in principle read a registered host
buffer directly. **[PM-VERIFIED]** against `mlx/mlx/distributed/jaccl/lib/jaccl/rdma.h`
and `rdma.cpp` (note: the jaccl source tree is at `mlx/mlx/distributed/jaccl/lib/jaccl/`
within the submodule; W1's per-file line cites were re-verified by the PM at these paths).

## Q2 — Buffer registration: in place, or a host bounce by construction?

**Host bounce by construction — and this is a NEW blocking fact round 2 did not have.**
`SharedBuffer` owns its own `posix_memalign`'d, page-aligned region
(`rdma.cpp:22-28` `page_aligned_alloc` → `posix_memalign(&buf, page_size, num_bytes)`;
`rdma.cpp:73-77` the ctor allocates from it), and registers *that pointer* via
`ibv_reg_mr` in `SharedBuffer::register_to_protection_domain` (`rdma.cpp:90-96`).
Every collective copies the MLX array into a jaccl-owned registered slot before posting
(`mesh_impl.h:789-801`, `post_chunk` lambda:
`std::memcpy(p + V2_HDR, reinterpret_cast<const char*>(out_ptr) + off, len)` —
**[PM-VERIFIED]**, where `out_ptr` traces back to `input.data<char>()`/
`output.data<char>()` in `jaccl.cpp`'s `JACCLGroup::all_sum`), and copies back out of
the reassembly buffer on receive.

The MLX side is *not* the obstacle: the Metal allocator creates buffers with
`MTL::ResourceStorageModeShared | ResourceHazardTrackingModeUntracked`
(`mlx/backend/metal/allocator.cpp:17-18` — **[PM-VERIFIED]**), so MLX array storage is
host-visible unified memory that an ibverbs MR could in principle cover. **But no code
in this fork has ever registered an `MTLBuffer.contents()` pointer with `ibv_reg_mr`**
— every `reg_mr` site registers jaccl's own allocations. Register-in-place is therefore
*untested* on this librdma: plausible (malloc'd DRAM registers fine, and Shared storage
is ordinary wired-able DRAM) but with real unknowns (MR-table locking at QP INIT is
already a documented constraint in this codebase; page-granularity SGE requirements;
whether librdma pins IOKit pages behind Metal allocations). Killing the memcpys is
worth ~5 µs each at the 32 KiB decode payload (at the measured ~7 GB/s crossing rate,
`docs/phase0a-allsum-boundary-decomposition-2026-08-20.md:90-91`) — **~10 µs/layer,
0.7%**. It cannot approach the bar.

## Q3 — Coherence: what does `input_coherent` do, and is there a cheaper op?

**`input_coherent` is already the minimal mechanism — there is no cheaper public
primitive to replace it with. H-e1's premise (b) fails here.** The kernel source
(`mlx/backend/metal/kernels/fence.metal:15-26` — **[PM-VERIFIED]**, round 2 cited only
the dispatch site) is:

```metal
[[kernel]] void input_coherent(
    volatile coherent(system) device uint* input [[buffer(0)]], ...) {
  if (index < size) { input[index] = input[index]; }
  metal::atomic_thread_fence(metal::mem_flags::mem_device,
      metal::memory_order_seq_cst, metal::thread_scope_system);
}
```

Three facts, each load-bearing:

1. **It uses Metal *private* internals**: `#pragma METAL internals : enable`, the
   `coherent(system)` address-space qualifier, and `thread_scope_system` (a private
   enum, `__METAL_MEMORY_SCOPE_SYSTEM__`, defined locally in the file). The sibling
   `fence_wait` spin kernel carries the in-source purpose statement: *"System-scope
   atomic load to force GPU cache refresh from SLC"* (`fence.metal:54-58` —
   **[PM-VERIFIED]**). Apple ships **no public Metal API** that guarantees GPU-written
   lines become visible to a non-Metal DMA agent reading over Thunderbolt/PCIe:
   `didModifyRange:` covers CPU→GPU on Managed storage (a no-op on Apple Silicon);
   hazard tracking and `MTLFence` are GPU-encoder-scoped; `makeResident` governs
   residency, not cross-agent visibility. The fork built this kernel precisely because
   no public API does it.
2. **The encoder barrier is ordering, not visibility**: `compute_encoder.barrier()` →
   `memoryBarrier(MTL::BarrierScopeBuffers)` (`device.cpp:571-573` — **[PM-VERIFIED]**)
   is encoder-scoped — it orders the `fence_update` dispatch after the sweep within
   the command buffer. The *visibility* work (GPU L2/SLC → system scope) is done inside
   the kernel by the `thread_scope_system` fence before `barrier()` even runs. So even
   a perfect H-e1 could only drop the per-word self-store (the ~7 GB/s payload-linear
   term, ~4.7 µs at 32 KiB) — the system-scope fence itself must stay, and its cost is
   dispatch-level, not payload-level.
3. **The reverse direction (CPU→GPU) has no equivalent op at all today**: no
   `output_coherent` kernel, no `didModifyRange` on the receive path — only the fence
   *counter* is made coherent (`fence.cpp:118-122` seq_cst store + `__dsb(0xF)`). An
   H-e1 design that lets the GPU consume collective results directly would have to
   *add* a coherence op on the consume side; none exists to reuse.

**Not publicly determinable** (flagged, not smoothed): whether the TB5 DMA engine snoops
the SLC such that a registered-Shared-buffer read is coherent after a system-scope
fence alone. That is hardware behavior Apple does not document; the fork's own working
code implies the fence is necessary. Any H-e1 implementation would be betting on it.

## Q4 — Completion on the GPU: CQE → MTLSharedEvent → encodeWait?

**Mechanically yes, cheaply, with APIs that already exist in this fork — this is the one
H-e1 piece that is NOT blocked.** But see Q5: what it buys is bounded by notification
mechanics, not the dependency.

- The pattern already exists in production shape: `EventImpl::signal()` is a bare
  `mtl_event_->setSignaledValue(value)` callable from any CPU thread
  (`event.cpp:171-173` — **[PM-VERIFIED]**), and `Event::signal(stream)` on a CPU
  stream routes through `scheduler::enqueue` to call exactly that from an arbitrary CPU
  worker thread (`event.cpp:212-217` — **[PM-VERIFIED]**). The CPU branch of
  `Fence::update` already does "CPU thread signals a shared event" today.
- The GPU-side wait is real: `CommandEncoder::wait_event` → `buffer_->encodeWait(...)`
  (`device.cpp:654-659` — **[PM-VERIFIED]**); a kernel encoded after the wait cannot
  start until the event fires, and the host never blocks.
- jaccl's completion today is a CPU poll loop: `ibv_poll_cq` via `Connection::poll`
  from the reduction paths (`mesh_impl.h:2331` `int n = connections_[dst].poll(16, wc);`
  — **[PM-VERIFIED]**), running synchronously on the collective's CPU worker thread
  (the CPU stream from `jaccl.cpp:88` — **[PM-VERIFIED]** `communication_stream_(new_stream(Device::cpu))`).
- **Missing pieces: glue only.** jaccl has zero `MTLSharedEvent` usage today
  (`grep SharedEvent mlx/distributed/` → 0 hits — **[PM-VERIFIED]**). Wiring =
  hand the comm thread an `Event`, call `signal()` after the CQE poll observes
  completion, and have the consuming stream `Event::wait()` before its next dispatch.
  No new Metal capability required.

Important framing (task line 42-43 respected): a CPU thread that *signals an event
after the transfer completes* is indeed NOT the same cost as today's CPU thread the GPU
must *wait for before the transfer can start*. Q5 is where that distinction's value is
priced.

## Q5 — What does a GPU-resident collective look like here? H-e1 vs H-e2

### H-e1 (GPU reduce, CPU-driven transfer) — feasible to build, cannot clear the bar

H-e1's three items, priced against the measured record:

| Item | What it removes | Estimated saving (cites) |
|---|---|---|
| (a) register MLX storage in place | the two CPU `memcpy`s (send + reassembly) | ~10 µs/layer (2 × ~5 µs at 32 KiB @ ~7 GB/s, `phase0a:90-91`); risk: untested `reg_mr` on `MTLBuffer.contents()` |
| (b) minimal coherence op | nothing — **hard-blocked**: `input_coherent` already *is* the minimal op (Q3); only the self-store sweep (~4.7 µs payload term) is droppable, the system-scope fence must stay | ~0–5 µs/layer |
| (c) CQE → MTLSharedEvent → `encodeWait` | the host busy-spin *notification* mechanics (`fence.cpp:58-77`); the next layer's dispatches can be encoded ahead and released on event-signal latency | bounded by what the GPU could otherwise do during the wait: with cross-layer overlap blocked (`hyper_connection.py:486-489`), the GPU's only alternative is *starting the next layer's kernel encode* — a dispatch-latency effect, tens of µs (round 1's measured CPU→GPU dispatch 96.8 µs ± 4.6 is the outer bound of the whole notification class) |

**The controlling arithmetic.** The ~1400 µs/layer decomposes as: jaccl wire transport
36 µs (2.6%, `round1/REPORT.md:163`), crossing/coherency ≤ ~100 µs (≤7%, `MECHANISM.md`
Q5), RDMA poll 5 µs (0.36%), leaving ~1250 µs of *local drain+dispatch*. Round 1's
strongest fingerprint — the per-layer cost **alternates with each layer's own
`compress_ratio`** (1.50 vs 1.28 ms, `round1/REPORT.md:169-172`) and the no-peer control
reproduces the penalty with no peer — says the residual tracks the layer's own GPU
compute, not fence mechanics (fence/commit costs would be uniform across layers). Round
1 itself attributes ~98% of collective-span cost to the `mx.eval` *wait* — the wait for
the layer's own accumulated GPU work, which no collective redesign removes.

**What H-e1 credibly saves: ~10–150 µs/layer (≈1–11%), against a bar of ≥560 µs.**
Even taking the most generous reading — that (c) also recovers the full ~100 µs crossing
class plus dispatch-release latency — the design has no mechanism that touches the
~1250 µs of local compute drain, because with `hc_expand` folding the collective result
into all four hyper-connection streams, layer N+1 *cannot start* before layer N's
collective returns, no matter who posts the transfer or how completion is signalled.

**The one honest caveat (kept open, not buried):** the ~1250 µs residual has never been
directly decomposed (round 2's "what could NOT be determined" list). Two cheap probes
would falsify the "it's local compute" attribution if someone wants to re-open: (i) a
GPU-identity-kernel substitute for `all_sum` (isolates the sync overhead from the
collective entirely), (ii) `MTLCommandBuffer` GPUStartTime/GPUEndTime around one
all_sum to read GPU-idle vs GPU-busy directly. Both are *backend code*, prohibited this
round (constraint 3) — they are specified here as round-4 falsification probes for the
"local compute" attribution, not as H-e work. Note the async-fence history is itself
evidence on this question: fixing the async-fence gate (making `mx.eval` non-blocking,
`docs/async-fence-fix-validated-2026-08-22.md`) recovered **+58–67% decode throughput** —
i.e. removing the *blocking wait for local drain* was worth ~2/3 of decode, while every
transport-side lever ever measured is ≤5%. The stall's mass sits where H-e1 cannot reach.

### H-e2 (true GPU-initiated DMA) — INFEASIBLE on this platform

The GPU kernel itself cannot trigger the DMA: the entire ibverbs post path
(`ibv_post_send` etc.) is CPU-only in this stack, resolved through `dlopen` on the CPU
side; Metal exposes no API for a GPU kernel to reach an ibverbs QP; and there is zero
prior art of such a path in the fork's 108 jaccl commits, its branches, or upstream
(Q6). This would require private/undocumented Apple framework behavior not present
anywhere in this codebase. **INFEASIBLE without private APIs — stated with evidence:
`rdma.h:143-167` (verbs are CPU dlsym symbols), zero GPU-side post sites in the fork.**

## Q6 — Prior art

**None, anywhere.** In the fork: `git log --all --grep` for collective / all_reduce /
allreduce / jaccl / rdma surfaces 108 jaccl commits — all CPU-transport reliability and
protocol work (ARQ, QP budgeting, standing recv pools, reconnect races); **none touch
`AllReduce::eval_gpu`, MTLSharedEvent completion, or any GPU-resident path**. The file
that throws still throws: `mlx/backend/metal/distributed.cpp:17-19` (**[PM-VERIFIED]** —
and `AllGather`/`Send`/`Recv`/`ReduceScatter` `eval_gpu` all throw too, lines 21-33).
Fork↔upstream: the fork's recent all_reduce commits are all present in upstream
`ml-explore/mlx` (fork→upstream contribution flow — the fork is *not* secretly ahead);
upstream's `distributed.cpp` is the same throw. No GPU-collective issues/PRs found in
upstream search. Exo record: only round 2's MECHANISM/REPORT name H-e; nothing earlier
analyzed GPU-resident collectives or DMA-from-Metal-buffer. **GPU-resident collective
work on Metal is a clean slate — nobody has attempted it, which is consistent with the
platform facts in Q3/Q5 explaining why.**

---

## VERDICT

**INFEASIBLE at the ≥40% materiality bar.**

- **Blocking fact (Q3):** the coherence primitive H-e1 hoped to shrink *is* the minimal
  one — a private-internals Metal kernel because Apple exposes no public API
  guaranteeing GPU-write visibility to an external DMA agent. There is no headroom.
- **Blocking fact (Q2, new):** jaccl's registered buffers are a host bounce by
  construction; in-place registration is untested and worth only ~10 µs/layer.
- **Structural fact (carried from round 2):** with cross-layer overlap blocked by
  `hc_expand`, the GPU has no alternative work during the collective, so faster
  completion signalling (Q4 — genuinely cheap) cannot buy back wait time.
- **Sum:** H-e1 credibly removes ~10–150 µs of ~1400 µs/layer (~1–11%), far below the
  pre-registered ≥560 µs. H-e2 needs private APIs that do not exist in any code shipped
  on these machines.

**Disposition: close H-e; close the decode-stall thread at the stack level.** The
~1400 µs is dominated by local compute drain that no communication redesign touches.
If the thread is ever re-opened, the entry point is NOT a collective — it is the two
falsification probes above aimed at the local-drain attribution (and the
async-fence-gate class of wins, which the record shows is where the real 2/3 lived).

## Reconciliation with the prior record

- Round 2's mechanism is **confirmed, extended, and sharpened**: the CPU handoff round 2
  found is actually *two* CPU round-trips (fence + jaccl's own memcpy bounce) — both
  small; neither is where the 1400 µs lives.
- Round 2's Q3 correction (fence not load-bearing; 06-26 attempt-3 was an algebra bug)
  is settled **on evidence** this round: see REPORT.md Task 2 — the reorder with the
  second `all_sum` restores correct, coherent output (needle passes, no garbage),
  though NOT bit-identical output (FP reassociation drifts bits while staying correct;
  the gate's strict byte-identity leg fails, its quality leg passes decisively).
- Round 1's "~95% local drain + dispatch" attribution stands, now with the
  compress_ratio alternation fingerprint and the async-fence +66% natural experiment as
  corroborating evidence; the consult-challenged "unmeasured 1250 µs" is bounded above
  by what any notification/dispatch mechanism could recover (~150 µs), which is why the
  verdict does not hinge on its exact split.
- The 06-26 doc's "overlap primitive exists" claim (line 88-89), fence-bit-equiv claim
  (90-94), and "86 collectives" (43 layers × 2 = 86 *was* right for its 2026-06-26
  context of attn+MoE collectives, but the doc's own later framing used it per-layer;
  corrected per round-2's count) are patched at source with a dated CORRECTIONS block —
  see REPORT.md Task 3 for the commit SHA.

*Every cite above marked [PM-VERIFIED] was read line-by-line by the PM against the
primary source in `~/repos/exo/mlx` @ `e40a416b2` / `mlx-lm` @ `37260bbd6`+patch, not
accepted from worker summaries. Worker raw artifacts: `W1-jaccl-transport-buffer.md`
(this dir) and `/Users/adam.durham/tmp/w3_gpu_collective_prior_art.md`.*