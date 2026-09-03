# W1 — jaccl transport-identity and buffer-registration feasibility (Task 1, Q1-Q4)

Scope: Task 1 questions 1-4 of `pm-campaign2-round3-task.md`. Q5 (H-e1/H-e2 architecture
+ us/layer estimate) and Q6 (prior-art grep) are covered by another worker's overall
FEASIBILITY.md; the prior-art grep result is included below since I ran it as
supporting evidence for Q1-4's "no existing GPU path" claim.

## Q1 — What is jaccl's transport, exactly?

**libibverbs-style RDMA, not Apple's own Thunderbolt networking API and not sockets
for the data path.** `mlx/distributed/jaccl/lib/jaccl/rdma.h` declares an `IBVWrapper`
that `dlopen`s `librdma.dylib` and resolves real ibverbs symbols: `ibv_get_device_list`,
`ibv_open_device`, `ibv_alloc_pd`, `ibv_create_qp`, `ibv_create_cq`, `ibv_query_port`,
`ibv_query_gid`, `ibv_modify_qp`, `ibv_reg_mr`, `ibv_dereg_mr` (rdma.h:143-167). The
`Connection` struct's `post_send`/`post_recv`/`poll` wrap `ibv_post_send`,
`ibv_post_recv`, `ibv_poll_cq` directly (rdma.h:271-296,314-315). TCP
(`lib/jaccl/tcp.cpp`) exists only as a control-plane side channel for connection setup
and small coordination messages (`CoordGroup`, `SideChannel`), not for tensor data.
So jaccl is a genuine QP/CQ RDMA transport running over Thunderbolt's RDMA-capable
fabric — same programming model as InfiniBand/RoCE — which is exactly the "RDMA path"
the gating question (round-3 task, line 15) assumes.

## Q2 — Buffer registration: in-place or host-bounce-by-construction?

**Host-bounce by construction; never registers Metal storage in place.** `SharedBuffer`
owns its own heap allocation obtained via `posix_memalign` at page granularity
(`rdma.cpp:22-28,75-76`: `page_aligned_alloc` → `posix_memalign(&buf, page_size,
num_bytes)`), completely independent of MLX's Metal allocator. `SharedBuffer::
register_to_protection_domain` registers *this* self-owned pointer with `ibv_reg_mr`
(`rdma.cpp:93-...`), not an `MTLBuffer`'s `contents()` pointer. The collective path
(`mesh_impl.h`'s `post_chunk`/`post_status` lambdas, e.g. lines 789-801) does
`std::memcpy(p + V2_HDR, reinterpret_cast<const char*>(out_ptr) + off, len)` — i.e. it
copies from the MLX array's buffer (`out_ptr`, which traces back to `input.data<char>()`
/ `output.data<char>()` in `jaccl.cpp`'s `JACCLGroup::all_sum`) into the RDMA-registered
`SharedBuffer` slot before posting. Same pattern for reassembly on receive
(`mesh_impl.h:768,883` etc. copy wire chunks into `asm_buf`, then a final
`std::memcpy(out_ptr, in_ptr, size*sizeof(T))` when `in_ptr != out_ptr`, e.g.
`reliable_all_reduce_v2` line ~697-699 and `mesh_impl.h:1476`).

For reference, MLX's Metal allocator (`mlx/backend/metal/allocator.cpp:17-18`) *does*
create buffers with `MTL::ResourceStorageModeShared` (`resource_options` combines
`ResourceStorageModeShared | ResourceHazardTrackingModeUntracked`), so the MLX-array
side is `StorageModeShared` and thus host-visible/DMA-able in principle — but jaccl
never touches that buffer's memory region for registration. Every all_sum/all_gather/
send/recv does at minimum one CPU-side `memcpy` into a separately `posix_memalign`'d,
separately `ibv_reg_mr`'d region on the send side, and a second on the receive/
reassembly side. This is a host bounce by construction, independent of and in addition
to the GPU→CPU handoff (fence busy-spin) round 2 already identified.

## Q3 — Coherence: what does `input_coherent` actually do?

`input_coherent` (`mlx/backend/metal/kernels/fence.metal:15-23`) is a GPU-side kernel
that does a **trivial self-store per element** (`input[index] = input[index]`) over a
`coherent(system)`-qualified device pointer, then issues
`atomic_thread_fence(mem_device, memory_order_seq_cst, thread_scope_system)`. This is
not a classic "flush the GPU cache to DRAM" op in the CUDA sense — Apple Silicon's GPU
and CPU share the SLC (system-level cache), so there's no separate device memory to
flush to. The self-store-plus-system-scope-fence pattern forces the GPU's per-element
writes to be visible at *system* scope (i.e., to any other agent — CPU, or in principle
an RDMA-capable DMA engine — snooping the SLC) rather than resting at GPU-only cache
scope. It is invoked from `Fence::update` (`fence.cpp:128-144`) only `if
(cross_device)`, sized to cover the *entire payload* (`nthreads` derived from
`x.data_size() * x.itemsize()`), immediately followed by a full `compute_encoder.
barrier()` (fence.cpp:146) before the tiny 1-thread `fence_update` kernel that flips the
timestamp the CPU spins on. So today's coherence guarantee is: touch every element of
the payload from the GPU, seq_cst system-scope fence, full compute-encoder barrier —
i.e. a whole-payload op plus a full device barrier, exactly as round 2 characterized it.

If a DMA engine (RDMA NIC or its associated DMA logic) read the buffer directly instead
of a CPU thread, it would need the same guarantee: GPU writes visible at system scope
before the DMA read starts. Metal does expose narrower-scoped mechanisms for managed
(non-shared) storage — `didModifyRange` for CPU→GPU direction on `StorageModeManaged`
buffers, and `synchronizeResource` in a blit encoder for GPU→CPU on Managed storage —
but **`StorageModeShared` buffers (which is what MLX/Metal allocator actually uses,
`allocator.cpp:17-18`) are documented as CPU/GPU coherent without those APIs on Apple
Silicon's unified memory, EXCEPT** that "coherent" in Apple's docs is at *normal* GPU
cache scope, not *system* scope reachable by a third-party DMA agent that isn't part of
Metal's own coherency domain. There is no public Metal API to get a narrower-than-
whole-resource, narrower-than-full-barrier system-scope fence — `MTLResource` hazard
tracking governs GPU-encoder-to-GPU-encoder and GPU-to-CPU-via-Metal ordering, not
ordering against an external RDMA NIC's independent DMA reads. The existing
`input_coherent` kernel plus device-scope barrier is very likely close to the *minimum*
publicly-documented mechanism to make a `StorageModeShared` buffer's GPU writes visible
to a non-Metal system-scope reader (like an RDMA NIC's DMA engine) — Apple does not
expose a cheaper, narrower "just this range, just to system scope" primitive. This is a
material data point against the "replace input_coherent with something cheaper" half of
H-e1 (task line 47's item b): there's no public API basis to assume a cheaper coherence
op exists.

## Q4 — Completion on the GPU: can RDMA completion become an `MTLSharedEvent` signal?

**Mechanically yes, cheaply, on the CPU side** — MLX already has `Event` wrapping
`MTL::SharedEvent` (`mlx/backend/metal/event.h:8-29`, backed by
`d.mtl_device()->newSharedEvent()` at `event.cpp:45`), with `wait(value)`/
`signal(value)` methods a compute encoder can `encodeWait`/`encodeSignalEvent` against.
Nothing in jaccl's CQ-polling path (`Connection::poll` → `ibv_poll_cq`, rdma.h:314-315)
is GPU-visible today; it's a plain CPU function call returning `ibv_wc` structs. But
turning "CPU thread observes a CQE" into "CPU thread calls `sharedEvent->
setSignaledValue(v)`" is a cheap, already-proven pattern in this codebase — it's
literally the CPU branch of `Fence::update` (fence.cpp:114-121), which does exactly
this today for the CPU-stream case. So the *shape* of H-e1's item (c) — CPU reacts to
completion and signals an event the next kernel `encodeWait`s on, rather than the GPU
polling/spinning for the CPU to notice — is architecturally sound and cheap: signaling a
`MTLSharedEvent` from a CPU completion handler is O(1) and does not require the GPU to
ever return control to the CPU or busy-spin. This is the one piece of H-e1 not blocked
by a hard platform fact.

## Prior-art grep (supporting Q1-4; full Q6 owned by teammate)

`git log --all --oneline` in `~/repos/exo/mlx` for `--grep=collective|all_reduce|jaccl`
(case-insensitive) surfaces the entire jaccl commit history — all reliability/protocol
fixes on the existing CPU-transport (QP budget, standing recv pools, reconnect races,
TCP coordinator splitting) — **none touch `AllReduce::eval_gpu`, MTLSharedEvent-driven
completion, or any GPU-resident collective path.** `mlx/backend/metal/distributed.cpp`
still unconditionally throws for `AllReduce`/`AllGather`/`Send`/`Recv`/`ReduceScatter`
`eval_gpu` (lines 16-33). No prior work in this fork has attempted H-e in any form.

## Bottom line for the gating question

Two separate, independent facts each currently force a CPU round-trip:
1. **The GPU→CPU handoff** (round 2's finding: MLX's `AllReduce` has no GPU eval path
   at all; jaccl's collective stream is CPU, `jaccl.cpp:88`).
2. **A structural host bounce inside jaccl itself**, found here: `SharedBuffer` is a
   `posix_memalign`'d region wholly separate from the MLX/Metal `StorageModeShared`
   buffer, registered via `ibv_reg_mr` on its own address — every collective call does
   at least one CPU `memcpy` MLX-array→RDMA-buffer on send and RDMA-buffer→MLX-array on
   receive, regardless of what a future GPU-resident scheme did about coherence or
   completion signaling.

H-e1 (keep CPU-driven RDMA post, GPU-resident coherence + MTLSharedEvent completion)
is not blocked by Metal API availability for the completion-signaling piece (Q4 is
solvable cheaply), but a credible >=40% removal of the 1400us/layer requires the
buffer-registration bounce (Q2) to also be eliminated (register `MTLBuffer.contents()`
directly with `ibv_reg_mr` instead of copying into a separate `SharedBuffer`) — that is
new engineering scope not addressed by the coherence/completion changes alone, and
Q3's finding that there's no public narrower-than-whole-barrier coherence primitive
caps how much the coherence side of H-e1 can save regardless. Both facts should feed
directly into the teammate's VERDICT and us/layer estimate for Q5.
