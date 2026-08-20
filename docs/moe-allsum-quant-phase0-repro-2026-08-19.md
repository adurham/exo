PHASE 0 + 2.1: Quantized moe.all_sum — failure signature frozen, offline repro built
=====================================================================================
Date: 2026-08-19 (offline, no live-cluster contact)
Repro harness: `bench/moe_allsum_quant_repro.py`

TL;DR
-----

1. **Our quantize/all_gather/dequant logic is NOT the bug.** A real 2-rank
   offline repro (`mlx.launch -n 2 --backend ring`) runs the exact code from
   `mlx-lm@feat/moe-allsum-quant-2026-08-19` end-to-end and **passes**, both
   in eager-staged form and in the production-faithful lazy/32-layer form.
   Both ranks produce bit-identical output. rel err 0.008 vs exact all_sum.

2. **Both hypotheses in the failure doc are REFUTED**:
   - *"shape/rank contract mismatch"* — no. all_gather's contract is exactly
     `result_shape[0] *= group.size()`, concat on axis 0, which is what
     `_dequant_sum_shards` assumes. Verified in `mlx/distributed/ops.cpp:79-101`
     and empirically at 2 ranks.
   - *"the 3 all_gathers are not ordered/paired across ranks"* — no. This is an
     MLX-graph-scheduling property and therefore transport-independent. 32
     sequential quantized all_sums with NO intermediate `mx.eval` produced
     bit-identical results on both ranks. There is no ordering drift.

3. **The bandwidth premise HOLDS at world=2** (~47% wire reduction, verified).
   The failure is therefore confined to the jaccl/RDMA transport layer.

**Net: the bug is in the transport interaction, not in our logic. The lever is
still viable in principle at world=2, but the specific mechanism remains
undetermined and is NOT safe to retry live without the diagnostic fix below.**

Frozen failure signature
------------------------

```
[Event::wait] Timed out: GPU event not signaled and no stream exception
(peer rank stuck on an abandoned c>=2 collective); surfacing a clean
fault for in-place reconnect / restart.
```

**This message is a SYMPTOM, not the root cause.** Source:
`mlx/mlx/backend/metal/event.cpp:161-165`. It is a generic 40s watchdog
(`MLX_EVENT_WAIT_TIMEOUT_MS`, default 40000) that fires on *any* wedged peer.
Its twin lives at `mlx/mlx/scheduler.h:246-250`. Neither knows anything about
all_gather. The actual jaccl-level error would have been on the OTHER rank's
stderr — jaccl throws specific, named errors:
- `[jaccl] all_gather wc.status=... wr_id=... byte_len=...` (mesh_impl.h:1911)
- `[jaccl] all_gather recv byte_len=N expected=M ...` (mesh_impl.h:1917)
- `all_gather STALLED ... no forward progress` (StallWatch, mesh_impl.h:1903)

**This is the single most important gap.** The existing failure report captured
only the victim rank's generic timeout, which is precisely why root cause was
undeterminable. It was never a mystery — the evidence simply wasn't collected.

> **Mandatory precondition for any future live attempt: capture BOTH ranks'
> stderr.** Without the initiating rank's jaccl error, a retry will produce the
> same uninformative timeout and burn another cluster cycle.

What the offline repro proves
-----------------------------

`bench/moe_allsum_quant_repro.py`, three modes:

| mode | what it does | result |
|---|---|---|
| `--mode local` | shape/dtype/byte audit + 2-shard simulated gather math | PASS, rel err 0.0058 |
| `--mode dist` | real 2-rank, stage-by-stage (control all_sum, control all_gather bf16, all_gather uint32, then the full quantized path) | ALL STAGES PASS |
| `--mode dist-lazy` | production-faithful: fp32-safe wrapper installed, 32 sequential layers, no intermediate eval | PASS, both ranks bit-identical |

Run:
```bash
cd ~/repos/exo
.venv/bin/python bench/moe_allsum_quant_repro.py --mode local
.venv/bin/mlx.launch -n 2 --backend ring bench/moe_allsum_quant_repro.py \
    --mode dist --batch 256 --hidden 1024
.venv/bin/mlx.launch -n 2 --backend ring bench/moe_allsum_quant_repro.py \
    --mode dist-lazy --layers 32 --batch 512 --hidden 2048
```

**Important capability discovery:** the prior doc assumed "this dev machine has
no multi-rank capability" and "any retry is inherently a live-cluster risk."
That is **false**. `.venv/bin/mlx.launch -n 2 --backend ring` gives a genuine
2-rank collective on a single Mac. Multi-rank collective *logic* can and should
be validated offline before any live deploy. The ring backend does not exercise
jaccl/RDMA, so it cleanly isolates *our logic* from *the transport* — which is
exactly the discrimination Phase 2.1 needed, and it discriminated.

Bandwidth premise: CORRECT at world=2, but only at world<=3
-----------------------------------------------------------

An earlier draft of this document claimed the optimization moved *more* bytes
and was arithmetically dead. **That was wrong** and is retracted. It conflated
all_gather's materialized output buffer (which is `world_size x` the input)
with wire traffic — a rank's own shard is copied locally and never crosses the
wire. Corrected per-rank **wire** bytes, H=7168 L=2048:

```
 N=2: quant all_gather in=  15.60 MB | ring all_sum in=  29.36 MB -> WIN  (0.53x)
 N=3: quant all_gather in=  31.20 MB | ring all_sum in=  39.15 MB -> WIN  (0.80x)
 N=4: quant all_gather in=  46.79 MB | ring all_sum in=  44.04 MB -> LOSE (1.06x)
 N=8: quant all_gather in= 109.18 MB | ring all_sum in=  51.38 MB -> LOSE (2.12x)
```

The design comment's "~47% smaller" is accurate for the current 2-node cluster.
But note the scaling asymmetry: all_gather cost grows as `(N-1)`, while ring
all-reduce saturates at `2(N-1)/N -> 2`. **The lever inverts at N=4.** It is
correct only for this specific 2-node topology and must be gated/removed if the
cluster ever grows. Worth a hard assert on `group.size() <= 3` if revived.

Caveat: bandwidth saved is not automatically latency saved. Three collectives
instead of one adds per-call setup latency, and the scales/biases calls are
tiny and latency-bound rather than bandwidth-bound. The prior compute-overhead
analysis covered the quant/dequant GPU cost (~1.5%, negligible) but not the
extra collective *call* overhead. End-to-end wall-time at the combine point is
still unmeasured.

Secondary finding: silent fp32 downcast of scales/biases
--------------------------------------------------------

`deepseek_v4.py`'s `_collective_fp32_safe` wrapper (applied to `all_gather` in
the module-level loop) intercepts the **fp32 scales and biases** and silently
downcasts them to bf16 on the wire, then upcasts back. `mx.quantize` emits
fp32 scales/biases, so this fires **twice per MoE layer** (confirmed: 64
downcasts over 32 layers in `--mode dist-lazy`).

Not the crash cause — it round-trips fine and quantization error dominates —
but it is an *unintended* interaction nobody designed for: the quantization
scale factors themselves are being requantized to bf16. Note it is also what
makes the 1.06 B/elem figure achievable (fp32 scales would be 1.12).

Ruled out
---------

- all_gather axis-0 concat contract mismatch — verified correct
- 3-collective cross-rank ordering/pairing drift — verified bit-identical
- uint32 payload dtype unsupported by all_gather — control stage 0c passes
- 3-D `(B, L, H)` reshape incompatibility in `_dequant_sum_shards` — verified
- the local dequant+sum math — already unit-tested, re-confirmed
- the bandwidth premise being false — retracted, premise holds at N=2

Still open (jaccl/RDMA-specific, untestable on the ring backend)
----------------------------------------------------------------

Ranked by how well they fit the evidence:

1. **Size-class buffer-slot collision (strongest candidate).**
   `buffer_size_from_message` (`rdma.h:118-127`) maps a message to one of
   `BUFFER_SIZES=8` classes over `FRAME_SIZE=4096`. This file has a documented
   history of *exactly* this bug class — see the `data_pool_recv_buffers_`
   member comment in `mesh.h` (2026-08-16): a small collective landing in size
   class 0 shared a slot with the standing data-recv pool, and "under Tensor
   sharding that corrupted the FIRST all_gather of warmup ... 100%
   reproducible." The quantized path introduces **two new, differently-sized,
   much smaller messages** (scales, biases) per layer that the baseline path
   never emitted — new size classes on a code path with prior art for exactly
   this failure. Fits "hangs on first real prefill" well.

2. **3x the collective call rate.** `next_call_id()` is a per-process counter
   (`mesh.h:248-251`). Tripling collectives per layer triples call-id churn and
   in-flight QP pressure on a transport whose own comments repeatedly describe
   wedges from that pressure.

Both are consequences of *issuing three collectives where there was one*.

Recommended next steps (in order)
---------------------------------

1. **Fix the diagnostics first, not the code.** Ensure both ranks' stderr is
   captured on any live run. Enable jaccl's own tracing
   (`trace_call`/`trace_hash` are already instrumented in `mesh.cpp:1313-1326`).
   Without this, any retry is uninformative by construction.
2. **Reduce 3 collectives to 1.** Pack `wq`/`scales`/`biases` into a single
   contiguous byte buffer and issue one `all_gather`, unpacking locally. This
   directly addresses both open hypotheses (no new small size classes, no
   tripled call rate), removes the per-call latency concern, and is a small,
   self-contained change to `_quantized_moe_all_sum`. **This is the highest-
   value next action** — it is likely to be both the fix and the perf
   improvement.
3. Re-validate the packed version offline with this harness (`--mode dist-lazy`)
   before any live deploy.
4. Only then consider a live attempt, with both-rank logging, and with a
   `group.size() <= 3` guard given the N=4 inversion.

Do NOT re-attempt the current 3-collective implementation on the live cluster
as-is.

Files
-----

- `bench/moe_allsum_quant_repro.py` — new; the 3-mode offline repro harness
- `docs/moe-allsum-quant-phase0-repro-2026-08-19.md` — this document
- `mlx-lm@feat/moe-allsum-quant-2026-08-19` — unchanged, still unmerged
- Live cluster — untouched; standing pin `bd5d67648e` remains checked out
