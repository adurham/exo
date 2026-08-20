PHASE 0a: moe.all_sum cost decomposition — the GPU→CPU boundary is REAL, PAYLOAD-PROPORTIONAL, and ~2.4 ms/call (2026-08-20)
============================================================================================================================

Question
--------

`moe-all-sum-178ms-artifact-real-bottleneck-2026-08-20.md` established that
`moe.all_sum`'s own wall time is ~5-12 ms/call, and *attributed* the residual
NOP-ablation delta to "collateral serialization the collective imposes on the
surrounding graph" — a GPU→CPU→GPU pipeline drain/refill, 43x per prefill
chunk. That attribution was **source-grounded but never isolated by
measurement**. Its own closing caveat says so: *"not yet confirmed by a
gate-toggle A/B."*

This is that isolation.

Method — the control the repo did not have
------------------------------------------

The doc's own source citations imply the key insight: nothing about the
drain is **collective-specific**. It is imposed by the *stream device
change*, which any CPU-stream op inherits:

* `mlx/backend/metal/distributed.cpp:17` — `AllReduce::eval_gpu` is a hard
  `throw`; every MLX collective is `eval_cpu`-only.
* `mlx/distributed/jaccl/jaccl.cpp:88-95` — `JACCLGroup` pins every
  collective to one owned `new_stream(Device::cpu)`, ignoring the caller's
  stream.
* `mlx/transforms.cpp:159-164` — MLX records `needs_fence` with
  `device_switch = (a.stream().device != in.stream().device)` whenever a
  consumer's stream differs from its input's.
* `mlx/backend/metal/fence.cpp:129-140` — a **cross-device** `Fence::update`
  additionally launches the `input_coherent` kernel over the WHOLE payload.
* `mlx/backend/metal/fence.cpp:57-76` / `event.cpp:56-167` — a CPU-stream
  consumer of GPU-produced data spin/sleep-polls until the GPU signals.

So a **non-collective op pinned to a CPU stream, at the identical shape**,
pays the exact same machinery with **zero wire bytes and zero cross-rank
lockstep**. Differencing that against the same op on the GPU stream isolates
the boundary from the collective. That is the probe:
`bench/phase0a_allsum_boundary_decompose.py`.

Conditions are measured back-to-back in one process, medians over 11 reps
with 4 warmups. The `layered_*` conditions run N sequential
`GPU chain -> op -> blocking mx.eval` cycles — exactly the Phase-H Lever 1
fence shape at `deepseek_v4.py:2836-2894` — which is where any per-layer
drain would *accumulate*.

RESULT 1 — the boundary is real, and it is NOT collective-specific
------------------------------------------------------------------

Single rank, production payload (2048 x 4096 bf16 = 16.8 MB), 12 layers:

```
  condition                median_ms    iqr_ms
  gpu_compute                  1.979     0.044
  gpuop_ready                  0.239     0.028      <- op on GPU stream
  cpuop_ready                  1.149     0.017      <- SAME op, CPU stream
  gpu_then_gpuop               1.978     0.064      <- no boundary
  gpu_then_cpuop               5.398     0.401      <- boundary crossed
  layered_gpuop               23.755     0.081
  layered_cpuop               63.232     0.663      <- 2.66x, zero wire bytes
```

`layered_cpuop / layered_gpuop = 2.66x` on **identical arithmetic with no
collective, no wire, no peer rank**. This alone confirms the doc's redirect:
the cost the NOP-ablation attributed to `all_sum` is substantially a
*stream-boundary* cost that `all_sum` merely happens to trigger.

Note how close 2.66x sits to the NOP A/B's 2.6x speedup. That is not proof of
identity, but it is the first independent measurement in the same magnitude
band, produced without any collective at all.

RESULT 2 — the boundary is PAYLOAD-PROPORTIONAL, not a fixed drain bubble
-------------------------------------------------------------------------

**This corrects the prior doc's mechanism.** "43 pipeline drain/refill
cycles" reads as a fixed-latency bubble per layer. It is not. Sweeping the
payload and subtracting the CPU op's own arithmetic
(`boundary = (layered_cpuop - layered_gpuop)/L - (cpuop_ready - gpuop_ready)`):

```
     MB  cpuop_ready  gpuop_ready  perlayer_raw  op_arith  BOUNDARY    GB/s
   1.05        0.148        0.153         0.238    -0.005     0.242    4.33
   4.19        0.356        0.149         0.770     0.207     0.563    7.45
  16.78        1.154        0.221         3.379     0.933     2.447    6.86
  67.11        4.445        0.559        12.776     3.886     8.891    7.55
```

64x the payload gives ~37x the boundary cost — **linear, at a stable
~7 GB/s**. A fixed drain/refill latency would be flat across this sweep. It
is not flat. The boundary is *bytes being made coherent across the
GPU/CPU stream split*, not a scheduling bubble.

~7 GB/s is far below M4 Max unified-memory bandwidth, and is consistent with
several full passes over the buffer (the `input_coherent` kernel over
`x.data_size() * x.itemsize()`, plus the CPU-side touch) rather than one
memcpy.

RESULT 3 — poll granularity is NOT the driver (200x sweep, flat)
-----------------------------------------------------------------

The obvious "it's the sleep-poll" hypothesis is **refuted**. `EventImpl::wait`
sleeps `MLX_EVENT_WAIT_POLL_US` (default 50 us) between polls. Sweeping it
200x:

```
  MLX_EVENT_WAIT_POLL_US     1      10      50     200
  per-layer boundary (ms)  3.230   3.263   3.341   3.330
```

Flat within noise. Tightening the poll to 1 us buys ~3%. **Do not spend
effort on poll-granularity tuning** — it cannot touch a payload-proportional
cost.

RESULT 4 — `MLX_METAL_FAST_SYNCH=1` is a REGRESSION here, not a lever
----------------------------------------------------------------------

The fast-synch fence path (`fence.cpp:14-27`, busy-wait on a shared buffer
with `__dsb(0xF)` instead of a `SharedEvent`) is the natural "make the
boundary cheaper" knob. Measured, it is worse:

```
  MLX_METAL_FAST_SYNCH=0   layered_cpuop  64.439 ms  (iqr  0.477)  boundary 3.390
  MLX_METAL_FAST_SYNCH=1   layered_cpuop  95.655 ms  (iqr 34.546)  boundary 6.000
```

~1.5x slower with ~70x the IQR. The full-system `__dsb(0xF)` barrier per
fence, and the CPU-side spin burning a core the GPU submission thread wants,
plausibly explain it. Flagging this so nobody spends a cycle on it.

RESULT 5 — two-rank numbers (reported, but weak evidence)
----------------------------------------------------------

`mlx.launch -n 2 --backend ring` on this laptop puts **both ranks on the same
GPU**, so compute and comms contend and the IQRs explode:

```
  layered_gpuop    99.853   iqr 38.505
  layered_cpuop   119.935   iqr 56.829
  layered_allsum  161.808   iqr 13.319
  -> per-layer boundary 1.674 ms ; per-layer all_sum 5.163 ms  (boundary = 32%)
```

The 5.16 ms/call for a real collective lands inside the doc's 5-12 ms band,
which is a consistency check. But single-GPU loopback is a known confound
(same caveat as `phase0b_collective_overlap_probe.py`), the IQRs are 30-60%
of the medians, and the boundary term here (1.67 ms) is *lower* than the
clean single-rank measurement (2.45 ms) — an ordering artifact, not a real
reduction. **Treat RESULT 5 as directional only.** The single-rank isolation
(RESULTS 1-4) is where the confidence is, precisely because it needs no
peer.

Decomposition at the production shape
--------------------------------------

Per `moe.all_sum` call at 16.8 MB, from the clean single-rank isolation plus
the prior doc's directly-measured collective time:

| component | ms/call | evidence | scales with |
|---|---|---|---|
| GPU→CPU stream-boundary coherency | **~2.4** | this doc, RESULT 2 | **payload (linear, ~7 GB/s)** |
| CPU-side per-call work (reduce, memcpy, memset) | ~0.9-1.7 | this doc `op_arith`; prior doc's `v2bench.cpp` (1.72 ms) | payload |
| wire + collective proper | ~2-9 | prior doc: 5.0 ms median / 9.0 ms p90 jaccl ENTER→EXIT | payload |
| cross-rank skew absorbed into the wait | remainder | not isolated here | rank jitter |

That totals into the 5-12 ms/call band from three independent instrumentation
layers. **No per-call term anywhere near 166 ms is reproducible.** The
178 ms figure remains an artifact, as the prior doc established.

What this changes
-----------------

CONFIRMED (was "strong candidate", now measured):

* The GPU→CPU stream boundary is a real, first-order cost, and it is
  **not collective-specific** — a plain `mx.abs` on a CPU stream reproduces
  2.66x, with zero wire bytes. The prior doc's redirect to the stream
  boundary was correct.

CORRECTED:

* The boundary is **payload-proportional (~7 GB/s), not a fixed drain
  bubble.** Any lever must reduce *bytes crossing the boundary* or *the
  number of crossings*, not "hide latency".
* This partially **rehabilitates payload-shrinking levers**, which the prior
  doc retired. It retired them on the reasoning that at a genuine ~12 ms
  cost, a 2x payload cut buys only ~6 ms — but it did not know the boundary
  term is *also* linear in payload. A 2x payload cut now cuts boundary
  (~2.4 ms) + CPU work (~0.9 ms) + wire together. **Caveat: the quantized
  all_sum family measured no speedup live** (`moe-allsum-quant-*`,
  `moe-allsum-sharedscale-*`), so this is a reason to re-examine *why* those
  nulled — quantize/dequantize adds its own GPU passes and may itself cross
  the boundary — not a reason to re-run them as-is.

RETIRED (measured null, do not re-attempt):

* `MLX_EVENT_WAIT_POLL_US` tuning — flat across 200x.
* `MLX_METAL_FAST_SYNCH=1` — 1.5x slower, 70x the variance.

Where the lever now points
---------------------------

Reduce **bytes across the boundary** or **crossings per chunk**:

1. **Fewer crossings.** 43 layers x 1 collective = 43 crossings/chunk. Any
   restructuring that batches multiple layers' reductions into one crossing
   attacks the term directly. (Distinct from OPT-7, which deferred the
   *fence* while leaving the crossing count unchanged — which is why OPT-7's
   -23% does not rule this out.)
2. **Fewer bytes**, judged against the corrected cost model above — with the
   quant-family null results explained first.
3. **A GPU-side collective.** `AllReduce::eval_gpu` throwing is the root
   cause of the crossing existing at all. This is the highest-ceiling and
   highest-cost option, and the `JACCLGroup` ctor comment
   (`jaccl.cpp:64-88`) documents exactly why the CPU pin exists (cross-rank
   encoder-thread race → UC FIFO corruption) — so it is a correctness
   constraint, not an oversight.

NOT verified
------------

* All RESULTS 1-4 are **single-rank on a shared MacBook Pro M4 Max**, with
  other user workloads running. Ratios are back-to-back and IQRs are tight
  (<2% except where noted), but absolute ms are not cluster numbers.
* The boundary→`all_sum` attribution is measured on a **synthetic** graph
  (matmul chain -> op), not on DSv4's real MoE tail. The shapes and the
  fence structure match; the surrounding graph does not.
* No cluster relaunch, no node touched, no runner killed, no model loaded.
* RESULT 5's two-rank numbers share one GPU and are directional only.
* The ~7 GB/s figure's *mechanism* (how many passes over the buffer, and
  which of `input_coherent` / CPU touch dominates) is **not** isolated — only
  its linearity is measured. A Metal capture would be needed to split it.

Files
-----

* Created: `bench/phase0a_allsum_boundary_decompose.py` (the probe).
* Created: this doc.
* No source modified.
