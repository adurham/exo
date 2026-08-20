moe.all_sum LEVER 3: the 178ms/call figure is an ARTIFACT — the collective itself costs ~12ms/call (2026-08-20)
================================================================================================================

Question
--------

Given `MLX_JACCL_RELIABLE_MAX_SZ=2->3` measurably changed nothing
(`jaccl-sz3-tested-no-improvement-2026-08-20.md`) and bytes-on-wire
explain only ~1.5% of the modeled cost
(`moe-all-sum-skew-vs-comms-2026-08-19.md`), what is the real
bottleneck inside `moe.all_sum`'s 178ms/call?

VERDICT
-------

**There is no 178ms/call. The number is derived, not measured, and it
is contradicted by three independent DIRECT measurements already in
this repo.** The collective's own wall time is ~12ms/call at the exact
same 16.8MB payload shape. The remaining ~166ms/call attributed to
`all_sum` by the NOP ablation is **collateral serialization cost the
collective imposes on the surrounding graph**, not time spent moving
or reducing bytes.

That single correction explains the sz=3 null result, retires the
chunking hypothesis, retires the "94 MB/s effective bandwidth"
finding, and redirects the lever.

The arithmetic reductio (settles it on its own)
-----------------------------------------------

From `dsv4-220k-prefill-span-profile-2026-08-18.md`, a real 220K-token
TP prefill at the standing config (`EXO_PREFILL_STEP_SIZE=2048`, so the
all_sum payload is `2048 * 4096 * 2 = 16.8 MB` — **the identical shape
the 178ms was derived at**):

```
moe.all_sum   4730 calls   12446.73 us avg   58873.05 ms total   9.5% of wall
```

Total prefill wall for that request: 612s (clean unprofiled run).

    4730 calls x 178 ms/call = 842 s

**842s of `all_sum` inside a 612s prefill is impossible.** The 178ms
figure exceeds the entire measured wall-clock of a run containing 4730
of those calls, by 38%. It cannot be the per-call cost of the
collective under any model.

Three independent direct measurements, all agreeing, all ~12ms
---------------------------------------------------------------

| source | instrumentation layer | per-call | share of wall |
|---|---|---|---|
| `dsv4-220k-prefill-span-profile-2026-08-18.md` | MLX model-code `span()` (wraps `all_sum` **and** its `mx.eval(y)` fence) | **12.4 ms avg** | 9.5% |
| `dsv4-220k-prefill-rdma-wait-breakdown-2026-08-18.md` | jaccl transport `[jaccl-v2] ENTER`→`EXIT` pairing, 14,919 calls | **5.0 ms median, 9.0 ms p90** | 19.1% (all collectives) |
| `moe-all-sum-dominant-cost-2026-08-19.md` (NOP A/B) | end-to-end tok/s delta | **178 ms derived** | 61-64% |

The first two are *different instrumentation layers measuring the same
thing* and agree with each other to within the fence overhead you would
expect (span includes `mx.eval`; ENTER→EXIT doesn't). The third is the
outlier by 14x — and it is the only one of the three that never
observes the collective directly.

At 12.4ms for 16.8MB that is **~1.35 GB/s each way** on the wire, which
is an entirely sane TB5/RDMA number at 16KB chunks. The prior doc's
"**92-96 MB/s effective bandwidth**, running at 1.6% of link
capability" was computed as `payload / 178ms` — it is the same artifact
divided through, not an independent confirmation. **The two "independent
arithmetic routes" that promoted the chunking story from fitting to
mechanism were not independent: both used the 178ms numerator.**

Why the chunk-round model was wrong on its own terms
----------------------------------------------------

The prior doc modeled 1029 chunks / 4 parity slots = **258 stop-and-wait
rounds per call**, cross-checking to 690us/round. Reading
`reliable_all_reduce_v2` (`mlx/distributed/jaccl/lib/jaccl/mesh_impl.h`
:670-1100), that is not how the large path works:

* `round` is incremented ONLY at the TCP `coordinator_->reliable_barrier`
  rendezvous, which is reached once at the **exit check** after
  `all_recv >= num_chunks && chunks_posted >= num_chunks` — i.e. after
  every chunk is already sent and received. It is not a per-4-chunk
  gate.
* `top_up_sends()` runs every loop pass and posts into any free parity
  slot; slots free as send CQEs arrive. The path is a **continuously
  topped-up pipeline over 4 slots**, not 258 serialized barrier rounds.
* Direct empirical confirmation already exists and was overlooked:
  `dsv4-220k-prefill-eventwait-rootcause-triage-2026-08-18.md` grepped
  the live logs and found **"All 8,988 16MB calls are `rounds=1`"** and
  "`rounds=1` is the ARCHITECTURAL BASELINE for the large path." One
  barrier per call, measured, at exactly this payload size. Not 258.

The 690us/round cross-check "matching exactly" was inevitable: both
sides of that check were `178ms` divided by the same assumed round
count. It validated an assumption against itself.

This also fully explains the sz=3 null result. Going sz=2->3 halves
`num_chunks` (1029 -> 513) but `rounds` was already 1, so it changes
nothing structural — only a second-order slice of a 12ms cost, i.e.
under ~4% of the 178ms that was expected to move. **The sz=3 experiment
was a correct experiment run against a wrong model, and its null result
is itself confirming evidence for this correction.**

CPU-side per-call work is also NOT the bottleneck (measured)
------------------------------------------------------------

Before concluding, I ruled out the obvious alternative — that the
178ms is CPU memcpy/reduce work inside `reliable_all_reduce_v2` rather
than wire time. Built a standalone replica of the function's exact
per-call CPU work (`/tmp/v2bench.cpp`, disposable) at the real 16.8MB
bf16 shape on an M4 Max:

```
                          sz=2      sz=3      sz=7
  asm_buf alloc+zero      0.28      0.12      0.12   ms
  memcpy in->out          0.01      0.01      0.02
  post_chunk (send copy)  0.32      0.26      0.30
  consume + zero_recv     0.46      1.75      0.75
  reduce_op bf16 scalar   0.65      0.65      0.59
  ----------------------------------------------
  TOTAL                   1.72      2.80      1.78   ms
```

**~1.7ms of CPU work per call** — including the scalar `SumOp<bfloat16_t>`
reduction over 8.4M elements and every `memset`/`memcpy` in the chunk
loop. That is ~1% of 178ms and ~14% of the real 12ms. Consistent with
the collective being ~12ms; nowhere near explaining 178ms. (Note the
non-monotonic `consume+zero_recv` column: `zero_recv_buffer` memsets the
FULL buffer on every re-arm, so larger size classes zero more bytes per
chunk — a mild real inefficiency, but bounded at ~1-2ms, not a lever.)

So where does the other ~166ms/call actually go?
------------------------------------------------

The NOP ablation is real and correctly executed — 2.6x speedup is not
noise. It just does not measure what it was read as measuring. NOP'ing
`all_sum` to an identity does far more than delete a 12ms transfer:

**1. It deletes a mandatory GPU-pipeline drain and refill, per layer.**
`AllReduce::eval_gpu` in `mlx/backend/metal/distributed.cpp:17` is a
hard `throw` — *"has no GPU implementation."* All MLX collectives run
`eval_cpu` only. And `JACCLGroup::communication_stream()`
(`mlx/distributed/jaccl/jaccl.cpp:88-95`) **pins every collective to one
owned `new_stream(Device::cpu)`**, deliberately ignoring the caller's
stream (the ctor comment explains why: cross-rank encoder-thread race
→ UC FIFO corruption). Consequence: every one of the 43 `all_sum` calls
per chunk is a **GPU→CPU→GPU stream round trip**. The whole preceding
MoE GPU graph must fully materialize; the CPU stream reduces; the next
layer's GPU work then depends on a CPU-stream output. The GPU pipeline
empties and refills 43 times per chunk. NOP'ing `all_sum` removes that
cross-device dependency entirely and lets the GPU stay saturated across
the whole forward pass — a gain that is attributed to `all_sum` by the
A/B but is not *inside* `all_sum`.

**2. It deletes per-layer cross-rank lockstep.** With the collective
present, each layer re-syncs both ranks; any per-layer jitter on either
node is absorbed into the next collective's wait rather than being
overlapped. NOP removes the lockstep, so both ranks free-run.

**3. It deletes the blocking `mx.eval(y)` fence's teeth.** The fence
(`deepseek_v4.py` ~2855-2894) still executes under NOP, but on a
purely local graph with no cross-device/cross-rank dependency — a
qualitatively cheaper flush.

This is also *exactly* what `gpu-util-vs-allsum-cost-reconciled-2026-08-19.md`
independently concluded from the opposite direction (97% "GPU busy"
telemetry coexisting with a 61-64% NOP delta = the GPU submission
thread is parked in a wait that reads as occupancy). That doc's
reconciliation was right; this one supplies the missing quantity — the
split is roughly **12ms genuine collective / ~166ms induced
serialization**, not 178ms of transport.

What this retires, and what it redirects to
-------------------------------------------

RETIRED (do not re-attempt without new evidence):

* **Chunk-size / `MLX_JACCL_RELIABLE_MAX_SZ` sweeps.** The projection
  table in `moe-all-sum-skew-vs-comms-2026-08-19.md` (sz=7 -> ~6ms/call)
  is built entirely on the 258-round model and the 178ms numerator, both
  now falsified. `rounds=1` is already the measured baseline. sz>=4 also
  carries the documented Apple librdma hang risk — that risk now buys a
  projected win that does not exist.
* **The "1.6% of link capability / 92-96 MB/s" framing.** Real figure is
  ~1.35 GB/s. The link is not the problem and neither is the protocol's
  chunking.
* **Payload-shrinking levers judged by this cost model.** The whole
  quantized-all_sum family (`moe-allsum-quant-*`, `moe-allsum-sharedscale-*`,
  all of which measured no speedup) was chasing a 178ms bytes-bound
  cost. Their null results are consistent with this correction and
  should be read as *confirming* it, not as separate failures. At 12ms
  genuine cost, even a perfect 2x payload cut buys ~6ms/call.

REDIRECTED — the actual lever is the GPU↔CPU stream boundary, not the wire:

* The dominant term is that MoE `all_sum` is structurally a
  **CPU-stream-only op inside a GPU graph**, forcing 43 pipeline
  drain/refill cycles per prefill chunk. Attacking bytes or chunks
  cannot touch it.
* Note this reframes the overlap idea too. `OPT-7` (deferring the
  `mx.eval` fence via `_fence_every_n`) failed at -23% because MLX's
  lazy graph accumulation cost dominated — but OPT-7 attacked the
  *fence*, while the drain is imposed by the **stream device change**,
  which no fence-gating can remove. These are different mechanisms;
  OPT-7's failure does not rule this one out.
* Cheapest next probe, and it needs no cluster relaunch: the existing
  `EXO_DSV4_ALLSUM_PROBE` instrumentation (`deepseek_v4.py:75`) already
  times `mx.eval(y)` immediately after the collective, per layer. Run
  it at 12K and compare its per-call number against 12.4ms. If the probe
  reports ~12ms while the NOP A/B still implies 178ms, the ~166ms is
  confirmed to live OUTSIDE the span — which localizes it to the
  drain/refill and makes the stream-boundary lever the confirmed target.
  This is a read-only diagnostic run, far cheaper than any further
  transport experiment.

Confidence and caveats
----------------------

* The reductio (842s of collective inside a 612s prefill) is
  arithmetic on numbers already in this repo's own docs and is not
  sensitive to any model of the collective. **High confidence.**
* The ~12ms/call figure is directly measured by two independent
  instrumentation layers that agree. **High confidence.**
* `rounds=1` at 16MB is directly measured from live logs (8,988 calls).
  **High confidence.**
* The *attribution* of the residual ~166ms specifically to GPU-pipeline
  drain/refill is source-grounded (`eval_gpu` throws; the CPU stream is
  pinned) and consistent with the independent GPU-utilization
  reconciliation — but it is **not yet confirmed by a gate-toggle A/B**,
  which per this repo's standing discipline is what would promote it
  from strong candidate to settled. The `EXO_DSV4_ALLSUM_PROBE` run
  above is the cheap first confirmation step.
* One depth caveat, flagged rather than smoothed: the 12.4ms span
  number comes from a 220K run, while the 178ms was derived at 12K/38K.
  The payload shape per call is identical (2048-token chunk), and the
  NOP result was itself flat across 12K/38K (62.0% vs 61.1%), so a
  depth confound would have to affect only the derivation and not the
  direct measurement. The reductio also does not depend on this — it
  uses the 220K run's own wall clock against its own call count.

Files
-----

* Created: this doc.
* No source modified. No cluster relaunch. No node touched.
  `/tmp/dsv4_nop_targets` not written.
* `/tmp/v2bench.cpp` + `/tmp/v2bench` — disposable local CPU-cost
  replica, not committed.
