moe.all_sum cost attribution: NOT skew, NOT link bandwidth — jaccl chunking config (2026-08-19)
================================================================================================

Task P2 asked whether `moe.all_sum`'s measured 61-62% TP prefill cost is
rank-imbalance/straggler cost rather than genuine communication cost, to be
tested via a cheap 4-byte all_sum BARRIER ablation.

VERDICT (reached WITHOUT a cluster relaunch, from arithmetic + source):
the answer is NEITHER of the two hypotheses in the question. The cost is a
real, on-wire transfer cost — but it is running at ~1.6% of link capability
because of a jaccl chunk-size/pipeline-depth configuration, not because of
rank skew and not because the link is saturated.

The arithmetic that settles it
------------------------------

Payload per all_sum call (prefill chunk 2048, hidden 4096, bf16):
    2048 * 4096 * 2 = 16.8 MB
Calls per prefill: ceil(tokens/2048) * 43 layers.

From the already-measured warm-vs-warm NOP A/B in
`moe-all-sum-dominant-cost-2026-08-19.md`:

| depth | delta (all_sum cost) | calls | per-call | total payload | EFFECTIVE BW |
|---|---|---|---|---|---|
| 12,066 tok | 46.0 s | 258 | 178 ms | 4.25 GB | 92 MB/s |
| 38,066 tok | 139.1 s | 817 | 170 ms | 13.4 GB | 96 MB/s |

TB5/jaccl realistic throughput per this repo's own docs is ~6-10 GB/s.
At 6 GB/s the entire 13.4 GB would take 2.2 s, not 139 s.
**Bytes-on-wire explain ~1.5% of the measured cost.** Doubling the payload
for any reasonable 2-rank all-reduce algorithm (ring or naive exchange, TB5
full-duplex) still leaves it at ~3%. The bandwidth exoneration is robust to
the collective model.

Where the other ~98% goes — the actual mechanism
------------------------------------------------

`mlx/mlx/distributed/jaccl/lib/jaccl/mesh_impl.h` + `rdma.h`:

* `FRAME_SIZE = 4096`, and `v2_size_class()` returns
  `min(jaccl_reliable_max_sz(), BUFFER_SIZES-1)` — i.e. ONE uniform size
  class for every message, taken straight from the env knob.
* The live cluster runs `MLX_JACCL_RELIABLE_MAX_SZ=2`
  => chunk = 4096 * (1<<2) - V2_HDR ~= **15.9 KB**.
* A 16.8 MB prefill all_sum therefore becomes **~1029 chunks per call**.
* `jaccl_reliable_small_chunks()` defaults to 3, so with 1029 chunks the
  prefill collective is NEVER on the optimistic no-rendezvous path — it
  takes the LARGE path (`small == false`), which keeps the per-collective
  TCP-coordinator barrier rendezvous and rotates over
  `data_slots = NUM_BUFFERS/2 = 4` parity slots.
* 1029 chunks / 4 slots = **258 stop-and-wait rounds per call**.

Cross-check: 178 ms / 258 rounds = **690 us per round**, i.e. ~173 us per
16 KB chunk => ~94 MB/s. This matches the independently-derived effective
bandwidth (92-96 MB/s) from the throughput A/B **exactly**. Two independent
routes to the same number is what promotes this from a fitting story to the
mechanism.

Why the "skew/straggler" hypothesis is actively disfavored
----------------------------------------------------------

1. **The cost is flat per call across depth** — 178 ms @12K vs 170 ms @38K,
   over a 3.2x context increase. Per this repo's own perf-hypothesis
   discipline ("what does the curve SHAPE say"), flat-with-depth implies a
   fixed structural cost, not accumulating imbalance.
2. **TP on two identical machines is symmetric** — both ranks run the same
   compute on the same tokens, and a blocking collective every layer
   re-locksteps them, so skew cannot accumulate. Skew at this magnitude
   would require one node to be systematically ~170 ms/layer slower, which
   would be visible everywhere else too.
3. **Skew would be noisy and workload-dependent**; the measurement is a
   near-constant.

Why the proposed 4-byte BARRIER test was NOT run
------------------------------------------------

Besides needing a cluster relaunch (explicit approval required; a sibling
P1 task shares the cluster), a `consult` review identified that the barrier
ablation **cannot discriminate the live hypotheses**:

* A 4-byte message has `num_chunks = 1 <= 3`, so it takes the *optimistic
  small* path — a **different code path**, not merely fewer bytes. So a
  cost collapse would be ambiguous between "payload size mattered" and
  "path switched".
* "Barrier retains cost" is consistent with BOTH a sync/skew mechanism AND
  a fixed protocol/poll latency, so that outcome would not have been
  decisive either.
* A 4-byte value not data-dependent on `y` can also be reordered by MLX's
  lazy scheduler relative to local compute, firing at a different timeline
  position than the real collective.

The chunk-count arithmetic above answers the question more cleanly and at
zero cluster cost.

The real, testable lever (NOT yet run — needs approval + relaunch)
------------------------------------------------------------------

Raise `MLX_JACCL_RELIABLE_MAX_SZ`. Projected per-call cost if round count
scales as chunks/slots:

| sz | chunk | chunks/call | rounds | projected ms/call |
|---|---|---|---|---|
| 2 (current) | 15.9 KB | 1029 | 258 | 178 (measured) |
| 4 | 63.9 KB | 257 | 65 | ~45 |
| 5 | 127.9 KB | 129 | 33 | ~23 |
| 6 | 255.9 KB | 65 | 17 | ~12 |
| 7 | 511.9 KB | 33 | 9 | ~6 |

**CAUTION — this knob exists for a real reason.** Its own comment states:
"Large UC sends (>= ~64KB / sz>=4) do not reliably COMPLETE on Apple's
librdma (they stick, which is likely the same failure the UC all_reduce
wedged on); cap the chunk to a size class that reliably completes."
So sz>=4 is precisely the regime previously found to WEDGE. This is not a
free win — it is a correctness/throughput tradeoff that must be A/B'd
carefully, and the stall-timeout machinery (`jaccl_stall_timeout_us`,
`MLX_JACCL_ACK_RETRANSMIT_US=500000`) is what would catch a regression.

A cheaper first probe than raising chunk size: raise the in-flight/pipeline
depth or `MLX_JACCL_RELIABLE_SMALL_CHUNKS`, which changes round count
without changing per-send size class into the known-unreliable regime.

Status
------

* Attribution: **CONFIRMED by two independent arithmetic routes + source
  reading.** Not yet confirmed by a live gate-toggle A/B, which per this
  repo's standing discipline is what would promote it from strong candidate
  to settled. The A/B is the `MLX_JACCL_RELIABLE_MAX_SZ` sweep above.
* No cluster relaunch was performed. No files on the Mac Studios were
  modified. `/tmp/dsv4_nop_targets` was confirmed EMPTY on entry and left
  untouched.
