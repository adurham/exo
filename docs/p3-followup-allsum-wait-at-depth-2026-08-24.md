# P3 follow-up — the all_sum-wait-at-depth measurement, closed: collective wait does NOT grow with depth; the residual is on-GPU busy work — 2026-08-24

**One-line result**: with a **same-build, two-depth, dual-rank** xctrace decode
capture (the measurement C2 lost to the crash), GPU occupancy **rises** with
depth — **83.70–83.86% @100K → 85.91–86.22% @352.6K** — so derived per-rank
idle **falls** by **0.07–0.14 ms/token** across the depth step while busy grows
**+5.0 ms/token**. The +1.0..+1.8 ms/tok residual is therefore **NOT collective
wait**: the collective/idle term moves the wrong way and is ~20× too small.
The residual lives in **unattributed on-GPU busy time**.

**Cost**: the deep capture **crashed both runners** (Metal GPU Timeout, 16 s
after tracer detach) even at a 12 s window. The retry at 100K with a 10 s window
**survived cleanly**. Cluster was relaunched and is **healthy** on the current
default build. Full incident record in §6.

**The ≤15 s protocol is NOT sufficient at 352.6K depth.** That is a new,
important operational finding and the HAZARD row is updated accordingly.

---

## 0. What was asked vs what exists

| deliverable | status |
|---|---|
| busy/idle ms/token, rank × depth, at ~352.6K | **DONE** — both ranks, n=1 (C2's missing point) |
| busy/idle ms/token, rank × depth, at ~100K, SAME BUILD | **DONE** — both ranks, n=1 |
| two-depth decision rule evaluated | **RESOLVED** — see §5 |
| additivity check against the depth budget | **DONE** — §7, does not close; overdraw on the kernel side |
| collective-wait-hidden-in-busy confound | **PARTIALLY** closed — §5.3, honest residual uncertainty |
| protocol viability at depth | **ANSWERED: ≤15 s is still unsafe at 352.6K** — §6 |

---

## 1. Method

Same read-only attach methodology as C2 (`docs/p3-worker-c2-depth-busy-idle-capture-2026-08-23.md`),
with four changes forced by worker D's forensics
(`docs/p3-worker-d-metal-timeout-crash-forensics-2026-08-23.md`):

1. **Window ≤15 s** (12 s deep, 10 s at 100K), not 50 s.
2. **Rank labels corrected.** C2's capture script hard-coded m4-1 → rank0.
   Worker D proved that is backwards; re-verified live from each node's own
   `mlx_distributed_init` line on **both** the pre-crash and post-relaunch
   cluster:

   | host | rank | role |
   |---|---|---|
   | `adams-mac-studio-m4-1.local` | **rank 1** | API node |
   | `adams-mac-studio-m4-2.local` | **rank 0** | jaccl coordinator / master |

3. **1 Hz memory sampler on both nodes**, running from before attach through
   the end of finalize (worker D's highest-value "instrument next time" item
   that did not need sudo). This turns the crash mechanism from an inference
   into a measured curve (§6.2).
4. **Headroom gate at attach time.** Available RAM is read on both nodes at the
   instant before attach; a node below threshold is dropped from the capture
   rather than attached. (Both nodes passed at both depths; the gate never
   fired. It is in the harness for the next worker.)

**Load generator**: C2's `/tmp/c2_probe.py` **unchanged** — it `importlib`-loads
`bench/p3_depth_anchor_probe.py` and reuses `build_prompt()` verbatim (same
nonce-at-front, same 8-topic non-degenerate filler, same tokenizer-targeted
binary search, `use_prefix_cache: False`, EOS-banning `/bench/chat/completions`
route, `max_tokens=2000`). Same script both depths.

**Capture orchestration**: `/tmp/p3e_capture.py` (new, this work). Blocks on the
probe's decode-start marker, waits 8 s, then fires both ranks' `xctrace` in
parallel over ssh. **Prefill is never traced** — attaching during a live deep
prefill wedged the collective 3/3 and is a documented hard rule.

**Export + parse**: `xctrace export --xpath .../table[@schema="metal-gpu-intervals"]`,
parsed by C2's `/tmp/c2_occ_parse.py` **unchanged** — streaming `iterparse`,
resolves Instruments' id/ref back-reference compression, keeps only rows whose
`process` field carries our runner PID, and computes **interval-union** busy
time (merged overlaps, never a naive sum).

**Pipeline validated before the real runs**: a 1,517-token / 10 s rank0 smoke
capture exported and parsed cleanly (40,814 own-process rows, 77.81% whole-span
/ 82.41% trimmed), and the cluster survived it with zero GPU errors.

---

## 2. Timeline

### 2.1 Deep point (352.6K) — captured, then crashed

```
21:19:41  probe starts; prompt 1,908,431 chars, predicted 352,645 tokens
21:37:26  DECODE_START  ttft=1064.26s (17.7 min prefill)
21:37:34  attach both ranks (decode_start + 8.0s), --time-limit 12s
            rank0 m4-2 xctrace pid 12732   } launched within 0.80s
            rank1 m4-1 xctrace pid 11950   } of each other
21:37:46  12s limit expires, both "Reached specified time limit"; finalize begins
21:37:51  probe stream ends early (624 events, no usage block)
21:38:02  rank1 (m4-1) runner dies: [METAL] GPU Timeout Error  <- T+16s after detach
21:38:11  rank0 (m4-2) runner dies (collateral); Worker plan: Shutdown
21:40     both traces finalize OK (2.2 GB -> ~399 MB each)
```

### 2.2 100K reference — captured, survived, ran to completion

```
21:50:58  cluster relaunch via start_cluster.sh (production env, no POOL_GROW vars)
21:55     READY (2/2), EXIT=0
21:56:25  probe starts; predicted 100,067 tokens
22:01:01  DECODE_START ttft=276.44s
22:01:09  attach both ranks (decode_start + 8.0s), --time-limit 10s
22:01:19  10s limit expires; finalize begins
          probe RUNS TO COMPLETION: finish_reason=length, completion_tokens=2000,
          REAL usage.prompt_tokens=100,067, cached_tokens=0
          both runners ALIVE, zero GPU timeouts
```

The 100K run is the **cleaner** of the two: it returned a real `usage` block, so
its depth is **measured, not inferred** — unlike C2's 100K run and unlike this
work's deep run.

---

## 3. Node memory at attach (measured, not inferred)

Available = (free + inactive + speculative) pages × 16 KiB, read on each node at
the instant before attach.

| depth | rank0 (m4-2) | rank1 (m4-1) | gate |
|---|---|---|---|
| 352.6K | **23.5 GB** | **25.9 GB** | both PASS |
| 100K | **33.1 GB** | **33.5 GB** | both PASS |

The deep point leaves **~8–10 GB less headroom** than 100K — the depth-scaling
memory risk worker D flagged, now quantified on this build.

**Measured tracer cost**: during the smoke capture, `xctrace`'s own RSS peaked at
**13.2 GB for a 10-second trace**. That is the quantity worker D inferred from
swapfile timestamps; it is now directly observed. A 10–12 s Metal System Trace is
**not** a small perturbation on a node with ~24 GB free.

---

## 4. Occupancy results — the 2×2 table, now complete

**Measured occupancy (%), interval-union, own-process rows only:**

| depth | rank0 (m4-2) | rank1 (m4-1) | window | own-process rows |
|---|---|---|---|---|
| **~100K** (100,067 real) | **83.86** / 83.93 trimmed | **83.70** / 83.76 trimmed | 10.57 s / 10.54 s | 46,546 / 46,416 |
| **~352.6K** | **86.22** / 86.30 trimmed | **85.91** / 86.12 trimmed | 12.54 s / 12.55 s | 44,120 / 44,121 |

Both depths, both ranks, **same build** (exo `10357e570`+, mlx-lm `0854b39`,
chunked pool growth as the code default, no `POOL_GROW*` env on either node).

**Client-side in-window ms/token**, restricted to exactly the aligned trace span:

| depth | events in window | span | ms/token | median gap |
|---|---|---|---|---|
| 100K | 274 | 9.91 s | **36.29** | 31.69 ms |
| 352.6K | 287 | 11.90 s | **41.60** | 36.50 ms |

**Same-build untraced anchors** (Part III §18, no tracer attached):
34.55 ms/tok @100,067 and 39.49 ms/tok @352,643.

### 4.1 Derived busy/idle ms/token

Arithmetic on the two measured quantities above, kept explicitly separate from
them. Two bases, because neither is unambiguously "the" answer — tracing inflates
wall time, but occupancy is a ratio and is the more transferable quantity.

**Basis (a): traced in-window ms/token** (what the trace itself saw)

| depth | rank | busy ms/tok | idle ms/tok | idle ÷ 43 all_sums |
|---|---|---|---|---|
| 100K | rank0 | 30.43 | **5.858** | 0.136 ms |
| 100K | rank1 | 30.38 | **5.915** | 0.138 ms |
| 352.6K | rank0 | 35.87 | **5.733** | 0.133 ms |
| 352.6K | rank1 | 35.74 | **5.864** | 0.136 ms |

**Basis (b): Part III untraced same-build anchors** (removes tracing overhead by
applying the measured occupancy fraction to a clean anchor)

| depth | rank | busy ms/tok | idle ms/tok | idle ÷ 43 all_sums |
|---|---|---|---|---|
| 100K | rank0 | 28.97 | **5.577** | 0.130 ms |
| 100K | rank1 | 28.92 | **5.631** | 0.131 ms |
| 352.6K | rank0 | 34.05 | **5.442** | 0.127 ms |
| 352.6K | rank1 | 33.92 | **5.566** | 0.129 ms |

### 4.2 The depth deltas — the actual answer

| basis | rank | Δbusy | **Δidle** |
|---|---|---|---|
| traced in-window | rank0 | +5.436 | **−0.125** |
| traced in-window | rank1 | +5.362 | **−0.051** |
| untraced anchor | rank0 | +5.075 | **−0.135** |
| untraced anchor | rank1 | +5.005 | **−0.065** |

**Idle per token does not grow with depth. It shrinks slightly, on both ranks,
on both bases.**

### 4.3 Rank symmetry / arrival skew

| depth | idle rank0 | idle rank1 | skew | occupancy gap |
|---|---|---|---|---|
| 100K | 5.577 | 5.631 | **0.054 ms/tok** | 0.157 pp |
| 352.6K | 5.442 | 5.566 | **0.124 ms/tok** | 0.315 pp |

Skew **does** grow with depth — by **+0.070 ms/token**. That is a real,
directionally-correct signal for the arrival-skew hypothesis, and it is **1.4–2.6%
of the +1.0..+1.8 residual**. It is nowhere near large enough to be the residual.

---

## 5. Verdict on the residual

The pre-registered decision rule:

> if idle ms/token at 352.6K ≈ idle ms/token at 100K (within ~±1 ms) while busy
> ms/token grows by roughly the full observed wall delta, then
> GROWING-COLLECTIVE-WAIT IS RULED OUT and the residual is on-GPU busy work.

**Both clauses are satisfied, with room to spare.** Idle changes by
**−0.05 to −0.14 ms/tok** (an order of magnitude inside the ±1 ms band) while
busy grows **+5.0 to +5.4 ms/tok**, i.e. **101–103% of the total wall delta**.

### **VERDICT: residual NOT collective. (option 2 of the three)**

The residual is **unattributed on-GPU busy work**, not waiting.

Numbers supporting it:

- collective/idle contribution to the depth delta: **−0.07 to −0.14 ms/tok**
  — **wrong sign**, and **~20× too small** in magnitude vs the +1.0..+1.8 band.
- arrival-skew growth: **+0.070 ms/tok** = **4–7%** of the residual band.
- **upper bound** on all_sum wait at 352.6K if *all* idle were charged to the
  collective (which it must not be — ordinary per-kernel CPU-dispatch latency
  lives in there too): **≤0.127–0.129 ms per call**, *below* the ≤0.130–0.131 ms
  bound at 100K.

Combined with worker A's code-proof that the **payload is L-independent**
(fixed (1,1,4096) bf16, 43×/token), the `moe.all_sum` question is now closed on
both axes: **payload flat (proven in code), wait flat-to-shrinking (measured
live, two depths, same build, both ranks).**

### 5.1 T5/C2 occupancy series, extended

| capture | window | depth | rank0 | rank1 |
|---|---|---|---|---|
| T2 | ~30 s | short (~512) | 78.64% | 78.86% |
| C2 | 50.5 s | ~100K | 83.06%\* | 82.98%\* |
| T5 | ~9 s | ~300K | 82.43% | 82.70% |
| **this, 100K** | **10.6 s** | **100,067** | **83.86%** | **83.70%** |
| **this, deep** | **12.5 s** | **~352.6K** | **86.22%** | **85.91%** |

\* C2's labels flipped per worker D's correction.

This run **reproduces C2's 100K point to within 0.8 pp on a 5× shorter window**
(83.86/83.70 vs 83.06/82.98) — a useful independent check that the short-window
protocol does not bias the occupancy ratio much. It also **overturns C2's
tentative "step up then plateau" reading**: occupancy keeps climbing from 100K to
352.6K. T5's 300K figure now looks low, consistent with C2's own warning that
T5's 9 s non-EOS-banned window was the weakest point in the series.

### 5.2 Cross-methodology corroboration

This is the third independent line of evidence pointing the same way. B1's
per-token latency distribution showed a **uniform rightward shift** with depth
(p10 and p50 move together, no tail fattening) — the signature of every token
paying more compute, not of intermittent waiting. Worker C's kernel microbench
found attention wall time growing **linearly** with L. And now occupancy rises
with depth. A growing collective wait predicts the opposite of all three.

### 5.3 The one honest hole: could a wait be hidden *inside* "busy"?

Raised in review, and it is a fair challenge. The occupancy method assumes a
rank waiting on the collective shows up as GPU **idle**. If MLX instead encodes
an event-wait *inside* a command buffer, the wait would be counted as **busy**
and a growing collective wait would be invisible to this method.

I tested it from the interval data (`/tmp/p3e_dur_hist.py`, depth-0 intervals
only). If the growth were a padded wait on the 43 collective calls, that
population would absorb essentially all of it while the rest of the distribution
stayed put:

| | rank0 100K → 352.6K | rank1 100K → 352.6K |
|---|---|---|
| intervals/token | 124.3 → 112.1 (**−12.2**) | 123.9 → 112.4 (**−11.5**) |
| busy/token (naive sum) | 31.505 → 36.771 (+5.266) | 31.401 → 36.687 (+5.286) |
| top-43/token population | 24.983 → 30.117 (+5.134) | 24.925 → 30.063 (+5.138) |
| **bulk (everything else)** | 6.522 → 6.654 (**+0.132**) | 6.477 → 6.625 (**+0.148**) |
| p50 / p90 / p99 duration | +9.3% / +22.5% / +28.3% | +9.0% / +22.4% / +53.6% |

**Reading, honestly**: the growth *is* concentrated in the large-interval
population — but that population is ~43/token, the **same cadence as all_sum**,
so cadence alone cannot discriminate the two hypotheses. What does discriminate,
weakly:

- the **whole distribution shifts right** (p50 +9%, p90 +22%), including
  intervals that are not in the top-43 population — a padded wait on 43 specific
  buffers should not move the median of a 112–124-interval-per-token population;
- **intervals per token FALLS by ~12** while busy per token rises, i.e. **fewer,
  bigger** GPU work items — the signature of kernels processing more data, not
  of a fixed-size op acquiring wait padding;
- the per-call growth in that population is **~119 µs/call**, which **overshoots**
  the 23–42 µs/call the collective-wait hypothesis needs by ~3×. If it were the
  collective, it would have to be *more* than the entire residual — but the
  residual band was derived assuming the kernel term is right, and this
  population also contains the attention kernels.

**So: consistent with kernel growth, not proof against a hidden wait.** The
decisive experiment is a direct CPU-side timer on the collective. The code
already has one — `EXO_DSV4_ALLSUM_PROBE` (`deepseek_v4.py:3026`, dumps per-layer
`mx.eval(y)` timings) — but it **forces a blocking `mx.eval` in place of the
production `mx.async_eval` fence**, so it changes the fence discipline it is
trying to measure and would need its own A/B to interpret. I did not run it;
flagged as the clean next step if anyone wants to upgrade this verdict from
"strongly supported" to "proven".

**What is NOT in doubt regardless of this hole**: the *idle* channel does not
grow with depth. Any surviving collective-wait story has to live entirely inside
command-buffer time, and has to explain why the median interval also grew.

---

## 6. The incident — the ≤15 s protocol is still not safe at 352.6K

### 6.1 What happened

The deep capture killed both runners. rank1 (m4-1) died first at **21:38:02**,
**16 s after tracer detach**, in the same `[METAL] Command buffer execution
failed: Caused GPU Timeout Error (00000002:***)` signature as the 2026-08-23
incident, preceded by `[wait_for_one] slow: elapsed=3.1s n_active=6`. rank0
(m4-2) followed at **21:38:11** as collateral, and the worker plan went to
`Shutdown`.

**This is the second crash of this exact shape, at a 4× shorter window.** C2's
was 50 s at 100K → death 6.5 s after detach. This was **12 s at 352.6K** → death
16 s after detach. The window cap alone did not save it. **Depth is the variable
the 12–15 s rule failed to account for**: at 352.6K the node has ~8–10 GB less
headroom (§3) and the tracer's ~13 GB peak no longer fits.

### 6.2 The mechanism, now measured rather than inferred

Worker D's memory-pressure hypothesis was ranked best-supported on retrospective
kernel-log evidence. The 1 Hz sampler makes it a curve, and it **confirms D and
sharpens the attribution to finalize, not recording**:

| t (rel. detach) | rank1/m4-1 avail | compressor | rank0/m4-2 avail | compressor | swap |
|---|---|---|---|---|---|
| attach −12 s | 26.2 GB | 0.0 GB | 23.7 GB | 11.0 GB | 140 M |
| during recording | 26–30 GB | 0.0 GB | 23–27 GB | 11.0 GB | 140 M |
| **detach (T+0)** | 26.1 GB | 0.0 GB | 23.3 GB | 10.9 GB | 140 M |
| T+5 s | 20.9 GB | **7.0 GB** | 16.6 GB | 15.5 GB | 140 M |
| T+8 s | **18.4 GB** | **12.0 GB** | **12.9 GB** (min) | — | — |
| T+20 s | 66.8 GB | **35.2 GB** | 24.3 GB | **82.5 GB** | **12.6 GB** |
| **crash** | — | — | 27.1 GB | 79.4 GB | **15.0 GB** |

**Memory is flat during recording and collapses only at finalize.** Compressor
goes 0 → 35 GB on m4-1 and 11 → 82.5 GB on m4-2; swap goes 140 MB → **15 GB**.
Both runners die inside that window. This is exactly worker D's mechanism #1,
now with the curve he asked for — and it **localizes the danger to the
stop/finalize phase**, which is the part the window-length cap does *not*
shorten proportionally.

### 6.3 Handling (per the task's crash rule)

Reported here; last log lines saved to `/tmp/p3e_crash/`. Cluster relaunched via
`start_cluster.sh` with production env and **no `POOL_GROW*` vars** (default is
in code): `READY (2/2)`, `EXIT=0`, nodes synchronized on `bbfd53e66`. The single
permitted retry ran at **10 s** (shorter than the crashed 12 s) at the shallower
100K depth and **survived cleanly** — probe ran to completion, both runners
alive, zero GPU timeouts. **No second crash.**

### 6.4 Revised guidance

- **≤15 s is NOT sufficient at 352.6K.** The correct constraint is worker D's
  memory form, not a wall-clock form: `trace_peak_GB + resident_GB < RAM − margin`,
  with `trace_peak_GB ≈ 13 GB even for a 10 s Metal System Trace`.
- At 352.6K the deep node has ~23–26 GB available, so a 13 GB tracer plus a
  finalize that inflates the compressor by tens of GB does not fit. **Assume any
  Metal System Trace attach at ≥350K depth will kill the instance.**
- If a deep capture is required anyway: **budget the run as sacrificial**,
  capture **one rank only**, and expect to relaunch. The traces do survive the
  crash — both deep traces finalized fine and parsed cleanly, which is why this
  measurement exists at all.
- Finalize for a 10–12 s trace is **~1–5 min**, not the ~25 min a 50 s trace
  needed. Export is ~6–9 s. So the *cost* of the protocol is fine; the *risk* is
  not.

---

## 7. Additivity — does the budget close?

Using **same-build** numbers throughout (this is the first time both depths and
the occupancy split come from one code state).

**Measured total depth cost, 100K → 352.6K, post-pool-fix**: the task brief
carried **+4.35 ms/tok**. The same-build Part III anchors give
**39.49 − 34.55 = +4.94 ms/tok**. I use **+4.94**, and flag the discrepancy: the
+4.35 predates the final default-flip re-verification.

| component | ms/tok | evidence class |
|---|---|---|
| Attention/indexer kernel wall time | **+2.56 … +3.34** | worker C microbench, production silicon |
| Collective / GPU-idle growth | **−0.07 … −0.14** | **this work**, measured, both ranks |
| Pool-growth concat term | **0** (already removed) | shipped as code default; the +4.94 is post-fix |
| **Sum of explained** | **+2.42 … +3.27** | |
| **Measured total** | **+4.94** | same-build anchors |
| **Residual** | **+1.67 … +2.52** | **on-GPU busy, unattributed** |

**The budget does NOT close.** It is short by **+1.67 to +2.52 ms/tok**, and the
residual **grew** relative to the +1.0..+1.8 band the task carried — because the
same-build total (+4.94) is larger than the +4.35 the band was computed against,
while the collective term contributes nothing.

**Where the missing time most likely is.** Busy growth is **+5.0 to +5.4 ms/tok**,
which is **101–103% of the total wall delta** — i.e. essentially the entire depth
cost is on-GPU busy, and the independently-measured kernel band accounts for only
**~51–67%** of it. The leading candidate is the one worker C flagged against
himself and R2 repeated: **C's harness instantiates one layer per class and
scales by census, so it captures no inter-layer pipelining loss and biases the
attention estimate DOWN.** The interval data supports that reading — fewer,
bigger GPU work items at depth (§5.3), with the median interval up 9%. MoE-at-depth
interplay and the ~90 GB-resident allocator regime remain untested.

**Additivity sanity check passes in the weak sense**: Δbusy + Δidle = +4.94 =
the measured total, by construction, with no overdraw. The problem is not
double-counting; it is that the kernel microbench under-predicts the on-GPU term.

---

## 8. Limitations — read before citing anything above

- **n=1 per depth.** One capture at each depth, one run each.
- **Deep run has no `usage` block.** The stream died before it (the crash), so
  **352.6K depth is inferred**, not returned: same builder, same target, same
  tokenizer, locally predicted 352,645; Part III's identical-target run landed
  at a real 352,643. Strong but indirect. The **100K point is fully measured**
  (`usage.prompt_tokens = 100,067`, `finish_reason = length`, 2000 tokens).
- **Windows are 10 s and 12 s** — shorter than C2's 50 s, and the two depths
  differ from each other by 2 s. Occupancy is window-length sensitive (C2's own
  smoke read 75.59% whole-span vs 82.06% trimmed); I report whole-span and
  1 s-trimmed, and they agree to <0.25 pp everywhere, which is the reassuring
  case. Cross-depth window mismatch (10 vs 12 s) is a real if small asymmetry.
- **Traced decode ran ~5% slower than untraced** (36.29 vs 34.55 @100K; 41.60 vs
  39.49 deep) — the expected, documented Instruments overhead. The **differential**
  between depths (~1.74 vs ~2.11 ms/tok) is ~0.4 ms/tok and lands somewhere in
  the busy/idle split. **This is larger than the Δidle I report (−0.05..−0.14).**
  So the precise claim "idle *shrinks*" is **overprecise**; the robust claim is
  **"idle growth is ≪ +1.0 ms/tok"**, which is all the verdict needs.
- **The hidden-wait confound (§5.3) is narrowed, not eliminated.** A direct
  CPU-side collective timer is the clean closer.
- **Per-kernel attribution remains impossible** with this template: 99.98% of
  intervals are generic `"Compute"`. The "idle ÷ 43" figures are **ceilings
  derived by division, not measurements of the collective**.
- **Idle here is whole-process GPU idle**, which includes ordinary per-kernel
  CPU-dispatch latency. Attributing all of it to all_sum would be wrong.
- **The deep point's own throughput number is unusable** (crashed run, 624
  events, no usage), so §7 uses Part III's clean same-build anchor for the deep
  ms/token rather than this run's. The **occupancy ratio** from the deep run is
  unaffected — the trace window closed 5 s before the stream ended and 16 s
  before the crash, and contained no stall above 289 ms.
- **`powermetrics` still not captured** — passwordless sudo unavailable on both
  studios, unchanged since C2.
- **Different runner PIDs across the two depths** (the relaunch sat between
  them). Same build, same env, same rank assignment — verified — but not the
  same process lifetime, so allocator-age effects are not controlled.

---

## 9. Files

Local (this Mac): `/tmp/p3e_capture.py` (new orchestrator: corrected rank
labels, ≤15 s cap, memory sampler, headroom gate), `/tmp/p3e_dur_hist.py` (new,
interval-duration discriminator), `/tmp/p3e_deep.json`, `/tmp/p3e_100k.json`,
`/tmp/p3e_{deep,ref100k}_capture_meta.json`, `/tmp/p3e_crash/` (crash forensics
+ both nodes' 1 Hz memory logs).

On the studios (`/tmp` only): `p3e_{deep,ref100k}_rank{0,1}.trace`,
`p3e_{deep,ref100k}_rank{0,1}_occ.json`, `p3e_{deep,ref100k}_rank{0,1}_dur.json`,
`p3e_memsample.sh`.

Related: `docs/p3-worker-c2-depth-busy-idle-capture-2026-08-23.md` (the capture
this completes), `docs/p3-worker-d-metal-timeout-crash-forensics-2026-08-23.md`
(the forensics whose protocol this executes, and whose mechanism this confirms
with live telemetry),
`docs/p3-synthesis-500k-decode-decay-decomposition-2026-08-23.md` (the budget),
`docs/p3-followup-poolgrow-ab-2026-08-23.md` §18 (the same-build anchors).
