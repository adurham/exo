# exo Performance History — Consolidated Reference

**Compiled 2026-08-21.** This document consolidates every performance,
benchmark, timing, throughput, and optimization investigation found across
~197 markdown files in this repository (docs/, .hermes/plans/, bench/,
root-level handoffs), spanning 2026-04-24 through 2026-08-21 (~4 months of
active work). Built by systematically scrubbing the full repo history, not
just recent sessions.

**Purpose: prevent re-litigating settled questions.** Every entry below
states what was tried, the real numbers, why it worked or didn't, and a
one-line reusable lesson. Read the relevant section before starting new
optimization work in that area.

**Cross-checked and merged (2026-08-21):** a second, fully independent
consolidation of the same raw source data was separately written by
Claude Fable 5 — given only the raw per-file findings, never this
document, so it was a genuine independent synthesis, not a review. The
two versions agreed on essentially every substantive finding (same
negative results, same contradictions flagged — e.g. the gate+up
fusion's -3.8% vs +3.01% discrepancy across sessions), which is a real
corroboration signal, not just two summaries of the same source text.
One real gap was found this way and fixed: this doc originally omitted
the `mlx#3596` allocator-coalescing RSS finding (now in §10). Fable's
independent §12.5 "cross-domain recurring patterns" cut and its
closed-levers quick-reference table (bottom of this doc) were merged in
directly since they added genuine, non-duplicate value beyond what this
doc already had. The standalone Fable doc has been retired now that its
unique content lives here — this is the single reference going forward.

**How to use this doc:** find your topic area below, read the WINs to know
what's already banked, read the NEGATIVE/DEAD-END entries to know what NOT
to re-attempt (or what would need genuinely new evidence to justify
retrying), and check INCONCLUSIVE entries for open threads that were never
fully closed out.

---

## Table of contents

1. [Current known-good baseline](#1-current-known-good-baseline)
2. [MoE / all_sum collective — the dominant cost center](#2-moe--all_sum-collective--the-dominant-cost-center)
3. [Prefill throughput](#3-prefill-throughput)
4. [Decode throughput & dispatch overhead](#4-decode-throughput--dispatch-overhead)
5. [Speculative decoding (MTP / Eagle / token-tree / DSpark)](#5-speculative-decoding-mtp--eagle--token-tree--dspark)
6. [Kernel fusion attempts (general pattern)](#6-kernel-fusion-attempts-general-pattern)
7. [Quantization (KV cache, weights, collectives)](#7-quantization-kv-cache-weights-collectives)
8. [jaccl / RDMA / Thunderbolt transport](#8-jaccl--rdma--thunderbolt-transport)
9. [Concurrency (c=2+) and PP vs TP topology](#9-concurrency-c2-and-pp-vs-tp-topology)
10. [Memory leaks](#10-memory-leaks)
11. [Correctness bugs found during perf work](#11-correctness-bugs-found-during-perf-work)
12. [Measurement methodology lessons (meta)](#12-measurement-methodology-lessons-meta)
    - [12.5 Cross-domain recurring patterns](#125-cross-domain-recurring-patterns)
13. [Open / never-finished threads](#13-open--never-finished-threads)

---

## 1. Current known-good baseline

**LOCKED IN 2026-08-22** (tags `known-good-decode-fenceasync-20260822`
on both `exo` and `mlx-lm` repos; `mlx` repo unchanged since
`known-good-prefill-20260821-165048`, still current). Config:
`EXO_DSV4_MOE_FUSED_GATE_UP=1` + `EXO_DSV4_FENCE_ASYNC=1`, the latter
now GENUINELY functional after the §2.8 fix (was silently broken for
the entire campaign before this date — see below). All figures
re-measured fresh against this exact config, needle-verified where
applicable.

| Metric | Value |
|---|---|
| Prefill @ 100K ctx | 359.7 tok/s (needle PASS) |
| Prefill @ 300K ctx | 348.2 tok/s (needle PASS) |
| Prefill @ 500K ctx | 324.1 tok/s (needle PASS) |
| Decode @ 100K ctx | 26.91 tok/s (needle PASS) |
| Decode @ 300K ctx | 24.44 tok/s (needle PASS) |
| Decode @ 500K ctx | 21.51 tok/s (needle PASS) |
| Decode, short context (512-2000 tok prompt) | 29.2-31.1 tok/s |

Prefill is at parity with the pre-fix baseline (was 366.6/351.5/331.6 —
within normal run-to-run variance, confirming the async-fence fix
correctly left prefill untouched, as designed). **Decode is up
+23-67% depending on context depth** vs. the pre-fix numbers below,
because the async fence — live in production config since 2026-07-02
but silently non-functional the whole time — now genuinely engages.

**MTP/DSpark status, confirmed live 2026-08-22:** NOT wired up for this
cluster's TP sharding mode. `EXO_DSV4_MTP=0` (classic single-head MTP
disabled). `EXO_DSV4_DSPARK=1` is set but structurally inactive —
DSpark's real decode loop (`pp_dspark_decode_loop`) is PP-only; live
log check shows zero `"PP speculation using DSpark"` lines against 12
`"DSpark ctx warmed"` lines (module loads/warms, decode loop never
fires). This is a real, standing gap — DSpark/MTP speculative decode
represents unrealized throughput upside on this cluster, not yet
attempted for TP. See §12 and §13 for prior investigation.

> **CORRECTION 2026-08-24 (P4v2, machine-verified against exo `6dcd61e4a`
> / mlx-lm `0854b39`; supersedes the paragraph above on three points).**
> See `docs/p4-scoping-mtp-for-tp-2026-08-24.md` for the full trail.
>
> 1. **"NOT wired up for TP" is wrong.** A TP-mode draft+verify cycle
>    exists and is live-reachable: `DSv4MTPBatchGenerator._speculative_next`
>    (`speculative/dsv4_mtp.py:3415`), constructed at
>    `generator/batch_generate.py:847`, dispatched via `self._mlx_gen.next()`
>    at `:4228`. It contains a **first-class DSpark branch**
>    (`dsv4_mtp.py:3481-3486`) that drafts with the DSpark 3-stage head and
>    feeds it ctx at `:3984-3991`. Nothing needs to be built for TP.
>    What is actually off is the env: `EXO_SPECULATIVE=0` **and**
>    `EXO_DSV4_MTP=0` (both verified live via `ps eww` on both nodes,
>    2026-08-24).
> 2. **The zero-`"PP speculation using DSpark"` evidence proves nothing
>    about TP.** That line is emitted only on the PP path
>    (`batch_generate.py:3466`, reached from `pp_speculation.py`'s
>    `pp_dspark_decode_loop`). Its absence under TP is *expected by
>    construction*, not evidence of missing TP capability. The TP DSpark
>    branch has **no log line at all** — that is an observability gap, not
>    a wiring gap.
> 3. **`EXO_DSV4_DSPARK=1` is not free under TP, and it is not "warmed but
>    ready" either.** Today it (a) attaches a ~10 GB draft head into unified
>    memory on every node (`utils_mlx.py:383-390`, live log: `DSpark draft
>    head attached ... 115 tensors, 3 stages, block_size=5, taps=[40,41,42]`),
>    (b) arms per-layer hc-mean capture taps at layers 40/41/42 that run on
>    **every** target forward including all prefill
>    (`mlx-lm/mlx_lm/models/deepseek_v4.py:6705-6724`), and (c) does a
>    per-request `append_ctx` warm (`batch_generate.py:2748-2760`). All of
>    that output is discarded because the consumer
>    (`_speculative_next`) is never constructed while `EXO_SPECULATIVE=0`.
>    See the P4v2 memo §2 for the pure-waste cost breakdown.

At parity with an earlier-campaign 339 tok/s @ 500K prefill baseline
(i.e. months of jaccl/transport hardening work did not regress raw
throughput — it fixed correctness/stability, see §8).

<details>
<summary>Historical baseline, 2026-08-21 — superseded, decode figures
measured under the silently-broken async fence (click to expand)</summary>

**As of 2026-08-21** (tag `known-good-prefill-20260821-165048`):

| Metric | Value |
|---|---|
| Prefill @ 100K ctx | 366.6 tok/s (needle PASS) |
| Prefill @ 300K ctx | 351.5 tok/s (needle PASS) |
| Prefill @ 500K ctx | 331.6 tok/s (needle PASS) |
| Decode @ 100K ctx | 17.48 tok/s |
| Decode @ 300K ctx | 18.60 tok/s |
| Decode @ 500K ctx | 17.26 tok/s |
| Decode, short context (512-tok prompt) | ~18.6-18.9 tok/s |

`EXO_DSV4_FENCE_ASYNC=1` had been silently, permanently non-functional
(always falling back to the blocking `mx.eval(y)` fence) for the
entire multi-session optimization campaign prior to the 2026-08-22 fix
(§2.8, `docs/async-fence-cache-owner-dead-code-root-cause-2026-08-22.md`,
`docs/async-fence-fix-validated-2026-08-22.md`) — real numbers at the
time, not representative of current production throughput.

</details>

**Known operational constraint:** Thunderbolt RDMA degrades under repeated
rapid restart/teardown cycles — GPU power asymmetry (~7W vs ~20W) and
throughput crashing to ~130-165 tok/s after ~10 rapid relaunches. A full
reboot of both nodes clears it completely. Check GPU power symmetry
(~45-52W both nodes) as a cheap canary before trusting any throughput
number gathered after several rapid cluster restarts.

---

## 2. MoE / all_sum collective — the dominant cost center

This is the single most-investigated area in the repo — at least 20
distinct docs, spanning 2026-05-14 through 2026-08-21. **Read this section
in full before touching all_sum/collective code again.**

### 2.1 Establishing it's the dominant cost (WIN, 2026-08-19)

`docs/moe-all-sum-dominant-cost-2026-08-19.md` — NOP-ablation (file-toggle
identity pass-through on `moe.all_sum`) confirmed at two depths:
- 12,066-12,069 tok: baseline 162.5 tok/s vs NOP(all_sum off) 427.5 tok/s
  = **2.63x speedup, all_sum share 62.0%**
- 38,066-38,067 tok: baseline 167.3 tok/s vs NOP 430.3 tok/s = **2.57x,
  share 61.1%**

Consistent at two depths, ruling out cold-start confounds. **A prior
skill-note claim of "~12% comms cost, ~0% at high context" was wrong by a
large margin** — always re-measure inherited cost-attribution claims, don't
trust them.

Independently reconfirmed 2026-08-18 in `docs/dsv4-220k-prefill-span-profile-2026-08-18.md`:
span breakdown at 220K context showed `moe.all_sum` = 9.5% of wall
(narrower window, real 220,321-token prefill, attn 58.4% + ffn 41.6% with
moe.switch_mlp 26.9%) — the percentage varies by measurement window/depth
but the collective is consistently large.

Reconfirmed again 2026-08-21 (this project's own session,
`docs/moe-allsum-collective-cost-confirmed-2026-08-21.md`): sync-span
decode-isolated measurement (SIGUSR1 mid-decode) showed **moe.all_sum =
21.4% of decode wall time** — higher than the blended prefill+decode
14.4% figure from an earlier same-day measurement, confirming decode-only
windows show a bigger share than blended windows.

**CORRECTION (2026-08-22, same continued session): both the 21.4% and
14.4% figures above are very likely methodology artifacts, not real
per-collective cost shares.** A later same-session arithmetic
reconciliation (§2.7, `docs/allsum-sync-span-artifact-arithmetic-check-2026-08-22.md`)
found that the underlying ~4094µs/call sync-span average is
mathematically impossible as a real per-call cost — 43 layers ×
4094µs = 176ms/token, but real measured decode wall time is only
53.48ms/token (a 3.29x impossibility: the collective alone would have
to consume 3.29x the ENTIRE token budget). Real jaccl-internal
`steady_clock` timing (same section) instead measured a genuine
median of 36µs/call, accounting for only ~2.9-5.3% of real wall time.
The likely mechanism: `mx.synchronize()` at a sync-span boundary
drains MLX's ENTIRE pending lazy graph (all upstream unevaluated
compute since the last sync point), not just the spanned op — so a
span ending right after `all_sum` misattributes real upstream
GPU-compute time to the collective. Treat the 21.4%/14.4%/9.5%
sync-span-derived percentages in this section as upper-bound
measurement artifacts, not actionable per-collective cost shares; the
real, trustworthy figure is the jaccl-internal one in §2.7.

### 2.2 The "178ms/call" artifact (2026-08-20) — measurement trap

`docs/moe-all-sum-178ms-artifact-real-bottleneck-2026-08-20.md`: a
previously-reported 178ms/call figure for all_sum was **itself a
measurement artifact**, not the true collective cost — real collective
cost is closer to ~12ms/call. **Always verify whether a large measured
per-call cost reflects the operation itself or includes queuing/waiting
artifacts before optimizing the wrong thing.**

### 2.3 GPU-stream-boundary decomposition (WIN, 2026-08-20)

`docs/phase0a-allsum-boundary-decomposition-2026-08-20.md` — isolated that
the GPU→CPU stream-boundary coherency cost (required because MLX
collectives are CPU-stream-only) is **payload-proportional, not a fixed
drain bubble**, and is **NOT collective-specific** — a plain non-collective
CPU-stream op reproduces the same 2.66x penalty. Boundary cost linear at
~7 GB/s (1.05MB→0.242ms, 67.11MB→8.891ms). `MLX_EVENT_WAIT_POLL_US` sweep:
flat, no effect. `MLX_METAL_FAST_SYNCH=1`: **1.5x SLOWER with 70x more
variance** — dead end, don't retest. Production 16.8MB call: ~2.4ms
boundary + ~0.9-1.7ms CPU work + ~2-9ms wire matches the prior 5-12ms/call
band.

**Reusable lesson: when attributing a stall to a specific op (e.g. a
collective), test whether a non-collective op on the SAME stream
reproduces the same cost before concluding the collective itself is
expensive.**

### 2.4 Skew vs. wire cost (WIN, 2026-08-19)

`docs/moe-all-sum-skew-vs-comms-2026-08-19.md` — determined via arithmetic
+ source reading (no cluster relaunch) that the real bottleneck at the
time was a **chunking config knob**, not rank skew or bandwidth:
- 16.8MB all_sum payload: 178ms @ 12K tok depth, 170ms @ 38K tok depth
  (flat with depth — rules out rank imbalance, since cost doesn't scale
  with depth and machines are symmetric)
- Effective bandwidth only ~92-96 MB/s vs Thunderbolt5/jaccl realistic
  ~6-10 GB/s (bytes-on-wire explain only ~1.5% of cost)
- `MLX_JACCL_RELIABLE_MAX_SZ=2` → chunk=15.9KB → ~1029 chunks/call → 258
  stop-and-wait rounds; 690us/round cross-checks independently to ~94 MB/s
- Projected fix: sz=4 → chunk=63.9KB → ~45ms/call; sz=7 → ~6ms/call

**BUT sz≥4 has a documented hang risk on Apple's librdma** — flag any
chunk-size widening as needing careful A/B, not blind widening.

**Follow-up test (NEGATIVE, 2026-08-20):** `docs/jaccl-sz3-tested-no-improvement-2026-08-20.md`
— bumped `MLX_JACCL_RELIABLE_MAX_SZ` from 2→3 (32KB, still inside the safe
zone below sz≥4's hang risk). Result: **no measurable throughput change**
(~166-169 tok/s vs baseline's ~162-172 tok/s, statistically identical).
**Real clean null result — do not re-attempt sz=3 without new evidence.**
The bottleneck likely has a fixed per-round latency floor or is dominated
by a per-call barrier cost (e.g. `ack_sync_pre`), not chunk count.

### 2.5 Quantized all_sum — the most-attempted, most-failed lever (NEGATIVE, closed 2026-08-19)

A multi-doc chain (`moe-allsum-quant-phase0-repro`, `moe-allsum-sharedscale-root-cause-found`,
`local-absmax-fence-artifact-confirmed`, `moe-allsum-sharedscale-CORRECTED-final`,
`moe-allsum-quant-root-cause-and-closure`, `moe-allsum-quant-live-test-failed`,
`moe-allsum-sharedscale-live-test-no-speedup`, `moe-allsum-quant-compute-overhead-analysis`,
all 2026-08-19) tried replacing the bf16 `moe.all_sum` payload with a
quantized (int8, shared-scale) version to cut wire bytes.

**Timeline of the investigation (a good example of iterative
self-correction):**
1. Initial live test: quantized all_sum **hung the collective on the
   first real prefill request** (GPU event timeout, peer rank stuck on an
   abandoned c≥2 collective) — `docs/moe-allsum-quant-live-test-failed-2026-08-19.md`.
2. Root-cause attempt #1 blamed `local_absmax` reduction costing
   ~400-420ms/call (`moe-allsum-sharedscale-root-cause-found`) — **later
   found to be a probe-fence artifact**: `mx.eval()` fences flush the
   ENTIRE pending lazy graph, not just the requested output, so a probe's
   first `mx.eval()` after an unevaluated backlog silently absorbs
   upstream cost (`docs/local-absmax-fence-artifact-confirmed-2026-08-19.md` —
   real reduction cost only ~0.4ms vs backlog-flush ~57-60ms, ~130x ratio).
3. Corrected final measurement (`moe-allsum-sharedscale-CORRECTED-final-2026-08-19.md`):
   an EARLIER version of this same investigation had wrongly claimed a
   ~148x speedup by misreading the log tail as prefill when it was
   actually decode. Real corrected numbers: **PREFILL: shared-scale
   ~265ms/call vs baseline unquantized 178ms/call (~1.49x SLOWER)**;
   DECODE: shared-scale ~1.1-1.6ms/call (fast, but decode's collective is
   already tiny so this doesn't matter). Real prefill throughput
   ~168-169 tok/s ≈ baseline ~162-172 tok/s — **no speedup**.
4. Structural closure (`docs/moe-allsum-quant-root-cause-and-closure-2026-08-19.md`):
   `moe.all_sum` is a true cross-rank **partial-sum reduction**, not
   gather-semantics — the zero-padded int8 trick that works for GATHER
   collectives (like the seq-split reconstruction, see §8) mathematically
   **cannot** apply to a genuine elementwise sum reduction. Also: jaccl's
   `all_gather` lacks the reliability layer `all_sum` has, explaining the
   original crash when substituting two all_gathers.
5. Compute-overhead check (`moe-allsum-quant-compute-overhead-analysis-2026-08-19.md`):
   even setting aside the above, GPU-side quantize/dequant compute
   overhead at production hidden_size was found negligible relative to
   the (moot) wire savings.

**Reusable lesson: before attempting a quantized/reduced-bytes redesign
of a collective, verify whether it's a genuine GATHER (disjoint-slice
combine, where zero-padding tricks work) or a genuine elementwise
partial-sum REDUCTION (where they mathematically cannot) — this is a
structural, not engineering, constraint. Also: correlate probe timestamps
against an independent ground-truth log signal before trusting a
prefill-vs-decode phase split in mixed data — a superficial "tail of log
looks fast" read produced a false, more exciting conclusion here twice.**

**Do not re-attempt quantized moe.all_sum without a fundamentally
different mechanism** (this specific approach — shared-scale int8 payload
reduction on the existing all_sum primitive — is conclusively closed).

### 2.6 Comm/compute overlap — the real remaining lever (WIN, multi-phase, 2026-08-20/21)

This is the one direction that DID pan out, across a careful multi-phase
validation:

**Phase 0b (WIN, 2026-08-20)** — `docs/phase0b-collective-overlap-gate-2026-08-20.md`:
proved overlap is structurally achievable. `all_sum` runs on a CPU stream
in MLX (no Metal `eval_gpu` exists for it); overlap is destroyed only by
ordinary same-GPU-stream FIFO ordering, **not a device-wide barrier**.
Probe1: COMPUTE_ONLY 77.75ms, COMM_ONLY 36.93ms, BOTH 77.34ms,
overlap_ratio=0.995 (~33% of a 115ms serial budget recovered). Probe4
(2nd GPU stream): near-perfect overlap (b/max=1.002-1.008) vs same-stream
serialization (b/max=1.2-1.43). **Escape hatch: either pre-evaluate the
collective's input before issuing overlapping compute, OR issue
overlapping compute on a dedicated `mx.new_stream(mx.gpu)`. Doing neither
gives ~0% overlap and LOOKS EXACTLY LIKE (but isn't) a hardware drain —
likely why prior sessions wrongly concluded collectives "block
everything."**

**Phase 0c (correctness gate, MIXED/critical, 2026-08-20)** —
`docs/phase0c-collective-order-determinism-2026-08-20.md`: before
building any overlap, established the correctness constraint. 6 scenarios
× 20 trials: `same_order`/`interleaved`/`async_eval_same`/`issue_skew` all
PASS; `async_eval_skew` and `eval_arg_skew` **FAIL with 100% deterministic
silent corruption** (no crash/hang, just cross-paired wrong-rank tags).
**MLX collectives are matched positionally by EVAL order, not Python
program order — any rank-dependent branching or divergent `async_eval`
call order across ranks produces silent, deterministic wrong results
with no crash/hang/NaN signal. Any overlap/pipelining design MUST
guarantee a rank-invariant eval schedule.**

**Lever 1 (small-M headroom check, NEGATIVE, 2026-08-20)** —
`docs/lever1-moe-smallm-headroom-2026-08-20.md`: checked whether the
gather_qmv_rhs/gather_qmm_rhs small-M kernels have headroom vs an ideal
dense grouped-GEMM ceiling. Live kernels already **match or beat** the
idealized ceiling at production per-expert token counts (0.79x at L=512,
0.63x at L=1024 meaning live is 1.6x FASTER than the naive ceiling, 1.07x
at L=2048). `MLX_GATHER_QMV_RHS=0` ablation changes production shape
<4%. Widening `MAXBE=64` **regressed -33% at L=1024, -78% at L=2048**
vs default MAXBE=6. **No kernel-level lever here — bottleneck is
weight-streaming-bound arithmetic intensity from the routing
distribution, not kernel inefficiency.**

**Lever 2 (sequence-chunk pipelining, MIXED, 2026-08-20)** —
`docs/lever2-seqchunk-overlap-2026-08-20.md`: layer-pipelining is
**invalid** (RMSNorm inter-layer dependency breaks it structurally);
sequence-CHUNK pipelining (overlap chunk A's all_sum with chunk B's
independent same-layer compute) is the valid axis instead. Real but
modest overlap of **~1.1-1.15x**, noisy on the (laptop) test environment.

**Prefill-chunk-overlap live cluster test (INCONCLUSIVE, 2026-08-20)** —
`docs/prefill-chunk-overlap-live-test-2026-08-20.md`: deployed the real
TP-native compute/comm overlap mechanism (commit `b7aa41920`) live. No
crash/hang/HTTP errors, but **a genuine correctness anomaly appeared on
one of two trials** — flagged, not shipped as-is.
`docs/prefill-chunk-overlap-race-fix-2026-08-20.md`: root-caused and fixed
a cross-stream ordering race, but **the throughput lever itself was
declared DEAD/CLOSED** — fixing the correctness bug did not resolve into
a real performance win (`docs/prefill-optimization-campaign-handoff-2026-08-18.md`
closes this line).

**GPU utilization reconciliation (INFO_ONLY, 2026-08-19)** —
`docs/gpu-util-vs-allsum-cost-reconciled-2026-08-19.md`: explains why
96-97% GPU utilization telemetry and all_sum costing 61-64% of wall time
LOOKED contradictory but aren't — GPU-busy metrics measure occupancy
(including blocked-but-scheduled submission threads during a collective
wait), not useful compute throughput. **Don't conflate occupancy with
achieved compute.**

**Phase 3 real-cluster validation (INCONCLUSIVE, blocked, 2026-08-20)** —
`docs/phase3-cluster-validation-blocked-resolved-2026-08-20.md`: a
standalone active RDMA overlap microbenchmark failed because the
cluster's only two RDMA interfaces were already held by production
runners. **Resolved by using PASSIVE instrumentation of live production
instead of standalone active probing** — on hardware with limited
physical RDMA interfaces already claimed by production, don't attempt
standalone concurrent RDMA sessions. Theoretical ceiling for perfect
overlap: ~10.5% end-to-end speedup best case (from a real 9.5%-of-wall
measurement), realistically 5-7% after imperfect overlap/scheduling
overhead — **flagged that this modest a ceiling may not justify a risky
live deployment.**

**Confirmed already implemented and active (2026-08-21, this project's
own session)** — `docs/comm-compute-overlap-already-exists-2026-08-21.md`:
discovered `EXO_DSV4_FENCE_ASYNC` (uses `mx.async_eval(y)` instead of
blocking `mx.eval(y)` after the collective, two-owner-gated for cross-rank
safety) was already implemented by a PRIOR session (2026-07-02) and
already `=1` in production — this IS the comm/compute overlap design.
Historic comment claims **+28% decode (28.9→37.0 t/s)** from 2026-07-02;
this session's clean re-A/B on the current baseline measured only
**+1.04%** (18.664 vs 18.471 tok/s, n=8 each side) — a real, consistently
positive, but much smaller effect than claimed.

**RESOLVED (2026-08-22, same continued session, read-only, zero cluster
risk)** — `docs/fence-async-28pct-claim-traced-to-artifact-2026-08-22.md`:
traced the +28% claim to its origin commit (`mlx-lm` `1e808319f`,
2026-07-02), which cites **"MTP-PROF"** as its measurement tool.
MTP-PROF's own code comment (`dsv4_mtp.py`) explicitly states: "brackets
the draft/verify/accept phases with `mx.eval` + `perf_counter`...
inserts evals at phase boundaries which serialises pipelining —
**measurements are upper bounds on real production walls**." This is
the SAME methodology-artifact class conclusively proven this session
for the sync-span profiler (§2.7 arithmetic reconciliation) — forced
per-boundary synchronization destroys real async pipelining and
inflates measured cost. The historic +28% figure is now understood to
be an inflated upper-bound artifact, not a real number this session's
clean +1.04% A/B should have been expected to reproduce. **Internally
consistent with the real transport-cost finding**: perfect all_sum
overlap can recover at most ~2.9-5.3% of wall time (real measured
transport cost, §2.7) — a real +1.04% capturing a meaningful fraction
of that small ceiling is coherent; the old +28% claim against a
collective that costs only ~3-5% of wall time was implausible on its
face. No further live-cluster overlap investigation pursued this
session given the now-small remaining ceiling (≤2% residual upside)
does not justify relaunch risk with no one available to monitor it.

**Older, harder OPT-7 attempt (NEGATIVE, tested + reverted twice)** —
gating the per-layer `mx.eval` on `_fence_every_n` (rather than doing
`async_eval` at every layer) was tried and reverted: **made B=2 prefill
23% SLOWER (111 vs 144 tok/s)**. Referenced repeatedly across multiple
docs (`docs/gpu-util-vs-allsum-cost-reconciled-2026-08-19.md`,
inline code comments in `deepseek_v4.py`) as a settled negative — "without
the per-layer eval, MLX builds a larger lazy graph that's more expensive
to evaluate at the fence point than incremental evals; the overlap
benefit doesn't materialize, the graph accumulation cost dominates."
**`EXO_DSV4_FENCE_EVERY_N_LAYERS` is dead/unused config as of 2026-08-21**
(the OPT-7 mechanism that consumed it was reverted; the variable is set
but never read).

### 2.7 Other all_sum/collective investigations

- `docs/moe-vs-dense-qmm-isolation-2026-08-19.md` — isolated qmm kernel
  cost between MoE and dense configs (INFO_ONLY).
- `docs/moe-quant-vs-bf16-dequant-attribution-2026-08-19.md` — attributed
  overhead split between quantized compute path and bf16 dequant step
  (MIXED).
- `docs/moe-per-stage-gpu-breakdown-2026-08-18.md` — per-stage GPU time
  breakdown for MoE forward (gather/qmm/allsum) (INFO_ONLY).
- `docs/moe-gpu-time-overlap-bandwidth-bound-2026-08-19.md` — concluded
  MoE GPU time is bandwidth-bound, not compute-bound (INFO_ONLY).
- `docs/moe-all-sum-payload-size-causal-test-2026-08-19.md` — causal test
  varying payload size (INCONCLUSIVE).
- `docs/mxfp4-gather-qmm-rhs-lhs-kernel-2026-08-19.md` — MXFP4 quantized
  gather+qmm RHS/LHS operand-layout kernel work (MIXED).
- `.hermes/plans/2026-05-19_allsum_tail_findings.md` (NEGATIVE) — a
  chained-collective peer-CQE-arrival-tail hypothesis was **falsified**:
  forcing per-layer fences (`FENCE=1`) made the tail WORSE (verify mean
  63.16ms vs production `FENCE=43`'s 57.10ms, +10.6%), the opposite of
  the prediction. True cost is uniform per-layer all_sum drain (~1.4ms ×
  43 layers); earlier apparent 35% spread was iter-0 cold-compile-cache
  contamination. **Design hypothesis-falsifying experiments — a result
  that's the OPPOSITE of the prediction is strong evidence the mechanism
  is wrong, not just inconclusive.**
- `.hermes/plans/2026-05-19_phase_f_findings.md` (NEGATIVE) — invalidated
  an entire plan's premise that jaccl/RDMA ACK-barrier optimization would
  meaningfully help decode: jaccl poll wall was only 1.9% of verify cost
  (mean 8.15us, median 5.00us) vs ~98% attributable to the `mx.eval`
  fence itself (median p50=37.4ms). **Don't assume the collective/RDMA
  layer is the bottleneck without direct per-call instrumentation.**
- Session-own 2026-08-21 work (this project): real Instruments Metal
  System Trace measured **CPU-to-GPU dispatch latency = 96.8µs ± 4.6µs**
  (n=19, steady-state matmul), cross-validating two independent prior
  estimates (a MiniMax fusion docstring's "~100-200µs" and this session's
  own roofline-inferred "~150µs"). Real jaccl `allreduce_bench` isolated
  microbenchmark: raw hardware floor at the exact 8KB decode message size
  = **~120µs**, vs in-model sync-span average of ~4094µs — a **34x gap**
  confirming the cost is NOT the RDMA wire transport but overhead in the
  model's call context (skew and/or CPU-dispatch/scheduling around the
  collective). See `docs/offline-collective-microbenchmark-2026-08-21.md`
  and `docs/instruments-metal-trace-real-dispatch-latency-2026-08-21.md`.
- **Live two-rank Instruments trace of real production decode (WIN, new
  session, 2026-08-21)** — following an independent Fable review of this
  doc that ranked a live two-rank trace as the single highest-EV next
  step, traced BOTH TP ranks simultaneously (not a synthetic probe) via
  `xctrace --attach` on the real running runner PIDs during a real
  decode request. Found and worked around a real `xctrace` bug along the
  way: `--toc` and `remodel` fail with "Missing Template Error" on
  attach-mode trace packages even though the trace genuinely contains
  full Metal GPU data — direct `--xpath` export using a schema name
  known from a working launch-mode trace succeeds cleanly regardless.
  **Real measured GPU occupancy: 30.4% (rank0) / 28.4% (rank1), i.e.
  ~70% genuine GPU idle time** — independently corroborates the earlier
  roofline estimate (~12% of theoretical peak) using a completely
  different instrument (direct Metal telemetry vs. architectural FLOPs
  math). Idle gaps in the 1-10ms bucket cluster tightly on both ranks
  (mean ~2909-3010µs), tentatively flagged at the time as suggestively
  close to the `moe.all_sum` sync-span average (~4094µs) — **this
  tentative correlation was tested directly in the very next step below
  and found WRONG: the real jaccl transport call itself is far too fast
  (median 36µs) to be the primary contributor to a ~3ms-scale GPU-idle
  gap.** Known limitation: the two captures used independent,
  non-synchronized system clocks, so true cross-rank wall-clock gap
  correlation was not attempted this session (only aggregate per-rank
  statistics, which don't require synced clocks). See
  `docs/live-decode-two-rank-instruments-trace-2026-08-21.md`.
- **jaccl-internal timing decomposes the moe.all_sum 34x gap: transport
  is fast, overhead is elsewhere (WIN, decisive, same session,
  2026-08-21/22)** — directly answers the open thread above and open
  thread #6 in §13. Added real `std::chrono::steady_clock` timing
  INSIDE jaccl's C++ transport call itself (immune to MLX/Python-level
  graph laziness, unlike a `perf_counter`-around-call-site approach an
  earlier review correctly rejected as methodologically unsound — see
  the rejected-approach note in §13). Took 5 relaunches to get right,
  each one a real, separately-diagnosed bug, not repeated guessing:
  (1) `MeshGroup` instrumented correctly but produced zero output; (2) a
  wrong-class detour to `RingGroup` based on an unconfirmed premise
  about which jaccl `Group` subclass the topology uses (later confirmed
  `MeshGroup` was right all along — `MLX_JACCL_RING` unset in
  production means `prefer_ring` is false); (3) real bug: `MeshGroup`'s
  trace-file-open gates on `JACCL_TRACE_CALLS`, not the new
  `JACCL_TRACE_TIMING`, so a relaunch that set only the latter silently
  wrote nothing — confirmed both new symbols WERE present in the
  rebuilt `.dylib` via `nm` before assuming a code bug; (4) real bug:
  cleanup `rm -f` on the trace file path between a relaunch and the
  benchmark run unlinked a file the long-lived runner process had
  already opened at construction time (opens once per process
  lifetime, not per-call) — classic Unix unlink-while-open, silently
  orphaning ~1.68MB of real writes into an inode unrecoverable on macOS
  (no `/proc`). **Real result once correctly measured: 45,666 real
  decode-time 8192-byte `moe.all_sum` transport calls, median 36.1µs
  (rank0) / 36.0µs (rank1), mean 66.3µs / 58.9µs — FASTER than the
  earlier isolated microbenchmark's ~120µs wire floor, and only
  0.04% of calls exceed the ~4094µs sync-span-measured average.** The
  jaccl transport is conclusively NOT the software-overhead bottleneck
  — the 34x gap lives almost entirely in whatever surrounds the call
  (MLX's `mx.eval` fence, CPU/GPU dispatch coordination, or
  Python-level scheduling), not the RDMA collective itself. Reusable
  lesson: never `rm` a log file a long-lived process has already opened
  — no error signal, no recovery path on macOS, silently orphans all
  future writes. See
  `docs/jaccl-internal-timing-allsum-transport-fast-2026-08-21.md`.
- **Arithmetic reconciliation: the sync-span 21.4%/14.4%/4094µs figures
  were methodology artifacts, not real costs (WIN, decisive, offline,
  zero cluster risk, same session, 2026-08-22)** — per an independent
  Fable review's suggestion to check the sync-span figures against real
  measured wall-clock arithmetic BEFORE attempting any further live
  instrumentation (production uses non-blocking `mx.async_eval`, so a
  naive timing wrapper around it would be uninformative, and the user
  had gone to bed with no one available to approve/monitor a further
  relaunch). Real check: 43 layers × the sync-span-measured ~4094µs/call
  average = **176ms/token predicted**, vs. **53.48ms/token real
  measured wall time** (18.7 tok/s baseline) — a 3.29x impossibility,
  since a per-call cost cannot exceed the total per-token budget it's
  supposedly a component of. Using the REAL jaccl-internal-measured cost
  instead (previous entry, 36-66µs/call): 43 × 36-66µs = 1.55-2.85ms/token
  = a plausible 2.9-5.3% of real wall time. **Conclusively corrects the
  earlier "moe.all_sum = 21.4%/14.4% of decode wall time" claims (§2.1)
  as sync-span methodology artifacts** — `mx.synchronize()` at a span
  boundary drains MLX's ENTIRE pending lazy graph since the last sync
  point (not just the spanned op), so a span ending right after
  `all_sum` misattributes real upstream GPU-compute time from prior
  layers to the collective specifically. Also explains sync-span
  profiling's own previously-documented overhead (~15% prefill/~77%
  decode, §12) as a direct consequence of this per-layer forced-drain
  destroying pipelining, not just adding a flat measurement tax.
  **Reusable lesson: before trusting a sync-span/forced-synchronization
  percentage breakdown, sanity-check it against real end-to-end
  wall-clock arithmetic (n_events × measured_avg_cost vs. total real
  wall time) — an impossible ratio (predicted > actual) is a strong,
  free, zero-risk signal that the measurement technique is
  misattributing cost, not that the op is actually expensive.** See
  `docs/allsum-sync-span-artifact-arithmetic-check-2026-08-22.md`.

### 2.8 The async fence was permanently, silently broken — root cause found and FIXED (WIN, decisive, 2026-08-22)

**This is the single biggest real throughput finding in this entire
document.** `EXO_DSV4_FENCE_ASYNC=1` had been live in production
config since 2026-07-02 — but the gate requiring its two owner flags
(`"engine"` + `"cache"`) to both be `True` was structurally unsatisfiable
under this cluster's TP sharding mode: `"cache"` is owned exclusively
by `dsv4_mtp.py`'s `DSv4MTPPredictor`, MTP/DSpark-specific code that is
never instantiated under TP (confirmed dead, see §12's DSpark finding).
The fence had therefore ALWAYS fallen back to the blocking `mx.eval(y)`
path for the entire multi-session optimization campaign, regardless of
the flag's value — meaning every decode number in §1's original
baseline table, and every decode measurement anywhere in this document
before 2026-08-22, was taken under this silently-broken condition.

Found via a real in-process Python stack sampler (built as a
zero-privilege alternative to `py-spy`, which needs root/sudo
unavailable on this cluster): ~95% of the compute thread's real
decode-time wall time was spent blocked in `mx.eval(y)`. Live
diagnostic logging (`EXO_DSV4_FENCE_GATE_DIAG=1`) confirmed the
mechanism directly: zero `"cache"` setter calls logged across a real
request.

**Fix** (`docs/async-fence-cache-owner-dead-code-root-cause-2026-08-22.md`):
a registration-based, fail-closed gate — an owner key is only REQUIRED
to be `True` if something has actually registered as its live owner.
`"engine"` always registers (unconditional request-lifecycle code).
`"cache"` now only registers when `DSv4MTPPredictor.__init__` actually
succeeds (i.e. MTP/DSpark is genuinely active) — preserving the
original two-owner safety property exactly for that config, while
letting the fence arm on `"engine"` alone when MTP/DSpark is
structurally absent. Rejected a simpler getattr/hasattr
structural-sniffing alternative per an independent Fable design review
— that approach fails OPEN (silently drops the safety check on an
unrelated rename/refactor), unacceptable for a subsystem with a real
documented corruption history (the 2026-07-02 c=2 stream-join bug).

**Real, validated result** (`docs/async-fence-fix-validated-2026-08-22.md`):
decode throughput went from ~18.5 tok/s to **29.2-30.9 tok/s — a real
+58-67% improvement**, confirmed across two context depths (512-tok and
2000-tok prompts), with output correctness validated via three
independent checks including an exact-match needle-in-haystack test
(given this subsystem's history of fast-but-corrupted output under
related bugs). This single fix is larger than every other decode-side
throughput win found in this document's entire history combined.

**Locked in as the new baseline (§1), including at real depth**: a
full needle-verified 100K/300K/500K re-benchmark confirmed decode
+23-54% at depth too (17.48→26.91 @100K, 18.60→24.44 @300K,
17.26→21.51 @500K) — the fix's benefit shrinks somewhat at deeper
context (the fixed-cost prefill/transition window, still using the
blocking fence by design, becomes a larger fraction of total time
relative to the growing KV-cache decode cost) but remains large and
real at every depth tested. Prefill confirmed unaffected (within normal
run-to-run variance of the pre-fix numbers), exactly as the fix's
design predicted.

**Reusable lesson**: a feature flag being `=1` in production config
does not mean the feature is actually active — always verify the real
runtime GATE state (not just the top-level env var) with live
diagnostic logging before trusting a flag's documented behavior,
especially for any multi-owner/multi-condition gate.

**Triple-confirmed via a fresh dual CPU+GPU capture (2026-08-22, Phase
C of the post-fix investigation)** —
`docs/phase-c-dual-capture-confirms-fix-2026-08-22.md`: since the
original idle-gap data (§12) was captured under the broken fence, a
fresh simultaneous CPU-side (in-process Python sampler) + GPU-side
(Instruments Metal trace) capture was run on the fixed baseline. **Real
post-fix GPU occupancy (request-window-isolated): 85.42%** — up from
~28-30% pre-fix, a massive direct hardware-level confirmation. **Real
CPU-side sampler**: the compute thread's dominant hot-line genuinely
shifted from the blocking `mx.eval(y)` call (was 67% pre-fix) to the
intended non-blocking `mx.async_eval(y)` call (45.75% post-fix; the
blocking fallback is now only 13.19%, matching the design intent of
firing only during prefill/transitions). **Three independent
measurement methods — throughput benchmark, GPU hardware trace, and
CPU-side stack sampling — all converge on the same conclusion**, using
genuinely different instrumentation. This is about as decisively
validated as a single fix can get.

---

## 3. Prefill throughput

### 3.1 Major wins

**hc_expand fused Metal kernel (WIN, 2026-08-24)** —
`docs/hc-expand-kernel-ab-2026-08-24.md`. Env-gated fused Metal kernel
for `HyperConnection.expand` (used inside every layer's `attn_residual`
and `ffn_residual`); previously the op path was measured at 4.4% of
prefill wall time at 220K real tokens
(`docs/t10-final-decomposition-closed-2026-08-22.md` Check 1). Laptop
microbench: 8.66x faster than the op path at true prefill shape
`[1,2048,4,4096]` (3645µs → 421µs). Live 2×2 A/B on the cluster at
~70.5K real tokens (`bench/phase3_precheck_depth_throughput.py`,
`target_tokens=100000`), production env (`EXO_SPECULATIVE=0
EXO_DSV4_MTP=0 EXO_DSV4_DSPARK=1`):
- Arm A (kernel OFF, env absent): mean **359.89 tok/s** prefill (355.03,
  364.75; arm A r2 landed at the P5 known-good baseline of 366.5)
- Arm B (kernel ON): mean **373.80 tok/s** prefill (373.18, 374.43;
  0.33% spread)
- **Δ = +13.91 tok/s (+3.87%) mean-to-mean**; pairwise +5.11% / +2.65%;
  even the conservative "worst B vs best A" bound is +2.31% —
  above the pre-registered +1.5% ship threshold in every framing
- Quality: needle FALCON-MERCURY-7749 recovered exact on all 4 probes,
  no U+FFFD, no BOS spam, `finish_reason=stop`
- Kernel path is fp32-accumulate + single-cast-to-output-dtype at the
  end (mean rel err 2.77e-7 vs the reference op path, laptop-measured
  — fp32-exact class, NOT the bf16-comb variant rejected earlier for
  1.08% mean rel err in `docs/hc-expand-rejection-relitigated-multiseed
  -2026-08-22.md`; that one is dead, this is a different mechanism)
- The measured +3.87% matches the `span_share × kernel_reduction`
  prediction (4.4% × 7/8 = 3.85%) almost exactly — first e2e prefill win
  since the 2026-06-24 breakthrough that isn't a NULL/dead-end
- **Default flipped 2026-08-24** in `start_cluster.sh`
  (`: "${EXO_DSV4_HC_EXPAND_KERNEL:=1}"`); cluster now serves production
  with kernel ON. Reversion recipe: `EXO_DSV4_HC_EXPAND_KERNEL=0
  ./start_cluster.sh` gives the pre-kernel op path bit-identical (arm A
  of the A/B). SHAs: exo `deb1c8a6d` (env-forwarding +
  default-flip landed same-day, previous SHA `e3df799c0` had only the
  forwarding), mlx-lm `7a1a4e8` (unchanged this session; the kernel
  itself was shipped by a prior worker via exo `ecce148ff`
  submodule-bump)
- Not tested at deeper context (300K, 500K) or under PP mode — expected
  to transfer since the op is per-layer-per-token and the code path is
  shared, but not verified
- **Depth verification (2026-08-24, INCONCLUSIVE)** —
  `docs/hc-expand-depth-verification-2026-08-24.md`. Live A/B on the
  same production config, at 300K target (2 pairs) and 500K target
  (1 pair): Δ = **+1.28% @300K** (356.72 ON vs 352.21 OFF mean),
  **+0.85% @500K** (336.74 ON vs 333.91 OFF, n=1 per arm). BOTH
  DEPTHS INSIDE ±1.5% — pre-registered NEUTRAL/INCONCLUSIVE per the
  task brief. No regression flagged (neither trips −1.5%). Needle
  recovered exact on all 6 probes. Direction matches the mechanistic
  prediction (smaller relative share of the op as SDPA/indexer grow
  with depth) but magnitude is below the predicted +2.5%–4% range at
  300K — either the 70.5K span share overestimated the op's
  contribution at depth, or the noise floor at these depths (~±0.7%
  within-arm at 300K, unbounded at 500K with n=1) hides a real ~+2%
  effect. **The 70.5K ship decision is not undermined**; the kernel
  is neutral/safe at depth in this sample. Default stays ON;
  production restored to kernel-ON and depth-scale-verified with a
  final 211K-token production probe (ON@300K r2, real 211,022
  tokens, coherent needle-recovery decode). One anomaly recorded:
  first ON@500K probe hung client-side after successful server-side
  prefill; re-run cleanly. n=1 at 500K is this test's biggest
  limitation.

**Reusable lesson: a small-share prefill op (single-digit % of wall
time) with a large per-op inefficiency (~8x) can still be worth
shipping as a fused kernel; the span-share × per-op-reduction math
predicts the e2e win within noise, so triage kernels by that product
BEFORE writing the fused version. Also: bit-identity when disabled is
what makes an env-gated fused kernel safe to ship default-on — the
gate itself must remain and default-off must be provably equivalent
to the pre-kernel code path (verified here max_abs=0.0 laptop-side,
and the live A/B arm A independently confirmed by matching the P5
same-code baseline).**

**Prefill throughput breakthrough (WIN, 2026-06-24)** —
`docs/prefill-throughput-breakthrough-2026-06-24.md`: five stacked fixes.
- c=1 500K prefill: before 167 t/s avg (crossed below 200 at ~250K) →
  after **251 t/s avg, never below 200 through 500K**
- c=2 500K prefill: before sequential (not concurrent) → after **317
  tok/s aggregate (B=2)**
- Indexer weight fold (OPT-6): **64x compute reduction** (130 GFLOP→2
  GFLOP/chunk at 380K), bit-equivalent (max diff 6e-5 fp32)
- Command-buffer fix (`MLX_MAX_MB_PER_BUFFER` 50→200MB): eliminated
  bimodal stalls, B=2 aggregate 144→317 t/s (+120%), per-chunk 1.78s avg
  (bimodal 0.77/2.3s) → steady 0.81s, 100% fast chunks (was 32%)
- **Reverted OPT-8** (broadcast sorted `gather_qmm_rhs` extended to
  prefill): fast chunks hit 349 t/s individually but average unchanged at
  142 t/s due to 3.2GB/chunk broadcast allocation causing bimodal stalls

**Reusable lesson: a persistent bimodal fast/slow timing pattern with
IDENTICAL memory between fast and slow runs rules out memory-pressure
causes; a consistent Nx ratio matching per-layer op count points to GPU
command-buffer scheduling. Apple Silicon's default `max_mb_per_buffer`
(50MB on M4 Max 's' variants) can trigger non-deterministic mid-forward
command-buffer flushes under large batched forwards — raise
`MLX_MAX_MB_PER_BUFFER`/`MLX_MAX_OPS_PER_BUFFER` to fix.**

**MoE kernel handoff (WIN, ~2026-07-01)** — `MOE_KERNEL_HANDOFF.md`:
- Prefill 100K: **255→353 tok/s (+38%)**; 495K: ~200→306 tok/s; 727K: 215
  tok/s; decode 29.0 tok/s mean
- Eliminated the 340K prefill cliff (270→40 tok/s) by replacing the
  indexer's O(P log P) **argsort with argpartition**
- Breakdown: argpartition alone 255→289 tok/s, +lm_head last-row →295,
  +chunk256 →353
- Falsified alternatives (don't retry): chunk 512 lost to OPT-4 tiling
  overhead; gamma=3 decode didn't pay off (25.2 vs 29.0 tok/s); qmm tile
  retuning falsified (all alternatives worse); fused topk kernel proven
  numerically incorrect via bitexact test

**SEQ_SPLIT re-validated at long context (WIN, 2026-08-18)** —
`docs/dsv4-220k-prefill-seqsplit-ab-2026-08-18.md`: SEQ_SPLIT=1 220,318
tok/614.5s = **358.6 tok/s** vs SEQ_SPLIT=0 191,330 tok/688.775s = **277.8
tok/s** — SEQ_SPLIT=1 is **~29% faster** at 190-220K context (confirms
the default holds beyond the previously-only-tested 100K case).

### 3.2 Prefill cliff investigation (mechanism CLOSED 2026-08-24; symptom fixed in production since 2026-06-24)

`PREFILL_CLIFF_HANDOFF.md` (2026-06-21, NEGATIVE): documented the
~340K-context sharp throughput cliff (270 t/s → 40-48 t/s). Root cause of
the SHARP discontinuity was never identified. A memory-allocation fix
(tiled-P indexer, `INDEXER_TILED_P_PLAN.md`) reduced peak memory per call
**4.36GB→0.40GB (-91%)** but throughput was **~2% WORSE**, not better —
**allocation pressure was not the actual bottleneck for the cliff.**
`attn.indexer` avg cost grew +18% from 300K→360K context (4532us→5362us),
max/avg ratio ~4x (22ms spikes) — consistent with, but not proven to
fully explain, the cliff.

**This session's own `EXO_DSV4_INDEXER_PBLOCK` re-test (NEGATIVE,
2026-08-21)** confirmed: no prefill win at ANY tested depth (100K/300K/
500K all at parity with baseline), and a REAL DECODE REGRESSION at small
p_block (13.67→11.55→10.03 tok/s across depths) because the design's own
"decode pays zero overhead" claim breaks once pooled length exceeds
p_block at deep context. See §13 for what's still open here.

**Later re-validated (2026-08-18, INFO_ONLY):**
`docs/dsv4-220k-prefill-span-profile-2026-08-18.md` found the July
gather-based bottleneck theory (indexer.score/topk/attn.gather) is now
**0.0% of wall** — already optimized away by subsequent refactors. At
220K: unprofiled clean run 220,321 tok/612s = **360 tok/s**; compute-bound
verdict (attn+ffn sum to ~100% of wall, 97-98% live GPU utilization,
superseding a stale July memory-bandwidth-bound finding). **Re-validate
old bottleneck theories against current code before trusting them —
significant refactors can invalidate a profile from even a month prior.**

**MECHANISM CLOSURE (2026-08-24)** — `docs/prefill-cliff-mechanism-2026-08-24.md`
consolidates a 3-round investigation into a threshold-family mechanism
identification:

- **Mechanism family DEMONSTRATED**: the era cliff belongs to the
  `active_memory > threshold` family, with TWO independent triggers
  riding the SAME crossing under era `MLX_MAX_MB_PER_BUFFER=50`
  per-primitive commits — (a) MLX allocator gc-release
  (`mlx/backend/metal/allocator.cpp:149-151`, fires when
  `mem_required = active + cache + size ≥ gc_limit_ ≈ 106.7 GB` on
  Studios), and (b) fork's eval-driver memory-branch throttle
  (`mlx/transforms.cpp:285-299`, fires on `active > memory_limit &&
  n_active_tasks > 0`, draining outstanding cbufs synchronously).
  Local repro (§14 of the mechanism doc): cache_gb collapses
  lockstep with ballast (direct evidence gc-release firing);
  memory-branch throttle produces +48% per-chunk overhead in
  isolation.
- **Amplitude and bimodality at Studio scale INFERRED/UNREPRODUCED**:
  the era ~6-8× stall + bimodal 8-32s per-chunk signature do NOT
  reproduce at local 36 GB M4 Max scale (+13-48% smooth overhead
  only). The Studio-resident-set (~30×) + queueing divergence
  (`1/(1-ρ)` wait-time explosion) hypothesis is offered as inference,
  not evidence.
- **PROVEN attribution correction**: `EXO_DSV4_PREFILL_ARGPARTITION=1`
  **cannot have fixed the cliff on Metal** — `sort.cpp:342-353`
  (`ArgPartition::eval_gpu` delegates to identical
  `gpu_merge_sort(...,argsort=true)`) + microbench parity +
  timeline (cliff was already gone in the 2026-06-24 500K = 251 t/s
  measurement, a week BEFORE argpartition shipped ~2026-07-01).
  `MOE_KERNEL_HANDOFF.md`'s attribution to argpartition is
  factually wrong; correction banner added to that doc.
- **Positive attribution strongly-supported but co-shipped-candidate-
  ambiguous**: `MLX_MAX_MB_PER_BUFFER=50→200` (exo `463ac5d`) is the
  *most likely* proximate positive fix; the same 2026-06-24
  breakthrough batch also shipped OPT-6 (indexer weight fold, 64×
  compute reduction, mlx-lm `453daa5` / exo `d26dc013`) and OPT-9
  (broadcast elimination, -3.2 GB/chunk); attribution weight cannot
  be cleanly split among the three from available evidence
  (mechanism doc §15.1 provenance audit).
- **Sync-span observer-effect rescoping**: the era tiled-P A/B
  (`PREFILL_CLIFF_HANDOFF.md:108` documented launch:
  `EXO_PROFILER=spans EXO_PROFILER_SYNC_SPANS=1 ...`) remains
  **INTERNALLY VALID** as evidence for the narrow claim "tiled-P
  does not help in a sync-serialized regime". What is INVALID is
  only the broader INFERENCE "allocation pressure ruled out for the
  unprofiled cliff regime" — because sync-spans structurally disable
  both throttle branches and gc-release stacking; extrapolating from
  that cliff-suppressed regime to the cliff-manifesting regime is
  not licensed by the data. The A/B measurement is fine; its
  conclusion's scope was overreached.
- **Live confirmation (2026-08-24)**: cliff-band probe at 381,619
  tokens (exo `34478792b` / `0854b39` era stack, MB=200 live) —
  328.6 tok/s aggregate, needle PASS. **Symptom fully gone in
  production.**
- Bench artifacts: `bench/prefill_cliff_mechanism_local.py`,
  `bench/prefill_cliff_throttle_repro_local.py`,
  `bench/prefill_cliff_gclimit_repro_local.py` (+v2),
  `bench/cliffband_380k_probe.py` — all with `_results.jsonl`
  companions. Full analysis: `docs/prefill-cliff-mechanism-2026-08-24.md`.
- **Correction banners added**: `MOE_KERNEL_HANDOFF.md` (argpartition
  attribution wrong) and `PREFILL_CLIFF_HANDOFF.md` (symptom-resolved
  banner + sync-span rescoping note).

### 3.3 Step-size / chunk-size tuning


`docs/dsv4-prefill-step-size-4096-retest-2026-08-18.md` (NEGATIVE): a
retest of `EXO_PREFILL_STEP_SIZE=4096` (previously rejected). Quality now
PASSES (previously-broken finding is stale), but throughput at ~191K
tokens: **STEP_SIZE=4096 331.2 tok/s vs STEP_SIZE=2048 baseline 358.6
tok/s — 4096 is ~8% SLOWER**, despite isolated MoE-GEMM microbench
showing 4096 should be ~15% more GEMM-efficient (72.0% vs 62.6% of dense
ceiling). **The sparse indexer's score-transient cost scales with BOTH
chunk size L and pooled window P and dominates at high context — an
isolated component-level microbenchmark win does NOT guarantee
end-to-end throughput.** `EXO_PREFILL_STEP_SIZE=2048` remains correct
default.

`.hermes/plans/2026-06-12-prefill-optimization-fable5-analysis.md`
(MIXED) — **measurement-harness bug produced a fake win.** Initial
claimed progression 236→280→290→317.8 tok/s (+35%, "OPT-4 two-level
chunking") was from a BUGGY harness (`fire_and_measure`) scraping the
wrong log line via `grep|tail -1`. Corrected harness
(`measure_clean.py`) revealed the REAL result at ~25K ctx: baseline
(chunk 128) ~258 tok/s, seq-split ON ~285 tok/s (+10-11% real, shipped),
**OPT-4 chunk 256 ~140 tok/s — a REGRESSION**, reverted (commit
`ff1d3f42`). **Always cross-check log-parsed throughput against
independently measured wall-clock throughput; flag >30% disagreement.**

### 3.4 MoE tile geometry / GEMM efficiency (all closed NEGATIVE)

`docs/moe-tile-geometry-retune-dead-end-2026-08-18.md`: tile-waste model
at production ragged routing distribution — bm=16 (current) 20.4% mean
waste, bm=32 34.8% waste, bm=64 52.1% waste (**monotonically worse**).
Production's median expert receives only 8-14 rows, below current
bm=16's tile height — larger tiles increase per-tile padding waste since
most experts are small. Corrected MoE efficiency vs proper mxfp4 ceiling:
**72.0-72.5% at L=2048** (28% gap, not an earlier miscalculated 37%).
This closed the third and final investigated idea (after hybrid dispatch
NO-GO and gather/scatter overhead ruled out) for the MoE efficiency gap.
**Re-verify old "falsified" conclusions against the ACTUAL production
distribution shape, not a stale uniform-distribution sweep.**

`docs/lever1-moe-smallm-headroom-2026-08-20.md` (also see §2.6) — live
kernels already match/beat the idealized dense-GEMM ceiling; widening
MAXBE regressed severely.

**NEW (2026-08-23, P4): TP=2 width-sharding does NOT create skinny-GEMM
inefficiency — the sign is inverted, and the lever is closed.**
`docs/p4-tp-width-shard-gemm-efficiency-2026-08-23.md`. Hypothesis was
that exo's MoE-only sharding (`gate/up` all-to-sharded → N 2048→1024;
`down` sharded-to-all → K 2048→1024) leaves per-rank GEMMs too skinny for
good tile efficiency. Measured with chained-eval methodology (one
`mx.eval` per 12 calls, per the 2026-08-22 P0 retraction) and routing
indices held EXACTLY constant across width arms:
- MoE `SwitchGLU` arm, achieved TFLOPS across a **4x width sweep**
  (inter=512/1024/2048): **6.851 / 7.049 / 6.739 — flat**, max/min ratio
  1.046; wall time scales linearly (actual÷linear = 1.000/0.972/1.017).
- Dense mxfp4 `qmm` at production shapes: sharded is **FASTER** per unit
  work — gate/up N-halved **+3.6%** (9.939 vs 9.598 TFLOPS), down
  K-halved **+2.6%** (9.510 vs 9.267). Kernels sit at **76.7–85.2% of
  measured dense compute peak**.
- A real skinny-K penalty does appear, but only **beyond** the cluster's
  config (TP=4, down K=512: −3.5%) — not at TP=2.
- Both per-rank and unsharded widths are exact multiples of the 32-wide
  tile → zero partial-tile waste analytically; the bench confirms no
  second-order latency-hiding penalty either.
- Side confirmation: **mxfp4 dequant IS fused in-kernel**
  (`QuantizedBlockLoader` inside `fp_gather_qmm_rhs`, no separate pass,
  no materialized bf16 weights) — so the packed-bytes roofline
  denominator used elsewhere is **correct as used**. Also confirmed
  empirically that prefill dispatches **tier-2 `gather_qmm_rhs`** at the
  production shape (B/E=48 vs the tier-1 gate of ≤6), tile geometry
  bm=16/bn=32/bk=32 (the bm=64 `_nax` variant is gated off on M4 Max,
  GPU gen 16 < 17).
Lesson: **check the SIGN before assuming a structural penalty exists** —
"TP makes our shapes skinny" sounded obviously true and measured
backwards.

### 3.5 Prefill planning docs never fully executed

`PREFILL_THROUGHPUT_PLAN.md` (2026-07-13, INFO_ONLY) — pure planning doc.
Key finding used to justify the plan: per-chunk cost is near-FLAT
(~2.3-2.8s) regardless of chunk tokens/depth — **fixed cost dominates**,
not depth scaling. 147/200 logged prefills are <2K tokens (agentic-loop
common case). **When throughput is dominated by a near-flat fixed
per-chunk cost, the highest-leverage fix targets the common-case SMALL
request bucket, not deep-context scaling.**

`docs/pp-prefill-tp-decode-phase-swap-design-2026-08-16.md` (NEGATIVE,
superseded design) — a PP-prefill→TP-decode phase-swap design was
rejected. Full math: ~11.1% gain on a cold 500K request (TP-only 1584s
vs PP+swap+TP 1409s), **0% gain on cached follow-up turns**; ~37.4s swap
cost. PP 364 / TP 319 tok/s at 500K — **neither topology alone even hits
the 400 tok/s depth requirement**, so the swap converts one miss into a
smaller miss. Requires two weight layouts (~125.4GB of 128GB) that can't
co-reside. Kept only the DSv4 CacheList/PoolingCache wire codec as
salvage.

`docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md` (NEGATIVE) —
earlier version of the same investigation. Found and fixed a measurement
bug (chars//4 instead of true ~5.68 chars/token, inflating all prefill
tok/s by 1.42x). Corrected honest numbers: prefill 225/214/202 tok/s at
100K/300K/500K; PP 364 vs TP 319-427 tok/s prefill. TP chosen for BOTH
prefill and decode — PP structurally idles one node per single-session
request. Hybrid swap and expert-locality placement both ruled
architecturally dead (weight layouts can't co-reside on 128GB nodes).

---

## 4. Decode throughput & dispatch overhead

### 4.1 MoE gate+up fusion — the clearest recent win (WIN, 2026-08-21, IN PRODUCTION)

`docs/moe-gate-up-fusion-validated-2026-08-21.md` — `EXO_DSV4_MOE_FUSED_GATE_UP=1`
fuses `SwitchGLU`'s gate_proj+up_proj into ONE `gather_qmm` dispatch
instead of two. Decode: Fusion ON **18.879 tok/s** (n=8, σ=0.158) vs OFF
**18.328 tok/s** (n=8, σ=0.173) = **+3.01%**, means differ by ~3.2x
combined stdev — statistically clean. Prefill at parity (dispatch-count
reduction matters proportionally more for decode's small per-forward-pass
compute than prefill's large per-pass compute). **This is the current
recommended production default for c=1.**

### 4.2 The direct analogue that DIDN'T work (NEGATIVE, 2026-08-21)

`docs/qa-kv-fusion-no-measurable-gain-2026-08-21.md` — tested the direct
structural analogue on attention projections: fusing `wq_a`+`wkv`
(`EXO_DSV4_QA_KV_FUSED`). Bit-exact, correct on real hardware, but
**-0.48% incremental** vs gate+up alone (18.789 vs 18.879 tok/s, within
noise). Combined gain over baseline (+2.52%) was actually LESS than
gate+up alone (+3.01%). **A validated fusion pattern does not
automatically generalize to smaller matmuls — wq_a/wkv are much smaller
than gate_proj/up_proj, so the same fixed per-dispatch overhead is a
smaller fraction of their cost. Never declare a fusion win by only
checking correctness — always do the real A/B against the current best
baseline.**

### 4.3 Roofline / dispatch-bound analysis (WIN, 2026-08-21)

`docs/decode-roofline-dispatch-bound-2026-08-21.md` — using confirmed
public model specs (DeepSeek-V4-Flash: 284B total/13B active MoE, mixed
FP4/FP8), decode is running at **~12% of the theoretical
bandwidth-bound ceiling** (8.4x slower than roofline). This rules out
"already near the hardware ceiling" and reprioritized the search toward
dispatch-count-reducing fusions and comm/compute overlap. Later
cross-validated by a real Instruments trace (§2.7): measured dispatch
latency 96.8µs matches this roofline's inferred ~150µs order of
magnitude.

**Sanity-checked and confirmed correct (2026-08-22, same continued
session)** — `docs/roofline-sanity-check-inputs-confirmed-2026-08-22.md`:
checked both roofline inputs against live production state. (1) Is
decode's "1 real forward pass per token" assumption right given
`EXO_DSV4_DSPARK=1`/`EXO_DSV4_MTP_EAGLE_K=8` are live? Traced the code:
DSpark's decode loop (`pp_dspark_decode_loop`) is genuinely PP-only —
confirmed via a live log check that its actual usage log line never
fires under TP (only the harmless module-attach "ctx warmed" line
does). **These flags are dormant under the TP topology tonight's
production runs — real finding, not a bug, but worth knowing they
provide zero effect here.** Decode is confirmed plain autoregressive,
the roofline's assumption was already right. (2) Is the 0.588
bytes/active-param ratio correct given the model card's coarse
`"quantization": "fp8"` label? Checked against real on-disk size from
`/state` (166,878,536,440 bytes / 284B params = 0.5876 bytes/param,
between pure FP4 and FP8, confirming genuine mixed precision) — this
IS what the original roofline calculation already used, sourced from
real on-disk size, not the coarse label. **Both inputs confirmed
correct; the ~12%-of-ceiling headroom finding stands unchanged.**

**RECALCULATED post-async-fence-fix (2026-08-22, new session)** —
`docs/roofline-recalculated-post-fix-2026-08-22.md`: the ~12% figure
above was measured under the since-fixed silently-broken async fence
(§2.8) — stale once decode throughput changed by +23-67%. Recalculated
using the exact same bandwidth-floor methodology (6.51ms/token/node,
unchanged — compute/bandwidth-side, not touched by the fence fix)
against real post-fix decode numbers: **14.0-20.2% of theoretical peak
depending on depth/context shape** (100K: 17.5%, 300K: 15.9%, 500K:
14.0%, short-context: 19.0-20.2%) — up from 11.9% pre-fix, but **real,
substantial headroom (~5-7x slower than roofline) still remains**.
Confirms Fable's prediction ("still likely ~15-20% of peak") made
before this calculation was run. The dispatch-overhead hypothesis
(implied dispatches/token from the residual gap) remains broadly
consistent post-fix (171-267 implied dispatches/token depending on
depth, down from the pre-fix ~320 but not eliminated) — the underlying
mechanism (real synchronization/dispatch overhead, not raw
compute/bandwidth) still looks like the right frame, just with a
smaller absolute gap now that the fence genuinely engages.

### 4.4 Older decode-stall investigation (NEGATIVE, three failed overlap attempts, 2026-06-26)

`docs/dsv4-decode-stall-2026-06-26.md` — confirmed decode's 73%
`moe.switch_mlp` cost is almost entirely GPU IDLE time during the
cross-rank all_sum collective (envelope 2935ms vs ~7ms real GPU compute,
>99% idle) — but **three separate attempts to capture that headroom all
failed**:
- MoE gate+up fusion (3→2 dispatches): **-3.8%** (37.2→35.8 tok/s) —
  note this predates and CONTRADICTS the 2026-08-21 validated win (§4.1);
  worth investigating whether something else changed between these two
  measurements before assuming either is wrong
- all_sum/shared_experts stream-overlap: **broke quality hard** (B=2 200K
  needle failed, near-zero output ~0.42 tok/s aggregate)
- OPT-7 fence-gating: **-23% prefill** (see §2.6)

**A large measured GPU-idle window during a collective does NOT mean the
idle time is capturable via naive dispatch-fusion or fence-reordering —
MLX's lazy-graph + comm/GPU-stream + fence interaction has load-bearing
fence POSITIONS for cross-rank bit-equivalence not visible from reading
source alone. Three independent failed attempts at "overlap the stall"
is strong evidence it needs deeper MLX-stream-scheduling understanding
(see §2.6's later phase0b/0c work, which DID succeed with proper stream
separation), not another surface-level retry.**

### 4.5 Compile-boundary fusion attempts (NEGATIVE, hard "do not touch")

From `.hermes/plans/2026-05-18_1830-dsv4-verify-tail-investigation.md`
and `.hermes/plans/2026-05-19_to_35tps.md`:
- Fused SDPA L≤8 fold (Lever 1): 30.06→28.9 tok/s — regressed because
  MLX's compiled shapeless kernel had higher per-call Metal launch
  overhead than `mx.fast.sdpa` at small batch×L
- Fuse `_raw_post_attn`+`_raw_ffn_pre` (Lever 2): **30.06→7.2-10.5 tok/s
  — CATASTROPHIC.** Capturing an `mx.fast.metal_kernel` inside another
  `mx.compile` boundary triggers pathological behavior.

**Do not nest `mx.fast.metal_kernel` calls inside another `mx.compile`
boundary — causes catastrophic (3-4x) throughput regression. Always gate
new fusion attempts with a single-layer microbench before cluster
deployment.**

### 4.6 The dispatch-overhead / pipelined-vs-isolated trap (recurring pattern, 3+ instances)

This exact trap recurs across THREE separate fusion projects — worth
flagging as a pattern, not just individual results:

1. **MoE fused Metal kernel** (`.hermes/plans/2026-05-14_113951-dsv4-moe-fused-metal-kernel.md`,
   NEGATIVE, killed at phase-1 spike): pipelined 43-layer chain measured
   at 207us/layer vs 187us memory-bandwidth floor — only 20us/layer of
   dispatch-overhead headroom = ~3% throughput, below the 1.5x decision
   gate. The MoE-NOP probe showed 8K c=1 35.0→53.2 tok/s (+52%) and 100K
   c=1 29.47→41.2 tok/s (+40%) headroom when MoE fully bypassed — but
   that headroom is **memory-bandwidth-bound, not dispatch-bound**, so
   kernel fusion couldn't recover it.
2. **DSv4 indexer fused kernel** (`.hermes/plans/2026-05-14_185010-dsv4-indexer-fused-kernel.md`,
   NEGATIVE, abandoned): pipelined 21-call microbench showed the fused
   kernel (numerically correct, 159/160 topK overlap) at 215us/call vs
   the EXISTING chain's 117us/call pipelined = **0.54x, SLOWER once
   pipelined**. An INDEXER-NOP isolation probe had shown 28.4→32.3 tok/s
   (+13.7%) motivating the plan, but the real fusion ceiling was only
   ~0.9% once pipelining was accounted for.
3. **DSv4 sparse-attn fused kernel** (`.hermes/plans/2026-05-14_140936-dsv4-sparse-attn-fused-kernel.md`,
   NEGATIVE): abandoned after a phase-1 microbench spike failed the gate
   before any real Metal kernel was written.

**THE PATTERN: MLX's async graph executor already overlaps dispatch
across the op chain. Per-call-in-isolation dispatch-overhead estimates
(e.g. "~250 launches/token × 20-30us = 5-7ms recoverable") badly
overestimate savings because they don't account for this pipelining —
only a PIPELINED chain-level microbench reveals the true ceiling.
Per-call analysis lies; pipelined chain tells the truth. Apple's own
matmul/argsort kernels are already near-optimal — hand-rolled Metal
rarely beats them for well-tuned ops, only for eliminating true
dispatch-BOUNDARY overhead between many small ops (which is why the gate
+up and (attempted) wq_a+wkv fusions, which reduce dispatch COUNT rather
than trying to out-kernel MLX's own primitives, are the pattern that
actually works — see §4.1/4.2).**

### 4.7 gather_qmm M=1 dispatch check (NEGATIVE = no bug found, 2026-08-21)

`docs/gather-qmm-m1-dispatch-confirmed-correct-2026-08-21.md` — zero-risk
read-only check confirmed MLX's `GatherQMM::eval_gpu` dispatch ladder
already correctly routes decode's M=1 shape to a dedicated `gather_qmv`
gemv kernel, not the general tiled `gather_qmm` gemm path. `vector_limit`
(hardware-gen-aware, 10-32) confirms M=1 never reaches the general path.
**No bug — closes this specific question with confidence.**

---

## 5. Speculative decoding (MTP / Eagle / token-tree / DSpark)

Second-most-investigated area after all_sum. Long, iterative history —
**the throughput ceiling here has proven very hard to move past ~30-35
tok/s** despite dozens of attempts.

### 5.1 Timeline of "champion" throughput claims (cautionary tale)

Multiple sessions across May-June 2026 chased a "35 tok/s" target. Key
lesson from this whole arc, stated explicitly in
`.hermes/plans/2026-05-19_to_35tps.md`: **historical "champion" claims
repeatedly failed to reproduce**:
- `champion-2026-05-17-fenced` claimed **31.5 tok/s** — NOT reproducible
  later (`FENCE=8` retest got 29.5 tok/s)
- `champion-2026-05-18-acksync` claimed **32.3 tok/s** — redeploy
  **catastrophically cratered to 4.3 tok/s**, cause never found
- Actual reproducible baseline held at **30.06 tok/s (σ=0.06)**

Separately, `.hermes/plans/2026-05-19_quality_findings.md` (NEGATIVE)
found that some of these "champion" numbers were measured at
**`EXO_DSV4_INDEX_TOPK=160`, which produces BROKEN quality** (only BOS
token output at 100K context, complete needle-recall failure) vs the
quality-correct `TOPK=512`'s 30.06 tok/s. **A throughput gain that trades
away output quality is not a real win.**

**Reusable lesson (critical): never trust a champion throughput number
without ≥10 iterations at σ<0.3-0.5. Treat unreplicated single-sample
champions as suspect until re-verified. Always pair speed benchmarks
with a correctness/quality probe (needle-in-haystack) before accepting a
throughput result as valid.**

Eventually stabilized at a real, reproducible **gamma=2 MTP-on ~30.06-31.5
tok/s** production baseline (`.hermes/plans/2026-05-17-session-retrospective.md`,
WIN: gamma=2 MTP-on 31.5 tok/s shipped, +3.6% over gamma=1, ~30 hours wall
across two sessions, ~15 bench runs). Later further refined via rollback
cost fix and Eagle K=8 no-renorm (below).

### 5.2 Real, shipped, quality-neutral wins

**MTP verify-forward losslessness (WIN, 2026-07-10)** —
`docs/dsv4-rowseq-followups-plan-2026-07-10.md`: `EXO_DSV4_VERIFY_ROWSEQ`
makes L>1 MTP verify attention bitwise-identical to sequential decode,
proven bitwise-zero on a dedicated harness, **at ZERO throughput cost**
(c=1 27.4 tok/s matches the old lossy baseline). Fixed via
`MLX_GEMV_BATCH_INVARIANT` + cache-level spec rollback. **It IS possible
to fix MTP losslessness bugs at zero throughput cost when the root cause
(gemv/gemm M-dispatch rounding drift) is fixed at the source.**

**Rollback cost fix (WIN, 2026-07-12)** —
`docs/rollback-cost-campaign-handoff-2026-07-12.md`: target was cutting
rollback cost ~32ms→~5ms; **root cause was actually different from the
assumed one** — pruned-champion rollback was already ~2ms, real cost was
a ~72ms commit-forward FALLBACK path. Fixed at the cache level
(`mlx-lm f00a9a9`). Result: 29.0-29.3 tok/s held, rollback cost slashed
to **0.79ms mean** (`rb_commitfwd n=0`). **Premise revision via
sub-phase profiling found the real cost center differed from the assumed
one — always profile before assuming which code path dominates.**

**Eagle K=8 no-renorm (WIN, small but rigorous, 2026-05-24)** —
`.hermes/plans/2026-05-24_w3_K8_norenorm_results.md`: **+0.83%** decode
(+0.24 tok/s), Welch t=6.19, p<0.001, Cohen's d=2.77. Baseline 28.7998
mean → K=8+no-renorm 29.0375 mean. **Small but statistically significant
gains (via proper Welch t-test/effect size) are valid to ship even at
<1% improvement, provided quality is preserved and the effect is
rigorously verified.**

**Drain elimination (WIN, 2026-05-20/21)** — the key unlock for c=2
concurrent MTP throughput. `.hermes/plans/2026-05-20_phase12_drain_elim_results.md`
+ `.hermes/plans/2026-05-21_phase13_c2_milestone.md`: yielding all
N×(γ+1) responses in ONE `_next()` call removed ~50ms per-token-drain
overhead at TP c>1, unlocking **22.9→34.5-34.6 aggregate tok/s** at c=2
100K MTP-on (8/10 iters clean at σ=0.04, 2/10 outliers traced to a
bench-side asyncio timing artifact, not a server regression). Required
also bumping `EXO_BATCHED_PREFILL_RENDEZVOUS_MS` 200ms→2000ms to reliably
catch the 2nd concurrent POST arrival — without this, intermittent
fallback to serial submission caused σ≈10 tok/s variance
(`.hermes/plans/2026-05-20_phase11_c2_progress.md`... wait see
`phase12_drain_elim_results.md` for the pre-fix variance table: iter1
wall=790s agg=11.7, iter3 wall=737s agg=34.6 — same code, wildly
different outcome depending on whether the rendezvous window caught the
race).

### 5.3 Attempts that did NOT beat linear/baseline decode

**Token-tree drafting (INCONCLUSIVE→NEGATIVE across a multi-week arc,
May 2026):** planned in `.hermes/plans/2026-05-19_token_tree_drafting.md`
as "the only remaining realistic structural path" after compile-fusion
levers were exhausted. Implemented, but found broken at production
config first (`.hermes/plans/2026-05-19_phase6_findings.md`, NEGATIVE —
unit microbenches passed but production config output was garbage/looped,
100K needle FAILED; root cause: microbenches didn't exercise the sparse/
pooled-attention path, PoolingCache under tree input, or Indexer with
tree-rotated Q). After 3 correctness bugs fixed
(`.hermes/plans/2026-05-20_phase6b_findings.md`, MIXED): correctness
restored (cluster bench 75K: tree 29.95 tok/s ≈ baseline 30.06 tok/s,
100K needle passed), but **tree drafting showed NO throughput lift over
linear baseline** — commit-forward overhead (~6% of wall, ~100ms/cycle)
ate the gain. Perf-tuning follow-up
(`.hermes/plans/2026-05-20_phase7_perf_findings.md`, NEGATIVE): 6 tuning
variants (DFS reorder, greedy tree, K sweep) all landed in [29.7, 29.95]
tok/s vs the 30.06 linear champion — **none beat linear.** Root cause:
verify phase is bounded by per-token KV attention access at long context
(constant regardless of tree/draft size), not by query-row count, so
shrinking L_q barely helps.

**`.hermes/plans/2026-05-20_phase8_beating_linear.md`** (INCONCLUSIVE) —
best tree config (K=2 γ=2) still only 29.95 tok/s vs 30.06 linear:
+44% verify-wall growth (L_q=7) ate a +13% draft-acceptance win. **Verify
cost decomposes as ~30ms floor (KV attention over long context) + ~5.3ms
per L_q token — tree's wider verify erases its per-slot acceptance gain
because the FLOOR, not marginal cost, dominates at long context.**

**Per-row-SDPA verify-vec hypothesis (NEGATIVE, 2026-07-12)** —
`docs/vec-rowsdpa-campaign-2026-07-12.md`: the "lossless ~34 tok/s via
per-row sdpa variant" hypothesis was found DEAD. Don't re-attempt without
new evidence.

**Eagle K1 debug (NEGATIVE, 2026-05-22)** — a K=1 c=2 throughput
regression was root-caused via 5-hypothesis elimination
(`.hermes/plans/2026-05-22_eagle_k1_debug_report.md`) to un-synced
`prev_logits` breaking cross-rank determinism: MLX produces tiny
per-rank logit drift at cycle 5+ that flips argmax on near-tied logits;
any new tensor computed from rank-local logits without inheriting the
existing cross-rank broadcast reintroduces divergence. K=1 c=2 100K:
21.48 tok/s vs FENCE=4 baseline's symmetric 23.29 tok/s. A DEEPER root
cause was then found (`eagle_k1_fix_report.md`, NEGATIVE): the real
mechanism was a `broadcast_from_canonical(soft_emb)` 16KB collective
placed directly on the per-chain-step critical path with ZERO compute to
hide behind — turned a ~150s c=2 100K iter into a **~6.5-min iter (~17x
slowdown)**. **A collective placed immediately before a dependent compute
op with no other work to overlap becomes fully exposed serialized
latency — 16KB is "small" in absolute terms but catastrophic when
directly on the critical path with nothing to hide the RTT behind.**

**Gamma=3 bistability (never fully resolved, 2026-05-23)** —
`.hermes/plans/2026-05-23_gamma3_bistability_fix_plan.md`: γ=3 c=2 100K
showed a symmetric case at 40.57 tok/s but bistable behavior across
iterations. `.hermes/plans/2026-05-23_session_eagle_to_gamma3_findings.md`
(WIN, partial) landed a production champion: **γ=2 K=1 FENCE=4 @ 34.19
tok/s, σ=0.05, 5/5 clean** — 0.81 tok/s short of the 35 tok/s target.

**MTP head investigation (NEGATIVE, 2026-05-19)** —
`.hermes/plans/2026-05-19_mtp_head_investigation.md`: DSv4-Flash ships
only ONE MTP head (`num_nextn_predict_layers=1`) — **cannot add a second
head without training** (weeks of cluster time). `gamma=3 alpha_3=0.37`
acceptance measured. Cheaper alternative identified: Eagle-style
token-tree drafting using the existing single head (this is what
motivated the token-tree work above, which ultimately didn't beat
linear).

**DSpark FULLBLOCK context-depth collapse (NEGATIVE, severe, 2026-08-04)** —
`docs/dspark-fullblock-context-scaling-cliff-2026-08-04.md`: DSpark
ON+FULLBLOCK+ROWSEQ=shared showed **27.56 tok/s at depth~500 but 1.73
tok/s at depth~14K — a 15.9x collapse** (later found narrower, ~2800-token
band). DSpark OFF: normal 1.28x scaling (25.11→19.57 tok/s). A separate
MoE fix (`EXO_DSV4_MOE_PARTS_ROWSEQ=shared`) gave a real win ONLY at
near-zero context (300-500 tokens, +12%) but was never tested at real
depth before this — DSpark+FULLBLOCK turned out catastrophically worse
than no speculation at depth. **Cluster now runs DSpark OFF as a safety
default.** **Always benchmark speculative-decode fixes across a RANGE of
context depths, not just short prompts, before shipping.**

> **CORRECTION 2026-08-24 (P4v2): this verdict attaches to a decode-loop
> MODE, not to the DSpark head, and the "cluster runs DSpark OFF" line is
> stale.** Two separate things share the name "DSpark":
> - **DSpark the head** — `DeepseekV4DSparkModule`
>   (`mlx-lm/mlx_lm/models/deepseek_v4.py:6340`), a 3-stage
>   semi-autoregressive draft head, `block_size=5`, taps `[40,41,42]`.
>   Nothing in the 15.9x measurement indicts the head: the cited
>   `r1_verify_fwd=1455.8ms` outlier occurred at **94% draft acceptance**,
>   i.e. the head was drafting *well* while the verify path was collapsing.
> - **FULLBLOCK the verify mode** — `EXO_DSV4_ROWSEQ_FULLBLOCK=1`
>   (`deepseek_v4.py:1591`), which runs the ENTIRE non-MoE block **once per
>   drafted row** instead of batched. Its cost scales with `k` × (per-row
>   attention incl. the Indexer top-k over compressed KV, which itself grows
>   with context) — that product is the collapse mechanism.
>
> Consequence: the collapse is a `k`-multiplier problem, and DSpark's
> `block_size=5` simply put it at the worst end of that multiplier. It is
> **not** a reason to reject the DSpark head at small `k`.
>
> Also stale: **the cluster does NOT run DSpark off.** Live `ps eww` on
> both nodes 2026-08-24: `EXO_DSV4_DSPARK=1`, `EXO_DSV4_ROWSEQ_FULLBLOCK=1`,
> `EXO_DSV4_ROWSEQ_FULLBLOCK_MOE=0`, `EXO_DSV4_MOE_PARTS_ROWSEQ=shared`.
> What is actually off is `EXO_SPECULATIVE=0` + `EXO_DSV4_MTP=0`, which
> prevents the *generator* from ever being constructed — so FULLBLOCK never
> executes even though its flag reads `1`. The safety property holds, but
> by a different mechanism than this entry claims.

**DSpark native head (INCONCLUSIVE, implemented but never A/B tested,
2026-08-03)** — `docs/dsv4-0731-dspark-native-head-plan-2026-08-03.md`:
`EXO_DSV4_DSPARK_NATIVE` implemented and validated standalone, but never
live-A/B-tested. **This session (2026-08-21) confirmed it's STILL unset
in production** and flagged it as a real, untested candidate — but noted
it's decode-only (affects MTP/DSpark draft acceptance rate) and NOT
applicable to prefill-focused optimization work. Genuinely open.

**Verify-cost restructuring plan (INCONCLUSIVE, 2026-05-18)** —
`.hermes/plans/2026-05-18_1505-dsv4-verify-forward-toward-35tps.md`:
target was 35 tok/s (+17% from a 30 tok/s quality-correct baseline; the
32.35 tok/s `TOPK=160` number was quality-broken, see §5.1). Flagged
**hard "do not touch" items**: ACK barrier optimization, removing the
per-step `mx.eval` fence, and swapping to `mx.async_eval` were all
PROVEN to cause severe regressions in this specific plan's context —
note this predates and is superseded by 2026-07-02's successful,
carefully-gated `EXO_DSV4_FENCE_ASYNC` (§2.6/§4) which DID make
async_eval work safely, via proper two-owner gating that this earlier
naive attempt lacked.

**Critical-path NOP-probe methodology trap (NEGATIVE finding about the
METHOD, 2026-05-19)** — `.hermes/plans/2026-05-19_critical_path_findings.md`:
a `build_probe`-style per-op profiler attributed 46.51ms/forward (92.7%)
to FFN/MoE, but a critical-path NOP test (zeroing MoE entirely) only
saved ~7.5ms of verify wall — the other ~39ms ran CONCURRENTLY with other
work on the async MLX pipeline. **Per-op profiler attribution can
overstate a component's true critical-path cost when the framework
overlaps ops asynchronously — a microbenchmark speedup does not imply an
equivalent end-to-end wall-clock speedup.**

### 5.4 MTP correctness fixes tied to performance work

- `docs/mtp-tiebreak-losslessness-fix.md` (WIN, shipped default-on
  2026-06-04) — fixed a losslessness bug in MTP tie-break logic.
- `docs/deepseek-v4-c2-mtp-verify-fixes.md` (WIN, 2026-06-06/07) — made
  MTP speculative decode correct at c≥2 (BS>1).
- `docs/deepseek-v4-mtp-performance.md` (WIN, 2026-06-04) — MTP
  self-speculation roughly doubles per-forward token yield since verify
  dominates cycle cost (93.4% of an 81.7ms MTP cycle). 30.77 tok/s mean
  (σ=0.067) c=1 100K gamma=2 with MTP-on + tiebreak fix vs MTP-off ~27
  tok/s. mean_accept 1.04/2 drafts = 68% of gamma=2 ceiling. **gamma has
  a sweet spot determined by acceptance-rate falloff, not a monotonic
  increase** — gamma=1 is -6%, gamma=3 is -18% vs gamma=2's baseline.

---

## 6. Kernel fusion attempts (general pattern)

See §4.6 for the detailed "pipelined vs isolated dispatch overhead" trap
that recurred 3+ times. Summary table of ALL fusion attempts found in
the repo history:

| Fusion | Outcome | Result |
|---|---|---|
| MoE gate_proj+up_proj (`EXO_DSV4_MOE_FUSED_GATE_UP`) | **WIN** | +3.01% decode, in production (2026-08-21) |
| Attention wq_a+wkv (`EXO_DSV4_QA_KV_FUSED`) | NEGATIVE (null) | Correct, bit-exact, but -0.48% incremental (2026-08-21) |
| MoE fused Metal kernel (gate+up+SwiGLU+down, one kernel) | NEGATIVE | Killed at phase-1 spike, dispatch headroom only ~3% once pipelined (2026-05-14) |
| DSv4 indexer fused kernel | NEGATIVE | Abandoned, fused kernel SLOWER once pipelined (2026-05-14) |
| DSv4 sparse-attn fused kernel | NEGATIVE | Abandoned at phase-1 microbench gate (2026-05-14) |
| Fused SDPA L≤8 fold | NEGATIVE | 30.06→28.9 tok/s regression (2026-05-18) |
| Fuse post_attn+ffn_pre (compile-boundary) | NEGATIVE (severe) | 30.06→7.2-10.5 tok/s catastrophic (2026-05-18) |
| Fused softmax (`EXO_DSV4_FUSED_SOFTMAX`) | **NEGATIVE (correctness break)** | Real needle failure at 100K, never A/B'd for 5 weeks before this was caught (2026-08-21) |
| Fused sparse gather-SDPA kernel (decode/verify) | mentioned but outcome not captured in this scrub | See `mlx-lm` git log `0b07134` if revisiting |
| MLX head_dim 192/256 support (upstream) | **WIN** | Enables fused SDPA for larger head dims, gated by key-seq-length threshold |

**Cross-cutting lesson: fusions that reduce DISPATCH COUNT on
already-small, frequently-called ops (gate+up, in production) tend to
win. Fusions that try to out-kernel MLX's own already-tuned primitives
(matmul, argsort, SDPA) tend to lose once measured in a pipelined
context, because MLX's async graph executor already hides much of the
per-call overhead a naive isolated microbench suggests is recoverable.**

---

## 7. Quantization (KV cache, weights, collectives)

### 7.1 KV cache quantization

`docs/kv-cache-architecture.md` (INFO_ONLY) — documented quality-vs-perf
tradeoff: **bf16 KV is the default despite being SLOWER** (bf16: 11.4
tok/s/stream vs 4-bit: 11.9 tok/s/stream, +4% faster at c=2 100K MTP=0
sparse-attention regime) because 4-bit quantization noise compounds and
degrades long-context quality. 4-bit's advantage is bandwidth-driven, not
compute-driven, on Apple Silicon (SDPA is bandwidth-fed) — and that
advantage **shrinks specifically for sparse-attention (top-K read)
regimes** vs full attention, where it's ~2.5x. **Keep KV quantization
decisions tied to measured quality regression risk, not just raw
throughput.**

### 7.2 TurboQuant (NEGATIVE, all configs regressed)

`docs/turboquant-integration.md` — tested as an alternative to
dequantize-then-SDPA. Baseline (4-bit dequant + fused SDPA): **40.9
tok/s**. Every TurboQuant variant tested REGRESSED:
- 4-bit+quantized_matmul: 37.0 tok/s (**-10%**)
- 3-bit+quantized_matmul: 38.3 tok/s (**-6%**)
- 3-bit+dequant+inverse-rotate+fused SDPA: 36.5 tok/s (**-11%**)

**Despite theoretically avoiding full dequantization, added compute
overhead (rotation, polar conversion) outweighed bandwidth savings on
this hardware/kernel combo. Theoretical bandwidth wins don't guarantee
measured wins when compute overhead is added — do not adopt without a
fundamentally different implementation approach.** (Also separately
evaluated as a no-win on Qwen3.5 for a different reason — see
`docs/prefill-optimization.md`, §3.5-adjacent — only 8/30 layers are
attention on that model, so KV-cache-focused optimizations have limited
reach there.)

### 7.3 Quantized MoE all_sum

See §2.5 — comprehensively tested and closed NEGATIVE across 8 docs.

### 7.4 Weight quantization comparisons (data points, not investigations)

`bench/quant_compare_results/` contains benchmark result cards (decode
tok/s, TTFT, total time) for coding-eval tasks across quant variants on
Qwen3.5-397B-A17B. These are mostly INFO_ONLY reference data points, not
investigations:
- 4bit: 35.8 tok/s decode
- 4bit-qkv: 35.6 tok/s decode
- nvfp4: 35.6-37.0 tok/s decode (varies by sample)
- 4bit (arch_design sample): 31.5-31.7 tok/s
- 4bit (debug_fix sample): 34.9 tok/s

(Note: these numbers vary noticeably across different eval samples of the
same quant scheme — treat any single comparison as noisy; use only for
rough cross-scheme ordering, not precise deltas.)

### 7.5 bf16→fp16 compute dtype

Referenced in `docs/fork-vs-upstream-inventory.md` as a **~7% faster
quantized_matmul** win, applied across 7 kernel files in the
`qwen3_5_moe` kernel set. **BUT** `docs/profiling/request_lifecycle_trace.md`
notes fp16 compute dtype was **reverted for DSv4** — caused a **7x decode
slowdown** because JACCL/RDMA lacks fp16 support. **This dtype switch is
model/kernel-specific — a documented win for one model's MoE kernels is
a severe regression for a different model on the RDMA transport path.
Do not blanket-apply across models without re-testing the transport
interaction.**

---

## 8. jaccl / RDMA / Thunderbolt transport

(This session's own 2026-08-21 jaccl transport hardening work — 9 bugs
fixed, dual-cable topology split, QP-less TCP CoordGroup — is
extensively documented in `docs/dual-cable-topology-and-qp-budget-2026-08-21.md`
and is the most recent, most complete writeup. Summarizing OLDER related
work here for context.)

**c=2 serving handoff (MIXED, 2026-07-06)** —
`docs/dsv4-c2-serving-handoff-2026-07-06.md`: jaccl transport-level c=2
wedge SOLVED with a reliable ARQ all_reduce over UC plus a framed
coordinator barrier. A SEPARATE, unsolved residual instability was found
in the exo/mlx-lm batched-generation admission path: admitting a request
mid-batch diverges the two TP ranks, causing false-positive hangs. ~20
tok/s decode at c=2; c=2 MTP-on 14.3 tok/s/stream (28.6 aggregate). Real
measured: 1.45ms of a 6.6ms/token budget (~22%) attributed to jaccl
overhead, matching 25% GPU idle. **c=2-from-start (streams starting
together) is stable; admitting a new request into an already-running
batch mid-decode is not.**

**RDMA reliability bug cascade (2026-08-08/10)** — a good example of
iterative masking:
- `docs/handoff-2026-08-08-section22.md` / `section23.md` (NEGATIVE) —
  Section 22's bounded-blocking-ack fix for a chunk-boundary race was
  deployed, but its FIRST real validation attempt uncovered a NEW stall:
  hard stall after chunk 0's 11-advance sequence, jaccl "recv() deadline
  in drain", clean reconnect, then **ZERO activity for 8+ minutes** (test
  killed before the 30-min self-abort ceiling). GPU idle during the stall
  (23mW, ~6% active residency). Also found: `EXO_PP_METAFRAME=1` is an
  UNDOCUMENTED prerequisite for `EXO_PP_BATCHED_DECODE=1` — missing this
  caused two false-negative validation attempts that silently ran the
  old fallback path.
- `docs/handoff-2026-08-09-section39.md` (MIXED) — removing an internal
  fatal retransmit cap (60s shared deadline → 300s dedicated) fixed
  premature crashes on slow-but-healthy transfers, but ALSO removed the
  only (accidental) mechanism that force-cleared a genuine, separate
  scheduler-protocol-layer deadlock — a call_id then stayed stuck for
  21+ minutes. **Fixing a redundant/miscalibrated timeout can unmask a
  real, previously-hidden bug that the old timeout was accidentally
  recovering from.**
- `docs/handoff-2026-08-10-section43.md` / `section43-part2.md` (MIXED) —
  RDMA p2p_retry_exchange bugs: Bug 1 (wrong timeout constant, 8s generic
  vs 300s dedicated, crashed live requests at ~18s), Bug 2 (PP-spec-decode
  cancel() lifecycle state unhandled, causing full runner process exit).
  Verification after both fixes: no crash, cancel POST 200 in 5.29s, but
  **rank0 CPU kept climbing, never converged to idle — a THIRD
  undiagnosed symptom.** **When debugging a reliability/stall issue
  iteratively, expect a "masking cascade" — fixing an earlier crash can
  simply expose a later, previously-unreached bug. Don't declare victory
  until the ORIGINAL reported symptom is soak-tested, not just absence of
  newly-found bugs.**
- `docs/handoff-2026-08-10-section42.md` (INCONCLUSIVE) — a prior finding
  (TCP starving under RDMA contention) was DISPROVEN via a live interface
  check, which triggered then REVERSED a decision to shelve an RDMA
  migration for jaccl's control-plane traffic. **Disproving ONE stated
  justification for a decision does not disprove the decision itself.**

**Metal Event::wait stalls at 220K+ context** — multi-doc investigation
(2026-08-18): `dsv4-220k-prefill-rdma-wait-breakdown`,
`dsv4-220k-prefill-eventwait-rootcause-triage`,
`dsv4-220k-prefill-eventwait-ringdiag-nonrepro`. The ring-diag
reproduction attempt found: **zero Event::wait stalls across 2 runs**
(~440,644 combined prompt tokens, ~1,452s combined prefill wall) vs 8
stalls in a prior single 220K run — estimated per-call stall rate
**~0.17%** (8 in ~4730 MoE all_sum calls), consistent with a low, noisy
Bernoulli probability, not falsified by 0/9460 in these runs. **For
low-probability intermittent stalls, compute expected trials-to-reproduce
rather than burning cluster time on blind short repro attempts — either
run much longer single sessions, or leave cheap always-on diagnostics
enabled and wait for organic reproduction.**

**subgroup all_gather (NEGATIVE, still faults post-fix, 2026-08-21, this
session)** — `docs/allgather-lever-negative-result-2026-08-21.md`: tested
whether the same night's TCP-coordinator fix retired the precondition for
a known subgroup all_gather reliability bug. It did NOT — faulted
~0.4s into prefill (jaccl `wc.status=1`), request lost, needle FAIL. The
existing all_sum-based 2x-wire-bytes workaround remains necessary. **A
partial fix to a related transport issue does not guarantee it covers
all fault paths for a similar-looking lever — test small and cheap
before committing.**

**Chunk size / MTU tuning** — see §2.4 (`MLX_JACCL_RELIABLE_MAX_SZ`
sz=2→3 tested NEGATIVE, sz≥4 has documented hang risk, don't attempt
without new evidence).

---

## 9. Concurrency (c=2+) and PP vs TP topology

### 9.0 NEW(2026-08-24): the "~22% TP-vs-PP prefill gap" is an ARTIFACT — live head-to-head INVERTS it (WIN, decisive)

`docs/p5-tp-prefill-gap-2026-08-24.md` — the long-standing, never-root-caused
"TP ~350 vs PP ~450 prefill" lead was chased to ground with provenance
archaeology + the first-ever apples-to-apples live PP-vs-TP measurement on the
production checkpoint. **Verdict (b): the gap does not reproduce; it runs the
OPPOSITE direction.**

Live, identical probe (`bench/phase3_precheck_depth_throughput.py`, tokenizer
ground truth, needle-verified), same fp8 `DeepSeek-V4-Flash-0731`, same 100K
depth, fresh cluster for each arm:

| Mode | Prefill @100K | TTFT | Needle |
|---|---|---|---|
| **TP** (production) | **366.5 tok/s** | 192.49 s | PASS |
| **PP** | **277.0 tok/s** | 255.06 s | PASS |

**TP is +32.3% over PP.** (TP also re-confirms the §1 baseline: 366.5 vs
359.7, +1.9%, and matches the 2026-08-21 known-good 366.6 to 0.03%.)

**Four compounding provenance defects manufactured the phantom 22%:**
1. **Different checkpoint + quantization** — every PP ~450 figure was measured
   on `mlx-community/DeepSeek-V4-Flash` (**affine int8 PREVIEW**); production is
   `deepseek-ai/DeepSeek-V4-Flash-0731` (**fp8 e4m3**). `start_cluster.sh`'s own
   comment warns preview vs production differ structurally.
2. **The 1.42x `chars//4` counting artifact** — fact 1450 explicitly names fact
   1018 (the entire PP 364–512 curve) as contaminated; §3.5's own header already
   forbade quoting those numbers without checking the numerator.
3. **Depth swap** — the ~450s are 500K/94K/10K points quoted as if at 100K.
   (One source, `docs/profiling/request_lifecycle_trace.md`'s 467/439, is a
   *different model entirely* — Qwen3.5-397B.)
4. **THE DECISIVE ONE — chunk-loop rate vs TTFT rate.** PP's opening chunk-loop
   rate measured live today is **523 tok/s**, reproducing the historical
   490/512 claims *exactly*. But **55.8 s of PP-only first-token pipeline drain
   sits OUTSIDE the chunk loop** (runner logs "Prefill complete: 199.29s" while
   client TTFT was 255.06 s). TP's chunk-loop and TTFT rates diverge by **2.9%**;
   PP's by **34%**. Comparing PP's loop rate to TP's TTFT rate invents the gap.

**Exact additive decomposition of PP's chunk loop (closes to 188.1s measured):**
`PP1 baseline at PP's own opening rate 135.0s + PP2 stall outliers 19.5s +
PP3 depth-decay 33.6s`. TP's total for the same 70,656 tokens is **187.1s**.
PP1 alone *beats* TP by 52s — PP simply cannot hold it: PP decays **+63.4%**
across 70K with **5 stall outliers** (worst 9.6s), vs TP's **+5.1% and ZERO
outliers**.

**The structural asymmetry is real but self-cancelling.** TP moves 43x the
collective bytes per chunk (43 × 16.78 MB all_sum vs PP's 1 boundary transfer),
worth **4–9%** — bounded two independent ways: §2.3 arithmetic (215–516 ms of a
measured 5502 ms chunk) and fact 939's 93–95% GPU-busy (275–385 ms idle
headroom). PP repays all of it in bubble + decay, so **at the chunk-loop level
the modes are 377 (TP) vs 372 (PP) — statistically identical.**

**Consequences:**
- **The 2026-08-18 "prefill ceiling ~350–360, compute-bound" finding is UPHELD
  and STRENGTHENED.** A PP 450 was the only evidence straining it; PP is 277.
- **No TP-side prefill lever is proposed, and none should be.** Any comm-side
  lever addresses a ≤9% pool against ≤7% GPU idle; the collective/compute
  overlap cousin additionally inherits §2.6's correctness-race death and §0c's
  silent eval-order corruption hazard. fp8-native collective payload is capped
  under ~2% (§2.3: wire is only 2–9 ms of the 5–12 ms/call; the rest is
  payload-proportional but *non-collective-specific* stream-boundary cost).
  **Remaining prefill headroom is in COMPUTE, not communication.**
- **Side-finding: PP mode is NOT bit-rotted** — it launched clean on current
  code (`PipelineShardMetadata` 0–22 / 22–43, both runners READY, needle PASS).

**Reusable methodology lesson (→ §12):** *an instantaneous per-chunk progress
rate is only a valid throughput proxy when out-of-loop time is small. TP hides
2.8% outside its chunk loop, PP hides 25.5%. Never compare a chunk-loop rate
from one topology against a TTFT rate from another — always compare the same
denominator, and state which one it is.*

### 9.1 PP vs TP structural tradeoff (settled, WIN as a design decision)

`docs/fork-notes.md` (2026-07-31): PP gives **27-33 tok/s single-request**
with DSpark vs TP's ~15-20 tok/s; TP wins CONCURRENCY because collective
layers are stateless per-request while PP's pipeline layers hold mutable
per-request state, making concurrent requests wire-indistinguishable at
the `mx.distributed.send/recv` level. **This is structural, not a bug to
fix** — confirmed on both fork and upstream exo. Cluster uses TP for both
prefill and decode (per §3.5's hybrid-design rejection).

### 9.2 c=2 concurrency bugs (mostly fixed, some residual)

**B=2 quality bugs (WIN, resolved 2026-06-24)** —
`docs/b2-quality-handoff-2026-06-24.md` + `docs/b2-mtp-resolution-2026-06-24.md`:
root cause was NOT the originally-suspected cache-merge theory — it was a
**seq-split all_gather batch-unsafe bug.** Two root-cause bugs fixed,
verified across a full B=2 100K-500K needle sweep with MTP on, allowing
removal of the previously-required c≥2 safety gate.

**c=2 100K quality bug (NEGATIVE, invalidated a prior champion, 2026-05-23)** —
`.hermes/plans/2026-05-23_c2_100k_quality_bug_discovery.md`: discovered a
quality bug UNDERMINING the previously-shipped gamma=2 **34.16 tok/s
(σ=0.07)** "production champion" claim from 2026-05-22. **A throughput
win must be re-verified for quality/correctness at larger context/
concurrency before declaring it a production champion** — same lesson as
§5.1, recurring.

**MTP cache lifecycle for c≥2 (MIXED, 2026-05-20)** —
`.hermes/plans/2026-05-20_phase10_mtp_c2_fix.md`: fixed MTP cache
lifecycle bug (commit `cc200799`), reverted KV-bits default 4→0 (bf16)
per canonical prod constraint — but **c=2 100K bench still at 5.7
aggregate tok/s after the fix**, pointing to a SEPARATE verify-cost
bottleneck at B=2 long context, not resolved by this fix alone.

**c=2 MTP structurally broken at first attempt (NEGATIVE, 2026-05-20)** —
`.hermes/plans/2026-05-20_phase9_c2_findings.md`: c=2 MTP-OFF (bf16 KV)
31.4 aggregate tok/s (target 35, below). **c=2 MTP-ON (γ=2): STRUCTURALLY
BROKEN, 3.5-5.8 aggregate tok/s.** B=2 draft 147.34ms (30x B=1's 4.40ms),
verify 176.85ms (4x B=1's 42.81ms). Root cause: MTP cache lifecycle
assumed single-stream use, silently corrupted across concurrent streams
(cache reset on new submit clobbers other in-flight streams' state).
Also: 4-D masks (per-stream batch-rotating KV caches) force SDPA out of
fused causal kernel paths, multiplying verify cost — a uniform-offset
fast path (2D mask) is the mitigation. **Speculative-decoding/MTP caches
designed for single-stream use will silently corrupt across concurrent
streams — must be made per-stream-extendable before enabling
concurrency>1.**

**c=2 at 100K catastrophic slowdown / wedge (NEGATIVE, 2026-05-19)** —
`.hermes/plans/2026-05-19_phase_j_findings.md`: iter 0 warmup 4.5 tok/s
aggregate (2.3 per stream), wall 794s; **bench killed at iter 1, cluster
WEDGED, required hard restart.** **Docstring/marketing claims of relative
scaling (e.g. "2.7x scaling") can be relative to a weak baseline (naive
c=2), not vs single-stream c=1 — verify absolute throughput at target
context length, not just relative multipliers.**

**Batched-decode N=2 admission race (WIN, fixed, 2026-08-05)** —
`docs/batched-decode-n2-admission-handoff-2026-08-05.md`: 7 real bugs
found and fixed in the N=2 concurrent admission path (bypassing a
single-writer channel gate, conflating NACKs with timeouts in the retry
guard, rank 1 never propagating eviction signal). Verified with 4 rounds/
8 concurrent requests, zero crashes/500s/wire errors.
`EXO_PP_BATCHED_DECODE=1` is a verified working opt-in path for N=2
concurrency but **stays OFF by default.**

### 9.3 Co-hosting multiple models

`.hermes/plans/2026-06-10_dsv4_hyperconnection_fix_and_cohost_bench.md`
(WIN) — c=1 co-host (DSv4+Qwen3.6): total aggregate decode mean **78.5
tok/s** (Qwen ~51.2 tok/s/stream, DSv4 ~27.3 tok/s/stream). c=2: total
aggregate mean 73.0 tok/s — Qwen per-model went UP to 56.4 tok/s but DSv4
per-model roughly HALVED to 16.6 tok/s under contention. **Recommend the
balanced concurrency point (c=1) rather than assuming more concurrent
streams always helps — a lightweight model can starve a heavy one under
GPU contention with no net throughput gain.** (This doc also root-caused
a separate DSv4 garbage-output bug: silently-dropped Hyper-Connection
weights from a checkpoint/code weight-key naming mismatch under
`strict=False` loading — confidently-wrong output, not a numerical bug.)

### 9.4 TP decode capability audit (INFO_ONLY reference)

`bench/section101_tp_decode_capability_audit.md`: 43 real all_sum
collectives per decode token (1/MoE layer; attention fully replicated/
unsharded at decode, zero attention collectives). Cold shard+load ~18.7s
per rank. PP+MTP is not structurally blocked (a separate PP-native
speculative path already coexists). Sharding mode is baked into weight
tensors at one-shot load time — switching TP↔PP needs a full process
restart, no live re-dispatch seam, and no retained full-precision backup
after in-place sharding.

---

## 10. Memory leaks

**Correction (2026-08-21, added after an independent Fable cross-check
of this same raw data caught a real gap):** upstream MLX PR `mlx#3596`
(metal allocation coalescing, referenced in `docs/upstream-prs.md`) was
missing from this section entirely in the first pass. Real data: **RSS
growth rate improved 770→155 KB/token** with the coalescing change (A/B
throughput itself was flat at 32.1 t/s both configs — this is a memory
footprint win, not a throughput win). Filed here since it's the same
"Apple Silicon allocator defaults are tuned for small workloads and
cause stalls/aborts at this scale" theme as `MLX_MAX_MB_PER_BUFFER`
(§3.1) and the IOGPU residency abort below.

Two separate, unrelated leak investigations, both eventually WIN:

**Multi-turn memory leak #1 (WIN, resolved 2026-06-29)** —
`docs/dsv4-memory-leak-handoff-2026-06-29.md` +
`docs/dsv4-memory-leak-resolution-2026-06-29.md`: **+29.5 GB over a
139-msg/68-call session** (77.13GB→106.61GB), ~0.2-0.4 GB/turn, **+21
PoolingCache objects leaked per turn** (= layer count). Root cause: two
never-pruned `CacheSnapshot` accumulators — a `leaf_snapshots` merge
filter that never dropped entries because `restore_pos` climbs
monotonically, and dead write-only node-level snapshots in
`_build_edge_node`/`_split_edge` with zero live readers. Fixed by
bounding retention (cap=4) and removing the dead write path. Post-fix:
total GB pinned FLAT at 79.04GB for 11 consecutive turns. **Three earlier
fix attempts based on plausible-sounding hypotheses (deepcopy, generator
closing) all failed because they weren't the actual accumulation site —
walk gc referrer chains for growing `len()` on nested containers per
turn rather than guessing at fix sites. Bound any cache/snapshot list
that grows per-turn with an explicit retention cap rather than relying on
a filter condition that may never evaluate true in a monotonically-
growing session.**

**Multi-turn memory leak #2 (WIN, resolved 2026-06-27)** —
`docs/dsv4-memory-leaks-2026-06-27.md`: four DISTINCT leak sites (in the
prefix-cache/prefill path, separate from leak #1's MTP-cache path)
identified, fixed, and verified — memory plateaus across sequential
requests. **Memory leaks in prefix-cache paths compound across sequential
requests; verify fixes by confirming memory plateaus over REPEATED
requests, not a single run.**

**Related: IOGPU residency-set abort** — `docs/iogpu-residency-set-abort.md`
(NEGATIVE, environmental): a `MTLResidencySet::addAllocation` silent
process abort deterministically reproduced on Apple M4 Max 128GB under
LONG SUSTAINED workloads — this blocks long-running performance
benchmarks and is a distinct systemic issue from throughput tuning, not
something to fix via memory-leak-style debugging.

---

## 11. Correctness bugs found during perf work

Performance investigations repeatedly surfaced correctness bugs as a
side effect — worth tracking as its own category since several caused
false throughput conclusions elsewhere in this doc:

- **DSv4 confidently-wrong output** (2026-06-10) — silently-dropped
  Hyper-Connection weights (checkpoint `hc_attn`/`hc_ffn` vs model code
  `attn_hc`/`ffn_hc` naming mismatch, dropped under `strict=False`
  loading). Produced confident-but-wrong output, not numerical
  instability. §9.3.
- **Eagle K=1 cross-rank divergence** (2026-05-22) — un-synced
  `prev_logits` broadcast. §5.3.
- **Fused softmax kernel correctness break** (2026-08-21, this session) —
  `EXO_DSV4_FUSED_SOFTMAX` sat at default-off with a "needs A/B
  validation" comment for ~5 weeks (added 2026-07-14) before being
  actually tested — found to break needle-in-haystack correctness at
  100K context on first real test. **Optimization flags left
  default-off with an "unvalidated" comment can hide real correctness
  bugs for a long time — test opt flags before they accumulate
  untested.**
- **Collective eval-order determinism** (2026-08-20) — silent,
  deterministic cross-rank corruption if async_eval call order diverges
  between ranks. §2.6.
- **Token-tree drafting production-config bugs** (2026-05-19/20) — 3
  bugs (row-causal pmask for tree siblings, compressor pool-cache
  mutation during verify, KV/rollback discarding accepted-path context).
  §5.3.
- **Seq-split all_gather batch-unsafe bug** (2026-06-24) — masqueraded
  as a cache-merge quality bug. §9.2.
- **Thinking-parser delimiter fusion bug** — `docs/thinking-parser-fused-delimiter-fix.md`:
  exact-string-equality bug in `parse_thinking_models` caused a
  delimiter fused/split across streaming chunks to leak chain-of-thought
  into visible content. Pure correctness fix, not perf-motivated, but
  listed here since it's the kind of bug that could otherwise be
  mistaken for a streaming-performance artifact.

---

## 12. Measurement methodology lessons (meta)

> **⚠️ TOOLING HAZARD (2026-08-23, P2) — `xctrace` attach vs live
> prefill.** Do NOT attach `xcrun xctrace record --template "Metal System
> Trace"` to a live DSv4-Flash TP=2 runner during a long/deep prefill. It
> preceded a cross-rank collective stall and cost BOTH runners in **3 of
> 3 attempts** (including the "safe" simultaneous dual-rank design), at
> depths of 61K–78K tokens. This is **separate from and additional to**
> the SIGSTOP/watchdog false-positive fixed in `fc954293` — that fix is
> sound and its defer branch never even fired here (the stalls began 2–3s
> AFTER the tracer detached, and a 15.9s capture can never produce the
> 45s of silence the watchdog needs). Untraced controls at the same and
> greater depth are clean (135K+ this session with zero stalls; a full
> 300014/300014 completion on 2026-08-22). Decode-window and idle-window
> captures remain FINE (many prior successes, §2.7/T2) — the hazard is
> scoped to deep prefill. Mechanism NOT root-caused; leading unproven
> hypothesis is unified-memory pressure (large KV cache + non-trivial
> system-wide trace buffers), which is checkable from existing
> `memory_pressure`/`[MEM]` logs without another capture. Until then,
> profile prefill via idle-window or synthetic-load captures, or
> `mx.metal.start_capture()`. See
> `docs/p2-xctrace-prefill-collective-wedge-2026-08-23.md`.
>
> Useful figure recovered from the captures that DID complete: **prefill
> rank0 GPU occupancy is ~87–88%** (two independent short-window
> measurements: 87.40% / 88.32%), with ranks near-symmetric (76.12% vs
> 76.63% over a matched 120s window — but those longer windows are
> contaminated by the wedge onset and are a floor, not steady state).
> This refutes any lingering "prefill is mostly idle" framing.

These recur across many of the sections above — consolidated here as a
standalone checklist to run through before trusting ANY new throughput
number:

1. **Verify tokenizer chars-per-token assumptions** in any script that
   estimates token counts — a chars//4 assumption inflated prefill
   numbers by 1.42x for an entire investigation (§3.5).
2. **Cross-check log-parsed throughput against independent wall-clock
   measurement.** A `grep|tail-1` harness bug fabricated an entire false
   optimization narrative (§3.3).
3. **Never trust a champion number without ≥10 iterations at low σ.**
   Multiple "champions" in the 31-32 tok/s range failed to reproduce,
   one catastrophically (§5.1).
4. **Always pair a throughput claim with a quality/correctness probe**
   (needle-in-haystack). Several fast-but-broken configs were nearly
   shipped (§5.1, §5.3, §7.1).
5. **`mx.eval()` fences flush the ENTIRE pending lazy graph**, not just
   the requested output — a probe's first eval after an unevaluated
   backlog silently absorbs upstream cost and inflates the timed phase
   (§2.5, discovered as a repeat trap across 3 docs in the same
   investigation).
6. **Sync-span / forced-synchronization profiling has its own overhead**
   (measured this session: ~15% prefill / ~77% decode) — never trust
   ABSOLUTE tok/s from an instrumented run, only RELATIVE kernel
   percentages.
7. **Per-op isolated microbenchmarks can badly overstate recoverable
   savings** when the framework (MLX) already pipelines/overlaps the op
   chain — always validate with a PIPELINED chain-level microbench, not
   per-call-in-isolation (§4.6, the single most-repeated trap in this
   whole history, hit independently 3+ times across different fusion
   projects).
8. **GPU utilization/occupancy ≠ useful compute throughput** — a
   submission thread blocked in an uninterruptible collective wait can
   still read as "busy" (§2.6).
9. **Correlate probe/bench timestamps against an independent
   ground-truth signal** (e.g. a "Prefill complete" log line) before
   trusting a phase split in mixed prefill+decode data (§2.5, happened
   twice in the same investigation).
10. **A blended prefill+decode measurement window can mask which phase a
    cost belongs to** — always try to isolate the phase you actually
    care about (this session's own decode-isolation via SIGUSR1 showed a
    meaningfully different number than an earlier blended measurement,
    §2.1).
11. **Re-validate old bottleneck theories against CURRENT code** before
    trusting them — significant refactors can invalidate a profile from
    even a month prior (§3.2).
12. **Component-level microbenchmark wins don't guarantee end-to-end
    wins** if a different pipeline stage scales unfavorably with the
    same knob (§3.3, hit at least twice with STEP_SIZE tuning).
13. **Establish a theoretical ceiling (roofline) before an open-ended
    optimization sweep**, so effort has a clear termination condition
    (this session, §4.3).

---

## 12.5 Cross-domain recurring patterns

*(Consolidated in from an independent second pass over the same source
data — see the note at the top of this doc. These are the same
underlying facts as §12 above, but cut across topic-area boundaries
rather than within a single measurement technique, which surfaces a few
different groupings worth keeping.)*

1. **Microbenchmark wins do not transfer end-to-end — the single most
   common failure mode in this whole history, hit independently 4+
   times across unrelated domains**: STEP_SIZE=4096 (isolated MoE-GEMM
   +15% → e2e -8%, §3.3), fused-topk (5x per-call microbench → +0.13%
   e2e, below noise, §5.1/§6.5), fused indexer kernel (dispatch-overhead
   analysis predicted savings → 0.54x SLOWER once pipelined, §4.6),
   TurboQuant (bandwidth-savings math → -6% to -11% measured, §7.2).
   **No lever ships on a component-level number alone.**
2. **"GPU is idle/busy" reasoning misleads in both directions.** A
   2935ms GPU-idle envelope during `moe.switch_mlp` was NOT capturable
   by naive dispatch-fusion or fence-reordering (three independent
   failed attempts, §4.4); conversely, 96-97% GPU "utilization" telemetry
   coexisted with `moe.all_sum` eating 61-64% of wall time, because a
   blocked-but-scheduled submission thread still reads as "busy" (§2.6).
   Utilization telemetry alone is neither necessary nor sufficient
   evidence of where time actually goes.
3. **Correctness fixes and throughput fixes are independent axes —
   fixing one never implies the other.** Token-tree drafting (fixed,
   zero throughput lift, §5.3), MTP c≥2 cache fix (fixed correctness,
   throughput unchanged at 5.7 agg t/s, §9.2), shared-scale int8 all_sum
   (correctness passed, 1.49x SLOWER, §2.5), `TOPK=160` (fast, but
   quality-broken, §5.1). Every claimed champion needs BOTH a
   needle/quality probe AND a σ-qualified throughput run — neither alone
   is sufficient.
4. **Optimization verdicts rot across time and don't transplant across
   model/topology.** The identical MoE gate+up fusion measured -3.8%
   on 2026-06-26 and +3.01% on 2026-08-21 (§4.4/§6) — same code pattern,
   different codebase state, opposite sign. STEP_SIZE tuning has
   opposite signs on Qwen3.5/PP vs DSv4/TP (§3.3/§3.5). bf16→fp16 compute
   dtype is a ~7% win on `qwen3_5_moe` kernels but a 7x decode
   REGRESSION on DSv4 because JACCL/RDMA lacks fp16 support (§7.5).
   **Never trust an old result across a model, topology, or multi-week
   code-gap boundary without re-measuring on the current state.**
5. **Untested default-off flags are a real, recurring liability, not a
   theoretical one.** `EXO_DSV4_FUSED_SOFTMAX` hid a genuine correctness
   break for ~5 weeks behind a "needs A/B validation" comment before
   anyone actually tested it (§6). `EXO_DSV4_INDEXER_PBLOCK`'s own
   "decode pays zero overhead" docstring claim was false (§3.2).
   DSpark Native Head was implemented and never live-A/B'd (§5.3).
   **Either validate a flag promptly or delete it — don't let it sit
   untested indefinitely.**
6. **Speculative decoding's binding constraint at long context is the
   verify-phase KV-attention floor (~30ms, independent of draft
   width/depth) — every widening scheme loses to it.** Token-tree,
   higher gamma, and wider L_q all failed for the same underlying reason
   (§5.3). Every real win in this domain instead came from removing
   overhead around an unchanged draft shape: tie-break losslessness,
   rollback-cost fix, drain-elimination (§5.2) — never from a cleverer
   drafting algorithm.
7. **Exposed collectives sitting directly on the critical path are the
   recurring transport killer — not raw bandwidth.** Eagle K=1's 16KB
   `broadcast_from_canonical` collective with zero compute to hide
   behind caused a 17x regression (§5.3); `moe.all_sum`'s effective ~92-96
   MB/s (vs a realistic 6-10 GB/s wire capacity) traced to a chunking
   config knob, not the wire itself (§2.4); the decode-fence `mx.eval`
   is 98% of its own wait, not the jaccl poll (§2.7). **Fix collective
   ordering/overlap/chunking-config, not the wire, when a collective
   looks expensive.**
8. **Concurrency-tier transitions (c=1→c≥2) break things silently and
   repeatedly, across unrelated subsystems.** MTP caches (§9.2), output
   quality (twice, §5.1/§9.2), request admission (§9.2), and prior
   champion throughput claims (§5.1) all broke independently when
   concurrency was raised. **Any change validated only at c=1 is
   unvalidated at c≥2, and any change validated only at 100K context is
   unvalidated at 500K — re-verify explicitly at both axes, don't
   assume a lower-tier result generalizes upward.**

---

## 13. Open / never-finished threads

**NEW (2026-08-22, session 4): triaged 4 external tip items against
real repo history — 1 genuinely new+high-value (prefill FLOPs
roofline, promoted to T7, see below), 1 cheaply confirmed closed (T8,
prefill fence-gate audit, see above), 2 already substantially/fully
investigated and STALE as stated:**
- *"switch_mlp batch-size sweep B=1→8, untested at prefill's larger
  M"* — **already done, just not by that name.** §3's MoE-GEMM
  efficiency work (`docs/moe-vs-dense-qmm-isolation-2026-08-19.md`,
  `docs/lever1-moe-smallm-headroom-2026-08-20.md`) measured this exact
  kernel across the real prefill M-range: 62.6-72.0% of dense-mxfp4
  ceiling at L=2048-8192, and confirmed the existing small-M kernels
  already MATCH OR BEAT an idealized dense grouped-GEMM ceiling at
  production per-expert counts (0.63-1.07x). A follow-on kernel
  extension (`gather_qmv_rhs_lhs` for M>1) was actually IMPLEMENTED,
  deployed, and found NO-GO (`docs/lever1-moe-smallm-headroom-2026-08-20.md`)
  — real headroom exists only at decode's B=1 (T3's 27.7%), not at
  prefill's larger M. Do not re-run this sweep; it would reproduce
  already-committed numbers.
- *"sequence-chunk pipelining overlap never validated on real
  cluster"* — **false as stated; it WAS live-cluster tested and
  permanently closed.** `docs/prefill-chunk-overlap-live-test-2026-08-20.md`
  deployed the real mechanism on hardware, found a correctness race,
  which `docs/prefill-chunk-overlap-race-fix-2026-08-20.md` root-caused
  and fixed — but the throughput lever itself measured FLAT and was
  declared **"DEAD. This avenue is CLOSED... Do not re-litigate this
  lever for throughput."** (explicit standing decision, `EXO_PREFILL_CHUNK_OVERLAP`
  permanently OFF). Do not re-test.

Lesson for future external tips: always cross-check against
`PERFORMANCE_HISTORY.md` + a targeted `git log --grep` before acting —
this session's 197-file doc scrub means most "obvious next steps" have
already been tried, and re-running a closed lever wastes real cluster
time for a result already on record.

**NEW (2026-08-22, session 4): "82.5-84.9% unattributed" figure
recomputed against the post-async-fence-fix baseline, per a Fable
consult's flagged first step.** That figure was computed against the
pre-fix ~18.7 tok/s decode baseline and went stale the moment the fence
fix landed (decode -> 26.9-31.1 tok/s). Recomputed: **~73-81%
unattributed depending on context depth** (short ctx 72.9-74.6%, 100K
76.6%, 300K 78.7%, 500K 81.3%) — the headline number moved but the core
conclusion (large majority of wall time still unexplained) survives.
Per Fable's reframe: since GPU occupancy jumped 29-30% → 85.42% with
the fence fix, the investigation has now flipped from "why is the GPU
idle" to **"why is GPU-busy time still ~4-5x the roofline compute
floor"** — a kernel-efficiency/achieved-bandwidth question, not a
dispatch-gap question. Also flagged a real methodology gap not yet
resolved: the 6.51ms roofline floor used is active-MoE-weight-bytes
only and excludes KV-cache/attention read cost, which grows with
context — meaning the 500K-ctx 81.3% figure is likely overstated (real
compute floor is higher there than 6.51ms once KV-read cost is
counted); short-context/100K figures are less affected. See
`docs/decode-attribution-recompute-postfix-2026-08-22.md`. Next:
fresh post-fix Instruments capture with per-kernel labels (existing
capture data lacks them) + concurrent GPU clock/power log, to directly
attribute the real busy-vs-idle split now that the fence bug is fixed.

**NEW (2026-08-22, session 4, T2): fresh post-fix Instruments capture
confirms occupancy/gap/clock all dramatically improved, but per-kernel
attribution still structurally unavailable from this template.**
Attached `xctrace --template 'Metal System Trace'` to both live
production runner PIDs during two real `decode_probe.py` requests,
concurrent `powermetrics` on m4-1. Real interval-union occupancy
(own-process rows only, request-window-isolated): **78.64% (rank0) /
78.86% (rank1)** — up from pre-fix ~29-30%, roughly matching (not
identical to, different capture window) the Phase C pysampler
dual-capture's 85.42% figure. Gap-length distribution collapsed from
pre-fix median 520-528µs/mean 1,700-1,961µs (dominated by 0.5-20ms
buckets) to **post-fix median 89-95µs/mean 137-139µs** — now in the
range consistent with ordinary per-kernel CPU-dispatch latency, not
fence/collective-boundary-scale stalls. GPU clock: **median 1578MHz
(exact peak spec), 88.3% of busy samples at peak**, vs pre-fix
819-1122MHz never reaching peak; power draw ~19.5W median vs pre-fix
4.6-7.1W — confirms the earlier "clock is a downstream symptom of
bursty dispatch" prediction by showing it resolve once the bursty
pattern was fixed. **Decision-gate result** (per the Fable plan's
explicit criteria): occupancy 78.6-78.9% (idle ~21-21.4%) falls between
the <15%-closes and >25%-promotes thresholds — gap-chasing is not
fully closed, but its ceiling has shrunk to roughly +27% (1/0.786) at
most, a much smaller prize than the pre-fix "70% idle" framing implied.
Clock clearly clears the 80%-of-peak threshold, closing that decision
branch (clock is not an independent lever). **Real limitation
re-confirmed against fresh data**: `metal-gpu-intervals`'s
`gpu-channel-name` field is only ever "Compute"/"Fragment"/"Vertex"
(100.0% of our process's real GPU time was "Compute"); the
`formatted-label` field shows only positional "Command Buffer
N:Compute Command M" labels, never MLX kernel names — true per-kernel
attribution (isolating `moe.switch_mlp`/`GatherQMM`) needs
`mx.metal.start_capture()`/Xcode GPU Frame Capture, not this `xctrace`
template. See `docs/gpu-occupancy-clock-gap-postfix-2026-08-22.md`.

**RETRACTED 2026-08-22 (session 5, P0) — the 27.7% figure below is a
measurement artifact, do not cite it. The T3 bench called `mx.eval(y)`
inside its per-iteration loop ("serial-sync"), charging ~172µs/call of
host/dispatch overhead to the kernel. Re-measured with a
dependency-chained graph (one eval per 300 calls) and cache-defeating
rotated routing indices: 116-117µs/call = ~404 GB/s = 74% of peak
(87% with fully independent calls). The ablation matrix (dense-vs-gather,
bf16-vs-mxfp4-vs-affine8, B=1..32 sweep) found NO meaningful cost from
gather machinery, dequant, or B=1 sparse access — efficiency is flat in
B. Real switch_mlp cost ≈ 5.0ms/token ≈ 15% of decode wall time, not
39%. NO kernel optimization lever exists here. See
`docs/switch-mlp-bandwidth-artifact-retraction-2026-08-22.md`. Original
(wrong) finding kept below for the historical trail only.**

**NEW (2026-08-22, session 4, T3): switch_mlp/FusedSwitchGLU kernel
achieves only 27.7% of theoretical peak bandwidth — confirmed via a
real pipelined microbench at exact production shape/config.** Built a
standalone microbench replicating the REAL deployed kernel exactly
(verified before building, not assumed): `EXO_DSV4_MOE_FUSED_GATE_UP=1`
is live (must use `FusedSwitchGLU`'s single fused gather_qmm, not
vanilla 3-dispatch `SwitchGLU`); TP=2 shards MoE intermediate WIDTH not
expert identity (both ranks hold all 256 experts at HALF width,
1024 not 2048, per `auto_parallel.py`'s own 2026-08-16 corrected
comment); expert weights are mxfp4 (group_size=32, bits=4) per
`make_quantization_config()`. Real pipelined (300 iters,
`mx.synchronize()`-bracketed, per the standing pipelined-not-per-call
methodology rule) wall time: **290-312µs/call**. Real bytes touched/token
(top_k=6, per-rank, mxfp4+scale overhead): 47.186MB. **Achieved
bandwidth: ~151.4 GB/s vs 546GB/s M4 Max peak = 27.7% efficiency.**
Sanity-checked against the known decode budget: 43 layers × ~300µs ≈
12.9ms/token, which is 38.9% of post-fix short-ctx wall time (T1's
32.15-34.25ms/token) — directly matches the historically-cited
"~30-45% of wall time" span-breakdown figure, cross-validating the
microbench's fidelity to real production behavior. **Decision-gate
result**: 27.7% clearly falls in the plan's <40%-of-peak bucket —
confirmed real, substantial headroom at this specific kernel (3.6x gap
between theoretical and achieved bandwidth), not a "kernel already at
its floor" dead end. Root cause of the shortfall NOT yet isolated
(candidates: gather_sort/scatter_unsort overhead around the core
gather_qmm call; mxfp4 dequant overhead; B=1 decode's inherently thin/
scattered top-6-of-256 sparse-gather access pattern, most likely given
this is fundamentally a gather not a dense read) — flagged as the
natural next sub-step (e.g. a batch-size sweep from B=1 to B=8 to test
whether efficiency scales with batch, which would implicate the
access-pattern hypothesis directly). A real Metal GPU Frame Capture was
saved (`/tmp/switch_mlp_capture_session4.gputrace`, m4-1) but not yet
opened/analyzed in Xcode this session. See
`docs/switch-mlp-kernel-bandwidth-efficiency-2026-08-22.md`.

**NEW (2026-08-22, session 4, T8): audited prefill's blocking-fence
gate for the same silent-multi-owner-gate failure class that broke
decode's async fence — CONFIRMED genuinely by-design, closed cheaply.**
Prompted by an external tip (correctly informed by the real cache-owner
bug precedent) suggesting prefill's "blocking fence by design" claim
might be another stale, un-re-audited gate. Real gate code
(`deepseek_v4.py` line ~3050): `y.shape[1] <= 8` — an explicit, always-
visible numeric shape check, not a hidden multi-owner boolean.
Prefill's `y.shape[1]` equals the active chunk length
(`EXO_PREFILL_STEP_SIZE`, 2048 standing default, tested up to 4096/8192
— §3), always ≥64x over the threshold. This trivially and structurally
excludes every prefill call by construction, independent of the
`engine`/`cache` owner state that caused the DECODE bug. **Different
failure class from the cache-owner bug**: that one was invisible
(unregistered owner silently defaulting False forever, needed a stack
sampler to find); this one is a single visible comparison, settled by
reading the code once. Confirmed: no fix needed, no further
investigation warranted on this specific angle. See
`docs/prefill-fence-gate-audit-2026-08-22.md`.

**NEW (2026-08-22, session 4, T7): prefill FLOPs compute-bound roofline
— aggregate ~70% of ceiling on the GEMM-bound majority, but honest
end-to-end figure is ~49.8% with a real, un-decomposed 1.40x-headroom
tail flagged as genuinely open.** First attempt used a decode-style
top-down formula (2×active_params/peak-TFLOPS/TP) and got a nonsensical
"36-40% of peak" — caught via cross-check against already-measured
per-span data: reconciling it implied the unstudied remainder would
need -35.8% efficiency, physically impossible. Root cause: that formula
is a decode-style weight-touch-only heuristic that ignores attention's
context-length-scaling FLOPs (real evidence: prefill tok/s genuinely
decreases with depth, 359.7→324.1 @100K→500K, which a flat top-down
formula can't explain) — **do not reuse the decode roofline formula for
prefill; the regime is fundamentally different (compute-bound and
context-scaling vs decode's bandwidth-bound and context-independent).**
Corrected bottom-up approach aggregated 5 already-measured real spans
(attn.sdpa 13.6%wall/61.7%ceiling, attn.sdpa.compressed 11.8%/79.1%,
attn.o_proj 10.0%/83.2%, attn.proj_qkv 8.9%/84.7%, moe.switch_mlp
26.9%/62.6%), covering 71.2% of real prefill wall time, into a
wall-time-weighted figure never before explicitly synthesized: **9.00
TFLOPS achieved vs 12.86 TFLOPS blended ceiling = 70.0%.** A Fable
consult caught two real gaps before this was accepted as settled: (1)
70% is span-conditional, not end-to-end — the honest end-to-end figure,
counting the 28.8% non-GEMM remainder as ~0 useful FLOPs against wall
time, is **~49.8%**, implying up to **1.40x** theoretical headroom if
that remainder were fully addressed — comparable magnitude to decode's
real fence-fix win, explicitly flagged as NOT yet decomposed/checked
for a similar hidden bug (only the `moe.all_sum` 9.5% slice of it is
separately closed elsewhere); (2) the achieved-TFLOPS ceiling
denominators for `attn.sdpa`/`attn.sdpa.compressed` were never audited
for causal-masking/MLA-absorption FLOP-count inflation — flagged,
not yet checked. See `docs/prefill-flops-roofline-aggregate-2026-08-22.md`.
**Genuinely open next step**: decompose the ~19.3% non-all_sum portion
of the 28.8% remainder (layer.attn_hc/ffn_hc, residuals, norms,
attn.indexer, moe.gate/post_combine) with the same rigor as the decode
async-fence investigation.

**NEW (2026-08-22, session 4, T4): cross-rank all_sum skew — bulk
distribution symmetric (closes the primary question), but a real
4.2x rank0-leaning straggler asymmetry found in the rare tail.**
Reused the existing jaccl-internal `steady_clock` trace files from the
§2.7 investigation (no relaunch needed — the async-fence fix only
changes Python-side handling of `y` after `all_sum` returns, not the
C++ transport call being measured, so the transport-layer data isn't
stale for this question). Matched all 45,666 real decode-time 8192-byte
calls by `call_id` across both ranks (100% match rate). **Bulk result**:
median 36.1µs (rank0) vs 36.0µs (rank1), essentially a coin-flip on
which rank is momentarily slower per call (50.2%/49.4%) — clears the
plan's "<symmetric ~36-60µs → closed>" decision criterion cleanly, no
systematic one-rank-waits-for-the-other pattern in the bulk. **Real
secondary finding**: filtering to severe outlier calls (>1000µs
rank-to-rank difference, ~0.2% of all calls, 93 total) found **75
rank0-straggling vs only 18 rank1-straggling — a 4.2x asymmetry**,
consistent with rank0's slightly higher mean (66.3µs vs rank1's
58.9µs). Small aggregate magnitude (doesn't change the primary
conclusion), but a real, non-random, direction-consistent pattern —
root cause (TCP-coordinator role, hardware/thermal asymmetry, or
upstream scheduling artifact) not investigated further given the small
impact.
**UPDATE (2026-08-22, session 5, P6): aggregate impact QUANTIFIED from
the same traces** (`docs/allsum-straggler-aggregate-impact-2026-08-22.md`,
calc in `/tmp/p6_skew_calc.py`): summing per-call `max−min` over the 93
severe events gives 0.595s of fast-rank idle wait = 14.3% of total
effective all_sum time = **~1.7% of total decode wall time (upper
bound; ~1.3% attributable to the rank0-leaning portion)** — this
EXCEEDS the 0.5% "ignore forever" threshold at the upper bound, but is
an upper bound (assumes lateness is fully removable overhead, not
arrival jitter — for scale, summing max−min over ALL pairs gives 7.4%,
mostly un-removable jitter). Verdict: not conclusively ignorable;
bounded future investigation (correlate severe-event timestamps vs
rank0 master/control-plane activity) justified only if decode wins are
needed again. No action taken within the P6 timebox by design.
impact. See `docs/cross-rank-allsum-skew-2026-08-22.md`. **T4 CLOSED**
on the primary question; the tail asymmetry is flagged but not deemed
worth independent follow-up at this time.

**NEW (2026-08-22, session 4, T5): long-context (300K) GPU occupancy
capture — a real, informative NEGATIVE result for its hypothesis:
occupancy INCREASES with depth (82.4-82.7%), not decreases.** Fired a
real 300K-token prefill + decode against the live cluster, captured
fresh `xctrace` traces on both ranks (same methodology as T2). Real
decode: 22.03 tok/s, consistent with T1's known 300K baseline (24.44
tok/s). **Occupancy at 300K ctx (82.4-82.7%, both ranks) is HIGHER than
short-ctx (78.6-78.9%, T2)** — directly contradicting the naive
hypothesis this check set out to test ("decode slows at depth because
the GPU sits MORE idle waiting on growing attention/KV work"). Gap
median collapsed from ~90-95µs (short ctx) to ~1µs (300K) while
mean/p95 stayed comparable — consistent with far more back-to-back
sub-microsecond dispatch gaps within larger per-token attention
computation, not growing idle time. **Real interpretation**: the
context-scaling throughput drop (29.2→21.51 tok/s, T1) is NOT an
idle-time problem — it's straightforward increased real per-token
compute cost from larger KV/pooled attention shapes at depth,
independently corroborating T1's flagged caveat that the roofline
compute floor (6.51ms/token) excludes KV-read cost, which grows with
context. Same per-kernel-attribution limitation as T2/T3 re-confirmed
(channel names still 100% generic "Compute"). Real methodology gap
noted: decode window was only ~9s (a `bench=True`-routing quirk meant
EOS wasn't banned as intended) — result is internally consistent
across both ranks but would benefit from a longer confirmatory capture
if this becomes higher priority. See
`docs/long-context-gpu-occupancy-2026-08-22.md`. **T5 substantially
answered** — context-scaling slowdown mechanism identified as real
compute growth, not idle time; does not reprioritize T6/T10.

**NEW (2026-08-22, session 4, T6): MTP/DSpark TP-port decision gate —
NOT recommended, for two independent reasons, one of which is a
significant previously-unflagged staleness finding.** Per the plan's
gate ("<1.5x realistic ceiling → MTP becomes highest-EV"): (1) a narrow
kernel-fix-only ceiling estimate using T3's real switch_mlp finding
(38.9% of wall time, 27.7%→optimistic-65% of peak BW) gives **1.29x**
— below the 1.5x threshold, but incomplete (single-kernel-only, no
decode-side equivalent of T10's remainder investigation exists yet).
(2) **More significant**: the entire "MTP/DSpark is worth porting"
premise rests on `docs/fork-notes.md`'s 2026-07-23 PP+DSpark sweep
(27-33 tok/s single-request) compared against a TP baseline of
"~15-20 tok/s" — **that TP baseline is now known stale**, predating
both the MoE gate+up fusion AND the async-fence fix (+58-67%, this
session's headline finding). Redone honestly: PP+DSpark's unchanged
27-33 tok/s vs current TP's real 29.2-31.1 tok/s (T1) is only **-8% to
+13%**, not the +35-120% the stale comparison implied. **Nobody has
re-measured PP+DSpark since the fence fix** — it's equally plausible
PP+DSpark has its own undiscovered async-fence-class bug as it is that
TP has caught up. Recommended cheap next step (not executed this
session): re-run the 2026-07-23 PP+DSpark sweep as-is against current
cluster state before any porting decision — resolves the premise
itself before committing to multi-day TP-native speculative-decode
engineering. See `docs/mtp-dspark-tp-port-decision-gate-2026-08-22.md`.
**Standing recommendation**: T10 (prefill's 28.8% non-GEMM remainder,
real quantified 1.40x headroom via a methodology with a proven track
record this session) remains the single highest-EV item on the active
list — MTP/DSpark should not be prioritized ahead of it without first
doing the cheap re-validation sweep flagged above.

**NEW (2026-08-22, session 4, T10 first sub-investigation):
HyperConnection training-gate — a promising decode-fence-shaped lead,
FALSIFIED by direct production verification.** Investigating prefill's
28.8% non-GEMM remainder (T7), read `HyperConnection.__call__`'s
`use_ops = self.training or (not gpu) or (no metal) or env_var` gate
(chooses between a slow pure-MLX 20-Sinkhorn-iteration path `_hc_ops`
and a fast fused Metal kernel `_hc_kernel`). A standalone pipelined
microbench at real production shape found a real **4.32x speedup**
(_hc_ops 1517µs/call vs _hc_kernel 351µs/call) with near-identical
output (max diff 4.9e-4) — looked exactly like the async-fence bug's
shape. **Before treating this as a real bug, ran a Fable consult**
which correctly flagged that static code reading can't rule out any of
the 4 gate conditions differing in the live process, and load-time
state doesn't guarantee nothing flips it later. **Direct production
verification** (loading the REAL DSv4-Flash checkpoint via the exact
`mlx_lm.utils.load_model` function exo's real TP path calls, at the
real model path, `/Users/adam.durham/.exo/models/deepseek-ai--DeepSeek-V4-Flash-0731`):
`training=False` throughout (confirmed `load_model` calls `model.eval()`
before `load_weights()`, and this correctly propagates to submodules
constructed before that call — verified experimentally); device/metal
both clear; `EXO_HC_USE_OPS` confirmed unset via `ps eww` on the live
runner PID. **All four gate conditions clear the fast path — `_hc_kernel`
already fires in production. NOT a bug.** Root cause of my own false
lead: my standalone microbench never called `.eval()` on the
constructed module, and MLX's `nn.Module.training` defaults to `True`
— a genuinely reusable methodology lesson, not previously documented
in this repo: **any `self.training`-gated fast-path check must be
verified against `load_model`'s real propagation, not a standalone
reconstruction**, or it silently benchmarks the wrong branch. See
`docs/hyperconnection-training-gate-false-lead-2026-08-22.md`.
**T10 NOT closed** — this investigated only the single largest
remainder-span candidate (attn_hc/ffn_hc, 4.6% combined) and found it
already optimized; `layer.attn_residual`/`ffn_residual` (hc_expand,
4.4%), `attn.indexer` (4.0%), and `moe.gate`/`post_combine` (5.1%)
remain unread/unbenchmarked — a future continuation should apply the
same rigor to those.

**NEW (2026-08-22, session 4, T10 second sub-investigation):
attn.indexer's fused top-k lever closed cheaply via code reading alone,
no live test needed — second real NULL result.** Following the
HyperConnection false lead, checked `attn.indexer`'s existing
`EXO_DSV4_TOPK_FUSED` opt-in flag (live-toggleable via
`/tmp/dsv4_nop_targets`, docstring claims "~5x speedup at the pipelined
chain level") since a prior A/B of it (`docs/fork-notes.md`, older
MTP-era session) tested only decode and might not have been re-checked
for the current TP prefill regime. **Real finding, settled by reading
the gate condition alone**: `_fused_topk`'s dispatch requires
`scores.shape[1] == 1` — a structural decode-only shape condition
(`scores.shape[1]` equals `L`, and real prefill chunks run at
`L=2048`). **The fused top-k path cannot engage during prefill by
construction, regardless of the flag's value** — no live cluster test
was needed to settle this. Confirmed clean baseline via `ps eww` on the
live runner (neither the env var nor the NOP-target file were active
before this check). This upgrades the prior decode-only A/B from
"possibly stale for current architecture" to "confirmed still fully
applicable and closed" — real prefill top-k always uses `argpartition`
(a separate, already-tested lever) or fallback `argsort+slice`, never
`topk_fused`. See `docs/indexer-topk-fused-decode-only-2026-08-22.md`.

**T10 CLOSED (2026-08-22, same session, continued and completed after
a Fable-provided task list) — full decomposition, no async-fence-class
bug found.** Continued immediately in the same session rather than
deferring: ran a sum-check (Fable's flagged blind spot — real leaf
spans sum to 101.5% of real wall time using raw ms totals, no hidden
gap-between-spans bug; the earlier apparent 110% was a rounding
artifact from re-summing pre-rounded percentages), then investigated
all remaining candidate spans:
- **hc_expand** (attn/ffn_residual, 4.4%): corrected the FLOP-vs-
  bandwidth framing error Fable flagged (this op is memory-bound,
  ~150MB traffic, not FLOP-bound) — real gap is 1.93x over a corrected
  ceiling, not the initially-computed ~8.5x/100x. Found a real 1.41x
  speedup (cast the tiny `comb` tensor to bf16 instead of upcasting the
  large `residual` tensor) but **REJECTED on quality grounds**: real
  precision-check found ~1.08% mean relative error (max abs diff 0.125
  on realistic ±20-range activations), confirmed via a bf16-roundtrip
  control that this is genuine numerical divergence, not ordinary bf16
  noise — compounds across 43 layers, payoff is only ~1.3pp of prefill
  wall time, not worth shipping without full quality validation (not
  performed this session). Documented as investigated-and-rejected.
  **UPDATE (2026-08-22, session 5, P5): rejection RE-LITIGATED and
  UPHELD with multi-seed evidence** — the 1.08% error is stable
  (1.08–1.14% mean rel err, identical 0.125 max-abs, across 7 seeds),
  so the original single-check rejection was not a fluke. Root cause
  is quantizing `comb` itself to bf16: an fp32-accumulation variant
  with bf16 comb is WORSE (1.30–1.37%), and the only variant clearing
  the <0.2% gate (fp32 comb, broadcast-multiply+sum accumulation,
  ~0.000% err) is 1.6x SLOWER than production (5018µs vs 3104µs).
  No shippable variant exists in this design space; P5 CLOSED. See
  `docs/hc-expand-rejection-relitigated-multiseed-2026-08-22.md`.
- **moe.post_combine** (4.2%): `@mx.compile` gives ~0 speedup (1.01x,
  not broken). The apparent 6.1x span-vs-microbench gap was fully
  explained by reading the real code: the span genuinely wraps the
  full `shared_experts` MLP forward too, not just the elementwise
  combine (documented in the code's own comment) — my initial
  microbench mis-scoped what it was comparing against. Real implied
  shared_experts cost is **1.05x of theoretical peak — already
  optimal.** Not a bug.
- **moe.gate** (0.9%): clean code (argpartition, no host sync, already
  compiled), 1.54x gap is normal overhead. Timeboxed per the plan
  given the 0.9% cap on any possible win.
- **7 tail spans** (~2.5% combined): batch-triaged against the
  dispatch-latency floor; 4 exceeded 3x but each plausibly wraps 2+
  real sub-ops (e.g. `attn.rope_in` does both q AND kv rope calls) —
  not pursued to individual depth given the small aggregate magnitude.

**Final conclusion**: no async-fence-class hidden bug exists in
prefill's non-GEMM remainder. It decomposes into legitimate real
compute already near its own ceilings, one real-but-unshippable gap
(hc_expand), and the already-separately-closed `moe.all_sum` collective
cost (§2). **T7's 1.40x theoretical-headroom figure is reframed**: this
is a genuine architectural/dispatch-count ceiling for this model's
current design, not a to-be-found bug — the honest remaining margin
from everything investigated is closer to ~1.3 percentage points
(hc_expand, unshipped) plus low-single-digit-percent residual
efficiency in already-near-peak GEMMs. See
`docs/t10-final-decomposition-closed-2026-08-22.md` for the full
writeup. **No further prefill dispatch/gate-hunting recommended**
without new evidence (different architecture, different context
regime, or a new methodology beyond span-profiling).

**RESOLVED (2026-08-22, end of session 3, root cause found):** the
previous interim synthesis (below, superseded) flagged CPU profiling as
blocked pending `py-spy` install approval. Approval was given; `py-spy`
requires root/sudo which isn't available passwordlessly on either node,
so built a zero-privilege in-process Python stack sampler instead (per
a Fable consult — strictly more informative than an external sampler,
since it can also detect GIL-held-by-native-code vs genuinely-idle via
its own wakeup latency). **This found the actual root cause**: ~95% of
the real compute thread's decode-time wall time is spent blocked inside
the SYNCHRONOUS `mx.eval(y)` fence branch, not the intended
non-blocking `mx.async_eval(y)` path, despite `EXO_DSV4_FENCE_ASYNC=1`
being live all session. Live diagnostic logging (deployed the same
session) confirmed why: the gate requires two owner flags
(`"engine"` + `"cache"`) both `True`; `"cache"` is owned exclusively by
`dsv4_mtp.py`'s MTP/DSpark-specific code, which is confirmed DEAD under
this cluster's TP sharding mode (zero setter calls logged across a real
request) — so the fence can structurally never arm, regardless of
`EXO_DSV4_FENCE_ASYNC`'s value or real batch/sequence shape. See
`docs/pysampler-blocking-eval-root-cause-2026-08-22.md` and
`docs/async-fence-cache-owner-dead-code-root-cause-2026-08-22.md` for
full detail, including a real self-caught-and-corrected methodology
detour (a manual line-counting error that cost real investigation time
before being found via careful re-verification — documented as a
reusable lesson, not glossed over).

This is very likely the single largest concrete, fixable contributor to
decode's ~82.5-84.9% unattributed wall time identified this session —
a real, structural code defect, not a mysterious hardware/dispatch
limit.

**FIX IMPLEMENTED AND VALIDATED, same session (2026-08-22):** despite
the corruption-history caution above, a careful, registration-based,
fail-closed fix was designed (per an independent Fable design review),
implemented (`mlx-lm` `1fea494` + `exo` `6e427b549`), deployed, and
live-validated. **Real result: decode throughput 18.5 → 29.2-30.9
tok/s (+58-67%)**, correctness confirmed via three independent checks
including an exact-match needle-in-haystack test. See
`docs/async-fence-fix-validated-2026-08-22.md` and §2.8 above for full
detail. This closes what was flagged as "future session" work within
the same session it was found, once a genuinely careful fix design
(not a rushed patch) was available.

Cross-rank skew correlation (previously the other queued item) was
assessed as superseded by this direct-evidence finding — no longer
needed to distinguish "local stall" from "straggler wait" once a
structural cause is confirmed. A real technical limitation was also
found in the process: the sampler used `time.monotonic_ns()` (per-process
clock), not directly cross-rank-comparable even with a real measured
NTP clock-skew estimate (~1.1-1.2ms between nodes) — this remains a
real gap for future cross-rank timing work, though moot for THIS
specific question now that a direct-evidence root cause and validated
fix both exist.

---

**Prior interim synthesis (2026-08-22, superseded by the above):** see
`docs/decode-idle-time-investigation-interim-synthesis-2026-08-22.md`
for the full honest status — 5 of 7 planned investigation steps
completed with real findings this session (ruled out: `moe.all_sum`,
memory/expert-weight paging, pure CPU-dispatch latency; confirmed real:
GPU occupancy/clock/power reconciliation, real gap-length distribution,
sharpened 82.5-84.9% unattributed figure). 2 steps remain genuinely
open: true kernel-level attribution (needs a fresh Instruments capture
with per-kernel labels — existing data lacks them) and CPU-side
profiling (needs `py-spy` install approval, not given this session).
Next-session priority order is spelled out in that doc.

**Current highest-priority open question (2026-08-22, progressively
sharpened across this session, still not fully answered):** decode runs
at only ~12% of the theoretical bandwidth-bound roofline (§4.3) and
~28-30% real GPU occupancy on both ranks (§2.7 Instruments trace).
Stacking two independently-verified real measurements (real
`moe.all_sum` cost + theoretical compute floor) against real wall time
gives a sharper number: **82.5-84.9% of decode's real per-token wall
time is neither the collective nor unavoidable compute-floor cost**
(`docs/decode-time-budget-synthesis-2026-08-22.md`). Tonight's work
conclusively RULED OUT `moe.all_sum` as the cause (real transport cost
is only 2.9-5.3% of wall time, not the 21.4% earlier tooling claimed —
§2.7 arithmetic reconciliation).

**Further real progress this session** (`docs/gpu-idle-gap-deep-dive-2026-08-22.md`):
computed the real GPU idle-gap length distribution from existing trace
data (both ranks) and found gap TIME is dominated by 0.5-20ms gaps
(83.9% of gap time combined), NOT the 50-500µs range that would signal
pure CPU-dispatch latency — ruling out simple per-kernel dispatch
overhead as the dominant cause. Also reconciled a real apparent
contradiction between the Instruments occupancy figure (~30% busy) and
a live `powermetrics` reading (100% "HW active residency", 0% "idle
residency") — these measure different things (work-duration vs.
power-gating state); the GPU's real 4.6-7.1W power draw during decode
(far below a genuinely saturated workload) confirms the Instruments
occupancy figure, not the naive powermetrics idle-residency reading.
Found GPU clock frequency reduced to 819-1122 MHz (vs ~1.5GHz+ peak)
during decode — a real but likely downstream SYMPTOM of the same
bursty-low-load pattern (DVFS never ramps without sustained queue
pressure), not an independent root cause, though it IS a real
multiplicative amplifier while the gap pattern persists. A rough
decomposition (occupancy × clock-fraction ≈ 0.20) doesn't fully explain
the measured 0.12 roofline efficiency — a real, smaller, still-open
sub-mystery (~0.08 unaccounted) likely in per-kernel bandwidth
efficiency, not yet investigated.

**CONFIRMED via a direct real test (2026-08-22, Phase B of the
post-async-fence-fix investigation)** —
`docs/gpu-clock-symptom-confirmed-2026-08-22.md`: ran a sustained-load
probe (2000 back-to-back 4096×4096 bf16 matmuls, `mx.eval`-forced, no
artificial gaps) on the same physical node used for all other real
measurements this campaign. Real `powermetrics` samples during the run:
**GPU HW active frequency locked at 1578 MHz — 100% of that (the
topmost) P-state bucket, every single sample**, with real 55-57W power
draw (vs. 4.6-7.1W during real decode). **The same GPU hardware reaches
its real peak clock under guaranteed sustained load — the 819-1122MHz
range during decode is conclusively a downstream symptom of the bursty
dispatch pattern, not an independent throttling/thermal/firmware
limitation.** Closes this as a standalone lever; any future fix to the
underlying idle-gap pattern should raise clock as a side effect.

**Still genuinely open, not yet resolved**: (1) whether the dominant
0.5-20ms gaps align with per-layer `moe.all_sum`/fence boundaries
specifically — a per-token gap-rate check (~20-22 gaps/token from
existing data) did NOT cleanly confirm the expected 43/token match, but
this is inconclusive rather than a clean refutation (some layers' stalls
may be too short to register as separate merged gaps in the existing
low-granularity trace data); (2) true cross-rank skew-vs-shared-overhead
correlation was NOT performed — the two existing traces used
independent, non-synchronized clocks on separate machines, and a proper
test requires a clock-synced capture that hasn't been done; (3) the
existing `metal-gpu-intervals` trace data lacks per-kernel operation
labels (confirmed this session — only command-buffer-level granularity,
generic "Compute"/"Fragment"/"Vertex" channel names, no
`gather_qmm`/`switch_mlp`/`all_sum` labels), so true kernel-level
attribution requires a FRESH Instruments capture with a different
template/config, not further mining of existing data. Most promising
unstarted angles remain: a real Instruments trace of the
`moe.switch_mlp` kernel internals specifically (flagged below, never
done, needs the richer capture config), and/or a clock-synced two-rank
capture for the skew test.

**RULED OUT this session** (`docs/memory-residency-check-ruled-out-2026-08-22.md`):
memory residency / expert-weight paging from disk. Real pageins delta
across a full real decode request (495 tokens, 1.97s TTFT + 26.73s
decode) was only 1.0MB with zero swapins — not consistent with
per-token expert-weight paging. Confirmed the model shard is fully
resident (87.4GB via `vmmap --summary`'s `IOAccelerator (graphics)`
line, matching the expected ~83.5GB TP=2 shard size) after an initial
`ps aux` RSS reading (16.5GB) looked alarmingly low — resolved as a
real measurement-tool gap: `ps`'s RSS column excludes GPU/Metal-owned
unified memory on Apple Silicon, not a genuine residency problem.
**Reusable lesson: use `vmmap --summary`, not `ps aux`, to check real
memory footprint for MLX/Metal processes** — `ps`'s RSS undercounts
GPU-resident unified memory by the full size of the GPU working set.

Things flagged in the source docs as incomplete, unresolved, or worth
future investigation — check here before assuming a topic is fully
closed:

- ~~The 2026-07-02 `EXO_DSV4_FENCE_ASYNC` +28% claim vs this session's
  re-measured +1.04%~~ **RESOLVED 2026-08-22**: traced to MTP-PROF, a
  measurement tool whose own code comment documents it inflates costs
  via forced per-boundary synchronization — same artifact class as the
  sync-span profiler. See §2.6.
- **The ~340K prefill cliff's true root cause** — the sharp discontinuity
  was never fully explained; the tiled-P indexer memory fix didn't
  actually help (§3.2). This session's `INDEXER_PBLOCK` retest closed
  out ONE specific angle (small p_block causes decode regression at
  depth) but the original prefill cliff mechanism remains open.
- ~~`EXO_DSV4_DSPARK_NATIVE`~~ **SUPERSEDED 2026-08-22**: this entry's
  original framing ("out of scope for prefill-focused work, decode-only
  mechanism") is stale. Confirmed this session
  (`docs/roofline-sanity-check-inputs-confirmed-2026-08-22.md`):
  DSpark's decode loop is PP-only and never invoked under the TP
  topology production actually runs — `EXO_DSV4_DSPARK_NATIVE` (a
  DSpark sub-flag selecting which draft head to use) is moot for the
  same structural reason, not just out of scope. A live A/B of this
  flag would be testing a no-op under the current config. Not worth
  live-cluster time unless/until production switches to PP sharding.
- **Decode stall's "third undiagnosed symptom"** (rank0 CPU never
  converging to idle after two other bugs were fixed, §8) — investigation
  chain was abandoned mid-cascade.
- **Section 110's decode-stall root cause** — 6 hypotheses refuted, root
  cause of the 550-686ms/token PP decode stall still unknown as of that
  doc; two unproven leading candidates flagged (`ForwardStepInfo.queue_sends`
  context-var inconsistency, chunked-prefill KV-cache dependency-graph
  fragmentation) (§4, referenced via `bench/section110_decode_stall_last_candidate.md`).
- **DeltaNet kernel auto-selection** for Qwen3.5 pipeline-parallel prefill
  — projected 5-15% of DeltaNet time, never started (§3.5,
  `docs/prefill-optimization.md`).
- **Dual-stream overlap** for Qwen3.5 prefill — projected 0.5-1ms/chunk,
  flagged as high correctness risk, never started (§3.5).
- **A real Instruments Metal trace of the `moe.switch_mlp` GatherQMM
  kernel internals specifically** (as opposed to the generic matmul probe
  this session ran) — the single largest individual span (~30-45% of
  wall depending on phase) remains unexplored below the kernel-dispatch
  level.
- ~~The genuine unsync per-call `moe.all_sum` cost decomposed into
  skew-wait vs. real dispatch/scheduling overhead~~ **RESOLVED
  2026-08-21/22 (new session)**: real jaccl-internal `steady_clock`
  timing (not the rejected `perf_counter`-around-call-site approach)
  measured 45,666 real decode-time transport calls at median 36µs, mean
  58-66µs — the transport itself is fast (faster than the earlier
  isolated microbenchmark's ~120µs floor); the ~4094µs sync-span average
  and its 34x gap live almost entirely OUTSIDE the transport call
  (MLX's `mx.eval` fence, dispatch coordination, or Python-level
  scheduling — not yet further decomposed, now the correct next target).
  See §2.7 and `docs/jaccl-internal-timing-allsum-transport-fast-2026-08-21.md`.

**NEW (2026-08-23, P3 worker A): attention-path read bandwidth is NOT the
500K decode decay — code-derived byte inventory, and the TP all_sum
payload is provably depth-independent.** Pure code/config derivation (no
cluster load; read-only `cat` of the checkpoint config on m4-1). Derived
the exact per-decode-step attention memory-read inventory as a function
of context depth L: **`bytes_per_rank(L) = 5.298 GB + 1930 B · L`** →
5.491 GB @100K, 5.978 GB @352.6K, 6.263 GB @500K. Per-component scaling
laws: core attention on the 21 `SparseCompressedAttention` layers is
**CONSTANT in L** — it gathers exactly `index_topk=512` pooled rows
(`deepseek_v4.py:2527-2549`, OPT-10 comment: "does NOT scale with P");
the 2 `LocalAttention` layers are constant (128-entry rotating window);
the 20 `CompressedAttention` layers are linear but at slope L/128. **The
only material depth term is the Indexer's full-pool score GEMM** —
21 layers × L/4 entries × 128 dims × bf16 = **1344 of the 1930 B/token
(70%)** (`deepseek_v4.py:3800,3833`), plus 42 B/token for the exact-topk
kernel's 4 strided passes over the scores array (`deepseek_v4.py:3455,
3480,3510,3545`). KV dtype at runtime is confirmed **bf16, unquantized**
(`start_cluster.sh:151` `EXO_KV_CACHE_BITS:=0`; the 0-sentinel is honored
at `mlx/cache.py:2393`). **PREDICTION, not measurement**: at the
repo-measured 297 GB/s streaming BW
(`docs/dsv4-attention-kernel-efficiency-2026-08-18.md:28`) the byte
deltas imply only **+1.64 ms/token going 100K→352.6K** and +0.96 ms
going 352.6K→500K — ~3% of the ~54.6 ms/token baseline. **all_sum
verdict: shape is independent of L.** The ONLY collective active in TP
single-token decode is `moe.all_sum` (`deepseek_v4.py:3007`) at a fixed
`(1, 1, 4096)`; both attention-tail all_sums are dead because production
sets `EXO_DSV4_ATTN_ALLSUM:=0` (`start_cluster.sh:1755`), and both
seq-split all_gathers need `L >= 16` (`deepseek_v4.py:225`) so never fire
at decode. `auto_parallel.py:1141-1152` confirms experts are sharded by
intermediate WIDTH, so the reduced tensor is fixed "regardless of which
experts fired". If measured all_sum time grows with depth it must be
**arrival skew, not payload**. **Incidental correction worth flagging**:
exo never calls mlx-lm's `Model.shard()` — its TP path is MoE-only
(`auto_parallel.py:1032-1034`), so attention is fully REPLICATED and
TP=2 does not halve any attention byte figure;
`docs/dsv4-attention-kernel-efficiency-2026-08-18.md:38,52-55` assumes
the head-halving and is wrong on that specific point (depth verdicts
unaffected). **Scope discipline: this rules out ONE mechanism
(attention-path read bandwidth), not the attention path** — the indexer
scan may still be latency/occupancy-bound at large P, which bytes cannot
capture. See `docs/p3-worker-a-kv-read-inventory-2026-08-23.md`.

**NEW (2026-08-23, P3 worker B1): fresh live depth anchors with a REAL
EOS ban — the depth penalty is a UNIFORM per-token shift, and
`decode_probe.py`'s EOS ban is a no-op (root cause of T5's 9s window).**
Three live probes, one HTTP POST each, cluster untouched (`RunnerReady`
before/after). Real depths from `usage.prompt_tokens`, decode window =
last−first streamed event so prefill is excluded by construction:
**520 tok → 29.63 tok/s (33.75 ms/tok), 67.5s window; 100,026 tok →
27.94 tok/s (35.79 ms/tok), 71.6s; 352,599 tok → 23.48 tok/s (42.59
ms/tok), 85.1s.** All three delivered exactly 2000/2000 completion
tokens with `finish_reason=length` and `cached_tokens=0`. **Probe bug
found**: `bench/decode_probe.py` posts `{"bench": true}` to
`/v1/chat/completions`, but `ChatCompletionRequest` has **no `bench`
field** — pydantic silently drops it, so EOS is never banned; the ban
(`batch_generate.py:2658`) only fires via the separate
`/bench/chat/completions` route. Verified side-by-side live: `/v1`+bench
gave `finish=stop` at 56/60 tokens, `/bench` gave `finish=length` at
60/60. **This is the mechanism behind T5's flagged "`bench=True`-routing
quirk" 9s window** — T5 recorded the symptom, not the cause. **Main
finding — the depth penalty is uniform, not bursty**: the entire gap
distribution translates rightward (p10 18.0→26.4ms, p50 31.9→39.2ms,
mean 36.6→42.9ms short→deep) while the tail does NOT fatten (outliers
>3× median: 1.25% short vs 0.55%/0.76% deep) and dispersion FALLS
(stdev 55.6→16.3→19.6ms); the only multi-second stalls are at SHORT
context. This corroborates T5's "real compute growth, not idle time"
conclusion **by a different instrument on a 9x longer window**,
discharging its caveat. Decay: **+8.84 ms/token total (−20.8%) 520→352.6K**,
+2.05 ms/100K over the first 100K and +2.69 ms/100K over the next 253K
(mildly super-linear, not saturating). **vs T1**: short and 100K
reproduce within noise (+3.8% @100K); the deep point is **+9.2% above
T1's 21.51** — NOT claimed as an improvement, most likely T1's deep
number was measured on the same short-window/unbanned-EOS path;
confounds (window length, prompt content, n=1) not separated. **Note for
worker A**: its prediction arithmetic used a "~54.6 ms/token baseline"
at depth; the real measured value here is **42.59 ms/token @352.6K**, so
its ~3% bandwidth share should be recomputed against 42.59. Limitations:
n=1 per depth, 300K not re-run. New additive probe
`bench/p3_depth_anchor_probe.py` (read-only vs cluster); `decode_probe.py`
deliberately NOT patched mid-investigation — **recommend repointing it at
`/bench/chat/completions` and re-examining past results that used it**.
See `docs/p3-worker-b1-live-depth-anchors-2026-08-23.md`.

**NEW (2026-08-23, P3 reviewer R1): independent re-verification of workers
A and B1 — both hold; one incidental number in A refuted.** Re-derived
every central claim from the primary sources (code, an independent local
copy of the -0731 `config.json`, and the workers' own raw pasted output);
read-only, no cluster contact. **Worker A: 6/6 central claims CONFIRMED.**
MoE-only TP / replicated attention holds — `grep '\.shard(' src/exo/`
returns one unrelated hit, and mlx-lm's `n_heads //= N`
(`deepseek_v4.py:7170`) is reachable only from `sharded_load`
(`mlx_lm/utils.py:759`), which exo never calls; **`docs/dsv4-attention-kernel-efficiency-2026-08-18.md:38,52-55`
is confirmed WRONG on the head-halving.** bf16-unquantized KV, top-k=512
bounded gather, the 21-sparse-layer L/4×128 indexer scan (census recomputed
from config: 21×r4 / 20×r128 / 2×r0), and L-independent all_sum payloads all
verified at their cited (or ±3-line) locations. **The byte arithmetic is
exactly self-consistent**: components sum to 5,297,553,408 + 1930.25·L, giving
5.978 GB @352,599 and +1.6417 ms @297 GB/s — no table-vs-formula drift.
**REFUTED**: A's §4 aside that 6.9 KB/token is "~74× smaller than a naive
dense-MLA estimate (44 KB/token)" — the real ratio is **6.39×**. Cosmetic
only; it touches no formula and was never carried into this file. **Worker
B1: probe bug CONFIRMED from code alone** — `ChatCompletionRequest`
(`api/types/api.py:243`) has no `bench` field and no `ConfigDict`, so
pydantic's default `extra='ignore'` drops it; only `/bench/chat/completions`
(`main.py:1192-1197`) force-sets it for `batch_generate.py:2658`. **Anchors
arithmetically sound**: all three tok/s, ms/token and deltas reproduce from
the raw windows to ±0.03, decode-only accounting verified by the identity
`wall − TTFT == decode window` at every depth, and all three points show
2000/2000 with `finish_reason=length` and `cached_tokens=0`. Uniform-shift
distribution claims are internally consistent. **Both worker entries above
are correctly placed and free of drift vs their source docs.** One carry-
forward for the synthesis: use B1's measured **42.59 ms/token** denominator,
so A's attention-read share is **~3.9%**, not the ~3% its entry states.
See `docs/p3-reviewer-r1-verification-2026-08-23.md`.

**NEW (2026-08-23, P3 worker C): attention-path kernel wall time MEASURED at
depth — kernels are ~2x bigger than the byte model but explain only ~43% of
the live +6.80 ms; scaling above 100K is LINEAR, not superlinear.** Built one
real instance of each production attention class (`v4_attention_factory`, real
mxfp8/affine quantization per `make_quantization_config`, bf16 KV, all
`start_cluster.sh` env defaults) with a synthetic pre-filled KV/pool cache and
timed 256 consecutive real B=1 L_q=1 decode steps, then scaled by the true layer
census (2x r0 / 21x r4 / 20x r128). Run on **`adams-mac-studio-m4-2.local`
(rank1, production silicon)** from `/tmp`; nothing under `~/repos/exo` on either
studio touched; both runners `RunnerReady` before AND after; bench peak GPU
allocation **0.96 GB**. **43-layer attention-path ms/token: 12.88 @520, 16.57
@100,026, 19.13 @352,599, 21.52 @500,000.** Δ(100K→352.6K) = **+2.56 ms**
(+2.96 and +3.34 in two earlier runs under a different fencing mode) vs Worker
A's byte-model +1.19–1.64 and vs the live **+6.80**. So attention explains
**38–49%** of the live depth cost; **~3.5–4.2 ms/token lives outside the
attention block.** Absolute scale validated independently: the 520-token point
(12.88 ms) matches A's constant term 5.298 GB / 405 GB/s measured streaming =
13.1 ms to within 2%, and attention is 45–46% of B1's live per-token budget at
both depths — no 2–4x microbench/production mismatch of the kind
`dsv4-attention-kernel-efficiency-2026-08-18.md` warns about. **Scaling verdict:
LINEAR** — 3-point fit above 50K is `15.22 + 1.21 ms/100K` with residuals
within ±2% (two other runs ±2.4%), and the marginal rate goes +3.71 → +1.01 →
+1.62 ms/100K, i.e. a large fixed 520→100K step then a flat regime. **B1's
mildly superlinear end-to-end decay (+2.05 then +2.69 ms/100K) is NOT reproduced
by the attention kernels.** Component attribution: `_indexer_score` is the only
monotonic depth term (+0.405 ms x21 over the range) and it is **already at the
bandwidth roof — 477 GB/s @352.6K, 558 @500K, vs 405 GB/s measured streaming —
so it has zero headroom**; `_exact_topk` (+0.088) and compressed-SDPA (+0.261)
are latency-bound (8–12% of peak) but too small to matter; sparse gather and
sparse core SDPA are **flat in L**, empirically confirming A's O(1) claim.
Achieved GB/s *rises* with depth for every kernel — the kernels get **more**
efficient at depth, the opposite of the "kernels degrade at L" hypothesis.
Fork's own `EXO_DSV4_SECTION_TIME` sub-span fences agree: of the sparse layer's
depth growth, **94% is the `indexer` sub-span**; `proj_qkv`/`qk_prep`/`out_proj`/
`compressor` are flat. **Methodological finding worth carrying forward: fencing
discipline changes the answer by up to 2x.** A per-step `mx.eval+synchronize`
adds a measured **0.197 ms round-trip floor** (= 8.5 ms/token over 43 layers);
and naively chaining K steps under ONE fence keeps K pool-storage views alive,
defeating `PoolingCache`'s donation (`cache.py:1547-1556`) and inflating the
352.6K number as K grows (0.461→0.494→0.555 for K=4/8/32) while barely moving
100K. Use per-step `mx.async_eval`, which is what production does. Residual
+4.24 ms/token is NOT attributed here — candidates are MoE all_sum arrival skew,
inter-layer pipelining loss this bench structurally cannot see (biases the
attention estimate DOWNWARD), 85 GB-resident allocator pressure, and
intermittent pool-write donation failure (an isolated probe shows that failure
mode costs up to +6.35 ms/token over the same range — most testable follow-up).
New additive benches `bench/p3_attn_depth_walltime_microbench.py` and
`bench/p3_attn_subspan_attribution.py`; no existing script modified; nothing
committed. See `docs/p3-worker-c-attn-kernel-walltime-2026-08-23.md`.

**NEW (2026-08-23, P3 worker C2): live two-rank GPU busy/idle capture at ~100K
— occupancy 82.98%/83.06%, idle ~6.1 ms/token, ranks symmetric to 0.03
ms/token; the 352.6K counterpart was NEVER CAPTURED because the capture itself
probably killed a runner. Decision rule UNRESOLVED.** Reused T2/T5's read-only
`xctrace --template 'Metal System Trace' --attach` methodology on both live
runner PIDs (m4-1 46718, m4-2 45206), driven by B1's `build_prompt` + the
EOS-banning `/bench/chat/completions` route, with the capture fired 6.02s AFTER
the first decode token (prefill never traced, per §12's HAZARD) and both ranks
attached within 0.40s. **Measured**: occupancy **82.98% (rank0) / 83.06%
(rank1)** over a 50.55s decode-interior window, 214,182/213,816 own-process
interval-union rows; in-window decode **38.02 ms/token** traced (median gap
35.79 ms) vs B1's 35.79 ms/token untraced. **DERIVED** (arithmetic, kept
separate): on B1's untraced anchor, busy **29.70/29.73** and idle
**6.09/6.06 ms/token**, i.e. ≤0.142 ms per all_sum if ALL idle were the
collective (it is not — ordinary dispatch latency lives there too). **The two
ranks agree to 0.08pp of occupancy — no measurable arrival skew at 100K.**
**T5's 9s-window gap is closed at 100K, not at 300K**: a 5.6x longer window
confirms T5's qualitative claim (deep ~83% >> short-ctx 78.6-78.9%) and its
gap shape (median 0.92µs vs ~90µs short-ctx), but my 100K figure sits slightly
ABOVE T5's 300K 82.43-82.70%, suggesting a step-then-plateau, not monotone
climb. **Decision rule NOT evaluable on one depth.** Cross-run/cross-window
SUGGESTIVE only: derived idle is 7.21/7.13 (short, T2) → 6.09/6.06 (100K, this)
→ 7.48/7.37 (300K, T5) ms/token — a 6.1-7.5 band with NO monotone growth while
busy climbs 26.5→29.7→35.1; that is the direction of the "collective-wait ruled
out" branch, but three windows (30s/50s/9s), T5's non-EOS-banned 9s point, and
n=1 everywhere forbid calling it. **moe.all_sum verdict downgraded**: payload
L-independence stands (worker A, code-verified); wait-growth is NOT bounded —
only bounded ≤6.09 ms/token total idle AT 100K, symmetric. **INCIDENT + safety
update**: the traced 50s were metronomic (no stall >200ms), then 6.5s AFTER the
tracer detached a cascade of 12 stalls (200ms-2.6s) hit and rank1 died with
`[METAL] Command buffer execution failed: Caused GPU Timeout Error` in
`mx.async_eval(y)`; instance now `instances: []`, one `RunnerFailed` — NOT
relaunched (hard rule). Signature matches §12's post-detach pattern (stalls
began 2-3s after detach there). Suspected mechanism is **stop/finalize** load,
not recording: each trace was **10 GB while recording**, finalizing to 1.7 GB
over ~25 min, on an 85 GB-resident node. **This narrows §12's "decode-window
captures remain fine" claim — it does not extend to 50s captures at depth; the
risk scaler looks like trace size/duration, not prefill-vs-decode** (T5's 300K
capture was deep but only 9s and survived). Retry protocol: 12-15s windows,
decode-interior, deep point FIRST, expect ~25min finalization and post-detach
pressure. powermetrics SKIPPED (no passwordless sudo on either studio). n=1;
no `usage` block returned (depth inferred: same builder/target predicted
100,021 here, landed 100,026 on B1's identical target); channel names 99.98%
generic "Compute" so no per-kernel attribution, same limit as T2/T3/T5. Studio
`/tmp` traces+XML deleted (~24 GB reclaimed); nothing committed. See
`docs/p3-worker-c2-depth-busy-idle-capture-2026-08-23.md`.

**NEW (2026-08-23, P3 worker D): rank1 Metal GPU timeout during C2 capture —
crash forensics say MEMORY PRESSURE, not thermal, not a pre-existing fault;
kernel logged 2 GPURestarts. NEW OPEN ITEM — instance still DOWN, needs
restoring.** **Host↔rank correction first: the dead runner was PID 46718 on
`adams-mac-studio-m4-1.local`, which was rank1 this run — C2's doc has the
labels swapped** (proved by each node's own `mlx_distributed_init: Starting
initialization for rank N`, the jaccl coordinator bind/dial roles, and the PID
carrying the Metal error). C2's `100k_rank0`/`100k_rank1` occupancy blocks are
therefore swapped too; since they agree to 0.08pp, **no C2 conclusion changes**.
**Crash**: 13:51:43.416 CDT, `mx.async_eval(y)` in the DSv4 MoE ffn
(`deepseek_v4.py:3061`) on a plain `L_q=1` decode step (`inputs[:, None]`) at
~100K context, 2h41m into the runner's life. **Full traceback recovered** and
the failure is kernel-real, not bookkeeping: `kernel[0] (IOGPUFamily) … Deny
submissions/ignore app[] with **2 GPURestarts in 398 submissions**` at
13:51:48. Note `MTL_DISABLE_TIMEOUT=1`/`MTL_COMMAND_BUFFER_TIMEOUT=0`/
`EXO_DISABLE_METAL_TIMEOUT=1` were all set and **a timeout fired anyway** — those
knobs don't cover the IOGPUFamily watchdog. **Telemetry**: thermal **NEGATIVE**
(zero events either node). Memory **POSITIVE and tightly aligned** — jetsam
cascade at T−31s (≥25 daemons, `JETSAM_REASON_MEMORY_LONGIDLE_EXIT`, compressor
≈23.6 GB), then **26 swapfiles created in 18s ending in the same second as the
GPU timeout**. Baseline kills the "it always swaps" objection: m4-1's swapfile
bursts today occur **only** at 03:16-03:24, 10:55-11:10 and 13:51 — every one an
xctrace window; **the box does not swap in ordinary operation**. History clean:
**0 GPU-timeout/GPURestart events in 7 days on m4-1 and ever on m4-2**, 0 across
all 5 rotated `exo.*.log.zst`. Peer rank0 saw **no jaccl/RDMA fault** — healthy
37.5-38.5ms steps until one 85.10ms step at 13:51:13.975 (waiting on its dying
peer), then blocked in `wait_for_one` until the 45s hang-watchdog SIGKILLed it;
its thread dump gives **footprint 90.3 GB, peak 115.3 GB of 137 GB**.
**RANKED**: (1) xctrace stop/finalize **memory** pressure on a ~90GB-resident
node — `runningboardd` logged xctrace is **"not RunningBoard jetsam managed"/
"not memory-managed"**, so its 10 GB buffer takes RAM with no OS backstop;
(2) latent MoE-kernel fragility under allocator stress — **not separable** from
(1) with 99.98% generic "Compute" channels; (3) coincidence/pre-existing —
**refuted**; (4) finalize **disk** I/O — demoted, the apfs activity in-window is
swapfile truncation, i.e. an *effect* of memory pressure (refines C2's I/O
suspicion to a memory axis); (5) thermal — **ruled out**. **PRODUCTION vs
TRACING: primarily a TRACING-PROCEDURE risk** — trigger chain requires the
tracer's buffer, the box only swaps when tracing, and 7 days of untraced deep
decode (incl. worker C's 352.6K/500K benches) produced zero such events. **But
the caveat is real and P3-relevant**: 90.3 GB resident / 115.3 GB peak of 137 GB
at 100K means headroom is thin, and nothing here proves an untraced 500K decode
can't reach the same jetsam/swap regime on its own — measure it directly, don't
infer it from this crash. **Next time**: background `powermetrics` logger (the
one lever that would separate cause 1 from 2), 1Hz `vm_stat`/`memory_pressure`/
`swapusage` sampling through finalize, per-N-step `mx.get_active_memory()`, and
a live `GPURestart` tripwire; budget traces against **free RAM**, not just wall
time. **CLUSTER STATE (read-only, not restored)**: `instances: {}`; `6ac91846`
(m4-1/rank1) `RunnerFailed`; `f85456ee` (m4-2/rank0) shows `RunnerRunning` but
**the process is gone** (SIGKILL -9 13:52:20) — stale state. Both `python -m exo`
supervisors alive with full env; **memory fully released on both nodes (~133.5 GB
of 137.4 GB free, 97%, swap drained)**; no zombie runner, no stuck tracer, GPU
recovered. Needs a fresh DSv4-Flash instance placement — cluster processes do
**not** need restarting. Nothing modified/killed/committed on either studio; all
`/tmp` scratch removed. See
`docs/p3-worker-d-metal-timeout-crash-forensics-2026-08-23.md`.

**NEW (2026-08-23, P3 worker C3): PoolingCache donation-failure hypothesis
TESTED and REFUTED — production does NOT defeat donation. But the real
production cache class is `BatchPoolingCache`, which reallocates the ENTIRE
pool via `mx.concatenate` on EVERY flush: +1.91 ms/token of depth cost, ADDITIVE
with worker C's kernel delta, no double-count.** Worker C flagged donation
failure as the most testable candidate for the ~3.5-4.2 ms/tok residual (live
+6.80 minus kernel +2.56-+3.34). **Tested, not cited.** Method: drove the REAL
production decode loop (exo `ExoBatchGenerator.step()` →
`self._mlx_gen.next()`, `batch_generate.py:4228` → mlx-lm `BatchGenerator._next()`
→ `GenerationBatch._step()`, `generate.py:1564`) over all 43 REAL DSv4
attention blocks + real `model.make_cache()` pre-filled to depth, real sampler +
real bench-mode `ban_token_ids`, production env verified off the LIVE rank-1
process command line. MoE replaced by a depth-INDEPENDENT stub (out of scope,
cancels exactly in every delta) — **residual living in MoE-at-depth or
collective interplay is NOT covered**. Harness on `adams-mac-studio-m4-2`,
12.9 GB peak, 3 reps/point (spread ≤0.016 ms/tok). **KEY STRUCTURAL FINDING**:
`insert()` does not keep the caller's caches — `_merge_caches`
(`generate.py:1261`) converts `PoolingCache` → **`BatchPoolingCache`**
(`cache.py:1822`), verified at runtime (`LIVE CACHE CLASS: BatchPoolingCache`).
Worker C's microbench measured `PoolingCache`, which grows storage in
**256-entry chunks** (`cache.py:1522-1528`, so growth costs 1 flush in 256);
production's `BatchPoolingCache` grows via `mx.concatenate` to **exactly
max_pool, i.e. +1 entry, EVERY flush** (`cache.py:1899-1903`) — an
unconditional O(P·D) copy donation can never address. **MEASURED** (mod-4
flush-phase split, amortized ms/token, 100,026 → 352,599): (a) production as-is
+0.557 → **+2.504**; (c) donation maximally enabled (sync per step) +0.553 →
+2.493; (b) donation deliberately defeated (`EXO_DSV4_POOL_DEFER_COPY_MAX_BYTES=1<<40`)
+1.850 → +3.437; (d) **concat suppressed** (pool pre-padded) +0.019 → **+0.055**.
**(a)≈(c) within 0.011 ms/tok at both depths → donation is RULED OUT as the
residual's source, plainly.** (a)→(d) removes **98%** of the pool cost and the
mod-4 periodicity vanishes (p90 51.72→42.11 ms at 352.6K); per-step transient
`get_active_memory` drops **107.1 MB → 10.1 MB (10.6x)**, matching the 90.3 MB
compressor + 22.6 MB indexer pool exactly. Control (e) (concat suppressed AND
donation defeated) leaves +1.00 → +1.07 — **FLAT in depth**, i.e. the residual
donation-sensitive cost is fixed overhead, not O(P·D). **ADDITIVITY**: concat
cost (a)−(d) = +0.538 @100K → +2.449 @352.6K, **depth delta +1.91 ms/tok**.
Disjoint from C's +2.56 (C's 256-step window amortized its growth 1/64 vs
production's 64/64; had C paid it, C's r=4 layer would have been ~+0.48 ms/step
higher, which C did not observe). **C (+2.56) + C3 (+1.91) = +4.47, vs B1 live
+6.80 — NO overshoot, no double-count.** Explained fraction rises 38-49% →
**66%**; unexplained residual narrows **~3.5-4.2 → ~1.5-2.3 ms/tok** (still
open: MoE all_sum skew, inter-layer pipelining, 85 GB allocator regime,
MoE-at-depth). **NEXT ATTACK VECTOR (needs relaunch authorization)**: one-line
change at `cache.py:1899-1903` making `BatchPoolingCache` growth step-chunked,
gated by new `EXO_DSV4_POOL_GROW_STEP` (=1 is today's behavior bit-for-bit; =256
matches `PoolingCache.step`). Safe by the existing `_visible_width` mask clamp
(`cache.py:2174-2176`). **Expected live signature** via B1's probe: 352.6K
42.59 → **~40.14 ms/tok** (23.48 → ~24.91 tok/s, +6.1%), 100K 35.79 → ~35.25;
depth delta +6.80 → **~+4.89**. The **asymmetry (deep helped ~4.5x more than
shallow) is the diagnostic**; p90 at 352.6K should collapse ~9-10 ms toward p50.
**Falsification stated up front**: no change at 352.6K ⇒ mechanism is off the
live critical path and +1.91 is a stub-MoE-schedule artifact. Instance was DOWN
throughout — **no relaunch, no runner touched, no live A/B possible; all numbers
are harness-on-production-silicon, not live**. Nothing under `~/repos/exo` on
either studio touched; nothing committed. See
`docs/p3-worker-c3-donation-failure-insitu-2026-08-23.md`.

**NEW (2026-08-23, P3 reviewer R2): independent verification of workers C, C3
and D — C3's mechanism and arithmetic HOLD, but its A/B correctness rationale
is WRONG and one sanity number is 4x off; C's 520→100K delta OVERSHOOTS the
live delta and its doc never says so.** Read-only re-derivation from code and
the workers' own raw output (R1's A/B1 pass not repeated). **CONFIRMED**:
C3-1 `_merge_caches` (generate.py:1261) → `PoolingCache.merge` → **`BatchPoolingCache`**
(cache.py:1823); C3-2 the growth asymmetry is real — `max_pool = max(_pool_lengths)
+ max_new` so the concat pads **exactly +1 entry every flush** (cache.py:1899-1903)
vs `grow_by = max(self.step=256, …)` 1-in-256 (cache.py:1517-1528), and
`BatchPoolingCache` *cannot* carry slack because its length IS its capacity
(`size()` = `pooled.shape[1]`, :2532) while `PoolingCache` hides slack behind a
storage/offset view (:1360-1364); C3-3 all four no-holder cites (generate.py:1639,
:1651, batch_generate.py:971-981); C3-5 all arithmetic exact (0.557−0.019=0.538,
2.504−0.055=2.449, Δ**+1.911**, C 2.56+1.911=**4.471**=65.8% of live 6.80,
residual 2.33; (a)≈(c) to 0.004/0.011; mod-4 spikes verified at indices ≡3 mod 4
in C3's raw series; pool 90.27+22.57=112.8 MB vs 107.1 observed). **REFUTED (C3-4,
matters — PM is forwarding this A/B)**: growing the pool past `max_pool` is NOT
made safe by the `_visible_width` clamp — in both branches `visible = self.pooled`
**in full**, so `_visible_width == P` and `min(P, _visible_width)` (:2175-2176) is
a **no-op** on a trailing pad. The real invariant is the **length mask**
`pool_idx < pool_lengths` (:2177-2181). And `PoolingCache` relies on no such
invariant — it has **zero** `_visible_width` references (all 11 are inside
`BatchPoolingCache`). Patch still looks safe (pads score `finfo.min`, k=512 ≪ P),
but **the stated rationale must be replaced**, and the A/B has an unflagged
confound: padding flips `make_mask` from `None` to a `valid` array, switching the
indexer to the full `mx.where` path (deepseek_v4.py:3840/3858/3883) — arm B changes
two things, not one. **REFUTED (C3-6, cosmetic)**: "C's r=4 would have been ~+0.48
ms/step higher, ~+10 ms/token over **43** layers" — +10.014 is the whole-step
excess for all **21** sparse layers, and C's 0.5403 is a 256-step average with only
64 flushes, so the like-for-like adder is **+0.12 ms/step → +2.50 ms/token over 21
layers** (= C3's own concat cost, self-consistent). Still 13% of C's 19.13, far
outside its ±2% residuals, so **disjointness holds on corrected arithmetic**.
**C-1 CONFIRMED** (12.88/16.57/19.13/21.52, Δ+2.562, +2.962/+3.344 alt-fencing,
fit 15.2182+1.2139/100K resid +0.82/−1.92/+1.08%; totals recompute from per-class
medians to 19.129). **C-2 CONFIRMED AND UNADDRESSED — carry as a caveat**: C's
kernel delta over 520→100K is **+3.69 ms** but B1's *live end-to-end* delta on the
same span is **+2.04 ms**; a component cannot outgrow the whole budget. C's doc
compares the two spans only in *shape* (line 443-451, "the reverse curvature") and
never notices the overshoot; §6's sanity check only inspects the 100K/352.6K
points. This bounds how literally C's absolute deltas read — quote "explained ~66%"
with a one-sided error bar, not as a measurement. **D-1/D-2/D-3 CONFIRMED**: the
rank correction is backed by each node's own `mlx_distributed_init` line plus three
independent corroborations, the timeline is internally consistent (43.416 Metal →
48.043 GPURestarts → 52:04.34 reap → 52:06.17 peer SIGKILL, and the 45s watchdog
maths out from the 13:51:13.975 last-healthy step), and §4/§5/§8 explicitly separate
"best supported" from "not separable"/"proven" while arguing against its own
convenient conclusion. **But C and C3 both carry the SAME host↔rank swap D fixed
in C2** — both call m4-2 "rank1"; D proves m4-2 was **rank 0** this launch.
Label-only (attention is replicated; C3 is single-rank B=1), but don't propagate it
a third time. **HISTORY FILE INTACT**: all seven P3 entries present, contiguous,
chronological, none duplicated/truncated, zero conflict markers — A 2343-2383,
B1 2385-2424, R1 2426-2456, C 2458-2506, C2 2508-2553, D 2555-2609, C3 2611-2666.
No drift vs source docs on any headline number. **Two inherited errors live in this
file and need fixing**: line ~2648 repeats the "+0.48 ms/step" slip, and line ~2656
repeats "Safe by the existing `_visible_width` mask clamp" — the second is a
correctness rationale attached to a patch someone will write. Not fixed here
(read-only). See `docs/p3-reviewer-r2-verification-2026-08-23.md`.

**NEW (2026-08-23, P3 SYNTHESIS): 500K-context decode decay decomposed —
~66% explained (attention/indexer kernels +2.56 ms/tok + BatchPoolingCache
per-flush concat +1.91 ms/tok of the live +6.80 ms/tok 100K→352.6K),
residual ~1.5-2.3 ms/tok open; all_sum payload proven L-independent;
core sparse attention proven O(1); scaling LINEAR, not superlinear.**
Full synthesis: `docs/p3-synthesis-500k-decode-decay-decomposition-2026-08-23.md`.
Key corrections to the historical picture: (1) the "31% drop" was partly
probe artifact — decode_probe.py's EOS ban silently never worked via /v1
(no `bench` field in ChatCompletionRequest), so historical deep anchors
had short decode windows; clean 2000-token-window anchors give 29.63 →
27.94 → 23.48 tok/s (520/100,026/352,599 real tokens) = **-20.8%**
short→352.6K. (2) The depth cost is NOT idle time (C2: occupancy ~83%
both ranks at 100K, idle ~6.1 ms/tok, ranks symmetric to 0.08pp — no
arrival skew; corroborates T5 on a 5.6× longer window) and NOT KV-read
bytes alone (worker A: 1930 B/ctx-token/step linear slope predicts only
+1.19-1.64 ms; core attention reads top-k=512, O(1) in L). It is: real
indexer L/4-pool scan kernel time (linear, already at/above streaming-BW
ceiling — no kernel headroom) + an O(P·D) `mx.concatenate` pool realloc
on EVERY 4th-token flush in `BatchPoolingCache` (production converts
PoolingCache→BatchPoolingCache at generate.py:1261; chunked-growth
asymmetry vs cache.py:1517-1528 confirmed by R2). Donation-failure
hypothesis REFUTED in-situ (production loop ≈ donation-optimal within
0.011 ms/tok). NEXT: (a) live A/B env-gated chunked pool growth
(EXO_DSV4_POOL_GROW_STEP=256; expected ~+6% tok/s at 352.6K; carry R2's
two corrections: correctness rests on the length mask cache.py:2177-2181
NOT `_visible_width`, and padding flips the indexer make_mask path —
needs a control); (b) deep busy/idle capture under the new ≤15s-window
protocol to close all_sum-wait-at-depth; (c) probe-bug fix via git.
OPEN ITEM: rank1 Metal GPU timeout during C2's capture (worker D
forensics: tracer-finalize memory pressure best-supported, thermal ruled
out, primarily tracing-procedure risk BUT node runs 90.3 GB resident /
115.3 GB peak of 137 GB at 100K — headroom itself is a depth-scaling
risk). Cluster left DOWN pending operator relaunch authorization.

**NEW (2026-08-23, P3 FOLLOW-UP LIVE A/B — CONFIRMED WIN):
`EXO_DSV4_POOL_GROW_STEP=256` (BatchPoolingCache chunked pool growth) is a
REAL decode win — +9.79% tok/s at 352.6K ctx and +3.46% at 100K, measured
live on the 2-node TP=2 cluster. The pre-registered falsification condition
was NOT met.** Full report:
`docs/p3-followup-poolgrow-ab-2026-08-23.md` (Part II).

Measured, 2 arms x 2 depths, `bench/p3_depth_anchor_probe.py`, EOS genuinely
banned via `/bench/chat/completions` (all four runs `finish_reason=length`,
2000 completion tokens, `cached_tokens=0`, decode window >= 68s):

| depth (REAL prompt_tokens) | arm A (`GROW_STEP` unset=1) | arm B (`=256`) | delta |
|---|---|---|---|
| 100,022 / 100,023 | 28.09 tok/s / 35.60 ms/tok | 29.06 tok/s / 34.41 ms/tok | **+3.46% / -1.19 ms** |
| 352,602 / 352,601 | 23.50 tok/s / 42.55 ms/tok | 25.80 tok/s / 38.76 ms/tok | **+9.79% / -3.79 ms** |

Depth delta (100K->352.6K) in ms/tok: arm A **+6.95** (C3 predicted +6.80),
arm B **+4.35** (predicted +4.89) — the pre-registered deep>>shallow asymmetry
fingerprint is PRESENT and the depth-delta numbers were hit closely, though the
asymmetry ratio came in 3.2x vs the predicted 4.5x and absolute magnitudes ran
~2x larger than predicted at both depths (cost model underweighted the fixed
per-flush overhead relative to the size-dependent copy). Secondary signature
PARTIALLY met: p90 inter-token gap at 352.6K fell 70.53 -> 64.89 ms (-5.64,
predicted -9..10) and the p90-p50 spread narrowed 31.00 -> 26.68 ms.

VALIDITY. Arm A reproduced B1's independent anchors to **+0.53%** (100K) and
**+0.09%** (352.6K) against the pre-registered +-5% gate — that reproduction
sets a ~+-0.5% empirical noise floor, so the arm-B effects are ~7x and ~20x
noise. Both arms deployed by identical `start_cluster.sh` relaunch at exo
`7acf74c57` / mlx-lm `643d42d`, differing ONLY in the one env var. The uv.lock
pin-drift caveat was discharged directly: the venv `mlx_lm/models/cache.py` the
runner actually imports is byte-identical (md5 `f6b4201d…`) to the submodule on
BOTH nodes and contains the lever, and the var has exactly one consumer site.
`ps eww` on the real runner pid on BOTH nodes confirmed the var ABSENT in arm A
and `=256` in arm B — the §2 check that would otherwise have made the arms
silently identical. Zero foreign chat-completion requests in all four probe
windows; no probe rerun. TTFT unchanged between arms (1068.3s vs 1058.3s deep),
consistent with a decode-path-only mechanism. Arm B ran ~40 min AFTER arm A, so
thermals were, if anything, biased against it.

R2 CONTROL PASSED (live half). Arm B's deep output is not degraded: zero
U+FFFD, no repetition loops, coherent and correctly structure-aware at 352.6K
in both arms. The static half (padded columns always mask to False; k=512 <<
min pool length, so pads can never enter the top-k) was already proven — **but
that top-k claim is scoped to the SPARSE branch ONLY (correction 2026-08-23).**
`CompressedAttention` (deepseek_v4.py:4249 — the 20 ratio-128 layers) has NO
top-k at ANY context and attends the full padded pool; so does
`SparseCompressedAttention`'s compressed branch (:4614) below ctx ~2048. Only
`pooled.shape[1] > index_topk` reaches the top-k gather the claim describes.

ESTIMATOR CAVEAT (added 2026-08-23). +9.79% at 352.6K is the **usage-based**
estimator; the **events-based** estimator on the same run gives **+6.91%**. Both
are real. Quote the deep win as **+7 to +10%**, not a bare +9.79%.

CAVEAT ON ATTRIBUTION. Arm B changes TWO things — it removes the per-flush
concat AND raises the `make_mask` masked-path DUTY CYCLE (correction: not a
`None`→valid flip from nothing — arm A already builds a real validity array on
~25% of r=4 steps and ~0.8% of r=128 steps; arm B takes those to ~99% and ~75%).
Because the extra masked steps ADD work, they cannot
manufacture the observed speedup, so the causal claim about the env var is
sound; but attributing the gain specifically to concat elimination still wants
R2's slice-the-pad-off variant. That is now the follow-up for ATTRIBUTION, not
for disambiguating a null. Other limits: n=1 per cell, step=256 only (no sweep),
two depths only, `MLX_GPU_TIME` mod-4 spike check not run (would have broken
A/B parity).

⚠ SCOPE WARNING ON THE MEASUREMENT (reviewer, 2026-08-23). This win was
measured at **100K and 352.6K only**. Short and mid context were **never A/B'd**
by the raw lever, and the raw lever is **not numerics-preserving** there: the two
post-deploy temp-0 smoke generations, same prompt, same seed params, produced
**DIFFERENT text** (at ctx=20 the pool pads 5→256, K goes 133→384 with 251 masked
columns, and both consumers at that width have no top-k). Consequently the
default flip proceeded **ONLY in a gated form and ONLY after byte-identity
gates** — never as the raw lever measured here. See the next entry.

DEFAULT NOT CHANGED *by this run* — the lever remained opt-in and unset in the
committed default path at `643d42d`. **SUPERSEDED the same day — see the
default-flip entry below.** Cluster left RUNNING in arm B
(`EXO_DSV4_POOL_GROW_STEP=256`) as a runtime state only. Two relaunches this
session, both clean (`EXIT=0`, READY 2/2 in 8.6 and 6.7 min), no crashes, no
runner deaths.

**NEW (2026-08-23, DEFAULT FLIPPED — GATED FORM, BYTE-IDENTITY VERIFIED):
chunked `BatchPoolingCache` growth is now ON BY DEFAULT.** mlx-lm `0854b396`
(`EXO_DSV4_POOL_GROW_STEP` default 1 -> 256), exo `10357e570`. Tags
`known-good-poolgrow-default-20260823` on both repos. Full report:
`docs/p3-followup-poolgrow-ab-2026-08-23.md` **Part III**.

The reviewer approved a flip gated only on width (`_POOL_GROW_MIN=512`), on the
claim that past 512 "every consumer is on the sparse/top-k path." **That claim is
FALSE for 20 of 43 layers and the flip shipped with a third gate to fix it.**
`CompressedAttention` (deepseek_v4.py:4249, the ratio-128 layers) has NO top-k at
ANY context — it concatenates the whole pool into SDPA. An r=128 pool crosses 512
at ctx ~65.5K, so a width-only gate would still pad it at depth (2755 -> 2816 at
352.6K) and drift 20 layers. Shipped fix: `EXO_DSV4_POOL_GROW_MAX_RATIO=4` —
never pad a ratio-128 pool. Costs ~0.1-3% of the win (realloc byte-volume scales
as 1/ratio^2); measured cost was ~0.5 points of the ~9.8-point deep gain.

Neutrality now holds at EVERY context, not just below 65K: r=128 never pads;
r=4 below 512 valid columns never pads; r=4 above it pads but both arms are
`> index_topk` (same sparse branch, no flip), pads score `finfo.min`, and
`k = min(512, P)` never reaches past the valid columns, so the top-k selects
identical indices -> identical gather -> identical SDPA. The gate keys on
`min(_pool_lengths)`, not `max_pool`, so a ragged batch's short stream also
blocks padding (removes any dependence on MLX argsort tie-break stability).

BYTE-IDENTITY GATES: **3/3 PASS.** temp-0, fixed nonce-free prompt (prompt md5
verified equal), `use_prefix_cache=False`, 250-token windows, default build vs
`EXO_DSV4_POOL_GROW_STEP=1` escape-hatch relaunch, compared as RAW BYTES:
ctx 1,825 identical (1007 B); ctx 10,025 identical (1150 B); **ctx 70,016
identical (1081 B)**. The 70K point is the positive control — it is past the
r=128/512 crossover and was PREDICTED to diverge under the reviewer's
width-only gate; under the shipped ratio gate it does not. Contrast §10.5, where
the raw ungated lever produced DIFFERENT text at ctx~20 on an identical prompt.

DEEP RE-VERIFICATION on the final default build (no POOL_GROW var on either
node), `bench/p3_depth_anchor_probe.py`, EOS banned, 2000 completion tokens:

| depth (REAL) | tok/s | ms/tok | p50 | p90 | vs arm A |
|---|---|---|---|---|---|
| 352,643 | **25.32** (events 25.31) | 39.49 | 38.82 ms | 53.54 ms | **+7.7%** |
| 100,067 | **28.94** (events 28.73) | 34.55 | 33.82 ms | 52.95 ms | **+3.0%** |

Within noise of raw arm B (25.80 / 29.06) and far above arm A (23.50 / 28.09).
Both estimators now agree to 0.04% at depth. Text is real and on-task: both runs
recover the corpus structure AND its correct section count (7,923 at 352.6K,
2,263 at 100K — two different correct numbers), a content-dependent read that a
mis-masked pool could not produce. Zero U+FFFD, no repetition loops.

Three relaunches (default -> escape -> default), all `READY (2/2)` / `EXIT=0`,
no crashes, no runner deaths. Verified on BOTH nodes each time: SHAs, venv
`cache.py` md5 == submodule md5 (`5bc32bc0...`), the three new constants present
at :1312-1314, and `ps eww` showing ZERO `POOL_GROW*` vars on the default arm
(so the flip comes from the committed default, not a stale export).

STILL OPEN: attribution (R2's slice-the-pad variant was rejected as a shipping
form and never run as an experiment, so concat-elimination vs make_mask
duty-cycle is not isolated); no step-size sweep; `EXO_DSV4_POOL_GROW_MAX_RATIO`
remains env-tunable and setting it to 128 re-opens the closed hole; gates were
250-token windows at 3 depths with batch=1 and MTP/DSpark OFF, so they are
strong positive controls rather than a proof over all inputs.

---

**NEW (2026-08-24, P3 FOLLOW-UP — the all_sum-wait-at-depth measurement is
CLOSED: collective wait does NOT grow with depth; the residual is on-GPU busy
work).** Full report:
`docs/p3-followup-allsum-wait-at-depth-2026-08-24.md`.

The measurement C2 lost to the crash now exists: **same-build, two-depth,
dual-rank** xctrace decode-window captures. GPU occupancy **RISES** with depth,
so derived per-rank idle **falls**:

| depth (REAL) | rank0 (m4-2) busy % | rank1 (m4-1) busy % | window |
|---|---|---|---|
| 100,067 | **83.86** | **83.70** | 10.6 s |
| ~352.6K | **86.22** | **85.91** | 12.5 s |

Derived on Part III's clean same-build anchors (34.55 / 39.49 ms/tok):

| depth | rank | busy ms/tok | idle ms/tok | ≤ per all_sum call |
|---|---|---|---|---|
| 100K | rank0 / rank1 | 28.97 / 28.92 | **5.577 / 5.631** | 0.130 / 0.131 ms |
| 352.6K | rank0 / rank1 | 34.05 / 33.92 | **5.442 / 5.566** | 0.127 / 0.129 ms |

**Δidle = −0.07 to −0.14 ms/tok** (wrong sign, ~20x too small vs the
+1.0..+1.8 residual) while **Δbusy = +5.0 to +5.4 ms/tok** = 101–103% of the
total wall delta. Arrival skew DOES grow with depth (0.054 → 0.124 ms/tok) but
that **+0.070 ms/tok is only 4–7% of the residual**. VERDICT: **residual NOT
collective.** With worker A's code-proof that the payload is L-independent, the
`moe.all_sum` question is closed on BOTH axes — payload flat (code), wait
flat-to-shrinking (live, 2 depths, 2 ranks, same build).

ADDITIVITY DOES NOT CLOSE, and the residual GREW. Same-build total is
**+4.94 ms/tok** (39.49 − 34.55), not the +4.35 previously carried. Kernel band
+2.56..+3.34 and collective −0.07..−0.14 leave **+1.67 to +2.52 ms/tok
unexplained**, all of it on-GPU busy. Leading candidate is the bias worker C
flagged against himself (one-layer-per-class harness scaled by census captures
no inter-layer pipelining loss). Interval data agrees: at depth there are
**FEWER, BIGGER** GPU work items (112 vs 124 intervals/token, median interval
+9%, p90 +22%).

Also reproduces C2's 100K occupancy to within 0.8 pp on a 5x shorter window
(83.86/83.70 vs C2's 83.06/82.98), and **overturns C2's tentative "step up then
plateau" reading** — occupancy keeps climbing 100K → 352.6K. T5's 300K/9s figure
now looks low, as C2 warned it might.

HONEST HOLE: the method assumes collective wait appears as GPU *idle*. If MLX
encodes an event-wait INSIDE a command buffer it would be counted as *busy* and
hidden. Interval-shape analysis narrows but does not eliminate this (the whole
distribution shifts right, incl. the median, and intervals/token FALLS — the
signature of bigger kernels, not padded waits). The clean closer is a direct
CPU-side collective timer; `EXO_DSV4_ALLSUM_PROBE` exists (`deepseek_v4.py:3026`)
but forces a blocking `mx.eval` in place of the production `mx.async_eval`
fence, so it perturbs what it measures and needs its own A/B. Not run.

⚠ **THE ≤15s CAPTURE PROTOCOL IS NOT SAFE AT 352.6K — second crash of this
shape.** The 12 s dual-rank deep capture killed **BOTH** runners 16 s after
detach (`[METAL] GPU Timeout Error`, rank1 first, rank0 collateral). The 1 Hz
memory sampler run through the whole window makes worker D's mechanism a
measured curve and **localizes it to FINALIZE, not recording**: memory is flat
during the 12 s recording, then compressor goes 0→35 GB (m4-1) / 11→82.5 GB
(m4-2) and swap 140 MB→**15 GB**, with both deaths inside that window. Tracer
peak RSS is **~13 GB even for a 10 s trace**, against only **23.5/25.9 GB free**
at 352.6K (vs 33.1/33.5 GB at 100K). The real constraint is
`trace_peak_GB + resident_GB < RAM − margin`, NOT wall-clock. The single
permitted retry (10 s, at 100K) **survived cleanly** — probe ran to completion,
`finish_reason=length`, 2000 tokens, zero GPU timeouts. Cluster relaunched via
`start_cluster.sh` (production env, no `POOL_GROW*`): `READY (2/2)`, `EXIT=0`.
Traces DO survive the crash, which is why the deep point exists.

Rank labels: worker D's correction re-verified live on both the pre-crash and
post-relaunch cluster — **m4-1 = rank1 (API node), m4-2 = rank0 (coordinator)**.
C2's capture script had these backwards.

LIMITS: n=1 per depth; deep depth INFERRED (crashed run returned no `usage`
block; 100K is fully measured at 100,067); windows 10 s vs 12 s across depths;
~5% tracing overhead whose cross-depth *differential* (~0.4 ms/tok) exceeds the
Δidle reported, so the robust claim is **"idle growth is ≪ +1.0 ms/tok"** rather
than "idle shrinks"; 99.98% generic "Compute" channel = still no per-kernel
attribution; idle is whole-process GPU idle (includes ordinary dispatch
latency), so the per-call figures are CEILINGS by division, not measurements of
the collective; different runner PIDs across the two depths (relaunch sat
between them).

---

## 2026-08-24 — P4v2 M1 shadow gate: measured, verdict HOLD; incident recovery; cluster reverted to production

Full writeup: `docs/p4v2-m1-shadow-gate-results-and-recovery-2026-08-24.md`.
Build `34478792b` (M0 head-load gate + M1 env-gated DSpark shadow measurement).

The 2026-08-23 measurement session was killed mid-flight by API-provider
overload, leaving the cluster serving on the experimental shadow config with
what looked like a 3.4x decode regression (8.51 tok/s @100K) and an empty
diagnostic log. Recovery session findings: **neither was a malfunction.**
8.51 tok/s is shadow mode's designed cost (full draft+verify every cycle,
forced n_accepted=0, 1 token/cycle: 117.5 ms wall/cycle ≈ 110.3 ms measured
draft+verify + ~8 ms bookkeeping); the "empty" log was an artifact of (a) the
worker's relaunch #2 dropping `EXO_DSV4_SPEC_SHADOW_LOG` from the env and
(b) checking the wrong host — the real 782-cycle jsonl from relaunch #1 was
intact on m4-1 and is preserved at
`bench_data/shadow_gate_20260823/dspark_shadow_relaunch1.jsonl`.

**M1 gate numbers (the point of the exercise), 100K ctx, τ=0.5, block=5:**
a = 2.256 accepted/cycle (γ_mean 3.31, accept rate 0.681), draft 11.3 ms,
verify 99.0 ms (γ-linear: 53.5@γ=1 → 134.4@γ=5, ≈20.2 ms/row — +22% over
the cost model's A@100K=16.57). Projected speculative decode =
1000·(1+a)/110.3 = **29.5 tok/s vs 29 sequential = +1.8%**, break-even
a\*=2.199, margin 0.057 ⇒ **§D.4 HOLD band. DSpark speculation at 100K is
knife-edge at best under real verify cost; M2+ not funded.** At 2K it's
+17.7% (a=2.995) — but the north-star is 100K. Byte-identity gate FAILS
under shipped MoE config (`ROWSEQ_FULLBLOCK_MOE=0`/`PARTS_ROWSEQ=shared`
0.023%/row residual, documented) — deterministic across reruns but diverges
from production trajectory; any future byte-exact shadow run needs
`EXO_DSV4_ROWSEQ_FULLBLOCK_MOE=1`. Acceptance stats remain valid
(self-consistent vs walked trajectory). LIMITS: n=1/depth, 200/385-token
windows, no 10K or 352.6K shadow point, τ=0.9 never measured.

**Cluster reverted to production defaults** (relaunch #3,
`EXO_SPECULATIVE=0 EXO_DSV4_MTP=0 ./start_cluster.sh`, READY 2/2, EXIT=0).
This relaunch also live-validated **M0**: both nodes log `DSpark head load
SKIPPED (~10 GB/node reclaimed)` with the exact gate reasoning, 0 attach
lines. Post-revert verification: identity probe @2K = 30.05 tok/s,
byte-identical to the pre-shadow production output; decode probe @100K
below.

**Post-revert 100K decode probe: 28.73 tok/s usage (28.44 events), gap
median 34.2 ms / p95 60.4 ms, 600/600 tokens, coherent text — baseline
restored** (vs 8.51 / p95 189 ms under shadow config).

---

## 2026-08-25 — Launcher incident: Xcode removed from studios; CLT-fallback fix (no perf impact)

Full writeup: `docs/xcode-removal-launcher-clt-fallback-2026-08-25.md`.
Fix commit `70e0423bc` (`start_cluster.sh` only).

The arm-A launch for the `hc_collapse` fused-pre A/B aborted with
`Failed to sync on macstudio-m4-1`. Root cause: `/Applications/Xcode.app` had
been removed from **both** studios (only CommandLineTools remain), while
`start_cluster.sh` hardcoded
`DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer` at three remote-build
sites. Latent until commit `782c8cf97` touched `start_cluster.sh` and
invalidated uv's wheel cache — the first launch that actually had to *build* hit
it, dying inside `uv sync`'s maturin build of `exo_rs`
(`maturin` → `cargo` → `cc` → `xcrun: error: missing DEVELOPER_DIR path`).

Fix: `DEVELOPER_DIR` is now resolved **on each node** (Xcode if present, else
`/Library/Developer/CommandLineTools`), `xcode-select -s` is guarded on Xcode
existing, the `dirname $(xcrun -f metal)` PATH segment is added only when metal
resolves, and the mlx stamp-check's `NEED_BUILD=1` branch fails fast with an
explicit reinstall-Xcode banner (CLT has `cc` but no Metal compiler).

**No performance impact** — launcher-only, no model/kernel/config change. Perf
baselines from 2026-08-24 and earlier remain valid and directly comparable.
Residual risk: any future mlx C++ rebuild (an `mlx` submodule pin advance,
`MLX_FORCE_REINSTALL`, or a stale venv stamp) now requires reinstalling Xcode on
the studios first; the launcher will abort loudly rather than produce a broken
build. The current pin `e40a416b2` is stamped-good on both nodes, so the guard
does not fire on ordinary launches.

---

## 2026-08-25 — hc_collapse fused precursor kernel: live 2x2 A/B PASSES all framings (+1.89% mean prefill @ ~70.5K)

Full writeup: `docs/hc-collapse-kernel-ab-2026-08-25.md`. Kernel: fused Metal
precursor (`astype fp32` + `rms_norm` + `matmul fn.T`) for HyperConnection
collapse (`layer.attn_hc` / `layer.ffn_hc`), mlx-lm branch
`kernel/hc-collapse-roofline` @ `8d5de181d`, gate
`EXO_DSV4_HC_COLLAPSE_KERNEL=1`, default OFF (unset = bit-identical classic
path). Env forwarding: exo `782c8cf97` (opt-in, no default flip).

**Numbers** (`bench/phase3_precheck_depth_throughput.py --targets 100000
--max-tokens 128`, 2 runs/arm, ~70.5K real tokens, both arms exo `cd254d15a`,
`EXO_DSV4_HC_EXPAND_KERNEL=1` held ON, TP worldSize=2 fp8): arm A mean
**376.1681** tok/s (spread 0.03%), arm B mean **383.2700** (spread 0.37%).
Gate `mean(A) x 1.015 = 381.8106` ⇒ **PASS**. Deltas: +2.0947% B1−A1,
+1.6814% B2−A2, **+1.8880% mean-mean (+7.10 tok/s)**, conservative min-B vs
max-A +1.6814%. **All pre-registered framings pass.** Quality clean: needle
`FALCON-MERCURY-7749` byte-identical on 4/4 runs, zero U+FFFD, zero BOS spam,
zero `RunnerFailed`. Deploy discrimination verified per-PID on all 8 runner
PIDs (arm A env absent, arm B `=1`) plus venv greps (0 vs 3 hits/node).

**Caveats**: n=2/arm and the tightest passing framing clears the gate by only
+0.18 pp — smaller than arm B's own 0.37% spread; depth spread 235 tok (0.33%)
unnormalized; B1 reasoning-token outlier 60 vs 32/34/34 (same trajectory-
variance class as hc_expand's documented delta, final answers identical);
measured +1.89% is only **~70% of the predicted +2.73%** (span share 4.6% x
(1 − 1/2.47) from `docs/hc-collapse-roofline-2026-08-24.md`); decode (A 26.4677
→ B 28.8293) is noise-dominated at 41-69 completion tokens and NOT load-bearing.

**PM verdict SHIP — and SHIPPED.** Supervisor GO ~11:05; production flip executed
and verified **2026-08-25 ~11:20**. exo ship commit **`99f5f96b8`** = mlx-lm
submodule pin bump `7a1a4e868` → `8d5de181d` + `start_cluster.sh` default flip
`: "${EXO_DSV4_HC_COLLAPSE_KERNEL:=1}"` (revert = set `0`); mlx-lm fork `main`
fast-forwarded `7a1a4e8` → **`8d5de18`** and pushed. Production relaunched with a
bare `./start_cluster.sh` (no explicit gate — the script default promoted it),
READY 2/2 at 11:19:55 (~7.3 min); verified **kernel-ON on all 8 runner PIDs**
(m4-1 25937/25938/25939/25949, m4-2 27261/27262/27263/27272), both nodes on exo
`99f5f96b8` + mlx-lm `8d5de181d`, venv `HC_COLLAPSE` grep = 3/node, /state 2x
RunnerReady TP worldSize=2. Serving smoke clean (finish_reason `stop`, 206
completion tokens, zero U+FFFD, no BOS spam). Full record: §14 of the A/B doc.
Rollback: `EXO_DSV4_HC_COLLAPSE_KERNEL=0 ... ./start_cluster.sh`.

**Depth verification (same day, 2026-08-25 afternoon):** HOLDS at depth —
**+1.97% @300K-target** (211.7K real tokens, 364.97 vs 357.93 tok/s,
OFF×1.015 = 363.30) and **+1.89% @500K-target** (352.3K real, 345.53 vs
339.12 tok/s, gate 344.20). n=1 per arm per depth, verdicts per the
pre-registered gate (written before any probe, `/tmp/hccol_depth_preregistration.txt`);
no repeats ran because both deltas cleared +1.5% (the repeat branch was
reserved for the inconclusive case). Needle `FALCON-MERCURY-7749` exact
on 5/5 probes, zero U+FFFD, zero BOS spam. **No attenuation with depth**
(+1.89% @70.5K → +1.97% @300K → +1.89% @500K) — the pre-registered
mechanistic prediction (~+0.6%/+0.4%, hc_expand-style decay) was WRONG
in the pleasant direction. All four arms inside pre-registered sanity
bands; env gate verified 8/8 runner PIDs per arm with zero code delta
between arms (pure env flip, submodule NOT rolled back); SHAs unchanged
(exo `f7ef1180e`, mlx-lm `8d5de181d`). Production restored kernel-ON
via bare relaunch (script default `:=1` promoted it), READY 2/2 in
~330s, verified on 8/8 PIDs, smoke needle-exact. Full record:
`docs/hc-collapse-depth-verification-2026-08-25.md`.

---

## Quick-reference: closed levers, one line each

*(Added from the independent second pass's appendix table — a fast
scan-list for "has this been tried" before reading the full section.
Every row is expanded with real numbers and the "why" in its home
section above; use the section refs to jump there.)*

| Lever | Verdict | Section |
|---|---|---|
| Quantized/shared-scale int8 `moe.all_sum` | Infeasible (true reduction, not gather) / measured 1.49x slower | §2.5 |
| Token-tree speculative drafting (all variants) | Never beat linear baseline; verify-floor bound | §5.3 |
| Hybrid PP-prefill/TP-decode phase swap | Rejected twice, independently | §3.5 |
| TurboQuant KV compression | Every tested config regressed | §7.2 |
| MoE tile-geometry retune (bm>16), MAXBE widening | Kernel already at/above theoretical ceiling | §3.4 |
| Fused topk, fused indexer kernel, wq_a+wkv fusion | Noise-level or measurably slower once pipelined | §4.2, §4.6, §6 |
| `EXO_PREFILL_STEP_SIZE=4096` on DSv4/TP | ~8% end-to-end regression, confirmed twice | §3.3 |
| bf16→fp16 compute dtype on DSv4 | 7x decode slowdown — JACCL/RDMA lacks fp16 support | §7.5 |
| `MLX_JACCL_RELIABLE_MAX_SZ=3` | Statistically identical to sz=2, clean null | §2.4 |
| Expert co-location to reduce TP all_sum traffic | Traffic volume independent of expert placement | §2.1 |
| `EXO_PREFILL_CHUNK_OVERLAP` | Correctness race fixed, but lever itself dead/closed | §2.6 |
| DSpark FULLBLOCK at real context depth | 15.9x throughput collapse. **Corrected 2026-08-24 (P4v2): indicts the FULLBLOCK verify MODE at k=4 (block_size=5), not the DSpark head — the outlier occurred at 94% acceptance. Cluster does NOT run "DSpark OFF": `EXO_DSV4_DSPARK=1` is live; `EXO_SPECULATIVE=0` is what disables the loop.** | §5.3 |
| `MLX_METAL_FAST_SYNCH=1` | 1.5x slower, 70x more variance | §2.3 |
| `EXO_DSV4_FUSED_SOFTMAX` | Real correctness break (needle failure at 100K) | §6 |
| Nesting `mx.fast.metal_kernel` inside `mx.compile` | Catastrophic 3-4x regression | §4.5 |
| `EXO_DSV4_INDEXER_PBLOCK` (small block size) | Real decode regression at depth | §3.2 |
| Prefill cliff (~340K, 270→40 t/s, bimodal 8-32s stalls) | **RESOLVED IN PRODUCTION** since 2026-06-24 (breakthrough batch); mechanism identified 2026-08-24 as `active_memory > threshold` family — allocator gc-release + fork memory-branch throttle riding same crossing under era MB=50 per-primitive commits. Argpartition attribution PROVEN WRONG (`sort.cpp:342` identity + microbench parity + timeline). Positive attribution to MB=50→200 strongly-supported but shares credit with co-shipped OPT-6 / OPT-9. Amplitude / bimodality at Studio scale remain INFERRED / UNREPRODUCED. Live-confirmed 381K @ 328.6 tok/s 2026-08-24. | §3.2, `docs/prefill-cliff-mechanism-2026-08-24.md` |
| Subgroup `attn.all_gather` (vs all_sum workaround) | Still faults post-transport-fix | §8 |
| `all_sum` NOP ablation as a live measurement technique | Unsafe — destabilizes the cluster | §13 |
| TP=2 width-sharding as a skinny-GEMM efficiency tax | No penalty — sharded is 2.6-3.6% FASTER per unit work; sign inverted | §3.4 |
| `xctrace` Metal trace attach during live deep prefill | **HAZARD** — wedged the collective 3/3, killed both runners; use idle/synthetic captures | §12 |
| `xctrace` LONG (50s) decode-window attach at 100K depth | **HAZARD (2026-08-23, C2)** — runner died of Metal GPU Timeout 6.5s after detach; 10 GB trace, ~25min finalize | §12 |
| `xctrace` SHORT (≤15s) decode-window attach at 352.6K depth | **HAZARD (new 2026-08-24)** — the 12–15s cap is NOT sufficient at depth: a **12s** dual-rank capture killed BOTH runners 16s after detach (same GPU-Timeout signature). Live 1 Hz telemetry shows memory is flat DURING recording and collapses at **finalize** (compressor 0→35 GB / 11→82 GB, swap 140 MB→15 GB). Tracer peak RSS is **~13 GB even for a 10s trace**, vs only ~23–26 GB free at 352.6K. Real constraint is `trace_peak_GB + resident_GB < RAM − margin`, NOT wall-clock. Assume any Metal trace attach at ≥350K kills the instance; budget the run as sacrificial (traces DO survive the crash). A 10s attach at 100K (~33 GB free) survives cleanly | see `docs/p3-followup-allsum-wait-at-depth-2026-08-24.md` §6 |
| `EXO_DSV4_POOL_GROW_STEP=256` (BatchPoolingCache chunked growth) | **CONFIRMED WIN** — +9.79% tok/s @352.6K, +3.46% @100K, live A/B, output clean; default still opt-in | see `docs/p3-followup-poolgrow-ab-2026-08-23.md` |


---

## 2026-08-25 — DSpark Native + MTP Enablement: Phase 0 Static Audit (pre-registration, no cluster)

**PM:** GLM-5.2 (Ollama Cloud). **Repo HEAD:** `61efad499` (main, clean).
**Plans:** `/tmp/glm_plan.md` (authoritative), cross-ref `/tmp/dspark_plan.md`,
`/tmp/kimi_reasoning.md`. Cluster currently DOWN (unrelated debugger investigating
"no nodes available" placement for DeepSeek-V4-Flash-0731). No relaunch, no flag
flip, no baseline capture performed — only static code audit + pre-registration.
Full doc: `docs/dspark-mtp-ab-preregister-2026-08-25.md`.

### ⚠️ CRITICAL FINDING — DSpark head is REPLICATED, not SHARDED (~10 GB/node)

`DeepseekV4ShardingStrategy.shard_model` (`auto_parallel.py:1049-1180`) does NOT
shard `model.model.dspark`. It iterates only `model.model.layers` (loop
1062-1133, sets `layer.ffn.sharding_group` at :1087) and `model.model.mtp`
(`mtp_blocks` at :1059, loop 1153-1178, sets `mtp.ffn.sharding_group` at :1156).
**Definitive grep: zero `dspark` references in the sharding code** (only
mentions are PP tap-capture comments at :622-633). No `super().shard_model()`
call (base `TensorParallelShardingStrategy.shard_model` is `@abstractmethod`,
:869-873). No generic `modules()`/`children()` walk. The dspark stages DO
contain `DeepseekV4MoE` ffns (`deepseek_v4.py:6320` `DeepseekV4DSparkStage.ffn
= DeepseekV4MoE(...))`) that WOULD shard if reached, but the loop never sets
their `sharding_group` → cross-rank `sum_gradients` never fires → **head runs
replicated full-size on every rank (~10 GB/node, ~20 GB total), not sharded
(~5 GB/node).**

The overlay comment at `utils_mlx.py:370-372` ("Attached BEFORE tensor sharding
so its DeepseekV4MoE ffns shard exactly like the native mtp head's") is **FALSE**
— the sharding strategy was never updated to recurse into `dspark`. Confirmed
via second-opinion consult (no missed generic path). **Memory implication: the
enablement plan's ~5 GB/node sharded assumption is wrong; per-node cost is
~10 GB/node, 2× the assumption. A sharding code fix is required before a
sharded trial can run; until then any trial runs the head replicated.**

### Phase 0 audit results (all file:line verified)

| Item | Verdict | Evidence |
|---|---|---|
| Native head reads `mtp.0/1/2.*` from checkpoint | ✅ | `utils_mlx.py:879` (`_overlay_dsv4_dspark_native`), gated `:441` |
| Head attaches BEFORE sharding | ✅ | `utils_mlx.py:866,:1016` (`inner.dspark=mod`); dispatch at `:500` |
| DSpark head sharded across TP | ❌ **REPLICATED ~10GB/node** | `auto_parallel.py:1049-1180` no dspark; `deepseek_v4.py:6320` MoE; consult-confirmed |
| MTP head double-load | two distinct modules (sum costs) | `deepseek_v4.py:6500-6507` (`self.mtp`) vs `utils_mlx.py:866` (`dspark`); different roles |
| `EXO_DSV4_MTP_DEDICATED` default | ✅ unset (native MTP default) | `utils_mlx.py:358-362` |
| TP consumer double-gate (SPEC + MTP) | ✅ both required | `utils_mlx.py:421`; `dsv4_mtp.py:370-371` |
| No PP fallthrough in TP mode | ✅ | `utils_mlx.py:422-426` requires `PipelineShardMetadata` |
| Head-load gate single var | ✅ (FORCE_LOAD is measurement override, not 2nd key) | `utils_mlx.py:418-420` |
| `spec_degen_capture.py` intact + system+user triggers | ✅ | `bench/spec_degen_capture.py:37-89`; `--help` runs |
| Config dspark params | ✅ block_size=5, target=[40,41,42], markov=256, nextn=1 | `~/.cache/huggingface/.../config.json` |
| Native head weights on disk (node 0) | ❌ **ABSENT — blocker** | HF cache 6.1 MB, no safetensors |

### Proposed flag set (two-stage staging, Kimi-K3 idea — indicated given REPLICATED finding)

**Stage 1 (head-load validation, SPECULATIVE=0, zero decode risk):**
`EXO_DSV4_DSPARK=1 EXO_DSV4_DSPARK_NATIVE=1 EXO_DSV4_MTP=1 EXO_SPECULATIVE=0
EXO_DSV4_HC_COLLAPSE_KERNEL=1`. Unset: `EXO_DSV4_DSPARK_DIR`,
`EXO_DSV4_MTP_DEDICATED`, `EXO_PP_DRAFT_MODEL`, `EXO_DSV4_POOL_GROW_STEP` (isolate
per delta (a)). Gate: head loads without OOM, `mx.metal.get_active_memory()` +~10GB
(replicated, per node), config matches, decode works, `spec_degen_capture.py`
no diff. **If OOM → STOP; head doesn't fit replicated.**

**Stage 2 (full speculative, only if Stage 1 passes):** add `EXO_SPECULATIVE=1`,
`EXO_SPECULATIVE_GAMMA=2` (conservative — Eagle K=8 degenerated),
`EXO_SPECULATIVE_TEMP=0.0`, `EXO_SPECULATIVE_ALPHA=1.0`.

Acceptance (strict): ≥15% decode t/s @100K, ≥10% @352.6K, draft acceptance ≥60%,
TTFT ≤+10%, zero quality regression on trigger set, RSS <126GB/node, 0 swap, 50
clean steps. **No approval requested — all flag flips pre-registered only.**

### Blocked on live cluster

Phase 0.1 memory audit (peak RSS per node), Phase 0.7 weight verification on
node 1 (`mtp.0/1/2.*` safetensors absent on node 0), Phase 1 baseline capture
(Treatment A, no relaunch), Phase 2.2 single-node dry run, Stage 1/2 relaunch.
Cluster is DOWN; even if up, user must explicitly approve each relaunch
(session runs through the cluster).


---

## 2026-08-25 — DSpark Native + MTP Enablement: Phase 0 Static Audit (pre-registration, no cluster)

**PM:** GLM-5.2 (Ollama Cloud). **Repo HEAD:** `61efad499` (main, clean).
**Plans:** `/tmp/glm_plan.md` (authoritative), cross-ref `/tmp/dspark_plan.md`,
`/tmp/kimi_reasoning.md`. Cluster currently DOWN (unrelated debugger investigating
"no nodes available" placement for DeepSeek-V4-Flash-0731). No relaunch, no flag
flip, no baseline capture performed — only static code audit + pre-registration.
Full doc: `docs/dspark-mtp-ab-preregister-2026-08-25.md`.

### ⚠️ CRITICAL FINDING — DSpark head is REPLICATED, not SHARDED (~10 GB/node)

`DeepseekV4ShardingStrategy.shard_model` (`auto_parallel.py:1049-1180`) does NOT
shard `model.model.dspark`. It iterates only `model.model.layers` (loop
1062-1133, sets `layer.ffn.sharding_group` at :1087) and `model.model.mtp`
(`mtp_blocks` at :1059, loop 1153-1178, sets `mtp.ffn.sharding_group` at :1156).
**Definitive grep: zero `dspark` references in the sharding code** (only
mentions are PP tap-capture comments at :622-633). No `super().shard_model()`
call (base `TensorParallelShardingStrategy.shard_model` is `@abstractmethod`,
:869-873). No generic `modules()`/`children()` walk. The dspark stages DO
contain `DeepseekV4MoE` ffns (`deepseek_v4.py:6320` `DeepseekV4DSparkStage.ffn
= DeepseekV4MoE(...))`) that WOULD shard if reached, but the loop never sets
their `sharding_group` → cross-rank `sum_gradients` never fires → **head runs
replicated full-size on every rank (~10 GB/node, ~20 GB total), not sharded
(~5 GB/node).**

The overlay comment at `utils_mlx.py:370-372` ("Attached BEFORE tensor sharding
so its DeepseekV4MoE ffns shard exactly like the native mtp head's") is **FALSE**
— the sharding strategy was never updated to recurse into `dspark`. Confirmed
via second-opinion consult (no missed generic path). **Memory implication: the
enablement plan's ~5 GB/node sharded assumption is wrong; per-node cost is
~10 GB/node, 2× the assumption. A sharding code fix is required before a
sharded trial can run; until then any trial runs the head replicated.**

### Phase 0 audit results (all file:line verified)

| Item | Verdict | Evidence |
|---|---|---|
| Native head reads `mtp.0/1/2.*` from checkpoint | ✅ | `utils_mlx.py:879` (`_overlay_dsv4_dspark_native`), gated `:441` |
| Head attaches BEFORE sharding | ✅ | `utils_mlx.py:866,:1016` (`inner.dspark=mod`); dispatch at `:500` |
| DSpark head sharded across TP | ❌ **REPLICATED ~10GB/node** | `auto_parallel.py:1049-1180` no dspark; `deepseek_v4.py:6320` MoE; consult-confirmed |
| MTP head double-load | two distinct modules (sum costs) | `deepseek_v4.py:6500-6507` (`self.mtp`) vs `utils_mlx.py:866` (`dspark`); different roles |
| `EXO_DSV4_MTP_DEDICATED` default | ✅ unset (native MTP default) | `utils_mlx.py:358-362` |
| TP consumer double-gate (SPEC + MTP) | ✅ both required | `utils_mlx.py:421`; `dsv4_mtp.py:370-371` |
| No PP fallthrough in TP mode | ✅ | `utils_mlx.py:422-426` requires `PipelineShardMetadata` |
| Head-load gate single var | ✅ (FORCE_LOAD is measurement override, not 2nd key) | `utils_mlx.py:418-420` |
| `spec_degen_capture.py` intact + system+user triggers | ✅ | `bench/spec_degen_capture.py:37-89`; `--help` runs |
| Config dspark params | ✅ block_size=5, target=[40,41,42], markov=256, nextn=1 | `~/.cache/huggingface/.../config.json` |
| Native head weights on disk (node 0) | ❌ **ABSENT — blocker** | HF cache 6.1 MB, no safetensors |

### Proposed flag set (two-stage staging, Kimi-K3 idea — indicated given REPLICATED finding)

**Stage 1 (head-load validation, SPECULATIVE=0, zero decode risk):**
`EXO_DSV4_DSPARK=1 EXO_DSV4_DSPARK_NATIVE=1 EXO_DSV4_MTP=1 EXO_SPECULATIVE=0
EXO_DSV4_HC_COLLAPSE_KERNEL=1`. Unset: `EXO_DSV4_DSPARK_DIR`,
`EXO_DSV4_MTP_DEDICATED`, `EXO_PP_DRAFT_MODEL`, `EXO_DSV4_POOL_GROW_STEP` (isolate
per delta (a)). Gate: head loads without OOM, `mx.metal.get_active_memory()` +~10GB
(replicated, per node), config matches, decode works, `spec_degen_capture.py`
no diff. **If OOM → STOP; head doesn't fit replicated.**

**Stage 2 (full speculative, only if Stage 1 passes):** add `EXO_SPECULATIVE=1`,
`EXO_SPECULATIVE_GAMMA=2` (conservative — Eagle K=8 degenerated),
`EXO_SPECULATIVE_TEMP=0.0`, `EXO_SPECULATIVE_ALPHA=1.0`.

Acceptance (strict): ≥15% decode t/s @100K, ≥10% @352.6K, draft acceptance ≥60%,
TTFT ≤+10%, zero quality regression on trigger set, RSS <126GB/node, 0 swap, 50
clean steps. **No approval requested — all flag flips pre-registered only.**

### Blocked on live cluster

Phase 0.1 memory audit (peak RSS per node), Phase 0.7 weight verification on
node 1 (`mtp.0/1/2.*` safetensors absent on node 0), Phase 1 baseline capture
(Treatment A, no relaunch), Phase 2.2 single-node dry run, Stage 1/2 relaunch.
Cluster is DOWN; even if up, user must explicitly approve each relaunch
(session runs through the cluster).

---

## 2026-08-25 21:39 CDT — Post-reboot TB link verification + Stage-1 launch command prep (GLM-5.2 PM, commit pending)

**Context:** the TB/RDMA wedge diagnosed earlier (AppleThunderboltRDMA teardown
20:23:22 after a runner SIGKILLed; TB link dropped; all ports "No device
connected") was fixed by the user rebooting both Studios. This entry records
the read-only post-reboot verification + the exact Stage-1 launch command now
written into `docs/dspark-mtp-ab-preregister-2026-08-25.md`.

### Post-reboot TB/RDMA link — HEALTHY on both nodes (read-only SSH)

- **Node1** (`adams-mac-studio-m4-1.local`, uptime 4 min): `ifconfig` shows
  `inet 192.168.200.1` on the TB bridge, `status: active`; `system_profiler
  SPThunderboltDataType` shows 2 ports `Status: Device connected`;
  `pgrep python -m exo` = none; `ping -c5 192.168.200.2` = **0.0% packet loss,
  avg 0.849 ms**.
- **Node2** (`adams-mac-studio-m4-2.local`, uptime 4 min): `ifconfig` shows
  `inet 192.168.200.2` on the TB bridge, `status: active`; `system_profiler`
  shows 2 ports `Status: Device connected`; `pgrep python -m exo` = none;
  `ping -c5 192.168.200.1` = **0.0% packet loss, avg 0.608 ms**.
- **Verdict:** link fully healthy both directions, sub-ms latency, no stale
  exo procs. Wedge fix confirmed. exo NOT yet started (user's launch gate).

### Checkpoint + DSpark weights — PRESENT on both nodes (blocker resolved)

- The earlier "ABSENT — blocker" finding was checking the WRONG machine's HF
  cache. The cluster loads from `~/.exo/models/`, not `~/.cache/huggingface/`.
- Node1 `~/.exo/models/deepseek-ai--DeepSeek-V4-Flash-0731/` = 155 GB, 48
  shards; node2 = 165 GB, 48 shards. Identical `model.safetensors.index.json`
  SHA-1 `810e55576e2d29570d6b9a0ffaa8202f7cec1ea2` on both. 155G/165G delta =
  APFS sparse accounting, not content.
- **4705 `mtp.*` keys (mtp.0/1/2) packed inside shards 46-48** — the Phase 0
  `ls | grep mtp` returned 0 because it looked for standalone files; the
  weights live inside the unified shards. Native head weights ARE present.

### Stage-1 launch command — TWO corrections vs the original pre-reg

1. **`EXO_DSV4_DSPARK_FORCE_LOAD=1` added.** With `EXO_SPECULATIVE=0` alone,
   the head-load gate (`utils_mlx.py:427` `_dspark_usable = _tp_consumer or
   _pp_consumer or _dspark_force`) is False → head SKIPPED, not loaded. The
   original "head loads but no speculative drafting" was wrong for this code
   path. FORCE_LOAD=1 bypasses the consumer-reachability gate so the head
   actually loads; the draft cycle itself is gated separately at
   `batch_generate.py:813,822` (`use_speculative = SPECULATIVE==1`, cycle
   constructed only `if use_speculative:`), so SPECULATIVE=0 still guarantees
   zero decode risk. Confirmed via consult.
2. **`EXO_DSV4_MTP_DEDICATED=0` explicitly set.** `start_cluster.sh:468` has
   `: "${EXO_DSV4_MTP_DEDICATED:=1}"` inside the `DSV4_ENABLED=1` block, so
   the launch path defaults it to 1 (not unset, as the Phase 0 audit claimed
   by reading only the Python env default at `utils_mlx.py:361`). Without
   explicit `=0`, the external `mlx-community/DeepSeek-V4-Flash-MTP-bf16` head
   overlays `mtp[0]` before DSpark native runs — conflicting with
   `EXO_DSV4_DSPARK_NATIVE=1`'s intent.

### Exact approved Stage-1 launch command (user runs in tmux — approval gate)

```bash
tmux new-session -d -s dspark_s1 \
  'cd ~/repos/exo && \
   EXO_DSV4_DSPARK=1 EXO_DSV4_DSPARK_NATIVE=1 EXO_DSV4_DSPARK_FORCE_LOAD=1 \
   EXO_DSV4_MTP=1 EXO_DSV4_MTP_DEDICATED=0 EXO_SPECULATIVE=0 \
   EXO_DSV4_HC_COLLAPSE_KERNEL=1 \
   ./start_cluster.sh 2>&1 | tee /tmp/dspark_s1.log'
```

Each inline var maps to a verified `start_cluster.sh` EXO_ENV allowlist line
(DSPARK→1784, NATIVE→1819, FORCE_LOAD→1787, MTP→1728, MTP_DEDICATED→1921,
SPECULATIVE→1631, HC_COLLAPSE→2202). `DSPARK_DIR`/`PP_DRAFT_MODEL`/
`POOL_GROW_STEP` deliberately not exported → not forwarded → code sees its
`os.environ.get(..., "0")` default.

### Post-launch verification checklist (written into the pre-reg doc)

7 steps for the post-launch dispatch: (1) wait for `READY (2/2)` in
`/tmp/dspark_s1.log`; (2) env-var propagation audit via `ps eww` on both
nodes; (3) head-load log greps (`DSpark draft head attached (NATIVE...)`,
must NOT see `SKIPPED`/`overlay failed`); (4) memory audit (`footprint <pid>`
not `ps RSS` — Apple Silicon unified memory; <126 GB/node, 0 swap); (5)
decode smoke (coherent "hello world"); (6) **spec_degen baseline+diff
QUALITY GATE FIRST** — zero BOS-spam, zero period-≥3 loops on all 6
system+user triggers; (7) 50 clean decode steps, no desync/OOM.

### Status

**No relaunch run.** Cluster is back up, link healthy, weights present, exact
command + checklist written. **Waiting on:** user to explicitly approve and
run the Stage-1 `tmux new-session` command themselves (their established
pattern — relaunch kills this session). A separate post-launch dispatch will
run the verification checklist once they confirm.

---

**NEW (2026-08-26, DSPARK/MTP TWO-STAGE ENABLEMENT — BASELINE (Treatment A,
spec-off) MEASURED):** the spec-off baseline against which Stage-1 (head-load)
and Stage-2 (full speculative) DSpark/MTP arms will be compared. Cluster relaunched
fresh (`dspark_base` tmux) at exo HEAD `d0ef1f7f0` / mlx-lm `643d42d`,
`EXO_SPECULATIVE=0 EXO_DSV4_MTP=0 EXO_DSV4_HC_COLLAPSE_KERNEL=1` (HC_COLLAPSE
now default-ON since `61efad499`). DSpark head-load gate SKIPPED the head on
both nodes (confirmed in `~/.exo/exo_log/exo.log`: "DSpark head load SKIPPED
(~10 GB/node reclaimed): EXO_DSV4_DSPARK=1 but no runtime consumer is reachable
— EXO_SPECULATIVE=0 EXO_DSV4_MTP=0"). Env audited via `ps eww` on both runner
pids: `SPECULATIVE=0 MTP=0 HC_COLLAPSE_KERNEL=1 DSPARK=1 MTP_DEDICATED=1`
(DSPARK=1 + MTP_DEDICATED=1 are start_cluster.sh defaults, inert with
SPECULATIVE=0 + MTP=0). Sharding=Tensor, 2/2 runners READY.

Measured, `bench/p3_depth_anchor_probe.py`, one probe at a time, EOS banned via
`/bench/chat/completions` (`finish_reason=length`, 2000 completion tokens,
decode window >= 70s), `.venv/bin/python`:

| depth (REAL prompt_tokens) | tok/s (usage) | tok/s (events) | ms/tok | TTFT | decode window | gaps p50/p90 |
|---|---|---|---|---|---|---|
| 100,025 | **28.46** | 28.17 | 35.14 | 269.0s | 70.2s | 0.03 / 0.06 ms |
| 352,601 | **25.07** | 24.91 | 39.88 | 1011.4s | 79.7s | 0.04 / 0.07 ms |

Peak memory (footprint on the real `spawn_main` weight-holding child, sampled
during deep-probe decode): node1 91 GB (peak 104), node2 90 GB (peak 103).
`vm_stat` Swapouts=0 on both nodes (zero swap pressure). Headroom ~24-38 GB/node.

ANCHOR SANITY GATE (vs pre-registered 27.94 @100K, 23.48 @352.6K, ±5%):
- **100K: PASS** — 28.46 vs 27.94 = +1.9% (within ±5%).
- **352.6K: +6.8%** (25.07 vs 23.48) — outside the strict ±5% gate, but FASTER
  (healthier), not slower. Explained: the 23.48 anchor was measured 2026-08-23
  (poolgrow arm A, exo `7acf74c57`) BEFORE `EXO_DSV4_HC_COLLAPSE_KERNEL=1`
  shipped as default (`61efad499`/`f7ef1180e`, +1.89% @500K per the hc_collapse
  entry). The current baseline includes hc_collapse ON; the historical anchor
  did not. The +6.8% shift at 352.6K (vs +1.9% at 100K) is consistent with the
  hc_collapse depth-scaling gain. This does NOT invalidate the A/B: both
  Stage-1 and Stage-2 arms run at the SAME HEAD with the SAME hc_collapse=1,
  so the comparison isolates the DSpark/MTP effect. The pre-reg anchors are
  historical reference points, not the control — this measured baseline IS
  the control. Baseline @100K is the cleaner anchor reproduction (+1.9%).

QUALITY BASELINE (`bench/spec_degen_capture.py`, 7 trigger prompts, temp=0
greedy, max-tokens 200): **7/7 clean.** Zero BOS-spam, zero U+FFFD, zero
special-token leaks, all `finish_reason` normal (3 stop, 4 length-truncated).
Short prompts: "Paris", "One, two, three, four, five.", "red, yellow, blue".
Long prompts start coherent (Roman Empire essay, TCP handshake, 20 languages).
Saved to `/tmp/ab/baseline_degen.json`. This is the ground-truth the Stage-2
spec-degen diff will be compared against.

Raw artifacts: `/tmp/ab/baseline_100k.json`, `/tmp/ab/baseline_352k.json`,
`/tmp/ab/baseline_degen.json`, `/tmp/ab/baseline_degen_samples.txt`,
`/tmp/dspark_base.log`. Cluster left RUNNING in `dspark_base` (spec-off,
head not resident) pending Stage-1 relaunch.

---

**NEW (2026-08-26, DSPARK/MTP STAGE 1 — HEAD-LOAD VALIDATION PASSED):**
native DSpark head loaded + resident on both nodes, SPECULATIVE=0 (zero decode
risk), memory fits with ~39 GB headroom. First-ever on-cluster load of the
NATIVE checkpoint-bundled DSpark head (per enablement doc, "NATIVE has NEVER
been run on-cluster"). Relaunched `dspark_s1` tmux at exo HEAD `9ed2ee218` /
mlx-lm `643d42d`, flags: `EXO_DSV4_DSPARK=1 EXO_DSV4_DSPARK_NATIVE=1
EXO_DSV4_DSPARK_FORCE_LOAD=1 EXO_DSV4_MTP=1 EXO_DSV4_MTP_DEDICATED=0
EXO_SPECULATIVE=0 EXO_DSV4_HC_COLLAPSE_KERNEL=1`.

ENV AUDIT (`ps eww` on runner pids, both nodes identical):
`DSPARK=1 DSPARK_NATIVE=1 DSPARK_FORCE_LOAD=1 MTP=1 MTP_DEDICATED=0
SPECULATIVE=0 HC_COLLAPSE_KERNEL=1`. All 7 flags reached both runners.

HEAD-LOAD GREP (both nodes, `~/.exo/exo_log/exo.log`):
```
DSpark draft head attached from .../deepseek-ai--DeepSeek-V4-Flash-0731
  (NATIVE checkpoint-bundled head, 115 tensors, 3 stages, block_size=5,
   taps=[40, 41, 42]).
[DSPARK-GUARD] provenance=native ... missing=0 extra=34 ... block_size=5
  markov_rank=256 n_stages=3 taps=[40,41,42] noise_token_id=128799
  CHECKPOINT_PROVENANCE=MATCHED
```
No `SKIPPED`, no `overlay failed`, no fallback warnings. The
`param_tree_assert=FAIL` + `extra=34` is a WARNING about MXFP4 `.scales`
quantization tensors the assertion doesn't enumerate — `missing=0` (no missing
weights), `CHECKPOINT_PROVENANCE=MATCHED`; the head loaded correctly.

MEMORY AUDIT (footprint on `spawn_main` child, post-load idle):

| node | baseline (spec-off) | Stage 1 (head loaded) | delta | peak | swap |
|---|---|---|---|---|---|
| m4-1 | 80-83 GB | **89 GB** | **+6-9 GB** | 94 GB | 0 |
| m4-2 | 80-83 GB | **89 GB** | **+6-9 GB** | 94 GB | 0 |

Delta is the replicated native head (~10 GB estimate per Phase 0.2, measured
~6-9 GB — under estimate). Headroom **~39 GB/node** (128-89), well above the
1 GB min and 126 GB safety ceiling. `vm_stat Swapouts=0` both nodes. **Head
fits; no OOM; no headroom fail.**

DECODE SMOKE (temp=0): "The capital of France is Paris, notable for being a
global hub of art, fashion, and culture, renowned for iconic landmarks like the
Eiffel Tower and the Louvre." — coherent, `finish=stop`, zero U+FFFD, no
`<|begin_of_sentence|>` leak.

QUALITY GATE FIRST — `spec_degen_capture.py` 7 trigger prompts diff vs baseline:
**0/7 diffs.** All 7 prompts byte-identical between baseline (spec-off, no head)
and Stage 1 (spec-off, head loaded). Confirms the pre-reg prediction: with
`EXO_SPECULATIVE=0` the draft cycle is NOT constructed (`batch_generate.py:813`
gates `use_speculative` on `EXO_SPECULATIVE=1`), so loading the head via
`FORCE_LOAD=1` has zero decode effect. Head is resident but inert. Quality
unaffected.

STAGE-1 GATE: **PASS** (all 6 checks). Proceed to Stage 2 (full speculative).
Raw artifacts: `/tmp/ab/s1_degen.json`, `/tmp/dspark_s1.log`. Cluster left
RUNNING in `dspark_s1` (head resident, spec off) pending Stage-2 relaunch.

---

**NEW (2026-08-26, DSPARK/MTP STAGE 2 — FULL SPECULATIVE: QUALITY PASS,
THROUGHPUT FAIL → REVERT):** Stage-2 relaunched `dspark_s2` at exo HEAD
`5078e9018` / mlx-lm `643d42d`, flags = Stage 1 + `EXO_SPECULATIVE=1
EXO_SPECULATIVE_GAMMA=2 EXO_SPECULATIVE_TEMP=0.0 EXO_SPECULATIVE_ALPHA=1.0`
(FORCE_LOAD dropped — `SPECULATIVE=1` makes `_tp_consumer=True` so the
head-load gate passes naturally). Spec mechanism confirmed engaged on both
nodes: log shows `DSv4 MTP speculative decoding enabled (γ=2, T=0.0)`.
Memory 89 GB/node (same as Stage 1 — spec-on adds no resident weight, only the
draft cycle), 0 swap.

QUALITY GATE FIRST — **PASS** (the one gate that would force immediate revert
did NOT trigger):
- `spec_degen_capture.py` 7 trigger prompts (max-tokens 2000, temp=0): **all
  coherent.** Zero BOS-spam, zero U+FFFD, zero special-token leaks. Short
  prompts: "Paris", "One, two, three, four, five.", "Red, Yellow, and Blue".
  Long prompts produce coherent structured content (Roman Empire essay, TCP
  handshake with `##` headers, 20-language list outline). Note: at max-tokens=200
  the long prompts had empty `content` (all 200 tokens consumed by reasoning);
  at 2000 tokens they emit correct content. This is the DSv4 reasoning/content
  split, not degeneration.
- **`math_digit_sum` self-verification control (the batched-verify landmine
  test): PASS.** `hard_eval.py --tasks math_digit_sum --max-tokens 8000
  --temperature 0`: **answer = 115 (CORRECT), finish=stop, 0 leaks, 0
  truncations, reasoning_len=3383c, latency=40.4s.** On the PP cluster, the
  spec-correctness skill documented ALL THREE spec mechanisms (classic, MTP,
  DSpark) looping on this exact prompt (18-24x repetition, finish=length,
  never converged). On this TP cluster with DSpark native + MTP γ=2, the model
  **converges and answers correctly**. The PP batched-verify landmine does NOT
  reproduce on TP with this config — a genuine, positive quality finding.

THROUGHPUT GATE — **FAIL at both depths** (per `p3_depth_anchor_probe.py`):

| depth (prompt_tokens) | baseline (spec-off) | Stage 2 (spec-ON) | delta | gate | verdict |
|---|---|---|---|---|---|
| 100,025 | 28.46 tok/s (2000 tok, length) | **27.57** tok/s (760 tok, stop) | **−3.1%** | ≥+15% | **FAIL** |
| 352,601 | 25.07 tok/s (2000 tok, length) | **22.13** tok/s (340 tok, stop) | **−11.7%** | ≥+10% | **FAIL** |

- **TTFT**: 100K 269.0→266.6s (−0.9%, within +10% gate PASS); 352.6K
  1011.4→1026.8s (+1.5%, within +10% gate PASS). Prefill unaffected by spec.
- **Acceptance**: `usage.completion_tokens_details.accepted_prediction_tokens=0`
  and `rejected_prediction_tokens=0` on both probes. Either drafts have ~0%
  acceptance (fails the ≥60% gate) or the TP code path does not populate these
  OpenAI-usage counters (the pre-reg doc anticipated "if not logged, note
  that"). Either way, no measurable acceptance → no speedup mechanism.
- **Gap distribution**: bimodal — many 0.1ms "burst" gaps (accepted draft
  batches) interspersed with 100–400ms slow gaps (verify+reject overhead);
  37–39% of gaps are >3× median. This is the signature of speculative decode
  with poor draft acceptance: the verify overhead dominates the occasional
  accepted-batch savings.
- **Early-stop anomaly**: both Stage-2 probes show `finish_reason=stop` (760
  and 340 tokens) despite EOS being banned via `/bench/chat/completions`.
  Baseline ran the full 2000 tokens (`finish=length`). The tok/s is a rate
  (completion_tokens / decode_window), so the shorter window doesn't inflate
  it — but the premature stop is a behavioral change worth investigating (a
  possible spec-decode token-drop at an MTP cycle boundary, per the
  degeneration-sampler skill's known residual). It does not rescue the
  throughput result: the sign is negative at both depths and the regression
  worsens with depth (−3.1% → −11.7%), which is not plausibly a
  short-window artifact.

**DECISION: REVERT** (confirmed via second-opinion consult). The pre-reg doc's
"Throughput DECREASE vs baseline → immediate revert" criterion triggered at
both depths. Quality and memory gates passing do not override a throughput
regression — a config that is slower at both depths, uses more memory (+6-9
GB/node for the inert head), and has unmeasurable draft acceptance should not
be kept. The cluster was reverted to spec-off baseline flags
(`EXO_SPECULATIVE=0 EXO_DSV4_MTP=0 EXO_DSV4_HC_COLLAPSE_KERNEL=1`,
`dspark_revert` tmux session) — the production-safe config.

**Positive findings to bank** (do NOT re-litigate without new evidence):
1. The NATIVE DSpark head loads cleanly on both nodes (first-ever on-cluster
   native load) — `CHECKPOINT_PROVENANCE=MATCHED`, 115 tensors, 3 stages,
   block_size=5, taps=[40,41,42]. The `.scales` `param_tree_assert=FAIL` is a
   benign quantization-metadata warning, not a load failure.
2. The replicated head fits: +6-9 GB/node (under the ~10 GB estimate), 89 GB
   total, ~39 GB headroom, 0 swap. Memory is NOT the blocker for DSpark.
3. The PP batched-verify landmine (self-doubt loop on math_digit_sum) does
   NOT reproduce on TP with DSpark+MTP γ=2 — the model converges correctly.
   This is the single most important quality finding: the correctness fear
   that kept spec decode off production (per the spec-correctness skill) does
   not apply to this TP config.

**Why it's slow (leading hypothesis, not confirmed)**: draft acceptance is
near-zero. The DSpark native head's drafts (trained on this checkpoint's
hidden-state distribution) are being rejected by the TP verify forward. The
bimodal gap pattern (0.1ms bursts = accepted batches, 100-400ms = verify
overhead) and `accepted_prediction_tokens=0` both point here. The γ=2 setting
means at most 2 draft tokens/step — even if all accepted, the ceiling is
~2× sequential; with near-zero acceptance, the verify+reject overhead
(~per-step extra forward) makes it slower. A future trial would need (a)
proper accepted/rejected counters on the TP path to measure acceptance, (b)
a higher γ if acceptance is actually decent, and/or (c) investigation of
whether the native head's draft distribution actually matches the TP
verify path's expectations. These are separate follow-ups, not blockers for
this revert decision.

Raw artifacts: `/tmp/ab/s2_degen.json`, `/tmp/ab/s2_degen_2k.json`,
`/tmp/ab/s2_math_digit_sum.json`, `/tmp/ab/s2_quality_samples.txt`,
`/tmp/ab/s2_100k.json`, `/tmp/ab/s2_352k.json`, `/tmp/dspark_s2.log`.
Cluster reverted to `dspark_revert` (spec-off, production-safe).

---

## 2026-08-26 — DSpark/MTP Stage 2c (width-3 + EOS ban): QUALITY FAIL 2/7 → REVERT

**Stage 2c config:** Stage 2b flags (DSpark native + MTP, `EXO_SPECULATIVE=1
EXO_DSV4_MTP=1 EXO_DSV4_DSPARK=1 EXO_DSV4_DSPARK_NATIVE=1
EXO_DSV4_MTP_DEDICATED=0 EXO_DSV4_HC_COLLAPSE_KERNEL=1 EXO_SPECULATIVE_GAMMA=3`)
**plus** the EOS-ban fix (`commit ebe272fd7`, `EXO_DSV4_SPEC_EOS_BAN=1` default
ON). The EOS ban mirrors the non-spec baseline's `ban_token_ids(eos_ids)`
logits-processor by applying it to the spec verify logits BEFORE any argmax —
so neither accepted drafts NOR the bonus token can be EOS. Applied at three
verify sites in `dsv4_mtp.py` (lines 2532, 3927, 5184), gated by
`EXO_DSV4_SPEC_EOS_BAN` (default `"1"`). HEAD `8668e2616` (pre-reg commit +
EOS-ban fix). Launched `dsp_s2c` on both studios (screen `exorun`), API 200,
runners READY 2/2, model `deepseek-ai/DeepSeek-V4-Flash-0731` loaded, 89
GB/node, 0 swap.

### QUALITY GATE — FAIL 2/7 (vs 7/7 clean in Stage 2b)

`bench/spec_degen_capture.py` 7-prompt trigger set (max-tokens 256, temp=0)
against the spec-ON cluster. **2 of 7 prompts degenerated** — both hit the
built-in degeneration kill-switch (HTTP 500 "repetition-loop degeneration
detected"):

| label | Stage 2b (no ban) | Stage 2c (EOS ban) | delta |
|---|---|---|---|
| `sys_primary_colors` | clean ("Red, Yellow, Blue") | clean (finish=length) | — |
| **`sys_capital_france`** | **clean ("Paris", stop)** | **HTTP 500 repetition-loop** | **FAIL** |
| **`sys_count_to_five`** | **clean ("One, two, three, four, five.", stop)** | **HTTP 500 repetition-loop** | **FAIL** |
| `sys_long_essay` | clean (length) | clean (length) | — |
| `sys_long_steps` | clean (length) | clean (length) | — |
| `sys_long_list` | clean (length) | clean (length) | — |
| `control_user_only` | clean (length) | clean (length) | — |

**Stage 2b baseline (commit `d0f1db1e1`, DSpark native at γ=3, NO EOS ban):
7/7 clean** — `sys_capital_france`→"Paris", `sys_count_to_five`→"One, two,
three, four, five.", all 7 prompts coherent, `math_digit_sum` PASS (115). The
ONLY delta between 2b (clean) and 2c (2/7 fail) is the EOS-ban fix
(`ebe272fd7`). Ban is the sole regressor.

### Degeneration evidence — real repetition samples (from both nodes' exo logs)

The degeneration detector (`batch_generate.py:4381`) caught both failures
with exact per-cycle state. Identical timestamps + cycle_token_ids on BOTH
nodes (rank-consistent — the failure is cluster-wide, not node-specific):

**`sys_capital_france` (uid=3)** — degenerated at completion_token=61,
cycle period=3, repeated ≥6×:
```
DEGENERATION DETECTED uid=3 at completion_token=61: token cycle period=3
  repeated>=6x. action=error
  cycle_token_ids=[16, 128822, 51119]
  cycle_text='.<|end|>Paris'
  in_thinking=False | temp=0.0 | gen_engine=DSv4MTPBatchGenerator
```
Fired at MTP cycle ~125 (mean_accept=1.848/3, hist 0:20,1:30,2:24,3:51).

**`sys_count_to_five` (uid=4)** — degenerated at completion_token=96,
cycle period=11, repeated ≥6×:
```
DEGENERATION DETECTED uid=4 at completion_token=96: token cycle period=11
  repeated>=6x. action=error
  cycle_token_ids=[16, 128822, 6111, 14, 1234, 14, 2038, 14, 2689, 14, 3818]
  cycle_text='.<|end|>One, two, three, four, five'
  in_thinking=False | temp=0.0 | gen_engine=DSv4MTPBatchGenerator
```
Fired at MTP cycle ~154 (mean_accept=1.935/3, hist 0:22,1:33,2:32,3:67).

**Key observation:** token `128822` = `<|end|>` (think_end; confirmed from
the verify-audit code's own special-token set at `dsv4_mtp.py:4394`:
`_special = {128822, 128821}  # <|end|>, <|done|>`). The think_end special
token appears in BOTH degenerate cycles' committed streams — the model
generates `.<|end|>` then restarts the answer, looping. The EOS ban was
intended to prevent EOS from entering the committed stream; instead it
appears to have pushed the argmax to the next-best special token
(`<|end|>`/think_end, 128822), which enters the committed stream and the
model re-answers indefinitely.

### Verify-audit evidence gap — `EXO_DSV4_MTP_VERIFY_AUDIT` was NOT set

The built-in verify-audit (`dsv4_mtp.py:4388+`, env
`EXO_DSV4_MTP_VERIFY_AUDIT=<path>`) fires rank-0 JSONL whenever a special
token (think_end/eos, the `{128822, 128821}` set) appears in the draft,
accepted-target, or bonus at temp=0 — exactly the smoking-gun detector for
this failure class. It would have captured the per-cycle draft/target/bonus
token ids, the verify-logit top-2 margin at the bonus position, and the
pool-cache flush state at the degeneration onset.

**This env was NOT set during the s2c run.** Confirmed via
`ps eww <pid> | grep EXO_DSV4_MTP_VERIFY_AUDIT` on both nodes (absent from
the running process env), and it was not in the Stage 2c flag list. The
audit's per-cycle JSONL was therefore never written — the evidence gap is
documented here, not papered over.

**What the audit WOULD have captured** (per the code at `dsv4_mtp.py:4394+`):
for each cycle where `128822`/`128821` appeared in the draft, target_argmax,
or bonus, a JSONL record with `cycle`, `uid`, `gamma`, `n_accepted`,
`draft` (token id list), `target_argmax`, `all_next`, `bonus`, `bonus_pos`,
`bonus_argmax`, `bonus_top2_logits`, `bonus_margin`, `bonus_special`,
`draft_special`, and `pools` (per-pool offset/remainder/snapshot state).
This would have pinned whether `<|end|>` was being DRAFTED (DSpark head
emitting it), ACCEPTED (verify argmax choosing it), or chosen as the BONUS
(raw argmax on the banned-EOS logits), and whether it won by a healthy
margin (corrupted context) or a hair (numerical/ban side-effect).

**What we DO have** (independent corroboration): the degeneration detector's
own `cycle_token_ids` field confirms `128822` (`<|end|>`) is in the committed
output stream at both failing prompts. Combined with the fact that Stage 2b
(no ban) produced clean output on the same prompts, the ban is implicated as
the cause — but the per-cycle draft-vs-accept-vs-bonus attribution that the
audit would have provided is absent.

### The EOS-ban-fix-causes-degeneration finding

The EOS-ban fix (commit `ebe272fd7`) was intended to kill the Stage 2b
early-stop anomaly (completion 1148/2000, finish=stop) by preventing EOS
from entering the committed stream via the spec verify path. The
hypothesis: the non-spec baseline is immune because `_step` applies
`ban_token_ids(eos_ids)` as a logits-processor before sampling, but the
spec verify path (`dsv4_speculative_forward`) calls the model RAW with no
logits-processors — so the raw-argmax bonus and per-row accepted-draft
argmax could both be EOS.

**The fix worked for its stated goal** (no early-stop observed in 2c), but
**introduced a new regression**: banning EOS from the spec verify logits
changes the argmax distribution. The baseline applies the ban to the
*sampling* path (the model samples after the ban); the spec path applies
the ban to *raw argmax* — these are not equivalent. Banning EOS from raw
argmax can push the argmax to the next-best token, which for these two
prompts is `<|end|>` (think_end, 128822), another special token that
triggers the model to re-emit its answer in a loop. The ban is applied at
three verify sites (`dsv4_mtp.py:2532`, `:3927`, `:5184`) — all gated by
`EXO_DSV4_SPEC_EOS_BAN` (default `"1"`).

**Mechanism (suspect, not fully confirmed without the audit data):** the
spec path's raw argmax + ban changes the argmax distribution in a way the
non-spec path's sample-after-ban does not. The non-spec path samples from
the banned distribution (EOS mass redistributed proportionally across all
non-EOS tokens via softmax); the spec path takes argmax on the banned
logits (EOS mass simply removed, argmax jumps to whatever was second —
often another special token like `<|end|>`). This is a structurally
different operation, and for these 2/7 prompts it lands on a token that
triggers a repetition loop.

### REVERT executed

The pre-registered bars mandated revert on quality FAIL. Executed
2026-08-26 09:38 CDT:

1. **SIGTERM** (graceful, never `kill -9`) the `dsp_s2c` exo processes on
   both studios. `pkill -TERM -f 'python.*exo'` on each node — both exited
   cleanly within 1s (jaccl RDMA teardown ran via static destructors, TB
   link stayed healthy: 0% packet loss, sub-ms latency, 2 active ports
   each). Quit the `exorun` screen sessions. Verified NO exo processes
   remained on either node (pgrep clean, lsof 52415 clear) before relaunch.
2. **Relaunch** spec-off production config via tmux pattern
   `tmux new-session -d -s dspark_revert2 'cd ~/repos/exo && EXO_SPECULATIVE=0
   EXO_DSV4_MTP=0 EXO_DSV4_DSPARK=0 EXO_DSV4_HC_COLLAPSE_KERNEL=1
   ./start_cluster.sh 2>&1 | tee /tmp/dspark_revert2.log'`.
3. **READY (2/2)** reached at poll 34 (~340s). HEALTHY (Nodes: 2,
   Identities: 2), commit `8668e2616` synchronized on both nodes.
4. **ENV AUDIT** (both nodes, `ps eww`, rank-consistent):
   `EXO_SPECULATIVE=0 EXO_DSV4_MTP=0 EXO_DSV4_DSPARK=0
   EXO_DSV4_HC_COLLAPSE_KERNEL=1` — spec fully OFF, prefill opt retained.
   `EXO_DSV4_MTP_VERIFY_AUDIT` and `EXO_DSV4_SPEC_EOS_BAN` absent (correct
   for spec-off).
5. **API live** on both nodes (port 52415), 125 models including
   `deepseek-ai/DeepSeek-V4-Flash-0731`.

### Spec-off quality confirmation (proves ban+spec interaction, not the prompts)

Re-ran `bench/spec_degen_capture.py` (7 prompts, max-tokens 256, temp=0)
against the spec-off cluster: **7/7 clean.** The 2 previously-failing
prompts are now clean:

- `sys_capital_france` → "Paris" (finish=stop, no leak) — was HTTP 500
  repetition-loop under s2c
- `sys_count_to_five` → "One, two, three, four, five." (finish=stop, no
  leak) — was HTTP 500 repetition-loop under s2c
- All other 5 prompts: coherent, no leak, no error

This confirms the degeneration requires the spec path + the EOS-ban
interaction — the prompts themselves are benign under greedy/sequential
decode. The ban is the sole regressor vs the 7/7-clean Stage 2b baseline.

### Next steps — candidate alternatives for the next campaign

The ban mechanism itself is suspect: the baseline applies the ban because
the non-spec path *samples* (EOS mass redistributed via softmax); the spec
path's *raw argmax* + ban changes the argmax distribution differently, and
for 2/7 prompts it lands on a loop-triggering special token. Candidate
alternatives to explore (not yet tested):

1. **Skip-EOS-at-commit-boundary (intercept, don't ban):** instead of
   banning EOS from the verify logits (which distorts the argmax), let the
   draft/verify accept EOS normally but intercept it at the committed-stream
   boundary — when EOS would be committed, *skip* the EOS token and continue
   decoding (as if the model "changed its mind"). This preserves the
   argmax distribution while still preventing premature finish. Closest to
   how a sampler with `ban_token_ids` behaves, but applied post-acceptance
   rather than pre-argmax.
2. **Bonus-only ban (don't ban the draft-acceptance rows):** apply the EOS
   ban ONLY to the bonus-token argmax (the raw `all_next`), NOT to the
   per-row accepted-draft argmax (`target_tokens`). The early-stop anomaly
   was specifically the bonus token being EOS; banning it there may suffice
   without distorting the draft-acceptance distribution that the 2 failing
   prompts depend on. The three ban sites (`dsv4_mtp.py:2532`, `:3927`,
   `:5184`) could be selectively gated.
3. **Load-aware scheduler (compare against the paper):** the DSpark paper's
   mechanism may handle EOS differently — a load-aware scheduler that
   decides when to stop drafting vs. when to commit EOS may avoid both the
   early-stop anomaly AND the repetition degeneration. Worth a literature
   check before re-attempting a ban-based fix.

**Before any re-attempt:** set `EXO_DSV4_MTP_VERIFY_AUDIT=<path>` in the
launch env so the per-cycle draft/accept/bonus state is captured to JSONL.
The audit was built exactly for this failure class and its absence this run
left the mechanism attribution incomplete. Also consider extending the
audit's `_special` set beyond `{128822, 128821}` if other special tokens
(like `<|done|>` variants) become loop triggers.

### Raw artifacts

`/tmp/dspark_s2c.log`, `/tmp/dspark_s2c_specdegen.log`,
`/tmp/dspark_s2c_specdegen.json`, `/tmp/dspark_revert2.log`,
`/tmp/dspark_revert2_specdegen.json`, `/tmp/dspark_revert2_specdegen.log`,
node exo logs (`~/.exo/exo_log/exo.log` on both studios — degeneration lines
at `2026-08-26 09:32:13.941` and `09:32:18.485`).

**Cluster final state: production spec-off (`dspark_revert2` tmux session,
both studios, API live, env audited).** Campaign closes here per the failed
quality gate.


## 2026-08-26 — DSpark/MTP Stage 3 (width-3 + EOS_BAN default-off fix): PRIMARY FAIL → REVERT

**Auditor:** GLM-5.2 (Ollama Cloud), acting as PM.
**Repo HEAD:** `1c26dad08` (pre-reg commit) — cluster running Stage 3 config,
NO relaunch. Fix under test: `d8c671501` (default `EXO_DSV4_SPEC_EOS_BAN` 1→0,
opt-in now). Stage 3 = Stage 2b flags + the ban default-OFF fix.

**Stage 3 config (verified live via env audit on both nodes):**
`EXO_SPECULATIVE=1 EXO_SPECULATIVE_GAMMA=3 EXO_DSV4_MTP=1 EXO_DSV4_DSPARK=1
EXO_DSV4_DSPARK_NATIVE=1 EXO_DSV4_DSPARK_FORCE_LOAD=1 EXO_DSV4_MTP_DEDICATED=0
EXO_DSV4_MTP_LOG_INTERVAL=1 EXO_DSV4_HC_COLLAPSE_KERNEL=1`. `EXO_DSV4_SPEC_EOS_BAN`
**ABSENT** from the runner env on both nodes (ban OFF — the fix). DSpark head
attached NATIVE checkpoint-bundled (115 tensors, 3 stages, block_size=5,
taps=[40,41,42]) on both nodes. HEAD `1c26dad08` on both nodes.

### Verdict: REVERT — PRIMARY FAIL at BOTH probe depths

Per the pre-registered Stage 3 bars (docs/dspark-mtp-ab-preregister-2026-08-25.md):
PRIMARY (hard gate) = "Where baseline runs to 2000/length, spec must run to
2000/length" and "If completion < 2000 or finish=stop at 100K/352.6K probes →
hard fail, revert immediately." Stage 3 tripped this at BOTH depths.

### Depth-probe results (p3_depth_anchor_probe, /bench/chat/completions, temp=0)

| metric | baseline 100K | stage3 100K | baseline 352K | stage3 352K |
|---|---|---|---|---|
| prompt_tokens | 100,025 | 100,022 | 352,601 | 352,646 |
| **completion_tokens** | **2000** | **369** | **2000** | **462** |
| **finish_reason** | **length** | **stop** | **length** | **stop** |
| TTFT (s) | 269.0 | 265.7 | 1011.4 | 1026.8 |
| tok/s (usage) | 28.46 | 29.54 | 25.07 | 24.30 |
| tok/s (events) | 28.17 | 29.22 | 24.91 | 24.14 |
| streamed events | 1980 | 365 | 1987 | 459 |
| decode window (s) | 70.24 | 12.46 | 79.73 | 18.97 |

**Deltas vs baseline:**
- 100K: completion 369 vs 2000 (−1631) → **PRIMARY FAIL** (early-stop anomaly).
  tok/s +3.8% (but measured over a truncated 12.46s decode window — not a
  clean 2000-token comparison).
- 352K: completion 462 vs 2000 (−1538) → **PRIMARY FAIL**. tok/s 24.30 vs 25.07
  (−3.1%) → **hard rollback criterion tripped** (throughput DECREASE vs baseline).

Both probes hit the early-stop anomaly (finish=stop, completion < 2000). The
log line itself flagged it: `finish_reason: stop  (must be 'length' -- EOS banned)`.
The text tails are NOT clean natural ends — 100K tail ends mid-thought
("...no narrative or analytical content." — a period, but the summary is
truncated relative to what the model continues to produce across runs); 352K
tail ends mid-word in the first run ("...underlying content i") and mid-sentence
in the repeat.

### Committed-stream divergence (losslessness failure — deeper than finish alone)

The spec-ON output diverges from the spec-OFF baseline **from token 1** —
common-prefix length 0 chars (baseline "1.  The user has provided..." vs
stage3 "The user wants a brief summary..."). This is not "same text, different
stop point"; the committed stream is a different generation. Per the
`exo-speculative-decode-correctness` skill, spec-ON producing different output
than spec-OFF on the identical prompt at temp=0 is a correctness divergence,
not merely a finish-behavior difference.

### Determinism check — NONDETERMINISTIC at temp=0 (root-cause clue)

A repeat of the 100K probe (same config, same prompt, same temp=0) produced
**completion=504** vs the first run's **369** — a 37% difference in stop
point, with the same text opening ("The user wants a brief summary...") but
different length. Greedy/temp=0 decode should be deterministic; the spec path
is not. This run-to-run variance also explains the Stage 2b (1148) vs Stage 3
(369) discrepancy: it is NOT the ban code's presence changing no-ban behavior
(the ban gate `env == "1"` is a true no-op when unset — verified at
`dsv4_mtp.py:2532,3940,5235`), it is nondeterminism in the spec verify path
itself (likely floating-point / RDMA-ordering / async-timing sensitivity in
the draft+verify cycle at 100K+ depth).

### Quality gate — 7/7 spec_degen PASS (natural-end parity on SHORT prompts)

`bench/spec_degen_capture.py` 7-prompt trigger set (max-tokens 256, temp=0)
on Stage 3 config. Compared vs baseline (`/tmp/ab/baseline_degen.json`):

| label | baseline finish | stage3 finish | base content | stage3 content | parity |
|---|---|---|---|---|---|
| sys_primary_colors | length | length | — | — | PASS |
| **sys_capital_france** | **stop** | **stop** | **Paris** | **Paris** | **PASS** |
| **sys_count_to_five** | **stop** | **stop** | **One, two, three, four, five.** | **One, two, three, four, five.** | **PASS** |
| sys_long_essay | length | length | — | — | PASS |
| sys_long_steps | length | length | — | — | PASS |
| sys_long_list | length | length | — | — | PASS |
| control_user_only | length | length | — | — | PASS |

**7/7 PARITY (finish + content + no leak): PASS. 0 U+FFFD across all 7.**
The natural-end equality holds on short prompts: `sys_capital_france`→"Paris"
finish=stop (vs Stage 2c's ban-ON "Paris.Paris.Paris..." period-3 loop — the
ban-OFF fix restored natural-end here). This is the losslessness signal the
fix was designed to deliver — but it does NOT hold at the deep probes.

### math_digit_sum control — PASS (3/3)

`bench/hard_eval.py --tasks math_digit_sum --max-tokens 8000 --temperature 0`:
3/3 trials pass=1.00, finish=stop, leak=False, snippet="115", 0 leaks, 0
truncations. No loop.

### Acceptance (from [MTP] LOG_INTERVAL logs) — 71% (exceeds 50% bar)

`grep "[MTP]" exo.log` final aggregate: cycles=2266, mean_accept=2.134/3 =
**71.1%**, hist 0:277, 1:407, 2:318, 3:1264. Good draft acceptance — but
irrelevant to the finish/stop PRIMARY gate.

### Memory — PASS (95/93 GB, 0 swap)

`footprint` on the real `python -m exo` worker PID (the spawn_main child, not
the screen wrapper — see `exo-cluster-debugging` skill pitfall):
- m4-2 (master): 95 GB footprint, sysctl `vm.swapusage` = 0.00M used, 0
  swapins/swapouts.
- m4-1 (worker): 93 GB footprint, `vm.swapusage` = 0.88M/1024M used (56
  historical swapouts, 0 swapins — negligible background, not runner paging).
Both under the 126 GB bar. Memory gate PASS (but irrelevant to the finish
gate).

### Generated-text samples

**Short prompt (natural stop, PARITY with baseline) — sys_capital_france:**
> finish=stop, content="Paris"

**Short prompt (natural stop, PARITY) — sys_count_to_five:**
> finish=stop, content="One, two, three, four, five."

**100K probe (early-stop anomaly) — text head:**
> "The user wants a brief summary of the corpus. The corpus is a long list of
> sections (0 to 2262) that follow a repetitive pattern. Each section describes
> a practice (e.g., \"distributed inference schedulers allocate pipeline stages
> across nodes\"..."
>
> **completion=369, finish=stop** (baseline 2000/length)

**352K probe (early-stop anomaly) — text tail:**
> "...The configuration and stage numbers vary across entries, but the
> underlying content is essentially identical, with no narrative, argument,
> or additional information."
>
> **completion=462, finish=stop** (baseline 2000/length)

### Audit excerpt (ban gate, verified OFF)

```
dsv4_mtp.py:2532  if (self._spec_eos_ids and os.environ.get("EXO_DSV4_SPEC_EOS_BAN", "0") == "1"):
dsv4_mtp.py:3940      if (self._spec_eos_ids and os.environ.get("EXO_DSV4_SPEC_EOS_BAN", "0") == "1"):
dsv4_mtp.py:5235      if (self._spec_eos_ids and os.environ.get("EXO_DSV4_SPEC_EOS_BAN", "0") == "1"):
```
Env audit on both nodes: `EXO_DSV4_SPEC_EOS_BAN` ABSENT (ban OFF, correct for
Stage 3). The gate is a true no-op when unset — verified the ban code's
presence does not change no-ban behavior; the 369-vs-1148 (Stage 2b)
discrepancy is nondeterminism, not a code-path regression.

### Protocol-calibration note (separate from the verdict)

The `/bench/chat/completions` endpoint bans EOS in the committed stream
(`ban_token_ids(eos_ids)` at `batch_generate.py:2658`). The spec-OFF baseline
ran to 2000/length at both depths **because it was forced to** — it literally
cannot emit EOS. Stage 3's spec verify path (ban OFF) CAN emit EOS, so it
stops where the model's true greedy argmax is EOS. The 369/stop and 462/stop
may be the model's TRUE natural greedy end — which the baseline was
artificially prevented from reaching. The discriminator would be: does the
spec stream 1..N match an UNBANNED greedy stream? We cannot get an unbanned
greedy stream from the `/bench` endpoint (it always bans). This is the
next-protocol gap: a probe that runs spec-OFF WITHOUT the `/bench` EOS ban
(e.g. `/v1/chat/completions`, no bench flag) at 100K to capture the model's
true natural-stop point, then compare. Per the pre-registered bars, this
calibration issue does NOT excuse the fail — the bars were written against
the `/bench` baseline behavior and Stage 3 diverged. Noting it as the path
to a fairer Stage 4 bar.

### Decision: REVERT to spec-off

Per the pre-registered rollback criterion ("ANY divergence from baseline on
finish/stop behavior OR throughput DECREASE vs baseline → REVERT"), Stage 3
FAILS: early-stop anomaly at both depths (369/stop, 462/stop vs 2000/length)
PLUS throughput −3.1% at 352K. The 7/7 short-prompt parity and 71% acceptance
do not override the deep-probe finish-gate failure. **Cluster does NOT stay
on Stage 3.** The natural-end theory held on short prompts but broke at deep
context — the spec path's EOS emission at the post-acceptance bonus position
is depth-correlated and nondeterministic.

### Raw artifacts

`/tmp/ab/s3_degen.json`, `/tmp/ab/s3_degen.log`,
`/tmp/ab/s3_math_digit_sum.json`, `/tmp/ab/s3_math.log`,
`/tmp/ab/s3_100k.json`, `/tmp/ab/s3_100k.log`,
`/tmp/ab/s3_352k.json`, `/tmp/ab/s3_352k.log`,
`/tmp/ab/s3_100k_repeat.json` (determinism check),
node exo logs (`~/exo.log` on both studios — `[MTP]` acceptance lines
through `2026-08-26 13:05+`).

**Cluster final state: Stage 3 config still RUNNING (screen `exorun` on both
studios, API live) — pending REVERT to spec-off per this verdict.** The PM
session does not relaunch (SIGTERM-only rule + approval gate); the user must
run the `dspark_revert` relaunch.

## 2026-08-26 — DSpark/MTP corrected verdict protocol (24-run measurement): REVERT

**Step 4 (measurement) of the corrected spec-decode verdict protocol.**
Follows step 1 (C_s profile, `docs/dspark-cs-profile-2026-08-26.md`) and
step 3 (Tier-1 byte-identity, `docs/dspark-tier1-byte-identity-2026-08-26.md`).
Full writeup: `docs/dspark-verdict-measurement-2026-08-26.md`.

Ran 12 spec-ON + 12 spec-OFF runs of `bench/golden_v1_probe.py` at 100K
context (`--target-tokens 100000 --max-tokens 2000`, temp=0 greedy), one
probe at a time, ~60s cooldown. Metric: **256-token fixed-window decode
tok/s** — amortizes away prefill+startup so it measures pure decode.

**Result (REVERT):**

| metric | value | bar | pass? |
|---|---|---|---|
| median % delta (on−off) | +1.87% | ≥ +10% | FAIL |
| 95% bootstrap CI (10K resamples) | [−0.82%, +9.45%] | lower ≥ +5% | FAIL |
| CI includes 0? | YES | no | FAIL (REVERT trigger) |
| Tier-1 byte-identical | 2/3 | all 7 | FAIL |
| Gate A (acceptance = strict argmax) | clean | clean | PASS |

Per-arm fixed-window tok/s: **ON median 28.30 (IQR 27.44–29.90, range
21.6–32.3), OFF median 27.49 (IQR 27.19–27.61, range 27.0–28.2).** The ON
arm has ~3.7× wider spread — the spec-decode cycle's bimodality (long
verified chains vs early-reject cycles). Run #02-ON (64 tokens,
`finish_reason: null`) is the EOS-bypass anomaly (the spec verify path
applies no logits processors, so the raw-argmax bonus token can be EOS —
same family as the Stage-2c early-stop). Excluding it, median % delta is
+2.14%, still ≪ +10%.

**Tier 2 (natural-EOS):** ON 10 stop / 1 length / 1 null-anomaly, median
length 374; OFF 12/12 stop, median 535. ON shorter by 161 tokens median
(consistent with the EOS-emission tendency). rep16-gram fraction ON lower
(0.019 vs 0.069) but length-confounded (shorter → fewer 16-gram windows).
1 loop flag each (task-structural, maxrep=3 on a repetitive-corpus summary
task — not degeneration). scipy unavailable → descriptives only.

**Decision: REVERT.** All three independent inputs agree: (1) throughput
+1.87% ≪ +10%, CI straddles 0; (2) Tier-1 2/3 (MoE-rowseq 0.023%/row residual
flips a near-tie, so spec-ON is NOT bit-identical); (3) C_s arithmetic says
break-even is the ceiling at C_s=3.20 / a≈2.26 — +10% is impossible without
verify-path batching (the real fix direction, per the C_s doc). The
arithmetic prediction from step 1 held: a clean fixed-window measurement
lands near break-even, not +10%.

**Cluster final state: production spec-off** (screen `exorun_specoff` both
nodes, `EXO_SPECULATIVE=0 EXO_DSV4_MTP=0 EXO_DSV4_HC_COLLAPSE_KERNEL=1`,
DSpark head loaded but not drafting — the `dspark_prod`/`dspark_revert`
pattern). Env verified via `ps eww` on both nodes. The measurement phase
left the cluster in the production state; no further relaunch needed.

Pre-reg doc updated: `docs/dspark-mtp-ab-preregister-2026-08-25.md`
(corrected-protocol section). The corrected fixed-window protocol's bars
(median ≥ +10%, lower CI ≥ +5%) supersede the Stage-2/3 bars for the
final verdict.

---

## 2026-08-27 — Verify-path batching (EXO_DSV4_VERIFY_BATCH) G0 FAIL → REVERT

**Campaign:** verify-path batching — the proposed fix to bring C_s from
3.20 (rowseq break-even) down to ~1.3, targeting +55-80% throughput.
Implementation: **indexer-stream-sharing** (Phase 0 design,
`docs/verify-batch-phase0-2026-08-26.md`) — snapshot the compressed-KV
stream once per cycle, reuse for rows 1..L-1. Gated behind
`EXO_DSV4_VERIFY_BATCH=1` (default OFF, submodule `93afab7`).

**Implementation SHAs:** super `a1ba8c27e18bed8d26a430325357851b5cf29492`,
submodule `93afab74a27f40ec747663833407b215de653366` (both pushed, clean).

**G0 (cycle-level bitwise, VERIFY_BATCH=1 vs 0, expect 0-ulp): FAIL.**
VERIFY_BATCH=1 crashes deterministically on the first verify cycle:
`ValueError: [broadcast_shapes] Shapes (1,1,3) and (1,1,2) cannot be
broadcast.` in `Indexer.__call__` (`deepseek_v4.py:4008`), call chain
`_speculative_next` (`dsv4_mtp.py:3893`) → `dsv4_speculative_forward`
(`:1420`) → `_forward_steps` (`:6856` activates
`_set_verify_batch_ctx`) → `Indexer` (`:4008` `mx.where(pmask, scores)`).
4 crashes per node (8 total) on the first warmup (`"Say hi"`). The
runner supervisor restarts the worker each time; the parent survives.

**Root cause:** the Phase 0 design doc and the in-code comment
(`deepseek_v4.py:3870-3880`) assume "the verify-time pmask is None
(PoolingCache.make_mask returns None for L<=_POOL_VERIFY_MAX_L)". This
is **violated**: `_dispatch_pmask` returns a non-None, 3D pmask sized
`(1,1,L_full=3)`. The `_tail_ok` fast-path (`:3960`) requires
`pmask.ndim == 2`, so it falls to the `else` branch (`:4007`) which
broadcasts pmask `(1,1,3)` against per-row scores `(1,1,2)` → last-axis
3 vs 2 mismatch → crash. The stream-sharing reuses row 0's `pooled`
snapshot correctly, but the companion pmask handling was written for the
None-pmask case only.

**VERIFY_BATCH=0 (rowseq baseline) CLEAN:** identical env minus the 3
verify-batch flags — 0 errors, 0 crashes, 256-token probe coherent
(`completion_tokens=256, finish_reason=length`). Confirms the bug is
isolated to the `EXO_DSV4_VERIFY_BATCH=1` path.

**REAL C_s:** NOT OBTAINABLE — the verify-batch path crashes before
`[MTP-PROF]` phase-timer lines emit (crash is inside the verify forward,
before the dump). The rowseq-baseline G0-OFF run was launched without
`MTP_PROFILE`, so no `[MTP-PROF]` lines there either. Documented
rowseq C_s=3.20 (`docs/dspark-cs-profile-2026-08-26.md`) stands: at
C_s=3.20 the +10% bar is unreachable (break-even a*=2.199 vs a≈2.256).

**G2 / G3 / sanity A/B / 24-run: CANCELLED (moot).** The verify-batch
path crashes before any downstream gate can execute — there is no
functioning VERIFY_BATCH=1 to A/B test.

**Verdict: REVERT.** `EXO_DSV4_VERIFY_BATCH` stays default OFF. The fix
direction is in the Indexer pmask handling (`deepseek_v4.py:3894-4010`):
slice pmask to the per-row band when `_VERIFY_BATCH_CTX["active"]` and
`pmask is not None`, OR extend the `_tail_ok` fast-path to the 3D-pmask
case the stream-sharing produces. Scoped to the mlx-lm submodule fork;
rowseq baseline untouched.

**Cluster final state: production spec-off** (screen `exorun_specoff`
both nodes, `EXO_SPECULATIVE=0 EXO_DSV4_MTP=0 EXO_DSV4_DSPARK=1
EXO_DSV4_DSPARK_NATIVE=1 EXO_DSV4_DSPARK_FORCE_LOAD=1`, DSpark head
resident not drafting). Env verified via `ps eww` both nodes.
`EXO_DSV4_VERIFY_BATCH` absent (default OFF). Warmup `"Say hello"` →
`"Hello!"` clean.

**Docs:** `docs/verify-batch-g0-fail-2026-08-27.md` (full root-cause +
call chain + next-step direction). Phase 0 design doc
(`docs/verify-batch-phase0-2026-08-26.md`) unchanged — its design
decision (indexer-stream-sharing) stands; the implementation has a pmask
bug that blocks G0.

**Sync/launch method:** `start_cluster.sh` blocked (expired sudo both
nodes). Manual per-node screen pattern: SIGTERM old runners → per-node
`git fetch + reset --hard` (super + submodule) → `zsh -l -c 'uv pip
install --no-deps --force-reinstall ./mlx-lm'` (copy install) →
`grep -c EXO_DSV4_VERIFY_BATCH` = 8 verify → scp launch file →
`screen -dmS exorun_specoff bash -c 'bash /tmp/specoff_launch.sh'`.

---

## 2026-08-27 — DSPARK MTP PROMOTED: Depth-gated batched verify (24-run verdict, +36.7%)

### The result that ships
| metric | batched-ON | spec-OFF | delta |
|---|---:|---:|---:|
| median fixed-window tok/s @100K | **36.63** | 27.15 | **+36.71%** |
| 95% bootstrap CI (10K resamples) | | | **[+28.26%, +51.02%]** |
| pairs where ON > OFF | | | **12/12** |
| weakest / strongest pair | | | +14.17% / +56.17% |

Pre-registered PROMOTE bar: median >= +50% of wall-model prediction at C_s=2.14 (+13%) AND lower CI > 0. Cleared on both counts.

### Full 24-run table (fixed-window 256-tok tok/s, 100K ctx, temp=0, golden_v1_probe)
| pair | ON | OFF | delta% |
|---|---:|---:|---:|
| 0 | 41.93 | 27.12 | +54.60 |
| 1 | 35.75 | 26.44 | +35.19 |
| 2 | 40.66 | 27.35 | +48.65 |
| 3 | 36.94 | 26.72 | +38.23 |
| 4 | 38.41 | 27.25 | +40.99 |
| 5 | 42.41 | 27.15 | +56.17 |
| 6 | 30.86 | 27.03 | +14.17 |
| 7 | 36.31 | 27.24 | +33.27 |
| 8 | 34.69 | 27.15 | +27.77 |
| 9 | 35.77 | 27.78 | +28.76 |
| 10 | 41.82 | 27.27 | +53.39 |
| 11 | 32.08 | 25.27 | +26.96 |

### The mechanism (why this time it's real)
- The corrected depth-gated batched verify (EXO_DSV4_VERIFY_BATCH=1, MIN_CTX=8192) reintroduces the pre-rowseq batched M=4 forward (submodule dda9237, parent 6eba31ff1).
- Verify: 83.76 -> 60.60ms mean (MTP-PROF n=1550), C_s 3.20 -> 2.14, acceptance 2.250 (parity with rowseq 2.118 — the snapshot hack that killed acceptance -19% is REMOVED).
- G0'' gate PASSED: batched-vs-rowseq drift 74.7% <= base-vs-base run-to-run drift 99.3% at 100K — the batched path adds less noise than the base already has.
- G2/G3: Tier-1 short-ctx byte-identity rides the rowseq path below the 8192 gate (untouched); soak clean.
- 12/12 paired wins is a distribution, not a cherry-pick: this protocol pairs time-adjacent runs on the same prompt, cancelling the base's run-to-run drift (the 295-vs-977-token wobble).

### Operational notes
- Cold-start: first batched verify cycle can trip the jaccl GPU-event fence under memory pressure if a second model is resident; exo's load warmup covers kernel JIT, but avoid concurrent-resident-model placement during the first request. Warm kernels = clean (proven across 12 runs).
- Cluster now runs the promoted config: EXO_SPECULATIVE=1 EXO_DSV4_MTP=1 EXO_DSV4_DSPARK=1 EXO_DSV4_VERIFY_BATCH=1 EXO_DSV4_VERIFY_BATCH_MIN_CTX=8192 gamma=3 temp=0 alpha=1.0 HC collapse+expand=1.
- 352.6K-depth measurement still pending (the +36.7% is @100K); bar for deep: >= +15% median (regression risk of the batched path at max depth is the open question).
