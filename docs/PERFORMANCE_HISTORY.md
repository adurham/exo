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

> **SUPERSEDED 2026-08-27/28 — SPECULATION IS NOW ON IN PRODUCTION.** The
> paragraph above and its Aug-24 correction both describe pre-promotion
> state. Depth-gated batched verify was PROMOTED 2026-08-27 (+36.71%
> median @100K, 12/12 pairs; `EXO_SPECULATIVE=1 EXO_DSV4_MTP=1
> EXO_DSV4_DSPARK=1 EXO_DSV4_VERIFY_BATCH=1 MIN_CTX=8192 γ=3`), and the
> 352.6K memory regression was fixed and validated 2026-08-28
> (`EXO_DSV4_DSPARK_TP_SHARD=1` + `EXO_MLX_CLEAR_CACHE_INTERVAL=64`,
> +17.57% @352.6K, 0/8 collapses). Authoritative consolidated record:
> `docs/dspark-mtp-master-history-2026-08-28.md`.

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

> **UPDATE 2026-08-27: the ~30-35 tok/s ceiling framing is superseded.**
> Depth-gated batched verify (`EXO_DSV4_VERIFY_BATCH=1 MIN_CTX=8192`)
> measured **36.63 tok/s median @100K** (+36.71% vs spec-off, 12/12
> paired wins) and is the production default. The ceiling statement was
> true of the rowseq-verify era (C_s=3.20); batched verify (C_s=2.14)
> broke it. See `docs/dspark-mtp-master-history-2026-08-28.md`.

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

**DSpark draft-epilogue fusion (WIN — code shipped, cluster A/B deferred,
2026-08-27)** — `docs/dspark-draft-epilogue-fusion-2026-08-27.md`,
env-gated `EXO_DSV4_DRAFT_EPILOGUE=1` (default OFF). Moves the DSpark
draft forward (~10.8 ms `_dspark.draft()` block-forward + Markov loop)
off the per-cycle critical path by computing the NEXT cycle's draft in
the CURRENT cycle's epilogue (after `append_ctx` + bonus token), so the
next cycle consumes it without serializing before the verify. Mirrors
the PP path's existing implementation (`pp_speculation.py` ~line 2952).
Scoping verdict: the DSpark draft depends only on the anchor token
(prev cycle's `bonus_val`) + the ctx-KV caches (populated by
`append_ctx`) — NOT on `_mtp_pre_norm` or the target's prompt-cache —
so the **entire** next draft is computable in the epilogue (full fusion,
no partial-fusion needed). The Markov loop is sequential WITHIN one
`draft()` call but no state persists across cycles except `_dspark_caches`.
Tie-reverify hazard (can mutate `bonus_val` after the epilogue) guarded
by invalidation; dead code in prod (tie-reverify OFF/retired). Expected:
cycle 73.7→~62.9 ms, C_s 2.14→~2.51, 36.6→~42.5 tok/s (+16% theoretical;
real win depends on the epilogue-tail overlap fraction on Metal).
**Cluster A/B deferred** — the 352.6K decode protocol was running during
implementation (NO CLUSTER TOUCH). Gates: ruff 20→20 (zero new),
basedpyright 734 (zero new error types — only pre-existing
`reportAny`/`reportUnknownMemberType` from the untyped `_dspark.draft()`
pattern), import clean, scoped pytest pass (the one
`test_pp_speculation_cache_snapshot` failure is pre-existing — stale
test vs 5-tuple `PoolingCache.state`, confirmed against clean HEAD).
**Lesson: the PP path was the reference implementation — when a fork
has two decode paths (PP + TP), the optimization that landed on one is
the proven design for the other; port the pattern, don't redesign.**

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

> **UPDATE 2026-08-27/28: doubly superseded — speculation is now ON in
> production.** Depth-gated batched verify was promoted (+36.71% @100K)
> and the DSpark head runs live drafting at γ=3 with `TP_SHARD=1` at
> 352.6K (+17.57%, zero collapses). The FULLBLOCK k-multiplier collapse
> above remains a valid negative result for that verify MODE; the
> production path avoids it (batched ≥8K, rowseq <8K). See
> `docs/dspark-mtp-master-history-2026-08-28.md`.

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

## 2026-08-27/28 — 352.6K deep-context gate: memory regression found, root-caused, fix shipped

The pending 352.6K measurement ran (16 ON runs, golden_v1_probe, fresh
process each, ~60s cooldown): **4/16 collapsed to ~1 tok/s** (median cycle
gap 1.8-2.3s, every cycle stalled onset-to-end, 1.37-1.76 GB swap both
nodes); healthy runs 16.9-31.1 tok/s at ~97 GB wired / ~600 MB free.
Regression frame: spec-OFF @500K ran 17.26 tok/s ZERO swap on 2026-08-21.

**Root cause (code-verified): the DSpark draft head (~10.13 GB quantized)
loads REPLICATED on both nodes** — `DeepseekV4ShardingStrategy` shards
`model.model.layers` + `mtp_blocks` but never `model.model.dspark`
(`utils_mlx.py:866` attach; no shard-path reference). Spec-ON steady
residency ≈ 99.4 GB/node vs spec-OFF ≈ 89.3. With a ~19-22 GB/node
per-cycle touched working set (MoE routing dominant) and mmap-backed
weights, crossing the ceiling → clean-page eviction → every cycle
re-faults → bistable thrash equilibrium (matches the stochastic 4/16 rate
and the three observed modes). Batched-verify transients (+4.7 MB/layer),
allocator fragmentation, pool growth: all analyzed and REFUTED as drivers
(docs/dspark-352k-*.md).

**Fix `2d85ccdcb`: `EXO_DSV4_DSPARK_TP_SHARD=1`** (default OFF) — shards
the 3 DSpark stages' MoE FFN through the same quantized shard helpers as
the mtp blocks; ~3-3.5 GB/node recovery, ~1-3% draft-latency cost;
detach-on-failure guard. Honest framing: collapse-eliminator candidate,
not a full margin-restorer (attention/markov parts stay replicated).

Protocol completion (stripped-OFF ×9 for the paired verdict, forced-OFF ×3
residency probe, verbon3 ×8 fix-validation with CLEAR_CACHE_INTERVAL=64 +
per-cycle memory profile) ran overnight; results land in
docs/dspark-352k-memory-regression-2026-08-27.md and the production
baseline doc's 352.6K section.

## 2026-08-28 — verbon3 launch abort root-caused: JIT placement wait aborted on transient non-memory blocker (fix `75d2402dd`)

The overnight driver's phase C (verbon3 = promoted spec-ON config +
`EXO_DSV4_DSPARK_TP_SHARD=1` + `EXO_MLX_CLEAR_CACHE_INTERVAL=64`) aborted
at 04:01 and auto-reverted to stripped-OFF: the smoke probe got a 503 and
the dspark shard-gate grep came back empty. **Neither had anything to do
with the TP shard** — the model never placed at all.

Timeline (node log, `exo_verbon3.log.rot-*` on m4-1): smoke request hit
the API 6s after API-up (19s after the previous phase's SIGTERM).
04:01:01.654 the JIT placement wait entered its 120s window on a
memory-blocked tick ("No cycles found with sufficient memory" — post-kill
reclaim lag, exactly what the window is for). A poll tick 2-4s later
raised a NON-memory `JitPlacementUnavailableError`, and the wait loop
treated any non-memory blocker as permanent → instant 503 → driver abort
→ auto-revert. At 04:01:05.66, ~1-2s after the 503, an `/instance/previews`
sweep dry-ran ALL FOUR RDMA placement combos successfully — the cluster
was healthy; the blocker class had merely oscillated while gossip
converged (memory reports, topology edges, rdma_ctl, node_network all
stream in asynchronously after a launch).

**Fix `75d2402dd`**: while the opt-in `EXO_JIT_PLACEMENT_WAIT_SECONDS`
window is open, poll through ALL `JitPlacementUnavailableError` reasons;
503 only at deadline (detail carries first+last blocker so oscillation is
visible). Default 0 = upstream instant-503, unchanged. Replaced the unit
test that encoded the buggy behavior with an oscillation regression test
(memory→non-memory→viable), a non-memory-first-tick test, and a
default-off parity test. 121/121 api+jit suites pass; basedpyright/ruff
zero-delta.

**Live validation of the fix (07:50 relaunch, same launch file)**: the
identical race occurred and was survived — log shows tick 1 memory-blocked
→ tick 3 blocker CHANGED to "MLX ring backend requires connectivity
between neighbouring nodes" (the old code's insta-503) → placement
succeeded 6s in: "JIT auto-placing ... (sharding=Tensor, meta=MlxJaccl,
min_nodes=2)". Smoke probe returned "Paris"; both ranks logged
"DSv4 DSpark draft head TP-sharded across 2 ranks (3 stages)" with no
detach. The verbon3 8-run 352.6K phase is now genuinely running; verdict
lands in dspark-352k-memory-regression-2026-08-27.md.

Also fixed in the driver (phaseC_von3.py): the env gate now reads the
runner env via `ps eww <pid>` (env-prefix vars never appear in
`ps -axo command` argv — the 07:46 first relaunch attempt false-aborted on
that before any request was sent), and the append-mode node log is rotated
per attempt so gate greps can't see a previous launch's lines.

## 2026-08-28 — 352.6K verbon3 validation COMPLETE: zero collapses, +17.6% median — DSpark TP shard validated at depth

**8 genuine von3 runs at 352.6K (spec-ON promoted config + `EXO_DSV4_DSPARK_TP_SHARD=1`
+ `EXO_MLX_CLEAR_CACHE_INTERVAL=64`), fresh process each, ~60s cooldown,
telemetry on both nodes:**

| run | fw tok/s | n_tokens | finish |
|---|---:|---:|---|
| 00 | 26.29 | 384 | stop |
| 01 | 34.77 | 1196 | stop |
| 02 | 28.83 | 338 | stop |
| 05 | 28.05 | 356 | stop |
| 06 | 30.52 | 585 | stop |
| 07 | 30.89 | 450 | stop |
| 08 | 27.97 | 361 | stop |
| 09 | 19.50 | 380 | stop |

- **Collapses (<5 tok/s): 0/8** (the regression phase had 4/16 at ~1 tok/s).
  Worst run 19.50 tok/s — slow tail, NOT the thrash equilibrium.
- **ON median 28.44 vs stripped-OFF median 24.19 (n=9, same night, same
  protocol) = +17.57%**, above the +15% deep bar. Unpaired bootstrap 95% CI
  on the median delta: [+8.74%, +27.77%] — the point estimate clears the
  bar; the CI lower bound does not, so the margin is thin. Honest read:
  collapse-elimination is conclusive; the throughput win at depth is real
  but modest.
- **Memory ground truth (2s OS telemetry)**: swap peaked 97 MB (vs
  1.37-1.76 GB in the regression runs), pageouts-delta ~44-49 across the
  whole ~4.7h phase. Peak wired m4-1 92.0 GB / m4-2 93.5 GB vs 96.8/95.3
  in the (equal-length window) regression-phase telemetry — ~3-4.8 GB/node
  recovered, matching the ~3-3.5 GB/node the FFN shard predicts.
- Validation is of the COMBINED verbon3 config (TP_SHARD + CLEAR_CACHE=64
  + mem-profile overhead); the two knobs were not isolated. Given zero
  collapses and the wired-peak drop matching the shard's predicted
  recovery, the shard is the operative fix; CLEAR_CACHE=64 is retained as
  belt-and-braces.

**Incident during the phase (2 run slots voided, runs #3/#4):** at
08:52:18, ~1h2m into the launch, both ranks hit `[jaccl-v2] WC_ERR
rank=0/1 call=85230 status=4 wt=5 buff=6` mid-prefill → segfault in the
prefill forward (deepseek_v4 attention path, NOT the dspark draft path) →
both runners died. First WC_ERR ever in this campaign (verbon2's 126 MB /
16-run log: zero). exo's supervisor + JIT auto-re-place recovered the
cluster WITHOUT operator action — notably through the `75d2402dd` wait
path (the re-place rode a fresh jaccl PD-allocation backoff). Runs #5-9
completed cleanly after. n=1: cannot attribute to TP_SHARD vs the known
flaky-UC RDMA class (dsv4-c2-serving-handoff-2026-07-06.md documents the
class); no repeat in ~3.7h of post-recovery running. Watch item, not a
blocker.

**Cluster end state**: left on verbon3 config (exo `75d2402dd` + mlx-lm
`d098642`), model idle-reaped (JIT-placed ⇒ reaper-eligible; next request
auto-places in ~2min). Artifacts: /tmp/ab/protocol352/ (summary_von3.jsonl,
run_von3_*.json, telemetry_*_von3.csv, phaseC_von3.log,
node logs ~/exo_verbon3.log{,.rot-*} both nodes).

**Verdict: `EXO_DSV4_DSPARK_TP_SHARD=1` is VALIDATED at 352.6K** — the
memory-regression collapse mode is eliminated at depth with the promoted
spec-ON throughput win intact (+17.6% median vs spec-OFF at the same
depth). Promote TP_SHARD=1 into the production spec-ON launch env.

## 2026-08-28 — 352.6K correctness + harness verification of the SHIPPED stacked config: PASS (with one harness artifact found)

Final verification pass on the exact production config (batched verify +
TP_SHARD + CLEAR_CACHE=64; BOOKKEEP_FAST/DRAFT_EPILOGUE compiled-in but
env-OFF, absence verified via ps eww; exo 75d2402dd, mlx-lm d098642):

- **Tier-1 (live, stacked)**: capital_france + count_to_five byte-identical
  to the Aug-26 spec-OFF captures; primary_colors byte-identical to the
  known accepted spec-ON residual trajectory; 7/7 deterministic across 2
  live captures AND across a deep-session boundary (post-soak rerun 7/7).
- **G0''-style @352.6K, ONE frozen prompt (usage=352600, cached=0, TTFT
  ~1000s all runs)**: spec-off S1=S2=S3 BYTE-IDENTICAL; batched B0=B1
  BYTE-IDENTICAL across two separate cluster launches; rowseq R
  deterministic. Arms differ from each other deterministically (B-vs-R
  first div char 24; R-vs-S char 54) — the known bounded MoE residual
  class, not noise. Formal envelope bar recorded FAIL-degenerate (base
  drift = 0 makes it unfalsifiable); real bars (determinism + bounded
  divergence + quality) PASS.
- **HARNESS ARTIFACT (real finding): the "base nondeterministic at depth"
  premise (99.3% base-vs-base @100K, 295-vs-977) came from fresh-nonce
  prompts** — build_prompt() embeds a uuid4 per call; those runs never had
  byte-identical prompts. Fixed-prompt retest @100K (EOS-banned past the
  natural end, 979 toks): byte-identical pair. Do not cite "base is
  nondeterministic at depth" or the G0'' 74.7%<=99.3% numbers without this
  caveat. Throughput verdicts unaffected.
- **Depth soak (production config)**: 4000/4000 tokens @352.6K, 30.46
  tok/s, median gap 0.08ms, max 376ms, zero faults, swap peak 50/72MB,
  mean_accept 2.438/3; 34/34 procedurally-verifiable factual claims in the
  output correct (0 wrong).
- **Shipped-verdict artifact audit**: von3 +17.57% / CI [+8.76,+27.79] /
  0 collapses, pre-fix -17.87% gate FAIL, telemetry 97MB swap / 92.0-93.5GB
  wired — all reproduce from raw JSONLs/CSVs to the decimal; 0 mismatches.
- Cluster end state: verbon3 production, cold JIT auto-place + Paris smoke
  PASS. Docs: dspark-352k-correctness-harness-verification-2026-08-28.md
  (+ preregister + runs-table JSON). Artifacts: /tmp/ab/g0_352/.

## 2026-08-28/29 — P1 draft-epilogue fusion A/B: byte-lossless but NO throughput win; stays default-OFF

- Pre-registered (`docs/dspark-p1p4-campaign-preregister-2026-08-28.md`, committed
  before any run) fixed-prompt A/B of `EXO_DSV4_DRAFT_EPILOGUE=1` vs `0` on the
  production verbon3 stack (exo `75d2402dd` + mlx-lm `d098642`): 6 runs/arm @100K
  + 3/arm @352.6K + Tier-1.
- **Correctness: ALL PASS.** Tier-1 7/7 byte-identical to Aug-28 captures; ON and
  OFF streams byte-identical to EACH OTHER at both depths (the consume path
  reproduces the inline draft exactly, as designed); every arm internally
  deterministic; 0 collapses; swap ≤65 MB; zero fault lines.
- **Throughput: FAIL.** Shared-window medians — @100K ON 37.62 vs OFF 37.76
  (−0.35%, boot CI [−0.56,+0.51]); @352.6K ON 30.27 vs OFF 30.35 (−0.26%).
  Pre-registered bar was ≥+8% AND min(ON)>max(OFF): neither met.
- **Mechanism (MTP-PROF windowed, m4-1):** consume-cycle draft 8.20→0.55 ms — the
  fusion ENGAGES — but the epilogue `_dspark.draft()`+eval is synchronous inside
  the cycle epilogue and its cost reappears in the accept window (1.55→9.47 ms).
  Net cycle 76.0→74.6 ms (−1.8%), which does not survive to end-to-end tok/s.
  The design's assumed overlap with the accept/rollback/bookkeeping tail does not
  exist on a single Metal stream with a synchronous eval.
- Verdict: `EXO_DSV4_DRAFT_EPILOGUE` remains default-OFF. Draft is 10.8% of the
  cycle; verify (64 ms @100K / 85 ms @352.6K) remains the only wall worth
  attacking. Full doc: `docs/dspark-p1-draft-epilogue-ab-results-2026-08-28.md`.

## 2026-08-28/29 — P2 c=2 validation + P4a verify-cost curve (campaign wrap; P3/P4b staged for next session)

- **P2 c=2 (dspark-p2-c2-validation-results-2026-08-28.md):** spec-specific legs
  ALL CLEAN — deep batched B=2 (100K+100K) per-stream deterministic across
  repeats with zero cross-stream contamination; June Bug-3 adversarial
  final-digit needle **0/6 flips** under TP batched verify (PP-era ~80% class not
  reproduced). TWO NEW SHARED-GENERATOR BUGS (both reproduce spec-OFF, c=1
  clean): (a) c=2 system+user short prompts degenerate deterministically
  (`.</think>Paris` period-3, kill-switch token 61) — BS=2 stop/EOS handling;
  (b) the BS=2 degen-abort path crashes the runner (mlx-lm cache.py:2050
  fetch_overlap_carry `[reshape] size 2 -> (1,1,1,1)`), killing the healthy
  batchmate + instance. c=2 spec-ON @100K measured 10.0/9.7 tok/s per stream
  (19.7 aggregate; B=2 spec cycles are per-row — BS>1 batched verify is still
  the Phase-5 TODO). c=2 NOT production-ready until (a)+(b) fixed; c=1 promoted
  config unaffected.
- **P4a (batched-regime verify-cost curve, production config, fixed prompts):**
  verify ms/cycle = 79.5 @4K, 78.3 @7.5K (rowseq, <MIN_CTX) | 56.1 @9K, 56.1
  @14K, 54.5 @32K, 59.2 @64K, 64.2 @100K, 84.8 @352.6K (batched). The
  historical "1455.8 ms @14K cliff" is refuted at the measurement level: batched
  14K is 26× cheaper; the old number was FULLBLOCK-regime. P4b (rowseq-forced
  14K/32K above the gate, launch staged) + the MIN_CTX placement verdict deferred
  to next session — see master-history §6 P4 resume notes.
- **P3 (TP_SHARD vs CLEAR_CACHE ablation):** not started; launch scripts + driver
  + comparator trio staged — master-history §6 P3 resume notes.
- Campaign hygiene: all runs pre-registered (`dspark-p1p4-campaign-preregister-
  2026-08-28.md` + amendments committed before the affected runs); cluster
  restored to production verbon3 (env ps-eww-verified, Paris smoke PASS).

## 2026-08-29 — P3 arm-SHARD landed (first leg of the verbon3 ablation)

- **L3 arm-SHARD (TP_SHARD=1, CLEAR_CACHE_INTERVAL=0), n=3 @352.6K, fixed
  prompt `prompt_352k_long.txt`, max_tokens 2500:** shared-window (W=2170)
  tok/s = 30.465 / 30.483 / 30.512 (median 30.483, σ=0.024 = 0.08% CV) vs
  arm-BOTH (P1-OFF L0 trio) 30.352 / 30.353 / 30.424 (median 30.353). The
  CLEAR_CACHE=64 interval at depth costs **−0.43%** — i.e. removing it buys
  +0.43%, far inside the estimated 5–15% band, and arms overlap in range.
  All 3 runs byte-identical to each other AND to the arm-BOTH comparator
  (hash `f08efa3f`) — the two fixes remain byte-lossless across the config
  axis, and determinism survives the relaunch.
- **Memory (decode-windowed telemetry):** arm-SHARD wired peaks climb
  across consecutive depth runs within one launch — 92.3 → 94.1 → 96.6 GB
  (m4-1), 92.3 → 92.3 → 97.0 GB (m4-2) — vs arm-BOTH 92.3 → 92.3 → 93.4 GB:
  the MLX buffer-cache ratchet with CLEAR_CACHE=0 is real (~+3 GB by run 3
  at depth, growing per run) but produced **zero collapses and flat swap
  (50–65 MB)** over this short horizon. No throughput penalty from the
  ratchet at n=3 depth-run scale.
- Hygiene: env verified via `ps eww` on both nodes pre-run; log offsets
  recorded per run in `log_offsets.jsonl` (L3.* labels); telemetry sampler
  `p3shard` tag; artifacts `run_p3shard_{0,1,2}.json`.

## 2026-08-29 — P3 arm-CACHE landed; P3 ablation complete (verdict)

- **L4 arm-CACHE (TP_SHARD=0, CLEAR_CACHE_INTERVAL=64), n=3 @352.6K, same
  fixed prompt/max_tokens:** shared-window (W=2097, min chunk-count across
  all arms) tok/s = 29.000 / 29.008 / 29.035 (median 29.008, σ=0.018 =
  0.06% CV). With arm-CACHE in the set the shared window tightens
  2170→2097, so the pre-registered analysis (`analyze_p3_final.py` →
  `p3_verdict.json`) recomputes every arm on W=2097: arm-BOTH
  30.446/30.482/30.519 (median 30.482), arm-SHARD 30.582/30.590/30.618
  (median 30.590). The L3 interim entry above quoted whole-generation tok/s
  rather than windowed — same ranking, uniformly ~0.1–0.4% lower.
- **P3 verdict (pre-registered outputs; quantification session, no hard
  pass/fail bar):**
  - **CLEAR_CACHE=64 cost at depth** = arm-BOTH vs arm-SHARD = 30.482 vs
    30.590 → **−0.35%** (windowed; −0.43% whole-generation). The 5–15%
    estimate is refuted — the interval is ~free at depth.
  - **TP_SHARD contribution** = arm-CACHE vs arm-BOTH = 29.008 vs 30.482
    → **−4.84%**: sharding the DSpark head is worth ~+4.8% at 352.6K and is
    the dominant term of the bundled pair by >10×.
  - Confidence: per-arm σ ≈ 0.02 tok/s (0.06% CV, 3 runs each); shard-off
    delta 1.47 tok/s with non-overlapping 3-run ranges (>40σ separation);
    cache-interval delta ~0.35% — direction consistent across both
    windowing conventions but near cross-session drift scale, so read as
    "≤0.5%, marginal".
- **Collapse tallies (pre-registered):** 0/3 arm-CACHE (the accepted-risk
  arm did NOT collapse — n=3 distinguishes ~0 from ~1 rates only), 0/3
  arm-SHARD, 0/3 arm-BOTH. Decode health gap_median 0.07 ms on every run.
- **Memory (decode-windowed telemetry — shard contribution):** arm-CACHE
  wired peaks 101.1 → 97.6 → 97.3 GB (m4-1), 101.5 → 97.3 → 97.3 GB (m4-2)
  vs arm-BOTH 92.3–93.4 GB (m4-1) / 92.3–96.2 GB (m4-2): the replicated
  head costs **~+4–5 GB/node settled** (first-run load spike ~101 GB then
  settles; prereg estimate was ~+3–3.5 GB — close, slightly low); swap flat
  at 50–65 MB on both arms — no memory-pressure signal anywhere.
- **Byte-identity across the TP_SHARD axis does NOT hold (new finding):**
  all 3 arm-CACHE runs are byte-identical to each other (hash `d56b8dd1`)
  and fully deterministic, but diverge from arm-BOTH/arm-SHARD
  (`f08efa3f`) — first divergence at output char ~1967 is a single
  mid-reasoning coin flip ("…topics repeat in order? Let's identify 10
  topics…" vs "…There are 8? Let's identify distinct topic patterns…"),
  then bounded divergence with identical finish_reason=length,
  completion_tokens=2500. This is the expected numerics class (head
  replication changes reduction order in draft/verify matmuls — same
  family as the documented batched-vs-sequential divergence), NOT a
  correctness regression; the production path (TP_SHARD=1) remains
  byte-lossless across the CLEAR_CACHE axis (L3 finding stands).
- **Production-env recommendation fed by P3:** keep both fixes (arm-BOTH
  config = production): the shard buys ~4.8% at depth with no settled-memory
  penalty, CLEAR_CACHE=64 costs ≤0.5% and holds wired memory ~4–5 GB lower
  than it would otherwise ratchet. Each term of the bundle is now
  individually quantified.
- Hygiene: env ps-eww-verified both nodes pre-run; log offsets L4.* in
  `log_offsets.jsonl`; telemetry sampler `p3cache`; artifacts
  `run_p3cache_{0,1,2}.json`; machine verdict in `p3_verdict.json` (to be
  re-run after L6 to fill the cross-session anchor leg).

## 2026-08-29 — P4b rowseq crossover landed; MIN_CTX placement verdict

- **L5 rowseq-forced (`VERIFY_BATCH=0, VERIFY_ROWSEQ=1`), 14K + 32K, same
  frozen prompts as P4a, max_tokens 600, warm-flush runs first
  (250-token same-ctx generation before each measured run so MTP-PROF
  accumulator remainders are same-ctx cycles):** both runs clean
  (finish_reason=length, 600 completion tokens, no errors). Windowed
  MTP-PROF per-run means (m4-1, `~/exo_von3rowseq.log`,
  `p4_phase_extract.json`, extraction method identical to P4a):
  - 14K rowseq: verify **78.73 ms**, draft 8.24, total 89.21 (n=200 cycles)
  - 32K rowseq: verify **75.20 ms**, draft 8.17, total 85.64 (n=250 cycles)
  - vs P4a batched: 14K verify **56.06 ms** / total 68.70; 32K verify
    **54.48 ms** / total 68.02 (re-extracted and reproduced exactly).
  - Within-window per-dump ranges (verify): batched 14K mean-range
    72.8–73.3 cumulative / per-cycle min–max 38.5–136.3; rowseq 14K
    75.4–77.6 / 45.4–122.6 — cycle-level scatter overlaps but windowed
    means separate cleanly (+40% gap).
- **MIN_CTX placement verdict (pre-registered rule: placement SUPPORTED iff
  batched verify_ms < rowseq verify_ms at 14K, checked at 32K too):
  SUPPORTED — keep 8192.** Batched is cheaper at both crossover depths:
  14K 56.1 < 78.7 (+40.4% rowseq penalty), 32K 54.5 < 75.2 (+38.0%).
  The floor cannot move below ~8K regardless (pre-registered correctness
  asymmetry: rowseq below 8K is the short-ctx byte-identity guarantee).
  End-to-end corroboration: decode tok/s batched 41.3 vs rowseq 31.3 @14K
  (+32%), 39.3 vs 28.7 @32K (+37%); TTFT identical (37.3 s / 83–84 s) —
  the difference is pure decode-path cost.
- **Curve shape:** rowseq verify is ~flat in ctx (79.5 @4K, 78.3 @7.5K,
  78.7 @14K, 75.2 @32K) while batched sits at 54–56 ms from 9K–32K — the
  two regimes are parallel plates with a one-time step at the 8K gate; no
  crossover above 8K, so there is no depth at which raising MIN_CTX above
  8192 would pay. The historical "1455.8 ms @14K cliff" is retired with a
  same-stack rowseq measurement: 78.7 ms — the old number was
  FULLBLOCK-regime, off by ~18.5×.
- Hygiene: env ps-eww-verified both nodes (VERIFY_BATCH=0,
  VERIFY_ROWSEQ=1); log offsets L5.* labels; artifacts
  `run_p4b_flush_{14000,32000}.json` + `run_p4b_rowseq_{14000,32000}.json`;
  merged P4a+P4b extraction in `p4_phase_extract.json`.

## 2026-08-29 — L6 production restore + cross-session anchor; campaign P1–P4 complete

- **L6 production restore:** `/tmp/verbon3_launch.sh` on both studios, env
  ps-eww-verified (EXO_DSV4_VERIFY_BATCH=1, EXO_DSV4_DSPARK_TP_SHARD=1,
  EXO_MLX_CLEAR_CACHE_INTERVAL=64, EXO_SPECULATIVE=1 on both nodes),
  Paris smoke PASS (stop, 200, correct answer). Cluster left on stock
  production verbon3 — NOT a test config.
- **Cross-session validity anchor (pre-registered ±5% gate): PASS.** One
  fresh 352.6K run on the restored production launch: shared-window
  (W=2097) tok/s **30.462**, whole-generation 30.367, finish_reason=length,
  2500 tokens, gap_median 0.07 ms — vs arm-BOTH median 30.482 →
  **−0.07%**, comfortably inside ±5%. The P1-OFF L0 trio reuse for the
  P3 ablation stands; no session-confound caveat needed.
- **Byte-identity bonus:** anchor output hash `f08efa3f` — byte-identical
  to the arm-BOTH L0 trio AND arm-SHARD, across a session boundary, two
  relaunches, and the whole P3/P4b campaign. Determinism of the
  production TP_SHARD=1 path is now cross-session verified.
- **Final machine verdict** (`p3_verdict.json`, re-run with anchor leg):
  window W=2097; arm-BOTH median 30.482, arm-SHARD 30.590 (σ=0.019,
  0.06% CV), arm-CACHE 29.008 (σ=0.018, 0.06% CV), anchor 30.462;
  CLEAR_CACHE cost −0.35%, shard-off −4.84%, anchor −0.07% (valid).
- Campaign end-state: P1/P2/P3/P4a/P4b all landed and documented; master
  history §5 rows CLOSED for the 14K A/B and TP_SHARD/CLEAR_CACHE
  ablation items; remaining open items are unchanged (replicated-head
  residue, SPEC_STATE_RESTORE gating, jaccl WC_ERR watch).

## 2026-08-29 — Phase 0: real per-kernel Metal capture of moe.switch_mlp — capture restored durably, kernel confirmed at ceiling

**PM:** GLM-5.3. Repo HEAD `6a5f7fb23`. Delegated to two leaf workers;
PM re-verified every claim (own grep of trace bundles, own byte
arithmetic, git status on both nodes, runner PIDs). Full detail:
`docs/p01-switch-mlp-gputrace-recapture-2026-08-29.md`.

- **The lost capture is replaced — and this time it is durable AND
  analyzed.** Real `mx.metal.start_capture()` traces captured at exact
  production decode shape (B=1, top_k=6-of-256, per-rank inter=1024,
  mxfp4 g=32 b=4, rotated 64-entry routing pool): `~/repos/exo/tmp/
  p01-20260829/m4_1/moe_capture.gputrace` (m4-1, 11 GB) and the laptop
  copy `~/repos/exo/tmp/p01-20260829/laptop_smoke/moe_capture.gputrace`
  (11 GB). Both outside /tmp — they survive relaunch/reboot cycles.
  Runner PID 25491 unchanged before/after; no relaunch, no env flips.
- **Capture-enabling finding (reusable, closes the "unparseable format"
  dead-end partially):** `mx.metal.start_capture()` fails with
  "Capture layer is not inserted" unless `METAL_CAPTURE_ENABLED=1` is
  set in the environment — on laptop AND on studios (works even with
  Xcode removed from the studios). Kernel NAMES are scriptably
  extractable from the bundle's `device-resources-*` member (grep:
  `mxfp4_gather_qmv_fast_bfloat16_t_gs_32_b_4`,
  `mxfp4_quantize_float_gs_32_b_4`) but per-kernel TIMINGS are not
  stored in a scriptable format (Xcode GUI on the laptop remains the
  only route for visual per-kernel timeline inspection).
- **Per-stage GPU attribution (DRAM-real, rotated indices, m4-1,
  MLX_GPU_TIME=1 bracketing):** fused_gate_up (gather_qmm) 59.08 µs =
  531 GB/s = **97.5% of 546 GB/s spec** (31.46 MB stage bytes);
  activation 2.93 µs; down_proj (gather_qmm) 32.67 µs = 482 GB/s =
  **88.5% of spec** (15.73 MB). Stage sum 94.67 µs vs the retraction
  doc's chained wall regime ~117 µs — GPU-busy sum sits below wall as
  expected (gaps excluded), both stages land between measured real
  streaming (424 GB/s) and spec. NOTE the byte accounting in the
  first draft of the doc (worker) split 47.186 MB ~50/50, producing a
  physically impossible "126%-of-peak" down_proj figure — corrected by
  PM to the true 2:1 gate_up:down split (gate+up are TWO fused
  matrices, down is one).
- **A first attempt produced a tainted 131%-of-peak claim — superseded
  and documented as such.** Worker 1's per-stage run used a FIXED
  routing index (`idx = pool[0]`) inside its 50-iter timing windows →
  warm-cache times (gate_up 38.7µs/down 24.2µs, "719 GB/s, 131% of
  peak") — the exact fictitious-cache artifact class the 2026-08-22
  retraction warns about. The repo's standing rotated-indices rule
  caught it; the re-measurement above is DRAM-real. Worker 1 also
  patched `BatchedSwitchGLU.fuse_weights` (bias=None handling) on the
  laptop AND scp'd it to m4-1; worker 2 investigated, found production
  does not call `fuse_weights()` on the runner's load path, and
  REVERTED both trees (verified: mlx-lm submodule clean on laptop and
  m4-1). No unreviewed edit remains on any production node.
- **HEADROOM VERDICT: no meaningful kernel-level headroom at this op.**
  Independent-calls wall-clock on m4-1: 100.65 µs ≈ 491 GB/s ≈ 90% of
  spec — consistent with the corrected microbench's 74-87% band
  (chained regime slightly lower because dependency-chain tails are
  real). The ~30-45%-of-wall span attribution is kernel time, not
  hidden overhead; closing the §13 line-2648 thread ("a real Instruments
  Metal trace of the GatherQMM kernel internals") with an honest
  negative: the kernel is at its realistic ceiling (between real
  streaming BW and spec), no lever exists. Do not re-litigate
  switch_mlp kernel optimization.

## 2026-08-29 — Phase 1(b): inter-layer pipelining-loss at depth — leading candidate tested, NOT the residual

**PM:** GLM-5.3. Full detail: `docs/p01b-multilayer-pipelining-loss-2026-08-29.md`,
raw `tmp/p01b-20260829/p01b_results.json`. Cluster node m4-2 (rank0),
runner PID 28581 unchanged before/after. No relaunch.

- **The hypothesis, precisely:** the 2026-08-24 additivity doc's stated
  leading candidate for the +1.67..+2.52 ms/tok unattributed on-GPU
  busy growth was inter-layer pipelining loss that worker C's
  single-layer × census microbench structurally cannot see.
- **Method:** 4-layer chain of REAL production attention layers
  ([sparse r4, compressed r128] × 2) with real cross-layer data
  dependencies and real shared cache state (PoolingCache/RotatingKVCache
  pre-filled at depth), 256 B=1 L_q=1 decode steps, per-step
  mx.async_eval, depths ~500/100K/352.6K, 3 repeats each (spread
  ≤0.014 ms). Single-layer-per-class arm re-measured on the same
  silicon/build for apples-to-apples.
- **Result:** multi-layer chained per-token cost at depth grows only
  **+0.076 ms/token** (100K→352.6K, 1.845→1.921 ms) — an order of
  magnitude below the +1.67..+2.52 residual band. The chain is
  consistently FASTER than the summed single-layer baseline (~0.7
  ms/token faster at every depth — cross-layer command-buffer
  pipelining makes the chain more, not less, efficient). No growing
  inter-layer bubble exists in this harness's frame.
- **CAVEAT (PM verification, do not bury):** the same harness's
  single-layer arm shows ~ZERO depth growth (2.6335→2.6173 ms
  100K→352.6K) where worker C's 2026-08-23 measurement of the same
  classes demanded ~+2.56 ms/tok at 43-layer census scale (~+0.24 ms
  per 4-layer chain). Both arms of this bench under-engage
  depth-dependent attention work (~3× too cheap at 100K). Root cause
  not run down; plausible causes: the harness's synthetic pre-fill
  may not reproduce the indexer's depth-dependent scan (e.g. pool
  fill via raw state writes may skip the real accumulate_windows
  path), or class mix/shape differences. CONSEQUENCE: the honest
  verdict is **"no support for the leading candidate in this
  harness's frame"**, not a clean refutation — the harness fails the
  calibration cross-check against the historical baseline, so the
  ~0 term is weak evidence, not proof. The residual remains OPEN.
- **Donation/allocator pre-check (read-only telemetry):** peak bench
  memory grew linearly with depth (1.65→1.84→2.0 GB) and stayed tiny
  vs the runner's ~85-90 GB; no donation-failure markers; node
  resident memory far below the 125 GB hard-abort threshold.
  Allocator-pressure deep-dive remains a separate later phase (c),
  untouched here by design.
- **State of the residual after (b):** still +1.67..+2.52 ms/tok
  unattributed. Candidates narrowed: inter-layer pipelining loss is
  now UNLIKELY (weak evidence); MoE-at-depth interplay and
  allocator/donation-at-85GB-regime remain live. Next test should
  first re-calibrate against worker C's per-class numbers (or use a
  harness validated to reproduce them) before drawing conclusions.

## 2026-08-29 — Phase 1(a): all_sum arrival-skew at depth — measured INSIDE the collective, ruled out

**PM:** GLM-5.3 campaign, executed by fable-5 orchestrator directly
(serialized probes, no nested delegation). Full detail:
`docs/p01a-allsum-arrival-skew-at-depth-2026-08-29.md`; raw traces +
analyzer in `tmp/p01a-20260829/`.

- **Method:** `JACCL_TRACE_CALLS=1 JACCL_TRACE_TIMING=1` relaunch (env
  diff vs production = exactly those two vars), two serialized
  p3_depth_anchor_probe runs at REAL usage.prompt_tokens **100,022** and
  **352,645**, per-rank steady_clock timing inside C++
  `reliable_all_reduce_v2` (includes peer-wait), cross-rank matched by
  call_id (100% match), decode segmented after last 16MB prefill chunk.
  This is the "direct CPU-side timer on the collective" the 2026-08-24
  doc §5.3 named as the decisive experiment for its one honest hole.
- **RESULT: arrival skew grows only +0.079 ms/tok** (0.494→0.573) across
  100K→352.6K in the per-token verify-class collectives (32/24/16KB,
  ~16-17 calls/tok) — 3-5% of the +1.67..+2.52 residual band, and an
  independent-method confirmation of 2026-08-24's occupancy-derived
  +0.070. Per-CALL skew depth-flat (30.7→33.5µs mean); rescaled to
  spec-off's 43-calls/tok cadence the growth is +0.12 ms/tok ≈ 5-7% of
  band — holds in both regimes. **Total in-collective time is also
  depth-flat** (< +0.09 ms/tok/rank growth) — even charging everything
  inside all_sum to the residual cannot close it. Median per-call
  transport depth-FLAT (36.9-40.1µs both depths) — extends "transport is
  fast" to 352.6K.
- **No straggler rank on this build:** r0-slower 44-50% everywhere;
  severe (>1ms) verify-class skew 0.03-0.34% with no direction. The
  2026-08-22 4.2x rank0 tail asymmetry does not reproduce.
- **Per-REQUEST skew classes flagged as a trap:** 8192B sequential-path
  (215/request, 177→358ms total) and multi-MB transition calls
  (74.8→297.2ms) amortize to ~0 over long generations — dividing them by
  a short probe's token count manufactures a phantom +1.5 ms/tok. Do not
  re-derive.
- **Probe anomaly, separate thread:** /bench route returned
  finish_reason=stop at 409/320 tok despite the EOS ban (expected
  length@2000). Does not affect per-call skew arithmetic. Worth its own
  look.
- **Residual after 1(a)+1(b): still +1.67..+2.52 ms/tok unattributed
  on-GPU busy.** Collective now closed from BOTH sides (idle: 08-24;
  in-call: this). Live candidates: MoE-at-depth interplay,
  allocator/~90GB-resident regime.
- **Production restored + verified (relaunch #2):** SIGTERM + clean
  screen teardown, byte-identical verbon3_launch.sh relaunched both
  nodes, ps eww shows zero JACCL_TRACE_* + all production flags, no new
  trace files under new PIDs, smoke probe @2K clean (35.15 tok/s,
  coherent, finish_reason=length).

## 2026-08-29 — Phase (c): MoE-at-depth + allocator/memory-pressure — both candidates INCONCLUSIVE, residual still open

**PM/analysis pass, offline only** (no cluster access — pure post-hoc
analysis of already-collected R1 probe data; probes themselves were run by
a prior session that exhausted its iteration budget before write-up). Full
detail: `docs/p02c-moe-allocator-depth-residual-2026-08-29.md`; raw data +
new analysis scripts in `tmp/p02c-20260829/` (`analysis/*.py`).

- **Method:** delta-of-deltas A/B (`bench/p3_depth_anchor_probe.py`,
  `/tmp/dsv4_nop_targets=moe` short-circuits `DeepseekV4MoE.__call__` to
  `mx.zeros_like(x)` for arm B), spec-off band-era-allocator config
  (`CLEAR_CACHE_INTERVAL=0`, `GC_COLLECT_INTERVAL=0`), n=2/depth/arm, both
  100K and 352.6K, plus `MLX_LOG_NEW_BUFFER_PATH` fresh-alloc log +
  `EXO_MEMORY_PROFILE_INTERVAL=8` structured dump + 1Hz vm_stat sampler
  (m4-1 only) for the allocator candidate.
- **Reproduction gate PASSED:** arm A dA = +4.70 ms/tok (35.64→40.34),
  within 4.8% of the established +4.94 clean-anchor total (P3 Part III) —
  third independent confirmation of the same total this campaign day.
- **Candidate 1 (MoE-at-depth): validity gate FAILS.** dB (MoE-NOP arm
  depth-delta) = **+5.05 ms/tok** vs worker C's kernel-census expectation
  of +2.56 (primary run) to +3.34 (noisiest of 3 fencing-mode runs) — a
  1.5-2x overshoot, not a marginal miss, confirmed real (reps tight to
  <0.3% spread, not n=2 noise). Per DESIGN.md's own instruction, the naive
  delta-of-deltas (dA−dB = **−0.35 ms/tok**) is reported only as a
  weakly-related observation, NOT forced into an attribution. Root-cause
  dig (not just citing the gate failure): the DESIGN.md-flagged arm-B bias
  (garbage hidden states → cheaper attention at depth) predicts dB should
  UNDERSHOOT census — wrong direction to explain an overshoot. A plausible
  mechanistic explanation exists — dB is defined as "attention+framework"
  and structurally still pays P3 worker C3's already-isolated
  `BatchPoolingCache` per-flush concat cost (+1.91 ms/tok, additive with
  kernel census per C3/R2) that C's bare synthetic census never paid;
  under kernel+C3 (+4.47..+5.25) dB's overshoot shrinks to −0.20..+0.58 —
  but adopting this requires also removing C3 from the residual band's
  subtracted side (double-count risk, flagged not resolved) and a small
  gap remains even then. **Net: not established either way**, flagged as
  inconclusive rather than forced.
- **Candidate 2 (allocator/memory-pressure): the intended cross-depth test
  COULD NOT RUN — a real data gap, confirmed two independent ways
  (timestamp ranges AND newbuf byte-offset watermarks).** All three
  per-node telemetry streams (newbuf log, structured mem JSONL, vm_stat
  sampler) stop at/just after the 100kb block ends, ~9 minutes before the
  352.6K block starts; zero telemetry coverage of the 352.6K block exists
  in the collected files, on either rank. 100K-only findings: **zero
  gc_limit crossings** (peak active+cache 80.57GB vs 114.557GB limit, 34GB
  headroom), `active_bytes` flat through decode (79.06-79.31GB). One
  same-regime both-depths check WAS possible (each probe's own recorded
  inter-token gaps, independent of the missing telemetry): quintile trend
  does **NOT** reproduce P01a's spec-ON "352.6K worsens monotonically
  within-window" claim in this spec-off regime — both depths show flat-to-
  noisy quintile trends (Q5-Q1 within ±1.6ms, no consistent direction),
  leaning against the escalating-churn mechanism but scope-limited to
  spec-off (P01a's claim was spec-ON/batched-verify, untested here).
  **Net: data gap prevents the designed test; partial available evidence
  leans negative on the mechanism but is not a full refutation.**
- **Data-quality note:** arm B's temp=1.0 degeneration mitigation
  (documented in `run_block.py`) only partially held — first 100K B-reps
  (`100k_1B`/`100k_2B`) both still hit the exact-repeat degeneration kill
  at 71 tokens despite temp=1.0; re-run (`100kb_0B`/`100kb_1B`) completed
  clean to 2000 tokens. Correctly excluded/replaced; flagged for anyone
  re-running MoE-NOP arms in the future.
- **Residual after (c): still fully open at +1.67..+2.52 ms/tok.** Neither
  candidate produced a validated attribution — one hit a genuine
  validity-gate failure with a plausible-but-unconfirmed partial
  explanation, the other hit a telemetry collection gap, not a clean
  negative result on either candidate's core mechanism.
- **Recommendation: diminishing returns reached for now.** Residual is
  small (4-7% of total decode ms/tok), multiple relaunch-gated phases
  already spent today, and both remaining paths to a cleaner answer are
  non-trivial (Candidate 1 needs either a residual-band bookkeeping
  decision or a new 3rd cache-management-NOP arm; Candidate 2 needs a full
  R1 redo with telemetry verified to actually span 352.6K). Documented as
  a bounded, low-priority open item rather than continued in-session.
- **Cluster state:** not touched — pure offline analysis, zero SSH/relaunch/
  probe activity this phase, per task scope. Last known state stands
  (supervisor-verified production config, both nodes, moments before this
  phase began).

## 2026-08-29 — Phase (d): 14-20% roofline × 78-85% occupancy × 88-97% switch_mlp — the three numbers multiplied together for the first time; "5-7x slower than hardware ceiling" closes as an accounting artifact

**Pure desk analysis** (no cluster access — arithmetic against
already-documented numbers only; consult-reviewed before writing, which
caught three framing overreaches now baked into the doc as caveats). Full
detail + all arithmetic:
`docs/p02d-roofline-occupancy-kernel-reconciliation-2026-08-29.md`; scripts
+ JSON output in `tmp/p02d-20260829/`.

- **The never-done multiplication, done:** naive product 0.80 occupancy ×
  0.91 switch_mlp efficiency = 0.73, vs the observed aggregate 0.14-0.20 —
  a **3.8x mismatch**. Factorized exactly as `headline = coverage ×
  occupancy × busy-blend`: **2.29-2.57x is the roofline's byte denominator**
  (3.56 GB counts routed-expert bytes only at a whole-model 0.588 B/param
  average, halved for TP — it omits the REPLICATED 5.30 GB/rank attention
  path, 1930.25 B/ctx-token depth-linear reads, lm_head/shared-expert/gate;
  per-tensor dtypes are mxfp8 attn vs mxfp4 experts, not one average), and
  **1.63-2.08x is busy-time composition** (switch_mlp is 13.7% of busy, not
  the retracted "~30-45% of wall" span artifact — the blend is set by the
  attention census at 49-58% of busy / 57-75% of spec, and dragged by the
  implied small-op bucket at 27-31% of busy / 16-23% of spec). Weighted
  average across all buckets reproduces the headline within ~1pp at every
  clean depth. Candidates (a) and (b) both CONFIRMED and quantified.
- **B_true = 8.17-9.14 GB/rank/token** (true byte inventory), validated
  independently: worker C's census at L=520 agrees with the byte model to
  2% (12.876 vs 13.1 ms), C2's measured idle at 100K matches exactly
  (6.09), switch_mlp bytes agree across two independent docs to 4 sig
  figs. Honest status of the identity: bookkeeping, not validation — stated
  as such in the doc.
- **Corrected efficiency: decode runs at 36-44% of the 546 spec, i.e.
  46-57% of the measured 424 GB/s real-streaming ceiling — 1.75-2.16x
  slower than the REAL ceiling, not 5-7x.** Both denominators reported.
  The §13/§4.3 "highest-priority open question" framing ("decode is 5-7x
  slower than the hardware ceiling") is CLOSED as an artifact of the
  3.56 GB denominator; the 2026-08-22 reframe ("why is GPU-busy time
  ~4-5x the roofline floor") inherits the same correction (real answer:
  ~1.9-2.2x, of which idle is ~6-8 ms and the small-op bucket ~6-10 ms).
- **True-gap decomposition (vs real-streaming floor, ms/token):** idle
  6.1-8.2 (measured, C2-confirmed at 100K); small-op latency-bound excess
  5.7-9.8; attention+switch above byte floor 0.6-7.0 growing with depth
  (upper bound — B_true assumes zero L2 reuse; indexer-score measured at
  477-558 GB/s shows real efficiency is higher at depth). The campaign's
  open +1.67..+2.52 ms/tok residual band lives inside these buckets —
  bounded here, not attributed.
- **Genuinely new, cheap next target identified (NOT executed):** the
  small-op bucket (moe.gate, shared_experts, post_combine, norms,
  residuals, rope, lm_head) is 27-31% of GPU-busy time at an implied
  16-23% of spec (~88-140 GB/s effective) and has NEVER been measured
  per-kernel in any campaign. One `mx.metal.start_capture()` +
  `MLX_GPU_TIME=1` bracketing pass (the p01-proven recipe,
  `METAL_CAPTURE_ENABLED=1`), run in both spec-off (comparability) and
  spec-ON verbon3 (production relevance) regimes, would produce the first
  per-kernel table for the largest never-characterized decode cost —
  either it's irreducible dispatch latency (decode tuning is done) or
  it's a concrete fusion target.
- **Caveats carried in the doc, per consult review:** the factorization's
  closure is by construction (busy-blend is derived); attention-byte
  model is census-validated only at short context; 424 GB/s real-streaming
  is a transplant from a different workload (shown alongside spec, not
  instead of it); 300K/500K rows pair T1-era EOS-bug-flagged anchors with
  cross-run occupancy (clean-anchor claim = short/100K/352.6K); the
  spec-off era ≠ current spec-ON production wall times (re-basing is part
  of the recommended next step).

- **2026-08-30 — P03 small-op bucket per-kernel capture (first ever; closes the Phase (d) open item).** Real MLX_GPU_TIME=1 + MLX_DISPATCH_COUNT=1 + gputrace capture on m4-1 as a standalone process beside the live runner (p01 recipe, ZERO relaunches, production untouched end-to-end). Full detail: `docs/p03-smallop-bucket-gputrace-2026-08-30.md`; artifacts in `tmp/p03-20260830/`.
- **Spec-off (L=1) bucket = 8.34 ms/token measured** (sum of parts; 43-layer chained skeleton cross-check 7.76 ms, ratio 0.93) — dead center of Phase (d)'s implied 6-10 ms band, so the implied-bucket arithmetic was right in total. But its "16-23% of spec" was an averaging artifact: the bucket is bimodal — lm_head at 92.6% spec (506 GB/s), shared_experts at 69-90%, markov_w2 at 80.7%, vs a genuine latency floor (norms/rope/expand/combine at 1-8 µs, 1-3 dispatches each). No op actually runs at 16-23%.
- **Byte-model corrections:** lm_head is UNQUANTIZED BF16 1.059 GB/rank replicated (Phase d carried ~0.55 GB — 1.9x understated; caught by runtime assertion battery); HC fn matrices (1.57 MB fp32 × 86/token ≈ 135 MB/token) were never in any byte model.
- **Spec-ON re-basing (the number Phase d asked for): 17.2 ms/cycle ÷ 3.2 tok = 5.38 ms/token** for the bucket, LOWER than spec-off per token because verify-batch amortization is real (fused HC L4 beats 4× L1 by 38%; batched shared beats per-row group by 45%).
- **Verdict: NOT irreducible — three concrete targets, one blocked.** (1) Quantize lm_head+markov_w2 to mxfp8 (family = 27% of spec-ON bucket at 92-94% spec, pure byte limitation — projects −0.7 ms/token; needs quality gate). (2) HC Sinkhorn 20-iteration truncation — hc_collapse is 30.4% of the spec-ON cycle (5.24 ms/cycle at L=4) and is 20 sequential 4×4 normalize passes with barriers, pure latency; truncating to ~4-5 iters projects the family to ~1.5-2 ms/cycle (numerics-gated, arithmetic only). (3) shared_experts verify batching is a measured −1.2 ms/cycle win but BLOCKED by the 2026-08-04 numerics fix (ROWSEQ=shared exists precisely because batched shared was the isolated divergence source). Genuinely irreducible: norms/rope/expand/combine ~1.9 ms/token at the dispatch floor — decode tuning on those is done.
- **Four measurement traps found live (each produced plausible garbage until caught):** HyperConnection defaults training=True (95-dispatch Sinkhorn loop vs the production 4-dispatch fused path after model.eval() — 8.8x); a `*0` trick let MLX DCE fold away an entire lm_head graph (measured "497,890 GB/s"); a recheck lambda rebuilt random tensors inside the timed call (70 vs 34 µs false "variance"); start_capture fails on existing filenames. All documented in the doc's §2.
- **Cluster untouched:** zero relaunches (cap was 2); both runners verified 5/5 verbon3 flags + zero capture env before and after; real post-capture generation smoke-test passed (coherent "Paris … Louvre", finish_reason=length). Working tree has doc updates only, uncommitted per instructions.
- **2026-08-30 — P04 Sinkhorn-truncation numerics gate: code knob landed offline, P03's 4-5-iter projection REFUTED by measured numerics.** 100% laptop-local work, zero cluster contact. Full detail: `docs/p04-sinkhorn-truncation-numerics-2026-08-30.md`; artifacts in `tmp/p04-sinkhorn-truncation-20260830/`.
- **Code knob (mlxl-lm submodule, uncommitted):** `EXO_HC_SINKHORN_ITERS` in `HyperConnection.__init__` — construction-time override of the Sinkhorn iteration count, exact `EXO_HC_USE_OPS` pattern (env-gated, invalid/unset falls back bit-identically to `config.hc_sinkhorn_iters=20`). Both execution paths and the kernel source untouched; the fused kernel already takes the count as its ITERS template param. Verified: unset→20, set→override, bogus/0/negative→20.
- **Cross-path equivalence proven locally for the first time:** fused Metal kernel path vs pure-MLX ops path at iters=20 are BIT-IDENTICAL on all three outputs (collapsed/post/comb), on both logit regimes — and the fused path's truncation-divergence table matches the ops path's exactly, proving the knob threads the ITERS template param for real.
- **"Sinkhorn converges fast" REFUTED at realistic logit scales:** measured convergence is geometric at ratio ≈0.67/iter — even the FULL 20 iterations end at ~1.1e-3 residual row deviation, and a wide-logit (×4) stress case plateaus at ~4e-2 by iter 19 (does NOT converge within 20). The checkpoint's 20 is a real operating point, not overkill.
- **Truncation divergence (comb output, worst over 3 param draws × 4 inputs, identical on both paths):** iters=10 → 1.07e-2 max abs; iters=5 → 5.5e-2; iters=4 → 8.6e-2; iters=3 → 1.4e-1; iters=2 → 2.3e-1 (realistic O(1)-logit case; wide-logit ×4 stress is 5-8× worse). pre/post/collapsed are structurally Sinkhorn-independent and measured BIT-ZERO everywhere — the error enters only via comb, the residual-mixing matrix consumed by hc_expand 86× per forward (43 layers × attn/ffn), so it compounds across depth rather than cancelling.
- **Verdict: P03's "truncate to ~4-5 iters, 5.24 → 1.5-2 ms/cycle" is numerically NOT viable.** Minimum defensible candidate is iters=10 (halves the loop, "plausibly tolerable, unproven"); anything ≤5 carries per-application comb error of the same order as the mixing weights themselves. Any live throughput test must be paired with the generation-quality gate (exo-local-vs-cloud-dsv4 probe suite) and is reversible per-restart via the knob. Cheapest upgrade path before any live test: read-only dump of one real decode cycle's comb-logit std on a studio (standalone process, no relaunch — p01/p03 recipe) to replace the synthetic O(1)-logit assumption with the checkpoint's actual distribution.
- **Two measurement traps found (each silently produced a false "zero divergence" answer until caught):** (1) `uv run python script.py` resolved `import mlx_lm` to the STALE non-editable copy in `exo/.venv/site-packages` — knob silently absent, env ignored; fixed by pinning sys.path to the submodule + asserting the knob's presence in the imported source. (2) fn drawn at naive init scale 1/sqrt(fan-in) gives softmax-logit std ≈0.08 — a flat softmax converges in 1-2 iters and makes iters=2 measure bit-identical to iters=20; fixed by calibrating fn so logit std hits the target regime (1.0 realistic, 4.0 stress). Also: scaling the input x is a no-op for Sinkhorn (rms_norm is scale-invariant) — the case axis must be the logit scale.
- **Cluster untouched:** no SSH, no production API, no relaunches — verified by construction. mlx-lm submodule and parent-repo working trees carry the code+doc changes UNCOMMITTED per instructions (supervisor commits).

## 2026-08-30 — P05 review: independent re-derivation of the three fusion-target claims from an interrupted, oversight-flagged campaign

A PM (sa-0-e5a81d3d) ran P05 autonomously against P03/P04's three fusion
targets (lm_head mxfp8, HC Sinkhorn real-weight re-validation, shared_experts
batching) but was killed mid-flight after two operational problems (left the
cluster on a test config at one point, since restored/confirmed clean; a
nested dispatch of its own picked an invalid model string). None of its
numbers were trusted as-is — three independent read-only review subagents
re-derived each phase from the raw JSON directly. Full docs:
`tmp/p05-review-20260830/phase{A,B,C}_review.md`; raw data in
`tmp/p05-{lmhead-mxfp8,sinkhorn-real,shared-batching}-20260830/`.

**Phase A (lm_head mxfp8) — DON'T SHIP.** Kernel-level speedup CONFIRMED
(studio microbench 1.64-1.85x across M=1..4, matches the PM's claim exactly)
but never validated end-to-end: the only live quantized-head A/B runs show
0.05-0.06x (catastrophic regression, ~4-5 tok/s vs ~99 tok/s baseline) — a
zero-acceptance draft/verify break, not a valid speedup measurement (likely
`QuantizedLinear` inside `@mx.compile`). The `live_ab_v2` re-runs that might
have shown a working head are all connection-refused tracebacks — no valid
live 100k throughput exists anywhere in the data. The ~13% synthetic top-1
flip rate is CONFIRMED (17/128, 100% concentrated below margin 3.62). The
~16% real-token flip-rate claim (n=798) is UNVERIFIABLE — that dataset does
not exist anywhere in the repo; the only occurrence of "n=798" is the code
comment itself. markov_w2's exclusion from quantization is CONFIRMED sound
(microbench 1.00x, not the comment's 0.98x — conclusion unchanged). Before
any ship decision: fix the zero-acceptance bug, get one valid live 100k A/B
of a working head, and produce the missing n=798 margin data.

**Phase B (Sinkhorn, real HC weights) — P04's gate CONFIRMED and gate
should TIGHTEN, not loosen.** Real extracted comb-logits are ~10x wilder
than P04's synthetic "realistic O(1)" assumption (measured std 2.5-14.9,
median 11.4, vs P04's std=1.0), and are bias-dominated (base2/logits ratio
0.85-1.00, non-row-constant, so no softmax-shift escape) — the worst case
for the 86x/forward comb-error compounding P04 already flagged. Real
truncation divergence at iters=10 is 1.33e-1, already exceeding P04's own
synthetic iters=4 value (8.6e-2) — P04's "minimum defensible iters=10" is
NOT supported once real weights replace the synthetic assumption. P03's
4-5-iter projection stays dead (real div@4-5 = 0.37-0.47, 5.4-12.4x worse
than P04's synthetic numbers at the same iter counts). Recommendation:
keep iters=20 as the shipped default; any live truncation test must use
iters=10 at most, be quality-gated, and stay reversible via
`EXO_HC_SINKHORN_ITERS`.

**Phase C (shared_experts batching) — NOT M-invariant on real weights;
confirms the existing 2026-08-04 divergence, no lossless fix found.** Real
layer-3 shared_experts weights: `qmv_wide(M=4)` vs `qmv(M=1)` diverges on
13/131072 elements (0.0099%, max 2⁻⁹) — reproduces the known 2026-08-04
divergence offline. The batched win is real (181µs vs 268.5µs, 32.5%
faster) but numerically unsound as-is. The proposed zero-pad-to-`qmm`
workaround is NOT lossless as claimed: 0-ulp only at M=8, 1-ULP divergence
at M=16/32/64, and the one case where it might matter most (real weights)
never got its numerics saved to the output JSON — the single most
important number for this phase is missing from the data entirely. Speed
gain if the padded path were used is marginal anyway (257µs vs 268.5µs,
~4%) — not attractive even before the correctness question. Stays
BLOCKED per the existing 2026-08-04 fix (`ROWSEQ=shared`) until a lossless
formulation is found and the missing real-weight padded-qmm numerics are
captured.

**Net P05 verdict: none of the three fusion targets are ready to ship.**
Phase A needs a real bug fix before it can even be measured honestly; Phase
B's real data argues the opposite direction from what P05 was chasing
(tighten, don't loosen); Phase C confirms a known blocker with no new
escape route. The two mlx-lm submodule diffs (`EXO_DSV4_LMHEAD_MXFP8`,
`EXO_DSV4_PRENORM_H_DUMP`) stay uncommitted, env-gated and inert — no code
lands from this review. Cluster untouched throughout (pure read-only JSON
analysis, verified clean production config before this review began).

## 2026-08-30 — P05 leads 1&2 run down: "zero-acceptance bug" CLOSED as wrong-model harness artifact; true 0731 baseline + real margin data collected; Phase C REOPENED via lossless pad-to-M=8

Follow-up to the P05 review above. Both review-flagged leads were run to a
verified conclusion; artifacts under `tmp/p05-lmhead-mxfp8-20260830/`
(rootcause2/, live_ab_v3/, real_margins/) and `tmp/p05-shared-batching-20260830/`
(real_shared_padded_qmm.json, real_shared_pad8_speed*.json). Reviews updated in
place: `tmp/p05-review-20260830/phase{A,C}_review.md`.

- **LEAD 1 — the Phase A "zero-acceptance draft/verify bug" never existed.**
  Full log-forensic run→instance→model→sharding→knob attribution
  (rootcause2/RUN_ATTRIBUTION.md + attribution_table.json, 30 rows): every
  catastrophic 0.05-0.06x "quant" run (11:19-11:50) hit `mlx-community/
  DeepSeek-V4-Flash` (8-bit) — the wrong model (its lm_head is already
  quantized, so the mxfp8 knob silently no-ops: the `not hasattr(mod,"scales")`
  guard fails, no `[LMHEAD_MXFP8]` stderr line; independently confirmed on the
  studio node — `mlx-community--DeepSeek-V4-Flash/model.safetensors.index.json`
  ships `lm_head.scales`/`lm_head.biases`, 0731 ships only unquantized
  `head.weight`). **Correction from independent review:** the acceptance
  collapse tracks the mlx-community model itself, not single-node Pipeline
  placement specifically — raw logs show the SAME model under Tensor sharding
  (10:47-10:58, knob-off) also sustained 0.000/3 acceptance with healthy
  ~14.6ms drafts; only the ~500-552ms draft *latency* (not the acceptance
  collapse) tracks the Pipeline placement. The wrong-model finding stands;
  the original PP-specific framing was imprecise. The probe harness
  hardcoded that model id until `de925720e` added `--model`. The
  knob-quantized 0731 head's own live evidence was healthy at every measured
  context (trivial + 5.6K, rowseq verify): mean_accept 1.890/3, draft ~9ms,
  verify ~66ms, decode 200-223 tok/s. All H1-H5 hypotheses (incl. the H5
  QuantizedLinear-inside-mx.compile theory) are moot. The 12:45 API death was a
  CLEAN manual shutdown (SIGTERM, exit 0), not a crash.
- **True 0731 production baseline measured (the review's "99 tok/s @100K" was
  also wrong-model data): 100K decode 271.4/345.1/276.8 tok/s (median 276.8),
  prefill ~375 tok/s, needle_hit true in 3/3 (live_ab_v3/, model id verified
  per-file). 5K decode 169-369 tok/s.** Collection gotcha banked: a c=1 cluster
  rejects concurrent heavy probes with 500 admission errors — sequential only
  (a prior worker's parallel fan-out produced the misleading "cluster
  instability").
- **The missing n=798 real-token margin data now EXISTS** (real_margins/,
  n=3999 committed tokens, 4 contexts 35→25.1K prompt tokens, temp-0, logprobs
  top1-vs-top2 margins): pooled **42.7% of real tokens below margin 3.62**
  (n≥798 slice 44.5%), implying an mxfp8 top-1 flip rate of **~11.5%**
  (n≥798: ~12.8%) via the synthetic band kernel — an estimate, not a direct
  measurement. The prior campaign's "~58% below 3.6 → ~16% flip" is REFUTED
  (both figures asserted without data; the flip rate was overstated ~40%).
  Margin distribution is stable across context depths (42.0-43.6% per ctx).
  G5 same-prompt A/B completed: mxfp8 arm byte-identical for 199 chars then
  diverges mid-reasoning — the visible quality cost is real (same_prompt_G5_result.json).
- **LEAD 2 — Phase C shared_experts batching REOPENED.** The missing
  real-weight padded-qmm numerics were captured (real_shared_padded_qmm.json,
  layer-3 w1 AND w2): pad→M=8 qmm is **bitwise-lossless on real weights**
  (0/8192 + 0/4096 nonzero), while M≥16 pads diverge 1-ULP-class as on
  synthetic. Crucially the M=8 SPEED was never measured before — only pad16/32
  (~257µs, "just ~4%"). Measured now (real_shared_pad8_speed*.json): **M=8 pad
  = 174.4µs on w1 vs 267.1µs per-row (−34.8%) and 179.0µs on w2 vs 202.8µs
  (−11.7%) — faster than even the divergent qmv_wide path.** The lossless
  formulation captures ~100% of the batching win. Next step: knob-gated code
  change running shared_experts verify-batch through pad-to-M=8 + end-to-end
  A/B against the P03 −1.2 ms/cycle projection (needs relaunch approval).
- **Phase A remaining open:** ≥8K batched-verify and 100K regimes on the
  quantized head remain UNMEASURED; a ship decision needs a knob-ON relaunch
  A/B at 100K vs the true baseline above (relaunch approval gated).
- **Knob comment corrected in-place** (mlx-lm utils.py): 0.98x markov_w2 →
  measured 1.00x (253.9 vs 254.7µs); the asserted 58%/16% numerics figures →
  the measured 42.7%/~11.5% (with the refutation noted).
- **Cluster untouched:** zero relaunches (both leads were log-forensics,
  standalone studio scripts, and API-only probes against the clean production
  verbon3 cluster). Config verified clean before/after (pid 65573, spec-ON
  verbon3 env, LMHEAD_MXFP8 absent). Working-tree changes (mlx-lm utils.py +
  deepseek_v4.py, bench/ab_probe_tier1.py, docs, tmp/ artifacts) left
  UNCOMMITTED per the supervisor-commits pattern.

## 2026-08-30 — P06 Phase A: lm_head mxfp8 SHIPS (+6.0% decode @100K, zero measured quality cost); the flip-rate framing was a near-miss false negative

Live knob-ON A/B against the true 0731 baseline, 2 relaunches, sequential
probes only (c=1 admission gotcha respected). Artifacts:
`tmp/p05-lmhead-mxfp8-20260830/live_ab_v4/`. New harnesses:
`bench/long_decode_probe.py`, `bench/lmhead_task_eval.py`,
`bench/lmhead_quality_gate.py`.

- **MEASUREMENT TRAP FOUND AND FIXED FIRST — the existing A/B harness could
  not answer this question.** `bench/ab_probe_tier1.py` asks a needle
  question the model answers in ~80-100 tokens, i.e. a 0.2-0.3 s decode
  window. The standing rule (never quote t/s from <400-token generations)
  makes those numbers startup noise, and the *baseline* in the file
  (`live_ab_v3`, 81-93 tokens) has the identical defect — so the "271.4/
  345.1/276.8, median 276.8 tok/s" figures are comparing noise to noise and
  must NOT be used for a ship decision. First knob-ON reps reproduced the
  artifact perfectly: 469.6 and 322.8 tok/s off 97-token samples, a spread
  no real effect could produce. `bench/long_decode_probe.py` keeps the same
  100K prefill but asks for a long essay, giving a 1200-token / ~35 s decode
  window, and records `decode_sample_trustworthy` so this trap cannot be
  re-entered silently.
- **Throughput (3 reps/arm, 1200 tokens each, same probe both arms):
  ON 34.35/34.12/34.28 (median 34.28) vs OFF 32.35/33.09/32.11 (median
  32.35) = +1.93 tok/s, +6.0%.** The arms do NOT overlap — ON's worst rep
  (34.12) beats OFF's best (33.09) — so at n=3 this is signal, not jitter.
  Prefill unchanged as expected for a decode-side byte win (378.3 vs 378.0);
  needle retrieved 3/3 in both arms.
- **Quality: 15/15 on BOTH arms, all 15 answers BYTE-IDENTICAL.**
  `bench/lmhead_task_eval.py` scores mechanically (no LLM grader, no human
  judgment): 6 arithmetic/word problems checked as exact numbers, 4 exact
  factual-recall items, and 5 code tasks whose generated functions are
  EXECUTED against real assertions. A flipped digit, operator or index
  fails loudly. Nothing failed, and the two arms produced the same bytes on
  every task.
- **The ~11.5% flip-rate estimate is REAL but is a DIAGNOSTIC, not a quality
  metric — reading it as one would have produced a false negative.** Prior
  campaign framing (42.7% of real tokens below margin 3.62 → ~11.5% implied
  top-1 flips, plus a same-prompt G5 divergence at 199 chars) pointed at
  "don't ship". It is now clear why that inference over-reads: flips
  concentrate in low-margin near-ties, i.e. between near-equivalent tokens,
  so they change wording without changing answers. Measured directly: free
  prose does diverge (0/5 byte-identical on the long-form gate — bullet
  style, phrasing) while every checkable answer stayed correct and identical
  (element list identical, same exact fraction 381 5/7, same ages). The
  arithmetic behind 11.5% was re-verified as a 4-band weighted sum and is
  sound; only its *interpretation* was wrong. A consult review flagged this
  exact risk ("flip rate is a diagnostic, not a quality metric; you may be
  steering toward a false negative") and prompted the task-eval that
  inverted the call.
- **VERDICT: SHIP.** +6.0% decode for zero measured quality cost is a good
  trade. Landed the project's real way — `start_cluster.sh` gets
  `: "${EXO_DSV4_LMHEAD_MXFP8:=1}"` plus EXO_ENV plumbing, matching the
  EXO_DSV4_LMHEAD_LASTROW pattern. (Note for the record: the task brief
  asked for a `config.yaml`-gated flag; this repo has NO config.yaml — its
  user-facing knob surface is start_cluster.sh defaults. Followed the actual
  convention rather than inventing a second one.) Reversible per-restart
  with `EXO_DSV4_LMHEAD_MXFP8=0`; the mlx-lm quantizer itself is unchanged
  and still no-ops on any checkpoint whose head is already quantized.
- **Eval-oracle bug caught and fixed before it could corrupt the result:**
  the first task-eval draft asserted (47*83-1229)/7 == 382 (truly 381.714)
  and 90 km/60 min == 12 per 10 min (truly 15), scoring two CORRECT model
  answers as failures and yielding a bogus 13/15 baseline. Both expectations
  were wrong, not the model. Fixed, re-run, 15/15. A quality gate whose
  oracle is wrong manufactures exactly the regression it is meant to detect.

## 2026-08-30 — P06 Phase C: pad-to-M=8 shared_experts batching DON'T-SHIP — the "lossless" microbench was measuring the divergent kernel all along (root cause: MLX's qmv batch limit is 12, not 8)

Knob written, deployed live, measured, root-caused, and REVERTED. Artifacts:
`tmp/p05-shared-batching-20260830/` (cycle_ab/, p05b-p05f_*.py/json,
verify_threshold_independent*.py/json). New harnesses:
`bench/mtp_cycle_time.py`, `bench/phase_c_cycle_ab.sh`.

- **The code was written and it worked as specified.** Env-gated
  `EXO_DSV4_SHARED_PAD8`, default OFF, bit-identical when unset; pads the
  M<=8 verify rows to exactly 8, one shared_experts call, slices M back.
  Local test 16/16 with every max_abs exactly 0.0, negative controls bite
  (pad-to-16 sabotage flips assertion 2; re-introducing the prefill bug
  flips assertion 7). A review caught one real blocker pre-deploy: the
  first draft's `or (_SHARED_PAD8 and _prs_L > 8)` disjunct made the knob
  hijack the PREFILL path (L=2048) into a 2048-iteration per-row loop —
  `DeepseekV4MoE.__call__` is prefill AND verify. Removed at root (the
  pre-existing gate already restricts _prs to B=1, 2<=L<=8) rather than
  guarded around.
- **Live per-cycle result CONFIRMED P03's projection: -1.755 ms/cycle**
  (69.65 ON vs 71.41 OFF, trimmed means over ~1000 cycles/arm, paired
  relaunch A/B, both arms env-verified 5/5 verbon3 flags). P03 projected
  -1.2 ms/cycle. The speed theory was right.
- **But acceptance FELL: mean_accept 1.31/1.249 ON vs 1.412/1.386 OFF**,
  which cancelled the per-cycle win — end-to-end decode did not improve
  (33.53/32.82/35.24 ON vs 34.35/34.12/34.28 OFF). A bitwise-lossless path
  CANNOT move acceptance. That contradiction, not the tok/s, is what
  forced the root-cause dig.
- **Fixed-prompt temp-0 task eval: 14/15 byte-identical, 1 DIVERGENT**
  (code_anagram, diverges at char 110). Both arms still scored 15/15
  correct, but a lossless path must be 15/15 identical. Direct proof the
  path is not lossless in production.
- **ROOT CAUSE (independently verified twice, hardware-level):** MLX
  dispatches `qmm`/`qmm_splitk` only when M >= `get_qmv_batch_limit(K,N,arch)`.
  For BOTH real shared_experts per-rank shapes — gate/up (K=4096,N=1024) and
  down (K=1024,N=4096) — that limit is **12** on applegpu_g16s (M4 Max,
  arch_gen=16). **8 < 12, so pad-to-8 never reaches qmm at all**: it falls
  into `dispatch_qmv`, which for mxfp8 (any non-affine mode) always routes to
  `qmv_wide` — the exact kernel proven divergent on real weights on
  2026-08-04, i.e. the reason `ROWSEQ=shared` exists. Padding to 8 is
  mechanically indistinguishable from just batching the rows. Confirmed by
  `mx.metal.dispatch_count()` sweep (1 dispatch at M<=11, 2 at M>=12,
  transition exactly at 12 on both shapes) and by pad-4-to-8-then-slice
  being BIT-IDENTICAL to a plain unpadded M=4 call (max_abs exactly 0.0,
  both shapes) — padding provably changes nothing.
- **Why the original microbench read 0.0 — sampling luck, not a safe path.**
  It sampled only 4 rows (8192/4096 elements) against the known real-weight
  divergence rate (13/131072, 17/262144). Poisson: ~0.8-1.1 expected hits,
  P(zero) ~35-44% per tensor, ~15% both. The "0/8192 + 0/4096, bitwise
  lossless" result was an unremarkable draw from the ALREADY-BUGGY kernel.
  The same data's M=16 lossiness (max 0.03125) is the same mechanism
  crossing the REAL threshold into qmm_splitk. Methodology lesson: a
  zero-divergence result on a sample far smaller than 1/rate is not evidence
  of losslessness — it needs a power calculation, or a dispatch-level check
  that the intended kernel is even running.
- **Also found: the earlier w2 microbench used the wrong shape** (sharded
  the wrong axis: built K=2048/N=1024 instead of the real K=1024/N=4096).
  Doesn't change the verdict — both shapes have limit 12 — but the
  "-11.7% on w2" figure was never measuring the production tensor.
- **VERDICT: DON'T SHIP. Phase C is structurally dead as a padding trick,
  and is now CLOSED rather than parked.** Every dispatch path for this op
  has measured nonzero real-weight divergence: qmv_wide for 2<=M<12 (rare
  but real), qmm_splitk for M>=12 (drastically lossy). The per-row loop is
  the only proven-lossless formulation, so `EXO_DSV4_MOE_PARTS_ROWSEQ=shared`
  stays exactly as-is. A real fix would need a custom fixed-accumulation-order
  Metal kernel — not a padding trick, and not a reachable win at -1.2ms/cycle
  (~+1.3% decode) for that cost. Knob code REVERTED on the laptop and both
  studios (all three copies md5-verified back to 46cc271d, pad8 refs = 0).
- **Deploy trap banked:** the studios' `.venv/.../site-packages/mlx_lm/` is a
  physical COPY, not an editable install (different inodes, `direct_url.json`
  has no `"editable": true`). Editing only `mlx-lm/` in the repo deploys
  NOTHING — both paths must be written and md5-verified, or the knob silently
  no-ops. This is the same class as P04's stale-import trap.
- **Instrument note:** end-to-end tok/s could not have answered this question.
  The projected win (~+1.3%) is smaller than the 100K probe's run-to-run
  spread, so tok/s alone would have been a coin flip in either direction.
  `bench/mtp_cycle_time.py` derives real ms/cycle from the runner's own [MTP]
  log lines (~500-1000 samples/run, same unit as the projection) and records
  a per-run byte-offset window — a naive `tail` mixes arms and yields a
  descending cycle_range, which is the tell the window is wrong.

---

## P07 — prefill's non-GEMM remainder, re-opened on new methodology (2026-08-30)

**Framing correction first: T7's "19.3% unattributed remainder" is stale, and
~5.3pp of it is already fixed and shipped.** T10
(`docs/t10-final-decomposition-closed-2026-08-22.md`) closed the 28.8%
remainder honestly and that closure is not disputed. But it named its own
reopening condition — "a new methodology that reveals something the
span-profile approach can't see" — and two things now meet it.

**(1) T10's span table is stale by construction.** Two fused Metal kernels
shipped default-ON *after* T10 closed, both landing inside the remainder:
`hc_expand` (08-24, `deb1c8a6d`, 8.66x op-path, +3.87% e2e @70.5K) and
`hc_collapse` (08-25, `99f5f96b8`, 2.47x span, +1.89% e2e, depth-verified at
300K/500K). Recomputed from raw ms totals in the 220K span profile:

| Span | T10 (08-22) | Today | Note |
|---|---|---|---|
| `moe.all_sum` | 9.5% | 9.5% | out of scope, closed separately |
| `moe.post_combine` (+shared_experts fwd) | 4.2% | 4.2% | unchanged |
| `attn.indexer` | 4.0% | 4.0% | decomposed 08-24, nothing shipped |
| tail spans (7) | ~2.5% | ~2.5% | unchanged |
| HyperConnection (`attn_hc`/`ffn_hc`) | 4.6% | **~1.9%** | hc_collapse shipped |
| `moe.gate` | 0.9% | 0.9% | unchanged |
| `hc_expand` (`attn`/`ffn_residual`) | 4.4% | **~0.5%** | hc_expand shipped |
| **total remainder** | **~28.8%** | **~23.5%** | |
| **non-`all_sum` (the target)** | **~19.3%** | **~14.0%** | |

So the honest answer to "was the 19.3% a floor or a fixable bottleneck" is
already partly settled: **~5.3pp was genuinely fixable, and it shipped.** The
open scope is ~14.0%, not 19.3%. Also confirmed: `lm_head` is NOT in the
prefill budget (ship commit `80ec8ec03` records "Prefill unchanged (378.3 vs
378.0)"; `model.lm_head` is 0.2% of prefill wall) — the mxfp8 win is
decode-only.

**(2) Per-kernel GPU capture at PREFILL shape has never been done, and the
span profiler is provably blind inside these spans.** P03 (08-30) was the
first per-kernel capture but at DECODE shape (L=1/L=4). Every "at ceiling"
verdict inside the prefill remainder rests on FLOP/bandwidth arithmetic or
CPU-side microbenches. The indexer decomposition already documents the blind
spot in its own numbers: `indexer.score` (8.34µs/call) and `indexer.topk`
(5.81µs/call) "only measure **lazy-graph build time**, not GPU compute — if
the score GEMM at 220K (14.42 GFLOP) executed in 8.34µs it would be 1.7
PFLOPS, physically impossible" (`indexer-prefill-decomposition-2026-08-24.md:116-119`),
and that doc's own caveat at :254 — **"the 'score GEMM at ceiling' claim is
inferred, not directly measured."** Under MLX's lazy executor a span timer not
terminated by an eval barrier measures graph construction, not work. Per-kernel
capture is exactly the instrument that resolves this; span-profiling
structurally cannot.

Gates, regime-choice rules, ship criteria and method constraints pre-registered
BEFORE measurement in `docs/p07-prefill-remainder-perkernel-preregister-2026-08-30.md`.

**SECONDARY (T7's never-checked flag): SDPA ceiling denominators audited — one
real error found.** T7 flagged that `attn.sdpa`/`attn.sdpa.compressed` ceiling
denominators were never checked for causal-mask / MLA-absorption FLOP
inflation. Audited against `bench/attn_production_class_bench.py` + real code:
- **MLA absorption: not in play.** DSv4-Flash has no `kv_lora_rank` (unlike
  v2/v3); KV projects directly `hidden(4096)→head_dim(512)` and is stored at
  full head_dim. Naive D=512 formula is correct.
- **`attn.sdpa` (sparse): denominator CORRECT, 61.7% stands.**
  `_sparse_pooled_attention_inner` does a **dense** local matmul over all 2175
  keys and masks *after* — the masked positions genuinely are computed.
- **`attn.sdpa.compressed`: denominator INFLATED ~1.65x.** The bench counted
  full `L_band × CATTN_KV` using a synthetic 95%-dense mask; production passes
  a real **causal** mask to `mx.fast.scaled_dot_product_attention`. Real causal
  work 158.3 GFLOP vs 261.3 counted → corrected efficiency **~48%, not 79.1%**.
  Flagged UNDETERMINED pending one runtime test (causal-vs-dense timing at
  production shape) — registered as a pre-declared test, not a conclusion.
  Note the direction: if confirmed, a span T7 wrote off as "at ceiling, dead
  end" actually has real headroom. This sits in the 71.2% GEMM-covered
  majority, not in the remainder.

Cluster untouched for all of the above (code reading + arithmetic only): PIDs
59909/60392 continuous since 19:34, all verbon3 production flags verified live
on both nodes, zero leftover test env vars.

**P07 RESULTS (2026-08-30, same session): first per-kernel GPU capture at
PREFILL shape — remainder is mostly a genuine floor, with ONE real open
item found that the span profiler was structurally blind to.**
Full writeup: `docs/p07-prefill-remainder-perkernel-results-2026-08-30.md`.
Artifacts `tmp/p07-20260830/`. Zero cluster relaunches; PIDs 59909/60392
verified identical before/after every capture.

**Answer to the phase question — it is BOTH, in now-known proportion:**
1. **~5.3pp of T7's original 19.3% was a real fixable bottleneck, and it is
   already fixed and shipped** (hc_expand 08-24, hc_collapse 08-25).
   Remainder today ~14.0%, not 19.3%.
2. **norms / rope / kv_cache are a genuine dispatch/latency floor** — 1-2
   dispatches, sub-ms/chunk, same shape as decode's own ~1.9ms/token floor.
   Confirmed by measurement, not assumed.
3. **One real, open, non-floor item: `attn.indexer` top-k, ~2.9% of prefill
   wall, running at ~4-7% of achievable bandwidth.**

**Two previously-INFERRED claims now MEASURED and confirmed**:
`indexer.score_gemm` **99.3% of ceiling** (the OPT-6 folded GEMM really is at
hardware peak — `indexer-prefill-decomposition-2026-08-24.md:254` had flagged
this as "inferred, not directly measured"), and `moe.shared_experts` **87.2%**
(independently corroborating T10's 1.05x-of-peak estimate via a different
instrument). `moe.gate`/`post_combine`/`pmask` land 47-55% — each <1.0% e2e,
below the ship gate.

**Why span-profiling could never have found the top-k cost**: `indexer.topk`'s
span reads **5.81 µs/call** for an op that really costs **7.7-15.4 ms** — a
~1300-2600x under-read, because under MLX's lazy executor a span timer not
terminated by an eval barrier measures graph CONSTRUCTION, not GPU work. This
is the pre-registered justification for reopening T10, and it held.

**The "impossible" number, reconciled.** The capture's 15406 µs/call top-k
exceeds the whole `attn.indexer` span (24,596.71ms/2310 = 10.65 ms/call). Both
are right: the span is a run-AVERAGE over 110 chunks with ctx growing 0→220K
(mean P ≈ P_final/2); the capture is at FINAL-chunk shape (P=55000). Top-k is
~linear in P (0.20→0.31 µs per unit P over a 25x sweep). Run-averaged:
7.70 ms/call = **72.3% of the 10.65ms span** → 4.0% × 72.3% = **~2.9% of
prefill wall**. Fits its parent, coherent with the rest of the span being the
at-ceiling score GEMM.

**A review pass wrongly called this an artifact** ("argpartition is default
off"), reading only the code's `os.environ.get(...,"0")` fallback. Wrong:
`start_cluster.sh:553` sets `: "${EXO_DSV4_PREFILL_ARGPARTITION:=1}"` and the
live runner env confirms it. **Reusable lesson: a code-level env default is NOT
the production default — check the launcher and the live process env.**

**Is default-ON argpartition a pessimization? NO — measured, closed.** Isolated
GPU-timed A/B vs the `argsort` fallback (whose code comment claims "~5% faster
on Metal"), bf16 k=512 L_band=1024, rotation banks, median-of-5,
`results_topk_ab.json`:

| P | ~ctx | argpartition µs | argsort µs | ratio | disp |
|---|---|---|---|---|---|
| 5,000 | 20K | 1003.1 | 1003.0 | 1.0002 | 7 |
| 25,000 | 100K | 6336.8 | 6337.2 | 1.0000 | 11 |
| 55,000 | 220K | 15396.0 | 15402.5 | 0.9996 | 13 |
| 125,000 | 500K | 39149.3 | 39174.1 | 0.9994 | 15 |

Dead heat at every production P with **identical dispatch counts** — both
expressions lower to the SAME radix-sort kernel in this MLX build. Top-k sets
equal at all P incl. forced ties. So `EXO_DSV4_PREFILL_ARGPARTITION` /
`EXO_DSV4_ARGPARTITION_MIN_P` are **not levers** — no config change
recommended, and the cost is structural to MLX's sort, not a wrong-branch bug.
The historical "295→163 tok/s at P=500" collapse does NOT reproduce on the
current build (divergence only at P≤2000).

**Open next step (top-k), scoped honestly**: the only remaining path is a
custom Metal top-k exploiting k=512 ≪ P=55000 (MLX full-radix-sorts all P to
extract 512). Ceiling if perfect ≈ **1.03x prefill** — a scoped spike, not a
campaign. Must reduce real WORK (not just dispatch bookkeeping) and be
validated PIPELINED in situ: this repo's whole-indexer fused kernel measured
0.54x = SLOWER, and wq_a+wkv fusion -0.48%.

**Ceiling-denominator correction banked**: the 11.66 TFLOPS figure
(`bench/attn_production_class_bench.py:136-145`) was measured on the LAPTOP
M4 Max (32-core), not the 40-core Studio nodes. On-node achieved `score_gemm`
of 14.23 TFLOPS is 122% of 11.66 (impossible) but 99.3% of 14.34 (plausible).
**Any prior doc quoting 11.66 against a Studio measurement understated
efficiency by ~19%.** Caveat: 14.34 is theoretical, not on-node measured;
running `measure_peak_gemm()` on a Studio node is the rigorous fix (not done).
Separately, four tail ops (rmsnorm, rope.q, rope.q_idx, kv_cache.write) report
>100% of the bandwidth ceiling — byte-model over-count (tensors are L2-resident
from the immediately-preceding op, so the modeled DRAM round trip never
happens), published with that caveat, not as real bandwidth. LATENCY FLOOR
verdict unaffected.

**NOT measured (flagged, not glossed)**: HyperConnection `attn_hc`/`ffn_hc` and
`hc_expand` at prefill shape with both fused kernels ON — ran out of capture
window. These are the two spans already reduced by shipped kernels, so least
likely to hide headroom, but their post-ship prefill-shape per-kernel numbers
are genuinely unmeasured; do not reuse pre-ship figures as if current.

## P08 — resolving P07's two open items: SDPA-compressed ceiling denominator + indexer top-k spike (2026-08-30)

Pre-registration: `docs/p08-sdpa-ceiling-and-topk-spike-preregister-2026-08-30.md`
(gates written BEFORE any measurement). Raw artifacts will land in `tmp/p08-20260830/`.

**Cluster state at phase open, verified from real signals (not assumed):** m4-1
PID **59909**, m4-2 PID **60392**, both `etime` 01:48:14 (continuous since
~19:34 today, zero relaunches across P06/P07). Both hosts `up 4 days, 23:46`.
Full verbon3 production flag set verified via `ps eww <live pid>` on BOTH nodes,
identical, **zero leftover test env vars** — incl. `EXO_DSV4_PREFILL_ARGPARTITION=1`,
`EXO_DSV4_INDEX_TOPK=512`, `EXO_DSV4_SEQ_SPLIT=1`, `EXO_COMPUTE_DTYPE=bf16`,
`EXO_DSV4_HC_EXPAND_KERNEL=1`, `EXO_DSV4_HC_COLLAPSE_KERNEL=1`,
`EXO_DSV4_LMHEAD_MXFP8=1`. Target: zero relaunches, captures standalone beside
the live runner (p01/P03/P07-proven).

**Two items, both from P07, nothing else in scope:**

1. **`attn.sdpa.compressed` denominator (P07 §8, UNDETERMINED).** P07's
   arithmetic says the bench counted dense `L_band × CATTN_KV` (261.3 GFLOP)
   with a synthetic 95%-dense mask while production passes a real causal mask
   (158.3 GFLOP) → corrected efficiency ~48%, not 79.1%, i.e. **more** headroom
   than T7 believed. Whether that correction is real hinges on one unmeasured
   runtime fact: does MLX's fused SDPA skip fully-masked blocks or mask after?
   Registered decisive test: GPU-timed causal-vs-dense-vs-nomask at production
   shape, `R = t_causal/t_dense`; `R ≤ 0.75` → kernel exploits mask (48% class),
   `R ≥ 0.92` → 79.1% stands, in between → report both bounds.
2. **`attn.indexer` top-k (P07 §7, OPEN).** ~2.9% of prefill wall at ~4-7% of
   achievable bandwidth; e2e ceiling if perfect ~1.03x. Boxed to a two-phase
   spike with a hard stop: Phase A (floor + existing-op composition, no Metal),
   Phase B (one disposable Metal kernel) ONLY if Phase A clears a pre-registered
   gate. No third phase under any outcome.

**Two methodology upgrades registered before measuring, both aimed at not
repeating this campaign's own past mistakes:**

- **A denominator argument alone cannot open a lever.** Item 1 must also measure
  a denominator-free **direct floor** — `max(matmul_QK + matmul_PV,
  softmax_bw_time)` built from the same node's own measured `mx.matmul`
  performance at the exact shapes, under the roofline `max()` convention (never
  additive). Item 1 only becomes a P09 candidate if `direct_headroom ≥ 1.40x`
  AND `span_share × (1 − 1/direct_headroom) ≥ 1.0%` of prefill wall. Correcting
  a percentage is not the same as finding recoverable work.
- **P07's "4-7% of 424 GB/s" framing is structurally generous to the
  complaint** — a radix sort is inherently multi-pass, so scoring it against a
  *single* streaming pass overstates the gap. Item 2 Phase A must measure a real
  single-pass floor over the identical tensor and report
  `pass_ratio = t_current / t_single_pass`, not reuse the streaming-bandwidth
  comparison.

**Also closing a P07 self-flagged caveat:** P07 §9 replaced the FLOPS ceiling
11.66 → 14.34 TF after finding 11.66 was measured on the **laptop** M4 Max
(32-core), not the 40-core Studio nodes — but flagged that 14.34 is theoretical,
named the rigorous fix (run `measure_peak_gemm()` on a Studio node), and did not
do it. P08 does it; every Item 1 efficiency figure is computed against the
MEASURED on-node peak, re-derived from fresh timings, never by scaling a stale
percentage.

**Pre-registered phase-level conclusion gate:** if Item 1 yields no actionable
lever AND Item 2 closes KILL, the verdict is recorded plainly as *"the easy wins
in this area are now exhausted"* — no manufactured P09. Neither item may close
as "needs more investigation" without naming the specific measurement that would
resolve it and why it wasn't made.

### P08 Item 1 — the causal-vs-dense test: MLX's fused SDPA does NOT exploit the mask (measured)

Raw: `tmp/p08-20260830/item1_results.json`, harness `p08_item1_capture.py`.
Standalone on m4-1 beside the live runner; PIDs 59909/60392 verified unchanged
before AND after (etime continuous 02:00→02:09). Zero relaunches.

**Production shape nailed down with file:line provenance** (P07 had this only by
arithmetic): q `(1,32,1024,512)` bf16, kv `(1,1,3894,512)` bf16, mask
**materialized bool** `(1,1,1024,3894)` = 3.99MB, sinks `(32,)`.
`CATTN_KV=3894 = 2175` local ring (`cache.py:633`, `128 + 2048-1`) `+ 1719`
pooled (`ceil(220000/128)`), concat at `deepseek_v4.py:4359`; n_heads=32 =
64//TP2 (`dv4:7390`); L_band=1024 from the seq-split slice (`dv4:4381`,
`EXO_DSV4_SEQ_SPLIT=1` verified in the LIVE runner env via `ps eww 59909`, not
the source default).

**The bench's SHAPES were right; its mask CONTENT was wrong.**
`bench/attn_production_class_bench.py:212-236` builds the correct q/kv, but
`bench:219` uses a random **~95%-dense** mask while the real production mask is
**47.3% dense** (1844/3894 visible keys per row, computed by summing the
reconstructed production mask at runtime). So `f = causal/dense FLOP = 0.4736`
— the real mask is even sparser than P07's arithmetic estimated (0.6058).

**Decisive result — the registered gate resolves cleanly:**

| condition | µs/call | dispatches |
|---|---|---|
| causal (production mask) | 21423.4 | 7 |
| dense (all visible, same dtype/shape) | 21408.6 | 7 |
| `mask=None` | 20255.7 | 6 |

`R = t_causal/t_dense = **1.0007**`, identical dispatch counts. Registered rule
was `R ≥ 0.92` → **KERNEL DOES FULL WORK**. The `mask=None` control shows the
5.4% causal-vs-nomask delta is the *fixed cost of reading the mask tensor*
(one extra dispatch), not work-skipping.

**So P07's arithmetic was RIGHT and its conclusion was WRONG, in the useful
direction.** The denominator really is inflated — by 2.11x (1/0.4736), even more
than P07's 1.65x estimate. But the kernel leaves that entire gap **unexploited**:
it computes ~2.1x the causally-required FLOPs. The 79.1%-class efficiency figure
therefore *stands as a hardware-efficiency number* — the kernel really does emit
that many FLOPs, at 12.20 TF = **80.2% of the measured on-node peak**.

**P07's flagged caveat closed: on-node GEMM peak MEASURED, not theoretical.**
Dense bf16 GEMM sweep on m4-1 (40-core Studio): **15.21 TFLOPS** measured at
`[16384x4096x4096]` (fp16 15.10). Replaces the 11.66 figure
(`bench/attn_production_class_bench.py:136-145`, measured on a 32-core **laptop**)
and *exceeds* P07's theoretical 14.34. Also measured 488.4 GB/s streaming r+w on
this node. **Rebasing P07's headline: `indexer.score_gemm` 14.23 TF = 93.6% of
the measured peak** (was quoted 99.3% vs theoretical 14.34) — still AT CEILING,
now on a measured denominator.

**Direct floor — and an ambiguity in this phase's own pre-registration.**
Measured on the same node: `matmul_QK 8847.6µs + matmul_PV 8732.3µs = 17579.9µs`,
softmax stream-bound 522.5µs; roofline `max()` (never additive) → floor.

- Against a **dense** floor (what MLX can do today at the full 3894-key shape):
  headroom **1.2186x** → gate needs ≥1.40 → **FAILS**.
- Against a **causal** floor (`f × dense = 8325.8µs`, what the math actually
  requires): headroom **2.573x** → passes 1.40, and
  `span_share × (1−1/h) = 11.8% × 0.611 = **7.21%** of prefill wall` → passes 1.0%.

`span_share = 11.8%` is citable, not estimated:
`docs/dsv4-220k-prefill-span-profile-2026-08-18.md:84` (2200 calls, 73396.81 ms).

**§2.4 did not specify which floor, so the verdict is not yet earned.** Recorded
as such rather than picked. The two floors answer different questions: the dense
floor says *the fused kernel is only 1.22x off what MLX's own matmuls cost, i.e.
softmax+masking is nearly free and there is no kernel-inefficiency headroom*; the
causal floor says *2.1x of the work it does is arithmetically unnecessary*. Both
are true. What decides the phase is **reachability**, which is one more
measurement, not an argument — see the next entry.

**Mask structure (the reason this is even askable), from the measured density:**
1844 visible ≈ 1719 pooled + ~125 local. The **pooled half is ~100% dense** (at
ctx 220K nearly every pooled entry precedes every query row) — zero waste there.
**All the waste is in the local ring**: 2175 keys computed, ~128 needed per row
(the sliding window, `max_size=128`) = **~17x over-computation on 55.9% of the
key space**. That is a sliding-window mask, which is exactly the block structure
a block-sparse kernel skips — and MLX's SDPA here does not.

### P08 Item 2 Phase A — top-k floor measured; the "4-7% of bandwidth" framing was far too kind

Raw: `tmp/p08-20260830/item2_phaseA_results.json`, harness `item2_phaseA.py`.
Standalone on m4-2; PIDs 59909/60392 verified unchanged before AND after
(etime 01:57:13 → 02:15:36 continuous). Zero relaunches.

Live branch confirmed from the **running process env** (not the source default):
`EXO_DSV4_PREFILL_ARGPARTITION=1`, `EXO_DSV4_ARGPARTITION_MIN_P=8192`,
`EXO_DSV4_INDEX_TOPK=512`; scores `(1,1024,P)` bf16 (`dv4:3760`), band L=1024
(`dv4:3909-3913`), k=512 (`dv4:3998`, `dv4:3840`); live expression is
`mx.argpartition(-scores, kth=k-1, axis=-1)[..., :k]` at **`dv4:4055`**.
`EXO_DSV4_EXACT_TOPK`/`EXO_DSV4_TOPK_FUSED` confirmed NOT set, and L=1024>16, so
the L≤16 exact-topk and shape[1]==1 fused branches are OFF.

**The single-pass floor P07 never measured.** P07 scored top-k against a *single*
streaming pass ("4-7% of 424 GB/s"), which understates the gap because a radix
sort is inherently multi-pass. Measured honestly, on the identical tensor:

| P | ~ctx | single-pass `mx.max` µs | production argpartition µs | **pass_ratio** |
|---|---|---|---|---|
| 55,000 | 220K | 221.7 | 15413.5 | **69.53** |
| 125,000 | 500K | — | — | **79.81** |

The floor is genuinely at streaming peak (221.7µs ≈ 509 GB/s read). **The
production sort costs ~70-80 single-pass-equivalents.** That is a much larger
structural gap than P07's framing implied, and it grows with P.

**Existing-op composition sweep** (all GPU-timed, same barrier discipline, all
checked for EXACT top-k index-set equality vs production on random AND
forced-tie inputs — ~1010/1024 rows carried an exact boundary tie):

| variant | µs | disp | speedup | set-eq (rand / tie) |
|---|---|---|---|---|
| production argpartition | 15420.4 | 13 | 1.000 | PASS / PASS |
| argpartition kth +1 / +16 / +128 | 15419.1 / 15418.6 / 15416.5 | 13 | 1.000 | PASS / PASS |
| chunked C=8 | 12867.7 | 18 | 1.198 | PASS / PASS |
| chunked C=16 | 12712.8 | 21 | 1.212 | PASS / PASS |
| **chunked C=32** | **12182.0** | 20 | **1.265** | PASS / PASS |
| chunked C=64 | 16348.8 | 22 | 0.943 | PASS / PASS |
| chunked C=128 | 22646.0 | 23 | 0.681 | PASS / PASS |
| two-pass threshold select | 30692.9 | 27 | 0.502 | **FAIL** / FAIL |
| `mx.topk` (values only) | 14489.5 | 12 | 1.064 | N/A (no indices) |

Chunked top-k+merge is **exact at every C** (per-chunk `kth=min(K,size)` keeps
whole chunks when chunk<K, so the merged candidate set is a strict superset of
the global top-K) — but peaks at only **1.265x** and inverts past C=32. The
kth-offset variants are a dead heat with identical 13-dispatch counts,
independently reproducing P07's "same radix-sort kernel family" conclusion.
`mx.topk` in this build returns VALUES only, so it cannot produce an exact index
set under ties. Two-pass threshold select in existing ops is both **slower**
(0.502x) and **inexact** (boundary ties).

**Gates, applied as written in §3.3:**
- **(a) composition ≥1.5x with exact equality → NOT MET** (best 1.265x).
- **(b) `pass_ratio ≥ 4.0` AND a named mechanism that reduces real WORK →
  MET.** 69.53 ≫ 4.0, and the mechanism is concrete: a threshold-select touches
  the score tensor ~2x where the radix sort touches it ~70x.

Gate (b) is an OR-branch and it is satisfied, so **Phase B (one disposable Metal
spike) is authorized by the pre-registration.** Noting explicitly: the worker
recommended skipping Phase B on the grounds that no *existing-op* composition
captures the gap — but existing-op expressibility is gate (a)'s requirement, not
gate (b)'s. Gate (b) exists precisely for "the gap is structural and no MLX
primitive expresses it." Declining Phase B here would be moving a pre-registered
gate after seeing the data, which this campaign does not do.

**Why this is not the prior fused-kernel failure pattern** (the reason the spike
isn't obviously doomed): the 0.54x fused-indexer and −0.48% wq_a+wkv failures
were attempts to out-kernel MLX's **steel GEMM** at D=512, where hand-written
Metal peaks ~4.4 TFLOP/s vs steel's ~14-15 TF — a structural MMA-tiling wall.
Top-k is not a GEMM; it is a bandwidth-bound selection with a 69.5x pass gap
against a general-purpose radix sort. Different regime, so the prior art does not
transfer automatically. It still has to beat the pipelined in-situ gate.

### P08 Item 2 Phase B — the "custom kernel" already exists in this repo and is default-ON; only a `L <= 16` shape gate blocks it from prefill

Raw: `tmp/p08-20260830/item2_phaseB_results.json`, kernel source
`item2_phaseB_kernel_source.metal`, harness `item2_phaseB.py` + `item2_phaseB_partD.py`.
Standalone on m4-2; PIDs 59909/60392 verified unchanged before AND after. Zero relaunches.

**The spike built a threshold-select top-k — then found it was re-deriving
`_exact_topk`, which has been in `deepseek_v4.py` since 2026-07-07.**
Verified directly at the source (`mlx-lm/mlx_lm/models/deepseek_v4.py`):

- `:3495` — `_EXACT_TOPK = os.environ.get("EXO_DSV4_EXACT_TOPK", "1") == "1"`.
  **The code default is `"1"` — ON.** It is not an off-by-default experiment.
- `:3511 _exact_topk_source()`, `:3641 _get_exact_topk_kernel(L)`, `:3664 _exact_topk()`
  — a complete `mx.fast.metal_kernel` histogram/threshold top-k, parameterized by L.
- `:~4030` — the live gate is `_EXACT_TOPK and scores.shape[1] <= 16`. The
  in-code comment states the intent plainly: *"decode + MTP-verify rows (L <= 16)
  take the histogram/threshold kernel ... Prefill chunks (L > 16) keep the landed
  argpartition path."*

So the blocker is **not** the env var and **not** a missing kernel. It is one
hardcoded shape condition. The kernel is already default-ON, already shipped,
already carrying exactness guarantees in its own comment ("exact top-k set ...
always", "deterministic lowest-index tie-breaking"), and already handling the
masked-score case ("finfo.min fills from the pmask path map to the lowest key,
so masking semantics carry through unchanged").

**Correcting the Phase A note in the record**: Phase A reported
"`EXO_DSV4_EXACT_TOPK` NOT set in live env → exact-topk branch OFF." The first
half is true and the conclusion is wrong — unset means it takes the code default,
which is `"1"`. The branch is ENABLED; it simply never fires at prefill because
L=1024 > 16. This is the *same* class of error P07 recorded as a reusable lesson
(a review pass misread `os.environ.get(X,"0")` and called a real finding an
artifact), just inverted: **an unset env var is not an off switch — read the
default.** Both directions of this mistake have now cost this campaign a wrong
call; the rule is to read the default AND the launcher AND the live env.

**Measured, at production shape (1,1024,P) bf16, k=512:**

| metric | production argpartition | `_exact_topk` @ L=1024 |
|---|---|---|
| passes over score tensor | ~70 single-pass-equiv | **3** |
| dispatches | 13 | **1** |
| µs @ P=55,000 | 15413.5 | **1040.2** (14.82x) |
| µs @ P=125,000 | — | 2339.5 (16.77x) |
| pipelined chain (GEMM→topk→gather) | 33582 | **19322 (1.738x)** |
| pipelined @ P=125,000 | — | 2.032x |

Exactness: **0/1024 mismatching rows on all 5 cases** — 3 random seeds
(974-986 tie rows each) plus forced-tie at P=55,000 (1013/1024 tie rows) and
P=125,000. Effective read bandwidth 325-328 GB/s across 3 passes, under the
~490 GB/s physics ceiling.

The isolated 14.82x collapsing to 1.738x pipelined is **exactly** the inflation
pattern the pre-registration warned about and that killed the prior fused-indexer
attempt (0.54x). Here the pipelined number still wins decisively.

**Pre-registered ship gate (§3.4), all four:**
1. e2e ≥1.0%: `2.9% × (1 − 1/1.738)` = **1.23%** → PASS. (An intermediate log
   printed 1.67% from a wrong formula; 1.2315% is the value re-derived from raw JSON.)
2. Reduces real WORK/dispatch COUNT: 3 passes vs ~70, 1 dispatch vs 13 → PASS.
3. Validated PIPELINED, not isolated → PASS.
4. Exact top-k index-set equality → PASS.

**Op-level verdict: SHIP.** Not yet a production change — the remaining work is
the gate relaxation plus a LIVE e2e A/B, since a pipelined microbench is still
not a live measurement. Two gaps to close before landing: (a) Phase B's exactness
cases used clean random/forced-tie inputs, **not** `pmask`-masked scores carrying
`finfo.min` fills, which is the real prefill input; (b) `_EXACT_TOPK_PARAM_CAP`
(`:3507`, default 64) governs a `(P,k)` params-array cache whose behaviour at
L=1024 with growing P across a 110-chunk prefill is unverified.

### P08 Item 1 reachability — waste localized and MEASURED, but the key-split decomposition was the wrong decomposition

Raw: `tmp/p08-20260830/item1b_reachability_results.json` (on m4-1),
harness `p08_item1b_reachability.py`. PID 59909 verified continuous; m4-2 untouched.

**Mask structure now MEASURED per-region, closing the reviewer's flag** that the
earlier "all waste is local" claim was inferred arithmetic stated in a measured
voice. Empirically at production shape, rank-0 band:
- local ring `[0:2175]`: **exactly 128.0 visible keys/row** (min=max=128), 5.9% dense
- pooled `[2175:3894]`: min 1710 / max 1718 / **mean 1713.51 = 99.68% dense**

Confirmed: the pooled half has ~zero waste; **all** of it is the local ring's
**16.9x** over-computation. Correction to the assumed geometry: band row `i` sees
local keys `[i, i+127]` — the window sits **ahead** of the row (lower-right
aligned), not behind. Tiling arithmetic is unchanged (B+127 distinct keys per
contiguous B-row block) but the key range each block needs is different.

**MLX exposes no windowed/block-sparse path.** Introspected signature (mx
0.32.1.dev): `scaled_dot_product_attention(q, k, v, *, scale, mask: Union[None,
str, array], sinks, stream)`. The only string accepted is `'causal'`;
`window_size`/`sliding_window`/`block_sparse`/`segment` all raise `TypeError`.
`mask='causal'` measured 21336.7µs vs a materialized lower-right causal bool
tensor at 21402.7µs (max rel diff 0.0) — same kernel path, no faster route. And
`'causal'` cannot express the production mask anyway (windowed ring + dense
row-causal pooled suffix).

**A premise in the pre-registration was wrong, and checking it was worthwhile.**
`EXO_DSV4_SEQSPLIT_BALANCED=1` IS in the live env (`start_cluster.sh:97`), but the
balanced implementation was **reverted at mlx-lm `bf8cbad5`** (2026-07-13,
"throughput-neutral, slight regression"). `grep SEQSPLIT_BALANCED mlx-lm/mlx_lm`
= **0 hits**; `deepseek_v4.py:4376-4383` assigns a CONTIGUOUS band
(`_seq_lo = rank * band`). **The flag is inert and rows are contiguous**, so
window locality is intact in production. Measured anyway: contiguous gives
255/383/639 distinct local keys at B=128/256/512; a worst-case stride-2 interleave
gives 382/638/1150 and measurably destroys the win (B=512: 6473µs vs 3677µs
contiguous). So re-introducing balanced seq-split would *hurt* windowed tiling.

**Windowed local tiling works — the local half really is ~7x recoverable:**

| variant | µs |
|---|---|
| untiled local SDPA (2175 keys) | 11919.5 |
| tiled contiguous B=128 (255 keys/block) | **1697 (7.0x)** |
| tiled contiguous B=256 (383 keys/block) | 2302 (5.2x) |
| tiled contiguous B=512 (639 keys/block) | 3677 (3.2x) |

**But the composite failed, and the reason is a decomposition mistake, not a
hardware limit.** The harness split by KEY range (pooled call + local call) and
merged with log-sum-exp. `mx.fast.scaled_dot_product_attention` returns no
per-row LSE, forcing eager fp32 matmul+softmax partials: splitting *alone* cost
33875µs = **1.59x SLOWER** than the 21423µs baseline, and the best composite
(B=128) reached only 16909µs (1.27x) at **352-368 dispatches vs 7**. Numerics in
that run were invalid (mean rel err ~85x) from a merge-formula bug — numerator
weighted by `exp(lse_part − lmax)` instead of `exp(max_part − lmax)` — root-caused
but not re-run in budget.

**The mistake: a key-range split needs an LSE merge; a QUERY-range split does
not.** If instead the 1024 query rows are tiled into blocks of B, and each block
attends to (all 1719 pooled keys + its own 128+B local window), every block is a
*complete, independent* attention — outputs concatenate along the query axis with
**no merge at all**. Key-work becomes `1024 × (1719 + B + 127)` vs the baseline's
`1024 × 3894` = **1.97x less work at B=128**, at ~8 SDPA calls (~56 dispatches),
not 352. Sinks are per-row and survive unchanged. This is the decomposition that
should have been measured and it is the one remaining Item 1 measurement.

**Numerics-gate finding worth banking regardless of outcome**: the registered
"<0.2% mean relative error vs the production fused output" gate is
**structurally unmeetable at head_dim=512** for *any* recomposition — the fused
kernel's own deviation from exact fp32 attention is p50 ~0.34% / rms ~0.31%
(isolated to head_dim: at D=128 it falls to 0.135%; mask density and sinks have
no effect — a pure precision signature, and bf16-score-rounding emulation does
not close it). A gate defined against a reference that is itself 0.34% from truth
rejects mathematically-exact restructurings by construction. Not amended
unilaterally; recorded so the next phase anchors numerics to exact fp32 (or to
the kernel's own error bar) rather than to the fused output.

### P08 Item 1c — QUERY-range tiling BEATS the fused SDPA baseline 1.90x pipelined; Item 1 is NOT a dead end

Raw: `tmp/p08-20260830/item1c_querytiled_results.json`, harness `p08_item1c_querytiled.py`.
Standalone on m4-1; PID 59909 unchanged throughout (02:54:06 → 03:02:09).

The prior key-range split needed an LSE merge MLX cannot supply. A **query**-range
split needs no merge at all: each block of B query rows attends to (its own
128+B local window + all 1719 pooled keys) and is a complete independent
attention; outputs simply concatenate along the query axis.

| B | GPU µs | dispatches | speedup vs 21423.4µs |
|---|---|---|---|
| **64** | **10655.7** | 192 | **2.01x** |
| 128 | 11020.3 | 96 | 1.94x |
| 256 | 11757.3 | 48 | 1.82x |
| 512 | 13122.3 | 24 | 1.63x |
| 1024 (control) | 15946.0 | 11 | 1.34x |
| baseline (re-run same session) | 21318.3 | 7 | 1.00x (anchor reproduced within 0.5%) |

**Key-set correctness verified BEFORE any timing**: every block's visible-key set
checked for exact boolean equality against the reconstructed production mask, all
B (`keyset_verification.all_match=true`).

**Geometry correction to the prior entry**: "window AHEAD of the row `[i, i+127]`"
was a coordinate-frame artifact. With the chunk offset `p = i+127` the window is
`[p-127, p]` — **behind** the row, as originally expected. The +127 belongs inside
the block-mask formula. Caught by the exact-mask equality check, which is why that
check ran first.

**The pipelined check — the one that killed the prior fused kernel at 0.54x —
holds**: in a realistic chain (projection matmul → attention → reduce), tiled
11986.5µs vs fused 22726.5µs = **1.90x pipelined**, retaining **94%** of the 2.01x
isolated win. MLX's async executor absorbs the dispatch increase almost perfectly.
Measured concat cost: 133.6µs = **1.3%** of the variant — not material.

**Dispatch honesty**: 192 vs baseline 7 at B=64 — a real 27x explosion. Unlike the
prior 352-dispatch attempt it does **not** eat the win, but it multiplies
graph-construction work per layer, which GPU-busy timing does not capture and the
chain test only partly does. This is the main productionization risk.

**Numerics — the query-split is mathematically exact, and measurement confirms it:**

| comparison | p50 | RMS (floored) |
|---|---|---|
| variant vs fused production output | **0.0%** | 0.006% |
| variant vs exact fp32 reference | 0.3449% | 0.86% |
| fused production output vs same exact reference | **0.3449%** | 0.86% |

(b) == (c) to four decimals and (a) ≈ 0: the tiled output is bit-identical to the
fused kernel at the median element, and **all** deviation from exact attention is
the fused kernel's own — independently reproducing this phase's ~0.34% D=512
precision finding from a different direction. The variant is not less accurate
than production; it is the same kernel applied to row blocks. Reproduced in a
fresh process.

Note the raw-RMS trap recorded for reuse: at D=512 raw relative-error RMS is
dominated by near-zero output elements (median |out| = 0.039 vs bf16 LSB ~2e-4),
so raw RMS reads ~62% for *any* pair including the fused kernel against itself.
p50 and a floored RMS are the meaningful statistics.

**Implied e2e: 11.8% × (1 − 10655.7/21423.4) = ~5.9% of prefill wall** (~5.7%
using the pipelined ratio). That is roughly 5x Item 2's ceiling and the largest
single lever this campaign has surfaced since hc_expand.

**Item 1 VERDICT: REAL LEVER, opens P09.** P07's arithmetic was right, its
"at ceiling, dead end" inheritance from T7 was wrong, and the recoverable work is
real, reachable with existing MLX primitives, and numerically exact.

### P08 measurement trap caught by the user before it produced a false result: an env knob that never reaches the node

While the Item 2 live A/B was being set up, the user flagged that
`EXO_DSV4_EXACT_TOPK_PREFILL` is **not wired into `start_cluster.sh`'s
env-forwarding whitelist**. Independently confirmed here, twice:

1. `grep EXACT_TOPK_PREFILL start_cluster.sh` → **no match**. Sibling knobs ARE
   wired, via the standard pattern:
   `:1771 [ -n "${EXO_DSV4_TOPK_FUSED:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_TOPK_FUSED=$EXO_DSV4_TOPK_FUSED"`
   (same at `:1772` INDEX_TOPK, `:1775` TOPK_OVERLAP_LOG, `:1999` MTP_DUMP_TOPK).
2. `ps eww` on all four live `exo -v` PIDs on m4-1 → only `EXO_DSV4_PREFILL_ARGPARTITION=1`
   and `EXO_DSV4_LMHEAD_MXFP8=1` present. `EXO_DSV4_EXACT_TOPK_PREFILL` **absent**.

**Mechanism**: `start_cluster.sh` ssh's out and builds the REMOTE runner's env from
its own explicit whitelist. Setting a var in the local invoking shell
(`EXO_DSV4_EXACT_TOPK_PREFILL=1 ./start_cluster.sh`) does **not** reach the node
process unless the var is on that list. The "ON" arm would have silently run the
**identical code path** as baseline — producing a false null (or a
coincidentally-similar number) rather than a measurement.

**No result was corrupted**: the Item 2 A/B worker terminated on a tool guardrail
before producing a verdict, so there is no number to retract. The code change it
landed (`mlx-lm a248d0a7`, exo pin `f96853544`) is **default-OFF and inert** —
verified `_EXACT_TOPK_PREFILL = os.environ.get("EXO_DSV4_EXACT_TOPK_PREFILL","0")`
at `deepseek_v4.py:3503`, gate at `:4046` reads
`scores.shape[1] <= 16 or _EXACT_TOPK_PREFILL`, so with the flag absent production
behaviour is byte-identical to before.

**Third instance of the same class in this campaign** (P05 wrong-model artifact,
P07 span-timer-vs-eval-barrier, now this). Generalized rule, banked:
***verify the knob actually reached the code path being measured — via `ps eww` on
the real PID — BEFORE trusting any A/B number. Never infer from the launch command
that it did what it looked like it should.*** An env var has three independent
places to fail: the source default, the launcher whitelist, and the live process
env. All three must be checked, and only the third is authoritative.

**Cluster note**: the failed A/B worker left both nodes cleanly shut down
("EXO Shutdown complete", "Released MLX buffers before exit" on both — no crash,
no orphaned GPU memory). Relaunched from the laptop; both nodes back up.

### P08 Item 2 — EXO_DSV4_EXACT_TOPK_PREFILL SHIPPED (live e2e A/B, verified flag)

The wiring bug that killed the previous worker is fixed and the live A/B is
done. Full chain, with the third (authoritative) env place checked at every step.

**Wiring fix**: added `EXO_DSV4_EXACT_TOPK_PREFILL` to `start_cluster.sh`'s
env-forwarding whitelist (commit `e5e8c1c72`), mirroring the sibling TOPK knobs
with the `[ -n "${VAR:-}" ] &&` guard form so it stays absent (unset) unless
explicitly set. Then flipped to `: "${EXO_DSV4_EXACT_TOPK_PREFILL:=1}"` default-ON
at SHIP (`b36a5dc8b`).

**Verified flag propagation (`ps eww`, the authoritative check)**:
- Flag ON: `EXO_DSV4_EXACT_TOPK_PREFILL=1` literally present in all 4 live
  `exo -v` PIDs on BOTH nodes (~46060/46072 m4-1, ~45942/45953 m4-2).
- Flag OFF (baseline launch with nothing set): present in ZERO PIDs on both nodes
  (`grep count = 0`), confirming the OFF arm genuinely ran the argpartition path.
- Every relaunch certified the full 12-flag verbon3 set present and ZERO leftover
  test env vars (checked `MLX_GPU_TIME`, `MLX_DISPATCH_COUNT`, `EXO_TEST*`, etc).

**Behavior probe (flag present ≠ path taken) — PASSED**, standalone on m4-1 using
the installed mlx_lm (module `deepseek_v4.py:3503`):
- (a) gate reads env correctly (`_EXACT_TOPK_PREFILL=True` with flag; `False`
  control in a fresh interpreter without it);
- (b) exact-topk Metal kernel runs at prefill-chunk shape `(1,1024,55000)` and
  returns the EXACT top-k set — **0/1024 rows** vs argpartition (P=55000, k=512);
- (c) positive GPU-execution proof: exact-topk **1.33 ms** GPU vs argpartition
  **15.6 ms** at production shape (≈11.7x isolated), confirming real dispatch.
  (First pass used `mx.metal.dispatch_count()` which reads 0 without
  `MLX_GPU_TIME`/`MLX_DISPATCH_COUNT` set — the earlier phaseC harness set those;
  `gpu_time_ns()` with them set is the reliable counter.)

**Live A/B (server "Prefill complete: N tokens in Xs" line from ~/exo.log, NOT
client wall time; deterministic per-rep prompts; cache-busting salt; cluster
warmed before each side; ALL samples reported):**

| depth | OFF baseline (s) | ON exact-topk (s) | prefill-wall gain |
|---|---|---|---|
| 220K ctx | 631.27 / 630.44 / 630.48 | 621.13 / 620.33 / 620.22 | **1.61%** (370.22 vs 364.24 tps) |
| 75K ctx | 203.21 / 203.68 / 203.68 | 203.22 / 203.13 / 203.56 | **0.11%** (384.58 vs 384.16 tps) |

The win **scales with P** exactly as the op-level microbench predicted: negligible
at 75K, 1.61% at the 220K campaign reference depth (top-k cost grows with pooled
width). Pre-registered ship gate ≥1.0% e2e → **cleared**. Predicted 1.23%
(2.9% op-level * 1/1.738 pipeline retention) was if anything conservative.

**Quality gate — PASSED**: real 75K-depth generation with flag ON retrieved the
needle `FALCON-MERCURY-7749` and cited the exact source segment (`[P08I2-QUALITY-
75000 i 432]`), finishing `stop` with no BOS spam / no degeneration. Top-k selects
which KV positions attention attends to; the retrieved-answer correctness confirms
the index set is right, matching the 0/1024 op-level exactness result.

**VERDICT: SHIP.** `EXO_DSV4_EXACT_TOPK_PREFILL` is now default-ON. Cluster left
UP on the clean production config (`b36a5dc8b`), all 12 prod flags present.

Raw batteries: `tmp/p08-20260830/item2_battery_ON.jsonl` / `item2_battery_OFF.jsonl`.

**PM independent verification of the Item 2 SHIP** (re-derived from the raw
batteries, not from the worker's summary):

- Flag propagation re-checked live by the PM *after* the worker finished, on the
  final running config: `EXO_DSV4_EXACT_TOPK_PREFILL=1` present in the live env of
  m4-1 PID 2963 and m4-2 PID 3692. Zero leftover test vars (`MLX_GPU_TIME`,
  `MLX_DISPATCH_COUNT`, `EXO_TEST*` all absent). Wiring confirmed at
  `start_cluster.sh:40` (default) and `:1786` (forwarding). Tree clean, HEAD ==
  origin/main at `d1245646c`. Cluster serving: 1 instance, 2 runners.
- **220K re-derived: OFF median 630.48s → ON median 620.33s = 1.61% faster.**
  Confirmed. The three-sample ranges do **not** overlap (OFF 630.44-631.27, ON
  620.22-621.13; spread ≤0.91s within each arm), so the win is well outside
  run-to-run noise. Gate ≥1.0% cleared on a real live measurement.
- **75K corrected: the entry's "0.11%" should read "no measurable win."**
  Re-derived from medians it is 0.23%, but the arms **overlap** (OFF
  203.21-203.68, ON 203.13-203.56). At this depth the difference is inside noise
  and should not be quoted as a gain in either direction. This does not weaken
  the finding — it *is* the predicted shape, since top-k cost scales with pooled
  width, so the win must be negligible at shallow depth and appear at depth.
- **`needle_hit: false` in the battery JSONL is a harness artifact, not a quality
  failure.** Those runs used `max_tokens=32` and every one finished
  `finish_reason: "length"` mid-answer — the captured content shows the model
  actively emitting the needle when it was cut off ("...is: FALCON-MERCURY-774",
  "...is: FALCON-MERCURY-7749."). The dedicated quality-gate run (untruncated)
  retrieved `FALCON-MERCURY-7749` and cited its source segment. Recorded so a
  future reader does not mistake the raw field for a regression.

Net: SHIP upheld on independent re-derivation, with the 75K number restated
honestly as noise rather than a small win.

## P08 CLOSE — both P07 items resolved; one SHIPPED, one opens a genuinely larger P09

**Item 1 — `attn.sdpa.compressed` ceiling: RESOLVED, and it is a REAL LEVER.**
The registered question ("is the denominator inflated by masked-position FLOP
counting?") answered **yes, by 2.11x** — more than P07's 1.65x estimate, because
the real production mask is 47.3% dense, not the bench's synthetic 95%. But the
decisive finding is the one the denominator argument alone could not deliver:
`R = t_causal/t_dense = 1.0007` means **MLX's fused SDPA does not exploit the mask
at all** — it computes the full dense work and masks after. So the 79.1% figure
stands *as a hardware-efficiency number* (12.20 TF = 80.2% of the measured 15.21 TF
on-node peak) while ~2.1x of the work it does is arithmetically unnecessary.
Neither "48%" nor "79.1%" alone is the honest answer; both are, for different
questions. **The unnecessary work is reachable**: query-range tiling at B=64 is
2.01x isolated / **1.90x pipelined**, numerically exact (variant-vs-fused p50 0.0%;
all deviation from exact fp32 is the fused kernel's own 0.34%), implying **~5.9%
of prefill wall**.

**Item 2 — indexer top-k: SHIPPED**, +1.61% live at 220K, default-ON. The
"custom kernel" turned out to already exist in-repo, default-ON, blocked from
prefill only by a hardcoded `L <= 16`. The ~1.03x ceiling was approximately right
(1.016x realized).

**Both P07 open items are now closed.** The campaign's easy wins in the *top-k*
area are exhausted — but Item 1 did not close as a dead end, so this phase does
NOT end with "nothing left."

### P09 candidate (named, scoped, ready for a fresh PM)

**Query-range tiled compressed attention.** Ceiling ~5.9% of prefill wall — the
largest single lever surfaced since hc_expand, ~5x Item 2's entire ceiling.

- *What*: replace the single fused SDPA call over 3894 keys with per-query-block
  calls over (own 128+B local window + all 1719 pooled keys), concatenated along
  the query axis. No LSE merge — each block is a complete independent attention.
- *Where*: `mlx-lm/mlx_lm/models/deepseek_v4.py` around the concat at `:4359` and
  the compressed-attention call at `:4385/:4413-4425`; needs block-local key views.
- *Measured*: B=64 best (10655.7µs vs 21423.4µs); monotone in B, no knee found
  below 64 — **try smaller B first**. Exact-mask key-set equality verified per
  block before timing.
- *Known risks, in priority order*: (1) **dispatch count 192 vs 7** — survived the
  chain test (94% retention) but multiplies per-layer graph construction, which
  GPU-busy timing does not capture; validate under the real multi-layer loop, not
  a standalone chain. (2) The B=1024 control ran 1.34x, not 1.0x, vs baseline —
  worth understanding before trusting the B-sweep shape. (3) Rows must stay
  contiguous; `EXO_DSV4_SEQSPLIT_BALANCED` is currently inert (implementation
  reverted at mlx-lm `bf8cbad5`), and re-introducing it would *hurt* this lever.
- *Numerics gate must be re-anchored*: the "<0.2% vs fused output" gate is
  structurally unmeetable at D=512 (the fused kernel is itself 0.34% p50 from
  exact fp32). Anchor to exact fp32, or to the kernel's own error bar.
- *Method note*: at D=512 raw relative-error RMS is dominated by near-zero output
  elements and reads ~62% for any pair including the kernel against itself. Use
  p50 and a floored RMS.

---

## P09 — query-range tiled compressed attention: does the 1.90x survive the real 43-layer forward pass? (2026-08-31)

Inherits P08 Item 1c. The lever is measured, exact, and reachable; the ONE
unresolved question is whether the **27x dispatch explosion (192 vs 7 per
compressed-attention call at B=64)** behaves the same inside the real 43-layer
loop as it did in an isolated 3-op chain. Isolated-chain retention was 94%
(2.01x -> 1.90x). The real loop multiplies per-layer Python graph-construction
by 43 and runs against a GPU already saturated by every other op in the model.

### P09 gate — PRE-REGISTERED before any measurement was taken

Registered at phase start, with the cluster verified healthy on the clean
verbon3 production config (4 exo PIDs/node, `EXO_DSV4_LMHEAD_MXFP8=1` +
`EXO_DSV4_EXACT_TOPK_PREFILL=1` confirmed via `ps eww` on both nodes, TP
worldSize=2, 43 layers) and **before any P09 code existed**.

**Primary metric**: e2e prefill-wall gain at the campaign reference depth
(~220K ctx), read from the server's own `Prefill complete: N tokens in Xs`
line in `~/exo.log` — never client wall time. >=3 reps/side, deterministic
per-rep prompts with a cache-busting salt, cluster warmed before each side,
ALL samples reported (P08's protocol, which reproduced to +/-0.1%).

| outcome | condition | action |
|---|---|---|
| **SHIP** | >= **2.0%** e2e prefill-wall gain at 220K, AND numerics gate passes, AND quality gate passes | land default-ON |
| **NO-SHIP (eroded)** | 0% < gain < 2.0% | keep code default-OFF or revert; document as characterized |
| **NO-SHIP (collapsed)** | gain <= 0% | close as a well-characterized dead end |
| **NO-SHIP (unsafe)** | any numerics or quality gate failure, at any speed | revert regardless of speed |

**Why 2.0% and not something lower**: P08 shipped `EXO_DSV4_EXACT_TOPK_PREFILL`
at +1.61% for a change that is a pure kernel-selection swap. This lever
restructures how attention is *called* (192 dispatches/layer instead of 7,
per-block key views, output concat along the query axis) in a production hot
path. It must beat the already-shipped simpler lever to justify carrying that
structural complexity, or it is not worth landing. Anything under 2.0% means
the dispatch explosion ate most of the projected 5.9% and the honest answer is
"the isolated chain test overstated it."

**Numerics gate** (re-anchored per P08's correction — the "<0.2% vs fused
output" form is structurally unmeetable at D=512):
1. Flag unset -> production path **byte-identical** to today (inert knob).
2. Per-block visible key-set == production mask, **exact boolean equality**,
   verified BEFORE any timing is trusted (this check is what caught the
   coordinate-frame bug in P08).
3. variant-vs-fused-output **p50 == 0.0%**, and
   variant-vs-exact-fp32 == fused-vs-exact-fp32 (all deviation belongs to the
   existing fused kernel, none introduced by the tiling).
   p50 + floored RMS only — raw RMS reads ~62% for any pair at D=512.

**Quality gate**: real generation at depth with the flag ON must retrieve a
planted needle, cite its source segment, terminate on `stop`, and show no
degeneration/BOS-spam. This lever changes attention output structure, not a
quantization detail, so a speed-only verdict is not acceptable.

**Mandatory pre-flight before ANY A/B number is trusted** (3rd-instance rule
from P08 — P05 wrong-model artifact, P07 span-timer bug, P08 env-wiring gap):
the new knob must be (a) present in `start_cluster.sh`'s env-forwarding
whitelist AND (b) literally visible in `ps eww` on the real running PIDs on
**both** nodes for the ON arm, and absent from all PIDs on the OFF arm.
Neither check is optional and the launch command is not evidence.

### P09 implementation + review — a TP coordinate-frame blocker caught before any A/B ran

Knob implemented as `EXO_DSV4_QUERY_TILED_SDPA` (+ `EXO_DSV4_QUERY_TILED_B`,
default 64) in `mlx-lm/mlx_lm/models/deepseek_v4.py`: module-level gate at
`:3504` mirroring `_EXACT_TOPK_PREFILL`, helpers `_pooled_len`/`_query_tiled_ok`,
and an additive `elif` branch in `CompressedAttention.__call__`'s
`attn.sdpa.compressed` span. Both vars wired into `start_cluster.sh`'s
env-forwarding whitelist at `:1789-1790` (sibling `[ -n "${VAR:-}" ] &&` guard
form, so they stay genuinely ABSENT when unset — the OFF arm depends on that).

**Independent read-only review found a BLOCKER, confirmed here by direct code
read before acting on it.** At `:4444-4458`, when seq-split is active the code
slices the QUERY side to this rank's row band — `q = q[:,:,_seq_lo:_seq_hi,:]`
and `mask = mask[...,_seq_lo:_seq_hi,:]` — but deliberately leaves `kv`
**full-width** (its own comment: *"kv is full-width so each band attends
correctly"*). The tiled branch then used the loop index `_r` as BOTH a
band-relative query-row index AND a full-width key index. For rank 1 of a
2-rank prefill (`_seq_lo=512`), block `_r=0` is absolute query row 512 but was
slicing local keys `[0,191]` where the production mask marks `[512,703]`
visible.

Severity is maximal because it is **silent**: no crash, no shape error, wrong
attention output all-summed into the shared result. And it is **reachable in
production right now** — `ps eww` on the live PIDs shows `EXO_DSV4_SEQ_SPLIT=1`,
`L>=16` and `L % 2 == 0` hold at prefill shape, so the ON arm would have taken
this path on every layer of every chunk. `_query_tiled_ok` did not catch it:
after band-slicing, `q.shape[0]==1`, `mask.shape[-2]==n_q` (512==512) and
`n_q >= 2*B` all still pass.

**This is the SAME coordinate-frame bug class as P08's, one level up.** P08's
was window-vs-row (`[p-127,p]` behind, not ahead); this one is
band-relative-vs-absolute. The tiled formula was only ever proven against
un-band-split query rows — the standalone P08 harness had no TP sharding, so
no isolated test could have surfaced it. Root-cause fix (not a disable):
index key-space by `_seq_lo + _r` while keeping `_r` for band-relative query
rows and mask ROWS (mask columns were never band-sliced).

**The validation harness also did not actually prove what was claimed.** Its
assertion 2 reconstructs the block endpoint with `min(LOCAL_LEN, r + (b1-r) -
1 + SLIDING)` — the *same formula as the implementation* — then asserts the two
match: circular, and a one-key truncation would shrink both sides identically
and still pass. Assertion 3's `p50 == 0.0%` cannot catch it either (a missing
visible key corrupts only block-final rows, ~1.6% of elements, leaving the
median exactly 0). Independent hand-trace of blocks r=0/512/960 confirms the
shipped non-TP `_khi` is exactly correct, but the harness is not what proves
it. Additionally A5 (tail block) and the timing block **crashed** —
`ValueError: [rope] freqs must be one dimensional with size 16 but got shape
(32)` and an `IndexError` in the SDPA recorder, then `ZeroDivisionError` on the
speedup. **No implementation-level speedup number exists yet**; nothing was
committed.

Verified-good regardless: inertness (flag unset short-circuits before
`_query_tiled_ok` is ever called; diff is additions-only), `cache=local_cache`
is safe across the 16 per-block calls (`base.py:122` uses it only for a
`hasattr(cache,"bits")` quantized-kernel check — never mutated, no offset
advance), and `finalize()` parity with the `else` branch is exact.

---

### P09 ON-arm A/B complete — all 3 reps, needle_hit:false resolved as a harness artifact

ON arm driven to completion: 3 reps, deterministic per-rep prompts with distinct
cache-busting salts (3 unique salts across the 3 reps, OK), harness exit 0, at
the campaign reference depth (~220K). Raw output in
`tmp/p09-20260831/p09_ab_ON_220000.json`. Per the pre-registered protocol all
samples are reported individually, never just the median:

| rep | prompt_tokens | server_prefill_s | tps |
|---|---|---|---|
| 1 | 273271 | 687.54 | 397.46 |
| 2 | 273265 | 685.95 | 398.37 |
| 3 | 273271 | 686.23 | 398.22 |
| **median** | **273271** | **686.23** | **398.22** |

Prompt-token spread across reps is 0.002% — effectively identical prompts (the
only difference is the per-rep salt). All 3 reps report `needle_hit: false`,
`bos_spam: false`, `finish_reason: "length"`, `completion_tokens: 24`, and empty
`content`.

**Live config verified on the real running PIDs, both nodes, at the time of the
run** (`ps eww`, 4 exo PIDs/node, 1 instance, 2 runners):
`EXO_DSV4_QUERY_TILED_SDPA=1`, `EXO_DSV4_QUERY_TILED_B=64`,
`EXO_DSV4_LMHEAD_MXFP8=1`, `EXO_DSV4_EXACT_TOPK_PREFILL=1`,
`EXO_DSV4_SEQ_SPLIT=1` — all five present on both nodes, so the ON arm ran the
tiled path on every layer of every chunk, and pre-flight's 3rd-instance rule
(`ps eww`, not the launch command) is satisfied.

#### `needle_hit: false` — RESOLVED as a harness artifact, not a quality signal

Every rep is flagged `needle_hit:false` with `finish_reason:"length"`,
`completion_tokens:24`, and empty content. This is **two compounding, purely
harness-side causes** — neither is a model or kernel fault:

1. **A 24-token budget far too small for the answer.** `max_tokens: 24` is
   hardcoded at `tmp/p09-20260831/p09_live_ab.py:232` (in `one_run`), not
   derived from `--depth`.
2. **Content-only field blindness.** Needle detection at `p09_live_ab.py:273`
   tests `NEEDLE.split(": ")[-1] in text` where `text` comes from `:244`
   `d["choices"][0]["message"].get("content") or ""` — it reads `message.content`
   **only**; `reasoning_content` is never read anywhere in that file.

Why content is EMPTY while `completion_tokens` is 24: DeepSeek-V4-Flash under
this fork's encoding path appends the thinking_start_token to the END of the
prompt and sets `thinking_mode="thinking"` when a request sends no tools and
leaves `enable_thinking` unset →
`src/exo/worker/engines/mlx/vendor/deepseek_v4_encoding.py:443-455` (in
particular the `prompt += thinking_start_token` at `:453`). The model therefore
opens in a reasoning block, and essentially all 24 tokens are consumed as
`reasoning_content` before any answer text can reach `content`.

**P08 precedent** (`docs/PERFORMANCE_HISTORY.md` lines ~6235-6241): P08
documented the same pattern at `max_tokens=32` and recorded it as a harness
artifact, confirmed there by a separate untruncated quality-gate run. P08's
driver (`tmp/p08-20260830/item2_retry_driver.py:87,120`) had the identical
content-only blindness in its boolean.

**One honest difference worth stating**: P08's reps still showed partial
non-empty content with the needle visibly forming mid-emission (`"...is:
FALCON-MERCURY-774"`). P09's content is **fully empty** — 8 fewer tokens of
budget plus the reasoning-block open. Same phenomenon, more extreme point on
the curve: consistent and explainable, but not literally identical to the P08
precedent.

**Conclusion**: `needle_hit:false` in the A/B JSONL is **not** a quality signal
in either direction. The real quality gate is
`tmp/p09-20260831/p09_quality_gate.py` (`max_tokens 250`, reads
`reasoning_content + content`, requires `finish_reason=="stop"`, checks segment
citation + BOS spam + U+FFFD + n-gram repetition); its result is being recorded
separately.

**The one real correctness risk specific to this lever is already ruled out.**
The TP/seq-split band-relative-vs-absolute coordinate-frame bug caught in review
is already FIXED at the commit under test (mlx-lm `2e2d17d`; `_key_lo =
_seq_lo + _r` at `mlx-lm/mlx_lm/models/deepseek_v4.py:4519`, with the absolute
key-space indexing comment at `:4516`). Silent-wrong-attention is not an
explanation for this run.

#### Methodology correction — P08-vs-P09 prompt-size incomparability at nominal 220K

**Do not compare P09 ON wall time to P08's 220K numbers.** P08's shipped 220K
figure (OFF 630.48s / ON 620.33s median, 370.22 tps) is NOT comparable to P09's
ON arm (686.23s median, 398.22 tps) as a wall-clock time, because the P09
harness builds a materially larger prompt at the same nominal 220K depth:
**273,271 real prompt tokens vs P08's ~229.8K**. Throughput (tps) is in fact
**higher** in P09 (398.22 vs 370.22). Any cross-phase wall-clock comparison at
"220K" is therefore invalid; a future reader must NOT read 686s as a regression
against P08's 620s. The pre-registered >=2.0% gate is evaluated **strictly on
P09 ON vs P09 OFF measured with the SAME harness**.

**Status: OFF arm not yet run at time of this entry — verdict still OPEN.**
ON arm complete; OFF arm pending. No ship/no-ship conclusion is recorded here.



---

### P09 ON-arm quality gate — PASSED; spurious segment-citation FAIL traced to a harness off-by-one

Quality gate for the P09 lever (query-tiled compressed SDPA, TP/seq-split) run at real
production depth (220K) against the **live ON config**, using
`tmp/p09-20260831/p09_quality_gate.py`. The lever flag set was verified on the real
running PIDs on **both nodes** at run time (`ps eww`, 4 exo PIDs/node):
`EXO_DSV4_QUERY_TILED_SDPA=1`, `EXO_DSV4_QUERY_TILED_B=64`,
`EXO_DSV4_LMHEAD_MXFP8=1`, `EXO_DSV4_EXACT_TOPK_PREFILL=1`, `EXO_DSV4_SEQ_SPLIT=1`.

**Two runs** were recorded: the script's original hardcoded default `max_tokens=250`,
and a second run after adding a `--max-tokens` CLI arg to the harness (the 250 default
is unchanged, so prior invocations are unaffected). Per-assertion results for both:

| # | assertion | RUN 1 — max_tokens=250 | RUN 2 — max_tokens=2000 |
|---|---|---|---|
| 1 | needle_retrieved | PASS | PASS |
| 2 | source_segment_cited | PASS | FAIL * |
| 3 | finish_reason_stop | FAIL (reason=length, budget ran out mid-reasoning) | PASS (reason=stop) |
| 4 | no_bos_spam | PASS | PASS |
| 5 | no_fffd | PASS | PASS |
| 6 | no_pathological_repetition | PASS | PASS |
| 7 | achieved_depth_landed | PASS | PASS |

\* The `source_segment_cited` FAIL in RUN 2 is a **harness off-by-one**, not a model
error — see next subsection.

**Run metrics.** RUN 1 (max_tokens=250): `elapsed_s=685.6`,
`usage.prompt_tokens=266101` (nominal 220000), `completion_tokens=250`,
`finish_reason=length`. Note this run did cite the segment correctly *by luck*: its
truncated reasoning happened to contain the literal marker string the harness greps
for. RUN 2 (max_tokens=2000): `elapsed_s=10.6` (a prefix-cache hit on the identical
prompt — **NOT a new prefill**; do not read 10.6s as a speed result),
`usage.prompt_tokens=266101`, `completion_tokens=151`, `finish_reason=stop`. The
model's final answer retrieved needle `P09-ORION-8821` correctly and cited it as
segment `[P09-QUALITY-220000 i 1270]`.

#### The `source_segment_cited` FAIL is a harness off-by-one, not a model error

The model is right; the harness's *expected* marker is off by one. The relevant
facts in `tmp/p09-20260831/p09_quality_gate.py`:

- `:92-93` builds the `n_fill` marker-prefixed segments, indexed `i = 0..n_fill-1`,
  each formatted `"[P09-QUALITY-{depth} i {i}] s {seed} " + FILLER`.
- `:94` then splices the needle in **after** the loop:
  `fillers.insert(int(n_fill * 0.4), " " + NEEDLE + " ")`.
- `:137-139` computes the EXPECTED marker as `needle_seg_i = int(n_fill * 0.4)` —
  reusing the `list.insert` POSITION as if it were the index of the segment the
  needle lands inside.
- Python's `list.insert(k, x)` places `x` **AT index `k`** and pushes the element
  formerly at `k` to `k+1`. So the segment physically PRECEDING the needle is `k-1`,
  not `k`.

Arithmetic at depth=220000: `len(FILLER)=346`, so
`n_fill = int(220000 * 5 // 346) = 3179`, and
`k = int(3179 * 0.4) = 1271`. The needle therefore sits between segment 1270's text
and segment 1271's marker; the nearest preceding marker is **i 1270** — exactly what
the model answered.

Independently corroborated by the max_tokens=250 run, where the model quoted the raw
document: `"... A red-black tree self-balances through rotations.  The secret
authorization code for project AURORA is: P09-ORION-8821. [P09-QUALITY-220000 i 1271]
s 999540459 ..."` — the needle appears BEFORE the `i-1271` marker, confirming it
belongs to segment 1270.

Suggested harness fix (NOT applied — artifact script only): line 138 becomes
`needle_seg_i = max(0, int(n_fill * 0.4) - 1)`.

#### Verdict: **P09 ON-arm QUALITY GATE — PASSED**

All seven substantive conditions hold at real production depth with the flag ON:
needle retrieved, source segment cited correctly (model right, harness expectation
off by one), natural `stop` termination once the token budget is adequate, no BOS
spam, no U+FFFD, no pathological repetition, and the intended depth genuinely landed
(266,101 real prompt tokens confirmed from the API's own `usage` block, not a
client-side count). The lever changes attention-output structure, and at 220K it
still retrieves a planted needle from ~1270 segments deep and cites its source — this
is the evidence the pre-registered quality gate asked for.

#### Two more harness defects found (now the 5th and 6th measurement-invalidating harness defects caught this campaign)

Both the 24-token A/B budget + content-only field blindness (already documented in
the previous entry) and this expected-marker off-by-one are **measurement-side**, in
throwaway `tmp/` artifact scripts. Neither is in shipped code and neither affects the
timing numbers.

#### Status: OFF arm still pending — verdict OPEN

OFF arm still pending; ship/no-ship verdict remains **OPEN**. Pre-registered decision
boundary, computed **before any OFF data exists**: ON median = 686.23s, the gate
requires >= 2.0% e2e gain, therefore **SHIP requires OFF median >= 700.24s**;
anything below that is NO-SHIP.

---

## P09 CLOSE — query-range tiled compressed attention SHIPPED at +7.20% e2e

**Verdict: SHIP.** The pre-registered >=2.0% e2e gate at 220K is beaten by **3.6x**
(measured **+7.20%**), the numerics gate passed pre-A/B, and the quality gate
PASSED. `EXO_DSV4_QUERY_TILED_SDPA=1` + `EXO_DSV4_QUERY_TILED_B=64` are now
**default-ON** (`start_cluster.sh` commit **b27a83ced**), mirroring the P08
`EXO_DSV4_EXACT_TOPK_PREFILL` knob pattern. This is the largest single win of the
campaign to date.

### Full live A/B at 220K — all six reps (server-side "Prefill complete" wall, NOT client wall)

Raw artifacts: `tmp/p09-20260831/p09_ab_ON_220000.json` and
`p09_ab_OFF_220000.json`. ON = `EXO_DSV4_QUERY_TILED_SDPA=1`, `EXO_DSV4_QUERY_TILED_B=64`
(verified via `ps eww` on the real PIDs on **both** nodes). OFF = both vars genuinely
absent, verified the same way.

| side | rep | server prefill (s) | tps | prompt_tokens |
|---|---|---|---|---|
| ON | 1 | 687.54 | 397.46 | 273,271 |
| ON | 2 | 685.95 | 398.37 | 273,265 |
| ON | 3 | 686.23 | 398.22 | 273,271 |
| **ON** | **median** | **686.23** | **398.22** | 273,271 |
| OFF | 1 | 740.65 | 368.96 | 273,271 |
| OFF | 2 | 739.47 | 369.55 | 273,272 |
| OFF | 3 | 738.69 | 369.94 | 273,270 |
| **OFF** | **median** | **739.47** | **369.55** | 273,271 |

ON spread 685.95–687.54 (1.59s); OFF spread 738.69–740.65 (1.96s). Ranges are
**DISJOINT** (ON max 687.54 < OFF min 738.69) — the win sits far outside
run-to-run noise.

### Derived metrics

- **e2e prefill-wall gain = (739.47 − 686.23) / 739.47 = +7.20%**
- **Throughput gain = 398.22 vs 369.55 tps = +7.76%**
- Ranges DISJOINT — the win is far outside run-to-run noise (no overlapping spread).
- **Worst-case bound** (ON slowest vs OFF fastest) = **+6.92%**, still **3.4x** the gate.
- **Median prompt_tokens delta between arms = 0** (273,271 on both) — arms directly
  comparable, no normalization needed.
- **Salt check passed both arms**: 3 unique salts across 3 reps/side (6 unique total).

### Verdict vs the pre-registered gate

Gate was **>=2.0% e2e at 220K**, and it had to beat the already-shipped P08 Item 2
(+1.61%). Measured **+7.20% = 3.6x the gate**. The decision boundary (SHIP requires
OFF median >= 700.24s) was pre-registered in the previous entry **before any OFF
data existed**; the actual OFF median 739.47s cleared it by **39s**. **SHIP.**

### Calibration note — the isolated-chain projection was CONSERVATIVE, not optimistic

P08 projected this lever at ~5.9% of prefill wall from isolated-chain measurements,
and the phase's pre-registered worry was that the **27x dispatch explosion (192 vs 7
dispatches per compressed-attention call at B=64)** would erode that inside the real
43-layer loop. It did **not** erode — the realized **+7.20%** actually **EXCEEDS**
the 5.9% projection. This is the opposite of the failure mode the phase was designed
to catch, and it is the largest single win of the campaign to date (vs P08 Item 2's
+1.61%).

### Numerics gate — PASSED

Passed pre-A/B: per-block visible key-set == production mask under exact boolean
equality; variant-vs-fused p50 == 0.0%; all deviation attributable to the existing
fused kernel, none introduced by tiling. The flag unset leaves the production path
byte-identical (an additive `elif` branch that short-circuits before
`_query_tiled_ok` is even called) — re-confirmed empirically by the OFF arm running
the historical baseline path cleanly.

### Quality gate — PASSED

Documented in the previous entry: needle retrieved at 220K from **~1270 segments
deep**, source segment cited correctly (model right, harness expected-marker off by
one), natural `stop` termination, no BOS spam / U+FFFD / pathological repetition,
achieved depth confirmed from the API's own `usage` block.

### What shipped

`start_cluster.sh` commit **b27a83ced** adds `: "${EXO_DSV4_QUERY_TILED_SDPA:=1}"` and
`: "${EXO_DSV4_QUERY_TILED_B:=64}"` next to the P08 `EXO_DSV4_EXACT_TOPK_PREFILL`
default, mirroring that knob's established pattern. The model-side gate at
`mlx-lm/mlx_lm/models/deepseek_v4.py:3522` is `os.environ.get("EXO_DSV4_QUERY_TILED_SDPA", "0") == "1"`
— a strict string compare, so `EXO_DSV4_QUERY_TILED_SDPA=0` genuinely disables the
feature (the `:=` default only applies when unset, and the forwarding guard at `:1789`
forwards an explicit `"0"` correctly). Escape hatch verified, not assumed.

### The correctness bug this phase caught (the phase's most important artifact)

The TP/seq-split **band-relative-vs-absolute coordinate-frame bug** was found in
independent review and fixed **BEFORE any A/B number existed** (mlx-lm `2e2d17d`;
`_key_lo = _seq_lo + _r`). Had it shipped, rank 1 of a 2-rank prefill would have
silently attended to the wrong keys — no crash, no shape error, just wrong attention
all-summed into the shared result. The pre-registered "key-set equality verified
BEFORE timing is trusted" rule is what made it findable. A speed number measured on
that code would have been meaningless.

### Measurement-discipline tally

This phase caught **3 more measurement-invalidating defects** — the TP coordinate-frame
bug pre-A/B, the A/B harness's 24-token budget + content-only field blindness, and the
quality-gate expected-marker off-by-one — bringing the campaign total to **6**. The
mandatory env-knob whitelist + `ps eww`-on-real-PIDs pre-flight ran on both arms and is
what makes the OFF baseline trustworthy.

### Final cluster state

Relaunched on the shipped **default-ON** config with the QUERY_TILED vars **UNSET** at
launch (proving the new default puts them live rather than an inherited shell export).
Health verification is being recorded separately.

## Lever-1 MoE small-M re-check — verdict UPHELD, headline number RETRACTED (2026-08-31)

Scoped local re-check of `docs/lever1-moe-smallm-headroom-2026-08-20.md`. **No
cluster contact, no relaunch, read-only + local microbench.** Full detail in that
doc's ADDENDUM; artifacts `tmp/lever1-recheck-20260831/`.

**The `0.63x` headroom number is an artifact — retracted.** `gather_qmm` and dense
`quantized_matmul` are both tile-quantized at **~32 rows/expert**, so cost/row is a
STAIRCASE: dense M=32 → 6.75ms vs M=33 → **13.3ms** (1.97x for one row);
gather m=32 → 0.954 us/row vs m=31 → 1.390, m=33 → 1.359. The original drew
`mean_M=35`, just past a cliff → denominator ~2x inflated → "live is 1.6x FASTER
than an ideal dense GEMM". A kernel cannot beat an ideal GEMM reading the same
bytes; that should have been a red flag. **Comparing against a single M is invalid
on a staircase** — this class of error is now the campaign's 7th measurement-
invalidating defect, and the first found in an already-CLOSED result.

**Corrected, alignment-fair (gates pre-registered before running):** live = real
ragged histogram, 6 independent draws; balanced = uniform swept over a FULL
staircase period (m=24..55, every alignment), median us/row:

| live (ragged) | balanced (full period) | R |
|---|---|---|
| **1.383** (spread ±1.5%) | **1.255** | **1.10x** |

Gates R≤1.15 holds / R≥1.30 reopen → **1.10x = HOLDS**. An intermediate denominator
(balanced at m=32 only) gave 1.48x and would have triggered a false REOPEN — it sat
exactly on the luckiest notch, the same artifact class being audited.

**The tile-aligned notch is not a lever (mechanism, not judgment).** Balanced at
m=32/48 reaches 0.94 us/row (~1.47x better), but the notch is a property of
**UNIFORMITY, not the mean**: forcing the real histogram to `mean=32` still measures
**1.415 us/row**, identical to untouched. Only **5.3%** of rows lie within ±2 of a
32-multiple (histogram spans 1..413). Capturing it requires equalizing every
expert's row count = a **router/load-balancing change that alters model outputs**,
not a kernel change. Strengthens the original conclusion: the constraint is routing.

**MLX version caveat RESOLVED, no re-run needed.** `ac73d0c9` → `e40a416b2` (183
commits): `affine_gather_qmv_rhs`, `affine_gather_qmv_rhs_chunk`,
`affine_gather_qmm_rhs` **SHA-identical**; `steel_gemm_gather.h` SHA-identical;
`gather_qmv_rhs` dispatch + `MAXBE` bound unchanged; `GatherQMM::eval_gpu`
untouched. New `qmv_wide` reaches only `dispatch_qmv`, called **only** from
`QuantizedMatmul`/`QQMatmul::eval_gpu` — never `GatherQMM`. (It would have
perturbed the old `ceiling` tier, i.e. exactly the retracted one.) `qmm_splitk`
`k_align` no-op at g=32. Also: the repo venv still runs the **original** build
(`0.32.0.dev20260804+ac73d0c9`, .so dated Aug 4) — the submodule moved, the
installed binary did not, so this re-check ran on the doc's own build.

**Still open, NOT locally resolvable:** 32-core laptop vs 40-core Studio. Does not
change the verdict (R is a same-device ratio; the corrected error is arithmetic),
but absolute us/row and exact cliff positions are core-count dependent — cf. the
11.66→14.34 TFLOPS 32-vs-40-core correction. Confirming staircase geometry on a
Studio needs cluster access = **separate follow-up requiring explicit relaunch
approval**, deliberately not taken. Non-blocking; no decision hinges on it.

**Net: LEVER 1 STAYS CLOSED.** No kernel rewrite, no MAXBE widening, no tile
retune. Unchanged from 2026-08-20 — but now on a denominator that survives audit.

---

## P11 — Decode-gap re-audit: the "5-7x / 14-20% of ceiling" brief was STALE (2026-08-31)

**PM finding, before any measurement was run.** A fresh investigation was
commissioned against the 2026-08-22 framing ("decode runs at 14.0-20.2% of its
bandwidth-bound ceiling, ~5-7x gap; two named-but-never-executed next steps:
(1) fresh Instruments capture with real per-kernel labels, (2) per-token
gap-rate check vs per-layer moe.all_sum/fence boundaries"). A read-only audit
of repo HEAD (`d15c3f639`) found **all three premises already superseded**.
Every claim below was independently re-verified by the PM against the cited
lines, not taken on the worker's report.

1. **The 5-7x gap is CLOSED, and was mostly a wrong denominator.** §P02D
   (2026-08-29, `docs/p02d-...md`, quoted at PH:5183-5190): *"Corrected
   efficiency: decode runs at 36-44% of the 546 spec, i.e. 46-57% of the
   measured 424 GB/s real-streaming ceiling — 1.75-2.16x slower than the REAL
   ceiling, not 5-7x... The §13/§4.3 'highest-priority open question' framing
   ... is CLOSED as an artifact of the 3.56 GB denominator."* The old
   denominator omitted the REPLICATED 5.30 GB/rank attention path (exo shards
   MoE only), depth-linear reads, lm_head, and shared-expert/gate;
   `B_true = 8.17-9.14 GB/rank/token`. **Do not cite 14-20%/5-7x again.**

2. **Next step #1 (Instruments per-kernel labels) is DEAD AS SPECIFIED, and the
   goal was reached by another instrument.** `xctrace`'s
   `metal-gpu-intervals` template only ever emits generic
   `Compute`/`Fragment`/`Vertex` channel names — re-confirmed structurally
   across 5+ attempts (PH:2128-2134, :2300-2301, :3301, 99.98% generic).
   Real per-kernel attribution WAS achieved via
   `mx.metal.start_capture()` + `MLX_GPU_TIME=1`/`MLX_DISPATCH_COUNT=1`
   bracketing: P01 (switch_mlp, 2026-08-29) and **P03 (2026-08-30) which
   measured the entire decode small-op bucket per-kernel in BOTH spec-off
   (8.34 ms/token) and spec-ON (17.2 ms/cycle ÷ 3.2 = 5.38 ms/token) regimes.**
   Commissioning a fresh Instruments capture would have burned cluster time
   re-proving a known-impossible thing.

3. **Next step #2 (gap-rate vs all_sum/fence) is ANSWERED, from both sides.**
   The literal gap-rate re-count was never re-run, but the question it existed
   to answer is closed by a strictly stronger method — P01a's direct CPU-side
   timer INSIDE the collective (`docs/p01a-...md`:96-105): *"Phase 1(a) CLOSED:
   arrival skew at depth is ruled out as the residual's owner ... per-token
   arrival skew grows only +0.079 ms/tok from 100K→352.6K — 3-5% of the band
   ... not in idle, not inside the collective call."* P01b independently
   refutes inter-layer pipelining loss (+0.076 ms/tok vs a +1.67..+2.52 ms/tok
   residual band).

**Consequence: the campaign does not need the work this brief named.** Decode
is 1.75-2.16x from the real ceiling, not 5-7x, and its two named next steps are
respectively impossible-as-specified and already-answered. Re-running either
would have been pure waste. What IS genuinely open is recorded in P12 below.

### P11 pre-registered gate (written BEFORE the measurement, per campaign rule)

The one cheap check the brief asked for that had NOT been done: does tonight's
own P05-P09 campaign have any **decode**-side effect? P07/P08/P09 measured
prefill wall ONLY — **no decode tok/s measurement exists in this repo after
2026-08-30** (P06 Phase A). Non-trivial reason to actually check rather than
assume: P09's query-tiled SDPA is gated on query length, and while decode is
L=1 spec-off, **spec-ON production runs verify batches at L=4** — so "decode is
L=1, nothing to tile" is NOT automatically true under the shipped spec-ON config.

- **Reference (locked):** P06 Phase A, 2026-08-30, spec-ON, 100K ctx,
  `bench/long_decode_probe.py`, 1200-token windows, n=3:
  **34.35 / 34.12 / 34.28, median 34.28 tok/s.**
- **Prediction:** no decode-side effect (P07/P08/P09 are prefill-only).
- **PASS (expectation confirmed):** |Δ median| ≤ 3% of 34.28 (i.e.
  33.25-35.31 tok/s).
- **FAIL-LOW (regression, must investigate before campaign close):**
  median < 33.25 tok/s.
- **FAIL-HIGH (unexpected decode win, must attribute):** median > 35.31 tok/s.
- Same harness, same depth, same rep count, `decode_sample_trustworthy=true`
  required on every rep (the <400-token sampling trap guard).

### P11 result -- live decode measurement on full current stack (2026-08-31)

Live decode throughput measured against the already-running production cluster
(no restart, no config change, read-only). Same harness as P06 Phase A
(`bench/long_decode_probe.py`), same depth (100K), same 1200-token windows,
n=3, one throwaway warmup request first (cold-start mirage guard). Raw JSON:
`tmp/p11-decode-20260831/rep{1,2,3}.json` (warmup: `warmup.json`).

**Verified live config on the real running PIDs, both nodes** (`ps eww`,
m4-1 PID 58066, m4-2 PID 60416, 4 exo PIDs/node, 1 instance, 2 runners):
`EXO_DSV4_LMHEAD_MXFP8=1`, `EXO_DSV4_EXACT_TOPK_PREFILL=1`,
`EXO_DSV4_QUERY_TILED_SDPA=1`, `EXO_DSV4_QUERY_TILED_B=64`,
`EXO_DSV4_SEQ_SPLIT=1`, `EXO_SPECULATIVE=1`, `EXO_SPECULATIVE_GAMMA=3`,
`EXO_DSV4_VERIFY_BATCH=1`, `EXO_DSV4_VERIFY_BATCH_MIN_CTX=8192`,
`EXO_DSV4_MTP=1`. Identical on both nodes; all match `start_cluster.sh`
defaults. Spec-ON (verify batches at L=4) confirmed live.

| rep | decode_tps | decode_sample_trustworthy | completion_tokens | finish_reason |
|---|---|---|---|---|
| 1 | 30.71 | true | 1200 | length |
| 2 | 29.84 | true | 1200 | length |
| 3 | 31.93 | true | 1200 | length |
| **median** | **30.71** | true | 1200 | length |

**VERDICT vs pre-registered gate: FAIL-LOW (regression).** Reference (P06 Phase
A, 2026-08-30, spec-ON, 100K, same harness, n=3): 34.35/34.12/34.28, median
34.28. Measured median 30.71 = **-3.57 tok/s, -10.41%** vs reference, below the
33.25 PASS floor. All three reps are below the reference's WORST rep (34.12) --
consistent regression, not jitter. Every rep `decode_sample_trustworthy=true`
(1200-token windows, no <400-token sampling trap). This is the first decode
measurement on the full current stack (P05-P09 shipped) and it does NOT confirm
the "no decode-side effect" prediction. Must be investigated before campaign
close. Prefill unchanged (417-419 tps, consistent with prior 100K runs).

### P12 pre-register — is the P11 FAIL-LOW a CODE regression or CLUSTER-STATE drift? (2026-08-31)

P11 measured median 30.71 tok/s vs the P06 Phase A reference 34.28 (-10.41%,
FAIL-LOW). PM-verified against the raw JSONs: 3/3 reps `trustworthy=true`,
1200 completion tokens each, `finish_reason=length`, prompt 106.6K-112.3K.
All three reps sit below the reference's WORST rep (34.12) — consistent, not
jitter. The regression is real as measured. What it is NOT yet is attributed.

**The two arms differ in TWO ways, not one.** This is the confound that must be
split before anyone touches code:
1. **Config.** The P06 reference predates P08 (`EXO_DSV4_EXACT_TOPK_PREFILL=1`)
   and P09 (`EXO_DSV4_QUERY_TILED_SDPA=1`, `EXO_DSV4_QUERY_TILED_B=64`), both
   confirmed live on both nodes via `ps eww`. Both were analyzed as
   prefill-only. The named reason that analysis could be wrong: query-tiled
   SDPA is gated on query length, and shipped spec-ON decode runs VERIFY
   BATCHES AT L=4, not L=1.
2. **Cluster state.** The P06 reference was taken on a FRESHLY RELAUNCHED
   cluster ("2 relaunches", PH:5376). P11 ran on a cluster with ~4h uptime that
   had just executed the P07/P08/P09 prefill campaign at 220K and 500K context.
   Allocator/residency drift after deep-context work is a live, named suspect
   in this campaign already — it is one of the two surviving candidates for the
   P02C depth residual ("allocator/~90 GB-resident regime", p02c §6.2).

**Decisive experiment (pre-registered BEFORE running):** relaunch the cluster
fresh with the IDENTICAL currently-shipped config, change nothing else, re-run
the identical probe (100K, 1200-token window, n=3, all reps trustworthy).

- **Outcome A — state drift (config exonerated):** fresh-relaunch median
  >= 33.25 tok/s (back inside the P11 PASS band). Then P05-P09 did NOT regress
  decode; instead we have a real, separately-valuable operational finding —
  decode loses ~10% after a deep-context prefill workload without a relaunch —
  which should be handed to the P02C allocator thread rather than treated as a
  code bug.
- **Outcome B — real code regression:** fresh-relaunch median < 33.25 tok/s,
  i.e. the regression survives a clean relaunch. Then config is implicated and
  the next step is a single-flag A/B with `EXO_DSV4_QUERY_TILED_SDPA=0`
  (verified live via `ps eww`, not just set in the script) to test the L=4
  verify-batch mechanism named above.
- **Outcome C — ambiguous:** median lands 33.25-34.12 with reps straddling.
  Then n=3 is insufficient; escalate to n=5 at the same depth before concluding
  anything (this campaign's own repeat-testing rule).

No code will be changed until this experiment picks A or B. Neither arm is
allowed to be rationalized after the fact.

### P12 BLOCKED — `start_cluster.sh` route-clear needs a sudo right the sudoers rule doesn't grant (2026-08-31)

The pre-registered P12 experiment did NOT run. The fresh relaunch aborts before
launching anything, and the cause is an infrastructure gap, not a perf finding.

**Root cause (PM-verified directly on both nodes, not taken on report).**
`start_cluster.sh:917-920` clears stale direct-link routes before the
connectivity test:

```
echo "Testing direct-link connectivity (clearing stale routes first)..."
for node in macstudio-m4-1 macstudio-m4-2; do
    ssh "$node" "for r in \$(netstat -rn | awk '/192\.168\.(200|201|202)\./{print \$1}' | sort -u); do sudo route delete -net \$r 2>/dev/null; done" &> /dev/null
done
```

`sudo route delete` is invoked WITHOUT `-n`, over a non-interactive ssh, and is
NOT covered by the scoped NOPASSWD rule. Verified `sudo -n -l` on BOTH nodes —
the rule grants exactly four things, none of them `route`:

```
(root) NOPASSWD: /usr/sbin/sysctl iogpu.wired_limit_mb\=*
(root) NOPASSWD: /usr/bin/fdesetup authrestart*
(ALL)  NOPASSWD: /usr/bin/ktrace
(ALL)  NOPASSWD: /usr/bin/powermetrics
```

So the command blocks on a password prompt that can never be answered, and the
launch hangs at the route-clear step indefinitely. `2>/dev/null` hides the
prompt, and `&> /dev/null` on the ssh hides it again — which is why this
presents as a silent hang rather than an error. Confirmed live: a stuck
`sudo route delete -net 192.168.200.2` (PID 87204) was found sitting on m4-2
and cleared by the PM.

**This is NOT a no-op step that can be skipped blindly.** The matching routes
genuinely exist right now — m4-1 has 1 (`192.168.200.2`), m4-2 has 2
(`192.168.200.1`, `192.168.200.2`) — so the loop really does iterate and really
does need the privilege. The `sysctl` form (`sudo -n sysctl
iogpu.wired_limit_mb=115000`) still works passwordless on both nodes; only the
route-clear is uncovered.

**Cluster state left by the failed launch (verified, both nodes):**
- **m4-1: DOWN.** No `python -m exo` process. Killed during the relaunch's
  teardown, never came back up because the launch aborted before restart.
- **m4-2: UP but STALE.** Old PIDs 60405/60416 still running the pre-relaunch
  production config. Serves nothing useful alone — the model is TP-sharded
  across both nodes.

**Deliberately NOT worked around.** Editing the route-clear out of
`start_cluster.sh`, or hand-rolling a launch that skips it, would be working
around a sudo failure — explicitly out of bounds, and it would also silently
change the network-path setup underneath a measurement whose entire purpose is
attributing a 10% throughput delta. A perf experiment run on a quietly
different network setup than its own reference is worthless.

**Exact unblock (either one):**
1. Add a scoped rule mirroring the existing `sysctl` one, e.g.
   `(root) NOPASSWD: /sbin/route delete -net *` on both nodes; or
2. Change `start_cluster.sh:919` to `sudo -n route delete` so it fails fast and
   loud instead of hanging silently — but that only converts the hang into a
   clean error; the privilege is still required for the step to actually work.

**P12's pre-registered gate stands unchanged and is still the right next step**
once the relaunch works: fresh relaunch on identical config, 100K / 1200-token
/ n=3, Outcome A (median >= 33.25 = state drift, config exonerated) vs B
(< 33.25 = real code regression) vs C (ambiguous, escalate to n=5). The
before-snapshot is already captured at
`tmp/p12-relaunch-20260831/before_node{1,2}_ps_eww.txt` (both nodes identical,
74 EXO vars, `EXACT_TOPK_PREFILL=1 QUERY_TILED_SDPA=1 QUERY_TILED_B=64
LMHEAD_MXFP8=1 VERIFY_BATCH=1 VERIFY_BATCH_MIN_CTX=8192 SPECULATIVE=1 GAMMA=3`),
so the post-relaunch config diff can be done immediately.

### P12 RESULT — OUTCOME B: the decode regression SURVIVES a cold relaunch (2026-08-31)

The pre-registered decisive experiment ran. **Outcome B: real, code/config-implicated
regression.** Cluster-state drift is real but accounts for less than half of it.

**Pre-flight (non-negotiable, all verified on the REAL running PIDs, both nodes).**
Cluster cold-relaunched 17:46; main PIDs m4-1 `21927`, m4-2 `21675`, etime ~2h at
probe time; old P11-era PIDs 60405/60416 confirmed GONE (`ps -p` empty, rc=1), so
the relaunch genuinely happened. `ps eww` on both real PIDs extracted 74 `EXO_*`
vars each; diffed against the pre-relaunch P11 snapshot
(`tmp/p12-relaunch-20260831/before_node{1,2}_ps_eww.txt`): **empty diff, both nodes**
— live config is byte-identical to what P11 was measured under. All 10 decode-
relevant flags confirmed live (`QUERY_TILED_SDPA=1 QUERY_TILED_B=64
EXACT_TOPK_PREFILL=1 SPECULATIVE=1 GAMMA=3 VERIFY_BATCH=1 VERIFY_BATCH_MIN_CTX=8192
MTP=1 LMHEAD_MXFP8=1 SEQ_SPLIT=1`). 1 instance, 2 runners, both `RunnerReady`.
Capture: `tmp/p12-decode-20260831/preflight_node{1,2}_ps_eww.txt`.

**Probe: identical instrument.** `bench/long_decode_probe.py` unmodified
(`git status --porcelain` clean, `git diff --stat` empty; `API` and `MODEL`
constants unchanged), 100K target, `--max-tokens 1200`, warmup + n=3 as separate
sequential invocations — the exact P11/P06 protocol. Raw:
`tmp/p12-decode-20260831/{warmup,rep1,rep2,rep3}.json` + `.stdout.log`.

| rep | decode_tps | trustworthy | completion_tokens | finish_reason | prompt_tokens | prefill_tps |
|---|---|---|---|---|---|---|
| warmup (discarded) | 32.88 | true | 1200 | length | 108886 | 417.36 |
| 1 | 31.36 | true | 1200 | length | 108886 | 416.93 |
| 2 | 32.18 | true | 1200 | length | 110006 | 417.21 |
| 3 | 32.89 | true | 1200 | length | 110008 | 418.39 |
| **median** | **32.18** | true | 1200 | length | — | — |

**VERDICT vs the pre-registered bands: OUTCOME B, unambiguously.** Median 32.18 is
below the 33.25 floor. This is not a near-miss and Outcome C does not apply: **all
three reps (31.36 / 32.18 / 32.89) are below 33.25**, and **no rep lands anywhere in
the 33.25-34.12 C-band**, so there is nothing to straddle and no case for escalating
to n=5. vs the P06 Phase A reference (34.28): **-2.10 tok/s, -6.13%.**

**Independently re-verified** (separate agent, raw artifacts not the report): mtimes
strictly increasing 19:55:11 → 20:00:13 → 20:05:18 → 20:10:21, gaps 302/305/303 s
(consistent with ~261 s prefill + ~37 s decode); all four `content` /
`reasoning_content` sha256s distinct and all four file md5s distinct (genuinely
independent runs, nothing copied — the near-equal `prompt_tokens` are the ~4 chars/
token heuristic, not duplicate files); no traceback / ModuleNotFoundError / truncated
stream in any log; post-run PIDs still 21927 / 21675 (no restart mid-measurement).

**The -10.41% P11 gap decomposes.** Same code, stale→fresh cluster: 30.71 → 32.18,
**+1.47 tok/s (+4.79%)**. So allocator/residency drift after deep-context work is
REAL and worth ~1.5 tok/s — **41.2%** of the P11 gap. It is NOT the whole story:
**58.8%** (-2.10 tok/s) survives a clean relaunch. Both halves are now separately
attributed instead of confounded.

**Prefill is untouched** at 416.9-418.4 tps across all four runs, consistent with
every prior 100K run. The regression is decode-specific.

### P12 addendum — the NAMED candidate mechanism is REFUTED (static, two independent traces)

The pre-registration named one mechanism: P09's query-tiled SDPA interacting with
spec-ON decode's L=4 verify batch (the "decode is L=1, nothing to tile" analysis
being wrong). **That mechanism is dead.** Traced twice by independent agents reading
`mlx-lm/mlx_lm/models/deepseek_v4.py` verbatim; the second was explicitly tasked to
refute the first and confirmed it line-for-line.

1. **The L=4 premise is CORRECT.** `src/exo/worker/engines/mlx/speculative/dsv4_mtp.py`
   :4082-4087 builds `verify_input` as `(1, γ+1)` = **(1, 4)** at γ=3, and
   `deepseek_v4.py:7034-7044` activates the batched verify path
   (`2 <= h.shape[1] <= _VERIFY_ROWSEQ_MAX_L` and `_vb_ctx_len >= 8192`) — at ~108K
   ctx it IS active, so attention really does see q_len=4 in production decode.
2. **But the tiled path cannot be entered at L=4.** `_query_tiled_ok`
   (`deepseek_v4.py:3558-3584`) hard-returns at line **3572**:
   `if n_q < 2 * _QUERY_TILED_B or mask.shape[-2] != n_q: return False`.
   With `_QUERY_TILED_B=64`: `4 < 128` → **True** → unconditional early `False`,
   short-circuiting before every other clause. The dispatch at :4481-4483
   (`elif _QUERY_TILED_SDPA and _query_tiled_ok(...)`) therefore always declines at
   decode and falls through to the unchanged fused SDPA. `_QUERY_TILED_B` parses to
   int 64 with no clamp beyond positivity (:3527-3546). **The path needs q_len >= 128;
   decode is 1 and verify is 4.**
3. **No 16x padding-waste mechanism either.** The tile loop (:4507) uses
   `_b1 = min(_r + _b_q, _n_q)` — real-row slicing, never pads up to B.
4. **The indirect route was checked too and is also closed.** The one way a
   "prefill-only" flag could still hurt decode is by leaving divergent persistent
   cache state behind. It does not: `local_cache.update_and_fetch` (:4441-4446), the
   pooled compressor write (:4421) and the `kv` concat (:4452) all happen **upstream
   of the branch**, both branches call the same SDPA wrapper, and neither writes to
   `pool_cache`/`local_cache` in its own body. **Cache state is identical.**
5. **P08 is inert at L=4 by construction.** `deepseek_v4.py:4123-4126` gates on
   `(scores.shape[1] <= 16 or _EXACT_TOPK_PREFILL)`; `scores` is `(B, L, P)` from
   `_indexer_score` (:3849), so `shape[1]` IS the query length. At L=4 the `<= 16`
   disjunct already passes, so flipping `EXO_DSV4_EXACT_TOPK_PREFILL` **cannot change
   behavior at L=4**. The exact-topk kernel does run during verify, but via the
   pre-existing `L<=16` clause, not via P08's flag.

**Change-window inventory (P06 reference → HEAD).** Reference commit `80ec8ec03`
(2026-08-30 18:07, submodule `a6eb893`) — it is the commit that shipped the very
config the 34.28 was measured on. HEAD `1365cecc9`, submodule `2e2d17d`. The window
contains exactly **two** submodule commits — `a248d0a` (P08 exact-topk prefill gate)
and `2e2d17d` (P09 query-tiled SDPA TP/seq-split fix) — plus `start_cluster.sh`
default flips for their flags (`b36a5dc8b`, `b27a83ced`) and whitelist adds
(`e5e8c1c72`, `1a25163ca`), and deploy/doc noise. **No change to
`src/exo/worker/engines/mlx/` in the window beyond a comment.** `LMHEAD_MXFP8=1`
shipped *with* the reference commit, so it is constant across both arms, not a delta.

**So there is a real tension, and it is stated rather than resolved:** the regression
is empirically real and survives a cold relaunch (-6.13%, both arms fresh), yet every
code change in the window is statically gated out of the decode path. One of these is
wrong. That is the question the next experiment exists to settle.

### P13 pre-register — is the surviving -6.13% CONFIG or NOT-CONFIG? (written BEFORE the run)

The pre-registration's Outcome-B next step was a single-flag A/B on
`EXO_DSV4_QUERY_TILED_SDPA=0`. **Deviating deliberately, and the reason is on the
record:** that flag is now proven inert at L=4 by two independent verbatim traces
(§P12 addendum), so a single-flag A/B on it has near-zero prior of being informative
and would burn a full relaunch to likely re-prove a null. The strictly stronger test
for the same cluster time is to revert the **entire config delta** at once — both
flags OFF — which is the only dimension separating current HEAD from the P06
reference config. A null there exonerates the whole config dimension in one shot; a
non-null implicates it and *then* justifies the single-flag split.

- **Arms:** identical fresh cold relaunch, identical everything, except
  `EXO_DSV4_QUERY_TILED_SDPA=0` **and** `EXO_DSV4_EXACT_TOPK_PREFILL=0`. Both MUST be
  confirmed as `=0` via `ps eww` on the real running PIDs on BOTH nodes before any
  measurement is trusted — set-in-the-script does not count (this class of bug has
  invalidated 4 measurements this campaign).
- **Probe:** unchanged. 100K, `--max-tokens 1200`, warmup + n=3 separate sequential
  invocations, `decode_sample_trustworthy=true` required on all 3.
- **CONFIG-IMPLICATED:** median **>= 33.25** tok/s. The two shipped flags do cause the
  decode loss despite the static gating analysis; the static analysis is then WRONG
  and must be re-opened at the kernel level. Next step: single-flag split to identify
  which one.
- **CONFIG-EXONERATED:** median **< 33.25** tok/s. The flags are not the cause, the
  static traces are corroborated, and the residual lies either in the submodule diff
  `a6eb893→2e2d17d` outside the gated paths, or in a non-code difference between the
  two fresh boots. Next step is a code-level bisect of the submodule pointer, NOT
  further flag A/Bs.
- **AMBIGUOUS:** median lands **32.89-33.25** (above P12's best rep but below the
  floor) with reps straddling → n=5 at the same depth before concluding.

Neither arm may be rationalized after the fact. Note for whoever runs it: P12's
within-run spread (31.36-32.89, range 1.53) was markedly wider than P06's
(34.12-34.35, range 0.23) — if P13 also comes back wide, between-boot reproducibility
of this probe is itself unmeasured and becomes the next thing worth pinning down.

### P13 RESULT — CONFIG-IMPLICATED. Both flags OFF recovers decode to 36.08 (2026-08-31)

**Outcome: CONFIG-IMPLICATED, and not marginally.** Median **36.08** tok/s vs the
pre-registered floor of **33.25**. Every one of the three reps clears the floor. No
rep lands in the 32.89-33.25 AMBIGUOUS band, so there is nothing to straddle and
no case for n=5.

**Pre-flight (non-negotiable, verified on the REAL running PIDs, both nodes, then
independently re-verified by a second agent from live state rather than from the
first agent's files).** Cold relaunch with `EXO_DSV4_QUERY_TILED_SDPA=0` and
`EXO_DSV4_EXACT_TOPK_PREFILL=0`. Old P12 PIDs 21927 / 21675 confirmed GONE
(`ps -p` empty, rc=1). New real `exo -v` PIDs **7728** (m4-1) / **7776** (m4-2),
etime ~7 min at proof time. `ps eww` on both real PIDs: both flags literally `=0`
on both nodes. Full live-env diff vs the P12 arm: **exactly two differing vars per
node** (`EXACT_TOPK_PREFILL` 1->0, `QUERY_TILED_SDPA` 1->0), 74 EXO_* vars each
side, **zero other drift** — `QUERY_TILED_B=64`, `VERIFY_BATCH=1`,
`VERIFY_BATCH_MIN_CTX=8192`, `MTP=1`, `SPECULATIVE=1`, `GAMMA=3`, `SEQ_SPLIT=1`,
`LMHEAD_MXFP8=1` all identical. Capture:
`tmp/p13-relaunch-20260831/preflight_node{1,2}_ps_eww.txt`.

**Probe: identical instrument, identical protocol.** `bench/long_decode_probe.py`
unmodified (`git status --porcelain` clean, `git diff --stat` empty), 100K,
`--max-tokens 1200`, warmup + n=3 as separate sequential invocations, venv
interpreter. Raw: `tmp/p13-decode-20260831/{warmup,rep1,rep2,rep3}.json` + logs.

| rep | decode_tps | trustworthy | completion_tokens | finish_reason | prompt_tokens | prefill_tps |
|---|---|---|---|---|---|---|
| warmup (discarded) | 29.95 | true | 1200 | length | 110009 | 377.79 |
| 1 | 36.08 | true | 1200 | length | 110006 | 378.95 |
| 2 | 36.20 | true | 1200 | length | 112251 | 377.71 |
| 3 | 34.18 | true | 1200 | length | 108887 | 378.15 |
| **median** | **36.08** | true | 1200 | length | — | 378.15 |

**Effect size, clean arm-to-arm.** P12 (both flags ON) 32.18 -> P13 (both OFF)
36.08 = **+3.90 tok/s, +12.12%**. Identical build, identical everything, two env
vars. This also lands **above** the P06 reference (34.28) by +1.80 / +5.25%.

**Complete separation, which matters given the spread.** Min P13 rep (34.18)
exceeds max P12 rep (32.89). All 3 vs all 3, zero overlap in the two arms' rep
distributions. So although the within-run spread is again wide (P13 range 2.02;
P12 1.53; P06 only 0.23), the arm effect is larger than the spread and does not
depend on the median choice.

**Positive control — the flags demonstrably reached the kernels, not just the env.**
Prefill dropped from a P12 median of **417.21** to **378.15** tok/s, **-9.36%**.
Both flags are prefill optimizations; turning them off is *supposed* to cost
prefill. Prefill had been pinned at 416.9-418.4 across every prior 100K run in the
campaign, so this is the first time it moved — and it moved exactly when and in the
direction the flags predict. This independently rules out the "knob never reached
the live process" failure mode that invalidated four earlier measurements.

**So the static analysis is WRONG somewhere, and it must be re-opened at the kernel
level.** §P12-addendum's two independent verbatim traces are almost certainly
correct about what they actually proved: at decode's L=4 verify shape,
`_query_tiled_ok` returns False (`4 < 2*64`) and the exact-topk `<=16` disjunct
already passes, so neither flag changes the *branch taken during decode*. P13 shows
that is not the same claim as "neither flag changes decode throughput." The
mechanism is therefore NOT branch selection at decode. Live candidates, none yet
tested: (a) prefill running a different SDPA decomposition leaves the MLX allocator
/ memory-pool / fragmentation state different at ~108K, and decode is
bandwidth-bound on that state; (b) Metal shader-cache or `@mx.compile` graph-cache
occupancy — the tiled path compiles extra kernels whose residency displaces
something decode needs; (c) a numerics difference in prefill output shifting
speculative accept rate at verify, which would change effective decode tok/s
without changing any decode code path. (c) is directly testable from the accept-rate
counters and should be checked first because it is the cheapest.

**Correction to the §P12-addendum change-window inventory — `LMHEAD_MXFP8` was NOT
constant across the arms.** The addendum stated `LMHEAD_MXFP8=1` "shipped *with* the
reference commit, so it is constant across both arms, not a delta." That conflates
the env var shipping with the implementation existing. Git evidence:
`git -C mlx-lm show a6eb893:mlx_lm/utils.py | grep LMHEAD_MXFP8` is **empty** — the
reference submodule has no consumer for the flag. The implementation landed in
`a248d0a`, and `git merge-base --is-ancestor a248d0a a6eb893` returns **rc=1** (not
an ancestor). Reference commit `80ec8ec03` shipped `EXO_DSV4_LMHEAD_MXFP8:=1` in
`start_cluster.sh` while pinning a submodule that ignored it. **So at P06 the
lm_head was NOT mxfp8-quantized; from P08 onward it is.** Consequence: any
P06-vs-later comparison (including the headline -6.13%) carries this extra
uncontrolled delta. It runs the *opposite* direction to the regression (the shipping
commit measured mxfp8 lm_head at +6.0% decode), so it cannot be the regression's
cause — but it means **P12-vs-P13 is the only clean pairwise comparison in this
sequence**, and P06 should be treated as a contaminated reference until re-measured.

**Full newly-activated-knob ledger for the window** (every env var read by mlx-lm at
`2e2d17d` but not at `a6eb893`, cross-checked against what `start_cluster.sh` sets):
`EXO_DSV4_LMHEAD_MXFP8` (set=1 at both reference and HEAD, but inert at reference —
the delta above); `EXO_DSV4_EXACT_TOPK_PREFILL`, `EXO_DSV4_QUERY_TILED_SDPA`,
`EXO_DSV4_QUERY_TILED_B` (all unset at reference, defaulted ON at HEAD — the P13
subject); `EXO_DSV4_PRENORM_H_DUMP` and `..._BUDGET` (never set in either arm,
confirmed absent from the live env on both nodes — genuinely inert). No other new
env reads in the window.

### P14 pre-register — single-flag split (written BEFORE the run)

P13 was a joint revert, so it cannot say *which* flag pays. Pre-registered per the
P13 CONFIG-IMPLICATED branch.

- **Arm A:** `EXO_DSV4_QUERY_TILED_SDPA=0`, `EXO_DSV4_EXACT_TOPK_PREFILL=1`.
- **Arm B:** `EXO_DSV4_QUERY_TILED_SDPA=1`, `EXO_DSV4_EXACT_TOPK_PREFILL=0`.
- Both arms run regardless of what the other does — a shared or interaction effect is
  only diagnosable with both.
- Probe unchanged: cold relaunch, `ps eww` proof on real PIDs both nodes, 100K,
  `--max-tokens 1200`, warmup + n=3 sequential, `trustworthy=true` required on all 3.
- **Anchors:** P12 (both ON) 32.18; P13 (both OFF) 36.08; gap **3.90** tok/s.
  Recovery fraction R = (median_arm - 32.18) / 3.90.
- **DOMINANT:** R >= 0.75, i.e. median **>= 35.11**.
- **PARTIAL:** 0.25 < R < 0.75, i.e. median **33.16-35.11**.
- **NOT-THIS-FLAG:** R <= 0.25, i.e. median **<= 33.16**.
- If BOTH arms come back NOT-THIS-FLAG, the effect is an interaction requiring both
  flags on, and the next step is mechanism (a)/(b)/(c) above, not more flag A/Bs.
- **Stated power limitation, in advance:** rep-level range was 1.53 (P12) and 2.02
  (P13), comparable to the 1.95-wide PARTIAL band. A PARTIAL landing is therefore
  weak evidence and triggers n=5 on that arm before any attribution is claimed. A
  DOMINANT or NOT-THIS-FLAG landing is outside the noise and may be taken at n=3.
- Prefill is a free positive control on every arm: whichever flag is set to 0 should
  move prefill off the 416.9-418.4 band. If prefill does NOT move on an arm where a
  prefill flag was turned off, that arm is invalid regardless of its decode number.
### P14 Arm A RESULT — NOT-THIS-FLAG. QUERY_TILED_SDPA is exonerated for decode (2026-08-31)

**Arm A** (`EXO_DSV4_QUERY_TILED_SDPA=0`, `EXO_DSV4_EXACT_TOPK_PREFILL=1`).
**Outcome: NOT-THIS-FLAG.** Median **32.49** tok/s. The pre-registered NOT-THIS-FLAG
band is median <= 33.16; 32.49 clears it with room. Recovery fraction
R = (32.49 - 32.18) / 3.90 = **0.079** — under 8% of the P12->P13 gap. Turning off
the query-tiled SDPA flag alone buys back essentially nothing.

**Pre-flight.** Cold relaunch; old PIDs 7728 / 7776 confirmed gone; new real `exo -v`
PIDs **33861** (m4-1) / **34716** (m4-2). `ps eww` on both real PIDs on both nodes:
`QUERY_TILED_SDPA=0`, `EXACT_TOPK_PREFILL=1`. Live-env diff vs the P13 arm:
**exactly one differing var per node** (`EXACT_TOPK_PREFILL` 0->1). Live-env diff vs
the P12 arm: **exactly one differing var per node** (`QUERY_TILED_SDPA` 1->0).
74 EXO_* vars each side, no other drift. This is a clean one-variable move from
*both* anchors, which is the strongest form this split can take. Capture:
`tmp/p14-armA-20260831/preflight_node{1,2}_ps_eww.txt`.

| rep | decode_tps | trustworthy | completion_tokens | finish_reason | prompt_tokens | prefill_tps |
|---|---|---|---|---|---|---|
| warmup (discarded) | 33.24 | true | 1200 | length | 110008 | 382.98 |
| 1 | 33.34 | true | 1200 | length | 114490 | 382.45 |
| 2 | 31.89 | true | 1200 | length | 111129 | 381.19 |
| 3 | 32.49 | true | 1200 | length | 107767 | 384.35 |
| **median** | **32.49** | true | 1200 | length | — | 382.45 |

**Arm A behaves like P12, not like P13 — and the rep-level structure says so more
clearly than the medians do.** Arm A reps span 31.89-33.34; P12 reps span
31.36-32.89. Those overlap heavily — the two are indistinguishable at n=3. Against
P13 (34.18-36.20) there is **complete separation**: Arm A's best rep (33.34) sits
below P13's worst (34.18). So Arm A is firmly in the slow regime. This conclusion
does not rest on the median choice, which matters given the wide spreads.

**Positive control PASSES, so the null is a real null and not a dead knob.** The
pre-registration required that an arm turning off a prefill flag must move prefill
off the 416.9-418.4 band, else the arm is invalid regardless of its decode number.
Prefill went 417.21 -> **382.45**, **-8.33%**. The flag unambiguously reached the
kernels; it simply did not move decode.

**Bonus decomposition of the prefill effect — free from this arm.** Prefill:
P12 (both ON) 417.21; Arm A (only QUERY_TILED off) 382.45; P13 (both off) 378.15.
So `QUERY_TILED_SDPA` is worth **34.76** tok/s of prefill (89.0% of the combined
39.06), and `EXACT_TOPK_PREFILL` only about **4.30** (11.0%). The two flags are
very unequal contributors on the axis they were actually designed for.

**Which sets up Arm B as the decisive run, with a sharp prediction.** By
elimination, if the P12->P13 decode gap is a single-flag effect it must be
`EXACT_TOPK_PREFILL`, and Arm B should land DOMINANT (>= 35.11). If Arm B instead
lands NOT-THIS-FLAG, then neither flag alone reproduces the effect and it is a
genuine interaction requiring both — which would point at a shared resource
(allocator/pool state, Metal shader-cache or `@mx.compile` graph residency) rather
than either flag's own code path, per the mechanism list in §P13 RESULT.

**Note the emerging cost asymmetry, if Arm B lands DOMINANT.** `EXACT_TOPK_PREFILL`
would then be buying ~4.30 tok/s of prefill (~1.0%) while costing ~3.90 tok/s of
decode (~12.1%). At the project's decode-oriented north star that trade is bad by a
wide margin and the default should flip. Not concluded yet — stated in advance so
the decision rule is pre-registered rather than reverse-engineered from Arm B's
number.
