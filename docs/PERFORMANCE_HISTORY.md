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

### 3.2 Prefill cliff investigation (root cause never fully found)

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
| DSpark FULLBLOCK at real context depth | 15.9x throughput collapse; cluster runs DSpark OFF | §5.3 |
| `MLX_METAL_FAST_SYNCH=1` | 1.5x slower, 70x more variance | §2.3 |
| `EXO_DSV4_FUSED_SOFTMAX` | Real correctness break (needle failure at 100K) | §6 |
| Nesting `mx.fast.metal_kernel` inside `mx.compile` | Catastrophic 3-4x regression | §4.5 |
| `EXO_DSV4_INDEXER_PBLOCK` (small block size) | Real decode regression at depth | §3.2 |
| Subgroup `attn.all_gather` (vs all_sum workaround) | Still faults post-transport-fix | §8 |
| `all_sum` NOP ablation as a live measurement technique | Unsafe — destabilizes the cluster | §13 |
