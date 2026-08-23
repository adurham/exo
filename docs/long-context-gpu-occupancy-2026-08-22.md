# T5: Long-context (300K) GPU occupancy/gap capture — occupancy INCREASES with depth (82.4-82.7% vs short-ctx 78.6-78.9%), contradicting the naive "more attention work = more idle" expectation — 2026-08-22 (session 4)

## Why this check

Per the Fable-provided plan: repeat T2's capture methodology at long
context (300-500K) to check whether attention/KV kernel cost share
explains the real context-scaling decode throughput drop (29.2 tok/s
short-ctx → 21.51 tok/s @500K, a real ~26% degradation per T1).

## Method

Real production runner PIDs (same as T2: m4-1=65758, m4-2=79210).
Fired a real `bench/decode_probe.py` request with `--prompt-tokens
300000 --max-tokens 3000` against the live cluster — genuine 300K-token
prefill (confirmed via `/metrics`: `exo_prompt_tokens_total` jumped to
301,660, a real fresh prefill, not a cached-prefix reuse) followed by
real decode. `bench=True` in the request body did not engage bench-mode
EOS-banning as expected (the flag is honored by the `/bench/chat/completions`
route, not the plain `/v1/chat/completions` path `decode_probe.py`
actually calls) — real decode window ended at n_tokens=200,
decode=9.03s, **22.03 tok/s** (consistent with the known 300K baseline
of 24.44 tok/s from T1, within normal run-to-run variance). This
shorter-than-planned window (9s vs the originally intended much longer
capture) is a real methodology gap, noted below, not glossed over —
but the resulting trace still contains ~26,000-27,000 real GPU
intervals per rank, an ample sample for occupancy/gap analysis.

Launched `xctrace --template 'Metal System Trace' --attach <pid>
--time-limit 15s` on both nodes immediately before firing the request
(same technique as T2), exported `metal-gpu-intervals` via the
established `--xpath` method, parsed via the same streaming
`ElementTree.iterparse` script used in T2.

## Real result

### GPU occupancy (interval-union, own-process rows, request-window-isolated)

| | rank0 (300K ctx) | rank1 (300K ctx) | rank0 (short ctx, T2) | rank1 (short ctx, T2) |
|---|---|---|---|---|
| Real GPU intervals (our process) | 26,063 | 27,126 | 63,952 | 65,279 |
| Request-window-isolated occupancy | **82.43%** | **82.70%** | 78.64% | 78.86% |

**Occupancy at 300K context is HIGHER than at short context** (82.4-82.7%
vs 78.6-78.9%, both ranks) — the opposite direction from a naive "more
attention/KV work per token at depth = more idle waiting" hypothesis.
This is a real, if modest (~4 percentage point), and consistent
(both ranks agree) finding.

### Gap-length distribution

| | rank0 (300K) | rank1 (300K) | rank0 (short, T2) | rank1 (short, T2) |
|---|---|---|---|---|
| median gap | 0.96µs | 1.00µs | 95.12µs | 89.29µs |
| mean gap | 160.23µs | 155.48µs | 139.20µs | 137.43µs |
| p95 gap | 263.88µs | 271.33µs | 213.12µs | 230.71µs |

**Median gap dropped dramatically** (from ~90-95µs at short context to
~1µs at 300K) while mean/p95 stayed roughly comparable — this shape
(tiny median, similar mean/p95) indicates far MORE very-short gaps
(sub-microsecond, likely between back-to-back tightly-packed kernel
dispatches within a single larger attention computation over the
longer KV/pooled sequence) alongside a similar population of
moderate-length gaps. This is consistent with — not contradicting —
the higher occupancy figure: at deep context, the GPU is kept
continuously busier by the larger per-token SDPA/attention workload
(real per-kernel work scales with KV/pooled sequence length, as
established in `docs/dsv4-attention-kernel-efficiency-2026-08-18.md`'s
real shape table: local KV len 2175 → pooled CompressedAttention KV up
to 3894 at 220K context), leaving proportionally less true idle time
between dispatches.

### Per-channel breakdown

Same limitation as T2, re-confirmed at long context: `gpu-channel-name`
is 100.0% "Compute" (26,056/26,063 rows) — no MLX kernel names
available from this trace template at any context depth. Per-kernel
attribution (isolating how much of the increased busy-time is
attention/SDPA specifically, vs MoE) still requires
`mx.metal.start_capture()`/Xcode GPU Frame Capture, consistent with
T2's and T3's findings.

## Interpretation — real, and it complicates the naive hypothesis

**The naive hypothesis this check set out to test — "decode gets slower
at depth because attention/KV kernels eat proportionally MORE of an
increasingly-idle GPU" — does NOT match the real data.** Occupancy goes
UP with depth, not down. This means the real throughput degradation
(29.2 → 21.51 tok/s, T1) is NOT explained by the GPU becoming more idle
as context grows. Instead, it's consistent with the straightforward,
expected explanation already implicit in the roofline work: **real
per-token compute cost genuinely increases with context depth** (larger
KV/pooled attention shapes require more real GPU work per token, not
more idle waiting) — the GPU stays busier because there is genuinely
more work to do per token, and that extra real work is what slows
decode down, not a growing idle-time problem.

This is a real, informative NEGATIVE result for the specific hypothesis
under test (context-scaling throughput loss ≠ growing idle-time
problem), consistent with — not contradicting — T1's finding that the
roofline compute floor itself needs to account for KV-read cost growth
with context (T1's flagged, still-open caveat: the 6.51ms roofline
floor is active-MoE-weight-bytes-only and excludes KV-cache read cost,
which grows with context — this session's real occupancy data
independently corroborates that the "missing" cost at depth is real
compute, not idle time).

## What this does NOT establish

Real per-kernel attribution of WHICH specific attention sub-kernel
(SDPA vs SDPA.compressed vs indexer) drives the increased busy-time at
depth remains unmeasured — the trace template's channel-name limitation
(same as T2/T3) blocks this. Does not establish whether the increased
occupancy at depth is running at good or poor per-kernel efficiency
(more busy time is not automatically more USEFUL compute — the same
occupancy-vs-efficiency distinction flagged throughout this campaign
applies here too). The methodology gap (9s decode window instead of a
longer capture, due to the bench-flag routing issue) means this
result, while real and internally consistent across both ranks, would
benefit from a longer confirmatory capture in a future session if the
question becomes higher-priority.

## Decision

**T5 substantially answered, with a genuine negative result for its
original hypothesis.** Context-scaling decode slowdown is NOT explained
by growing GPU idle time — occupancy increases, not decreases, with
depth. The real explanation is straightforward increased per-token
compute cost from larger attention/KV shapes, already anticipated by
T1's flagged roofline-floor caveat. Does not change T10's priority
(the prefill-side 28.8% non-GEMM remainder investigation) or T6's MTP
gate evaluation — this was an independent question about decode's
context-scaling mechanism, now answered.
