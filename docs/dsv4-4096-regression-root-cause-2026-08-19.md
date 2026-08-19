# EXO_PREFILL_STEP_SIZE 2048-vs-4096 regression: ROOT CAUSE FOUND (2026-08-19)

## Summary

**Root cause: SDPA cost scales worse than linearly with per-rank sequence
length.** At STEP_SIZE=4096, SEQ_SPLIT halves the nominal chunk to 2048
real rows per TP rank (vs 1024 rows/rank at STEP_SIZE=2048). Both SDPA
sub-kernels (`attn.sdpa`, `attn.sdpa.compressed`) get measurably worse
per-token at the larger per-rank L: `attn.sdpa` costs 78% more ms/token,
`attn.sdpa.compressed` costs 100% more ms/token (i.e. roughly double) --
consistent with the expected quadratic-ish scaling of attention cost with
sequence length. This SDPA regression outweighs MoE's real, confirmed
efficiency GAIN at the larger chunk (moe.switch_mlp is 11% cheaper per
token at 4096, matching the isolated microbenchmark's prediction), and
the net effect is the observed ~7-8% end-to-end regression.

## Method

Ran the SAME 41,346-token prompt against the live 2-node cluster at both
STEP_SIZE=2048 and STEP_SIZE=4096, with `EXO_PROFILER=spans` enabled,
letting prefill run to natural completion (no signals -- the profiler
auto-dumps stats at prefill completion via a log line in
`generator.generate.prefill`, no SIGUSR1 needed). Compared per-span
`total_ms` normalized by total prompt tokens (ms/token) for a fair
apples-to-apples comparison despite the two runs having different chunk
counts.

**Measurement caveat**: this profile did NOT use
`EXO_PROFILER_SYNC_SPANS=1`, so per the standing profiler pitfall
documented earlier tonight, absolute span times can absorb adjacent lazy
-compute time into whichever call forces the next GPU sync. This means
individual spans' ABSOLUTE ms values should not be over-trusted. However,
since BOTH runs (2048 and 4096) suffer this same measurement artifact
equally, the RELATIVE comparison between them (ratio of ms/token) is a
valid signal -- and the SDPA ratios found here (1.78x, 2.0x) are large
enough to be a real effect, not an artifact of sync-mode ambiguity.

## Results (ms/token, normalized by 41,346 total prompt tokens)

| span | @2048 (L=1024/rank) | @4096 (L=2048/rank) | ratio |
|---|---|---|---|
| `attn` (parent) | 4.946 | 5.672 | **1.15x worse** |
| `ffn` (parent) | 4.657 | 3.668 | **0.79x (better)** |
| `moe.switch_mlp` | 0.820 | 0.729 | 0.89x (better, confirms isolated bench) |
| `attn.sdpa` | 0.387 | 0.689 | **1.78x worse** |
| `attn.sdpa.compressed` | 0.258 | 0.514 | **2.00x worse** |
| `attn.all_gather` | 3.680 | 3.885 | 1.06x worse (minor) |
| `moe.all_sum` | 3.695 | 2.813 | 0.76x (better) |

## Interpretation

This is architecturally expected, not a bug: attention (specifically the
SDPA kernels) does more work as the effective per-rank sequence length
grows -- both the raw dense-local-window term and the compressed/pooled
attention term scale with L in ways that outpace MoE's per-token
efficiency gain from larger batches. MoE benefits from bigger token
batches (more rows per expert, better GEMM utilization); attention does
NOT benefit from a bigger per-call L in the same way -- if anything it's
penalized, since SDPA cost grows with L while MoE cost per activated
token stays roughly flat (or improves) with M.

This directly explains why the naive "4096 should be faster because MoE
is more efficient" intuition from earlier tonight was wrong: it only
looked at MoE in isolation and never accounted for attention's opposite-
direction response to the same chunk-size change. The two effects don't
cancel -- attention's regression (60.7% of wall at 4096 vs 51.5% at 2048,
i.e. a LARGER fraction of a comparable total, on top of the per-token
figures above) dominates MoE's improvement.

## UPDATE 2026-08-19 (later): the "quadratic-ish" mechanism above is WRONG -- Option A tested and killed, real cause still open

A follow-up investigation designed and tested the "decouple attention's
per-rank L from MoE's batch size" idea flagged below. Two subagents
(code map + isolated laptop microbenchmark, see
`docs/dsv4-sdpa-subtiling-code-map-2026-08-19.md` and
`bench/sdpa_subtile_microbench.py`) found:

1. **The sparse attention class (`SparseCompressedAttention`, emits
   `attn.sdpa`) already internally tiles its SDPA calls** into
   `EXO_DSV4_SPARSE_SDPA_TILE=128`-row sub-chunks (default on), with a
   single upfront gather + per-tile dispatch + one final concatenate. So
   "sub-tile the SDPA call" (Option A) is *already happening* for this
   class -- there was nothing new to build here.
2. **Isolated laptop SDPA microbenchmark (same M4 Max architecture as
   the cluster) found SDPA cost is EXACTLY LINEAR in query-row count**,
   not quadratic: doubling rows (1024->2048) gives ~1.86-2.00x cost
   across every shape and KV-length tested (sparse-equivalent gathered
   KV=512, compressed-equivalent pooled KV 2128-20128). Tiling one big
   call into two smaller sequential calls measured 0.998-1.047x --
   neutral to slightly worse, no recoverable kernel-shape penalty
   exists. **This falsifies the "quadratic-ish scaling" explanation
   given in the original Summary above and confirms Option A
   (SDPA sub-tiling) is a dead end -- there is nothing to decouple at
   the SDPA-kernel level.**
3. **A real, still-unresolved discrepancy was found while cross-checking**:
   the live cluster's raw `attn.sdpa` span data showed an average
   per-call cost ratio of **3.15x** (95,245us/call @4096 vs 30,218us/call
   @2048) -- not the ~2.0x pure linear-row-scaling predicts. Depth-based
   confounds (the 4096 run's fewer/bigger calls occur at a slightly
   greater average prefill depth than 2048's more/smaller calls, so the
   sparse indexer's pooled-KV selection pool is marginally larger on
   average) were checked and are too small to explain this: bounded at
   roughly STEP/2 extra depth, ~5% effect max, nowhere near 1.58x.
   Candidate remaining causes (not yet isolated): the per-rank-L-scaled
   upfront gather step (`attn.gather` span, materializes a
   `(B, L_q, 512, 512)` tensor that's 2x larger at L_q=2048, timed as a
   SEPARATE span from `attn.sdpa` so not visible in the numbers above),
   lazy-eval misattribution absorbing adjacent work into whichever call
   forces the next sync, or a cluster-environment effect (memory
   pressure/allocator threshold) not present in the isolated laptop
   test. **This gap was not resolved before the raw log data needed to
   investigate it further was lost to a subsequent cluster relaunch's
   log rotation** -- re-investigating it requires a fresh matched-prompt
   A/B run capturing the `attn.gather` span specifically alongside
   `attn.sdpa`, which was not preserved from the original run.

## SECOND UPDATE 2026-08-19 (same day, later still): ruled out attn.gather/indexer/mask/compressor -- gap is inside SDPA itself

Ran a fresh matched-prompt (41,346 tokens, same prompt as all prior runs)
capture at STEP_SIZE=4096 with `EXO_PROFILER=spans`, specifically to
check `attn.gather` (the OPT-11 upfront-gather step, a span SEPARATE from
`attn.sdpa`) against `attn.sdpa`'s own numbers -- the leading unresolved
candidate from the update above.

**Result: `attn.gather` is negligible (2.17ms total, 0.0% of wall time)
and does not explain anything.** Also checked `attn.indexer` (1622ms @
4096 vs 1618ms @ 2048 -- essentially flat, ratio 1.002x), `attn.mask`,
and `attn.compressor` (both <0.005ms/token) -- none of these candidates
scale meaningfully between configs. The fresh capture also reproduced
the original finding almost exactly (attn.sdpa total_ms ratio 1.774x,
attn.sdpa.compressed ratio 2.001x -- both within 1% of the first run),
confirming this is a real, stable, reproducible effect, not run-to-run
noise.

**This means the unexplained ~1.58x extra cost is genuinely INSIDE the
SDPA kernel call itself** (or its immediate wrapper), not in any
surrounding gather/indexer/mask step -- which directly CONTRADICTS the
isolated laptop microbenchmark's finding that raw SDPA cost scales
exactly linearly with query-row count. Attempted the natural next test
(EXO_PROFILER_SYNC_SPANS=1, to rule out lazy-eval misattribution
absorbing adjacent work into the span boundary) but it did not complete
cleanly on the live cluster -- one attempt returned an HTTP 500 with
"Runner shutdown before completing command (signal=9)", a second attempt
hung with no prefill progress logged at all. Killed the attempt rather
than continue fighting cluster instability; this remains the next
concrete step for a future session.

## What this means for the standing config

**`EXO_PREFILL_STEP_SIZE=2048` remains the correct standing default** --
that conclusion is unchanged and well-supported (4096 measurably
regresses end-to-end, confirmed multiple times tonight, including via a
completely independent second matched-prompt capture). The mechanistic
explanation is narrower than before but still not fully closed: the
"extra" cost beyond pure linear FLOP scaling is now confirmed to live
specifically inside the SDPA call/wrapper itself, with gather, indexer,
mask, and compressor all ruled out as contributors. Option A
(SDPA-kernel-level sub-tiling) is DEAD -- do not revisit it. The next
concrete step for a future session: get `EXO_PROFILER_SYNC_SPANS=1`
working reliably on the live cluster (it crashed/hung twice tonight,
unrelated to the sync flag itself as far as could be determined) to
distinguish "real extra SDPA cost" from "lazy-eval misattribution
artifact" -- this is the last remaining decisive test that was
identified but not completed.

## Files

Raw span dumps captured via server log at prefill completion (auto-dump,
no signal needed -- discovered mid-session that the profiler already
dumps automatically when `generator.generate.prefill` completes, making
the earlier SIGUSR1 approach unnecessary and, as it turned out, unsafe --
a mistargeted SIGUSR1 signal crashed one cluster rank during this
investigation and required a full relaunch to recover; no data lost, but
flagging the safer method for future reference: just let a real request
complete naturally and read the log).

Cluster restored to standing config
(`EXO_PREFILL_STEP_SIZE=2048`, no profiler) after this investigation.
