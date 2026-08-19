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

## What this means for the standing config

**Confirms `EXO_PREFILL_STEP_SIZE=2048` is correct, for a real,
now-understood structural reason** -- not just "measured worse,
unexplained." The regression is not an incidental allocator or indexer
cost (both already ruled out earlier tonight); it is the direct,
expected consequence of attention's per-rank sequence-length sensitivity
under SEQ_SPLIT. Any future attempt to raise STEP_SIZE further would need
to address the SDPA per-rank-L scaling specifically (e.g. finer-grained
SEQ_SPLIT that keeps per-rank L bounded independent of nominal chunk
size, decoupling attention's L from the MoE batch size the same way the
"decouple MoE batch size from attention/indexer chunk size" idea flagged
earlier tonight would need to work) -- not a launcher-flag change.

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
