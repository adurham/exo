# EXO_PREFILL_STEP_SIZE=4096 retest at long context: quality passes, throughput REGRESSES (2026-08-18)

## Context

Continuing the queued investigation from earlier tonight: commit `1185ae740`
(2026-07-13) raised `EXO_PREFILL_STEP_SIZE` 1024->2048 specifically because
of MoE-GEMM small-M inefficiency, and that same commit tested 4096 and
rejected it for a quality break ("needle misses, garbled output at 1K/32K").
That finding was never root-caused and predates SEQ_SPLIT and other
subsequent changes, so tonight it was retested on current `main`
(commit `13a7f116e`) before investing further effort chasing a possibly-stale
bug.

Separately, a fresh production-class MoE-GEMM microbenchmark tonight
(`bench/moe_production_class_bench.py`, see git history) measured the
ragged-small-M mxfp4 SwitchGLU forward at 43-72% of matched-shape dense
ceiling depending on chunk size (43% at L=512, 62.6% at L=2048, 72.0% at
L=8192) -- suggesting a naive throughput win from raising the chunk size,
IF the quality break turned out to be stale.

## Method

Needle-in-haystack correctness test at 1K and 32K context (matching the
original July failure's context sizes), each run at both STEP_SIZE=2048
(control, already running) and STEP_SIZE=4096 (test, required a cluster
relaunch -- `EXO_PREFILL_STEP_SIZE=4096 MLX_JACCL_DATA_RECV_POOL=0
./start_cluster.sh`, all other launcher defaults including
`EXO_DSV4_SEQ_SPLIT=1` unchanged). Fresh random secret code + random-word
filler per test, `POST /v1/chat/completions`, `max_tokens=50`, `temp=0`.
Verified fresh prefill via `cached_tokens: 0` in each response (one 32K
trial hit a false cache-collision from reusing an identical prompt across
two curl attempts after a client-side timeout -- exo does not cancel
server-side prefill on client disconnect, a previously-documented pitfall
-- that trial was discarded and rerun with a genuinely fresh prompt).

Additionally ran a large-context (~191K token) real-throughput comparison
at STEP_SIZE=4096 against tonight's earlier STEP_SIZE=2048/SEQ_SPLIT=1
baseline (358.6 tok/s, 220,318 tokens, from the SEQ_SPLIT A/B doc), using
the same needle-in-haystack methodology and server-side log timestamp
verification used throughout tonight.

## Results

**Correctness: PASSES at both 1K and 32K, STEP_SIZE=4096, current code.**
- 1K context: identical numeric code returned as the 2048 control
  (`7516-CHK-6079`), same reasoning trace, same truncation artifact
  (50-token budget cuts off the tag prefix -- a harness limitation seen
  throughout tonight's session, not a correctness failure).
- 32K context: exact match, `finish_reason: stop`, `cached_tokens: 0`
  (`K32B-9247-CHK-8316` returned verbatim).

**The July 2026-07-13 quality-break finding is STALE and does not
reproduce on current code.** Whatever combination of code changes since
then (SEQ_SPLIT, argpartition, or other fixes) resolved it, or the
original finding was itself measurement noise -- either way, 4096 is
quality-clean today at the tested context sizes.

**However: real-world throughput at 4096 REGRESSES, contradicting the
naive MoE-GEMM-efficiency-implies-throughput-win expectation.**
- STEP_SIZE=4096, ~191K tokens: **331.2 tok/s** (191,330 tokens,
  21:01:37.027 -> completion, live-computed rate at finish per server log,
  `finish_reason: length` truncation artifact only, code correctly
  identified in the reasoning trace).
- STEP_SIZE=2048 baseline (same session family, SEQ_SPLIT=1): **358.6
  tok/s** (220,318 tokens, from the earlier SEQ_SPLIT A/B doc tonight).
- **4096 is ~8% SLOWER than 2048 at long context**, despite the isolated
  MoE-GEMM microbenchmark showing 4096 should have ~15% better GEMM
  efficiency than 2048 (72.0% vs 62.6% of dense ceiling) in isolation.

## Interpretation

The isolated MoE-GEMM microbenchmark measures ONE component of the
pipeline in isolation and its result does NOT transfer to end-to-end
throughput. Something else in the pipeline gets more expensive at 4096
faster than the MoE-GEMM gets more efficient. The leading suspect,
already documented in this repo's own code comments
(`start_cluster.sh` ~line 74-77, the "Context-adaptive prefill chunk
sizing" note): **the sparse indexer's scores transient scales with BOTH
chunk size L and pooled window P**, and at high context this makes larger
chunks a net loss -- the exact mechanism already documented there for why
256-chunk is a "-30% at 380K vs 128" regression at HIGH context despite
being a "+39% win at 100K" LOW-context. The same shape of tradeoff
plausibly applies one order of magnitude up (2048 vs 4096) at long
context, and 191K tokens is well into "high context" territory by this
codebase's own established heuristics.

This is consistent with, not contradictory to, the existing
`EXO_PREFILL_STEP_SIZE_HIGH_CTX` / `EXO_PREFILL_STEP_SIZE_CROSSOVER`
context-adaptive chunking mechanism already built into this codebase
(currently unset/disabled by default) -- the infrastructure to use a
DIFFERENT chunk size at low vs high context already exists, unused.

## Conclusion

**Do not change the standing `EXO_PREFILL_STEP_SIZE=2048` default.** The
quality concern that originally blocked 4096 is resolved, but 4096 is a
net throughput LOSS at the long-context regime this cluster is actually
used for (~191-220K tokens), not a win. The MoE-GEMM efficiency gain is
real but is outweighed by a different, larger cost elsewhere in the
pipeline (indexer/attention-side, chunk-size-dependent) that was not
isolated by tonight's microbenchmark alone.

**This closes the immediate "is 4096 safe/free" question** with a clean
negative at the current default other settings. It does NOT close the
underlying opportunity: the isolated 72%-of-ceiling MoE-GEMM number at
L=8192 confirms real headroom exists if the chunk size used for MoE could
be decoupled from the chunk size used for attention/the sparse indexer
(process attention/indexer at 2048, but batch multiple such chunks
together before feeding the MoE block a larger effective L) -- a real,
scoped, non-trivial follow-up architectural idea, not investigated
further tonight. Also worth a follow-up: sweep intermediate values
(2560, 3072) rather than jumping straight to the previously-rejected 4096,
now that quality is confirmed clean at 4096 -- an intermediate size might
land in a sweeter spot between the MoE-GEMM win and the indexer-cost
regression than either 2048 or 4096 do individually.

## Cluster state

Cluster was relaunched twice for this test (4096 for testing, then
restored to the standing `EXO_PREFILL_STEP_SIZE=2048` default) and left
healthy, 2-node, `READY (2/2)`, commit `13a7f116e` on both nodes,
verified via `/metrics` on both nodes post-restore.
