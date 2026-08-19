# EXO_PREFILL_CLEAR_CACHE_INTERVAL=2 test at STEP_SIZE=4096: does NOT recover the regression (2026-08-19)

## Context

Earlier tonight, the 2048-vs-4096 throughput regression (331.2 vs 358.6
tok/s, ~8% worse at 4096) was investigated by direct measurement rather
than accepting an inherited "sparse indexer" explanation -- that
explanation was refuted (every indexer sub-stage measured CHEAPER per
token at 4096). The measured driver found instead was the Metal
allocator: `EXO_PREFILL_CLEAR_CACHE_INTERVAL=1` clears the buffer cache
every chunk, and at 4096 each chunk's working set is ~2x bigger, so
despite half as many clears the net per-token allocator cost was measured
+2.4% worse at 4096 vs -1.1% favorable at 2048 (an isolated microbenchmark
result, not a full end-to-end one). The hypothesis: relaunching at
`EXO_PREFILL_STEP_SIZE=4096 EXO_PREFILL_CLEAR_CACHE_INTERVAL=2` should
restore a similar bytes-cleared-per-token cadence to the 2048 baseline
and recover most of the regression.

## Method

Relaunched the live 2-node cluster with
`EXO_PREFILL_STEP_SIZE=4096 EXO_PREFILL_CLEAR_CACHE_INTERVAL=2
MLX_JACCL_DATA_RECV_POOL=0`, all other launcher defaults unchanged. Ran
the same needle-in-haystack methodology used throughout tonight's
session: fresh ~180K-token prompt with a unique random secret code,
`POST /v1/chat/completions`, `max_tokens=50`, `temp=0`. Verified fresh
prefill via `cached_tokens: 0`. Rate taken from server-side
`Prefill progress: 0/N` to `N/N` log timestamps, matching the
methodology used for every other throughput number tonight.

## Result

**332.8 tok/s** (179,720 tokens, 00:11:25.498 -> 00:20:25.414 =
539.9s), essentially unchanged from the interval=1 baseline:

| config | tok/s | vs 2048 baseline (358.6) |
|---|---|---|
| STEP_SIZE=2048, interval=1 (standing default) | 358.6 | -- |
| STEP_SIZE=4096, interval=1 | 331.2 | -7.6% |
| STEP_SIZE=4096, **interval=2** | **332.8** | **-7.2%** |

**interval=2 vs interval=1: +0.48%** -- statistical noise, not a real
recovery. The allocator-interval theory does NOT explain the bulk of the
regression, contrary to the earlier hypothesis. The isolated allocator
microbenchmark's +2.4%-worse-at-4096 finding was real as far as it went,
but it is evidently a small contributor to the full ~7-8% end-to-end gap,
not the dominant one as hoped.

**Correctness note**: this run's 50-token completion budget was consumed
by the model's reasoning trace before it reached the answer -- the
secret code was not recalled in the visible output this time (unlike
most other needle tests tonight, which either passed cleanly or hit the
same truncation-of-the-tag-suffix artifact after having already
identified the code). This is a single data point, not confirmed via
repeat, and throughput was the priority for this specific test -- flagged
for awareness, not treated as a new quality finding without
re-verification.

## Conclusion

**The clear-cache-interval fix does not work.** The 2048-vs-4096
regression's true cause remains only partially attributed: the earlier
session's isolated stage-by-stage measurement ruled out the indexer
(cheaper at 4096) and found a real but small (+2.4% isolated) allocator
effect that, per this end-to-end test, is clearly not the dominant term.
The bulk of the ~7-8% gap is still unexplained and most likely lives
inside the distributed model forward (MoE dispatch behavior, replicated
attention, or cross-rank collective timing at the larger chunk size) --
exactly the "needs a live 2-node cluster A/B to fully close" caveat
flagged (correctly) in the earlier investigation, now confirmed by
this negative result rather than resolved.

**Standing `EXO_PREFILL_STEP_SIZE=2048 EXO_PREFILL_CLEAR_CACHE_INTERVAL=1`
config reconfirmed correct.** No further chunk-size lever remains
scoped or testable tonight without a deeper, harder investigation into
the distributed-forward-path cost at 4096 specifically -- not attempted
here.

## Cluster state

Relaunched twice for this test (interval=2 test, then restored to
standing interval=1/STEP_SIZE=2048 config), left healthy, 2-node,
`READY (2/2)`, commit `575111d15` on both nodes, verified via `/metrics`
post-restore.
