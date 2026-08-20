Shared-scale int8 all_sum: CORRECTED FINAL ANSWER -- slower during real prefill, fast only in decode (2026-08-19/20)
========================================================================================================================

Correction notice
---------------------

An earlier read of this same data (during this session) concluded the
design achieved a ~148x speedup based on the TAIL of the SS-PROBE-LITE
log (p50 ~1.2ms/call). That conclusion was WRONG -- caught and
corrected within the same session by cross-checking call-count
timestamps against the actual `Prefill progress`/`Prefill complete`
log lines, which the earlier read had not done. This doc supersedes
that number entirely.

What actually happened
--------------------------

Deployed the lite (entry/exit-timing-only, no per-phase mx.eval
fences) probe live
(mlx-lm branch feat/moe-allsum-sharedscale-2026-08-19, commit
`b7e3db6`, `EXO_DSV4_MOE_ALLSUM_SHAREDSCALE_PROBE=lite`) to get a
trustworthy unfenced number, per the plan from
`local-absmax-fence-artifact-confirmed-2026-08-19.md`.

Fired a real 38,067-token fresh prefill request (correctness confirmed:
clean secret-code recall, HTTP 200). Cumulative throughput: ~168-169
tok/s -- statistically identical to the unquantized baseline measured
repeatedly all night (~162-172 tok/s), i.e. genuinely NO speedup, same
as the earlier (fenced) live test found.

Pulled the SS-PROBE-LITE per-call log and found a dramatic split:
~200-280ms/call for roughly the first 800-1000 calls, then a sharp
drop to ~1.1-1.6ms/call for the remaining ~2260 calls in the log.

**Correlated against `Prefill progress`/`Prefill complete` timestamps:
the slow phase (~200-280ms/call) exactly spans the real prefill request
(03:10:25 start -> 03:14:12 `Prefill complete: 38067 tokens in
226.70s`). The fast phase (~1.2ms/call) begins immediately AFTER
`Prefill complete` fires -- i.e. it is DECODE, not prefill.**

Real numbers, correctly attributed
---------------------------------------

| phase | payload | measured p50 |
|---|---|---|
| PREFILL (shape (1,2048,4096), the real target) | large, ~8MB int8 | ~200-280ms/call |
| DECODE (single/few-token steps) | tiny | ~1.1-1.6ms/call |

Comparison against the known unquantized baseline (178ms/call,
measured earlier tonight at the same prefill shape via NOP-ablation):

```
shared-scale int8 all_sum, PREFILL:  ~265ms/call (avg of the slow phase)
baseline (unquantized) all_sum:       178ms/call
ratio:                                ~1.49x -- SLOWER, not faster
```

**This directly and finally explains the "no speedup" result from the
earlier live test** (`moe-allsum-sharedscale-live-test-no-speedup-2026-08-19.md`)
-- it wasn't a measurement artifact, and it isn't fixed by removing
the probe's eval fences. During actual prefill, at the real production
payload size, this design is genuinely SLOWER than the plain
unquantized path.

Why: latency-vs-bandwidth crossover (a caveat Fable flagged from the
very first consult on this design and which was under-weighted at the
time)
-----------------------------------------------------------------------------

At the small decode payload size, the fixed per-call overhead
(computing local_absmax, agreeing on a shared scale via a second
all_sum, quantize/dequant) is negligible relative to almost-zero
baseline cost, so the design looks essentially free. At the large
prefill payload size, that same fixed overhead is now competing
against a baseline that's ALSO fast in relative terms (178ms is
already reasonably efficient for an 16.8MB bf16 payload at this
cluster's real RDMA throughput) -- and the shared-scale design's TWO
all_sum calls (one tiny scalar for the scale agreement, one large for
the payload) plus the local reduction cost more in aggregate than one
plain all_sum, even with the smaller wire payload.

Status: CLOSED
------------------

The shared-scale int8 all_sum design is a real, correctness-validated,
architecturally sound piece of engineering (int8 wire dtype confirmed
supported, overflow-safety proven, 100% on the reliable path, no
all_gather) -- but it does not deliver a prefill speedup at this
cluster's real production shape. It may be genuinely useful for DECODE
collectives specifically (where the ~1.2ms number is real and
representative), but that's a different, unexplored use case from what
this investigation targeted (prefill throughput). Do not redeploy for
prefill without a fundamentally different mechanism that reduces the
FIXED per-call overhead, not just the payload bytes.

Session process note (for future reference)
-------------------------------------------------

This finding required cross-checking the probe's own call-count
timestamps against the independent `Prefill progress`/`Prefill
complete` log lines to correctly attribute which phase (prefill vs.
decode) the fast numbers belonged to. A superficial read of "the tail
of the log is fast" without that cross-check produced a wrong, and
much more exciting-sounding, conclusion. Always correlate probe/bench
timestamps against an independent ground-truth signal (here, the
prefill-completion log line) before trusting a phase split in
per-call data from a mixed prefill+decode request.
