Quantized all_sum: LIVE CLUSTER TEST FAILED -- hangs the collective (2026-08-19)
====================================================================================

Summary
-------

Deployed the quantized `moe.all_sum` replacement (mlx-lm branch
`feat/moe-allsum-quant-2026-08-19`, `EXO_DSV4_MOE_ALLSUM_QUANT=1`) to the
live 2-node cluster for the first real end-to-end test. **It failed
immediately on the first real prefill request** with:

```
[Event::wait] Timed out: GPU event not signaled and no stream exception
(peer rank stuck on an abandoned c>=2 collective); surfacing a clean
fault for in-place reconnect / restart.
```

HTTP 500, prefill never advanced past 0/38066 tokens. This confirms
exactly the gap flagged as untested in
`docs/moe-allsum-quant-compute-overhead-analysis-2026-08-19.md`: the
actual `mx.distributed.all_gather` call, never tested end-to-end with a
real second rank, does not work correctly in production -- the local
quant/dequant math (validated with 6 passing unit tests) was necessary
but not sufficient. Something about `all_gather`'s wire behavior for
this quantized int8+scale payload shape either hangs or produces a
collective-desync that the cross-rank lockstep detection catches.

Root cause NOT yet determined
----------------------------------

Did not debug further live -- correctly stopped and reverted rather
than iterating on a hung/faulted collective on production hardware.
Plausible causes (unranked, need code-level investigation before any
retry):
- `all_gather` on a TP=2 group may have a different expected tensor
  rank/shape contract than `all_sum`, and the quantized payload's shape
  (int8 tensor + separate fp32/uint8 scale tensor, gathered together or
  sequentially?) may not match what jaccl's `all_gather` implementation
  expects.
- The two SEPARATE collective calls (`all_gather` on the quantized data,
  `all_gather` on the scales) may not be properly ordered/paired across
  ranks, causing one rank to get ahead of the other -- exactly the kind
  of cross-rank graph-position drift the `Phase H Lever 1` forced-eval
  comment (deepseek_v4.py) exists to prevent for the UN-quantized path.
  The quantized replacement may not have an equivalent ordering
  guarantee.
- jaccl's `all_gather` may simply be less exercised/more fragile than
  `all_sum` in this codebase generally (worth checking git history/
  existing usage elsewhere in the model).

Recovery
-----------

Cluster **self-healed automatically** -- exo's own fault-detection
("surfacing a clean fault for in-place reconnect / restart") recovered
both nodes to healthy state within ~15-30s without manual intervention.
Confirmed both nodes healthy via `/metrics` afterward. No hard hang, no
reboot needed, no data loss -- the "clean fault" framing in the error
message was accurate.

Actions taken
----------------

1. Confirmed both nodes healthy post-fault via `/metrics`.
2. Reverted the `mlx-lm` submodule pointer back to the pinned commit
   (`bd5d67648e82069168314f95fad8f0f2a7b67ea1`) -- the quantized code
   remains on its own unmerged branch, untouched, available for future
   debugging.
3. Relaunched the cluster at the standing baseline config
   (`EXO_PREFILL_STEP_SIZE=2048`, no quant flag) to restore known-good
   state.
4. Did NOT attempt to debug/retry the quantized path live tonight --
   this needs code-level investigation of jaccl's `all_gather`
   semantics (offline, code-reading) before any further live attempt.

What this means for the quantized-all_sum lever
-----------------------------------------------------

**Status: real, promising idea on paper (compute overhead confirmed
negligible), but the actual implementation has a real bug in the
distributed collective call that was never caught by local-only
testing.** This is exactly why the "not yet verified end-to-end" caveat
in the prior doc mattered -- it caught a real, cluster-breaking issue
before it could be mistaken for a working optimization. Do not
re-attempt on the live cluster without first: (a) reading jaccl's
`all_gather` implementation to understand its exact shape/ordering
contract, (b) fixing whatever mismatch caused the hang, (c) testing the
FULL quantize->all_gather->dequant round trip in some way that doesn't
risk the live cluster (unclear if this codebase has any 2-rank
simulation/testing capability outside real hardware -- if not, any
retry is inherently a live-cluster risk and should be scoped and
approved explicitly as such).

Standing config confirmed unaffected -- baseline (unquantized)
`moe.all_sum` continues to work correctly, as it has all night.
