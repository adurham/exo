RDMA chunk size sz=3 tested live: no measurable improvement, no hang (2026-08-20)
======================================================================================

Summary
----------

Tested the one remaining bounded/safe lever flagged hours earlier but
never actually executed: `MLX_JACCL_RELIABLE_MAX_SZ=3` (32KB chunks,
one step up from the standing sz=2/16KB, still inside the documented
safe zone -- the source comment's danger threshold is sz>=4/~64KB,
"do not reliably COMPLETE on Apple's librdma").

Result
---------

- **No hang, no wedge.** Cluster came up READY (2/2) cleanly, RDMA
  connected without issue. Confirmed sz=3 genuinely active on the wire
  (`[jaccl-v2] ENTER ... sz=3` in the live log, not silently reverted).
- **Correctness: clean.** Fresh 38,066-token prefill, correct secret-
  code recall, HTTP 200, `cached_tokens: 0`.
- **Throughput: NO measurable improvement.** ~166-169 tok/s cumulative
  -- statistically identical to the standing sz=2 baseline (~162-172
  tok/s measured repeatedly all night at this shape).

Interpretation
------------------

This is a real, clean null result, not an inconclusive one -- the test
ran correctly, the chunk size genuinely changed, and throughput simply
didn't move. Possible explanations (not investigated further tonight):
- The `moe.all_sum` cost measured via NOP-ablation earlier
  (`moe-all-sum-dominant-cost-2026-08-19.md`, 61-64% of wall time) may
  be dominated by something other than chunk count at this shape --
  e.g. a fixed per-round latency floor that doesn't shrink with larger
  chunks, or a different code path than the one `MLX_JACCL_RELIABLE_MAX_SZ`
  actually gates.
- P2's earlier chunk-count arithmetic
  (`moe-all-sum-skew-vs-comms-2026-08-19.md`) derived the 178ms/call
  cost from a MODEL of chunk-round latency, not a direct measurement of
  sz-vs-latency scaling -- that model may not hold, or the real
  bottleneck may be elsewhere in the round-trip (e.g. `ack_sync_pre`
  barrier cost, which is per-CALL not per-chunk and wouldn't shrink
  with fewer, larger chunks).

Status
---------

Reverted to standing baseline (`EXO_PREFILL_STEP_SIZE=2048`, sz=2)
immediately after the test. This closes the sz=3 lever as tested and
negative -- do not re-attempt sz=3 expecting a different result without
new evidence. sz>=4 remains explicitly out of scope (documented hang
risk, no clean baseline to compare against per earlier consult
guidance).
