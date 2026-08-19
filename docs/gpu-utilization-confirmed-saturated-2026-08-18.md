# GPU utilization confirmed near-saturated throughout real prefill (2026-08-18)

## Context

The prior "compute-bound" conclusion from earlier tonight was inferred from
span-time percentages (attn+ffn summing to ~100% of wall time). Span time
measures wall-clock of a code region, which is not the same thing as GPU
business -- a region could be "slow" due to CPU-side dispatch/serialization
while the GPU itself sits idle. This was flagged as a real, previously
unverified gap: nobody had directly measured aggregate GPU utilization
percentage during a real prefill run.

A dedicated task to measure this hit a tool timeout mid-run, but its
already-launched probe (a real 573K-token needle-in-haystack request
against the live cluster) was still executing server-side when discovered
-- salvaged by polling `exo_gpu_usage_ratio` from both nodes' `/metrics`
endpoint directly for the remainder of the run, rather than losing the
in-flight measurement.

## Method

Polled `curl http://<node>:52415/metrics | grep exo_gpu_usage_ratio` on
both cluster nodes at ~8-9 second intervals, correlated against server-side
`Prefill progress: N/573540 tokens` log timestamps confirming the request
was actively mid-prefill (182K-242K tokens processed, 358-365 tok/s live
rate) throughout the sampling window.

## Results

16 samples across both nodes during active prefill:

- **m4-1 (master, rank 0)**: mean 97.0%, min 93.7%, max 98.7%
- **m4-2 (worker, rank 1)**: mean 96.6%, min 93.4%, max 98.4%

No sustained dips below ~93% on either node. No idle bubbles found.

## Conclusion

**Confirms the compute-bound conclusion from earlier tonight with direct
utilization telemetry, not just span-time inference.** There is no hidden
idle-time lever -- both GPUs are genuinely, continuously near-saturated
during real long-context prefill. This closes the "4th unmeasured thing"
gap Fable flagged when this investigation was scoped: the ~29% of wall
time not covered by the top-two named spans (attn 44.3% + MoE 26.9%) is
NOT idle GPU time; it's real GPU work happening under other/smaller spans
(all_gather, all_sum, indexer, norms, embed, etc. -- all individually
small but summing to real busy time), consistent with the full span table
in `docs/dsv4-220k-prefill-span-profile-2026-08-18.md` which does account
for essentially all of it once every row is summed, not just the top two.

No code changes. Raw sample log preserved at
`/tmp/gpu_util_probe/manual_poll.log` (not committed -- ephemeral /tmp
data, reproducible via the same polling method against any live prefill
request).
