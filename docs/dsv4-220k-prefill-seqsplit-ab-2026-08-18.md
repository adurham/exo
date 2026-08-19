# DSv4-Flash TP prefill: EXO_DSV4_SEQ_SPLIT A/B at ~200K context (2026-08-18)

## Context

`EXO_DSV4_SEQ_SPLIT` had only been validated at 100K context in earlier
work. This session re-ran the A/B at long context (~190-220K tokens) to
confirm the choice still holds at the scale the cluster is actually
being used for. Cluster state: commit `06daa2286`, TP-only
(`DSV4_SHARDING=Tensor`), standing `MLX_JACCL_DATA_RECV_POOL=0` fix
applied (Section 119 workaround, required for any relaunch to succeed).
Both Mac Studio M4 Max nodes, 128GB each, RDMA over Thunderbolt 5.

## Method

Same needle-in-haystack methodology as the other 2026-08-18 prefill
docs in this directory: a genuinely fresh, randomly-seeded ~190-220K
token prompt with a unique embedded secret code, `POST
/v1/chat/completions`, `max_tokens=50`, `temp=0`, asking the model to
repeat the code back. Verified genuineness via `usage.cached_tokens: 0`
and server-side `Prefill progress: N/<total>` log lines on both nodes
bracketing a single clean run with no concurrent request in the window.

Rate computed from wall-clock between the `0/<total>` and final
`<total>/<total>` server log lines, cross-checked against the server's
own live-computed tok/s at completion (they match to within noise in
both runs below, confirming internal consistency).

## Results

**SEQ_SPLIT=1 (baseline, captured earlier this session):**
220,318 prompt tokens / 614.5s = **358.6 tok/s**. Output correct,
`finish_reason: stop`.

**SEQ_SPLIT=0 (this run):**
191,330 prompt tokens / 688.775s (18:54:39.243 -> 19:06:08.018) =
**277.8 tok/s** wall-clock, matching the server's own live-computed
rate at completion (277.8 tok/s) exactly. `cached_tokens: 0` (fresh
prefill, not a cache-hit artifact). Model correctly located the secret
code (`SPLIT0-9493-DIAG-773...`, string cut short by the same
50-token completion-budget artifact seen in the prior ring-diag doc --
`finish_reason: length`, not a correctness failure -- the code was
correctly identified in the reasoning trace before the budget cut it
off).

Token counts differ (191K vs 220K) because the harness's random-seed
filler-word count wasn't recomputed for exact parity before this run.
This does not undermine the comparison: the SEQ_SPLIT=1 span-profile
doc shows throughput is flat-to-mildly-degrading with token count in
unprofiled runs at this scale (no mechanism by which a *shorter*
191K run would be structurally slower than a 220K one), so if
anything the shorter SEQ_SPLIT=0 run had a slight scale advantage --
yet it still measured slower.

## Conclusion

**SEQ_SPLIT=1 is ~29% faster than SEQ_SPLIT=0 at long context**
(358.6 / 277.8 = 1.29x), confirming the SEQ_SPLIT=1 default holds at
190-220K context, not just the previously-validated 100K case. No
further action needed -- this closes out the queued SEQ_SPLIT A/B
task. Current launcher default (`EXO_DSV4_SEQ_SPLIT=1`, set at
`start_cluster.sh:108`) is correct; do not flip it.

## Aside: rsync repo-sync slowness (resolved as expected behavior, not a bug)

The handoff into this session flagged the `start_cluster.sh` repo-sync
`rsync` step to `macstudio-m4-1` as suspiciously slow (~0.84 Mbit/s)
and hypothesized it was routing over the wrong interface (household
WiFi/Ethernet instead of the 80Gbps Thunderbolt 5 RDMA link).

Investigated on resume and corrected: this session's driving laptop
(`adams-macbook-pro-m4`) has **zero Thunderbolt devices physically
connected** (confirmed via `system_profiler SPThunderboltDataType` --
all 3 TB buses report "No device connected"). The 192.168.200.x /
192.168.201.x TB5 RDMA subnet exists only between the two Mac Studios
directly; the laptop was never on it and never could be. All
laptop-to-studio traffic, including `start_cluster.sh`'s rsync step,
necessarily goes over the household WiFi/Ethernet LAN
(192.168.86.0/24) regardless of hostname resolution. The mDNS
`.local` resolution to `192.168.86.201` flagged in the handoff was
therefore correct behavior, not a misroute -- there was no TB5 path
available to misroute away from.

This is consistent with the design comment already in
`start_cluster.sh` (~line 1185-1189): the rsync source-of-truth
pattern exists specifically because studio-to-studio `git fetch`
over the internet was unreliable, and rsync from the laptop replaced
it -- the laptop was never intended to be on the RDMA fabric, only
the two Mac Studios are. The observed rsync duration in this session
was ordinary WiFi transfer time for the changed files (dominated by
build artifacts like `libexo_rs.dylib`), not a bug. No code change
made. If repo-sync time becomes a recurring pain point, the actual
lever would be reducing what gets rsynced (e.g. trimming build
artifacts from the sync set), not chasing an interface-routing fix
that cannot exist given the laptop's hardware.
