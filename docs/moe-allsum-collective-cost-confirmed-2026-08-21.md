# TP collective cost: moe.all_sum confirmed real and substantial via sync-span — 2026-08-21 (session 2, part 9)

## Why this check

Per a second Fable review checkpoint: the wq_a+wkv fusion's null result
(-0.48%, not the predicted ~15% under a naive "fixed cost × dispatch
count" model) actively falsifies the simple dispatch-overhead
explanation for the earlier roofline finding (decode at ~12% of
bandwidth-bound peak). Fable flagged TP collective (all_gather/all_sum)
round-trip cost over the Thunderbolt interconnect as the more likely
dominant factor, unaccounted for in the per-node-only bandwidth
roofline, and asked for real measured collective cost rather than more
extrapolation.

## Method

Relaunched with `EXO_DSV4_MOE_FUSED_GATE_UP=1 EXO_PROFILER=spans
EXO_PROFILER_SYNC_SPANS=1` (sync-span mode forces `mx.synchronize()` at
every span boundary, so each span measures REAL wall time including any
collective wait, not MLX's lazy/async graph-build time — this is the
same methodology used for the earlier prefill kernel breakdown, applied
here to decode). Ran a short-prompt, long-generation request
(`bench/decode_probe.py`, 300 max tokens) and pulled the resulting
`[PROFILER] span breakdown:` dump.

Also confirmed via live `ps aux`: `EXO_DSV4_ATTN_ALLSUM=0` in the current
production config — the attention-side `all_sum` collective is DISABLED
in this deployment (only fires if explicitly re-enabled), so the only
real per-layer TP collective actually executing in this model's forward
pass is `moe.all_sum` (the MoE branch's cross-rank reduce).

## Result

From the real sync-span dump (n=129 forward-pass calls, spanning this
request's full prefill+decode — not decode-isolated, see caveat below):

| Span | % of measured wall time |
|---|---|
| ffn (top-level) | 66.6% |
| moe.switch_mlp | 45.2% |
| attn (top-level) | 33.4% |
| **moe.all_sum** | **14.4%** |
| attn.o_proj | 10.5% |
| attn.proj_qkv | 7.0% |

**`moe.all_sum` — a real cross-rank TP collective, not a compute
kernel — is 14.4% of measured wall time under forced-accurate (sync)
timing.** This is a genuine, measured cost, not an estimate or roofline
inference. It's larger than `attn.proj_qkv` (7.0%) and comparable to
`attn.o_proj` (10.5%) — i.e., in the same league as real compute
projections, not a rounding error.

## Caveat

This particular dump's `n=129` spans the WHOLE request (a multi-chunk
525-token prefill plus a 244-token decode phase), not a decode-isolated
window — the KV-cache-added log line for 525 tokens appears immediately
after this dump in the raw log, confirming it captures both phases
combined. The 14.4% figure is therefore a blended prefill+decode
average, not a decode-only number. A cleaner decode-only isolation
(e.g., resetting/dumping the span profiler at the prefill/decode
boundary, similar to how `request_trace` already does via
`reset()`/`dump()` in `prefill()`) was not done this session — noted as
a concrnormal next step if this line of investigation continues.

## Interpretation

This is consistent with — and materially strengthens — Fable's
alternative hypothesis over the naive dispatch-count model: TP
collective cost (specifically `moe.all_sum`, since `attn.all_sum` is
disabled) is a real, substantial, and previously under-weighted
component of both prefill and decode wall time. Combined with the
already-documented `attn.all_gather` (8.5% of prefill, per the earlier
sync-span kernel breakdown) and this newly-confirmed `moe.all_sum`
(14.4% blended), TP collectives plausibly account for a meaningfully
larger fraction of total wall time than the individual-kernel view
suggested.

## What this changes for next steps

**Comm/compute overlap — launching the collective asynchronously
alongside independent, data-independent compute — is the highest-
priority remaining lever, ahead of further kernel-level fusion work.**
This is NOT a quick env-var toggle test like tonight's other levers; it
is a genuine engineering change (restructuring the forward pass so an
MoE all_sum for layer N can overlap with layer N+1's attention
prologue, or similar), requiring careful correctness verification (the
reduce must still complete before its result is consumed) and is a
larger scope than what was completed in any single lever test tonight.

**Not started this session** — flagged as the clear, evidence-backed
target for continued work: (1) get a clean decode-only span isolation to
confirm the 14.4% holds (or is even higher) in pure decode without
prefill's blended average, (2) investigate whether MLX's async graph
scheduling already provides any overlap for independent
same-layer/adjacent-layer work or whether the collective is a genuine
hard synchronization barrier, (3) if a real overlap opportunity exists,
design and implement it with the same rigor (offline correctness proof,
real hardware A/B, needle verification) applied to every other lever
tonight.
