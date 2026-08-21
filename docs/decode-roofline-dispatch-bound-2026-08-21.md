# Roofline analysis: decode is dispatch-bound, not bandwidth-bound — 2026-08-21 (session 2, part 7)

## Why this analysis

Per a Fable review checkpoint: "optimize every microsecond" has no
termination condition without knowing the theoretical ceiling.
Established one before continuing the kernel-by-kernel sweep.

## Method

DeepSeek-V4-Flash public architecture facts (web search, cross-referenced
across 5+ independent sources published June-August 2026): 284B total
parameters, 13B active per token (MoE, top-6 of 256 routed experts + 1
shared expert), 43 layers — the layer count matches the cluster's own
`/state` endpoint model card. Mixed FP4 (MoE expert weights) / FP8 (rest)
quantization per the model's HF card.

Real on-disk footprint from the cluster's own `/state` endpoint:
166,878,536,440 bytes = 155.4 GB for the full 284B-param model — giving
an effective 0.588 bytes/param average (consistent with the mixed
FP4/FP8 quant scheme, not pure FP8's 1.0 byte/param).

Applying that same effective ratio to the 13B active parameters (a
conservative choice — the active set is expert-weight-heavy, which ships
in the LOWER-precision FP4, so if anything this overestimates active
bytes): 13B × 0.588 ≈ 7.11 GB total active weight bytes per decode step,
split across TP=2 → 3.56 GB/token/node.

At the M4 Max's public unified-memory-bandwidth spec (546 GB/s):
bandwidth-bound roofline = 3.56 GB ÷ 546 GB/s ≈ 6.51 ms/token/node.

Real observed baseline decode from tonight's A/B probes: 18.3 tok/s ≈
54.64 ms/token.

## Result

**Observed decode is 8.4× slower than the bandwidth-bound roofline —
decode is running at ~12% of theoretical peak.** This rules out "decode
is already near the memory-bandwidth ceiling, nothing left to claim" as
an explanation. There is real, large headroom.

The gap (48.13 ms/token) is consistent with dispatch/kernel-launch
overhead, not raw compute or bandwidth: at the ~100-200µs/dispatch
overhead figure already documented in this repo's own code
(`FusedSwitchGLU`'s docstring, referring to MiniMax's 62-layer case:
"each with ~100-200 µs of dispatch+sync overhead"), the 48.13ms gap
implies roughly 320 dispatches/token at 150µs each. A rough per-layer
GPU-op count for DSv4's decode path (proj_qkv, rope×2, kv_cache update,
sdpa, o_proj, residual×2, norm×2, moe.gate, switch_mlp's 2-3 gather_qmm
calls, post_combine, all_sum) — roughly 10-15 ops × 43 layers ≈ 430-650
dispatches/token — is the same order of magnitude as the implied count.
Consistent, not proof, but a strong signal.

## Implication for the rest of tonight's campaign

This directly explains WHY the MoE gate+up fusion (halving MoE's
dispatch count from 3→2 gather_qmm calls per layer) produced a real
+3.01% decode win despite doing mathematically identical work — it cut
real dispatch-overhead time, and dispatch overhead is apparently the
dominant cost, not the underlying compute.

**This reprioritizes the remaining search: dispatch-count-reducing
fusions (the gate+up pattern) are the highest-leverage remaining lever
class, not deeper kernel-internals optimization of already-cheap ops.**
Per Fable's flagged analogue: DSv4's Q/KV-down projections (`wq_a` +
`wkv`, both consuming the SAME input activation `x`, currently two
separate `nn.Linear`/matmul calls per `_project_qa_kv`) are a direct
structural match to the gate+up pattern and the next candidate to test.

Also flagged for future work, not started this session: an Instruments
Metal System Trace of one real decode step, to directly observe GPU
idle/dispatch-gap time rather than infer it from this back-of-envelope
roofline. The sync-span Python profiler used earlier tonight serializes
the graph at span boundaries by design and can misattribute or hide
genuine pipeline-bubble time — this roofline estimate is a useful prior
but not a substitute for a real trace.
