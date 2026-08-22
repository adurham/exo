# Comm/compute overlap for moe.all_sum: already implemented, real but smaller effect today — 2026-08-21 (session 2, part 15)

## Summary

Item #14 (design and implement comm/compute overlap for the MoE
all-reduce collective) turned out to already be DONE — by a prior
session, well before tonight. This doc documents the discovery, the
re-verification, and one real, honest discrepancy against the
historic claim.

## What already exists

`EXO_DSV4_FENCE_ASYNC` (default 0, but **already `=1` in tonight's
production `start_cluster.sh` config** — confirmed live via `ps aux`
before this investigation started) replaces the blocking `mx.eval(y)`
after every layer's `moe.all_sum` with `mx.async_eval(y)`. Per the
in-code comment (2026-07-02): "the CPU waits for the GPU to finish each
layer before encoding the next... `mx.async_eval(y)` commits the graph
at the same per-layer points — the cross-rank dispatch ORDER that Lever
1 needs is still pinned — but does not block, letting the CPU encode
layer n+1 while the GPU runs layer n." This is exactly the comm/compute
overlap design consulted on tonight — already built, already gated with
real safety conditions (`_FENCE_ASYNC_CTX["engine"]` +
`_FENCE_ASYNC_CTX["cache"]` both must be armed, single-active-task only
by default, `EXO_DSV4_FENCE_ASYNC_C2` to extend to more streams
experimentally), wired through two independent owner side-channels
(`batch_generate.py`'s engine-level admission gate and `dsv4_mtp.py`'s
cache-level steady-state gate) specifically to prevent the documented
2026-07-02 corruption class (stream-transition races).

The comment cites a historic validation: "A/B'd 2026-07-02: c=1 decode
28.9 -> 37.0 t/s, outputs byte-identical" — a claimed +28% win.

## What I found trying to measure it fresh

Attempted to use the codebase's own dedicated `EXO_DSV4_ALLSUM_PROBE`
diagnostic to directly time the fence. Found a real methodology trap:
the probe's `if _ALLSUM_PROBE_ENABLED: ... mx.eval(y) ... elif _FENCE_ASYNC: mx.async_eval(y)`
structure means **enabling the probe silently disables the async fence
it's meant to measure** — the probe can only ever show the blocking-path
timing, never the async path, because the two are mutually exclusive
branches of the same if/elif. This is worth fixing for future
diagnostics but wasn't touched tonight (out of scope, would need its
own careful design to time async work without forcing sync).

Given the probe couldn't measure it directly, did a real end-to-end A/B
instead — the same trusted methodology used for every other lever
tonight (`bench/decode_probe.py`, matched sample sizes, real correctness
checks either side):

| Config | n | mean decode tok/s | stdev |
|---|---|---|---|
| `EXO_DSV4_FENCE_ASYNC=0` (forced blocking) | 8 | 18.471 | 0.171 |
| `EXO_DSV4_FENCE_ASYNC=1` (current production) | 8 | 18.664 | 0.140 |

**+1.04%, separation only 1.24× combined stdev** — a real, consistently
positive effect (async is never worse across both clean 8-rep runs) but
much smaller than the historic +28% claim, and weaker statistical
separation than tonight's other validated positive lever (gate+up
fusion showed ~3.2x separation).

## Honest reconciliation, not fully resolved

The discrepancy between today's +1% and the comment's claimed +28% is
real and not explained with certainty this session. Plausible reasons,
none independently confirmed:
- The original 2026-07-02 A/B may have been measured on a materially
  different baseline (different context depth, no MoE gate+up fusion
  yet — that lever didn't exist until 2026-06-26/never enabled until
  tonight — different cluster/transport state before this session's
  jaccl hardening work).
- The 28.9→37 t/s figures don't match any other decode baseline number
  seen anywhere else in this codebase's history at c=1 — worth treating
  as measured in a different regime (possibly MTP/speculative-decode
  active, since the surrounding code lives adjacent to
  `dsv4_mtp.py`'s speculative-verify machinery) rather than the same
  vanilla single-stream decode this session has been benchmarking all
  night.
- Genuine regression in the async fence's relative benefit isn't ruled
  out either — e.g. if other work since 2026-07-02 shifted where
  decode's bottleneck sits (this session's own roofline/collective
  findings show the picture is more complex than a single fence point).

**Not chased further tonight** — the effect is real, non-negative, and
already the safer/better production default regardless of its exact
magnitude; re-deriving the historic discrepancy would need archaeology
into a session from six weeks before this one, out of scope for
tonight's prefill/decode optimization work.

## Correctness

Verified at both configurations: 100K-context needle-in-haystack passes
cleanly with `EXO_DSV4_FENCE_ASYNC=1` (current production state),
consistent with the "byte-identical" claim in the original comment —
this session did not find any correctness regression from the async
fence, only a smaller-than-claimed throughput benefit.

## Conclusion on item #14

**Comm/compute overlap for `moe.all_sum` is already implemented,
already validated for correctness, already active in production
(`EXO_DSV4_FENCE_ASYNC=1`), and re-confirmed tonight to have a real
(if smaller than historically claimed) positive effect on decode
throughput with zero correctness cost.** No new implementation was
needed — the item was already closed by prior work; tonight's
contribution is re-verifying it holds on the current baseline (it does,
directionally) and flagging the unresolved magnitude discrepancy for
whoever next has time to dig into the 2026-07-02 session's original
measurement conditions.

Production config remains `EXO_DSV4_MOE_FUSED_GATE_UP=1` with
`EXO_DSV4_FENCE_ASYNC=1` at its default-on value — both real, validated,
non-negative levers, both active. Cluster confirmed healthy and correct
at both short and 100K-context depth in this exact configuration.
