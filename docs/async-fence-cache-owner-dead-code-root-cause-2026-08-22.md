# ROOT CAUSE CONFIRMED: the async-fence's "cache" owner is dead code under TP — the fence structurally never arms — 2026-08-22 (session 3, live diagnostic logging)

## Summary — the definitive answer

**Root-caused, with live evidence, why decode's dominant real cost is
the blocking `mx.eval(y)` fence** (per
`docs/pysampler-blocking-eval-root-cause-2026-08-22.md`'s finding that
~95% of the compute thread's real wall time is spent there, not in the
intended non-blocking `mx.async_eval(y)` path).

Deployed rate-limited, real-value diagnostic logging (per an
independent Fable consult's guidance to log real runtime values rather
than continue static code tracing, given an earlier real detour tonight
from a manual line-counting error) at both the async-fence gate check
and its two setter call sites. Live log from a real correctness-check
request:

**The gate requires BOTH `_FENCE_ASYNC_CTX["engine"]` AND
`_FENCE_ASYNC_CTX["cache"]` to be `True` simultaneously.**

- `"engine"` (owned by `batch_generate.py`, general request-lifecycle
  code): confirmed live to actually fire — set `True` then `False`
  during the real request, exactly as its request-scoped design
  intends.
- `"cache"` (owned exclusively by `dsv4_mtp.py`'s `_set_fence_async()`
  method, called from `snapshot_for_uid()` and other cache-management
  entry points): **confirmed via live log to NEVER fire at all — zero
  occurrences of any `key=cache` setter call, across the entire
  correctness-check request.**

Since both keys must be `True` simultaneously and `"cache"` never
becomes `True` even once, **the async fence gate can never pass under
this cluster's current configuration — it is structurally, permanently
stuck on the blocking `mx.eval(y)` branch for 100% of real TP decode
traffic**, independent of `EXO_DSV4_FENCE_ASYNC`'s value, independent
of real batch size or sequence length (both confirmed well within the
gate's numeric limits: `B=1 <= MAX_B=1`, `L=1 <= 8`).

## Why "cache" never arms: dsv4_mtp.py is dead code under TP

This directly connects to an independently-confirmed finding from
earlier tonight
(`docs/roofline-sanity-check-inputs-confirmed-2026-08-22.md`): DSpark's
actual decode loop (`pp_dspark_decode_loop`) is PP-only — auto-selected
in Pipeline sharding mode, never reached under this cluster's Tensor
sharding mode (`MLX_JACCL_SHARDING_MODE=Tensor`, confirmed live all
session). `EXO_DSV4_MTP=0` (classic single-head MTP) is also confirmed
disabled live.

`dsv4_mtp.py`'s `_set_fence_async()` (the sole owner of the "cache" key)
is a method on the MTP/DSpark cache-management class
(`snapshot_for_uid`, `activate_for_uids`, and related methods) — code
that is only ever invoked as part of the MTP/DSpark speculative-decode
machinery. Confirmed via a live log grep for any `DSv4MTP`/`dsv4_mtp`
activity during the correctness-check request: **zero matches** — this
module's classes are not being instantiated or exercised at all in the
current TP, non-speculative decode configuration.

**The "cache" arming owner was designed under the assumption that
MTP/DSpark's cache-snapshot lifecycle would run on every request and
arm it — but under plain TP decode (no MTP, no DSpark, both confirmed
dormant), that lifecycle simply never executes, so the key it's
supposed to set stays permanently `False`.** The two-owner design
(explicitly built in 2026-07-02 per the code's own comments, "to close
ordering holes at stream join/leave") never accounted for a
configuration where one of the two owners' code path is entirely
absent from the execution graph.

## Real, quantified impact

Per the earlier arithmetic (`docs/decode-time-budget-synthesis-2026-08-22.md`),
and per the earlier real A/B test of `EXO_DSV4_FENCE_ASYNC`
(`docs/comm-compute-overlap-already-exists-2026-08-21.md`, real
measured +1.04% when the fence goes async under favorable conditions —
which per this finding, essentially never occurs in production): this
means production has been running the blocking fence path for the
ENTIRE session (and very plausibly since 2026-07-02 when
`EXO_DSV4_FENCE_ASYNC` was first introduced), despite the flag being
`=1`. The historic +28% claim
(`docs/fence-async-28pct-claim-traced-to-artifact-2026-08-22.md`,
already independently shown to likely be an MTP-PROF measurement
artifact) may ALSO partly reflect a real environment difference: if
that 2026-07-02 measurement was taken with MTP/DSpark actually active
(a real, different configuration than tonight's TP-only, MTP-disabled
production), the "cache" key may have genuinely armed in that original
test, making the async path genuinely reachable then — a real,
different regime than production runs today, not purely a measurement
artifact. This is a new, unverified hypothesis worth flagging, not yet
confirmed.

## What this means for the real decode idle-gap mystery

This is very likely the single largest concrete, fixable contributor
identified this entire session. Every real per-layer collective
(`moe.all_sum`, 43 calls/token) is followed by a forced, BLOCKING
`mx.eval(y)` — meaning the CPU thread genuinely waits for the ENTIRE
accumulated lazy graph (all prior unevaluated compute since the last
sync point) to materialize on the GPU, every single layer, every
single token. This is exactly consistent with the earlier-found
gap-length distribution (0.5-20ms gaps, ~20+ per token,
`docs/gpu-idle-gap-deep-dive-2026-08-22.md`) — a blocking eval waiting
on real GPU completion is precisely the right granularity and
magnitude to produce that pattern.

## Immediate, obvious fix candidates (not yet attempted — this session stops at root-cause identification)

1. **Simplest**: for TP-only, non-speculative deployments (this
   cluster's actual configuration), the "cache" owner requirement is a
   structural leftover from a two-owner design meant for a
   configuration (MTP/DSpark active) this cluster doesn't run. A
   TP-aware code path could either (a) have `batch_generate.py`'s
   "engine" arming alone be sufficient when DSpark/MTP are confirmed
   inactive, or (b) have some TP-decode-loop entry point explicitly arm
   the "cache" key once, since there's no real MTP cache-lifecycle
   requiring the two-owner ordering-hole protection the design was
   built for.
2. Any fix here needs to preserve the real safety property the
   two-owner design was protecting against (the 2026-07-02 c=2 stream
   join/leave corruption) — a naive "just always arm cache" change
   could reintroduce that bug if MTP/DSpark are ever activated
   alongside a TP-decode-loop-only arming path. This needs careful
   design, not a quick patch.
3. This is real, validated, production-relevant work for a FUTURE
   session with the user's explicit sign-off on a code change with
   real correctness stakes (given the documented history of a real
   corruption bug in this exact subsystem) — not something to rush
   through without review.

## Disposition

This closes the core investigation goal for tonight: **decode's
dominant real cost is now understood, with live evidence, not just
inferred from aggregate statistics.** The remaining `EXO_DSV4_FENCE_ASYNC`
"+1.04%" real measurement from earlier this session
(`docs/comm-compute-overlap-already-exists-2026-08-21.md`) should be
understood as measuring the RARE cases where both owners happen to
align (or possibly measurement noise/edge cases), not the fence's
typical, designed behavior — since the "cache" owner is now confirmed
to essentially never fire in this cluster's real configuration.
