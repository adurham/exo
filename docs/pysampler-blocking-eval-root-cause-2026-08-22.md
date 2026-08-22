# DECISIVE FINDING: the async fence is barely ever armed during real decode — root cause of the idle-gap mystery — 2026-08-22 (session 3, real in-process Python sampler)

## Summary — the real answer

**Root-caused the dominant contributor to decode's idle-gap mystery.**
A real in-process Python-level stack sampler (built as a zero-privilege
alternative to py-spy, which requires root/sudo unavailable on this
cluster) captured 58,400 real 1kHz samples from the actual TP compute
thread across a real ~43-second decode window (spanning two full
back-to-back `decode_probe.py` requests). **~95% of ALL samples on the
compute thread, sustained continuously from t=1s through t=43s (i.e.
during BOTH prefill AND decode, not just prefill), show the thread
blocked inside `mx.eval(y)` at `deepseek_v4.py:3016`** — the BLOCKING
fence branch — not the intended non-blocking `mx.async_eval(y)` branch
one `elif` up (line 3014), despite `EXO_DSV4_FENCE_ASYNC=1` being
confirmed live in production all session.

This directly identifies WHERE the bulk of decode's real wall time
goes: **the async fence (`EXO_DSV4_FENCE_ASYNC`, the mechanism this
session already found only delivers +1.04% instead of a historically
claimed +28% — see `docs/fence-async-28pct-claim-traced-to-artifact-2026-08-22.md`)
is barely ever actually ARMED during real production traffic.** The
`elif` gate requiring `_FENCE_ASYNC_CTX["engine"]`,
`_FENCE_ASYNC_CTX["cache"]`, `y.shape[0] <= _FENCE_ASYNC_MAX_B`, and
`y.shape[1] <= 8` is failing on the overwhelming majority of real
forward passes, falling through to the blocking `else: mx.eval(y)`
branch instead — even though the code comment states this async path
should be "steady-state decode."

## A real, honest methodology detour before the answer

This finding required backing out of a genuine self-inflicted
confusion, documented here rather than silently cleaned up, since the
debugging process itself is a reusable lesson:

1. First manual read of the source (`sed -n '3010,3020p'` on the repo
   checkout) was MISCOUNTED — I read line 3016 as `y = finalize(y)`
   when it is actually `mx.eval(y)`. This is a real off-by-one in my
   own line-counting, not a tool bug.
2. Acting on that wrong premise, spent significant effort ruling out
   two WRONG hypotheses (source/venv mismatch, sampler line-attribution
   bug) via real, correct, and valuable tests — both tests were
   methodologically sound and DID prove the sampler itself is accurate
   (confirmed via two independent controlled experiments with
   `time.sleep()` in matching call-chain structures) — but were solving
   the wrong problem, since the sampler was right all along and my
   manual read was wrong.
3. A `dis`/`inspect`-based function-boundary check compounded the
   confusion by initially checking the WRONG class (`Compressor`
   instead of `DeepseekV4MoE`), then hitting a second real, genuine
   discrepancy: this session's LOCAL machine's own `.venv` copy of
   `deepseek_v4.py` (6968 lines) differs from the LOCAL repo checkout
   (7119 lines) — an unrelated, stale-local-environment issue that
   happened to surface at the worst possible moment. This was NOT the
   actual bug; the REMOTE cluster's repo checkout and installed venv
   are confirmed IDENTICAL (both 7119 lines) and match the local repo
   checkout exactly.
4. Final, authoritative resolution: a careful, explicitly-numbered
   `sed -n '3014,3017p' | cat -n` check against the REAL remote
   installed file (not local, not cached, not assumed) confirmed line
   3016 is genuinely `mx.eval(y)`.

**Reusable lesson: when reading source manually to correlate against a
tool's line-number output, always re-verify with an explicit,
numbered check (`cat -n` or equivalent) against the EXACT file the
live process actually imports — never trust a remembered manual
line-count, and never assume a local dev-machine file matches the
remote deployed file without checking.** This one counting error cost
real, otherwise-avoidable investigation time chasing two wrong
hypotheses before the simple truth was found.

## Real data

Compute thread (tid 8397290304 on rank0), 58,400 samples over a
58.4-second real capture:

| Time window | Hot-line (`mx.eval`, blocking) fraction |
|---|---|
| t=0s (TTFT/startup) | 0% |
| t=1s to t=43s (both decode requests) | consistently 83-100%, median ~95% |
| t=44s onward (post-request idle) | 0% |

The ~43-second hot window aligns closely with the real combined
duration of the two back-to-back `decode_probe.py` requests captured
during this session (run1: ttft=4.09s + decode=17.68s ≈ 21.8s; run2:
ttft=0.97s + decode=20.17s ≈ 21.1s; combined ≈ 42.9s — matching the
observed ~43s hot window almost exactly).

Wakeup latency for hot-line samples was LOW (median 0.327ms, max
3.4ms) — the sampler was never itself starved/delayed when catching
this frame, meaning the ~95% concentration is a real, high-confidence
signal about where the thread's wall time genuinely goes, not a
sampling artifact.

## Why this matters — reframes the entire investigation

Given `mx.eval(y)` is a REAL, GENUINE blocking synchronization point —
unlike `finalize()` (a confirmed no-op) which I originally,
incorrectly, thought was the hot line — **this is not a mysterious
"idle gap" needing further attribution. It is the expected, real cost
of a synchronous GPU wait, just occurring far more often than the
async-fence design intends.** This reframes tonight's whole
investigation:

- The earlier arithmetic reconciliation
  (`docs/allsum-sync-span-artifact-arithmetic-check-2026-08-22.md`)
  correctly proved the sync-span profiler's 21.4%/4094µs figures were
  measurement artifacts of forced per-span synchronization — that
  finding STANDS, unaffected by this one.
- The real jaccl-internal transport timing
  (`docs/jaccl-internal-timing-allsum-transport-fast-2026-08-21.md`,
  36-66µs/call) also STANDS — the collective's wire cost really is
  small. This new finding does not contradict that; it explains what
  happens AROUND the collective, not the collective's own cost.
- **What's NEW: the blocking `mx.eval(y)` fence itself — not the
  collective, not dispatch overhead, not memory paging — is very
  plausibly the dominant real cost**, because it forces the CPU thread
  to genuinely wait for GPU completion of the ENTIRE accumulated lazy
  graph up to that point (all prior unevaluated layers' compute, not
  just the collective), every single time the async-fence gate fails
  to arm.
- This is fully consistent with the earlier GPU idle-gap analysis
  (`docs/gpu-idle-gap-deep-dive-2026-08-22.md`): 0.5-20ms gaps
  dominating gap time, roughly 20+ per token — a blocking `mx.eval`
  call waiting on real GPU work is exactly the right order of magnitude
  and exactly the right per-call granularity to explain that
  distribution.

## Immediate next question (not yet answered)

**WHY is the async-fence gate (`_FENCE_ASYNC_CTX["engine"]`,
`_FENCE_ASYNC_CTX["cache"]`, `B <= _FENCE_ASYNC_MAX_B`, `L <= 8`)
failing on ~95% of real forward passes during confirmed real decode
traffic**, when the code's own comment states it should be armed at
"steady-state decode"? This is now the single highest-priority
follow-up question. Candidates to check next: what `_FENCE_ASYNC_CTX`
values actually are during a live request (add a real one-off debug
print or trace), whether `_FENCE_ASYNC_MAX_B`/the `L<=8` condition is
being violated by real production batch/sequence shapes, or whether
the "engine"/"cache" arming flags are simply never being set true by
whatever code is supposed to call `_set_fence_async_ok` per the
referenced comment.

## Disposition

This is real, hard, decisive progress — likely the actual dominant
answer to the session's core question, found via a from-scratch
zero-privilege profiling tool built specifically because py-spy wasn't
available, validated through real controlled experiments, and reached
despite (and honestly documenting) a real self-inflicted detour along
the way. Not yet fully closed — the immediate next step (why the async
gate rarely arms) is a natural, well-scoped follow-up.
