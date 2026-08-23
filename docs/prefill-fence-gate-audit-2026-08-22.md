# T8: Audited prefill's blocking-fence gate for the same failure class as the decode async-fence bug — confirmed genuinely by-design, not a stale re-audit gap — 2026-08-22 (session 4)

## Why this check

A tip from another session flagged: the decode async-fence bug was
"flag says 1, but a silent gate condition made it structurally
impossible to ever engage" (the `cache` owner defaulting to False
forever, invisible without diagnostic logging — see
`docs/async-fence-cache-owner-dead-code-root-cause-2026-08-22.md`).
`PERFORMANCE_HISTORY.md` §1 states prefill "still using the blocking
fence by design" but that claim was asserted in the same doc that
found the decode bug, not independently re-verified post-fix. Worth a
cheap, read-only recheck for the same failure class.

## Method

Read the real gate logic directly in `deepseek_v4.py` (lines 3046-3051,
the exact `elif` branch immediately following the async-fence probe
code):

```python
elif (
    _FENCE_ASYNC
    and _fence_key_ok("engine")
    and _fence_key_ok("cache")
    and y.shape[0] <= _FENCE_ASYNC_MAX_B
    and y.shape[1] <= 8
):
    mx.async_eval(y)
else:
    ...  # blocking mx.eval(y)
```

## Real finding

**The gate has an explicit, always-visible numeric condition:
`y.shape[1] <= 8`.** `y.shape[1]` is the sequence-length dimension of
the layer's output at that call site — for prefill, this equals the
active chunk length (`EXO_PREFILL_STEP_SIZE`, standing default 2048,
historically tested at 4096/8192 too — see §3's chunk-size-tuning
history). **Every real prefill chunk length ever used in this codebase
is at least 64x larger than 8** — this condition trivially and
structurally fails for every single prefill call, by construction,
completely independent of the `"engine"`/`"cache"` owner registration
state that caused the DECODE-side bug.

This is a genuinely **different failure class** than the bug this
tip's pattern was based on:
- The decode `"cache"` bug: an *invisible*, multi-owner boolean flag
  defaulted to `False` forever because nothing ever registered as its
  owner — required diagnostic logging (`EXO_DSV4_FENCE_GATE_DIAG=1`)
  and a real stack-sampler investigation to discover.
- Prefill's gate: a *single, explicit, immediately-visible* numeric
  shape comparison, directly readable in the same `and` chain, with no
  hidden state, no owner-registration indirection, and no possible
  silent-default failure mode — reading the code once settles the
  question completely.

## Conclusion

**Confirmed: prefill's "blocking fence by design" claim in
`PERFORMANCE_HISTORY.md` §1 is correct and load-bearing, not a stale
assumption that needed re-auditing.** The tip's underlying concern (a
reasonable one, given the real precedent) does not reproduce here —
there is no silent gate for prefill to be broken by. This closes the
question cheaply and definitively; no further investigation needed on
this specific angle.

## What this does NOT establish

This confirms the gate correctly EXCLUDES prefill from the async-fence
path — it says nothing about whether prefill's compute-bound wall time
itself has other headroom (that is exactly T7's question: a real
FLOPs-based roofline check, not yet done, is the correct next step for
prefill, analogous to the bandwidth-roofline work that found decode's
headroom in the first place).
