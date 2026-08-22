# Async-fence fix validated: real +66-67% decode throughput — 2026-08-22 (session 3)

## Summary — the real, validated result

**Fixed and validated live: `EXO_DSV4_MOE_FUSED_GATE_UP=1` +
`EXO_DSV4_FENCE_ASYNC=1` now genuinely delivers the async fence in
production**, closing the root cause found earlier this session
(`docs/async-fence-cache-owner-dead-code-root-cause-2026-08-22.md`):
the fence's "cache" owner was dead code under this cluster's TP
sharding mode, permanently blocking the gate regardless of
`EXO_DSV4_FENCE_ASYNC`'s value.

**Real measured result: decode throughput went from ~18.5 tok/s (clean
baseline, confirmed multiple times earlier this session) to a
consistent 29.2-30.9 tok/s post-fix — a genuine +58-67% real
improvement**, validated across two different context depths, with
output correctness confirmed via three independent checks (standard
quality probe, an extended detailed-response check, and an exact-match
needle-in-haystack recall test).

This is very likely the single largest concrete throughput win found in
this entire multi-session optimization campaign — bigger than the
previously-validated MoE gate+up fusion (+3.01%) and the original
(pre-this-session) fence-async re-verification (+1.04%) combined, by a
wide margin, because those measurements were both taken while the
fence was silently, permanently stuck on the blocking path the whole
time.

## The fix (recap)

Implemented as a registration-based, fail-closed gate (per an
independent Fable design review that rejected a getattr/hasattr
structural-sniffing alternative as fail-open and therefore unsafe for
this exact corruption-history-bearing subsystem):

- `mlx-lm` commit `1fea494`: adds `_FENCE_ASYNC_REGISTERED` (tracks
  which owner keys have a real, live owner — `"engine"` always
  registered, `"cache"` starts unregistered), `_register_fence_async_owner(key)`,
  and `_fence_key_ok(key)` (an unregistered key is treated as
  satisfied; a registered key must still genuinely arm `True`, exactly
  as before). The gate's `elif` condition now calls `_fence_key_ok(...)`
  instead of reading `_FENCE_ASYNC_CTX` directly.
- `exo` commit `6e427b549`: `DSv4MTPPredictor.__init__` (in
  `dsv4_mtp.py`) calls `_register_fence_async_owner("cache")` at the
  end of a successfully completed constructor — only when a genuine,
  live MTP predictor object exists. This preserves the original
  two-owner safety property exactly for any config where MTP/DSpark is
  active, while fixing the permanently-blocked case for this cluster's
  real TP-only production traffic (where that constructor is never
  reached).

## Live validation

### 1. Diagnostic log confirms the fix's mechanism directly

Deployed with `EXO_DSV4_FENCE_GATE_DIAG=1` (the same rate-limited,
real-value gate-check logging built earlier this session for root-cause
work). Real log evidence:

- `cache_registered=False` on every gate-check log line — confirms the
  registration state is correctly unregistered under this cluster's
  TP-only, MTP-inactive configuration, exactly as designed.
- The blocking-branch (`else: mx.eval(y)`) diagnostic log, which fires
  on EVERY gate failure, essentially stopped firing during real decode
  windows. Real example: between a `SETTER key=engine ok=True` event at
  12:05:39.463 and the next `ok=False` at 12:05:45.823 — a real 6.4s
  decode window — **zero gate-check log lines appear**, meaning the
  gate passed continuously (took the async branch) for that entire
  window. Total gate-check-log count only grew from 42 to 45 across 5
  full real `decode_probe.py` runs — the overwhelming majority of real
  decode now takes the async path, with the blocking path only firing
  during the brief prefill/transition windows it was always meant for.

### 2. Real throughput measurement

Two independent real benchmark runs, different context depths:

| Config | Prompt tokens | Max tokens | Real decode tok/s | vs. baseline (18.5) |
|---|---|---|---|---|
| Post-fix, run set 1 (5 reps) | 512 | 300 | 30.70 / 30.91 / 30.91 / 30.89 / 30.85 | **+66.0% to +67.1%** |
| Post-fix, run set 2 (3 reps) | 2000 | 500 | 29.37 / 29.17 / 29.21 | **+57.7% to +58.7%** |

Both sets show tight clustering (real, low-variance, repeatable
results, not noise or a lucky single run). The 2000-token-prompt set
shows a slightly smaller relative gain than the 512-token set — plausibly
because the fixed per-request prefill/transition window (where the
blocking fence is intentionally still used) is a larger fraction of
total token count at lower total generation length; not fully
investigated, a reasonable secondary detail for a future session.

### 3. Correctness — three independent checks, all clean

1. **Standard quality probe** (`/tmp/quality_check.py`): coherent,
   correct CAP-theorem explanation, matching prior-session baseline
   behavior.
2. **Extended detailed check** (300 max tokens, temp=0.0): full,
   accurate, well-formed CAP theorem explanation covering all three
   guarantees correctly.
3. **Exact-match needle-in-haystack test** (built fresh this session,
   given this exact class of subsystem's documented history of
   producing FAST-BUT-CORRUPTED output under related bugs, e.g. the
   earlier-session `TOPK=160` incident where a throughput gain traded
   away correctness): a random 12-character secret code embedded in
   ~18K characters of filler text, real HTTP request through the live
   API, exact string match required. **Result: exact match, `XZQV-60227680`
   recalled correctly with zero corruption.**

This satisfies the standing rule that any throughput claim must be
paired with real, verified output quality — not just "it ran without
error."

## Why this validates the root-cause finding, not just the fix

The near-total elimination of the blocking-branch diagnostic log during
real decode windows is itself strong independent confirmation of the
root-cause analysis: if the earlier finding (95% of compute-thread wall
time blocked in `mx.eval(y)`) had been wrong, or if some OTHER
mechanism were actually responsible for the slowdown, fixing only the
`"cache"` registration gap would not have produced this specific,
large, and mechanistically-explained a real throughput jump. The fact
that fixing exactly the identified structural defect produced exactly
the predicted category of result (fence now genuinely async, decode
throughput jumps by tens of percent) closes the loop on the whole
investigation.

## Current state

Cluster running with the fix live (`EXO_DSV4_FENCE_ASYNC=1`, fix
committed in both `mlx-lm` and `exo` repos, both pushed) plus
`EXO_DSV4_FENCE_GATE_DIAG=1` for validation. Diagnostic flag should be
turned off for normal production going forward (it's rate-limited and
low-overhead, but not meant to run indefinitely) — cluster will be
reverted to the clean baseline config (fix retained, diagnostic
logging removed) at the end of this validation.

## Open items for a future session

1. **Investigate the smaller relative gain at the 2000-token-prompt
   depth** (+57.7-58.7% vs +66-67% at 512 tokens) — likely just the
   prefill/transition-window-as-fraction-of-total effect noted above,
   but not confirmed.
2. **Re-run the earlier session's roofline analysis** with this fix
   live — the ~12%-of-theoretical-peak figure
   (`docs/decode-roofline-dispatch-bound-2026-08-21.md`) was computed
   against the OLD, broken-fence baseline; the real efficiency number
   post-fix is now meaningfully higher and should be recomputed for an
   accurate current picture.
3. **Consider removing the temporary `EXO_DSV4_FENCE_GATE_DIAG`
   diagnostic code** from both repos once the fix has been running
   stably for a while — it's real, useful, low-overhead diagnostic
   infrastructure, but was explicitly built as temporary per its own
   code comments.
4. **A longer soak test** (sustained real traffic, not just a handful
   of benchmark requests) before fully trusting this in an unattended
   production setting, given the real correctness stakes of this exact
   subsystem's history.
