# PREFILL ROUND 3 — PM report (feedback-loop round 3)

**Date:** 2026-09-02 · **exo repo:** `17d427b01` (main, unmodified) · **PM:** delegation subagent
**Client change shipped:** hermes-agent `bdc9b6f1fc` (committed + pushed, FORK.md entry included)
**Mode:** A executed (offline + 3 live API calls, ~0 GPU cost). B PREP-ONLY. C DESIGN-ONLY.
**Pre-registration:** `PRE-REGISTRATION.md` in this directory, written BEFORE any measurement.

---

## HEADLINE

**The pad-strip gate PASSED on live evidence, the fix shipped, and the re-probe now PASSES the
identical round-2 bands — but Fix B is CONDITIONAL-GO, not GO.**

The strongest result this round is not the re-probe. It is a direct, non-proxy observation from
the live server's own accounting: with the pad present the server reported
`cached_tokens = 0`; with the pad absent it reported `cached_tokens = 351`. **The pad
demonstrably destroyed real prefix-cache reuse in production, measured by the server itself,
entirely independent of our probe.** That is the mechanism proof.

The re-probe's role is narrower and should not be oversold: it confirms the pad was the *sole*
cause of the round-2 zeros in this dataset. That was a falsifiable prediction — had only 5 of 7
zeros flipped, the causal story would have been incomplete. All 7 flipped and nothing else moved.

| # | Item | Status | Result |
|---|---|---|---|
| A1 | Live API verification (a/b/c) | **DONE** | **GATE: PASS** — all four criteria, live matches offline exactly |
| A2 | Provider-scoped pad-strip | **SHIPPED** | `bdc9b6f1fc` — additive, exo-only, fails safe, 40 tests green |
| A3 | LCP re-probe, identical bands | **DONE** | **PASS** (median 1.000, p10 1.000) — but under an optimistic proxy |
| B | SDPA 2-length timing | **PREPARED** | Patch + runbook + analyzer ready; NOT run. ~15-20 min cluster ask |
| C | Decode instrumentation | **DESIGNED** | Patch applies cleanly; all 4 pitfalls handled structurally; NOT applied |

---

## A1. API VERIFICATION — the gate for the pad-strip

Three requests to the live exo server (`http://192.168.86.201:52415/v1`, model
`deepseek-ai/DeepSeek-V4-Flash-0731`), identical except a prior assistant message's
`reasoning_content`. Both halves were required and both were done: an offline render through
exo's real encoder path, and live confirmation on the server.

| variant | `reasoning_content` | offline tokens | **live `prompt_tokens`** | **live `cached_tokens`** | HTTP |
|---|---|---:|---:|---:|---|
| a | key ABSENT | 353 | **353** | 0 | 200 |
| b | `""` | 353 | **353** | **351** | 200 |
| c | `" "` (today's client) | 354 | **354** | **0** | 200 |

**Live matched offline exactly — 353/353/354 predicted, 353/353/354 observed.** No constant
offset, no disagreement. The server is not doing anything the offline path does not.

**The divergence is exactly one inserted token.** (a) vs (c) differ by a single space token
(id 223) at index 294. Removing index 294 from (c) yields (a) *exactly*: common prefix 294 +
common suffix 59 = 353 = len(a). Rendered text differs only as `<think></think>` vs
`<think> </think>`. Variants (a) and (b) are byte-identical token-id lists.

**The `cached_tokens` column is the find of the round.** It was present in the raw responses but
missed in the worker's own summary. It is a direct production measurement of the defect: the
padded variant got zero cache reuse; the unpadded variant hit 351 cached tokens against the
identical prefix. This is independent of our reconstruction pipeline entirely.

### Gate criteria — evaluated verbatim against the pre-registration
- **G1 no error** — PASS. Three HTTP 200s, well-formed completions.
- **G2 absent is clean** — PASS. No `None`/`null`/`NoneType`, no dangling or empty reasoning
  delimiters, no duplicated or dropped turn markers in variant (a).
- **G3 divergence is localized** — PASS. Exactly one inserted token, zero re-segmentation of
  surrounding text, no cascade. Live `prompt_tokens` deltas corroborate the offline token diff.
- **G4 absent is not worse** — PASS. (a) is one token *shorter* than (c), never longer.

**GATE: PASS.** Fable's named "silent mishandling" risk — a template folding the field into the
prompt such that three variants produce three materially different sequences — is **ruled out on
live evidence**, not on a code read.

---

## A2. THE PAD-STRIP — what shipped

**Commit `bdc9b6f1fc`** on `adurham/hermes-agent` main, pushed (local and origin HEAD match).
5 files, +409/-11: 3 source files, the new test file, and the FORK.md entry.

Scoped exactly as directed — **additive, at message-serialization time, keyed off the provider
identity the client already resolves.** Not a global config flag. Not a blocklist.

New predicate `omits_reasoning_pad_for_provider()` matches only `exo` / `custom:exo`, threaded as
an `omit_pad` kwarg (default `False`) through the existing reasoning-policy pathway. All call
sites already had `agent.provider` in scope — no new plumbing was threaded.

**All four `reasoning_content` pad-emission sites were found and handled:**
`chat_completion_helpers.py:2282` (build-time pad), `message_sanitization.py:960/983/1008`
(policy pad-injection), `message_sanitization.py:1110` (echo re-pad).

**PM-verified fail-safe behavior** (I ran the predicate directly, not from the worker's report):

```
exo -> omit    custom:exo -> omit    EXO -> omit
anthropic, ollama-cloud, openai, some-new-provider, None, "", exotic, exo-remote -> UNCHANGED
```

Exact-match only. `exotic` and `exo-remote` do **not** match — no prefix/substring bug. Every
unverified provider keeps today's behavior byte-for-byte.

**Tests: 40 passed**, run by me across the full reasoning/sanitization surface, not just the new
file. Genuine non-whitespace reasoning is still echoed verbatim on exo — the fix strips only the
synthetic pad.

*Correction to the worker's own claim, carried up rather than buried:* it reported assertion 3 as
"anthropic still emits the pad." That is wrong, and the worker caught it itself — anthropic and
unknown providers do not pad today (they strip). The correct statement is that their behavior is
**unchanged**, which is what the assertion was actually protecting. `ollama-cloud` is the non-exo
provider that genuinely still pads, and it still does.

---

## A3. LCP RE-PROBE — identical bands, only the pad handling changed

Same 54 pairs, same script, same tokenizer, same variant A/B bounding, same percentile
conventions. **Every number below was recomputed by me from the raw per-pair JSON.**

### Methodology-drift control (this is what makes the comparison trustworthy)
Re-running round 3's script with the pad *still in* reproduces round 2 **exactly**: 7 zeros,
47 ones, mean 0.8704, p10 0.0000. **Zero drift.** The only thing that changed is the pad.

### Distribution (variant A, primary)

| run | min | p10 | median | mean | max | zeros / ones / between |
|---|---:|---:|---:|---:|---:|---|
| round 2 (pad) | 0.0000 | **0.0000** | 1.0000 | 0.8704 | 1.0000 | 7 / 47 / 0 |
| round 3 replay (pad) | 0.0000 | 0.0000 | 1.0000 | 0.8704 | 1.0000 | 7 / 47 / 0 |
| **round 3 (pad omitted)** | 1.0000 | **1.0000** | 1.0000 | 1.0000 | 1.0000 | **0 / 54 / 0** |
| round 3 (pad as `""`) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 / 54 / 0 |

p10 = 1.0000 under **all three** percentile conventions (linear, nearest-rank, inclusive) — the
gate does not hinge on a convention, same discipline as round 2.

**The 9 no-reasoning pairs:** all 7 padded pairs went 0.000 -> 1.000; both `null` pairs stayed
1.000. The omit and empty-string variants are identical, so the result is robust to whichever
form the client fix took.

### Variant B — the pessimistic lower bound
Variant B treats the 8 multi-invoke tool-call pairs as unpinned (their exact wire shape is not
persisted). **It also passes:** median 1.0000, p10 0.8431, min 0.5715, mean 0.9702. The residual
sub-1.0 mass is the known multi-invoke ambiguity, unrelated to the pad.

### VERDICT AGAINST THE BANDS
Median 1.000 >= 90% **AND** p10 1.000 >= 70% -> **PASS**, on both the primary and the
pessimistic lower bound.

### OPTIMISM HANDLING — option (b), chosen and registered in advance
Fable required this be handled, not waved off. **I took option (b): explicitly downgrade the
claim and file the proxy fix as a shipping prerequisite.**

Option (a) — persist raw decode ids so the probe stops being optimistic — **was not available for
this dataset.** The 54 pairs came from a session that already ended; raw decode token ids for
those turns exist nowhere (`state.db`, `requests.jsonl`, the exo event-log archive were all
searched in round 2) and cannot be created retroactively. Fable also explicitly preferred the
same-data re-probe over waiting for a new session, since n=9 pad-related pairs is too thin for a
statistical claim and the mechanism is what convinces. Same-data therefore forces (b).

**The claim this round supports is exactly: "FAIL is ruled out under optimistic reconstruction."
It is not "Fix B works in production."**

### On the perfect separation — the honest reading
7 zeros -> 0 zeros looks emphatic. It should not be leaned on rhetorically, for a reason worth
stating plainly: **a score of exactly 1.0000 across all 54 pairs is the predicted outcome under
*both* hypotheses** — "the system is now clean" *and* "the proxy is blind to every remaining
divergence." The proxy has no resolution in precisely the regions where problems could still
hide (decode-time BPE segmentation quirks; the tool-call region, which matches *by construction*
because both sides render from the same parsed arguments).

So the perfect separation is a necessary checkbox, not a strength. **The probe can no longer
detect any problem, and the probe's detection floor is now the binding constraint.** That is
exactly what C1 exists to fix.

The claim is *not* tautological, though — two checks rule that out. The fix changed the
production client, **not the measurement instrument** (the replay proves the instrument is
unchanged), and variant B still produces 0.5715 and 0.8431, so 1.0 is not structurally forced.

---

## FIX B VERDICT: **CONDITIONAL-GO**

Not GO. The pre-registration fixed this ceiling before any measurement: a PASS under an
optimistic proxy can never produce a GO, because the PASS inherits the proxy's optimism. A FAIL
would have been robust; a PASS is soft. That asymmetry was registered in advance precisely so a
cheap re-probe could not silently upgrade itself into a shipping-grade signal.

### The pre-registered conditions

**C1 — de-optimize the proxy (correctness gate).** Persist raw decode token ids for >= 1 fresh
session; re-run this probe against **real ids, not reconstructions**; pass these same bands.
*Amended after results, labeled as such in the pre-registration: the fresh session must contain
multi-invoke tool-call pairs.* Without that, C1 could be satisfied by a short, tool-light session
that never exercises the region the current proxy is structurally blind to — satisfying the
condition while routing around the blind spot. The amendment only makes C1 harder.

**C2 — establish the fleet value (value gate).** The headline 21.9% is a *long-session* figure
from n=1 session, not a fleet figure. Round 2 measured it as strongly length-dependent: 2.09% at
2 turns, 4.23% at 6, 12.25% at 31, reaching 21.91% only at 55. A session-length distribution over
N >= 20-30 sessions is needed. **A mechanism that works is not the same as a mechanism worth
shipping**, and a clean mechanism result must not be allowed to launder the headline number.

Both conditions are named up front rather than discovered later.

---

## B. SDPA 2-LENGTH TIMING — prepared, NOT run

Implements fable's design exactly: **two** single-context-length runs (12K and 64K), SDPA
**directly** wrapped in `perf_counter`, n >= 5 calls per length, **per-call absolute time**
reported. No sweep. No resurrection of the original attribution methodology.

**Deliverables (all verified by me):**
- `artifacts/sdpa_2length_timing.patch` — **applies cleanly** to HEAD (`git apply --check`), and
  is **not applied**; `src/` and `mlx-lm/` are clean.
- `artifacts/sdpa_2length_run.sh` — exact copy-pasteable commands, both arms. `bash -n` clean.
- `artifacts/sdpa_2length_analyze.py` — compiles; separates warmup from steady-state, computes
  per-call absolute times, and **hard-fails the reductio** (calls x per-call must be <= wall).
- `findings/b-sdpa-2length-prep.md` — design, citations, runbook, what each outcome decides.

**Call site:** the `attn.sdpa` span in `mlx-lm/mlx_lm/models/deepseek_v4.py:4865`
(`SparseCompressedAttention.__call__`), instrumented at entry and after `out = finalize(out)`
(`:5018`), tagged per branch. The ratio compares `local` (2048 rows) vs `sparse` (1024 rows) —
recovering the same 1024->2048 per-rank row doubling that produced the round-2 4.06x **without
varying step size**. `attn.sdpa.compressed` (`:4478`, Apple's fused kernel, already at ceiling) is
explicitly excluded.

**Lazy-eval handled:** the existing `finalize`/`span` helpers are no-ops when no profiler hook is
registered, so the probe does its own explicit `mx.eval(out)` immediately before the end
timestamp. Enqueue-vs-execute is not being measured.

**Documented gotchas baked into the runbook:** `EXO_PROFILER_SYNC_SPANS=1` paired with
`EXO_RUNNER_HANG_TIMEOUT_SECONDS=600`; batched path only (rejects any run lacking
`Starting batched prefill:`); no SIGUSR1; the chunk-halving `// min(4, group.size())` is PP-loop
only and does not apply to `prefill_batched`.

**Pre-registered decision rule (fixed before any timing exists):** real multiplicative constant
if the ~4.06x per-call ratio holds at both lengths (both in [3.0, 5.0], and
|R(64K)-R(12K)|/R(12K) <= 25%) — then it matters at 250K. Fixed-overhead artifact if R(64K) <= 2.2
or R(64K) < 0.6*R(12K) — then it doesn't. Otherwise INDETERMINATE, reported as such with no story
attached.

**Cluster ask: ~15-20 min, two short runs.**

---

## C. DECODE INSTRUMENTATION — designed, NOT applied

Direct per-collective timestamps for candidate A (`dsv4_mtp.py` fenced coord collectives) and
candidate B (`batch_generator.py` `agree_on_tasks` / `agree_on_cancellations_fast`).
`artifacts/decode_instrumentation.patch` **applies cleanly** and is **not applied**.

**Line ranges re-verified at current HEAD `17d427b01` — zero drift** from round 2's `80db9a855`
(the intervening commits are docs-only). A: gate `:2259`, all_sum `:2269`, all_max `:2306-2308`.
B: **BatchGenerator** copies at `:507` / `:561` — the `SequentialGenerator` duplicates at
`:159`/`:208` were confirmed by grep and deliberately left alone.

Direct timing, not ablation — settled and not relitigated: `agree_on_tasks` is the only path that
fills `_queue` (skipping it hangs the cluster), `agree_on_cancellations_fast` is load-bearing
(skipping it reintroduces a measured 133.7s cancellation-latency bug), and A's fences guard
against rank drift.

### All four pitfalls handled structurally
1. **Lazy eval** — every timer ends at the interval's *own existing unconditional*
   materialization (A: `mx.eval(counted)` / `mx.eval(synced)`; B: `mx_any`'s internal
   `eval + .item()`, `mx_all_gather_tasks`' internal `.tolist()`). Enqueue is never measured.
2. **Cross-rank clock skew** — records carry duration-µs + rank id + per-process seq only.
   **No absolute timestamp exists anywhere in the schema**, making cross-rank arithmetic
   structurally impossible rather than merely discouraged.
3. **First-call warmup** — per-instance counters tag first occurrences `warmup`, bucketed out of
   steady-state means.
4. **Disjoint placement** — proven, not asserted: B's collectives fire in `BatchGenerator.step()`
   before `self._gen.step()` (`:849`) or in post-cycle callbacks (`batch_generate.py:4266`); A's
   fence sits inside `_next()` before any forward. `dsv4_mtp.py` has **zero** `agree_on*` call
   sites. **No nesting in either direction**, so no double-counting and no subtraction rule needed.

**The key design finding:** because every measured interval already terminates at a
production-unconditional sync, **no forced `mx.eval` needs to be added**. This sidesteps the
perturbation risk that a forced synchronize would serialize previously-overlapped work — the
exact mechanism behind the documented `EXO_PROFILER_SYNC_SPANS=1` watchdog incident. Overhead is
~40-100 ns plus ~1-3 µs emission that happens *after* the end timestamp, against a ~1.2 ms/call
signal. Perturbation detection is still pre-registered as an on/off decode-tok/s comparison at
matched strata (±2% gate), with `EXO_RUNNER_HANG_TIMEOUT_SECONDS=300` as insurance.

Env-gated on `EXO_DSV4_DECODE_COLLECTIVE_PROFILING=1` (strict no-op when off). Strata S1 0-4K /
S2 4-16K / S3 16-32K / S4 32K+; decision on S3-S4 means, no post-hoc threshold tuning. Bands
unchanged: A costs if `(all_sum + all_max) >= 600 µs/call` AND >= 10% of decode-step wall in
S3-S4.

**Cost when approved: ~1.5-2 h.** Not requested this round.

---

## ROUND-4 AGENDA (priority order)

1. **C1 — persist raw decode token ids and re-probe against real ids.** The single thing standing
   between CONDITIONAL-GO and GO. Must include multi-invoke tool-call pairs, or it tests around
   the blind spot instead of through it. Cheap; no GPU.
2. **C2 — session-length distribution over N >= 20-30 sessions.** Read-only, cheap, and it gates
   the value of the entire workstream. Converts 21.9% into a real fleet number or kills it.
3. **SDPA 2-length run** — approved-and-go, ~15-20 min. Decides whether the 4.06x per-call
   superlinearity is a real multiplicative constant (matters at 250K) or amortizes away.
4. **Decode instrumentation run** — ~1.5-2 h, ready to apply on approval.
5. **NEW — server-side defect class survives the client fix.** The shipped fix is client-side, but
   *any* client feeding a leading-divergent byte into the re-fed region silently zeroes the cache,
   and `cached_tokens = 0` is the only symptom. Either normalize server-side or add a
   `cached_tokens` monitoring signal — otherwise this exact failure mode gets re-litigated with
   the next client. Not verdict-blocking; genuinely new, surfaced by A1's evidence.

**Also carried forward, unchanged from round 2:** bucket A (49% of uncached prefill, cold start)
is the *more* survivorship-laden bucket — its entire value is set by relaunch cadence, still
unmeasured.

---

## THE APPROVAL ASK

**SDPA 2-length timing run: 2 short runs, ~15-20 min cluster time.** Scripts, patch, env, and
runbook are ready and verified; the decision rule is pre-registered.

Everything else this round was zero-GPU. Fix B remains **CONDITIONAL-GO** until C1 and C2 are
discharged.

---

## Appendix — provenance and verification

**PM verification performed directly, not delegated** (four of five worker claims needed
checking, two were wrong):
- Recomputed the full LCP distribution, all three p10 conventions, and both variant A and
  variant B band evaluations from the raw per-pair JSON.
- Confirmed the pad-replay reproduces round 2 **exactly** — the methodology-drift control.
- Re-derived the A1 token diff from the raw render JSON: confirmed single-token insertion,
  prefix 294 + suffix 59 = 353.
- Read the raw live response JSON and **found the `cached_tokens` 351-vs-0 evidence the worker
  had missed** — now the strongest result in the round.
- Ran the hermes-agent tests myself (40 passed) and probed the provider predicate directly for
  fail-safe behavior, including the `exotic` / `exo-remote` near-miss cases.
- Verified both patches apply cleanly and are **not** applied; `src/` and `mlx-lm/` are clean.
- Verified the commit contains exactly 5 intended files and that the unrelated modified
  `contributors/emails/...` file was **not** staged.

**Two worker claims corrected rather than passed upward:** the A1 worker mis-summarized its own
clean single-token-insertion result as "60 differing positions" (it is a pure insertion —
everything after index 294 is shifted by one, not changed) and never ran the live half at all,
requiring a re-dispatch; the A2 worker's assertion-3 framing about anthropic padding was wrong
and is corrected above.

**Constraints honored:** no commits to `~/repos/exo`; exo `src/` untouched; B not run; C not
applied; no boot experiments; no P16; no NOT-FUNDED items re-proposed. The only cluster contact
was three `max_tokens:1` verification requests against an already-resident model (no cold load,
sub-second to ~3s each).

**Artifacts:** `PRE-REGISTRATION.md` (+ labeled post-hoc amendment), `findings/a1-api-verification.md`,
`findings/a3-lcp-reprobe.md`, `findings/b-sdpa-2length-prep.md`,
`findings/c-decode-instrumentation-design.md`, `findings/lcp_probe_round3{,_omit,_empty,_pad}.json`,
`a3_probe_lcp.py`, and `artifacts/` (a1 render + live responses, both patches, SDPA runbook +
analyzer).
