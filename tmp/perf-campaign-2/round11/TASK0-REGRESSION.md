# Task 0 Regression — Real-Usage Residual Decomposition

**Round:** campaign 2 / round 11
**Interpreter used:** `/usr/bin/python3` (pure `statistics`/stdlib, no numpy/scipy)
**Script:** `tmp/perf-campaign-2/round11/task0_regression.py`
**Machine-readable output:** `tmp/perf-campaign-2/round11/task0_regression.json`
**Pre-registration:** `tmp/perf-campaign-2/round11/PREDICTION.md` §1 (committed before this analysis was run; not edited)

---

## 0. Data and validation

Source: `tmp/real-usage-capture-20260902/phase1/requests.jsonl` (57 records, each
field `{value, provenance}`, unwrapped by the script). 55 of the 57 records have
non-null client fields (`wall_seconds_client`, `client_started_ts`) — these are
the "main rows"; the 2 excluded aux rows have `prompt_tokens` 18/17 and null
client fields, matching the study's own partition.

Residual was **recomputed from raw fields**, not read from any stored column,
per the identity in `tmp/real-usage-capture-20260902/REPORT.md` lines 55–75:

```
prefill_uncached = (prompt_tokens - cached_tokens) / prompt_tps
decode           = completion_tokens / generation_tps
residual          = wall_seconds_client - prefill_uncached - decode
```

**Recomputation matches the published numbers:**

| | recomputed | published (`partition_verified.json`) | diff |
|---|---|---|---|
| median | 0.9432 s | 0.94 s | 0.0032 s |
| min | 0.7515 s | 0.75 s | 0.0015 s |
| max | 1.3223 s | 1.32 s | 0.0023 s |

All diffs are well under the 0.01 s tolerance — validation **PASSED**. The
identity used and the client/server join are confirmed sound; the regression
below proceeds on a residual that reproduces the study's published figures.

---

## 1. Exclusions reconciliation (E)

- 2 aux rows excluded (prompt 18/17, null client fields) — always excluded, matching the study's own partition.
- `partition_verified.json` reports `prefix_cache: {'partial': 54, 'none': 3}` over **all 57 captured records**, not the 55 main rows.
- Of those 3 `none`-hit rows, **2 are the aux rows** (excluded above). Only **1** `none`-hit row remains among the 55 main rows.
- That 1 remaining row is exactly the pre-registered cold outlier: `task_id=19282ba9-c98c-45f5-a72b-1822080d7597`, `prompt_tokens=92594`, `cached_tokens=0`, recomputed `prefill_uncached≈222.1 s` — matching the pre-registration's description (prompt 92,594, prefill ~222 s) exactly.
- **Conclusion:** the 3-vs-1 discrepancy is fully explained by aux-row overlap; no additional exclusion beyond the pre-registered 2 aux rows + the cold-outlier sensitivity split was needed. Fits below are run on 55 rows (with cold outlier) and 54 rows (without it), as pre-registered.

---

## 2. Descriptive stats (F)

| Quantity | median | min | max |
|---|---|---|---|
| residual (s) | 0.943 | 0.751 | 1.322 |
| transit = server_received_ts − client_started_ts (s) | 0.191 | 0.077 | 0.394 |
| residual_ex_transit (s) | 0.781 | 0.560 | 1.045 |
| prompt_tokens | 145,918 | 92,594 | 188,902 |

Transit range matches the study's published [0.077, 0.394] s, median 0.191 s exactly.

---

## 3. Regression results

All slopes reported in **µs/token**; intercept in seconds; 95% CI via t-distribution
on the OLS slope (n−2 dof, table-based t critical value).

| Fit | n | slope (µs/tok) | 95% CI | intercept (s) | r² |
|---|---|---|---|---|---|
| **(A)** residual_ex_transit ~ prompt_tokens, **with cold outlier** | 55 | 0.862 | [−0.244, 1.968] | 0.653 | 0.044 |
| **(A)** residual_ex_transit ~ prompt_tokens, **without cold outlier** | 54 | 0.683 | [−0.454, 1.821] | 0.681 | 0.027 |
| **(B)** residual ~ prompt_tokens (raw), with cold outlier | 55 | 1.674 | [0.413, 2.935] | 0.726 | 0.118 |
| **(B)** residual ~ prompt_tokens (raw), without cold outlier | 54 | 1.591 | [0.278, 2.904] | 0.739 | 0.102 |
| **(C)** residual ~ cached_tokens (raw), with cold outlier | 55 | 1.356 | [0.302, 2.410] | 0.776 | 0.112 |
| **(C)** residual ~ cached_tokens (raw), without cold outlier | 54 | 1.521 | [0.211, 2.831] | 0.752 | 0.094 |

### (D) Collinearity

| | Pearson r (prompt_tokens, cached_tokens) |
|---|---|
| with cold outlier (n=55) | 0.931 |
| without cold outlier (n=54) | **0.997** |

The pre-registration's r > 0.95 threshold is met on the fitted (cold-excluded)
rows: **prompt_tokens and cached_tokens are essentially collinear (r = 0.997)
among the 54 warm-cache rows.** The one cold row (cached=0, prompt=92,594) is
what pulls the *full-55* correlation down to 0.931 — it is the only point where
the two regressors diverge.

**Answer: NO**, the trie-walk (∝cached) and tokenization (∝prompt) hypotheses
**cannot be told apart on this data.** Fits (B) and (C) are statistically
almost the same regression in disguise (same 54 points, near-identical x
values) — their similar slopes/r² (1.59 vs 1.52 µs/tok, r²=0.102 vs 0.094) is
exactly what perfect collinearity predicts, not independent confirmation of
either mechanism.

### (G) Robustness — Theil-Sen vs OLS for fit (A)

| | slope (µs/tok) | intercept (s) |
|---|---|---|
| OLS, with cold outlier | 0.862 | 0.653 |
| Theil-Sen, with cold outlier | 0.814 | 0.658 |
| OLS, without cold outlier | 0.683 | 0.681 |
| Theil-Sen, without cold outlier | 0.572 | 0.702 |

Theil-Sen agrees with OLS in **sign and order of magnitude** in both cases
(within ~0.05–0.11 µs/tok, i.e. well inside the OLS 95% CI). The cold-outlier
row has real leverage (drops the OLS slope from 0.862→0.683 µs/tok, ~21%) but
does not flip the qualitative picture — both estimators land in the same
sub-1-µs/tok, low-precision regime with CIs spanning zero. **Conclusion: the
OLS slope for fit (A) is directionally consistent but too imprecise (CI
crosses zero in both variants) to be trusted as a point estimate.**

---

## 4. Verdict (pre-registered decision rule)

Decision rule: slope ≈ 1–2 µs/tok ⇒ O(context) work dominates; flat + high
jitter ⇒ IPC/polling ticks dominate; anything in between ⇒ MIXED, naive fit
cannot rank hypotheses.

**Fit (A) without cold outlier (primary, as pre-registered): slope = 0.683
µs/tok, 95% CI [−0.454, 1.821], r² = 0.027.**

- The point estimate (0.683) sits below the 1–2 µs/tok "O(context) dominates"
  band and above the "flat" (~0) case.
- The CI is wide enough to include both 0 and values up to ~1.8 µs/tok — it
  **cannot exclude** either the flat-IPC hypothesis or a genuine sub-2-µs/tok
  context-scaling term.
- **VERDICT: MIXED.** The naive fit (A) does not cleanly discriminate between
  IPC/polling-dominated and O(context)-dominated explanations. This matches
  the pre-registration's own expected outcome for the verdict category
  (though not every sub-claim below scored correct — see §5).

The **intercept is large and robust across every fit variant (0.65–0.78 s)**,
present in fits (A), (B), (C), with and without the cold outlier, and
confirmed by Theil-Sen. This is the more solid finding of this analysis: **a
substantial context-independent residual floor exists** regardless of what
the slope turns out to be. Per the interpretation guard, the low r² values
throughout do **not** by themselves argue against O(context) work — the
residual bakes in up to 0.32 s of transit-time noise (removed in fit A) plus
whatever fixed IPC/rendezvous cost remains, both of which dilute r² without
bearing on whether a context-scaled term is also present.

---

## 5. Pre-registered prediction scoring

Scored against the **without-cold-outlier** fits (the pre-registration treats
the cold row as a leverage outlier to be excluded from primary
interpretation).

| # | Sub-claim | Predicted | Observed | Verdict |
|---|---|---|---|---|
| 1 | Slope of raw residual (B) on prompt_tokens | 0.5–2.0 µs/tok | 1.591 µs/tok | **CORRECT** |
| 2 | r² of raw fit (B) | < 0.35 | 0.102 | **CORRECT** |
| 3 | Intercept (B) | 0.4–0.8 s | 0.739 s | **CORRECT** |
| 4 | r² of ex-transit fit (A) | > 0.4 | 0.027 | **WRONG** |

**3 of 4 sub-claims correct; 1 (the "sharper test" tie-breaker) is WRONG, and
it is the one the pre-registration itself flagged as the primary
discriminator.**

The pre-registration explicitly staked its MIXED verdict on r²(A) rising
materially above r²(B) once transit noise is removed — that is exactly what
did *not* happen. r² for fit (A) (0.027) is actually **lower** than for fit
(B) (0.102), not higher. Read literally, the pre-registration's own
falsification criterion ("if r² does **not** rise, that is real evidence the
residual floor is dominated by fixed IPC/polling ticks rather than
context-scaled work") was met: **removing transit did not sharpen the
prompt_tokens signal, which weakly favors the IPC/polling-fixed-cost
explanation over an O(context) explanation for whatever residual variance
transit wasn't hiding.** This does not overturn the large, robust intercept —
it means the *slope* evidence for an O(context) term is weaker than
pre-registered, while the evidence for a large fixed floor is, if anything,
strengthened.

---

## 6. Limitations

1. **n=54–55 is small** for a regression meant to resolve a sub-µs/tok effect
   against a residual with 0.32 s of injected transit noise and further
   unmeasured IPC jitter. Both OLS and Theil-Sen CIs are wide and cross zero
   in the primary fit; this analysis can rule out large slopes (e.g. >3
   µs/tok) but cannot precisely pin down small ones.
2. **prompt_tokens and cached_tokens are collinear at r=0.997** among the 54
   fitted warm rows. Fits (B) and (C) are not independent evidence for
   competing mechanisms (trie-walk vs tokenization) — they are near-identical
   regressions on near-identical regressors. Any claim distinguishing these
   two hypotheses from this dataset alone should be rejected.
3. **The cold-outlier row has real leverage** (single point at prompt_tokens
   92,594 vs a 92,594–188,902 range for the rest, and cached_tokens=0 vs
   otherwise ~partial-hit). Its inclusion/exclusion moves the OLS point
   estimate for fit (A) by ~26% (0.862→0.683 µs/tok). Theil-Sen is somewhat
   more robust but still shifts (0.814→0.572 µs/tok) — a single boot's worth
   of cold-cache data cannot anchor a leverage-robust estimate on its own.
4. **`residual_ex_transit` still contains unmeasured constants** beyond
   client→server transit — e.g. any rendezvous sleep or IPC/polling tick
   noted in `PREDICTION.md` §0 (R10 removed ~0.2 s of a 200 ms rendezvous
   sleep; ~0.55 s/request was still unexplained going into this round). The
   ~0.65–0.78 s intercept found here is consistent with, but does not by
   itself decompose, that remaining fixed cost — that decomposition is
   Task 1/Task 2's job (source read + instrumented replay), not this offline
   regression.
5. **r² near-independence guard**: per the pre-registration's own
   interpretation guard, a low r² anywhere in this analysis must not be read
   as evidence against an O(context) term — it can equally reflect the
   injected transit/IPC noise. Conversely, this analysis's failure to find
   the *predicted rise* in r² after removing transit (§5, sub-claim 4) *is*
   informative and should not be waved away by the same guard — the guard
   protects against over-reading a low r² as anti-O(context) evidence, not
   against reading a *stalled* r² (A ≤ B) as weak evidence against the slope
   hypothesis specifically.
6. **Timestamp precision is millisecond-granularity** (`server_received_ts`,
   `client_started_ts` are parsed at `.mmm` precision), which floors
   measurement resolution for `transit` and hence `residual_ex_transit` at
   ~1 ms — negligible next to the ~0.3–1.3 s magnitudes involved here, but
   worth noting for any future sub-10ms-scale analysis on this data.
7. This is a **naive linear regression on an observational dataset with one
   plausible confound (collinearity) and one leverage point** — it is a
   screening tool for Task 0, not a causal decomposition. Task 1 (source
   read) and Task 2 (instrumented replay) are the appropriate next steps to
   actually attribute the ~0.55 s residual floor to specific code paths.
