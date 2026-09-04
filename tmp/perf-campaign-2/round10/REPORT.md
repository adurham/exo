# CAMPAIGN 2, ROUND 10 — `EXO_BATCHED_PREFILL_RENDEZVOUS_MS` 200→0, residual-governed design

## OUTCOME: SHIP

**One-line reason:** the governing full-set pooled short residual gap of **224.4 ms** landed
inside the pre-registered **[150, 250] ms** band, **C1 passed**, and both hard gates
(**byte-identity**, **clean-logs**) **passed**.

The governing statistic for this round — the RESIDUAL, not raw TTFT — was fixed in advance in
`PRE-REGISTRATION.md`, committed `b06fb1c29` **BEFORE any round-10 measurement**. This is the
exact inverse of R9: R9 governed on raw TTFT and demoted the residual to diagnostic-only, and
HELD because raw TTFT carried a noisy in-`prefill()` compute term the window cannot enter. Round
10 pre-registered the opposite mapping — residual governs, raw TTFT is diagnostic — and did so in
writing before collecting a single fresh data point this round.

The verdict is decided by applying THE BAND (§4) to the full 6-boot set, now that the fresh
confirmatory pair (boot P, RV=200; boot Q2, RV=0) is complete. That application is **SHIP**: the
full-set pooled short residual gap is **224.4 ms**, inside [150, 250] ms, C1 passes at
149.6 > 56.6, and all three applications of the band (R9-only, fresh-pair-only, full set) agree —
there is no sub-analysis disagreement on the governing instrument.

---

## 1. THE INSTRUMENT AND WHY

Governing statistic, exactly as PRE-REGISTRATION section 1 defines it:

```
residual_ms = prefill_s*1000 - ((prompt_tokens - 1) / prompt_tps) * 1000
```

Field provenance (all from `bench/long_decode_probe.py`'s output JSON):
- `prefill_s` = client-observed TTFT, JSON field `prefill_s`
- `prompt_tokens` = JSON field `prompt_tokens`
- `prompt_tps` = JSON field `server_stats.prompt_tps` (server-side, timed INSIDE `prefill()`)

`(prompt_tokens - 1)` is used because `prefill()` receives `prompt_tokens[:-1]`.

**Justification (on record before measuring):**
- The rendezvous sleep is pre-prefill by code: `runner.py:580` gates it, `:582` arms the
  deadline, and the drain loop `:594-620` completes before `prefill()` is entered. The window
  therefore **cannot** enter the in-prefill compute term.
- That in-prefill compute term is proven arm-INDEPENDENT noise by its sign flip across R9's two
  independent pairs on the 2K instrument: **+861 ms** on pair 1 versus **−132 ms** on pair 2. A
  constant sleep cannot change sign.
- Therefore raw TTFT = (arm-sensitive residual) + (arm-independent noisy compute term), and the
  residual is the only bucket the knob can occupy.

**Disclosure (PRE-REGISTRATION section 3.1, reproduced honestly, not softened):** R9's REPORT
section 2.2 had already published the short and 2K residual MEDIANS. This pre-registration is
therefore **not written blind** to them. It IS written blind to (i) all round-10 fresh-pair data,
and (ii) the per-rep residual distributions and ranges, which R9 never published. The instrument
choice (short prompt as governing) is inherited from the round brief and R9 section 8, not
selected this round.

---

## 2. PRIMARY RESULT — R9 recompute at zero cluster cost

Full recompute of R9's 60 raw rep JSONs against the residual formula above, PM-verified, all 8 of
R9's published residual medians reproduced within ±1 ms. All 60 reps had `prefix_cache_hit=none`.

### SHORT instrument (GOVERNING)

| boot | arm | n | short residual median (ms) |
|---|---|---|---|
| A | RV=200 | 10 | 685.9 |
| Z1 | RV=0 | 10 | 484.7 |
| B | RV=200 | 10 | 634.3 |
| Z2 | RV=0 | 10 | 469.4 |

spread(RV200) = max − min across A, B short residual medians = **51.7 ms**

**C1**: min(RV200 medians) − max(RV0 medians) > spread(RV200)
- LHS = min(685.9, 634.3) − max(484.7, 469.4) = 634.3 − 484.7 = **149.6 ms**
- RHS = spread(RV200) = **51.7 ms**
- **C1 result: PASS** (149.6 > 51.7)

**C2**: pooled short residual gap = median(all RV200 short reps, n=20) − median(all RV0 short reps, n=20), reported as a positive magnitude, sign must be RV=0 LOWER
- pooled gap magnitude = **205.3 ms**
- sign direction: RV=0 LOWER
- in [150, 250] band: True
- **C2 result: PASS**

### 2K instrument (SECONDARY DIAGNOSTIC — non-governing)

**R9-only, 4 boots:**

| boot | arm | n | 2K residual median (ms) |
|---|---|---|---|
| A | RV=200 | 5 | 697.0 |
| Z1 | RV=0 | 5 | 431.3 |
| B | RV=200 | 5 | 674.8 |
| Z2 | RV=0 | 5 | 400.4 |

spread(RV200)_2k (R9-only) = **22.3 ms** (note: this is the R9-only spread; the full-set spread,
computed with boot P included, is also 22.3 ms — see below)

- **C1 (2K, R9-only, non-governing)**: LHS = 243.5 ms, RHS = 22.3 ms → **PASS**
- **C2 (2K, R9-only, non-governing)**: pooled gap magnitude = **252.8 ms**, sign = RV=0 LOWER, in
  [150,250] band: False → **FAIL** (2.8 ms outside the band)

**Full set, 6 boots (A, Z1, B, Z2, P, Q2) — supersedes the R9-only sub-analysis above:**

| boot | arm | n | 2K residual median (ms) |
|---|---|---|---|
| A | RV=200 | 5 | 697.0 |
| B | RV=200 | 5 | 674.8 |
| P | RV=200 | 5 | 684.7 |
| Z1 | RV=0 | 5 | 431.3 |
| Z2 | RV=0 | 5 | 400.4 |
| Q2 | RV=0 | 5 | 442.3 |

spread(RV200)_2k (full set) = **22.3 ms**

- **C1 (2K, full set, non-governing)**: LHS = 232.5 ms, RHS = 22.3 ms → **PASS**
- **C2 (2K, full set, non-governing)**: pooled gap magnitude = **249.4 ms**, sign = RV=0 LOWER, in
  [150,250] band: True → **PASS**

> **Honest reconciliation of the two 2K sub-analyses:** the R9-only 4-boot 2K sub-analysis FAILED
> C2 at 252.8 ms (2.8 ms outside the band). Adding the fresh pair (boot P, boot Q2) brought the
> full-set 2K pooled gap to 249.4 ms, which PASSES by a margin of 0.6 ms. Both numbers are
> reported here in full, unedited. A statistic sitting within roughly 3 ms of a band edge —
> whether just inside (249.4) or just outside (252.8) — is **weak evidence in either direction**;
> it should not be read as a confident pass or a confident fail. This is exactly why the 2K
> instrument was **pre-registered as non-governing**: the round's ship/hold decision does not turn
> on which side of the line this borderline number falls, and it did not need to here — the short
> instrument (§4) is decisive on its own.

This entire section reproduces `R9-RESIDUAL-RECOMPUTE.md` at zero cluster cost for the R9-only
part — it is the "R9-only recompute, 4 boots" sub-analysis required by PRE-REGISTRATION section
4.1 — extended with the fresh-pair full-set recompute once boot P and boot Q2 data were available.

---

## 3. CONFIRMATORY PAIR — fresh round-10 boots

| boot | arm | RV verified via `ps eww` (m4-1 / m4-2) | n | short residual median (ms) | short residual full RANGE (ms) | prompt_tokens range | prefix_cache_hit audit |
|---|---|---|---|---|---|---|---|
| P | RV=200 | **200** / **200** | 25 | 690.8 | [571.9, 1182.9] | 220–240 | all `none` |
| Q2 | RV=0 | **0** / **0** | 25 | 451.9 | [383.1, 557.3] | 220–240 | all `none` |

Boot P was launched and `ps eww`-verified on the REAL runner PIDs on BOTH nodes:
`EXO_BATCHED_PREFILL_RENDEZVOUS_MS=200`, `MLX_STEEL_BATCH_INVARIANT=1`, `EXO_DSV4_MTP=1`,
`EXO_DSV4_BATCHED_PREFILL=1`. The 300s idle actually slept after READY. 25 short + 5 2K reps were
collected.

Boot Q2 was launched and `ps eww`-verified on the REAL runner PIDs on BOTH nodes
(`macstudio-m4-1` and `macstudio-m4-2`): `EXO_BATCHED_PREFILL_RENDEZVOUS_MS=0` on all four matched
runner PIDs per node, `MLX_STEEL_BATCH_INVARIANT=1`. READY at `2026-09-04T20:35:51Z`, 300s idle
actually slept, reps started after. 25 short + 5 2K reps were collected. For comparison, boot P's
2K residual median was 684.7 ms, range [605.9, 733.2]; boot Q2's 2K residual median was 442.3 ms,
range [371.2, 483.7].

---

## 4. THE BAND, APPLIED VERBATIM

Per PRE-REGISTRATION section 4.1, C1/C2 are applied three times; the full set is governing.

**(i) R9-only, 4 boots** (§2 above):
- C1: LHS 149.6 > RHS 51.7 → **PASS**
- C2: pooled gap 205.3 ms, RV=0 LOWER, in [150,250] → **PASS**

**(ii) Fresh-pair-only, 2 boots** (P, Q2):
- spread(RV200) fresh-pair-only = **0.0 ms** (single RV=200 boot in this pair, so max − min = 0)
- C1: LHS = **238.9 ms**, RHS = **0.0 ms** → **PASS**
- C2: pooled gap = **238.9 ms**, sign = RV=0 LOWER, in [150,250]: **True** → **PASS**

**(iii) FULL SET, 6 boots (A, Z1, B, Z2, P, Q2) — THE GOVERNING APPLICATION:**
- spread(RV200) full set (A, B, P short residual medians: 685.9, 634.3, 690.8) = **56.6 ms**
- C1: LHS = **149.6 ms**, RHS = **56.6 ms** → **PASS**
- C2: pooled gap = **224.4 ms**, sign = RV=0 LOWER, in [150,250]: **True** → **PASS**
- **SHIP requires C1 AND C2 on the full set.** Both pass → **SHIP**

The full-set application in (iii) governs the ship/hold decision. All three applications — (i)
R9-only, (ii) fresh-pair-only, (iii) full set — agree with each other; there is **no
sub-analysis disagreement** on the short (governing) instrument, so **no caveat is owed** on that
front. The code-predicted rendezvous effect was 200 ms; the governing measured gap is 224.4 ms,
i.e. **12.2% above prediction**, comfortably inside the pre-registered ±25% band.

`prefix_cache_hit` was `none` on **all 90 short reps across all six boots** — the short instrument
is uncontaminated by cache effects.

### Demoted diagnostic — raw TTFT (reported, does not govern)

Raw TTFT, full 6-boot set, short instrument: RV=200 median **1940.0 ms** vs RV=0 median
**1570.0 ms**, gap **370.0 ms**. This is again far above the 200 ms prediction, reproducing
**exactly** the contamination R9 diagnosed in its own raw-TTFT statistic — which is precisely why
the residual, not raw TTFT, was pre-registered as governing for this round. This 370.0 ms number
is **corroboration of R9's diagnosis**, not a new anomaly: it shows the same in-`prefill()`
compute confound inflating raw TTFT here just as it did in R7 and R9, and the residual (§2, §4)
correctly strips it out.

---

## 5. BYTE-IDENTITY GATE

Three prompts (short / 2K / 89K), temp=0, fixed `--run-id r10id`, matched `--max-tokens` across
arms. Comparison is `reasoning_content` + `content` CONCATENATED (DSv4 spends small budgets
entirely in `reasoning_content`; comparing `content` alone would compare two empty strings and
falsely report a PASS). Two captures per arm per prompt; within-arm compared FIRST. Every rep
must show `prefix_cache_hit = none`.

**Decision rule, restated verbatim from PRE-REGISTRATION section 6:**
> - within-arm identical on BOTH arms AND cross-arm identical → PASS
> - within-arm identical on BOTH arms AND cross-arm differs → HARD FAIL → HOLD
> - within-arm differs on EITHER arm → that prompt's cross-arm comparison is VOID (nondeterministic
>   regime); it neither passes nor fails and does not block a ship otherwise supported. An arm
>   that cannot reproduce ITSELF cannot testify against the other arm.

### RV=200 within-arm identity self-controls (capture 1 vs capture 2) — GIVEN

| prompt | chars | sha256[:12] | within-arm result |
|---|---|---|---|
| short | 276 | `29d8a6dbaf29` | IDENTICAL |
| 2K | 283 | `8682a9ec8a9f` | IDENTICAL |
| 89K | 898 | `0a2cc063d4df` | IDENTICAL |

### RV=0 (Q2) within-arm identity self-controls (capture 1 vs capture 2)

| prompt | chars | sha256[:12] | within-arm result |
|---|---|---|---|
| short | 276 | `29d8a6dbaf29` | IDENTICAL |
| 2K | 283 | `8682a9ec8a9f` | IDENTICAL |
| 89K | 898 | `0a2cc063d4df` | IDENTICAL |

### Cross-arm comparison

| prompt | RV=200 sha256[:12] | RV=0 sha256[:12] | cross-arm result |
|---|---|---|---|
| short | `29d8a6dbaf29` | `29d8a6dbaf29` | IDENTICAL |
| 2K | `8682a9ec8a9f` | `8682a9ec8a9f` | IDENTICAL |
| 89K | `0a2cc063d4df` | `0a2cc063d4df` | IDENTICAL |

Both arms reproduce themselves AND match each other; all 10 captures across both arms share the
same three hashes (`29d8a6dbaf29` / `8682a9ec8a9f` / `0a2cc063d4df`). Per PRE-REGISTRATION section
6 this is the unambiguous PASS branch.

Overall byte-identity gate verdict: **PASS**

---

## 6. TASK 3 — R7 STEEL-BI 89K SELF-CONTROL (COMPLETE — no placeholders)

Per PRE-REGISTRATION section 9: on the RV=200 production boot with `MLX_STEEL_BATCH_INVARIANT=1`
(default per `start_cluster.sh:269`), the 89K prompt at temp=0, fixed `run-id r10id`, was captured
THREE times with identical `max-tokens`.

**All three byte-identical: 898 chars, sha256[:12] = `0a2cc063d4df`.**

Two further 89K captures on the same boot — `identity_RV200_c1_89k` and `identity_RV200_c2_89k`
— ALSO matched that same hash. **Five independent 89K captures agree**, all `0a2cc063d4df`.

**Verdict, per PRE-REGISTRATION section 9's decision rule:** all three (and in fact all five)
captures are byte-identical → **R7 section 4's 89K leg STANDS as-is.**

**Nuance for the record:** R9 observed a within-arm 89K outlier on its RV=200 boot (its original
`identity_RV200_89k` capture diverged from three same-arm re-runs and from all RV=0 output — see
R9 REPORT section 4.1) and concluded 89K generation was not reliably reproducible run-to-run on
this cluster. This round's five-for-five agreement on the production config shows 89K **is**
reproducible here, so R9's outlier was a rarer event than it appeared at the time. This does
**not** reinstate any R7 claim beyond the 89K leg standing — R7's `<8192` and 5-fixed-prompt legs
were never in question, and R9's retroactive weakening of R7's 89K identity finding (R9 REPORT
section 4.2) is the finding this round's control was designed to check.

---

## 7. CLEAN-LOGS VETO (HARD)

Per PRE-REGISTRATION section 7: on the RV=0 boot, under mixed short + 89K traffic, the gate
requires zero errors, zero rank disagreement, zero task-set mismatch, zero "out of sync" /
"closed communication". The following pre-existing background warnings (from R9 REPORT section
2.3) are excluded BY NAME and do not trip the veto:

- HF catalog poll for `GLM-4.7-8bit-gs32`
- invalid model cards
- `mx.metal.get_*_memory` deprecations
- transformers rope notice
- normal `[jaccl-v2]` trace
- the `error.svelte.js` build-artifact filename

**RV=0 (Q2) boot clean-logs result: PASS.** Both nodes showed ZERO hits for rank disagreement,
task-set mismatch, "out of sync", or "closed communication". Tracebacks WERE present in the logs
— this is stated plainly, not hidden — and every one was individually enumerated and classified:
11 per node of `Exception: Failed to fetch file list: 404` from the background HF catalog poll for
`mlx-community/GLM-4.7-8bit-gs32` (fires on a roughly 77-second timer, arm-independent), plus
pydantic `ValidationError` for `ModelCard` (the "invalid model cards" exclusion). No other error
class appeared. Every traceback present matched a PRE-REGISTRATION section 7 exclusion named
above; none fell outside the excluded list. Veto: **PASS**.

Any error not on the excluded list above → HOLD.

---

## 8. SHIP OR HOLD

**SHIP.** All gates passed: C1/C2 on the full-set short residual instrument (§4) both pass, the
byte-identity gate (§5) is a clean PASS, and the clean-logs veto (§7) is a clean PASS.
`EXO_BATCHED_PREFILL_RENDEZVOUS_MS` default changed 200 → 0 and the cluster was relaunched onto
it.

`start_cluster.sh` line 136 (now line 145 after the added comment) changed from
`: "${EXO_BATCHED_PREFILL_RENDEZVOUS_MS:=200}"` to `: "${EXO_BATCHED_PREFILL_RENDEZVOUS_MS:=0}"`,
with a dated 2026-09-04 comment citing R7/R9/R10, the 224.4 ms residual gap, the [150,250] ms
band, and the V4 c≥2 rationale. The `:=` form is preserved so the knob remains overridable; the
propagation line (now line 1706) is untouched; the original 200 ms rationale text was kept, not
deleted. `bash -n` passes. Commit `096a00a58`.

```
- : "${EXO_BATCHED_PREFILL_RENDEZVOUS_MS:=200}"
+ # 2026-09-04 R10: SHIP — governing full-set pooled short residual gap 224.4 ms,
+ # inside pre-registered [150,250] ms band (R7 §2.3/§8.2, R9 §2.2/3.2/3.3/8, R10 §4/§9).
+ # V4 c>=2 rationale retained below; knob remains overridable via env.
+ : "${EXO_BATCHED_PREFILL_RENDEZVOUS_MS:=0}"
```

**Decisive verification:** the cluster was then relaunched with **no environment override at
all** (boot label PROD, launched via the DEFAULT path so no
`EXO_BATCHED_PREFILL_RENDEZVOUS_MS` was exported), and `ps eww` on the REAL runner PIDs on BOTH
nodes shows `EXO_BATCHED_PREFILL_RENDEZVOUS_MS=0` on every matched PID. This proves the changed
DEFAULT itself took effect, not an env var. READY `2026-09-04T20:54:10Z`, 300s idle slept.

`API /v1/models` returns HTTP 200.

Live probe through the existing harness on the production boot: `prompt_tokens` 194,
`completion_tokens` 16, TTFT 1910.0 ms, residual 374.9 ms, `prefix_cache_hit` none, and the model
returned non-empty generated text. This short probe request serves in place of the
coherent-completion curl (which the approval gate blocked in R9); the clean-logs veto plus this
probe together serve that purpose.

PROD boot logs: zero hard-veto patterns on both nodes; tracebacks only the same named-excluded
GLM-404 and invalid-model-card classes.

---

## 9. RECONCILIATION

**Against R7 section 2.3 (the −480 ms).** R7 reported a raw-TTFT gap of −480 ms and declined to
ship an unexplained number. R9 section 1.2 showed ~270 ms of that 480 sat in in-`prefill()`
compute — a term the rendezvous window structurally cannot enter (it closes before `prefill()` is
entered). Round 10's entire design choice — govern on the residual, not raw TTFT — is the direct
consequence of that finding: by construction the residual excludes exactly the term that
contaminated R7's number.

**Against R7 section 8.2 (the paired-boot prescription).** R7 prescribed alternating paired boots
per arm as the fix for boot-to-boot bias. R9 executed that prescription (four boots, two per arm)
and it worked — between-boot bias fell from ~270 ms (R7) to ~122 ms — but R9 section 3.2 showed
the *residual* contamination (the in-prefill compute term) is *within-arm* variance, which paired
boots cannot remove, because it is noise on both arms independently, not a systematic difference
between boots. Round 10 inherits the paired-boot design from R7/R9 for the fresh confirmatory
pair, and additionally isolates the residual bucket so the within-arm noise doesn't leak into the
governing statistic at all.

**Against R9's residual analysis (sections 2.2, 3.2, 3.3, 8).** R9 section 2.2 first published
the short/2K residual medians as a secondary, non-governing diagnostic. R9 section 3.2 showed the
in-prefill compute term flips sign across R9's own two independent pairs (+861 ms vs −132 ms at
2K) — proof that term is arm-independent noise, since a constant sleep cannot change sign. R9
section 3.3 concluded in writing: raw TTFT is the wrong decision statistic for a 200 ms effect on
this cluster, because it carries a term the window provably cannot influence. R9 section 8
explicitly recommended, as the design to close the round in one more cheap round: (1) pre-register
the residual as governing statistic before measuring, (2) raise n to ~25 short reps, (3) two boots
suffice (one per arm) since the residual already reproduces across R9's own pairs, (4) byte-identity
does not need re-running except any future 89K claim needs a same-arm self-control. **Round 10 is
exactly that recommendation, executed**: PRE-REGISTRATION.md pre-registers the residual as
governing (committed b06fb1c29 before measurement), the confirmatory pair collects n=25 short
reps per boot, exactly two fresh boots (P, Q2) are used, and Task 3 runs the same-arm 89K
self-control R9 flagged as the outstanding requirement.

**On R9's HOLD itself.** R9's HOLD was **correct on its own pre-registered terms**: R9 governed on
raw TTFT, which section 3.2/3.3 showed carries an arm-independent noise term the rendezvous window
cannot influence, and HOLDing rather than shipping on a contaminated statistic was the right call
given that instrument. Round 10 did **not** overturn R9 by relaxing a standard, re-running the same
test until it passed, or moving the band. It changed the **instrument** — in advance, in writing,
with the justification above (§1) — and then met a band of the **same width** ([150,250] ms, ±25%
around the 200 ms prediction) that R9's raw-TTFT statistic could not meet. R9's diagnosis (raw TTFT
is the wrong statistic here) and R9's prescription (§8: pre-register the residual, raise n, use two
fresh boots) are both vindicated by this round's result, not contradicted by it.

---

## 10. DEVIATIONS AND HONESTY

1. **Void boot Q.** The first RV=0 launch attempt (boot Q) was aborted by `start_cluster.sh`'s
   node-commit-consistency gate because a commit from a concurrent session landed mid-rsync
   (nodes reported `99c74a27b` vs `ea4d65ab2`). **No reps ran on boot Q.** It was relaunched
   clean as boot Q2. This is the identical failure mode R9 hit and documented (R9 REPORT section
   7e.3); recorded here for the same reason.
2. **Interpreter bug found and fixed mid-round.** The driver scripts invoked bare `python3`,
   which under some shells resolved to Homebrew python lacking `httpx`, causing instant probe
   failure. Fixed by pinning `PY=/usr/bin/python3` in all four driver scripts — the same
   interpreter that produced the already-collected arm data, chosen for cross-arm consistency.
   `bench/long_decode_probe.py` itself was **not** modified.
3. **PRE-REGISTRATION section 3.1 disclosure**, reproduced without softening: the pre-registration
   was written after R9's REPORT section 2.2 had already published the short and 2K residual
   medians, so it is not written blind to those medians. It is written blind to all round-10
   fresh-pair data and to the per-rep residual distributions/ranges R9 never published. The
   instrument choice (short prompt as governing) predates this round, inherited from the brief
   and R9 section 8.

---

## 11. ARTIFACTS AND LOCAL COMMITS

Files in `tmp/perf-campaign-2/round10/`:
- `PRE-REGISTRATION.md` — governing statistic and verbatim ship band, committed before measurement
- `R9-RESIDUAL-RECOMPUTE.md` — the primary zero-cost result (§2 above)
- `recompute_r9_residual.py`, `r9_residual_recompute.json` — recompute script and machine-readable output
- `run_boot.sh`, `run_reps.sh`, `run_identity.sh`, `run_89k_selfcontrol.sh`, `summarize10.py` — drivers (harness `bench/long_decode_probe.py` itself unmodified)
- `compare_identity.py` — byte-identity hashing/comparison tool
- `results/` — raw JSONs from boot P and (pending) boot Q2, plus identity captures
- `REPORT.md` — this report

Local commits known so far:
- `b06fb1c29` — PRE-REGISTRATION.md
- `0a98ce693` — R9 recompute + R10 drivers
- `99c74a27b` — boot P data + interpreter pin
- `096a00a58` — the start_cluster.sh default change (200 → 0)

A final commit for this report will follow.

**No pushes.**

---

## 12. WHAT IS NOW SETTLED

- **Mechanism.** One consumer, window bounded by `W` (the rendezvous timeout), and the wait is
  pre-prefill by code: `runner.py:580` gates it, `:582` arms the deadline, and the drain loop
  `:594-620` completes before `prefill()` is entered.
- **The residual is the right instrument** for sub-300ms effects on this cluster: it excludes the
  arm-independent in-`prefill()` compute noise term that raw TTFT cannot avoid, and it is what
  closed this round inside a ±25%-of-prediction band where raw TTFT (§4, demoted diagnostic)
  could not.
- **Byte-identity holds** at short, 2K, and 89K, across both arms (§5): within-arm self-controls
  reproduce on both arms, and cross-arm comparisons match, for all three prompt sizes.
- **R7's 89K steel-BI leg stands** (§6, Task 3): five independent 89K captures on the production
  config all agree at `0a2cc063d4df`, confirming R7 section 4's 89K finding.
- **RV=0 is now the shipped default.** `start_cluster.sh`'s
  `EXO_BATCHED_PREFILL_RENDEZVOUS_MS` default is `0` (commit `096a00a58`), the cluster is running
  on it with no env override, and API + probe + clean-logs checks all pass on that boot.

---
