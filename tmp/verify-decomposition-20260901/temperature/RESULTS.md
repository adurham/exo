# RESULTS — sampling-temperature A/B (temp=1.0 vs temp=0.8)

**Run:** 2026-09-02, 10:00:55–10:22:58 CDT. 1 arm, 5 scored deep-context requests
(mode=repetitive, 75000 words, temperature=1.0 forced in the request body).
**Pre-registration:** `PREREGISTRATION_TEMPERATURE.md` (decision rule fixed
before any temp=1.0 request was sent; STEP 0 premise check also fixed there).
**Question:** do real Hermes sessions sample hotter than the bench
(assumed 1.0 vs 0.8), lowering MTP draft acceptance and explaining part of the
~14 t/s real-usage decode gap?

Statistics computed independently by the analysis dispatch from
`raw/temp_temp10.json` (parsed, not hand-copied); the PM's figures were used
only as a cross-check (§2.4). t-distribution CDF/quantiles from **scipy 1.17.1**
(`/Users/adam.durham/repos/exo/.venv/bin/python`).

---

## VERDICT

**No — sampling temperature does not explain the real-usage decode gap. The
hypothesis dies at the premise, before the measurement even matters.**

The verdict has two distinct layers and they must not be collapsed:

**LAYER 1 — the premise is FALSE (decisive).** The hypothesis assumed real
Hermes sessions run at temperature 1.0 while the bench runs at 0.8. They do
not. Real Hermes sessions send **no temperature field at all** to this cluster
(grep of `~/.hermes/config.yaml`, every profile config, the exo provider
profile, the chat-completions transport, and the main-loop caller: zero
emitters; `agent.adaptive_sampling` is enabled in config but read by zero
installed client files — inert). The exo server resolves each sampling field
independently (`request → instance → card → cluster-env → hardcoded`) and the
card pins `temperature = 0.8`, so **real sessions and the bench both resolve
to 0.8. There is no temperature delta in production to explain anything.**
This was established by direct grep before the arm ran (pre-registration
§STEP 0) and kills the hypothesis regardless of what the temp=1.0 arm showed.

**LAYER 2 — the measured contrast is PARTIAL and UNDERPOWERED (moot for
production, reported for completeness).** Even had a delta existed, forcing
0.8 → 1.0 moved decode by only **−1.04 t/s (−3.11%)**: 32.54 vs pooled
baseline 33.58 (n=5 vs n=7). Welch one-sided p = 0.101 — not significant.
Point estimate explains **7.45%** of the 14 t/s gap. Per the pre-registered
decision rule this lands in **PARTIAL** (T=32.54 is in the 32.0–33.0 dead
band, and although p ≥ 0.05 the 95% CI upper bound on the drop, **+2.79 t/s,
exceeds the 2.5 t/s NOT-CAUSE bar**). The arm is underpowered: the MDE is
**~2.7 t/s** at the baseline's sample sd, so this contrast cannot exclude
drops smaller than that. The honest worst case is the CI upper bound: a drop
of up to **~2.8 t/s ≈ 20% of the gap** is not excluded by this data. Per the
pre-registration this is **not rounded to a clean null**.

But Layer 1 makes Layer 2 moot for the real-usage question: **an effect that
never fires in production — because production and bench run the identical
temperature — explains 0% of the production gap**, whatever its true size.
Temperature is **closed** as a cause of the real-usage 20 t/s, on both layers.

---

## 1. The temp=1.0 arm

| arm | mode | prompt tok | decode mean | median | min | max | sd (ddof=1) | prefill t/s | cycle ms (n1/n2) |
|---|---|---|---|---|---|---|---|---|---|
| temp10 | repetitive, **temp=1.0** | 89,408 ×5 | **32.54** | 32.22 | 32.05 | 33.42 | 0.58 | 425.9 | 67.36 / 67.40 |

Per-iteration decode (t/s): 32.816, 32.045, 32.222, 33.417, 32.176.
All five iterations scored, `rc=0`, prompt_tokens 89,408 exactly (matches the
baseline's 89,408 — same calibrated 75,000-word prompt). Config echo confirms
`"temperature": 1.0` reached the server; by the per-field resolution order
(F2/F4) top_p/top_k/min_p stayed at card values, so this is a clean
single-variable contrast against the temp=0.8 baseline.

*(Note on the sd column: the entropy study's table reported population sd
(ddof=0); this table uses sample sd (ddof=1), which is the primary convention
for every statistic in §2. See anomaly 2.)*

### Cycle-phase breakdown (`[MTP-PROF]`, interval-weighted, node 1 / node 2)

| phase | n1 | n2 |
|---|---|---|
| verify | 54.89 | 54.76 |
| draft | 8.99 | 8.97 |
| accept | 2.72 | 2.83 |
| rollback | 0.79 | 0.79 |
| **total** | **67.36** | **67.40** |
| cycles | 650 | 650 |
| `rb_pool_restores` *(count, not ms)* | 16.73 | 16.73 |

De-aggregated per the pre-registered arithmetic
`(mean_k·n_k − mean_{k−1}·n_{k−1})/(n_k − n_{k−1})`, cycle-count weighted,
anchored on the last pre-window dump. Reused `../analyze_prof.py` +
the `../entropy/build_entropy_table.py` pattern unchanged.

### Anchor trustworthiness (validated before publishing any cycle number)

The standard 200KB pre-window anchor came back **empty on both nodes**
(`raw/prof_anchor_temp10_n{1,2}.txt` are 0 bytes) because the last pre-window
MTP-PROF dump sits 460,699 bytes (n1) / 617,856 bytes (n2) back from the
window start — beyond the 200KB harvest window. The supplemental **2MB
anchors** were used instead and were validated before use:

1. **Monotonic counters.** n1 anchor dumps run 2200→2250→2300→2350→2400;
   n2 runs 2250→2300→2350→2400. No restarts, no resets.
2. **Immediate predecessor.** The last anchor dump is cycles=2400 on both
   nodes (timestamps 09:52:51.529 / 09:52:51.533); the first in-window dump is
   cycles=2450 (10:04:29.865 / 10:04:29.864). Nothing between.
3. **No pre-window cycle bleed.** A direct read of the full pre-window log gap
   (the 460,699 / 617,856 bytes, live over ssh) shows **exactly one MTP-PROF
   line** — the tail of the 2400 dump itself — and **zero generation activity**
   between 09:52:53 ("runner idle … runner ready") and 10:00:55 (arm start).
   No cycles ran in the gap, so cycles 2401–2450 are genuinely this arm's.

**The anchor is trustworthy. Cycle phases are publishable.**

One structural note: the first 50-cycle interval (2401–2450) contains the
**warmup iteration's** decode, whose verify phase de-aggregates to 61.49 ms vs
54.3 ms for the rest. This is the same structure the baseline's arm 1 had
(counters starting fresh at cycles=50, warmup inside the first interval), so
the comparison remains apples-to-apples. Full-window figures above include it;
excluding the first interval gives n1 66.80 / n2 66.79 ms over 600 cycles.

---

## 2. Contrast vs temp=0.8 pooled — the statistics

### 2.1 Distributions

| condition | n | mean | sd (ddof=1) | sd (ddof=0) | median | min | max |
|---|---|---|---|---|---|---|---|
| temp=0.8 pooled (repetitive + recheck) | 7 | **33.5786** | 1.8567 | 1.7189 | 33.33 | 31.72 | 37.23 |
| temp=1.0 | 5 | **32.5353** | 0.5750 | 0.5143 | 32.22 | 32.05 | 33.42 |

Per-iteration values were parsed from `raw/temp_temp10.json` (treatment) and
`../entropy/RESULTS.md` §1 (baseline: 31.72, 31.79, 33.02, 34.00, 33.33, 37.23,
33.96). The treatment JSON's own summary block (mean 32.5353, median 32.2225)
matches the independent recomputation exactly.

### 2.2 Effect size

| quantity | value |
|---|---|
| absolute drop B−T | **1.0432 t/s** |
| relative drop (B−T)/B | **3.11%** |
| share of the 14 t/s real-usage gap (B−T)/14 | **7.45%** (point estimate) |

### 2.3 Welch test, CI, MDE

| quantity | value |
|---|---|
| Welch se of the drop | 0.7474 |
| Welch t (two-sample, B−T) | **1.3958** |
| Satterthwaite df | **7.5163** |
| one-sided p (B > T), scipy t.sf | **0.1013** |
| 95% CI on the drop (t-crit 2.3321 at df 7.5163) | **[−0.700, +2.786] t/s** |
| MDE (α=0.05 one-sided, 80% power, z-approx, **sample sd 1.857**) | **2.70 t/s** |
| MDE (same, population sd 1.7189 — the pre-reg's input) | 2.50 t/s |
| MDE (same, Welch se 0.7474, df 7.52) | 1.86 t/s |
| MDE, t-based multipliers at df 7.52 (sample sd) | 3.01 t/s |

Sample-sd figures are primary. The pre-registration's MDE of 2.49 used the
population sd; recomputed with the same convention it is 2.50 — consistent
within rounding. With the correct sample sd it is 2.70 t/s (≈19% of the gap):
**this arm cannot exclude a temperature drop smaller than ~2.7 t/s.** Even the
most favourable power calculation (Welch se) needs a 1.86 t/s effect to detect
at 80% power — larger than the 1.04 t/s observed. The MDE using Welch's
per-group variances (1.86 t/s) is smaller only because the temp=1.0 arm is
much less noisy (sd 0.58) than the pooled baseline (sd 1.86); the pooled-sd
planning figure is the honest one for what this design could resolve.

### 2.4 Cross-check against the PM's independent figures

| quantity | this dispatch | PM cross-check | agree? |
|---|---|---|---|
| T mean | 32.5353 | 32.5353 | ✓ |
| T sample sd | 0.5750 | 0.5750 | ✓ |
| B mean | 33.5786 | 33.579 | ✓ |
| B sample sd | 1.8567 | 1.857 | ✓ |
| drop / % | 1.0432 / 3.11% | 1.0432 / 3.11% | ✓ |
| share of gap | 7.45% | 7.5% | ✓ (rounding) |
| Welch t / df / se | 1.3958 / 7.5163 / 0.7474 | 1.3958 / 7.52 / 0.7474 | ✓ |
| 95% CI on drop | **[−0.700, +2.786]** | **[−0.676, +2.762]** | **✗ small** |

**One discrepancy, reported loudly:** the CI bounds. The PM's bounds imply a
t-critical of ≈2.300; the exact quantile at df = 7.5163 is **2.3321**
(scipy `t.ppf(0.975, 7.5163)`), giving [−0.700, +2.786]. The PM likely used a
rounded t-critical (e.g. df≈8 → 2.306, or a 2.30 table value). The difference
(±0.02 t/s) is immaterial to every conclusion — both intervals straddle zero
and both upper bounds exceed the 2.5 t/s NOT-CAUSE bar — but the exact-quantile
interval [−0.700, +2.786] is the one adopted here, and it is the value used in
the decision below. Every other number agrees.

### 2.5 Pre-registered decision rule, applied mechanically

| branch | condition | met? |
|---|---|---|
| CONFIRMED | T ≤ 32.0 **and** Welch one-sided p < 0.05 | **No** — T=32.54, p=0.101 |
| NOT-CAUSE | T ≥ 33.0, **or** (p ≥ 0.05 **and** CI upper < 2.5) | **No** — T<33.0; p≥0.05 ✓ but CI upper 2.79 > 2.5 ✗ |
| PARTIAL | anything else | **← this branch fires** |

Formally **PARTIAL**, exactly as the PM's reading anticipated, and not to be
rounded toward a clean outcome: the CI genuinely admits a temperature-driven
drop of up to ~2.8 t/s (~20% of the gap). The honest statement is "a real
effect up to ~20% of the gap cannot be excluded; the point estimate is ~7.5%".

---

## 3. Mechanism check (pre-registered, secondary, non-gating)

The pre-registration: if decode falls, **flat cycle time (<10% change) while
decode moves implicates ACCEPTANCE; cycle time moving in proportion implicates
CYCLE COST** and the temperature-acceptance story is wrong even if decode
falls.

| condition | decode t/s | cycle ms (n1) | Δ cycle vs baseline | inferred tok/cycle |
|---|---|---|---|---|
| temp=0.8 pooled | 33.5786 | 67.06 | — | 2.251 |
| temp=1.0 | 32.5353 | 67.36 | **+0.44%** (n2: +0.50%) | 2.192 |

**Acceptance signature confirmed.** Cycle time moved +0.44% — nowhere near the
10% band — while decode moved −3.11%. The *direction* of the small decode drop
is consistent with temperature acting on draft acceptance, and only on
acceptance, exactly as hypothesised. (Corroborating, not independent: the arm
burned **650 cycles for the same 6 requests** — warmup + 5 scored, 256 tokens
each — vs the baseline's 600, i.e. +8.3% cycles for the same tokens, which is
what lower acceptance forces.)

The inferred tok/cycle figures are **INFERRED, not measured** — decode_tps ×
cycle_ms / 1000 — and are reported only because the *direction* is
informative, exactly as `../entropy/RESULTS.md` §Acceptance established:
acceptance is not independently measurable with the current instrumentation
(the count-based estimator is biased by profiler coverage; the
self-consistent estimator is circular).

**Prefill:** 425.93 t/s vs baseline 425.8 — +0.03%, flat, as expected
(temperature cannot affect prefill). Confirms the contrast stayed clean.

---

## 4. Validity gates

| gate | criterion | result |
|---|---|---|
| **G1** | runner PIDs + lstart byte-identical before AND after | **PASS** — genuine runners 83029 (m4-1) / 85554 (m4-2), lstart `Tue Sep 1 16:19:35 / 16:19:37 2026`; driver diffed before/after files, IDENTICAL |
| **G2** | probe rc=0, 5/5 scored iterations returned | **PASS** — rc=0, 5/5, zero probe-side errors |
| **G3** | 0 DEGENERATION/PPSpec/crash/SIGKILL in window; benign noise classified | **PASS** — zero matches on both nodes; 17 Tracebacks per node, all the recurring benign HF-404 (`download_utils:fetch_file_list_with_cache`), independently re-verified by reading the arm's exact log byte-window |
| **G4** | prompt_tokens within 0.5% of 89,408 | **PASS** — 89,408 exact on all 5 iterations |
| **G5** | idle gate < 0.10 gpu_usage_ratio both nodes pre-arm | **PASS** — n1=0.0272, n2=0.0283 |
| **anchor** | (this dispatch) 2MB anchor trustworthy, no pre-window bleed | **PASS** — §1: monotonic counters, 2400 is the immediate predecessor of 2450, zero generation activity in the pre-window gap (live-verified on both nodes) |
| contention | no third-party GPU load during the arm | **PASS** — GPU ~0.94–0.99 on both nodes in lockstep for the full 22-min window, back to ~0.03 idle after |

---

## 5. Anomalies

1. **The 200KB pre-window anchors were empty on both nodes** (0 bytes): the
   last pre-window MTP-PROF dump sits 460,699 bytes (n1) / 617,856 bytes (n2)
   back from the window start — 13% / 12% of the harvest window short of the
   driver's 200KB anchor reach. Handled per the pre-registered caveat: the
   supplemental 2MB anchors were used, and their trustworthiness was
   established before any cycle number was published (§1). Had they failed,
   cycle phases would have been reported UNAVAILABLE; the mechanism check
   would have lost its cycle-time leg but the Layer-1 verdict would be
   unaffected.
2. **The prior study's published sd 1.72 is the population sd (ddof=0); the
   sample sd (ddof=1) of the same 7 baseline values is 1.857.** The entropy
   RESULTS table's "sd" column is ddof=0 throughout (verified against its
   per-iteration lists). The pre-registration's MDE (2.49) inherited 1.72.
   Both are reported in §2.3; the sample-sd version (MDE 2.70) is primary.
   The discrepancy changes no branch — it makes the arm slightly *less*
   powerful than the pre-registration stated.
3. **The baseline's own within-condition session drift (8.62%) is larger than
   this arm's entire treatment effect (~3.1%).** The pooled baseline mixes two
   sub-arms 65 minutes apart (32.77 → 35.60) that the entropy study flagged as
   unexplained upward drift. A single 5-iteration arm measured hours later
   inherits whatever state produced that drift; a ~3% effect is within the
   session's own demonstrated noise, which is a material reason the Welch test
   cannot separate it (p=0.101) and the MDE is ~2.7 t/s. Any future attempt to
   measure small decode effects on this cluster must first explain the drift.
4. **The model card's own comment undercuts the card it lives in.**
   `deepseek-ai--DeepSeek-V4-Flash-0731.toml` carries a comment that DeepSeek
   officially recommends **temp=1.0 / top_p=0.95** (agentic) while the card
   pins temperature=0.8 / top_p=0.9 — and that the card's sampling parity was
   **carried forward from the PREVIEW checkpoint and never re-validated
   against the -0731 re-post-trained weights** (the preview's hard_eval A/B —
   Ollama 100% vs exo 78.8% at the official temp recommendation — is the
   reason the values exist, but it was run on the preview). This is a
   **separate open question about output quality/correctness, explicitly NOT
   part of this verdict**: this study measured throughput at 1.0 vs 0.8 and
   found no production-relevant premise; whether 0.8 is the *right quality
   setting* for -0731 is untested. Flagged for a future quality A/B.
5. **First-interval warmup contamination is present but symmetric.** The first
   50-cycle interval includes the warmup iteration (verify de-aggregates to
   61.49 ms in that slice vs ~54.3 steady-state). The baseline arm 1 has the
   identical structure, so the comparison is fair; excluding the first
   interval gives 66.80/66.79 ms — still flat vs the baseline's 67.06/67.09.

---

## 6. What this leaves open

Sampling temperature is now **closed** as an explanation for real-usage
20 t/s (Layer 1: the production premise never existed; Layer 2: even forced,
the effect is small, insignificant, and underpowered). It closes as suspect 3
of `../entropy/RESULTS.md` §6. Standing suspects, in the order the evidence
now supports:

1. **The 11–16% of decode wall outside profiled phases** (`../REPORT.md` §3) —
   **now the primary suspect by elimination.** Entropy is closed, temperature
   is closed; the cycle phases themselves (verify+draft+accept+rollback ≈
   67 ms) do not move between conditions, so whatever separates bench from
   real usage lives in the unprofiled remainder of decode wall or outside
   decode entirely.
2. **Serving-path overhead** — streaming, detokenization, per-request setup,
   and TTFT on 150K+ prompts. The bench measures server-side
   `generation_tps`; the user experiences `total_tokens / total_wall`. This
   measurement gap is structural: no temperature or acceptance change can
   touch it. (H3 in the original pre-registration, already SUPPORTED.)
3. **Context depth beyond 89K on real prompts** — weak lead (depth scan found
   decode flat to 250K), but it was measured on repetitive prompts only.
4. **The card's sampling-parity provenance** (anomaly 4) — a *quality* A/B
   against DeepSeek's official temp recommendation for -0731, unrelated to
   throughput. Cheap, and worth doing once, but it does not bear on the 14 t/s
   gap.

Measuring acceptance directly would still need `EXO_DSV4_MTP_LOG_INTERVAL`
(relaunch — out of scope for this run), but note that the mechanism check
motivation has evaporated: with temperature closed at the premise, there is no
production acceptance shift left to chase via this lever.

---

## 7. Provenance

- Arm: `run_temp_arm.sh` (driver; mirrors entropy `run_entropy_ab.sh`),
  `temp_probe.py` (byte-copy of `entropy_probe.py` + single additive
  `--temperature` arg), `PREREGISTRATION_TEMPERATURE.md` (decision rule +
  STEP 0 client grep, written before the arm). All pre-existing in
  `temperature/`; this dispatch modified none of them.
- Raw: `raw/temp_temp10.json|.log` (treatment measurements; config echo
  confirms temperature=1.0), `raw/prof_temp10_n{1,2}.txt`,
  `raw/prof_anchor_temp10_n{1,2}.txt` (empty — anomaly 1),
  `raw/prof_anchor_temp10_n{1,2}_2MB.txt` (supplemental, used),
  `raw/errs_temp10_n{1,2}.txt`, `raw/pids_temp10_{before,after}_n{1,2}.txt`,
  `raw/gpu_samples_temp10.log`, `raw/temp_arm_driver.log`,
  `raw/anchor_gap_check.py`, `raw/sanity_{body.json,check.py}` (canary F5).
- Baseline: `../entropy/RESULTS.md` §1 pooled repetitive (n=7) and its raw
  files; cycle-phase reference 67.06/67.09 ms and prefill 425.8 t/s.
- Analysis: de-aggregation via `../analyze_prof.py` (`analyze`,
  `anchors_from`) — the implementation that reproduced the published V3
  figure (56.087 ms verify / 650 cycles) before adoption in the entropy study.
  Statistics via `/Users/adam.durham/repos/exo/.venv/bin/python` +
  scipy 1.17.1 (`t.sf`, `t.ppf`); Welch arithmetic independently coded and
  cross-checked against the PM's figures (§2.4, one small CI discrepancy,
  resolved to the exact-quantile value).
- Anchor forensics: pre-window log gaps (460,699 / 617,856 bytes) read live
  over ssh on both nodes to verify zero pre-window cycle bleed (§1);
  G3 error-window re-verified from the exact driver-recorded byte offsets
  (141198005→146034317 n1, 166599375→172266340 n2).
- Constraints honoured: no relaunch, no kill, no config change, no restart,
  no new benchmark, no commits; no file outside `temperature/` touched
  (entropy study read-only).