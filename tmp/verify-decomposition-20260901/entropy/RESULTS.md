# RESULTS — matched-depth entropy A/B

**Run:** 2026-09-02, 08:24–09:41 CDT. 4 arms, 21 scored deep-context requests.
**Pre-registration:** `PREREGISTRATION_ENTROPY.md` (written before any request).
**Question:** is the 34 t/s decode baseline inflated by the benchmark's
low-entropy repetitive prompt?

---

## VERDICT

**No.** The 34 t/s baseline is **not** meaningfully inflated by prompt entropy.

Natural high-entropy prose runs at **33.58 t/s** — statistically
indistinguishable from the repetitive benchmark prompt's pooled **33.58 t/s**
(Welch t = −0.00). The pre-registered "inflated" outcome required natural to
fall to 20–24 t/s. It did not fall at all.

The entropy hypothesis from `../REPORT.md` §5 — that the benchmark overstates
real throughput via inflated speculative acceptance — is **refuted for
realistic text**. The `_fixed_prompt` repetitive prompt, despite containing
only 23 distinct words, is a fair throughput proxy for natural prose at 89K
depth.

A real but modest entropy effect **does** exist, but only at adversarial
entropy: uniform random letter-strings drop decode to 28.66 t/s (−12.6%).
That accounts for **35% of the 14 t/s gap**, and random letter-strings are not
what real traffic looks like. Natural prose accounts for **0%**.

**The 20 t/s the user experiences in real sessions is still unexplained.**
This experiment closes prompt entropy as the cause and hands the question back
to the remaining suspects (below).

---

## 1. Per-mode table

| arm | mode | prompt tok | decode mean | median | min | max | sd | prefill t/s | cycle ms (n1/n2) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | repetitive | 89,408 | **32.77** | 33.02 | 31.72 | 34.00 | 0.89 | 425.8 | 67.06 / 67.09 |
| 2 | natural | 89,298 | **33.58** | 32.83 | 30.95 | 38.90 | 2.76 | 430.6 | 66.09 / 66.11 |
| 3 | random | 89,410 | **28.66** | 27.77 | 27.51 | 30.19 | 1.24 | 429.6 | 65.45 / 65.49 |
| 4 | repetitive_recheck | 89,408 | **35.60** | 35.60 | 33.96 | 37.23 | 1.63 | 426.4 | 68.51 / 68.55 |

Per-iteration decode (t/s):

- repetitive: 31.72, 31.79, 33.02, 34.00, 33.33
- natural: 32.83, 38.90, 30.95, 32.17, 33.07
- random: 27.77, 27.51, 30.19, 27.67, 30.16
- repetitive_recheck: 37.23, 33.96

Arms 1 and 4 are the **same condition** measured 65 minutes apart, so the
honest repetitive distribution is the pool of both: **n=7, mean 33.58,
sd 1.72, range 31.72–37.23**.

### Cycle-phase breakdown (`[MTP-PROF]`, interval-weighted, node 1)

| phase | repetitive | natural | random | recheck |
|---|---|---|---|---|
| verify | 54.49 | 53.66 | 53.10 | 55.77 |
| draft | 9.06 | 8.98 | 8.89 | 9.12 |
| accept | 2.74 | 2.70 | 2.59 | 2.82 |
| rollback | 0.76 | 0.74 | 0.86 | 0.79 |
| **total** | **67.06** | **66.06** | **65.45** | **68.51** |
| cycles | 600 | 650 | 750 | 300 |
| `rb_pool_restores` *(count, not ms)* | 16.30 | 16.42 | 16.96 | 17.69 |

De-aggregated per pre-registered arithmetic
`(mean_k·n_k − mean_{k−1}·n_{k−1})/(n_k − n_{k−1})`, cycle-count weighted.
Counters ran a clean monotonic 50→600 from a fresh start (cluster idle ~9 h
before the run), so no pre-window anchor was required for arm 1 and no idle
cycles leaked in.

### Acceptance

Not independently measurable with current instrumentation, exactly as
`../REPORT.md` §4 established (count-based estimator biased by profiler
coverage; self-consistent estimator circular). The figures below are
**inferred**, not measured, and are reported only because the *direction* is
informative:

| arm | decode t/s | cycle ms | inferred tok/cycle |
|---|---|---|---|
| repetitive | 32.77 | 67.06 | 2.198 |
| natural | 33.58 | 66.09 | 2.220 |
| random | 28.66 | 65.45 | 1.876 |
| recheck | 35.60 | 68.51 | 2.439 |

---

## 2. Pre-registered interpretation, applied

The decision rule, fixed in advance:

| branch | condition | met? |
|---|---|---|
| INFLATED | N ≤ 24 **and** X ≤ 24 | **No** — N=33.58, X=28.66 |
| NOT INFLATED | N ≥ 30 **and** X ≥ 30 | **Partly** — N=33.58 ✓, X=28.66 ✗ |
| PARTIAL | anything else | **← this branch** |

Formally this lands in **PARTIAL**, because random (28.66) falls in the 24–30
band. Per the pre-registration this is reported as partial and **not rounded
toward a clean outcome**. But the two sub-results point in clearly different
directions and must not be averaged into a mush:

- **Natural prose: complete null.** −2.48% effect vs arm 1; **0.0%** of the
  14 t/s gap explained. Against pooled repetitive the means are identical to
  two decimals (33.58 vs 33.58, t = −0.00, df = 6.1). There is no entropy
  effect on realistic text at all.
- **Random letter-strings: real, modest.** −12.55% vs pooled repetitive,
  t = +5.25, df = 9.9; distributions do not overlap (repetitive floor 31.72 >
  random ceiling 30.19). Explains **35.1%** of the 14 t/s gap.

The pre-registered consequence text for the INFLATED branch — "every campaign
decode number built on the repetitive baseline needs re-examination" — **does
not apply**. Campaign decode numbers built on the repetitive prompt stand.

### Mechanism check (pre-registered, secondary)

The pre-registration stated: if decode falls, it must be **acceptance**, not
cycle time, and named the signature to distinguish them.

**Acceptance confirmed as the mechanism.** Random has the *lowest* cycle time
of any arm (65.45 ms vs repetitive's 67.06) yet the *lowest* throughput.
Cycle time varies only 65.45–68.51 ms across all four arms (4.7% spread,
inside the pre-registered <10% flat band) while decode moves 28.66–35.60.
Throughput is being set by tokens-accepted-per-cycle, not by cycle cost.

So the *mechanism* proposed in `../REPORT.md` §5 is real — entropy does act on
acceptance, and only on acceptance. Its *magnitude on realistic text is zero*.
The hypothesis was right about the pathway and wrong about the effect size.

---

## 3. Validity gates

| gate | criterion | result |
|---|---|---|
| **G1** baseline reproduction | repetitive inside V3 band 30.4–37.6 t/s | **PASS** — 32.77 (and 35.60 on recheck) |
| **G2** no drift | repetitive vs recheck within 10% | **PASS** — 8.62% |
| **G3** integrity | 0 errors, no restart, no degeneration | **PASS** |
| **G4** matched depth | prompt_tokens spread ≤ 0.5% | **PASS** — 0.125% (89,298–89,410) |

- **G3 detail:** runner PIDs **83029** (m4-1) / **85554** (m4-2), lstart
  `Tue Sep 1 16:19:35 / 16:19:37 2026` — byte-identical before *and* after all
  four arms. Probe `rc=0` on every arm. 21/21 scored iterations returned; zero
  probe-side errors. The 32–68 lines matched by the error grep per arm are
  **entirely one benign recurring warning** —
  `download_utils:fetch_file_list_with_cache` HTTP 404 against HuggingFace,
  firing once per ~78 s from a background catalog-refresh task unrelated to
  inference. Zero matches for degeneration, PPSpec, crash, or SIGKILL.
- **Idle gate** held before every arm (all four ≤ 0.032 on both nodes).
- **No user contention observed.** GPU sat at ~0.95 throughout each arm and
  returned to ~0.027 after. No third-party request appeared in any arm window.

---

## 4. Robustness of the central claim

The natural arm has the widest spread (sd 2.76), driven by one 38.90 outlier.
The conclusion does not depend on it:

| natural, treatment | n | mean | vs 24 t/s bar |
|---|---|---|---|
| all iterations | 5 | 33.58 | +40% above |
| outlier dropped | 4 | 32.26 | +34% above |

Either way natural is indistinguishable from repetitive and nowhere near the
20–24 band the hypothesis predicted.

**The random effect is not a drift artifact.** Arms ran in temporal order
repetitive (32.77) → natural (33.58) → random (28.66) → recheck (35.60). The
baseline trend across the session is *upward*, and random sits below **both**
of its temporal neighbours. Drift pushes against the random deficit, so the
true effect is if anything slightly larger than measured — and correspondingly,
the natural null is not being propped up by a favourable drift.

---

## 5. Anomalies

1. **Upward drift within the repetitive condition (8.62%).** 32.77 at 08:24
   vs 35.60 at 09:30, same prompt, same config, PIDs unchanged. Passes the
   10% gate but is *comparable in magnitude to the random effect itself*
   (12.55%), which is the main reason the random result is reported as modest
   rather than decisive. Cycle time moved with it (67.06 → 68.51 ms), so the
   drift is not purely acceptance-side. Unexplained; consistent with warm-up
   or allocator/residency state. n=2 on the recheck arm limits confidence
   (t = −1.67, df = 1.2 — not significant).
2. **Natural arm variance** (sd 2.76 vs 0.89/1.24 elsewhere), single 38.90
   iteration. Does not change the conclusion (§4).
3. **Random arm ran more cycles** (750 vs 600–650) for the same 256 max_tokens
   — the expected consequence of lower acceptance needing more cycles per
   token. Internally consistent with the acceptance mechanism.
4. **Benign HF 404 warning storm** in every arm window (§3). Pre-existing
   background behaviour, not caused by this run, but it makes a naive
   `grep -i error` on these logs misleading. Worth fixing separately.

---

## 6. What this leaves open

Prompt entropy is now **closed** as an explanation for real-usage 20 t/s.
Standing suspects, in the order the evidence supports:

1. **The 11–16% of decode wall outside profiled phases** (`../REPORT.md` §3) —
   now the primary suspect by elimination. The pre-registered NOT-INFLATED
   branch named this explicitly.
2. **Serving-path overhead** — streaming, detokenization, per-request setup.
   The bench measures server-side `generation_tps`; the user experiences
   `total_tokens / total_wall` including TTFT on a 150K+ prompt (H3 in the
   original pre-registration, already SUPPORTED).
3. **Sampling temperature** — real sessions sample at temperature, the bench
   may not. This independently lowers acceptance and was never controlled for
   here. Cheap to test and never tested.
4. **Context depth beyond 89K** — real sessions run 150K+; the depth scan
   found decode never left the 89K noise band up to 250K, so this is a weak
   lead, but it was measured on repetitive prompts only.

**Not recommended:** further entropy work. The effect on realistic text is
zero and the adversarial-entropy effect is smaller than the session drift.

Measuring acceptance directly would need `EXO_DSV4_MTP_LOG_INTERVAL`, which
requires a relaunch — out of scope for this run.

---

## 7. Provenance

- Driver: `run_entropy_ab.sh` (4 arms, idle-gated, PID + error + anchor
  harvest per arm). Analysis: `build_entropy_table.py`. Both new, in
  `entropy/`; no existing study artifact was modified.
- Raw: `raw/entropy_{repetitive,natural,random,repetitive_recheck}.json|.log`,
  `raw/prof_ent_*_n{1,2}.txt`, `raw/errs_*_n{1,2}.txt`,
  `raw/pids_*_{before,after}_n{1,2}.txt`, `raw/entropy_ab_driver.log`,
  `raw/entropy_table.json`.
- Word counts independently re-calibrated offline against the real DSv4-Flash
  tokenizer before the run. Both prior claims reproduced **exactly**
  (repetitive 75,000 → 89,404; natural 65,646 → 89,294); random derived by the
  identical method (23,525 → 89,406). Server-reported depths came in at
  89,408 / 89,298 / 89,410 — within 0.125%.
- De-aggregation validated by reproducing the published V3 figure
  (56.087 ms verify / 650 cycles) from the V3 run-1 logs before use.
- Constraints honoured: no relaunch, no process kill, no config change, no
  code change to the calibrated probe, no commits. API requests only.
