# PRE-REGISTRATION — matched-depth entropy A/B

**Written:** 2026-09-02, BEFORE any entropy-arm request was sent.
Extends `../PREREGISTRATION.md`. Fixes the decision rule *in advance* so no
result can be rationalized after the fact.

## Environment anchor (verified at write time)

- Runner PIDs **83029** (m4-1) / **85554** (m4-2), lstart
  `Tue Sep 1 16:19:35 / 16:19:37 2026` — **unchanged** from the V3 and
  depth-scan runs. No relaunch between those runs and this one.
- `EXO_DSV4_MTP_PROFILE=50`, `EXO_DSV4_RB_PROFILE=1` confirmed live via
  `ps eww` on the **genuine runner PIDs** (not the SCREEN/login/zsh wrappers
  that also match `-m exo`).
- GPU idle at write time: n1 `0.0268`, n2 `0.0295` (both < 0.10).
- Two other live Hermes CLI sessions exist on this host. The user reports being
  on ollama-cloud. Contamination will be checked per-arm (idle gate + GPU
  sampling + log error scan), and any overlap disclosed.

## Design

Single variable: **prompt entropy**. Context depth held at ~89.4K tokens.
Word counts calibrated OFFLINE against the real DSv4-Flash tokenizer
(`transformers` `AutoTokenizer`, `local_files_only=True`, raw-prompt
`tokenizer.encode()`), independently re-verified for this run:

| mode | words | tokens | Δ vs repetitive |
|---|---|---|---|
| repetitive | 75,000 | 89,404 | — |
| natural | 65,646 | 89,294 | −0.123% |
| random | 23,525 | 89,406 | +0.002% |

All three land inside ±0.13% — matched depth.
The prior run's two calibration claims (89,404 / 89,294) **reproduced exactly**,
so the identical method was used to derive `random`.

Per arm: **1 warmup + 5 scored iterations**, `max_tokens=256`, concurrency 1,
sequential, idle-gated. Seed fixed at 1234 (prompts deterministic).

## Validity gates (checked BEFORE interpreting the contrast)

The A/B is only interpretable if all hold:

1. **G1 — baseline reproduction.** The `repetitive` arm must land inside the V3
   89K noise band **30.4 – 37.6 t/s**. Outside ⇒ cluster state drifted since
   V3; the contrast is confounded (config AND state differ) and no verdict is
   issued.
2. **G2 — no drift across the run.** A `repetitive_recheck` arm (1 warmup +
   2 scored) runs **last**. If it deviates from the opening `repetitive` arm by
   **> 10% relative**, the run is drift-contaminated and the verdict is
   downgraded to provisional.
3. **G3 — integrity.** 0 request errors, no runner restart (PID + lstart
   identical before/after every arm), no `DEGENERATION DETECTED` in the log
   window.
4. **G4 — matched depth realized.** Server-reported `prompt_tokens` within
   ±0.5% across all arms. (Guards against a chat-template or truncation
   surprise that offline calibration cannot see.)

## Decision rule (pre-committed)

Let `R` = mean decode t/s of `repetitive`, `N` = `natural`, `X` = `random`.

- **INFLATED (hypothesis confirmed):** `N ≤ 24` **and** `X ≤ 24`.
  ⇒ The benchmark baseline was inflated by prompt entropy via speculative
  acceptance. The cluster's real-traffic speed is the natural/random number.
  Every campaign decode figure resting on the repetitive baseline needs
  re-examination. To be stated plainly, without hedging.
- **NOT INFLATED (hypothesis refuted):** `N ≥ 30` **and** `X ≥ 30`.
  ⇒ Entropy is not the mechanism. The gap lives elsewhere — primary suspect
  the 11–16% of decode wall outside profiled phases (REPORT §3), secondary the
  serving path (streaming/detokenization/temperature).
- **PARTIAL:** anything else (incl. `N` and `X` disagreeing, or either landing
  in the 24–30 band). Reported as partial with the exact numbers and the
  fraction of the 34→20 gap explained. **Not** rounded toward either clean
  outcome.

**Effect-size reporting is mandatory in all three branches:** report
`(R − N)/R` and `(R − X)/R` as percentages, and state what fraction of the
34→20 t/s gap (14 t/s) each accounts for.

## Mechanism check (secondary, non-gating)

If decode falls, the pre-registered mechanism is **acceptance**, not cycle time.
Signature to confirm it:

- cycle time (`[MTP-PROF]` verify + draft + accept + rollback) stays **flat**
  (< 10% change) across arms, while decode t/s moves — implicates acceptance.
- cycle time grows in proportion to the decode drop — implicates cycle cost,
  and the entropy story is **wrong even if decode falls**.

Acceptance is **not** independently measurable with current instrumentation
(REPORT §4: the count-based estimator is biased by profiler coverage, the
self-consistent one is circular). Any acceptance figure derived here is
labeled **inferred**, never asserted as measured. `EXO_DSV4_MTP_LOG_INTERVAL`
would fix this but needs a relaunch — out of scope.

## Method constraints (inherited, re-affirmed)

- `[MTP-PROF]` dumps are **cumulative running means**; per-interval values
  reconstructed as `(mean_k·n_k − mean_{k−1}·n_{k−1})/(n_k − n_{k−1})`,
  cycle-count weighted, **anchored on the last pre-window dump**.
- `rb_pool_restores` is a **COUNT** mislabeled with an `ms` suffix — never read
  as a time.
- No relaunch, no process kill, no config change, no code change to the
  calibrated probe. New harness code is additive and lives in `entropy/`.
