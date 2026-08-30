# Phase B Review — HC Sinkhorn truncation, real-weights re-validation (P05)

**Date:** 2026-08-30 (post-hoc review of interrupted campaign)
**Scope:** Read-only analysis of already-collected files in
`tmp/p05-sinkhorn-real-20260830/`. No cluster contact, no live benchmarks.
**Data provenance:** Real trained HC weights (`fn`/`base`/`scale`, fp32) for all
86 modules (43 layers × attn/ffn) extracted from the production
`DeepSeek-V4-Flash-0731` checkpoint via `extract_hc_weights.sh` (header-only
scan + range reads, safe beside production). Two analysis scripts:
`fn_sigma_screening.py` (all 86 modules) and `real_weights_sinkhorn.py`
(16 representative modules × 2 input modes). Both ran through the real
`HyperConnection` ops path with the real `EXO_HC_SINKHORN_ITERS` knob and real
checkpoint config (`hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6`). Run log
`run_real_weights.log` shows `EXIT=0` — the run completed cleanly.

---

## 1. The P04 synthetic claim (baseline to re-validate)

P04 (`docs/p04-sinkhorn-truncation-numerics-2026-08-30.md`, commit 2b0824a3c)
used **synthetic** logit distributions (P04's own words: "the actual trained
comb-logit distribution" was the open question) and concluded:

- **"Sinkhorn converges fast" is REFUTED** at realistic logit scales:
  convergence is geometric at ratio ≈0.67/iter; even the full 20 iterations end
  at ~1.1e-3 residual row deviation; a wide-logit (×4) stress case plateaus at
  ~4e-2 by iter 19 (does not converge within 20).
- **Truncation divergence (comb output, worst over 3 param draws × 4 inputs):**
  iters=10 → 1.07e-2 max abs; iters=5 → 5.5e-2; iters=4 → 8.6e-2; iters=3 →
  1.4e-1; iters=2 → 2.3e-1 (realistic O(1)-logit case; wide ×4 is 5-8× worse).
- **Verdict:** P03's "truncate to ~4-5 iters, 5.24 → 1.5-2 ms/cycle" is
  numerically NOT viable. **Minimum defensible candidate = iters=10**
  (1.07e-2 max, "plausibly tolerable, unproven"); anything ≤5 carries
  per-application comb error of the same order as the mixing weights
  themselves. Error enters only via `comb` (the residual-mixing matrix consumed
  by `hc_expand` 86×/forward), so it compounds across depth.
- **Explicit open question P04 flagged:** the real trained comb-logit
  distribution could be tamer (supporting deeper truncation) or wilder
  (refuting even 10). Phase B was meant to answer this.

---

## 2. Phase B real-weight numbers

### 2a. Real logit scale (the key new fact)

`fn_sigma_screening.json` (all 86 modules) — σ_max of the comb-slice fn
(rows 8:24), the exact upper bound on mixes std over all unit-RMS x:

| metric | min | median | max |
|---|---|---|---|
| σ_max(comb_fn) | 4.25 | 8.78 | 21.04 |
| implied max logit std = σ_max·\|scale2\| | 1.01 | 2.42 | 4.91 |
| base2_std (comb bias) | 2.13 | 4.81 | 18.66 |
| row_norm_mean (isotropic mixes std) | 0.89 | 4.32 | 8.49 |

`real_weights_sinkhorn.json` (16 modules × 2 modes) — **measured** comb-logit
std (softmax input):

- **isotropic x** (natural scale): logits_std **2.49–14.90, median 11.4**
- **worstcase x** (top singular direction, adversarial upper bound):
  logits_std **33.9–155.7, median 75.7**

**Critical structural finding:** in the realistic (isotropic) regime the
measured comb logits are **dominated by the fixed bias** `base[8:]` —
`base2_std / logits_std` ratio is **0.85–1.00** across all 16 modules, while
the input-dependent fn-mixes contribution (`scale2·row_norm_mean`) is only
0.03–0.73. The bias is **not** row-constant (0/86 modules), so the
softmax-shift-invariance escape does not apply — the bias genuinely widens the
logits. The worstcase mode is an adversarial bound, not the operating regime.

**Bottom line on scale:** the real comb logits sit at std ~2.5–15 (isotropic,
bias-dominated) to ~34–156 (worstcase) — i.e. **~10–100× larger than P04's
"realistic O(1)" assumption (std=1.0)**. The real distribution is at the
*wilder* end of P04's uncertainty, not the tamer end.

### 2b. Truncation divergence (comb, max_abs, worst over modules)

| iters | P04 synthetic O(1) | P04 synthetic ×4 | **Phase B real isotropic** | **Phase B real worstcase** |
|---|---|---|---|---|
| 10 | 1.07e-2 | 8.70e-2 | **1.33e-1** | 8.00e-2 |
| 8 | — | — | **1.97e-1** | 1.37e-1 |
| 6 | — | — | **2.97e-1** | 2.63e-1 |
| 5 | 5.55e-2 | 5.30e-1 | **3.68e-1** | 3.84e-1 |
| 4 | 8.61e-2 | 5.95e-1 | **4.65e-1** | 5.61e-1 |
| 3 | 1.37e-1 | 6.79e-1 | **6.03e-1** | 7.48e-1 |

Real isotropic divergence is **12.4× worse than P04's O(1) at iters=10**
(1.33e-1 vs 1.07e-2) and **5.4× worse at iters=4** (4.65e-1 vs 8.61e-2). It
lands between P04's O(1) and ×4 stress cases, closer to the ×4 end. The worst
single module is **L00 attn** (div@10=1.33e-1, div@4=4.65e-1, div@3=6.03e-1);
L19/L20/L21 attn+ffn are the next-worst (div@4 ≈ 0.19–0.28).

### 2c. Convergence curve (row_err)

- **isotropic:** iter5 row_err max 3.23e-1 (med 1.34e-1); iter19 row_err max
  8.05e-2 (med 4.29e-2). **Does not converge within 20** — matches P04's
  geometric-slow finding, at a larger scale.
- **worstcase:** bimodal — most modules converge to ~1e-6 by iter 19 (9/16
  have div@10 < 1e-3, i.e. instant convergence), but the worst (L11 ffn, L19/20
  ffn) still show div@4 up to 5.6e-1. The worstcase mode is dominated by a few
  adversarial modules, not representative.

---

## 3. Verdict

**P04's gate is CONFIRMED and STRENGTHENED — the real-weight data does NOT
support loosening it; it argues for tightening.**

- **The real comb-logit distribution is wilder than P04's "realistic O(1)"
  assumption, not tamer.** Real isotropic logits_std ≈ 2.5–15 (median 11.4),
  ~10× P04's std=1.0. P04's explicit open question ("could be tamer, supporting
  deeper truncation") is answered in the negative.
- **Truncation divergence is worse with real weights at every tested point.**
  At iters=10 the real isotropic max divergence (1.33e-1) already exceeds P04's
  iters=4 synthetic value (8.61e-2) — i.e. **even P04's "minimum safe iters=10"
  is not supported by the real data.** Real iters=10 error is ~12× P04's O(1)
  iters=10 estimate.
- **The compounding concern is amplified, not reduced.** The real comb logits
  are bias-dominated (base2_std/logits_std ≈ 0.85–1.00), so the truncation
  error is a largely **input-independent, systematic perturbation** of the
  mixing matrix — the worst case for 86×/forward depth compounding (a fixed
  bias in the same direction every application, rather than something that
  could average out).
- **P03's 4–5-iter projection remains dead.** Real div@4–5 is 0.37–0.47 max
  abs — same order as the mixing weights themselves, confirming P04.

**Recommendation:** Do **not** ship any truncation below 20 as a default. The
real data makes even iters=10 look risky (1.33e-1 max per-application error,
bias-dominated and compounding). If a live throughput test is still desired,
it must (a) use iters=10 at the very most aggressive, (b) be paired with the
generation-quality gate (exo-local-vs-cloud-dsv4 probe suite), and (c) be
reversible per-restart via the `EXO_HC_SINKHORN_ITERS` knob. The honest
position is that the numerics gate now argues for **keeping iters=20** unless
a quality-gated live test proves otherwise.

---

## 4. Caveats / data sufficiency

- **Sufficient to conclude on the direction** (real weights are wilder, gate
  holds/needs tightening). The 16-module sample covers representative depths
  (0,1,3,11,19,20,21,41) plus the 4 widest by σ_max — a reasonable spread.
- **Not measured:** 43-layer compounding with real weights (still unmeasured,
  as in P04); live throughput delta; generation-quality impact. The worstcase
  mode is an adversarial bound (top singular direction), not the real residual
  stream — the true operating regime is bracketed by isotropic (bias-dominated,
  logits_std ~11) and worstcase, and the isotropic numbers are the ones that
  matter for the verdict.
- **One residual unknown:** the real residual-stream x is neither isotropic nor
  the top singular direction. But since the isotropic (realistic) case already
  shows logits_std ~11 and divergence worse than P04, the conclusion is robust
  to this uncertainty.
- **No anomalies found in the data:** run completed cleanly (EXIT=0), all 86
  modules present in the screening, 32 rows in the sinkhorn analysis, numbers
  internally consistent (divergence monotonic in truncation, convergence curve
  matches divergence at each point).

---

## 5. Files reviewed

- `tmp/p05-sinkhorn-real-20260830/real_weights_sinkhorn.json` (32 rows)
- `tmp/p05-sinkhorn-real-20260830/fn_sigma_screening.json` (86 rows)
- `tmp/p05-sinkhorn-real-20260830/run_real_weights.log`
- `tmp/p05-sinkhorn-real-20260830/real_weights_sinkhorn.py` / `fn_sigma_screening.py`
- `tmp/p05-sinkhorn-real-20260830/extract_hc_weights.sh` (read only, not run)
- `tmp/p05-sinkhorn-real-20260830/p05_weights/manifest.json` + `.bin` artifacts
- `docs/p04-sinkhorn-truncation-numerics-2026-08-30.md`, `docs/PERFORMANCE_HISTORY.md` (P04 section)
