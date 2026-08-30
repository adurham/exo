# Phase A Review — lm_head mxfp8 quantization (independent re-derivation)

**Reviewer:** mid-coder subagent (read-only analysis of already-collected data)
**Date:** 2026-08-30
**Data dir:** `tmp/p05-lmhead-mxfp8-20260830/`
**Claim under test (from uncommitted `mlx-lm/mlx_lm/utils.py` comment):**
mxfp8 lm_head gives 1.64–1.84x speedup at production call shapes, ~13% top-1
flip rate on synthetic inputs (concentrated in near-ties), extrapolated ~16%
real-token flip rate from live generation margins (n=798). markov_w2
deliberately NOT quantized (0.98x microbench, latency-bound, no benefit).

---

## 1. Recomputed numbers

### 1a. Kernel-level speedup (studio microbench) — CONFIRMED
`studio_lmhead_microbench.json` (host Adams-Mac-Studio-M4-1, applegpu_g16s):

| op | M | BF16 us | mxfp8 us | speedup |
|----|---|---------|----------|---------|
| lm_head | 1 | 2249.9 | 1215.0 | **1.85x** |
| lm_head | 3 | 2244.9 | 1293.0 | **1.74x** |
| lm_head | 4 | 2238.0 | 1367.8 | **1.64x** |
| markov_w2 | 1 | 253.9 | 254.7 | **1.00x** |
| sampler (L=4) | 4 | 2328.7 | 1455.7 | 1.60x |

- The **1.64–1.84x** range matches the microbench exactly (M=4 1.64x, M=1 1.85x).
- Cycle projection: BF16 7.57 ms vs mxfp8 4.88 ms → **−2.69 ms/cycle = −0.84 ms/token** (matches the comment's −2.7 ms/−0.84 ms).
- **markov_w2 = 1.00x, not 0.98x** as the comment claims. The conclusion (no benefit, don't quantize) is still correct, but the specific "0.98x" figure is wrong — it's a wash, not a small loss.

### 1b. Live A/B throughput — NOT CONFIRMED / CONTRADICTED
`live_ab/` (v1) decode_tps (quant vs base):

| condition | base (med) | quant (med) | ratio |
|-----------|-----------|-------------|-------|
| 5k decode | 68.6 (75.4/68.6/7.4) | 4.0 (4.0/4.0/4.4) | **0.06x** |
| 5k prefill | 235.7 | 44.8 | 0.19x |
| 100k decode | 99.1 (86.8/99.1/100.5) | 5.0 (5.0/5.1/5.0) | **0.05x** |
| 100k prefill | 377.8 | 256.9 | 0.68x |

The live quant arm is **16–20x SLOWER** in decode, not 1.64–1.84x faster. This is
the catastrophic zero-acceptance state documented in `diagnose_zero_accept.py`
("5K+ probes: acceptance 0.000/3 on EVERY cycle, ~550ms draft, 5 tok/s"). The
quantized head broke the draft/verify match in production, so these throughput
numbers are **NOT a valid measurement of a working quantized head** — they are a
failure signature, not a speedup.

`live_ab_v2/` (v2, arm A = knob ON): A_5k healthy (~200 tps decode, ~310 tps
prefill), but **all three A_100k files are connection-refused tracebacks** (no
data). So there is **no valid live 100k throughput measurement of a working
quantized head anywhere in the data**.

`quant_100k_retry1/2`: prefill back to ~379 (matches base) but decode 8.78 and
78.54 — inconsistent, no clean signal. `quant_8k_diag`/`quant_16k_diag` show
healthy prefill (303/341) and decode (101/108) — these look like base-like
behavior, not a quantized-head measurement.

**Bottom line on speedup:** the 1.64–1.84x is a *kernel microbenchmark* number
that was **never validated end-to-end in the live data present**. The only live
quantized-head runs show a catastrophic regression (0.05–0.06x), and the v2
100k re-runs all failed to collect data.

### 1c. Synthetic top-1 flip rate — CONFIRMED (~13%)
`lmhead_numerics_v2.json` (128 real-weighted rows, M=1): **17/128 = 13.28%**
flips. Margin stratification:

| margin band | n | flips | flip_rate |
|-------------|---|-------|-----------|
| [0.00, 1.44) | 32 | 13 | 40.6% |
| [1.44, 3.62) | 32 | 4 | 12.5% |
| [3.62, 7.50) | 31 | 0 | 0.0% |
| [7.50, inf) | 33 | 0 | 0.0% |

Flips are **100% concentrated in margin < 3.62** (0% for margin > 3.62). Logit
err: mean 0.53, rms 0.68, max 7.5, vs logit std 11.3. top-5 set overlap 90%.
This confirms the "~13% synthetic, concentrated in near-ties" claim.

### 1d. ~16% real-token flip rate (n=798) — CANNOT BE VERIFIED
The n=798 live-generation margin distribution is **not present anywhere in the
data directory** (nor in sibling p05 dirs, docs, or bench). The only occurrence
of "n=798" in the repo is the `mlx_lm/utils.py` comment itself. The
`PRE_REGISTERED_GATES.md` references "expected real-token flip rate ~16%
(near-ties only)" and "~58% of real tokens below 3.6" but **no raw margin
array, histogram, or per-token data backing these figures exists in the
collected files**. This specific claim is unverifiable from the data present.

### 1e. markov_w2 decision — CONFIRMED as sound, figure slightly off
The knob (`EXO_DSV4_LMHEAD_MXFP8`) quantizes **only** `Model.lm_head`, never
markov_w2 (verified in the utils.py diff). The live A/B data therefore reflects
the "markov_w2 not quantized" decision. Microbench shows markov_w2 = 1.00x
(comment says 0.98x — minor discrepancy, conclusion unchanged).

---

## 2. Verdict: **DON'T SHIP** (based on data present)

- The **kernel-level 1.64–1.84x speedup is real** (microbench-confirmed) but was
  **never validated end-to-end**. The only live quantized-head runs show a
  catastrophic 0.05–0.06x regression (zero-acceptance draft/verify break), and
  the v2 100k re-runs that might have shown a working head all failed to
  collect data (connection refused).
- The **~13% synthetic flip rate is confirmed**, and it is a real quality cost
  (100% of flips in near-ties, but 13% of greedy tokens flip).
- The **~16% real-token flip rate is unverifiable** — the n=798 data is absent.
- **markov_w2 not quantizing is the right call** (1.00x, no benefit).

A ship decision would require: (1) a valid live 100k A/B of a *working*
quantized head (the zero-acceptance bug must be fixed first — see
`diagnose_zero_accept.py` H5: QuantizedLinear inside `@mx.compile`), and
(2) the n=798 real-margin data to substantiate the ~16% real flip rate.

---

## 3. Data quality issues found

1. **All `live_ab/*.json` files have a trailing `saved -> ...` line** appended
   after the JSON object — not valid JSON; requires tolerant parsing (the
   `json.load` in the harness would fail on these).
2. **`live_ab_v2/A_100k_1/2/3.json` are connection-refused tracebacks**, not
   data (cluster was down). `A_100k_1` has a JSON object with all-null/NaN
   fields; `A_100k_2/3` are pure tracebacks.
3. **`warmup_a1.json` is a 503 traceback**; `warmup_a2.json` has NaN prefill/decode.
4. **`base_5k_3.json` is an outlier** (decode 7.39 tps, decode_s 9.61s vs ~1s
   for the other base runs) — likely a broken run; distorts the base_5k median.
5. **The v1 quant arm was in a broken zero-acceptance state** (per
   `diagnose_zero_accept.py`), so its throughput numbers are invalid as a
   speedup measurement — they were never a valid A/B of a working head.
6. **No n=798 real-margin data** — the ~16% real flip rate and "~58% below 3.6"
   figures are asserted in comments/gates but have no backing raw data file.
7. **G5 same-prompt divergence gate not completed**: `same_prompt_A.json` exists
   (arm A output) but there is no paired arm-B file and no byte-diff computed.
8. **`quant_100k_retry1/2` and `quant_8k/16k_diag` are inconsistent** with the
   main quant runs (retry prefill matches base; diag decode ~100 tps) — no
   clean interpretation, suggests the knob state / cluster state varied across
   runs.
9. **Tiny sample sizes**: synthetic flip rate is n=128 (one seed); live A/B is
   n=3 per arm per condition with one broken run each; surrogate replay is
   n=12 rows total (diagnostic only, no signal).

---

## 4. Files created
- `tmp/p05-review-20260830/phaseA_review.md` (this document)

---

## Follow-up (root-cause investigation)

**2026-08-30 · LEAD 1 forensics (read-only) · full evidence in `tmp/p05-lmhead-mxfp8-20260830/rootcause2/`**

### Verdict: the zero-acceptance regression was a wrong-model harness artifact, not a QuantizedLinear bug

The catastrophic 0.05-0.06x zero-acceptance regression was **never a property of the mxfp8-quantized 0731 head**. It was measured against a **different model** — `mlx-community/DeepSeek-V4-Flash` (8-bit) — JIT-loaded under **pipeline parallelism** (single-node, `min_nodes=1`) for the 11:19-11:50 window, a configuration with a documented depth-dependent spec-decode collapse (100% zero-acceptance, draft ~550ms).

**Root cause of the false claim:** `bench/ab_probe_tier1.py` hardcoded `MODEL='mlx-community/DeepSeek-V4-Flash'` until commit `de925720e` (13:04) added `--model`. Every "quant" run in `live_ab/` hit the 8-bit mlx-community model — which the knob **cannot** quantize (its `lm_head` already has `.scales`, so the `not hasattr(mod,"scales")` guard at `mlx_lm/utils.py:620` no-ops; `model_type=='deepseek_v4'` and `EXO_DSV4_LMHEAD_MXFP8=1` both pass) — under a degenerate single-node Pipeline placement. The 0.05-0.06x decode and ~550ms draft signature were the Pipeline-parallel mlx-community instance's spec-decode collapse, wrongly attributed to the quantized 0731 head.

**Healthy quantized-head evidence (12:32-12:45, instance 99e6a0a5, knob ON):** `[MTP] cycles=237 mean_accept=1.890/3 hist=0:43,1:37,2:60,3:97`; `[MTP-PROF]` draft ~9ms / verify ~66ms / total ~78ms; decode ~200-223 tok/s at 5.6K prompt tokens. The 11:13 "Say OK" probe (instance 6a7f098e, knob ON) was also healthy (mean_accept ~1.3-1.6/3).

**Unmeasured regimes on the quantized head:** ≥8K batched verify (`EXO_DSV4_VERIFY_BATCH=1`, `MIN_CTX=8192`) was **never** exercised; 100K was killed mid-prefill (88064/111074) by the 12:45 clean manual shutdown (SIGTERM, exit 0; new cluster pid 65573 started 12:46:02).

**Deliverables:** `rootcause2/attribution_table.json`, `rootcause2/v2_0731_acceptance.json`, `rootcause2/RUN_ATTRIBUTION.md`.

---

## Follow-up 2 (PM, same day): zero-acceptance bug CLOSED as harness misattribution; true 0731 baseline + real margin data collected

**The "zero-acceptance draft/verify bug" never existed.** Full log-forensic
attribution in `../p05-lmhead-mxfp8-20260830/rootcause2/` (RUN_ATTRIBUTION.md,
30-row attribution_table.json): every catastrophic 0.05-0.06x "quant" run
(11:19-11:50) was served by `mlx-community/DeepSeek-V4-Flash` (8-bit, lm_head
already quantized → the mxfp8 knob silently no-ops on it) JIT-loaded under
SINGLE-NODE PIPELINE parallelism — a placement with a documented depth-dependent
spec-decode collapse (draft ~500-552ms, mean_accept 0.000/3, knob-independent:
the knob-off base arm showed the same early zero-acceptance). The probe harness
hardcoded that wrong model id until `de925720e` (13:04) added `--model`. The
knob-quantized 0731 head's own live measurements were healthy: trivial ctx +
5.6K ctx (rowseq verify) at mean_accept 1.890/3, draft ~9ms, decode ~200-223
tok/s. H1-H5 (including the H5 QuantizedLinear-in-mx.compile theory) are all
moot — there was no bug to explain.

**Correction to this review's own §1b:** the "99.1 tok/s base 100k decode"
figure was ALSO mlx-community data and is not a valid production baseline.
The TRUE 0731 production baseline (live_ab_v3/, clean verbon3 cluster, model
id verified in every file, needle_hit true in all runs):
- 100K decode: **271.4 / 345.1 / 276.8 tok/s (median 276.8)**, prefill
  374.8-376.5 tok/s — ~2.8x the figure this review carried.
- 5K decode: 169.4 / 289.0 / 368.6 tok/s; 1K warmup decode 133.6 tok/s.

**Missing n≥798 margin data — collected (real_margins/, n=3999 committed
tokens across 35 / 2.5K / 6.3K / 25.1K-ctx temp-0 generations, top1-vs-top2
logprob margins via /v1 logprobs; every run reached 1000 committed tokens):**
- Pooled fraction with margin < 3.62: **42.7%** (n≥798 slice: **44.5%**) —
  the prior campaign's "~58% below 3.6" is **REFUTED**.
- Implied mxfp8 top-1 flip rate (synthetic-band kernel — an estimate, not a
  direct measurement): **~11.5% pooled / ~12.8% at n≥798** — the asserted
  "~16%" is **REFUTED** (overstated by ~40%).
- Median margin 4.875, mean 6.75, p10 0.5, p90 15.5. Stable across contexts
  (per-context frac<3.62 within 42.0-43.6%) — the flip risk does NOT grow
  with context depth in this sample.

**G5 same-prompt divergence (completed):** arm A (mxfp8 head, 12:37 capture)
vs arm B (BF16, live) on the fixed temp-0 prompt: byte-identical for 199
chars, then diverge mid-reasoning ("all colors" vs "different wavelengths")
— same_prompt_G5_result.json. Real visible quality cost, as expected.

**Remaining open on Phase A (unchanged):** ≥8K batched-verify and 100K on the
quantized head are still UNMEASURED (the healthy evidence covers ≤5.6K,
rowseq). A ship decision still needs a knob-ON relaunch A/B at 100K against
the true baseline above — requires supervisor relaunch approval, not taken.
