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
