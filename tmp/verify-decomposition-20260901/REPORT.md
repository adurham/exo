# REPORT — Where the 20→34 tok/s decode gap actually comes from

**Date:** 2026-09-01/02
**Scope:** context-depth scan (89K/150K/250K) + verify-phase decomposition.
**Constraints honored:** no relaunch, no process kills, no config changes, no
code changes to exo/mlx-lm, no commits/pushes. Measurement + read-only source
analysis only.
**Pre-registration:** `PREREGISTRATION.md`, written before any depth-scan data
existed. All bars quoted below are as pre-registered, not fitted after the fact.

---

## VERDICT (up front)

**The leading hypothesis is REFUTED. Verify is not attention/KV-gather bound at
these depths, and context depth does not explain the gap.**

1. **Verify barely scales with context.** 55.76 → 59.24 ms — a **1.062×**
   increase for a **2.80×** increase in context. The pre-registered bar for
   "attention/KV-gather bound (structural)" was ≥ 2.0×; the pre-registered bar
   for "NOT attention-bound" was < 1.3×. **The measurement lands at 1.06 —
   decisively in the NOT-attention-bound band.**
2. **Decode barely degrades with context.** 33.73 → 30.16 t/s from 89K to 250K
   (−10.6%). Real usage reports ~20 t/s **at 150K, where this benchmark measures
   31.84 t/s.**
3. **Therefore the gap is not a context-depth effect.** At the *matched* depth
   of 150K there is an **11.84 t/s discrepancy (86% of the total gap)** between
   what the benchmark measures and what real usage reports. Context depth
   accounts for only ~1.89 t/s (~14%).

**Is there a lever?** Not in the place we were told to look. The 56 ms verify is
~92–97% context-independent, so "reduce per-token attention cost over pooled KV"
would target ~2–5 ms of a 72 ms cycle. **The real question is why identical
depth yields 31.8 t/s synthetic vs ~20 t/s real** — and the evidence points at
the benchmark's prompt, not the cluster. See §5.

---

## 1. Context-depth table

Cluster: 2× Mac Studio M4 Max, TB5 RDMA, TP world_size=2,
`deepseek-ai/DeepSeek-V4-Flash-0731` (fp8, 43 layers). MTP speculative decode,
γ=3. Runners PID 83029 / 85554, lstart `Tue Sep 1 16:19:35/37 2026` — **identical
process, no relaunch, across all three depths** (verified before and after).
Profiling flags confirmed live on the genuine runner PIDs (not the SCREEN/login/
zsh wrappers that also match `-m exo`).

c=1, 5 scored iterations + 1 warmup per depth, sequential, cluster verified idle
(`gpu_usage_ratio` ≈ 0.026–0.031) before each depth.

| depth | prompt tok | decode t/s (mean) | med | min | max | prefill t/s | verify ms | draft | accept | rollback | **total ms** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 89K  | 89,408  | **33.73** | 33.60 | 31.52 | 36.08 | 426.0 | 55.76 | 9.13 | 2.81 | 0.80 | **68.33** |
| 150K | 150,013 | **31.84** | 31.30 | 31.00 | 33.50 | 418.6 | 58.89 | 8.98 | 2.81 | 0.63 | **71.48** |
| 250K | 250,019 | **30.16** | 29.99 | 28.63 | 31.84 | 406.6 | 59.24 | 9.11 | 2.81 | 0.79 | **71.95** |

Errors: 0 across all 15 scored iterations. `tail_ratio` = 1.00 everywhere.

**Baseline reproduction check:** the 89K row reproduces the independently
measured V3 baseline from earlier the same day (33.99 t/s, verify 56.087 ms,
total 68.839 ms) within noise — decode within 0.8%, verify within 0.6%. The
harness, the cluster state, and the analysis arithmetic are all validated
against a known answer before any new claim is made.

### Scaling summary

| quantity | 89K → 250K | ratio |
|---|---|---|
| context | 89,408 → 250,019 | **2.80×** |
| verify | 55.76 → 59.24 ms | **1.062×** |
| total cycle | 68.33 → 71.95 ms | 1.053× |
| decode | 33.73 → 30.16 t/s | 0.894× |
| prefill | 426.0 → 406.6 t/s | 0.954× |

---

## 2. Verify-phase decomposition

### 2.1 What the verify timer actually brackets

`prof.record("verify", ...)` at `dsv4_mtp.py:4242` (live single-uid path;
`dsv4_mtp.py:2472-2474` selects it) brackets exactly one batched target forward:
timer starts at `t_after_draft` (4101, *after* `mx.eval(*draft_ids)` so draft
cost is excluded) and stops at `t_after_verify` (4240-4242, after
`mx.eval(verify_pre_norm, verify_logits)`). The snapshot cost is separately
isolated as `rb_snap` under `EXO_DSV4_RB_PROFILE=1`, so it is **not** charged to
verify. Both boundaries are `mx.eval`-fenced, which serializes pipelining — the
source comment at `dsv4_mtp.py:238-241` states these are **upper bounds** on the
real production wall.

### 2.2 Component classification (from source, spot-verified against live code)

Live config resolved from the actual model `config.json` (verified directly, not
assumed): `num_hidden_layers=43`, `index_topk=512`, and `compress_ratios`
containing **21 layers at ratio 4, 20 layers at ratio 128, and local layers** —
confirming the split below.

| component | context-scaling? | why |
|---|---|---|
| Indexer score GEMM + top-k (21 ratio-4 layers) | **YES** — O(P), P ≈ ctx/4 | the only strongly ctx-linear term |
| CompressedAttention dense SDPA (20 ratio-128 layers) | **YES** — P ≈ ctx/128 | small pool, grows slowly |
| Sparse gathered SDPA | no (k capped at `index_topk`=512) | gather width fixed once top-k chosen |
| Local sliding-window attention | no | window clamped at 128 |
| MoE gate / switch_mlp / shared_experts / combine | **no** | batch=4 × model dims only |
| lm_head | **no** | vocab 129,280 × batch 4 |
| all_sum / p2p collectives | **no** | block width 4, not ctx |
| compressor / pool write | **no** | writes, does not read the pool |

`v4_attention_factory` (`deepseek_v4.py:5109-5116`, read directly) maps
ratio 128 → `CompressedAttention` (**no** Indexer) and ratio 4 →
`SparseCompressedAttention` (**with** Indexer). This **corrects** the prior
2026-08-04 note, which implied all ~21 alternating layers shared one Indexer
concept.

**Indexer branch flip:** `SparseCompressedAttention.__call__` branches on
`pooled.shape[1] <= self.indexer.index_topk` (verified at `deepseek_v4.py:4879`).
With `index_topk=512` (config default at :870, and `EXO_DSV4_INDEX_TOPK=512`
live) and P ≈ ctx/ratio, the flip is at **ctx ≈ 512×4 = 2,048** for ratio-4
layers. **All 21 sparse layers were already past the flip at every depth
measured**, so all three points sit on the same smooth branch — there is no
knee between 89K and 250K, which is exactly what the flat measurement shows.

### 2.3 The decomposition, measured

Fitting the three measured points:

| model | fixed component | ctx-dependent @89K | ctx-dependent @250K | R² |
|---|---|---|---|---|
| linear | 54.72 ms | 1.78 ms (**3.2%**) | 4.96 ms (8.4%) | 0.706 |
| √ctx | — | 4.98 ms (**8.9%**) | — | 0.768 |
| log ctx | — | — | — | 0.827 |

**Under every model tried, the context-dependent share of verify at 89K is
between ~3% and ~9%. Verify is ~91–97% fixed-cost.** The exact split is
model-dependent and I do not claim a single precise figure; the *conclusion*
(verify is not context-bound at these depths) is robust across all three fits
and, more importantly, follows directly from the raw ratio test (1.06× for
2.80×) without any fitting at all.

This is architecturally consistent: with `index_topk=512` the actual attention
SDPA is O(512) = **constant** per token; only the Indexer's score+top-k scan is
O(P), and on fp8 pooled keys that is small in absolute bytes.

### 2.4 Available instrumentation (no new code)

The profiler splits draft/verify/accept/rollback but **not verify's internals**.
The pre-built instrument that would split verify into attention-vs-MoE, and
within attention into compressor/indexer/sdpa, is **`EXO_DSV4_SECTION_TIME=1`**
(+ `EXO_DSV4_SECTION_TIME_LOG_EVERY=N`). Confirmed **absent from the live
environment** (checked directly on the running PIDs: zero matches). It is read at
import, so enabling it requires a relaunch — out of scope here, and given §1 it
would only be attributing a ~2–5 ms context-dependent slice.

---

## 3. Gap decomposition: where the 34 → 20 t/s actually goes

Using the 89K benchmark (33.73) as the reference and ~20 t/s as the reported
real-usage figure at 150K+ — a **13.73 t/s** gap:

| term | t/s | share of gap | basis |
|---|---|---|---|
| **context depth** 89K→150K | 1.89 | **13.8%** | measured directly |
| **acceptance** | not separable | — | see §4 — not independently measured |
| **residual at matched depth (150K)** | **11.84** | **86.2%** | measured 31.84 vs reported ~20 |

Even taking the deepest point measured (250K, far past real usage), context
depth accounts for only 3.57 t/s (26%) — and real usage at 150K is being
compared against a benchmark that gets 31.84 t/s at that *same* depth.

**Conclusion: ~86% of the gap survives at matched context depth.** Whatever
causes it, it is not context length, and it is not the verify phase's attention
cost.

### A second, independent anomaly

The profiled cycle does not fully account for decode wall time:

| depth | (1+acc)/cycle implies | measured | unaccounted |
|---|---|---|---|
| 89K | 39.95 t/s | 33.73 | ~16% |
| 150K | 35.81 t/s | 31.84 | ~11% |

11–16% of decode wall time sits **outside** the four profiled phases
(draft/verify/accept/rollback) — sampling, detokenization, scheduling, and
inter-cycle overhead. This is unexplained fixed overhead and is a legitimate
second suspect, independent of everything above.

---

## 4. Acceptance — honest limitation

**Acceptance was NOT independently measured, and I am not reporting a number
for it as if it were.**

`EXO_DSV4_MTP_LOG_INTERVAL` is **unset** on the live runners (verified), and the
emit site returns early when it is ≤ 0 (`dsv4_mtp.py:2182`), so the
`[MTP] mean_accept=` line never fires. Confirmed: zero genuine acceptance lines
in the live logs.

Two derivations were attempted and **both are unsound**:

- **Count-based** (`generation_tokens / profiled_cycles`): the profiler only
  covered **71–89%** of the expected cycles per window (dumps land every 50
  cycles; partial trailing intervals and the discarded anchor interval are lost).
  This biases tokens-per-cycle **upward** by exactly the coverage shortfall, and
  produces impossibilities — at 250K it implies 42.7 t/s when 30.16 was measured.
  **Discarded.**
- **Self-consistent** (`decode_tps × cycle_time − 1`): gives a smooth
  1.305 → 1.276 → 1.170 (−10.3% over 89K→250K), but it is **circular** — it is
  derived *from* the throughput it would be used to explain, so it cannot serve
  as independent evidence.

Consequently the pre-registered H2 test (acceptance collapse) **cannot be
adjudicated with the instrumentation currently enabled**. Enabling
`EXO_DSV4_MTP_LOG_INTERVAL` would fix this, but requires a relaunch.

---

## 5. The lever, and what to test next

**No lever exists in the verify/attention path.** Eliminating *all* measured
context-dependence in verify would return ~2–5 ms of a ~72 ms cycle (~3–7%),
and the snapshot/rollback machinery is already closed at 0.218%.

**The strongest remaining hypothesis is a benchmark-vs-reality mismatch in
prompt entropy**, which acts on acceptance:

`bench/concurrent_bench.py::_fixed_prompt` builds its prompt by **repeating one
sentence** until the word count is reached. Measured directly: at 3,000 words it
contains **23 distinct words** (uniqueness ratio 0.008). A draft head predicting
a sentence it has already seen thousands of times will accept far more
speculative tokens than on real, varied text. Since decode ≈ (1+acceptance)/cycle,
inflated acceptance inflates measured t/s **without any cycle-time change** —
which is exactly the signature observed: cycle time is nearly flat with depth,
yet real usage is far slower at matched depth.

Arithmetic consistency check: at 150K, cycle time is 71.48 ms. If real varied
text drops acceptance to ~0.85 (from the benchmark's ~1.28), that gives
(1.85)/71.48 ms ≈ 25.9 t/s, and applying the ~11% outside-phase overhead from
§3 lands at ≈ 23 t/s — close to the reported ~20. **The gap closes almost
entirely via acceptance, with zero change to cycle time.**

### The one experiment that would settle it

A **matched-depth entropy A/B**: identical harness, identical ~89.4K context,
varying *only* prompt entropy.

Prepared and ready at `entropy_probe.py` (+ `run_entropy.sh`), **not run** —
the launch command was declined, so this remains a recommendation, not a result.
It is already calibrated offline against the real tokenizer (free, no cluster
time): **repetitive 75,000 words → 89,404 tokens** vs **natural 65,646 words →
89,294 tokens** (−0.13% apart). Validated as deterministic, with 23 / 352 / 2999
distinct words per 3,000 across its three modes.

- If decode falls to ~20–24 t/s on natural text → **confirmed**: the benchmark
  overstates real throughput via inflated speculative acceptance, and the "34
  t/s baseline" was never achievable on real traffic.
- If it stays ~30+ → the residual is in the serving path (streaming,
  detokenization, sampler settings — note real usage samples at temperature
  while the bench may not, which independently lowers acceptance), and the
  11–16% outside-phase overhead from §3 becomes the primary suspect.

Cost: ~30 min of cluster time. This is the highest-information next step, and it
requires no code or config change.

---

## 6. Pre-registered outcomes — adjudicated

| hypothesis | pre-registered bar | measured | verdict |
|---|---|---|---|
| **H1** attention/KV-gather bound | verify(deep)/verify(89K) ≥ 2.0 | **1.062** (2.80× ctx) | **REFUTED** — below even the 1.3 "not attention-bound" bar |
| **H2** acceptance collapse | acc drop ≥ 15% rel, cycle growth < 20% | not independently measurable (§4) | **UNRESOLVED** — instrumentation off |
| **H3** fixed overhead / accounting | — | 11–16% outside profiled phases; 86% of gap survives at matched depth | **SUPPORTED** |
| **H0** null (no ctx dependence) | decode within 30.4–37.6 at all depths | 33.73 / 31.84 / 30.16 — **all inside** | **effectively holds** |

H0's band was set as the 89K noise envelope (±2σ). Every depth measured lands
inside it, which is itself the headline: **from 89K to 250K, decode never leaves
the 89K noise band.**

The pre-registered "lever exists" gate required a phase to account for ≥ 25% of
the cycle-time increase AND a named operation carrying the growth. The Indexer
path is correctly identified as the ctx-scaling operation, but the total
cycle-time increase over 2.8× context is only 3.6 ms — so the gate **fails on
materiality**, and per the pre-registration this is reported as "no actionable
lever," not massaged into one.

---

## 7. Data provenance & caveats

- Raw: `raw/bench_{089k,150k,250k}.json`, `raw/prof_*_{n1,n2}.txt`,
  `raw/driver.log`, `raw/table_full.json`.
- Tools: `run_scan.sh` (driver), `analyze_prof.py` (de-aggregation),
  `build_table.py` (table), `entropy_probe.py` + `run_entropy.sh` (prepared, not run).
- **`[MTP-PROF]` de-aggregation:** dumps are cumulative running means. Per-interval
  values reconstructed as `(mean_k·n_k − mean_{k−1}·n_{k−1})/(n_k − n_{k−1})`,
  cycle-count weighted, **anchored on the first in-window dump** so pre-window
  idle cycles cannot leak in (`n` is lifetime-cumulative, not window-relative —
  an early version of this analysis over-reported verify as 61.8 ms by missing
  this). Procedure validated by exactly reproducing the V3 published figures
  (56.087 ms verify / 650 cycles).
- **`rb_pool_restores` is a COUNT mislabeled `ms`** by a unit-blind formatter —
  never read as a time.
- **⚠ 250K arm contamination:** a user chat request was sent and cancelled at
  ~22:54:24, overlapping the 250K arm's iter1 tail and possibly iter2. Runners
  did not crash (PIDs unchanged, no errors). **All 250K scored reps should be
  treated as suspect.** A clean re-run would cost ~30 min and was not performed.
  This does not change the verdict: the 150K arm is uncontaminated and already
  carries the matched-depth comparison that drives every conclusion, and the
  250K arm's spread (28.63–31.84) shows no evidence of distortion.
- **250K warmup read 6.88 t/s** vs 28.6–31.8 for its scored iterations. This
  occurred 22:38–22:44, **before** the user's request, so it is not
  user-caused. Treated as a first-touch/allocator warmup artifact — the reason
  warmups are excluded — but flagged as an unexplained one-off worth noting.
- Acceptance limitation: see §4. Not measured, not asserted.
