# Q1e — mining the 531-token × 40-layer MoE routing trace (DSv4.1, E=384, topk6)

**Source (read-only).** `docs/benchmarks/phase10-planb-gate-2026-09-28/raw/p30_exl3_trace_clamp.json`
sha256 `b4f1b142a5c5ef779e87b8eb3f041f4e822fdc5de8fe858ea9c8fba1bde91a11`.
Real captured **production clamp** routing (`silu_clamp`); 531 tokens × 40 layers = 21 240
`[layer, [6 expert ids]]` records, **layer-major** (`records[L*531:(L+1)*531]` = layer L,
token t=0..530 — verified). Script `c_trace_mining.py` (numpy 2.0.2, local CPU, seconds).

**gate_meta.** Keys `'0'..'39'` each hold **only** `{"n_experts": 384, "topk": 6}` — no
temperature / bias / scale fields. (Note: `bench/section108_tp_expert_locality_analysis.md`
talks about **256** experts — that doc predates/describes plain DSv4; this trace's DSv4.1
checkpoint has **384**.)

---

## 1. Per-layer unique-expert union + frequency-rank curve

Per layer, union over the 531×6 = **3186** picks:

| | distinct experts (of 384) |
|---|---:|
| min | **227** (L18) |
| median | **276.5** |
| mean | **273.8** |
| max | **331** (L0) |
| **all-layers union** | **384 / 384** (every expert is hit at least once) |

Coverage points — smallest expert-count *k* (ranked by hit count, desc) whose cumulative
count reaches the share of the 3186 picks (per-layer, then aggregated over 40 layers):

| cov of picks | min k | median k | mean k | max k | mean as frac of union |
|---|---:|---:|---:|---:|---:|
| 50 % | 11 | **32** | 31.98 | 63 | 0.117 |
| 80 % | 59 | **96** | 97.65 | 151 | 0.357 |
| 90 % | 102 | **144** | 144.2 | 202 | 0.527 |
| 95 % | 136 | **183.5** | 182.2 | 242 | 0.666 |
| 99 % | 196 | **245.5** | 242.9 | 300 | 0.887 |

Pooled (all 40 layers): top-10 hit counts `891, 884, 770, 629, 620, 598, 589, 589, 588,
587`; coverage k = **145 / 271 / 322 / 350 / 376** of 384 at 50/80/90/95/99 %.

**Read.** Routing is *diffuse*: half the mass is spread over ~32 distinct experts (≈12 % of
the layer's touched set), and it takes ~184 of 384 at 95 %. Per-expert full curve is in the
JSON (`analysis1_union_and_rank_curve.per_layer[L].top10_counts`).

---

## 2. Hot-set concentration + STABILITY

**Concentration** — share of the 3186 per-layer picks covered by the top-k most-used experts
(mean across 40 layers; uniform baseline = k/384):

| top-k | mean share | uniform |
|---|---:|---:|
| 1 | 5.9 % | 0.3 % |
| 2 | 10.3 % | 0.5 % |
| 4 | 16.9 % | 1.0 % |
| 8 | 25.5 % | 2.1 % |
| 16 | 36.9 % | 4.2 % |
| **24** | **45.0 %** | 6.25 % |
| 48 | 61.4 % | 12.5 % |
| 96 | 79.4 % | 25.0 % |

So the top-24 experts (6.25 % of the vocab of experts) carry **~45 %** of traffic — **≈7×
uniform**. A real hot set exists.

**Stability** — top-24 expert **set** Jaccard between split windows (per layer, aggregated):

| split | Jaccard(mean / median) | range |
|---|---:|---:|
| 2 contiguous windows (266 / 265) | **0.273 / 0.263** | 0.171–0.455 |
| 3 contiguous windows w0·w1 | **0.186 / 0.200** | 0.091–0.333 |
| 3 contiguous windows w1·w2 | 0.217 / 0.200 | 0.091–0.455 |
| 3 contiguous windows w0·w2 | 0.289 / 0.297 | 0.143–0.412 |
| **parity (even vs odd tokens)** | **0.678 / 0.714** | 0.333–0.920 |
| *random 24-subsets of 384* | *0.032* | — |

Fraction of a **later** window's picks landing in the **earlier** window's top-24:

| split | mean | median |
|---|---:|---:|
| 2 windows: w1 ∈ w0-top24 | 0.344 | 0.318 |
| 3 windows: w1 ∈ w0-top24 | 0.264 | 0.244 |
| 3 windows: w2 ∈ w1-top24 | 0.282 | 0.248 |
| 3 windows: w2 ∈ w0-top24 | 0.371 | 0.364 |
| parity: odd ∈ even-top24 | 0.437 | 0.429 |

**Read (headline).** The hot set is **concentrated but NOT globally stable**.
- It survives the *interleaved* (even/odd) split almost intact (Jaccard 0.68, 44 % of the
  other parity's picks still land in the hot-24) → **not a parity/position artifact**.
- But across *long contiguous spans* it **churns hard**: only ~19–27 % of the top-24 set is
  shared between the two halves of the prompt (≈2/3 of the hot-24 membership turns over),
  and a later window's picks hit the earlier hot-24 only ~34 % of the time.
- A **static / model-wide** hot-q4 set would capture ~45 % of traffic and lose ~2/3 of its
  membership between prompt halves → **marginal for a fixed hot/cold split**. A
  **windowed / periodically re-selected** hot set is the defensible version.

---

## 3. Per-rank touched-expert imbalance

**Production geometry (verified, not assumed).** DSv4.1's tensor-parallel split is
**intermediate-WIDTH sharding**, not expert-id EP: `auto_parallel.py:1164-1175` +
`bench/section108_tp_expert_locality_analysis.md` confirm both ranks hold **all 384 experts
at half width** (gate/up sliced on `ndim-2`, down on `-1`); the "each rank holds half the
experts" comment is explicitly flagged **WRONG/stale**. Under this scheme per-rank touched
sets are *identical* on both ranks → **cross-rank routing-skew imbalance ≡ 1.0 exactly.**

The task's `rank0=0..191 / rank1=192..383` split is therefore **NOT the production
partition**; it is computed below as a **counterfactual** for a *hypothetical future EP*
scheme (what §Q4 of section108 says EP would require).

Counterfactual EP 192/192, distinct experts touched per rank per batch:

| R | rank0 toc mean | rank1 toc mean | max/mean ratio p50 / p95 / max | frac ≥2× | max/min ratio p50 / p95 / max | frac ≥2× |
|---|---:|---:|---:|---:|---:|---:|
| **4** | 8.10 | 8.23 | **1.17 / 1.50 / 2.00** | **0.019 %** (1/5280) | 1.4 / 3.0 / 14.0 | 20.1 % |
| 1 | 2.96 | 3.04 | 1.33 / 1.67 / 2.00 | 2.6 % | — (discrete) | larger |

Pick-count imbalance (R=4) `max/min` = 1.4 / 3.0 at p50/p95.

**Read.** Under the counterfactual EP split the two ranks are **balanced on average**
(8.10 vs 8.23 touched) and a *heavy* (≥2×) imbalance on the max/mean metric is essentially
**never** (0.02 % of R=4 batches; 2.6 % at R=1). The p95 max/min of 3× is small-count
discreteness (one rank touching 4 vs 9 experts), not a systemic skew. **Conclusion:** on
this trace, routing skew would **not** justify a traffic-aware expert placement — and under
the *real* width-sharded architecture the question doesn't arise at all (imbalance ≡ 1.0).

---

## 4. Batch-to-batch hot-set overlap (R=4)

Batch = 4 consecutive tokens (a spec-verify group); set = unique experts it touches
(mean **16.33** unique/batch — cross-checks q1d's 16.25). Jaccard between **consecutive**
batches' sets, 132 batches/layer × 40 layers:

| | value |
|---|---:|
| consecutive-batch Jaccard mean | **0.268** |
| median | 0.241 |
| p5 / p95 | 0.086 / 0.538 |
| min / max | 0.0 / 0.923 |
| per-layer median range | 0.114 – 0.333 |
| trend (late-half mean − early-half mean) | **+0.005** (flat) |

**Read.** Consecutive 4-token batches share only ~24 % of their unique-expert sets (vs
~2 % random for two 16-subsets of 384) — routing **churns** batch-to-batch, but the churn
rate is **flat over the prompt** (no acceleration). Consistent with §2: no stable global
hot set; moderate local overlap.

---

## Headline verdicts

1. **Union:** per-layer distinct experts **227 / 276.5 / 331** (min/med/max of 384); overall
   **384/384**. Coverage: ~**32** experts = 50 % of picks, ~**96** = 80 %, ~**184** = 95 %.
2. **Hot set:** **concentrated but churny.** top-24 = **45 %** of picks (~7× uniform), but
   top-24 Jaccard across contiguous halves = **0.27** and across thirds = **0.19** (parity
   0.68). → **NOT flat**, but a *static* hot-q4/cold-EXL3 split is **marginal**; a
   *windowed/dynamic* hot set is the viable form.
3. **Rank imbalance:** real architecture = width-shard → imbalance **≡ 1.0**, no expert
   ownership. Counterfactual EP 192/192: mean load balanced (8.10 vs 8.23), ≥2× on the
   max/mean metric in **0.02 %** of R=4 batches → **no traffic-aware placement warranted.**
4. **Batch-to-batch:** median Jaccard **0.24**, trend **flat** (+0.005).

## Surprises / anomalies

- The task's expert-id rank split is **refuted** by the codebase (production is width-shard).
- `gate_meta` is bare (`n_experts`, `topk` only) — no gate temperature/bias captured.
- section108 doc says **256** experts; this trace's DSv4.1 checkpoint has **384**.
- Every expert (384/384) is touched at least once across the prompt — zero dead experts.
- R=4 batch union 16.33 reproduces q1d's independent 16.25 (good cross-check).

## Artifacts (this dir, NOT committed)

- `c_trace_mining.py` — mining script (numpy, local CPU).
- `c_trace_mining.json` — all numbers (per-layer curves, stability, imbalance, overlap).
- `c_trace_mining.stdout.txt` — run digest.
- `c-trace-mining.md` — this doc.

## Caveats

- **One prompt, one run** (531-token multi-domain prefill). Per-layer windows are ~177–266
  tokens; "stability" is measured within a single context, not across prompts.
- All numbers are **real clamp routing**; only the 192/192 rank split is a modeling
  assumption (flagged; the *production* split is width-shard).
- `records` order is trusted as layer-major and asserted per block; token t=0..530.
