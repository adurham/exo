# Q2c — Hot-set Jaccard & turnover vs window length (DSv4.1 EXL3 real routing trace)

**Mode.** DESK-ONLY, CPU/numpy. No engine load, no GPU, no cluster touch. Script
`q2c_jaccard_window.py` (numpy 2.0.2, local CPU, seconds). Branch `p20/q2-gamma-jacc`.

**Source (read-only).** `docs/benchmarks/phase10-planb-gate-2026-09-28/raw/p30_exl3_trace_clamp.json`
sha256 `b4f1b142a5c5ef779e87b8eb3f041f4e822fdc5de8fe858ea9c8fba1bde91a11` (verified at run: `sha_match: true`).
Real captured **production clamp** routing (`silu_clamp`); 531 tokens × 40 layers = 21 240
`[layer, [6 expert ids]]` records, **layer-major** (`records[L*531:(L+1)*531]` = layer L,
token t=0..530), topk6, E=384.

---

## Definitions (frozen; kept comparable to Q1E `c-trace-mining.md`)

- **Batch** = one **R = 4 consecutive-token spec-verify group** (γ = 3 → verify shape R = γ+1 = 4;
  a spec-verify group *is* consecutive positions in one forward pass — q1d convention).
  531 tokens → **nb = ⌊531/4⌋ = 132 full batches/layer** (last 3 tokens dropped).
- **Window** = **W consecutive batches** (= W·4 tokens), W ∈ {1,2,4,8,16,32}. Per layer the 132
  batches are **tiled** into `nwin = 132 // W` **non-overlapping** windows; **adjacent pairs**
  = `nwin − 1` (± the boundary of consecutive windows). (Tiling matches Q1E's `array_split`
  "contiguous windows".) Note: the two windows of an adjacent pair **touch** but their *centres*
  are W·4 tokens apart — so the compared spans separate as W grows (see Read).
- **Hot set** = the top-k experts **by pick frequency within the window** (stable descending
  `argsort`, ties → lowest expert id — identical to Q1E `top_set`). Primary **k = 24** (Q1E
  precedent); sensitivity k ∈ {8, 24, 48}. Per-window pick pool = W·4 tokens × 6 = 24·W picks.
- **Jaccard** = |A ∩ B| / |A ∪ B| between the hot sets of **adjacent** windows.
- **Turnover** = **1 − |A ∩ B| / k** = fraction of the *previous* window's hot-k not in the
  current window (both sets are size k; when a window is distinct-capped the denominator is
  max(|A|,|B|)). Also reported: mean fraction replaced.
- **Pooling.** "per-layer pooled" = collect the adjacent-pair statistic over all 40 layers and
  report its distribution (this is the primary curve). "global (union across layers)" = build the
  window hot set from **all 40 layers' picks pooled by token position** (one expert-popularity
  signal for the whole model at that position window).
- **Controls.** (a) **Random baseline** = E[Jaccard] of two independent random k-subsets of E=384,
  analytic **(k²/E) / (2k − k²/E)**, + a 200 000-draw Monte-Carlo check. (b) **Parity control** =
  even-index tokens vs odd-index tokens within a layer (Q1E's interleaved control).

---

## 1. Primary curve — adjacent-window hot-set Jaccard vs W, k = 24

Per-layer pooled (n_pairs = 131 pairs ×40 = 5240 at W=1, down to 3 ×40 = 120 at W=32).

| W (batches / tokens) | n_pairs | Jaccard median | IQR (q25–q75) | min | max | turnover mean |
|---:|---:|---:|---:|---:|---:|---:|
| 1 (4 tok) | 5240 | **0.371** | 0.237 | 0.021 | 0.920 | 0.453 |
| 2 (8 tok) | 2600 | **0.263** | 0.171 | 0.000 | 1.000 | 0.571 |
| 4 (16 tok) | 1280 | **0.297** | 0.171 | 0.000 | 0.778 | 0.559 |
| 8 (32 tok) | 600 | **0.231** | 0.190 | 0.000 | 0.714 | 0.614 |
| 16 (64 tok) | 280 | **0.171** | 0.147 | 0.021 | 0.500 | 0.686 |
| 32 (128 tok) | 120 | **0.171** | 0.115 | 0.021 | 0.371 | 0.711 |

**Read.** Adjacent-window top-24 Jaccard sits at **~0.23–0.37** for short windows and decays to
**~0.17** at W=32 — i.e. **~63–83 % of the hot-24 membership turns over between adjacent windows**,
**6–11× the random baseline of 0.032**. The curve is **not monotone** (a local bump at W=4):
at small W the hot-set estimate is *noisy* (a 4-token window holds only ~16 distinct experts, so
its "top-24" is under-sampled), while at large W the two compared windows drift apart in token
space — two competing effects (estimation noise ↓ at small W, temporal drift ↓ at large W). The
**turnover** column is the cleaner monotone signal: it rises steadily **0.45 → 0.71** as the
re-window interval lengthens.

## 2. k sensitivity (per-layer pooled, adjacent-window median Jaccard)

| W | k=8 | k=24 | k=48 | rand k=8 | rand k=24 | rand k=48 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.231 | 0.371 | 0.655* | 0.0105 | 0.0323 | 0.0667 |
| 2 | 0.231 | 0.263 | 0.500* | | | |
| 4 | 0.333 | 0.297 | 0.333 | | | |
| 8 | 0.231 | 0.231 | 0.247 | | | |
| 16 | 0.143 | 0.171 | 0.200 | | | |
| 32 | 0.067 | 0.171 | 0.193 | | | |

\* **Filler artifact** — at W=1 (24 picks) and W=2 (48 picks) a window has *fewer than k distinct*
experts (mean distinct: W1≈16, W2≈26), so literal "top-48" pads with **zero-count tie-fillers**
(always the lowest-id unseen experts) and *inflates* Jaccard. See §3 for the corrected rows.

## 3. Distinct-capped sensitivity (no zero-count filler)

Hot set = top-**min(k, D)** experts, D = distinct experts in the window. (k=8 never saturates;
this only bites k=24 at W=1–2 and k=48 at W=1–4.)

| W | k=8 | k=24 | k=48 | mean D / window |
|---:|---:|---:|---:|---:|
| 1 | 0.231 | **0.241** | **0.241** | 16 |
| 2 | 0.231 | 0.263 | **0.278** | 26 |
| 4 | 0.333 | 0.297 | **0.307** | 42 |
| 8 | 0.231 | 0.231 | 0.250 | 64 |
| 16 | 0.143 | 0.171 | 0.200 | 98 |
| 32 | 0.067 | 0.171 | 0.193 | 149 |

Capping collapses the k=24/k=48 rows onto one another at W=1–2 (both windows hold only ~16–26
distinct experts) and removes the spurious k=48 = 0.655 point. The corrected short-window
value is **~0.24**, concordant with the k=24 story.

## 4. Turnover (per-layer pooled, 1 − |A∩B|/k)

| W | k=8 | k=24 | k=48 |
|---:|---:|---:|---:|
| 1 | 0.610 | 0.453 | 0.212* |
| 2 | 0.572 | 0.571 | 0.336* |
| 4 | 0.554 | 0.559 | 0.497 |
| 8 | 0.630 | 0.614 | 0.580 |
| 16 | 0.724 | 0.686 | 0.640 |
| 32 | 0.779 | 0.711 | 0.675 |

\* filler-inflated (see §3). Cleaner signal: **40–70 % of the hot set is replaced between
adjacent windows**, rising with W. On the capped definition the k=24 W=1 turnover is 0.621.

## 5. Global — union across all 40 layers

Window hot set built from **all-layer pooled** picks at the token window (one model-wide
popularity signal). n_pairs is small (nwin−1 ∈ {3 … 131}), so this is **noisy**; report as
secondary.

| W | k=24 median | k=24 turnover | k=48 median | k=48 turnover |
|---:|---:|---:|---:|---:|
| 1 | 0.231 | 0.633 | 0.280 | 0.551 |
| 2 | 0.297 | 0.569 | 0.371 | 0.487 |
| 4 | 0.352 | 0.543 | 0.391 | 0.479 |
| 8 | 0.116 | 0.658 | 0.215 | 0.560 |
| 16 | 0.171 | 0.726 | 0.171 | 0.661 |
| 32 | 0.200 | 0.736 | 0.247 | 0.639 |

Same qualitative shape (short-window ~0.3, decaying, turnover rising); the layer pool is *wider*
so model-wide hot sets churn a little harder than per-layer ones.

## 6. Per-token (window = 1 token)

Cheap extra unit: window = **a single token**, hot set = the **6 picked experts** (topk6 ≤ k for
all k≥6, so k-independent). 531 tokens ×40 layers → 21 200 adjacent-token pairs.

| statistic | value |
|---|---:|
| adjacent-token Jaccard median | **0.200** |
| IQR (q25–q75) | 0.242 |
| mean | 0.228 |
| mean turnover (|A\B|/|A|) | 0.660 |
| mean unique experts/token | 6.00 |

Two adjacent tokens share only **20 %** of their 6-expert sets — routing is essentially
*uncorrelated at 1-token granularity*; the 4-token batch (W=1) Jaccard of 0.24–0.37 is the first
scale at which meaningful overlap appears.

## 7. Sliding-window robustness (step = 1 batch)

Same statistic on **overlapping** windows (step 1). W=1 reproduces the tiled W=1 median exactly
(0.371, k=24) — a consistency check. As W grows, consecutive sliding windows overlap by (W−1)/W
and converge to ~0.8–1.0: this measures *within-span smoothness*, not the re-selection question,
and is reported only as a robustness companion.

| W | k=24 sliding median |
|---:|---:|
| 1 | 0.371 |
| 2 | 0.500 |
| 4 | 0.655 |
| 8 | 0.778 |
| 16 | 0.846 |
| 32 | 0.920 |

---

## Controls

- **Random baseline** (two independent random k-subsets of 384):
  analytic **(k²/E)/(2k − k²/E)** → **k=8: 0.0105, k=24: 0.0323, k=48: 0.0667**
  (200 000-draw MC: 0.0112 / 0.0330 / 0.0672). The k=24 value **reproduces Q1E's 0.032 exactly**.
- **Parity control** (even vs odd tokens, top-k, per layer, n=40):

  | k | Jaccard median | IQR | mean | turnover |
  |---:|---:|---:|---:|---:|
  | 8 | 0.778 | 0.178 | 0.707 | 0.188 |
  | **24** | **0.714** | 0.114 | **0.678** | 0.198 |
  | 48 | 0.655 | 0.114 | 0.652 | 0.214 |

  k=24 parity median **0.714 / mean 0.678** **reproduces Q1E §2 (`parity even/odd 0.678 / 0.714`)
  bit-for-bit** — an independent confirmation that the mining is faithful. Parity stays **~0.7**
  while every contiguous-window Jaccard is **~0.2–0.4** → the hot set churns with *position/context*,
  **not** with a 1-token parity artifact.

---

## Linkage to the PARKED dynamic-hot-set path (do NOT reopen)

This curve is the **supply-side input** for the parked *dynamic-hot-set* path (top-k experts
re-selected per window for a hot-quant / expert-tiering scheme). It shows a real, concentrated hot
set exists (Q1E: top-24 ≈ 45 % of picks, ~7× uniform) but is **locally churny**: adjacent-window
top-24 Jaccard **0.17–0.37** (turnover **45–71 %**) vs a 0.032 random floor, so a *static* model-wide
hot-24 would lose ~2/3 of its membership between windows and must instead be **re-selected every
W batches** — and this curve quantifies the W→stability trade-off a re-selection interval would buy
(short W ≈ 0.24–0.37, long W ≈ 0.17). **The path itself stays PARKED**: its reopen condition is
"paired with a tiering/mixed-bit mechanism," and that mechanism is **absent** here. This doc
supplies the number; it does **not** reopen the path, propose a hot set, or claim a win.

---

## Caveats / NOT verified

- **One prompt, one run** (531-token multi-domain prefill, single context). No cross-prompt
  generalisation; all "stability" is within one context.
- **Tiling convention.** Non-overlapping tiles drop remainder tokens and, at large W, leave only
  3 pairs/layer (W=32) — the tail of the curve is low-n. Sliding variant (§7) is a companion, not a
  replacement.
- **Non-monotonicity** in the small-W region is partly estimation noise (windows shorter than k
  distinct experts), mitigated by the capped variant (§3) but not eliminated.
- **Filler artifact** flagged for literal top-k at W=1–2; primary table is literal top-k (as asked,
  for Q1E comparability), corrected rows in §3.
- Batch unit = **R=4 = γ+1 at γ=3** only; other γ not traced.
- No engine/cluster involvement, so **no end-to-end routing change is verified** — this is trace
  arithmetic on a pre-existing artifact only.

**Artifacts (this dir):** `q2c_jaccard_window.py` (script), `q2c-jaccard-window.json` (all numbers),
`q2c_jaccard_window.stdout.txt` (run digest), `q2c-jaccard-window.md` (this doc).
