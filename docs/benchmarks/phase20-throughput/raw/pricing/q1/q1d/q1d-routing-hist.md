# Q1d — TRUE per-layer unique-expert routing histogram (DSv4.1 MoE, 40 layers × 384 experts, topk6)

**Question.** Settle the N1 claim that at `R=4, topk6` the 24 (row,slot) pairs land on
exactly **6 distinct experts × 4 rows** (`hist = [4,4,4,4,4,4]`), and report the *real*
per-layer unique-expert / per-expert-`m` distribution the Round-1 native-quant
microbench shapes should use.

**Method.** CPU / numpy ONLY, no mlx model load, no GPU, no engine touch. Script
`bench/p20_q1d_routing_hist.py`, run on **studio1** with the node venv python; staged
under `/tmp/q1d_hist`, nothing written under `~/repos/exo`. Idle-guarded (exit 0 before
and after). Two arms:

- **A. Real captured trace (FOUND — not synthetic).** The phase-10 full 40-layer routing
  trace of the **real EXL3 checkpoint**, `raw/pricing/.../phase10-planb-gate-2026-09-28/
  raw/p30_exl3_trace_clamp.json` (sha256 `b4f1b142…`, 531 tokens × 40 layers = 21 240
  real gate decisions, clamped `silu_clamp` = production routing). `R` consecutive tokens
  per layer are grouped (a spec-verify group *is* consecutive positions in a forward pass)
  and the unique-expert count + per-expert row-count (`m`) histogram are measured.
  **No live capture was done** (no boots / instrumentation budget) — the trace already
  existed. NOTE: the node's `~/p30_exl3_trace.json` is a **pre-clamp** run (its sha differs;
  README §5: the clamp flipped 8.1 % of entries) and is *not* production; the committed
  `…_clamp.json` is the production-correct one used here.
- **B. Synthetic correlated rows × real gate.** For all 40 layers, `R∈{1,4,8}`,
  `corr∈{0.01,0.03,0.1,0.2,0.3,0.5,1.0}`, 16 seeds/cell (640 groups/cell): rows =
  `base N(0,1)[5120] + corr·N(0,1)[R,5120]`, gate = the real
  `sqrt(softplus(x@Wᵀ/temp)) + gate.bias` → `argpartition(-biased, 5)[:, :6]`
  (identical to `bench/p20_q1c_opclass.py::make_rows`/`route_idx`).
- **C. Sanity.** numpy `route_idx` compared against the **real production
  `mlx_lm/models/deepseek_v41/moe.py::Gate`** (imported CPU-side, `mx.set_default_device(mx.cpu)`):
  `set_match` and `sorted_exact` both **True** for layers 0/20/39 → the numpy gate is the
  production gate. It also reproduces q1c exactly (layer 20, corr 0.03, seed 1000 →
  6 unique, `[4,4,4,4,4,4]`).

---

## 1. Real trace — the answer

| R | unique experts (mean / med) | p5 / p95 | min / max | `m` mean | `m` dist (1/2/3/4) |
|---|---|---:|---:|---:|---:|---|
| **1** | **6.00 / 6** | 6 / 6 | 6 / 6 | 1.00 | 1.00 / — / — / — |
| **4** | **16.25 / 16** | **12 / 21** | **7 / 24** | **1.48** | **0.703 / 0.172 / 0.071 / 0.054** |
| **8** | 26.19 / 26 | 17 / 36 | 10 / 46 | 1.83 | 0.630 / 0.182 / 0.078 / 0.038 (+ m5-8) |

**`R=4` unique-expert histogram** (count : #groups of 21 120):

`7:2  8:26  9:84  10:269  11:644  12:1121  13:1740  14:2231  15:2535  16:2675
17:2644  18:2285  19:1778  20:1368  21:933  22:503  23:226  24:56`

**6×4 claim: `[4,4,4,4,4,4]` occurs 0 / 21 120 groups = 0 %.**
`#groups with unique==6` = **0**. The claim is **FALSE on real routing.**

Per-layer (R=4) mean unique: **min 13.70 (L6), median 16.05, max 20.84 (L0), mean 16.25**
— every one of the 40 layers is 13.7–20.8, *no* layer is anywhere near 6.
Per-layer distinct experts over all 531 tokens: min 227, med 277, max 331 of 384
(matches README §2) — routing is diffuse; the small per-`R` unique count is purely the
narrow 4-token window, not a hot-6.

## 2. Synthetic correlated rows × real gate (why the claim appeared)

| corr | R=4 unique mean | `[4,4,4,4,4,4]` groups | `m` mean |
|---|---:|---:|---:|
| 0.01 | 6.16 | **536/640 = 83.8 %** | 3.89 |
| 0.03 | 6.49 | **358/640 = 55.9 %** | 3.70 |
| 0.10 | 7.49 | 95/640 = 14.8 % | 3.21 |
| 0.20 | 8.82 | 10/640 = 1.6 % | 2.72 |
| 0.30 | 10.17 | 1/640 = 0.16 % | 2.36 |
| 0.50 | 12.66 | 0 | 1.90 |
| **1.00** | **16.96** | 0 | 1.42 |
| *(real trace)* | *16.25* | *0* | *1.48* |

The `6×4` shape is a **degenerate near-zero-perturbation artifact**: only for
`corr ≲ 0.1` do the 4 rows collapse onto the same 6 experts, and even at `corr = 0.1`
only 15 % of groups do. Real adjacent-token hidden states are only weakly correlated —
the equivalent synthetic `corr` that reproduces the real 16-unique count is ≈ **1.0**
(this parameterization: per-dim perturbation std ≈ base std), *not* the 0.03 q1c assumed.
q1c's "6.0 unique / `[4,4,4,4,4,4]`" is therefore an artifact of `corr = 0.03`.

**Q4's "14 unique" is also wrong** — its `route_topk` used `sigmoid(x@Wᵀ + b)`, *not* the
real `sqrt(softplus(x@Wᵀ/temp)) + bias`, so it neither confirms nor refutes the real gate.

## 3. Corrected shapes for the Round-1 microbench

| shape | draft `R=1` | **verify `R=4`** | prefill-class |
|---|---:|---:|---:|
| **realistic (mean)** unique experts | **6** | **16** | R=8 → 26 |
| mean rows-per-hit-expert `m` | 1.00 | **1.48** (70 % m=1 / 17 % m=2 / 7 % m=3 / 5 % m=4) | 1.83 |
| working band (p5–p95) | 6 | **12–21** | 17–36 |
| **conservative (upper bound)** | 6 | **24** (all slots distinct, m=1) | 48 |

- **Gating point: 16 unique experts at R=4, m ≈ 1.48** (NOT 6 / `[4,4,4,4,4,4]`).
- **Conservative bound: 24 unique** at R=4 (uniform routing; equals the trace max).
- The old `6`-unique / `[4,4,4,4,4,4]` assumption **understates the verify gather width by
  ≈2.7×**. The realistic range is **12–21** (the "10–18 unique at m=1–3" hypothesis is
  right in spirit; m ≥ 2 covers ~30 % of hit-experts, m=4 the tail ~5 %).
- Cost note: q4 showed the R=4 expert *module* cost is slot-bound (~1.06–1.08 ms for
  unique ∈ {6…24}); this correction changes the **gather/width shape**, not the GEMM ms.

## Artifacts

- `bench/p20_q1d_routing_hist.py` — node script (CPU/numpy; `Exl3Checkpoint` gate load,
  real-trace arm, sanity vs production `Gate`).
- `raw/pricing/q1/q1d/q1d_routing_hist.json` (+ `.stdout.txt`) — all distributions above.
- Real trace used: `docs/benchmarks/phase10-planb-gate-2026-09-28/raw/p30_exl3_trace_clamp.json`
  (sha256 `b4f1b142a5c5ef779e87b8eb3f041f4e822fdc5de8fe858ea9c8fba1bde91a11`).

## Caveats

- The real trace is a **single 531-token multi-domain prefill pass**; per-position picks
  are causally-valid decode routing (README §2), and a spec-verify group *is* consecutive
  positions, so `R=4` grouping is exact — but it is one prompt, one run. The synthetic arm
  spans corr/seed space so the shape is not single-sample.
- `R=8` is not a production shape (verify is γ+1; γ=3 → R=4) — reported for the curve only.
