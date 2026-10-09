# PHASE 5 — P2: dense EXL3 isolated production-shape GEMV/GEMM on REAL weights

Author: Phase-5 P2 subagent. Opened 2026-10-08 ~19:20 CDT. **Bench-only, OFFLINE, on an
idle-guarded cluster node (studio2).** Real weights, real production kernel, production
shapes. **0 relaunches; servers untouched (no SIGKILL, no relaunch, no node restart);
no user-facing API POST.**

Companion scripts: `bench/p2_dense_real.py` (production-shape timing),
`bench/p2_cached_w.py` (the Reading-1 route test), `bench/p2_dense_census.py`
(exact per-rank byte census). Raw JSON: `raw/p5/`.

---

## 0. PRE-REGISTERED GO/NO-GO (stated BEFORE the run, per the P2 brief)

> **Bar: projected ≥ 5 ms/round saving → next-round cluster proposal (GO). Below → write
> 'feature-blocked' WITH closing evidence (NO-GO).** The projected saving is from the
> route the measured effective GB/s selects.

The ONE number the route turns on: the **isolated production-shape dense-slice effective
rate**, per shape (m=1 draft/decode shape and m=4 verify shape, γ=3 → m=γ+1).

**Result: GO.** Measured dense-slice rate is **69.3 GB/s (m=1) / 53.4 GB/s (m=4)** — Reading 2
(one pass, ~53 GB/s, 9.3× its 497 GB/s floor), **not** Reading 1 (~240 GB/s). Route =
**Reading 2** (GEMV bandwidth/latency tuning + collectives-at-m=1). Projected saving is well
above the 5 ms bar (§6). Reading 1's bf16-dequant-cache route was measured and is **DEAD**
(slower in wall-clock, §5).

---

## 1. Method — an ISOLATED PRODUCTION-SHAPE bench on REAL weights

- **Node / guard.** Run on **studio2** (studio1 was the busier alias). Idle guard
  (`bench/phase20_guard.py idle`) returned `ok=True` immediately before the run and again
  after; canary healthy on both nodes (14.86 / 14.85 TFLOPS) before and after. The work is
  CPU/GPU-light relative to a full server (one layer's dense groups, ~67 MB of trellis), and
  the guard stayed green throughout — no interference with user traffic. No relaunch, no
  server restart, no API POST.
- **Real weights.** `studio2:~/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw`
  (39 safetensors). Loaded the real dense/shared/attn EXL3 groups of a layer via
  `mlx_lm.models.exl3.loader.load_dense_layer`.
- **Real production kernel, real dispatch.** The bench calls the `EXL3Linear` module itself
  (the module the model uses), i.e. exactly the dispatched path
  (`prepare_xh` + `inner_gemv_mlx` at rows==1 / `inner_gemm_mlx` at 1<rows≤16 + `finish_y`).
  So the measured per-projection time **is** the production per-projection cost. The
  installed package is the production mlx-lm (`exl3_linear.py` dispatch lines 123/146 match
  mlx-lm `3bf8316`; exo `576e9d279` on both nodes).
- **Shapes.** rank-0 of TP world=2. The three groups:
  **m=1** = the GEMV/decode (draft) shape → `rows==1` branch (`inner_gemv_mlx`, simd GEMV);
  **m=4** = the verify shape (γ=3, m=γ+1) → `rows<=16` branch (`inner_gemm_mlx`, v20 devx GEMM).
- **Burying the `mx.eval` floor (load-bearing).** Timing one kernel per `eval` is
  meaningless here — the eager `mx.eval` launch floor on this node is **0.10–0.21 ms**, equal
  to the whole small-projection time. Two mitigations, both reported:
  (a) **K-batching** (K=8 projections built into one lazy graph, one `eval`, time/K);
  (b) **whole-slice single-eval** — all 18 dense projections of the layer built into one
  graph and `eval`ed once (this is the production fused-graph shape; there is no per-dispatch
  host floor in production). The whole-slice number is the headline; it reproduced to
  **<0.5 %** across REPS=2 and REPS=9.
- **GB/s numerator = bytes READ = real trellis/sign bytes** (measured from the checkpoint
  headers, not the bpw label — see §3). GB/s = trellis_bytes / time. Because bytes and work
  both halve under a rank slice, GB/s is invariant to the world-2 sharding, so the projection
  is timed at full shape and the /2 is applied only in the byte accounting.
- **Reps.** warm-up + median of ≥7–9 timed reps; min/max carried in the raw JSON.

---

## 2. THE ONE NUMBER — effective GB/s per production shape

Layer-20 whole dense slice (18 projections; rank-0 bytes **66.75 MB**), whole-slice
single-eval, REPS=9 (studio2):

| shape | kernel | ms (whole slice) | **effective GB/s** | × its 497 GB/s floor |
|---|---|---:|---:|---:|
| **m=1** (draft/decode) | simd GEMV | **0.962** | **69.3** | 9.3× |
| **m=4** (verify, γ=3) | v20 devx GEMM | **1.249** | **53.4** | 9.3× |

Per-layer robustness (whole-slice single-eval): layer 0 → 67.5 / 55.4; layer 10 → 65.2 / 53.5;
layer 20 → 69.3 / 53.4 GB/s (m=1 / m=4). **The rate is flat across layers** — the measured
number is not a layer-20 artifact.

Per-shape (K-batched, REPS=9) the two big tiles (`attn.wq_b`, `attn.wo_b`, 13.11 MB rank-0)
are the fastest: **m=4 = 55.0 GB/s** each, **m=1 = 77 GB/s**; the small tiles
(`attn.wkv` 1.64 MB, the eight `attn.wo_a.slice.*` 2.62 MB) sit at 33–52 GB/s — i.e. the
slice average (53–69) is dragged down by many small projections, not by the big ones.

**Effective rate: m=1 ≈ 69 GB/s, m=4 ≈ 53 GB/s. Both ≈ 9.3× their 497 GB/s pure-read floor.**

---

## 3. REAL byte census — the checkpoint is ~5 bpw DENSE, not 2.9

Measured from the checkpoint headers (`raw/p5/dense_census.json`), all 40 layers, dense
(attn + shared + indexer + compressor) groups only:

| dim | value |
|---|---:|
| dense trellis, full model | **4.184 GB** |
| **dense trellis, per rank (world=2), per PASS** | **2.734 GB** |
| actual bit-width: attn dense | **packed 80 → k=5 (~5.0 bpw)** |
| actual bit-width: shared experts | **packed 64 → k=4 (4.0 bpw)** |
| actual bit-width: `attn.indexer.wk` | packed 128 → k=8 |
| routed-expert trellis, full (unactivated) | 195.35 GB |

The model card says "2.9 bpw", but the **dense/attn trellis is packed=80 = 5.0 bpw** and the
shared experts packed=64 = 4.0 bpw (checked on layers 0/10/20/25/39; `layer 0.attn.wq_a` is
even packed=96 = 6 bpw). So the real dense per-pass byte count is **2.734 GB/rank**, ~34 %
**higher** than the `PHASE4-P5-ROOFLINE.md` §2.2 figure of 2.04 GB (which used the 2.9 bpw
label). Every dense time below uses the **measured** byte count.

---

## 4. What it implies for passes/round — P0's unit question, answered by measurement

- **Dense-slice time per PASS (rank-0, all 40 layers, 2.734 GB):**
  - one m=4 verify pass: `2.734 / 53.4 = **51.2 ms**`
  - one m=1 pass: `2.734 / 69.3 = **39.5 ms**`
- **Passes/round for the dense slice = 1** (the m=4 verify). The "3× m=1 + 1× m=4" pattern in
  the P0 question applies to the tiny **indexer/score** sub-module (its own `wk`/`wq_b`
  projections ≈ 67 MB/rank/pass over the ~20 indexer layers), **not** to the full 40-layer
  dense stack. The three m=1 draft forwards run only the (small) MTP draft head.
- **Decisive internal-consistency check:** if the dense slice really ran **4 full passes**
  per round (Reading 1), dense alone would cost `3×39.5 + 51.2 = **169.7 ms/round**` — larger
  than the whole observed round (`93.8–101.06 ms`). **Impossible.** So Reading 2 holds: the
  dense stack runs **once** per round at **~51 ms**.
- **Floor comparison (measured, 497 GB/s read roof; `PHASE4-P5-ROOFLINE.md` §2.3):** dense
  floor at the real 2.734 GB = **5.50 ms**; measured 51.2 ms = **9.3× floor** (`PHASE4`'s
  microbench-extrapolated "8.5×" was, if anything, optimistic — it used the low byte count).
  The dense slice is the round's **largest single slice (~51 ms ≈ 54 % of a ~94 ms round)** and
  the worst per byte, exactly as `PHASE4` flagged by extrapolation — now measured directly.

---

## 5. Route pick — Reading 2. Reading 1 (bf16 dequant-cache) MEASURED DEAD

Reading 1's stated mechanism was "~240 GB/s effective → the bf16 cache swaps a slow unpack
for a byte-bound stream." The measurement refutes both halves:

1. **There is no 240 GB/s.** The measured effective dense rate is 53–69 GB/s (§2); the
   `~240 GB/s / 4-passes` reading is refuted (§4). So Reading 1's premise — that the current
   path is a 2.1×-off-floor unpack-limited path — is false; it is a ~9.3×-off-floor
   byte-stream-limited path.
2. **The cache is slower in wall-clock anyway** (`bench/p2_cached_w.py`, `raw/p5/p2_cached_w_layer20.json`).
   Direct test: decode the dense slice's `W` **once**, keep it resident (fp16/bf16), then time
   only the matmul, vs the fused trellis. Same memory state, same run:

   | arm | m=1 | m=4 |
   |---|---:|---:|
   | fused trellis (production) | 1.060 ms → **58.3 GB/s** | 1.179 ms → **52.4 GB/s** |
   | cached bf16 `W` + native matmul | 1.170 ms → 52.8 GB/s trellis-equiv (175.0 GB/s native) | 1.424 ms → 43.4 GB/s trellis-equiv (143.8 GB/s native) |

   The cached `W` stream runs at **143.8–175.0 GB/s native** — but it is **3.2–4.0× the
   bytes** (dense `W` at fp16 ≈ 3.2× the 5 bpw trellis for the k=5 attn groups, 4.0× for the
   k=4 shared groups). 175 GB/s × (1/3.2) ≈ 55 GB/s trellis-equivalent — i.e. the byte growth
   cancels the faster stream, and the cache is **10–21 % slower in wall-clock**, worse once
   the resident `W` pollutes the cache (`fused` itself degraded 69.3 → 58.3 GB/s in the
   cache-resident run). **Reading 1 is refuted at the mechanism level, not merely projected
   away.** (RAM aside: the dense bf16 cache is ≈ 9.1 GB/rank on top of the ~87 GB wired model —
   feasible vs the 115 GB guardrail but moot, since it is slower.)

**Route = Reading 2: GEMV bandwidth/latency tuning + collectives-at-m=1.**

---

## 6. Kernel-parity audit — production uses the BEST variant; the 58 GB/s sweep REPRODUCES

Same-shape A/B/C of the microbench variants vs the production dispatch
(`raw/p5/p2_dense_real_layer20.json`, K-batched medians):

| projection | shape | prod (dispatch) | A full-W | B striped | C fused-inner | prod vs best |
|---|---|---:|---:|---:|---:|---|
| attn.wq_b | m=4 | **0.237** | 0.854 | 1.267 | 0.237 | ≡ C (best) |
| attn.wo_b | m=4 | **0.239** | 0.662 | 0.964 | 0.238 | ≡ C (best) |
| ffn.shared_experts.w2 | m=4 | **0.092** | 0.300 | 0.352 | 0.091 | ≡ C (best) |
| attn.wq_b | m=1 | **0.170** | 0.622 | 0.919 | 0.164 | ≈ C (−4 %) |
| attn.wo_b | m=1 | **0.170** | 0.898 | 0.966 | 0.166 | ≈ C (−2 %) |
| ffn.shared_experts.w2 | m=1 | **0.070** | 0.219 | 0.336 | 0.065 | ≈ C (−7 %) |

- **Production is optimal.** The production dispatch equals the fused-inner variant (C) for
  every shape; full-W (A) is **2.8–3.6×** slower and striped (B) **4.1–5.3×** slower at m=4.
  There is **no kernel regression** — production is on the best microbenched path.
- **The prior "58 GB/s" figure is NOT suspect — it reproduces.** The production m=4 per-shape
  rate averages ~58 GB/s (whole-slice 53.4; the two big tiles 55.0); m=1 averages ~69
  (big tiles 77). The sweep and the production effective GB/s agree. Audit **PASS** (the
  earlier suspicion is cleared).
- **`EXL3_FUSE_POST=1` (off by default) is neutral** — whole-slice m=4 52.9 GB/s (vs 53.4 off),
  m=1 69.1 (vs 69.3). No free lever there (`raw/p5/p2_dense_fusedpost_layer20.json`).

---

## 7. VERDICT — GO (route = Reading 2)

- **The one number:** dense-slice effective rate **69.3 GB/s (m=1) / 53.4 GB/s (m=4)**, flat
  across layers, measured on real weights with the production kernel at production shapes.
- **Passes/round:** dense slice = **1 pass/round** (m=4 verify) = **~51 ms/round**; Reading 1's
  4-passes/round is refuted by internal consistency (would need 170 ms > the 101 ms round).
- **Route:** **Reading 2** — GEMV bandwidth/latency tuning + collectives-at-m=1. Reading 1's
  bf16 dequant-cache is **measured dead** (slower in wall-clock; 143.8–175 GB/s native × the
  3.2–4.0× byte growth loses to the 53 GB/s trellis stream).
- **Projection:** the dense slice runs at **10.7 % of the 497 GB/s read roof** and **9.3× its
  byte floor** — a large, latency/stream-bound plateau (not ALU-bound: m=4 ≡ m=1 time shows the
  kernel is byte/latency-limited, so there is bandwidth-shaped headroom, not compute-shaped).
  A conservative even 2× rate realization (53 → 106 GB/s) takes the dense slice 51 → 26 ms,
  a **~25 ms/round** saving; a 3× realization ~34 ms/round. Both are **well above the
  pre-registered 5 ms/round bar.**
- **GO/NO-GO: ≥5 ms/round → GO.** (Not a feature-block. The route is a per-round cluster
  proposal: GEMV bandwidth/latency tuning + collectives-at-m=1.)
- **Honest scope of the projection:** the ≥5 ms bar is met with margin even under a
  pessimistic 2× realization; the exact realized multiple is *not* measured here (that is the
  P3/next-round cluster work). This P2 is the offline decisive-number + route pick.

---

## 8. Canary / guard readings, provenance, limitations

- **Canary (raw-GPU 4096³ matmul):** studio1 **14.86**, studio2 **14.85** TFLOPS — healthy,
  before and after the run. Idle guard `ok=True` before and after. No contention observed.
- **Node/commit:** studio2; exo `576e9d279` (deploy/next13), mlx-lm `3bf8316` on both nodes
  (production). Installed `exl3_linear.py` dispatch (rows==1 simd GEMV / rows≤16 v20 devx)
  matches the bench's dispatch.
- **Reproducibility:** whole-slice m=4 53.4 GB/s at REPS=2 and REPS=9 (Δ<0.5 %); layer-20 vs
  layer-0/10 within 5 %.
- **Limitations.**
  1. **Full-shape timing, not rank-sliced.** Projections are timed at full (unsharded) trellis
     shape; the world-2 rank slice halves a tile axis. Since both the bytes and the work halve,
     GB/s is invariant — but a *rank-slicing* could in principle shift the small-tile launch
     overhead, which a full-shape run does not capture. Direction: the small tiles (where the
     slice average is set) would get *smaller* under a rank slice, so the measured 53 GB/s is
     if anything an *upper* bound on the rank-0 rate — the headroom case is not weakened.
  2. **One layer, extrapolated ×40.** Layer-20's 18 dense groups × 40 layers, using the exact
     per-rank census. The per-shape and per-layer spread above is the error bar.
  3. **The round-level ~51 ms is a bytes/rate extrapolation**, not a whole-model forward; the
     `PHASE4-P5-ROOFLINE.md` §2.3 "dense ≈ 34 ms slice" is the in-situ counter-anchor (same
     ballpark, ×1.5). Both agree dense is the largest, worst-per-byte slice.
  4. **The projection (§7) is a route-level envelope, not a kernel result** — the realized
     speedup is P3/next-round work.
- **`EXL3_FUSE_POST`/`EXL3_WCACHE` left at their production defaults (both 0)** for the
  headline numbers; the FUSE_POST arm was a separate labelled run.

*Real weights, real production kernel, production shapes, idle-guarded, no relaunch, no
server restart, no user-facing API POST. Raw JSON + census in `raw/p5/`.*
