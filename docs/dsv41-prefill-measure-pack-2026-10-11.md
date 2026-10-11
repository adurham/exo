# DSv4.1 — P20 prefill measurement pack (2026-10-11)

Author: Phase-20 PM (delegation). Branch `p20/prefill-measure-pack` (worktree
`/private/tmp/p20-measure-pack`, off `origin/main` `e657e0f94`). **Measurements only — no
fit/build work started** (mixed-format layers, wired-limit raise, requant shipping are all
gated on Task A's result and were NOT begun).

**Budget / disturbance: 0 boots, 0 config changes, 0 generation requests.** Offline benches
on **studio2** only, idle-guarded (`bench/phase20_guard.py`), 0 boots. Production
`gemv_metal.py` md5 **`649d2d6a0a68c8d6eab2a8a5394b7ab9` BEFORE == AFTER on BOTH nodes**
(independently re-checked by the PM after the runs) → no production contact. Nodes left
idle, no strays.

Trigger: a Fable review (2026-10-11) found ONE credible ≥5% lever (native-format experts,
previously closed on the q4 memory technicality) plus two open measurement gaps (a ~44%
"rest" bucket in the wall attribution with no per-component share; the unmeasured
prefill form of the native-vs-EXL3 gap).

---

## Task A — THE DECISIVE TEST: native vs EXL3 expert GEMM at REAL PREFILL shapes

### VERDICT: **YES** — native beats EXL3 on the expert BLOCK at real prefill routing by **22.1–23.0 % WALL** (≥20 % gate), across all four real-routing cells.

Margin is real but modest: it clears the 20 % bar by ~2 pp on WALL. On GPU-kernel-work the
delta is larger (29–43 %), so the verdict holds under **either** metric — but WALL is the
decisive, clean number (see the counter caveat). The lever is **ALIVE**.

### Method
- Harness `bench/p20_pmp_taskA_prefill.py` (reuses `bench/p20_q1d_native_experts.py`); node
  studio2, offline, idle-guarded; median of 5 interleaved rounds (warmup 2); one drained
  eval/round under `MLX_GPU_TIME=1`.
- Per-rank production shapes: E=384 experts, TP-split H=1152 (of 2304), D=5120, top-6,
  rank0/world2. Chunks 2048 and 4096. Layers **20 (k=2)** and **0 (k=3)**.
- Routing: the REAL captured 40-layer production trace
  `raw/trace/p30_exl3_trace_clamp.json` (531 tok × 40 layers; sha256 `b4f1b142…`), per-expert
  M-profile built the autopsy way (autopsy `real_chunk_idx`); a uniform/synthetic fill is the reference.
- Arms: (1) EXL3 production kernel (`EXL3_MOE_V2=1 EXL3_MOE_MM=1 EXL3_MOE_FUSED=1`,
  silu_clamp); (2) native **q4g64** (bits=4 g=64); (3) native **q3g128** (bits=3 g=128, the
  3.25 bpw bpw-fair arm). Native arms use the fused gate+up sorted `gather_qmm`. Routing glue
  IDENTICAL across arms (argsort → gather_qmm(sorted) → scatter-unsort).
- Dense bf16 roofline, in-session calibrated (4096×8192×8192 `mx.matmul`): **15.14 TF wall /
  15.22 TF gpu**.

### Results — real routing (ms/layer; %roofline on GPU-work)

| cell | arm | WALL ms | GPU ms | TF(gpu) | %roofline |
|---|---|---:|---:|---:|---:|
| L20 k=2, S=2048 | EXL3 prod | **59.56** | 80.02 | 5.43 | 35.7 % |
| | native q4g64 | **46.37** | 46.09 | 9.44 | 62.0 % |
| | native q3g128 | 46.46 | 46.23 | 9.41 | 61.8 % |
| L20 k=2, S=4096 | EXL3 prod | **104.04** | 140.70 | 6.18 | 40.6 % |
| | native q4g64 | **80.80** | 80.57 | 10.80 | 71.0 % |
| L0 k=3, S=2048 | EXL3 prod | **65.92** | 71.13 | 6.11 | 40.2 % |
| | native q4g64 | **50.74** | 50.49 | 8.61 | 56.6 % |
| L0 k=3, S=4096 | EXL3 prod | **109.44** | 148.00 | 5.88 | 38.6 % |
| | native q4g64 | **85.24** | 84.99 | 10.23 | 67.2 % |

### Deltas (best native vs EXL3, real routing)

| cell | Δ WALL | Δ GPU-work | native/EXL3 GPU ratio |
|---|---:|---:|---:|
| L20 S=2048 | **22.1 %** | 42.4 % | 0.576 |
| L20 S=4096 | **22.3 %** | 42.7 % | 0.573 |
| L0  S=2048 | **23.0 %** | 29.1 % | 0.709 |
| L0  S=4096 | **22.1 %** | 42.6 % | 0.574 |

Uniform-fill reference Δ WALL is larger (24.8–38.7 %): the real per-expert skew (a few very
hot experts) erodes the advantage from ~37 % to ~22 %, but it stays above the gate.

### Numerics sanity (A6b method, R=4 natural, layer 20, vs EXL3 fused kernel)
- native **q4g64**: cosine **0.9881**, rel-L2 0.155
- native **q3g128**: cosine **0.9423**, rel-L2 0.347

q4g64 reproduces the q1d verify-shape cosine (0.9881) exactly — a good cross-harness check.
**q3g128 (the memory-fitting bpw-fair arm) is materially lossier** (0.9423) — a quality
caveat to carry into any fit plan, not a perf-harness bug (the reconstruction/transpose/sign
path is the same one q1d validated near-exact).

### Caveats (do not over-read the win)
1. **GPU-time counter over-reports the EXL3 arm.** `mx.metal.gpu_time_ns()` for EXL3 reads
   80.0 ms at S=2048 vs its own WALL 59.6 ms (ratio 1.34, σ up to 12 ms) — the segmented path
   issues many dispatches whose summed Metal durations exceed wall (overlap/pipelining). The
   native arms are clean (GPU≈WALL). ⇒ **WALL is the decisive metric**; the 29–43 % "less GPU
   work" view is secondary and not directly comparable to q1d's verify GPU ratio (though
   native/EXL3 GPU ratio 0.57–0.71 is consistent with q1d's 0.564).
2. **Standalone single-node module bound** (same shape as q1d's caveat): the live round
   overlaps the expert block with comm/other work, so the realized e2e win is ≤ this. Amdahl:
   at ~33 % expert-block share of the chunk (Task B), a 22 % block win ≈ ~7 % per chunk.
3. **The real M-profile is more skewed than the brief's estimate.** The autopsy's real chunk
   had `max_rows_per_expert` 341 (L0); this run's L0 S=2048 hits 330 experts, Mmax 402 — the
   same order. (The brief's "M_max ~64" was the autopsy's BM=64 *block* histogram max, not
   rows/expert.) The real profile was used as required.
4. **q3g128 ties q4g64 on speed** (both ~22 %) — consistent with q1d's finding that the speed
   lever is the 4-bit-class packed stream, not precision.

---

## Task B — full per-op-class GPU-time attribution over one whole chunk

### Method (re-specified by a design review; NOT a microbench sum)
REAL-WEIGHT **LAYER-SLICE REPLAY** of one decoder layer built by the production builder
`mlx_lm.models.deepseek_v41.exl3_build.build_block` (real per-rank TP2 rank0/world2 weights;
EXL3 experts + production affine6 dense + shared-expert q8g64), driven through the real
`Block.__call__` with a 2048-row chunk and a ~100K prefix cache state. Two measurements:
(a) **unbracketed** whole-layer GPU-busy + WALL (production graph/compile boundaries intact);
(b) **bracketed** per-op-class via `MLX_GPU_TIME=1` + eval-fenced module wrappers (phase-22
methodology) with a running counter. `Σ(bracketed)` vs `(a)` = the **bracketing distortion**.
`collective.all_sum` is **stubbed to identity** (no RDMA on the live link) and bounded
separately. Harness `bench/p20_pmp_taskB_opclass.py`; raw `raw/B/`.

### Share table — one 2048-row chunk, 40-layer weighted fold (~100K ctx, the deliverable)
Weighted 40-layer fold of sampled layers {0,1,2,3,20,21,24} (counts 1/1/2/15/1/15/4, +L14≈L2).
**GPU-busy-sum 7974 ms, chunk WALL 7975 ms, Σ(bracketed) 7208 ms → gap vs WALL −9.6 %.**

| op-class | share (100K) | share (16K, method validation) |
|---|---:|---:|
| **indexer** | **31.1 %** | 13.5 % |
| MoE expert (down) | 19.1 % | 23.4 % |
| sdpa (sparse/compressed) | 13.6 % | 15.9 % |
| dense proj (attn o) | 7.2 % | 8.8 % |
| MoE expert (up) | 7.2 % | 9.9 % |
| **hyper-connections (HC)** | **5.2 %** | 6.2 % |
| dense proj (attn qkv+rope) | 4.3 % | 5.2 % |
| MoE experts total (routing glue: sort/gather/scatter) | 3.3 % | 2.2 % |
| dense proj (shared expert) | 3.2 % | 3.9 % |
| MoE expert (gate) | 3.0 % | 7.4 % |
| norms (RMSNorm) | 1.7 % | 2.1 % |
| dense proj (other) | 0.7 % | 0.9 % |
| MoE routing gate (score+topk) | 0.5 % | 0.5 % |
| CSA/HCA compression | 0.08 % | 0.05 % |
| engram (LazyEngramTable) | ~0 % (SSD/host-bound) | ~0 % |
| **dispatch / GPU-idle** (WALL − Σ(bracketed)) | **~9.6 %** | ~8.3 % |
| **TP all_sum** (unmeasurable via gpu_time — range) | **0.08–1.9 s/chunk ≈ 1–24 %** | (same census) |

The ~44 % "rest" bucket is now **decomposed**: dense projections (attn o 7.2 + qkv 4.3 +
shared 3.2 + other 0.7 = 15.4 %), **hyper-connections 5.2 %**, norms 1.7 %, routing glue
3.3 %+0.5 %, compression 0.08 %, dispatch/GPU-idle ~9.6 %, all_sum bounded 1–24 %.

### LARGEST UNEXPLAINED BUCKET: **hyper-connections (HC), ~5–6 % of layer GPU** — a real
production class (the `hyper_connections.py` module) that is *not in the requested taxonomy*.
(The only other unnamed residuals are MoE routing glue 2–3 % and dense-proj-other ~0.7 %;
Σ(bracketed) closes, so nothing else is unexplained.)

### Closure / caveats
- **Σ(bracketed) vs (a): −8.3 % (16K) / −9.6 % (100K)** — borderline under the ~10 %
  threshold ⇒ per-class shares are reliable to **~10 % resolution**, no better. Direction is
  negative because the *unbracketed* busy-sum can exceed wall (unfenced buffers overlap:
  busy/wall 96.9–118.9 % per layer), while eval-fencing serializes.
- **The indexer diverges sharply from the phase-22 fenced table (3.6 % @16K) — 3.7× higher
  here (13.5 % @16K, 31.1 % @100K).** Context-INDEPENDENT classes reconcile (sdpa 15.9 vs
  15.6; dense 18.8 vs 16.6; MoE 42.9 vs 30.7). The indexer is genuinely O(context) (nb =
  end_pos/ratio) and is the **single largest 100K bucket** in this harness; the divergence
  from phase-22 is flagged, not resolved — it needs reconciliation (likely the M2/HIER path
  vs the phase-22 capture) before 31.1 % is treated as the production indexer share. It is,
  however, **directionally consistent** with the earlier sync-span finding that
  `attn.indexer.score` reaches ~43 % of deep prefill wall at ~340K.
- **`DSV41_INDEXER_HIER` was not explicitly exported** in the harness env (code default
  applies) — noted for the reconciliation.
- **all_sum is NOT measured** (stubbed): the range 0.08–1.9 s/chunk is a bound
  (analytic RDMA-payload floor 84–336 ms; by-difference vs the live unfenced 8380 ms/16K
  ≈ 1907 ms upper, confounded with dispatch + seq-split) — hence 1–24 %, wide.
- **Engram (L1) is host/SSD-bound** (wall 584 ms vs GPU-busy 206 ms) — its cost is ~0 GPU and
  is a WALL term, not a GPU-attribution term.
- The 40-layer fold weights L14 ≈ L2 (not sampled).

---

## Task C — Metal hardware counters (xctrace, bounded ≤30 min)

**SUCCEEDED (no failure).** `xctrace record --template 'Metal System Trace' --time-limit 30s
--attach <pid>` works **headless** on studio2 (REC_RC=0; 30.88 s trace, 347 MB; export rc=0).
Captured 1681 command-buffer submissions / 2377 completed / 2521 app intervals;
`metal-gpu-intervals` "Command Buffer 0:Compute" sums 19616 ms of nested compute over a
30.6 s span (≈64 % GPU compute-busy). **Limitation, not failure:** the default template's
counter set is `(null)` and Shader Timeline is Disabled → only the RT-Unit counter plus
command-buffer-level GPU intervals; **no per-kernel / per-op-class GPU-time attribution**
(so it does not substitute for Task B). Raw: `raw/C/` (tables + `xctrace_summary.json` +
`notes.md`).

## Task D — GPU clock / residency (powermetrics, ~45 s load)

Passwordless `sudo -n powermetrics` confirmed. Sustained load 1240 matmuls / 45.0 s = 15.14 TF.

| run | GPU clock MHz (min/max/mean) | active residency % (min/max/mean) | power mW |
|---|---|---|---|
| studio2 idle | 766 / 771 / 768.7 | 3.86 / 4.46 / 3.96 | 15 |
| **studio2 LOAD (40 s)** | **1578 / 1578 / 1578.0** | **100.0 / 100.0 / 100.0** | 54786 |
| studio1 idle ref | 703 / 774 / 764.1 | 3.43 / 9.14 / 3.99 | 16 |
| studio1 during load (ref) | 764 / 771 / 768.2 | 3.39 / 3.91 / 3.51 | 12 |

**Confirmed: no clock sag** — under load the GPU pins max clock 1578 MHz (σ=0) at 100 %
residency the whole window; idle ≈769 MHz / ~4 %. studio1 (idle reference) showed no spill.
Raw: `raw/D/`.

---

## Open items / next steps (NOT acted on — measurements only)
- **Task A is a measured YES at the ≥20 % gate (thin margin).** The build decision (mixed-format
  layers / transient reclaim / wired-limit raise to make a native arm *fit*) is now armed by
  data, but the q4 arm still **does not fit** (q1d: q4g64 = 155.4 GB = 113 % of 128 GiB) and
  **q3g128 (the fit-class arm) is lossier** (cosine 0.9423). Owner decision.
- **Reconcile the indexer share** (Task B 31.1 % @100K vs phase-22 3.6 % @16K) before treating
  it as the production number — it is the single largest 100K bucket if real.
- **Hyper-connections (~5–6 %)** is an un-taxonomized real class now visible.
- **all_sum** remains unmeasurable via `gpu_time` (JACCL buffers filtered) — only bounded.

## Artifacts
- Doc: `docs/dsv41-prefill-measure-pack-2026-10-11.md` (this file).
- Fragments: `docs/dsv41-prefill-measure-pack-2026-10-11/fragments/{A,B}.md`.
- Raw: `docs/dsv41-prefill-measure-pack-2026-10-11/raw/{A,B,C,D}/` + `raw/trace/`.
- Harnesses: `bench/p20_pmp_taskA_prefill.py`, `bench/p20_pmp_taskB_opclass.py` (+ reused
  `bench/p20_q1d_native_experts.py`, `bench/p20_q1d_routing_hist.py`, `bench/p20_moe_autopsy.py`,
  `bench/phase20_guard.py`).
- Provenance: node studio2 (macstudio-m4-2), mlx `0.32.3.dev20260918+603f16eb7`; production
  kernel md5 unchanged on both nodes; trace sha256 `b4f1b142…`.
