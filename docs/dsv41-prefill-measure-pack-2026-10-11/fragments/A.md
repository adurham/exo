# Task A — native vs EXL3 expert GEMM at REAL PREFILL shapes (DECISIVE)

## VERDICT: **YES** — the lever is ALIVE at prefill shapes.

native beats EXL3 on the expert BLOCK at **real** prefill routing by **22.1–23.0 %
WALL** (>=20 % threshold) across all four real-routing cells (layer 20 k=2 and
layer 0 k=3; chunks 2048 and 4096).  GPU-kernel-work ratio 0.57–0.71 (native does
29–43 % less GPU work).  Uniform fill (reference) gives a larger 37 % wall saving;
the REAL per-expert M-profile (hot experts, M_max up to ~1.5k rows) erodes the
advantage from 37 % to ~22 %, but it stays above the 20 % gate.

Margin note: 22 % is a real but not large pass — it clears the gate by ~2 pp.  The
decisive production metric is WALL (see GPU-time caveat below).

## Measurements (node studio2, offline, idle-guarded; median of 5 interleaved rounds)

Dense bf16 roofline, in-session calibrated (4096×8192×8192 `mx.matmul`):
**15.22 TF GPU / 15.14 TF wall** (gpu_ms 36.13).  FLOPs/block = 6·S·topk·H·D
(H=1152, D=5120 per rank).

### Real routing (autopsy trace method: S draws WITH replacement from the 531 layer rows)

| cell | arm | WALL ms | GPU ms | TF(gpu) | %roofline(gpu) | x40 WALL ms |
|---|---|---|---|---|---|---|
| **L20 k=2, S=2048** (hit 237, M̄ 52, Mmax 781) | EXL3 prod | **59.56** | 80.02 | 5.43 | 35.7 % | 2382 |
|  | native q4g64 | **46.37** | 46.09 | 9.44 | 62.0 % | 1855 |
|  | native q3g128 | **46.46** | 46.23 | 9.41 | 61.8 % | 1858 |
| **L20 k=2, S=4096** (hit 237, M̄ 104, Mmax 1573) | EXL3 prod | **104.04** | 140.70 | 6.18 | 40.6 % | 4162 |
|  | native q4g64 | **80.80** | 80.57 | 10.80 | 71.0 % | 3232 |
|  | native q3g128 | **81.46** | 81.21 | 10.71 | 70.4 % | 3258 |
| **L0 k=3, S=2048** (hit 330, M̄ 37, Mmax 402) | EXL3 prod | **65.92** | 71.13 | 6.11 | 40.2 % | 2637 |
|  | native q4g64 | **50.74** | 50.49 | 8.61 | 56.6 % | 2029 |
|  | native q3g128 | **50.73** | 50.44 | 8.62 | 56.7 % | 2029 |
| **L0 k=3, S=4096** (hit 331, M̄ 74, Mmax 752) | EXL3 prod | **109.44** | 148.00 | 5.88 | 38.6 % | 4377 |
|  | native q4g64 | **85.24** | 84.99 | 10.23 | 67.2 % | 3410 |
|  | native q3g128 | **85.88** | 85.64 | 10.16 | 66.8 % | 3435 |

### Uniform / synthetic reference fill (equal rows/expert)

| cell | arm | WALL ms | GPU ms | TF(gpu) | %roofline | x40 WALL ms |
|---|---|---|---|---|---|---|
| L20 S=2048 (M=32 uniform) | EXL3 | 57.47 | 77.14 | 5.64 | 37.0 % | 2299 |
|  | native q4g64 | 36.08 | 35.86 | 12.13 | 79.7 % | 1443 |
| L20 S=4096 (M=64) | EXL3 | 95.09 | 128.39 | 6.77 | 44.5 % | 3803 |
|  | native q4g64 | 71.51 | 71.27 | 12.20 | 80.2 % | 2860 |
| L0 S=2048 | EXL3 | 58.91 | 64.42 | 6.75 | 44.4 % | 2356 |
|  | native q4g64 | 36.11 | 35.87 | 12.12 | 79.7 % | 1444 |
| L0 S=4096 | EXL3 | 95.72 | 129.48 | 6.72 | 44.1 % | 3828 |
|  | native q4g64 | 71.45 | 71.23 | 12.21 | 80.2 % | 2858 |

### Deltas (best native vs EXL3, real routing)

| cell | Δ WALL | Δ GPU-work | native/EXL3 GPU ratio |
|---|---|---|---|
| L20 S=2048 | **22.14 %** | 42.40 % | 0.576 |
| L20 S=4096 | **22.34 %** | 42.74 % | 0.573 |
| L0  S=2048 | **23.04 %** | 29.09 % | 0.709 |
| L0  S=4096 | **22.11 %** | 42.57 % | 0.574 |

Uniform fill Δ WALL: L20 S=2048 **37.2 %**, S=4096 **24.8 %**; L0 S=2048 **38.7 %**, S=4096 **25.4 %**.

## Numerics sanity (A6b method, R=4 natural routing, layer 20, vs EXL3 fused kernel)
- native **q4g64**: cosine 0.9881, rel-L2 1.55e-1
- native **q3g128**: cosine 0.9423, rel-L2 3.47e-1
(Quantization error of the affine format vs the 2.9 bpw trellis reference; not a
harness bug — the reconstruction/transpose/sign path is the same one the q1d
harness validated with its A6a true-fp16 near-exact check.  Perf comparison only;
bit-exactness not required.)

## Metrics caveat — GPU-time counter
`mx.metal.gpu_time_ns()` is trustworthy for the native arms (GPU≈WALL, sigma
0.02–0.11 ms).  For the **EXL3** arm it over-reports: GPU 80.0 ms vs WALL 59.6 ms
at S=2048 (ratio 1.34) with sigma up to 12 ms — the segmented EXL3 path issues
many dispatches whose summed Metal GPU durations exceed wall (overlap / counter
summing).  So **WALL is the decisive, clean metric**; the GPU-work numbers are a
secondary "how much less kernel work" view and the GPU-% deltas are not directly
comparable to the q1d VERIFY GPU ratio (which used the same inflated EXL3 counter
— native/EXL3 GPU ratio here 0.573–0.709 is consistent with the q1d 0.564).

## Geometry / method
- Per-rank: E=384 experts, H=1152 (TP-split of 2304), D=5120, top-6, rank 0, world 2.
  gu N=2H K=D ; dn N=D K=H.
- Arms: (1) EXL3 production kernel (`EXL3_MOE_V2=1 EXL3_MOE_MM=1 EXL3_MOE_FUSED=1`,
  activation silu_clamp) at layer 20 (k=2) and layer 0 (k=3); (2) native q4g64
  (bits=4 group=64); (3) native q3g128 (bits=3 group=128, the 3.25 bpw bpw-fair arm).
  Both native arms use the fused gate+up sorted `gather_qmm` variant.
- Routing glue IDENTICAL across arms: **both** the EXL3 segmented path and the
  native `sorted` path do argsort → `gather_qmm(sorted_indices=True)` → scatter-unsort.
  (The native `unsorted` variant was not used.)
- Timing: one drained eval per round under `MLX_GPU_TIME=1`; arms interleaved with
  alternating order per round; ROUNDS=5, WARMUP=2; median + sigma reported.  Hot
  (prefill is compute-bound); no 1.5 GiB flush.
- Native weights built ONCE from layer-20 experts (streamed requant, ~5 s/arm,
  3.82 GB q4g64 / 2.76 GB q3g128); native timing is shape-driven → reused across
  layers.  EXL3 module loaded per layer.

## Provenance / integrity
- Node: studio2 (macstudio-m4-2), OFFLINE, idle-guarded (`bench/phase20_guard.py`,
  idle window reached 13:36).  Run under `phase20_guard.py watch`; EXIT=0 (no abort).
- Production kernel md5 (site-packages `mlx_lm/models/exl3/gemv_metal.py`):
  BEFORE `649d2d6a0a68c8d6eab2a8a5394b7ab9` (studio2 + studio1);
  AFTER `649d2d6a0a68c8d6eab2a8a5394b7ab9` (studio2 + studio1) — **MATCH, no production contact**.
  (The `~/repos/exo/mlx-lm` source-tree copy is byte-identical md5 too.)
- Trace sha256 `b4f1b142a5c5ef779e87b8eb3f041f4e822fdc5de8fe858ea9c8fba1bde91a11`
  (verified on node).
- Raw: `raw/A/taskA.json`, `raw/A/taskA_stdout.log`.  Harness: `bench/p20_pmp_taskA_prefill.py`
  (reuses `bench/p20_q1d_native_experts.py`).  Elapsed 27.2 s.

## Open item flagged (not fixed — out of Task A scope)
The autopsy/real-trace M-profile at layer 20 is far more skewed than the brief
anticipated (experts_hit 237 not ~330; M_max 781–1573 not ~64).  Layer 0 matches
the brief's "~330 hit / M̄ ~31" better (hit 330, M̄ 32 all-experts).  The brief's
"M_max ~64" does not match either layer's real profile (sampling with replacement
concentrates hot experts); I used the real profile as required.

---

# Task C — Metal hardware counters (xctrace, bounded) — **SUCCEEDED**

`xctrace record --template 'Metal System Trace' --time-limit 30s --attach <benchpid>`
**works headless** on studio2 (no failure).  REC_RC=0; 30.88 s trace saved (347 MB);
`xctrace export` rc=0.  Captured: **1681 command-buffer submissions**, 2377 completed,
2521 app intervals; `metal-gpu-intervals` "Command Buffer 0:Compute" category summed
**19616 ms** of nested compute over a 30.6 s span (≈64 % GPU compute-busy for the load).

Limitation (not a failure): the default template's counter set is `(null)` and
**Shader Timeline: Disabled** → only 1 HW counter (RT Unit Active, ~0 %);
shader-profiler/APS/device-state tables empty.  So command-buffer-level GPU
intervals ARE available, but **not** per-kernel / per-op-class GPU-time attribution
(Task-B style).  Raw: `raw/C/` (tables + summary + `notes.md`).

# Task D — GPU clock / residency (powermetrics) — **NO CLOCK SAG**

Passwordless `sudo -n powermetrics` confirmed.  Sustained load 1240 matmuls /
45.0 s = **15.14 TF** (matches roofline).

| run | GPU clock MHz (min/max/mean) | active residency % (min/max/mean) | power mW |
|---|---|---|---|
| studio2 idle | 766 / 771 / 768.7 | 3.86 / 4.46 / 3.96 | 15 |
| **studio2 LOAD (40 s)** | **1578 / 1578 / 1578.0** | **100.0 / 100.0 / 100.0** | 54786 |
| studio1 idle ref | 703 / 774 / 764.1 | 3.43 / 9.14 / 3.99 | 16 |
| studio1 during load (ref) | 764 / 771 / 768.2 | 3.39 / 3.91 / 3.51 | 12 |

**Confirmed: no clock sag** — under load the GPU pins the max clock 1578 MHz
(σ=0) at 100 % residency for the full window; idle is ~769 MHz / ~4 %.  studio1
stayed fully idle (no cross-node spill).  Raw: `raw/D/`.

# Node cleanliness
- studio2: OFFLINE, no boots, no config changes, no model/generation requests;
  idle restored (guard `idle` exit 0 at 13:49); no lingering bench/powermetrics/
  xctrace processes; system memory freed back to 24 %.
- GPU released: empty marker `/tmp/p20_pmp_gpu_release` written at 13:49 (after
  all node GPU work finished).
- Production kernel md5 BEFORE==AFTER on both nodes (no production contact).
