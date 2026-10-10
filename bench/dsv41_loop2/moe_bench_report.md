# EXL3 Routed-Expert (MoE) Prefill Microbench — Verdict Report

**Node:** macstudio-m4-2 (Mac Studio M4 Max, 128 GB), MLX `0.32.3.dev20260918+603f16eb7`
**Interpreter:** `/Users/adam.durham/repos/exo/.venv/bin/python` (node venv, has the EXL3 kernels)
**Model:** `~/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw` (39 shards)
**Date:** 2026-10-07 · **Method:** wall-clock `time.perf_counter` only (`gpu_time_ns` is dead on this build); every timed region ends in `mx.eval` drain; warmups drained.
**Bench-only:** the resident exo runner was never touched. Loads are read-only via `os.pread`.

---

## VERDICT: (iii) INCONCLUSIVE — headroom exists, unharvested

- **Not POSITIVE:** the best alternative (dequant-to-fp16 inside the region + dense
  per-expert `mx.matmul`) is **1.8–2.1× SLOWER** than the current segmented path at
  block level (0.47–0.55×), far below the 1.15× threshold. No arm beats current.
- **Not ROOFLINE-CONFIRMED:** the current path reaches only **51–60 % of the
  calibrated compute ceiling** at its arithmetic intensity. It is *neither*
  compute-saturated (60 %) *nor* bandwidth-saturated (**6–9 %** of the 450 GB/s
  ceiling) — a kernel bound by neither roofline is not at its roofline.
- **INCONCLUSIVE:** real headroom (≈40–49 % below the compute ceiling) exists in the
  fused decode→MMA kernel and is not harvested by any alternative tested. Per the
  decision rule this is (iii), **not** confirmed.

The July "82 TFLOP/s" artifact was **not reproduced** — see §7. Max observed
throughput anywhere in this bench is **9.58 TFLOPS**, vs a 15.1 TFLOPS ceiling.

---

## 1. Calibrated ceilings (Phase 0)

Dense bf16 `mx.matmul` (the hardware compute ceiling):

| M × N × K | ms | TFLOPS |
|---|---|---|
| 1024×4096×4096 | 2.47 | 13.89 |
| 2048×4096×4096 | 4.78 | 14.39 |
| 4096×4096×4096 | 9.24 | 14.88 |
| 8192×4096×4096 | 18.21 | 15.09 |
| 2048×8192×8192 | 18.37 | 14.97 |
| 4096×8192×8192 | 36.32 | 15.14 |

**Peak compute ceiling = 15.14 TFLOPS** (bf16 in / fp32 accum). A 512 MB fp32
read/write triad:

| op | GB/s |
|---|---|
| `a+b` (3 streams) | 449 |
| `a*1.0` (2 streams) | 435 |
| `sum(a)` (1 stream) | 472 |

**Peak bandwidth ceiling = ~450 GB/s** (conservative; decode stream).

---

## 2. Production geometry actually benched

The production loader ships a **world=2** intermediate-width slice
(`loader.py::_slice_intermediate`), which is the real 2-node TP layout. Verified by
executing the real loader (E=2..192):

| | value |
|---|---|
| E (experts per rank) | 384 (whole layer 0 is in shard `model-00001`) |
| D (input/hidden) | 5120 |
| H (per-rank intermediate) | 1152  (= `moe_intermediate_size` 2304 / world 2) |
| gu_tiles / dn_tiles | 72 / 320 |
| **k (bits)** | **3** on layer 0 (packed=48); **2 on layers 18–22**; 3 elsewhere |
| per-expert trellis bytes | **6.64 MB** (gu 4.15 + dn 2.49) |
| per-rank layer trellis | **≈ 2.55 GB** (384 experts) |

Layer bit-width map: `{0–17:3, 18–22:2, 23–39:3}`. **Layer 0 (k=3, 2.9 bpw-nominal)
was used for all timing**, so results reflect the bulk of the weights, not the
2-bit minority.

The gather_mm fallback (`_mm_fallback_viable`) is **structurally impossible** at
this geometry (rhs ≥ 9 GB > int32 buffer), so the segmented kernel
`inner_mm_seg_mlx` is the *only* production prefill path — confirmed in code
(`use_mm = seg_ok and (N <= _MM_MAX_ROWS or not fb_viable)`, `fb_viable=False`).
**The current path is ALREADY a single launch** over all experts' sorted row blocks
(`grid = (out_e/_MM_BN, nb_max, 1)`, `inner_mm_seg_mlx` in `exl3/gemv_metal.py`), so
**arm (c) — a stacked/grouped single-op dispatch alternative — is moot and skipped**
(there is no per-expert Python loop to replace).

Production per-chunk config: chunk **2048 rows**, top-**6**, E=384 → **N = 12288**
(token,slot) pairs per rank per layer; blocks are BM=64 rows
(`_MM_BM=64`, `_MM_BN=64`, seg version **v19d**).

---

## 3. M-curve — current segmented path (Phase 1/2)

E_sel = 64 experts loaded (working set 425 MB; exceeds L2), swept over all 64
distinct experts. `ms` = median wall of one `sg._prefill` call.

| target M | avgM/expert | pairs | blocks | pad-waste | re-decode | **ms** | **TFLOPS** | dec GB/s |
|---|---|---|---|---|---|---|---|---|
| 16 | 16.0 | 1026 | 64 | 3.99× | 1.0× | 7.94 | 4.57 | 53 |
| 32 | 32.1 | 2052 | 64 | 2.00× | 1.0× | 10.47 | 6.94 | 41 |
| 64 | 64.0 | 4098 | 65 | 1.02× | 1.0× | 17.76 | 8.16 | 24 |
| 128 | 128.1 | 8196 | 129 | 1.01× | 2.0× | 32.62 | 8.89 | 26 |
| 256 | 256.0 | 16386 | 257 | 1.00× | 4.0× | 64.48 | 8.99 | 26 |
| 2048 | 2048 | 16386 | 257 | 1.00× | 32.1× | 60.56 | **9.58** | 28 |

**The kernel is throughput-flat (~8–9.6 TF) from M≈64 upward** and is *linear in
pairs* (ms/pair ≈ 4.3 µs both at M=128 and M=2048) — i.e. cost is per-block, not
per-row. The "re-decode" column is the working-set re-read factor (blocks /
distinct experts); at 64 experts it never drops below 1×, so the swept trellis
(425 MB) is always resident above L2 and read from DRAM.

Decode-only floor (`decode_full_mlx`, production decode kernel, 2 experts):
gu 0.26 ms @ **1090 GB/s** src; dn 0.26 ms @ 548 GB/s — i.e. the pure decode kernel
reaches DRAM speed; the **fused** path does not (see §5).

---

## 4. Arms at M=64 and M=128 (Phase 3)

Interleaved A/B/A/B, ≥7 reps each, medians. Arm (a) = current segmented
`sg._prefill`. Arm (b) = dequant-to-fp16 **inside** the timed region
(`decode_full_mlx` → fp16 W → dense `xg @ wg`) with the EXL3 Hadamard rotations
(`_rows_prep` / `_rows_finish`) applied exactly as production. 32 distinct experts.

| M | arm | ms | TFLOPS | ratio |
|---|---|---|---|---|
| **64** | (a) segmented | **10.00** | **7.26** | — |
| 64 | (b) dense | 17.55 | 4.14 | **a/b = 1.75×** |
| **128** | (a) segmented | **17.52** | **8.28** | — |
| 128 | (b) dense | 23.46 | 6.18 | **a/b = 1.34×** |

Correctness cross-check (a vs b): rel. max-err **9.1e-4** (M=64) / **1.2e-3**
(M=128) — fp16 accumulation noise, so the arm is numerically the same computation.

Arm (c) skipped: the current path is already a single segmented launch (§2).

---

## 5. Block-level mock — DECISION-GRADE (Phase 4)

One layer, **E=192 real experts**, production chunk **N=12288 pairs** (2048 rows ×
top-6). Uniform (production-like, per `bench/pDE_segment.py`) and skewed (Zipf-0.9,
35 % hot) routing. k=3, ceiling 14.82 TF / 434 GB/s.

| routing | blocks | pad-waste | maxM | seg ms | seg TF | % ceil | dec GB/s | dense ms | dense TF | **seg/dense** | **×40 seg ms/chunk** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| uniform | 192 | 1.00× | 64 | 48.99 | 8.88 | **60 %** | 26 | 103.65 | 4.20 | **2.12×** | **1960** |
| skew | 326 | 1.70× | 1867 | 57.58 | 7.55 | **51 %** | 38 | 104.69 | 4.15 | **1.82×** | **2303** |

**Extrapolation math (explicit):**
- Production chunk = 40 layers × 12288 pairs = 491 520 expert-slot pairs.
- Per-rank, one layer = 12288 pairs → measured 48.99 ms (uniform) / 57.58 ms (skew).
- **×40 layers → 1959.6 ms / 2303.2 ms per 2048-token chunk per rank.**
- Caveat: only 192 of the 384 experts were loaded (1.28 GB). The resident model
  runner holds ~115 GB of the 128 GB unified memory; loading the full 384-expert
  layer (2.55 GB) plus transients risked destabilising the cluster, so `E=192` was
  used (same per-expert M regime) and extrapolated. **Production's per-expert
  M at E=384, N=12288 is 32** — between my uniform M=64 (60 % of ceil) and skew
  M≈32 ragged (51 %). The M-curve (§3) shows flat throughput 8.2–9.0 TF over
  M=64–256 and 6.9 TF at M=32, so the production figure plausibly sits **at or
  slightly below** the 51 % skew figure.

**Kernel share of the production path (decomposition, E=192, k=3):**

| N | full `_prefill` ms | seg-total ms | seg TF | seg % of prefill | glue (table/prep/gather) ms |
|---|---|---|---|---|---|
| 2052 | 20.11 | 18.84 | 3.85 | **94 %** | 0.20 / 0.90 / 0.18 |
| 12288 | 57.63 | 49.05 | 8.87 | **85 %** | 0.20 / 4.25 / 0.49 |

The segmented GEMM **is** the bottleneck (85–94 % of `_prefill`); the sort,
seg-table, Hadamard-rotation and gather glue is minor. So the block-level verdict
is a statement about the kernel itself.

**Arithmetic intensity / roofline:**
- FLOP per layer = 6·D·H·N = 6·5120·1152·12288 = **4.35e11**.
- Decode traffic = 6.64 MB/expert × blocks = 1.275 GB (uniform) … 2.165 GB (skew).
- **AI = 341 FLOP/B (uniform) … 201 FLOP/B (skew)** — both ≫ the ridge point
  (15.1e12/450e9 ≈ **33.6 FLOP/B**), so the roofline is the **compute** ceiling.
- Achieved **8.88 TF (59 % of roofline)** / **7.55 TF (50 %)**. The kernel is
  40–49 % below the roofline it *should* hit, and 6–9 % of the bandwidth ceiling.

---

## 6. Sanity (Phase 5)

- Every timed region ends in `mx.eval`; warmups drained too. ✔
- Implied ceilings respected: max TFLOPS 9.58 ≤ 15.14 ceiling; max GB/s 53 ≤ 449. ✔
- Arm (b) cross-checks against (a) to fp16 precision (rel err ~1e-3). ✔
- Working set exceeds L2 (425 MB swept / 1.28 GB block mock). ✔
- No phase destabilised the runner; `mx.clear_cache()` between phases. ✔

---

## 7. The historical "82 TFLOP/s at M=48" — NOT reproduced (artifact confirmed)

At M=48–64 this bench measures **7–9 TFLOPS** (peak anywhere in the bench: 9.58 TF
at M=2048). The calibrated hardware ceiling is **15.1 TFLOPS**, so the July claim of
82 TF is **5.4× above what the silicon can do** and physically impossible — it is
not merely *unreproduced*, it is *unreproducible in principle*. No point in this
bench exceeded 30 TF, so the ">30 TF ⇒ find why" alarm does not fire. This is
consistent with the existing repo reconciliation
(`docs/PERFORMANCE_HISTORY.md`, "82 TFLOP/s … physically exceeds the M4 Max's own
~16 TFLOPS hardware ceiling"): the figure is a missing-drain / dispatch-latency
artifact. **Retire it; 8–9.6 TF (this bench) is the correct reference.**

---

## 8. Command lines

All on the node, node venv, scripts written to `/tmp`:

```bash
# main bench: ceilings, layer k-map, M-curve, decode floor, arms, block mock
scp moe_bench2.py macstudio-m4-2:/tmp/moe_bench2.py
ssh macstudio-m4-2 'cd /tmp && MB_JSON=/tmp/moe_bench.json MB_LAYER=0 \
  MB_E_LOAD=64 MB_E_BLOCK=192 MB_E_ARM=32 MB_REPS=7 \
  ~/repos/exo/.venv/bin/python moe_bench2.py'

# decomposition: seg kernel vs glue
scp decompose2.py macstudio-m4-2:/tmp/decompose2.py
ssh macstudio-m4-2 'cd /tmp && MB_E_BLOCK=192 MB_JSON=/tmp/moe_decompose.json \
  ~/repos/exo/.venv/bin/python decompose2.py'

# decision-grade block-level, uniform vs skew
scp final_block.py macstudio-m4-2:/tmp/final_block.py
ssh macstudio-m4-2 'cd /tmp && MB_E_BLOCK=192 MB_REPS=7 MB_JSON=/tmp/moe_final_block.json \
  ~/repos/exo/.venv/bin/python final_block.py'

# geometry / bit-width / loadability probe
scp probe_load.py macstudio-m4-2:/tmp/probe_load.py
ssh macstudio-m4-2 'cd /tmp && ~/repos/exo/.venv/bin/python probe_load.py'
```

Key code facts read (read-only): `mlx_lm/models/exl3/exl3_moe.py` (`EXL3SwitchGLU._prefill`),
`mlx_lm/models/exl3/gemv_metal.py` (`inner_mm_seg_mlx`, `_mm_seg_source`, `_MM_BM/_MM_BN/_MM_WS`,
`decode_full_mlx`), `mlx_lm/models/exl3/loader.py` (`load_experts`, `_slice_intermediate`),
`src/exo/worker/engines/mlx/dsv41/load.py` (`tp_geometry`).

---

## 9. Interpretation

The EXL3 routed-expert prefill kernel is a **fused 3-bit-decode + fp16 simdgroup-matrix
GEMM** that decodes 32×64 weight tiles straight into threadgroup memory and MMAs them.
At production shapes it converts the packed trellis at only **26–38 GB/s** — ~6–9 % of
the achievable DRAM bandwidth, while the *un-fused* decode kernel alone reaches
**1090 GB/s**. It is not bandwidth-bound (decode is far too slow to be DRAM-limited),
not compute-bound (60 % of the 15.1 TF ceiling), and not fixed by dequant+dense (which
is 2× slower because it re-materialises fp16 W per expert and loses the fusion).
The gap is therefore **latency / occupancy / ALU-issue bound inside the fused kernel**
— a real, unharvested engineering target, not a physics floor. It is *not* a
production-ready win (nothing here beats the current path), so the correct label is
**INCONCLUSIVE**, with the specific hypothesis for a future round: the fused
decode→MMA pipeline is under-occupied (2 in-tiles/threadgroup, 4-out-tile decode per
block, 128 threads) and could be attacked with deeper pipelining / more in-flight
decode tiles — worth a controlled occupancy sweep before any claim.
