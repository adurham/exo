# Q1c — per-op-class GPU-time split of the DSv4.1 MoE EXPERT block (OFFLINE, studio2)

**Question.** At production decode (draft `R=1`, verify `R=4`) and prefill (`S=2048`)
shapes, is the expert block dominated by the expert GEMM **KERNEL** (ALU/issue) or
by **non-kernel** overhead (gather / scatter / routing)?

**Method.** Offline single-node bench, no engine touch. Loaded layer-20
`EXL3SwitchGLU` via `load_experts(ckpt, 20, n_experts=384, rank=0, world=2,
activation="silu_clamp")` (per-rank H=1152, D=5120, E=384, k_trellis=2). Real
per-op GPU ms via `MLX_GPU_TIME=1`: `reset_gpu_time()` → build one sub-stage →
`mx.eval` → `mx.synchronize()` → read `mx.metal.gpu_time_ns()`. Median over
15 reps (decode) / 11 reps (prefill), 3 warmup, 3 correlated-draft seeds
(`make_rows` base + 0.03·N(0,1) per row → real gate → top-6 indices).
Router = the real `deepseek_v41/moe.py` `Gate` (sqrt(softplus(x·Wᵀ/temp)) →
bias-only argpartition top-6 → weights from unbiased scores → /(Σ+1e−20) → ×1.5).
**Two arms** (env is read at `exl3_moe` import, so each is its own process):
`fused` (`EXL3_MOE_FUSED=1`, production) and `unfused` (`=0`).
node `Adams-Mac-Studio-M4-2`, mlx `0.32.3.dev20260918+603f16eb7`, layer 20, 40 MoE layers.

---

## Decode split — per sub-stage GPU ms (medians)

| stage | fused R1 (draft) | fused **R4 (verify)** | unfused R1 (`_decode`) | unfused R4 (`_prefill`) |
|---|---:|---:|---:|---:|
| prep/gather + Hadamard (gate+up) | 0.0091 | 0.0089 | 0.0058 | 0.0358 |
| **gate_up KERNEL** | **0.2847** | **0.5405** | **0.2341** (`mapped_gemv`) | 0.9380 (`seg_mm`) |
| gu finish / act / dn prep / dn finish / sort / segtable / scatter | — | — | 0.0362 (5 ops) | 0.0859 (7 ops) |
| **down KERNEL** | **0.2241** | **0.3238** | **0.1155** (`mapped_gemv`) | 0.1967 (`seg_mm`) |
| **whole `module(x,idx)`** | **0.5350** | **0.9398** | **0.3888** | **0.9212** |
| sum(stages) | 0.5179 | 0.8732 | 0.3816 | 1.2563 |
| **UNATTRIBUTED** (whole−Σ) | **+0.0171** | **+0.0666** | **+0.0072** | **−0.3351** |
| router (outside module) | 0.0412 | 0.0481 | 0.0404 | 0.0477 |

Fused arm: the A2 gate_up kernel and B2 down kernel each hide the activation,
per-row Hadamard rotation and svh inside the launch → only 3 dispatches, so the
"gather/finish/act" decomposition is **not separable**; the unfused arm is run
precisely to expose those. The only pre-kernel op the fused arm exposes is
`_rows_prep` (per-row Hadamard + `gu_suh` gather) at **~0.009 ms (≈1% of the call)**.

Unfused R1 (`_decode`) is the cleanest full decomposition: of 0.3888 ms,
gate_up+down kernels = **0.3496 ms (90%)**, and *all* non-kernel ops
(gather+Hadamard prep, finish, activation, ×2 projections) = **0.0392 ms (10%)**.
(The unfused R4 path is `_prefill`, not a decode path — with `EXL3_MOE_FUSED=0`
the `R<=8` fused-decode branch is disabled — so its `seg_mm` sums overshoot the
whole; see caveats.)

## Prefill split — per sub-stage GPU ms (S=2048, identical in both arms)

| stage | GPU ms | share |
|---|---:|---:|
| sort/argsort gather | 0.0899 | 0.2% |
| gu prep/gather + Hadamard | 8.0926 | 15.3% |
| segtable | 0.0295 | 0.1% |
| **gate_up KERNEL (seg_mm)** | **39.0510** | **73.8%** |
| gu finish Hadamard | 1.2458 | 2.4% |
| act (clamped SwiGLU) | 0.4930 | 0.9% |
| dn prep Hadamard | 0.4974 | 0.9% |
| **down KERNEL (seg_mm)** | **12.7245** | **24.1%** |
| dn finish Hadamard | 2.1398 | 4.0% |
| scatter | 2.6511 | 5.0% |
| **whole `module`** | **52.8954** | — |
| Σ stages | 67.0146 | — |
| **UNATTRIBUTED** | **−14.1192** | — |

Router at S=2048: **0.8198 ms**.

---

## ×40-layer round totals (whole expert module, ms)

| shape | path | per-layer | ×40 layers |
|---|---|---:|---:|
| R1 draft (`topk6`) | `_decode_fused2` | 0.5350 | **21.40** |
| R4 verify (`topk6`) | `_decode_fused2` | 0.9398 | **37.59** |
| S=512 prefill | `_prefill` | 13.75 | 550.1 |
| S=2048 prefill | `_prefill` | 52.90 | 2115.8 |
| router R1 ×40 | — | 0.0412 | 1.65 |
| router R4 ×40 | — | 0.0481 | 1.92 |

Spec round ≈ γ·draft + 1·verify = 3×(R1) + 1×(R4) expert-module ms ≈
3·21.40 + 37.59 ≈ **101.8 ms** across all 40 layers (draft is 1 token/row; verify is γ+1=4 rows).

## Verify-shape (R=4) attribution

| bucket | ms | share of whole |
|---|---:|---:|
| gate_up KERNEL | 0.5405 | 57.5% |
| down KERNEL | 0.3238 | 34.5% |
| **KERNEL total** | **0.8643** | **92.0%** |
| prep/gather+Hadamard | 0.0089 | 0.9% |
| gather/scatter/act (hidden in fused kernels; from unfused arm: ≈0.32 ms of `_prefill`) | — | — |
| UNATTRIBUTED | 0.0666 | 7.1% |
| router (outside module) | 0.0481 | — |

---

## Verdict

**Expert KERNEL ms (×40) at the production verify shape = 34.57 ms → `>= 25 ms`**
(draft R=1 shape = 20.35 ms `< 25 ms`). The kernel **≳** non-kernel overhead — in
fact the two GEMM kernels are **~92%** of the whole expert-module call at verify
(90% in the fully-split unfused R1 arm), while gather/prep/finish/activation are
~1–10% and the *router itself (0.048 ms) is comparable to the entire non-kernel
share of the expert module*. Expert decode time is **KERNEL/ALU/issue-bound, not
gather-scatter-routing-bound.** → `verdict.verdict = "KERNEL-DOMINATED"`.

## Verify-shape per-expert m-distribution (best-effort)

At `R=4, topk6` (24 slots) with correlated drafts, the 24 rows land on exactly
**6 distinct experts, 4 rows each** (`hist = [4,4,4,4,4,4]`) — consecutive draft
rows route to the same hot experts, so the gather touches the minimum unique-expert
set. Unique-expert count: R1 6.0, R4 6.0, S2048 7.33 mean.

## Caveats (honest)

* **Prefill UNATTRIBUTED is negative** (−14.1 ms at S=2048, −3.8 ms at S=512):
  isolated per-stage GPU spans *sum higher* than the whole-call span, because in
  one eval MLX pipelines adjacent dispatches inside a command buffer and the
  cache-hot whole run hides per-stage cold-cache cost. The decode shapes (the ones
  the question is about) close to ≈0 (+0.017/+0.067/+0.007), validating the bracket.
* The fused production kernel hides act/rotation/svh inside the two GEMM launches,
  so the fused arm attributes only 3 ops; the **unfused arm** supplies the
  non-kernel attribution. The router is a *separate* module (not inside
  `EXL3SwitchGLU`) and is reported separately.
* `EXL3_MOE_CLAMP`/`EXL3_MOE_A2_TILES` were left at production defaults (unset);
  `EXL3_MOE_V2=1`, `EXL3_MOE_MM=1`, `EXL3_MOE_FUSED=1` = shipped production build.

_Artifacts: script `bench/p20_q1c_opclass.py`; raw `.../q1/q1c/q1c_opclass.json`; stdout `.../q1c_opclass.stdout.txt`._
