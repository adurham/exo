# MoE SwitchMLP GPUTrace & Attribution Recapture (2026-08-29)

## Overview
This report documents the recapture and honest re-measurement of the `moe.switch_mlp` kernel performance on production silicon (Mac Studio M4 Max), correcting the warm-cache artifacts and capture failures from the initial attempt.

## Capture Viability
The previous failure to start captures was caused by the absence of the `METAL_CAPTURE_ENABLED=1` environment variable, which is required for programmatic `MTLCaptureManager` calls outside of the Xcode debugger.

| Machine | Xcode Installed | `METAL_CAPTURE_ENABLED=1` | Result |
| :--- | :--- | :--- | :--- |
| MacBook (M4 Max) | Yes | Yes | **SUCCESS** |
| Mac Studio (m4-1) | No | Yes | **SUCCESS** |

**Finding:** `mx.metal.start_capture()` is viable on production silicon even without Xcode installed, provided the environment variable is set.

## GPUTrace Extraction
An attempt was made to script the extraction of per-kernel timings from the `.gputrace` bundle.
- **Extractable:** Kernel names (e.g., `mxfp4_gather_qmv_fast_bfloat16_t_gs_32_b_4`) are present in `device-resources` files.
- **Not Extractable:** High-resolution timing data for individual kernels is not stored in a scriptable format (plist/sqlite) within the bundle; it requires the Xcode GPU Debugger UI for visualization.
- **Utility:** The ability to extract kernel names allows correlation between stage-bracketed `gpu_time_ns` windows and the specific Metal kernels being dispatched.

## Per-Stage Attribution (DRAM-Real)
Measurements were performed on **m4-1** using a rotated index pool (64 entries) to defeat the L2 cache and ensure measurements reflect actual DRAM bandwidth. 

**Accounting (corrected by PM 2026-08-29 — the first draft of this table split bytes ~50/50, which produced a physically impossible >100%-of-peak figure for down_proj):**
- Total bytes touched per token: **47.186 MB** (matches `docs/switch-mlp-bandwidth-artifact-retraction-2026-08-22.md` ground truth exactly).
- `fused_gate_up` stage: **31.46 MB** (top_k=6 × 2 fused matrices (gate+up) × 1024×4096 × 0.5 B mxfp4 + fp32 scales @ g=32 = 30.0 MB weights + 1.5 MB scales). 66.7% of traffic.
- `down_proj` stage: **15.73 MB** (top_k=6 × 1 matrix × 1024×4096 × 0.5 B + scales = 15.0 MB weights + 0.75 MB scales). 33.3% of traffic.
- Spec Peak BW: 546 GB/s; measured real streaming on these machines ≈ 424 GB/s.

| Stage | Average Time ($\mu\text{s}$) | Stage Bytes | Implied BW (GB/s) | % of Spec Peak |
| :--- | :--- | :--- | :--- | :--- |
| `fused_gate_up` | $59.08$ | 31.46 MB | $531$ | $97.5\%$ |
| `activation` | $2.93$ | ~0.01 MB | N/A | N/A |
| `down_proj` | $32.67$ | 15.73 MB | $482$ | $88.5\%$ |
| **Total (Sum)** | **$94.67$** | **47.186 MB** | **$497$** | **$91.4\%$** |

Both stages land between the machine's measured real streaming bandwidth (424 GB/s) and the 546 GB/s spec — physically consistent for pure GPU-busy times (gaps excluded), and consistent with the retraction doc's independent-calls wall-clock regime (84 µs → 477 GB/s / 87.4%).

### Reconciliation
- **Stage Sum:** $94.67\mu\text{s}$
- **Chained Wall Time:** $\approx 117\mu\text{s}$ (Retraction Band)
- **Verdict:** The stage sum is within $\approx 20\%$ of the chained regime. The results are DRAM-real and supersede the previous $65\mu\text{s}$ (131% peak) result, which is now confirmed as a warm-cache artifact.

## Headroom Verdict
The measured efficiency ($\approx 76\text{--}91\%$ of spec peak) for the `gather_qmm` kernels is consistent with the corrected microbenchmarks. 
**Verdict:** No meaningful kernel-level headroom at this op.

## Artifacts & Cleanup
- **Durable Paths:**
    - Laptop: `/Users/adam.durham/repos/exo/tmp/p01-20260829/laptop_smoke/`
    - m4-1: `/Users/adam.durham/repos/exo/tmp/p01-20260829/m4_1/`
- **`fuse_weights` Finding:** The patch to `BatchedSwitchGLU.fuse_weights` to handle `biases=None` was found to be unnecessary for the current production DSv4 runner. The patch has been reverted on all nodes to maintain tree cleanliness.
- **Verification:** 
    - m4-1 `mlx-lm` submodule is clean.
    - Runner PID `25491` remains active.
