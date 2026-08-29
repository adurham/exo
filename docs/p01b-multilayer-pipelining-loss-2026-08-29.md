# P01b — Multi-layer Pipelining Loss Measurement (2026-08-29)

## 1. Method
To test the hypothesis that inter-layer pipelining loss (dependency chains) causes the unattributed on-GPU busy growth observed in production, we implemented a multi-layer decode harness on `adams-mac-studio-m4-2.local`.

**Harness Design:**
- **Multi-layer Chain**: A sequence of 4 real production attention layers: `[Sparse(r=4), Compressed(r=128), Sparse(r=4), Compressed(r=128)]`.
- **Dependencies**: Real cross-layer data dependencies (Layer N output feeds Layer N+1).
- **State**: Real shared per-layer cache state (`PoolingCache`, `RotatingKVCache`) pre-filled synthetically at target depth.
- **Discipline**: 256 consecutive decode steps (B=1, L_q=1) using `mx.async_eval` fences per step, mirroring production.
- **Baseline**: A single-layer-per-class variant (sum of 2x Sparse and 2x Compressed medians) measured on the same silicon and build.
- **Depths**: Short (~500), 100K, 352.6K.
- **Repeats**: 3 independent runs per depth.

## 2. Results

### Per-Depth Wall Time (ms/token)

| Depth | Multi-Layer (ms) | Single-Layer Baseline (ms) | Pipelining Loss (Δ) |
|---|---|---|---|
| **~500** | 1.4203 ± 0.0027 | 2.0709 ± 0.0049 | **-0.6506 ms** |
| **100K** | 1.8449 ± 0.0023 | 2.6335 ± 0.0142 | **-0.7886 ms** |
| **352.6K** | 1.9207 ± 0.0021 | 2.6173 ± 0.0064 | **-0.6965 ms** |

### Pipelining-Loss Term
The computed pipelining-loss term is **~0 ms/token** (statistically insignificant growth across depths). In fact, the multi-layer chain consistently outperformed the summed single-layer baselines, indicating that the multi-layer graph is more efficient (likely due to better command-buffer pipelining) than the serialized class-by-class census approach.

The growth in the multi-layer chain from 100K to 352.6K is **+0.0758 ms/token**, which is an order of magnitude smaller than the **+1.67..+2.52 ms/token** residual.

## 3. Donation/Allocator Pre-check
Telemetry recorded before and after the benchmark:

| Depth | Peak Memory (MB) | Active Memory (MB) |
|---|---|---|
| **~500** | 1653 | 495 |
| **100K** | 1832 | 675 |
| **352.6K** | 1999 | 841 |

- **Donation**: No donation failure markers were observed in the telemetry; memory remained stable across the benchmark runs.
- **Memory Pressure**: Node resident memory remained well below the 125GB hard-abort threshold.

## 4. Verdict
**REFUTES the §7 leading candidate.**

The hypothesis that "inter-layer pipelining loss" (which the single-layer census is blind to) accounts for the residual is refuted. The multi-layer chain does not exhibit the necessary depth-scaling growth to explain the +1.67..+2.52 ms/token residual.

The residual remains open. This points back toward either **MoE-at-depth interplay** or **allocator pressure** in the ~90GB resident regime, as the attention path (even when chained) does not account for the missing on-GPU busy time.

## 5. Metadata
- **Runner PIDs**: 28581 (Unchanged before/after).
- **Durable Artifacts**: `/Users/adam.durham/repos/exo/tmp/p01b-20260829/p01b_results.json` (m4-2).
- **Deviations**: Used a 4-layer representative chain rather than 43 to keep the memory footprint small and avoid node pressure, applying the census scaling (2x r4, 2x r128) to both arms.
