# MECHANISM.md — PHASE 1 verdict: the dense/EXL3 slice is STRUCTURALLY bound → **CLOSE**

Author: Phase-20 PM. 2026-10-09 ~03:25 CDT. Worktree `/private/tmp/phase20-campaign`.
All measurements **OFFLINE bench-only on studio2** (real weights, production kernel, layer-20
dense roster, per-rank TP=2 shapes). **0 relaunches; servers untouched; no API POST.**
Cluster verified still production `fb4f9290b`/`16830e1`, canary 14.86/14.84 before AND after.

Scripts: `bench/p20_dense_fork.py`, `p20_dense_msweep.py`, `p20_dense_splitk.py`,
`p20_dense_decode_arms.py`, `apply_p20_nsplits.py`. Raw JSON under `raw/p20/`.

---

## 0. THE VERDICT

**CLOSE the dense track.** Four independent experiments all say the same thing: the dense/EXL3
slice's 9.3× gap to the 497 GB/s read roof is **not harvestable by any tune-existing mechanism**.
The production fused kernel simultaneously (a) runs at the **decode-ALU cost** — the decode is
~100 % of the fused time and the *decode-free* alternative is **slower**, not faster — and (b) sits
at the **small-M structural floor** — the split-K auto-tune is already optimal and finer splits are
monotonically *worse*. No existing launch parameter, decode-mechanism env knob, or geometry change
clears the ≥ 5 ms/round gate. The best measured headroom from ANY arm is **≤ ~1 ms/round** and inside
noise. A further gain needs a **NEW custom kernel, which is scope-closed (NOT-FUNDED)**.

**Falsifier fired** (ROUND-DENSE-EXL3.md §4): no offline mechanism clears ≥ 5 ms. Honest close.

---

## 1. THE ONE NUMBER — the fork (decode-free isolation)

Layer-20 dense slice, whole-slice single-eval + K=8-batched per-call p95, REPS=9, studio2:

| arm | m4 per-call (p95) | m4 GB/s (trellis-equiv) | m1 per-call (p95) | m1 GB/s | vs prod |
|---|---:|---:|---:|---:|---:|
| **prod** (fused decode+matmul) | **1.048 ms** | **63.7** | **0.759 ms** | **87.9** | baseline |
| **native bf16 `x@W`** (decode NOT in loop) | 1.225 ms | 54.5 | 1.062 ms | 62.8 | **1.19× SLOWER** |
| decode-only (bulk decode_full) | 1.452 ms | 46.0 | — | — | 1.39× |
| pure-read control, 512 MB | — | **477 GB/s** | — | — | read roof |
| pure-read control, slice size (66.7 MB) | — | 264 GB/s | — | — | size-limited roof |

**Read the table carefully — this is the whole fork:**

- **The decode-free (native) arm is 1.19× SLOWER than the fused production kernel**, not faster.
  `decode_full_mlx` → resident bf16 W has `W_bytes/trellis = 5.1×` (339.9 MB vs 66.7 MB): the
  decode-free matmul streams **5.1× the bytes at only 1.19× the wall-clock** → a native stream of
  **279 GB/s** (339.9 MB / 1.216 ms) — i.e. **59 % of the 477 GB/s read roof** and it *still loses*.
  So the small-M `x@W` matmul itself is latency/occupancy-limited at these tiny shapes.
- **The bulk decode cost ≈ the whole fused cost** (decode-only m4 = 1.45 ms ≥ prod 1.05 ms). Decode
  writes 5× W (=5.1× bytes) so as a *bandwidth* number it is not directly comparable, but as a
  *timing floor* it says the decode+write path alone already accounts for the fused kernel's time.
- **Therefore the fork's numeric branch fired `PROCEED` (native 54.5 < 100 GB/s) for the wrong
  reason.** The `<100 GB/s` branch was pre-registered to mean "native is slow *because it has no
  decode to hide latency* → the fused kernel is latency-bound → split-K/geometry can harvest." That
  interpretation is **directly refuted** by experiments 2–4 below: every latency/occupancy/geometry
  lever was tested and none helps, AND the fused kernel is *faster than the decode-free alternative*.
  The correct reading is: **the fused kernel is at the coincident decode-ALU + small-M-structural
  floor, and it is already better than removing the decode could make it.** (This is the same class
  of correction the project already made once: the "82 TFLOPS" claim was retired for being 5.4× the
  silicon ceiling — a plausible-looking mechanism number that the calibrated evidence killed.)

---

## 2. m-sweep + occupancy (Exp 2) — m=4 is NOT specifically degraded

K-batched per-call p95, whole dense slice (one layer), and the kernel's auto-selected `n_splits`:

| m | K/call med | p95 | GB/s (p95) |
|--:|---:|---:|---:|
| 1 | 0.841 | 1.028 | **79.4** |
| 2 | 0.992 | 1.005 | 67.3 |
| 3 | 1.084 | 1.089 | 61.6 |
| **4** | 1.018 | 1.031 | **65.6** |
| 5 | 1.226 | 1.234 | 54.4 |
| 6 | 1.229 | 1.260 | 54.3 |
| 8 | 1.163 | 1.168 | 57.4 |
| 16 | 2.090 | 2.097 | 31.9 |

- **m=4 is flat in the m=2–8 band (54–67 GB/s); the ONLY special point is m=1 (79).** The P2
  "53.4 (m4) vs 69.3 (m1)" gap is reproduced (64–66 vs 79 here) as a **modest m=1 GEMV advantage**,
  not an m=4 defect. There is no m=4-specific occupancy anomaly to fix.
- Auto-split (the kernel's own launch tune): e.g. `attn.wo_b` in_t=512,out_t=320 → n_splits 4→8,
  grid 2560 threadgroups at m=4; the big `attn.wq_b` (out_t=2048) is single-split with grid 2048 —
  already ≥ the ~8k "measured optimum ~M5" comment's per-node share over the 18 projections.

## 3. Split-K sweep (Exp 3) — auto-tune is OPTIMAL; finer splits are monotonically WORSE

Call-time env override of the EXISTING kernel's `n_splits` (in-place launch-param tune; the kernel
source is not recompiled — `n_splits` is a runtime `dims[3]` value). `P20_XSPLIT = in_tiles/n_splits`:

| XSPLIT | m1 p95 | m1 GB/s | m4 p95 | m4 GB/s | Δm4 vs stock |
|--:|---:|---:|---:|---:|---:|
| 0 (stock) | 0.774 | 86.3 | 1.067 | 62.6 | — |
| 4 | 0.797 | 83.7 | 1.098 | 60.8 | +0.031 ms |
| 8 | 0.811 | 82.3 | 1.146 | 58.3 | +0.079 ms |
| 16 | 0.873 | 76.5 | 1.275 | 52.3 | +0.209 ms |
| 32 | 0.969 | 68.9 | 1.595 | 41.8 | +0.529 ms |

- **Monotonically worse** with more splitting, at both shapes. Cosine vs stock = **1.0** (correct
  numerics). The kernel is **already at the launch-parallelism optimum**; the plateau is not
  "too few threadgroups / under-filled GPU." A split-K *win* required the opposite result.
- This is the decisive refutation of the "latency/occupancy-bound → split-K harvests" hypothesis.

## 4. Decode-mechanism env arms (Exp 4) — no decode lever moves the plateau

No prefetch/double-buffer parameter exists in the dense dispatch (v19b measured double-buffering as
noise in the mm path already), so R2-as-written is N/A. The in-scope decode levers are the existing
**env knobs**; each measured in its own process (module-level env read is fresh):

| arm | m1 p95 | m1 GB/s | m4 p95 | m4 GB/s |
|---|---:|---:|---:|---:|
| stock | 0.759 | 87.9 | 1.048 | 63.7 |
| `EXL3_DECODE_SWAR=0` | 0.789 | 84.6 | 1.079 | 61.9 |
| `EXL3_GEMV_LUT=1` | 0.756 | 88.3 | 1.040 | 64.2 |
| `EXL3_FUSE_POST=1` | 0.755 | 88.4 | 1.021 | 65.4 |
| `EXL3_GEMV_SIMD=0` (sanity) | 0.828 | 80.6 | 1.554 | 42.9 |

- No decode mechanism beats stock by more than noise. `FUSE_POST=1` (+1.7 % at m4) reproduces P2's
  "neutral" call (±2 %, inside run noise). `SWAR=0` (the literal ALU path) is **slower** — the SWAR
  path is already the good decode. `SIMD=0` is 1.48× slower at m4, as expected (sanity that the
  harness is sensitive). **The decode is at its structural optimum.**

---

## 5. Gate arithmetic — why this is a CLOSE, not a proceed

Gate G1 = **offline p95 m=4 dense-pass improvement ≥ 5 ms/round**, m=1 regression ≤ 1 %.

- Best m4 arm (`FUSE_POST=1`): 1.021 vs stock 1.048 ms/layer = **−0.027 ms/layer** → **≈ −1.1 ms/round**
  over 40 layers — and it is inside run noise (P2 independently measured FUSE_POST neutral).
- Split-K: **+0.03 … +0.53 ms worse**. Decode arms: ≤ noise. m=1 GEMV advantage (79 vs 66) exists
  but the dense slice's per-round pass **is the m=4 verify** (P2 §4: 1 pass/round at m=4) — an m=1
  number does not move the round.
- **No arm reaches 5 ms/round. Gate FAILS. Falsifier fires.**

## 6. Chosen mechanism (what a fix WOULD require)

There is no tune-existing mechanism. The two coincident walls:
1. **Decode ALU/issue-bound** for this quant: the trellis mul1 decode is ~the whole fused cost, the
   SWAR path is already the best decode, LUT/SWAR/FUSE variants don't help.
2. **Small-M structural latency floor**: at m≤16 the `x@W`/trellis GEMM is latency-limited; the fused
   kernel beats the decode-free alternative, and split-K is already optimal — so MLX's scheduler has
   nothing left to reorder.

A real gain requires a **new custom Metal kernel** (e.g. a fused decode+MMA at a different tile/M
geometry, or a fundamentally different small-M dataflow). That is **scope-closed (NOT-FUNDED)**.
**CLOSE the dense track here**, with the evidence above attached.

### What was NOT closed (honest boundary)
- The P2 route "collectives-at-m=1" is a *separate* slice from the dense GEMV; the m=1 GEMV advantage
  (79 vs 66 GB/s) means a **collective/comm** lever at m=1 could still be live, but that is out of
  this round's dense-GEMM scope and was not measured here.
- Everything in this verdict is an **isolated microbench on layer 20**. The live round-level effect
  of "no lever" is therefore "no shippable change," which is what triggers the falsifier — not a
  claim that the round itself is at 100 % of a theoretical floor.

---

## 7. Budget / cluster state

- **Promotions spent: 0 / 3 (+1 reserve).** All of Phase 0–3 here is bench-only (no relaunch,
  no restart, no API POST).
- Cluster end state: production `fb4f9290b`/`16830e1` live on BOTH nodes, gates unset, canary
  healthy (14.86/14.84), no stray processes. Nothing to restore (nothing was deployed).
