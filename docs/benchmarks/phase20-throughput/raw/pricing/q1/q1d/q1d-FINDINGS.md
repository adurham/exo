# Q1d — Round 1 / GATE 1: native-quant EXPERT microbench (OFFLINE, studio2)

**Question.** Can a NATIVE MLX quant (`mx.quantize` + `mx.gather_qmm`) drop-in replace the
EXL3 2.9 bpw trellis fused expert kernel and save **≥30 % WALL** on the ×40-layer expert
stack — correctness clean, mem-fit OK?

**Answer: GATE 1 = PASS** — anchor arm **q4g64 (bits=4, group=64, fused gate+up)** saves
**32.6 %** WALL at the headline condition (native @ 24-unique FLUSHED vs same-session EXL3
@ 24-unique FLUSHED).  Margin over the 30 % threshold is **thin (+2.6 pp)** — see caveats.

Script: `bench/p20_q1d_native_experts.py`.  Raw: `q1d_native_experts.json`,
`q1d_native_experts.stdout.txt` (this dir).  Node `Adams-Mac-Studio-M4-2`, mlx
`0.32.3.dev20260918+603f16eb7`, layer 20, per-rank H=1152, D=5120, E=384, k_trellis=2.

---

## Method

- **EXL3 baseline** — `load_experts(ckpt, 20, n_experts=384, rank=0, world=2,
  activation="silu_clamp")`; timed at the MODULE boundary `mod(x[1,R,D], idx[1,R,6])`.
- **TRUE fp16** — per expert `reconstruct_public_mlx(load_dense_layer("…experts.{e}.w{1,2,3}"))`
  → `[in,out]`; intermediate sliced to rank-0 (cols 0:1152 w1/w3, rows 0:1152 w2);
  transposed to native `[out,in]`: `Wgu=[E,2H,D]=[gate|up]`, `Wdn=[E,D,H]`.
- **Native forward** — follows the `mlx_lm` `switch_layers` `BatchedSwitchGLU` reference
  shape convention: `sorted` (`_gather_sort` + `gather_qmm(sorted_indices=True)` +
  scatter-unsort; geometry `x[N,1,in]`/`rhs[N]`) and `unsorted` (`sorted_indices=False`;
  geometry `x[R,1,1,in]`/`rhs[R,kk]`).  Both **fuse gate+up into one gather_qmm**;
  compiled `silu_clamp` activation.  An unfused (2-dispatch) gate+up variant is also run.
  The faster LEGITIMATE variant was chosen (`sorted`, 0.779 vs 0.918 ms flushed).
- **Mem-fit** — the node is production-live (~98 GiB wired, ~22 GiB free), so the first
  attempt (full 27 GiB fp16 + copy) GPU-timed-out.  Rebuilt **memory-lean**: reconstruct
  one expert at a time, quantize it (per-expert == whole-tensor; groups never span
  experts) and **in-place slice-assign** into pre-allocated quantized buffers (~7 GiB peak).
- **Routing** — REAL DSv4.1 gate (sqrt(softplus(x·Wᵀ/temp)); bias-reordered top-6) on
  correlated draft rows (base N(0,1) + 0.03·N per row).  Unique-expert sweep forces
  exactly {6,12,18,24} distinct experts.
- **A2 cache-busting** — every timed rep re-draws a fresh hidden state (fresh routing);
  warm (back-to-back) and **flushed** (1.5 GiB unrelated scratch read between reps,
  OUTSIDE the timed region).  **The GATE number is the FLUSHED one.**
- **Timing** — median of 14 reps (≥15 with warmup 3), 3 seeds; per-call WALL
  (`perf_counter` around eval+synchronize) and per-op GPU ms (`reset_gpu_time` bracket).

---

## Arm table — R4 verify (top-6), u24 flushed (the GATE condition)

| arm | wall ms/layer | GPU ms/layer | ×40 wall ms | % saved vs EXL3 | GB/s | cosine (A6b) | leg? |
|---|---:|---:|---:|---:|---:|---:|:--:|
| **EXL3 baseline @u24 (flush)** | **1.2243** | 0.9063 | **48.97** | — | — | ref | — |
| **native q4g64** (anchor) | **0.8250** | 0.5116 | **33.00** | **32.6 %** | 289.6 | 0.9881 | ✅ |
| native q4g32 | 0.8265 | 0.5590 | 33.06 | 32.5 % | 321.1 | 0.9905 | ✅ |
| native q4gu_q6dn | 0.8928 | 0.5751 | 35.71 | 27.1 % | 307.2 | 0.9919 | ✅ |
| native q5g64 | 0.9098 | 0.6116 | 36.39 | 25.7 % | 320.9 | 0.9972 | ✅ |
| native q6g64 | 0.9824 | 0.7152 | 39.30 | 19.8 % | 351.2 | 0.9993 | ✅ |
| EXL3 baseline @natural (flush) | 1.3959 | 0.8867 | 55.84 | — | — | ref | — |

- **Only q4g64 (32.6 %) and q4g32 (32.5 %) clear the 30 % gate**; q4gu_q6dn / q5 / q6 miss.
- R4 **natural** (real gate, ~6 unique) native walls: q4g64 0.7803, q4g32 0.8240,
  q4gu_q6dn 0.9296, q5g64 0.9361, q6g64 1.0113 ms/layer.
- Unfused-gu q4g64 (R4 natural flush) = 0.8046 ms → **fused gate+up is faster** (chosen).
- Arm footprint: q4g64 3.82 GB, q4g32 4.25 GB, q4gu_q6dn 4.39 GB, q5g64 4.67 GB, q6g64 5.52 GB.

## Unique-expert sweep (q4g64, flushed wall ms/layer)

| unique experts | 6 | 12 | 18 | **24** |
|---|---:|---:|---:|---:|
| wall ms | 0.5860 | 0.6604 | 0.7434 | **0.8250** |
| GPU ms | 0.3257 | 0.3394 | 0.4256 | 0.5116 |
| GB/s | 101.9 | 180.9 | 241.0 | 289.6 |

Monotonic in unique-expert count; the 24-unique point (the conservative headline) is the
slowest — used for the gate.

## Correctness

- **A6a (reconstruction, must be near-exact)** — reconstructed-fp16 TRUE-W path vs EXL3
  fused kernel on a 24-expert subset: `cosine_min = 1.000000`, `rel_l2_max = 9.5e-4`,
  `max_abs_err = 5.86e-3` → **near-exact, PASS** (validates reconstruct/transpose/sign/
  activation order).  A transposed/missing-sign projection would fail loudly here.
- **A6b (each arm vs EXL3)** — cosine / relative-L2 (float64): q4g64 0.9881 / 0.155;
  q4g32 0.9905 / 0.138; q4gu_q6dn 0.9919 / 0.128; q5g64 0.9972 / 0.074; q6g64 0.9993 /
  0.037.  All arms legitimate (requant of an already-2.9 bpw-lossy tensor; fidelity to
  EXL3, not to the original model).  No fast-but-wrong arm.
- **Prefill S=2048 (no-regress CHECK, not a gate)** — native q4g64 wall 85.2 ms (GPU 35.6)
  vs EXL3 99.3 ms (GPU 56.9) → **no regress** (native faster).

## GATE 1 arithmetic

| quantity | value |
|---|---:|
| same-session EXL3 @u24 flush wall ×40 | 48.972 ms |
| native q4g64 @u24 flush wall ×40 | 33.000 ms |
| **saved** | **15.972 ms (32.61 %)** |
| implied round = 86.03 − 15.972 | 70.058 ms |
| net t/s @3 % acceptance loss | 40.86 (=0.97·34.299·86.03/70.058) |
| net t/s @1 % acceptance loss | 41.70 (=0.99·34.299·86.03/70.058) |
| kernel-only GPU ratio (native/EXL3) | 0.564 (native 44 % faster on GPU) |

## Read + caveats

**GATE 1 = PASS (anchor q4g64, 32.6 % wall saved; correctness clean; mem-fit OK).**  The
native drop-in IS faster than the fused EXL3 trellis kernel at the production verify shape.

Caveats (bound the *upside*, not the verdict):
1. **Thin margin (+2.6 pp).**  The PASS depends on the same-session EXL3 WALL baseline
   (1.2243 ms/layer).  Its GPU time (0.9063) matches N1's recorded 0.9398, so the baseline
   is credible — but WALL is 1.35× GPU for EXL3 and 1.61× for native, i.e. both arms carry
   ~0.31 ms/layer of Python dispatch that the format change does not remove.  The wall win
   (32.6 %) is therefore smaller than the kernel win (44 %).  A same-session EXL3 WALL near
   its N1 GPU value would flip the verdict to ~12 % → FAIL.  Report the wall baseline as the
   load-bearing number.
2. **q4 only.**  q4g32 (32.5 %) ≈ q4g64 (32.6 %) — group 32 buys nothing (more scale bytes).
   q5/q6/mixed are correct but miss 30 %.  The speed lever is the 4-bit packed stream, not
   precision headroom.
3. **Standalone single-node ⇒ UPPER bound.**  This is the expert module's standalone cost;
   the live round overlaps it with comm/other work (GATE-0 caveats 2–4).  The realized win is ≤ this.
4. **Reconstruction check is a subset (24 experts).**  It validates orientation/signs/
   transpose/activation; it does not re-audit all 384 experts' trellises (they are read by
   the production EXL3 module unchanged).
5. `sorted_indices=True` and `False` are numerically identical here (cos 0.9881 both);
   `sorted` measured faster in wall and was chosen.

_Safety: idle-guarded before/after (both ok, no active TextGeneration); no engine touch,
no POST, no writes under `~/repos/exo`; `MTL_DISABLE_TIMEOUT=1` set to match the production
runner; `/tmp/q1d_bench` cleaned after.  Artifacts: script `bench/p20_q1d_native_experts.py`;
raw `…/q1/q1d/{q1d_native_experts.json,q1d_native_experts.stdout.txt}`._
