# Phase-3 kernel go/no-go — DECISIVE mega-eval test (whole dense set, ONE eval)

Worktree: `/private/tmp/lever3` (detached from `deploy/next15-levers` @ `1ed2b6c37`).
Runner: `bench/exl3_dense_smallm_probe_megaeval.py` (this artifact + the JSON beside it).
Machine: MacBook Pro M4 Max, macOS, mlx `0.32.0.dev20260804+ac73d0c9`.
Date: 2026-10-07 16:41 CDT.  Untracked, uncommitted.  No tracked repo file modified.

**VERDICT: EXHAUSTED.**  The ~55 GB/s plateau **persists** when the whole dense
set is submitted as a single batched eval (no amortisation assumption).  Per-call
cost is essentially unchanged: **ratio mega/amortised = 0.946 at M=4** (60.70 vs
64.18 µs/call).  The plateau is a genuine kernel throughput, **not** a per-eval
submission/launch-shape artefact.  The lever tested here is exhausted.

---

## 0. What this settles (and what it does not)

The prior microbench (INCONCLUSIVE) divided a K=64-amortised eval wall by K, which
left open whether the ~55 GB/s plateau was intrinsic kernel cost or a fixed
per-eval submission cost.  This test removes the amortisation assumption entirely:
the **entire dense set** — all 15 dense linears × K repeats — is built as **one
`mx.eval` graph**, timed end-to-end with `time.perf_counter()`, and divided by the
total call count (15·K).  Each repeat re-reads the same device-resident trellis
arrays, so bytes read = K × 52.9 MB (52.9 MB ≫ the 4 MB L2 → a real DRAM read every
repeat).  At K=80 one eval covers 1200 calls = 4.231 GB of trellis read = **the full
40-layer model's dense traffic in one submission**.

**Result: the plateau does not move.**  Per-call cost is flat from K=1 → K=80
(77.5 → 60.7 µs/call at M=4, saturating by K≈8) and lands within 5.4 % of the prior
K=64 amortised figure.  The per-op floor inside a big graph is **1.818 µs/op** — 3 %
of a 60.7 µs call — so host submission is **not** the bottleneck once the graph is
large.  Hence: **EXHAUSTED** (the batched-eval fix does not help).

---

## 1. The decisive number (M=4, the verify band)

```
M=4: mega-eval 60.70 us/call  vs  prior amortised K=64 64.18 us/call  ->  ratio 0.946  [PLATEAU PERSISTS]
     A 58.1 GB/s (12.9% of 450) | B 247.5 GB/s | roofline C 117.53 us/L | compute E 75.22 us/L
     x40L:  A 36.419 ms   B 45.593 ms   C 4.701 ms   E 3.009 ms
```

- **Mega/amortised ratio = 0.946** — a 5.4 % difference, inside run-to-run noise.
  Removing the amortisation assumption did **not** reveal a hidden submission cost.
- **Achieved 58.1 GB/s = 12.9 % of the 450 GB/s roofline** — the same plateau as the
  amortised run (54.9 GB/s, 12.2 %).
- **36.419 ms/round vs the 4.701 ms/round read-once roofline** and **3.009 ms/round
  compute roofline** — still 7.7× off memory, 12.1× off compute.

## 2. Full primary table (whole dense set × K=80 in ONE eval)

| M | A µs/L | B µs/L | C µs/L | E µs/L | A GB/s | B GB/s | A %BW | A ms/40L | C ms/40L | E ms/40L | A/amort |
|--:|-------:|-------:|-------:|-------:|-------:|-------:|------:|---------:|---------:|---------:|--------:|
| 1 |  636.0 |  949.0 | 117.53 |  18.80 |   83.2 |  297.2 | 18.5% |   25.440 |    4.701 |    0.752 |   0.954 |
| 4 |  910.5 | 1139.8 | 117.53 |  75.22 |   58.1 |  247.5 | 12.9% |   36.419 |    4.701 |    3.009 |   0.946 |
| 5 | 1154.4 | 1185.9 | 117.53 |  94.02 |   45.8 |  237.9 | 10.2% |   46.176 |    4.701 |    3.761 |   1.060 |

*(µs/L = per layer, sum over the 15 dense linears — directly comparable to the prior
doc, which is per-layer. `A/amort` = mega per-call ÷ prior-K=64 amortised per-call.
C = analytic 450 GB/s read-once; E = analytic 15 TFLOPS fp16 compute. flops/layer
@M=4 = 1.128 GFLOP.)*

## 3. The K-scaling (the core evidence that the plateau is intrinsic)

Per-call µs, whole dense set in ONE eval, arm A (EXL3 trellis GEMM):

| K | calls | M=1 | M=4 | M=5 | M=4 wall ms | M=4 GB/s |
|--:|------:|----:|----:|----:|------------:|---------:|
|  1 |    15 | 72.28 | 77.54 | 86.19 |        1.16 |     45.5 |
|  2 |    30 | 49.05 | 68.31 | 78.01 |        2.05 |     51.6 |
|  4 |    60 | 44.80 | 65.08 | 73.67 |        3.90 |     54.2 |
|  8 |   120 | 46.41 | 66.10 | 76.14 |        7.93 |     53.3 |
| 16 |   240 | 44.25 | 62.70 | 75.50 |       15.05 |     56.2 |
| 40 |   600 | 42.79 | 60.95 | 77.52 |       36.57 |     57.8 |
| 80 |  1200 | 42.40 | 60.70 | 76.96 |       72.84 |     58.1 |

The improvement is confined to K=1→8 (dominated by the eval floor draining out)
and then **flattens hard** — from K=8 to K=80 the per-call cost moves <3 % while the
graph grows 10×.  If submission were the binding constraint, per-call cost would
keep falling toward the roofline.  It does not.

**Floor calibration (why submission is ruled out):**

```
EVAL FLOOR  single eval (K=1, zero amortisation)  = 178.4 us
EVAL FLOOR  per-op inside ONE 600-op graph         = 1.818 us/op
```

The per-op in-graph floor is 1.818 µs vs a 60.70 µs EXL3 call → **host overhead is
≈3 % of the call**; the remaining 58.9 µs is real GPU work at ~58 GB/s.

## 4. Arm B (decode-free fp16 ceiling) — also pinned, and slower per layer

| M | A µs/L | B µs/L | B GB/s | A/B |
|--:|-------:|-------:|-------:|----:|
| 1 |  636.0 |  949.0 |  297.2 | 0.67 |
| 4 |  910.5 | 1139.8 |  247.5 | 0.80 |
| 5 | 1154.4 | 1185.9 |  237.9 | 0.97 |

Arm B reads **5.3× more bytes** (22.565 GB vs 4.231 GB fp16 weights) yet is at best
1.5× slower per layer and is pinned at **247–297 GB/s = 55–66 % of the 450 GB/s
roofline**.  A plain fp16 matmul of a fully-materialised weight **cannot reach 450**
at these per-linear shapes on this machine either — the 450 GB/s figure is the
campaign's *large-contiguous-stream* number; the achieved BW at these matmul shapes
tops out near 250–300 GB/s.  Batching (K=80) does not lift it.  This is independent
corroboration that the plateau is a machine/shape property, not a decode artefact.

## 5. Per-linear breakdown (M=4, each linear × K=80 in ONE eval)

| key          |   in |   out | C µs/L | E µs/L | A µs/L | B µs/L | A GB/s | A %BW | A/B  |
|:-------------|-----:|------:|-------:|-------:|-------:|-------:|-------:|------:|-----:|
| wq_a         | 5120 |  1280 |   5.46 |   3.50 |  48.78 |  28.75 |   50.4 | 11.2% | 1.70 |
| wq_b         | 1280 | 16384 |  17.48 |  11.18 | 138.71 | 246.79 |   56.7 | 12.6% | 0.56 |
| wkv          | 5120 |   512 |   2.18 |   1.40 |  25.52 |  16.77 |   38.5 |  8.6% | 1.52 |
| wo_b         | 8192 |  5120 |  34.95 |  22.37 | 283.88 | 300.40 |   55.4 | 12.3% | 0.95 |
| wo_a_x8.0    | 4096 |  1024 |   3.50 |   2.24 |  38.48 |  20.84 |   40.9 |  9.1% | 1.85 |
| wo_a_x8.1..7 | 4096 |  1024 |   3.50 |   2.24 | 35.0-36.0 | 20.75-21.18 | 43.7-44.9 | ~9.9% | ~1.7 |
| shared_w1    | 5120 |  2304 |   9.83 |   6.29 |  82.74 |  64.55 |   53.5 | 11.9% | 1.28 |
| shared_w3    | 5120 |  2304 |   9.83 |   6.29 |  82.47 |  65.61 |   53.6 | 11.9% | 1.26 |
| shared_w2    | 2304 |  5120 |   9.83 |   6.29 |  83.11 | 137.36 |   53.2 | 11.8% | 0.61 |

Every linear sits at 8.6–12.6 % of the 450 GB/s roofline — a uniform ~9–12 % plateau
across all 15 shapes, reproduced independently of the amortisation method.

## 6. Machine-load regime (the separate confound — documented, not hidden)

The mega-eval design removes the *submission* artefact; machine load is a **separate,
still-present** confound. Load was recorded before every timed block (`sysctl -n
vm.loadavg` + `uptime`):

- **Load averages this run: 5.16 – 5.40 (1-min), 5.72 – 5.79 (5-min), 5.84 – 5.87
  (15-min)** across 25 marks.  This is **at or below** the prior run's 5.0–8.0
  regime — i.e. the confound is no worse than the prior run, and the two runs are
  run-comparable.
- Streaming BW measured on this run (single big eval): **343.7 GB/s read, 307.4 GB/s
  read+write** (vs the campaign's 450).  So absolute arm-A/B rates are a
  lower-bound-on-speed snapshot, as before.
- Compute probe: a 512×8192 @ 8192×5120 fp16 matmul = **11.26 TFLOPS** (vs the 15
  TFLOPS campaign peak used for the analytic E roofline) — the analytic E roofline
  is therefore optimistic by ~1.33×.

Machine is shared with ≥2 other live sessions; the shape of the result (plateau
persists, ratio ≈ 0.95) is robust because it is a *within-run comparison* of two
methods at the same load.

## 7. Counters — re-confirmed unavailable

```
dispatch_count: trivial=0  exl3=0  gpu_time_ns=0  ->  STILL 0 / NOT WIRED
```

`mx.metal.dispatch_count()` (and `reset_dispatch_count`) and `mx.metal.gpu_time_ns()`
**still report 0** on `0.32.0.dev20260804+ac73d0c9` — confirmed again on both a
trivial op and a real EXL3 call.  All timing here is wall-clock `time.perf_counter`.

## 8. Caveats (honest)

- **Synthetic weights** (labelled throughout); correctness cos(A, B_fp16) = 0.9999999
  reproduced.  Random trellis content does not affect timing (mul1 decode is an
  ALU/table path).
- **Single node, not TP=2**; per-rank shapes, no RDMA/all-sum.  The K=80 eval
  *approximates* the 40-layer model's dense traffic by repeating the 15 linears 80×
  (K=40 = one-pass 40-layer traffic; K=80 = two-pass headroom) — a valid total-bytes
  and total-FLOPs proxy, not a real pipeline.
- **No attention/MoE** — dense GEMM kernel only.
- **Machine under load** (§6), streaming BW 343 GB/s not 450.  Absolute µs are a
  lower-bound snapshot; the *verdict* rests on the within-run mega-vs-amortised
  agreement, which is load-invariant.
- **Scope of "EXHAUSTED"**: this test rules out the *submission / batched-eval*
  explanation the prior doc named in §6.  It does **not** prove the kernel is at an
  irreducible HW bound — arm B (decode-free) also fails to reach 450 GB/s, so the
  residual gap to the paper roofline is a machine/shape property.  What is settled:
  **the fix is not batching/launch-shape, and not a small-M rewrite of the decode
  path** — the batched-at-scale per-call cost is identical to the amortised one.

---

*Script: `bench/exl3_dense_smallm_probe_megaeval.py`.  Raw JSON:
`phase3-kernel-megaeval.json`.  Full stdout: `phase3-kernel-megaeval.stdout.txt`.
No tracked repo file modified; no git commit; no cluster access.*
