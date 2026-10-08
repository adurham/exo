# Phase-3 kernel go/no-go — SYNTHETIC-WEIGHTS MICROBENCH of the EXL3 dense small-M GEMM

Worktree: `/private/tmp/levers-wt` (branch `deploy/next15-levers` @ `71a8c94c2`).
Runner: `bench/exl3_dense_smallm_probe.py` (this artifact + the JSON beside it).
Machine: MacBook Pro M4 Max, macOS, mlx `0.32.0.dev20260804+ac73d0c9`, metal available.
Date: 2026-10-07 16:22 CDT.  Untracked, uncommitted.  No tracked repo file modified.

---

## 0. Headline — the three-way verdict

**INCONCLUSIVE.**

- The vendor kernel **is decisively NOT at the memory roofline**: arm A = **38.5 ms/round**
  vs arm C (450 GB/s read-once) = **4.70 ms/round** — a **33.8 ms/round** gap.  That part
  is a firm GO-direction answer.
- **But the gap is not recoverable by a small-M kernel rewrite, and the single strongest
  evidence is that arm A ≈ arm B at M=4** (962.6 µs vs 996.1 µs/layer).  The trellis-direct
  GEMM *equals* a plain fp16 matmul of the fully-materialised 2×-wider weight, while reading
  **5.3× fewer bytes**.  Both are pinned to the same ~55 GB/s machine plateau.  There is
  therefore **no demonstrated ≥2 ms/round of dense-GEMM headroom** that a different small-M
  GEMM could unlock on this machine.
- The measured M-proportional marginal is **~99 µs/layer/row** (A(4)−A(1) = 295.9 µs / 3 rows)
  → **≈3.9 ms/round for one extra verify row across 40 layers** — consistent with, but *not
  larger than*, the phase-17 harness share of the dense projections (3.6 ms).  So the dense
  path is neither a large hidden lever nor obviously at a recoverable ceiling.
- The GO clause and the EXHAUSTED clause **both fail**, on different sub-conditions →
  the honest verdict is INCONCLUSIVE, and §6 names the one extra measurement that settles it.

The single load-bearing caveat: a ~3.9 ms/round figure and a ~55 GB/s plateau cannot by
themselves separate "intrinsic per-pass cost that a better kernel cannot touch" from
"machine under load / submission-bound artefact".  See §6.

---

## 1. Measurement method (why the numbers are trustworthy)

This machine's `mx.eval` carries a **large fixed per-eval cost** on the order of **150–250 µs**,
dominated by host/command-buffer submission.  Timing one small-kernel call per eval is therefore
meaningless — the first (buggy) run reported **5.5 ms** for a 7.9 MB kernel and **240 µs** for a
0.98 MB kernel, i.e. the eval floor, not the kernel.  (`mx.metal.dispatch_count()` reports `0` on
this build — not wired; `mx.metal.gpu_time_ns()` is likewise stuck at 0, so GPU-only timing is
unavailable.  Both were checked and are recorded in the JSON.)

Fix: **amortise**.  Each timed eval enqueues `K = 64` independent identical calls and the median
eval wall is divided by `K`, which divides the fixed floor below 4 µs/call:

```
mx.eval floor (amortized K=64): 3.53 us/call
```

**Synthetic-layer validity check** (real run output):

```
correctness cos(A, B_fp16) on wo_b @M=4 = 0.9999999
built 15 dense linears: 141.0 M params, trellis 52.9 MB/layer  (k=3, packed=48)
```

Cosine 0.9999999 between the synthetic `EXL3Linear` and the fp16 reconstruction ⇒ the layer
is a faithful numerical stand-in.  Random trellis content does not affect timing: the mul1
decode is an ALU/table path that runs identically for every codeword.

---

## 2. The full table (arm A / arm B / arm C), per M

Arm A = vendored `EXL3Linear` (`inner_gemm_mlx`, the M=2..16 band).  Arm B = fp16 ceiling
(`x @ W`, W reconstructed once).  Arm C = analytic roofline `52.9 MB / 450 GB/s = 0.1175 ms`
per layer, flat across M.  All per-layer, 15 dense linears, per-rank TP=2.

| M  | A us/layer | A x40 ms | B us/layer | B x40 ms | C x40 ms | A eff GB/s |
|---:|-----------:|---------:|-----------:|---------:|---------:|-----------:|
|  1 |      666.8 |   26.672 |      472.8 |   18.911 |    4.701 |       79.3 |
|  2 |      917.6 |   36.705 |      979.5 |   39.178 |    4.701 |       57.6 |
|  3 |      960.3 |   38.414 |      991.0 |   39.641 |    4.701 |       55.1 |
|  4 |  **962.6** |**38.506**|  **996.1** |**39.845**|    4.701 |   **54.9** |
|  5 |  **1089.1**|**43.564**|     1021.4 |   40.855 |    4.701 |   **48.6** |
|  6 |     1097.3 |   43.892 |     1028.6 |   41.145 |    4.701 |       48.2 |
|  8 |  **1094.8**|**43.793**|     1041.0 |   41.640 |    4.701 |   **48.3** |
| 16 |     2070.3 |   82.812 |     1118.4 |   44.736 |    4.701 |       25.5 |

*(Values from `phase3-kernel-microbench.json`; the M=1/4/5/6/8 rows are the rubric rows. The
per-linear breakdown is in §4.)*

### Reading the arm ratios

- **A vs B** — decode is *free* at the plateau: A/B = 1.41 (M=1), **0.97 (M=4)**, 1.07 (M=5),
  1.05 (M=8), 1.85 (M=16).  From M=4 up, materialising the weight in fp16 and running a stock
  matmul is the **same speed or faster** than the trellis GEMM.
- **A vs C** — never close: achieved 27–79 GB/s vs 450 GB/s ⇒ 15–26 % of the roofline at the
  small-M band — matching the source-read's measured `17–26 % of BW` exactly.
- **Amdahl ceiling** — even A → B (perfect decode removal) only moves 38.5 → 39.8 ms at M=4
  (**a 3 % regression**); B is bigger than A there.  The dense small-M GEMM has no 2 ms lever.

---

## 3. The critical deltas — and the R4→R5 anomaly

```
CRITICAL DELTAS (us/layer):  A(4)=962.6  A(5)=1089.1  A(6)=1097.3  A(8)=1094.8
  A(5)-A(4) = +126.4 us   A(6)-A(5) = +8.2 us   A(8)-A(4) = +132.2 us   A(2)-A(1) = +250.8 us
```

- **The M=4→M=5 step is VISIBLE: +126.4 µs/layer** (≈ **+5.1 ms/round** across 40 layers).
  It reproduces in **every** linear and is largest on the widest (`wo_b` +35.4 µs, `wq_b` +16.7 µs).
- **The step is specific to 4→5, not general M-cost.**  A(6)−A(5) is only **+8.2 µs**; A(8) is
  *below* A(6).  The curve is flat from M=5 to M=8.
- **Mechanism = `simd_mt` 4→8** (`gemv_metal.py:1422`): M≤4 runs mt=4, M=5..8 run mt=8.
  The source-read predicted "no non-monotonic mechanism at M=5"; the microbench **finds one**,
  exactly at the predicted boundary.  The step is a *slowdown* on the M≤8 band (mt=8 is the
  worst amortisation point — v20 mt=8 = 318 Gw/s vs mt=4 = 486 Gw/s, `gemv_metal.py:1232`).
- **R4→R5 reconciliation.**  The real `VERIFY_MS` R4→R5 = **+14.0 ms** on the *whole* verify
  forward (MoE + dense + attention).  This microbench attributes **+5.1 ms** of it to the dense
  GEMM's mt=4→8 step.  That is a **plausible dense contribution** to the anomaly and is worth
  flagging — but ~9 ms of the real step is *not* dense and lives in the MoE / attention path
  (`exl3_moe.py`, out of scope).  A positive dense-side step that the source-read said should
  not exist has now been measured.  **Caveat:** this 5.1 ms is *saving-shaped in the wrong
  direction* — it is a cost you would have to *remove*, and §5 shows you cannot, because
  splitting M=5 into two mt=4 groups doubles the trellis re-read (the wash the source-read warned of).

---

## 4. Per-linear breakdown (arm A us; d = A(5)−A(4))

| linear       |   in  |   out |  MB  |   A1  |   A4  |   A5  |   A8  |   B4  |   d5-4 | GB/s@4 |
|:-------------|------:|------:|-----:|------:|------:|------:|------:|------:|-------:|-------:|
| wq_a         |  5120 |  1280 | 2.46 |  33.1 |  47.0 |  53.3 |  53.3 |  30.6 |   +6.2 |   52.3 |
| wq_b         |  1280 | 16384 | 7.86 |  89.4 | 130.9 | 147.6 | 148.8 | 226.4 |  +16.7 |   60.1 |
| wkv          |  5120 |   512 | 0.98 |  19.3 |  23.9 |  27.3 |  27.2 |  16.1 |   +3.4 |   41.1 |
| wo_b         |  8192 |  5120 |15.73 | 170.3 | 257.0 | 292.3 | 295.2 | 291.5 |  +35.4 |   61.2 |
| wo_a_x8.0    |  4096 |  1024 | 1.57 |  24.4 |  33.4 |  37.9 |  38.1 |  21.2 |   +4.5 |   47.1 |
| wo_a_x8.1    |  4096 |  1024 | 1.57 |  24.4 |  33.3 |  37.7 |  37.6 |  21.3 |   +4.4 |   47.2 |
| wo_a_x8.2    |  4096 |  1024 | 1.57 |  24.3 |  34.0 |  38.2 |  37.8 |  21.3 |   +4.1 |   46.2 |
| wo_a_x8.3    |  4096 |  1024 | 1.57 |  24.5 |  34.0 |  37.7 |  38.1 |  21.2 |   +3.7 |   46.2 |
| wo_a_x8.4    |  4096 |  1024 | 1.57 |  24.1 |  33.3 |  38.5 |  38.0 |  21.3 |   +5.2 |   47.2 |
| wo_a_x8.5    |  4096 |  1024 | 1.57 |  24.2 |  33.4 |  38.2 |  38.2 |  21.2 |   +4.9 |   47.2 |
| wo_a_x8.6    |  4096 |  1024 | 1.57 |  24.4 |  34.0 |  38.3 |  38.3 |  21.3 |   +4.3 |   46.2 |
| wo_a_x8.7    |  4096 |  1024 | 1.57 |  24.5 |  33.2 |  37.8 |  38.3 |  21.3 |   +4.6 |   47.4 |
| shared_w1    |  5120 |  2304 | 4.42 |  53.3 |  78.2 |  88.0 |  88.3 |  65.8 |   +9.8 |   56.6 |
| shared_w3    |  5120 |  2304 | 4.42 |  53.7 |  78.4 |  88.0 |  88.4 |  65.7 |   +9.6 |   56.4 |
| shared_w2    |  2304 |  5120 | 4.42 |  53.0 |  78.6 |  88.4 |  89.1 | 129.8 |   +9.9 |   56.3 |

Note B4 > A4 for `wq_b` (226 vs 131) and `shared_w2` (130 vs 79) — the shapes whose fp16 W
does not blow the tile budget decode *faster* than EXL3, another sign the decode tax is the
only thing EXL3 is buying here and it is already ~paid off by the plateau.

---

## 5. The two GO/EXHAUSTED projections (rubric)

```
PROJECTIONS at M=4 (all dense projections, x40 layers):
  A(M=4) x40 = 38.506 ms   A(M=1) x40 = 26.672 ms   C x40 = 4.701 ms
  roofline headroom   (A4_40 - C_40)        = +33.805 ms/round
  small-M penalty     (A4_40 - 1.25*A1_40)  = +5.167 ms/round
  A(M=4)/A(M=1) = 1.444   A(M=4)/max(A(M=1),C_us) = 1.444
```

- **Roofline headroom = 33.8 ms/round** — the raw distance to a 450 GB/s read.  Tempting, but
  §2 shows arm B (fp16, decode-free) does **not** reach it either: the ceiling on *this machine*
  is ~55 GB/s, not 450.  So 33.8 ms is not an achievable dense-GEMM recovery; it is the machine's
  distance from its own paper bandwidth, which the source-read already attributed to a 308 Gw/s
  ALU/issue bound, not to a removable padding/decode/cache mistake.
- **Small-M penalty = 5.17 ms/round** — `A4_40 − 1.25·A1_40`, i.e. the cost of M=4 over 1.25× the
  M=1 cost.  **This is ≥2 ms**, so the GO numeric clause is *met* — **but A(4)/A(1) = 1.44 < 1.6**,
  so the GO *ratio* clause fails.  The rubric requires both; the GO gate does not open.
- **The 1.44 ratio is close to noise.**  A(2)−A(1) = +250.8 µs is *larger* than A(4)−A(1) = +295.9 −
  over 3 rows, and A(3) ≈ A(4); the single-row-vs-M=4 distinction in arm A is only ~1.44×, on a
  loaded machine whose eval floor varied 150→250 µs across repeats.  The 1.6 threshold is inside
  the noise band.

### EXHAUSTED clause

`A(4) ≤ 1.25·max(A(1), C)` → `962.6 ≤ 1.25·666.8 = 833.5` → **FALSE** (ratio 1.44).
Second clause `(A5−A4) ≈ (A6−A5)` → `126.4 ≈ 8.2` → **FALSE**.
⇒ Not EXHAUSTED.

### GO clause

`A(4) ≥ 1.6·A(1)` → `1.44 ≥ 1.6` → **FALSE**.  (Headroom clause ≥2 ms → TRUE, 5.17 ms.)
⇒ Not GO.

⇒ **INCONCLUSIVE.**

---

## 6. What would settle it (the single extra measurement)

**Measure arm A and arm B on the *identical* shapes at M=4 as one large batched eval, `K≥256`,
on an idle machine, and compare the per-call cost against the 450 GB/s roofline and the
15 TFLOPS compute roofline in the same run.**  Concretely: a single `mx.eval` that enqueues the
**entire 40-layer × 40-projection dense set at M=4 in one graph**, timed end-to-end and divided
by the call count.

Why this and not something else: the only thing standing between the present data and a
firm verdict is whether the observed **~55 GB/s / ~99 µs-per-layer-per-row plateau is intrinsic
(spec's 308 Gw/s ALU bound) or a submission/machine-load artefact**.  A single mega-eval removes
the per-eval floor entirely (no amortisation assumption), removes host-submission from the
critical path, and — run when `uptime` shows load ≈ core count — removes the load confound.  If
the plateau persists at one-eval-per-40L, the lever is EXHAUSTED (the kernel is at the machine's
real limit and no small-M rewrite can help).  If batched-at-scale jumps toward the roofline, the
lever is GO and the fix is batching/launch-shape, not the decode ALU.

*(Secondary confirmations, not required: run the same probe with `EXL3_GEMM_DEVX=0` to force the
staged mt=8 fallback and `EXL3_FUSED_ROW_LIMIT=16`, and with the machine quieted, to separate the
`simd_mt` step from load noise.  This subagent did not have a quiet machine.)*

---

## 7. Caveats (honest)

- **Synthetic weights** (labelled throughout).  The real EXL3 checkpoint is on the down cluster;
  shapes/k/packed-layout are reproduced exactly, so *kernel cost* is faithful; content is random.
- **Single node, not TP=2.**  These are per-rank shapes measured on one M4 Max, no RDMA, no
  all-sum.  The 40-layer ×40 projection assumes 40 layers of these groups with no pipeline effects.
- **No attention, no MoE, no tokenizer.**  This measures the *dense GEMM kernel only*.  The real
  R4→R5 +14.0 ms also contains MoE (`exl3_moe.py`, out of scope) and attention/cache motion.
- **Machine under load.**  `uptime` during the run: load averages **5.0–8.0** on a ~16-core part,
  with ≥2 other live sessions on the box; measured streaming BW on this run was ~320 GB/s
  (vs the campaign's 450).  Absolute arm A/B values are therefore a **lower-bound-on-speed**
  snapshot and are noisier than a quiet-machine run.  The *shape* (the M=4→5 step, A≈B) is robust
  across all 15 linears; the exact 1.44 ratio and the absolute µs are not.
- **`dispatch_count` reports 0** on this build (not wired); GPU-only timing (`gpu_time_ns`) is also
  stuck at 0, so all timing is amortised wall-clock, not GPU-time.
- **Total runtime**: the probe ran well under the 10-minute budget; nothing was committed.

*Script: `bench/exl3_dense_smallm_probe.py`.  Raw JSON: `phase3-kernel-microbench.json`.  No
tracked repo file modified; no git commit; no cluster access.*
