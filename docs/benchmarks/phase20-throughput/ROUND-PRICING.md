# ROUND-PRICING.md — Fable-contested lever pricing round (Q1–Q5)

Author: Phase-20 PM (delegation). Date: 2026-10-09 (CDT). Worktree `/private/tmp/phase20-campaign`.
Branch `deploy/phase20-campaign`.

This round **PRICES** the five levers a Fable review named against prior closures. It is **NOT a ship
round**: nothing is re-quantized, no static default is changed, no code is deployed. Outputs are price
tables, attribution verdicts, and — for anything that would be a quality-gated config change — an explicit
**OWNER-DECISION memo**.

---

## 0. DECLARED BUDGET (fixed BEFORE spending, per the brief)

- **Boots: ≤2 TOTAL.**
  - **Q2 (gamma re-pricing) = 1 boot.** VERIFIED: per-request `spec_gamma` is **NOT** in the deployed build
    (`fb4f9290b`; it lives on the divergent `deploy/next14-gamma` branch, `git merge-base --is-ancestor
    01c416b10 fb4f9290b` = FALSE), and `EXO_SPECULATIVE_GAMMA` is read in generator `__post_init__`
    (`batch_generate.py:845`, `dsv4_mtp.py:3969`) → **one γ per boot**. So Q2 = 1 boot with arms
    γ2 + γ3 (control) interleaved × {benign 20K, agentic 91K}; a 2nd boot (reserve) only if a γ4 add is
    worth an extra boot.
  - **Q4 (MoE-drain differential) = 0 boots.** VERIFIED runnable as a **bench-only** process: `load_experts`
    loads the layer-20 `EXL3SwitchGLU` in the node venv on studio2; decode-class forward smoke: E=16,
    D=5120, H=1152, R=1 topk3 0.510 ms / topk6 0.682 ms, R=4 0.671/1.065 ms (this PM smoke, real weights).
    No engine, no deploy needed.
- **Everything else OFF (0 boots):** Q1 (offline microbench), Q3 (offline analysis), Q5 (offline read +
  prototype).
- **No ship of anything.** Cluster stays production `fb4f9290b` + mlx-lm `16830e1`, gates unset.

### Entry state (verified ~10:57 CDT)
Production `fb4f9290b` (both nodes; `git rev-parse HEAD` on studio2 = `fb4f9290b4e0b…`), mlx-lm `16830e1`;
2 runners Ready, loadavg 1.33/1.54/2.79 (idle); model
`~/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw`; node mlx `0.32.3.dev20260918+603f16eb7`
(`mx.quantize`/`mx.quantized_matmul` present). Local mlx source for Q5: `~/repos/mlx` HEAD
`ac73d0c9eeb2240725fb203d3e6745c516a3dd8e`.

### 2nd-opinion fold-in (auxiliary.consult, pre-dispatch)
Adopted: (Q1) **graph-chained timing** (not per-call at m=1 — per-call is launch-overhead-dominated at
tiny shapes; chain the whole layer's dense roster, one eval, /n_reps); **TP-sharded shapes**; **quality is
UNMEASURED** (re-quant of an already-2.9bpw-lossy tensor ≠ model-quality proxy) and **capacity check**
before the expert-side memo. (Q4) the isolated topk differential is a **compute, not a drain/overlap**
proxy → frame it as a **totals comparison** (standalone expert-block ms × n_moe_layers vs the ~37 ms
unattributed in verify_block); **cost tracks unique experts touched, not k·R** → use real-ish routing, not
uniform random; optional **forced-sync arm** to bound exposure. (Q3) confirm the 101.06 client number's
build before attributing 7.26 ms; the engine-loop read is a correctness answer → **sr-coder**. (Ordering)
run Q1/Q3/Q4/Q5 before the Q2 boot; fold any live claim into it. (Contention) benches share studio2's GPU
with production → **idle-guard before and after, bounded reps**.

---

## Q1 — Quant-format pricing (EXL3-fused vs native quantized `mx.quantized_matmul`)  [STATUS: DONE — OWNER DECISION]

Offline microbench on studio2 (real layer-20 dense roster, TP=2 rank-0 sharded, real EXL3 weights
decoded then re-quantized in-memory g64). Script `bench/p20_pricing_q1_dense.py`; raw
`raw/pricing/q1/`. Whole-slice = 18 linears chained into one lazy graph + one eval; also K-batched per-call.

| arm (ms per dense layer) | m4 whole | m4 K/call med (p95) | ratio m4 | projected ms/round (×40) |
|---|---:|---:|---:|---:|
| **EXL3 prod (fused)** | 0.945 | 0.774 (0.780) | 1.00× | **37.8** |
| native q4 **+Hadamard (fair drop-in)** | 0.483 | 0.314 (0.317) | **2.47×** | **19.3** |
| native q5 +Hadamard | 0.482 | 0.338 (0.341) | 2.29× | 19.3 |
| native q6 +Hadamard | 0.482 | 0.336 (0.339) | 2.30× | 19.3 |
| native q4 raw (no Had — not bit-fair) | 0.374 | 0.229 (0.240) | 3.38× | 15.0 |

- **OWNER DECISION — TRIGGERED.** The native-qN rate at m=4 is **≥2×** the EXL3 rate (2.47× fair /
  3.38× raw), projecting **~18.5 ms/round** off the dense slice. A quant-format change (EXL3 2.9bpw
  trellis → native q4/q5 g64) is **QUALITY-GATED — NOT acted on this round.** Gate = real-quality
  validation against the **original** weights + a full-model two-node run (layer-level wins have not
  always survived full-depth — row-limit / GPU-timeout history).
- **Quality is UNMEASURED.** Cosine to the **bf16 EXL3-decoded W** (not to the model): q4 0.9959,
  q5 0.9990, q6 0.9998. A qN re-quant of an already-2.9bpw-lossy tensor measures fidelity to EXL3,
  not to the original model.
- **The win is ALU/issue, not bandwidth:** q4 streams 62 MB vs EXL3's 66.7 MB trellis (not fewer
  bytes) yet is 2.5× faster — the fused trellis decode (k=7 SWAR) is the cost. Corroborates
  `MECHANISM.md`'s decode-ALU-bound reading, but shows a **different kernel escapes it** where
  tune-existing levers did not.
- **Experts arm PARKED** (not measured): 384 experts/layer × ~71 MB fp16 = ~27 GB to re-quantize;
  native path is `gather_qmm` on a stacked layout. Would need its own round.
- Caveat: the sharded per-rank dense slice measured **37.8 ms/round**, not the older "~51 ms" (that
  was unsharded/heavier-load).

## Q2 — Gamma re-pricing on the current build  [STATUS: pending — 1 boot]

## Q3 — The 7.26 ms client↔server boundary  [STATUS: DONE — ARTIFACT]

Verdict doc `raw/pricing/q3/q3-boundary.md`. Offline source-read at `fb4f9290b` + raw-data
re-derivation; **0 boots**. **PM independently re-verified the two load-bearing claims below.**

- **VERDICT: ARTIFACT / NOT ON THE CRITICAL PATH. Addressable as a production lever: ~0 ms.**
- **Provenance:** 101.06 is the **next17 + `DSV41_INDEXER_HIER=0`** client number (a *different build*,
  agentic 91K, from `PHASE3B-SHIP-VALIDATION.md:161`); 93.8 is the **next16-instr benign** server
  `round_total`. `101.06 − 93.8` subtracts across **build × boot × shape × statistic**.
- **Same-boot, same-request** (raw/p3): client ms/round sits **+0.23…+0.51 ms** above the server
  loop's own per-round wall, both shapes. The dominant component of the doc's 7.26 is **agentic-vs-benign
  shape** (~5.5 ms; server medians 99.23/99.28 vs 93.69/93.71), not a boundary.
- **Mechanism (PM-verified):** the deployed instance is served by **`Dsv41Engine`**
  (`dsv41/dispatch.py` → `Dsv41Builder` → `Dsv41Engine`; `batch_generate.py`/`dsv4_mtp.py` are the
  batched/PP path, not imported here). `engine.py:_rounds` is a **yield-based streaming generator**;
  round N+1 is gated **only on the in-process consumer drain + the cross-rank cancel collective**,
  never on the client. One HTTP request → one long SSE stream. The channel is unbounded.
- Honest boundary: the doc's **1.40 ms in-round residual is real** (`round_total − verify_block`, an
  inside-round bracket). The correct decomposition is ~1.5 ms in-round + ~0.65 ms inter-round
  in-engine gap + ~0.3–0.5 ms client-statistic delivery offset.

## Q4 — MoE-expert drain differential  [STATUS: DONE — re-attributes the residual to MoE COMPUTE]

Bench-only offline microbench on studio2 (real layer-20 `EXL3SwitchGLU`, E=384, rank0/world2).
**0 boots** (no engine needed). Script `bench/p20_pricing_q4_modrain.py`; raw `raw/pricing/q4/`.

| arm | unique experts | median ms | dispatch-only ms |
|---|---:|---:|---:|
| R1 × topk3 | 3 | 0.530 | 0.010 |
| R4 × topk3 (correlated) | 8 | **0.615** | 0.009 |
| **R4 × topk6 (correlated, PROD)** | 14 | **1.077** | 0.009 |
| R4 × topk6 (uniform) | 24 | 1.078 | 0.009 |

- **(a)** Halving experts **at the verify shape (R=4) moves the block a LOT**: 1.077 → 0.615 ms =
  **−0.462 ms/layer (−43 %)**, **1.75×**. At the draft shape (R=1) it barely moves.
- **(b) TOTALS: `1.077 ms × 40 MoE layers = 43.08 ms` vs the ~37 ms unattributed → ratio 1.16×.**
  The MoE expert GEMM **alone** is the right order of magnitude to be the *entire* unattributed slice
  (slightly over-fills it). So the answer to "what is the ~37 ms?" is **"the MoE expert GEMMs at the
  verify shape × 40 layers"** — the mystery is attributed to **compute**.
- **(c) Lead-A impact:** this does **not** re-validate a *comm* bound — it **re-attributes the residual
  away from comm to MoE COMPUTE**. Lead-A's "≤5.25 ms exposed non-compute" is not contradicted
  (compute ≠ non-compute), but the **premise** of the comm lead (large residual = possibly comm/drain)
  is **weakened**. **The MoE path re-opens with numbers** (~18 ms/round standalone from halving at R=4).
- Honest boundary: a **1-node** bench **cannot** show cross-rank overlap/exposure. Cost is
  **slot-bound, routing-distribution-independent** (14 vs 24 unique experts → identical ms).

## Q5 — GPU-time instrument feasibility  [STATUS: DONE — GO (per-op-class), not a flag-flip to per-kernel]

Memo `raw/pricing/q5/q5-gpu-instrument.md` + prototype `proto_gpu_time.py`/`proto_granularity.py`.
Offline read of `~/repos/mlx` @ `ac73d0c9` + a live prototype on studio2. **0 boots.**

- **GO — but the existing mechanism is per-COMMAND-BUFFER, not per-kernel.**
- **Already present & turn-on-able on the node build:** `MLX_GPU_TIME=1` → `mx.metal.gpu_time_ns()`
  = Σ(`GPUEndTime − GPUStartTime`) per completed `MTL::CommandBuffer` (`eval.cpp:94-109`,
  `device.cpp:896-916`, `metal.h:32-43`; Python `python/src/metal.cpp:118-139`). PM verified the env
  var + `gpu_time_ns`/`accumulate_gpu_time_ns` symbols exist in the node's `libmlx.dylib`.
- **Prototype RAN** (studio2, offline): nonzero per-op GPU ms — matmul 1024³ gpu 0.447/wall 0.611;
  4096³ gpu 13.285/wall 13.482 (GPU-bound); quantized_matmul mxfp4 gpu 0.053/wall 0.191
  (launch-bound). `MLX_MAX_OPS_PER_BUFFER=1` gives ≈ per-kernel; compiled + custom kernels are counted
  (covers the compiled-prefill "not in spans" gap).
- **Effort:** Path 1 (per-op-class, no rebuild) **~2-4 h**; Path 2 (per-buffer label→time ring buffer +
  `mx.metal.gpu_time_records()`, ~1 round incl. node wheel rebuild); Path 3 (true per-dispatch
  `MTLCounterSampleBuffer`) **2-4 rounds**. **Recommendation: Path 1 now, file Path 2 next round.**
- Caveat: the filter **drops RDMA/CPU-only buffers** → JACCL collective transport GPU cost is not
  attributed (jaccl has no GPU timestamp hook). Overhead of per-op sync ≈ 1.15× on hot loops.

---

## End state
Production `fb4f9290b` + mlx-lm `16830e1` live on BOTH nodes, gates unset, canary healthy, no stray
processes. Budget spent: **1 boot (Q2)** of ≤2; Q1/Q3/Q4/Q5 all offline (0 boots). Nothing shipped.

