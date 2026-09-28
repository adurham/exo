# Phase 10 -- Plan B's day-2 gate: full-concentration routing trace + device soak

Date: 2026-09-28
Host: macstudio-m4-1 / macstudio-m4-2 (measurement) + hermes-gw-01 (analysis)
Status: gate experiment complete. B (streamed native) measured **marginal**; a
correctness bug found en route is **fixed**; the 40-layer forward is **coherent**.

## Bottom line

1. **The gate numbers.** On the real 40-layer routing trace (531 tokens, all 40
   layers of the EXL3 checkpoint) at a 256-experts/layer LRU budget the steady
   hit rate is **89.8%** -- the full-model number, below phase-5's reduced-model
   92.8%. Streamed native therefore costs **230 MB/token/rank**, needing
   **5.76 GB/s at 25 tok/s against 6.2-6.3 GB/s measured** -- a **1.08x margin**
   -- and **7.37 GB/s at 32 tok/s, which the SSD does not have**. Ceiling:
   **26.9 tok/s**.
2. **MTP is I/O-negative on the streamed path.** Spec verify does (gamma+1)
   positions per step for (a+1) accepted tokens; the per-position miss rate is
   unchanged by batching (measured W=5: 0.6131 vs W=1: 0.6124 miss/token/layer),
   so streaming I/O per produced token multiplies by **(gamma+1)/(a+1) ~ 1.7-2.0x**
   at realistic acceptance (gamma=5, a~2.5 -> 1.71x; gamma=3, a~1.15 -> 1.86x).
   Streamed-native + MTP lands **~14-16 tok/s** I/O-implied.
3. **EXL3 needs no streaming at all**: experts fully resident = **98.9 / 97.2 GB
   per rank** (exact per-layer tiered byte counts), **104.3-106.0 GB/rank** with
   non-expert bytes -- fits the 115 GB wired limit with **9-10.7 GB headroom**.
   Its open question is kernel speed (2.65x decode / 3.05x verify vs production
   MXFP4), not memory and not I/O.
4. **A real correctness bug was found and fixed**: the EXL3 expert path was
   missing V4.1's clamped SwiGLU (`swiglu_limit=10`). Unclamped, layer 39's
   activations explode (rms 150.45, absmax 20401); with the clamp the rms curve
   is smooth through L39 (34.58). Fixed as an additive `silu_clamp` mode in
   PonyExl3 (stock modes bit-identical, backup kept).
5. **The trace doubles as the first real 40-layer V4.1 forward, and it is
   coherent**: teacher-forced NLL mean **1.003** (median 0.029), top-1 **78.3%**;
   the EXL3 6-bit head reconstructs to cos **0.9997** vs the native head; routing
   is deterministic across runs.

## 1. Device soak (the disk half)

35 minutes of shuffled 13.32 MB expert fetches over the real **196.0 GB** EXL3
working set, depth 8, with per-second `iostat` as device ground truth:

| node | core-window median | p10 | p90 | all-window n |
|---|---|---|---|---|
| macstudio-m4-1 | 6206 MB/s | 6064 | 6237 | 1501 |
| macstudio-m4-2 | 6284 MB/s | 6159 | 6314 | 1517 |

The AP1024Z spec is ~6.7 GB/s; the soak sustains **92-94% of spec** under the
app's access pattern. Take-1's process-level rates (21-26 GB/s) were
cache-served and rejected -- `F_NOCACHE` does not truly bypass on APFS; iostat
is truth.

## 2. Routing trace (the concentration half)

`p30_exl3_trace.py` builds each of the 40 layers from the real EXL3 checkpoint
(PonyExl3 kernels, gate weights at full precision), runs one 531-token
multi-domain prompt through the stack layer-major, and records every gate's
top-6. 21,240 records; one checkpoint pass in ~80 s. Determinism: L0/L20/L39
first-96 picks identical across runs.

| budget/layer | cap | steady hit (full model) | miss/token/rank | native MB/token/rank | native tok/s @6.2 |
|---|---|---|---|---|---|
| 48 GB | 128 | 84.6% | 18.5 | 348 | 17.8 |
| 72 GB | 192 | 88.1% | 14.2 | 268 | 23.2 |
| 96 GB | 256 | **89.8%** | **12.25** | **230** | **26.9** |
| 120 GB* | 320 | 90.0% | 12.0 | 225 | 27.6 |

\* cap-320 does not fit native (120 GB > headroom); shown for the curve's shape.
The curve is flat above cap 256 -- more cache buys almost nothing until full
residency. Routing is diffuse (per-layer distinct experts: min 227, med 277,
max 331 of 384), exactly the direction phase-5's "optimistic ceiling" caveat
predicted.

Required bandwidth vs the two targets: **5.76 GB/s @ 25 tok/s** (1.08x margin
on 6.2), **7.37 GB/s @ 32 tok/s** (does not fit). EXL3 byte counts for the same
trace: **157.7 / 155.0 MB/token/rank** (rank0/rank1) -> 3.9/3.9 GB/s @25 (1.6x
margin), ceilings **39.3 / 40.0 tok/s**.

## 3. The MTP correction

A spec-decode step runs (gamma+1) positions to accept (a+1) tokens. Batching
does not change the per-position miss rate (0.6131 vs 0.6124 at W=5 vs W=1),
and the union fetch is 18.8 experts/step vs 30 naive picks (37% saved) -- but
the I/O **per produced token** scales with positions/token:

| config | positions/token | native MB/token/rank | tok/s @6.2 (I/O) |
|---|---|---|---|
| plain R=1 | 1.0 | 230 | 26.9 |
| MTP gamma=3, a~1.15 | 1.86 | 428 | 14.5 |
| MTP gamma=5, a~2.5 | 1.71 | 395 | 15.7 |

So on the streamed path MTP trades compute latency for extra streaming volume;
it needs near-perfect acceptance (~100% of gamma) merely to break even on I/O.

## 4. Residency re-check with exact per-layer bytes

From safetensors headers, per 384-expert layer (tiered: 35 layers k=3 at
13.32 MB/expert, 5 layers 18-22 at 8.89 MB/expert):

| format | experts/rank (20 layers) | + non-expert (~7.18 GB) | vs 115 GB wired |
|---|---|---|---|
| native mxfp4, full | 144.4 GB | -- | does not fit |
| native, cap-256 LRU | 96.3 GB | -- | fits (streaming) |
| EXL3, cap-256 LRU | 65.9 / 64.8 GB | ~73 GB | fits easily |
| **EXL3, full** | **98.9 / 97.2 GB** | **~104.3-106.0 GB** | **fits, 9-10.7 GB headroom** |

## 5. The clamp bug

The port's own routed experts use `ClampedSwiGLU(swiglu_limit=10)` (config:
`swiglu_limit: 10.0`); the EXL3 library had only silu/gelu, so the expert path
ran unclamped. Symptom: L39 experts turn a 0.473-rms input into 44.9-rms output
and the layer rms jumps 4.21 -> 150.45 (absmax 20401). Fix:
`patch_exl3_clamp.py` adds an additive `silu_clamp` activation (gate upper-only,
up two-sided, limit from `EXL3_MOE_CLAMP`, default 10.0) with bit-identical
stock paths. Post-fix L39 rms 34.58; the unclamped-vs-clamped routing diff is
1725/21240 entries (8.1%), confined to layers >= 17 -- consistent with
depth-growing activations crossing the clamp threshold. L39's residual elevation
vs trend (~34.6 vs ~4.4) is unexplained; the model's norm absorbs it (collapsed
7.95 -> norm-out rms 0.206 ~ L38's 0.269) and output quality is unaffected.

## 6. Coherence of the first real 40-layer forward

Teacher-forced over the prompt, from the clamped trace's L39 state:
NLL **mean 1.003 / median 0.029**; top-1 **78.3%** overall (77.7% for pos>100);
predictions track the text ('IP' -> 'IP', 'sc' -> 'sc', 'aling' -> 'aling').
The EXL3 6-bit head (`head.trellis/suh/svh`, `head_bits=6`) reconstructs with
cos **0.9997** vs the native head and identical NLL; the mul1=F control gives
NLL 28.2, proving the test discriminates. (The EXL3 checkpoint DOES carry a
quantized head -- `embed.weight` is NOT the head; `tie_word_embeddings=false`.)

## 7. What this does to the decision

- The 2026-09-28 consult called **B (streamed native), provisionally, gated on
  this experiment**. The gate does **not** confirm a comfortable margin for B:
  1.08x at the 25 tok/s bar device-level (before any application overhead),
  fails the 32 tok/s stretch, and MTP -- a hard requirement -- makes the
  streaming bill ~1.7-2x worse per token.
- **A (EXL3 resident)** is memory-feasible with no streaming; its open question
  is kernel throughput (2.65x decode / 3.05x verify vs production MXFP4; the
  structural floor from the day-1 spike is ~1.55-1.68x).
- The remaining question is a build-path call, not a measurement.

## Artifacts

- `scripts/p30_exl3_trace.py` -- 40-layer routing-trace runner (v3: clamped activation)
- `scripts/p31_trace_analyze.py` -- LRU/union simulation + gate arithmetic
- `scripts/patch_exl3_clamp.py` -- additive `silu_clamp` patch for PonyExl3 `exl3_moe.py`
- `scripts/p33_collapse.py` -- head-collapse check on saved states
- `raw/p24-iostat-summary.txt` -- device soak, per-node medians/p10/p90
- `raw/p30-gate_summary2.txt` -- machine-generated gate numbers
- `raw/p30_exl3_trace_clamp.json` -- the full trace (21,240 records)
- `raw/p33-full.log`, `raw/p33-clamp-rms.txt` -- per-layer rms, pre/post clamp
- `raw/p34-clamp-l39-deep.txt` -- component split + weight norms for L37-39
- `raw/p38-head-nll.txt` -- head cross-check + NLL evidence

## Corrections carried into this phase

| claim | source | now |
|---|---|---|
| 92.8% steady hit / 162.7 MB / 39.9 tok/s @cap256 | phase-5 (reduced-model trace) | full-model trace: 89.8% / 230.2 MB / 26.9 tok/s |
| "Plan B's SSD ceiling is ~40 tok/s" (comfortably above bar) | phase-5 | 26.9 tok/s, 1.08x device margin at the bar |
| (unpriced) MTP effect on streaming volume | -- | (gamma+1)/(a+1) ~ 1.7-2.0x per token |
| EXL3 clamp | presumed present | was missing; now added + verified |
