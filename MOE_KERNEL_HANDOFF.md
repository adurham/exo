# MoE Prefill Kernel Optimization — Handoff (2026-07-01/02)

## WHERE THIS SESSION LANDED (all deployed + quality-gated)

Prefill at 100K-target (76,213 real tokens): 255 -> 353 t/s (+38%)
Prefill at 495K real tokens:                 ~200* -> 306 t/s
Prefill at 727K real tokens:                 215 t/s (needle still hit)
Decode (1200-tok essays, 3x):                29.0 t/s mean (top of 25-30 band)
The pre-existing "340K prefill cliff" (270->40 t/s collapse): GONE.
  Mechanism was the indexer's O(P log P) argsort; argpartition removed it.
(* old baseline at 495K not directly measured; cliff made it far worse)

Every number above: cache-busted probe (bench/ab_probe_tier1.py), wall-clock
time-to-first-token cross-checked vs server "Prefill complete" log lines,
needle-recall + BOS-spam + short-prompt gates all clean.

## DEPLOYED STATE (as of end of session)

- exo origin/main = 09b259c1 (nodes were launched on 3fea0b36 — the bench-
  only commit 09b259c1 needs no relaunch; next start_cluster.sh picks it up).
- mlx-lm (adurham fork) main = 2254ae3, gitlinked from exo 3fea0b36.
- mlx: deployed = 980ac156a (unchanged this session). Branch
  `tile-sweep-wip` (03e33c791) adds MLX_GATHER_QMM_RHS_LHS_TILE env override
  — NOT deployed, default tiles unchanged, safe to merge anytime.
- Cluster: running the final config, healthy at session end.

New defaults in start_cluster.sh (commit 7f376920):
  EXO_PREFILL_STEP_SIZE=256, DSV4_PREFILL_STEP_SIZE=256
  EXO_DSV4_PREFILL_ARGPARTITION=1, EXO_DSV4_ARGPARTITION_MIN_P=8192
  EXO_DSV4_LMHEAD_LASTROW=1
New mlx-lm code (both default-safe):
  7c721d9  LMHEAD_LASTROW min-L gate (L>32) — the bare L>1 gate SLICED THE
           MTP VERIFY LOGITS -> repetition-loop degeneration (reproduced
           on-cluster, root-caused, fixed). _LMHEAD_LASTROW_MIN_L env.
  2254ae3  OPT-12 tail-restricted pmask (bit-exact, default ON,
           EXO_DSV4_TAIL_PMASK=0 kill switch). Perf-neutral at 495K
           (306 vs 305) — kept for reduced memory traffic.

## A/B LEDGER (what was tried, what happened)

WINS (deployed):
  argpartition (MIN_P=8192):      255 -> 289 t/s @76K
  + lm_head last-row:             -> 295 t/s
  + chunk 256:                    -> 353 t/s  (argpartition removed the
    L-scaling term that made 256 lose in the 2026-06-13 test)

LOSSES (rejected, do not retry without new mechanism):
  chunk 512:        293 @76K, 267 @495K — OPT-4 tiling overhead dominates.
  gamma=3 decode:   25.2 vs 29.0 t/s at gamma=2. Acceptance doesn't pay.
  fused topk kernel (EXO_DSV4_TOPK_FUSED): bench/topk_fused_verify_bitexact.py
    PROVED it drops non-tie candidates at the K boundary EVEN AT L=1
    (K_LOCAL=4 per-thread eviction). Never enable. Harness kept as gate.
  qmm tile sweep:   default 16,32,32,1,2 is local optimum (10 configs, all
    worse: 32x32 -63%, 64x64 -65%...). Falsified tile-retune.
  M-batched gather layout (M=6/expert): only 1.19x. Not the fix alone.
  scalar-LUT M-batch GEMV prototype (bench/qmv_mbatch_proto.py): 9 GB/s,
    wrong numerics — mxfp4 in-memory layout doesn't match naive LUT sketch.
    Use the existing qdot/Dequantize machinery instead (it's correct).

## THE ONE BIG OPEN ITEM — M-batched quantized GEMV (next session's job)

MEASURED FACTS (bench/qmm_bandwidth_bench.py + friends, on m4-1):
  fp_qmv_fast (M=1):        448 GB/s   <- near the 546 peak
  steel qmm dense M=6:      173 GB/s
  steel qmm dense M=16:     103 GB/s
  gather_qmm_rhs_lhs (prod prefill path, 1536 sorted pairs): 141 GB/s
  bf16 GEMM M=256 reference: 8.6x faster than quantized qmm at same M.

moe.switch_mlp is 43% of prefill chunk wall (spans, 495K) and runs at ~26%
of memory bandwidth. This is THE remaining prefill lever. Roughly:
recovering even 300 GB/s in this path ≈ +15-20% end-to-end prefill
=> 306 -> ~350-370 @495K = the user's stated target (350 @500K).

DESIGN (worked out this session, not yet written):
  Generalize fp_qmv_fast_impl (mlx/backend/metal/kernels/fp_quantized.h:325)
  from 1 to M<=8 activation rows:
  - Keep the EXACT weight-streaming structure: 2 simdgroups x 4 output rows,
    packs_per_thread=2, qdot<U, values_per_thread, bits> per row — this is
    what achieves 448 GB/s and is numerically the shipped path.
  - Add thread-register x_thread[M][values_per_thread] (M<=8 keeps register
    pressure OK at values_per_thread=16 fp32 -> 128 floats + accumulators;
    if occupancy tanks, template on M and stop at M=4).
  - result[M][4] accumulators; simd_sum reduce each; y is (M, N) block.
  - Gather entry point: iterate sorted same-expert runs (run-length encode
    rhs_indices like affine_gather_qmm_rhs_lhs does at quantized.h:2438),
    each run of rows shares one weight stream. Ragged runs: loop in chunks
    of M_TILE, last chunk partial via qdot_safe/load_vector_safe.
  - Dispatch: in GatherQMM::eval_gpu (quantized.cpp ~1628), route
    right_sorted && M_per_expert small && transpose to the new kernel.
    Keep gather_qmm_rhs_lhs for large-M; env kill switch for A/B.
  - AFFINE variant too if needed (quantized.h has parallel affine_* set) —
    but DSv4 experts are mxfp4/mxfp8 = fp_quantized.h family. Check
    shared_experts (mxfp8 g32) also route here; they do (same SwitchGLU? NO
    — shared_experts are DeepseekV4MLP dense nn.Linear -> QuantizedMatmul,
    fine, different path).
  VALIDATION GATES (non-negotiable):
  1. Bit-exactness vs existing path: same qdot => bitwise-equal expected.
     Write bench/qmv_mbatch_bitexact.py asserting exact equality vs
     gather_qmm output across M in {1..8}, ragged runs, both fp4/fp8.
  2. Microbench: must show >=2x on the 1536-pair production shape.
  3. Cluster: 495K probe vs 306 t/s reference + needle + BOS + smoke.
  BUILD NOTE: local Mac Metal toolchain missing (xcodebuild
  -downloadComponent MetalToolchain to fix); m4-1 builds fine:
  venv at ~/scratch/tilesweep-venv (py3.13, cmake, nanobind), source clone
  at ~/scratch/mlx-tilesweep, build cmd:
  CMAKE_BUILD_PARALLEL_LEVEL=14 CMAKE_ARGS="-DMLX_METAL_JIT=ON" \
    ~/scratch/tilesweep-venv/bin/pip install . --no-deps --no-build-isolation
  (JIT build needed only for tile-env experiments; the new kernel should be
  AOT-instantiated in fp_quantized.metal like the rhs kernels are.)

## SECOND OPEN ITEM — decode (25-30 t/s band)

Decode cycle budget mapped but NOT yet attacked:
  ~50 collectives + ~45 forced mx.eval per cycle (draft fence
  mtp_module.py:871, per-layer MoE all_sum eval deepseek_v4.py:~1470).
  ALLSUM probe (495K prefill run): per-layer fence wall p50 15-26ms,
  outliers to 207ms. FENCE_EVERY_N_LAYERS=4 is the deployed default; OPT-7
  (batching evals) FAILED for B=2 prefill historically — but the
  measurement of what fraction of DECODE cycle wall is fence overhead is
  still missing. Next probe: EXO_DSV4_MTP_PROFILE=200 launch + decode probe,
  read draft/verify/accept phase walls.
  Also unmeasured: EAGLE_K sweep (K=8 deployed; K affects draft only,
  quality-invariant by construction).

## THIRD OPEN ITEM — the smooth O(P) decline past 500K

353 (76K) -> 306 (495K) -> 215 (727K). NO stalls (verified: chunk times
uniform, max 2.2s — see gap_analysis.awk). Decline drivers by spans share
at 495K: moe.switch_mlp 43%, moe.all_sum 11%, all_gather 8.5%, o_proj 7.9%,
proj_qkv 7.7%, sdpa 7.0%, indexer 4.5%. The all_sum/all_gather shares are
fence-serialization, not wire time. After the M-batch GEMV lands, re-profile.

## MEASUREMENT PITFALLS DISCOVERED (also in warm memory 847/848)

1. LOG EXTRACTS THROUGH THE TOOL PIPE TRUNCATE SILENTLY (~50KB). Gap
   analysis on extracted lines manufactured phantom 224-930s "stalls" that
   burned ~3 probe cycles chasing allocator/realloc ghosts. ALWAYS compute
   on-node: scp bench/gap_analysis.awk + awk -f ... -v total=<N> ~/exo.log.
2. KV PREFIX CACHE serves byte-identical prompts -> "98,813 t/s prefill"
   (i.e., NO prefill). ab_probe_tier1.py now salts every run (uuid header +
   per-filler salts). Any future probe MUST cache-bust or relaunch.
3. SECTION-TIME probe (EXO_DSV4_SECTION_TIME) is DEAD CODE on current tree:
   its layer_count accumulator lived in the compiled-block fast path removed
   2026-06-18. Dumps never fire. Use EXO_PROFILER=spans (works, per-prefill
   dump) or the ALLSUM probe.
4. Spans max-times accumulate across the whole run (incl. load) — do not
   read "max 80s" as a runtime stall without corroborating chunk-gap data.

## NEW BENCH FILES (all in bench/, committed 09b259c1)

  ab_probe_tier1.py        cache-busted prefill+decode probe (the workhorse)
  decode_probe_ab.py       steady-state decode probe (3x1200 tok)
  gap_analysis.awk         on-node chunk-gap analyzer (run via ssh)
  topk_fused_verify_bitexact.py  proves fused-topk approximate; keep as gate
  tail_pmask_bitexact.py   OPT-12 exactness gate (all shapes OK)
  qmm_bandwidth_bench.py   MoE qmm achieved-bandwidth at production shape
  qmm_mbatch_bench.py      M-batched layout A/B (1.19x, not the fix)
  qmm_tile_sweep.py        tile sweep driver (needs JIT build + env override)
  qmv_mbatch_proto.py      failed scalar prototype (kept as negative result)
  stall_catch.sh           live stall catcher (gitignored; /tmp probe + sample)

## PRIORITY ORDER FOR NEXT SESSION

1. M-batched quantized GEMV kernel (the 350@500K play). Design above.
2. Decode MTP_PROFILE measurement -> fence/collective attack if it's >20%.
3. Re-run 727K after (1) — the O(P) MoE term shrinks, curve should lift.
4. Merge tile-sweep-wip into mlx main if keeping the env override.

===========================================================================
2026-07-02 SESSION — M-BATCH GEMV BUILT; PREMISE CORRECTED; NEW PLAY: MXFP4
===========================================================================

## WHAT SHIPPED (mlx main = cb539cda, exo main = 2347fffa, deployed)

gather_qmv_rhs — M-batched qmv over sorted expert runs, BOTH families:
  fp_quantized.h:  fp_gather_qmv_rhs   (mxfp4/mxfp8/nvfp4, bits 4/8)
  quantized.h:     affine_gather_qmv_rhs (affine, bits {4,8}, gs {32,64})
Structure: grid (B, N/(2*RPS)); only run-start threadgroups work; a run is
processed in M_TILE-row chunks holding x rows in registers (bf16 via
qdot_xt in fp; fp32 + per-group sums in affine), qmv_fast weight streaming.
BITWISE-IDENTICAL to the gather_qmv_fast path (gate:
bench/qmv_mbatch_bitexact.py — 72/72 exact incl. affine, both tiles).
Dispatch gate (GatherQMM::eval_gpu): M==1 && B>=16 && sorted && transpose
&& bits in {4,8} && dense x && N%8==0 && K%block==0 && B/E in [2,8].
Kill switch MLX_GATHER_QMV_RHS=0; MLX_GATHER_QMV_RHS_TILE (default 4),
MLX_GATHER_QMV_RHS_RPS (default 4).

## THE PREMISE CORRECTION (read this before optimizing MoE again)

THE DEPLOYED DSv4-Flash CHECKPOINT IS AFFINE 8-BIT g64 — config.json
quantization = {group_size: 64, bits: 8, mode: affine}. It is NOT
mxfp4/mxfp8. make_quantization_config() in deepseek_v4.py (mxfp4 experts)
is the CONVERSION RECIPE, not this checkpoint. Consequences:
- The 2026-07-01 "141 GB/s prod path" was measured on synthetic MXFP4
  weights. Production never ran that kernel or that mode.
- fp_qmv_fast's 448 GB/s was also the fp family — irrelevant in-situ.
- Actual prod prefill MoE shape: seq-split halves the 256-token chunk ->
  B=768 pairs / 256 experts (B/E=3), affine 8b. At that shape the OLD
  sorted path already runs at ~427 GB/s effective (SLC absorbs the 3x
  re-reads). THERE WAS NO 3x BANDWIDTH LEVER. The e2e probe confirmed:
  349 t/s @111K real = exactly the old 353@76K..306@495K curve.

## MEASURED KERNEL MATRIX (m4-1, bench/qmv_mbatch_bench.py, ms/layer)

  shape                      steel   qmv_rhs(best)  speedup
  mxfp4  B=1536 (B/E=6)      11.56   7.81 (mt4)     1.48x
  mxfp4  B=768  (B/E=3)       5.35   4.63 (mt6)     1.16x
  affine8 B=1536 (B/E=6)     ~15.3  ~13.2 (mt4)     1.16x
  affine8 B=768  (B/E=3)      8.56   8.48 (mt4)     1.01x
  affine8 B=4096 (B/E=16)    12.80  38.96 (mt4)     0.33x  <- hence B/E<=8
Kernel-iteration lessons (5 variants benched): threadgroup staging +
barriers kill the streaming pipeline (v2); explicit dequant-once regressed
— the compiler already hoists pack decode across the m-loop, w_vals just
added spill (v3); the register wall is x_thread fp32 (64 floats OK, 96
dead) — bf16 x registers fixed it (v5); RPS=8 x-amortization ~ +7% only.

## THE ACTUAL 350@495K PLAY: CONVERT EXPERTS TO MXFP4

mxfp4 B=768: 4.63 ms/layer vs affine8's 8.48 = 1.85x switch_mlp
=> e2e 1/(0.57 + 0.43/1.85) ~ 1.25x => ~380 t/s @495K projected.
Most of the win is the weight-byte halving itself; gather_qmv_rhs adds
1.16x on top (and is already deployed for it). Decode should also gain
(per-token expert reads halve). The recipe exists:
deepseek_v4.make_quantization_config — mxfp4 experts, mxfp8
shared_experts/attn. NEEDS: conversion run + quality gates (needle, BOS,
perplexity/eval battery) — MODEL-QUALITY DECISION, get user sign-off.

## VALIDATION STATE (this session)

- smoke 2K: needle+BOS clean (both deploys).
- 111K real: 349 t/s, needle hit, no BOS spam == old curve (expected
  after correction; kernel ~1.01x at the 768-pair shape).
- 559K real (495K-target run): 304.7 t/s, needle hit, no BOS spam.
  Old curve: 306@495K, 215@727K => ~280 interpolated at 559K. So the
  curve shifted up at depth (~+8%) — consistent with the adaptive
  prefill schedule growing chunks past 256 at deep context into the
  B/E 4-8 window where affine qmv_rhs wins ~1.16x.
- decode (3x1200 tok): 28.89 +/- 0.35 t/s vs 29.0 reference — unchanged
  (B/E < 2 at decode keeps the old path, as designed).
- Deployed: mlx cb539cda on both Studios, exo main 2347fffa, healthy.

## MEASUREMENT PITFALL #5 (add to the list)

5. CHECK THE CHECKPOINT'S ACTUAL QUANTIZATION (config.json) before
   benchmarking "the production shape". A model-file recipe function is
   not the deployed checkpoint. One wrong mode assumption cost a full
   kernel-optimization cycle aimed at the wrong 141 GB/s.

===========================================================================
2026-07-02 SESSION PART 2 — DECODE: ASYNC FENCE +28%; C=2 BUG ISOLATED
===========================================================================

## DECODE WIN (deployed, default ON): EXO_DSV4_FENCE_ASYNC

MTP-PROF (EXO_DSV4_MTP_PROFILE=200) decode cycle budget at c=1:
  total 62.3ms | verify 56.3 (90%) | draft 4.9 (8%) | accept+rb 1.1
  ALLSUM probe: 44 per-layer fence walls x ~1.1ms; ~0.5ms/layer weight
  floor => ~0.3-0.5ms/layer CPU-GPU serialization from the BLOCKING
  mx.eval(y) at the Phase H Lever 1 fence.

Fix: mx.async_eval(y) at the SAME per-layer commit points (dispatch order
preserved; CPU encodes layer n+1 while GPU runs n). This is NOT OPT-7
(which removed evals and paid batched-graph cost).
  c=1 decode: 28.89 -> 37.03 +/- 0.5 t/s (+28%), verify 56.3 -> 42.2ms.
  Outputs BYTE-IDENTICAL to blocking fence. Needle + BOS clean.

Arming (all in code, env only enables the feature):
  two-key side channel in deepseek_v4._FENCE_ASYNC_CTX — "engine" key
  (batch_generate: exactly 1 active task, disarm+mx.synchronize at
  submit/submit_batched entry, re-arm at registration/removal) AND
  "cache" key (dsv4_mtp: single-uid steady, disarm+sync around cache
  merges). Fence async only when BOTH true AND B==1 AND L<=8.
  UNCONDITIONAL async (no arming) corrupts + wedges c=2 within seconds —
  do not simplify this away.

## C=2 DECODE CORRUPTION — PRE-EXISTING BUG, ISOLATED TO MTP VERIFY

Repro: two concurrent decode_probe_ab runs. Signature: repetition /
keyword-list degeneration from the join window ("Okay,Irwin,John,John",
"* * * *" spam), occasionally wedges ranks after repeated joins.
Degenerate streams show INFLATED t/s (repetition -> high MTP acceptance).

Bisect matrix (all on 2026-07-02 stack, 3-round join batteries):
  async fence OFF (control):        CORRUPT  -> fence exonerated
  MLX_GATHER_QMV_RHS=0:             CORRUPT  -> new kernel exonerated
  2026-07-01 knobs reverted
  (LASTROW/TAIL_PMASK/ARGPART/192): CORRUPT  -> yesterday exonerated
  EXO_DSV4_DEGEN_PROBE=1:           extracts consistent, snapshots fresh
                                    -> MTP-cache scramble exonerated
  EXO_DSV4_MTP=0:                   CLEAN (20.5 t/s x2, coherent)
                                    -> BUG IS IN MTP BATCHED VERIFY (B=2)

Age: older than 2026-07-01; last known-good c=2 validation was the
2026-05-22 fence sweep (34.18 agg). Next tools: EXO_DSV4_C2_TRACE=1
per-chain-step JSONL tracer (built for exactly this), then read the B=2
verify/rollback path in dsv4_mtp.py (~lines 1861-2290, batched cycle).

## PRIORITY ORDER FOR NEXT SESSION (revised)

1. Fix c=2 MTP batched-verify corruption (correctness; affects any
   concurrent Hermes usage TODAY, and predates this session).
2. mxfp4 expert conversion (the 350@495K prefill play; needs user
   sign-off on quality + eval battery). Kernel support already deployed.
3. Decode: remaining verify wall is ~42ms (weights ~20ms + collectives);
   next lever after (1) is per-collective latency / jaccl smalls.

===========================================================================
2026-07-02 SESSION PART 3 — C=2 ROOT-CAUSE HUNT: FENCED TO B=2 L>1 ATTENTION
===========================================================================

## FAILING-CELL MAP (each cell direct-tested on-cluster)

              L=1            L=3 (MTP verify)
  B=1         OK             OK (c=1 validated all session)
  B=2         OK (MTP off)   CORRUPT from the FIRST batched cycles

Exonerated by direct A/B (each its own relaunch + c=2 battery):
  temp (probes ran greedy), acceptance divergence (min-acceptance clamp
  deployed — still corrupt from cycle ~2, BEFORE any divergence),
  indexer selection (EXO_DSV4_INDEX_TOPK=999999 — still corrupt),
  seq-split (gated L>=16, verify L=3 never bands), pool-vs-ring drift
  (fixed by the min-acceptance clamp, still corrupt), async fence,
  gather_qmv kernel, 2026-07-01 env knobs, MTP-cache scramble (DEGEN
  probe: extracts consistent).

=> The B=2, L>1 batched VERIFY FORWARD produces wrong logits from the
start: core batched attention — causal mask construction at B=2 L>1
over (PerStream)BatchRotatingKVCache, batched RoPE positions, or the
L>1 ring write at B=2. Matches the 2026-06-17 in-code note
("suspected per-stream verify mask/offset in the L>1 batched forward;
investigation ongoing") at dsv4_mtp.py's residual-sampling branch.

## SHIPPED THIS PART (kept as hardening, not the fix)

exo 67846af5: BS>1 min-acceptance clamp (EXO_DSV4_BS_MIN_ACCEPT=1
default). Repairs a REAL latent inconsistency (per-stream rotating-KV
trims vs batch-uniform pooling/indexer trim + whole-batch restore_meta)
that would corrupt once the forward is fixed. =0 reverts to per-stream
acceptance for A/B.

## NEXT SESSION: THE DECISIVE EXPERIMENT

B=1-vs-B=2 logit differential: instrument the batched cycle to run the
same verify_input rows through two B=1 forwards on cloned caches and
mx.diff the logits (then bisect per-layer if they differ). One cache
clone is GBs but feasible once. Alternatively EXO_DSV4_C2_TRACE=1
(per-chain-step JSONL, built for the bistability hunt) on a corrupting
run. Candidate code: mlx-lm cache.py BatchRotatingKVCache.update at
L>1 B>1 (ring insertion), make_mask with per-stream left_padding at
L>1, rope offset scalar-vs-per-stream in the batched attention.

Repro (30s): two parallel `uv run python bench/decode_probe_ab.py
--iters 1`; corrupt stream heads look like "Okay,Irwin,John,John" /
"* * *" prompt-echo salad. Degenerate streams INFLATE t/s.

===========================================================================
2026-07-02 SESSION PART 4 — C=2 FIXED IN PROD (GATE); REAL BUG = REGRESSION
===========================================================================

## SHIPPED FIX (validated): EXO_DSV4_MTP_C2_MAX_CTX=1 default

start_cluster.sh now arms the existing c>=2 spec gate for ALL contexts:
c=1 keeps MTP (37.45 +/- 0.57 t/s, needle+BOS clean), any c>=2 batch
falls back to non-spec batched decode. Validation: 3 consecutive c=2
join rounds, ALL streams coherent (byte-identical clean openings),
~14 t/s per stream at 1200 tokens. Set =0 to re-enable c>=2 spec after
the real fix.

## KEY DISCOVERY: THE B=2 L>1 CORRUPTION IS A REGRESSION (06-18..07-01)

The 2026-06-18 degeneration handoff (~/.hermes/pastes/paste_1_222627.txt;
umbrella doc dsv4-mtp-batch-degeneration-and-diagnosis-2026-06-17.md was
pruned by the skills curator) records BS=2 VALIDATED CLEAN via
bench/concurrent_bench.py after the fusion removal (exo f0bd448f,
mlx-lm 91c3a95). Today's corruption therefore regressed in
1ece2f27..5619827c (116 commits, mostly the 06-21..06-24 perf wave:
tiled-P indexer, argpartition, direct pooled gather + revert, OPT-9/10,
wide-ring PerStream work, chunk sizing).

## WHAT WAS EXONERATED BY DIFFERENTIAL HARNESSES (single-process, m4-1)

bitwise-equal single-vs-batch: PoolingCache/BatchPoolingCache (ratio 4;
ratio 128 trivially), RotatingKVCache/BatchRotatingKVCache (55-prefill +
6 spec cycles with trims), rope array-vs-int offsets (both thetas, even
unequal), _sparse_pooled_attention (L=1/3, masks on/off), BatchPooling
make_mask verify guard. Whole-model 4-layer random-weight E2E: only bf16
noise (argmax flips on random-logit near-ties — NOT the bug; don't chase
that again). The real bug needs a production ingredient the harness
lacks — most likely the 2-rank sharded execution.

## ALSO SHIPPED (correct-discipline hardening, kept)

- exo 67846af5: BS>1 min-acceptance (uniform commit lengths).
- exo db9a473e: batched pool rollback now mirrors the c=1 flush-gated
  restore + commit-forward discipline (the old unconditional restore
  really did lose committed pool tokens on every rejecting cycle — a
  genuine latent bug, just not THE trigger).

## NEXT SESSION: BISECT THE REGRESSION (bounded, mechanical)

git bisect exo 1ece2f27..5619827c (# ~7 steps), each step:
start_cluster.sh relaunch (~8 min) + the 30-second repro (two parallel
`uv run python bench/decode_probe_ab.py --iters 1`; corrupt = 'Okay,Irwin'
/ '* * *' heads, often with INFLATED t/s). Set
EXO_DSV4_MTP_C2_MAX_CTX=0 during bisect so spec actually runs at c=2.
Prime suspects in the window: d26dc013 (indexer fold weights into q),
fcb5c691/ffe00c8c/0c526dc6 (direct pooled gather + revert), 24059598
(tiled-P indexer), e3eb3499 (argpartition), the wide-ring PerStream
commits (096515f/48a4a3c era). bench/concurrent_bench.py is the original
validation harness if a richer gate is wanted.

---

# PART 5 (2026-07-02, later): c>=2 MTP CORRUPTION — ROOT-CAUSED AND FIXED

## THE BUG (mlx-lm `_bootstrap_per_stream_ring`, cache.py)

Not the batched verify forward at all. The corruption was injected at the
BatchRotatingKVCache → PerStreamBatchRotatingKVCache upgrade (which only
runs when spec dispatch engages at BS>1 — why MTP-off c>=2 was clean and
why every single-process differential passed). Two defects:

1. **Low-context prompt-KV wipe.** The bootstrap sliced the newest
   `valid` tokens as `keys[..., -valid:, :]` from the FULL base buffer.
   `_update_in_place` grows the buffer in `step`(256)-sized ZERO chunks,
   so at low context the buffer is wider than the written region and the
   slice reads the zero tail — the entire prompt KV of every stream
   replaced with zeros. Model continues from an empty local window →
   deterministic off-topic gibberish from ~token 3 ("Okay,Irwin…",
   "Okay,Irregular verbs…"). At high context the buffer is exactly
   max_size wide, so the 200K–500K needle validations passed — the bug
   only fires at SHORT context. (All our c=2 probes were 51-token
   prompts; all long-context validation was blind to it.)

2. **Join offset stamp.** The bootstrap assumed uniform streams
   ("no divergence has happened yet"), broadcast `offset[0]` to every
   row and zeroed `left_padding`. At a decode-time join (veteran ~264,
   newcomer ~74) the newcomer inherited the veteran's logical position
   (observed `[264,264]` in the spec trace) → wrong RoPE/mask → instant
   ' the.'-loop degeneration (killed by the kill-switch, then GPU
   timeout wedged the cluster).

The "regression window 06-18..07-01" was a red herring resolved: exo
7debac7f (06-24) removed the `is_bench` gate on batched prefill, exposing
/v1 traffic to the BS>1 spec path. Trigger, not cause.

## THE FIX (mlx-lm 8b7b5f9, exo 1ab31258)

Rebuild the per-stream ring from the base-class invariant (stream b's
newest real row = logical `offset[b]-1`): per stream, take the newest
`min(offset[b], max_size, real_w)` rows of the WRITTEN region
`[0:real_w)` (`real_w = _idx`, not buffer width) and scatter at
`pos % ring_width`; keep the true per-stream offset vector; zero
left_padding only because the modular ring layout genuinely has none.
Unit-verified bit-exact: uniform low-ctx pair, unequal-offset join,
rotated high-ctx (the 48a4a3c case still passes).

## VALIDATION (deployed, gate=0)

- Simultaneous pair (rendezvous batched prefill): CLEAN — both streams
  exact greedy text, 18.0 t/s/stream (was 8.4 corrupt).
- Staggered join (6s): CLEAN both; joiner coherent from token 1,
  21–22 t/s (was instant degeneration + cluster crash).
- c=1 regression: 34.2 t/s (baseline 34.4), coherent.
- B=2 acceptance: 0.80/stream vs 0.84 at B=1 (was pinned 0 — the
  min-clamp was dragging every batch to 1 tok/cycle off the corrupt
  streams' rejections).
- start_cluster.sh default flipped: EXO_DSV4_MTP_C2_MAX_CTX=0 (spec ON
  at c>=2 is production now).

## HOW IT WAS FOUND (method note)

EXO_DSV4_SPEC_TRACE=1 committed-token streams vs a c=1 ground-truth run:
corruption predated the first spec cycle → not the verify. The per-cache
offsets record showed `[264,264]` for a true `[264,74]` join → main
rotating cache, upgrade path. Two shell-level red herrings burned time:
a stale background watcher double-running the repro (made 2 requests
look like 4 phantom streams), and m4-1's clock ~4.5min ahead of the
laptop (made test requests look duplicated). Verify wall-clock skew
before reading multi-host logs.

## OPS NOTES (current state)

- m4-1 WAN TCP is BROKEN since its reboot: ICMP/DNS fine, all outbound
  TCP data stalls after handshake (ClientHello retransmits, no ACK).
  Not ECN, not TSO, not MTU, not pf, not dual-interface (Wi-Fi now
  off). Workaround in place: laptop pushes to bare mirrors in
  ~/gitmirror/{exo,mlx-lm}.git on m4-1 + global insteadOf rewrite
  git@github.com:adurham/ → /Users/adam.durham/gitmirror/. Deploys must
  push mirrors first: `git push ssh://macstudio-m4-1/~/gitmirror/exo.git
  main:main` (and mlx-lm HEAD:refs/heads/main). Router-side issue —
  needs user attention (possibly Google Wifi device state for .201).
- m4-2 exo process died earlier on a pydantic validation error when an
  event carried the GPU-timeout string (extra_forbidden) — robustness
  bug, unfixed, incidental.
- trim_per_stream sets `_idx = max % max_size` while update_and_fetch
  uses `% ring_width` — _idx is unused by PerStream paths, cosmetic.

---

# PART 6 (2026-07-02): c=2 DECODE LEVERS — +36%/stream (15.0 → 20.4 t/s)

Post-root-cause-fix A/B matrix (divergent-prompt staggered pair, 800 tok,
per-stream t/s; twin-prompt repros can't distinguish clamp settings —
identical streams accept identically, the clamp is a no-op there):

| config                              | t/s/stream | quality |
|-------------------------------------|-----------|---------|
| min-clamp ON + sync fence (old)     | 15.0      | clean   |
| per-stream acceptance (clamp off)   | 18.7      | clean   |
| async fence at B<=2 (clamp on)      | 17.2      | clean   |
| BOTH (new production default)       | 20.4      | clean   |

c=1 unchanged (34.2); twin-prompt repro CLEAN at ~21 t/s/stream in all.
Per-stream acceptance verified genuinely exercised: 65% of B=2 cycles had
unequal n_accepted (means 0.68/0.80).

- `EXO_DSV4_BS_MIN_ACCEPT=0` (now default): the clamp's "known-corrupt"
  rationale was contaminated by the bootstrap bug; per-stream rollback
  (`trim_per_stream`) handles divergence correctly post-fix.
- `EXO_DSV4_FENCE_ASYNC_C2=2` (now default, new env): extends the async
  decode fence arming to <=2 streams (engine + cache keys and the model-
  side B-gate all read it; exo b76b6a3e + mlx-lm submodule). c>2 arming
  untested — raise only with a validation pass.

## NEW BUG FOUND + FIXED: ragged rendezvous batched MTP prefill crash

Two /v1 requests with DIFFERENT prompt lengths inside the 200ms rendezvous
window crashed the runner: the batched prefill right-pads to max_L and the
captured last-chunk pre_norm is uniform, but the per-stream MTP cache
prefill paired it with each stream's unpadded token tail →
[broadcast_shapes] (1,21,4096) vs (1,72,4096) at dsv4_mtp.predict. Fixed in
batch_generate._submit_batched_eligible: slice both sides to the stream's
true rows (k_i = len_i - (max_L - S_pre)); <2 true rows → skip prefill
(cold draft cache, no crash). Equal-length behavior bit-identical.

## REMAINING VALIDATION GAPS

- Long-context c=2 CLOSED (2026-07-02): 2x~105K simultaneous divergent
  streams (rendezvous batched prefill at depth), needles 1/1 both, no BOS
  spam, 24.1/23.8 t/s/stream concurrent decode.
- c=3..5 concurrency battery under per-stream acceptance (user: not a
  priority).
- m4-2 exo pydantic extra_forbidden crash on GPU-timeout event strings
  (robustness, open).

## PART 6 ADDENDUM: B=1/B=2 cycle profile + the "rank0 rollback" red herring

MTP_PROFILE breakdown (ms/cycle, trace OFF): B=1 draft 4.7 / verify 42.2 /
accept 0.9 / rollback 0.2 / total 48.0. B=2: 5.6 / 75.6 / 0.5 / 0.2 / 81.9.

- Verify scales ~11ms/row (B=1 = 3 rows, B=2 = 6): each row routes to its
  own top-8 of 256 experts (~8% overlap at 6 rows) → per-row expert-weight
  bandwidth is intrinsic MoE physics. M-batched kernels can't help at
  decode shapes (need >=2 rows/expert). Same physics that killed gamma=3;
  tree drafting predicted to LOSE (each branch = +11ms row vs sub-row
  marginal acceptance) — skipped on that basis.
- EXO_DSV4_SPEC_TRACE costs ~10% end-to-end: 5.3ms/cycle of rank0-only
  .tolist()/str(offsets) syncs inside the rollback window, ON the critical
  path (rank totals 53.3 vs 48.0). Never benchmark with it on. All Part 6
  A/B numbers had it on — consistent relatively, ~10% low absolutely.

FINAL PRODUCTION NUMBERS (trace off, 2026-07-02): c=1 37.4/36.6 t/s;
c=2 divergent pair 24.4/24.6 t/s/stream (~49 aggregate); long-context c=2
(2x105K) needles 1/1, 24 t/s/stream. Session start: c=1 28.9, c=2 corrupt.

Remaining decode headroom beyond this: mxfp4 experts (halves per-row
expert bytes; quality-gated, user wants understanding first). Prefill:
gather_qmv_rhs bucket extension (B/E>8 tile-over-N variant).

## PART 7 (2026-07-02): gather_qmv_rhs bucket extension — RESOLVED NEGATIVE, gate bug fixed

Full B/E x M_TILE sweep (M4 Max, DSv4 shapes, affine8 g64 + mxfp4 g32,
MLX_GATHER_QMV_RHS_MAXBE lift): NO extension bucket exists. Larger M_TILE
makes long runs WORSE (mt=8 at B/E=16 is 0.17x — register pressure beats
restream savings), and steel's tile reuse owns everything past the
crossover. Measured crossover (TILE=4): B/E=6 wins (1.16x affine8 / 1.48x
mxfp4), B/E=7 parity, B/E=8 LOSES (0.63x / 0.86x), 12-32 lose more.

The useful find: the shipped B/E<=8 gate included the (6,8] regression
zone (only 3 and 6 were ever measured) — multi-stream batched prefill can
land there. Fixed: default bound 6 (mlx 0362d105), exo uv.lock pin bumped.
The mxfp4 crossover is also 6-7, so the bound holds for the mxfp4 play.
Raising EXO_PREFILL_STEP_SIZE for bigger B/E is a dead idea too: steel at
B/E=12 runs 206 GB/s vs 240-300 at smaller shapes.

## PART 8 (2026-07-02, evening): THE CHECKPOINT WAS MIXED ALL ALONG

Verified by reading safetensors shard headers (config.json's quantization
section is WRONG — claims uniform affine8 g64, no overrides):
- switch_mlp experts: **mxfp4 g32** (U32 packed 8/word, U8 E8M0 scales, no
  biases) — ~137e9 of the 155e9 bytes.
- attention / shared_experts / embed / lm_head: affine 8-bit g64 (~18e9 B).
- Model is ~284B params, not ~146B.

Fallout:
1. The mxfp4-conversion play (350@495K + decode-halving projections) is
   ALREADY REALIZED in prod. Dead. The part-1 "premise correction" of this
   handoff overcorrected — the original mxfp4 premise was right for the
   experts, which is what the kernels touch.
2. mlx-community/DeepSeek-V4-Flash-mxfp4 (uniform 4-bit) would only push
   the 8-bit remainder to 4-bit: ~9GB/node fewer once-per-forward reads,
   quality risk on attention/lm_head. Not worth an A/B. No download.
3. The B=1→B=2 verify scaling (~11ms/row) CANNOT be per-row expert reads
   (top-6/row ≈ 40MB/node ≈ 0.2ms at 200GB/s). The "MoE physics" verdict
   in part 6-addendum is RETRACTED. The row cost is elsewhere: batched
   sparse-attention/indexer path, kernel dispatch, or collectives. Decode
   B>=2 headroom is REOPENED (and tree-drafting is un-killed pending a
   real intra-verify profile).
4. gather_qmv_rhs B/E<=6 bound holds regardless (mxfp4 crossover also 6-7).
   Prefill expert gathers use the FP kernel family; the affine variant
   built this morning serves no prod tensor today (harmless, kept).

NEXT: intra-verify profile at B=1 vs B=2 (per-section: MLA attn, indexer,
pools, switch_mlp, collectives) to find the true +11ms/row owner.

## PRODUCTION BASELINE TABLE (2026-07-02, c=1, post-everything)

| ctx | prefill t/s | decode t/s | needle |
|-----|------------|------------|--------|
| 100K | 354.9 | 42.3 | OK |
| 200K | 345.4 | 33.9 | OK |
| 300K | 334.6 | 32.8 | OK |
| 400K | 322.5 | 29.2 | OK |
| 500K | 311.4 | 23.2 | OK |

(mtp_longctx_probe, exact server-side token counts, single stream, no
prefix cache, spam=no everywhere. Reference for any future A/B.)

## PART 9 (2026-07-02, late): B=2 verify row-cost hunt — partial attribution

Found + shipped: **MLX fused SDPA cliff at B>1 AND L>1** (the batched
verify shape). Sweep (M4 Max, heads=64/kv=1/dk=512): B=2 L=3 costs 2.9x
two B=1 calls at ctx512, 4.4x at 1024, 9.0x at 4096; B>1 L=1 and B=1 L>1
are both fine. Fix: mlx-lm base.py row-splits into B fused B=1 calls
(2<=B<=8, 1<L<=8, MLX_LM_SDPA_ROWSPLIT=0 kill switch). allclose, not
bitexact.

BUT production DSv4 barely hits it: main attention kv = ~136-slot local
ring (below cliff onset); sparse-pooled L>1 path uses the hand-rolled
split-softmax inner kernel, not fused SDPA. Result: B=2 verify 75.5 ->
69.7ms, e2e c=2 flat (~24 t/s/stream). Kept anyway (right for other
shapes/models, and for any future kv-length growth).

REMAINING: B=2 verify still ~69.7ms vs B=1 42.2ms (~27.5ms/cycle,
~640us/layer). Microbench-sized suspects each account for a fraction
(dense qmms get FASTER at B=2; expert gather +155us; ring-sized sdpa
+~100us; collectives 2x bytes but latency-bound). Attribution needs
per-kernel GPU traces: mx.metal.start_capture around ONE B=1 and ONE
B=2 verify, diff kernel-by-kernel in Xcode. Span profiling is USELESS
here (lazy eval scrambles attribution — verified empirically, don't
retry). Op microbenches mislead on absolute cost (launch overhead
pipelines away in the real graph) — only use them for RATIOS.

Session-close production state: c=1 37.4 t/s, c=2 24.5/stream divergent
(~49 aggregate), depth table in this doc, all quality gates green.
Deployed with EXO_DSV4_MTP_PROFILE=100 (negligible; drop on next deploy
if desired).

---

# NEW SESSION: START HERE (state as of 2026-07-02 session end)

## Current production state
- Cluster: 2x Mac Studio M4 (m4-1 .201 rank0 API, m4-2 .202), TB RDMA.
  Deploy: push to adurham/exo main FIRST, then `EXO_TARGET_BRANCH=main
  ./start_cluster.sh` from the laptop. Never push mid-deploy.
- Model: mlx-community/DeepSeek-V4-Flash, ~284B, **MIXED quant: experts
  mxfp4 g32, attention/shared/embed/lm_head affine8 g64**. config.json's
  quantization header LIES (says uniform affine8) — always read
  safetensors shard headers (part 8 has the how).
- Perf: c=1 37.4 t/s decode; c=2 24.5 t/s/stream divergent pair (~49
  aggregate); prefill/decode by depth in the PRODUCTION BASELINE TABLE
  above (355->311 prefill, 42->23 decode, 100K->500K, needles all OK).
- Defaults now in start_cluster.sh: EXO_DSV4_FENCE_ASYNC=1,
  EXO_DSV4_FENCE_ASYNC_C2=2, EXO_DSV4_BS_MIN_ACCEPT=0,
  EXO_DSV4_MTP_C2_MAX_CTX=0 (spec ON at c>=2 — the corruption is
  root-fixed, part 5). mlx pin: gather_qmv_rhs B/E<=6 (part 7).
  mlx-lm: SDPA row-split at B>1/L>1 (part 9, MLX_LM_SDPA_ROWSPLIT=0).

## Open work, in priority order
1. B=2 verify residual: ~640us/layer excess over B=1 (69.7 vs 42.2ms
   verify). Attribution plan: mx.metal.start_capture around ONE B=1 and
   ONE B=2 verify cycle, per-kernel diff in Xcode. Do NOT use span
   profiling (lazy eval scrambles it) or trust microbench absolute
   times (ratios only). Potential: c=2 -> ~30 t/s/stream.
2. Prefill re-profiling at depth (indexer/pool shares at 300-500K) —
   untouched since the +8% kernel win.
3. Reliability: m4-2 exo crashes on pydantic extra_forbidden when an
   event carries GPU-timeout strings; m4-1 WAN TCP broke once after
   reboot (fixed by another reboot; LAN git-mirror workaround procedure
   in part 5 ops notes if it recurs).
4. Tabled: mxfp4 (already shipped in the checkpoint — part 8); tree
   drafting (needs #1's data first); c>2 fence arming (validate before
   raising EXO_DSV4_FENCE_ASYNC_C2).

## Tooling that exists (don't rebuild)
- 30s c=2 quality repro: /tmp/c2_repro_check.sh (laptop).
- Decode probe: /tmp/decode_probe_ab.py; nonce/divergent variant in the
  session scratchpad pattern (append --suffix to defeat prefix cache;
  IDENTICAL twin prompts cannot A/B acceptance policies — use divergent
  same-length suffixes, staggered 6s to dodge rendezvous).
- Depth stress: bench/mtp_longctx_probe.py (target-tokens ~1.9x actual).
- Per-cycle phase timer: EXO_DSV4_MTP_PROFILE=100 (B-sliced draft/verify/
  accept/rollback). EXO_DSV4_SPEC_TRACE costs 10% on rank0 — never
  benchmark with it on.
- Kernel benches on m4-1: ~/scratch/tilesweep-venv + ~/scratch/mlx-tilesweep
  (adurham/mlx main), qmv_mbatch_bench.py (QMV_MODE/BITS/GROUP/PAIRS +
  MLX_GATHER_QMV_RHS_MAXBE), rowscale_bench.py, sdpa_sweep.py,
  rowsplit_check.py.
- Memory files (auto-recall): exo_dsv4_quantization, exo_dsv4_c2_mtp_corruption,
  exo_dsv4_decode_async_fence, exo_gather_qmv_rhs_kernel.

===========================================================================
2026-07-03 SESSION — c=2 DEEP-DEGEN FOUND; PYDANTIC CASCADE FIXED; B=2
RESIDUAL REPRODUCED LOCALLY
===========================================================================

## NEW BUG: c=2 DEEP-GENERATION DEGENERATION (OPEN, under A/B)

4000-token divergent c=2 pairs degenerate stochastically: 3/10 pairs on
the row-split-ON stack had a stream collapse into a repetition loop
(3 of 4 corrupt streams locked onto markdown-star ' *'/' **', ids
982/2619; onset tokens 1840/2106/3782/3905). c=1 is 6/6 clean at the
same depths. EXONERATED by direct A/B: xctrace attach, preceded-by-c1
session, tiebreak fix (correctly OFF since 06-09 — re-enabling would
corrupt this flat-logit checkpoint, see 6e978650). Part 6's 800-token
validation horizon was blind to this. Detection: server-side
DEGENERATION DETECTED kill-switch (action=error), degen streams end
without usage frame.

Current experiment: 6-pair battery with MLX_LM_SDPA_ROWSPLIT=0 (part 9's
row-split shipped AFTER the last deep c=2 quality validation and is the
prime suspect; passthrough added to start_cluster.sh, commit 25f61362).
Battery scripts: scratchpad c2_battery2.sh (readiness probe uses a
full-length prompt — a tiny 'hi' probe crashes the runner, see below).

## FIXED + DEPLOYED: THE CLUSTER-KILLING EVENT CASCADE (exo c0dc8456)

The 06-30 "m4-2 pydantic extra_forbidden" open bug root-caused: corrupt
stream → degen kill → GPU timeout → runner death → supervisor emits
ErrorChunk carrying RunnerMetalGpuTimeout diagnostics → EVERY receiving
node's router died on unhandled ValidationError → cluster 503 until
relaunch. Mechanism: TaggedModel's wrap-validator re-validates parsed
JSON in PYTHON mode, where strict tuple[str,...] (evidence field)
rejects the JSON array (a list). Fix: Field(strict=False) on evidence +
round-trip regression test (test_event_serialization.py). Validated in
anger: a later corrupt iteration no longer took the cluster down.
Wire-model rule: NO strict tuple fields in TaggedModels.

## NEW BUG: PREFIX-HIT PREFILL CRASH (OPEN, task: repro + fix)

max_tokens=1 'hi' request, then a ~69-token prompt sharing the chat
header → runner dies: [broadcast_shapes] (59,62) vs (1,64,59,186) in
fast SDPA, MTP draft block (sinks path). 62 = 59 new rows + offset 3
(readycheck's draft bookkeeping); 186 = 127 STALE draft-ring rows from
the PREVIOUS session + 59. The part-6 "<2 true rows → skip prefill
(cold draft cache)" path leaves the draft cache STALE, not cold.
_clamp_mask_to_kv only handles mask-wider-than-kv, not narrower.

## B=2 VERIFY RESIDUAL: REPRODUCED SINGLE-PROCESS (no collectives)

~/scratch/dsv4_bverify_perf.py (m4-1): 4-layer random-weight DSv4,
production quant (experts mxfp4 g32, rest affine8 g64), depth-1500
chunked prefill, PerStream upgrade at B=2, verify(B,3)+trim cycles.
  B=1: 11.48 ms/cycle   B=2: 14.20 ms/cycle  →  +680us/layer
Matches prod's +640us/layer/node — WITH FULL-WIDTH layers and no
collectives/MTP machinery. Width-independence suggests a fixed
per-layer cost (dispatch/serialization), not bandwidth. Also: prod
GPU-busy analysis (metal-gpu-execution-points, (1,X)=begin/(2,X)=end
per cmdbuf) shows B=2 excess is GPU EXECUTION, not fence idle (busy
83.5% c=1 vs 81.1% c=2).

## NEW TOOLING: XCODE-FREE PER-KERNEL GPU ATTRIBUTION

- mlx debug build (MLX_METAL_DEBUG=ON + JIT) in m4-1 tilesweep venv
  (mlx-0.32.0.dev+0362d105): labels every command buffer with primitive
  names; verified the labels flow into xctrace exports.
  Build: PATH needs ~/scratch/tilesweep-venv/bin (cmake) via ssh.
- Import into exo venv via PYTHONPATH=$HOME/scratch/tilesweep-venv/lib/
  python3.13/site-packages (same mlx rev as prod pin, debug-compiled).
- Pipeline: MLX_MAX_OPS_PER_BUFFER=1 + xctrace Metal System Trace,
  export metal-application-encoders-list (cmdbuf-id → label) +
  metal-gpu-execution-points (GPU begin/end per cmdbuf), join →
  named GPU intervals. Aggregator: scratchpad gpu_bylabel.py.
- PITFALL: xctrace --attach against short-lived harness processes hangs
  and writes empty stub traces; use --launch (validated) or attach only
  to long-lived processes (prod runner captures worked). Also pgrep -f
  matches watcher shells whose cmdline contains the pattern — attach to
  $! of the launched python, never pgrep.
- PITFALL: metal-shader-profiler-intervals is EMPTY in both attach and
  launch modes — per-shader profiling isn't available headless; the
  cmdbuf-label join is the workable granularity.

## OPS NOTES

- Both Studios rebooted 2026-07-03 (user-initiated; cleared xctrace
  zombies + paged-out weights). m4-1 WAN survived this reboot.
- Battery readiness probes MUST use full-length prompts (see prefix-hit
  crash above).
- exo pre-existing test failures (NOT from this session): pytest
  collection error in master/tests/test_routing_concurrency.py (stale
  import), failure in worker/tests/.../test_event_ordering.py; tests/
  conftest needs missing exo_tools module.

## PRIORITY ORDER (revised)

1. c=2 deep-degen: finish ROWSPLIT=0 battery → implicate/exonerate;
   if exonerated, next arms: EXO_DSV4_FENCE_ASYNC_C2=0, per-stream
   acceptance off, then spec-trace instrumented repro.
2. B=2 verify residual: labeled B=1/B=2 captures (launch mode) + 
   gpu_bylabel diff — harness already reproduces the excess.
3. Prefix-hit prefill crash: repro (30s, kills runner), then fix stale
   draft-ring reset; also fixes battery-harness fragility.

## 2026-07-03 (cont): ROWSPLIT EXONERATED; WEDGE ANATOMY; MTP-OFF ARM

- ROWSPLIT=0 battery: 4 clean, then iter 5 CORRUPT (uid=14, ' his' id 793,
  period-1 at token 1403). Rate ~ matches ROWSPLIT=1 (3/10). Row-split is
  NOT the corruption cause. Corrupt-token attractors so far: ' *' 982,
  ' **' 2619 (x2), ' his' 793 — varies, always period-1 or -8 loops,
  onset 1403-3905.
- WEDGE ANATOMY (second failure layer, reproduced + sampled): after the
  degen kill mid-batch, rank0's runner spins at 100% CPU inside an mlx
  collective (mlx core frames; sample in m4-1:~/scratch/wedge_sample_0044
  .txt) waiting on a peer that never comes — rank desync B=2→B=1 on the
  kill path. SIGTERM does NOT kill it (signal never reaches the
  interpreter inside the native spin) — use kill -9. exo then self-heals
  (worker respawns runner, JIT re-places, ~5 min).
- Readiness-probe pitfall: curl -w %{http_code} returns 200 when HEADERS
  arrive; against a wedged cluster the body then hangs. Battery
  wait_ready must require a body token, not a 200.
- pgrep -f "<pid>" matches command LINES, not pids — cost one bogus
  "runner killed" verification. Use ps -p.
- Current arm: EXO_DSV4_MTP=0 (prod defaults otherwise, row-split back
  to default ON), 4x 4000-tok c2only pairs. Discriminates: corruption in
  core batched decode (B=2 L=1) vs the spec verify/accept path. If spec
  implicated → port REFCHECK (exists only in c=1 _speculative_next,
  dsv4_mtp.py:3520) into _speculative_next_batch (1836): batched L=1
  reference forward (trim-all-streams-by-1 + refeed batch), per-stream
  argmax compare, trigger on rank-canonical values only (TP safety).

## 2026-07-03 (overnight): DEEP-DEGEN ROOT ARM FOUND — ASYNC FENCE AT C=2

Full A/B matrix (4000-tok divergent c2only pairs, degen kill-switch as
the detector):
  FENCE_ASYNC_C2=2 arms (any rowsplit/accept combo):  5 corrupt / 19
  FENCE_ASYNC_C2=0 (sync fence at c=2):               0 / 12  (~98% conf)
  EXO_DSV4_MTP=0 (fence never arms):                  0 / 4
  BS_MIN_ACCEPT=1 (async fence still on):             1 / 4  -> accept
                                                      policy exonerated
Shipped: start_cluster.sh default EXO_DSV4_FENCE_ASYNC_C2=0 (f028dc9c).
c=2 falls ~24.5 -> ~19 t/s/stream; c=1 keeps +28% (B==1-gated arming
validated deep all session). The corrupt+wedge pattern is one bug:
async-deferred graph races the batched cycle's cache mutations →
sometimes wrong logits (repetition attractor), and the subsequent
mid-batch degen kill rank-desyncs a collective (100% CPU spin, SIGTERM-
immune, kill -9 + self-heal).

SUSPECTED RACE (for the real fix): the "cache"/"engine" arming keys
disarm+synchronize around BS-transitions (reset/snapshot/activate,
dsv4_mtp.py:588-716) but NOT around the steady-state batched cycle's
per-stream ring trims / pool flush-restores — at B=2 those mutate cache
buffers every rejection cycle while a deferred async graph may still be
in flight. c=1 is safe because its cycle fully drains (single stream, no
per-stream trims). Fix sketch: disarm+mx.synchronize around
trim_per_stream/pool-restore in _speculative_next_batch (cost: bounded,
those are rare-ish sync points) then re-validate 12+ pairs deep.

Battery tooling: c2_battery3.sh (scratchpad) is hands-free — watchdog
kills probes at 480s, SIGKILLs >50%-CPU spinners on both nodes, exo
self-heals (~5 min), battery continues. wait_ready requires a body
token (a bare 200 lies — headers arrive from a wedged cluster).

## 2026-07-03: B=2 VERIFY RESIDUAL — ATTRIBUTED (task closed)

Labeled per-primitive GPU diff (4-layer harness, MLX_METAL_DEBUG build,
OPS_PER_BUFFER=1, positional queue-filtered join; scratchpad
gpu_bylabel3.py; validation: 0 encode-after-begin violations, 0.2% skew):

  B=1: 10.2 ms GPU-busy/cycle   B=2: 13.6   excess +3.4 ms (~860us/layer)

Where it goes: NOT the heavy kernels. QuantizedMatmul gets FASTER
per-dispatch at B=2 (78.7->52.9us); GatherQMM/Gather/RMSNorm scale ~2x
for 2x rows (bandwidth-proper). The excess is the LONG TAIL: ~490 tiny
dispatches/cycle (Sum alone ~220/cycle at 8-14us) whose per-op cost
floor rises with rows, plus inter-op serialization (busy AND idle gaps
both grow). An apparent 10x cliff on CompiledSigmoidMultiplyMultiply
(20->194us) did NOT reproduce in an isolated shape-exact microbench
(flat ~100us rows 1-12) — it is dependency-stall time attributed to the
dispatch at OPS=1, i.e. serialization, not a kernel bug.

CONCLUSION: the B=2 decode-shape verify graph is LAUNCH/SERIALIZATION-
BOUND. The production lever is op-count reduction / fusion in the
per-layer decode graph (compile broader regions; fold the MoE routing
Sum-chain — ~55 Sum dispatches per layer per cycle), not kernel tuning.
This also explains the width-independence of the +640us/layer/node and
why every part-9 microbench only accounted for a fraction.

Tooling shipped: gpu_bylabel3.py (join exec-points GPU intervals to
MLX_METAL_DEBUG cmdbuf labels POSITIONALLY — the two id columns are
DIFFERENT counters, never value-join them; filter exec rows to the
dominant queue id first; validate with encode<=gpu-begin and count
skew).

## 2026-07-03 (late): PLOT TWIST — SYNC-FENCE PAIR CORRUPTED AFTER A TINY REQUEST

A 13th sync-fence c=2 pair DEGENERATED (uid=40, ' his' id-793 again,
token 656 — much earlier than the 1400-3900 fence-era onsets). Unique
antecedent: a genuine ~10-token-prompt max_tokens=2 'hi' request (from
the prefix-crash repro) ran right before it. The 12 clean sync-fence
pairs all used FULL-LENGTH readiness prompts. Hypothesis: the task-3
stale-draft-ring state (tiny/skipped MTP prefill leaves the previous
session's draft KV with fresh offset bookkeeping) is an INDEPENDENT
corruption trigger — the crash ((59,62) vs 186) is its loud variant,
the ' his'-loop degeneration its quiet variant. The async fence may be
an AMPLIFIER (raises background rate via cache-state races) rather than
the sole root. Discriminator running: 3x ['hi' max_tokens=2 -> c=2 pair
800tok] on the sync-fence stack (scratchpad hi_trigger_test.sh).

Suspect code (from the crash arithmetic): MTP snapshot/reset lifecycle
in dsv4_mtp.py — snapshot_for_uid stores a REFERENCE to the live cache
(line ~635); reset_cache replaces self._cache but merge/extract paths
may alias buffers; a session with <2 true rows skips MTP prefill
"(cold draft cache)" (batch_generate.py:~1749 area, part-6 fix) leaving
STALE ring contents with reset offsets.

## 2026-07-03 (pre-dawn): PREFIX-CACHE RESTORE ROOT-CAUSED + FIXED (deploy pending)

THE MECHANISM (deterministic crash repro'd twice, unifies the crash and
plausibly the quiet degeneration):
- snapshot_ssm_states stores None for TRIMMABLE CacheLists. DSv4 layers
  are trimmable exactly when the PoolingCache is empty — i.e. SHORT
  sessions (<~128 tokens: a 'hi' readiness ping) and early-depth
  snapshots of long sessions.
- _materialize_cache_to_depth's fallback for a None state silently
  deepcopied the donor leaf's FINAL cache: rotating ring at final
  offset/contents while sliceable layers + restore_pos sat at snapshot
  depth. Next prefill: [broadcast_shapes] (63,66) vs (1,64,63,190)
  (mask = restore_pos+L, kv = stale-ring+L) — or, when shapes happened
  to broadcast, silent attention over the donor's stale ring.
- Trigger chain: tiny request seeds a short trie leaf -> next
  prefix-sharing prompt partial-hits it -> poisoned restore. Repro:
  round 1 of hi_trigger_test.sh crashed stream A's prefill exactly so.

FIX (exo e9609516, committed + pushed to m4-1 LAN mirror; GITHUB PUSH
PENDING — 1Password ssh agent locked): strict_snapshot on the restore
path — a non-sliceable layer without a snapshot state turns the lookup
into a full-prefill MISS (trace: "STRICT-MISS"). Never substitutes
final-state caches. Regression tests: TestStrictSnapshotRestore (miss
on None states; happy path still restores at snapshot offset).

## THE MORNING EXPERIMENT (decides the fence question)

Deploy e9609516, then re-run the deep c=2 battery WITH
EXO_DSV4_FENCE_ASYNC_C2=2 (c2_battery3.sh 12). Two outcomes:
- Clean: the async-fence correlation (5/19 vs 0/12) was the prefix
  poisoning wearing a fence costume (or fence-amplified); re-enable
  FENCE_ASYNC_C2=2 -> c=2 back to ~24.5 t/s/stream (+25%).
- Corrupt: fence race is real and independent; keep FENCE_ASYNC_C2=0,
  pursue the batched-cycle disarm fix.
Caveat kept in view: all fence-battery arms had identical trie traffic,
so 5/19 vs 0/12 is hard to explain by trie state alone (~2% chance) —
plausibly BOTH bugs are real.

## SESSION END STATE (2026-07-03 ~04:30)

- Cluster: healthy, 2 runners READY on f028dc9c (fence flip deployed;
  prefix fix NOT yet deployed). c=1 39.6 t/s validated.
- Commits: c0dc8456 (pydantic wire fix), 25f61362 (ROWSPLIT passthru),
  f028dc9c (FENCE_ASYNC_C2=0 default), e9609516 (prefix-cache strict
  restore — needs github push + relaunch).
- Open: task-3 adjacent — MTP draft-cache lifecycle across sessions was
  NOT implicated in the end (the crash lives in main-model prefill via
  prefix restore); the (59,62)-style crash is fully explained. The
  wedge (rank-desync collective spin on mid-batch kill) remains a real
  robustness hole — kill -9 + self-heal works, a clean B-transition on
  stream error is the proper fix.
- Batteries/tooling: c2_battery3.sh (hands-free), hi_trigger_test.sh
  (crash repro), dsv4_bverify_perf.py + gpu_bylabel3.py (labeled GPU
  attribution), all in session scratchpad; copies of harness+aggregator
  on m4-1:~/scratch{,/gpucap}.

## 2026-07-03 (morning): DEPLOYED + FENCE QUESTION SETTLED

- e9609516 (prefix-cache strict restore) pushed to github + deployed.
- Crash-repro validation on the fixed stack: 3/3 hi-trigger rounds
  CLEAN (previously: deterministic runner crash on round 1).
- FENCE RETEST (async fence at c=2 + prefix fix): 2 CORRUPT / 3 pairs
  (uid=1 ' **' @2721, uid=8 ' *' @2469 — the deep-onset signature).
  VERDICT: the c=2 async-fence race is REAL and independent of the
  prefix-cache bug. TWO distinct corruption bugs existed:
    prefix-restore poisoning — early onset (~656) + crash variant, FIXED;
    async-fence race       — deep onset (1400-3900), MITIGATED by
                             FENCE_ASYNC_C2=0 default (f028dc9c).
- Production restored to committed defaults and validated: c=1 42.5 t/s,
  clean. c=2 runs ~19-21 t/s/stream on the sync fence.
- Real fence fix remains open: disarm+mx.synchronize around the batched
  cycle's steady-state cache mutations (trim_per_stream / pool
  flush-restore) in _speculative_next_batch, then a 12-pair deep
  battery to re-earn FENCE_ASYNC_C2=2 (+25% c=2).

## 2026-07-03 (day): ASYNC-FENCE ROOT FIX SHIPPED; WEDGE ROOT-CAUSED (fix designed, not shipped)

- FENCE ROOT FIX (exo d4a50cf2 + 6978f56f): one mx.synchronize per
  batched cycle (via self.mtp._set_fence_async(False)) BEFORE the
  rollback's cache mutations, re-armed with the standard B-gate after
  the MTP-cache trim. Mechanism: mx.eval(verify_logits) only waits for
  the logits' dependency chain; pool/indexer side-chain writes consumed
  by FUTURE forwards were still in flight when trim_per_stream /
  restore_meta mutated the same buffers. B=1 immune (offset-decrement
  trims). First deploy crashed: _set_fence_async is on DSv4MTPPredictor
  (self.mtp), NOT the DSv4MTPBatchGenerator hosting the batched cycle.
  Validation battery (async fence armed, 12 pairs) in flight.

- WEDGE ROOT CAUSE (found, NOT fixed): the degen kill-switch eviction
  in batch_generate.step() (~line 2465) calls self._mlx_gen.remove()
  DIRECTLY on rank0 only. Degen detection lives in the detokenization
  loop that only rank0 runs; rank1 never learns → rank0 at B=1, rank1
  at B=2 → collective mismatch → 100% CPU jaccl spin (SIGTERM-immune).
  Natural completions are safe (token-level, rank-deterministic on the
  broadcast committed stream); client cancels are safe
  (agree_on_cancellations). ANY rank0-only termination (degen kill,
  string-level stop sequences?) has this hazard at c>=2.
  FIX DESIGN: an agree-on-evictions step at the batched-cycle boundary
  using the agree_on_cancellations_fast pattern (mx_any fast path on
  the coord group; all_gather only when someone has evictions), called
  in lockstep from _speculative_next_batch entry. NOT shipped: touches
  the decode hot loop's collective structure (known JACCL fragility:
  all_gather at decode rate has corrupted return buffers before) and
  deserves its own validation pass. Exposure is low post-fence-fix
  (degen kills should no longer occur). CHECK whether string-level stop
  sequences at c>=2 hit the same path before relying on them in prod.

## 2026-07-03 (final): DRAIN FIX INSUFFICIENT — FENCE RACE IS DEEPER

Validation (drain fix 6978f56f deployed, async fence armed, 12-pair
battery): 3 clean at FULL async throughput (23.4-24.2 t/s/stream — the
drain costs ~nothing), then iter 4 CORRUPT (uid=10, ' **"**' period-3
loop at token 2098 — a NEW multi-token loop variant). So the
rollback-vs-deferred-graph window is real but NOT the whole bug. The
remaining race is upstream of the rollback: most likely INSIDE the
B=2 forward — async per-layer commits vs CPU-side pool/indexer
metadata (Python-held _pool_lengths / ring bookkeeping read while the
deferred graph that produces the matching arrays is still in flight),
which a cycle-boundary synchronize cannot reach.

Production restored: committed defaults (FENCE_ASYNC_C2=0), c=1 40.4
t/s validated. The drain code stays (harmless, principled, closes a
real window). NEXT session's tools for the residual race:
- Port REFCHECK into _speculative_next_batch (design in the 2026-07-03
  early section): per-cycle batched L=1 reference forward, catches the
  FIRST divergent cycle with the async fence on.
- Or bisect inside the forward: arm async for layer ranges
  (EXO_DSV4_FENCE_ASYNC layer-gating would need a small env) to fence
  which subsystem's deferred work corrupts.
- The corrupt-token attractors are always markdown-ish tokens (982
  ' *', 2619 ' **', 793 ' his', now [2619,4,666] ' **"**') — suggests
  the corrupted state biases a narrow logit region; a REFCHECK diff at
  the first divergence would show HOW wrong the logits are (1ulp tie
  flip vs gross corruption), discriminating metadata-desync from
  torn-buffer reads.

## FINAL SESSION STATE (2026-07-03)

- Production: exo 6978f56f (all fixes incl. harmless drain), defaults:
  FENCE_ASYNC_C2=0. c=1 ~40 t/s; c=2 ~19-21 t/s/stream, quality clean
  (12/12 + 3/3 deep pairs on this config).
- FIXED this session: pydantic event cascade (c0dc8456), prefix-cache
  stale restore (e9609516, crash repro 3/3 clean), fence flip
  (f028dc9c), drain (d4a50cf2+6978f56f, partial).
- OPEN: (1) residual c=2 async-fence race (above); (2) rank0-only
  eviction desync wedge (design in previous section); (3) B=2 verify
  perf = op-count/fusion direction (attribution complete).

## 2026-07-03 (afternoon): REFCHECK VERDICT — TIE-FLIPS, NOT CORRUPTION

Batched REFCHECK shipped (ba2c5c82, EXO_DSV4_MTP_REFCHECK_BATCH=<jsonl>;
per-cycle batched L=1 clean forward vs verify bonus row, argmax + top-2
margins, TP-safe, self-disabling on error).

ASYNC ARM RESULT (1001 cycles, one corrupting pair): verify-vs-clean
argmax disagreements on 2% of cycles — EVERY ONE a 1-ulp tie (max
disagreeing margin 0.125 = the bf16 logit quantum; zero > 1.0). The
B=2 verify does NOT read corrupted state. Mechanism reframed: at temp 0,
~44 tie-flips per 4000-token stream re-roll the greedy trajectory; some
walks enter the model's own repetition attractors (' *', ' **', ' his'
are legitimate greedy loops of this checkpoint). This explains why
corrupt tokens are always common markdown-ish tokens and why REFCHECK
margins near onset stay tiny.

IMPLICATIONS:
- The "corruption" is temp-0 trajectory INSTABILITY at c=2, not state
  damage. Production Hermes runs temp 1.0/0.3 where 1-ulp logit shifts
  are absorbed by sampling — likely unaffected (UNVALIDATED: run a
  temp-1.0 deep c=2 battery to confirm).
- The fence A/B (5/19+2/3 corrupt async vs 12/12+3/3 clean sync,
  uninstrumented) remains real but is now best read as the fence
  changing FLIP RATE (scheduling nondeterminism), not causing state
  corruption. NOTE: a 13th sync pair DID corrupt (with REFCHECK
  deployed) — sync is not absolutely immune.
- A true fix for temp-0 c=2 stability needs a batch-invariant argmax
  (e.g. fp32 bonus-row recompute on a fixed-shape path), NOT the old
  eps-window tiebreak (0.5-window corrupts this flat checkpoint —
  6e978650).

OPEN PUZZLES (for next session):
1. The sync-arm REFCHECK run wrote ZERO rows (env verified present in
   the runner, spec gates open, code identical to the async arm that
   wrote 31). Unresolved — first step: check the runner log for the
   self-disable warning and whether _speculative_next_batch executed
   at all in that launch (add a liveness marker row on cycle 1
   unconditionally).
2. Whether the REFCHECK trim(1)+refeed itself perturbs pool state at
   flush boundaries (tag rows landed pool_flushed=false throughout the
   async run, so no evidence yet — but the instrumented sync pair
   corrupted while the uninstrumented sync record was 12/12).

PRODUCTION (session close): committed defaults (FENCE_ASYNC_C2=0),
exo ba2c5c82 both nodes, c=1 39.8 t/s clean. All instruments off.

## 2026-07-03 (evening): TEMP>0 c=2 BATTERY — FENCE HYPOTHESIS FALSIFIED; SYNC NOT IMMUNE EITHER

The afternoon REFCHECK verdict left one load-bearing UNVALIDATED claim:
"production Hermes runs temp 1.0/0.3 where 1-ulp shifts are absorbed by
sampling — likely unaffected." Ran the deep temp-1.0 c=2 battery to settle
it (4000-tok divergent staggered pairs, server degen kill-switch + client
full-text repetition/chars-per-token detector as the signal).

### RESULT — temp 1.0 does NOT rescue c=2 (both arms corrupt)

| config (temp 1.0, 4000-tok c=2 pairs)      | corrupt / total |
|--------------------------------------------|-----------------|
| async fence (EXO_DSV4_FENCE_ASYNC_C2=2)    | 2 / 5   (~40%)  |
| sync  fence (EXO_DSV4_FENCE_ASYNC_C2=0, PROD) | 3 / 13  (~23%)  |
| sync  fence @ temp 0.3                     | 0 / 3   (underpowered) |

- Async arm corruptions were REAL degeneration, not tie-flip noise:
  t10-3 (server DEGENERATION kill uid=5, backtick period-1 @token 2226,
  partner stream hidden-degen cpt=1.29 @118 t/s) and t10-5 (protein-folding
  stream collapsed into a " . . . . ." whitespace-repetition attractor,
  partner truncated by the batch kill). Sampling at temp 1.0 sometimes
  WALKS INTO the corrupted-logit region rather than away from it — the rate
  (~40%) is if anything worse than the temp-0 async rate (5/19 ≈ 26%).
- => The REFCHECK "just tie-flips, prod likely unaffected" framing is
  FALSIFIED for the async fence. Arming FENCE_ASYNC_C2=2 would corrupt real
  sampled Hermes traffic. Keeping FENCE_ASYNC_C2=0 is correct, not merely
  conservative; the +25% c=2 stays locked out until the intra-forward race
  is fixed.
- SYNC fence (production default) ALSO corrupts at temp 1.0: 3/13. The
  three: t10-7 (uid13 period-1 @token1410, HARD WEDGE — see below),
  t10-3cont (uid5 period-2 "** ** **" markdown attractor @token2982,
  runners restarted clean), t10-6cont (uid '**"  "****' attractor, killed
  cleanly, NO wedge). All deep-onset, all markdown/dash repetition
  attractors, all server-degen-kill. So c=2 concurrent serving is NOT
  fully quality-safe in production at sampling temps regardless of fence —
  the async fence AMPLIFIES a race that is present under sync too. The
  afternoon note "a 13th sync pair DID corrupt" was not a fluke; the sync
  rate is real and non-trivial (~23%). temp 0.3: 0/3 (underpowered — do
  not read as safe).

### NEW: the wedge fires under the PROD default and does NOT self-heal

The wedge on a degen kill is STOCHASTIC (3 sync degens, 3 different
outcomes): t10-7 HARD-wedged (both runners GPU-timeout DIED,
IOConnectUnmapMemory teardown, master stuck 8+ min in "Selecting
coordinator" placement loop WITHOUT reloading — no self-heal, full
start_cluster.sh relaunch required); t10-3cont restarted the runners
cleanly ("bye from the runner" → fresh bootstrap, ~4 min back to READY);
t10-6cont killed only the degenerate stream (action=error) while its
partner finished normally, cluster stayed fully up. So the rank0-only
eviction desync fires only SOMETIMES.

Root of the no-self-heal: a runner that dies via GPU-timeout mid-collective
comes back as RunnerConnecting bound to the STALE instance, never
transitioning to RunnerFailed — so the plan.py:88 "RunnerFailed → Shutdown
→ re-place" recovery cycle never fires (verified in
worker/plan.py:_shutdown_failed / _create_runner). This upgrades wedge
open-item #2: it needs BOTH the agree-on-evictions fix (prevent the desync)
AND a runner-death detector that marks the peer RunnerFailed so re-placement
runs (or start_cluster's placement-retry needs to run continuously, not
just at launch).

### PRODUCTION-SAFETY IMPLICATION (needs user decision)

At c>=2 with temp>0, deep generations (>~1400 tok) can degenerate AND can
wedge the cluster requiring manual relaunch. Options, in order of effort:
1. Gate c=2 spec back off for long generations (revive
   EXO_DSV4_MTP_C2_MAX_CTX-style gate but keyed on max_tokens/output
   length, not context) — cheap mitigation, costs c=2 throughput on long
   outputs only.
2. Fix the wedge (agree-on-evictions + RunnerFailed-on-death) so a
   degen at least fails one request cleanly instead of taking the cluster.
3. Fix the underlying batch-nondeterminism (batch-invariant fp32 bonus-row
   argmax) so c=2 stops degenerating at all — the real fix.

### TOOLING (session scratchpad, hardened)
- c2_pair_probe.py: streaming c=2 pair probe. Full-text saved to
  m4-1:~/scratch/c2_texts/<pair>_s<i>.txt; detects tail-loop, body-loop,
  and chars-per-token collapse; requires repeated unit to fill >=200 tail
  chars (ellipsis/rule false-positive fixed). printf (not echo) write so
  backslashes in essay text don't mangle the JSONL.
- c2_temp_battery.sh: N_T10/N_T03 pairs; wait_ready requires a body token;
  quick_ready gates spinner-killing on CONFIRMED API unresponsiveness
  (>90% CPU only); lenient JSONL summary.
- Battery watchdog LESSON (cost a healthy-runner kill this session): a
  loop detector that flags trailing "..."/"---" plus a >50%-CPU spinner
  rule will kill a live decoding runner. Both fixed. Always gate destructive
  watchdog actions on an independent liveness signal.

## 2026-07-03 (evening): PHANTOM-STREAM LEAK ON STOP-SEQUENCE FINISH — CONFIRMED + QUANTIFIED

Verified the code-traced hypothesis on-cluster (m4-1, sequential requests,
no c=2/no degen — benign; cluster stayed 2/2 Ready throughout).

### THE LEAK (proven)

A request that finishes via a STRING STOP-SEQUENCE match (OpenAI `stop`
param) leaks a phantom decode stream: exo's wrapper sets
finish_reason="stop", del's the uid from ITS _active_tasks and sends
FinishedResponse, but NEVER removes the uid from the underlying mlx-lm
BatchGenerator. mlx-lm keeps the uid in self.uids and keeps decoding it to
ITS OWN max_tokens. Every subsequent step emits that uid →
"response uid N was not found - should be active" (batch_generate.py:2054),
and the uid keeps occupying a batch slot.

Only the DEGEN kill-switch path evicts (via _killswitch_evict_uids →
_mlx_gen.remove()); the stop-sequence path has no equivalent. EOS/length
finishes are clean — those are mlx-lm's OWN stop conditions so it self-drops
the uid.

### MEASURED (phantom_leak_test2.py, on m4-1)

One stop-terminated request (stop=["\n"], max_tokens=2000) RETURNED after
33 real tokens, then leaked ~1292+ phantom decode steps ("should be active"
count 644→1936). Throughput cost on a CONCURRENT request: 12.2 t/s vs 16.4
baseline = ratio 0.74 (~26% slower — the phantom occupies a 2nd batch slot,
forcing B=2). (Absolute t/s here are wall-inclusive-of-prefill on short 250-
tok probes; the RATIO is the valid signal, not the absolutes.)

### WORSE THAN "BOUNDED WASTE": THE PHANTOM FREEZES + RE-ATTACHES

The runner's outer loop is `while self.active_tasks:` (runner.py:424), so
the phantom is NOT stepped while no real request is in flight — it FREEZES.
The counter proved it: +690 steps during probe C, +0 during a 15s idle
poll, then +602 MORE during probe E. So one leaked phantom degrades EVERY
subsequent request (~25% each) until it burns its full max_tokens budget of
co-scheduled steps — it does not expire on wall-clock. (My first analysis
misread the idle-freeze flat counter as "phantom completed"; corrected.)

### PRODUCTION IMPACT

- Any client using `stop` with a generous max_tokens leaves a slot-occupying
  phantom that taxes throughput on following requests.
- CRITICAL COUPLING: the phantom raises B. The c=1 async decode fence and
  the c>=2 spec gates are B==1-gated — a phantom silently disarms them, so a
  nominally-single-stream workload with stop sequences loses the +28% fence
  AND can trip the c>=2 corruption path documented above. (Inferred from the
  B-gate arming, not separately measured.)
- EXPOSURE UNKNOWN: could not confirm from the log whether real Hermes
  clients set `stop` (only my own probes were in recent traffic). CHECK the
  Hermes gateway / frontend request params before rating severity.

### THE FIX (rank-safe direction; NOT shipped — needs c=2 validation)

Root: exo does string-level stop matching in its OWN layer
(potential_stop_sequence_text / _stop_sequences, batch_generate.py:2211) and
does NOT pass the stops to mlx-lm's native token-level matcher. mlx-lm's
BatchGenerator.next() ALREADY has stop state-machines (generate.py:1588,
state_machines[i].match → finish_reason="stop" → self.filter(keep)) that
self-drop the uid SYMMETRICALLY on all ranks (the TP _step() forward is
deterministic, so every rank matches identically — no collective needed).

Preferred fix: wire the request's stop sequences into mlx-lm's insert()/
submit() so its native matcher fires and filters the uid. Rank-safe BY
CONSTRUCTION; also lets exo drop (or keep as belt-and-suspenders) its own
string layer.

Why NOT a naive rank0 evict: appending stop-finished uids to
_killswitch_evict_uids and calling _mlx_gen.remove() risks the SAME wedge as
the degen kill-switch (handoff open #2) IF exo's stop detection is not
symmetric across ranks. The DSv4 path uses DSv4MTPBatchGenerator.
next_generated() (not the vanilla next() read above) whose rank behavior for
detok/sampling is the subtle part the wedge note warns about. Do not ship a
hot-loop eviction without a c=2 concurrent-stop-sequence validation pass
(which itself can trigger the SIGKILL-immune wedge). The native-matcher
route sidesteps this entirely.

NOTE: this UNIFIES with wedge open-item #2 — both the phantom leak and the
degen-kill wedge are the missing "symmetric mid-batch eviction" primitive.
The native-matcher fix solves the stop-sequence half cleanly; the degen-kill
half still needs agree-on-evictions (or its own native-matcher equivalent).

Tooling: scratchpad phantom_leak_test.py (v1, spam-count proof) +
phantom_leak_test2.py (v2, throughput ratio + lifetime), copies on
m4-1:~/scratch.
