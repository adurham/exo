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

- Long-context (200K+) c=2 needle under the new defaults (levers don't
  touch cache layout, but unsoaked).
- c=3..5 concurrency battery under per-stream acceptance.
- m4-2 exo pydantic extra_forbidden crash on GPU-timeout event strings
  (robustness, open).
