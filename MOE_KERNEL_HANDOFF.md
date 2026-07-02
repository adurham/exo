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
