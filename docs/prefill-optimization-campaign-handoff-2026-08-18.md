# DSv4-Flash TP prefill optimization campaign -- consolidated handoff (2026-08-18)

## What this session was

A ground-up hunt for further TP prefill throughput on DeepSeek-V4-Flash-0731
across 2x Mac Studio M4 Max (RDMA/TB5), starting from the prior session's
measured ~358.6 tok/s ceiling. Investigated every candidate lever
identified from a full span-level profile, closed several definitively,
and landed one confirmed-real, precisely-scoped, not-yet-implemented
kernel-level opportunity.

## Levers investigated, in order, with verdicts

### 1. EXO_DSV4_SEQ_SPLIT (attention query-row split across TP ranks)
**CLOSED -- confirmed optimal at default.** Re-validated at long context
(190-220K tokens, previously only validated at 100K): SEQ_SPLIT=1 measured
358.6 tok/s vs SEQ_SPLIT=0 at 277.8 tok/s -- ~29% faster. Standing default
is correct. See `dsv4-220k-prefill-seqsplit-ab-2026-08-18.md`.

### 2. attn.sdpa.compressed (11.8% of prefill wall time)
**CLOSED -- dead end, not a bug.** Three independent audits (code-path
audit, span-instrumentation audit, git-history provenance check) confirmed
this is a distinct, architecturally-mandated attention class
(`CompressedAttention`, ~21 of 43 layers) that never runs redundantly
alongside the sibling `SparseCompressedAttention` class (~20 layers) --
each layer runs exactly one. Spans are correctly measured (no
double-counting). Not a regression -- unchanged since April 2026, just
given its own profiler label in July. No fix exists here.

### 3. rsync repo-sync "stuck at 0.84 Mbit/s" (from the incoming handoff)
**RESOLVED -- not a bug.** The driving laptop has zero Thunderbolt
hardware connected (confirmed via `system_profiler`); the TB5 RDMA subnet
is studio-to-studio only and was never reachable from the laptop. The
observed slowness is ordinary WiFi transfer time for changed build
artifacts, not a misrouted interface. No code change.

### 4. fp8/mxfp4 MoE GEMM efficiency (moe.switch_mlp, 26.9% of wall time --
the single largest span)
**REAL FINDING, partially actionable, partially deep unfinished work.**
- Corrected a framing error: routed MoE experts are **mxfp4** (4-bit,
  group_size=32), not fp8 as originally assumed. Only the small always-on
  shared-expert path is mxfp8.
- Production's real per-expert-group M at 2048-token prefill chunks is
  severely ragged: range 1-1653, median just 14 rows, mean ~48 -- far
  below the fixed 16-row GEMM tile every group gets processed through
  regardless of size.
- Built a clean production-class benchmark
  (`bench/moe_production_class_bench.py`, committed to main) using the
  real `SwitchGLU`/`QuantizedSwitchLinear` classes at real shapes: measured
  62.6% of matched-shape dense-GEMM ceiling at the standing 2048-token
  chunk size, improving to 72.0% at L=8192 (43.4% at L=512) -- a real,
  reproducible, chunk-size-dependent inefficiency, not a benchmark
  artifact.
- Retired a misleading prior claim: a July 2026 commit message claimed
  "82 TFLOP/s" for this same kernel at "the same" M=48 shape -- reconciled
  and found this number **physically exceeds the M4 Max's own ~16 TFLOPS
  hardware ceiling** and has zero corroborating record anywhere in the
  repo. It's not trustworthy; retire it. Tonight's 9.37 TFLOPS / 62.6%
  finding is the correct reference number going forward.

### 5. EXO_PREFILL_STEP_SIZE=4096 (raise chunk size to improve MoE-GEMM
efficiency, per finding #4's chunk-size curve)
**RETESTED, CLOSED -- quality issue is stale, but throughput regresses.**
The July 2026-07-13 "4096 breaks quality" finding does NOT reproduce on
current code (clean needle-in-haystack passes at 1K and 32K context).
However, real end-to-end throughput at 4096 measured **331.2 tok/s at
~191K tokens -- an 8% REGRESSION vs the 358.6 tok/s 2048 baseline**,
despite the isolated MoE-GEMM benchmark showing 4096 should be ~15% more
efficient in isolation. The gain is real but is outweighed by a larger,
chunk-size-dependent cost elsewhere in the pipeline (leading suspect: the
sparse indexer, whose cost this codebase's own `start_cluster.sh` comments
already document as scaling unfavorably with chunk size at high context).
Standing `EXO_PREFILL_STEP_SIZE=2048` default reconfirmed correct. See
`dsv4-prefill-step-size-4096-retest-2026-08-18.md`.
**Not fully closed as an opportunity**: decoupling MoE batch size from
attention/indexer chunk size (run attention at 2048, batch several such
chunks before feeding MoE a larger effective L) remains a real, scoped,
unexplored architectural idea.

### 6. Rare severe RDMA collective stalls (Metal Event::wait, ~0.17%
per-call rate, up to 11.7s each)
**Investigated from 3 Metal/Apple-Silicon-specific angles, all ruled out
or already-adequately-instrumented:**
- **Memory pressure / GPU residency / MTL_DISABLE_TIMEOUT**: the disabled
  Metal watchdog was already investigated in July and found to be inert
  (nothing in this codebase/macOS build reads those env vars) -- ruled
  out as the mechanism. Memory pressure is plausible in principle
  (wired-memory limit is legitimately close to the model+KV footprint)
  but unconfirmed -- no live spike was captured correlating with an actual
  stall.
- **Thread QoS scheduling**: `EXO_RUNNER_QOS` already exists as a knob and
  is deliberately disabled -- it was tried and empirically caused a
  measured 16% regression + 25% bad-run rate from Metal command-queue
  contention. Don't re-enable it; this was a real, already-answered
  question, not an oversight.
- **Command-buffer scheduling-vs-execution timing discriminator**: already
  fully implemented (`EXO_CMDBUF_RING_DIAG`, wired to the exact stall
  detection point) -- nothing to build. The retransmit-round=1 correlation
  that looked suspicious is fully explained as architectural noise (a
  mandatory per-call counter, not an ARQ retransmit indicator), not a
  causal clue.
- **Net**: this remains a rare, low-priority, already-adequately
  instrumented issue. Recommendation unchanged from earlier tonight:
  leave `EXO_CMDBUF_RING_DIAG=1 JACCL_TRACE_PROGRESS=1` on passively,
  drop `EXO_PROFILER=spans` (not free, ~15% throughput tax), and pull the
  diagnostic dump organically next time a stall fires naturally.

### 7. MLX gather_qmm small-M kernel dispatch (the leading Metal/M4-Max-
specific theory for the MoE-GEMM inefficiency in finding #4)
**REAL, CONFIRMED, PRECISELY SCOPED -- not yet implemented.** This is
tonight's most concrete unfinished lever.

Confirmed via direct code read of this fork's vendored MLX
(`mlx/mlx/backend/metal/quantized.cpp`, `quantized.h`, `fp_quantized.h`):

- The production M>1 sorted-prefill dispatch path (`GatherQMM::eval_gpu`)
  uses one fixed 16-row GEMM tile for every expert group in a call,
  regardless of how few rows (down to 1) any individual group actually
  has.
- A GEMV-style fast kernel (`gather_qmv_rhs`) already exists and already
  handles ragged small-M internally in its Metal shader logic (loops over
  chunk sizes 1-7 plus a tile size) -- but is gated to fire only when the
  OUTER call's M==1 (pure decode), never for prefill's M=2048 calls, even
  though the ragged PER-EXPERT run-lengths inside those calls are exactly
  what this kernel already handles well.
- **The real blocker, precisely identified**: `gather_qmv_rhs` reads/writes
  via direct positional pointer arithmetic, requiring physically
  contiguous, pre-broadcast input rows. Extending it to the M>1 sorted
  path requires **indirect addressing via `lhs_indices`** (the same
  gather-by-index pattern the existing tiled kernel `gather_qmm_rhs_lhs`
  already uses successfully) -- this is genuine new kernel-indirection
  work across two shader files (affine + fp quant modes), not a
  guard-condition tweak. A prior attempt at a related idea (OPT-8,
  physically broadcasting instead of gathering) was tried and reverted for
  causing Metal allocator stalls from the broadcast memory cost -- the
  `lhs_indices` gather approach avoids that specific failure mode by
  construction, since the tiled kernel already proves it works.
- **Effort estimate**: a focused half-day-to-day implementation task, not
  a few-hours fix. A precise implementation spec was produced (exact
  files, exact kernel/dispatch changes, exact validation plan) so a future
  session can execute directly without re-deriving tonight's analysis --
  see the full spec in the delegation transcript
  `~/.hermes/cache/delegation/live/deleg_b70fec44/task-0.log`, summarized
  here:
  - `quantized.h:895-974` (affine) + `fp_quantized.h:1673-1750` (fp): add
    a `gather_qmv_rhs_lhs` kernel variant with an `lhs_indices` buffer
    param; change the chunk-load addressing from `x + m*in_vec_size` to
    `x_base + lhs_indices[m0+m]*in_vec_size`.
  - `quantized.cpp`: new `gather_qmv_rhs_lhs()` C++ wrapper modeled on the
    existing `gather_qmm_rhs_lhs` (lines 1465-1583); replace the `M==1`
    gate in `GatherQMM::eval_gpu` (currently line ~1889) with an M>1-safe
    condition bucketed on B/E (avg rows per expert), routing small-B/E
    groups through the new qmv-lhs path regardless of outer M, and large
    groups through the existing tiled path.
  - Validation: standalone harness with synthetic ragged run-lengths
    matching production's real distribution (median 14, range 1-1653,
    top-6-of-256 @ 2048 tokens), bit-for-bit/tolerance comparison against
    the existing tiled kernel's output on BOTH quant modes, correctness
    FIRST, before any performance claim -- then A/B bench against the
    current baseline (a real risk exists that dispatch overhead from
    splitting into two kernel launches could outweigh the tile-waste
    savings; this must be measured, not assumed).

This is the single most concrete, highest-expected-value unfinished item
from tonight's entire investigation.

## What's genuinely closed vs what remains open

**Closed (no further action needed):**
- SEQ_SPLIT tuning
- attn.sdpa.compressed (not a bug)
- rsync slowness (not a bug)
- EXO_PREFILL_STEP_SIZE at 4096 as a simple flag flip (regresses)
- Rare RDMA stalls as a Metal-watchdog/QoS/timing-instrumentation question
  (all three angles ruled out or already adequately handled)

**Open, real, and worth returning to:**
1. **The gather_qmv_rhs small-M MoE kernel extension** (finding #7) --
   precisely scoped, real expected payoff (recovering meaningful ground
   toward the 72%-of-ceiling number), needs a focused implementation
   session with correctness-first Metal kernel work across two shader
   files. This is the top recommendation for next session.
2. **Decoupling MoE batch size from attention/indexer chunk size**
   (finding #5's residual opportunity) -- would let the MoE-GEMM
   efficiency win from larger effective batches be captured without
   paying the indexer's chunk-size penalty. Architecturally bigger and
   less scoped than #1, but real.
3. **Rare RDMA stall organic capture** -- zero-effort, already-instrumented,
   just needs to fire naturally during real usage; pull the
   `EXO_CMDBUF_RING_DIAG` dump when it does.

## Cluster state at end of session

Healthy, 2-node, `READY (2/2)`, commit `5c4ba9ce8` on both nodes, standing
config (`EXO_PREFILL_STEP_SIZE=2048`, `EXO_DSV4_SEQ_SPLIT=1`,
`MLX_JACCL_DATA_RECV_POOL=0`). Repo tree clean, all real findings
committed and pushed to `origin/main`.
