# T3: switch_mlp (FusedSwitchGLU) real achieved bandwidth — 27.7% of peak, confirmed via pipelined microbench — 2026-08-22 (session 4)

## Why this check

Per the Fable-provided plan, T2 confirmed occupancy/gap/clock all
improved dramatically post-fence-fix but could not attribute per-kernel
cost (the `xctrace` Metal System Trace template only exposes generic
"Compute"/"Fragment"/"Vertex" channel names, not MLX kernel names). T3
targets the single largest-known busy-time consumer, `moe.switch_mlp`
(historically ~30-45% of wall time per earlier span breakdowns), via a
standalone `mx.metal.start_capture()` microbench that exactly
replicates the real production kernel call.

## Method — built to match production exactly, not a generic MoE guess

Read the actual deployed code before building anything (per the
project's standing "pipelined microbench, not per-call-isolated"
methodology rule, and the standing rule to verify sharding-axis
assumptions before estimating):

- Confirmed **`EXO_DSV4_MOE_FUSED_GATE_UP=1` is live in production
  right now** (§2.8/§13) — the microbench must use `FusedSwitchGLU`
  (one `gather_qmm` for gate+up combined), not vanilla `SwitchGLU`
  (which would measure the wrong, no-longer-deployed 3-dispatch path).
- Confirmed via `auto_parallel.py`'s own code comment (dated
  2026-08-16, correcting an earlier wrong assumption that had misled a
  prior investigation): **TP=2 shards the MoE intermediate WIDTH, not
  expert identity** — both ranks hold all 256 experts, each at HALF
  `moe_intermediate_size` (2048 → 1024/rank). Built the microbench at
  this real per-rank shape, not the full pre-shard width.
- Confirmed quantization via `make_quantization_config()`: MoE expert
  weights (`.ffn.switch_mlp.*_proj`) are **mxfp4** (`group_size=32,
  bits=4`), not the model-wide default 8-bit affine.
- Copied `FusedSwitchGLU.__call__`'s exact logic (gather_sort → single
  fused `gather_qmm` → split → SwiGLU → `down_proj` → scatter_unsort)
  from `auto_parallel.py` verbatim into a standalone script — same
  code path, not a reimplementation that could silently diverge.
- Real decode shape: B=1, L=1, `top_k=6` random routing indices (5 lines earlier
  confirmed random routing doesn't change per-token bytes touched —
  always exactly `top_k` experts' weights are read regardless of which
  ones).
- Ran on a real Studio node (m4-1) via SSH, standalone process — zero
  risk to the live production cluster (confirmed cluster still healthy
  and unaffected via a post-run `/state` check, not assumed).
- **Pipelined, not per-call-isolated**: 300 real back-to-back iterations
  timed via `mx.synchronize()` bracketing, per the project's standing
  rule that per-call-isolated dispatch-overhead estimates have
  repeatedly and independently misled three earlier fusion investigations
  in this repo's history (§6 in `PERFORMANCE_HISTORY.md`).
- Real Metal GPU Frame Capture (`mx.metal.start_capture()`, required
  `MTL_CAPTURE_ENABLED=1` env var — first attempt without it failed
  with `"Capture layer is not inserted"`, a real, simple environment
  gate, not a code bug) — captured 20 iterations to `/tmp/switch_mlp_capture_session4.gputrace`
  for future Xcode Metal Debugger inspection (not yet opened/analyzed
  this session — the achieved-bandwidth arithmetic below doesn't
  require it, but the capture is available for deeper per-encoder
  attribution if needed later).

## Real result

**Pipelined wall time: 290-312µs/call** (300 real iterations, two runs
for consistency — 290.12µs and 311.66µs, well within normal run-to-run
variance).

Real bytes touched per token, per rank (post-TP-shard shapes,
mxfp4 + fp16-scale-per-32-param-group overhead):

| | Value |
|---|---|
| `moe_intermediate_size` per rank | 1024 (2048 / TP=2) |
| Fused gate+up params/expert | 2 × 1024 × 4096 = 8,388,608 |
| `down_proj` params/expert | 4096 × 1024 = 4,194,304 |
| Total params/expert | 12,582,912 |
| Bytes/expert (mxfp4 + scale overhead) | 7.864 MB |
| Bytes touched/token (top_k=6) | **47.186 MB** |

**Achieved bandwidth: 47.186 MB ÷ 300µs ≈ 151.4 GB/s.**

**Efficiency vs M4 Max's 546 GB/s peak spec: 27.7%.**

## Sanity check against known decode-time budget

43 layers × ~300µs/layer (this microbench's per-call time) ≈ 12.9ms/token
of pure `switch_mlp` cost. Against the post-fix short-context real wall
time (32.15-34.25ms/token, from T1's recompute): **switch_mlp alone
accounts for ~38.9% of total decode wall time** — directly consistent
with (not just vaguely similar to) the historically-cited "~30-45% of
wall time" span-breakdown figure for this kernel, cross-validating this
microbench's shape/config against real production behavior rather than
measuring an unrelated synthetic workload.

## Decision gate (per the Fable-provided plan)

Plan criteria: "**<40% of peak BW** → reproduce with a pipelined
microbench at exact decode shapes to confirm; **>60-70% of peak** →
kernel near its floor, weight shifts to MTP."

**27.7% of peak clearly falls in the <40% bucket.** This microbench
*is* that confirming pipelined test — real, not per-call-isolated,
built against verified production shapes/config. **Confirmed: the
`switch_mlp` kernel itself has real, substantial achieved-bandwidth
headroom** (546 GB/s theoretical vs 151.4 GB/s achieved — a genuine
3.6x gap at the single largest busy-time kernel). This is NOT a
"kernel is already near its floor, nothing left to claim" situation —
there is a concrete, quantified target for further work.

## What this does and doesn't establish

**Does establish**: the switch_mlp/FusedSwitchGLU kernel, at the exact
production shape and quantization, achieves only ~27.7% of the M4 Max's
theoretical peak memory bandwidth in a clean pipelined microbench. This
is a real measured ceiling gap, not an estimate.

**Does NOT establish**: WHY the kernel falls short of peak bandwidth.
Candidate causes not yet distinguished: (1) the `gather_sort`/
`scatter_unsort` overhead around the core `gather_qmm` call (this
microbench includes that full path, not an isolated `gather_qmm` call —
a follow-up could isolate just the `gather_qmm` op to see how much of
the 300µs it alone consumes vs. sort/scatter overhead); (2) mxfp4
dequantization overhead specific to the `gather_qmm` kernel's internal
implementation, independent of raw byte-transfer bandwidth; (3) B=1
decode-shape inefficiency (very thin/tall access pattern relative to
GPU's preferred wide-batch memory access pattern) — this is plausible
given B=1's inherent memory-access-pattern disadvantage vs the wide
batched reads a bandwidth-bound roofline assumes; (4) real per-expert
weight-fetch scatter (top_k=6 experts out of 256 means non-contiguous
memory reads across a very large weight table, which is a structurally
different — and typically less efficient — access pattern than the
roofline's implicit assumption of one contiguous read).

Candidate (3)/(4) — decode's inherently scattered, thin-batch memory
access pattern for MoE gather — is the most likely root cause given
this repo's own architecture (top-6-of-256 sparse routing is
fundamentally a gather operation, not a dense read), but this session
did not isolate which factor dominates. Flagged as the natural next
sub-step if this line of investigation continues (e.g., a fatter-batch
microbench varying B from 1 to 8 to see if bandwidth efficiency scales
with batch size, which would implicate (3)/(4) directly).

## Real limitation acknowledged

This measures ONE rank's local compute in isolation — it does not
include the `moe.all_sum` collective that follows in real production
(already separately measured at ~2.19ms/token total, see §2.7), nor
any inter-op scheduling/dispatch overhead from the surrounding decode
loop (attention, norm, residual, etc.) that the real per-token wall
time also includes. This is a targeted, single-kernel measurement, not
a full-decode-loop reproduction.

## Next step

Per the plan, T3's decision output feeds into a scoping decision for
this kernel's optimization (deferred — no code change made this
session), OR proceeding to T4 (cross-rank skew) which is independent
and can run in parallel. Given T3 confirmed real headroom, per the
plan's T6 gate ("if realistic ceiling <~1.5x current throughput, MTP
becomes highest-EV"): this switch_mlp gap alone, if fully closed,
would only address ~12.9ms of a wall time whose current unattributed
remainder is ~23-38ms (T1) — a meaningful but partial win, not enough
on its own to close the full gap. T4/T5 remain needed before the T6
MTP-port gate can be evaluated with real numbers.
