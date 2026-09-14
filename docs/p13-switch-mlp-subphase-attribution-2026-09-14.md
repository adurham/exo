# P13: moe.switch_mlp/GatherQMM sub-phase attribution — gather/scatter closed as negligible, prefill compute-efficiency gap flagged as a genuinely new open thread (2026-09-14)

## Scope and what this closes

§13's line "A real Instruments Metal trace of the `moe.switch_mlp` GatherQMM
kernel internals specifically ... remains unexplored below the
kernel-dispatch level" is **substantially answered, but not via
Instruments/xctrace** — per P11 (2026-08-31), `xctrace`'s `metal-gpu-intervals`
template is a dead end (generic Compute/Fragment/Vertex labels only,
structurally incapable of per-kernel names), and the doc's own hazard table
records 3 separate production-runner kills from `xctrace` attach. No new
Instruments capture was attempted here for that reason — this would have
re-proven a known-closed negative.

Real GUI-only tooling (Xcode's GPU Frame Capture "Performance" tab, which
*does* show a limiter classification) was also evaluated and ruled
impractical: this investigation runs entirely over SSH with no VNC/screen-
sharing access and no passwordless sudo on either node, so the GUI is
unreachable. This is a genuine environment constraint, not a methodology
choice — flagged honestly rather than worked around.

**What was actually done**: a source-level read of `GatherQMM::eval_gpu`
(`mlx/backend/metal/quantized.cpp`) and the Python-level MoE routing wrapper
(`mlx_lm/models/switch_layers.py`) to determine, from first principles, what
the internal structure of a `gather_qmm`/`gather_qmv` dispatch actually is —
followed by real `mx.metal.start_capture()` + `MLX_GPU_TIME=1` +
`MLX_DISPATCH_COUNT=1` bracketing (the p01/p03-proven, zero-incident
methodology) of each Python-visible sub-stage, run as a **standalone process
on m4-1**, never attached to the live production runner PID. Zero relaunches,
zero xctrace, zero production risk. Runner PIDs on both nodes were confirmed
unchanged (43724 / 10062, ~3h47m uptime) before and after.

## Structural finding: gather/scatter are NOT inside the GatherQMM kernel

Reading `switch_layers.py` settles the "is gather/matmul/scatter a single
fused kernel or separable phases" question definitively, with no ambiguity
left for hardware counters to resolve:

```python
do_sort = indices.size >= 64
if do_sort:
    x, idx, inv_order = _gather_sort(x, indices)   # mx.argsort + fancy-index gather
...
gu = mx.gather_qmm(x, ..., rhs_indices=idx, sorted_indices=do_sort)  # ONE kernel dispatch
...
if do_sort:
    x = _scatter_unsort(x, inv_order, ...)          # fancy-index scatter
```

`_gather_sort`/`_scatter_unsort` are **separate MLX ops (real `mx.argsort` +
indexing dispatches) that run in Python BEFORE/AFTER `gather_qmm`** — they
are not internal sub-phases of the GatherQMM Metal kernel. Reading
`GatherQMM::eval_gpu`'s dispatch ladder and the `affine_gather_qmv_fast`
kernel source confirms the kernel itself does per-threadgroup pointer
arithmetic via `adjust_matrix_offsets` (indexed addressing into `w`/`scales`
via `rhs_indices`) fused directly into the same kernel invocation as the
tiled matmul — there is no separable "gather kernel" vs "matmul kernel"
*inside* `gather_qmm` at all. So the real "gather/matmul/scatter sub-phase"
decomposition the brief asked for is a decomposition across **five
Python-level stages**, not something requiring an Instruments trace to see
inside one dispatch:

`gather_sort → fused_gate_up (gather_qmm) → activation (SwiGLU) → down_proj (gather_qmm) → scatter_unsort`

**Critical shape-dependent finding, never previously flagged**: at decode
shape (M=1, top_k=6 → `indices.size=6 < 64`), `do_sort` is **False** —
`gather_sort`/`scatter_unsort` never fire. The Aug-29 P01 capture measured
decode shape exclusively, so it correctly never saw these stages (not a gap
in that work — those stages are inert at M=1). At **prefill shape** (M=2048,
`indices.size=12288 ≥ 64`), `do_sort` is **True** and these are real, every-call
dispatches that nobody had measured before this session.

## Real measurement: gather_sort + scatter_unsort are NOT the dominant cost

Standalone process on m4-1 (production live and busy throughout — see
Methodology Caveat below), rotated-index pools, chained/pipelined GPU-time
bracketing (not isolated per-call, per the repo's §4.6 standing rule), 3
independent runs at prefill shape (M=2048, top_k=6, 256 experts, mxfp4
g=32/b=4, matching production's real `make_quantization_config()` target and
the live `EXO_DSV4_MOE_FUSED_GATE_UP=1` fused-gate-up path):

| Stage | Run 1 (µs) | Run 2 (µs) | Run 3 (µs) | Dispatches/call |
|---|---|---|---|---|
| `gather_sort` | 351 | 1760 | 983 | 10 |
| `fused_gate_up` | 40146 | 37623 | 34872 | 3 |
| `activation` | 186 | 436 | 307 | 1 |
| `down_proj` | 20638 | 13219 | 14347 | 3 |
| `scatter_unsort` | 791 | 811 | 812 | 1 |
| **gather+scatter % of total** | **1.8%** | **4.8%** | **3.5%** | — |

Despite ~2x contention-driven variance in the absolute numbers (production
was running a live 349K-token prefill and concurrent decode traffic during
every measurement — this is a real cluster, not an idle testbed), the
**gather+scatter share is stable at under 5% of total GPU time across all
three runs**, and dispatch counts (contention-independent) are exactly
stable at 18/call across runs. This is a robust qualitative result:
**gather/scatter overhead is not the bottleneck at any measured shape.**
The dominant cost is squarely the two `gather_qmm` matmul stages
(fused_gate_up + down_proj), consistently ~95%+ of total GPU time.

This is a genuine, previously-uncaptured negative result: the doc's §13 had
flagged "candidates: gather_sort/scatter_unsort overhead around the core
gather_qmm call" as an unresolved hypothesis for the decode-time efficiency
shortfall (2026-08-22, T3 entry) — that specific candidate is now closed for
prefill too (it was already structurally impossible at decode's M=1 shape,
per the `do_sort` gate above).

## Compute-bound vs memory-bound: decode and prefill are in DIFFERENT regimes

Using the doc's own measured hardware ceilings (15.21 TFLOPS bf16 on-node
peak, PH:5819-5822; 424-488 GB/s measured streaming, PH:5822) computed a
proper roofline arithmetic-intensity check, not just bytes-vs-spec-peak:

- **Decode (M=1)**: 3.05 FLOPs/byte vs a ~28-36 FLOPs/byte ridge point →
  **0.085x the ridge point — deep in the memory-bound region.** This
  matches and is fully consistent with P01/P03's bandwidth-percentage
  framing (46.9-64.6% of peak GB/s measured this session, in the same
  91-97%-of-spec-under-light-contention ballpark P01 reported on 2026-08-29,
  once the live-production-contention variance this session hit is
  accounted for — see caveat below).

- **Prefill (M=2048, sorted/grouped)**: essentially every one of 256
  experts is touched (P(any expert untouched) ≈ 1.3e-21 at this M/top_k),
  so `gather_qmm`'s sorted path reads each expert's weight tile close to
  once and reuses it across ~48 co-routed tokens on average — this pushes
  effective arithmetic intensity to **~146 FLOPs/byte, ~4x the ridge
  point**, i.e. this regime *should* be compute-bound, not memory-bound.
  Cross-checked directly: implied bandwidth under the most generous
  "each expert read exactly once" assumption is only 28-33 GB/s (5-8% of
  peak) — memory clearly is NOT the limiter here, confirming the
  arithmetic-intensity prediction from the other direction.

- **But measured compute efficiency at prefill is only 33.5-41.3% of the
  15.21 TFLOPS peak** (5.09/6.08/6.28 TFLOPS achieved across 3 runs) —
  meaningfully below what "compute-bound and not memory-limited" alone
  would predict (which should trend toward high efficiency, not ~35-40%).

**This is the genuinely new, previously-uncharacterized finding**: prefill's
`gather_qmm` sub-phase is not bandwidth-limited (ruled out above) and not
dominated by gather/scatter overhead (ruled out above) — the remaining
~60-65% gap looks like an **occupancy/tile-efficiency** problem, consistent
with (but not previously connected at this specific shape to) the existing
"Lever-1 MoE small-M tile-staircase" finding (§ Lever-1 MoE small-M re-check,
2026-08-31): `gather_qmm`/dense `quantized_matmul` are tile-quantized at
~32 rows/expert, and a row count just past a tile boundary pays a
near-2x-for-one-row cliff. Simulating the real production prefill routing
distribution (M=2048, top_k=6, 256 experts, uniform routing — matching this
session's synthetic test, acknowledged as an approximation of the real
learned router's actual skew) gives a per-expert row-count distribution of
mean 48, stdev 6.7, range 31-68 — **86.7% of experts land on a partial final
8-row tile, 94.1% on a partial 16-row tile, 99.2% on a partial 32-row tile.**
The existing Lever-1 finding already established this class of tile-boundary
cost is real and structural (not fixable without a routing/load-balancing
change that alters model outputs, which that investigation correctly ruled
out as off-limits). This session's contribution is connecting that
already-known mechanism to the *specific, previously unmeasured* prefill
`gather_qmm` compute-efficiency gap (33-41% of peak) as a plausible
explanation, not a new lever — see "Not shipped" below.

**Roofline caveat, worth naming explicitly**: the roofline above was applied
at the aggregate-kernel level (total FLOPs / total bytes for the whole
`gather_qmm` call). But `gather_qmm`'s sorted path actually executes as
**256 separate small-M GEMMs** (M=31-68 per expert, per the simulation
above), not one large dense GEMM — and the 15.21 TFLOPS reference peak
(PH:5819) was measured on a large square dense GEMM sweep, not on a batch of
small irregular-M matmuls. Small-M GEMMs typically do not reach the same
peak FLOPS as large dense ones on any GPU, independent of tile-boundary
waste specifically. So "prefill should be compute-bound (AI ≈ 4x ridge)" is
a coarser, less certain label than it would be for a plain dense GEMM — part
of the 33-41%-of-peak gap may simply be "small-M GEMMs have a lower
achievable ceiling than the dense-GEMM reference peak," a distinct
mechanism from (and possibly compounding with) the tile-boundary-cliff
hypothesis. Both live under the umbrella "prefill runs irregular small
per-expert GEMMs," and this session did not separate their individual
contributions. Other unmeasured mechanisms (e.g., per-tile kernel-launch
overhead, small-M occupancy loss unrelated to the specific 32-row cliff)
could also produce some of this residual and were not ruled out — the
tile-staircase connection is offered as *a* plausible, documented-precedent
candidate, not established as *the* leading explanation by elimination.

## Novel dispatch-count finding

A single `gather_qmm` call (isolated, decode shape) dispatches **4 separate
Metal kernels**, not 1 — confirmed via `MLX_DISPATCH_COUNT=1` bracketing
around a bare `mx.gather_qmm(...)` call with no other MLX ops in the graph.
The full fused `BatchedSwitchGLU.__call__` (gate_up + activation + down_proj)
issues **8 total dispatches** at decode shape, and **18 total dispatches** at
prefill shape (10 for gather_sort's argsort+gather, 3+3 for the two
gather_qmm calls, 1 for activation, 1 for scatter_unsort). This is real,
previously-unquantified per-call dispatch-count data — relevant to
"occupancy-limited" in the sense of fixed per-dispatch launch overhead, and a
concrete number for any future dispatch-reduction fusion proposal to size
its ceiling against (per §4.6's rule: pipelined chain-level numbers, not
per-call-isolated estimates, are what determine a fusion's real ceiling —
these are the pipelined chained numbers).

## Methodology caveat: production was live and busy throughout

Every measurement in this session ran while production was actively serving
real prefill (up to 428.7 tok/s on a 349K-token request) and decode traffic
concurrently — this Mac Studio was NOT idle. This produced real, substantial
GPU contention: an independent elementwise-add bandwidth control measured
only 57.6% of the previously-established 424 GB/s streaming baseline during
one of the busier windows, and 8 back-to-back decode-shape `fused_gate_up`
trials swung from 128µs to 659µs (5.1x) purely from live-traffic contention.
This is why the absolute µs/GB/s numbers in this doc are noisier and
generally lower than P01's 2026-08-29 figures (which likely ran during a
quieter window) — **the qualitative findings (gather/scatter <5% of total;
prefill compute efficiency 33-41% of peak; dispatch counts 8/18 per call)
are robust across 3 runs and stable under 2-5x absolute-time swings, but the
specific percentage-of-peak numbers should be read as "this session's
live-contended measurement," not a clean isolated ceiling.** A follow-up
during a genuinely idle window (if one exists — this cluster appears to run
close to continuously busy) would tighten the absolute numbers but is
unlikely to change the qualitative conclusions given how stable the ratios
were despite the contention swings.

Separately, and orthogonally: the microbench itself is **synthetic** (built
from a standalone `BatchedSwitchGLU` instance at the correct shape/dtype/
routing-arity, not the live runner's actual weights or the real trained
router's output distribution) — per this doc's own repeatedly-learned lesson
that synthetic-only findings have not always held up live (e.g. the
2026-08-22 switch-mlp bandwidth artifact retraction). The decode-shape
numbers cross-check reasonably against P01's real-shape prior work
(same 8-dispatch structure, same stage ordering, bandwidth in the same
ballpark once contention is accounted for), which is the strongest
available validity check without live-runner instrumentation. The prefill
routing distribution used uniform-random expert assignment (this
microbench's own synthetic routing), not the real trained gate's actual
(likely skewed) distribution — the tile-boundary-cliff argument would only
strengthen under real skew (more experts get very small or very large
row-counts, both of which are worse for tile efficiency than a tight uniform
band), but this was not verified against real production routing logs.

## Not shipped / explicitly out of scope this session

Per the task brief, a fix was not required and none is proposed here. The
closed-levers table already rules out the two most obvious "fix" directions
for the prefill compute-efficiency gap found above:

- **MoE tile-geometry retune (bm>16), MAXBE widening**: already closed,
  "Kernel already at/above theoretical ceiling" (§3.4) — this was checked
  at decode shape/dense-comparison, not at this session's specific prefill
  tile-boundary framing, but the closure reasoning (kernel already
  well-tuned by Apple/MLX) plausibly transfers.
- **Tile-boundary/routing-uniformity fix**: Lever-1's own conclusion is
  explicit — "capturing it requires equalizing every expert's row count = a
  router/load-balancing change that alters model outputs, not a kernel
  change." Not revisited here; no new information changes that verdict.

No live A/B was run. This session's contribution is attribution data (what
the bottleneck category is), not a validated fix.

## Artifacts

- `bench/p13_switch_mlp_subphase_capture.py` — the probe script (repo,
  uncommitted, standalone process, no production dependency to run again).
- `tmp/p13-20260914/results.json` — raw JSON from the 3rd (final) run.
- This doc.

## Genuinely open after this session

1. **The 33-41%-of-peak prefill compute-efficiency gap's root cause is
   FLAGGED, not proven — and may be TWO compounding mechanisms, not one.**
   The tile-boundary-cliff connection (Lever-1) is a plausible,
   mechanism-consistent explanation reusing an already-established finding,
   not a newly-verified causal link; the roofline's 15.21 TFLOPS reference
   peak is itself a large-dense-GEMM number that may not transfer cleanly to
   a batch of 256 small irregular-M GEMMs, independent of tile-boundary
   waste. Nobody has directly measured per-expert-row-count vs
   achieved-TFLOPS, or isolated a small-M-only (tile-aligned) compute
   ceiling to separate these two candidate mechanisms. That decomposition
   would be the natural next step if this thread is prioritized further.
2. **A genuinely idle-cluster remeasurement** would tighten every absolute
   number in this doc; not performed (production never went idle during
   this session).
3. **Real (non-uniform) production routing distribution** was not checked
   against the trained gate's actual output skew — only assumed uniform as
   a first approximation.
4. Xcode's GPU Frame Capture Performance-tab limiter classification (the
   most direct possible answer to "compute/memory/occupancy-bound") remains
   unreachable from this environment (no VNC, no passwordless sudo) — flagged
   as a tooling gap, not attempted.
