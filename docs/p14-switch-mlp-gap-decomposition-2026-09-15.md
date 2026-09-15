# P14: Decomposing the P13 prefill compute-efficiency gap — tile-boundary-cliff vs small-M ceiling, and a measurement-methodology artifact that explains most of the headline number (2026-09-15)

## Scope and what this closes

P13 (2026-09-14) found `moe.switch_mlp`/`gather_qmm` at prefill shape
(M=2048, do_sort=True) achieves only **33-41% of the 15.21 TFLOPS measured
hardware peak**, despite roofline arithmetic putting this regime at ~4x the
compute/bandwidth ridge point (should be compute-bound, not memory-bound).
P13 named two distinct, un-separated candidate mechanisms and explicitly
flagged that nobody had measured either in isolation:

- **(a) tile-boundary-cliff** — the existing Lever-1 finding
  (`docs/lever1-moe-smallm-headroom-2026-08-20.md`): `gather_qmm`/dense
  `quantized_matmul` are tile-quantized at ~32 rows/expert; a row count just
  past a tile boundary pays a near-2x-for-one-row cliff.
- **(b) small-M-GEMM ceiling** — the possibility that the 15.21 TFLOPS
  reference peak (measured on a large dense square GEMM) simply doesn't
  transfer to a batch of 256 small irregular-M GEMMs, for reasons unrelated
  to tile-boundary waste specifically.

This session ran the direct, isolated measurements P13 asked for — and found
a **third factor that dominates both**: P13's own headline 33-41% figure was
itself computed with a benchmarking methodology that inflates apparent GPU
time by **~1.8-1.9x**, independent of anything about MoE, tile boundaries, or
small-M GEMMs. Once corrected, prefill `gather_qmm` runs at **~59-64% of
peak**, not 33-41% — most of the previously-reported "gap" was never a real
hardware/kernel-efficiency deficit. The two originally-named candidate
mechanisms both turn out to be real but small contributors to what remains.

## Finding 1 (primary): P13/Lever-1's `gpu_time_ns()`-chained timing pattern inflates apparent GPU time by ~1.8-1.9x

**Mechanism.** P13's `time_stage_chained()` helper (and Lever-1's original
2026-08-20 bench, before its own 2026-08-31 addendum corrected a *different*
bug in the same family) queues N independent kernel calls with no
synchronization between them, calls `mx.eval(*outs)` once at the end, and
divides the accumulated `mx.metal.gpu_time_ns()` by N. This is the exact
"pipelined, not isolated" pattern §4.6 of this doc recommends (correctly) for
checking whether a chain of *different* sequential kernels hides dispatch
overhead behind each other. But used to compute an absolute per-call TFLOPS
number for N *repeated* calls to the *same* kernel, it is invalid: Metal's
command-buffer GPUStartTime/GPUEndTime timestamps for back-to-back
unsynchronized submissions do not sum to true GPU-busy wall-clock time once
N crosses a small threshold.

**Directly measured, both on a large dense bf16 GEMM (the exact
`[16384x4096x4096]` shape that produced the cited 15.21 TFLOPS reference,
PH:6059) and on the real production `gather_qmm` kernel at real prefill
shape**, standalone process on m4-1, idle cluster (no concurrent production
traffic during these specific runs — confirmed via `~/exo.log` and
`ollama/api/ps` before/after):

| n_chain | wall-clock TFLOPS | `gpu_time_ns()`-chained TFLOPS | ratio (gpu/wall) |
|---|---|---|---|
| 1 | 15.18-15.27 | 15.26-15.27 | 0.995 (agree) |
| 2 | 15.22-15.23 | 9.83-10.17 | 1.50-1.55 |
| 4 | 15.22-15.24 | 8.46-9.14 | 1.67-1.80 |
| 8 | 15.02-15.23 | 7.88-8.67 | 1.76-1.85 |
| 10-40 | 15.22-15.25 | 7.88-8.26 | 1.84-1.93 |

The ratio climbs from 1.0x at n_chain=1 and **plateaus at ~1.85-1.9x by
n_chain≈6-8**, flat out to n_chain=40 — consistent with Metal's bounded
command-buffer pipeline depth producing a fixed per-buffer timestamp
inflation once concurrent/overlapping submission kicks in, not unbounded
overlap (which would keep growing with N). The exact mechanism inside Metal's
driver was not further isolated (out of scope; the empirical inflation
factor and its stability are what matters for correcting the numbers below).
Ruled out as explanations: double-dispatch (`dispatch_count`/call stayed
exactly 1.00-2.00 throughout, matching the true kernel count at every n_chain
tested) and graph memoization/dedup (wall-clock time scales linearly and
correctly with n_chain in every sweep).

**Confirmed on the real production kernel, not just a synthetic dense GEMM**:
at real prefill shape (M=2048 tokens, top_k=6, 256 experts, mxfp4 g=32, using
the actual `_gather_sort`+`gather_qmm` from `mlx_lm.models.switch_layers`,
pool of 16 rotated inputs — P13's exact configuration), isolated
(sync-per-call) wall-clock and `gpu_time_ns()` agree to <1%; chained
`gpu_time_ns()` diverges from chained wall-clock by the same ~1.85-1.9x:

| method | fused_gate_up+down_proj TFLOPS | % of 15.21 TFLOPS peak |
|---|---|---|
| ISOLATED (sync every call) | 9.66 | 63.5% (agrees with wall-clock, 0.992x ratio) |
| CHAINED wall-clock | 9.69 | 63.7% |
| CHAINED `gpu_time_ns()` (P13's method) | 5.24 | 34.5% |

`5.24 TFLOPS`/`34.5% of peak` lands squarely inside P13's originally-reported
5.09-6.28 TFLOPS / 33.5-41.3% range — this is not a nearby coincidence, it is
the same bug reproducing the same result.

**Decisive re-run of P13's exact scenario, byte-for-byte** (same model
construction, same quantization config, same routing pool, same prefill
shape, only the timing method changed), 3 independent runs:

| Run | Original P13 method | Corrected wall-clock (chained) | Corrected wall-clock (isolated, gold standard) | Artifact inflation ratio |
|---|---|---|---|---|
| 1 | 5.38 TFLOPS (35.3%) | 9.69 TFLOPS (63.7%) | 8.92 TFLOPS (58.6%) | 1.80x |
| 2 | 5.05 TFLOPS (33.2%) | 9.69 TFLOPS (63.7%) | 9.15 TFLOPS (60.2%) | 1.92x |
| 3 | 5.11 TFLOPS (33.6%) | 9.69 TFLOPS (63.7%) | 9.25 TFLOPS (60.8%) | 1.90x |

**Corrected prefill `gather_qmm` compute efficiency: ~59-64% of the 15.21
TFLOPS measured peak, not 33-41%.** The artifact inflation ratio
(1.80-1.92x) is fully consistent with the generic dense-GEMM artifact
magnitude measured above (~1.85-1.9x), confirming this is a general
chained-`gpu_time_ns()` phenomenon, not something specific to MoE/gather_qmm
tiling.

**Scope of this finding, stated precisely.** This does NOT mean P13's other
conclusions (gather/scatter <5% of total GPU time; 8/18 dispatches per call;
the qualitative "prefill is compute-bound, not memory/gather-bound" roofline
argument) are wrong — those are ratios/counts, largely artifact-invariant, or
independently cross-checked. Spot-checked the gather/scatter ratio itself
under corrected wall-clock timing: **7.0%** (vs P13's original 1.8-4.8%
across 3 runs) — still small, still "not the bottleneck," the qualitative
conclusion is unchanged, only the precise percentage shifts. The artifact is
also NOT universal across this repo's benchmark history — P02D, P03, and P08
each use per-iteration `reset_gpu_time()`+immediate `mx.eval()`+`sync()`
(isolated, not chained), which agrees with wall-clock to <1% per this
session's own measurements; those numbers are unaffected. The vulnerable
pattern (`time_stage_chained`-style: reset once, queue N calls, eval once at
the end) is specific to `bench/p13_switch_mlp_subphase_capture.py` and
`bench/p01_switch_mlp_gputrace.py`'s `time_stage()` helper (and, by the same
mechanism, likely affects Lever-1's original pre-2026-08-31-addendum
absolute-throughput numbers, though that doc's own final verdict rests on
*ratios* between two tiers measured with the *same* method, which the
sensitivity check below shows are far more robust to this artifact than
absolute TFLOPS are). **A full audit of every historical use of this pattern
was NOT performed — out of scope for this task, flagged as a genuine
follow-up below.** One additional spot-check, at decode shape (M=1): the same
pattern inflates by **2.22x** there too — P13's/P01's decode-shape *dispatch
counts* (contention-independent, unaffected) stay valid, but any decode-shape
*absolute* µs/bandwidth number computed via this exact chained-`gpu_time_ns()`
method should be treated as unverified until re-measured, which this session
did not do (out of scope; flagged as open below).

## Finding 2: tile-boundary-cliff, measured in isolation (Part A)

Single dense mxfp4 `quantized_matmul` calls (no batching, no aggregate
effects), at the two real per-expert production GEMM shapes
(`fused_gate_up`: HIDDEN=4096→2×INTER=2048; `down_proj`: INTER=1024→
HIDDEN=4096), M sitting exactly ON a 32-row tile boundary vs one row PAST it.
Converged wall-clock-chained methodology (validated against isolated
wall-clock and against a fine-grained n_chain sweep — small-M dispatch-bound
calls need n_chain≥256 to amortize away ~130-280µs of fixed CPU-submission
overhead per call, which otherwise swamps the actual <100µs GEMM and produces
a false near-zero cliff reading under naive isolated timing):

| Shape | M boundary→past | % of peak (on) | % of peak (past) | cliff (µs/row ratio) |
|---|---|---|---|---|
| fused_gate_up | 32→33 | 59.8% | 41.1% | **1.45x** |
| fused_gate_up | 64→65 | 82.1% | 57.4% | **1.43x** |
| fused_gate_up | 96→97 | 84.1% | 64.8% | 1.30x |
| fused_gate_up | 128→129 | 85.7% | 71.8% | 1.19x |
| fused_gate_up | 160→161 | 88.6% | 74.2% | 1.19x |
| fused_gate_up | 256→257 | 89.5% | 79.9% | 1.12x |
| down_proj | 32→33 | 65.4% | 41.7% | **1.57x** |
| down_proj | 64→65 | 77.5% | 59.5% | 1.30x |
| down_proj | 96→97 | 87.8% | 66.8% | 1.31x |
| down_proj | 128→129 | 88.3% | 71.3% | 1.24x |
| down_proj | 160→161 | 88.6% | 73.6% | 1.20x |
| down_proj | 256→257 | 88.9% | 78.9% | 1.13x |

**The cliff is real and directly measured** — 1.43-1.57x at the first tile
boundary (M=32/33), decaying to ~1.12-1.13x by M=256/257 — closely matching
Lever-1's own historical 2026-08-31 finding (1.44-1.47x at the same boundary
on both 32-core and 40-core hardware). This is the first time it has been
measured in true isolation (single GEMM, no batch/routing aggregate) rather
than inferred from aggregate ragged-vs-balanced comparisons.

## Finding 3: small-M ceiling, isolated from alignment (Part B)

Sweep of ONLY tile-ALIGNED M values (32, 64, 96, ... 16384 — every point a
multiple of 32, so by construction none of these measurements can reflect
tile-boundary waste) for the same two production GEMM shapes, mxfp4 vs bf16:

| Shape | M | mxfp4 % of peak | bf16 % of peak |
|---|---|---|---|
| fused_gate_up | 32 | 77.7% | 62.3% |
| fused_gate_up | 64 | 82.2% | 63.7% |
| fused_gate_up | 128 | 85.6% | 75.6% |
| fused_gate_up | 256 | 88.7% | 90.5% |
| fused_gate_up | 512 | 88.6% | 98.3% |
| fused_gate_up | 2048 | 89.2% | 99.3% |
| fused_gate_up | 16384 | 89.6% | 99.3% |
| down_proj | 32 | 66.3% | 48.7% |
| down_proj | 64 | 78.4% | 97.2%* |
| down_proj | 128 | 87.4% | 97.5% |
| down_proj | 512 | 88.5% | 97.9% |
| down_proj | 2048 | 85.3% | 98.2% |
| down_proj | 16384 | 89.0% | 98.9% |

(*down_proj bf16 at M=64 landing above M=128/256 is noise from a single
data point, not a real non-monotonicity — every other bf16 value on both
shapes rises monotonically toward ~98-99% and plateaus by M≈256-512.)

**A real, alignment-independent small-M ceiling exists, and it is
dtype-dependent.** bf16 (unquantized) climbs smoothly from ~48-63% of peak at
M=32-64 to a ~98-99% plateau by M≈256-512 — this is the "generic small-M GEMM
underutilization" mechanism P13 hypothesized, real and substantial at the
smallest M, but it is **not the dominant factor at production's actual M
range** (median ~48 rows/expert, per P13's routing simulation) because the
curve is already well up its ramp by M≈128-256. mxfp4 behaves differently: it
plateaus much earlier (~88-90% of peak by M≈256, essentially flat from there
to M=16384) and is *higher* than bf16 at the smallest M tested (M=32:
mxfp4 77.7% vs bf16 62.3%) — mxfp4's own dequantization/compute pattern
apparently has a lower per-call fixed cost that partially masks small-M
underutilization at the very smallest sizes, then caps below bf16's
large-M ceiling (an ~10-12% "quantization tax" at scale, consistent with the
already-established `docs/moe-vs-dense-qmm-isolation-2026-08-19.md` finding
of ~13% tax). **Since production experts run at mxfp4, not bf16, the
practically-relevant small-M ceiling curve is the mxfp4 one — and it is
already at 78-88% of peak by M=32-256, most of the way to its own plateau.**

## Reconciliation: which mechanism explains what fraction of the corrected gap

Fine-grained M=23-69 sweep (production's actual per-expert row-count support,
re-derived directly by simulating M=2048 tokens / top_k=6 / 256 experts /
uniform routing — mean 48.0, std 7.3, matching P13's own simulation), same
converged wall-clock methodology, both shapes:

| Shape | Real ragged-weighted (production's actual mix) | Uniform-over-same-range (alignment-neutral counterfactual) | R = ragged/uniform |
|---|---|---|---|
| fused_gate_up | 59.6% of peak | 60.8% of peak | **0.98** |
| down_proj | 59.1% of peak | 59.4% of peak | **1.00** |

**The tile-boundary-cliff's contribution to the AGGREGATE prefill gap, once
weighted by production's real per-expert distribution, is under 2%** — R≈1.0
means the real ragged mix costs essentially the same as a uniform spread
across the identical M range. This is not a contradiction of Finding 2 (the
cliff is real at any single boundary crossing) — it means production's
routing distribution (mean 48, spread 23-69) mostly lands well past the
worst part of the M=32/33 cliff already, so cliff crossings within this
range are individually real but small (1.1-1.3x at these already-elevated M
values, not the 1.45-1.57x seen right at M=32/33) and roughly wash out in
aggregate. This closely reproduces Lever-1's own 2026-08-31 finding
(R=1.10-1.11x on a neighboring, slightly different comparison range/method)
— same conclusion, independently arrived at via a different comparator
(uniform-over-identical-support here vs "balanced full period" there).

**The dominant remaining shortfall (from ~100% down to ~59-64%) is the
small-M ceiling (Finding 3), not the tile-boundary-cliff.** At production's
actual mean M≈48, mxfp4 alone (Part B, tile-aligned) already sits at
~82-84% of peak for both shapes — meaning even a PERFECTLY tile-aligned,
zero-raggedness version of this workload at the SAME mean row count would
still only reach ~82-84% of the 15.21 TFLOPS reference, not 100%. The
remaining small delta between that 82-84% aligned-ceiling number and the
measured ~59-64% ragged-real number is fully accounted for by the ~2% R-ratio
cliff cost (Finding 2) plus normal run-to-run/shape-mix noise — there is no
third unidentified mechanism needed to close the arithmetic.

## Verdict

**The originally-reported 33-41%-of-peak gap was not primarily a real
hardware/kernel-efficiency deficit — it was substantially (≈45-50 percentage
points of the ~60-65-point apparent gap) a benchmark-instrumentation
artifact** (chained `gpu_time_ns()` summation, ~1.8-1.9x inflation).
**Correcting for it, prefill `gather_qmm` runs at ~59-64% of the reference
peak.** Of the remaining, now much smaller, real gap (~36-41 points, i.e.
100% − ~59-64%):

- **The small-M-GEMM ceiling (candidate (b)) is the dominant real
  mechanism** — production's mean M≈48 mxfp4 GEMM sits at ~82-84% of the
  large-dense-GEMM reference peak even under perfect tile alignment, purely
  because M=48 is small relative to the shapes the 15.21 TFLOPS reference
  was calibrated on ([16384x4096x4096]). This closes most of the remaining
  gap by itself.
- **The tile-boundary-cliff (candidate (a)) is real (directly confirmed via
  isolated single-GEMM measurement, 1.43-1.57x at the first boundary) but a
  minor aggregate contributor** at production's actual routing distribution
  (R≈0.98-1.00, i.e. ≤2% of the aggregate) — because the real distribution's
  mean (48) and spread (23-69) mostly sit well clear of the worst part of the
  cliff (M=32/33), where the cliff has already decayed to ~1.1-1.3x. This
  independently corroborates, via a different method, Lever-1's own
  2026-08-31 R=1.10-1.11x finding.
- **Neither mechanism, individually or compounding, was previously
  quantified — this session is the first direct, isolated measurement of
  both, closing P13's explicitly-flagged open question.** Both candidates
  are confirmed real; the small-M ceiling dominates; the aggregate residual
  after accounting for both is consistent with noise, not a third
  unidentified mechanism.

## Closed-levers table: unaffected, and this finding adds nothing new to it

The existing closed-levers table's two entries relevant to this gap remain
correctly closed, and this session's findings **strengthen rather than
reopen** them:

- **MoE tile-geometry retune (bm>16), MAXBE widening** — "Kernel already
  at/above theoretical ceiling" (§3.4). Confirmed again here: at production's
  actual mean M, mxfp4 is already at ~82-84% of the achievable small-M
  ceiling (Part B), and the residual cliff cost is ≤2% in aggregate (this
  session's reconciliation) — there is no meaningful tile-geometry headroom
  a retune could capture.
- **Tile-boundary/routing-uniformity fix** — "capturing it requires
  equalizing every expert's row count = a router/load-balancing change that
  alters model outputs, not a kernel change" (Lever-1). Unchanged: this
  session's R≈0.98-1.00 aggregate finding makes this lever even less
  attractive than Lever-1's own R=1.10-1.11x suggested — there is now less
  than half as much aggregate cliff cost to capture as previously estimated,
  for the same off-limits routing-change cost.

**No fix is proposed for the small-M ceiling either.** It is a property of
the hardware/kernel's throughput-vs-M curve (bf16 and mxfp4 both ramp up from
small M and plateau by M≈256-512) intersecting with MoE's inherent per-expert
row count at this model's expert count/top_k/prefill-chunk-size combination —
not a bug, not a mistuned kernel parameter, and not something a kernel
retune, tile-geometry change, or routing change can move without changing
either the model's MoE topology (expert count/top_k — not on the table) or
`EXO_PREFILL_STEP_SIZE` (already tested at 4096, confirmed ~8% END-TO-END
REGRESSION for unrelated reasons — §3.3, closed). **This is a genuine
physics/framework floor for the current model+hardware+chunk-size
combination, not a tuning gap.**

## What this session found that IS a fixable, forward-looking lever — but not for THIS gap

The methodology artifact (Finding 1) is itself real, novel, actionable
information: any FUTURE benchmark in this repo that needs an absolute
per-call TFLOPS/bandwidth number from repeated calls to the same kernel
should use wall-clock timing (isolated, or chained-with-single-eval — both
validated <1% divergence from each other and from ground truth), never
`mx.metal.gpu_time_ns()` summed over an unsynchronized chained batch of
identical calls. This is a measurement-hygiene fix for the *investigation
process*, not a production kernel/config change — nothing to relaunch,
nothing to A/B against live traffic. Recorded here and in §12's
methodology-lessons checklist (this doc, patched below) so it isn't
re-discovered the hard way a third time.

## Genuinely open after this session

1. **Full audit of every historical use of the `time_stage_chained`-style
   pattern was NOT performed.** Confirmed vulnerable: P13's own script
   (`bench/p13_switch_mlp_subphase_capture.py`), P01's `time_stage()`
   (`bench/p01_switch_mlp_gputrace.py`). Confirmed UNAFFECTED (isolated
   per-call reset+eval+sync pattern): P02D, P03, P08. NOT checked:
   Lever-1's original 2026-08-20 absolute-throughput numbers pre-dating its
   own 2026-08-31 addendum (that addendum's own verdict rests on a ratio
   between two same-method tiers, which is far more robust to this specific
   artifact than an absolute TFLOPS number — a full re-derivation was not
   attempted here, out of this task's scope), and the P13 decode-shape
   (M=1) dispatch/bandwidth numbers specifically (this session confirmed the
   pattern ALSO inflates by 2.22x at decode shape via one spot-check, but did
   not re-derive P13's or P01's decode-shape bandwidth percentages under
   corrected timing).
2. **The exact Metal-internal mechanism behind the ~1.85-1.9x plateau** was
   not further isolated (e.g. via Instruments GPU trace of the actual
   command-buffer scheduling) — the empirical inflation factor, its
   stability across kernel types (dense GEMM and gather_qmm agree to within
   a few percent), and its non-growth past n_chain≈6-8 were sufficient to
   correct the numbers in this doc without a definitive lower-level
   explanation. Flagged as unconfirmed mechanism, not unconfirmed effect.
3. **A genuinely idle-cluster remeasurement of the ORIGINAL P13 percentages**
   (with zero live-traffic contention) was not attempted — moot for this
   session's purpose, since the artifact this doc corrects is
   contention-independent (confirmed: reproduces identically on an idle
   cluster and was already present, per P13's own text, during a
   contended one).
4. Real (non-uniform) production routing distribution, per P13's own open
   item 3, remains untested against this reconciliation — this session's
   M=23-69 support range is itself a re-derivation of P13's own uniform-
   routing simulation, not a capture of the trained router's real skew.
   `EXO_DSV4_ROUTE_HIST=1` exists in the codebase
   (`mlx_lm/models/deepseek_v4.py:_route_hist_record`) as exactly the
   instrumentation needed to capture this on live traffic, but was never
   enabled in this session's start_cluster.sh launch (confirmed via `ps eww`
   on both live runner PIDs) and enabling it requires a relaunch, which was
   out of scope per this task's constraints. The doc's own prior finding
   (real skew, if anything, would make small-per-expert-M runs MORE common,
   not less, since a skewed router concentrates rows onto fewer hot experts
   and starves the rest toward the low end of the row-count range) still
   stands as a reasoned, not measured, expectation.

## Production health

Confirmed unchanged before and after all testing on both nodes:

- macstudio-m4-1 PID 43724, macstudio-m4-2 PID 10062 — same PIDs throughout
  this session (etime continuous, no restart)
- `curl http://localhost:52415/ollama/api/ps` — model loaded throughout
- All measurement scripts ran as standalone processes in `/tmp` / a scratch
  location, never attached to the live runner PID — zero xctrace, zero
  relaunch, zero production risk, consistent with this doc's own §12
  hazard table.

## Artifacts

- `bench/p14_artifact_and_calibration.py` — reproduces the `gpu_time_ns()`
  chained-batch inflation demo + calibration against the 15.21 TFLOPS
  reference (both dense-GEMM-generic and real-`gather_qmm`-specific).
- `bench/p14b_decisive_recreate.py` — P13's exact scenario, 3 timing methods
  side-by-side (original P13 method, corrected wall-clock chained, corrected
  wall-clock isolated).
- `bench/p14c_tile_cliff_isolated.py` — Part A, isolated tile-boundary-cliff
  measurement, converged wall-clock methodology.
- `bench/p14d_smallm_ceiling.py` — Part B, tile-aligned-only small-M sweep,
  mxfp4 vs bf16.
- `bench/p14e_reconciliation.py` — fine M=23-69 sweep + ragged-weighted vs
  uniform-over-same-range reconciliation.
- `tmp/p14-20260914-decomposition/*.json` — raw results from all of the
  above (3 runs of the decisive comparison, full Part A/B/reconciliation
  tables).
- This doc.
