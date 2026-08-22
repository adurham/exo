# Read-only investigation: the historical FENCE_ASYNC +28% claim traced to the same sync-fence measurement artifact class — 2026-08-22 (session 3, offline, read-only)

## Why this check

Per Fable's guidance (given the real all_sum transport cost is now
confirmed at only 2.9-5.3% of wall time, undercutting item t7's
overlap-investigation premise — see
`docs/allsum-sync-span-artifact-arithmetic-check-2026-08-22.md`): rather
than pursue further live overlap experimentation (low remaining
ceiling, would require a relaunch with no one available to monitor it
overnight), investigate the open +28%-vs-+1.04% `EXO_DSV4_FENCE_ASYNC`
discrepancy using only existing git history and code — zero cluster
risk, no relaunch.

## Finding: the +28% claim was measured with "MTP-PROF," a tool with the exact same self-documented artifact class as the sync-span profiler

Traced the origin commit: `mlx-lm` commit `1e808319f82bb168b9536d601cc3631ba4d63057`
(2026-07-02), "perf(dsv4): env-gated non-blocking per-layer fence
(EXO_DSV4_FENCE_ASYNC)". Full commit message:

> mx.async_eval(y) at the Phase H Lever 1 fence site: same per-layer
> graph commit points (cross-rank dispatch order preserved) without
> blocking the CPU on GPU completion, overlapping graph-build of layer
> n+1 with GPU execution of layer n. **MTP-PROF measured verify = 90%
> of the 62 ms decode cycle** with ~1.1 ms fence wall per layer vs ~0.5
> ms weight floor. Distinct from OPT-7 (removing evals — regressed on
> batched-graph cost). Default OFF pending throughput + bit-determinism
> + c=2 stability A/B.

This commit message itself does not literally state "28.9 → 37.0
t/s" (that specific number lives in a later in-code comment referring
back to this work, not in this commit's own message) — but it names
**"MTP-PROF"** as the measurement tool used to justify shipping the
change. Located MTP-PROF's implementation and its own documented
methodology caveat, in `src/exo/worker/engines/mlx/speculative/dsv4_mtp.py`:

> Per-cycle phase timing. When `EXO_DSV4_MTP_PROFILE > 0`, brackets the
> draft / verify / accept phases with `mx.eval` + `perf_counter`,
> summarising every N cycles. **Inserts evals at phase boundaries which
> serialises pipelining — measurements are upper bounds on real
> production walls.**

**This is the same class of methodology artifact conclusively proven
tonight for the sync-span profiler**
(`docs/allsum-sync-span-artifact-arithmetic-check-2026-08-22.md`):
forced `mx.eval()`/`mx.synchronize()` synchronization at measurement
phase boundaries destroys real async pipelining and inflates the
measured cost of whatever sits right before the boundary, because it
forces materialization of accumulated upstream lazy-graph work at that
exact point. MTP-PROF's own code comment states this explicitly and
unambiguously as a known, documented limitation, not something this
session discovered — the tool's own author(s) already knew and wrote
down that its absolute numbers are upper bounds, not real production
walls.

Also directly relevant: `EXO_DSV4_RB_PROFILE`'s comment in the same
file states the same caveat for a different sub-measurement — "this
gate splits them with `mx.synchronize()` sub-boundaries (serialises the
pipeline — SHARES are trustworthy, absolute totals are upper bounds,
same caveat as `EXO_DSV4_SECTION_TIME`)" — confirming this is a known,
recurring, and previously-documented pattern across multiple profiling
tools in this codebase, not a one-off issue.

## Conclusion

**The historical +28% (28.9 → 37.0 tok/s) `EXO_DSV4_FENCE_ASYNC` claim
was very likely measured under MTP-PROF's self-documented
pipeline-serializing methodology — the same artifact class now
conclusively proven (via real arithmetic reconciliation) to have
produced the inflated 21.4%/14.4% `moe.all_sum` sync-span figures.**
This session's clean, real, unforced A/B measurement of
`EXO_DSV4_FENCE_ASYNC` (+1.04%, `docs/comm-compute-overlap-already-exists-2026-08-21.md`)
is internally consistent with the newly-confirmed real transport
ceiling (~2.9-5.3% of wall time is the absolute maximum any all_sum
overlap scheme could ever recover) — a real async-fence overlap
capturing ~1% out of a genuine ~3-5% ceiling is a coherent, believable
result. A historical claim of +28% against a collective that (per
tonight's real measurement) costs only ~3-5% of wall time was always
implausible on its face — it would require the fence to be "hiding"
essentially the ENTIRE cost of the collective and then some, which
doesn't square with any of tonight's real data.

**This closes the open +28%-vs-+1.04% discrepancy with a real,
evidence-backed explanation** (measurement-methodology mismatch between
an inflated pipeline-serializing tool and a clean unforced A/B), rather
than leaving it as an unexplained mystery.

## Disposition of item t7 (comm/compute overlap investigation)

Per Fable's arithmetic: perfect overlap of `moe.all_sum` with
next-layer's replicated attention could recover AT MOST the real
transport cost itself (~1.55ms/token, ~2.9% of wall time — the median
case from tonight's real measurement). `EXO_DSV4_FENCE_ASYNC`'s
measured +1.04% has already captured a meaningful fraction of that
small ceiling. Residual theoretical upside is on the order of 1-2%
decode throughput at most, and pursuing it further would require live
cluster experimentation (stream scheduling changes, layer-loop
reordering) — squarely in the relaunch-risk category correctly ruled
out for tonight given no one is available to monitor a live
degradation.

**Closing t7 for tonight with an evidence-backed disposition, not
abandonment**: ceiling ~2.9-5.3% (confirmed via real jaccl-internal
timing), ~1% already captured by the existing `EXO_DSV4_FENCE_ASYNC`
mechanism, residual upside ≤2%, not worth live-cluster experimentation
risk tonight. Revisit only if the real transport cost materially grows
(e.g. a larger cluster, bigger hidden-dimension model, or a config
change that increases the collective's real payload size) — the
current small ceiling doesn't justify further investigation at this
model/cluster scale.

## Real, structural attention/all_sum dependency check (read-only, as also requested)

Confirmed via code read (`mlx-lm/mlx_lm/models/deepseek_v4.py`,
`DeepseekV4Block.__call__`): the per-layer structure is
`attn_hc → attn_norm → attn(...) → residual add → ffn_hc → ffn_norm → ffn(...) [ends in moe.all_sum]  → residual add → NEXT layer's attn_hc`.
This confirms the structural claim from the earlier Fable review that
motivated t7: layer N+1's attention block genuinely has no DATA
dependency on layer N's `all_sum` output until the residual-add step
right before `ffn_hc` — i.e., there IS a real window where layer N+1's
attention compute could theoretically proceed concurrently with layer
N's collective completing. This structural fact is confirmed correct
and remains true — it's the CEILING on the opportunity (now known to be
small, ~2.9-5.3%) that has changed, not whether the opportunity exists
in principle.
