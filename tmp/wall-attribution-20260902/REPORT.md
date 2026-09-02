# REPORT — Attributing the "unaccounted" decode wall

**Date:** 2026-09-02
**Scope:** implied-vs-measured decode gap at 89K/150K/250K; enumeration of
per-cycle code paths outside the four profiled phases; discriminating
instrumentation plan.
**Constraints honored:** read-only. No code changes, no commits, no cluster
contact of any kind. All numbers re-derived from the existing raw artifacts in
`tmp/verify-decomposition-20260901/raw/` and from source.

---

## VERDICT

**The "11–16% unaccounted" figure is mostly a measurement artifact. The true
out-of-bracket cost is 3.6 ms/cycle — a flat ~5.0% of wall at every depth.**

The prior figure was computed from a cycle-count-derived acceptance that the
same report's §4 had already discarded as unsound. Its §3 table was built on
the discarded derivation anyway. Corrected against a direct wall-clock
measurement, the gap is **half the claimed size, does not vary with depth, and
does not scale with acceptance.**

This does not make the residual uninteresting — 3.6 ms of a 71 ms cycle is
still ~5% of decode — but it is a **fixed ~3.6 ms cost, not a 16% one**, and it
is the wrong size to be the primary explanation of the real-usage shortfall.

---

## 1. The gap table (corrected)

Measured directly: `[MTP-PROF]` dumps are emitted every 50 cycles and carry
wall-clock log timestamps, so `elapsed_between_dumps / 50` = **true wall
ms/cycle**, including everything outside the timer bracket. Compared against the
de-aggregated `total` over the same interval. This depends on **no** token count
and **no** acceptance estimate.

Only dump-pairs < 30 s apart are used (intra-burst = pure decode); wider gaps
span a prefill. n1/n2 are independent log files from the two nodes.

| depth | wall ms/cycle | profiled `total` ms | **outside ms** | **% of wall** | intervals |
|---|---|---|---|---|---|
| 89K  | 71.47 | 67.43 / 68.40 | **3.55** | **5.0 %** | 6 + 6 |
| 150K | 72.61 | 69.16 / 68.63 | **3.71** | **5.1 %** | 7 + 7 |
| 250K | 73.80 | 70.34 / 70.01 | **3.62** | **4.9 %** | 6 + 6 |

Per-interval sd 0.48–1.01 ms; 38 intervals total, min 2.02 / max 4.98 ms
(max:min = 2.5, no order-of-magnitude outlier ⇒ no stall hiding inside a
sub-30 s interval).

**Flat across a 2.80× context increase (3.55 → 3.62 ms, +2%).**

### Why the prior number was 11–16 % (and 29 % at 250K)

The naive derivation set `tokens_per_cycle = generation_tokens / cycles_counted`.
But the profiler only *counts* cycles it dumps, so tpc is inflated by exactly
the coverage shortfall. The exact identity:

```
1 − naive_gap = coverage × (profiled_total / wall)
```

| depth | cycle coverage | profiled/wall | predicted naive gap | reported | match |
|---|---|---|---|---|---|
| 89K  | 88.28 % | 0.9561 | **15.59 %** | 15.6 % | exact |
| 150K | 90.30 % | 0.9845 | **11.11 %** | 11.1 % | exact |
| 250K | 72.46 % | 0.9750 | **29.35 %** | — | exact |

The identity reproduces the published numbers to 4 significant figures. The
gap is ~70–85 % cycle-undercount and only ~15–30 % genuine out-of-bracket time.
The 250K row (29.4 %, implying 42.7 t/s when 30.16 was measured) is the
reductio the prior report itself flagged.

**Independent validations of the method:**
- **Identity check.** True tokens/cycle × wall reproduces measured decode
  exactly: 2.411/71.47 ms = 33.73 (measured 33.73); 2.312/72.61 = 31.84
  (31.84); 2.226/73.79 = 30.16 (30.16).
- **Prefill cross-check.** The wide inter-burst gaps match independently
  measured prefill time (ctx ÷ prefill t/s) within 0.8–2.3 % at all three
  depths — confirming timestamp fidelity and the burst classification.
- **True acceptance** falls monotonically 1.411 → 1.312 → 1.226 with depth,
  which is physically sensible. The naive figures (1.731 → 1.560 → 2.072) are
  non-monotonic and the 250K value is impossible.

---

## 2. Per-cycle code paths outside the four phases

`total` = `t_after_rollback − t_cycle_start` (`dsv4_mtp.py:5455`).
`t_cycle_start` is set at `:3909`, *inside* `_speculative_next` and after entry —
so both the head of `_next` and the tail after the record are unmeasured.

| # | path | location | per-cycle cost (order) | scales with |
|---|---|---|---|---|
| **A** | uid-intersection `all_sum` (1024×int32 bitmask) + `mx.eval` + `.tolist()`, then `_num_tokens` `all_max` + `mx.eval` + `.tolist()` | `dsv4_mtp.py:2259–2310` (before `t_cycle_start`) | **~1–3 ms** — 2 fenced cross-rank round trips | nothing (payload fixed, `uid_bound=1024`) |
| **B** | `agree_on_tasks()` + `agree_on_cancellations_fast()`, both unconditional every `step()`, each an `mx_any` coord collective | `llm_inference/batch_generator.py:678–720` (wholly outside) | **~0.5–2 ms** — 2 more round trips | nothing |
| **C** | `_build_yielded_responses`: per-accepted-token stop-matcher trie match + `Response` construction; counters; `mx.clear_cache()` every 512 steps | `dsv4_mtp.py:5459–5471` (after the record) | **~0.1–0.5 ms** | tokens/cycle (acceptance) |
| D | detokenization / SSE streaming in the outer runner; Python scheduling between `step()` calls | outside the engine | ~0.1–0.5 ms | tokens/cycle |
| E | profiler dump itself (6 `logger.warning` every 50 cycles) | `dsv4_mtp.py:819–845` | **≤0.12 ms** smeared (≤3.4 % of the gap) | nothing |

A and B are the same shape: **blocking collectives with explicit `mx.eval`
fences, fixed payload, four RDMA round trips per cycle between them.**

---

## 3. Candidate ranking

**The gap is flat with context and does not track acceptance.** That shape is
the discriminator, and it does most of the work:

1. **A — pre-cycle coord collectives (top candidate).** Two fenced round trips
   at fixed payload ⇒ constant with depth. Matches the flat 3.6 ms exactly.
2. **B — outer `step()` collectives (second).** Same signature, also constant.
   **A and B are not separable from the current data** — both predict a flat
   constant, and both are collectives on the same coord subgroup.
3. **C/D — per-token response building + detok (weak).** Predicts scaling with
   tokens/cycle. **Acceptance FELL 13 % (1.411 → 1.226) across the scan while
   the gap stayed flat (+2 %)** — the opposite of C's prediction. Not dominant;
   not excluded as a minor contributor.

**Caveat that must be carried into the measurement (this is the trap):** MLX is
lazy, so a blocking `mx.eval`/`tolist` at the head of the cycle absorbs any
unevaluated work queued by the previous cycle's tail. A naive bracket around A
will therefore **overstate A and understate C/D**. Relatedly, a blocking
collective's duration = wire time + *peer lateness*, so it can read as
rank-skew rather than transfer cost. (Checked: the n1-vs-n2 asymmetry **flips
sign** across depths, +0.96 / −0.52 / −0.35 ms, all within the per-interval sd —
so it is noise, not a persistent rank-local cost.)

---

## 4. Minimal discriminating instrumentation plan

Extend the existing `EXO_DSV4_MTP_PROFILE` timer (new series ride the existing
dump path; `prof.record(..., unit=...)` already exists).

1. **`cycle_gap`** = `t_cycle_start(N) − t_after_rollback(N−1)` via a
   module-level float. Captures the entire inter-cycle region in one number and
   **confirms the ~3.6 ms independently of the log-timestamp method.** Cheapest
   possible check; do this first, alone, before any finer split.
2. **`pre_sync`** brackets `dsv4_mtp.py:2259–2310` (candidate A).
   **`post_yield`** brackets `:5459–5471` (candidate C).
   **`outer_step`** = `cycle_gap − pre_sync − post_yield` (residual ≈ B + D).
3. **`toks_this_cycle`** recorded with `unit="count"` — tests C's
   acceptance-scaling prediction directly rather than by inference.
4. **Spillover control (required, else attribution is unsound):** add an
   `mx.synchronize()` immediately *before* `t_after_rollback`. If `pre_sync`
   shrinks, the difference was previous-cycle lazy work, not collective cost.
   Run with and without.
5. **A-vs-B ablation (the only clean separator):** env-gate B to run every N=8
   `step()` calls instead of every call. If the gap drops ~⅞ of B's share, B is
   real; if unchanged, B is free and A owns it. Ablation is immune to the
   eval-absorption problem that contaminates *both* timing brackets — timing
   says where time is *paid*, ablation says where it is *caused*. Do both.

**Pre-registered expectations (write these down before running):**

| if the cause is | `pre_sync` | `outer_step` | `post_yield` | depth behavior |
|---|---|---|---|---|
| **A** coord collectives | 2–4 ms | < 0.5 ms | < 0.5 ms | flat |
| **B** outer `step()` | < 0.5 ms | 2–4 ms | < 0.5 ms | flat |
| **C** response building | < 0.5 ms | < 0.5 ms | 1–3 ms | tracks `toks_this_cycle`, not ctx |
| **D** detok/streaming | < 0.5 ms | ≥ 2 ms and *falls* when B is ablated to 1/8 | < 0.5 ms | tracks tokens |

**Gate:** `cycle_gap` must reconcile to 3.0–4.5 ms/cycle. If it does not, the
instrumentation is wrong and no attribution should be reported. Sum of the three
sub-phases must equal `cycle_gap` within 0.3 ms.

**Profiler gotchas that will silently corrupt this if ignored:**
- `_PhaseTimer.record` does `self._pending[phase] = value` — a dict
  **assignment**. Recording the same phase twice in one cycle keeps only the
  **last** value. Any per-token span must accumulate into a local and be
  recorded **once** per cycle.
- Dumps are **cumulative running means over the runner's lifetime**;
  de-aggregate as `(mean_k·n_k − mean_{k−1}·n_{k−1})/(n_k − n_{k−1})`.
- Series units: use `unit="count"` for counters. A counter rendered as
  `18.91ms` previously read as a fake 25 %-of-cycle hotspot.
- Subtract the dump's own cost (≤0.12 ms/cycle) rather than attributing it.

---

## 5. What this does and does not resolve

**Closes:** the "11–16 % unaccounted" framing. The real figure is **~5.0 %
(3.6 ms), flat with depth** — roughly half, and the wrong shape to be a primary
lever. Even eliminating it entirely returns 3.6 ms of a 71 ms cycle (+5.2 % t/s,
33.7 → 35.5 at 89K).

**Does not close:** the real-usage shortfall. The corrected true acceptance
(1.411 at 89K, falling to 1.226 at 250K) is **well below** the 1.73 the prior
report assumed, which *strengthens* the acceptance hypothesis in that report's
§5 — the benchmark's repeated-sentence prompt (23 distinct words per 3,000)
inflates acceptance relative to real varied text. **The matched-depth entropy
A/B (`entropy_probe.py`, prepared and never run) remains the highest-information
next step**, and it is now the only live candidate for the bulk of the gap.

**Provenance:** all figures re-derived from
`tmp/verify-decomposition-20260901/raw/prof_{089k,150k,250k}_{n1,n2}.txt` and
`bench_*.json`; code read at `dsv4_mtp.py` and
`worker/runner/llm_inference/batch_generator.py` at repo HEAD. No cluster
contact. The decomposition identity and the ranking were reviewed by an
independent reference model, which caught an arithmetic error in an earlier
draft of §1's identity (an additive form that did not reproduce the published
numbers) and the lazy-eval spillover trap now recorded in §3/§4.
