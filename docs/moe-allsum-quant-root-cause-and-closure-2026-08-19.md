Quantized moe.all_sum: root cause found, then the whole approach found mathematically dead (2026-08-19, final)
====================================================================================================================

Summary
-------

Full investigation completed per the Fable-consulted plan (Phase 0-4).
Real root cause of the live crash found. Then, in checking whether a
fix exists, found the underlying premise of the entire lever is
mathematically incorrect for this specific collective -- not a bug,
a structural mismatch between int8-on-wire arithmetic and what
`moe.all_sum` needs to compute.

Root cause of the crash (found, not guessed)
--------------------------------------------------

Two parallel investigations converged:

1. **Our quant/dequant math is exonerated.** A 2-rank repro using
   MLX's `ring` distributed backend (`mlx.launch -n 2 --backend ring`)
   -- a genuine local multi-process test that does NOT touch jaccl --
   ran the full quantize -> gather -> dequant round trip end-to-end,
   bit-identical across ranks, including a production-faithful lazy-
   eval 32-layer simulation. Committed:
   `bench/moe_allsum_quant_repro.py`,
   `docs/moe-allsum-quant-phase0-repro-2026-08-19.md` (commit `a1bb80801`).

2. **jaccl's `all_gather` has NO reliability layer at all.** Confirmed
   directly from source (`mlx/distributed/jaccl/lib/jaccl/mesh_impl.h`,
   `jaccl.cpp`): `all_sum`/`all_max`/`all_min` route through
   `MeshGroup::all_reduce` -> soft-RC (ARQ retransmit, TCP-confirmed
   start barrier) -- the hardened path this whole cluster relies on all
   night. `all_gather` bypasses ALL of that: raw UC posts, a `StallWatch`
   that throws on any stall, no retransmit on a lost frame. A single
   dropped RDMA frame on `all_gather` = hard throw, not a self-healing
   retry. This directly explains why replacing one reliable `all_sum`
   with two `all_gather`s (data + scales) crashed almost immediately --
   it moved required traffic off the only path built to survive real
   packet loss on this hardware.

Why a fix doesn't exist for THIS collective
------------------------------------------------

Investigated whether a redesign could stay 100% on the reliable
`all_sum` path (which DOES natively support int8 as a wire dtype,
confirmed via `jaccl.cpp:20-47`'s `dtype_to_jaccl_dtype` switch) while
still reducing bytes. Consulted twice on this specifically.

**The answer: no, for a structural reason, not an engineering gap.**
`moe.all_sum` at DSv4-Flash's TP=2 MoE layer is a genuine PARTIAL-SUM
reduction -- under expert-parallel sharding inside TP, each rank
processes a disjoint subset of experts and produces a partial
contribution to the SAME full-hidden-dim output vector for every
token; the two ranks' outputs must be arithmetically summed, not
concatenated/gathered. This is confirmed directly from the call site
(`deepseek_v4.py` ~2834: `y = mx.distributed.all_sum(y, ...)` on the
POST-combine MoE output, same shape both ranks).

The one design that DOES work on the reliable path (zero-padded
int8 all_sum, exploiting that `0 + q = q` is exact and avoids overflow
entirely) only applies to GATHER-semantics collectives -- where each
rank holds a disjoint slice of the final output and is being combined
by placement, not arithmetic. `moe.all_sum` is not that; it's a true
elementwise sum of two full, overlapping-shape partials. Zero-padding
would require each rank to know in advance which OUTPUT ELEMENTS
belong to it exclusively -- which isn't true here; both ranks
contribute to every element.

Naive int8-code all_sum (summing the raw int8 codes directly across
ranks) is independently, separately broken: int8 addition overflows
immediately for realistic MoE partial magnitudes, and even if it
didn't, differing per-rank quantization scales make `q1 + q2` not
equal to `quantize(y1 + y2)` -- there is no scale-recovery step that
fixes this after the fact.

Conclusion
-------------

**The quantized moe.all_sum lever is closed as mathematically
infeasible for this specific collective, not merely buggy.** This is
a genuine, different outcome from "needs a bug fix" -- there is no
known-correct wire-format that both (a) reduces bytes and (b) stays on
jaccl's reliable path, for a true cross-rank partial-sum reduction.
Do not re-attempt without a fundamentally different mechanism (e.g. a
transport-level fix to make all_gather reliable, which is a much
larger jaccl-core engineering project, out of scope for a config-level
optimization).

The underlying finding that all_sum eats 61-64% of prefill wall time
(`docs/moe-all-sum-dominant-cost-2026-08-19.md`) remains real and
unchanged. The compute/comm-overlap lever
(`docs/gpu-util-vs-allsum-cost-reconciled-2026-08-19.md`) also remains
open but requires real graph-restructuring work, not a quick win.
The one still-open, bounded, lower-risk lever from earlier tonight is
the RDMA chunk-size bump (`MLX_JACCL_RELIABLE_MAX_SZ` from 2 to 3,
still inside the documented safe zone) -- that's the most promising
remaining concrete next step on this thread.

Cluster status: untouched throughout this investigation (all work was
code-reading + local 2-rank ring-backend testing, no live-cluster
contact). Standing baseline config confirmed healthy before this
investigation began and unaffected by it.
