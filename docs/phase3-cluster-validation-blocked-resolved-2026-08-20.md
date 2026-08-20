Phase 3 (real-cluster overlap validation): blocked by hardware, resolved via existing data (2026-08-20)
=============================================================================================================

What happened
-----------------

Attempted a standalone jaccl RDMA microbenchmark (real 2-rank all_sum
vs GPU compute overlap, Fable's Step 2 recommendation) on the live
2-node cluster, running ALONGSIDE the production exo runners rather
than replacing them.

**Result: real RDMA connection established (`jaccl init OK, size=2`
on both ranks), but `all_reduce` failed** with
`[jaccl] all_reduce wc.status=1` + `IOConnectUnmapMemory failed`
errors. Root cause: this cluster's Thunderbolt hardware has exactly
two RDMA interfaces (`rdma_en3`/`rdma_en4`), and production's live
runners already hold them open. A second, independent jaccl session
cannot claim the same physical RDMA queue-pairs concurrently -- this
is a genuine hardware/resource conflict, not a bug in the mechanism
being tested.

Both probe processes exited cleanly (no zombies, no stuck resources).
Confirmed production fully healthy (READY, correct master/worker
roles) immediately after -- no disruption, no wedge, no repeat of the
historical "any RDMA teardown drops the XDomain link" failure mode.
Cleaned up scratch files on all three machines.

Why this is not actually a blocker
---------------------------------------

Consulted Fable on the safe path forward. Key correction: **active
standalone RDMA testing was the mistake, not measurement itself** --
the safe way to get real overlap-cost data is PASSIVE instrumentation
of the live production all_sum calls, which this repo already has and
already ran.

Verified the theoretical ceiling number (Fable's Step 1) was measured
at a representative scale: the `moe.all_sum = 9.5% of wall time`
figure (`dsv4-220k-prefill-span-profile-2026-08-18.md`) comes from a
REAL 220,321-token prefill -- genuinely long context, exactly
DSv4-Flash's target regime, not a short/moderate prompt that could
understate the real comm fraction at production scale. This answers
Fable's flagged risk directly: the ceiling is real and representative,
not stale or under-scale.

**Theoretical ceiling for Phase 2's ENTIRE overlap mechanism, even
with perfect overlap: ~10.5% end-to-end speedup.** Realistic gains
after imperfect overlap, scheduling overhead, and MLX stream-ordering
constraints: likely 5-7%.

Decision
-----------

Given: (a) the ceiling is real, representative, and modest (~10.5%
best case), (b) standalone active testing is blocked by this
cluster's hardware (only 2 RDMA interfaces, both claimed by
production), (c) any further testing would require either a
maintenance window (stopping production, a real cost) or building
passive instrumentation INTO the actual Phase 2 integration and
deploying it live (the highest-risk option, deploying experimental
interruptible-generator-driven code to production for a ~10% ceiling)
-- **Phase 3 is deprioritized, not abandoned.** The effort-to-gain
ratio doesn't justify a risky live deployment tonight for a modest,
capped win.

What was still gained
-------------------------

This entire investigation (Levers 1-3, Phase 0-2) was not wasted even
without a Phase 3 cluster number:
- Lever 1: MoE small-M kernel work correctly closed as NO-GO (real
  headroom analysis, not a guess).
- Lever 3: found and corrected a real, load-bearing measurement error
  from 2026-08-19 (the "178ms/call, 61-64% of wall time" figure was
  arithmetically impossible) -- this alone changes how every future
  all_sum-related investigation should be scoped.
- Phase 1: found and FIXED a real, previously-unknown correctness bug
  (SparseCompressedAttention's chunk-boundary pooling state) --
  a genuine deliverable independent of whether chunked prefill or
  overlap ever ships.
- Phase 0b: proved the overlap mechanism is real and achievable in
  principle (not blocked by a device-wide GPU drain), with the exact
  two escape-hatch conditions needed (pre-evaluated input or second
  stream) -- reusable knowledge for any future comm/compute overlap
  work on this stack, not just this specific lever.
- Phase 0c: found a real, serious silent-corruption hazard (collective
  matching by eval-order, not issue-order) with a concrete guardrail
  -- critical for ANY future MLX distributed work on this fork, not
  just Phase 2.

Standing recommendation
---------------------------

If prefill throughput work resumes later, this thread's real
artifacts (the chunk-boundary fix, the Phase 0 gates, the corrected
all_sum cost model) are the foundation to build from -- do not
re-derive them. The next genuinely promising unexplored lever, per
tonight's own Alternative-Lever finding, is the padded seq-split
all_sum's 2x-wire-bytes tax (chosen over subgroup all_gather to dodge
a UC stuck-send wedge) -- flagged but not investigated tonight.
