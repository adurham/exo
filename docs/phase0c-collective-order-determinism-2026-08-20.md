PHASE 0c: cross-rank collective issue-order determinism — EVAL ORDER, not program order (2026-08-20)
======================================================================================================

Question
--------

Prerequisite for Lever 2 (sequence-chunk pipelining, see
`docs/lever2-seqchunk-overlap-2026-08-20.md`). If each rank builds TWO
INDEPENDENT subgraphs (chunk A and chunk B), each containing 3
`mx.distributed.all_sum` calls, are the 6 collectives matched across
ranks deterministically — and is the matching key (a) program/issue
order, (b) eval order, or (c) unspecified?

Answer (measured, 2 genuine ranks, `mlx.launch -n 2 --backend ring`)
--------------------------------------------------------------------

**Collectives are matched POSITIONALLY in EVAL ORDER. Program/issue
order is irrelevant. There is no tag/name matching. A cross-rank eval-
order divergence causes SILENT WRONG RESULTS — no hang, no error.**

Method: every `all_sum` carries a distinct constant tag payload
(A = 101/103/107, B = 211/223/227). Correct pairing ⇒ `size*tag`. Any
mispairing produces `tag_i + tag_j`, which uniquely identifies both
mispaired partners. 20 trials per scenario, per rank, both ranks'
observations dumped to JSON and diffed.

| scenario | issue order | eval order | result |
|---|---|---|---|
| `same_order` | same | same | **PASS** (202,206,214,422,446,454) |
| `interleaved` (A0,B0,A1,B1,A2,B2) | same | same | **PASS** |
| `async_eval_same` (async_eval A then B, both ranks) | same | same | **PASS** |
| `issue_skew` (r0 builds A first, r1 builds B first) | **DIVERGENT** | same | **PASS** |
| `async_eval_skew` (same build; r0 `async_eval(A);async_eval(B)`, r1 reversed) | same | **DIVERGENT** | **FAIL — silent corruption** |
| `eval_arg_skew` (same build; r0 `mx.eval(*a,*b)`, r1 `mx.eval(*b,*a)`) | same | **DIVERGENT** | **FAIL — silent corruption** |

Failure signature (identical on both ranks, all 20/20 trials,
1 distinct signature ⇒ *deterministically wrong*, not racy):

```
observed = [312, 326, 334, 312, 326, 334]
        312 = 101+211,  326 = 103+223,  334 = 107+227
```

i.e. rank0's A_k was matched with rank1's B_k. Pure positional pairing
of the *eval-issue* stream. Payloads were fully uniform (`uniform:
True`) — the wrong value looks perfectly well-formed.

Three conclusions that matter for Lever 2
-----------------------------------------

1. **Determinism holds, but it's keyed on eval order.** `issue_skew`
   passing is the load-bearing result: the two ranks built the
   subgraphs in *opposite Python order* and still paired correctly,
   because both called `mx.eval(*a, *b)`. So MLX does NOT record
   wire order at graph-construction time — it records it when the
   collective op is scheduled for evaluation.

2. **`async_eval` is safe only if every rank calls it in the same
   order.** `async_eval_same` passes; `async_eval_skew` corrupts.
   Since Lever 2's whole mechanism is staggered `async_eval` per
   chunk, this is the single highest-risk failure mode of the design.
   Any rank-dependent branch (`if rank == 0:`), any data-dependent
   chunk-scheduling heuristic, any early-exit that skips a chunk on
   one rank, silently produces garbage.

3. **The failure is silent and deterministic — worst possible shape.**
   No hang, no exception, no NaN, values are uniform and plausible.
   It will not show up as a crash; it shows up as slightly-wrong
   logits. Any Lever-2 integration MUST carry a tagged-collective
   correctness assertion in its test suite (this probe is that
   assertion), because normal loss/output eyeballing will not catch it.

Implication / guardrail for the real integration
------------------------------------------------

The chunked forward pass must have a **rank-invariant eval schedule**:
the sequence of `async_eval`/`mx.eval` calls (and their argument
order) must be a pure function of the *shape* of the workload, never
of the rank, never of runtime timing, never of any value that could
differ across ranks. Recommended: build the per-layer chunk eval
schedule as an explicit deterministic list up-front, assert it is
identical on all ranks (e.g. hash it and all_gather the hash) once at
startup, then replay it.

Caveats
-------

- Laptop loopback (`--backend ring`, MacBook Pro M4 Max, 2 ranks
  sharing one GPU). This is a *semantic* result about MLX's collective
  matching, not a perf number, so loopback is fine — but it has not
  been re-verified against the cluster's jaccl/RDMA backend, which is
  a different `Group` implementation. Worth a 5-minute re-run there
  before Lever 2 lands, since jaccl could in principle tag its ops.
- 2 ranks only (the laptop can't do more meaningfully). Positional
  matching should generalize, but N>2 mispairing modes were not
  enumerated.
- Only `all_sum` tested; `all_gather` / `sum_scatter` assumed to share
  the same scheduling path but not probed.

Files
-----

`bench/phase0c_collective_order_determinism.py` — 6 scenarios, tagged
payloads, per-rank JSON output to `$P0C_OUT` (default `/tmp/phase0c`),
exits 3 on mispairing so it works as a CI gate.

```
P0C_SCEN=async_eval_skew .venv/bin/mlx.launch -n 2 --backend ring \
    bench/phase0c_collective_order_determinism.py
```
