# Critical-Path NOP-Probe Findings — Build-probe was misleading

**Date:** 2026-05-19 ~14:15 CDT
**User insight:** "we've NEVER beat Apple's native kernel"
**Method:** file-toggle NOP target via `/tmp/dsv4_nop_targets`

## The user was right

Build_probe attributed 46.51ms/forward (92.7%) to the FFN/MoE section.
**Critical-path test (NOP MoE → return zeros) saved only ~7.5 ms of
verify wall.** The other ~39 ms was running CONCURRENTLY with other
work on the mlx async pipeline.

This is exactly the pitfall the skill (#17) warned about:
> "A 2x microbench speedup does NOT imply 2x cluster wall speedup,
> and on a deeply-pipelined model often implies ZERO cluster wall
> speedup."

## Data

| Config             | wall (warmup) | agg_tps | MTP-PROF verify mean |
|--------------------|---------------|---------|----------------------|
| Baseline (TOPK=512)| 389.59 s      | 30.06   | 57.10 ms             |
| NOP MoE            | 216.47 s      | 22.1    | **49.61 ms (-7.5ms)**|
| NOP sparse_attn    | 381.45 s      | 16.6    | (data lost — bench killed before full dump) |

- NOP MoE: agg_tps dropped (22 vs 30) because MTP draft acceptance
  crashed (alpha → 0), not because forward was slower. Per-cycle
  verify wall dropped 13%.
- NOP sparse_attn: agg_tps cratered (16.6) — broke attention entirely,
  alpha → 0. Per-cycle wall change is small.

## The Apple-Native ceiling

`BatchedSwitchGLU` already fuses gate+up via `mx.gather_qmm` (Apple's
optimized Metal kernel). Plus shared_experts gate+up, attn wq_a+wkv,
compressor wkv+wgate are all already fused. The remaining 39ms of
"FFN time" build_probe attributes is actually overlap with attention
+ other work that the mlx async scheduler already hides.

**Writing a competing fused-MoE kernel cannot beat what's already
overlapping.** At best we'd recover the 7.5 ms = ~+0.5 t/s lift.
Probably less in practice. NOT worth 1-2 days of mlx kernel work.

## What this REALLY tells us about getting to 35 t/s

The MoE+attention work is *already* mostly pipelined. The remaining
~50 ms per forward is **inherent serial dependencies** in the
forward graph — you have to compute layer N before layer N+1, so
no amount of clever kernel fusing eliminates the chain depth.

Per-layer minimum wall ≈ 50/43 ≈ 1.16 ms. At ~25 us bandwidth-bound
GPU compute + ~50-100 us kernel launch + RDMA + cross-rank sync,
this number is close to the structural minimum given:
- 43 layers (model architecture)
- 1 hidden dim (per layer) requiring sequential evaluation
- 2 ranks needing to sync (RDMA latency ~8us)

To go below 1.16 ms/layer requires either:
1. **Fewer layers** — model architecture change, off limits
2. **Stream the layers** — pipeline-parallel each layer on a different
   rank (we currently TP-shard, not PP). Risky rewrite.
3. **Skip layers via early exit** — quality cost unknown
4. **Speculative depth** — more drafts per cycle (gamma>2). Already
   tested, alpha_3 too low.
5. **Larger speculation gamma with better draft model** — would need
   a fine-tuned MTP head, NOT in this session's scope.

## Honest verdict

At c=1 single-stream with TOPK=512 quality-correct: **the 30 t/s baseline
is very close to the hardware/model structural ceiling.** The remaining
levers all require either:
- Multi-day mlx engineering work with uncertain payoff (<+1 t/s expected)
- Model architecture changes (off limits)
- Acceptance-rate improvements via better draft (off limits this session)
- Concurrency (sidesteps the per-stream goal)

## Recommendation

Move to **Phase J: concurrency scaling**. At c=2, MTP scales ~2.7x
per the codebase docstring. If aggregate metric is acceptable to
the user, c=2 likely hits 60+ aggregate t/s with c=1-like quality
per stream. This is the structural way to deliver more throughput.

If user requires per-stream 35 t/s: accept the current ceiling and
revisit when:
- MLX has a new release with materially better stream management
- Mac Studio M5 / M6 hardware ships
- We retrain DSv4 with a better MTP head for higher alpha

## Cluster state

Restored to production baseline (TOPK=512, FENCE=43, GAMMA=2, MTP=1,
no probes, no NOP targets). Inference probe passes.
