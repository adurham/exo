# Overnight: the 250K gap is CYCLE COST, and the decode-fence lever is already banked (2026-09-23)

## The decomposition (measured, counters are cumulative so DELTAS are used)

`tokens/cycle = 1 + acc/cycle`, and `tps = tokens_per_cycle / (ms_per_cycle/1000)`.

| depth | acc/cycle | tok/cycle | ms/cycle | tps |
|---|---|---|---|---|
| 30,174-32,687 (n=4) | 0.928 | 1.928 | 56.5 | 34.10 |
| 273,626 | 0.896 | 1.896 | 69.5 | 27.29 |

- acceptance: 0.928 -> 0.896 (**-3.5%**)
- ms/cycle: 56.5 -> 69.5 (**+22.9%**)

**The depth decline is cycle cost, not draft quality.** This corrects my own
earlier "acceptance cliff" story (which was built on a 46.30 t/s outlier that a
n=4 variance test later showed to be unrepresentative — real value at ~30K is
34.89 +- 0.64).

## What it takes to hit >30 t/s at 250K

From ms/cycle = 69.5 and tok/cycle = 1.896:
- need ms/cycle <= 63.2 (cut 9.0%), OR
- raise acc/cycle from 0.896 to 1.084 (+21%, i.e. beyond the 30K value)

## Candidate: the decode-fence lever — ALREADY ON

`mlx-lm/mlx_lm/models/deepseek_v4.py` (~line 131-150, 3809-3825) documents the
per-layer decode fence: a BLOCKING `mx.eval(y)` at the MoE all_sum site, paid
**44 times per decode cycle**. `EXO_DSV4_FENCE_ASYNC=1` replaces it with
`mx.async_eval(y)`, letting the CPU encode layer n+1 while the GPU runs layer n.

Its own A/B (2026-07-02, in the source comment): **c=1 decode 28.9 -> 37.0 t/s,
outputs byte-identical.** That is a +28% win and it is the single largest known
decode lever in this codebase.

**Status: already live.** Verified on the running runner:
```
EXO_DSV4_FENCE_ASYNC=1
EXO_DSV4_FENCE_ASYNC_C2=0
EXO_DSV4_FENCE_EVERY_N_LAYERS=4
```
So the 27-29 t/s measured at 250K is WITH the async fence already applied —
there is no unclaimed +28% sitting here.

## Remaining adjacency: FENCE_EVERY_N_LAYERS

The launcher comment (start_cluster.sh ~L496-502) states the fence cadence:
- `=1` -> per-layer fences: slower, maximally stable
- `=4` (current) -> costs ~0.7 t/s on c=1 (29.7 -> ~29.0) to unlock c=2 stability
- `=8` -> "recovers c=1 ceiling" but costs c=2 bistability
- `=43` -> only when running gamma=1

So `4 -> 8` is worth roughly +0.7-1.0 t/s (about +3%) on c=1. It is a real but
SMALL lever, and it trades away the c=2 stability property that the default was
deliberately chosen for. Our current workload is c=1 (all probes are single
requests), so the trade is arguably free for the user's actual usage — but the
gain (~+0.7 t/s) does not close a 3.2 t/s gap to 30, and it is a documented-
stability trade. Recommend NOT flipping it silently; surface it as an option.

## The per-cycle growth term, for future work

The one term that grows with depth per cycle is the indexer scoring over pooled
entries: ~234 pooled entries at 30K vs ~2,135 at 273K (9x). `EXO_DSV4_INDEX_TOPK`
would reduce SDPA work but is **FORBIDDEN below 512** (skill pitfall #49: quality
regression). There IS an unused diagnostic for exactly this question:
`EXO_DSV4_TOPK_OVERLAP_LOG=1` measures the Jaccard overlap between consecutive
steps' selected top-k sets — if overlap is consistently >90%, full O(context)
rescoring every step is largely redundant and "stale top-k reuse" (rescore every
N steps + cheap incremental scoring between) becomes viable. That is the
highest-value next investigation for the 250K target, and it is read-only.

## Status vs the user's two asks

- **(a) stop the 500K collapse** — root-caused and fixed: the byte cap was
  structurally unreachable (3570x accounting undercount on CacheList layers),
  fixed in 89ebdbff0, cap set to fire only above ~300K. Verification in flight.
- **(b) 250K > 30 t/s** — NOT met. Currently ~27-29 t/s at 250K. The gap is
  cycle cost (+22.9%), the big fence lever is already banked, and the remaining
  adjacent lever (FENCE_EVERY_N_LAYERS 4->8) is worth only ~+0.7 t/s with a
  stability trade. Honest position: needs the stale-top-k-reuse investigation.
