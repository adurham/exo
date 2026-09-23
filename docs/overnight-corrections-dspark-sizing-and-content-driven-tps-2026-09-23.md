# Overnight findings 2026-09-22/23 — two corrections to the plan

## Correction A: the DSpark "residual ~6.5-7 GB" target does not exist

The 352K memory doc recommends as lever 1: *"shard the non-expert projections
where dims divide"*, and frames the residual replicated DSpark footprint as
~6.5-7 GB/node. I measured the actual component sizes before writing any code:

| component (full-precision shape) | GB | share of head |
|---|---|---|
| stage FFN (switch_mlp + shared_experts) x3 | 77.613 | **97.8%** |
| everything else (attn x3, main_proj, markov, norms) | 1.770 | 2.2% |
| **whole head** | **79.383** | 100% |

Scaled to the quantized head as loaded (~10.876 GB on disk):

| piece | GB |
|---|---|
| FFN total | 10.633 |
| non-FFN total | **0.243** |
| per-rank FFN after the shipped TP_SHARD split | 5.317 |
| per-rank non-FFN (fully replicated today) | 0.243 |

**Sharding every non-FFN component recovers only ~0.12 GB/rank.**

Against the 4.9 GB of headroom at 565K, that is not a fix — it is ~2% of what
is needed. The doc's lever 1 is roughly an order of magnitude too small.

**The real remaining mass is the other half of the FFN.** `EXO_DSV4_DSPARK_TP_SHARD=1`
splits the FFN across 2 ranks (5.32 GB/rank instead of 10.63). The head is
already ~50% sharded. To recover more you would need either:
- >2-way sharding of the FFN (not available at TP=2), or
- lower-bit quantization of the head (quality-gated: the draft head drives
  acceptance), or
- not keeping the head resident at all (paging/spill — large change).

**Action: do NOT implement "shard non-expert projections".** It was the
plan's Phase 4 and it is now deprioritised on measured grounds.

## Correction B: tok/s is content-driven, not depth-driven

Overnight rung at 27,850 depth: **46.30 t/s, acceptance 2.077/cycle.**

That is the fastest decode measured on this cluster all session — at a depth
where earlier runs gave 31-33 t/s. Cross-referencing every rung measured today:

| run | depth | tps | acc/cyc | ms/cyc |
|---|---|---|---|---|
| overnight | 27,850 | **46.30** | **2.077** | 66.5 |
| earlier | 27,9622 | 26.72 | n/a | n/a |
| earlier | 21,855 | 33.22 | 0.810 | 54.6 |
| earlier | 68,002 | 31.78 | 0.815 | 57.2 |
| earlier | 115,614 | 36.74 | 1.415 | 65.8 |
| earlier | 2,216 | 29.17 | 1.033 | 61.8 |

- **Cycle cost spread: 54.6-66.5 ms (22%).**
- **Acceptance spread: 0.810-2.077 (2.6x).**

tok/s tracks acceptance, not cycle cost. And acceptance is set by **content**:
the overnight probe asks for "the integers 1 to 300 comma separated" (highly
predictable → acc 2.08), whereas the earlier ladder's higher rungs asked for a
900-word freeform essay (→ acc 0.81).

**This means the earlier "depth ladder" was substantially measuring CONTENT,
not depth.** The 26.7-26.8 t/s at 250-280K may be a content effect as much as
a depth effect.

## Consequence for the night's two targets

- **T2 (250K ≥ 30 t/s):** may already be met for predictable-output content —
  need the controlled same-depth test to separate it from prose content.
  Do NOT chase a compute lever until that is resolved.
- **T1 (500K collapse):** the refault mechanism stands (32,469 page-ins
  measured at 42K depth). But the fix is NOT "shard non-expert projections"
  (Correction A). Remaining honest options are head quantization or spill,
  both larger and both quality-gated.

## Next actions (revised)
1. Controlled same-depth content test (easy vs freeform, alternating) —
   settles whether T2 is real or a content artifact.
2. Let the depth ladder complete to get pageins/acceptance at 250K and 500K.
3. Re-scope T1 with the measured component sizes rather than the doc's estimate.
