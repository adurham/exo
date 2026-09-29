# Phase 21 -- what scales with context; post-prompt decode "collapse" explained

Date: 2026-09-29. Single-node ablation (layers 2 + 20, <=7.9 GB peak, next to
production), then one two-node run. Production stopped cleanly and restored.

## 1. Post-prompt decode collapse: not real

Per-step decode times after an 8K prompt (two nodes):
`1514 1860 1277 108 84 76 76 76 ...` ms. The first three steps are one-time
compile for the new shape; steady state is 76-79 ms/step (12.8-13.2 tok/s)
at 8K and 16K context -- vs 59 ms at short context. Phase 20's "1.6-5 tok/s"
came from averaging only 16 steps that included the compile. The warmup
should cover the decode shape after prefill; follow-up.

## 2. Prefill scaling

Single node, per 512-row chunk (2 layers):

| context | whole | indexer stubbed | sparse attn stubbed |
|---:|---:|---:|---:|
| 8K | 122.5 ms | 93.2 | 110.0 |
| 16K | 156.5 ms | 93.1 | 148.0 |

The indexer is the only term that grows with context (+29 ms at 8K, +63 ms
at 16K per chunk on 2 layers); everything else is flat. Decode per step is
flat with context (6.2 ms on 2 layers at 8K and 16K).

The 128-row chunks used above 8K were the other cost. With 512-row chunks
throughout (two nodes):

| prompt | phase 20 (128 rows above 8K) | 512 rows throughout |
|---:|---:|---:|
| 8K | 63 s = 130 tok/s | 63 s = 129 tok/s |
| 16K | 225 s = 73 tok/s | **127 s = 129 tok/s** |

Peak 105.7 GB/rank, same first tokens, 0 GPU timeouts. Now the default.

## Where this leaves the switch

Production prefills ~430 tok/s. DSv4.1 is ~129 tok/s up to 16K, and the
indexer term will grow further at 64K-100K. Remaining prefill levers:
(1) indexer score pass -- a fused score+top-k kernel instead of MLX ops per
tile; (2) EXL3 dense/MoE at 512 rows; (3) session reuse (built, not wired)
removes re-reading history on follow-up turns, which is what Hermes does.
