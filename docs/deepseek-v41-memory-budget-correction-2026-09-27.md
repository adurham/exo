# V4.1 memory budget — section 5 correction (measured, 2026-09-27)

Corrects `docs/deepseek-v41-exl3-plan.md` section 5. Its mitigation list was
drafted before phase 1 measured anything; two of its inputs turned out to be
wrong, one of them in a way that would have destroyed capability if followed.
Numbers here are measured from the real repo (all 39 shard headers) and from
reading the sharding code that is actually running.

## 1. The 105 GB figure, and why the real number is different

The plan's section 5 says "the EXL3 file is ~105 GB per rank under the same
sharding." 105.29 = 210.58 / 2 exactly — the file simply halved. But halving
is only valid for tensors the sharding rule actually splits, and section 5's
own stated rule ("experts split, everything else replicated") does not halve
the other 14.55 GB.

Two successive corrections, both measured:

| rule applied | per-rank weights |
|---|---:|
| plan's stated rule (experts split, rest replicated) | 112.56 GB |
| the rule the running code actually applies | **108.76 GB** |
| plan's number (naive file/2) | 105.29 GB |

The 108.76 figure is the correct one. `DeepseekV4ShardingStrategy`
(`src/exo/worker/engines/mlx/auto_parallel.py`) width-shards **three** things,
not one:

- `layer.ffn.switch_mlp.{gate,up,down}_proj` — the routed experts
- `layer.ffn.shared_experts.*` — the shared expert
- `mtp.ffn.*` and (behind `EXO_DSV4_DSPARK_TP_SHARD`, **default 1** in
  `start_cluster.sh` since 2026-08-27) every DSpark stage's `.ffn` — the
  same treatment, applied by explicit loops that carry the comment "each
  MTP block has the same DeepseekV4MoE ffn structure as a regular layer, so
  identical sharding applies."

Attention, embeddings, head, router and the vision tower are replicated whole
on both ranks (attention deliberately: sharding `wq_b` across heads crashes
the LoRA-decomposed output projection inside `mx.quantized_matmul`).

Per-rank, measured:

| component | GB/rank | note |
|---|---:|---|
| routed experts | 98.02 | width-sharded |
| MTP (3 DSpark stages) | 3.84 | width-sharded; expert tensors 3.41 + 0.43 non-expert |
| shared expert | 0.40 | width-sharded |
| attention + embed + head + router + vision + other | 6.50 | replicated |
| **total** | **108.76** | |

## 2. "MTP tensors are unused" is false — they are the DSpark decoder

Section 5's mitigation 2 reads "Drop the vision tower and unused MTP tensors
at load. Free but small." That describes 7.24 GB as free, and the plan's own
open question asked "whether the MTP tensors in the file are all three DSpark
stages or one."

Phase 1 answered it. The `mtp.{0,1,2}` modules carry `main_proj`, `main_norm`,
`markov_head`/`markov_w1`/`markov_w2`, `confidence_head`, `hc_head` — that is
the structure of `DeepseekV4DSparkModule` exactly, three stages, matching
`num_nextn_predict_layers: 3`. These are not spare tensors; they **are** the
speculative draft head whose acceptance rate the plan's 30-45 tok/s decode
estimate rests on. Dropping them would delete the decoder.

**MTP stays.** It is also already width-sharded (§1), so it costs 3.84
GB/rank, not 7.24. It should additionally be **exempted from expert
streaming**: if the draft stages can miss on the SSD, the prefetch oracle
stalls and the cascade hits verify throughput directly.

Vision is the one genuinely optional drop, at 0.97 GB/rank — about one point
of streaming fraction. Given a streaming budget is being spent regardless,
keeping it is cheap; it is the last GB to give up.

## 3. Corrected mitigation stack

Constraints: MTP must stay; vision preferred; Engram tables stream from SSD
regardless; expert streaming is part of the design, not a fallback.

| mitigation | GB/rank | status |
|---|---:|---|
| head + embed axis-0 shard | 0.91 | recommended — free over TB5, vocab-parallel |
| stream the coldest N% of expert bytes | 0.98 × N | primary lever, N adaptive to measured margin |
| precision-tier the cold resident set (3-bit → 2-bit) | 2.97 per 10% tiered | composes with streaming; see §4 |
| drop vision | 0.97 | reserve only |
| drop MTP | 3.84 | **not available — destroys DSpark** |
| raise `iogpu.wired_limit_mb` to 120 GB | — | last resort; also starves Engram's page cache |

Required N, with head+embed sharded, vision and MTP kept:

| runtime overhead | weights cap | gap | N (streaming only) | N with 10% tiered |
|---|---:|---:|---:|---:|
| 16 GB | 99 GB | 8.84 GB | 9.0% | 6.0% |
| 18 GB | 97 GB | 10.84 GB | 11.1% | 8.0% |
| 20 GB | 95 GB | 12.84 GB | 13.1% | 10.1% |
| 22 GB | 93 GB | 14.84 GB | 15.1% | 12.1% |

The 16 GB overhead is borrowed from V4-Flash. V4.1 adds Engram's hot-row LRU
(2-4 GB planned), FP4 KV at 890 B/token, and indexer/candidate-pool scratch —
hence the 18-22 GB rows. Each GB of overhead costs one point of N.

## 4. The precision-tier lever is already in this checkpoint

Section 5 assumed a uniform expert precision. The file is not uniform: layers
18-22 carry 2-bit routed experts, the other 35 layers 3-bit. Per rank that is
8.49 GB at 2-bit and 89.18 GB at 3-bit — 8.7% of expert bytes already tiered,
and the tiering is per-expert, which is exactly the granularity streaming
already requires. Extending the tier to a further 10% of the 3-bit expert
bytes saves 2.97 GB/rank (3.0 points of N), with quality risk landing on the
coldest experts.

This is cheaper than section 5's implicit assumption that all mitigations
beyond streaming are unavailable. It also composes: the 22 GB-overhead worst
case closes with head+embed + 12% streaming + a 10% tier, vision kept and the
wired limit untouched.

## 5. Section 108 does not conflict with any of this

`bench/section108_tp_expert_locality_analysis.md` concluded "do not pursue
expert-locality placement." That is about **cross-node traffic**: it shows
expert placement cannot reduce the per-layer `all_sum`, because width-sharding
computes a partial FFN for every active expert on every rank regardless of
which experts fired. It says nothing about **resident bytes**, which is this
problem. Placement schemes cannot take per-rank residency below total/2; only
fewer bytes (tiering, dropping), non-resident bytes (streaming), or lower
overhead can. So section 108 neither kills streaming's premise nor offers a
substitute — it is simply a different question, and it should not be cited as
a reason to abandon the streaming design.

One real interaction it does imply: because both ranks hold half-slices of the
same experts and activate them together, **eviction must be mirrored across
ranks** and each node needs its own copy of its half-slices (a TB5 RDMA pull
from the peer's copy is ~2 ms, no better than local SSD).

## 6. What must be measured before the stack is committed

Nothing above is a decision yet — it is the corrected arithmetic. The gates,
cheapest and most decisive first:

1. **Composition of the 16 GB baseline.** Some is likely releasable MLX cache.
   Every GB found here is a point of N. Cheap to measure on the running stack.
2. **Activation miss-rate vs resident-fraction, on real agentic/multilingual
   traces.** This single curve converts the whole design into arithmetic: if
   misses fall fast with residency, 9-13% streaming is comfortable; if the
   curve is flat, the stack is not viable at any N and the fallback is a
   wider tier plus device separation. Do not skip to implementation first.
3. **Engram SSD queue sharing.** Both streams share one device; expert loads
   are latency-critical foreground, Engram reads are throttle-able background.
   Measure Engram steady-state bandwidth to size the QoS.
4. **Heterogeneous per-expert quant support on the MX path** — gates §4.
5. **Timing sanity at the target rate.** 45 tok/s ≈ 65 ms/step ≈ 1.5 ms/layer
   of slack; a 6.4 MB expert slice off internal NVMe is ~1-1.3 ms, so a
   single-layer lookahead can hide one miss, and the draft oracle's 3-ahead
   window gives 2-3 layers of lead. Misses cluster at segment switches, where
   the draft is least accurate — test that case explicitly, not just steady
   state.
