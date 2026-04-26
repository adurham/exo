# Session prompt — DeepSeek V4 Flash on the cluster

Drop-in prompt to start a new session. Self-contained so the agent
can resume without prior transcript.

---

```
Goal: get DeepSeek V4 Flash running on the 2× M4 Max cluster, then
decide whether it's worth optimizing. The MiniMax-M2.7 decode
optimization arc just closed (ship 24.26 tok/s, no remaining levers
on stock Apple kernels). DeepSeek V4 Flash is the next model and
the question is "does it run, how does it perform, and is there
optimization headroom that didn't exist for MiniMax."

Read these first, in order:

1. memory/MEMORY.md — index of session knowledge. The cluster-
   hardware, bench-procedure, and trace-gotcha memories are load-
   bearing for this work.
2. memory/cluster_hardware_m4_max_not_ultra.md — Apple-published
   cluster specs. DO NOT make up numbers; look them up.
3. memory/feedback_minimax_bench_huihui_off.md +
   memory/feedback_minimax_trace_kills_tps.md +
   memory/feedback_minimax_trace_pct_normalization.md — bench
   procedure rules. These apply to DeepSeek V4 Flash benches too.
4. docs/minimax-rdma-moe-validation-2026-04-24.md — the precedent.
   Same hardware, same constraints; the conclusion ("stock Apple
   kernels are doing everything they can on this M4 Max + 5-bit
   MoE workload, custom kernels 0/3 cluster wins") frames what's
   plausible to attempt with a new model.
5. docs/kv-cache-architecture.md — exo's cache plumbing. DSv4's
   compressed/sparse attention may need a new cache type.

Cluster state:

- adurham/exo @ main = 0532b1b8 (post-MiniMax session, instrumentation
  + docs landed)
- adurham/mlx @ main = 22ef1101 (MLX_SDPA_BLOCKS env override +
  dispatch_count diagnostic; still carries the jaccl-#3412 revert)
- adurham/mlx-lm @ main = 872a271 (MiniMax forward-pass span
  instrumentation; otherwise tracking upstream)
- Hardware: 2× Mac Studio M4 Max (16-core CPU / 40-core GPU /
  128GB / 546 GB/s). 256GB total cluster memory.
- Cluster live config: MLX_SDPA_BLOCKS=88, EXO_MINIMAX_FUSED_ATTN=1,
  HUIHUI_INSTANCES_PER_STUDIO=1 (production; flip to 0 for any
  perf bench).

DeepSeek V4 Flash (real specs from
huggingface.co/deepseek-ai/DeepSeek-V4-Flash/raw/main/config.json):

- model_type: "deepseek_v4"
- 158B params total in BF16, 13B activated per token (MoE)
- 43 hidden layers
- hidden_size 4096, head_dim **512** (huge), 64 attention heads,
  **1 KV head** (extreme MQA — KV cache ~64x smaller per token
  than typical GQA)
- q_lora_rank=1024, o_lora_rank=1024 (V3-style LoRA-decomposed
  projections)
- 256 routed experts + 1 shared expert, top-6 per token,
  moe_intermediate_size=2048
- 1M context via YARN (factor=16, native 64K)
- Hybrid attention: Compressed Sparse Attention (CSA) + Heavily
  Compressed Attention (HCA), per-layer pattern in `compress_ratios`
  array. Layers 0-1 + last layer use ratio 0 (full attention),
  middle layers alternate 4 / 128 compression ratios.
- index_n_heads=64, index_head_dim=128, index_topk=512 (indexed
  attention selection mechanism, new for V4)
- num_nextn_predict_layers=1 (single MTP head)
- num_hash_layers=3
- sliding_window=128
- Native quantization: FP8 e4m3 with weight_block_size [128, 128];
  MoE experts are FP4 in the official release

MLX-community quantizations available (Apr 2026):
- mlx-community/deepseek-ai-DeepSeek-V4-Flash-3bit (smallest, ~60 GB)
- mlx-community/deepseek-ai-DeepSeek-V4-Flash-4bit (~80 GB)
- mlx-community/deepseek-ai-DeepSeek-V4-Flash-8bit (~160 GB — tight
  fit at 256 GB cluster total, but fits)

Memory math for cluster (256 GB total, 2-rank TP):
- 4-bit: ~80 GB / 2 ranks ≈ ~40 GB/rank → comfortable, fits with
  large KV cache headroom for 1M context
- 8-bit: ~160 GB / 2 ranks ≈ ~80 GB/rank → tight; KV cache needs
  to be sized carefully for long context
- 3-bit: ~60 GB / 2 ranks ≈ ~30 GB/rank → lots of headroom; use
  if quality tolerates

MQA + tiny KV cache is a meaningful change vs MiniMax. With 1 KV
head and head_dim=512, KV cache per token = 2 * 1 * 512 * 2 bytes
= 2 KB per layer × 43 layers = 86 KB/token at bf16. A 1M-context
session is 86 GB just for K and V — but with 4-bit KV that drops
to ~22 GB. Still substantial; needs cache-bits and prefix-cache
tuning per session.

Phase 1 — does it load?

The first concrete question: does mlx-lm 0.31.3+ have a
`models/deepseek_v4.py`? If yes, exo should be able to load
mlx-community/deepseek-ai-DeepSeek-V4-Flash-4bit out of the box
(after maybe a sharding strategy in auto_parallel.py).

Steps:

1. `grep -r 'deepseek_v4' mlx-lm/mlx_lm/models/` — does the model
   file exist?
2. If yes: try loading single-node first via
   `uv run exo` and the dashboard's LOAD MODEL flow against the
   3-bit quant (smallest, easiest to validate). Single-node fits
   3-bit easily.
3. If load works single-node: try 2-rank TP via start_cluster.sh
   placement. exo's `auto_parallel.py` may need a
   `DeepSeekV4ShardingStrategy` similar to `MiniMaxShardingStrategy`
   if the existing strategies don't cover the V3-style LoRA q/o
   projections + 1-KV-head + hybrid attention pattern.
4. If load fails: figure out why. Likely candidates: model file
   missing, hybrid-attention not implemented, FP4 dequant path
   missing, YARN scaling missing.

Phase 2 — measure

Once running, bench with `bench/minimax_cluster_ab.py` adapted for
DSv4 (or a new bench). Get:
- prompt_tps + decode tok/s at, say, 50K and 256K context
- macmon GPU power draw during a bench (mirror the methodology in
  docs/minimax-rdma-moe-validation-2026-04-24.md § "Live GPU power
  measurement")
- per-span breakdown if instrumentation exists or can be added

The headline question: at 1M context, is DSv4-Flash's hybrid
attention actually keeping `attn.sdpa` from being the dominant
slice? For MiniMax it was 55-69 % of wall time. If DSv4's CSA/HCA
shrinks attention's share to the point where MoE becomes the
bottleneck, that opens the qwen3_5_moe-style batched-fusion port
as a real lever (MiniMax couldn't justify it because attention
dominated).

Phase 3 — decide

Three likely outcomes, in order of likelihood:

A. Loads, runs, attention is still dominant → same conclusion as
   MiniMax, ship and move on.
B. Loads, runs, attention is shrunk by CSA/HCA → MoE is now the
   majority slice, and the qwen3_5_moe batched-fusion port (predicted
   +4-7 % when attention dominated; could be larger if MoE is the
   majority) becomes a real Phase-3 candidate.
C. Doesn't load or hybrid-attention isn't supported → write the
   model code in mlx-lm fork, or wait for upstream support. Ask the
   user before sinking real engineering time.

Hard scope constraints (carry over from MiniMax sessions):

- Cluster benches: HUIHUI_INSTANCES_PER_STUDIO=0. Always.
- Trace and tps measurement are mutually exclusive. Trace adds
  ~70 % overhead at decode (~3 % at prefill). Use trace only for
  per-span share-of-wall, work in absolute total_ms vs wall (the
  % column double-counts parents+children).
- ASK before running start_cluster.sh — each deploy disrupts the
  cluster ~5 min, and start_cluster.sh's git-reset behavior makes
  unrelated local changes a hazard.
- Custom-kernel work: 0/3 cluster wins on MiniMax. Do NOT propose
  writing custom Metal kernels for DSv4-Flash without first showing
  the dispatch-count or bandwidth analysis that says it'd land.
- M4 Max (NOT M4 Ultra). 40 GPU cores, 546 GB/s. Look up specs
  rather than approximate from memory.

Things explicitly out of scope at session start:

- Pivoting to other models (Qwen3.5, MTP work, etc.) as alternatives
  if DSv4-Flash is hard.
- Speculative decoding for DSv4-Flash unless the user explicitly
  greenlights it (MiniMax had it ruled out; DSv4 status unknown).
- Writing custom Metal kernels.

First user-facing decision point:

Once the load-test result is known (Phase 1 done), summarize:
"DSv4-Flash at 4-bit on 2-rank TP loads/doesn't, gets X tok/s decode
at 50K and Y at 256K, attention is Z % of wall, MoE is W %, GPU
power Q W at 99 % util. Worth optimizing further (Phase 3 path B)?"
That's the first user check-in.

Cluster restore: HUIHUI_INSTANCES_PER_STUDIO=1 + drop any DSv4-
specific env vars + start_cluster.sh.
```

---

## Notes for the prompt-writer (not part of the session prompt)

- The MiniMax-M2.7 deployment in `start_cluster.sh` is the
  production tenant. DSv4-Flash work likely needs a separate
  test cycle or a temporary swap. The user has the prediction-bot
  hitting MiniMax so coordinate before disrupting.
- DSv4 is brand-new (April 2026 release per Apple/news search).
  mlx-lm support may be incomplete or buggy — be ready for the
  Phase-1 flow to fail and have to triage upstream support.
- The hybrid attention / CSA / HCA mechanism is the unique
  architectural feature. If exo or mlx-lm don't support it, that's
  the bottleneck before any perf work matters.
