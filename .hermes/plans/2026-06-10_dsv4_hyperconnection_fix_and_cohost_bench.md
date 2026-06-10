# DSv4 garbage-output root cause + fix, and dual-model co-host benchmarks (2026-06-10)

Deployed: exo `4f40222e`, mlx-lm `90a799b` (both on `main`).

## TL;DR

DeepSeek-V4-Flash produced confident garbage output ("capital of France" ->
"hall"/"tr", Paris not in top-5), deterministic, on every prompt, identically
under both tensor and pipeline parallelism. Root cause was a **weight-key
naming mismatch**: the canonical `mlx-community/DeepSeek-V4-Flash` checkpoint
names the per-layer Hyper-Connection modules `hc_attn` / `hc_ffn`, but the
model code defines them as `attn_hc` / `ffn_hc`. Under `load_model(strict=False)`
the mismatched keys were silently dropped, so all 43 layers HC `fn`/`base`
weights stayed at their `mx.zeros` init. A zero mHC mix matrix collapses each
layers 4 hyper-connection streams to uniform `sigmoid(0)=0.5` gates instead of
the learned routing -> numerically healthy activations but a semantically
scrambled residual stream -> confident wrong tokens.

Fix: one rename in `DeepseekV4Model.sanitize()`, applied to incoming weight
keys before load:

    k.replace(".hc_attn.", ".attn_hc.").replace(".hc_ffn.", ".ffn_hc.")

Verified: layer-0 `attn_hc.fn`/`base` load non-zero (were 0.0), and end-to-end
output is coherent ("The capital of France is Paris.", valid haiku, correct
reasoning).

## Why it was hard to find

Every component checked out in isolation, so the bug hid behind a long process
of elimination. Verified correct and ruled OUT:

- Checkpoint structure, weights, quant (experts genuinely mxfp4 gs=32; attn
  affine-8; attn_sink a real learned vector).
- mlx-lm loader (experts load as `QuantizedSwitchLinear mode=mxfp4`).
- Tensor-parallel expert weight splits (dequant-lossless, max diff 0.0).
- MoE cross-rank reduction math (`sum_gradients` is a training-only no-op on
  the forward; partial routed + partial shared, single `all_sum` — sound).
- Quant-mode propagation to `gather_qmm` (mode=mxfp4 passed; mlx #3133 present).
- fp32 MoE gate accumulation (already fp32).
- Indexer scoring (had a real divergence from #1189 — extra ReLU + raw weights
  vs sigmoid — fixed it; **but it was not THE bug**, output byte-identical).
- RoPE (inv_freq convention correct; per-layer theta correct dense=10000 /
  compressed=160000+YaRN; output norm-preserving, no NaN/Inf, stable at S=1).
- HC Sinkhorn custom Metal kernel (EXO_HC_USE_OPS=1 pure-MLX fallback ->
  byte-identical garbage -> kernel exonerated).
- Output projection wo_a/wo_b grouped LoRA path (numerically identical to
  #1189, max diff 0.0).
- Prompt formatting (canonical `<|begin_of_sentence|><|User|>...<|Assistant|>
  <think>`, 16 tokens, correct special-token ids).
- Tensor vs pipeline parallelism (identical garbage under both -> sharding
  fully exonerated).

## The diagnostic that cracked it

Per-layer activation-RMS probe (`EXO_DSV4_ACT_PROBE=1`, env-gated, off by
default):

1. Forward RMS grew smoothly 0.098 (embed) -> 12.16 (layer 42), no explosion
   => not a numerical blow-up.
2. hc_head/post_norm/lm_head all healthy; logits confident (top ~17, clean
   margin) => model confidently predicts the WRONG token => semantic.
3. Same garbage under Tensor AND Pipeline => not the sharding stack.
4. Lazy single-node load + per-module weight-RMS check -> the per-layer
   `attn_hc`/`ffn_hc` weights sat at exactly 0.0 (while `hc_head`, whose name
   matched, loaded fine all along). That was the bug.

## What shipped to main

mlx-lm `90a799b`:
- THE fix: `hc_attn`/`hc_ffn` -> `attn_hc`/`ffn_hc` rename in `sanitize()`.
- Indexer scoring corrected to match #1189 (sigmoid head weights, no ReLU).
  Real correctness improvement; not the garbage bug.
- Off-by-default diagnostics: `EXO_DSV4_ACT_PROBE`, `EXO_HC_USE_OPS`.

exo `4f40222e`:
- Submodule bump to the mlx-lm fix.
- start_cluster.sh: off-by-default knobs `DSV4_SHARDING` (default Tensor),
  `EXO_HC_USE_OPS` / `EXO_DSV4_ACT_PROBE` EXO_ENV passthroughs.
- bench/dual_model_c2_100k.py: fixed hardcoded `DeepSeek-V4-Flash-8bit`
  (deleted repo) -> bare `DeepSeek-V4-Flash`.

## Dual-model co-host benchmarks (both MTP on, ~100K context)

DSv4 + Qwen3.6-35B-A3B-8bit co-hosted, 2-node Tensor + MlxJaccl, both MTP on.
`bench/dual_model_c2_100k.py`, 3 iters, max-tokens 200.

### c=1 (1 stream/model, 2 concurrent streams)

    TOTAL AGGREGATE decode: mean 78.5 tok/s  (median 81.7, range 70.4-83.4)
    Qwen3.6-35B-A3B-8bit:   ~51.2 tok/s/stream   needle 3/3   BOS-spam 0/3
    DeepSeek-V4-Flash:      ~27.3 tok/s/stream   needle 3/3   BOS-spam 0/3
    degenerate streams: 0/6

### c=2 (2 streams/model, 4 concurrent streams)

    TOTAL AGGREGATE decode: mean 73.0 tok/s  (median 68.3, range 32.1-118.5)
    Qwen3.6-35B-A3B-8bit:   per-model 56.4 tok/s (~28.2/stream)  needle 6/6  BOS-spam 0/6
    DeepSeek-V4-Flash:      per-model 16.6 tok/s (~8.3/stream)   needle 5/6  BOS-spam 0/6
    degenerate streams: 0/12

(DSv4 5/6 needle = 200-token-cap truncation, not corruption; output coherent.
High c=2 stdev is cold-prefill: iter1 wall 1049s = four sequential 100K
prefills; iter3 wall 9s = warm steady-state.)

### Interpretation

- **Cluster is GPU-saturated at 100K**: total aggregate basically flat
  c=1 -> c=2 (78.5 -> 73.0). Adding streams does not raise total throughput.
- Qwen scales with concurrency (51 -> 56 per-model agg); DSv4 per-stream
  **halves** under contention (27 -> 8.3). Lightweight Qwen-A3B dominates GPU
  scheduling; heavy 284B DSv4 with 100K sparse attention gets starved.
- **Recommendation: c=1 is the better balanced operating point** for the
  DSv4+Qwen co-host at 100K. c=2 mainly buys Qwen throughput at DSv4 expense
  with no net total gain.
- 100K prefill (~187-600 tok/s depending on batching) is the slow cold part;
  warm decode is fast.

## Operational notes

- No Thunderbolt link wedges across ~10 relaunches this session — graceful
  SIGTERM teardown held throughout.
- Launch plumbing: inline-prefix `VAR=x ./start_cluster.sh` inside
  `screen zsh -l -c "..."` works. To deploy a fork branch AND have a new
  start_cluster.sh passthrough take effect, the controller (MacBook) must be
  checked out on that branch (start_cluster.sh runs from the MacBook and builds
  EXO_ENV from its own copy), and `EXO_TARGET_BRANCH=<branch>` must be set (it
  deploys to the Studios and suppresses the interactive "HEAD not on
  origin/main, Continue? y/N" prompt that otherwise hangs the launch).
