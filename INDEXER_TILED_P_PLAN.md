# Tiled-P Indexer Fix — Plan

## Root cause (profiler-confirmed, 2026-06-21)
`attn.indexer` is the dominant context-scaling cost of DSv4 prefill.
Span profiler, A=300K vs B=360K completing prefills:
- attn.indexer per-call avg: 4532us -> 5362us (+18%); every non-scaling op flat (moe.switch_mlp +0%, proj_qkv 0%, o_proj 0%).
- attn.indexer max/avg ~4x (5.4ms avg, 22ms spikes) -> the spikes across ~21 indexer
  layers compound into multi-second per-chunk stalls -> prefill 270 t/s -> 40 t/s cliff at ~340K.

Mechanism: `_indexer_score` builds the full `(B, H=64, L_chunk, P)` scores tensor then the
caller argsorts over the entire pooled context P. P grows with context, so:
  - the (B,64,128,P) intermediate balloons (~GB/layer at high ctx) -> alloc pressure + spikes
  - argsort is O(P log P) over full P.

## Two independent costs, two fixes
1. **Top-k sort cost**: argsort O(P log P) over full P. Existing env-gated
   `EXO_DSV4_PREFILL_ARGPARTITION=1` already switches prefill (L>1) to argpartition O(P),
   top-k SET identical (downstream gathered-KV attention is order-invariant). OFF by default.
   -> low-risk lever to turn on; quality-equivalent by construction.
2. **Score-tensor materialization**: `_indexer_score` allocates full (B,64,L,P) then
   transpose + (B,L,P,H)@(B,L,H,1) batched matmul. The transpose of a (B,64,128,P) tensor
   at large P is the alloc spike. **Tiled-P fix**: process P in blocks of P_BLOCK, computing
   the collapsed (B,L,P_block) score per tile and concatenating, so the full (B,64,L,P)
   pre-collapse tensor never materializes — only (B,64,L,P_block) transient per tile.
   Output (B,L,P) identical (concatenation of tiles along P). Bit-exact: each tile does the
   SAME q@pf.T -> transpose -> @w math on a P-slice; concatenating P-slices == full op.

## Implementation (build on deployed baseline origin/main = 91c3a95)
- Add `_indexer_score_tiled(q, pooled, weights_x, scale, n_heads_inv_sqrt, p_block)` that
  loops P in chunks of p_block, computes the collapsed (B,L,p_block) per tile, concatenates
  along axis=-1 -> (B,L,P). Bit-identical to `_indexer_score` (same ops, P partitioned).
- Gate via `EXO_DSV4_INDEXER_PBLOCK` (int, 0/unset = OFF = current full-P path). When set
  >0 and P > p_block, use tiled path. Default OFF -> zero behaviour change until validated.
- Keep `@partial(mx.compile, shapeless=True)` on the inner per-tile score so the kernel is
  compiled once (shapeless handles the last ragged tile).

## Validation (MANDATORY — both, per user rule: throughput w/o quality = meaningless)
1. **Bit-exactness (local, no cluster)**: extend bench/indexer_score_microbench.py — assert
   tiled output vs full output: max abs diff <= 1 bf16 ulp AND top-k SET identical at
   k=512, P in {25k, 90k, 250k}, p_block in {8k,16k,32k}.
2. **Throughput (cluster)**: span profiler A/B (300K vs 360K) — attn.indexer avg + max
   must drop, prefill t/s cliff must flatten.
3. **Quality (cluster)**: bench/quality_probe_dsv4.py at long context + short-prompt curl
   ("capital of France" -> "Paris", no BOS spam). Output must be unchanged.

## Deploy path (submodule -> gitlink -> reinstall)
1. Edit mlx-lm/mlx_lm/models/deepseek_v4.py (submodule, on 91c3a95 baseline).
2. Commit + push in submodule (adurham/mlx-lm main).
3. Bump gitlink in exo, commit + push exo main.
4. On each node: start_cluster.sh resets exo + `git submodule update` + force-reinstalls
   ./mlx-lm into venv. Default OFF, so deploy is safe; enable via env for the A/B.
