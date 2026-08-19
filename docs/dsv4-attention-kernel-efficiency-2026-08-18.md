# DSv4-Flash attention sub-kernel efficiency: achieved vs ceiling (2026-08-18)

Companion to the MoE analysis (`bench/moe_production_class_bench.py`, 62.6% of
matched-shape dense ceiling). Same methodology applied to the 4 largest
attention sub-spans, which together are **44.3% of prefill wall time** --
larger than MoE's 26.9%.

Benchmark: `bench/attn_production_class_bench.py`
Raw results: `bench/attn_production_class_bench_results.json`

## Measurement environment

Run on the **laptop M4 Max (32-core GPU)** -- same architecture/generation as
the cluster's Mac Studio M4 Max (40-core GPU) nodes, so ratios transfer;
absolute ms are ~1.25x slower here (core-count scaling). No cluster contact.

Session-measured hardware baselines (not spec sheets):
- dense fp16 square GEMM peak: **11.66 TFLOPS**
- streaming memory bandwidth (read+write): **297 GB/s**

## Production shapes (read out of the code, not assumed)

From `config.json` (DeepSeek-V4-Flash-0731) + `DeepseekV4Model.shard()` +
`SparseCompressedAttention/CompressedAttention.__call__`:

| quantity | value | source |
|---|---|---|
| hidden_size | 4096 | config |
| num_attention_heads | 64 -> **32 per TP rank** | `shard()`: `attn.n_heads //= N` |
| head_dim | 512 | config |
| num_key_value_heads | 1 (MLA, single shared KV head) | config |
| q_lora_rank / o_lora_rank / o_groups | 1024 / 1024 / 8 | config |
| index_topk | 512 | config |
| sliding_window | 128 | config |
| attention weight quant | **mxfp8, group_size=32, bits=8** | `make_quantization_config()`: every `.attn.w*` key -> mxfp8 (NOT the mxfp4 the routed MoE experts use) |
| prefill chunk L | 2048 (`EXO_PREFILL_STEP_SIZE`) | -- |
| **L per SDPA/o_proj call** | **1024** | `EXO_DSV4_SEQ_SPLIT=1` halves the query row band per rank |
| L for wq_a/wkv | 2048 (kv side stays FULL on every rank) | seq-split v2 comment |
| local KV len during a chunk | 128 + 2048 - 1 = **2175** | `RotatingKVCache._update_concat` |
| pooled len @220K, r=128 | 1719 -> CompressedAttention KV = **3894** | -- |
| pooled len @220K, r=4 | 55000 (>> topk) -> sparse path, **512 gathered rows/query** | -- |

**Attention is replicated, not sharded, across TP ranks** -- confirmed: only
`wq_b` (all-to-sharded) and `wo_a` (sharded-to-all) are split, plus `n_heads//=N`.
So per-rank shapes are the head-halved ones above, and the wq_a/wkv/kv side is
full-width on both ranks (deliberate coherence, not waste).

Sub-kernel shapes actually benchmarked:
- `attn.proj_qkv`: wq_a (2048,4096,1024) + wkv (2048,4096,512) + wq_b (1024,1024,16384)
- `attn.o_proj`: wo_a (8192,2048,1024 batched over 8 groups) + wo_b (1024,8192,4096)

## Results

| span | %wall | kind | ceiling type | achieved | ceiling | % of ceiling | max e2e speedup if raised to ceiling |
|---|---|---|---|---|---|---|---|
| `attn.sdpa` | 13.6% | SDPA (sparse split-softmax) | analytic compute roofline @11.66 TF | 25.06 ms / 7.20 TFLOPS | 15.46 ms | **61.7%** | 1.055x |
| " | " | " | *same-FLOP single fused `mx.fast.sdpa` (achievable)* | " | 19.49 ms / 9.25 TFLOPS | **77.8%** | 1.041x |
| `attn.sdpa.compressed` | 11.8% | SDPA (dense fast-sdpa) | analytic compute roofline | 28.32 ms / 9.23 TFLOPS | 22.41 ms | **79.1%** | 1.025x |
| `attn.o_proj` | 10.0% | GEMM (mxfp8) | matched-shape dense fp16 GEMM, same session | 10.70 ms / 9.63 TFLOPS | 8.90 ms / 11.59 TFLOPS | **83.2%** | 1.017x |
| `attn.proj_qkv` | 8.9% | GEMM (mxfp8) | matched-shape dense fp16 GEMM, same session | 6.24 ms / 9.64 TFLOPS | 5.28 ms / 11.38 TFLOPS | **84.7%** | 1.014x |

Ranked by (headroom x wall-share): `attn.sdpa` (5.2) > `attn.sdpa.compressed`
(2.5) > `attn.o_proj` (1.7) > `attn.proj_qkv` (1.4).

## Headline conclusions

1. **Attention is markedly HEALTHIER than MoE.** The two attention GEMMs run at
   **83-85% of matched-shape dense-fp16 ceiling**, versus MoE's 62.6%. There is
   no "attention GEMM inefficiency" to fix. mxfp8 attention weights are
   large-M, dense, non-ragged -- exactly the regime MLX's quantized GEMM is
   good at, unlike MoE's ragged 14-median-row expert groups.

2. **The single real lever is `attn.sdpa` (the sparse split-softmax chain),
   and it is worth ~4-5% end-to-end, not more.** At 61.7% of the analytic
   roofline / 77.8% of an achievable fused-kernel ceiling, closing the whole gap
   buys at most **1.04-1.06x total prefill**. The mechanism is already known and
   already characterised in `exo-sdpa-fusion-analysis`: the hand-rolled
   split-softmax (`_sparse_pooled_attention_inner`: 2 separate score GEMMs +
   logsumexp/logaddexp + 2 value GEMMs, with a per-query-row gathered
   (L,512,512) tensor) is slower than one fused `mx.fast.sdpa` doing identical
   arithmetic. The reason it can't just call the fused kernel is architectural
   (each query row has its own gathered pooled KV at L_q>1), and that is exactly
   the "D=512 MMA wall" fusion project already scoped and previously measured
   at 1.23x isolated -- consistent with the 1.28x (25.06/19.49) headroom found
   here. **This measurement independently corroborates that project's size and
   confirms it is not a >10% e2e lever.**

3. **`attn.sdpa.compressed` at 79.1% of roofline is essentially at ceiling.**
   It is already a single `mx.fast.scaled_dot_product_attention` -- Apple's own
   fused kernel. 9.23 TFLOPS at D=512 against an 11.66 TFLOPS square-GEMM peak
   is normal SDPA efficiency, not a defect. Confirms the earlier "dead end, not
   a bug" verdict from the architectural audit; now backed by a number.

4. **Even summing ALL four spans to their ceilings gives only ~1.11x total
   prefill.** 44.3% of wall time running at an average ~78% of ceiling leaves
   ~10% of wall recoverable in the absolute best case. Attention is not where
   a large prefill win lives.

## Cross-check against the span profile (per Fable's guidance) -- one real anomaly

| span | span-profile avg/call | isolated (laptop) | isolated (40-core est, x0.8) | span/isolated |
|---|---|---|---|---|
| `attn.proj_qkv` | 11.73 ms | 6.24 ms | 4.99 ms | **2.35x** |
| `attn.o_proj` | 13.07 ms | 10.70 ms | 8.56 ms | 1.53x |
| `attn.sdpa.compressed` | 33.36 ms | 28.32 ms | 22.66 ms | 1.47x |
| `attn.sdpa` | 13.32 ms | 25.06 ms | 20.05 ms | **0.66x** |

Two of these are NOT efficiency findings and must not be framed as such:

- **`attn.proj_qkv` at 2.35x is the outlier.** Its span wall is more than twice
  what the isolated kernel costs. The span body is 3 quantized matmuls + 2
  RMSNorms + a reshape -- nothing that accounts for a 2.3x gap. Leading
  suspects, in order: (a) the span boundary absorbs cross-rank/queue wait time
  that isn't this kernel's compute (attention runs after a collective; MLX's
  lazy graph means the `finalize()`/eval inside the span can charge the
  *preceding* op's latency to this span), (b) `EXO_PROFILER=spans` overhead
  (documented ~15% throughput tax) concentrated on short spans, (c) mask/RMSNorm
  dispatch overhead at small op sizes. **This is worth ~5% of prefill wall time
  in unexplained-by-kernel-cost time and is a better-value follow-up than any
  of the kernel-efficiency gaps above.** Do NOT chase proj_qkv GEMM efficiency;
  it is already at 84.7% of ceiling.
- **`attn.sdpa` at 0.66x runs FASTER in production than isolated.** Expected and
  benign: the span aggregates 23 layers/chunk, but only 21 are
  `SparseCompressedAttention` (compress_ratio=4); 2 are cheap `LocalAttention`
  (compress_ratio=0, KV=sliding window only). Additionally, early prefill chunks
  have pooled length < index_topk and take the much cheaper dense-concat branch
  rather than the sparse gather branch. The isolated benchmark models the
  worst-case steady-state sparse path only. **Treat the 61.7%-of-roofline number
  as valid for the sparse kernel, but the 13.6% wall share as covering a mix,
  so the realisable e2e win is at the low end of the 4-5% band.**

## Measurement uncertainty (be honest about this)

- The two **GEMM** ceilings are solid: real matched-shape dense fp16 GEMMs,
  measured in the same process, same thermal state, same session. Low
  uncertainty.
- The **SDPA rooflines carry materially more assumption**: the compute-bound
  arm uses the square-GEMM peak (11.66 TF), which SDPA cannot generally hit
  even when perfectly implemented (softmax/exp is non-FMA work not counted in
  the 4*L*KV*H*D FLOP model). So the "% of roofline" numbers are a **lower
  bound on efficiency** -- the real kernels are healthier than 61.7%/79.1%
  suggests. This is why the *fused-equivalent* ceiling (77.8%) is reported for
  `attn.sdpa`: it is a real measured kernel doing the same arithmetic and is
  the more defensible ceiling. No such alternative exists for
  `attn.sdpa.compressed` because it already IS that kernel.
- Memory-bound arms are far from binding for both (0.27 ms and 3.86 ms vs
  22.4 ms / 15.5 ms compute), so the roofline choice is compute regardless.
- Masks are synthetic (95%-dense random boolean). Measured mask-on vs mask-off
  delta is small (5.7% for `sdpa.compressed`, 3.2% for `sdpa`), so mask
  modelling is not a major error source.
- top-k index locality was probed separately (random vs sorted vs contiguous
  pooled ids): 24.19 / 24.11 / 23.29 ms -- a **<4% effect**. The gather pattern
  is not the bottleneck; the split-softmax arithmetic is.
- 32-core -> 40-core scaling is applied as a flat 0.8x. It is a first-order
  estimate; it does not capture different memory-bandwidth-per-core or clock
  behaviour between laptop and Studio chassis. Only the cross-check table
  depends on it, not the %-of-ceiling numbers.

## Recommendation

- **Do not** open MoE-style kernel-efficiency work on any attention GEMM;
  83-85% of matched dense is a healthy number and there is nothing there.
- The `attn.sdpa` fusion project remains the only attention kernel lever, and
  this analysis sizes it honestly at **~1.04x e2e, not more** -- consistent
  with, and a useful cap on, the previously-measured 1.23x isolated figure.
- **The highest-value attention follow-up is not a kernel at all**: root-cause
  the 2.35x `attn.proj_qkv` span-vs-kernel gap. That is ~5% of prefill wall
  time currently attributed to a span whose kernels demonstrably cost less than
  half of it.
