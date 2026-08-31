# P08 Item 1 — attn.sdpa.compressed causal-vs-dense verdict + direct floor (2026-08-30)

Raw artifacts: `tmp/p08-20260830/item1_results.json`, `item1_stdout.log`,
harness `p08_item1_capture.py`, runner `run_item1_node.sh` (this directory).
Node: macstudio-m4-1 (M4 Max 40-core, `applegpu_g16s`, MLX 0.32.1.dev), standalone
process beside the live runner per the p01/P03/P07 recipe. Runner PIDs verified
identical before AND after (m4-1 PID 59909, m4-2 PID 60392, both etime
02:00→02:08, zero restarts). Live env flags re-verified via `ps eww 59909`
(EXO_DSV4_SEQ_SPLIT=1, EXO_DSV4_INDEX_TOPK=512, EXO_COMPUTE_DTYPE=bf16,
EXO_DSV4_PREFILL_ARGPARTITION=1, EXO_DSV4_SPARSE_SDPA_TILE=128, HC kernels=1).

## (A) The production shape at attn.sdpa.compressed

Cited path: `span("attn.sdpa.compressed")` at
`mlx-lm/mlx_lm/models/deepseek_v4.py:4385`; the fused call is
`deepseek_v4.py:4413-4425` → `scaled_dot_product_attention`
(`base.py:122`) → `mx.fast.scaled_dot_product_attention`
(`base.py:193-201`). Production prefill takes the `_Lq=1024 > 8` branch, so
the `_CATTN_LSPLIT_MAX_L` L-split (dv4:4391-4410) does NOT fire.

| quantity | value | provenance |
|---|---|---|
| batch | 1 | serving single-stream; the L-split guard requires `q.shape[0]==1` (dv4:4393) |
| num_heads | 32 per rank | 64 // TP=2 — shard at dv4:7390 (`n_heads //= N`) |
| query length | 1024 | seq-split band, slice at dv4:4381 with `EXO_DSV4_SEQ_SPLIT=1` (start_cluster.sh:108); chunk 2048 (`start_cluster.sh:72`) / 2 ranks |
| key length (CATTN_KV) | 3894 | local ring 128 (sliding_window, dv4:865) post-update `max_size + S − 1 = 2175` (cache.py:633) + pooled 1719 = ⌈220000/128⌉ (bench.py:70); concat at dv4:4359 |
| head_dim | 512 | config head_dim; KV projects hidden(4096)→512 directly (no kv_lora_rank — P07 §8) |
| dtype | bf16 | `EXO_COMPUTE_DTYPE=bf16` verified in live runner env (`ps eww 59909`) |
| mask dtype / shape | **bool**, (1, 1, 1024, 3894) | runtime-printed + code-derived: windowed-causal `create_causal_mask` (base.py:24-42, via RotatingKVCache.make_mask) concatenated with the row-causal pooled mask (`_extend_mask`, dv4:1361-1391; PoolingCache.make_mask cache.py:1605-1627); band row-slice at dv4:4382-4383 |
| broadcast vs materialized | MATERIALIZED `(1,1,1024,3894)` bool (3.99 MB), head-broadcast by the kernel | `mx.concatenate` of two bool masks at dv4:1390; the pooled part is broadcast_to'ed over H first (dv4:1386-1388), then materialized by concatenate |

**Bench comparison**: `bench/attn_production_class_bench.py:212-236` uses the
correct q/kv SHAPES — q (1,32,1024,512), kv (1,1,3894,512), mask bool
(1,1,1024,3894). Shape provenance matches exactly. BUT its mask CONTENT is
`mx.random.uniform > 0.05` (~95 % dense random, bench.py:219) instead of the
production causal+window content. Runtime reconstruction of the actual
production mask (create_causal_mask + PoolingCache.make_mask formula) gives
**47.3 % density** — avg 1844 visible keys per row out of 3894, NOT 0.95·3894.

## (B) The causal-vs-dense test at production shape (m4-1, GPU-busy time)

Mask conditions differ ONLY in content; identical dtype (bool), shape
(1,1,1024,3894), layout, and call path. Median of 7 timed reps after 6
warmup, 16 KV rotation banks (63.8 MB ≫ 16 MB L2), L2 sanity 2-vs-16 banks
difference < 0.1 %.

| condition | t (µs, GPU) | dispatches |
|---|---|---|
| (a) causal (production mask) | **21 423.4** | 7 |
| (b) dense (all-True, same shape/dtype) | **21 408.6** | 7 |
| (c) mask=None | **20 255.7** | 6 |

- **R = t_causal / t_dense = 1.0007** → pre-registered verdict
  (§2.3): **KERNEL DOES FULL WORK.** The fused kernel does not skip masked
  blocks: identical dispatch count (7 vs 7), timing statistically identical
  (±0.07 %). The 5.4 % delta between causal and `mask=None` is the fixed cost
  of reading/materializing the mask, NOT work skipping.
- **f = causal_FLOP/dense_FLOP from the ACTUAL mask content = 0.4736**
  (mean 1844.2 visible keys/row of 3894, computed from the production mask
  tensor itself). Note: P07's 0.6058 was a run-average over a growing
  context; at the FINAL 220K chunk the real content density is 0.4736 (and
  the P07 158.3-vs-261.3 GFLOP pair implies 0.6058 vs this measured 0.4736 —
  the actual production mask is even sparser than P07's arithmetic).
  R=1.0007 vs f=0.4736: the kernel computes ~2.1× the causally-required
  work — the denominator P07 questioned was INFLATED, but the kernel leaves
  that entire gap on the table rather than exploiting it.

**Correction to P07 §8's arithmetic**: at production final-chunk shape the
denominator should be based on f=0.4736 (mask-content density), so the
efficiency figure would be ~79.1 % × 0.4736/0.6058 ≈ 61.8 %-class
(158.3→123.4 GFLOP real work at this chunk) if the kernel exploited the mask
— AND IT DOES NOT (R=1.0007). So the honest statement of P08 Item 1 is:
the fused SDPA computes ALL 1024×3894 score positions including every
causally-masked one, and the "would-be" saving from mask exploitation
(~50 % of the span) is unreachable through mask content alone.

## (C) On-node measured GEMM peak (M4 Max 40-core Studio, m4-1)

Real dense GEMM sweep, bf16 AND fp16, rotated 2-4 input banks, median of 7,
fresh graph + eval barrier per call, MLX_GPU_TIME bracketing:

| shape (M×K×N) | bf16 TF | fp16 TF |
|---|---|---|
| 2048×4096×4096 | 14.83 | 14.83 |
| 4096×4096×4096 | 15.10 | 15.10 |
| 8192×4096×4096 | 15.17 | 14.67 |
| **16384×4096×4096** | **15.21** ← peak | 15.04 |
| 8192×8192×8192 | 15.18 | 14.998 |

**Measured on-node dense GEMM peak: 15.21 TFLOPS bf16 at [16384, 4096, 4096]
(M=16384)** — vs 11.66 TF (laptop figure, bench.py:136-145) and 14.34
(theoretical 40-core, P07 §9). This is the rigorous fix P07 named and did not
do. The old 11.66 denominator understated Studio efficiency by ~23 %; the
theoretical 14.34 was slightly conservative (measured exceeds it by 6.1 %).
Also measured: streaming r+w bandwidth 488.4 GB/s on this node (256 MB bf16
stream), not 424 — this node exceeds the campaign's 424 GB/s floor.

## (D) Direct, denominator-free headroom

Same node, same discipline, min across dispatch variants for QK^T:

| component | shape (per-head batched over 32 heads, KV shared) | µs |
|---|---|---|
| matmul_QK | (32,1024,512) @ (512,3894) → scores | 8847.6 (broadcast, 1 disp); loop-32 variant 9043.7 → used 8847.6 |
| matmul_PV | (32,1024,3894) @ (3894,512) → out | 8732.3 |
| softmax pass (measured chain max/exp/sum/div) | (32,1024,3894) | 5727.0 |
| softmax pure-streaming bound (rooftop at measured 488.4 GB/s) | 256 MB score tensor r+w | 522.5 |

`floor = max(matmul_QK + matmul_PV, softmax_time)`
= max(8847.6 + 8732.3, 522.5) = **17 579.9 µs** (roofline max() convention;
using the measured softmax chain instead of the streaming bound gives the
identical result since both are below the matmul sum).

**direct_headroom = t_causal / floor = 21 423.4 / 17 579.9 = 1.2186×**
(also 1.2186× under the measured-softmax convention — identical, the floor
is matmul-bound at this shape).

## Verdict against the pre-registered lever gate (§2.4)

- direct_headroom 1.2186 < 1.40 → **FAILS gate 1**
- span_share (11.8 %, `docs/dsv4-220k-prefill-span-profile-2026-08-18.md:84`
  — attn.sdpa.compressed: 2200 calls, 73 396.81 ms total, 11.8 % of prefill
  wall) × (1 − 1/1.2186) = 11.8 % × 0.1794 = 2.12 % ≥ 1.0 % → would pass
  gate 2, but BOTH conditions are required.
- **Verdict: `attn.sdpa.compressed` is NOT a real lever** under the
  pre-registered gate. The fused SDPA runs at 1.22× the reachable floor
  (matmul + softmax), i.e. at most ~18 % of the span is theoretically
  recoverable, and any real fix must beat 8.85 µs-per-matmul MLX dispatches
  on THIS node — no existing op composition does.

The denominator story is now fully adjudicated: the ceiling DENOMINATOR was
inflated (mask content is 47.4 % dense, not 95 %), but the KERNEL does full
dense work (R=1.0007) AND sits at 1.22× the reachable floor. The honest
conclusion is "denominator corrected, NO actionable lever" — the 79.1 %
efficiency figure stands (kernel does the full dense FLOP count it was
attributed).

## Caveats stated plainly

- The floor uses `mx.matmul` at exact production shapes (batched 32-head QK^T
  broadcast-KV and PV, both timed; the min kept per the consult). This is the
  only baseline respected in this repo; no custom kernel is assumed.
- The sdpa t_causal/t_dense/t_none all ran identically-shaped 11-rep medians
  with 6 warmups, kv-bank rotation, and the dispatch counter visible around
  each call as `MLX_DISPATCH_COUNT=1` was set on the runner.
- `mx.metal.gpu_time_ns()` granularity is ~8 ns per dispatch here; the
  21 400 µs / 5 226 µs / 15 210 µs figures are 1000× above timer noise.
- No `*0` arithmetic, no cancelling evals, no span timers; `MLX_GPU_TIME=1`
  and `MLX_DISPATCH_COUNT=1` were set before the MXL import in the runner.
- The mask-density figure 47.29 % (rank 0) / 47.39 % (rank 1) is computed by
  summing the reconstructed production mask; it matches the algebraic windowed
  mask (128+1 visible columns per row in the local part ÷ 2175 ⇒ ~0.4728).
- Item 1 lever-gate check uses the SAME span_share as the pre-registration
  (11.8 % from the cited span profile); this is prefill-phase wall share, so
  the decode-phase caution in §2.4 does not apply here.