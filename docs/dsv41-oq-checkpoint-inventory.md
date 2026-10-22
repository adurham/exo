# DeepSeek-V4.1-Flash-1017 oQ2.5e — checkpoint inventory

Measured from `mlx-community/DeepSeek-V4.1-Flash-1017-oQ2.5e-ngram-mtp` on
2026-10-22. Only the *metadata* was downloaded to produce this document:
`config.json`, `model.safetensors.index.json`, `quantization_report.json`,
the tokenizer files, and the safetensors **headers** of all 53 shards (fetched with HTTP range requests — no weights).
Two numeric test fixtures were then pulled as a byte-range subset: all of
layer 0, `embed`/`norm`/`head` and all of `mtp.0` (~9.7 GiB), which is what
`tests/dsv41_oq` runs against.

## 1. Shape of the pack

* 2282 tensors in 53 shards, 392.7 GiB of text-stack weights.
* 266 tensors belong to the vision tower (`vision.*`, `aligner.*`, `image_*`) and are not loaded by the text runtime.
* `config.json` has **no `quantization` key**. All quant metadata lives in
  `omlx_deepseek_v41`: version 1, 460 `quantized_modules`, `engram_in_index=True`, `preserve_mtp=True`.
* Model: dim 5120, 40 layers, 64 heads x 512, 384 experts top-6, moe_inter 2304, vocab 129280, engram layers [1, 14], 3 MTP stages.

### Two shards are 94 GiB each

| shard | size | tensors |
|---|---|---|
| `model-00001-of-00053.safetensors` | 94.4 GiB | 2 |
| `model-00002-of-00053.safetensors` | 94.4 GiB | 2 |
| `model-00051-of-00053.safetensors` | 4.7 GiB | 39 |
| …50 more | ~7.4 GiB each | |

The two giants hold the engram tables (`layers.1.engram.embed`,
`layers.14.engram.embed`) — 384 M rows x 256 fp8 each. Any plan that
downloads "the shards a layer lives in" must special-case them.

## 2. Tensor naming

The text stack is prefixed `language_model.`; everything else is the vision
tower. The loader strips the prefix so names match the EXL3 path
(`layers.N.…`, `head`, `mtp.N.…`).

| prefix | tensors |
|---|---|
| `language_model.layers` | 1901 |
| `vision` | 259 |
| `language_model.mtp` | 112 |
| `aligner` | 4 |
| `language_model.embed` | 1 |
| `image_start` | 1 |
| `image_end` | 1 |
| `image_newline` | 1 |
| `language_model.norm` | 1 |
| `language_model.head` | 1 |

## 3. One dense layer (layer 0), exact keys and shapes

| tensor | shape | dtype | quantized |
|---|---|---|---|
| `layers.0.attn.attn_sink` | `(64,)` | F32 | no |
| `layers.0.attn.kv_norm.weight` | `(512,)` | BF16 | no |
| `layers.0.attn.q_norm.weight` | `(1280,)` | BF16 | no |
| `layers.0.attn.wkv.biases` | `(512, 80)` | BF16 | no |
| `layers.0.attn.wkv.scales` | `(512, 80)` | BF16 | no |
| `layers.0.attn.wkv.weight` | `(512, 1280)` | U32 | yes |
| `layers.0.attn.wo_a.weight` | `(8192, 4096)` | BF16 | no |
| `layers.0.attn.wo_b.biases` | `(5120, 128)` | BF16 | no |
| `layers.0.attn.wo_b.scales` | `(5120, 128)` | BF16 | no |
| `layers.0.attn.wo_b.weight` | `(5120, 2048)` | U32 | yes |
| `layers.0.attn.wq_a.biases` | `(1280, 80)` | BF16 | no |
| `layers.0.attn.wq_a.scales` | `(1280, 80)` | BF16 | no |
| `layers.0.attn.wq_a.weight` | `(1280, 1280)` | U32 | yes |
| `layers.0.attn.wq_b.biases` | `(32768, 20)` | BF16 | no |
| `layers.0.attn.wq_b.scales` | `(32768, 20)` | BF16 | no |
| `layers.0.attn.wq_b.weight` | `(32768, 320)` | U32 | yes |
| `layers.0.attn_norm.weight` | `(5120,)` | BF16 | no |
| `layers.0.ffn.gate.bias` | `(384,)` | F32 | no |
| `layers.0.ffn.gate.bias_vl` | `(384,)` | F32 | no |
| `layers.0.ffn.gate.weight` | `(384, 5120)` | BF16 | no |
| `layers.0.ffn.shared_experts.w1.biases` | `(2304, 80)` | BF16 | no |
| `layers.0.ffn.shared_experts.w1.scales` | `(2304, 80)` | BF16 | no |
| `layers.0.ffn.shared_experts.w1.weight` | `(2304, 1280)` | U32 | yes |
| `layers.0.ffn.shared_experts.w2.biases` | `(5120, 36)` | BF16 | no |
| `layers.0.ffn.shared_experts.w2.scales` | `(5120, 36)` | BF16 | no |
| `layers.0.ffn.shared_experts.w2.weight` | `(5120, 576)` | U32 | yes |
| `layers.0.ffn.shared_experts.w3.biases` | `(2304, 80)` | BF16 | no |
| `layers.0.ffn.shared_experts.w3.scales` | `(2304, 80)` | BF16 | no |
| `layers.0.ffn.shared_experts.w3.weight` | `(2304, 1280)` | U32 | yes |
| `layers.0.ffn_norm.weight` | `(5120,)` | BF16 | no |
| `layers.0.hc_attn_base` | `(24,)` | F32 | no |
| `layers.0.hc_attn_fn` | `(24, 20480)` | F32 | no |
| `layers.0.hc_attn_scale` | `(3,)` | F32 | no |
| `layers.0.hc_ffn_base` | `(24,)` | F32 | no |
| `layers.0.hc_ffn_fn` | `(24, 20480)` | F32 | no |
| `layers.0.hc_ffn_scale` | `(3,)` | F32 | no |

Reading the dense layout (all of it standard MLX):

* `attn.wq_a`: in 5120, out 1280, affine g64 b8 -> `weight` `[1280, 1280]`
  uint32 = 5120*8/32 packed columns, `scales`/`biases` `[1280, 80]` = 5120/64.
* `attn.wq_b`: out 32768 (64 heads x 512), in 1280 -> `[32768, 320]`, scales
  `[32768, 20]`.
* `attn.wkv`: the shared 512-d latent, **not sharded**, `[512, 1280]`.
* `attn.wo_b`: out 5120, in 8192 (= o_groups 8 x o_lora 1024) -> `[5120, 2048]`,
  scales `[5120, 128]`.
* `attn.wo_a`: **plain BF16 `[8192, 4096]`** — one stacked
  `(o_groups, o_lora_rank, heads_per_group*head_dim)` tensor, not quantized and
  not stored per group. The port's grouped einsum reshapes exactly that,
  so it loads straight into one `nn.Linear(4096, 8192)`.
* `ffn.gate`: BF16 `[384, 5120]` plus F32 `bias` **and** `bias_vl` (the image-span
  routing bias is present in this pack — no sidecar needed).
* `hc_*`: F32 hyper-connection mixes, `[24, 20480]` and friends.
* `attn.attn_sink` F32 `[64]`, norms BF16.

## 4. One expert stack (layer 0)

| tensor | shape | dtype |
|---|---|---|
| `layers.0.ffn.experts.w1.biases` | `(384, 2304, 80)` | BF16 |
| `layers.0.ffn.experts.w1.scales` | `(384, 2304, 80)` | BF16 |
| `layers.0.ffn.experts.w1.weight` | `(384, 2304, 320)` | U32 |
| `layers.0.ffn.experts.w2.biases` | `(384, 5120, 36)` | BF16 |
| `layers.0.ffn.experts.w2.scales` | `(384, 5120, 36)` | BF16 |
| `layers.0.ffn.experts.w2.weight` | `(384, 5120, 216)` | U32 |
| `layers.0.ffn.experts.w3.biases` | `(384, 2304, 80)` | BF16 |
| `layers.0.ffn.experts.w3.scales` | `(384, 2304, 80)` | BF16 |
| `layers.0.ffn.experts.w3.weight` | `(384, 2304, 320)` | U32 |

Stack layout is `[n_experts, out, packed_in]` (SwitchGLU's), one mode/bits/group
for the whole stack — which is why `gather_qmm` can dispatch it, and why the
loader asserts the packed geometry against the metadata.

Per-layer expert bits are **not uniform**:
* `ffn.experts.w1`/`w3`: {2: 34, 3: 6} (bit -> layers)
* `ffn.experts.w2`: {3: 40}
* layers whose w1/w3 are not 2-bit: [14, 15, 16, 17, 18, 39]

## 5. Per-module mode/bits histogram (all 460 modules)

| module tail | mode | bits | group | quantize_input | count |
|---|---|---|---|---|---|
| `attn.compressor.wgate` | affine | 8 | 64 | no | 3 |
| `attn.compressor.wkv` | affine | 8 | 64 | no | 4 |
| `attn.indexer.weights_proj` | affine | 8 | 64 | no | 8 |
| `attn.indexer.wk` | affine | 8 | 64 | no | 4 |
| `attn.indexer.wq_b` | affine | 8 | 64 | yes | 8 |
| `attn.wkv` | affine | 8 | 64 | yes | 40 |
| `attn.wkv` | mxfp8 | 8 | 32 (implied) | no | 3 |
| `attn.wo_b` | affine | 8 | 64 | yes | 40 |
| `attn.wo_b` | mxfp8 | 8 | 32 (implied) | no | 3 |
| `attn.wq_a` | affine | 8 | 64 | yes | 40 |
| `attn.wq_a` | mxfp8 | 8 | 32 (implied) | no | 3 |
| `attn.wq_b` | affine | 8 | 64 | yes | 40 |
| `attn.wq_b` | mxfp8 | 8 | 32 (implied) | no | 3 |
| `engram.wkv` | mxfp8 | 8 | 32 (implied) | no | 2 |
| `ffn.experts.w1` | affine | 2 | 64 | yes | 34 |
| `ffn.experts.w1` | affine | 3 | 64 | yes | 6 |
| `ffn.experts.w1` | mxfp4 | 4 | 32 (implied) | no | 3 |
| `ffn.experts.w2` | affine | 3 | 64 | yes | 40 |
| `ffn.experts.w2` | mxfp4 | 4 | 32 (implied) | no | 3 |
| `ffn.experts.w3` | affine | 2 | 64 | yes | 34 |
| `ffn.experts.w3` | affine | 3 | 64 | yes | 6 |
| `ffn.experts.w3` | mxfp4 | 4 | 32 (implied) | no | 3 |
| `ffn.shared_experts.w1` | affine | 8 | 64 | yes | 40 |
| `ffn.shared_experts.w1` | mxfp8 | 8 | 32 (implied) | no | 3 |
| `ffn.shared_experts.w2` | affine | 8 | 64 | yes | 40 |
| `ffn.shared_experts.w2` | mxfp8 | 8 | 32 (implied) | no | 3 |
| `ffn.shared_experts.w3` | affine | 8 | 64 | yes | 40 |
| `ffn.shared_experts.w3` | mxfp8 | 8 | 32 (implied) | no | 3 |
| `main_proj` | mxfp8 | 8 | 32 (implied) | no | 1 |

Notes that shaped the loader:

* `group_size` is written for affine entries only. The MXFP entries omit it;
  32 is implied by the mode (`_qparams` refuses an unknown mode without a
  group instead of defaulting to affine's 64).
* `biases` exist **iff** mode is affine. No MXFP module in the pack has them.
* Only `attn.wo_a` is missing from the quantized set among attention's
  projections, and only because it is stored as plain BF16.
* `ffn.experts.w1/w3` use 2-bit on 34 layers and 3-bit on 6 layers, so a
  loader that hard-codes one expert bit width fails on 6/40 layers.

## 6. Shard map for the layers a harness usually builds

| what | shards |
|---|---|
| layer 0 | `00015`, `00016` |
| layer 1 (engram) | `00002`, `00022`, `00023` |
| layer 6 | `00003`, `00004` |
| layer 14 (engram) | `00001`, `00004`, `00005` |
| layer 39 | `00013`, `00014`, `00015` |
| mtp.0 | `00028`, `00029` |
| mtp.1 | `00035`, `00036` |
| engram tables | `00001`, `00002` |

(`model-000NN-of-00053.safetensors` abbreviated to its number.) Layers 1 and
14 additionally own the 94 GiB engram shards; a subset fetch must exclude
those tensors, which is what the test fixture script does.

## 7. Engram

* `engram_in_index = True`: the tables **are** in this pack's
  index, as `F8_E4M3` `[384006168, 256]` weight + `F8_E8M0` `[384006168, 8]`
  scales (one e8m0 exponent per 32 values).
* `language_model.layers.1.engram.embed`: mode mxfp8, group 32, in `model-00002-of-00053.safetensors`.
* `language_model.layers.14.engram.embed`: mode mxfp8, group 32, in `model-00001-of-00053.safetensors`.
* The loader does **not** read them from here: `LazyEngramTable` keeps reading
  rows on demand from the native FP8 release (`EXO_DSV41_ENGRAM_DIR`), exactly
  as the EXL3 path does. The tables are ~203 GB of the pack's 393 GiB, and
  materializing them would defeat the row-on-demand design.
* The layer's *gating* projection `engram.wkv` **is** quantized here
  (`mxfp8`, g32, `[25600, 1536]` weight + uint8 scales) and loads normally.
* If a future pack dropped the native release, `LazyEngramTable` would need a
  chunk map over these two 94 GiB shards — not built here.

## 8. MTP / DSpark draft

* `preserve_mtp = True`, `excluded_draft_tensors = 0`.
* MTP **is quantized**: 31 modules, modes {('mxfp8', 8): 22, ('mxfp4', 4): 9} —
  routed experts mxfp4 (g32), attention/shared/`main_proj` mxfp8 (g32).
  So `build_mtp` must use `from_quantized` for the draft, not the plain path.
* The markov and confidence heads are plain BF16, and only `mtp.2` carries them
  (`mtp.0` has `main_norm` + `main_proj`, `mtp.2` has `norm` + markov/conf);
  the loader keeps the EXL3 path's tail remap for that.
* No quantized draft module has biases.

## 9. The A8 question (`quantize_input`)

* 408 of 460 modules carry `quantize_input: true`.
* Key sets: every flagged module stores exactly `{weight, scales}` (MXFP) or `{weight, scales, biases}` (affine). Mismatches: **0**.
* Tensors matching `input_scale` / `smooth` / `smoothing` anywhere in the
  index: **0**.

**Verdict: `quantize_input` is a pure runtime flag.** Nothing in the pack
encodes per-channel activation smoothing, so bf16 activations are correct and
the A8 kernels' benefit is forgone, not a correctness risk. The check is
mechanical and re-runs on every load attempt
(`oq_build.check_a8_is_runtime_only`, exercised over all 460 modules in
`tests/dsv41_oq/test_coverage.py::test_a8_flag_is_runtime_only`).

## 10. Layout rules the loader depends on

| rule | why it matters |
|---|---|
| packed columns = `in*bits//32` | a rank slice must cut uint32 words, not floats |
| scale columns = `in//group` | biases share the scale shape exactly |
| slice boundary multiple of `lcm(32, group)` | otherwise a group or a word is cut |
| experts are `[E, out, packed_in]` | one mode per stack, or `gather_qmm` cannot dispatch |
| `wo_a` is `[groups*o_lora, heads_per_group*head_dim]`, BF16 | heads split contiguously by group |
| `head`/`embed` are BF16 | vocab sharding is a plain row slice |

## 11. What this machine cannot verify

* **No EXL3 1017 pack is present.** The published EXL3 builds are 210-415 GB
  and the in-tree `mlx_lm/models/exl3` has a decoder only (no encoder), so the
  cross-format leg `oQ tensor <-> EXL3 tensor` cannot be run here. The parity
  tests substitute an independent numpy decoder of the MLX quantized layout
  plus a `mx.quantize` round trip, and compare module/block outputs against a
  float reference built through `AffineProj.from_weight`.
* **No native FP8 release is present**, so engram rows cannot be exercised
  end-to-end; the loader only re-uses the already-tested `LazyEngramTable`.
* Layer 0 + `mtp.0` are the only layers with real tensors locally; every other
  structural claim in this document comes from the header snapshot committed at
  `tests/dsv41_oq/data/oq2.5e_text_index.json` (dtype/shape/nbytes per tensor),
  which is what the coverage tests read.
