# DeepSeek-V4-Flash-Vision-Exp — exo cluster port plan (Phases 0-2)

Target checkpoint: `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`
HF: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp
Created 2026-08-31, last modified 2026-09-01. 167,811,372,792 bytes
(167.8 GB), 48 shards, 72,633 tensors.

AUTHORIZED SCOPE FOR THIS DISPATCH: **Phases 0, 1, and 2 only.**
Phase 3+ is explicitly NOT authorized. Stop and report at end of Phase 2.

---

## Established facts (already verified by the supervisor — do not re-derive)

### Text side is architecturally identical to the running `-0731`

Diff of `config.json` (Vision-Exp vs 0731): identical for every text field.
43 layers, `hidden_size` 4096, 64 attention heads, `num_key_value_heads` 1,
head_dim 512, 256 routed experts / 1 shared / 6 active, `moe_intermediate_size`
2048, `scoring_func` sqrtsoftplus, `topk_method` noaux_tc,
`routed_scaling_factor` 1.5, `rms_norm_eps` 1e-20, `sliding_window` 128,
`swiglu_limit` 10.0, `q_lora_rank`/`o_lora_rank` 1024, `o_groups` 8,
`index_n_heads` 64 / `index_head_dim` 128 / `index_topk` 512, `hc_mult` 4,
`hc_sinkhorn_iters` 20, `num_hash_layers` 3, `num_nextn_predict_layers` 3,
yarn rope factor 16 / original 65536, `max_position_embeddings` 1048576,
`vocab_size` 129280, `compress_rope_theta` 160000, and the same 46-entry
`compress_ratios` list (truncated to 43 by ModelArgs.__post_init__).

Quantization identical: `quant_method` fp8, fmt e4m3, `scale_fmt` ue8m0,
`weight_block_size` [128,128], dynamic activation, `expert_dtype` fp4.

DSpark fields identical: `dspark_block_size` 5, `dspark_noise_token_id` 128799,
`dspark_target_layer_ids` [40,41,42], `dspark_markov_rank` 256.

`tokenizer_config.json` is **byte-identical** to 0731 (verified via diff — zero
output). `generation_config.json`: do_sample true, temperature 1.0, top_p 1.0.

Conclusion: the existing fp8 load path (`mlx_lm/utils.py` → `load_model()` →
`quant_method == "fp8" and model_type == "deepseek_v4"` branch →
`deepseek_v4.make_quantization_config()`) is the correct base. No text-side
loader changes are needed for the weights to load.

### New vision config keys (top-level in config.json, NOT a `vision_config` sub-dict)

```
vision_n_layers: 32          vision_dim: 1024
vision_n_heads: 16           vision_inter_dim: 2816
vision_patch_size: 14        vision_rope_theta: 10000.0
vision_downsample_ratio: 3   vision_max_n_token: 384
vision_min_pixels: 147456    vision_max_wh_ratio: 8
```

### New tensors (263 confirmed under vision./aligner. prefixes)

```
vision.patch_embed.proj.{weight,bias}      (1 each)
vision.blocks.{0..31}.norm1.weight         (32)
vision.blocks.{0..31}.attn.wqkv.{weight,bias}  (32 each)
vision.blocks.{0..31}.attn.wo.{weight,bias}    (32 each)
vision.blocks.{0..31}.norm2.weight         (32)
vision.blocks.{0..31}.mlp.w1.weight        (32)
vision.blocks.{0..31}.mlp.w2.weight        (32)
vision.norm.weight                         (1)
aligner.w1.{weight,bias}                   (1 each)
aligner.w2.{weight,bias}                   (1 each)
```

There are additionally 4 sentinel embedding parameters
(`image_start`, `image_end`, `image_newline`, `image_pad`, each shape (4096,))
per DeepSeek's `inference/model.py`. Their exact key names in
`model.safetensors.index.json` were NOT confirmed by the supervisor — verify
them directly from the index before writing the loader.

Vision tensors carry **no `.scale` / `.scale_inv` companions** (verified:
zero keys matching `scale` under the `vision.` prefix, and zero `scale_inv`
keys repo-wide). They ship as plain bf16 and are NOT fp8-block-quantized,
unlike the text weights which pair `.weight` with `.scale`.

### Reference implementation files in the HF repo

- `inference/vision.py` (118 lines) — ViT + Aligner, PyTorch
- `inference/image_processor.py` (184 lines) — resize solver + token layout
- `inference/model.py` (1046 lines) — full model; diff vs 0731's is 141 lines
- `encoding/encoding_dsv4.py` (957 lines vs 0731's 760)
- `inference/examples/example_vl.txt`, `inference/examples/example_vl_harmony.json`
- `inference/examples/images/carrots.jpeg`, `inference/examples/images/corn.jpeg`
- `inference/config.json` — the reference's own flat arg names (dim, n_layers,
  rope_head_dim, window_size, ...) differ from HF `config.json` names; map
  carefully.

The repo README states the TXT and JSON examples "encode to identical prompts
and token IDs" — that is the golden equality test for Phase 2.

---

## Phase 0 — free disk, download weights to both nodes

### 0a. Delete the superseded preview checkpoint on BOTH nodes

**Supervisor has verified and authorized this specific deletion.**

Delete: `~/.exo/models/deepseek-ai--DeepSeek-V4-Flash` (149 GB on each node).

This is the *preview* checkpoint, superseded by
`deepseek-ai--DeepSeek-V4-Flash-0731` per the repo's own model card comment
("Official release checkpoint, superseding the preview
deepseek-ai/DeepSeek-V4-Flash"). The live cluster instance was confirmed via
`http://adams-mac-studio-m4-1.local:52415/state` to be running
`deepseek-ai/DeepSeek-V4-Flash-0731`, NOT the preview. It is re-downloadable
from HF if ever needed, and its model card
(`resources/inference_model_cards/deepseek-ai--DeepSeek-V4-Flash.toml`)
stays in the repo — do NOT delete the card.

DO NOT delete anything else. Specifically **keep**
`mlx-community--DeepSeek-V4-Flash` (144 GB) — it is the historical
sampling-A/B baseline — and keep `local--DeepSeek-V4-Flash-DSpark-MTP`.

Re-verify immediately before deleting (belt and braces): confirm
`/state` still reports `-0731` as the loaded model, and confirm the preview
directory name exactly. If `/state` reports the preview is loaded, STOP and
report — do not delete.

Current free space: node1 (adams-mac-studio-m4-1.local) 111 GB,
node2 (adams-mac-studio-m4-2.local) 134 GB. After deletion expect
~260 GB and ~283 GB respectively. Confirm with `df -h ~` after.

### 0b. Download, disconnect-safe, both nodes in parallel

Target dir on each node:
`/Users/adam.durham/.exo/models/deepseek-ai--DeepSeek-V4-Flash-Vision-Exp`
(the `ModelId.normalize()` convention: `/` → `--`; `EXO_DATA_HOME` is `~/.exo`
on macOS).

Pattern (run as a detached background process that survives SSH disconnect —
this takes hours):

```bash
ssh <node> "cd ~/repos/exo && HF_HUB_ENABLE_HF_TRANSFER=1 nohup .venv/bin/python -c \"
from huggingface_hub import snapshot_download
snapshot_download('deepseek-ai/DeepSeek-V4-Flash-Vision-Exp',
                  local_dir='/Users/adam.durham/.exo/models/deepseek-ai--DeepSeek-V4-Flash-Vision-Exp',
                  max_workers=8)
print('DOWNLOAD_COMPLETE')
\" > ~/dsv4_vision_download.log 2>&1 < /dev/null &
disown
echo STARTED_PID:\$!"
```

Fire both nodes as independent parallel SSH calls, not sequentially.

**Do not block waiting on this.** Start it, then proceed to Phase 1 code work
(which needs no weights) and poll periodically with a bounded poll loop — never
a blind `sleep`.

Progress verification: `grep -c DOWNLOAD_COMPLETE ~/dsv4_vision_download.log`
for real completion, plus an explicit missing-shard check. Do NOT trust tqdm
file-count percentages — they treat a 1 GB and a 4 GB shard as equal progress:

```bash
cd ~/.exo/models/deepseek-ai--DeepSeek-V4-Flash-Vision-Exp
for i in $(seq -w 1 48); do f="model-000${i}-of-00048.safetensors"; [ -f "$f" ] || echo MISSING: $f; done
```

(Confirm the exact shard filename format from
`model.safetensors.index.json` first — do not assume the zero-padding width.)

Final acceptance for Phase 0: both nodes report DOWNLOAD_COMPLETE, zero
missing shards, and `du -sb` of the model dir on each node is within a few MB
of 167,811,372,792.

---

## Phase 1 — port ViT + Aligner to MLX

File to modify: `~/repos/exo/mlx-lm/mlx_lm/models/deepseek_v4.py`
(this is the `adurham/mlx-lm` fork, a git submodule of the exo repo — 7,618
lines, currently contains ZERO vision code).

Upstream `ml-explore/mlx-lm` has no DeepSeek-V4 vision support either
(checked) — there is nothing to sync from. This is original work.
Third-party HF repos named `...-Vision-Exp-MLX` (inferencerlabs, Solstice-AI)
are plain mirrors of DeepSeek's own files with a `chat_template.jinja` added —
they contain no MLX port. Do not waste time looking for one.

### Architecture, from `inference/vision.py`

Read the actual file; this summary is to prevent misreadings, not to replace it.

- **PatchEmbed**: `nn.Linear(3 * patch_size**2, vision_dim)` = Linear(588→1024),
  applied to `x.flatten(1)` where x is (n_patches, 3, 14, 14).
  NOTE: this is a **Linear on flattened patches, not a Conv2d**. The existing
  exo `vision.py` NHWC transpose hack for conv weights does NOT apply.
- **2D RoPE** (`get_vision_cos_sin`): `rope_dim = vision_dim // vision_n_heads
  // 2` = 32. `inv_freq = 1/(theta ** (arange(0,32,2)/32))`. h and w positions
  are stacked to (n_h*n_w, 2, 1), multiplied by inv_freq, then `flatten(1)` →
  interleaved h/w frequency pairs. cos/sin get `.unsqueeze(1)` → shape
  (n_h*n_w, 1, 32).
- **apply_rotary**: `x1, x2 = x.float().chunk(2, dim=-1)` — **half-split**
  style (first half / second half), NOT the interleaved even/odd style. Getting
  this wrong is silent and produces plausible-but-wrong embeddings.
  Computed in float32, cast back to input dtype.
- **Attention**: fused `wqkv` Linear(1024→3072) **with bias**, `wo`
  Linear(1024→1024) **with bias**. q,k,v via `.chunk(3, dim=-1)` then view
  (n, 16, 64). RoPE applied to q and k only. Full bidirectional
  `scaled_dot_product_attention` — **no causal mask, no attention mask** (one
  image attends to itself entirely). Default scale (1/sqrt(64)).
- **MLP**: `w1` Linear(1024 → 2*2816=5632) **no bias**, chunked into
  (gate, up); `w2` Linear(2816→1024) **no bias**. `silu(gate) * up`.
- **RMSNorm**: `weight` is float32, `eps=1e-6`. Computed in float32
  (`x * rsqrt(x.square().mean(-1,keepdim=True) + eps)`), result cast back.
  **CRITICAL**: the vision RMSNorm uses the constructor default `eps=1e-6`,
  NOT the text model's `rms_norm_eps=1e-20`. Do not wire `args.rms_norm_eps`
  into the vision tower.
- **Block**: pre-norm residual — `x = x + attn(norm1(x)); x = x + mlp(norm2(x))`.
- **ViT.forward**: patch_embed → 32 blocks (cos/sin shared across all) → final
  `norm`.
- **Aligner**: `downsample_ratio` r=3, `in_dim = vision_dim * r**2` = 9216.
  `w1` Linear(9216→4096) with bias, GELU, `w2` Linear(4096→4096) with bias.
  Forward: `x.view(n_h, n_w, -1).permute(2,0,1)` → `F.pad(x, (0, -n_w % r, 0,
  -n_h % r))` (pad right and bottom to a multiple of 3, zero fill) →
  `F.unfold(x.unsqueeze(0), 3, stride=3).squeeze(0).transpose(0,1)` →
  `w2(gelu(w1(...)))`.

  **MLX has no `F.unfold`.** It must be reimplemented as reshape/transpose.
  PyTorch's `unfold` output channel ordering is C-major then kernel-row then
  kernel-col — i.e. index = `c*(r*r) + kh*r + kw`. Reproduce that ordering
  exactly; a transposed ordering here is silent and wrong. Write a direct
  equality test against `torch.nn.functional.unfold` on random input as part
  of the port, not as an afterthought.

  Also note GELU: PyTorch `F.gelu` default is **exact erf** gelu, not tanh
  approximation. Match it (`mx.nn.gelu`, not `gelu_approx`) unless a numerical
  test shows otherwise.

### Quantization exclusion — MUST FIX BEFORE FIRST LOAD

`make_quantization_config()` at `mlx_lm/models/deepseek_v4.py:952` builds its
attention override as:

```python
attn = {k: mxfp8 for k, _ in flat_modules if ".attn.w" in k or ".attn.indexer.wq" in k}
```

The ViT modules are named `vision.blocks.N.attn.wqkv` and
`vision.blocks.N.attn.wo` — **both match `".attn.w" in k`**, so the vision
tower would be silently quantized to mxfp8 by the existing recipe. Worse, the
function's catch-all return `{"group_size": 64, "bits": 8, "mode": "affine", ...}`
would quantize `patch_embed`, `aligner.w1`, `aligner.w2` and the norms too.

The vision tower ships as plain bf16 (~466M params ≈ 0.9 GB) and should stay
bf16. Add an explicit exclusion so no key starting with `vision.` or
`aligner.` (and none of the 4 sentinel embeddings) receives a quant override
or the affine default. Verify by asserting on the constructed config dict in a
unit test — do not eyeball it.

### Phase 1 acceptance

A standalone numerical-parity test (CPU, no cluster, no full checkpoint
needed) comparing the MLX port against DeepSeek's PyTorch reference:

1. Instantiate both the torch `ViT`/`Aligner` from `inference/vision.py` and
   the new MLX versions with the real config values.
2. Copy identical random weights into both (torch → MLX).
3. Feed identical random patch input at a couple of realistic (n_h, n_w) grid
   shapes, including at least one where `n_h % 3 != 0` and `n_w % 3 != 0` so
   the aligner's padding path is exercised.
4. Report **actual max-absolute-difference and mean-absolute-difference
   numbers** for the ViT output and the Aligner output separately. bf16-level
   agreement (max abs diff ~1e-2 or better on bf16, ~1e-5 on fp32) is the
   target; report whatever it actually is, and if it is worse than that, do
   NOT declare success — find the discrepancy.
5. Separately assert the unfold reimplementation is exactly equal to
   `torch.nn.functional.unfold` (bitwise, on fp32).

Report the real numbers in the handoff. "Parity verified" with no numbers is
not an acceptable answer.

---

## Phase 2 — port the image processor

Reference: `inference/image_processor.py`. Port `grid_tokens`,
`solve_resize_ratio`, `safe_resize`, `load_image`, `build_image_block`, and
the placeholder-expansion half of `prepare_vl_inputs`.

Critical details:

- `COMPRESS_PAD_TO = 4` and `compress_pad = 3 - start_pos % 4` — the image
  block is padded so it aligns to a 4-token boundary. This exists because of
  the model's `compress_ratios == 4` pooling layers. Token counts are
  **derived from the prompt position**, so this cannot be approximated.
- `grid_tokens` returns `n_llm_h * (n_llm_w + 1) + 2`, plus a row if
  `n_llm_h` is odd, plus the `(n_llm_h+1)//2 * (n_llm_w+1) % 2 * 2` term.
  Reproduce the integer arithmetic exactly, including operator precedence in
  that last expression (`%` binds tighter than `*` — read it carefully).
- `build_image_block` produces a row-pair transpose permutation
  (`.view(rows//2, 2, row_len).transpose(1,2).reshape(-1)`) and a `perm`
  vector mapping aligner output rows into final token slots. Exact ordering
  matters.
- `load_image` normalization: RGB, `/255`, then `(x-0.5)/0.5`, cast bf16.
  Aspect clamp via `vision_max_wh_ratio`, min-pixel upscale via
  `vision_min_pixels`, then `ImageOps.pad` with fill color (127,127,127)
  — **except** when `width >= max_wh_ratio * height`, where it uses a plain
  `resize` instead of pad. Preserve that branch.
- Token IDs for sentinels are emitted as `vocab_size + type` where type is
  one of IMAGE_START/IMAGE_PAD/IMAGE/IMAGE_NEW_LINE/IMAGE_END = 0..4, i.e.
  **out-of-vocab IDs above 129280** used as an in-band signal. The embedding
  layer masks these (see `model.py`'s `mask` / `torch.where` logic) and
  `merge_image_embeddings` overwrites those positions. Phase 2 only needs to
  produce the correct token stream and `ImageInput` records; the embedding
  masking is Phase 1/4 territory, the routing/attention consumption is Phase 3
  and OUT OF SCOPE.

### Phase 2 acceptance

Golden test against the two examples DeepSeek ships:
`inference/examples/example_vl.txt` (compact `<image>path</image>` TXT
notation) and `inference/examples/example_vl_harmony.json` (OpenAI-style JSON
content blocks), with `inference/examples/images/carrots.jpeg` and
`corn.jpeg`. The repo README asserts these two "encode to identical prompts
and token IDs."

Required evidence in the handoff:
1. The two examples produce **identical** token ID sequences to each other
   under the port. Report the sequence length and a hash.
2. The port's token IDs match DeepSeek's own reference implementation run
   directly (torch, CPU — the processor needs no GPU and no model weights).
   Report exact match / first divergence index.
3. Reported per-image derived values (`n_llm_h`, `n_llm_w`, `n_vit_h`,
   `n_vit_w`, num_tokens) for both images.

An exact token-ID match against the reference is the bar. Anything less, say
so plainly and report the first divergence.

---

## HARD BOUNDARIES for this dispatch

1. **STOP AFTER PHASE 2.** Do not begin Phase 3. Phase 3 (the `bias_vl`
   second MoE routing bias, the hash-layer image-token routing change, and the
   `get_image_visible` / `get_window_topk_idxs_visible` image-span attention
   visibility) touches the most heavily tuned code in the fork and requires
   separate authorization. Do not modify `_hash_gate_route`, `MoEGate`, the
   attention/SDPA paths, the indexer, or anything related to
   `compress_ratios` pooling.

2. **DO NOT RESTART OR RELAUNCH THE CLUSTER.** Do not run `start_cluster.sh`.
   Do not kill the `exorun` screen session. Do not `git fetch && git reset` on
   either Mac Studio. The cluster is live serving `-0731` and must stay up.
   The only thing this dispatch does on the studios is delete one directory
   (0a) and download weights (0b).

3. **DO NOT EDIT FILES DIRECTLY ON THE MAC STUDIOS.** All code changes go in
   the local repo at `~/repos/adam.durham/...` — see git workflow below.

4. Phases 1 and 2 need **no weights and no cluster** — they are pure local
   code + CPU tests. Do not gate them on the download finishing.

## Git workflow

- exo repo: `/Users/adam.durham/repos/exo`, `origin` = `adurham/exo`.
- mlx-lm submodule: `/Users/adam.durham/repos/exo/mlx-lm`, `origin` =
  `adurham/mlx-lm`, and `ml-explore` = `https://github.com/ml-explore/mlx-lm.git`
  which is UPSTREAM — **never push, PR, or otherwise write to `ml-explore`.**
- Work on a branch named `dsv4-vision-port` in BOTH repos. Do not commit to
  `main` in either. Do not open a PR. Do not merge.
- Push the branch to `origin` only, so work survives.
- The exo repo working tree currently has pre-existing untracked files under
  `tmp/` and one modified file under `tmp/real-usage-capture-20260902/`.
  **Leave them alone** — do not commit, stash, clean, or reset them.
- If `git push` fails with `Permission denied (publickey)` (locked 1Password
  SSH agent), fix with `gh auth setup-git` and switch the remote to HTTPS
  rather than asking the user to unlock anything.
