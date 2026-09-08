# DSv4-Flash-Vision-Exp — exo port plan, Phases 3 & 4

Continuation of `DSV4_VISION_PORT_PLAN.md` (Phases 0-2, COMPLETE).
Read that file first for the checkpoint facts, tensor manifest, and
architecture summary — they are not repeated here.

AUTHORIZED SCOPE FOR THIS DISPATCH: **Phases 3 and 4 only.**
Phase 5 (cluster relaunch + on-hardware smoke test) is explicitly NOT
authorized and requires separate user approval, because it means restarting
the live cluster.

---

## Status carried in from Phases 0-2

- Weights present and byte-verified on BOTH nodes at
  `~/.exo/models/deepseek-ai--DeepSeek-V4-Flash-Vision-Exp`
  (48/48 shards, weights total 167,819,404,368 bytes, identical
  size-manifest hash `4f746810a918de91…` on both nodes).
- Branch `dsv4-vision-port` in BOTH `~/repos/exo` and `~/repos/exo/mlx-lm`.
  - mlx-lm HEAD `557b1df` — ViT + Aligner MLX port, quant exclusion, tests.
  - exo HEAD `f7712949a` — torch-free image processor port + tests.
- Phase 1 parity (fp32, vs DeepSeek torch reference):
  ViT max abs 4.29e-06 / mean 4.91e-07; Aligner max abs 3.04e-06 /
  mean 4.10e-07.
- Phase 2: exact token-ID match, len=457, sha256 `6d0bb5e0…faef8`, for both
  the TXT and JSON example encodings.
- Cluster is LIVE on `deepseek-ai/DeepSeek-V4-Flash-0731`, instance
  `c8a841d9`, TP world_size=2, RDMA over `rdma_en3`/`rdma_en4`
  (192.168.201.x). Untouched by all work so far.

### Known pre-existing breakage (NOT yours, do not fix unless asked)

`mlx-lm/tests/test_models.py:13` imports `gated_delta_chunkwise` from
`mlx_lm.models.gated_delta`, but that module (present, 9,751 bytes, dated
Apr 3, last touched by upstream merge `69f58c0`) defines only
`gated_delta_kernel`, `gated_delta_ops`, `gated_delta_update`. This import
error predates the vision work and is unrelated to it. Do not "fix" it as
part of this task; just don't let it mask your own test failures.

---

## Phase 3 — the two invasive text-model changes

This is the risky phase: it modifies the most heavily tuned code in the fork.
Reference is DeepSeek's own `inference/model.py` from the Vision-Exp repo
(fetch from
`https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp/raw/main/inference/model.py`).
Diff it against the same file in the `-0731` repo to isolate exactly what
vision changed — that diff is 141 lines and is the authoritative spec for
this phase.

### 3a. `bias_vl` — a second MoE routing bias for image tokens

In the reference `Gate.forward`, when the model is vision-enabled
(`args.vision_n_layers > 0`) a second bias parameter `bias_vl` of shape
`(n_routed_experts,)` float32 exists alongside the existing `bias`, and:

```python
image_mask = (input_ids >= self.vocab_size) if self.bias_vl is not None else None
...
if self.hash:                      # hash-routing layers (first num_hash_layers=3)
    if image_mask is None:
        indices = self.tid2eid[input_ids]
    else:
        indices = self.tid2eid[torch.where(image_mask, 0, input_ids)]
        vl_indices = (scores + self.bias_vl).topk(self.topk, dim=-1)[1]
        indices = torch.where(image_mask.unsqueeze(-1), vl_indices.to(indices.dtype), indices)
else:
    if image_mask is None:
        scores = scores + self.bias
    else:
        scores = scores + torch.where(image_mask.unsqueeze(-1), self.bias_vl, self.bias)
```

Two distinct behaviors to reproduce:

1. **Non-hash layers**: the additive routing bias becomes per-token —
   `bias_vl` for image tokens, `bias` for text tokens.
2. **Hash layers (layer_idx < num_hash_layers)**: image tokens BYPASS the
   `tid2eid` hash-table lookup entirely and do a real top-k over
   `scores + bias_vl`; text tokens keep the table lookup. Note the reference
   clamps the input id to 0 before the table lookup to keep the gather in
   range — image token ids are `>= vocab_size` and would otherwise index
   out of bounds.

Also note the reference moved `scores = scores + self.bias` from
unconditionally-before the hash branch to inside the non-hash branch. Read
the diff carefully: in `-0731` the bias was applied before the `if self.hash`
split; in Vision-Exp it is not applied on the hash path at all (the hash path
uses `bias_vl` only, for image tokens). Do not preserve the old ordering by
reflex.

Fork code sites:
- `mlx-lm/mlx_lm/models/deepseek_v4.py:3159` `class MoEGate`
- `mlx-lm/mlx_lm/models/deepseek_v4.py:3171` `self.tid2eid` allocation
- `mlx-lm/mlx_lm/models/deepseek_v4.py:1312` `def _hash_gate_route`
  (the fork's fused hash-routing fast path, `inds = tid2eid[input_ids]` at
  ~line 1337 — this is where the image bypass has to land)
- `mlx-lm/mlx_lm/models/deepseek_v4.py:3229` `class DeepseekV4MoE`

`_hash_gate_route` is a fork-only optimization that folds the matmul in;
upstream/DeepSeek have no equivalent. You must preserve its existing
text-only behavior bit-for-bit when no images are present — a text-only
regression here silently degrades the model the user runs every day.

**Mandatory guard**: add a test proving that with `images=None` /
no image tokens present, gate output (indices AND weights) is bitwise
identical to the pre-change implementation. Reconstruct the old function
from git if needed. This is the single most important test in Phase 3.

### 3b. Image-span attention visibility

Reference adds two functions (verbatim in the diff):

```python
def get_image_visible(input_ids, vocab_size, max_image_tokens):
    """Per-token visible counts to the left/right within each [IMAGE_START, IMAGE_END] span."""
    seqlen = input_ids.size(1)
    idx = torch.arange(seqlen, dtype=torch.int32).unsqueeze(0)
    is_start = input_ids == vocab_size + IMAGE_START
    is_end = input_ids == vocab_size + IMAGE_END
    valid = (is_start.cumsum(1) > is_end.cumsum(1)) | is_end
    starts = torch.where(is_start, idx, 0).cummax(1)[0]
    left = (idx - starts) * valid
    ends = torch.where(is_end, idx, seqlen).flip(1).cummin(1)[0].flip(1)
    right = (ends - idx) * valid
    return left.clamp(max=max_image_tokens - 1), right.clamp(max=max_image_tokens)


def get_window_topk_idxs_visible(window_size, seqlen, left, right, max_image_tokens):
    width = min(seqlen, window_size + max_image_tokens)
    idx = torch.arange(seqlen).unsqueeze(0)
    left_add = (left - (window_size - 1)).clamp(min=0)
    starts = (idx - (window_size - 1) - left_add).clamp(min=0)
    matrix = starts.unsqueeze(-1) + torch.arange(width)
    matrix = torch.where(matrix > (idx + right).unsqueeze(-1), -1, matrix)
    return matrix.int().contiguous()
```

Semantics: inside an `[IMAGE_START … IMAGE_END]` span, every token can see the
entire span (bounded by `vision_max_n_token` = 384) IN ADDITION to the normal
128-token sliding window. `-1` entries are the invalid/masked slots.

Wiring in the reference: `MLA.forward` gains a `visible=None` parameter and
switches
`topk_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos)` →
`get_window_topk_idxs_visible(win, seqlen, *visible, self.max_image_tokens)`
when `visible is not None`. `Transformer.forward` computes `visible` once
(only when `start_pos == 0`, i.e. prefill) and threads it down through every
layer. `self.max_image_tokens = args.vision_max_n_token` is set on both the
attention module and the top-level model.

**This is the hard part on this fork.** The fork does NOT have a single
`get_window_topk_idxs` to swap out — it has a family of heavily-optimized,
env-gated paths that all consume window/top-k geometry:
`EXO_DSV4_QUERY_TILED_SDPA` (query-tiled compressed SDPA),
`EXO_DSV4_EXACT_TOPK_PREFILL`, `EXO_DSV4_INDEX_TOPK`,
`EXO_DSV4_SPARSE_SDPA_TILE`, plus the `compress_ratios` pooling cache
(`PoolingCache`, overlap-carry logic) and the seq-split/TP coordinate frames.

Required approach — do NOT try to retrofit every optimized path at once:

1. **First** find and document the fork's equivalent of the reference's
   plain `get_window_topk_idxs`, and every call site that produces window/
   top-k index geometry. Write that inventory down before changing code.
2. Implement the visibility variant against the SIMPLEST correct path, and
   gate the whole feature behind a new env flag (default OFF) so no existing
   text path changes behavior by default.
3. For each optimized path, either (a) implement visibility correctly, or
   (b) explicitly and loudly fall back to the simple path when
   `visible is not None`. A documented fallback is acceptable for this
   phase; a silently-wrong fast path is not.
4. Prove text-only regression-freedom: with the flag OFF and with no image
   tokens, output must be bitwise identical to current `main`.

If it turns out the visibility semantics cannot be expressed in one of the
optimized kernels without a redesign, say so explicitly with the specific
blocking reason and propose the next attack vector. Do NOT silently disable
an optimization the user relies on, and do NOT declare it impossible without
having actually attempted a disposable prototype.

### Phase 3 acceptance

- Numerical parity vs DeepSeek's torch reference for the gate (routing
  indices + weights) on synthetic inputs containing BOTH text and image
  tokens, across hash layers (0,1,2) and non-hash layers. Paste real
  max/mean abs diff numbers and exact index-match counts.
- Numerical parity for `get_image_visible` and
  `get_window_topk_idxs_visible` against the torch reference — these are
  integer-valued, so require EXACT equality, not tolerance.
- The text-only bitwise-identity guard tests above, for BOTH 3a and 3b.
- Full existing mlx-lm DSv4 test suite still passes (modulo the
  pre-existing `gated_delta_chunkwise` import error noted above).

---

## Phase 4 — exo plumbing

### 4a. Refresh the vendored DeepSeek encoder

`src/exo/worker/engines/mlx/vendor/deepseek_v4_encoding.py` (872 lines) is a
frozen copy of DeepSeek's `encoding/encoding_dsv4.py`. The Vision-Exp version
is 957 lines vs `-0731`'s 760.

The drift is NOT vision-only — it changes the TEXT path too. In `-0731`, the
user-message merge path rebuilt a fresh 3-key dict and copied forward only
`("task", "wo_eos", "mask")`. In Vision-Exp it preserves the original message
object and all its metadata, and extends with existing `content_blocks`:

```python
content_blocks = msg.get("content_blocks")
if content_blocks is None:
    content_blocks = [{"type": "text", "text": msg.get("content", "")}]
...
new_msg = msg
new_msg["content_blocks"] = content_blocks
```

`encode_messages` was also split into `_encode_messages_text` plus a new
vision-aware `encode_messages` wrapper with a `return_multi_modal_data`
parameter returning `(prompt, media_data)`.

New vision surface to carry over: `IMAGE_PLACEHOLDER = "<｜deepseek_image｜>"`,
`IMAGE_TAG_PATTERN`, `parse_tagged_text`, `_is_image_block`, `_extract_image`,
`_process_image_blocks`, `_validate_no_image_sp_tokens`,
`process_image_messages`.

Diff all three versions (vendored copy, `-0731` upstream, Vision-Exp
upstream) before editing. Confirm the vendored copy still matches `-0731`
exactly first — if it does not, the fork has local modifications that must be
preserved through the refresh. Check `git log` on the vendored file.

Then trace every fork consumer of the changed functions (grep for
`encode_messages`, `_v4_reasoning_effort` in
`src/exo/worker/engines/mlx/utils_mlx.py`) and confirm the new signatures
are wired correctly.

### 4b. DSv4-native branch in exo's vision layer

`src/exo/worker/engines/mlx/vision.py` (844 lines) is entirely mlx-vlm-shaped
and CANNOT serve this model:
- `_load_weights` (line 277) reads `config["vision_config"]` — DSv4 has no
  such sub-dict, its vision keys are top-level.
- It constructs `mlx_vlm.config.VisionConfig` / `mlx_vlm.vision.VisionModel`
  — wrong architecture entirely.
- `_load_weights_from_model_repo` (line 417) looks for prefixes
  `vision_tower.` / `model.visual.` — DSv4 uses `vision.` and `aligner.`.
- It loads an HF `AutoImageProcessor` — DSv4 ships no HF processor config;
  Phase 2's port replaces it.

Add a DSv4-native path rather than contorting the mlx-vlm one. Keep the
existing `VisionProcessor` / `VisionResult` / `MediaRegion` interfaces intact
so `generator/generate.py:85`, `generator/batch_generate.py:68`,
`batch_generator.py:50`, `cache.py:35`, and `utils_mlx.py:311` keep working
unchanged. Entry point is `prepare_vision` at `vision.py:821`.

`VisionCardConfig` (`src/exo/shared/models/model_cards.py:134`) has fields
`image_token_id`, `model_type`, `weights_repo`, `image_token`,
`processor_repo` — none of which describe DSv4's scheme (sentinel token
types offset by `vocab_size`, no HF processor). Extend the model, don't
abuse the existing fields. Note `detect_vision_from_config`
(`model_cards.py:98`) and the autodetect at line 325 currently key off
`vision_config` + `image_token_id` in config.json, so DSv4 will NOT be
autodetected — it needs an explicit branch.

### 4c. Model card

Add `resources/inference_model_cards/deepseek-ai--DeepSeek-V4-Flash-Vision-Exp.toml`.
Base on `deepseek-ai--DeepSeek-V4-Flash-0731.toml` (read it — it documents
its own provenance and caveats well, follow that style).

- `storage_size.in_bytes = 167811372792` (from
  `model.safetensors.index.json` → `metadata.total_size`, NOT the HF API
  page figure).
- `capabilities` must add `"vision"` to the `-0731` set
  `["text", "thinking", "thinking_toggle"]`.
- Everything else (n_layers 43, hidden_size 4096, num_key_value_heads 1,
  context_length 1048576, quantization "fp8", family "deepseek",
  reasoning_dialect "tool_conditional", backends) carries over from `-0731`.
- Mark sampling defaults explicitly UNVERIFIED for this checkpoint. Note
  the repo's own README recommends temp=1.0 / top_p=0.95 for agentic use.

### 4d. Prefill chunk boundary guard

The reference asserts image spans must be prefilled in a single chunk:

```python
assert (input_ids < self.vocab_size).all(), "image spans must be prefilled in a single chunk"
```

The cluster runs `EXO_PREFILL_STEP_SIZE=2048`. A 384-token image span fits
inside one 2048 chunk, but only if the chunker never splits mid-span. Add an
explicit guard so a span is never split across chunk boundaries — either by
snapping the chunk boundary or by asserting loudly. Do not leave it to luck.
Sites: `src/exo/worker/runner/llm_inference/batch_generator.py:108,370,414,471`
(`prefill_step_size` plumbing) and the pp scheduler references at
`src/exo/worker/engines/mlx/pp_scheduler_protocol.py:659,855`.

### 4e. TP replication of the vision tower

The cluster runs Tensor-parallel world_size=2. The ViT + Aligner are dense
(~466M params, <1 GB bf16). Simplest correct approach is to replicate them
identically on every rank and run image encoding redundantly — no collective
needed, no sharding, and it avoids a cross-node round trip per image.
Confirm this against how the fork places non-sharded modules today
(check `TensorShardMetadata` handling in the builder) and state explicitly
which approach was taken and why.

### Phase 4 acceptance

- `uv run basedpyright` 0 errors, `uv run ruff check` clean, `nix fmt`
  applied, `uv run pytest` passing (per repo AGENTS.md).
- Model card parses and loads via the fork's own card loader.
- An offline end-to-end test: real image → Phase 2 processor → Phase 1 ViT
  + Aligner → embeddings merged into a token stream, with shapes and token
  counts asserted against the reference. This can run on CPU/one machine
  and must NOT touch the cluster.

---

## HARD BOUNDARIES

1. **STOP AFTER PHASE 4.** Do not run Phase 5. Do not attempt an
   on-cluster smoke test.
2. **DO NOT RESTART OR RELAUNCH THE CLUSTER.** Do not run
   `start_cluster.sh`. Do not kill the `exorun` screen sessions. Do not
   `git fetch`/`git reset` on either Mac Studio. The cluster is LIVE on
   `-0731` and the user is working against it.
3. **DO NOT EDIT FILES DIRECTLY ON THE MAC STUDIOS.** All changes go in the
   local repos.
4. Do not touch the 192.168.201.x network — it is the live RDMA
   interconnect.
5. Branch `dsv4-vision-port` in both repos, continuing from the existing
   commits. Never commit to `main`. No PRs, no merges. Push to `origin`
   (adurham fork) only — NEVER to the `ml-explore` remote in mlx-lm.
6. Leave the exo repo's pre-existing untracked `tmp/` files alone.
7. Root-cause fixes only. No mitigations, retry wrappers, defensive
   timeouts, or "good enough" approximations. If a fix is hard, scope the
   real fix and report the blocker with a next attack vector.
8. Every numerical claim must come with actual pasted numbers. "Parity
   verified" without numbers is not an acceptable report.
