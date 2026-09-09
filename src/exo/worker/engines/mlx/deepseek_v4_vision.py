# The `logger` re-exported by `exo.worker.runner.bootstrap` is annotated
# `loguru.Logger`, but loguru ships no stubs in this environment, so
# basedpyright resolves it as unknown. That is a pre-existing, repo-wide
# condition (62 occurrences across 30+ modules at the pre-Phase-4 baseline,
# e.g. api/main.py, download/coordinator.py, engines/mlx/cache.py), not
# something specific to this module. Suppressed narrowly, for the logger only.
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
"""DeepSeek-V4-Flash-Vision-Exp's native vision path.

`vision.py` proper is built around mlx-vlm's shape: an HF `AutoImageProcessor`,
a `config["vision_config"]` sub-dict, `mlx_vlm.vision.VisionModel`, and a
single `image_token_id` repeated once per output token. DSv4 matches none of
that, so it gets its own path here rather than having the mlx-vlm one contorted
around it. `vision.py` keeps the shared `VisionProcessor` / `VisionResult` /
`MediaRegion` interfaces and dispatches into this module on
`VisionCardConfig.scheme == "deepseek_v4"`.

How DSv4 differs, and what each difference implies:

* Config keys are TOP-LEVEL (`vision_dim`, `vision_n_layers`, ...) with no
  `vision_config` sub-dict and no `image_token_id`, so the model is not
  autodetected and its card declares `[vision]` explicitly.
* The tower ships INSIDE the main checkpoint under `vision.` / `aligner.`
  prefixes, already built and weight-loaded as part of the text model. There is
  nothing separate to construct or load here -- `Model.encode_image` is called
  directly on the live model.
* Image tokens are five SENTINEL TYPES offset past the vocabulary
  (`vocab_size + {0..4}`), not one repeated placeholder id. They are
  deliberately outside the embedding table: the reference never looks them up,
  it overwrites those rows.
* The prompt marker (`<|deepseek_image|>`, one token) is EXPANDED into a
  variable-length sentinel block whose geometry depends on the image's own
  aspect ratio, so the token stream is rewritten after tokenization rather than
  before it.

The merge reproduces the reference's `Transformer.merge_image_embeddings`::

    params = torch.stack([image_start, image_pad, image_pad, image_newline, image_end])
    for img in sample:
        embeds = self.encode_image(img.patches, img.n_vit_h, img.n_vit_w)[img.perm]
        block = params[img.types]
        block[img.types == IMAGE] = embeds
        h[i, img.start:img.start + block.size(0)] = block

Note the deliberate duplicate at index 1: `params` is indexed by sentinel TYPE,
and types 1 (IMAGE_PAD) and 2 (IMAGE) both map to `image_pad` in the stack --
type 2's rows are then overwritten wholesale by the aligner output, so the
value parked there only has to be shape-correct, never used.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any, Protocol, cast

import mlx.core as mx
import numpy as np
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.types.text_generation import Base64Image, TextGenerationTaskParams
from exo.worker.engines.mlx.types import Model
from exo.worker.engines.mlx.vendor.deepseek_v4_image_processor import (
    IMAGE,
    ImageInput,
    VisionEncoderConfig,
    expand_image_placeholders,
)
from exo.worker.runner.bootstrap import logger

if TYPE_CHECKING:
    from exo.shared.models.model_cards import VisionCardConfig

#: Sentinel TYPE -> which top-level parameter supplies that token's embedding.
#: Index order is the reference's `torch.stack([...])`, and must not be
#: reordered: it is indexed by raw type value.
_SENTINEL_PARAMETER_NAMES: tuple[str, ...] = (
    "image_start",  # 0 IMAGE_START
    "image_pad",  # 1 IMAGE_PAD
    "image_pad",  # 2 IMAGE      (overwritten by aligner output)
    "image_newline",  # 3 IMAGE_NEW_LINE
    "image_end",  # 4 IMAGE_END
)


class _DeepseekV4VisionModel(Protocol):
    """The subset of the DSv4 MLX model this module actually touches.

    `exo.worker.engines.mlx.types.Model` is the generic mlx-lm model type and
    knows nothing about `encode_image` or the sentinel parameters, which exist
    only when `vision_n_layers > 0`. Casting through this Protocol states the
    requirement precisely instead of casting to `Any` and losing every downstream
    type.
    """

    def encode_image(
        self, patches: mx.array, n_vit_h: int, n_vit_w: int
    ) -> mx.array: ...


class _EmbeddingModule(Protocol):
    def embed_tokens(self, input_ids: mx.array) -> mx.array: ...


class _HasInnerModel(Protocol):
    @property
    def model(self) -> _EmbeddingModule: ...


def load_vision_encoder_config(config: dict[str, Any]) -> VisionEncoderConfig:
    """Read the vision-relevant subset of a parsed DSv4 ``config.json``."""
    return VisionEncoderConfig.from_config_mapping(config)


def image_records_from_base64(images: list[Base64Image]) -> list[dict[str, object]]:
    """Adapt exo's base64 image payloads to the processor's record shape.

    The processor accepts several record shapes (raw bytes, base64 text, an
    Anthropic `source` block, a URL, a path). exo hands us base64 text, which
    is the `data` shape.
    """
    return [{"data": str(image)} for image in images]


def expand_prompt_tokens(
    prompt_token_ids: list[int],
    images: list[Base64Image],
    placeholder_token_id: int,
    config: VisionEncoderConfig,
) -> tuple[list[int], list[ImageInput]]:
    """Expand each placeholder token into its full sentinel block.

    Thin wrapper over the Phase 2 processor, kept here so the base64 adaptation
    lives next to the rest of the DSv4-specific plumbing.
    """
    return expand_image_placeholders(
        prompt_token_ids,
        image_records_from_base64(images),
        placeholder_token_id,
        config,
    )


def _sentinel_parameters(model: Model) -> mx.array:
    """Stack the four sentinel embeddings, indexed by sentinel TYPE.

    Mirrors the reference's
    ``torch.stack([image_start, image_pad, image_pad, image_newline, image_end])``.
    """
    rows: list[mx.array] = []
    for name in _SENTINEL_PARAMETER_NAMES:
        parameter = getattr(model, name, None)
        if parameter is None:
            raise ValueError(
                f"DeepSeek-V4 vision: model is missing the top-level `{name}` "
                "sentinel embedding. The vision tower is only constructed when "
                "`vision_n_layers > 0`; this looks like a text-only checkpoint "
                "loaded against a vision model card."
            )
        rows.append(cast(mx.array, parameter))
    return mx.stack(rows)


def encode_image_block(
    model: Model,
    image: ImageInput,
    sentinel_parameters: mx.array,
) -> mx.array:
    """Build one image's ``(n_block_tokens, hidden_size)`` embedding block.

    Reproduces the reference's per-image body:

    * run ViT + Aligner over the patches,
    * permute the aligner rows into final emission order with ``img.perm``,
    * lay the sentinel embeddings out by ``img.types``,
    * overwrite the ``IMAGE``-typed rows with the permuted aligner output.

    The reference mutates in place (``block[types == IMAGE] = embeds``); MLX
    arrays are immutable, so the same result is expressed as a `where` over a
    scatter of the aligner rows into their destination slots.
    """
    # `Model` is mlx-lm's generic model type and does not declare
    # `encode_image`, so basedpyright rejects a direct cast as non-overlapping;
    # go through `object`, exactly as it suggests.
    vision_model = cast("_DeepseekV4VisionModel", cast("object", model))
    encoded = vision_model.encode_image(
        mx.array(image.patches), image.n_vit_h, image.n_vit_w
    )
    permutation = mx.array(image.perm)
    permuted = encoded[permutation]

    types = mx.array(image.types)
    block = sentinel_parameters[types].astype(permuted.dtype)

    # `mx.equal` rather than `types == IMAGE`: the latter is typed as returning
    # `bool` (the `__eq__` signature), which then poisons every downstream use.
    is_image_token = mx.equal(types, mx.array(IMAGE))
    n_image_tokens = int(mx.sum(is_image_token.astype(mx.int32)).item())
    if n_image_tokens != permuted.shape[0]:
        raise ValueError(
            "DeepSeek-V4 vision: image block has "
            f"{n_image_tokens} IMAGE slots but the aligner produced "
            f"{permuted.shape[0]} rows. The token block and the ViT grid "
            "disagree, which means the patch geometry and the emitted block "
            "were computed from different image dimensions."
        )

    # Scatter `permuted` into the IMAGE slots: `cumsum - 1` numbers the IMAGE
    # slots 0..n-1 in order, and non-IMAGE rows read a clamped (unused) index
    # that the `where` discards.
    slot_index = mx.cumsum(is_image_token.astype(mx.int32)) - 1
    slot_index = mx.clip(slot_index, 0, permuted.shape[0] - 1)
    scattered = permuted[slot_index]
    return mx.where(is_image_token[:, None], scattered, block)


def merge_image_embeddings(
    model: Model,
    token_embeddings: mx.array,
    image_inputs: list[ImageInput],
) -> mx.array:
    """Write every image's block into ``token_embeddings`` at its own offset.

    ``token_embeddings`` is ``(1, n_tokens, hidden_size)`` -- the FULL prompt
    already embedded, text rows included. Returns a new array; the input is not
    mutated.

    This is the reference's ``merge_image_embeddings`` for a single sample.
    """
    if not image_inputs:
        return token_embeddings

    sentinel_parameters = _sentinel_parameters(model)
    n_tokens = token_embeddings.shape[1]
    merged = token_embeddings

    for image in image_inputs:
        block = encode_image_block(model, image, sentinel_parameters)
        start = image.start
        end = start + block.shape[0]
        if end > n_tokens:
            raise ValueError(
                f"DeepSeek-V4 vision: image block at {start} runs to {end}, "
                f"past the {n_tokens}-token prompt. The expanded token stream "
                "and the embedding buffer were built from different prompts."
            )
        merged = mx.concatenate(
            [
                merged[:, :start, :],
                block.astype(merged.dtype)[None],
                merged[:, end:, :],
            ],
            axis=1,
        )

    return merged


def image_span_bounds(image_inputs: list[ImageInput]) -> list[tuple[int, int]]:
    """Half-open ``[start, end)`` token spans, one per image.

    Consumed by the prefill chunk-boundary guard, which must never place a
    chunk boundary inside one of these.
    """
    return [(image.start, image.start + len(image.types)) for image in image_inputs]


def build_embeddings(
    model: Model,
    prompt_token_ids: list[int],
    image_inputs: list[ImageInput],
    vocabulary_size: int,
) -> mx.array:
    """Embed the expanded prompt and merge every image block into it.

    Returns ``(1, n_tokens, hidden_size)``, matching what
    ``patch_embed_tokens`` splices into the prefill.

    Sentinel ids are clamped before the table lookup. They sit at
    ``vocab_size + {0..4}``, outside the table; MLX's out-of-range gather
    happens to return zeros, but that is not a documented guarantee and those
    rows are overwritten by the merge regardless. Clamping keeps this dependent
    on defined behaviour only. The RAW ids are NOT clamped anywhere else --
    the model still receives them, because the MoE gate
    (``input_ids >= vocab_size`` selects ``bias_vl``) and the image-visibility
    mask both derive from them.
    """
    inner = cast("_HasInnerModel", cast("object", model)).model
    token_ids = mx.array(prompt_token_ids)[None]
    clamped = mx.minimum(token_ids, vocabulary_size - 1)
    token_embeddings = inner.embed_tokens(clamped)
    return merge_image_embeddings(model, token_embeddings, image_inputs)


def content_hash_for_image(image: Base64Image) -> str:
    """Stable per-image hash for the prefix cache's media regions."""
    from exo.worker.engines.mlx.vendor.deepseek_v4_image_processor import (
        load_image_bytes,
    )

    return hashlib.sha256(load_image_bytes({"data": str(image)})).hexdigest()


def process(
    images: list[Base64Image],
    chat_template_messages: list[dict[str, Any]],
    tokenizer: TokenizerWrapper,
    model: Model,
    task_params: TextGenerationTaskParams,
    vision_config: "VisionCardConfig",
    model_config: dict[str, Any],
) -> tuple[str, mx.array, mx.array, list[tuple[int, int]], list[str]]:
    """Run the DSv4 vision pipeline for one request.

    Returns ``(prompt, prompt_tokens, embeddings, image_spans, content_hashes)``.
    The caller (`VisionProcessor.process`) wraps this into a `VisionResult`.
    """
    from exo.worker.engines.mlx.utils_mlx import render_chat_template

    encoder_config = load_vision_encoder_config(model_config)

    placeholder = vision_config.placeholder_token
    if placeholder is None:
        raise ValueError(
            "DeepSeek-V4 vision: the model card must set "
            "`vision.placeholder_token`; it is the marker the prompt carries "
            "before expansion."
        )

    prompt = render_chat_template(tokenizer, chat_template_messages, task_params)

    placeholder_count = prompt.count(placeholder)
    if placeholder_count != len(images):
        # The encoder inserts one placeholder per image block. A mismatch means
        # the rendered prompt and the image list disagree, and expanding it
        # would silently pair images with the wrong spans.
        raise ValueError(
            f"DeepSeek-V4 vision: prompt carries {placeholder_count} "
            f"`{placeholder}` markers but {len(images)} image(s) were "
            "supplied."
        )

    prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    expanded_ids, image_inputs = expand_prompt_tokens(
        prompt_token_ids,
        images,
        vision_config.image_token_id,
        encoder_config,
    )

    logger.info(
        f"DSv4 vision: {len(images)} image(s), prompt {len(prompt_token_ids)} "
        f"-> {len(expanded_ids)} tokens after placeholder expansion"
    )
    for index, image in enumerate(image_inputs):
        logger.info(
            f"  image {index}: patches={image.patches.shape} "
            f"grid={image.n_vit_h}x{image.n_vit_w} "
            f"block={len(image.types)} tokens at [{image.start}, "
            f"{image.start + len(image.types)})"
        )

    embeddings = build_embeddings(
        model, expanded_ids, image_inputs, encoder_config.vocabulary_size
    )
    mx.eval(embeddings)

    prompt_tokens = mx.array(np.asarray(expanded_ids, dtype=np.int32))
    spans = image_span_bounds(image_inputs)
    hashes = [content_hash_for_image(image) for image in images]
    return prompt, prompt_tokens, embeddings, spans, hashes
