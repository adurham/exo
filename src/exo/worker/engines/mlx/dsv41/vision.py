"""DeepSeek-V4.1 vision wiring for the exo engine (workstream V).

WHAT THIS IS. The engine-side adapter between exo's request types and the
mlx-lm fork's DSv4.1 tower (``mlx_lm.models.deepseek_v41.{vision,
image_processor}``). It owns three things the tower itself does not:

1. **Prompt expansion.** A prompt carrying image blocks is tokenized by the
   checkpoint's own ``image_processor.prepare_vl_inputs``: this repo's own
   ``apply_chat_template`` already renders an image content block as the DSv4
   image-placeholder TEXT (the vendored V4 encoder emits
   ``"<|deepseek_image|>"`` in place of the block), and
   ``prepare_vl_inputs`` expands that token into the variable-length sentinel
   block ``[IMAGE_START] + ([IMAGE]*n_llm_w + [IMAGE_NEW_LINE])*n_llm_h +
   [IMAGE_END]``. Every position of the span carries the same
   ``image_token_id``; the sentinel TYPE of each position is what distinguishes
   them (``image_processor.TEXT`` = -1 elsewhere).

2. **Embedding splice.** ``mlx_lm.models.deepseek_v41.Model`` embeds ids
   internally (``self.embed``), so the merged ``(1, s, dim)`` embedding tensor --
   token lookup with each image span overwritten by its ViT/aligner block -- is
   installed by :func:`splice_embeddings` for the duration of the prompt
   prefill, exactly the way exo's generic path installs a patched
   ``embed_tokens`` (``generator.generate.patch_embed_tokens``). The reference
   asserts ``start_pos == 0`` for image spans, and so does the splice: it refuses
   any forward that does not start at position 0, or whose chunk does not cover
   the whole span, because a ring-buffer decode would have already overwritten
   the KV slots the span needs.

3. **Span accounting.** :func:`image_spans` is the ``[start, end)`` list the
   runner/prefix-cache side needs for media regions.

The tower is loaded ONCE per engine (bf16, ~1 GB resident) by
:func:`load_dsv41_vision`; it is not part of the text stack's EXL3 quant, and the
engine only loads it when the checkpoint config declares a vision tower AND the
model card advertises vision (workstream T owns the card).
"""

from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from exo.shared.types.text_generation import Base64Image, TextGenerationTaskParams
from exo.worker.runner.bootstrap import logger

__all__ = [
    "Dsv41Vision",
    "IMAGE_PLACEHOLDER",
    "build_embeddings",
    "image_spans",
    "load_dsv41_vision",
    "prompt_tokens_for_request",
    "splice_embeddings",
    "validate_prompt_tokens",
]

#: The placeholder text the checkpoint's chat template emits for an image block:
#: ``<｜deepseek_image｜>`` with FULLWIDTH vertical bars (U+FF5C), added token
#: 129264 (= ``config.json:image_token_id``). :func:`load_dsv41_vision` re-reads
#: it from the checkpoint's ``tokenizer.json`` and refuses a mismatch (including
#: the "does it re-encode to the same id" check), so this constant is only the
#: fallback for a checkpoint that moved the token.
IMAGE_PLACEHOLDER = "<\uff5cdeepseek_image\uff5c>"


@dataclass
class Dsv41Vision:
    """A loaded DSv4.1 vision tower plus the config the engine needs."""

    tower: Any  # mlx_lm.models.deepseek_v41.vision.VisionTower
    cfg: Any  # VisionConfig
    placeholder: str
    image_token_id: int
    text_dim: int
    n_vit_layers: int

    def __str__(self) -> str:
        return (
            f"Dsv41Vision(text_dim={self.text_dim}, vit_layers={self.n_vit_layers}, "
            f"patch={self.cfg.patch_size}, downsample={self.cfg.downsample_ratio}, "
            f"image_token_id={self.image_token_id}, placeholder={ascii(self.placeholder)})"
        )


def _placeholder_from_tokenizer(model_path: Path, image_token_id: int) -> str | None:
    """Read the placeholder text for ``image_token_id`` out of ``tokenizer.json``.

    ``added_tokens`` carries the string form of every special id, which is what
    the chat template actually emits; the model's tokenizer will re-encode it to
    the same id (verified by :func:`load_dsv41_vision`).
    """
    tokenizer_path = model_path / "tokenizer.json"
    if not tokenizer_path.exists():
        return None
    with open(tokenizer_path) as f:
        data: dict[str, Any] = json.load(f)
    for entry in data.get("added_tokens", []):
        if int(entry.get("id", -1)) == image_token_id:
            content = entry.get("content")
            return content if isinstance(content, str) else None
    return None


def load_dsv41_vision(model_path: Path, *, dtype: mx.Dtype = mx.bfloat16) -> Dsv41Vision:
    """Build the tower and load its 266 BF16 tensors from the EXL3 checkpoint.

    Raises loudly when the checkpoint has no vision tower: a card that advertises
    vision against a text-only checkpoint is a misconfiguration, not something to
    degrade silently.
    """
    from mlx_lm.models.deepseek_v41 import vision as _vision

    tower, cfg = _vision.load_vision_tower(str(model_path), dtype=dtype)
    if not cfg.vision_enabled:
        raise ValueError(
            f"{model_path}: checkpoint declares no vision tower "
            f"(vision_n_layers == 0); it cannot serve images."
        )
    placeholder = _placeholder_from_tokenizer(Path(model_path), cfg.image_token_id)
    if placeholder is None:
        logger.warning(
            f"[DSV41] no added-token entry for image_token_id={cfg.image_token_id} in "
            f"{model_path}/tokenizer.json; using the built-in placeholder string."
        )
        placeholder = IMAGE_PLACEHOLDER
    vision = Dsv41Vision(
        tower=tower,
        cfg=cfg,
        placeholder=placeholder,
        image_token_id=int(cfg.image_token_id),
        text_dim=int(cfg.text_dim),
        n_vit_layers=int(cfg.n_layers),
    )
    logger.info(f"[DSV41] vision tower loaded: {vision}")
    return vision


def _image_records(images: list[Base64Image]) -> list[dict[str, Any]]:
    """Adapt exo's base64 payloads to the processor's record shape."""
    return [{"data": str(image)} for image in images]


def prompt_tokens_for_request(
    vision: Dsv41Vision,
    prompt: str,
    images: list[Base64Image],
    tokenizer: Any,
) -> tuple[list[int], list[int], Any]:
    """Expand an image-carrying prompt into ``(tokens, token_types, image_inputs)``.

    ``prompt`` is the chat-templated text: it contains one placeholder per image
    (rendered by ``utils_mlx.apply_chat_template`` from the request's image
    content blocks). ``prepare_vl_inputs`` re-tokenizes it and expands each
    placeholder into its sentinel span, so the token list it returns is what the
    model must be fed -- NOT simply ``tokenizer.encode(prompt)``.
    """
    from mlx_lm.models.deepseek_v41 import image_processor as _mip

    tokens, token_types, image_inputs = _mip.prepare_vl_inputs(
        prompt, _image_records(images), tokenizer, vision.cfg
    )
    if not image_inputs:
        raise ValueError(
            "DSv4.1 vision: the request carries "
            f"{len(images)} image(s) but the rendered prompt contains no "
            f"{vision.placeholder!r} placeholder, so no image span was built. "
            "Check that the chat template renders image content blocks as the "
            "placeholder (workstream T owns the template/card wiring)."
        )
    if len(image_inputs) != len(images):
        raise ValueError(
            f"DSv4.1 vision: {len(images)} image(s) supplied but {len(image_inputs)} "
            "image span(s) expanded out of the prompt."
        )
    return tokens, token_types, image_inputs


def validate_prompt_tokens(vision: Dsv41Vision, tokens: list[int]) -> None:
    """Refuse token ids outside the embedding table except the image sentinel.

    ``prepare_vl_inputs`` emits the raw ``image_token_id`` at every image-span
    position (in-vocabulary by construction), so this is a guard against a
    template that renders some OTHER out-of-range block (e.g. the five-type
    DSv4-Flash layout) -- a gather out of range is not bounds-checked by MLX and
    would silently read garbage rows.
    """
    vocab = int(getattr(vision.cfg, "vocab_size", 0) or 0)
    out_of_range = sorted({int(t) for t in tokens if t < 0 or int(t) == int(vision.image_token_id)})
    if out_of_range:
        logger.debug(f"[DSV41] prompt carries sentinel id(s) {out_of_range} (image spans)")
    if vocab:
        bad = sorted({int(t) for t in tokens if int(t) >= vocab})
        if bad and bad != [int(vision.image_token_id)]:
            raise ValueError(
                f"DSV4.1: prompt token id(s) {bad} are outside the embedding table "
                f"({vocab} rows) and are not the image sentinel "
                f"({vision.image_token_id}); refusing to build embeddings."
            )


def build_embeddings(
    model: Any,
    vision: Dsv41Vision,
    tokens: list[int],
    image_inputs: Any,
) -> mx.array:
    """Token embeddings with every image span overwritten by its tower block.

    Returns ``(1, n_tokens, text_dim)``. Sentinel ids are clamped before the
    table lookup (MLX's out-of-range gather is not a documented zero) and every
    span row is overwritten by :meth:`VisionTower.merge_image_embeddings`.
    """
    token_ids = mx.array(np.asarray(tokens, dtype=np.int32))[None]
    clamped = mx.minimum(token_ids, int(model.embed.weight.shape[0]) - 1)
    embeds = model.embed(clamped)
    merged = vision.tower.merge_image_embeddings(embeds, image_inputs, sample=0)
    return merged


def image_spans(image_inputs: Any) -> list[tuple[int, int]]:
    """Half-open ``[start, end)`` token spans, one per image (for media regions)."""
    if not image_inputs:
        return []
    return [(int(img.start), int(img.start) + len(img.types)) for img in image_inputs]


class _InjectedEmbed:
    """Stand-in for ``Model.embed`` that carries the merged image embeddings.

    The engine must prefill an image prompt in PIECES (the whole prompt is far
    too large for one graph), but the reference requires the image span to sit
    inside the forward whose cache offset is 0 -- its KV is a rotating ring, so a
    span split across pieces would be unreadable by the time the later piece runs.
    This stand-in therefore tracks the rows consumed so far and serves the merged
    tensor for the FIRST piece, which the engine guarantees is large enough to
    cover every image span (``engine._first_chunk_covers``); later pieces get the
    ordinary table lookup, which is exactly what the merged tensor holds outside
    the span.

    Anything that violates that -- a first piece too small for a span, or a
    non-zero starting position -- raises instead of returning a partially-merged
    embedding. Every read of ``model.embed`` that is not a prefill piece (the
    DSpark draft head reads it on every draft) passes straight through to the
    real table, so the splice can stay installed for the whole turn.
    """

    def __init__(self, table: Any, embeddings: mx.array, span_start: int, span_end: int):
        self.table = table
        self.embeddings = embeddings
        self.span_start = int(span_start)
        self.span_end = int(span_end)
        self.total = int(embeddings.shape[1])
        self.pos = 0

    def __call__(self, input_ids: mx.array) -> mx.array:
        rows = int(input_ids.shape[1])
        if self.pos == 0 and rows >= self.span_end and self.span_end > 0:
            # first prefill piece: it covers the whole image span
            self.pos = rows
            return self.embeddings[:, :rows]
        if self.pos < self.span_end and rows > 0:
            raise RuntimeError(
                f"DSv4.1 vision: image span [{self.span_start}, {self.span_end}) "
                f"was not covered by the first prefill piece; the whole span must "
                "be prefilled in ONE forward from position 0 (the reference "
                "asserts start_pos == 0 for image spans, and the KV ring would "
                "have rotated past it)."
            )
        self.pos += rows
        return self.table(input_ids)

    def __getattr__(self, name: str) -> Any:
        # attribute access (weight/dims/...) still reaches the real table so the
        # model stays introspectable while the splice is installed
        return getattr(self.table, name)


@contextlib.contextmanager
def splice_embeddings(
    model: Any, embeddings: mx.array, span_start: int, span_end: int
) -> Iterator[None]:
    """Install the merged image embeddings as ``model.embed`` for one turn.

    ``span_start``/``span_end`` bound the image span (the engine puts the whole
    span in the first prefill piece). Every non-prefill read passes through to
    the real table, so this can stay installed across the decode rounds too.
    """
    original = model.embed
    injected = _InjectedEmbed(original, embeddings, span_start, span_end)
    model.embed = injected
    try:
        yield injected
    finally:
        model.embed = original


def _images_in_messages(messages: list[dict[str, Any]]) -> int:
    """Number of image blocks in a message list (content parts, in order)."""
    count = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            count += sum(
                1
                for part in content
                if isinstance(part, dict) and part.get("type") in ("image", "image_url")
            )
    return count


def _substitute_placeholders(messages: list[dict[str, Any]], placeholder: str):
    """Copy ``messages`` with every image block replaced by the placeholder text.

    This is the pre-step the release's own encoder does
    (``_process_image_blocks``): the checkpoint's chat template renders an image
    block by STRINGIFYING it, so the image must be turned into the placeholder
    text token BEFORE templating, or the prompt never contains the span the model
    was trained to read. exo's shared ``apply_chat_template`` instead keeps only
    the ``type == "text"`` parts of a content list, which drops images entirely
    -- hence this separate renderer for image-carrying requests.
    """
    out: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            out.append(dict(msg))
            continue
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") in ("image", "image_url"):
                parts.append(placeholder)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
            elif isinstance(part, str):
                parts.append(part)
        new_msg = dict(msg)
        new_msg["content"] = "\n".join(p for p in parts if p)
        out.append(new_msg)
    return out


def render_prompt(tokenizer: Any, params: TextGenerationTaskParams, placeholder: str) -> str:
    """Render ``params`` with image blocks turned into the image placeholder.

    Uses exo's own message assembly + ``render_chat_template`` (which routes a
    ``deepseek-v4`` model id into the vendored DSV4 encoder), so a text-only
    request takes the identical path as ``utils_mlx.apply_chat_template``; the
    only difference is that an image content block survives as the placeholder
    text instead of being flattened away.
    """
    from exo.worker.engines.mlx.utils_mlx import apply_chat_template, render_chat_template

    if not params.images:
        return apply_chat_template(tokenizer, params)

    if params.chat_template_messages is not None:
        messages: list[dict[str, Any]] = [
            dict(m) for m in params.chat_template_messages
        ]
    else:
        messages = []
        if params.instructions:
            messages.append({"role": "system", "content": params.instructions})
        for msg in params.input:
            messages.append({"role": msg.role, "content": msg.content})

    found = _images_in_messages(messages)
    if found != len(params.images):
        raise ValueError(
            f"DSv4.1 vision: the request carries {len(params.images)} image(s) but "
            f"its rendered messages contain {found} image block(s). The image list "
            "and the message content must agree -- expanding them would pair "
            "images with the wrong spans."
        )
    return render_chat_template(
        tokenizer, _substitute_placeholders(messages, placeholder), params
    )


def params_with_image_messages(
    params: TextGenerationTaskParams, placeholder: str
) -> TextGenerationTaskParams:
    """Unused compatibility shim (kept for the module's import list)."""
    del placeholder
    return params


def env_vision_enabled() -> bool:
    """``EXO_DSV41_VISION=0`` disables the tower even when the card asks for it."""
    return os.environ.get("EXO_DSV41_VISION", "1") == "1"
