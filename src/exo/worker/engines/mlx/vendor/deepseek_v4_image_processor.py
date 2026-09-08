"""Torch-free port of DeepSeek-V4-Flash-Vision-Exp's ``inference/image_processor.py``.

Ported from ``deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`` (HuggingFace), file
``inference/image_processor.py`` (184 lines, PyTorch). This module reproduces
that file's arithmetic and token layout **exactly**, using only the standard
library, ``numpy`` and ``PIL``. It deliberately does NOT import ``torch``: exo's
``src/`` tree must not gain a torch import dependency.

Scope: prompt-side work only -- deriving the resized ViT grid, cutting the image
into ViT patches, and expanding the ``<|deepseek_image|>`` placeholder token into
the model's in-band sentinel token stream. The downstream halves (embedding
masking, ``merge_image_embeddings``, attention visibility, compress-ratio
pooling) are intentionally not implemented here.

Numerical fidelity notes:

* The reference casts the normalized pixel tensor to ``torch.bfloat16``. numpy
  has no native bfloat16, so :func:`round_to_bfloat16_precision` performs the
  identical round-half-to-even reduction to 8 mantissa bits while keeping the
  values in ``float32`` storage. This was verified bit-for-bit against
  ``torch.Tensor.to(torch.bfloat16)`` over the pipeline's exact value domain,
  two million broad-magnitude random floats, all 65280 finite exact-tie values,
  a dense sweep around 1.0, and the zero/infinity/subnormal specials. An
  ``mlx.core`` bfloat16 cast was rejected: it flushes subnormal ties to zero and
  therefore diverges from the reference on exact-tie inputs.
* The sentinel token identifiers are emitted as ``vocabulary_size + token_type``,
  i.e. deliberately OUT OF VOCABULARY identifiers above 129279. That is the
  reference's in-band signalling scheme, not a defect; the embedding layer masks
  them downstream.
"""

from __future__ import annotations

import base64
import binascii
import io
import math
from collections.abc import Iterable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Final, Protocol, cast, final
from urllib.request import urlopen

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps

__all__ = [
    "COMPRESS_PAD_TO",
    "IMAGE",
    "IMAGE_END",
    "IMAGE_NEW_LINE",
    "IMAGE_PAD",
    "IMAGE_START",
    "ImageInput",
    "VisionEncoderConfig",
    "build_image_block",
    "expand_image_placeholders",
    "as_integer_list",
    "grid_tokens",
    "load_image",
    "load_image_bytes",
    "round_to_bfloat16_precision",
    "safe_resize",
    "solve_resize_ratio",
]

# Sentinel token TYPES. The reference spells this
# ``IMAGE_START, IMAGE_PAD, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(5)``;
# the values (0..4) are load-bearing because the emitted token identifier is
# ``vocabulary_size + type`` and the model's embedding table is indexed by it.
IMAGE_START: Final = 0
IMAGE_PAD: Final = 1
IMAGE: Final = 2
IMAGE_NEW_LINE: Final = 3
IMAGE_END: Final = 4

# The image block is left-padded so that it ends on a 4-token boundary, because
# the model's ``compress_ratios`` pooling layers operate with a ratio of 4. The
# amount of padding therefore depends on WHERE the placeholder sits in the token
# stream and cannot be precomputed independently of prompt position.
COMPRESS_PAD_TO: Final = 4


class SupportsRead(Protocol):
    """The single method this module needs from a ``urlopen`` response."""

    def read(self) -> bytes: ...


_HTTP_URL_SCHEMES: Final = ("http://", "https://")
_URL_READ_TIMEOUT_SECONDS: Final = 30

# The reference normalizes with ``x / 255`` then ``(x - 0.5) / 0.5``. Materialize
# the constants as float32 so the arithmetic can never widen to float64.
_PIXEL_VALUE_SCALE: Final = np.float32(255.0)
_NORMALIZE_SHIFT: Final = np.float32(0.5)
_NORMALIZE_SCALE: Final = np.float32(0.5)

# ImageOps.pad fill colour used by the reference for letterboxing.
_PAD_FILL_COLOR: Final = (127, 127, 127)


@final
@dataclass(frozen=True, slots=True)
class VisionEncoderConfig:
    """The vision-relevant subset of DeepSeek-V4-Flash-Vision-Exp's model config.

    The reference passes its whole ``ModelArgs`` around and reads flat
    ``vision_*`` attributes off it. Both the HuggingFace ``config.json`` and the
    reference's own ``inference/config.json`` spell these keys identically at the
    top level (there is no nested ``vision_config`` sub-dict), so
    :meth:`from_config_mapping` reads either one.
    """

    patch_size: int
    downsample_ratio: int
    max_token_count: int
    min_pixel_count: int
    max_width_height_ratio: int | None
    vocabulary_size: int

    @classmethod
    def from_config_mapping(cls, config: Mapping[str, object]) -> VisionEncoderConfig:
        """Build a config from a parsed ``config.json`` mapping.

        ``vision_max_wh_ratio`` is the one optional field: the reference guards
        every use of it with ``is not None``, so a checkpoint may omit it.
        """
        max_width_height_ratio = config.get("vision_max_wh_ratio")
        return cls(
            patch_size=_require_integer(config, "vision_patch_size"),
            downsample_ratio=_require_integer(config, "vision_downsample_ratio"),
            max_token_count=_require_integer(config, "vision_max_n_token"),
            min_pixel_count=_require_integer(config, "vision_min_pixels"),
            max_width_height_ratio=(
                None
                if max_width_height_ratio is None
                else _as_integer("vision_max_wh_ratio", max_width_height_ratio)
            ),
            vocabulary_size=_require_integer(config, "vocab_size"),
        )


@final
@dataclass(frozen=True, slots=True)
class ImageInput:
    """One expanded image's payload, mirroring the reference's ``ImageInput``.

    Attributes:
        start: Index of the image block's first token within the prompt token
            stream. This is the ``len(tokens)`` captured BEFORE the block was
            appended, exactly as the reference records it.
        patches: ``(n_vit_h * n_vit_w, 3, patch_size, patch_size)`` float32 array
            holding bfloat16-precision values (see
            :func:`round_to_bfloat16_precision`).
        n_vit_h: ViT patch-grid height.
        n_vit_w: ViT patch-grid width.
        types: The block's sentinel token TYPES in final emission order. Token
            identifiers are ``vocabulary_size + types``.
        perm: Maps aligner output rows onto the ``IMAGE`` token slots of
            ``types``, in final emission order.
    """

    start: int
    patches: NDArray[np.float32]
    n_vit_h: int
    n_vit_w: int
    types: NDArray[np.int64]
    perm: NDArray[np.int64]


def as_integer_list(array: NDArray[np.int64]) -> list[int]:
    """Narrow an ``int64`` array to ``list[int]``.

    ``numpy``'s ``ndarray.tolist()`` is typed as returning ``Any``, which under
    this repository's strict type-checking settings poisons every downstream
    expression. Narrowing it once, here, keeps the rest of the module (and its
    tests) free of scattered ignore comments.
    """
    return [int(value) for value in cast("list[int]", array.tolist())]


def _as_integer(key: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"Config key {key!r} must be an integer, got {type(value).__name__}"
        )
    return value


def _require_integer(config: Mapping[str, object], key: str) -> int:
    if key not in config:
        raise KeyError(f"Config is missing required key {key!r}")
    return _as_integer(key, config[key])


def round_to_bfloat16_precision(values: NDArray[np.float32]) -> NDArray[np.float32]:
    """Reduce float32 values to bfloat16 precision, keeping float32 storage.

    Reproduces ``torch.Tensor.to(torch.bfloat16)`` bit-for-bit: bfloat16 shares
    float32's sign and exponent fields and keeps only the top 8 mantissa bits, so
    the cast is a round-half-to-even truncation of the low 16 bits. The rounding
    bias is ``0x7FFF`` plus the least-significant retained mantissa bit, which is
    what makes exact ties round to even rather than away from zero.

    NaN inputs are restored explicitly because the rounding carry can otherwise
    propagate a NaN's mantissa into the exponent and turn it into an infinity.
    """
    bit_pattern = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    round_to_even_bias = ((bit_pattern >> np.uint32(16)) & np.uint32(1)) + np.uint32(
        0x7FFF
    )
    rounded = (
        ((bit_pattern + round_to_even_bias) & np.uint32(0xFFFF0000))
        .view(np.float32)
        .copy()
    )
    rounded[np.isnan(values)] = np.float32(np.nan)
    return rounded


def grid_tokens(
    best_height: int,
    best_width: int,
    patch_size: int,
    downsample_ratio: int,
) -> tuple[int, int, int]:
    """Number of LLM tokens the aligner grid occupies (N-layout, incl. padding).

    Returns ``(n_llm_h, n_llm_w, num_tokens)``.

    The final term is transcribed from the reference line

        ``num_tokens += (n_llm_h + 1) // 2 * (n_llm_w + 1) % 2 * 2``

    In Python, ``//``, ``*`` and ``%`` all sit at the SAME precedence level and
    associate LEFT TO RIGHT, so that expression parses as

        ``((((n_llm_h + 1) // 2) * (n_llm_w + 1)) % 2) * 2``

    and NOT as ``((n_llm_h + 1) // 2) * (((n_llm_w + 1) % 2) * 2)``. It is written
    fully parenthesized below so the grouping is not left to the reader.
    """
    n_llm_h = math.ceil((best_height // patch_size) / downsample_ratio)
    n_llm_w = math.ceil((best_width // patch_size) / downsample_ratio)
    num_tokens = n_llm_h * (n_llm_w + 1) + 2
    if n_llm_h % 2 == 1:
        num_tokens += n_llm_w + 1
    num_tokens += ((((n_llm_h + 1) // 2) * (n_llm_w + 1)) % 2) * 2
    return n_llm_h, n_llm_w, num_tokens


def solve_resize_ratio(
    height: int,
    width: int,
    patch_size: int,
    downsample_ratio: int,
    max_n_token: int,
) -> tuple[int, int, int, int, int]:
    """Solve for the largest aspect-preserving grid that fits ``max_n_token``.

    Returns ``(n_llm_h, n_llm_w, best_height, best_width, num_tokens)``.

    The reference asserts ``max_w > 1`` in the ``max_h_float < 2.0`` branch. That
    assertion is raised here as a ``ValueError`` instead: ``assert`` is stripped
    under ``python -O``, which would silently let a degenerate grid through.
    """
    aspect_ratio = height / width
    max_w_float = math.sqrt((max_n_token - 2) / aspect_ratio + 0.25) - 0.5
    max_h_float = max_w_float * aspect_ratio
    if max_w_float < 1.0:
        max_w = 1
        max_h = (max_n_token - 2) // (max_w + 1)
        if max_h % 2 == 1:
            max_h -= 1
        best_width = max_w * patch_size * downsample_ratio
        best_height = max_h * patch_size * downsample_ratio
    elif max_h_float < 2.0:
        max_h = 2
        max_w = ((max_n_token - 2) // max_h) - 1
        if max_w <= 1:
            raise ValueError(
                f"Token budget {max_n_token} is too small to encode a {height}x{width} image"
            )
        best_width = max_w * patch_size * downsample_ratio
        best_height = max_h * patch_size * downsample_ratio
    else:
        max_w = math.floor(max_w_float)
        max_h = math.floor(max_h_float)
        if max_h % 2 == 1:
            max_h -= 1
        beta = min(
            max_w * patch_size * downsample_ratio / width,
            max_h * patch_size * downsample_ratio / height,
        )
        best_width = math.floor(width * beta / patch_size) * patch_size
        best_height = math.floor(height * beta / patch_size) * patch_size
    n_llm_h, n_llm_w, num_tokens = grid_tokens(
        best_height, best_width, patch_size, downsample_ratio
    )
    return n_llm_h, n_llm_w, best_height, best_width, num_tokens


def safe_resize(
    height: int,
    width: int,
    best_height: int,
    best_width: int,
    patch_size: int,
    downsample_ratio: int,
    max_n_token: int,
) -> tuple[int, int, int, int]:
    """Shrink the grid until it fits the token budget.

    Returns ``(n_llm_h, n_llm_w, best_height, best_width)``.

    ``COMPRESS_PAD_TO - 1`` is subtracted from the budget up front to reserve room
    for the worst-case compress alignment padding, which is only known once the
    block's prompt position is known.
    """
    max_n_token -= COMPRESS_PAD_TO - 1
    n_llm_h, n_llm_w, num_tokens = grid_tokens(
        best_height, best_width, patch_size, downsample_ratio
    )
    budget = max_n_token
    while num_tokens > max_n_token:
        n_llm_h, n_llm_w, best_height, best_width, num_tokens = solve_resize_ratio(
            height, width, patch_size, downsample_ratio, budget
        )
        budget -= 1
    return n_llm_h, n_llm_w, best_height, best_width


def load_image_bytes(record: Mapping[str, object]) -> bytes:
    """Load image bytes from raw/base64 data, an Anthropic source, URL, or path."""
    data = record.get("data")
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        return _decode_base64(data)

    source = record.get("source")
    if isinstance(source, Mapping):
        # ``isinstance`` narrows only to ``Mapping[Unknown, Unknown]``; state the
        # key/value types once here rather than at every lookup below.
        source_mapping = cast("Mapping[str, object]", source)
        source_data = source_mapping.get("data")
        if source_data is not None:
            if not isinstance(source_data, str):
                raise TypeError(
                    f"Image source data must be base64 text, got {type(source_data).__name__}"
                )
            return _decode_base64(source_data)
        source_url = source_mapping.get("url")
        if source_url:
            return load_image_bytes({"url": source_url})

    url = record.get("url")
    if isinstance(url, str) and url:
        if url.startswith("data:"):
            header, _, payload = url.partition(",")
            if ";base64" not in header:
                raise ValueError(f"Unsupported data URL encoding: {header}")
            return _decode_base64(payload)
        if url.startswith(_HTTP_URL_SCHEMES):
            # ``urlopen`` is typed as returning ``Any`` for non-HTTP handlers, so
            # narrow the reader to the one method used.
            opened = cast(
                "AbstractContextManager[SupportsRead]",
                urlopen(url, timeout=_URL_READ_TIMEOUT_SECONDS),
            )  # noqa: S310
            with opened as response:
                return response.read()
        with open(url, "rb") as file:
            return file.read()

    raise ValueError(f"Cannot load image from record: {list(record.keys())}")


def _decode_base64(payload: str) -> bytes:
    try:
        return base64.b64decode(payload)
    except (binascii.Error, ValueError) as error:
        raise ValueError("Image payload is not valid base64") from error


def load_image(
    record: Mapping[str, object],
    config: VisionEncoderConfig,
) -> tuple[NDArray[np.float32], int, int, int, int]:
    """Load and transform one image record into ViT patches.

    Returns ``(patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w)``.

    Two details of the reference are load-bearing and preserved verbatim:

    1. The aspect clamp mutates the LOCAL ``width`` used for grid solving, while
       the geometry branch below re-reads ``image.width`` / ``image.height``,
       i.e. the UNCLAMPED size of the decoded image.
    2. When the decoded image is at least ``max_width_height_ratio`` times wider
       than it is tall, it is resized outright (distorting the aspect ratio)
       instead of being letterboxed. That is a real behavioural fork, not dead
       code, and it is what keeps an extremely wide image from being padded into
       a mostly-grey block.
    """
    patch_size = config.patch_size
    with Image.open(io.BytesIO(load_image_bytes(record))) as source:
        image = source.convert("RGB")
    width, height = image.size
    if (
        config.max_width_height_ratio is not None
        and width > height * config.max_width_height_ratio
    ):
        width = height * config.max_width_height_ratio
    if 0 < width * height < config.min_pixel_count:
        ratio: float = math.sqrt(config.min_pixel_count / (width * height))
        width = int(width * ratio)
        height = int(height * ratio)
    best_width = math.ceil(width / patch_size) * patch_size
    best_height = math.ceil(height / patch_size) * patch_size
    n_llm_h, n_llm_w, best_height, best_width = safe_resize(
        height,
        width,
        best_height,
        best_width,
        patch_size,
        config.downsample_ratio,
        config.max_token_count,
    )
    n_vit_h, n_vit_w = best_height // patch_size, best_width // patch_size
    if (
        config.max_width_height_ratio is not None
        and image.width >= config.max_width_height_ratio * image.height
    ):
        image = image.resize((best_width, best_height))
    else:
        image = ImageOps.pad(image, (best_width, best_height), color=_PAD_FILL_COLOR)

    channel_first = (
        np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / _PIXEL_VALUE_SCALE
    )
    normalized = round_to_bfloat16_precision(
        (channel_first - _NORMALIZE_SHIFT) / _NORMALIZE_SCALE
    )
    patches = (
        normalized.reshape(3, n_vit_h, patch_size, n_vit_w, patch_size)
        .transpose(1, 3, 0, 2, 4)
        .reshape(n_vit_h * n_vit_w, 3, patch_size, patch_size)
    )
    return patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w


def build_image_block(
    n_llm_h: int,
    n_llm_w: int,
    start_pos: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Build the N-layout token types (final order) and the aligner-row order.

    Returns ``(types, perm)``.

    ``order`` is the row-pair transpose the reference expresses as
    ``torch.arange(rows * row_len).view(rows // 2, 2, row_len).transpose(1, 2)
    .reshape(-1)``. torch's ``.transpose(1, 2)`` swaps the last two axes of a
    rank-3 tensor, which is numpy's ``.transpose(0, 2, 1)``; the trailing
    ``reshape(-1)`` then reads the permuted view in C order. The effect is to
    interleave each pair of adjacent grid rows column-by-column, so the two rows
    of a pair land adjacent in the token stream -- which is what the model's
    ratio-2 spatial pooling expects.

    ``pad_last`` is transcribed from ``rows // 2 * row_len % 2 * 2``, which under
    Python's equal-precedence left-to-right rule for ``//``, ``*`` and ``%``
    parses as ``((((rows // 2) * row_len) % 2) * 2)``. It is fully parenthesized
    below. Note ``rows // 2 == (n_llm_h + 1) // 2`` for every ``n_llm_h``, so this
    is the same quantity :func:`grid_tokens` adds.
    """
    compress_pad = COMPRESS_PAD_TO - 1 - start_pos % COMPRESS_PAD_TO
    pad_h = n_llm_h % 2
    rows = n_llm_h + pad_h
    row_len = n_llm_w + 1
    pad_last = (((rows // 2) * row_len) % 2) * 2

    grid_types = np.array(
        ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h
        + [IMAGE_PAD] * (row_len * pad_h),
        dtype=np.int64,
    )
    order = (
        np.arange(rows * row_len, dtype=np.int64)
        .reshape(rows // 2, 2, row_len)
        .transpose(0, 2, 1)
        .reshape(-1)
    )

    image_idx = np.full((rows, row_len), -1, dtype=np.int64)
    image_idx[:n_llm_h, :n_llm_w] = np.arange(
        n_llm_h * n_llm_w, dtype=np.int64
    ).reshape(n_llm_h, n_llm_w)
    perm = image_idx.reshape(-1)[order]
    perm = perm[perm >= 0]

    types = np.concatenate(
        [
            np.full((compress_pad,), IMAGE_PAD, dtype=np.int64),
            np.array([IMAGE_START], dtype=np.int64),
            grid_types[order],
            np.full((pad_last,), IMAGE_PAD, dtype=np.int64),
            np.array([IMAGE_END], dtype=np.int64),
        ]
    )
    return types, perm


def expand_image_placeholders(
    prompt_token_ids: Sequence[int],
    images: Iterable[Mapping[str, object]],
    image_placeholder_token_id: int,
    config: VisionEncoderConfig,
) -> tuple[list[int], list[ImageInput]]:
    """Expand image placeholder tokens into sentinel blocks and ``ImageInput``s.

    This is the placeholder-expansion half of the reference's
    ``prepare_vl_inputs``. The tokenizer half is deliberately not ported: the
    caller already holds the encoded prompt, and taking ``prompt_token_ids``
    directly keeps this module free of any tokenizer dependency.

    Returns ``(token_ids, image_inputs)``. ``image_inputs`` is empty when the
    prompt contains no placeholders.
    """
    image_records = list(images)
    placeholder_count = sum(
        1 for token_id in prompt_token_ids if token_id == image_placeholder_token_id
    )
    if placeholder_count != len(image_records):
        raise ValueError(
            f"Found {placeholder_count} image tokens but got {len(image_records)} images"
        )

    token_ids: list[int] = []
    image_inputs: list[ImageInput] = []
    image_iterator = iter(image_records)
    for token_id in prompt_token_ids:
        if token_id != image_placeholder_token_id:
            token_ids.append(token_id)
            continue
        patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = load_image(
            next(image_iterator), config
        )
        types, perm = build_image_block(n_llm_h, n_llm_w, len(token_ids))
        image_inputs.append(
            ImageInput(
                start=len(token_ids),
                patches=patches,
                n_vit_h=n_vit_h,
                n_vit_w=n_vit_w,
                types=types,
                perm=perm,
            )
        )
        token_ids.extend(
            token_type + config.vocabulary_size for token_type in as_integer_list(types)
        )
    return token_ids, image_inputs
