"""Prefill chunk scheduling that never splits a DeepSeek-V4 image span.

DeepSeek-V4's vision path carries a hard invariant, inherited from the
reference implementation's ``Transformer.forward``::

    if images is not None:
        if start_pos == 0:
            self.merge_image_embeddings(images, h)
            visible = get_image_visible(input_ids, self.vocab_size, self.max_image_tokens)
        else:
            assert (input_ids < self.vocab_size).all(), \\
                "image spans must be prefilled in a single chunk"

Read literally, the assert fires whenever a forward pass at ``start_pos > 0``
sees ANY image token. So the invariant is **not** merely "a span must not be
split across a chunk boundary" -- it is:

    every image token must be processed by the forward pass whose cache
    offset is ZERO, i.e. the first chunk, with no prefilled KV ahead of it.

That is stricter, and it is a real correctness constraint rather than a
reference quirk: the local KV is a ``RotatingKVCache`` capped at
``sliding_window`` (128), while image-span visibility reaches up to
``vision_max_n_token`` (384) positions back. A span is only retrievable while
the whole span is inside the current chunk; otherwise the ring slots it needs
have already been overwritten. mlx-lm's ``_apply_image_visibility`` enforces
the same thing from the model side and raises loudly when it is violated.

Two ways a chunker can violate it, and only the first is what "boundary guard"
suggests:

1. A uniform chunk boundary lands strictly inside a span.
2. An image sits in a LATER chunk. No boundary split it -- the span is intact
   -- but the chunk still runs at a non-zero cache offset. On this cluster
   ``EXO_PREFILL_STEP_SIZE=2048`` becomes 1024 after the world_size=2 divisor,
   so a 2000-token system prompt followed by an image is enough. This is the
   COMMON case, not an edge case.

Case 2 also covers a prefix-cache hit: restoring N cached tokens and prefilling
the remainder starts the first real forward at offset N.

`plan_prefill_chunks` handles both by extending the FIRST chunk to cover the
last image token. Every rank computes it independently from
``(token count, span list, step size)`` -- all of which are already required to
be identical across ranks for pipeline-parallel prefill to work at all -- so no
communication is needed and ranks cannot disagree about the schedule.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

#: Cap on how far the first chunk may be stretched to swallow an image span.
#: Beyond this the request is rejected rather than silently attempting a
#: single forward pass over an unbounded number of tokens.
#:
#: This cap is NOT a mitigation for a chunking bug -- it is the honest edge of
#: an architectural limit. Images can only be merged at cache offset 0, so the
#: entire prefix ahead of an image must go through one forward pass; there is
#: no valid way to sub-chunk it. A request that exceeds the cap is genuinely
#: outside the model's serving envelope.
DEFAULT_MAX_IMAGE_PREFILL_TOKENS: Final = 16384

_MAX_IMAGE_PREFILL_TOKENS_ENV: Final = "EXO_DSV4_MAX_IMAGE_PREFILL_TOKENS"


def max_image_prefill_tokens() -> int:
    raw = os.environ.get(_MAX_IMAGE_PREFILL_TOKENS_ENV)
    if not raw:
        return DEFAULT_MAX_IMAGE_PREFILL_TOKENS
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{_MAX_IMAGE_PREFILL_TOKENS_ENV} must be an integer, got {raw!r}"
        ) from exc
    if value <= 0:
        raise ValueError(
            f"{_MAX_IMAGE_PREFILL_TOKENS_ENV} must be positive, got {value}"
        )
    return value


class ImageSpanPrefillError(RuntimeError):
    """An image span cannot be prefilled at cache offset 0."""


@dataclass(frozen=True, slots=True)
class ImageSpan:
    """Half-open ``[start, end)`` token span covering one image block."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"invalid image span [{self.start}, {self.end})")


def plan_prefill_chunks(
    total_tokens: int,
    prefill_step_size: int,
    image_spans: list[ImageSpan] | None = None,
    *,
    cache_offset: int = 0,
    max_first_chunk: int | None = None,
) -> list[int]:
    """Chunk sizes summing to ``total_tokens``, image spans kept in chunk 0.

    ``total_tokens`` is the number of tokens actually being prefilled (exo
    prefills ``len(prompt) - 1``, holding the last token back for the first
    decode step). ``image_spans`` are positions within THAT stream.

    With no image spans this is byte-for-byte the existing uniform greedy
    schedule, so the text path is unchanged.

    Raises `ImageSpanPrefillError` when the invariant cannot be satisfied:
    a non-zero ``cache_offset`` with images still to prefill, or a first chunk
    that would have to exceed ``max_first_chunk``.
    """
    if prefill_step_size <= 0:
        raise ValueError(f"prefill_step_size must be positive, got {prefill_step_size}")
    if total_tokens <= 0:
        return []

    spans = sorted(image_spans or [], key=lambda s: s.start)

    if not spans:
        return _uniform_chunks(total_tokens, prefill_step_size)

    last_image_end = max(span.end for span in spans)
    if last_image_end > total_tokens:
        raise ImageSpanPrefillError(
            f"image span ends at {last_image_end} but only {total_tokens} "
            "tokens are being prefilled; the token stream and the span list "
            "were built from different prompts"
        )

    if cache_offset != 0:
        # A prefix-cache hit (or any already-populated KV) put us past offset
        # 0 while image tokens remain unprefilled. No chunk schedule can fix
        # this -- the caller must prefill this request from scratch.
        raise ImageSpanPrefillError(
            f"image tokens must be prefilled at cache offset 0, but the cache "
            f"already holds {cache_offset} tokens and an image span ends at "
            f"{last_image_end}. Prefill this request without the prefix cache."
        )

    limit = (
        max_first_chunk if max_first_chunk is not None else max_image_prefill_tokens()
    )
    first_chunk = max(prefill_step_size, last_image_end)
    first_chunk = min(first_chunk, total_tokens)
    if first_chunk > limit:
        raise ImageSpanPrefillError(
            f"the last image span ends at token {last_image_end}, so the first "
            f"prefill chunk would need {first_chunk} tokens, over the "
            f"{limit}-token limit ({_MAX_IMAGE_PREFILL_TOKENS_ENV}). DeepSeek-V4 "
            "can only merge image embeddings at cache offset 0, so every token "
            "before an image must be prefilled in one pass -- there is no valid "
            "way to split it. Move the image earlier in the prompt or raise the "
            "limit if the machine has the memory."
        )

    chunks = [first_chunk]
    chunks.extend(_uniform_chunks(total_tokens - first_chunk, prefill_step_size))

    _assert_schedule_valid(chunks, total_tokens, spans)
    return chunks


def _uniform_chunks(total: int, step: int) -> list[int]:
    """The pre-existing greedy schedule, unchanged."""
    chunks: list[int] = []
    remaining = total
    while remaining > 0:
        take = min(step, remaining)
        chunks.append(take)
        remaining -= take
    return chunks


def _assert_schedule_valid(
    chunks: list[int], total_tokens: int, spans: list[ImageSpan]
) -> None:
    """Postcondition check, run on every rank.

    Cheap, and it converts an off-by-one in the planner into a loud local
    failure instead of a pipeline-parallel hang: if ranks ever disagreed about
    chunk sizes, the ranks would block waiting on differently-shaped sends.
    """
    if sum(chunks) != total_tokens:
        raise ImageSpanPrefillError(
            f"chunk schedule sums to {sum(chunks)}, expected {total_tokens}"
        )
    if any(size <= 0 for size in chunks):
        raise ImageSpanPrefillError(f"chunk schedule has an empty chunk: {chunks}")

    boundaries: list[int] = []
    running = 0
    for size in chunks[:-1]:
        running += size
        boundaries.append(running)

    for span in spans:
        if span.end > chunks[0]:
            raise ImageSpanPrefillError(
                f"image span [{span.start}, {span.end}) is not contained in the "
                f"first chunk of {chunks[0]} tokens"
            )
        for boundary in boundaries:
            if span.start < boundary < span.end:
                raise ImageSpanPrefillError(
                    f"chunk boundary at {boundary} splits image span "
                    f"[{span.start}, {span.end})"
                )


def image_spans_from_media_regions(
    regions: "Iterable[object] | None",
) -> list[ImageSpan]:
    """Adapt `vision.MediaRegion`s to `ImageSpan`s.

    Kept structural rather than typed against `MediaRegion` so this module has
    no import dependency on the vision layer (which pulls in mlx-vlm).
    """
    spans: list[ImageSpan] = []
    for region in regions or ():
        start = getattr(region, "start_pos", None)
        end = getattr(region, "end_pos", None)
        if isinstance(start, int) and isinstance(end, int) and end > start:
            spans.append(ImageSpan(start=start, end=end))
    return spans
