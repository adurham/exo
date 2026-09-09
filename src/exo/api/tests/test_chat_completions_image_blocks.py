"""Regression tests for the API adapter's multimodal image-block emission.

BACKGROUND. `chat_request_to_text_generation` deliberately does NOT inline
image bytes into `chat_template_messages`: that structure is serialized and
shipped to every worker, so inlining would duplicate a multi-MB base64
payload on the wire alongside the copy already carried out of band in
`TextGenerationTaskParams.images`.

Until 2026-09-09 the adapter emitted a bare `{"type": "image"}` marker for
each image part. That block is indistinguishable from "an image was dropped
upstream", and the vendored DeepSeek-V4 encoder rejects it:

    _extract_image -> ValueError("Image block does not contain a valid source")

Both DSv4 generators wrap vision processing in
`except Exception: logger.warning("Vision processing failed, falling back to
text-only")`, so that ValueError never reached the client. EVERY image
request silently degraded to a text-only completion returning HTTP 200 --
the model would answer "there is no image attached" while usage.prompt_tokens
showed no image expansion at all.

That was found by the Phase 5 on-hardware smoke test
(docs/DSV4_VISION_PORT_PHASE5_PROCEDURE.md), the first thing to exercise the
HTTP -> adapter -> encoder seam end to end; the Phase 4 offline test called
the expansion helpers directly with `{"data": ...}` records and never went
through the adapter.

The fix emits an ordered reference -- `{"type": "image", "url":
"exo-image:<n>"}` -- which is a real non-empty source, so the encoder's
"an image block must declare where its content comes from" invariant is
satisfied WITHOUT weakening it for genuinely malformed blocks.

These tests pin that contract from both ends.
"""

from __future__ import annotations

import base64
from typing import Any, cast

from exo.api.adapters.chat_completions import chat_request_to_text_generation
from exo.api.types.api import (
    ChatCompletionMessage,
    ChatCompletionMessageImageUrl,
    ChatCompletionMessageText,
    ChatCompletionRequest,
)
from exo.shared.models.model_cards import ModelId

# A 1x1 PNG. Content is irrelevant here -- these tests are about block SHAPE,
# not pixels; decoding correctness is covered by the vendored processor tests.
_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
_DATA_URL = "data:image/png;base64," + _PNG_B64


def _image_request(n_images: int = 1) -> ChatCompletionRequest:
    parts: list[Any] = [ChatCompletionMessageText(type="text", text="Describe this.")]
    for _ in range(n_images):
        parts.append(
            ChatCompletionMessageImageUrl(
                type="image_url", image_url={"url": _DATA_URL}
            )
        )
    return ChatCompletionRequest(
        model=ModelId("test/model"),
        messages=[ChatCompletionMessage(role="user", content=parts)],
    )


async def test_image_bytes_travel_out_of_band_not_inline() -> None:
    """The bytes belong in `.images`; `chat_template_messages` must stay small.

    If this ever regresses to inlining, every worker pays the base64 payload
    twice on the zenoh wire.
    """
    params = await chat_request_to_text_generation(_image_request())

    assert len(params.images) == 1
    assert base64.b64decode(str(params.images[0]))[:8] == b"\x89PNG\r\n\x1a\n"

    messages = cast(list[dict[str, Any]], params.chat_template_messages)
    serialized = repr(messages)
    assert _PNG_B64 not in serialized, "image bytes must not be inlined"


async def test_image_block_declares_a_non_empty_source() -> None:
    """The regression guard proper.

    A bare `{"type": "image"}` is what broke live vision. The block must
    carry a non-empty source value so the vendored encoder accepts it.
    """
    params = await chat_request_to_text_generation(_image_request())
    messages = cast(list[dict[str, Any]], params.chat_template_messages)
    blocks = cast(list[dict[str, Any]], messages[0]["content"])

    image_blocks = [b for b in blocks if b.get("type") == "image"]
    assert len(image_blocks) == 1
    block = image_blocks[0]

    assert any(block.get(key) for key in ("source", "url", "data")), (
        "image block must declare a non-empty source -- a bare "
        '{"type": "image"} marker makes the DSv4 encoder raise, which the '
        "generators swallow into a silent text-only fallback"
    )
    assert block["url"] == "exo-image:0"


async def test_image_blocks_are_indexed_in_request_order() -> None:
    """Ordering is the contract binding block N to `images[N]`."""
    params = await chat_request_to_text_generation(_image_request(n_images=3))
    messages = cast(list[dict[str, Any]], params.chat_template_messages)
    blocks = cast(list[dict[str, Any]], messages[0]["content"])

    urls = [b["url"] for b in blocks if b.get("type") == "image"]
    assert urls == ["exo-image:0", "exo-image:1", "exo-image:2"]
    assert len(params.images) == 3


async def test_text_parts_are_preserved_alongside_images() -> None:
    """The text half of a multimodal message must survive untouched."""
    params = await chat_request_to_text_generation(_image_request())
    messages = cast(list[dict[str, Any]], params.chat_template_messages)
    blocks = cast(list[dict[str, Any]], messages[0]["content"])

    assert blocks[0] == {"type": "text", "text": "Describe this."}


async def test_adapter_blocks_survive_the_vendored_encoder() -> None:
    """End-to-end seam check: what the adapter emits, the encoder accepts.

    This is the exact hop that failed in production -- worth pinning both
    sides together rather than trusting each in isolation.
    """
    from exo.worker.engines.mlx.vendor.deepseek_v4_encoding import (
        IMAGE_PLACEHOLDER,
        process_image_messages,
    )

    params = await chat_request_to_text_generation(_image_request(n_images=2))
    messages = cast(list[dict[str, Any]], params.chat_template_messages)

    processed, images = process_image_messages([dict(m) for m in messages])

    assert len(images) == 2
    blocks = processed[0].get("content_blocks") or processed[0].get("content")
    texts = [b.get("text") for b in cast(list[dict[str, Any]], blocks)]
    assert texts.count(IMAGE_PLACEHOLDER) == 2, (
        "one placeholder per image -- deepseek_v4_vision.process asserts "
        "placeholder_count == len(images) before expanding"
    )
