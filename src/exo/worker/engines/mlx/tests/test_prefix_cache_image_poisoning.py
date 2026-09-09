"""Cross-request image cache-poisoning regression tests.

THE RISK, stated precisely. DeepSeek-V4 represents an image as a block of
SENTINEL token ids (``vocab_size + type``) whose layout depends only on the
image's PATCH GEOMETRY, not its pixels. Two different images at the same
resolution therefore expand to BYTE-IDENTICAL token id sequences -- measured
on the real processor:

    448x448 random noise, seed 11 -> 279 ids, image span [3, 277), grid 32x32
    448x448 random noise, seed 22 -> 279 ids, image span [3, 277), grid 32x32
    TOKEN IDS IDENTICAL      : True
    IMAGE BYTES IDENTICAL    : False
    PATCH TENSORS IDENTICAL  : False   (max |A - B| = 2.0)

So a prefix cache keyed on token ids ALONE would serve request B (image B) out
of request A's KV state -- one user's question answered against another user's
image. ``MediaRegion.content_hash`` is what actually distinguishes them, and
these tests pin the three ways that gate can be reached.

Cases 2 and 3 are REGRESSION tests for real fail-open holes found 2026-09-09:
before the fix, each reused 278/279 of the other request's cached tokens.
"""

import base64
import hashlib
import io

import mlx.core as mx
import numpy as np
from PIL import Image


VOCABULARY_SIZE = 129280
PLACEHOLDER_TOKEN_ID = 128815
BASE_PROMPT = [1, 2, 3, PLACEHOLDER_TOKEN_ID, 4, 5]


def _synthetic_png_base64(seed: int, width: int = 448, height: int = 448) -> str:
    """A deterministic noise image. Same dimensions for every seed, by design:
    equal dimensions are exactly the condition that makes the token ids equal.
    """
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def _expand(seed: int):
    """Return (tokens, media_regions, content_hash) for one synthetic image."""
    from exo.worker.engines.mlx.vendor import deepseek_v4_image_processor as processor
    from exo.worker.engines.mlx.vision import MediaRegion

    encoder_config = processor.VisionEncoderConfig(
        patch_size=14,
        downsample_ratio=2,
        max_token_count=2048,
        min_pixel_count=14 * 14 * 4,
        max_width_height_ratio=9,
        vocabulary_size=VOCABULARY_SIZE,
    )
    image_base64 = _synthetic_png_base64(seed)
    token_ids, image_inputs = processor.expand_image_placeholders(
        BASE_PROMPT,
        [{"data": image_base64}],
        PLACEHOLDER_TOKEN_ID,
        encoder_config,
    )
    content_hash = hashlib.sha256(base64.b64decode(image_base64)).hexdigest()
    image = image_inputs[0]
    region = MediaRegion(
        content_hash=content_hash,
        start_pos=image.start,
        end_pos=image.start + len(image.types),
    )
    tokens = mx.array(np.asarray(token_ids, dtype=np.int32))
    return tokens, [region], content_hash


def _fake_cache(num_layers: int, num_tokens: int, fill: float):
    from mlx_lm.models.cache import KVCache

    caches = []
    for _ in range(num_layers):
        entry = KVCache()
        keys = mx.full((1, 2, num_tokens, 4), fill, dtype=mx.float32)
        values = mx.full((1, 2, num_tokens, 4), fill, dtype=mx.float32)
        entry.update_and_fetch(keys, values)
        caches.append(entry)
    return caches


def _fake_model():
    from unittest.mock import MagicMock

    model = MagicMock()
    model.layers = [None]
    return model


def _serve(cached_regions, query_regions):
    """Cache image A under `cached_regions`, then query with image B's tokens
    under `query_regions`. Returns (matched_leaf_id, tokens_reused, total).
    """
    from exo.worker.engines.mlx.cache import KVPrefixCache

    tokens_a, _, _ = _expand(11)
    tokens_b, _, _ = _expand(22)

    cache = KVPrefixCache(None)
    cache.add_kv_cache(
        tokens_a,
        _fake_cache(1, int(tokens_a.shape[0]), 7.0),
        media_regions=cached_regions,
    )
    _, remaining, matched, _ = cache.get_kv_cache(
        _fake_model(), tokens_b, media_regions=query_regions
    )
    total = int(tokens_b.shape[0])
    return matched, total - int(remaining.shape[0]), total


def test_two_different_images_at_the_same_size_produce_identical_token_ids():
    """The PRECONDITION for the whole risk. If this ever stops holding, the
    cache-poisoning concern is moot -- and these tests would be vacuous, so
    assert it explicitly rather than assuming it.
    """
    tokens_a, regions_a, hash_a = _expand(11)
    tokens_b, regions_b, hash_b = _expand(22)

    assert np.array_equal(np.asarray(tokens_a), np.asarray(tokens_b)), (
        "two 448x448 images no longer expand to identical token ids"
    )
    assert hash_a != hash_b
    assert (regions_a[0].start_pos, regions_a[0].end_pos) == (
        regions_b[0].start_pos,
        regions_b[0].end_pos,
    )


def test_different_images_with_correct_hashes_are_served_cold():
    """Case 1 -- the designed path. Both sides carry real per-image hashes."""
    _, regions_a, _ = _expand(11)
    _, regions_b, _ = _expand(22)

    matched, reused, total = _serve(regions_a, regions_b)

    assert matched is None, "image B must not be served out of image A's leaf"
    assert reused == 0, f"reused {reused}/{total} of another image's cached KV"


def test_query_without_media_regions_cannot_reuse_cached_image_kv():
    """Case 2 -- REGRESSION. A caller that omits `media_regions` on the query
    used to sail straight past the hash gate (`query_r is None: continue`) and
    reuse 278/279 tokens of a different image's KV. Absence of a query region
    is absence of evidence, not evidence of a match.

    What is asserted is the exact, honest property: NOT ONE token of the
    cached IMAGE SPAN may be reused. The 3 text tokens that precede the span
    are still shared, and that is correct -- they are ordinary text whose KV
    genuinely does not depend on either image.
    """
    _, regions_a, _ = _expand(11)
    image_start = regions_a[0].start_pos

    matched, reused, total = _serve(regions_a, [])

    assert reused <= image_start, (
        f"reused {reused}/{total} tokens, which reaches into the cached image "
        f"span starting at {image_start} -- image KV was reused for a request "
        "that never identified an image"
    )
    if matched is not None:
        # A leaf may still be matched for the shared text prefix; what must
        # never happen is that the restore position enters the image span.
        assert reused == image_start


def test_unresolvable_hashes_do_not_alias_to_each_other():
    """Case 3 -- REGRESSION. `MediaRegion(content_hash="")` used to be the
    fallback whenever an image could not be hashed. Two such regions compared
    EQUAL, so every unhashable image aliased to every other one. The
    replacement placeholder must be unique per call.
    """
    from exo.worker.engines.mlx.vision import (
        MediaRegion,
        unresolvable_content_hash,
    )

    first = unresolvable_content_hash("pending")
    second = unresolvable_content_hash("pending")
    assert first != second, "unresolvable hashes must never compare equal"
    assert first != "" and second != ""

    _, regions_a, _ = _expand(11)
    _, regions_b, _ = _expand(22)
    placeholder_a = [
        MediaRegion(
            content_hash=unresolvable_content_hash("pending"),
            start_pos=regions_a[0].start_pos,
            end_pos=regions_a[0].end_pos,
        )
    ]
    placeholder_b = [
        MediaRegion(
            content_hash=unresolvable_content_hash("pending"),
            start_pos=regions_b[0].start_pos,
            end_pos=regions_b[0].end_pos,
        )
    ]

    matched, reused, total = _serve(placeholder_a, placeholder_b)

    assert matched is None
    assert reused == 0, f"reused {reused}/{total} tokens across unhashable images"


def test_control_identical_image_still_hits_the_cache():
    """CONTROL. Without this, an unconditional 'always serve cold' regression
    would pass every test above while destroying the prefix cache's value.
    """
    from exo.worker.engines.mlx.cache import KVPrefixCache

    tokens, regions, _ = _expand(11)
    cache = KVPrefixCache(None)
    cache.add_kv_cache(
        tokens, _fake_cache(1, int(tokens.shape[0]), 7.0), media_regions=regions
    )
    _, remaining, matched, is_exact = cache.get_kv_cache(
        _fake_model(), tokens, media_regions=regions
    )

    assert matched is not None, "the same image must still hit the prefix cache"
    assert is_exact
    assert int(remaining.shape[0]) == 1


def test_control_text_only_requests_still_share_prefixes():
    """CONTROL. The fail-closed rule keys on CACHED media regions, so text-only
    traffic -- every request on the cluster today -- must be untouched.
    """
    from exo.worker.engines.mlx.cache import KVPrefixCache

    shared = list(range(100, 400))
    tokens_a = mx.array(np.asarray(shared + [900, 901], dtype=np.int32))
    tokens_b = mx.array(np.asarray(shared + [902, 903], dtype=np.int32))

    cache = KVPrefixCache(None)
    cache.add_kv_cache(tokens_a, _fake_cache(1, int(tokens_a.shape[0]), 3.0))
    _, remaining, matched, _ = cache.get_kv_cache(_fake_model(), tokens_b)

    reused = int(tokens_b.shape[0]) - int(remaining.shape[0])
    assert matched is not None, "text-only prefix sharing must be unaffected"
    assert reused == len(shared), f"expected {len(shared)} shared tokens, got {reused}"
