# pyright: reportMissingImports=false, reportUnknownMemberType=false
# pyright: reportUntypedFunctionDecorator=false
"""Phase 4d: the prefill chunk planner must never orphan an image span.

The invariant under test is stricter than "don't split a span": DeepSeek-V4
merges image embeddings ONLY at cache offset 0, so every image token has to be
in the first chunk. See `prefill_chunking`'s module docstring.
"""

import pytest

from exo.worker.engines.mlx.prefill_chunking import (
    DEFAULT_MAX_IMAGE_PREFILL_TOKENS,
    ImageSpan,
    ImageSpanPrefillError,
    image_spans_from_media_regions,
    plan_prefill_chunks,
)


def _uniform(total: int, step: int) -> list[int]:
    """The ORIGINAL greedy schedule, transcribed from the pre-4d code."""
    chunks: list[int] = []
    remaining = total
    while remaining:
        n = min(step, remaining)
        chunks.append(n)
        remaining -= n
    return chunks


class TestTextPathUnchanged:
    """With no image spans the schedule must be byte-identical to before."""

    @pytest.mark.parametrize(
        "total", [1, 2, 127, 128, 129, 1023, 1024, 1025, 5000, 100_000]
    )
    @pytest.mark.parametrize("step", [1, 128, 256, 1024, 2048, 4096])
    def test_matches_original_greedy_schedule(self, total: int, step: int) -> None:
        assert plan_prefill_chunks(total, step) == _uniform(total, step)
        assert plan_prefill_chunks(total, step, []) == _uniform(total, step)

    def test_zero_tokens_is_empty(self) -> None:
        assert plan_prefill_chunks(0, 1024) == []

    def test_rejects_nonpositive_step(self) -> None:
        with pytest.raises(ValueError):
            plan_prefill_chunks(100, 0)


class TestImageSpansLandInFirstChunk:
    def test_span_inside_first_chunk_leaves_schedule_alone(self) -> None:
        # Span [50, 434) already fits in a 1024 chunk; nothing to stretch.
        chunks = plan_prefill_chunks(4000, 1024, [ImageSpan(50, 434)])
        assert chunks == _uniform(4000, 1024)
        assert chunks[0] == 1024

    def test_span_straddling_a_boundary_is_pulled_in(self) -> None:
        # Uniform would cut at 1024, splitting [900, 1284).
        span = ImageSpan(900, 1284)
        chunks = plan_prefill_chunks(4000, 1024, [span])
        assert chunks[0] == 1284
        assert sum(chunks) == 4000
        assert span.end <= chunks[0]

    def test_span_entirely_in_a_later_chunk_is_still_pulled_in(self) -> None:
        """The case a naive 'don't split a span' guard misses entirely.

        [2100, 2484) is not split by any 1024 boundary -- it sits wholly inside
        chunk 3 -- but that chunk runs at cache offset 2048, where DSv4 cannot
        merge image embeddings.
        """
        span = ImageSpan(2100, 2484)
        naive = _uniform(4000, 1024)
        boundaries = {sum(naive[: i + 1]) for i in range(len(naive) - 1)}
        assert not any(span.start < b < span.end for b in boundaries), (
            "precondition: no boundary splits this span"
        )

        chunks = plan_prefill_chunks(4000, 1024, [span])
        assert chunks[0] == 2484
        assert sum(chunks) == 4000

    def test_multiple_images_all_land_in_first_chunk(self) -> None:
        spans = [ImageSpan(100, 484), ImageSpan(1500, 1884), ImageSpan(3000, 3384)]
        chunks = plan_prefill_chunks(6000, 1024, spans)
        assert chunks[0] == 3384
        assert all(s.end <= chunks[0] for s in spans)
        assert sum(chunks) == 6000

    def test_first_chunk_never_shrinks_below_step_size(self) -> None:
        chunks = plan_prefill_chunks(4000, 1024, [ImageSpan(0, 384)])
        assert chunks[0] == 1024

    def test_image_at_very_end_of_prompt(self) -> None:
        chunks = plan_prefill_chunks(2000, 1024, [ImageSpan(1616, 2000)])
        assert chunks == [2000]

    def test_schedule_has_no_empty_chunks(self) -> None:
        for total in range(1, 60):
            for step in (1, 2, 7, 16):
                spans = [ImageSpan(0, min(4, total))] if total >= 4 else []
                chunks = plan_prefill_chunks(total, step, spans, max_first_chunk=10_000)
                assert all(c > 0 for c in chunks), (total, step, chunks)
                assert sum(chunks) == total


class TestRejectedCases:
    def test_nonzero_cache_offset_with_images_raises(self) -> None:
        """A prefix-cache hit ahead of an image cannot be chunked around."""
        with pytest.raises(ImageSpanPrefillError, match="cache offset 0"):
            plan_prefill_chunks(4000, 1024, [ImageSpan(100, 484)], cache_offset=2048)

    def test_nonzero_cache_offset_without_images_is_fine(self) -> None:
        assert plan_prefill_chunks(4000, 1024, [], cache_offset=2048) == _uniform(
            4000, 1024
        )

    def test_image_beyond_the_cap_raises_with_a_clear_message(self) -> None:
        with pytest.raises(ImageSpanPrefillError, match="over the"):
            plan_prefill_chunks(200_000, 1024, [ImageSpan(150_000, 150_384)])

    def test_cap_is_configurable(self) -> None:
        chunks = plan_prefill_chunks(
            200_000, 1024, [ImageSpan(150_000, 150_384)], max_first_chunk=200_000
        )
        assert chunks[0] == 150_384

    def test_default_cap_value(self) -> None:
        assert DEFAULT_MAX_IMAGE_PREFILL_TOKENS == 16384

    def test_span_past_the_prompt_raises(self) -> None:
        with pytest.raises(ImageSpanPrefillError, match="different prompts"):
            plan_prefill_chunks(500, 1024, [ImageSpan(400, 900)])

    def test_invalid_span_rejected(self) -> None:
        with pytest.raises(ValueError):
            ImageSpan(10, 10)
        with pytest.raises(ValueError):
            ImageSpan(-1, 5)


class TestCrossRankDeterminism:
    def test_same_inputs_give_same_schedule(self) -> None:
        """Ranks compute this independently; identical inputs must agree.

        A disagreement would desync pipeline-parallel prefill (each rank builds
        its own `real_chunk_sizes`), so this is a hard requirement, not a nicety.
        """
        spans = [ImageSpan(100, 484), ImageSpan(1500, 1884)]
        reference = plan_prefill_chunks(6000, 1024, spans)
        for _ in range(50):
            assert plan_prefill_chunks(6000, 1024, list(spans)) == reference

    def test_span_order_does_not_matter(self) -> None:
        a = plan_prefill_chunks(
            6000, 1024, [ImageSpan(100, 484), ImageSpan(1500, 1884)]
        )
        b = plan_prefill_chunks(
            6000, 1024, [ImageSpan(1500, 1884), ImageSpan(100, 484)]
        )
        assert a == b


class TestMediaRegionAdapter:
    def test_converts_media_regions(self) -> None:
        from exo.worker.engines.mlx.vision import MediaRegion

        regions = [
            MediaRegion(content_hash="a", start_pos=10, end_pos=394),
            MediaRegion(content_hash="b", start_pos=500, end_pos=884),
        ]
        assert image_spans_from_media_regions(regions) == [
            ImageSpan(10, 394),
            ImageSpan(500, 884),
        ]

    def test_none_and_empty(self) -> None:
        assert image_spans_from_media_regions(None) == []
        assert image_spans_from_media_regions([]) == []
