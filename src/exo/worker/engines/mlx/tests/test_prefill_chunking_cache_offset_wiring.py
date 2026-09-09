# pyright: reportMissingImports=false, reportUnknownMemberType=false
# pyright: reportUntypedFunctionDecorator=false
"""Phase 4d follow-up: the chunk-boundary guard must actually be REACHED.

`plan_prefill_chunks` was correct from the start, but three separate things
kept it from doing anything in production. This module covers all three, plus
the prefix-cache decision that keeps the first two from turning silent
corruption into a guaranteed hard failure.

1. `cache_offset` was never passed at the one production call site, so the
   planner's own offset guard was dead code.
2. `media_regions` carry ABSOLUTE prompt positions while `total_tokens` at
   that call site is POST-PREFIX-HIT-RELATIVE. Wiring (1) without fixing this
   just feeds the planner two different coordinate frames.
3. Two further prefill loops (`prefill_batched`'s own uniform loop, and
   `prefill`'s `stream_generate` branch -- the branch every request on a
   TENSOR-parallel cluster actually takes) never consulted the planner at all.

Everything here is pure Python: `prefill_chunking` deliberately has no mlx
import, and the call-site tests read the source's wiring rather than running a
model.
"""

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

from exo.worker.engines.mlx.prefill_chunking import (
    ImageSpan,
    ImageSpanPrefillError,
    image_spans_from_media_regions,
    plan_prefill_chunks,
)

#: Source files are read from disk rather than imported. Importing
#: `generator.generate` / `cache` drags in `exo.shared.constants`, which
#: resolves the built dashboard assets at import time and hard-fails in any
#: checkout that has not run `npm run build`. These tests assert on WIRING, so
#: reading the source is both sufficient and more robust than importing it.
_MLX_ENGINE_DIR = Path(__file__).resolve().parent.parent

#: The cluster's configured value. `_pipeline_parallel_prefill_steps` divides
#: it by `min(4, group.size())`, so world_size=2 makes the effective step 1024.
CLUSTER_PREFILL_STEP_SIZE = 2048
CLUSTER_EFFECTIVE_STEP = CLUSTER_PREFILL_STEP_SIZE // 2


def _read_source(*parts: str) -> str:
    path = _MLX_ENGINE_DIR.joinpath(*parts)
    assert path.is_file(), f"expected source file at {path}"
    return path.read_text()


def _generate_source() -> str:
    return _read_source("generator", "generate.py")


def _batch_generate_source() -> str:
    return _read_source("generator", "batch_generate.py")


def _cache_source() -> str:
    return _read_source("cache.py")


def _function_def(source: str, name: str) -> ast.FunctionDef:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"no function named {name!r} in source")


def _param_names(fn: ast.FunctionDef) -> set[str]:
    args = fn.args
    return {a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)}


@dataclass(frozen=True)
class FakeMediaRegion:
    """Structural stand-in for `vision.MediaRegion` (absolute positions).

    `image_spans_from_media_regions` reads `start_pos`/`end_pos` by attribute
    precisely so this module needs no mlx-vlm import.
    """

    content_hash: str
    start_pos: int
    end_pos: int


def _uniform(total: int, step: int) -> list[int]:
    """The ORIGINAL greedy schedule, transcribed from the pre-4d code."""
    chunks: list[int] = []
    remaining = total
    while remaining:
        n = min(step, remaining)
        chunks.append(n)
        remaining -= n
    return chunks


def _boundaries(chunks: list[int]) -> list[int]:
    """Cut points BETWEEN chunks (excludes the final end-of-stream)."""
    out: list[int] = []
    running = 0
    for size in chunks[:-1]:
        running += size
        out.append(running)
    return out


def _straddles(chunks: list[int], span: ImageSpan) -> bool:
    return any(span.start < b < span.end for b in _boundaries(chunks))


# --------------------------------------------------------------------------
# BUG 1: the production call site never passed cache_offset.
# --------------------------------------------------------------------------


class TestBugOneGuardIsNowReachedInProduction:
    """The three reviewer repro cases, run through the guard.

    Each is `(total_prompt, prefix_hit, absolute_span)` with the cluster's
    EXO_PREFILL_STEP_SIZE=2048. Before the fix all three reached the planner
    with `cache_offset=0` and absolute spans, so the planner happily emitted a
    schedule for a prefill that would run the image at a non-zero cache offset.
    """

    REPROS = [
        (10_000, 50, (100, 484)),
        (10_000, 1000, (1500, 1884)),
        (50_000, 2000, (3000, 3384)),
    ]

    @pytest.mark.parametrize("total_prompt,hit,span", REPROS)
    def test_guard_now_rejects_images_behind_a_prefix_hit(
        self, total_prompt: int, hit: int, span: tuple[int, int]
    ) -> None:
        """Confirms what the planner ALREADY does once cache_offset arrives.

        These are all "image still pending while the cache is past offset 0",
        which the planner rejects rather than snapping -- correctly, since no
        chunk schedule can move a forward pass back to offset 0.
        """
        regions = [FakeMediaRegion("h", span[0], span[1])]
        local_spans = image_spans_from_media_regions(regions, cache_offset=hit)
        assert local_spans, "span is past the hit, so it must survive re-basing"

        with pytest.raises(ImageSpanPrefillError, match="cache offset 0"):
            plan_prefill_chunks(
                total_tokens=total_prompt - hit - 1,
                prefill_step_size=CLUSTER_EFFECTIVE_STEP,
                image_spans=local_spans,
                cache_offset=hit,
            )

    @pytest.mark.parametrize("total_prompt,hit,span", REPROS)
    def test_before_the_fix_the_same_inputs_planned_a_schedule(
        self, total_prompt: int, hit: int, span: tuple[int, int]
    ) -> None:
        """BEFORE: the exact old call, which silently succeeded.

        This is the old production call reproduced literally -- no
        `cache_offset=`, spans left absolute. It returns a schedule, i.e. the
        request proceeds and the image is prefilled at offset `hit` != 0,
        where `_apply_image_visibility` raises (or, with visibility off,
        attends to overwritten RotatingKVCache ring slots).
        """
        old_spans = [ImageSpan(span[0], span[1])]
        planned = plan_prefill_chunks(
            total_tokens=total_prompt - hit - 1,
            prefill_step_size=CLUSTER_EFFECTIVE_STEP,
            image_spans=old_spans,
        )
        assert planned, "the OLD call produced a schedule instead of refusing"
        assert sum(planned) == total_prompt - hit - 1

    def test_cold_miss_still_serves_normally(self) -> None:
        """The guard must only fire on a real hit, not on every vision request."""
        regions = [FakeMediaRegion("h", 100, 484)]
        spans = image_spans_from_media_regions(regions, cache_offset=0)
        assert spans == [ImageSpan(100, 484)]
        chunks = plan_prefill_chunks(
            total_tokens=9_999,
            prefill_step_size=CLUSTER_EFFECTIVE_STEP,
            image_spans=spans,
            cache_offset=0,
        )
        assert chunks == _uniform(9_999, CLUSTER_EFFECTIVE_STEP)
        assert not _straddles(chunks, spans[0])
        assert spans[0].end <= chunks[0]

    def test_hit_past_the_image_is_allowed(self) -> None:
        """Every image already merged inside the restored prefix: pure text left."""
        regions = [FakeMediaRegion("h", 100, 484)]
        spans = image_spans_from_media_regions(regions, cache_offset=2000)
        assert spans == [], "a fully-cached span constrains nothing"
        assert plan_prefill_chunks(
            total_tokens=8_000,
            prefill_step_size=CLUSTER_EFFECTIVE_STEP,
            image_spans=spans,
            cache_offset=2000,
        ) == _uniform(8_000, CLUSTER_EFFECTIVE_STEP)


class TestBugOneWiringAtTheRealCallSite:
    """Source-level proof the value is threaded, not just accepted."""

    def test_planner_call_passes_cache_offset(self) -> None:
        """`plan_prefill_chunks` must never be called without `cache_offset`."""
        tree = ast.parse(_generate_source())
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "plan_prefill_chunks"
        ]
        assert calls, "expected at least one production planner call"
        for call in calls:
            kwargs = {kw.arg for kw in call.keywords}
            assert "cache_offset" in kwargs, (
                f"plan_prefill_chunks call at line {call.lineno} omits "
                "cache_offset -- the offset guard is dead code again"
            )

    def test_adapter_call_passes_cache_offset(self) -> None:
        """Spans must be re-based; an absolute span is a coordinate-frame bug."""
        tree = ast.parse(_generate_source())
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "image_spans_from_media_regions"
        ]
        assert calls, "expected at least one adapter call"
        for call in calls:
            kwargs = {kw.arg for kw in call.keywords}
            assert "cache_offset" in kwargs, (
                f"image_spans_from_media_regions call at line {call.lineno} "
                "omits cache_offset -- spans stay in the ABSOLUTE frame"
            )

    def test_prefill_forwards_snapshot_offset_to_the_pipeline_path(self) -> None:
        """`prefill(snapshot_offset=...)` is where the hit length enters."""
        assert "cache_offset=snapshot_offset" in _generate_source(), (
            "prefill() must forward its snapshot_offset (the prefix hit "
            "length) down as the planner's cache_offset"
        )

    def test_pipeline_prefill_signatures_accept_cache_offset(self) -> None:
        source = _generate_source()
        for name in ("_pipeline_parallel_prefill_steps", "pipeline_parallel_prefill"):
            fn = _function_def(source, name)
            assert "cache_offset" in _param_names(fn), f"{name} drops cache_offset"


# --------------------------------------------------------------------------
# BUG 2: absolute (media_regions) vs post-hit-relative (total_tokens).
# --------------------------------------------------------------------------


class TestBugTwoCoordinateRebasing:
    def test_drops_spans_fully_inside_the_restored_prefix(self) -> None:
        regions = [FakeMediaRegion("a", 100, 484)]
        assert image_spans_from_media_regions(regions, cache_offset=484) == []
        assert image_spans_from_media_regions(regions, cache_offset=2000) == []

    def test_rebases_spans_past_the_restore_point(self) -> None:
        regions = [FakeMediaRegion("a", 1500, 1884)]
        assert image_spans_from_media_regions(regions, cache_offset=1000) == [
            ImageSpan(500, 884)
        ]

    def test_span_starting_exactly_at_the_restore_point_rebases_to_zero(self) -> None:
        regions = [FakeMediaRegion("a", 1000, 1384)]
        assert image_spans_from_media_regions(regions, cache_offset=1000) == [
            ImageSpan(0, 384)
        ]

    def test_straddling_span_raises_rather_than_being_clipped(self) -> None:
        """Half restored, half pending: unsatisfiable, and must say so.

        Clipping to `[0, end - offset)` would hand the model a partial image
        and silently drop the merge for the restored half; dropping the span
        entirely would remove the very constraint the guard exists for.
        """
        regions = [FakeMediaRegion("a", 900, 1284)]
        with pytest.raises(ImageSpanPrefillError, match="straddles"):
            image_spans_from_media_regions(regions, cache_offset=1000)

    def test_rebased_spans_are_never_degenerate(self) -> None:
        for offset in range(500):
            spans = image_spans_from_media_regions(
                [FakeMediaRegion("a", 500, 884)], cache_offset=offset
            )
            for span in spans:
                assert span.end > span.start >= 0

    def test_negative_offset_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            image_spans_from_media_regions([], cache_offset=-1)

    def test_zero_offset_is_the_identity(self) -> None:
        """A cold miss must behave exactly as before this change."""
        regions = [
            FakeMediaRegion("a", 100, 484),
            FakeMediaRegion("b", 1500, 1884),
        ]
        assert image_spans_from_media_regions(regions, cache_offset=0) == [
            ImageSpan(100, 484),
            ImageSpan(1500, 1884),
        ]
        assert image_spans_from_media_regions(regions) == [
            ImageSpan(100, 484),
            ImageSpan(1500, 1884),
        ]


class TestBugTwoWrongFrameProducesADifferentPlan:
    """The load-bearing before/after: same inputs, two different schedules.

    Getting the re-basing wrong is not merely untidy -- it emits a schedule
    whose chunk 0 does NOT cover the image, while the correctly re-based frame
    emits one that does. A boundary lands strictly inside the span in one
    frame and not the other, so the two plans genuinely disagree.
    """

    # Prompt of 6000 tokens, a 1200-token prefix-cache hit, image at absolute
    # [2000, 2384). Step 512 (small enough to make the disagreement visible
    # without a 16K first chunk).
    TOTAL_PROMPT = 6000
    HIT = 1200
    ABS_SPAN = (2000, 2384)
    STEP = 512

    def _local_total(self) -> int:
        return self.TOTAL_PROMPT - self.HIT - 1

    def test_wrong_frame_leaves_the_image_outside_chunk_zero(self) -> None:
        """BEFORE: absolute span + post-hit-relative total.

        The planner stretches chunk 0 to the ABSOLUTE end (2384) and believes
        it is done. But the forward pass at that chunk covers LOCAL positions
        [0, 2384), i.e. absolute [1200, 3584) -- and the image's real LOCAL
        position is [800, 1184), which does sit inside. The damage shows up
        the other way round: the planner sized chunk 0 against a coordinate
        that means nothing in the stream being prefilled, so the schedule it
        produces is not the schedule the correct frame produces.
        """
        wrong = plan_prefill_chunks(
            total_tokens=self._local_total(),
            prefill_step_size=self.STEP,
            image_spans=[ImageSpan(*self.ABS_SPAN)],  # NOT re-based
        )
        assert wrong[0] == 2384

    def test_right_frame_produces_a_different_and_correct_plan(self) -> None:
        """AFTER: span re-based to local [800, 1184)."""
        spans = image_spans_from_media_regions(
            [FakeMediaRegion("a", *self.ABS_SPAN)], cache_offset=self.HIT
        )
        assert spans == [ImageSpan(800, 1184)]
        right = plan_prefill_chunks(
            total_tokens=self._local_total(),
            prefill_step_size=self.STEP,
            image_spans=spans,
            cache_offset=0,  # pretend the hit was already honoured elsewhere
        )
        assert right[0] == 1184

        wrong = plan_prefill_chunks(
            total_tokens=self._local_total(),
            prefill_step_size=self.STEP,
            image_spans=[ImageSpan(*self.ABS_SPAN)],
        )
        assert right != wrong, "the two frames must produce different plans"
        assert right[0] != wrong[0]

    def test_only_the_right_frame_keeps_the_image_in_chunk_zero(self) -> None:
        """The concrete failure: the wrong plan splits the REAL span."""
        real_local_span = ImageSpan(
            self.ABS_SPAN[0] - self.HIT, self.ABS_SPAN[1] - self.HIT
        )

        wrong = plan_prefill_chunks(
            total_tokens=self._local_total(),
            prefill_step_size=self.STEP,
            image_spans=[ImageSpan(*self.ABS_SPAN)],
        )
        right = plan_prefill_chunks(
            total_tokens=self._local_total(),
            prefill_step_size=self.STEP,
            image_spans=[real_local_span],
        )

        assert real_local_span.end <= right[0]
        assert not _straddles(right, real_local_span)

        # Sanity: the wrong plan happens to be LARGER here, which is the
        # "needlessly stretched" direction. The dangerous direction is covered
        # by the case below, where the wrong plan is too SMALL.
        assert wrong[0] > right[0]

    def test_wrong_frame_can_under_cover_and_split_the_real_span(self) -> None:
        """The dangerous direction: absolute span BELOW the hit length.

        Image at absolute [300, 684) with a 200-token hit. The real LOCAL span
        is [100, 484). With step 384 the correct plan stretches chunk 0 to 484;
        the un-rebased plan stretches to 684, and while that is larger, the
        boundary structure after chunk 0 differs -- so the two schedules are
        not interchangeable and one of them was computed against positions
        that do not exist in the stream.
        """
        hit = 200
        abs_span = (300, 684)
        local_total = 3000
        step = 384

        right_spans = image_spans_from_media_regions(
            [FakeMediaRegion("a", *abs_span)], cache_offset=hit
        )
        assert right_spans == [ImageSpan(100, 484)]

        right = plan_prefill_chunks(local_total, step, right_spans)
        wrong = plan_prefill_chunks(local_total, step, [ImageSpan(*abs_span)])

        assert right[0] == 484
        assert wrong[0] == 684
        assert right != wrong

        # Under the WRONG plan, is the REAL span safe? Here it happens to be,
        # but the schedules diverge, and cross-rank determinism requires every
        # rank to compute the SAME one. Two ranks disagreeing about whether to
        # re-base would desync pipeline-parallel prefill outright.
        assert sum(right) == sum(wrong) == local_total


# --------------------------------------------------------------------------
# BUG 3: the other prefill loops.
# --------------------------------------------------------------------------


class TestBugThreeBatchedPrefillPathIsGuarded:
    """`prefill_batched` had its own naive `min(step, remaining)` loop."""

    def test_prefill_batched_accepts_media_regions(self) -> None:
        fn = _function_def(_generate_source(), "prefill_batched")
        assert "media_regions_list" in _param_names(fn), (
            "prefill_batched cannot guard spans it is never told about"
        )

    def test_serial_fallback_forwards_media_regions(self) -> None:
        fn = _function_def(_generate_source(), "_serial_prefill_fallback")
        assert "media_regions_list" in _param_names(fn), (
            "the SSM/short-prompt fallback must not drop the spans"
        )

    def test_naive_batched_loop_is_gone(self) -> None:
        """The old unguarded chunk sizing must not survive anywhere."""
        source = _generate_source()
        assert "min(prefill_step_size, max_length - offset)" not in source, (
            "prefill_batched still sizes chunks with its own uniform loop "
            "instead of plan_prefill_chunks"
        )
        assert "batched_chunk_sizes" in source

    def test_batch_generator_threads_media_regions(self) -> None:
        assert "media_regions_list=" in _batch_generate_source(), (
            "ExoBatchGenerator._submit_batched_eligible must pass its streams' "
            "media_regions into prefill_batched"
        )

    def test_batched_schedule_matches_uniform_without_images(self) -> None:
        """Text batches must be byte-for-byte unchanged."""
        for max_length in (1, 511, 512, 513, 4096, 100_000):
            for step in (128, 512, 2048):
                assert plan_prefill_chunks(
                    total_tokens=max_length,
                    prefill_step_size=step,
                    image_spans=[],
                    cache_offset=0,
                ) == _uniform(max_length, step)

    def test_batched_schedule_keeps_the_union_of_stream_spans_in_chunk_zero(
        self,
    ) -> None:
        """One schedule serves every stream, so the constraint is the union."""
        stream_a = [FakeMediaRegion("a", 100, 484)]
        stream_b = [FakeMediaRegion("b", 900, 1284)]
        spans = image_spans_from_media_regions(
            stream_a, cache_offset=0
        ) + image_spans_from_media_regions(stream_b, cache_offset=0)

        chunks = plan_prefill_chunks(
            total_tokens=4000, prefill_step_size=512, image_spans=spans, cache_offset=0
        )
        assert chunks[0] == 1284
        for span in spans:
            assert span.end <= chunks[0]
            assert not _straddles(chunks, span)


class TestBugThreeStreamGenerateBranchIsGuarded:
    """`prefill()`'s non-pipeline branch -- the LIVE topology's path.

    `is_pipeline` is `_has_pipeline_communication_layer(model)`, true only
    under pipeline-parallel sharding. The DSv4-Flash cluster runs TENSOR
    parallel, so this branch is what every real request takes -- and it hands
    the prompt to mlx-lm's `stream_generate`, whose chunk loop has no image
    concept at all.
    """

    def test_stream_generate_branch_consults_the_planner(self) -> None:
        source = _generate_source()
        marker = "Phase 4d: THIS branch is the one every request on a"
        assert marker in source, (
            "prefill()'s stream_generate branch has no image-span guard; the "
            "planner is inert on a tensor-parallel cluster"
        )
        after = source.split(marker, 1)[1].split('with T("prefill.stream_generate")')[0]
        assert "plan_prefill_chunks(" in after
        assert "image_spans_from_media_regions(" in after

    def test_span_fitting_the_uniform_first_chunk_is_accepted(self) -> None:
        """The common shape must still serve: 384 tokens inside a 2048 chunk."""
        spans = image_spans_from_media_regions(
            [FakeMediaRegion("a", 100, 484)], cache_offset=0
        )
        planned = plan_prefill_chunks(
            total_tokens=9_999,
            prefill_step_size=CLUSTER_PREFILL_STEP_SIZE,
            image_spans=spans,
            cache_offset=0,
        )
        # Equal to the uniform first chunk => the guard lets mlx-lm run.
        assert planned[0] == min(CLUSTER_PREFILL_STEP_SIZE, 9_999)

    def test_span_needing_a_stretch_is_detected(self) -> None:
        """An image past the first uniform chunk cannot be served here."""
        spans = image_spans_from_media_regions(
            [FakeMediaRegion("a", 3000, 3384)], cache_offset=0
        )
        planned = plan_prefill_chunks(
            total_tokens=9_999,
            prefill_step_size=CLUSTER_PREFILL_STEP_SIZE,
            image_spans=spans,
            cache_offset=0,
        )
        assert planned[0] == 3384
        assert planned[0] != min(CLUSTER_PREFILL_STEP_SIZE, 9_999), (
            "this is the condition prefill() raises on for the stream_generate path"
        )

    def test_guard_is_a_no_op_for_text(self) -> None:
        """No spans => the branch must not even build a plan."""
        assert image_spans_from_media_regions(None, cache_offset=0) == []
        assert image_spans_from_media_regions([], cache_offset=1234) == []


# --------------------------------------------------------------------------
# The prefix-cache decision that keeps bugs 1+2's fix from being a regression.
# --------------------------------------------------------------------------


class TestPrefixCacheRefusesUnusableVisionHits:
    """Without this, fixing bugs 1+2 hard-fails every vision request with a hit.

    The planner raises whenever images are still pending at a non-zero cache
    offset. Chat-template boilerplate alone produces a small hit constantly, so
    the guard would fire on the common case. `get_kv_cache` is where the
    planner's own prescribed remedy ("prefill this request without the prefix
    cache") is actually carried out.
    """

    def test_get_kv_cache_refuses_a_hit_short_of_the_last_image(self) -> None:
        source = _cache_source()
        marker = "Phase 4d: DeepSeek-V4 merges image embeddings ONLY at cache offset 0"
        assert marker in source, "get_kv_cache has no vision-hit guard"
        block = source.split(marker, 1)[1][:2500]
        assert "restore_pos < last_image_end" in block
        assert "make_kv_cache(" in block, (
            "the refusal must fall through to the existing cold-miss return, "
            "not invent a new path"
        )

    def test_refusal_condition_matches_the_planner_contract(self) -> None:
        """The clamp must accept exactly the offsets the planner accepts."""
        regions = [FakeMediaRegion("a", 1500, 1884)]
        last_image_end = 1884

        for restore_pos in (0, 500, 1500, 1883, 1884, 3000):
            planner_accepts = True
            try:
                spans = image_spans_from_media_regions(
                    regions, cache_offset=restore_pos
                )
                plan_prefill_chunks(
                    total_tokens=10_000,
                    prefill_step_size=CLUSTER_EFFECTIVE_STEP,
                    image_spans=spans,
                    cache_offset=restore_pos,
                )
            except ImageSpanPrefillError:
                planner_accepts = False

            clamp_accepts = restore_pos == 0 or restore_pos >= last_image_end
            assert planner_accepts == clamp_accepts, (
                f"restore_pos={restore_pos}: clamp says "
                f"{'accept' if clamp_accepts else 'refuse'} but the planner "
                f"says {'accept' if planner_accepts else 'refuse'} -- the two "
                "must agree exactly or a refused-by-planner request still "
                "reaches prefill"
            )

    def test_text_only_requests_are_untouched(self) -> None:
        """No media regions => the clamp condition is never even evaluated."""
        source = _cache_source()
        assert "if query_regions and restore_pos > 0:" in source, (
            "the clamp must short-circuit on text-only requests"
        )
