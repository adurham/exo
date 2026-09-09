# pyright: reportMissingImports=false, reportUnknownMemberType=false
# pyright: reportUntypedFunctionDecorator=false, reportPrivateUsage=false
# `pytest.raises`' ExceptionInfo is generic-unresolved without pytest stubs in
# this environment -- the same repo-wide condition the sibling
# test_prefill_chunking_cache_offset_wiring.py header covers.
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Phase 4d residual gaps: the two call sites the first 4d pass missed.

An independent review after 4d landed found two more places the image-span
chunk guard was not reached. Both were INERT in production at the time, which
is exactly why they needed tests rather than a shrug.

GAP 1 -- ``prefill_interruptible_start`` called
``_pipeline_parallel_prefill_steps`` with NEITHER ``media_regions`` NOR
``cache_offset``, so that generator planned against zero spans at a hardcoded
offset 0. Structurally identical to the bug 4d fixed at the two other call
sites; double-gated inert (the live cluster is tensor-parallel so
``is_pipeline`` is False, and ``try_start_chunked_prefill`` declines every
vision request anyway).

GAP 2 -- the disaggregated prefill server had no image-span awareness at all.
It turned out not to be a chunking gap: the path structurally CANNOT serve
vision, because ``PrefillRequest`` carries no embeddings and the server never
installs ``patch_embed_tokens``. So the fix is a routing exclusion plus a
loud server-side boundary check, not a chunk guard.

Source files are read from disk rather than imported, matching
``test_prefill_chunking_cache_offset_wiring``: importing
``generator.generate`` drags in ``exo.shared.constants``, which resolves built
dashboard assets at import time and fails in any checkout that has not run
``npm run build``. These assert on WIRING, so reading the source is both
sufficient and more robust.
"""

import ast
from pathlib import Path

import pytest

from exo.worker.engines.mlx.prefill_chunking import (
    ImageSpanPrefillError,
    image_spans_from_media_regions,
    plan_prefill_chunks,
)

_MLX_ENGINE_DIR = Path(__file__).resolve().parent.parent


def _read_source(*parts: str) -> str:
    path = _MLX_ENGINE_DIR.joinpath(*parts)
    assert path.is_file(), f"expected source file at {path}"
    return path.read_text()


def _function_def(source: str, name: str) -> ast.FunctionDef:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"no function named {name!r} in source")


def _param_names(fn: ast.FunctionDef) -> set[str]:
    args = fn.args
    return {a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)}


def _call_name(call: ast.Call) -> str | None:
    """The called function's bare name, for both `f(...)` and `o.f(...)`."""
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _calls_named(fn: ast.FunctionDef, callee: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Call) and _call_name(node) == callee
    ]


def _keyword_source(call: ast.Call, name: str, source: str) -> str:
    for kw in call.keywords:
        if kw.arg == name:
            return ast.get_source_segment(source, kw.value) or ""
    raise AssertionError(
        f"call has no keyword {name!r}; keywords present: "
        f"{sorted(k.arg for k in call.keywords if k.arg)}"
    )


# ==========================================================================
# GAP 1: prefill_interruptible_start -> _pipeline_parallel_prefill_steps
# ==========================================================================


def test_prefill_interruptible_start_accepts_media_regions_and_cache_offset() -> None:
    """The signature gained both parameters.

    Before this fix it had neither, so there was no way for a caller to reach
    the planner from this path even if it wanted to.
    """
    fn = _function_def(
        _read_source("generator", "generate.py"), "prefill_interruptible_start"
    )
    params = _param_names(fn)
    assert "media_regions" in params, (
        "prefill_interruptible_start must accept media_regions -- without it "
        "its _pipeline_parallel_prefill_steps call plans against no image "
        "spans, which is the exact bug 4d fixed at the other call sites"
    )
    assert "cache_offset" in params, (
        "prefill_interruptible_start must accept cache_offset -- it is what "
        "re-bases spans into the local frame AND what reaches the planner's "
        "own non-zero-offset guard"
    )


def test_interruptible_start_forwards_both_to_the_chunk_generator() -> None:
    """The values are actually FORWARDED, not merely accepted.

    A parameter that is accepted and dropped is worse than none at all: it
    reads as fixed at the call site while the planner still sees nothing.
    """
    source = _read_source("generator", "generate.py")
    fn = _function_def(source, "prefill_interruptible_start")
    calls = _calls_named(fn, "_pipeline_parallel_prefill_steps")
    assert len(calls) == 1, (
        f"expected exactly one _pipeline_parallel_prefill_steps call in "
        f"prefill_interruptible_start, found {len(calls)}"
    )
    call = calls[0]
    assert _keyword_source(call, "media_regions", source) == "media_regions"
    assert _keyword_source(call, "cache_offset", source) == "cache_offset"


def test_interruptible_start_matches_the_already_fixed_pipeline_call_site() -> None:
    """Gap 1's wiring is the SAME shape as the 4d-fixed sibling.

    ``prefill`` -> ``pipeline_parallel_prefill`` is the call site 4d fixed.
    Pinning the two together is what stops them drifting apart again: if
    someone adds a third argument to the planner's inputs and threads it
    through only one, this fails.
    """
    source = _read_source("generator", "generate.py")
    wrapper = _function_def(source, "pipeline_parallel_prefill")
    interruptible = _function_def(source, "prefill_interruptible_start")

    planner_inputs = {"media_regions", "cache_offset"}
    for fn in (wrapper, interruptible):
        call = _calls_named(fn, "_pipeline_parallel_prefill_steps")[0]
        forwarded = {kw.arg for kw in call.keywords if kw.arg in planner_inputs}
        assert forwarded == planner_inputs, (
            f"{fn.name} forwards {sorted(forwarded)} to "
            f"_pipeline_parallel_prefill_steps, expected {sorted(planner_inputs)}"
        )


def test_batch_generate_caller_passes_both_from_real_variables() -> None:
    """The CALLER has the data, and passes the real variables.

    Confirms the claim rather than assuming it: ``try_start_chunked_prefill``
    lives inside ``_submit_batched_decode_deferred``, whose parameters include
    both ``media_regions`` and ``prefix_hit_length`` -- the same two values its
    sibling ``run_prefill`` already forwards to ``prefill``. Passing literals
    (``[]``/``0``) would type-check and silently reintroduce the gap, so the
    expressions themselves are asserted.
    """
    source = _read_source("generator", "batch_generate.py")
    enclosing = _function_def(source, "_submit_batched_decode_deferred")
    enclosing_params = _param_names(enclosing)
    assert {"media_regions", "prefix_hit_length"} <= enclosing_params, (
        "the enclosing method must have both values in scope for "
        "try_start_chunked_prefill to forward them"
    )

    inner = _function_def(source, "try_start_chunked_prefill")
    calls = _calls_named(inner, "prefill_interruptible_start")
    assert len(calls) == 1
    call = calls[0]
    assert _keyword_source(call, "media_regions", source) == "media_regions"
    assert _keyword_source(call, "cache_offset", source) == "prefix_hit_length"


def test_gap1_wiring_now_carries_a_verdict_the_old_call_could_not() -> None:
    """The behavioural consequence, computed against the real planner.

    The two arguments are not cosmetic. This is the exact input the fixed call
    site now produces for a request with an image span, versus what the old
    call (no regions, offset 0) produced: a schedule that silently splits the
    span out of chunk 0 vs. a stretched chunk 0.

    Effective step 1024 = the cluster's EXO_PREFILL_STEP_SIZE=2048 halved by
    ``_pipeline_parallel_prefill_steps``' ``// min(4, group.size())`` at
    world_size 2, which is the group size this path requires.
    """

    class _Region:
        def __init__(self, start: int, end: int) -> None:
            self.start_pos = start
            self.end_pos = end

    step = 1024
    total = 10_000
    regions = [_Region(1500, 1884)]

    # OLD call: media_regions never arrived, so the planner saw no spans.
    old = plan_prefill_chunks(
        total_tokens=total, prefill_step_size=step, image_spans=[]
    )
    assert old[0] == 1024
    assert old[0] < 1884, (
        "the un-wired call produced a schedule whose chunk 0 ends at "
        f"{old[0]}, leaving the image span at [1500, 1884) to be processed at "
        "a NON-ZERO cache offset -- case 2 of the invariant, where DeepSeek-V4 "
        "cannot merge the embeddings at all. This is what the guard exists to "
        "stop, and the un-wired call could not see it."
    )

    # NEW call: the same two arguments the fixed site now forwards.
    spans = image_spans_from_media_regions(regions, cache_offset=0)
    new = plan_prefill_chunks(
        total_tokens=total,
        prefill_step_size=step,
        image_spans=spans,
        cache_offset=0,
    )
    assert new[0] == 1884, (
        "with the spans threaded through, chunk 0 stretches to cover the last "
        f"image token; got {new[0]}"
    )
    assert new != old, "the two arguments must change the resulting schedule"


def test_gap1_cache_offset_reaches_the_planners_offset_guard() -> None:
    """``cache_offset``'s second job, exercised end to end.

    A non-zero offset with images still pending is unsatisfiable by any chunk
    schedule. Passing 0 (the old implicit default) hides that.
    """

    class _Region:
        def __init__(self, start: int, end: int) -> None:
            self.start_pos = start
            self.end_pos = end

    # Span entirely ahead of the restore point, still to prefill.
    spans = image_spans_from_media_regions([_Region(3000, 3384)], cache_offset=2000)
    with pytest.raises(ImageSpanPrefillError, match="cache offset 0"):
        plan_prefill_chunks(
            total_tokens=8000,
            prefill_step_size=1024,
            image_spans=spans,
            cache_offset=2000,
        )

    # And with the OLD implicit cache_offset=0 the same request is accepted --
    # which is precisely why the parameter had to be threaded, not defaulted.
    accepted = plan_prefill_chunks(
        total_tokens=8000, prefill_step_size=1024, image_spans=spans, cache_offset=0
    )
    assert accepted[0] == 1384, (
        "with the offset dropped the planner happily returns a schedule for a "
        "request the model cannot serve"
    )


# ==========================================================================
# GAP 2: the disaggregated / remote-prefill path.
#
# Not a chunking gap -- a capability gap. `should_use_remote_prefill` is the
# routing fix; `run_prefill_for_request`'s check is the boundary behind it.
# ==========================================================================


def test_should_use_remote_prefill_refuses_every_vision_request() -> None:
    """The routing rule itself, called directly.

    Imported rather than source-read: `generator.remote_prefill` does pull in
    mlx, but this whole test module already lives in the mlx test package and
    the function is pure.
    """
    from exo.worker.engines.mlx.generator.remote_prefill import (
        REMOTE_PREFILL_MIN_TOKENS,
        should_use_remote_prefill,
    )

    big = REMOTE_PREFILL_MIN_TOKENS + 1

    # CONTROL: a text request with the same shape DOES route remotely, so the
    # refusal below is caused by `has_vision` and not by the other terms.
    assert should_use_remote_prefill(
        uncached_token_count=big, prefill_endpoint="host:1234", has_vision=False
    )

    assert not should_use_remote_prefill(
        uncached_token_count=big, prefill_endpoint="host:1234", has_vision=True
    ), (
        "a vision request must never be routed to a prefill server: "
        "PrefillRequest carries no embeddings and the server never installs "
        "patch_embed_tokens, so the image tokens would be embedded from the "
        "wrong table rows and the resulting KV cache silently returned as "
        "authoritative"
    )

    # The pre-existing terms are unchanged for text.
    assert not should_use_remote_prefill(
        uncached_token_count=big, prefill_endpoint=None, has_vision=False
    )
    assert not should_use_remote_prefill(
        uncached_token_count=REMOTE_PREFILL_MIN_TOKENS,
        prefill_endpoint="host:1234",
        has_vision=False,
    )


def test_every_use_remote_site_goes_through_the_shared_rule() -> None:
    """No call site may recompute the routing rule inline.

    The gap existed because the rule was written out three times and none of
    the copies knew about vision. Asserting that the old inline shape is gone
    is what stops a fourth copy reintroducing it.
    """
    for parts in (
        ("generator", "generate.py"),
        ("generator", "batch_generate.py"),
    ):
        source = _read_source(*parts)
        for line in source.splitlines():
            stripped = line.strip()
            assert not stripped.startswith("use_remote = ("), (
                f"{'/'.join(parts)} still computes use_remote inline "
                f"({stripped!r}); it must call should_use_remote_prefill so "
                "the vision exclusion cannot be forgotten at one site"
            )
        assert "should_use_remote_prefill" in source


def test_use_remote_sites_pass_the_real_vision_expression() -> None:
    """`has_vision` is derived from the request, not hardcoded.

    `has_vision=False` at every site would satisfy the previous test while
    leaving the gap fully open.
    """
    for parts in (
        ("generator", "generate.py"),
        ("generator", "batch_generate.py"),
    ):
        source = _read_source(*parts)
        tree = ast.parse(source)
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "should_use_remote_prefill"
        ]
        assert calls, f"{'/'.join(parts)} has no should_use_remote_prefill call"
        for call in calls:
            expr = _keyword_source(call, "has_vision", source)
            assert expr == "vision is not None", (
                f"{'/'.join(parts)} passes has_vision={expr!r}; it must be the "
                "real per-request expression `vision is not None`. An empty "
                "media_regions list is a plausible representation for "
                "'images present but no spans resolved', so truthiness of "
                "media_regions is NOT a safe substitute."
            )
        assert len(calls) == (1 if parts[-1] == "generate.py" else 3), (
            f"{'/'.join(parts)} has {len(calls)} routing decisions; if a call "
            "site was added or removed, confirm the new one is guarded too"
        )


def test_serve_rejects_out_of_vocabulary_image_tokens() -> None:
    """The server-side boundary check, run against a real request.

    Uses a stub model whose `embed_tokens.weight` has a known row count, which
    is the same structural property `_embedding_table_size` reads off a real
    checkpoint. DeepSeek-V4's image sentinels sit at `vocab_size + {0..4}`, so
    a token at exactly `vocab_size` is the real failure shape.
    """
    from exo.worker.disaggregated.server import PrefillRequest
    from exo.worker.engines.mlx.disaggregated.serve import (
        RemotePrefillVisionUnsupportedError,
        _reject_if_vision_request,
    )

    vocabulary_size = 128

    class _Weight:
        shape = (vocabulary_size, 8)

    class _Embed:
        weight = _Weight()

    class _Inner:
        embed_tokens = _Embed()

    class _Model:
        model = _Inner()

    model = _Model()

    # CONTROL: an all-in-vocabulary (text) request passes untouched.
    _reject_if_vision_request(
        model,  # type: ignore[arg-type]
        PrefillRequest(request_id="text", token_ids=[0, 5, vocabulary_size - 1]),
    )

    with pytest.raises(RemotePrefillVisionUnsupportedError) as excinfo:
        _reject_if_vision_request(
            model,  # type: ignore[arg-type]
            PrefillRequest(
                request_id="vision",
                token_ids=[1, 2, vocabulary_size, vocabulary_size + 4, 3],
            ),
        )
    message = str(excinfo.value)
    assert "vision" in message
    assert str(vocabulary_size) in message
    assert "should_use_remote_prefill" in message, (
        "the error must name the routing rule that should have prevented it, "
        "so an operator can find the bypass rather than just the symptom"
    )


def test_serve_skips_the_check_when_the_table_size_is_unknowable() -> None:
    """An unresolvable model must not hard-fail every text request.

    `_embedding_table_size` returning None means "cannot check", and the
    guarantee lives in `should_use_remote_prefill` regardless. Test doubles and
    unusual architectures land here; turning that into a serving outage would
    be a worse bug than the one being closed.
    """
    from exo.worker.disaggregated.server import PrefillRequest
    from exo.worker.engines.mlx.disaggregated.serve import (
        _embedding_table_size,
        _reject_if_vision_request,
    )

    class _Opaque:
        pass

    assert _embedding_table_size(_Opaque()) is None  # type: ignore[arg-type]
    _reject_if_vision_request(
        _Opaque(),  # type: ignore[arg-type]
        PrefillRequest(request_id="unknown", token_ids=[1, 2, 999_999]),
    )


def test_run_prefill_for_request_checks_before_any_state_is_touched() -> None:
    """Order matters: reject BEFORE the prefix cache and the forward pass.

    `get_kv_cache` is called on the server with no `media_regions`, so the
    image-span clamp added to it in 4d cannot fire there. Checking after it
    would mean a rejected request had already mutated the trie.
    """
    source = _read_source("disaggregated", "serve.py")
    fn = _function_def(source, "run_prefill_for_request")

    def _first_lineno(callee: str) -> int:
        for node in ast.walk(fn):
            if isinstance(node, ast.Call) and _call_name(node) == callee:
                return node.lineno
        raise AssertionError(f"no call to {callee!r} in run_prefill_for_request")

    reject_line = _first_lineno("_reject_if_vision_request")
    for later in ("get_kv_cache", "make_kv_cache", "mlx_prefill"):
        assert reject_line < _first_lineno(later), (
            f"_reject_if_vision_request must run before {later}; a rejected "
            "request must not reach the prefix cache or a forward pass"
        )
