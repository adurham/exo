# pyright: reportPrivateUsage=false, reportAny=false
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false, reportArgumentType=false
# pyright: reportUnknownLambdaType=false, reportInvalidCast=false
"""Regression tests for Phase 2's live-wiring step: turning
``register_prefill_session()`` into the real production path
(2026-08-07). Two real correctness hazards were found via consult
review BEFORE this wiring existed (see
docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md's 2026-08-07
entry for the full design history) -- these tests target each one
directly and were verified, per this session's own established
discipline, to FAIL LOUDLY when their respective fix is reverted
(see each test's own docstring for the exact revert used).

HAZARD 1 (caller-assumed-completion): swapping the synchronous
``deferred.run_prefill()`` for ``register_prefill_session()`` makes
``_run_deferred_prefill_for_grant()`` return while the real prefill is
still in-progress across FUTURE ticks -- everything downstream
(``enqueue_admission``/``stage_local_cache``, and anything that reads
``_active_tasks``/decode state assuming "prefill done, cache ready")
must NOT be reachable until the real multi-chunk drive genuinely
completes.

HAZARD 2 (rank-registration skew): each rank calls
``register_prefill_session()`` independently, once per real chunk
boundary (not just at admission time) -- if one rank's registration for
chunk i+1 lands AFTER its peer has already sent (or could send) chunk
i+1's first ``PrefillAdvanceMessage``, the two ranks' chunk-drive state
machines desync. The fix (design doc's 2026-08-07 entry, Hazard 2
section) is ORDERING, not a new wire message: registering the next
chunk's session happens SYNCHRONOUSLY, INLINE, inside
``_advance_chunked_prefill_drive`` -- which itself only ever runs
strictly between one ``tick()`` return and the next ``tick()`` call in
the SAME runner event loop, so a peer physically cannot observe the
next chunk's first advance before this rank's own registration for it
has already happened.

These tests exercise ``ExoBatchGenerator``'s own real wiring
(``_run_deferred_prefill_for_grant``, ``_advance_chunked_prefill_drive``,
``_admit_completed_prefill``) against a REAL ``Rank0BatchedDecodeGlue``
(real ``register_prefill_session``/``has_active_prefill_session``
state), using a SYNTHETIC ``ChunkedPrefillDrive`` (a fake outer
generator + a real ``ResumablePrefillSession`` driving a fake
``_forward_steps``-shaped model) to avoid needing the real DSv4
checkpoint or a real distributed transport -- mirrors this session's
own established precedent (``test_pp_pipeline_parallel_prefill_session_
integration.py``, ``test_pp_batched_decode_glue_chunk_drive.py``) for
testing this exact class of state-machine correctness without real
hardware.
"""

from __future__ import annotations

from typing import Iterator, Literal, cast
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import pytest
from mlx_lm.tokenizer_utils import TokenizerWrapper
from transformers import AutoTokenizer

from exo.shared.types.common import ModelId
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.worker.engines.mlx.generator.batch_generate import (
    ChunkedPrefillDrive,
    ExoBatchGenerator,
)
from exo.worker.engines.mlx.pp_batched_decode_adapter import (
    BatchedDecodeResponseAdapter,
)
from exo.worker.engines.mlx.pp_batched_decode_glue import (
    GlueError,
    PrefillGrant,
    Rank0BatchedDecodeGlue,
)
from exo.worker.engines.mlx.pp_batched_decode_runtime import BatchedDecodeSession
from exo.worker.engines.mlx.pp_metaframe import ForwardPhase
from exo.worker.engines.mlx.pp_prefill_session import (
    ForwardStep,
    ResumablePrefillSession,
)

pytestmark = pytest.mark.filterwarnings("ignore")


def _make_tiny_llama() -> nn.Module:
    from mlx_lm.models.llama import Model as LlamaModel
    from mlx_lm.models.llama import ModelArgs

    args = ModelArgs(
        model_type="llama",
        hidden_size=256,
        num_hidden_layers=4,
        intermediate_size=512,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
        vocab_size=4096,
        rope_theta=10000.0,
        tie_word_embeddings=True,
    )
    mx.random.seed(42)
    model = LlamaModel(args)
    params = model.parameters()
    new_params = mlx.utils.tree_map(
        lambda p: mx.random.normal(shape=p.shape, dtype=p.dtype)
        if isinstance(p, mx.array)
        else p,
        params,
    )
    model.update(new_params)
    mx.eval(model.parameters())
    return model


def _make_tokenizer() -> TokenizerWrapper:
    from huggingface_hub import snapshot_download

    model_path = snapshot_download(
        "mlx-community/Qwen3.5-35B-A3B-4bit",
        allow_patterns=["tokenizer*", "*.jinja"],
    )
    hf_tokenizer = AutoTokenizer.from_pretrained(model_path)
    return TokenizerWrapper(hf_tokenizer)


class _FakeGroup:
    """Minimal mx.distributed.Group stand-in, N=2 -- Rank0BatchedDecodeGlue
    only stores it (tick()'s real wire calls are monkeypatched away in
    every test below, since these tests exercise ExoBatchGenerator's
    OWN drive-wiring logic, not the real wire protocol -- that is
    already covered by test_pp_batched_decode_glue_chunk_drive.py and
    the real 2-process subprocess tests)."""

    def rank(self) -> int:
        return 0

    def size(self) -> int:
        return 2


class _FakeInterruptibleModel:
    """Structural stand-in matching ``_InterruptibleForward`` -- same
    established pattern as test_pp_prefill_session.py's own
    ``_FakeInterruptibleModel``. ``n_layers`` is small and
    caller-controlled so a chunk's ``ResumablePrefillSession`` can be
    driven to ``done=True`` in very few ``advance()`` calls."""

    def __init__(self, n_layers: int) -> None:
        self.n_layers = n_layers

    def _forward_steps(
        self,
        inputs: mx.array,
        cache: object = None,
        *,
        interruptible: bool = False,
    ) -> Iterator[ForwardStep]:
        h = inputs
        for i in range(self.n_layers):
            h = h + mx.array([1.0])
            mx.eval(h)
            if interruptible:
                yield ("layer", i, h)
        yield ("done", None, h)


def _fake_outer_generator(
    n_chunks: int, model: _FakeInterruptibleModel
) -> Iterator[
    tuple[Literal["chunk"], int, mx.array] | tuple[Literal["done"], None, None]
]:
    """Synthetic stand-in for ``_pipeline_parallel_prefill_steps`` --
    same real yield shape (``("chunk", i, tokens)`` per real chunk,
    then ``("done", None, None)``), just without the real distributed
    pipeline-bubble bookkeeping that shape doesn't need for THESE
    tests (drive-wiring correctness, not chunk-loop mechanics -- the
    real generator's own chunk-loop is separately tested by
    test_pp_pipeline_parallel_prefill_session_integration.py)."""
    del model
    for i in range(n_chunks):
        yield ("chunk", i, mx.array([float(i)]))
    yield ("done", None, None)


def _make_drive(n_chunks: int, n_layers_per_chunk: int) -> ChunkedPrefillDrive:
    """Builds a real ``ChunkedPrefillDrive`` around the synthetic outer
    generator + a real ``ResumablePrefillSession`` for its first
    chunk -- exactly what ``prefill_interruptible_start`` would
    construct and return, just without needing the real
    ``_has_pipeline_communication_layer``/``group.size()==2``/
    ``supports_chunked_prefill_interruption`` eligibility gate or a
    real distributed group."""
    fake_model = _FakeInterruptibleModel(n_layers=n_layers_per_chunk)
    outer_gen = _fake_outer_generator(n_chunks, fake_model)
    kind, idx, chunk_tokens = next(
        cast("Iterator[tuple[str, int, mx.array]]", outer_gen)
    )
    assert kind == "chunk" and idx == 0
    session = ResumablePrefillSession(
        inner_model=fake_model, inputs=chunk_tokens, cache=[]
    )
    fake_top_level_model = cast("object", type("FakeModel", (), {"layers": []})())
    return ChunkedPrefillDrive(
        model=cast("object", fake_top_level_model),  # type: ignore[arg-type]
        outer_gen=cast("object", outer_gen),
        session=session,
        chunk_index=0,
        cache=[],
        num_tokens=10,
        start_time=0.0,
        has_ssm=False,
        snapshots=[],
    )


def _make_rank0_generator_with_glue() -> tuple[
    ExoBatchGenerator, Rank0BatchedDecodeGlue
]:
    model = _make_tiny_llama()
    tokenizer = _make_tokenizer()
    gen = ExoBatchGenerator(
        model=model,
        tokenizer=tokenizer,
        group=cast(mx.distributed.Group, _FakeGroup()),
        kv_prefix_cache=None,
    )
    session = BatchedDecodeSession.new(max_concurrency=2)
    adapter = BatchedDecodeResponseAdapter(session=session, eos_ids=frozenset())
    glue = Rank0BatchedDecodeGlue(
        session=session,
        adapter=adapter,
        dst_rank=1,
        group=cast(mx.distributed.Group, _FakeGroup()),
        peer_prefill_layer_count=2,
    )
    gen._batched_decode_active = True
    gen._batched_decode_rank0_glue = glue
    gen._batched_decode_rank1_glue = None
    return gen, glue


def _submit_one(gen: ExoBatchGenerator) -> int:
    task_params = TextGenerationTaskParams(
        model=ModelId("test-model"),
        input=[],
        max_output_tokens=5,
        temperature=0.0,
        seed=0,
    )

    def _identity_all_sum(x: mx.array, **_: object) -> mx.array:
        return x

    with patch("mlx.core.distributed.all_sum", side_effect=_identity_all_sum):
        return gen.submit(task_params, "What is the capital of France?")


def test_grant_with_chunked_drive_does_not_admit_until_genuinely_done() -> None:
    """HAZARD 1 direct proof: when ``try_start_chunked_prefill()``
    returns a real (multi-chunk) drive, ``_run_deferred_prefill_for_grant``
    must NOT call ``enqueue_admission`` and must NOT pop the deferred
    entry -- the request must stay ineligible for decode until the
    REAL outer generator reaches its own ``("done", ...)`` yield, many
    ``_advance_chunked_prefill_drive`` calls later (one per real
    chunk).

    Verified load-bearing (2026-08-07): reverting the fix (restoring
    the pre-2026-08-07 shape -- ``_run_deferred_prefill_for_grant``
    calling ``deferred.run_prefill()`` unconditionally, ignoring
    ``try_start_chunked_prefill`` entirely) makes this test FAIL
    LOUDLY: the request is admitted immediately on grant, its uid
    lands in ``_batched_decode_rank0_glue.session``'s active requests
    before ANY chunk has actually run, and the drive's own
    ``register_prefill_session`` is never called at all --
    ``glue.has_active_prefill_session()`` stays False, which is
    exactly the fail-loud signal proving the fix is what makes this
    assertion pass.
    """
    gen, glue = _make_rank0_generator_with_glue()
    uid = _submit_one(gen)

    drive = _make_drive(n_chunks=2, n_layers_per_chunk=3)
    deferred = gen._deferred_prefill_by_uid[uid]
    deferred.try_start_chunked_prefill = lambda: drive

    grant = PrefillGrant(
        request_id=uid,
        cache_slot=deferred.cache_slot,
        n_prompt_tokens=1,
        single_request_fallback=False,
    )
    gen._run_deferred_prefill_for_grant(grant, is_rank1=False)

    # THE core hazard-1 assertions: nothing downstream may treat this
    # grant as "prefill done, ready to decode."
    assert uid in gen._deferred_prefill_by_uid, (
        "a request with an in-flight chunked-prefill drive must NOT be "
        "popped from _deferred_prefill_by_uid -- its prefill is not "
        "done yet"
    )
    assert gen._deferred_prefill_by_uid[uid].drive is drive
    assert not glue.session.has_active_requests(), (
        "enqueue_admission must NOT have been called -- the request's "
        "prefill has not genuinely completed (only chunk 0's FIRST "
        "layer-segment session was registered, not run to completion)"
    )
    assert glue.has_active_prefill_session(), (
        "register_prefill_session must have been called for chunk 0's "
        "session -- this is what makes tick() able to drive it at all"
    )

    # Drive the FULL multi-chunk prefill to genuine completion by
    # repeatedly draining each chunk's session then advancing the
    # drive -- mirrors exactly what tick()'s real RANK1_DRAINING
    # completion (PrefillAdvanceCompleted) would trigger, one call to
    # _advance_chunked_prefill_drive per real chunk boundary. tick()'s
    # OWN real chunk-drive state machine clears _active_prefill_session
    # (glue.py's "the chunk is done, retire it" line) right before
    # returning PrefillAdvanceCompleted -- since this test drives each
    # chunk's session directly (bypassing tick()'s real RANK1_DRAINING
    # send-count bookkeeping, which is separately tested by
    # test_pp_batched_decode_glue_chunk_drive.py), it must replicate
    # that SAME clear here to accurately simulate tick()'s real
    # calling contract for _advance_chunked_prefill_drive.
    for _ in range(20):
        active_id, session = glue._active_prefill_session  # type: ignore[misc]
        assert active_id == uid
        done = session.done
        while not done:
            _layers, done = session.advance(
                max_layers=100, phase_for_pause=ForwardPhase.PREFILL_CONTINUE
            )
        glue._active_prefill_session = None
        gen._advance_chunked_prefill_drive(uid, is_rank1=False)
        if uid not in gen._deferred_prefill_by_uid:
            break

    # THE completion assertion: only NOW, after every real chunk has
    # genuinely finished, must the request be admitted.
    assert uid not in gen._deferred_prefill_by_uid, (
        "the deferred entry must be popped once the ENTIRE multi-chunk "
        "drive (not just one chunk) reaches real completion"
    )
    assert glue.has_pending_admissions(), (
        "enqueue_admission must have run NOW that the drive is "
        "genuinely, fully done -- if this fails, the completion path "
        "never actually calls _admit_completed_prefill. (Real admission "
        "onto the wire happens later, inside tick() -- enqueue_admission "
        "itself is pure in-memory queueing, per this glue's own "
        "documented no-wire-I/O guarantee.)"
    )
    assert not glue.has_active_prefill_session(), (
        "no session should remain registered once the drive is fully done"
    )


def test_advance_registers_next_chunk_session_before_returning() -> None:
    """HAZARD 2 direct proof (inner chunk-boundary ordering): when
    ``_advance_chunked_prefill_drive`` resumes the outer generator and
    it yields a NEXT real chunk (not yet done), the NEW chunk's
    ``ResumablePrefillSession`` must be registered with the glue
    (``glue.has_active_prefill_session() is True``) BEFORE
    ``_advance_chunked_prefill_drive`` returns control to its caller
    -- never deferred, never scheduled for a later tick. This is what
    makes the "tick() is the only recv site, so a peer cannot observe
    the next advance before registration" argument provably true
    rather than merely probably true (see design doc's 2026-08-07
    Hazard 2 entry).

    Verified load-bearing (2026-08-07): reverting the fix (making
    chunk-boundary registration a SEPARATE, later call the caller must
    remember to make -- e.g. returning the new session instead of
    registering it inline) makes this test FAIL LOUDLY:
    ``glue.has_active_prefill_session()`` reads False immediately
    after ``_advance_chunked_prefill_drive`` returns, exactly the
    unsynchronized-registration window Hazard 2 describes.
    """
    gen, glue = _make_rank0_generator_with_glue()
    uid = _submit_one(gen)

    drive = _make_drive(n_chunks=3, n_layers_per_chunk=2)
    deferred = gen._deferred_prefill_by_uid[uid]
    deferred.try_start_chunked_prefill = lambda: drive

    grant = PrefillGrant(
        request_id=uid,
        cache_slot=deferred.cache_slot,
        n_prompt_tokens=1,
        single_request_fallback=False,
    )
    gen._run_deferred_prefill_for_grant(grant, is_rank1=False)
    assert glue.has_active_prefill_session()

    # Drive chunk 0's session to completion, exactly as tick() would.
    active_id, session0 = glue._active_prefill_session  # type: ignore[misc]
    assert active_id == uid
    done = False
    while not done:
        _layers, done = session0.advance(
            max_layers=100, phase_for_pause=ForwardPhase.PREFILL_CONTINUE
        )
    assert session0.done

    # THE hazard-2 assertion: calling _advance_chunked_prefill_drive
    # for chunk 0's completion must leave chunk 1's session ALREADY
    # registered -- checked IMMEDIATELY after the call returns, no
    # intervening tick() or event-loop turn. Clear _active_prefill_
    # session first, exactly as tick()'s own real chunk-drive state
    # machine does right before returning PrefillAdvanceCompleted (see
    # test_grant_with_chunked_drive_does_not_admit_until_genuinely_
    # done's own comment for the full rationale).
    glue._active_prefill_session = None
    gen._advance_chunked_prefill_drive(uid, is_rank1=False)

    assert glue.has_active_prefill_session(), (
        "chunk 1's ResumablePrefillSession must be registered "
        "SYNCHRONOUSLY, inline, before _advance_chunked_prefill_drive "
        "returns -- Hazard 2's whole fix is that no tick() boundary "
        "may ever observe this rank without an active session while "
        "more real chunks remain"
    )
    active_id2, session1 = glue._active_prefill_session  # type: ignore[misc]
    assert active_id2 == uid
    assert session1 is not session0, (
        "chunk 1 must get a genuinely NEW session object, not a reused "
        "or stale reference to chunk 0's already-completed one"
    )
    assert not session1.done, (
        "chunk 1's session must be freshly constructed, not yet advanced at all"
    )

    # Drive chunk 1 and chunk 2 to completion the same way, confirming
    # the SAME inline-registration invariant holds at EVERY inner
    # chunk boundary, not just the first one.
    for expected_chunk in (1, 2):
        active_id, session = glue._active_prefill_session  # type: ignore[misc]
        assert active_id == uid
        done = False
        while not done:
            _layers, done = session.advance(
                max_layers=100, phase_for_pause=ForwardPhase.PREFILL_CONTINUE
            )
        glue._active_prefill_session = None
        gen._advance_chunked_prefill_drive(uid, is_rank1=False)
        if expected_chunk < 2:
            assert glue.has_active_prefill_session(), (
                f"chunk {expected_chunk + 1}'s session must be "
                f"registered inline immediately after chunk "
                f"{expected_chunk}'s completion is processed"
            )

    # After the FINAL chunk (index 2, n_chunks=3), the outer generator
    # reaches ("done", ...) -- no further session should be
    # registered, and the request must now be admitted.
    assert not glue.has_active_prefill_session()
    assert uid not in gen._deferred_prefill_by_uid
    assert glue.has_pending_admissions()


def test_cancel_refuses_a_uid_with_an_active_chunked_prefill_drive() -> None:
    """2026-08-07 fail-stop guard (Phase 2 live-wiring follow-up):
    ``ExoBatchGenerator.cancel()`` must raise ``GlueError`` for a uid
    whose ``_DeferredPrefill.drive`` is still active (i.e. registered
    with the glue, not yet reaching genuine completion) -- rather than
    silently popping ``_active_tasks``, which would leave
    ``glue._active_prefill_session`` permanently occupied by a request
    nothing will ever finish driving (a real correctness gap the
    design doc's 2026-08-06 entry explicitly flagged as NOT YET
    DESIGNED, not silently swept under a "just cancel it" shortcut).

    Verified load-bearing: reverting the guard (restoring
    ``cancel()``'s pre-2026-08-07 unconditional
    ``self._mlx_gen.remove(uids)``/``_active_tasks.pop`` body) makes
    this test FAIL LOUDLY -- no ``GlueError`` is raised, and the uid's
    entry silently disappears from every bookkeeping structure while
    ``glue.has_active_prefill_session()`` stays permanently ``True``
    for a session no caller will ever advance again.
    """
    gen, glue = _make_rank0_generator_with_glue()
    uid = _submit_one(gen)

    drive = _make_drive(n_chunks=2, n_layers_per_chunk=3)
    deferred = gen._deferred_prefill_by_uid[uid]
    deferred.try_start_chunked_prefill = lambda: drive

    grant = PrefillGrant(
        request_id=uid,
        cache_slot=deferred.cache_slot,
        n_prompt_tokens=1,
        single_request_fallback=False,
    )
    gen._run_deferred_prefill_for_grant(grant, is_rank1=False)
    assert deferred.drive is not None
    assert glue.has_active_prefill_session()

    with pytest.raises(GlueError, match="active ChunkedPrefillDrive"):
        gen.cancel([uid])

    # THE fail-loud contract: the guard must refuse BEFORE touching
    # any bookkeeping -- both the deferred entry and the glue's active
    # session must be completely untouched by the refused call.
    assert uid in gen._deferred_prefill_by_uid
    assert gen._deferred_prefill_by_uid[uid].drive is drive
    assert glue.has_active_prefill_session()
    assert uid in gen._active_tasks

    # A uid WITHOUT an active drive must remain cancellable, exactly
    # as before this guard existed -- the guard is scoped to the real
    # hazard only, never a blanket refusal.
    task_params = TextGenerationTaskParams(
        model=ModelId("test-model"),
        input=[],
        max_output_tokens=5,
        temperature=0.0,
        seed=0,
    )

    def _identity_all_sum(x: mx.array, **_: object) -> mx.array:
        return x

    with patch("mlx.core.distributed.all_sum", side_effect=_identity_all_sum):
        uid2 = gen.submit(task_params, "A second, unrelated request")
    assert gen._deferred_prefill_by_uid[uid2].drive is None
    gen.cancel([uid2])
    assert uid2 not in gen._active_tasks
