#!/usr/bin/env python3
# pyright: reportPrivateUsage=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownParameterType=false, reportMissingParameterType=false
# pyright: reportArgumentType=false, reportUnknownLambdaType=false
# pyright: reportCallIssue=false, reportAny=false
"""Worker script for the GENUINELY-INDEPENDENT 2-process chunk-drive
registration-ordering regression test (``test_pp_chunk_drive_subprocess.py``).

WHY THIS EXISTS (the gap it closes, mirroring
``_pp_admission_race_subprocess_worker.py``'s own precedent exactly):
every existing test for Hazard 1 (caller-assumed-completion) and
Hazard 2 (rank-registration skew) -- see
``test_batch_generate_chunked_prefill_live_wiring.py`` and
``test_pp_batched_decode_glue_chunk_drive.py`` -- drives BOTH ranks'
state from a SINGLE Python thread/process. Per this codebase's own
hard-won lesson (the original N=2 admission race was INVISIBLE to a
lockstep 2-process test and only reproduced once a genuinely-
independent-per-rank-event-loop subprocess harness was built), a
single-driver test cannot, by construction, exercise what happens when
each rank's decision to call ``tick()`` -- and therefore each rank's
own reactive ``register_prefill_session()`` call for the NEXT chunk
boundary -- happens on an independently-jittered schedule with no
cross-rank coordination on WHEN.

This worker reuses the SAME real production functions the in-process
tests already exercise (``Rank0/Rank1BatchedDecodeGlue.register_
prefill_session``/``tick()``/``enqueue_prefill``/``mark_prefill_
registered``) -- nothing about ``register_prefill_session``'s own
internal correctness is reimplemented here. What's inlined is the
exact CALL SEQUENCE ``ExoBatchGenerator._run_deferred_prefill_for_
grant``/``_advance_chunked_prefill_drive`` drive in production (chunk
0's session registered the instant this rank's own ``tick()`` returns
a real ``PrefillGrant``; each subsequent chunk's session registered
INLINE, synchronously, the instant THIS rank's own ``tick()`` reports
``PrefillAdvanceCompleted`` -- never scheduled, never deferred) --
matching the existing admission-race worker's own discipline of
inlining ``ExoBatchGenerator``'s known-correct call shape without a
real ``ExoBatchGenerator`` instance.

SCOPE BOUNDARY (deliberate, matches this session's established
precedent -- see ``test_pp_pipeline_parallel_prefill_session_integration.
py``'s own ``_InterruptibleLlamaWrapper``): the real DSv4
``_forward_steps`` split is not available on the currently-pinned
mlx-lm (see design doc's 2026-08-07 submodule-pin entry), so this
worker uses a SYNTHETIC one-real-forward-pass interruptible wrapper
around each rank's own REAL, metaframe-layer-patched half-model --
each "chunk" is ONE genuine forward pass through the REAL installed
``BatchedMetaFramedPipelineFirstLayer``/``LastLayer`` (same wire shape
production code emits), wrapped to yield exactly once before
"done" (mirrors the SAME session's own ``_InterruptibleLlamaWrapper``
precedent for the identical reason: proving the SESSION/GLUE STATE
MACHINE composition is correct is this test's job, not proving DSv4
specifically works). ``peer_prefill_layer_count=1`` on both ranks
follows directly from this: each rank's own synthetic session always
needs exactly ONE real ``advance()`` call to reach ``done=True``.

A SECOND, independent, ordinary (non-chunked) request is admitted
BEFORE and driven ALONGSIDE the chunked one -- proving ordinary
decode-step traffic keeps working correctly interleaved with a live
chunk-drive under real independent scheduling, not just in isolation.

A THIRD request is enqueued the INSTANT this rank's own tick() first
confirms the chunked request's drive is active (``glue.has_active_
prefill_session()`` reads ``True``) -- DETERMINISTICALLY, not
probabilistically (per a `consult` review: relying on random jitter
alone might never actually open the contention window this test
exists to stress) -- and this worker asserts, on every real ``tick()``
call for as long as the chunk-drive remains active, that the THIRD
request's real ``PrefillMessage``/grant is never delivered until the
chunk-drive genuinely, fully completes -- the priority-order guard
(``Rank0BatchedDecodeGlue.tick()``'s "no new prefill while
``_active_prefill_session`` is set" branch), now stressed under real
independent per-process scheduling pressure instead of a single-
threaded test's necessarily-limited interleavings.

Protocol with the parent: identical to the other subprocess workers --
write one JSON result file; the parent does all assertions.
"""

from __future__ import annotations

import json
import random
import sys
import time
import traceback
from typing import Iterator

import mlx.core as mx
import mlx.nn as nn
import mlx.utils

# Per-iteration jitter bound (seconds) -- same magnitude as the
# admission-race worker's own, for the same reason: small enough to
# keep the whole test fast, large enough that the two ranks' loops
# genuinely drift apart in wall time.
_MAX_JITTER_SECONDS = 0.01

# Real chunk count for the chunked request (D). Each chunk needs
# exactly ONE real tick() on each rank to complete (see module
# docstring's peer_prefill_layer_count=1 rationale) -- 3 chunks
# exercises the FIRST chunk-boundary (admission-grant-driven
# registration) plus TWO inner chunk-boundaries (advance-completion-
# driven registration), matching the design doc's own "2 chunks won't
# exercise an inner-boundary -> inner-boundary transition" caution.
_N_CHUNKS = 3

# Generous relative to the tiny model's real per-chunk work -- each
# chunk needs exactly 1 real tick() per rank, plus A's and E's own
# admission/decode overhead; this bounds the loop without depending on
# exact timing.
_MAX_ITERATIONS = 60

_D_REQUEST_ID = 2
_E_REQUEST_ID = 3


def _build_llama_model(seed: int):
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
    mx.random.seed(seed)
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
    return model, args.vocab_size


def _prefill(model, prompt):
    cache = model.make_cache()
    logits = model(prompt[None, :], cache=cache)
    mx.eval(logits)
    return cache, int(mx.argmax(logits[0, -1]).item())


class _OneChunkWrapper:
    """Synthetic ``_InterruptibleForward``-shaped wrapper (mirrors
    ``test_pp_pipeline_parallel_prefill_session_integration.py``'s own
    ``_InterruptibleLlamaWrapper`` precedent) around THIS rank's real,
    metaframe-layer-patched half-model -- one real forward pass (the
    SAME wire shape a real prefill chunk emits, through the REAL
    installed layers) per chunk, yielded once when interruptible
    (matching ``peer_prefill_layer_count=1``'s own arithmetic)."""

    def __init__(self, patched_model) -> None:
        self._model = patched_model

    def _forward_steps(
        self, inputs, cache=None, *, interruptible: bool = False
    ) -> Iterator[tuple[str, int | None, mx.array]]:
        out = self._model(inputs, cache=cache)
        mx.eval(out)
        if interruptible:
            yield ("layer", 0, out)
        yield ("done", None, out)


def main() -> int:  # noqa: C901 - one linear scenario, split would obscure it
    rank = int(sys.argv[1])
    out_path = sys.argv[2]
    seed = int(sys.argv[3])

    result: dict[str, object] = {"rank": rank}
    schedule_rng = random.Random(seed * 7919 + rank * 104729)
    trace: list[str] = []

    try:
        group = mx.distributed.init(backend="ring")
        if group.rank() != rank:
            raise RuntimeError(
                f"MLX ring group.rank()={group.rank()} does not match "
                f"expected rank={rank} from argv"
            )
        if group.size() != 2:
            raise RuntimeError(f"expected group.size()==2, got {group.size()}")

        model, vocab_size = _build_llama_model(seed)

        sys.path.insert(0, "src")
        from typing import Any, cast

        from exo.worker.engines.mlx.auto_parallel import (
            _set_layers,
            get_inner_model,
            get_layers,
        )
        from exo.worker.engines.mlx.pp_batched_decode_adapter import (
            BatchedDecodeResponseAdapter,
        )
        from exo.worker.engines.mlx.pp_batched_decode_glue import (
            GlueError,
            Rank0BatchedDecodeGlue,
            Rank1BatchedDecodeGlue,
        )
        from exo.worker.engines.mlx.pp_batched_decode_layers import (
            BatchedMetaFramedPipelineFirstLayer,
            BatchedMetaFramedPipelineLastLayer,
        )
        from exo.worker.engines.mlx.pp_batched_decode_runtime import (
            BatchedDecodeSession,
            RankOneMirrorSession,
        )
        from exo.worker.engines.mlx.pp_prefill_session import ResumablePrefillSession

        def greedy(logits: mx.array) -> mx.array:
            return mx.argmax(logits, axis=-1)

        inner = get_inner_model(cast(nn.Module, cast(Any, model)))
        layers = get_layers(inner)
        mid = len(layers) // 2

        # Request A: an ordinary, already-admitted decoding request --
        # prefilled BEFORE the layer swap, matching this session's
        # established discipline.
        mx.random.seed(seed + 1000)
        prompt_a = mx.random.randint(0, vocab_size, shape=(5,))
        cache_a, first_token_a = _prefill(model, prompt_a)

        # Request D: the CHUNKED request. Its own real per-chunk
        # "prompt" is arbitrary here (each chunk's forward pass is a
        # synthetic single-token step through the real patched
        # layers, not a real multi-chunk prompt split -- the outer
        # _pipeline_parallel_prefill_steps chunk loop is a SEPARATE,
        # already-tested concern; see module docstring's scope
        # boundary).
        mx.random.seed(seed + 2000)
        chunk_tokens_d = mx.random.randint(0, vocab_size, shape=(1,))[None, :]

        # Request E: an ordinary request enqueued the instant D's
        # chunk-drive is confirmed active -- proves the priority-order
        # guard holds under real independent scheduling.
        mx.random.seed(seed + 3000)
        prompt_e = mx.random.randint(0, vocab_size, shape=(4,))
        cache_e, first_token_e = _prefill(model, prompt_e)

        my_slice = slice(None, mid) if rank == 0 else slice(mid, None)
        my_layers = list(layers[my_slice])
        my_layers[0] = BatchedMetaFramedPipelineFirstLayer(
            my_layers[0], r=rank, group=group
        )
        my_layers[-1] = BatchedMetaFramedPipelineLastLayer(
            my_layers[-1], r=rank, s=2, group=group
        )
        _set_layers(cast(nn.Module, cast(Any, model)), my_layers)
        my_cache_a = cache_a[my_slice]
        my_cache_e = cache_e[my_slice]
        cache_d = model.make_cache()

        wrapped_model = _OneChunkWrapper(model)

        def new_chunk_session() -> ResumablePrefillSession:
            """A live, NOT-YET-ADVANCED session -- register_prefill_
            session's own documented contract (see that method's
            docstring: 'a live, not-yet-advanced ResumablePrefillSession
            for tick() to drive'). tick()'s own RANK0_LOCAL/reactive
            discipline drives this session's real advance() calls
            across SUBSEQUENT tick() calls -- this worker must NOT
            pre-drive it to completion before registering, or the
            very next tick() call raises PrefillSessionError trying
            to advance an already-done session (confirmed by a real
            crash caught running this worker for the first time --
            the earlier draft's drive_one_local_chunk_session() helper
            was a genuine bug in the TEST, not production code)."""
            return ResumablePrefillSession(
                inner_model=wrapped_model, inputs=chunk_tokens_d, cache=cache_d
            )

        d_chunk_index = 0
        d_done = False
        e_enqueued = False
        e_done = False
        tokens_a: list[int] = []
        tokens_e: list[int] = []

        if rank == 0:
            session = BatchedDecodeSession.new(max_concurrency=3)
            adapter = BatchedDecodeResponseAdapter(
                session=session, eos_ids=frozenset({999999})
            )
            glue = Rank0BatchedDecodeGlue(
                session=session,
                adapter=adapter,
                dst_rank=1,
                group=group,
                peer_prefill_layer_count=1,
            )
            glue.enqueue_admission(
                request_id=1,
                cache_slot=0,
                prefilled_cache=my_cache_a,
                initial_token=first_token_a,
                sampler=greedy,
                max_tokens=1000,
            )

            for iteration in range(_MAX_ITERATIONS):
                time.sleep(schedule_rng.uniform(0.0, _MAX_JITTER_SECONDS))

                if iteration == 2:
                    trace.append(f"it={iteration} enqueue_prefill_d")
                    glue.enqueue_prefill(
                        request_id=_D_REQUEST_ID,
                        cache_slot=1,
                        n_prompt_tokens=int(chunk_tokens_d.shape[1]),
                        single_request_fallback=False,
                    )

                trace.append(f"it={iteration} tick")
                responses, admitted_id, grant, advance_completed = glue.tick(model)
                del admitted_id
                if 1 in responses:
                    tokens_a.append(responses[1].token)
                if _E_REQUEST_ID in responses:
                    tokens_e.append(responses[_E_REQUEST_ID].token)

                if grant is not None and grant.request_id == _D_REQUEST_ID:
                    trace.append(f"it={iteration} grant_d chunk={d_chunk_index}")
                    d_session = new_chunk_session()
                    glue.register_prefill_session(
                        _D_REQUEST_ID, d_session, chunk_index=d_chunk_index
                    )
                    # DETERMINISTIC contention injection (per consult
                    # review): the instant D's drive is confirmed
                    # active, enqueue E -- never probabilistic, so
                    # every seed genuinely stresses the priority-order
                    # guard, not just some of them.
                    assert glue.has_active_prefill_session()
                    trace.append(f"it={iteration} enqueue_prefill_e")
                    glue.enqueue_prefill(
                        request_id=_E_REQUEST_ID,
                        cache_slot=2,
                        n_prompt_tokens=int(prompt_e.shape[0]),
                        single_request_fallback=False,
                    )
                    e_enqueued = True
                elif grant is not None and grant.request_id == _E_REQUEST_ID:
                    trace.append(f"it={iteration} grant_e")
                    if not d_done:
                        raise RuntimeError(
                            "PRIORITY-ORDER GUARD VIOLATION: rank 0's tick() "
                            "granted request E's prefill while request D's "
                            "chunk-drive was still active -- this is exactly "
                            "the cross-request interleaving hazard the "
                            "priority-order guard exists to prevent, now "
                            "reproduced under REAL independent per-process "
                            "scheduling"
                        )
                    glue.enqueue_admission(
                        request_id=_E_REQUEST_ID,
                        cache_slot=grant.cache_slot,
                        prefilled_cache=my_cache_e,
                        initial_token=first_token_e,
                        sampler=greedy,
                        max_tokens=1000,
                    )
                    e_done = True

                if (
                    e_enqueued
                    and not d_done
                    and glue.has_active_prefill_session()
                    and grant is not None
                    and grant.request_id == _E_REQUEST_ID
                ):
                    # While D's drive is active and E is queued, EVERY
                    # tick() must never grant E -- checked every
                    # iteration, not just at grant time, to catch a
                    # guard that fires only sometimes.
                    raise RuntimeError(
                        "PRIORITY-ORDER GUARD VIOLATION (redundant "
                        "check): grant.request_id == E while D active"
                    )

                if advance_completed is not None:
                    if advance_completed.request_id != _D_REQUEST_ID:
                        raise RuntimeError(
                            f"unexpected PrefillAdvanceCompleted for "
                            f"request_id={advance_completed.request_id}, "
                            f"expected {_D_REQUEST_ID}"
                        )
                    d_chunk_index += 1
                    trace.append(
                        f"it={iteration} advance_completed chunk={d_chunk_index}"
                    )
                    if d_chunk_index < _N_CHUNKS:
                        # HAZARD 2 fix under test: register the NEXT
                        # chunk's session SYNCHRONOUSLY, INLINE, same
                        # iteration -- never deferred to a later loop
                        # turn or a later tick() call.
                        next_session = new_chunk_session()
                        glue.register_prefill_session(
                            _D_REQUEST_ID, next_session, chunk_index=d_chunk_index
                        )
                    else:
                        d_done = True
                        glue.enqueue_admission(
                            request_id=_D_REQUEST_ID,
                            cache_slot=1,
                            prefilled_cache=cache_d,
                            initial_token=0,
                            sampler=greedy,
                            max_tokens=1000,
                        )

                if d_done and e_done and iteration > 10:
                    break

            if not d_done:
                raise RuntimeError(
                    f"rank=0: request D's chunk-drive never completed within "
                    f"{_MAX_ITERATIONS} iterations -- increase "
                    f"_MAX_ITERATIONS rather than treating this as a pass"
                )
            if not e_done:
                raise RuntimeError(
                    "rank=0: request E was never admitted -- the priority-"
                    "order guard scenario did not actually run to "
                    "completion"
                )

            result["tokens_a"] = tokens_a
            result["tokens_e"] = tokens_e
        else:
            session = RankOneMirrorSession.new(max_concurrency=3)
            glue = Rank1BatchedDecodeGlue(session=session, src_rank=0, group=group)
            glue.stage_local_cache(
                request_id=1, cache_slot=0, prefilled_cache=my_cache_a
            )

            d_registered = False
            e_marked = False
            d_cache_slot: int | None = None

            for iteration in range(_MAX_ITERATIONS):
                time.sleep(schedule_rng.uniform(0.0, _MAX_JITTER_SECONDS))

                if iteration == 2 and not d_registered:
                    trace.append(f"it={iteration} mark_prefill_registered_d")
                    glue.mark_prefill_registered(_D_REQUEST_ID)
                    d_registered = True

                trace.append(f"it={iteration} tick")
                try:
                    grant, evicted_request_id, advance_completed = glue.tick(model)
                except GlueError as ge:
                    raise RuntimeError(
                        f"rank=1 GlueError at it={iteration}: {ge}"
                    ) from ge
                del evicted_request_id

                if grant is not None and grant.request_id == _D_REQUEST_ID:
                    trace.append(f"it={iteration} grant_d chunk={d_chunk_index}")
                    d_cache_slot = grant.cache_slot
                    d_session = new_chunk_session()
                    glue.register_prefill_session(_D_REQUEST_ID, d_session)
                    if not e_marked:
                        trace.append(f"it={iteration} mark_prefill_registered_e")
                        glue.mark_prefill_registered(_E_REQUEST_ID)
                        e_marked = True
                elif grant is not None and grant.request_id == _E_REQUEST_ID:
                    trace.append(f"it={iteration} grant_e")
                    glue.stage_local_cache(
                        request_id=_E_REQUEST_ID,
                        cache_slot=grant.cache_slot,
                        prefilled_cache=my_cache_e,
                    )
                    e_done = True

                if advance_completed is not None:
                    if advance_completed.request_id != _D_REQUEST_ID:
                        raise RuntimeError(
                            f"unexpected PrefillAdvanceCompleted for "
                            f"request_id={advance_completed.request_id}, "
                            f"expected {_D_REQUEST_ID}"
                        )
                    d_chunk_index += 1
                    trace.append(
                        f"it={iteration} advance_completed chunk={d_chunk_index}"
                    )
                    if d_chunk_index < _N_CHUNKS:
                        next_session = new_chunk_session()
                        glue.register_prefill_session(_D_REQUEST_ID, next_session)
                    else:
                        d_done = True
                        assert d_cache_slot is not None
                        # Mirrors production's shared _admit_completed_prefill
                        # tail exactly (batch_generate.py): rank 1's real
                        # equivalent of enqueue_admission for a chunk-drive
                        # request that just genuinely completed is
                        # stage_local_cache -- confirmed missing in an
                        # earlier draft of this worker (a real TEST bug,
                        # caught by rank1's own fail-loud GlueError:
                        # "no staged local prefilled cache" -- production's
                        # existing guard did its job here, this was never a
                        # production bug).
                        glue.stage_local_cache(
                            request_id=_D_REQUEST_ID,
                            cache_slot=d_cache_slot,
                            prefilled_cache=cache_d,
                        )

                if d_done and e_done and iteration > 10:
                    break

            if not d_done:
                raise RuntimeError(
                    f"rank=1: request D's chunk-drive never completed within "
                    f"{_MAX_ITERATIONS} iterations"
                )
            if not e_done:
                raise RuntimeError("rank=1: request E was never staged")

        result["trace"] = trace
        result["ok"] = True
    except BaseException as e:  # noqa: BLE001 - report, don't crash silently
        result["ok"] = False
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
        result["trace"] = trace

    with open(out_path, "w") as f:
        json.dump(result, f)

    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
