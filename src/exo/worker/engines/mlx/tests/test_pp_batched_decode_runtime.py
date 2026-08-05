# pyright: reportAny=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownLambdaType=false, reportPrivateUsage=false
# pyright: reportInvalidCast=false, reportArgumentType=false
"""Tests for pp_batched_decode_runtime.py -- the real decode-loop
SESSION class that was still missing after this session's earlier
work: BatchedDecodeSession/RankOneMirrorSession actually CALL
model(...) with sampling and per-request generation state, rather
than the test harness doing that work itself around a lower-level
driver (test_pp_batched_decode_driver_full_stack.py's own pattern).

This file re-runs the SAME full-lifecycle scenario (admit both, batch
decode, evict one, continue solo) but through the actual session
class's own three-phase API (prepare_step/run_forward/finish_step) --
proving the session class itself (not just its building blocks)
produces correct results end-to-end against the same serial-plain-
forward golden reference this fork uses throughout Phase 0/1.

The three-phase split (rather than one step() call) exists SPECIFICALLY
so a 2-rank caller can hand rank 1 the StepMessage BEFORE either rank's
forward pass starts -- matching the real transport's actual ordering
constraint and avoiding a race in this simulated-2-thread harness
(see pp_batched_decode_runtime.py's own prepare_step() docstring).
"""

from __future__ import annotations

import threading
from typing import Any, cast
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
import mlx.utils
from mlx_lm.models.llama import Model as LlamaModel
from mlx_lm.models.llama import ModelArgs

from exo.worker.engines.mlx.auto_parallel import (
    _set_layers,
    get_inner_model,
    get_layers,
)
from exo.worker.engines.mlx.pp_batched_correctness import (
    _MLX_CALL_LOCK,
    SimPipelineTransport,
    _RankGroup,
)
from exo.worker.engines.mlx.pp_batched_decode_layers import (
    BatchedMetaFramedPipelineFirstLayer,
    BatchedMetaFramedPipelineLastLayer,
    batch_step_scope,
)
from exo.worker.engines.mlx.pp_batched_decode_runtime import (
    BatchedDecodeSession,
    RankOneMirrorSession,
)
from exo.worker.engines.mlx.pp_scheduler_protocol import EvictMessage

_RECV_TIMEOUT_SECONDS = 30.0

_ARGS = ModelArgs(
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


def _greedy_sampler(logits: mx.array) -> mx.array:
    """The simplest real sampler: greedy argmax, matching this whole
    fork's temp=0 golden-reference methodology."""
    return mx.argmax(logits, axis=-1)


def _seeded_model(seed: int) -> LlamaModel:
    mx.random.seed(seed)
    model = LlamaModel(_ARGS)
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


def _copy_weights(src: LlamaModel, dst: LlamaModel) -> None:
    dst.update(src.parameters())
    mx.eval(dst.parameters())


def _make_prompt(length: int, vocab_size: int, seed: int) -> mx.array:
    mx.random.seed(seed)
    return mx.random.randint(0, vocab_size, shape=(length,))


def _plain_prefill_and_decode(
    model: LlamaModel, prompt: mx.array, n_decode_steps: int
) -> list[int]:
    cache = model.make_cache()
    logits = model(prompt[None, :], cache=cache)
    mx.eval(logits)
    next_token = int(mx.argmax(logits[0, -1]).item())
    tokens = [next_token]
    for _ in range(n_decode_steps - 1):
        logits = model(mx.array([[next_token]]), cache=cache)
        mx.eval(logits)
        next_token = int(mx.argmax(logits[0, -1]).item())
        tokens.append(next_token)
    return tokens


def _single_request_prefilled_cache(
    golden_model: LlamaModel, seed: int, prompt: mx.array
) -> tuple[list[Any], int]:
    """Returns (cache, first_decode_token) -- the token the prefill
    forward pass itself produced, matching how a real caller (having
    just run the existing serial PP prefill path) would hand this
    session class an ALREADY-PREFILLED cache plus its first token."""
    prefill_model = _seeded_model(seed)
    _copy_weights(golden_model, prefill_model)
    cache = prefill_model.make_cache()
    logits = prefill_model(prompt[None, :], cache=cache)
    mx.eval(logits)
    first_token = int(mx.argmax(logits[0, -1]).item())
    return cache, first_token


def _build_two_rank_batched_split(
    rank0_model: LlamaModel,
    rank1_model: LlamaModel,
) -> tuple[LlamaModel, LlamaModel, SimPipelineTransport]:
    inner0 = get_inner_model(cast(nn.Module, cast(Any, rank0_model)))
    inner1 = get_inner_model(cast(nn.Module, cast(Any, rank1_model)))
    layers0 = get_layers(inner0)
    layers1 = get_layers(inner1)

    n_layers = len(layers0)
    mid = n_layers // 2

    transport = SimPipelineTransport()
    group0 = cast(mx.distributed.Group, cast(Any, _RankGroup(0, 2)))
    group1 = cast(mx.distributed.Group, cast(Any, _RankGroup(1, 2)))

    r0_layers = list(layers0[:mid])
    r1_layers = list(layers1[mid:])

    r0_layers[0] = BatchedMetaFramedPipelineFirstLayer(r0_layers[0], r=0, group=group0)
    r0_layers[-1] = BatchedMetaFramedPipelineLastLayer(
        r0_layers[-1], r=0, s=2, group=group0
    )
    r1_layers[0] = BatchedMetaFramedPipelineFirstLayer(r1_layers[0], r=1, group=group1)
    r1_layers[-1] = BatchedMetaFramedPipelineLastLayer(
        r1_layers[-1], r=1, s=2, group=group1
    )

    _set_layers(cast(nn.Module, cast(Any, rank0_model)), r0_layers)
    _set_layers(cast(nn.Module, cast(Any, rank1_model)), r1_layers)

    return rank0_model, rank1_model, transport


def _run_two_rank_session_step(
    rank0_session: BatchedDecodeSession,
    rank1_session: RankOneMirrorSession,
    rank0_model: LlamaModel,
    rank1_model: LlamaModel,
    transport: SimPipelineTransport,
) -> dict[int, tuple[int, bool]]:
    """Drive ONE real batched decode step through BOTH real session
    objects, using the three-phase split (prepare_step/run_forward/
    finish_step) specifically so rank 1 receives the StepMessage
    BEFORE either rank's forward pass starts -- no race, no polling,
    matching the real transport's actual ordering constraint.
    """
    prepared = rank0_session.prepare_step()

    result: dict[str, Any] = {}

    def _rank0() -> None:
        _MLX_CALL_LOCK.acquire()
        try:
            mx.eval(mx.zeros(1))
            result["logits"] = rank0_session.run_forward(rank0_model, prepared)
        except BaseException as e:  # noqa: BLE001 - surface on join
            result["error0"] = e
        finally:
            _MLX_CALL_LOCK.release()

    def _rank1() -> None:
        _MLX_CALL_LOCK.acquire()
        try:
            mx.eval(mx.zeros(1))
            rank1_session.step(rank1_model, prepared.message, prepared.tokens)
        except BaseException as e:  # noqa: BLE001 - surface on join
            result["error1"] = e
        finally:
            _MLX_CALL_LOCK.release()

    with (
        patch("mlx.core.distributed.send", transport.send),
        patch("mlx.core.distributed.recv_like", transport.recv_like),
    ):
        t0 = threading.Thread(target=_rank0)
        t1 = threading.Thread(target=_rank1)
        t0.start()
        t1.start()
        t0.join(timeout=_RECV_TIMEOUT_SECONDS + 5)
        t1.join(timeout=_RECV_TIMEOUT_SECONDS + 5)
        if t0.is_alive() or t1.is_alive():
            raise RuntimeError(
                "_run_two_rank_session_step: simulated rank thread deadlocked"
            )
    if "error0" in result:
        raise result["error0"]
    if "error1" in result:
        raise result["error1"]
    return rank0_session.finish_step(prepared, result["logits"])


def test_batched_decode_session_full_lifecycle_matches_serial_plain_forwards() -> None:
    """Re-runs the established full-lifecycle scenario (admit both,
    batch decode, evict one, continue solo) but through the ACTUAL
    session class API (prepare_step/run_forward/finish_step,
    evict_request, on_evict_ack) rather than a test harness manually
    calling model(...) and doing argmax itself -- proving the session
    class (sampling, per-request generation state, eviction
    bookkeeping) is correct end to end, not just its lower-level
    building blocks (already verified by
    test_pp_batched_decode_driver_full_stack.py).
    """
    vocab_size = _ARGS.vocab_size

    golden_model = _seeded_model(seed=99)
    prompt_a = _make_prompt(length=5, vocab_size=vocab_size, seed=1)
    prompt_b = _make_prompt(length=8, vocab_size=vocab_size, seed=2)

    golden_model_a = _seeded_model(seed=99)
    _copy_weights(golden_model, golden_model_a)
    golden_tokens_a = _plain_prefill_and_decode(
        golden_model_a, prompt_a, n_decode_steps=4
    )

    golden_model_b = _seeded_model(seed=99)
    _copy_weights(golden_model, golden_model_b)
    golden_tokens_b = _plain_prefill_and_decode(
        golden_model_b, prompt_b, n_decode_steps=6
    )

    rank0_model = _seeded_model(seed=99)
    rank1_model = _seeded_model(seed=99)
    _copy_weights(golden_model, rank0_model)
    _copy_weights(golden_model, rank1_model)

    cache_a_full, first_token_a = _single_request_prefilled_cache(
        golden_model, 99, prompt_a
    )
    cache_b_full, first_token_b = _single_request_prefilled_cache(
        golden_model, 99, prompt_b
    )
    assert first_token_a == golden_tokens_a[0]
    assert first_token_b == golden_tokens_b[0]

    n_layers = len(cache_a_full)
    mid = n_layers // 2
    rank0_cache_a = cache_a_full[:mid]
    rank1_cache_a = cache_a_full[mid:]
    rank0_cache_b = cache_b_full[:mid]
    rank1_cache_b = cache_b_full[mid:]

    rank0_model, rank1_model, transport = _build_two_rank_batched_split(
        rank0_model, rank1_model
    )

    rank0_session = BatchedDecodeSession.new(max_concurrency=2)
    rank1_session = RankOneMirrorSession.new(max_concurrency=2)

    msg_a = rank0_session.admit_request(
        request_id=1,
        cache_slot=0,
        prefilled_cache=rank0_cache_a,
        initial_token=first_token_a,
        sampler=_greedy_sampler,
    )
    rank1_session.admit_request(msg_a, cache_slot=0, prefilled_cache=rank1_cache_a)

    msg_b = rank0_session.admit_request(
        request_id=2,
        cache_slot=1,
        prefilled_cache=rank0_cache_b,
        initial_token=first_token_b,
        sampler=_greedy_sampler,
    )
    rank1_session.admit_request(msg_b, cache_slot=1, prefilled_cache=rank1_cache_b)

    candidate_tokens_a: list[int] = [first_token_a]
    candidate_tokens_b: list[int] = [first_token_b]

    for _ in range(3):
        step_results = _run_two_rank_session_step(
            rank0_session, rank1_session, rank0_model, rank1_model, transport
        )
        candidate_tokens_a.append(step_results[1][0])
        candidate_tokens_b.append(step_results[2][0])

    assert candidate_tokens_a == golden_tokens_a[: len(candidate_tokens_a)]
    assert candidate_tokens_b == golden_tokens_b[: len(candidate_tokens_b)]

    # Evict A: DRAINING/ack cycle through the real session API.
    final_cache_a, evict_info = rank0_session.evict_request(1)
    del final_cache_a  # not asserted on further -- production would save it
    evict_message = EvictMessage(
        step_id=evict_info.step_id,
        request_id=evict_info.request_id,
        cache_slot=evict_info.cache_slot,
    )
    rank1_session.evict(evict_message)
    rank1_session.release_slot(evict_info.cache_slot)
    rank0_session.on_evict_ack(request_id=1, cache_slot=evict_info.cache_slot)

    assert not rank0_session.driver.cache_router.is_occupied(0)
    assert rank0_session.has_active_requests()

    for _ in range(2):
        step_results = _run_two_rank_session_step(
            rank0_session, rank1_session, rank0_model, rank1_model, transport
        )
        candidate_tokens_b.append(step_results[2][0])

    assert candidate_tokens_b == golden_tokens_b


def test_single_rank_step_convenience_wrapper_matches_plain_forward() -> None:
    """The single-call ``step()`` convenience wrapper (prepare_step +
    run_forward + finish_step in one call, no second rank involved --
    matches an unsharded/single-process smoke-test use case) produces
    correct greedy tokens against a plain forward reference."""
    vocab_size = _ARGS.vocab_size
    golden_model = _seeded_model(seed=55)
    prompt = _make_prompt(length=4, vocab_size=vocab_size, seed=3)

    golden_tokens = _plain_prefill_and_decode(golden_model, prompt, n_decode_steps=3)

    model = _seeded_model(seed=55)
    _copy_weights(golden_model, model)
    cache, first_token = _single_request_prefilled_cache(golden_model, 55, prompt)
    assert first_token == golden_tokens[0]

    session = BatchedDecodeSession.new(max_concurrency=2)
    session.admit_request(
        request_id=1,
        cache_slot=0,
        prefilled_cache=cache,
        initial_token=first_token,
        sampler=_greedy_sampler,
    )

    candidate = [first_token]
    for _ in range(2):
        results = session.step(model)
        candidate.append(results[1][0])

    assert candidate == golden_tokens


def test_batch_step_scope_active_during_run_forward_matches_prepared_ctx() -> None:
    """Sanity check that run_forward's batch_step_scope really is
    active with the SAME request_uids prepare_step computed -- a
    regression guard for the prepare/run/finish split itself, in
    isolation from the full 2-rank machinery above."""
    vocab_size = _ARGS.vocab_size
    model = _seeded_model(seed=7)
    prompt = _make_prompt(length=3, vocab_size=vocab_size, seed=4)
    cache = model.make_cache()
    logits = model(prompt[None, :], cache=cache)
    mx.eval(logits)
    first_token = int(mx.argmax(logits[0, -1]).item())

    session = BatchedDecodeSession.new(max_concurrency=2)
    session.admit_request(
        request_id=42,
        cache_slot=0,
        prefilled_cache=cache,
        initial_token=first_token,
        sampler=_greedy_sampler,
    )
    prepared = session.prepare_step()
    assert prepared.active_ids == (42,)

    observed: dict[str, Any] = {}
    original_call = type(model).__call__

    def _spying_call(self: Any, *args: Any, **kwargs: Any) -> Any:
        from exo.worker.engines.mlx.pp_batched_decode_layers import (
            _require_batch_step_context,
        )

        observed["ctx"] = _require_batch_step_context()
        return original_call(self, *args, **kwargs)

    with patch.object(type(model), "__call__", _spying_call):
        session.run_forward(model, prepared)

    assert observed["ctx"] is not None
    assert observed["ctx"].request_uids == (42,)
    with batch_step_scope(prepared.ctx):
        pass  # exercised import path only; scope already validated above
