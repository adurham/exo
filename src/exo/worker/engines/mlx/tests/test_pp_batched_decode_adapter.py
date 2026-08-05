# pyright: reportAny=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportArgumentType=false, reportUnknownLambdaType=false
# pyright: reportPrivateUsage=false
"""Tests for pp_batched_decode_adapter.py -- the thin translation
layer between BatchedDecodeSession's raw {request_id: (token,
is_done)} output and real GenerationBatch.Response-shaped
finish_reason classification (mirrors _step_pp_spec's own EOS/
max_tokens decision logic, never reimplements it from scratch)."""

from __future__ import annotations

import mlx.core as mx
import mlx.utils
import pytest

from exo.worker.engines.mlx.pp_batched_decode_adapter import (
    BatchedDecodeAdapterError,
    BatchedDecodeResponseAdapter,
)
from exo.worker.engines.mlx.pp_batched_decode_runtime import BatchedDecodeSession


def _build_llama_model_and_cache(seed: int):
    from mlx_lm.models.llama import Model as LlamaModel
    from mlx_lm.models.llama import ModelArgs

    args = ModelArgs(
        model_type="llama",
        hidden_size=64,
        num_hidden_layers=2,
        intermediate_size=128,
        num_attention_heads=2,
        num_key_value_heads=1,
        rms_norm_eps=1e-6,
        vocab_size=256,
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

    mx.random.seed(seed + 1000)
    prompt = mx.random.randint(0, args.vocab_size, shape=(4,))
    cache = model.make_cache()
    logits = model(prompt[None, :], cache=cache)
    mx.eval(logits)
    first_token = int(mx.argmax(logits[0, -1]).item())
    return model, cache, first_token


def _greedy(logits: mx.array) -> mx.array:
    return mx.argmax(logits, axis=-1)


def test_admit_classifies_first_token_as_still_generating() -> None:
    _model, cache, first_token = _build_llama_model_and_cache(seed=1)
    session = BatchedDecodeSession.new(max_concurrency=2)
    adapter = BatchedDecodeResponseAdapter(session=session, eos_ids=frozenset({999}))

    _message, admit_response = adapter.admit(
        request_id=1,
        cache_slot=0,
        prefilled_cache=cache,
        initial_token=first_token,
        sampler=_greedy,
        max_tokens=50,
    )

    assert admit_response.token == first_token
    assert admit_response.finish_reason is None


def test_admit_classifies_eos_first_token_as_stop() -> None:
    _model, cache, _first_token = _build_llama_model_and_cache(seed=2)
    session = BatchedDecodeSession.new(max_concurrency=2)
    # Force the "first token happens to be EOS" edge case directly --
    # mirrors _step_pp_spec's own first-token EOS check, which must
    # fire even when max_tokens is nowhere close to reached.
    eos_token = 42
    adapter = BatchedDecodeResponseAdapter(
        session=session, eos_ids=frozenset({eos_token})
    )

    _message, admit_response = adapter.admit(
        request_id=1,
        cache_slot=0,
        prefilled_cache=cache,
        initial_token=eos_token,
        sampler=_greedy,
        max_tokens=50,
    )

    assert admit_response.finish_reason == "stop"


def test_admit_classifies_max_tokens_one_as_length() -> None:
    """A request submitted with max_tokens=1 must be classified as
    "length"-finished on its VERY FIRST token (the prefill token IS
    its only allowed token) -- mirrors _step_pp_spec's own
    tokens-generated>=max_tokens check, which _step_pp_spec applies
    starting from the first token too, not just steady-state steps."""
    _model, cache, first_token = _build_llama_model_and_cache(seed=3)
    session = BatchedDecodeSession.new(max_concurrency=2)
    adapter = BatchedDecodeResponseAdapter(session=session, eos_ids=frozenset({999}))

    _message, admit_response = adapter.admit(
        request_id=1,
        cache_slot=0,
        prefilled_cache=cache,
        initial_token=first_token,
        sampler=_greedy,
        max_tokens=1,
    )

    assert admit_response.finish_reason == "length"


def test_classify_step_results_tracks_tokens_generated_across_steps() -> None:
    """A real 3-step decode: step 1-2 still generating, step 3 hits
    max_tokens=3 exactly and is classified "length" -- proves the
    adapter's own _tokens_generated counter accumulates correctly
    across MULTIPLE classify_step_results calls (not just admit's
    first-token special case), matching real steady-state usage."""
    model, cache, first_token = _build_llama_model_and_cache(seed=4)
    session = BatchedDecodeSession.new(max_concurrency=2)
    adapter = BatchedDecodeResponseAdapter(session=session, eos_ids=frozenset({999}))

    _message, admit_response = adapter.admit(
        request_id=1,
        cache_slot=0,
        prefilled_cache=cache,
        initial_token=first_token,
        sampler=_greedy,
        max_tokens=3,
    )
    assert admit_response.finish_reason is None

    for step_num in (2, 3):
        prepared = session.prepare_step()
        logits = session.run_forward(model, prepared)
        step_results = session.finish_step(prepared, logits)
        classified = adapter.classify_step_results(step_results)

        assert set(classified.keys()) == {1}
        if step_num < 3:
            assert classified[1].finish_reason is None
        else:
            assert classified[1].finish_reason == "length"


def test_classify_step_results_detects_real_eos_token_mid_generation() -> None:
    """Uses a real forward pass's ACTUAL sampled token as the EOS id
    (rather than an arbitrary constant that never naturally occurs)
    to prove the membership test operates on the real token value
    finish_step produces, not a mocked/injected one. Registers the
    adapter's bookkeeping directly (bypassing admit()) since this
    test's focus is classify_step_results in isolation, not the
    admit-then-step lifecycle already covered by the test above."""
    model, cache, first_token = _build_llama_model_and_cache(seed=5)
    session = BatchedDecodeSession.new(max_concurrency=2)
    session.admit_request(
        request_id=1,
        cache_slot=0,
        prefilled_cache=cache,
        initial_token=first_token,
        sampler=_greedy,
    )
    prepared = session.prepare_step()
    logits = session.run_forward(model, prepared)
    step_results = session.finish_step(prepared, logits)
    real_step2_token = step_results[1][0]

    adapter = BatchedDecodeResponseAdapter(
        session=session, eos_ids=frozenset({real_step2_token})
    )
    adapter._tokens_generated[1] = 1
    adapter._max_tokens[1] = 50

    classified = adapter.classify_step_results(step_results)
    assert classified[1].token == real_step2_token
    assert classified[1].finish_reason == "stop"


def test_classify_step_results_raises_for_unadmitted_request() -> None:
    """A request_id classify_step_results has never seen via admit()
    must fail LOUDLY (BatchedDecodeAdapterError), never silently
    guess a finish_reason -- matches this session's established
    fail-stop discipline for unknown-state cases."""
    session = BatchedDecodeSession.new(max_concurrency=2)
    adapter = BatchedDecodeResponseAdapter(session=session, eos_ids=frozenset({999}))

    with pytest.raises(BatchedDecodeAdapterError, match="never admitted"):
        adapter.classify_step_results({7: (5, False)})


def test_forget_removes_bookkeeping_and_id_becomes_unclassifiable_again() -> None:
    _model, cache, first_token = _build_llama_model_and_cache(seed=6)
    session = BatchedDecodeSession.new(max_concurrency=2)
    adapter = BatchedDecodeResponseAdapter(session=session, eos_ids=frozenset({999}))
    adapter.admit(
        request_id=1,
        cache_slot=0,
        prefilled_cache=cache,
        initial_token=first_token,
        sampler=_greedy,
        max_tokens=50,
    )
    assert 1 in adapter._tokens_generated

    adapter.forget(1)

    assert 1 not in adapter._tokens_generated
    assert 1 not in adapter._max_tokens
    with pytest.raises(BatchedDecodeAdapterError):
        adapter.classify_step_results({1: (5, False)})
