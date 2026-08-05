# pyright: reportAny=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownLambdaType=false, reportPrivateUsage=false
# pyright: reportInvalidCast=false, reportArgumentType=false
"""Full real-transport correctness test: the batched-decode session
lifecycle (test_pp_batched_decode_runtime.py) but with the
StepMessage/EvictMessage/EvictAckMessage control messages ACTUALLY
crossing pp_scheduler_wire.py's real mx.distributed.send/recv_like
encoding -- not handed to rank 1 as a shared in-process Python object
the way every other test in this session does.

This is the last missing link before a real 2-node cluster A/B:
proof that rank 1 can reconstruct the SAME StepMessage rank 0 decided
purely from wire bytes (pp_scheduler_wire.py), independent of any
Python object identity/reference rank 0 happened to hold -- exactly
what a genuinely separate OS process would have to do.

Rank 1's own token VALUES are architecturally irrelevant to
correctness here: BatchedMetaFramedPipelineFirstLayer (r != 0)
unconditionally discards the local embedding-layer output and
replaces it with the received activation tensor (see that class's
__call__) -- so rank 1 only needs a correctly-SHAPED placeholder
token array to drive its own (discarded) embedding pass, not the
real values rank 0 is decoding. This test uses zeros for rank 1's
local tokens to make that architectural fact explicit rather than
accidentally relying on it.
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
)
from exo.worker.engines.mlx.pp_batched_decode_runtime import (
    BatchedDecodeSession,
    RankOneMirrorSession,
)
from exo.worker.engines.mlx.pp_scheduler_protocol import EvictMessage
from exo.worker.engines.mlx.pp_scheduler_wire import (
    recv_evict_ack_message,
    recv_evict_message,
    recv_step_message,
    send_evict_ack_message,
    send_evict_message,
    send_step_message,
)

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


def _run_two_rank_step_over_real_wire(
    rank0_session: BatchedDecodeSession,
    rank1_session: RankOneMirrorSession,
    rank0_model: LlamaModel,
    rank1_model: LlamaModel,
    transport: SimPipelineTransport,
) -> dict[int, tuple[int, bool]]:
    """Drive ONE real batched decode step where the StepMessage
    genuinely crosses the wire via pp_scheduler_wire.py (send on
    rank 0's thread, recv on rank 1's thread) -- NOT handed directly
    as a shared Python object, the one thing every other test in this
    session's suite does NOT prove."""
    prepared = rank0_session.prepare_step()
    result: dict[str, Any] = {}
    group0 = cast(mx.distributed.Group, cast(Any, _RankGroup(0, 2)))
    group1 = cast(mx.distributed.Group, cast(Any, _RankGroup(1, 2)))

    def _rank0() -> None:
        _MLX_CALL_LOCK.acquire()
        try:
            mx.eval(mx.zeros(1))
            # REAL wire send -- the control message, not a shared
            # reference. dst=1 matches this harness's rank1.
            send_step_message(prepared.message, dst=1, group=group0)
            result["logits"] = rank0_session.run_forward(rank0_model, prepared)
        except BaseException as e:  # noqa: BLE001 - surface on join
            result["error0"] = e
        finally:
            _MLX_CALL_LOCK.release()

    def _rank1() -> None:
        _MLX_CALL_LOCK.acquire()
        try:
            mx.eval(mx.zeros(1))
            # REAL wire recv -- rank 1 reconstructs the StepMessage
            # purely from bytes off the (simulated) transport, with
            # zero access to rank 0's actual Python StepMessage object.
            received_message = recv_step_message(src=0, group=group1)
            n_active = len(received_message.entries)
            placeholder_tokens = mx.zeros((n_active, 1), dtype=mx.int32)
            mx.eval(placeholder_tokens)
            rank1_session.step(rank1_model, received_message, placeholder_tokens)
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
                "_run_two_rank_step_over_real_wire: simulated rank thread deadlocked"
            )
    if "error0" in result:
        raise result["error0"]
    if "error1" in result:
        raise result["error1"]
    return rank0_session.finish_step(prepared, result["logits"])


def test_full_lifecycle_over_real_wire_matches_serial_plain_forwards() -> None:
    """THE final Phase 1 local checkpoint: the SAME full-lifecycle
    scenario (admit both, batch decode, evict one via a REAL
    EvictMessage/EvictAckMessage wire round-trip, continue solo) as
    test_pp_batched_decode_runtime.py's own full-lifecycle test, but
    with EVERY control message (StepMessage for decode steps,
    EvictMessage/EvictAckMessage for eviction) genuinely crossing
    pp_scheduler_wire.py's real transport -- proving the whole stack
    (scheduler protocol + cache router + batched metaframe layers +
    decode-loop session + wire encoding) works when rank 1 has NO
    privileged access to rank 0's Python objects, matching a real
    2-process deployment's actual constraint.
    """
    vocab_size = _ARGS.vocab_size

    golden_model = _seeded_model(seed=123)
    prompt_a = _make_prompt(length=5, vocab_size=vocab_size, seed=1)
    prompt_b = _make_prompt(length=7, vocab_size=vocab_size, seed=2)

    golden_model_a = _seeded_model(seed=123)
    _copy_weights(golden_model, golden_model_a)
    golden_tokens_a = _plain_prefill_and_decode(
        golden_model_a, prompt_a, n_decode_steps=4
    )

    golden_model_b = _seeded_model(seed=123)
    _copy_weights(golden_model, golden_model_b)
    golden_tokens_b = _plain_prefill_and_decode(
        golden_model_b, prompt_b, n_decode_steps=6
    )

    rank0_model = _seeded_model(seed=123)
    rank1_model = _seeded_model(seed=123)
    _copy_weights(golden_model, rank0_model)
    _copy_weights(golden_model, rank1_model)

    cache_a_full, first_token_a = _single_request_prefilled_cache(
        golden_model, 123, prompt_a
    )
    cache_b_full, first_token_b = _single_request_prefilled_cache(
        golden_model, 123, prompt_b
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

    # Admission's StepMessage ALSO crosses the real wire, not just the
    # steady-state decode steps -- rank 1 never sees rank 0's admission
    # StepMessage as a shared Python object either.
    admit_group0 = cast(mx.distributed.Group, cast(Any, _RankGroup(0, 2)))
    admit_group1 = cast(mx.distributed.Group, cast(Any, _RankGroup(1, 2)))

    def _admit_over_wire(
        request_id: int,
        cache_slot: int,
        rank0_cache: list[Any],
        rank1_cache: list[Any],
        initial_token: int,
    ) -> None:
        result: dict[str, Any] = {}

        def _rank0() -> None:
            _MLX_CALL_LOCK.acquire()
            try:
                mx.eval(mx.zeros(1))
                msg = rank0_session.admit_request(
                    request_id=request_id,
                    cache_slot=cache_slot,
                    prefilled_cache=rank0_cache,
                    initial_token=initial_token,
                    sampler=_greedy_sampler,
                )
                send_step_message(msg, dst=1, group=admit_group0)
            except BaseException as e:  # noqa: BLE001
                result["error0"] = e
            finally:
                _MLX_CALL_LOCK.release()

        def _rank1() -> None:
            _MLX_CALL_LOCK.acquire()
            try:
                mx.eval(mx.zeros(1))
                received = recv_step_message(src=0, group=admit_group1)
                rank1_session.admit_request(
                    received, cache_slot=cache_slot, prefilled_cache=rank1_cache
                )
            except BaseException as e:  # noqa: BLE001
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
            t0.join(timeout=_RECV_TIMEOUT_SECONDS)
            t1.join(timeout=_RECV_TIMEOUT_SECONDS)
            if t0.is_alive() or t1.is_alive():
                raise RuntimeError("_admit_over_wire: simulated rank thread deadlocked")
        if "error0" in result:
            raise result["error0"]
        if "error1" in result:
            raise result["error1"]

    _admit_over_wire(1, 0, rank0_cache_a, rank1_cache_a, first_token_a)
    _admit_over_wire(2, 1, rank0_cache_b, rank1_cache_b, first_token_b)

    candidate_tokens_a: list[int] = [first_token_a]
    candidate_tokens_b: list[int] = [first_token_b]

    for _ in range(3):
        step_results = _run_two_rank_step_over_real_wire(
            rank0_session, rank1_session, rank0_model, rank1_model, transport
        )
        candidate_tokens_a.append(step_results[1][0])
        candidate_tokens_b.append(step_results[2][0])

    assert candidate_tokens_a == golden_tokens_a[: len(candidate_tokens_a)]
    assert candidate_tokens_b == golden_tokens_b[: len(candidate_tokens_b)]

    # Eviction over the REAL wire: EvictMessage sent rank0->rank1,
    # EvictAckMessage sent rank1->rank0, both via pp_scheduler_wire.py.
    final_cache_a, evict_info = rank0_session.evict_request(1)
    del final_cache_a
    evict_message = EvictMessage(
        step_id=evict_info.step_id,
        request_id=evict_info.request_id,
        cache_slot=evict_info.cache_slot,
    )

    evict_result: dict[str, Any] = {}
    evict_group0 = cast(mx.distributed.Group, cast(Any, _RankGroup(0, 2)))
    evict_group1 = cast(mx.distributed.Group, cast(Any, _RankGroup(1, 2)))

    def _evict_rank0() -> None:
        _MLX_CALL_LOCK.acquire()
        try:
            mx.eval(mx.zeros(1))
            send_evict_message(evict_message, dst=1, group=evict_group0)
            ack = recv_evict_ack_message(src=1, group=evict_group0)
            evict_result["ack"] = ack
        except BaseException as e:  # noqa: BLE001
            evict_result["error0"] = e
        finally:
            _MLX_CALL_LOCK.release()

    def _evict_rank1() -> None:
        _MLX_CALL_LOCK.acquire()
        try:
            mx.eval(mx.zeros(1))
            received_evict = recv_evict_message(src=0, group=evict_group1)
            rank1_session.evict(received_evict)
            rank1_session.release_slot(received_evict.cache_slot)
            from exo.worker.engines.mlx.pp_scheduler_protocol import EvictAckMessage

            ack_message = EvictAckMessage(
                step_id=received_evict.step_id,
                request_id=received_evict.request_id,
                cache_slot=received_evict.cache_slot,
            )
            send_evict_ack_message(ack_message, dst=0, group=evict_group1)
        except BaseException as e:  # noqa: BLE001
            evict_result["error1"] = e
        finally:
            _MLX_CALL_LOCK.release()

    with (
        patch("mlx.core.distributed.send", transport.send),
        patch("mlx.core.distributed.recv_like", transport.recv_like),
    ):
        t0 = threading.Thread(target=_evict_rank0)
        t1 = threading.Thread(target=_evict_rank1)
        t0.start()
        t1.start()
        t0.join(timeout=_RECV_TIMEOUT_SECONDS)
        t1.join(timeout=_RECV_TIMEOUT_SECONDS)
        if t0.is_alive() or t1.is_alive():
            raise RuntimeError(
                "eviction over real wire: simulated rank thread deadlocked"
            )
    if "error0" in evict_result:
        raise evict_result["error0"]
    if "error1" in evict_result:
        raise evict_result["error1"]
    assert evict_result["ack"].request_id == 1
    assert evict_result["ack"].cache_slot == evict_info.cache_slot

    rank0_session.on_evict_ack(request_id=1, cache_slot=evict_info.cache_slot)

    assert not rank0_session.driver.cache_router.is_occupied(0)
    assert rank0_session.has_active_requests()

    for _ in range(2):
        step_results = _run_two_rank_step_over_real_wire(
            rank0_session, rank1_session, rank0_model, rank1_model, transport
        )
        candidate_tokens_b.append(step_results[2][0])

    assert candidate_tokens_b == golden_tokens_b
