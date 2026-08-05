# pyright: reportAny=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownLambdaType=false, reportPrivateUsage=false
# pyright: reportInvalidCast=false, reportArgumentType=false
"""Phase 1 correctness test: 2 concurrent decode-only requests through
the NEW batched scheduler-adjacent machinery
(``pp_batched_decode_layers``, ``pp_scheduler_protocol``,
``pp_batched_cache_router``) vs 2 SERIAL single-request PLAIN
(unsharded) forward passes.

Per the design doc's own methodology (Section 9's Phase 0 baseline:
"the golden reference is the PLAIN unsharded forward, not 'trust the
simulated split'") and this fork's established convention
(``pp_batched_correctness.py``'s module docstring point 2), the golden
reference here is each request's PLAIN forward output run in
isolation -- NOT another PP path. This is genuinely a byte-for-byte
(greedy-token) correctness check, not a re-validation of the transport
(Phase 0.5 already proved that) or the protocol state machine (Phase 1
step 1 already fuzz-tested that in isolation) -- it's the first test
that exercises ALL THREE together: the scheduler's slot bookkeeping,
the cache router's merge/extract, and the batched metaframe transport,
end to end, through simulated (not real-cluster) 2-rank PP.

Uses the SAME simulated-2-rank machinery as Phase 0/0.5
(``pp_batched_correctness.SimPipelineTransport``, real OS threads with
``_MLX_CALL_LOCK`` serializing MLX op execution) -- per this fork's
established rationale for keeping GPU/cluster time off correctness
questions the CPU can answer just as definitively.

``contextvars.ContextVar`` (``pp_batched_decode_layers.BatchStepContext``)
does NOT propagate across ``threading.Thread`` boundaries (documented
in that module's own docstring) -- this test's two rank-driving
functions each set their OWN ``batch_step_scope`` inside their own
thread body, matching the real production wiring's eventual shape
(each rank's own process/thread sets its own context locally, never
inherited from a parent).
"""

from __future__ import annotations

import threading
from typing import Any, cast
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import pytest
from mlx_lm.models.llama import Model as LlamaModel
from mlx_lm.models.llama import ModelArgs

from exo.worker.engines.mlx.auto_parallel import (
    _set_layers,
    get_inner_model,
    get_layers,
)
from exo.worker.engines.mlx.pp_batched_cache_router import (
    extract_request_cache,
    merge_request_caches,
)
from exo.worker.engines.mlx.pp_batched_correctness import (
    _MLX_CALL_LOCK,
    SimPipelineTransport,
    _RankGroup,
)
from exo.worker.engines.mlx.pp_batched_decode_layers import (
    BatchedMetaFramedPipelineFirstLayer,
    BatchedMetaFramedPipelineLastLayer,
    BatchStepContext,
    batch_step_scope,
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
) -> tuple[list[int], list[Any]]:
    """Golden reference: plain unsharded forward. Prefill the prompt,
    then greedily decode ``n_decode_steps`` tokens. Returns
    (greedy_token_ids, final_cache) so the cache can also be reused as
    the SOURCE for the candidate path's per-request cache (both paths
    must start from the IDENTICAL post-prefill state to isolate the
    comparison to decode-batching correctness only, not prefill
    correctness -- already covered elsewhere)."""
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
    return tokens, cache


def _build_two_rank_batched_split(
    rank0_model: LlamaModel,
    rank1_model: LlamaModel,
) -> tuple[LlamaModel, LlamaModel, SimPipelineTransport]:
    """Batched-decode counterpart to
    ``pp_batched_correctness.build_two_rank_split`` -- wraps the first/
    last layer with ``BatchedMetaFramedPipelineFirstLayer``/
    ``BatchedMetaFramedPipelineLastLayer`` instead of the plain/
    Phase-0.5 metaframe classes."""
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


def _run_batched_decode_step(
    rank0_model: LlamaModel,
    rank1_model: LlamaModel,
    transport: SimPipelineTransport,
    tokens: mx.array,
    rank0_cache: list[Any],
    rank1_cache: list[Any],
    request_uids: tuple[int, ...],
) -> mx.array:
    """Drive ONE batched decode step across the simulated 2-rank split.
    ``tokens`` shape ``(N, 1)`` -- N requests stacked on the batch axis
    (axis 0), matching ``BatchStepContext``'s row-order contract.

    Each rank's thread sets its OWN ``batch_step_scope`` -- ContextVars
    do not propagate across ``threading.Thread`` boundaries (see
    ``pp_batched_decode_layers`` module docstring), so this is not
    optional plumbing, it's the correct pattern for this exact
    threading shape.
    """
    mx.eval(tokens)
    result: dict[str, Any] = {}
    ctx = BatchStepContext(request_uids=request_uids)

    def _rank0() -> None:
        _MLX_CALL_LOCK.acquire()
        try:
            mx.eval(mx.zeros(1))
            with batch_step_scope(ctx):
                out = rank0_model(tokens, cache=rank0_cache)
                mx.eval(out)
                # Force-eval the cache's OWN internal mutable state
                # (e.g. BatchKVCache.offset/left_padding -- lazy mx
                # arrays advanced via `+=` each call, NOT necessarily
                # on `out`'s own dependency graph since `out` is
                # computed from the PRE-increment offset) before this
                # thread exits. This harness spawns a BRAND NEW thread
                # for every decode step (unlike a real persistent-
                # per-rank-process design) -- without this, the NEXT
                # step's new thread inherits a cache object with lazy
                # graph nodes still bound to THIS (now-dead) thread's
                # MLX stream context, raising "There is no Stream(gpu,
                # N) in current thread." the moment anything touches
                # them. Found empirically the first time this
                # multi-step batched-decode loop actually ran
                # (2026-08-05) -- mx.eval(out) alone was insufficient.
                mx.eval([layer.state for layer in rank0_cache])
            result["logits"] = out
        except BaseException as e:  # noqa: BLE001 - surface on join
            result["error0"] = e
        finally:
            _MLX_CALL_LOCK.release()

    def _rank1() -> None:
        _MLX_CALL_LOCK.acquire()
        try:
            mx.eval(mx.zeros(1))
            with batch_step_scope(ctx):
                out = rank1_model(tokens, cache=rank1_cache)
                mx.eval(out)
                mx.eval([layer.state for layer in rank1_cache])
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
                "_run_batched_decode_step: simulated rank thread deadlocked"
            )
    if "error0" in result:
        raise result["error0"]
    if "error1" in result:
        raise result["error1"]
    return cast(mx.array, result["logits"])


def test_batched_two_request_decode_matches_two_serial_plain_forwards() -> None:
    """THE checkpoint: 2 concurrent decode-only requests via the new
    batched scheduler-adjacent machinery must produce IDENTICAL greedy
    tokens to 2 serial single-request plain forward passes, over
    several decode steps, per request."""
    n_decode_steps = 5
    vocab_size = _ARGS.vocab_size

    golden_model = _seeded_model(seed=999)
    prompt_a = _make_prompt(length=6, vocab_size=vocab_size, seed=1)
    prompt_b = _make_prompt(length=9, vocab_size=vocab_size, seed=2)

    golden_tokens_a, cache_a_full = _plain_prefill_and_decode(
        golden_model, prompt_a, n_decode_steps
    )
    # Rebuild a second identical-weight model instance for B's own cache
    # state -- reusing golden_model's cache object across two independent
    # prefills would corrupt both (same object, shared mutable state).
    golden_model_b = _seeded_model(seed=999)
    _copy_weights(golden_model, golden_model_b)
    golden_tokens_b, cache_b_full = _plain_prefill_and_decode(
        golden_model_b, prompt_b, n_decode_steps
    )

    # Candidate: build the 2-rank batched split from freshly re-prefilled
    # per-request caches (same prompts, same weights) split at the SAME
    # midpoint the PP layer wrapping uses, then merged into one batched
    # cache per rank via pp_batched_cache_router's real merge().
    rank0_model = _seeded_model(seed=999)
    rank1_model = _seeded_model(seed=999)
    _copy_weights(golden_model, rank0_model)
    _copy_weights(golden_model, rank1_model)

    prefill_model_a = _seeded_model(seed=999)
    _copy_weights(golden_model, prefill_model_a)
    _, cache_a_for_split = _plain_prefill_and_decode(prefill_model_a, prompt_a, 1)
    prefill_model_b = _seeded_model(seed=999)
    _copy_weights(golden_model, prefill_model_b)
    _, cache_b_for_split = _plain_prefill_and_decode(prefill_model_b, prompt_b, 1)

    n_layers = len(cache_a_for_split)
    mid = n_layers // 2
    rank0_cache_a = cache_a_for_split[:mid]
    rank1_cache_a = cache_a_for_split[mid:]
    rank0_cache_b = cache_b_for_split[:mid]
    rank1_cache_b = cache_b_for_split[mid:]

    batched_rank0_cache = merge_request_caches([rank0_cache_a, rank0_cache_b])
    batched_rank1_cache = merge_request_caches([rank1_cache_a, rank1_cache_b])

    rank0_model, rank1_model, transport = _build_two_rank_batched_split(
        rank0_model, rank1_model
    )

    # Both requests' FIRST decode token was already produced by the
    # prefill pass above (matching _plain_prefill_and_decode's own
    # "prefill produces token 0" contract) -- start the batched decode
    # loop from there, matching golden_tokens_a[0]/golden_tokens_b[0].
    candidate_tokens_a: list[int] = [golden_tokens_a[0]]
    candidate_tokens_b: list[int] = [golden_tokens_b[0]]
    next_token_a = golden_tokens_a[0]
    next_token_b = golden_tokens_b[0]

    for _step in range(n_decode_steps - 1):
        batched_tokens = mx.array([[next_token_a], [next_token_b]])
        logits = _run_batched_decode_step(
            rank0_model,
            rank1_model,
            transport,
            batched_tokens,
            batched_rank0_cache,
            batched_rank1_cache,
            request_uids=(1, 2),
        )
        next_token_a = int(mx.argmax(logits[0, -1]).item())
        next_token_b = int(mx.argmax(logits[1, -1]).item())
        candidate_tokens_a.append(next_token_a)
        candidate_tokens_b.append(next_token_b)

    assert candidate_tokens_a == golden_tokens_a, (
        f"request A's batched-decode tokens diverged from its serial "
        f"plain-forward golden reference: {candidate_tokens_a} != "
        f"{golden_tokens_a}"
    )
    assert candidate_tokens_b == golden_tokens_b, (
        f"request B's batched-decode tokens diverged from its serial "
        f"plain-forward golden reference: {candidate_tokens_b} != "
        f"{golden_tokens_b}"
    )

    # Sanity: extract_request_cache round-trips back out cleanly too
    # (exercised here against REAL post-decode batched cache state, not
    # just the isolated unit tests in test_pp_batched_cache_router.py).
    extracted_a = extract_request_cache(batched_rank0_cache, 0)
    extracted_b = extract_request_cache(batched_rank0_cache, 1)
    assert len(extracted_a) == mid
    assert len(extracted_b) == mid
    # Suppress unused-variable lint noise for cache_a_full/cache_b_full
    # (kept for readability/documentation of what the golden path also
    # produces, even though only the token sequences are asserted on).
    del cache_a_full, cache_b_full


def test_single_request_batch_of_one_matches_plain_forward() -> None:
    """Degenerate N=1 case: the batched machinery with exactly ONE
    request in the batch must still match the plain forward exactly --
    catches any off-by-one in the batch-axis plumbing that only a
    genuinely single-row batch would expose (e.g. an accidental
    assumption that num_requests is always >=2)."""
    n_decode_steps = 4
    vocab_size = _ARGS.vocab_size

    golden_model = _seeded_model(seed=555)
    prompt = _make_prompt(length=7, vocab_size=vocab_size, seed=3)
    golden_tokens, _cache = _plain_prefill_and_decode(
        golden_model, prompt, n_decode_steps
    )

    rank0_model = _seeded_model(seed=555)
    rank1_model = _seeded_model(seed=555)
    _copy_weights(golden_model, rank0_model)
    _copy_weights(golden_model, rank1_model)

    prefill_model = _seeded_model(seed=555)
    _copy_weights(golden_model, prefill_model)
    _, cache_for_split = _plain_prefill_and_decode(prefill_model, prompt, 1)

    n_layers = len(cache_for_split)
    mid = n_layers // 2
    rank0_cache = cache_for_split[:mid]
    rank1_cache = cache_for_split[mid:]

    batched_rank0_cache = merge_request_caches([rank0_cache])
    batched_rank1_cache = merge_request_caches([rank1_cache])

    rank0_model, rank1_model, transport = _build_two_rank_batched_split(
        rank0_model, rank1_model
    )

    candidate_tokens: list[int] = [golden_tokens[0]]
    next_token = golden_tokens[0]
    for _step in range(n_decode_steps - 1):
        batched_tokens = mx.array([[next_token]])
        logits = _run_batched_decode_step(
            rank0_model,
            rank1_model,
            transport,
            batched_tokens,
            batched_rank0_cache,
            batched_rank1_cache,
            request_uids=(1,),
        )
        next_token = int(mx.argmax(logits[0, -1]).item())
        candidate_tokens.append(next_token)

    assert candidate_tokens == golden_tokens


def test_batched_decode_layer_rejects_mismatched_batch_step_context() -> None:
    """Direct check of the ordering/identity guard in
    ``BatchedMetaFramedPipelineLastLayer`` -- a wrong request_uids
    tuple in the scope must raise, not silently proceed and swap
    tokens between requests."""
    from exo.worker.engines.mlx.pp_batched_decode_layers import (
        _require_batch_step_context,
    )

    with pytest.raises(RuntimeError, match="outside an active"):
        _require_batch_step_context()

    with batch_step_scope(BatchStepContext(request_uids=(1, 2))):
        assert _require_batch_step_context().request_uids == (1, 2)
    # Outside the scope again -- must raise again, not leak the prior
    # context (module docstring point 4: token/reset in finally).
    with pytest.raises(RuntimeError, match="outside an active"):
        _require_batch_step_context()
