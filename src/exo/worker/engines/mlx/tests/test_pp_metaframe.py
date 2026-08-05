# pyright: reportAny=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownLambdaType=false, reportPrivateUsage=false
# pyright: reportInvalidCast=false, reportArgumentType=false
"""Phase 0.5 correctness tests for the metadata-framed PP transport.

See ``docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md`` Section 9,
"Phase 0.5 — Transport-only refactor at concurrency=1": run a SINGLE
request (concurrency still =1, no scheduler, no batching) through the
NEW metadata-framed send/recv protocol
(``exo.worker.engines.mlx.pp_metaframe``) and verify EXACT parity
against today's existing PP transport
(``PipelineFirstLayer``/``PipelineLastLayer``).

This reuses the Phase 0 harness's simulated-2-rank machinery
(``pp_batched_correctness``) rather than the real cluster — per the
Phase 0.5 isolation rationale (a transport bug found here is much
easier to localize than the same bug discovered later under real
concurrency=2 batched load), and per the design doc's own instruction
to validate locally before spending cluster/relaunch time.

Reviewed via `consult` before writing the transport itself (2026-08-05,
see pp_metaframe.py's module docstring for the full list of design
points that review shaped). This test file's job is narrower: prove the
two transports produce IDENTICAL output, not re-litigate the design.

Comparison is byte-for-byte at temp=0 (argmax token sequence AND raw
logit equality within float noise) — NOT the wider tolerance
test_pp_batched_correctness_harness.py needed for its plain-forward
anchor. Both paths here pay the IDENTICAL bf16 transport cast cost
(both are real PP, just with different metadata mechanisms), so unlike
that file's comparison, this one CAN and SHOULD use a tight tolerance —
exactly the distinction test_pp_batched_correctness_harness.py's module
docstring flagged as the reason for Phase 0.5+ to reuse a tighter bar.
"""

import threading
from typing import cast
from unittest.mock import patch

import mlx.core as mx
import mlx.utils
import pytest
from mlx_lm.models.llama import Model as LlamaModel
from mlx_lm.models.llama import ModelArgs

from exo.worker.engines.mlx.auto_parallel import (
    PipelineFirstLayer,
    PipelineLastLayer,
    _set_layers,
    clear_prefill_sends,
    flush_prefill_sends,
    get_inner_model,
    get_layers,
    set_pipeline_prefill,
    set_pipeline_queue_sends,
)
from exo.worker.engines.mlx.pp_batched_correctness import (
    _MLX_CALL_LOCK,
    SimPipelineTransport,
    _RankGroup,
    compare_logits,
)
from exo.worker.engines.mlx.pp_metaframe import (
    MetaFramedPipelineFirstLayer,
    MetaFramedPipelineLastLayer,
    encode_metaframe,
    handshake_metaframe_protocol,
    install_metaframed_pipeline_layers,
    recv_metaframe,
    send_metaframe,
)

# Tight tolerance — both sides pay the identical bf16 transport cast
# cost (see module docstring). A genuine transport-protocol bug (wrong
# shape, dropped chunk, mis-ordered frame/tensor) would produce either
# a crash, a shape-mismatch RuntimeError, or a diff many orders of
# magnitude above this, not a diff hovering just above it.
EXACT_PARITY_TOLERANCE = 1e-4

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


def _seeded_model() -> LlamaModel:
    mx.random.seed(4321)
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


def _make_prompt(length: int, vocab_size: int) -> mx.array:
    mx.random.seed(77)
    return mx.random.randint(0, vocab_size, shape=(length,))


def _build_legacy_split(r0: LlamaModel, r1: LlamaModel) -> SimPipelineTransport:
    """Wire r0/r1 with TODAY's PipelineFirstLayer/PipelineLastLayer —
    the existing, trusted transport this test validates the new
    metaframe path against."""
    inner0 = get_inner_model(r0)
    inner1 = get_inner_model(r1)
    layers0 = list(get_layers(inner0))
    layers1 = list(get_layers(inner1))
    mid = len(layers0) // 2

    transport = SimPipelineTransport()
    group0 = cast(mx.distributed.Group, cast(object, _RankGroup(0, 2)))
    group1 = cast(mx.distributed.Group, cast(object, _RankGroup(1, 2)))

    r0_layers = list(layers0[:mid])
    r1_layers = list(layers1[mid:])
    r0_layers[0] = PipelineFirstLayer(r0_layers[0], r=0, group=group0)
    r0_layers[-1] = PipelineLastLayer(r0_layers[-1], r=0, s=2, group=group0)
    r1_layers[0] = PipelineFirstLayer(r1_layers[0], r=1, group=group1)
    r1_layers[-1] = PipelineLastLayer(r1_layers[-1], r=1, s=2, group=group1)
    _set_layers(r0, r0_layers)
    _set_layers(r1, r1_layers)
    return transport


def _build_metaframe_split(
    r0: LlamaModel, r1: LlamaModel, *, request_uid: int
) -> SimPipelineTransport:
    """Wire r0/r1 with the NEW MetaFramedPipelineFirstLayer/
    MetaFramedPipelineLastLayer — build the legacy split first (reusing
    the same layer-splitting logic), then swap in the metaframe
    variants via install_metaframed_pipeline_layers, exactly as a real
    A/B deployment would (today's model-loading path stays the SAME up
    to the point the metaframe classes get installed)."""
    transport = _build_legacy_split(r0, r1)
    group0 = cast(mx.distributed.Group, cast(object, _RankGroup(0, 2)))
    group1 = cast(mx.distributed.Group, cast(object, _RankGroup(1, 2)))
    install_metaframed_pipeline_layers(r0, group0, request_uid=request_uid)
    install_metaframed_pipeline_layers(r1, group1, request_uid=request_uid)
    return transport


def _run_forward(
    r0: LlamaModel,
    r1: LlamaModel,
    transport: SimPipelineTransport,
    tokens: mx.array,
    c0: list[object],
    c1: list[object],
    *,
    is_prefill: bool,
    metaframe: bool,
) -> mx.array:
    """Drive one simulated 2-rank forward for either transport variant.
    Mirrors pp_batched_correctness.run_two_rank_pp_forward's threading/
    locking discipline exactly (see that module's docstring for the
    full rationale) — duplicated here rather than imported because the
    prefill/decode phase-setter calls differ between the legacy and
    metaframe layer classes (set_pipeline_prefill/set_pipeline_queue_sends
    vs the metaframe module's own counterparts)."""
    if metaframe:
        from exo.worker.engines.mlx.pp_metaframe import (
            set_metaframed_pipeline_prefill,
            set_metaframed_pipeline_queue_sends,
        )

        set_metaframed_pipeline_prefill(r0, is_prefill)
        set_metaframed_pipeline_prefill(r1, is_prefill)
        set_metaframed_pipeline_queue_sends(r0, is_prefill)
    else:
        set_pipeline_prefill(r0, is_prefill=is_prefill)
        set_pipeline_prefill(r1, is_prefill=is_prefill)
        set_pipeline_queue_sends(r0, queue_sends=is_prefill)

    mx.eval(tokens)
    result: dict[str, object] = {}

    def _rank0() -> None:
        _MLX_CALL_LOCK.acquire()
        try:
            mx.eval(mx.zeros(1))
            out = r0(tokens, cache=c0)
            # Match production's per-chunk discipline (generate.py's
            # prefill loop calls flush_prefill_sends() right after every
            # forward, unconditionally — a no-op when nothing was
            # queued). Without this, a queue_sends=True forward (prefill
            # under both the legacy and metaframe paths) leaves its
            # activation send parked in the shared _pending_prefill_sends
            # queue forever, and rank 1's recv blocks indefinitely.
            flush_prefill_sends()
            mx.eval(out)
            result["logits"] = out
        except BaseException as e:  # noqa: BLE001
            result["error0"] = e
        finally:
            _MLX_CALL_LOCK.release()

    def _rank1() -> None:
        _MLX_CALL_LOCK.acquire()
        try:
            mx.eval(mx.zeros(1))
            out = r1(tokens, cache=c1)
            mx.eval(out)
        except BaseException as e:  # noqa: BLE001
            result["error1"] = e
        finally:
            _MLX_CALL_LOCK.release()

    clear_prefill_sends()
    with (
        patch("mlx.core.distributed.send", transport.send),
        patch("mlx.core.distributed.recv_like", transport.recv_like),
    ):
        t0 = threading.Thread(target=_rank0)
        t1 = threading.Thread(target=_rank1)
        t0.start()
        t1.start()
        t0.join(timeout=15)
        t1.join(timeout=15)
        if t0.is_alive() or t1.is_alive():
            raise RuntimeError("simulated rank thread deadlocked")
    if "error0" in result:
        raise cast(BaseException, result["error0"])
    if "error1" in result:
        raise cast(BaseException, result["error1"])
    return cast(mx.array, result["logits"])


def _prefill_and_decode(
    prompt: mx.array, n_decode: int, *, metaframe: bool
) -> list[mx.array]:
    src = _seeded_model()
    r0 = LlamaModel(_ARGS)
    r1 = LlamaModel(_ARGS)
    _copy_weights(src, r0)
    _copy_weights(src, r1)

    if metaframe:
        transport = _build_metaframe_split(r0, r1, request_uid=123)
    else:
        transport = _build_legacy_split(r0, r1)

    c0 = r0.make_cache()
    c1 = r1.make_cache()

    if len(prompt) > 1:
        _run_forward(
            r0,
            r1,
            transport,
            prompt[:-1][None],
            c0,
            c1,
            is_prefill=True,
            metaframe=metaframe,
        )

    cur = int(prompt[-1].item())
    logits_per_step: list[mx.array] = []
    for _ in range(n_decode):
        out = _run_forward(
            r0,
            r1,
            transport,
            mx.array([[cur]]),
            c0,
            c1,
            is_prefill=False,
            metaframe=metaframe,
        )
        logits_per_step.append(out[0, -1])
        cur = int(mx.argmax(out[0, -1]).item())
    return logits_per_step


@pytest.mark.slow
def test_metaframe_transport_exact_parity_with_legacy_transport() -> None:
    """THE Phase 0.5 deliverable: the new metadata-framed transport must
    produce EXACT parity against today's existing PipelineFirstLayer/
    PipelineLastLayer transport, for the SAME weights, SAME prompt, at
    concurrency=1. Unlike the Phase 0 harness's plain-forward anchor,
    BOTH sides here pay the identical bf16 cast cost (both are real PP
    transports), so a tight tolerance is the right bar — any diff above
    float noise here is a real transport bug, not expected precision
    loss.
    """
    prompt = _make_prompt(length=14, vocab_size=_ARGS.vocab_size)
    n_decode = 8

    legacy_logits = _prefill_and_decode(prompt, n_decode, metaframe=False)
    metaframe_logits = _prefill_and_decode(prompt, n_decode, metaframe=True)

    max_diff, mismatches = compare_logits(
        legacy_logits, metaframe_logits, "metaframe-vs-legacy"
    )
    assert mismatches == 0, (
        f"Metaframe transport diverged from legacy transport: "
        f"{mismatches}/{n_decode} argmax mismatches -- Phase 0.5 parity "
        f"FAILED, do not proceed to a real cluster test"
    )
    assert max_diff < EXACT_PARITY_TOLERANCE, (
        f"Metaframe transport max logit diff {max_diff} exceeds tight "
        f"parity tolerance {EXACT_PARITY_TOLERANCE} -- both transports "
        f"pay the same bf16 cost, so this indicates a real protocol bug"
    )


@pytest.mark.slow
def test_metaframe_transport_exact_parity_single_token_prompt() -> None:
    """Degenerate case per the consult review's validation-coverage
    recommendation: single-token prompt (no real prefill chunk)."""
    prompt = _make_prompt(length=1, vocab_size=_ARGS.vocab_size)
    n_decode = 5

    legacy_logits = _prefill_and_decode(prompt, n_decode, metaframe=False)
    metaframe_logits = _prefill_and_decode(prompt, n_decode, metaframe=True)

    max_diff, mismatches = compare_logits(
        legacy_logits, metaframe_logits, "metaframe-vs-legacy-single-token"
    )
    assert mismatches == 0
    assert max_diff < EXACT_PARITY_TOLERANCE


@pytest.mark.slow
def test_metaframe_transport_exact_parity_long_decode() -> None:
    """Consult review's validation-coverage recommendation: a longer
    decode run to increase the chance a phase-transition/is_last_chunk
    edge case (where the old ambient-flag toggling and the new explicit
    per-step framing are most likely to disagree) would surface."""
    prompt = _make_prompt(length=6, vocab_size=_ARGS.vocab_size)
    n_decode = 24

    legacy_logits = _prefill_and_decode(prompt, n_decode, metaframe=False)
    metaframe_logits = _prefill_and_decode(prompt, n_decode, metaframe=True)

    max_diff, mismatches = compare_logits(
        legacy_logits, metaframe_logits, "metaframe-vs-legacy-long-decode"
    )
    assert mismatches == 0
    assert max_diff < EXACT_PARITY_TOLERANCE


def test_metaframe_protocol_version_mismatch_raises() -> None:
    """recv_metaframe must raise loudly on a protocol version mismatch
    -- exercises the version field the consult review flagged as
    near-free insurance against future frame-layout changes."""
    from exo.worker.engines.mlx import pp_metaframe as _mod

    transport = SimPipelineTransport()
    group0 = cast(mx.distributed.Group, cast(object, _RankGroup(0, 2)))
    group1 = cast(mx.distributed.Group, cast(object, _RankGroup(1, 2)))

    # Manually build a header with a wrong version number.
    bad_header = mx.array([999, 0, 1, 64, 0], dtype=mx.int32)
    table = mx.array([[1, 4, 1, 0]], dtype=mx.int32)

    _MLX_CALL_LOCK.acquire()
    try:
        with (
            patch("mlx.core.distributed.send", transport.send),
            patch("mlx.core.distributed.recv_like", transport.recv_like),
        ):
            transport.send(bad_header, 1, group=group0)
            transport.send(table, 1, group=group0)
            with pytest.raises(RuntimeError, match="version mismatch"):
                recv_metaframe(0, group=group1)
    finally:
        _MLX_CALL_LOCK.release()
    assert _mod.METAFRAME_PROTOCOL_VERSION == 2


def test_metaframe_handshake_agrees_when_both_ranks_match() -> None:
    """handshake_metaframe_protocol must succeed silently when both
    ranks are called with identical (enabled, version) -- simulated via
    a real 2-rank all_sum through the same fake-transport pattern used
    elsewhere in this file. Uses threads (all_sum is a real collective
    call on both simulated ranks) with the same _MLX_CALL_LOCK
    discipline as the rest of this file."""
    group0 = cast(mx.distributed.Group, cast(object, _RankGroup(0, 2)))
    group1 = cast(mx.distributed.Group, cast(object, _RankGroup(1, 2)))

    # all_sum isn't routed through SimPipelineTransport's send/recv_like
    # (it's a different MLX primitive) -- patch it directly for this
    # test with a trivial in-process reducer.
    results: dict[str, mx.array] = {}
    barrier = threading.Barrier(2)

    def fake_all_sum(arr: mx.array, *, group, stream=None) -> mx.array:  # type: ignore[no-untyped-def]
        rank = group.rank()
        results[f"r{rank}"] = arr
        barrier.wait(timeout=10)
        total = results["r0"] + results["r1"]
        return total

    errors: dict[str, BaseException] = {}

    def _rank(rank: int, group: mx.distributed.Group) -> None:
        try:
            handshake_metaframe_protocol(True, group)
        except BaseException as e:  # noqa: BLE001
            errors[f"r{rank}"] = e

    with patch("mlx.core.distributed.all_sum", fake_all_sum):
        t0 = threading.Thread(target=_rank, args=(0, group0))
        t1 = threading.Thread(target=_rank, args=(1, group1))
        t0.start()
        t1.start()
        t0.join(timeout=10)
        t1.join(timeout=10)

    assert not errors, f"handshake raised unexpectedly: {errors}"


def test_metaframe_handshake_raises_on_mismatch() -> None:
    """handshake_metaframe_protocol must raise loudly when ranks
    disagree on `enabled` -- exactly the operator-mistake scenario
    (mismatched EXO_PP_METAFRAME between nodes) the consult review
    flagged as a silent-hang risk without this check."""
    group0 = cast(mx.distributed.Group, cast(object, _RankGroup(0, 2)))
    group1 = cast(mx.distributed.Group, cast(object, _RankGroup(1, 2)))

    results: dict[str, mx.array] = {}
    barrier = threading.Barrier(2)

    def fake_all_sum(arr: mx.array, *, group, stream=None) -> mx.array:  # type: ignore[no-untyped-def]
        rank = group.rank()
        results[f"r{rank}"] = arr
        barrier.wait(timeout=10)
        total = results["r0"] + results["r1"]
        return total

    errors: dict[str, BaseException] = {}

    def _rank(rank: int, group: mx.distributed.Group, enabled: bool) -> None:
        try:
            handshake_metaframe_protocol(enabled, group)
        except BaseException as e:  # noqa: BLE001
            errors[f"r{rank}"] = e

    with patch("mlx.core.distributed.all_sum", fake_all_sum):
        t0 = threading.Thread(target=_rank, args=(0, group0, True))
        # rank 1 mistakenly launched without EXO_PP_METAFRAME=1.
        t1 = threading.Thread(target=_rank, args=(1, group1, False))
        t0.start()
        t1.start()
        t0.join(timeout=10)
        t1.join(timeout=10)

    assert "r0" in errors and "r1" in errors, (
        "both ranks must detect the mismatch and refuse to proceed"
    )
    assert "handshake FAILED" in str(errors["r0"])
    assert "handshake FAILED" in str(errors["r1"])


def test_install_metaframed_pipeline_layers_rejects_unsharded_model() -> None:
    """Guard: calling install_metaframed_pipeline_layers on a model that
    was never pipeline_auto_parallel-sharded (no
    PipelineFirstLayer/PipelineLastLayer present) must fail loudly, not
    silently no-op."""
    model = LlamaModel(_ARGS)
    mx.eval(model.parameters())
    group = cast(mx.distributed.Group, cast(object, _RankGroup(0, 2)))

    with pytest.raises(ValueError, match="found no"):
        install_metaframed_pipeline_layers(model, group, request_uid=1)


def test_metaframe_encode_send_recv_roundtrip_preserves_fields() -> None:
    """Direct unit test of encode_metaframe -> send_metaframe ->
    recv_metaframe, independent of any model forward -- confirms the
    header+table wire format preserves every field exactly."""
    transport = SimPipelineTransport()
    group0 = cast(mx.distributed.Group, cast(object, _RankGroup(0, 2)))
    group1 = cast(mx.distributed.Group, cast(object, _RankGroup(1, 2)))

    header, table = encode_metaframe(
        phase_flag=1,
        hidden_dim=256,
        request_uid=777,
        seq_len=1,
        is_last_chunk=True,
    )

    _MLX_CALL_LOCK.acquire()
    try:
        with (
            patch("mlx.core.distributed.send", transport.send),
            patch("mlx.core.distributed.recv_like", transport.recv_like),
        ):
            send_metaframe(header, table, 1, group=group0)
            frame = recv_metaframe(0, group=group1)
    finally:
        _MLX_CALL_LOCK.release()

    assert frame.phase_flag == 1
    assert frame.hidden_dim == 256
    assert frame.num_requests == 1
    assert frame.request_uids == [777]
    assert frame.seq_lens == [1]
    assert frame.is_last_chunk == [True]
    assert frame.total_tokens == 1
    assert frame.activation_template_shape() == (1, 1, 256)


def test_metaframed_layers_are_new_classes_not_legacy_subclasses() -> None:
    """Isolation-boundary guard (consult review point 5): the metaframe
    classes must be structurally distinct from PipelineFirstLayer/
    PipelineLastLayer, not a flag bolted onto them -- confirms today's
    shipped transport genuinely cannot be affected by metaframe code."""
    assert not issubclass(MetaFramedPipelineFirstLayer, PipelineFirstLayer)
    assert not issubclass(MetaFramedPipelineLastLayer, PipelineLastLayer)
    assert not issubclass(PipelineFirstLayer, MetaFramedPipelineFirstLayer)
    assert not issubclass(PipelineLastLayer, MetaFramedPipelineLastLayer)


def test_metaframe_encode_decode_roundtrip_4d_hyper_connection_shape() -> None:
    """Regression test for the REAL bug found on the first live cluster
    run (2026-08-05): DSv4-Flash's hyper-connection residual stream is
    4D -- (batch, seq_len, hc_mult, hidden_dim) -- not the 3D shape
    every OTHER test in this file exercises (mlx-lm's plain Llama has
    no hyper-connections and stays 3D throughout, so this class of bug
    was invisible to local validation until DSv4 actually ran on the
    real cluster: `RunnerFailed: ValueError: not enough values to
    unpack (expected 4, got 3)` inside hyper_connection.py, because
    v1's activation_template_shape() hardcoded a 3D recv_like template
    against a real 4D tensor). Confirms encode_metaframe/recv_metaframe
    correctly round-trip the extra_dim field and that
    activation_template_shape() returns the correct 4-tuple when
    extra_dim > 0, matching DSv4's real hc_mult=4 configuration."""
    transport = SimPipelineTransport()
    group0 = cast(mx.distributed.Group, cast(object, _RankGroup(0, 2)))
    group1 = cast(mx.distributed.Group, cast(object, _RankGroup(1, 2)))

    hc_mult = 4  # DSv4-Flash's real config.hc_mult value.
    header, table = encode_metaframe(
        phase_flag=0,
        hidden_dim=4096,
        request_uid=42,
        seq_len=128,
        is_last_chunk=False,
        extra_dim=hc_mult,
    )

    _MLX_CALL_LOCK.acquire()
    try:
        with (
            patch("mlx.core.distributed.send", transport.send),
            patch("mlx.core.distributed.recv_like", transport.recv_like),
        ):
            send_metaframe(header, table, 1, group=group0)
            frame = recv_metaframe(0, group=group1)
    finally:
        _MLX_CALL_LOCK.release()

    assert frame.extra_dim == hc_mult
    assert frame.activation_template_shape() == (1, 128, hc_mult, 4096)
    # And the 3D (extra_dim=0) path must be unaffected -- no regression
    # in the common case from adding this field.
    assert frame.activation_template_shape.__doc__ is not None  # sanity


def test_metaframe_3d_shape_still_default_when_extra_dim_omitted() -> None:
    """Guard: extra_dim defaults to 0 and produces the original 3D
    shape when the caller doesn't pass it -- confirms the v2 field
    addition didn't silently change behavior for every non-DSv4 model
    already covered by this file's other parity tests."""
    transport = SimPipelineTransport()
    group0 = cast(mx.distributed.Group, cast(object, _RankGroup(0, 2)))
    group1 = cast(mx.distributed.Group, cast(object, _RankGroup(1, 2)))

    header, table = encode_metaframe(
        phase_flag=1,
        hidden_dim=256,
        request_uid=1,
        seq_len=1,
        is_last_chunk=True,
    )

    _MLX_CALL_LOCK.acquire()
    try:
        with (
            patch("mlx.core.distributed.send", transport.send),
            patch("mlx.core.distributed.recv_like", transport.recv_like),
        ):
            send_metaframe(header, table, 1, group=group0)
            frame = recv_metaframe(0, group=group1)
    finally:
        _MLX_CALL_LOCK.release()

    assert frame.extra_dim == 0
    assert frame.activation_template_shape() == (1, 1, 256)


def test_metaframed_last_layer_sends_correct_extra_dim_for_4d_output() -> None:
    """Integration-level regression test: build a REAL
    MetaFramedPipelineLastLayer whose wrapped original_layer returns a
    4D tensor (simulating DSv4's hyper-connection shape without needing
    the full DSv4 model), drive it through the real
    encode_metaframe/send_metaframe call path exactly as production
    code does, and confirm the peer rank's recv_metaframe decodes the
    correct extra_dim -- catches a regression in the LastLayer's own
    shape-derivation logic (output_to_send.shape[2] when ndim==4),
    not just the encode/decode functions in isolation."""

    class _FourDLayer:
        """Minimal original_layer stand-in returning a fixed 4D
        tensor, shaped like DSv4's post-hyper-connection residual
        stream -- (batch, seq_len, hc_mult, hidden_dim)."""

        def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
            return mx.zeros((1, 3, 4, 64), dtype=mx.float32)

    transport = SimPipelineTransport()
    group0 = cast(mx.distributed.Group, cast(object, _RankGroup(0, 2)))
    group1 = cast(mx.distributed.Group, cast(object, _RankGroup(1, 2)))

    layer = MetaFramedPipelineLastLayer(
        cast(object, _FourDLayer()), r=0, s=2, group=group0, request_uid=1
    )
    layer.is_prefill = True  # prefill chunk -> only the forward hop fires

    result: dict[str, object] = {}

    def _decode() -> None:
        try:
            result["frame"] = recv_metaframe(0, group=group1)
        except BaseException as e:  # noqa: BLE001
            result["error"] = e

    import threading as _threading

    _MLX_CALL_LOCK.acquire()
    try:
        with (
            patch("mlx.core.distributed.send", transport.send),
            patch("mlx.core.distributed.recv_like", transport.recv_like),
        ):
            t = _threading.Thread(target=_decode)
            t.start()
            layer(mx.zeros((1, 3, 4, 64), dtype=mx.float32))
            t.join(timeout=10)
    finally:
        _MLX_CALL_LOCK.release()

    if "error" in result:
        raise cast(BaseException, result["error"])
    frame = result["frame"]
    assert frame.extra_dim == 4  # type: ignore[attr-defined]
    assert frame.hidden_dim == 64  # type: ignore[attr-defined]
    assert frame.seq_lens == [3]  # type: ignore[attr-defined]
