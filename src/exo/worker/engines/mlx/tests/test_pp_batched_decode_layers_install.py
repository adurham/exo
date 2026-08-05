# pyright: reportAny=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportPrivateUsage=false, reportUnknownLambdaType=false
"""Tests for install_batched_decode_pipeline_layers
(pp_batched_decode_layers.py) -- the load-time layer-swap function
that installs BatchedMetaFramedPipelineFirstLayer/LastLayer onto an
already pipeline_auto_parallel-sharded model, mirroring
pp_metaframe.install_metaframed_pipeline_layers's own established
test pattern exactly (see test_pp_metaframe.py's
test_install_metaframed_pipeline_layers_rejects_unsharded_model and
_build_metaframe_split)."""

from __future__ import annotations

from typing import cast

import mlx.core as mx
import mlx.utils
import pytest
from mlx_lm.models.llama import Model as LlamaModel
from mlx_lm.models.llama import ModelArgs

from exo.worker.engines.mlx.auto_parallel import (
    PipelineFirstLayer,
    PipelineLastLayer,
    _set_layers,
    get_inner_model,
    get_layers,
)
from exo.worker.engines.mlx.pp_batched_correctness import _RankGroup
from exo.worker.engines.mlx.pp_batched_decode_layers import (
    BatchedMetaFramedPipelineFirstLayer,
    BatchedMetaFramedPipelineLastLayer,
    install_batched_decode_pipeline_layers,
)

_ARGS = ModelArgs(
    model_type="llama",
    hidden_size=64,
    num_hidden_layers=4,
    intermediate_size=128,
    num_attention_heads=2,
    num_key_value_heads=1,
    rms_norm_eps=1e-6,
    vocab_size=256,
    rope_theta=10000.0,
    tie_word_embeddings=True,
)


def _random_model(seed: int) -> LlamaModel:
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


def _install_legacy_pp_split(model: LlamaModel, group: mx.distributed.Group) -> None:
    """Mirror pipeline_auto_parallel's own layer-wrapping (minus the
    generator/chunked-eval machinery, matching test_pp_metaframe.py's
    own _build_legacy_split pattern) -- installs today's
    PipelineFirstLayer/PipelineLastLayer directly, the state
    install_batched_decode_pipeline_layers expects as its
    precondition."""
    inner = get_inner_model(model)
    layers = list(get_layers(inner))
    layers[0] = PipelineFirstLayer(layers[0], r=0, group=group)
    layers[-1] = PipelineLastLayer(layers[-1], r=0, s=1, group=group)
    _set_layers(model, layers)


def test_install_batched_decode_pipeline_layers_rejects_unsharded_model() -> None:
    """Guard: calling this on a model that was never
    pipeline_auto_parallel-sharded (no PipelineFirstLayer/
    PipelineLastLayer present) must fail loudly, not silently no-op
    -- matches install_metaframed_pipeline_layers's own precondition
    check exactly."""
    model = LlamaModel(_ARGS)
    mx.eval(model.parameters())
    group = cast(mx.distributed.Group, cast(object, _RankGroup(0, 2)))

    with pytest.raises(RuntimeError, match="found no"):
        install_batched_decode_pipeline_layers(model, group)


def test_install_batched_decode_pipeline_layers_replaces_first_and_last() -> None:
    """After installation, the model's first/last layers must be the
    BATCHED classes specifically (not the Phase 0.5 single-request
    metaframe classes, and not the legacy PipelineFirstLayer/
    PipelineLastLayer) -- and every middle layer must be untouched."""
    model = _random_model(seed=1)
    group = cast(mx.distributed.Group, cast(object, _RankGroup(0, 2)))
    _install_legacy_pp_split(model, group)

    inner = get_inner_model(model)
    pre_middle_layers = list(get_layers(inner))[1:-1]

    install_batched_decode_pipeline_layers(model, group)

    inner_after = get_inner_model(model)
    layers_after = list(get_layers(inner_after))

    assert isinstance(layers_after[0], BatchedMetaFramedPipelineFirstLayer)
    assert isinstance(layers_after[-1], BatchedMetaFramedPipelineLastLayer)
    # Middle layers are the exact same objects, untouched by the swap.
    assert layers_after[1:-1] == pre_middle_layers


def test_install_batched_decode_pipeline_layers_preserves_r_s_group() -> None:
    """The batched layers' r/s/group must come from the ORIGINAL
    PipelineFirstLayer/PipelineLastLayer instances being replaced, not
    hardcoded defaults -- a real multi-rank deployment needs each
    rank's actual r/s carried through the swap correctly."""
    model = _random_model(seed=2)
    group = cast(mx.distributed.Group, cast(object, _RankGroup(1, 4)))
    inner = get_inner_model(model)
    layers = list(get_layers(inner))
    layers[0] = PipelineFirstLayer(layers[0], r=1, group=group)
    layers[-1] = PipelineLastLayer(layers[-1], r=1, s=4, group=group)
    _set_layers(model, layers)

    install_batched_decode_pipeline_layers(model, group)

    inner_after = get_inner_model(model)
    layers_after = list(get_layers(inner_after))
    first = layers_after[0]
    last = layers_after[-1]
    assert isinstance(first, BatchedMetaFramedPipelineFirstLayer)
    assert isinstance(last, BatchedMetaFramedPipelineLastLayer)
    assert first.r == 1
    assert first.group is group
    assert last.r == 1
    assert last.s == 4
    assert last.group is group


def test_install_batched_decode_pipeline_layers_preserves_original_layer_weights() -> (
    None
):
    """The swap must preserve the ORIGINAL layer's actual weights
    (wrapping, not rebuilding) -- confirmed via a real forward pass
    producing IDENTICAL output before/after the swap. Patches
    mx.distributed.send/recv_like to trivial pass-throughs for this
    test specifically: at s=1, r=0 IS both the first and the last
    rank, so PipelineLastLayer's gather-handoff branch
    (``self.r == self.s - 1``) fires unconditionally even with a
    single rank -- this is real, correct behavior of the ALREADY-
    established legacy/batched layer classes (not something this
    install function changes), and is irrelevant to what THIS test
    actually verifies (that install_batched_decode_pipeline_layers
    wraps rather than rebuilds the underlying layer's weights) --
    same rationale as pp_batched_correctness.py's own patching of
    these two functions for its simulated-transport tests."""
    from unittest.mock import patch

    def _passthrough_send(arr: mx.array, dst: int, **_: object) -> mx.array:
        del dst
        return arr

    def _passthrough_recv_like(template: mx.array, src: int, **_: object) -> mx.array:
        del src
        # Only reached if this test's r=0/s=1 config forces a receive
        # branch to fire (it shouldn't, given r=0 is also the first
        # rank), so return a zero tensor as an inert fallback rather
        # than crash with an unhelpful signature-mismatch error.
        return mx.zeros(template.shape, dtype=template.dtype)

    model = _random_model(seed=3)
    group = cast(mx.distributed.Group, cast(object, _RankGroup(0, 1)))
    inner = get_inner_model(model)
    layers = list(get_layers(inner))
    layers[0] = PipelineFirstLayer(layers[0], r=0, group=group)
    layers[-1] = PipelineLastLayer(layers[-1], r=0, s=1, group=group)
    _set_layers(model, layers)

    mx.random.seed(99)
    prompt = mx.random.randint(0, _ARGS.vocab_size, shape=(4,))

    with (
        patch("mlx.core.distributed.send", side_effect=_passthrough_send),
        patch("mlx.core.distributed.recv_like", side_effect=_passthrough_recv_like),
    ):
        cache_before = model.make_cache()
        logits_before = model(prompt[None, :], cache=cache_before)
        mx.eval(logits_before)

        install_batched_decode_pipeline_layers(model, group)

        from exo.worker.engines.mlx.pp_batched_decode_layers import (
            BatchStepContext,
            batch_step_scope,
        )

        cache_after = model.make_cache()
        with batch_step_scope(BatchStepContext(request_uids=(1,))):
            logits_after = model(prompt[None, :], cache=cache_after)
            mx.eval(logits_after)

    assert bool(mx.allclose(logits_before, logits_after, atol=1e-5).item())
