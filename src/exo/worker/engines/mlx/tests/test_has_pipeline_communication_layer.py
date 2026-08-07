# pyright: reportPrivateUsage=false
"""Regression test for `_has_pipeline_communication_layer`
(generate.py) -- 2026-08-06, the real bug found (not hypothetical)
auditing the chunk-drive fix before wiring it into live serving: this
check only ever matched the LEGACY `PipelineFirstLayer`/
`PipelineLastLayer` classes, never `MetaFramedPipelineFirstLayer`/
`MetaFramedPipelineLastLayer` (a DIFFERENT base class,
`CustomMlxLayer`, not a subclass of the legacy pair) -- meaning
`prefill()`'s `is_pipeline` gate was ALWAYS False under
`EXO_PP_METAFRAME=1`, so `pipeline_parallel_prefill()` (the only
function that can ever yield real chunk boundaries for the whole
chunked-prefill interruption mechanism built this session) was NEVER
reached, regardless of anything else fixed. Fixed by broadening the
isinstance check to include the metaframe classes too.

Reuses test_pp_metaframe.py's own established `_build_legacy_split`/
`install_metaframed_pipeline_layers` helpers -- no new transport
machinery needed, this is a pure structural detection check.
"""

from __future__ import annotations

from typing import cast

import mlx.core as mx
from mlx_lm.models.llama import Model as LlamaModel
from mlx_lm.models.llama import ModelArgs

from exo.worker.engines.mlx.generator.generate import (
    _has_pipeline_communication_layer,
)
from exo.worker.engines.mlx.pp_metaframe import install_metaframed_pipeline_layers
from exo.worker.engines.mlx.tests.test_pp_metaframe import (
    _build_legacy_split,
    _RankGroup,
)
from exo.worker.engines.mlx.types import Model as _ExoModel

_ARGS = ModelArgs(
    model_type="llama",
    hidden_size=32,
    num_hidden_layers=4,
    intermediate_size=64,
    num_attention_heads=2,
    num_key_value_heads=1,
    rms_norm_eps=1e-6,
    vocab_size=128,
    rope_theta=10000.0,
    tie_word_embeddings=True,
)


def test_detects_legacy_pipeline_layers() -> None:
    """Baseline: the pre-existing, always-correct case -- a model with
    legacy PipelineFirstLayer/PipelineLastLayer installed IS detected
    as pipeline-communicating."""
    r0 = LlamaModel(_ARGS)
    r1 = LlamaModel(_ARGS)
    _build_legacy_split(r0, r1)
    assert (
        _has_pipeline_communication_layer(cast("_ExoModel", cast("object", r0))) is True
    )
    assert (
        _has_pipeline_communication_layer(cast("_ExoModel", cast("object", r1))) is True
    )


def test_detects_metaframe_pipeline_layers() -> None:
    """THE regression test: a model with MetaFramedPipelineFirstLayer/
    MetaFramedPipelineLastLayer installed (EXO_PP_METAFRAME=1's real
    production layer classes) MUST ALSO be detected as pipeline-
    communicating -- confirmed BROKEN (always False) before this
    session's fix, which would have silently routed every real
    metaframe prefill through stream_generate() instead of
    pipeline_parallel_prefill(), making the whole chunked-prefill
    interruption mechanism built this session structurally
    unreachable regardless of anything else fixed."""
    r0 = LlamaModel(_ARGS)
    r1 = LlamaModel(_ARGS)
    _build_legacy_split(r0, r1)
    group0 = cast(mx.distributed.Group, cast(object, _RankGroup(0, 2)))
    group1 = cast(mx.distributed.Group, cast(object, _RankGroup(1, 2)))
    install_metaframed_pipeline_layers(r0, group0, request_uid=1)
    install_metaframed_pipeline_layers(r1, group1, request_uid=1)

    assert (
        _has_pipeline_communication_layer(cast("_ExoModel", cast("object", r0))) is True
    ), (
        "a model with MetaFramedPipelineFirstLayer/LastLayer installed "
        "must be detected as pipeline-communicating -- if this is False, "
        "prefill() silently routes through stream_generate() instead of "
        "pipeline_parallel_prefill(), and chunked-prefill interruption "
        "never runs at all"
    )
    assert (
        _has_pipeline_communication_layer(cast("_ExoModel", cast("object", r1))) is True
    )


def test_plain_unsharded_model_has_no_pipeline_layer() -> None:
    """Negative case: a plain, unsharded model (no PP layers installed
    at all -- e.g. a single-node run) correctly reports False, not a
    false positive from the broadened check."""
    model = LlamaModel(_ARGS)
    assert (
        _has_pipeline_communication_layer(cast("_ExoModel", cast("object", model)))
        is False
    )
