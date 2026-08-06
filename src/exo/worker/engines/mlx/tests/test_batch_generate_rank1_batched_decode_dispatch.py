# pyright: reportUnknownVariableType=false, reportAny=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownLambdaType=false, reportPrivateUsage=false
# pyright: reportArgumentType=false, reportInvalidCast=false
"""Regression test for the SECOND real bug found on the cluster
(2026-08-05, after the batch_step_scope fallback fix): rank 1's
submit() dispatch gate only checked ``_batched_decode_rank0_glue is
not None`` -- always False on rank 1, so rank 1 fell through to the
old serial ``_mlx_gen.insert()`` path and NEVER called
``Rank1BatchedDecodeGlue.stage_local_cache``. When rank 0's admission
for that request arrived over the wire, rank 1's glue had no staged
cache to bind it to and raised GlueError -- both runners crashed on
the very first real request.

This test constructs a real ExoBatchGenerator with a REAL
Rank1BatchedDecodeGlue attached (mirroring what utils_mlx.py's
__post_init__ branch builds on an actual rank-1 process) and proves
submit() dispatches into ``_submit_batched_decode`` -> the
``stage_local_cache`` branch, not the old serial insert() path.
"""

from __future__ import annotations

from typing import cast

import mlx.core as mx
import mlx.nn as nn
import mlx.utils
from mlx_lm.tokenizer_utils import TokenizerWrapper
from transformers import AutoTokenizer

from exo.shared.types.common import ModelId
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.worker.engines.mlx.generator.batch_generate import ExoBatchGenerator
from exo.worker.engines.mlx.pp_batched_decode_glue import Rank1BatchedDecodeGlue
from exo.worker.engines.mlx.pp_batched_decode_runtime import RankOneMirrorSession


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
    """Minimal mx.distributed.Group stand-in -- Rank1BatchedDecodeGlue
    only stores it (never calls send/recv in this test, since we only
    exercise submit()'s dispatch/staging, not a real tick())."""

    def rank(self) -> int:
        return 1

    def size(self) -> int:
        return 2


def test_rank1_submit_dispatches_to_stage_local_cache_not_serial_insert() -> None:
    """THE REGRESSION TEST for the second real cluster bug: with a
    REAL Rank1BatchedDecodeGlue attached (mirroring an actual rank-1
    process) and EXO_PP_BATCHED_DECODE=1, submit() must call
    stage_local_cache for the request rather than falling through to
    the old serial _mlx_gen.insert() path. Before the fix, the
    dispatch gate's ``self._batched_decode_rank0_glue is not None``
    check was always False here (this glue lives on
    ``_batched_decode_rank1_glue`` instead), so the request went
    through _mlx_gen.insert() and stage_local_cache was NEVER
    called -- exactly reproducing the real
    ``GlueError: ... has no staged local prefilled cache`` crash.

    Patches ``mx.distributed.all_sum`` for the duration of the call
    (identity pass-through) -- ``submit()``'s own ``prefill()`` call
    unconditionally invokes ``mx_barrier`` whenever ``group is not
    None`` (a real cross-rank collective this single-process test has
    no real peer for); this test's fake group only needs to satisfy
    the eligibility gate's own ``group.size() > 1`` check, not
    participate in a genuine collective -- same rationale as this
    session's established ``pp_batched_correctness.py``-style
    send/recv patching for other single-process glue-layer tests.
    """
    from unittest.mock import patch

    model = _make_tiny_llama()
    tokenizer = _make_tokenizer()

    gen = ExoBatchGenerator(
        model=model,
        tokenizer=tokenizer,
        group=cast(mx.distributed.Group, _FakeGroup()),
        kv_prefix_cache=None,
    )

    # Simulate what utils_mlx.py's __post_init__ branch builds on a
    # real rank-1 process (EXO_PP_BATCHED_DECODE=1, rank != 0) --
    # constructed directly here since this test targets submit()'s
    # OWN dispatch logic, not the load-time construction branch
    # (already covered by test_batch_generate_batched_decode_flag_off_smoke.py
    # for the flag-OFF case).
    session = RankOneMirrorSession.new(max_concurrency=2)
    glue = Rank1BatchedDecodeGlue(
        session=session, src_rank=0, group=cast(mx.distributed.Group, _FakeGroup())
    )
    gen._batched_decode_active = True
    gen._batched_decode_rank1_glue = glue
    gen._batched_decode_rank0_glue = None  # explicit: this IS rank 1

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
        uid = gen.submit(task_params, "What is the capital of France?")

    # The real assertion: stage_local_cache was called for this uid
    # (proves submit() took the batched-decode branch, not the old
    # serial insert() path) -- checked via the glue's own internal
    # staging dict, which is exactly what tick() would consult next.
    assert uid in glue._staged_local_caches
    assert glue._staged_slot_for_request[uid] == 0

    # And the OLD path must NOT have been taken: this uid must be
    # tracked in _active_tasks (the batched-decode registration path
    # in _submit_batched_decode), never handed to the old serial
    # mlx-lm BatchGenerator's own insert() at all -- if the bug were
    # still present, submit() would have fallen through past the
    # dispatch gate entirely and stage_local_cache above would never
    # have been called (the assertion on _staged_local_caches already
    # would have failed first).
    assert uid in gen._active_tasks
