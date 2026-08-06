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
submit() dispatches into the batched-decode path, not the old serial
insert() path.

UPDATED 2026-08-06 for the N=2 admission-race fix (see
pp_batched_decode_glue.py's module docstring, "UPDATE (2026-08-06...)"
section, for the full architecture): rank 1's submit() no longer calls
stage_local_cache SYNCHRONOUSLY -- that would just recreate a milder
version of the very race this fix closes (rank 1 independently
deciding to run prefill on its own schedule). Instead, submit() now
registers a DEFERRED prefill (``_deferred_prefill_by_uid``) and
returns; ``stage_local_cache`` only runs once rank 1 reactively
receives a real ``PrefillMessage`` (a ``PrefillGrant`` from
``Rank1BatchedDecodeGlue.tick()``) telling it rank 0 has decided to
admit this request. This test's assertion is therefore now COMPOSED,
matching the whole path end-to-end (submit() -> deferred registration
-> real PrefillGrant delivered via a real tick() -> stage_local_cache)
rather than a single synchronous check -- see the consult review
recorded 2026-08-06: testing only "deferred and not yet staged" would
NOT catch the modern equivalent of the original bug (a grant that
silently fails to route to staging); this test drives the real
grant-servicing path (``_run_deferred_prefill_for_grant``, the SAME
method ``_step_batched_decode`` calls on a real tick()-returned grant)
with an injected ``PrefillGrant``, not a private closure shortcut, so
a miswired grant dispatch still fails it the same way the original
bug would have.
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
from exo.worker.engines.mlx.pp_batched_decode_glue import (
    PrefillGrant,
    Rank1BatchedDecodeGlue,
)
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


def test_rank1_submit_registers_deferred_prefill_then_grant_stages_cache() -> None:
    """THE REGRESSION TEST for the second real cluster bug, updated for
    the 2026-08-06 admission-race fix's new (correct) two-phase
    contract.

    Phase 1: with a REAL Rank1BatchedDecodeGlue attached (mirrors an
    actual rank-1 process), submit() must register a deferred prefill
    for this uid -- NOT call stage_local_cache synchronously (that
    synchronous call is exactly the independent-per-rank decision this
    fix eliminates) -- and must NOT fall through to the old serial
    _mlx_gen.insert() path (uid must land in _active_tasks either way).

    Phase 2: delivering a real PrefillGrant for this uid (as
    Rank1BatchedDecodeGlue.tick() would produce upon reactively
    receiving rank 0's PrefillMessage) must cause the deferred prefill
    to run and stage_local_cache to be called -- proving the grant
    correctly routes through to staging, the modern equivalent of the
    original bug's failure mode (a request that never gets a staged
    cache to bind its admission to).
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

    # PHASE 1 assertion: submit() registered a DEFERRED prefill for
    # this uid and did NOT stage anything yet -- proves submit() took
    # the NEW batched-decode-deferred branch (not the old synchronous
    # stage_local_cache call, and not the old serial insert() path).
    assert uid in gen._deferred_prefill_by_uid, (
        "submit() must register a deferred prefill for an eligible "
        "request on rank 1 -- if this fails, either the eligibility "
        "gate rejected the request (falling through to the old serial "
        "path) or the dispatch gate regressed back to the pre-2026-08-06 "
        "shape"
    )
    assert uid not in glue._staged_local_caches, (
        "rank 1 must NEVER stage a cache synchronously inside submit() "
        "-- that is exactly the independent per-rank prefill decision "
        "this fix eliminates; staging must only happen reactively, "
        "after a real PrefillGrant is delivered via tick()"
    )
    assert uid in gen._active_tasks

    # PHASE 2: deliver a real PrefillGrant for this uid, exactly as
    # Rank1BatchedDecodeGlue.tick() would produce it upon reactively
    # receiving rank 0's PrefillMessage over the wire (constructed
    # directly here rather than driving a real 2-process transport --
    # the real-wire-transport case is already covered by
    # test_pp_admission_race_subprocess.py's genuinely-independent-
    # event-loop harness; this test's job is ExoBatchGenerator's own
    # grant-to-staging wiring, one layer up). Routed through the SAME
    # method (_run_deferred_prefill_for_grant) _step_batched_decode
    # itself calls on a real tick()-returned grant -- not a shortcut.
    deferred_cache_slot = gen._deferred_prefill_by_uid[uid].cache_slot
    grant = PrefillGrant(
        request_id=uid,
        cache_slot=deferred_cache_slot,
        n_prompt_tokens=1,
        single_request_fallback=False,
    )
    with patch("mlx.core.distributed.all_sum", side_effect=_identity_all_sum):
        gen._run_deferred_prefill_for_grant(grant, is_rank1=True)

    # THE core regression assertion, updated for the new contract:
    # the grant must have driven the deferred prefill to completion
    # and staged its result -- exactly the step the original bug
    # skipped entirely (silently falling through to the wrong path
    # with nothing ever staged).
    assert uid in glue._staged_local_caches
    assert glue._staged_slot_for_request[uid] == deferred_cache_slot
    assert uid not in gen._deferred_prefill_by_uid, (
        "the deferred prefill entry must be consumed (popped) once its "
        "grant has been serviced -- a lingering entry would mean this "
        "uid could be double-serviced by a later grant"
    )
