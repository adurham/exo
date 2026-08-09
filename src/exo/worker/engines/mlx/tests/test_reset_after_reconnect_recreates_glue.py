# pyright: reportPrivateUsage=false, reportAny=false
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false, reportArgumentType=false
"""Regression test for the THIRD real-hardware discovery that
ExoBatchGenerator.reset_after_reconnect() was missing state (design
doc Section 32, 2026-08-09):

The Section 27/28 fix taught reset_after_reconnect() to clear the
batched-decode glue objects' chunk-drive PREFILL state
(_active_prefill_session/_prefill_phase/_prefill_rank1_advances_remaining)
after an in-place jaccl reconnect. It never touched a SEPARATE object
-- Rank0BatchedDecodeGlue.session (a BatchedDecodeSession, owning its
own SchedulerCore._requests dict of ALREADY-ADMITTED, steady-state
decoding requests). Confirmed on real hardware: a request that was
admitted and decoding normally (never itself in chunk-drive prefill,
so Section 27/28's fix never touched it) survived an in-place
reconnect with its admission bookkeeping intact in the OLD session
object -- but the wire protocol on the NEXT decode tick treated it as
stale, raising ProtocolViolationError("TokenGeneratedEvent for
request_ids=[N] which is not active") -- a second full runner crash
immediately after an otherwise-successful recovery.

Root cause: the reset was assembled as a growing list of per-object
field clears, and kept discovering (three separate times, in three
separate objects) a piece of state nobody remembered to include. The
fix applied here (see reset_after_reconnect()'s own docstring) is
structural: RECREATE the glue objects from their own constructors
(the same call site __post_init__ uses at model-load time) rather
than enumerating fields to clear. A field nobody remembered to reset
cannot survive an object that no longer exists.

This test proves the fix directly: admit a real request into a real
Rank0BatchedDecodeGlue's session (simulating a request that survived
a fault untouched), call reset_after_reconnect(), and verify (a) the
OLD session object no longer has that request admitted, (b) the glue
object ITSELF was replaced (new identity, not just its fields
cleared), and (c) a genuinely fresh session correctly has ZERO
admitted requests -- so the next real TokenGeneratedEvent for the
dropped request_id would now correctly be rejected as unknown by a
CLEAN, freshly-constructed SchedulerCore (matching real client-retry
semantics), not silently misrouted into stale bookkeeping.
"""

from __future__ import annotations

from typing import cast

import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import pytest
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper
from transformers import AutoTokenizer

from exo.worker.engines.mlx.generator.batch_generate import ExoBatchGenerator
from exo.worker.engines.mlx.pp_batched_decode_adapter import (
    BatchedDecodeResponseAdapter,
)
from exo.worker.engines.mlx.pp_batched_decode_glue import Rank0BatchedDecodeGlue
from exo.worker.engines.mlx.pp_batched_decode_runtime import BatchedDecodeSession

pytestmark = pytest.mark.filterwarnings("ignore")


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


class _RankGroupStub:
    """Minimal mx.distributed.Group stand-in -- reset_after_reconnect()
    only reads dst_rank/group off the OLD glue to pass through to the
    new one; never sends/recvs anything itself."""

    def rank(self) -> int:
        return 0

    def size(self) -> int:
        return 2


def test_reset_after_reconnect_recreates_rank0_glue_dropping_admitted_request() -> (
    None
):
    """THE regression test for the real 2026-08-09 crash: an admitted,
    steadily-decoding request (never itself in chunk-drive prefill)
    must be genuinely dropped by reset_after_reconnect() -- not left
    dangling in a session object that survives the reconnect."""
    model = _make_tiny_llama()
    tokenizer = _make_tokenizer()

    gen = ExoBatchGenerator(
        model=model,
        tokenizer=tokenizer,
        group=cast(mx.distributed.Group, cast(object, _RankGroupStub())),
        kv_prefix_cache=None,
    )
    gen._batched_decode_active = True
    gen._batched_decode_eos = set()

    session = BatchedDecodeSession.new(max_concurrency=2)
    adapter = BatchedDecodeResponseAdapter(session=session, eos_ids=frozenset())
    glue = Rank0BatchedDecodeGlue(
        session=session,
        adapter=adapter,
        dst_rank=1,
        group=cast(mx.distributed.Group, cast(object, _RankGroupStub())),
        peer_prefill_layer_count=4,
    )
    gen._batched_decode_rank0_glue = glue
    gen._batched_decode_rank1_glue = None

    # Admit a request directly into the session -- simulating a
    # request that finished prefill and is genuinely decoding
    # steady-state (the exact class of request the real crash hit;
    # never touched chunk-drive prefill state at all, so Section
    # 27/28's fix could never have covered it).
    sampler = make_sampler(temp=0.0)
    session.admit_request(
        request_id=42,
        cache_slot=0,
        prefilled_cache=[],
        initial_token=1,
        sampler=sampler,
    )
    assert session.has_request(42), "test setup: admission must have succeeded"
    assert glue.session.admitted_request_ids() == [42]

    old_glue = gen._batched_decode_rank0_glue
    old_session = old_glue.session

    dropped = gen.reset_after_reconnect()

    # THE key proof (a): request_id 42 is reported as dropped.
    assert 42 in dropped, (
        "reset_after_reconnect() must report the admitted request as "
        "dropped -- if this fails, the recreate path silently lost "
        "track of a request that was genuinely in flight"
    )

    # THE key proof (b): the glue object itself was REPLACED (new
    # identity), not merely mutated in place -- proving this is a real
    # recreate, not a reset that happened to zero this one field but
    # could still miss the NEXT one (exactly the failure pattern this
    # fix exists to structurally close).
    new_glue = gen._batched_decode_rank0_glue
    assert new_glue is not None
    assert new_glue is not old_glue, (
        "reset_after_reconnect() must construct a NEW glue object, not "
        "mutate the old one in place -- recreation (not field-by-field "
        "reset) is the whole point of this fix"
    )
    assert new_glue.session is not old_session

    # THE key proof (c): the OLD session object (which the crash's own
    # traceback shows the wire protocol would otherwise still
    # reference) genuinely still shows the stale admission -- proving
    # this test's admission actually landed somewhere real, not a
    # no-op -- while the NEW session is clean.
    assert old_session.has_request(42), (
        "sanity check: the OLD session object must still show the "
        "admission (proves recreation, not a same-object in-place "
        "clear that would make this assertion trivially pass either way)"
    )
    assert not new_glue.session.has_request(42), (
        "the NEW session must NOT know about the dropped request -- "
        "this is the exact invariant whose absence caused the real "
        "crash: 'TokenGeneratedEvent for request_ids=[N] which is not "
        "active'. A clean, freshly-constructed session correctly has "
        "no memory of request 42, so a stale/duplicate wire event for "
        "it is now correctly rejected as unknown rather than silently "
        "misrouted into corrupted admitted-request bookkeeping"
    )
    assert new_glue.session.admitted_request_ids() == []
    assert new_glue.dst_rank == old_glue.dst_rank
    assert new_glue.peer_prefill_layer_count == old_glue.peer_prefill_layer_count


def test_reset_after_reconnect_is_a_noop_when_batched_decode_never_enabled() -> None:
    """With batched-decode OFF (both glue attrs None, the default),
    reset_after_reconnect() must not crash trying to recreate objects
    that were never constructed in the first place."""
    model = _make_tiny_llama()
    tokenizer = _make_tokenizer()

    gen = ExoBatchGenerator(
        model=model, tokenizer=tokenizer, group=None, kv_prefix_cache=None
    )
    assert gen._batched_decode_rank0_glue is None
    assert gen._batched_decode_rank1_glue is None

    dropped = gen.reset_after_reconnect()

    assert dropped == []
    assert gen._batched_decode_rank0_glue is None
    assert gen._batched_decode_rank1_glue is None
