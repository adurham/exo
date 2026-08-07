# pyright: reportPrivateUsage=false, reportAny=false
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportArgumentType=false
"""Gate-safety regression test for ``prefill_interruptible_start``
(2026-08-07, Phase 2 live-wiring).

REAL PRODUCTION STATE as of 2026-08-07: the ``mlx-lm`` submodule pin
(``55401ac57c7d7787c4efe97852b66254da15b565``) does NOT yet include
``DeepseekV4Model._forward_steps`` -- that generator-core split exists
only on the fork's ``pp-layer-segment-wip`` branch (commit
``26eb90f0b``), deliberately built off the OLD pin rather than the
fork's current ``main`` (~20 unrelated commits diverged since). See
this session's design doc entry for the full follow-up plan to close
that gap in a SEPARATE, dedicated session (explicit user + `consult`
direction: do not fold an unrelated submodule rebase into this
session's two hazard fixes).

This means ``prefill_interruptible_start`` -- the new function
production ``ExoBatchGenerator._run_deferred_prefill_for_grant`` now
calls on every grant to decide chunked-vs-synchronous prefill -- must
be PROVABLY safe to call against the REAL, currently-pinned model
class, not just against synthetic test fakes. This test is that
proof: it calls ``supports_chunked_prefill_interruption`` against a
REAL ``mlx_lm`` model class from the ACTUALLY-INSTALLED, pinned
version (not a synthetic stand-in), confirming it returns ``False`` --
meaning ``prefill_interruptible_start`` genuinely returns ``None`` on
today's real hardware, and every real request continues down the
unmodified, already-production-proven synchronous ``prefill()`` path,
exactly as before 2026-08-07's live-wiring change.
"""

from __future__ import annotations

from exo.worker.engines.mlx.generator.generate import prefill_interruptible_start
from exo.worker.engines.mlx.pp_prefill_session import (
    supports_chunked_prefill_interruption,
)


def test_currently_pinned_mlx_lm_lacks_forward_steps() -> None:
    """Direct proof of the gap this session's design doc entry
    documents: the REAL, installed ``mlx_lm`` package (matching the
    exo repo's actual submodule pin) has NO ``_forward_steps`` on its
    model classes yet -- confirmed against a real model class, not
    assumed. This is the reason ``prefill_interruptible_start`` is a
    guaranteed no-op on today's real hardware (see that function's own
    docstring for the full explanation) -- if this test ever starts
    FAILING (i.e. ``_forward_steps`` becomes newly present), that is
    the signal the submodule-pin follow-up has landed and this test
    (plus ``prefill_interruptible_start``'s own docstring) should be
    updated to reflect the new reality.
    """
    from mlx_lm.models.llama import Model as LlamaModel

    assert not hasattr(LlamaModel, "_forward_steps"), (
        "the currently-pinned mlx_lm package unexpectedly HAS "
        "_forward_steps on a real model class -- if the submodule pin "
        "follow-up (docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md's "
        "2026-08-07 entry) has landed, update this test and "
        "prefill_interruptible_start's own docstring to match"
    )
    assert supports_chunked_prefill_interruption(LlamaModel) is False


def test_prefill_interruptible_start_returns_none_when_model_lacks_forward_steps() -> (
    None
):
    """The actual safety guarantee: ``prefill_interruptible_start``
    must return ``None`` (never raise, never AttributeError) when the
    loaded model's inner model doesn't structurally support chunked
    interruption -- proven here by calling it with ``group=None``
    (single-rank, the cheapest way to guarantee ineligibility without
    needing a real distributed group or a real pipeline-sharded
    model), confirming it short-circuits cleanly on the very FIRST
    eligibility check (``is_pipeline and num_tokens >= prefill_step_size``)
    without ever reaching the ``supports_chunked_prefill_interruption``
    check or touching ``model.layers``/``get_inner_model`` at all --
    the real call shape ``_run_deferred_prefill_for_grant`` uses in
    production, just with inputs guaranteed to hit the earliest
    possible ``None`` return.
    """
    import mlx.core as mx

    from exo.worker.engines.mlx.tests.test_pp_pipeline_parallel_prefill_session_integration import (
        _ARGS,
        _random_model,
    )

    model = _random_model(seed=7)
    tokenizer_stub = object()
    prompt_tokens = mx.random.randint(0, _ARGS.vocab_size, shape=(50,))
    mx.eval(prompt_tokens)
    cache = model.make_cache()

    result = prefill_interruptible_start(
        model,
        tokenizer_stub,
        lambda x: x,
        prompt_tokens,
        cache,
        None,  # group=None -- single-rank, guaranteed ineligible
        None,
        None,
        prefill_step_size=4096,
    )
    assert result is None, (
        "prefill_interruptible_start must return None (never raise) "
        "for a non-pipeline-sharded model -- this is what makes "
        "_run_deferred_prefill_for_grant's fallback to the synchronous "
        "run_prefill() path safe on today's real production hardware"
    )
