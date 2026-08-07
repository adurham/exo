# pyright: reportPrivateUsage=false, reportAny=false
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportArgumentType=false
"""Gate-safety regression test for ``prefill_interruptible_start``
(2026-08-07, Phase 2 live-wiring; UPDATED 2026-08-07 same day, mlx-lm
submodule pin advanced).

HISTORY: the ``mlx-lm`` submodule pin was originally
``55401ac57c7d7787c4efe97852b66254da15b565``, which did NOT include
``DeepseekV4Model._forward_steps`` -- that generator-core split
existed only on the fork's ``pp-layer-segment-wip`` branch (commit
``26eb90f0b``), built off that old pin rather than the fork's
then-current ``main`` (~20 unrelated commits diverged at the time).
Per explicit user + `consult` direction, that gap was deliberately
left open in the session that built the chunked-prefill live-wiring,
to keep an unrelated submodule rebase out of the same review unit as
two subtle hazard fixes.

RESOLVED same day: ``git cherry-pick 26eb90f0b`` (verified
self-contained, single-file, zero merge conflicts) landed cleanly on
top of the fork's current ``origin/main`` at commit ``8df20cd7b``, the
mlx-lm submodule pin was advanced to that commit, and ``exo``'s own
submodule gitlink was bumped to match -- see this session's design doc
entry for the full verification trail (ruff/basedpyright diffed
against baseline pre/post cherry-pick, real module import +
structural checks against the ACTUAL installed package, not a
synthetic stand-in).

This test now proves the OPPOSITE of its original assertion:
``supports_chunked_prefill_interruption`` against the REAL, currently
pinned ``DeepseekV4Model`` returns ``True``, and
``prefill_interruptible_start`` is consequently no longer a
structural no-op on the real, currently-pinned mlx-lm -- the
NEXT gating item is real 2-node cluster deployment + validation
(``start_cluster.sh``'s own documented submodule-pin re-install step),
not a code-level gap.
"""

from __future__ import annotations

from exo.worker.engines.mlx.generator.generate import prefill_interruptible_start
from exo.worker.engines.mlx.pp_prefill_session import (
    supports_chunked_prefill_interruption,
)


def test_currently_pinned_mlx_lm_has_forward_steps_on_deepseek_v4() -> None:
    """Direct proof the submodule-pin follow-up landed: the REAL,
    installed ``mlx_lm`` package (matching exo's actual current
    submodule pin, post-cherry-pick) now HAS ``_forward_steps`` on
    ``DeepseekV4Model`` -- confirmed against the real model class, not
    assumed. This is the reason ``prefill_interruptible_start`` is no
    longer a guaranteed no-op for DeepSeek-V4 on real hardware (see
    that function's own docstring). If this test ever starts FAILING
    again (``_forward_steps`` missing), that is the signal the
    submodule pin has regressed/been reset and needs re-investigation.
    """
    from mlx_lm.models.deepseek_v4 import DeepseekV4Model
    from mlx_lm.models.llama import Model as LlamaModel

    assert hasattr(DeepseekV4Model, "_forward_steps"), (
        "the currently-pinned mlx_lm package unexpectedly LACKS "
        "_forward_steps on DeepseekV4Model -- the submodule pin may "
        "have regressed; see "
        "docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md's "
        "2026-08-07 entry for the expected state"
    )
    assert supports_chunked_prefill_interruption(DeepseekV4Model) is True
    # Llama was never part of the _forward_steps split (DSv4-only) --
    # confirms this isn't a false-positive from some unrelated
    # structural-check bug that would accidentally pass EVERY model.
    assert not hasattr(LlamaModel, "_forward_steps")
    assert supports_chunked_prefill_interruption(LlamaModel) is False


def test_prefill_interruptible_start_returns_none_for_non_pipeline_sharded_call() -> (
    None
):
    """A SEPARATE safety guarantee from the pin-gate test above:
    ``prefill_interruptible_start`` must return ``None`` (never raise,
    never ``AttributeError``) when the call itself isn't eligible for
    chunked interruption -- independent of whether the loaded model
    class structurally supports it. Proven here by calling with
    ``group=None`` (single-rank, the cheapest way to guarantee
    ineligibility without needing a real distributed group or a real
    pipeline-sharded model), confirming it short-circuits cleanly on
    the very FIRST eligibility check
    (``is_pipeline and num_tokens >= prefill_step_size``) without ever
    reaching the ``supports_chunked_prefill_interruption`` check or
    touching ``model.layers``/``get_inner_model`` at all -- the real
    call shape ``_run_deferred_prefill_for_grant`` uses in production,
    just with inputs guaranteed to hit the earliest possible ``None``
    return. This guarantee holds REGARDLESS of the submodule pin's
    ``_forward_steps`` availability (unlike the test above), since
    single-rank/non-pipeline calls are structurally excluded before
    that check is ever reached.
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
        "for a non-pipeline-sharded/single-rank call -- this is a "
        "SEPARATE, always-on safety net independent of the submodule "
        "pin's _forward_steps availability"
    )
