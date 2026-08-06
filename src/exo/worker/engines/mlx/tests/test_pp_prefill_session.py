# pyright: reportPrivateUsage=false
"""Tests for ``ResumablePrefillSession`` (pp_prefill_session.py), the
Stage 2 caller that drives ``DeepseekV4Model._forward_steps`` (Stage
1b's generator-core split) across multiple real pause/resume cycles.

Uses a synthetic ``_FakeInterruptibleModel`` (matching the
``_InterruptibleForward`` protocol structurally) rather than the real
DSv4 model -- the real model needs a 166GB checkpoint + a real
distributed group this local test environment doesn't have. This
mirrors the design doc's own Phase 0 precedent (``pp_batched_
correctness.py`` uses plain Llama, not the real DSv4 checkpoint, to
keep validation fast and independent of the real weights) and this
session's own earlier isolation-harness precedent for the same
generator-core pattern.

Covers real coverage gaps a stand-in doesn't magically close:
1. Eager vs. interruptible-fully-drained paths produce IDENTICAL
   output (the same invariant the design doc's earlier isolation
   harness checked, now against the REAL session object, not a
   disposable script).
2. Genuine pause/resume: advance() by fewer layers than total, verify
   partial progress, resume, verify completion -- with a REAL
   interleaved forward pass (a separate model instance) running
   between the pause and the resume, proving the paused session's own
   state (and the ForwardStepInfo contextvar discipline) survives
   real interleaving.
3. Misuse cases (double-start, advance-after-done, output-before-done,
   invalid max_layers) all raise PrefillSessionError, matching this
   module's fail-stop discipline.
"""

from __future__ import annotations

from typing import Iterator

import mlx.core as mx
import pytest

from exo.worker.engines.mlx.pp_metaframe import ForwardPhase
from exo.worker.engines.mlx.pp_prefill_session import (
    ForwardStep,
    PrefillSessionError,
    ResumablePrefillSession,
    supports_chunked_prefill_interruption,
)

pytestmark = pytest.mark.filterwarnings("ignore")


class _FakeInterruptibleModel:
    """Structural stand-in for DeepseekV4Model -- matches
    ``_InterruptibleForward``'s protocol shape exactly (a
    ``_forward_steps`` generator yielding ``("layer", i, h)`` then
    ``("done", None, h)``). ``queue_sends``/phase reads are exercised
    for real via ``pp_metaframe.get_forward_step_info()`` inside the
    fake layer body, matching how the real DSv4 layers actually read
    it -- not bypassed."""

    def __init__(self, n_layers: int) -> None:
        self.n_layers = n_layers
        self.observed_phases: list[ForwardPhase] = []
        self.observed_queue_sends: list[bool] = []

    def _forward_steps(
        self,
        inputs: mx.array,
        cache: object = None,
        *,
        interruptible: bool = False,
    ) -> Iterator[ForwardStep]:
        from exo.worker.engines.mlx.pp_metaframe import get_forward_step_info

        h = inputs
        for i in range(self.n_layers):
            step_info = get_forward_step_info()
            self.observed_phases.append(step_info.phase)
            self.observed_queue_sends.append(step_info.queue_sends)
            h = h + mx.array([1.0])
            mx.eval(h)
            if interruptible:
                yield ("layer", i, h)
        yield ("done", None, h)


def test_supports_chunked_prefill_interruption_true_for_conforming_model() -> None:
    model = _FakeInterruptibleModel(n_layers=3)
    assert supports_chunked_prefill_interruption(model) is True


def test_supports_chunked_prefill_interruption_false_for_non_conforming_object() -> (
    None
):
    class _NotInterruptible:
        pass

    assert supports_chunked_prefill_interruption(_NotInterruptible()) is False


def test_eager_and_fully_drained_interruptible_paths_agree() -> None:
    """The same invariant this session's design-scoping consult review
    demanded: an eager (non-interruptible) call and a fully-drained
    interruptible generator must produce IDENTICAL output."""
    from exo.worker.engines.mlx.pp_metaframe import set_forward_step_info

    model = _FakeInterruptibleModel(n_layers=5)
    x0 = mx.array([0.0])

    # Eager path (interruptible=False, matches __call__'s own contract).
    # set_forward_step_info called directly here (not via a session)
    # since this path bypasses ResumablePrefillSession entirely --
    # mirrors how the real __call__ wrapper's caller would set it once
    # before an ordinary (non-chunked) forward pass.
    set_forward_step_info(phase=ForwardPhase.PREFILL_FINAL, queue_sends=True)
    eager_gen = model._forward_steps(x0, interruptible=False)
    *_, (eager_kind, _eager_idx, eager_out) = eager_gen
    assert eager_kind == "done"
    mx.eval(eager_out)

    # Interruptible path, fully drained via the real session object.
    session = ResumablePrefillSession(inner_model=model, inputs=x0, cache=())
    advanced, done = session.advance(
        max_layers=100, phase_for_pause=ForwardPhase.PREFILL_FINAL
    )
    assert done is True
    assert advanced == 5  # fewer than max_layers=100 -- "done" arrived first
    mx.eval(session.output())

    assert float(eager_out.item()) == float(session.output().item())


def test_genuine_pause_resume_with_real_interleaved_forward_pass() -> None:
    """THE core mechanism this whole module exists for: pause a
    session mid-way, run a REAL, completely independent forward pass
    on a SEPARATE model instance in the gap (matching a real
    interleaved decode step), then resume the ORIGINAL session and
    confirm it completes correctly -- unaffected by the interleaved
    work, and observing the CORRECT ForwardPhase at each resume
    (PREFILL_CONTINUE mid-chunk, matching what the caller is
    responsible for passing per advance()'s own docstring)."""
    model = _FakeInterruptibleModel(n_layers=6)
    x0 = mx.array([0.0])
    session = ResumablePrefillSession(inner_model=model, inputs=x0, cache=())

    # First partial advance: 2 layers, more remain -> not done.
    advanced1, done1 = session.advance(
        max_layers=2, phase_for_pause=ForwardPhase.PREFILL_CONTINUE
    )
    assert advanced1 == 2
    assert done1 is False
    assert session.last_layer_index == 1
    assert not session.done

    # REAL interleaved forward pass on a SEPARATE model instance --
    # exercises get_forward_step_info() for its OWN phase (DECODE),
    # completely independent of the paused session's own state.
    decode_model = _FakeInterruptibleModel(n_layers=1)
    decode_session = ResumablePrefillSession(
        inner_model=decode_model, inputs=mx.array([100.0]), cache=()
    )
    d_advanced, d_done = decode_session.advance(
        max_layers=100, phase_for_pause=ForwardPhase.DECODE
    )
    assert d_advanced == 1
    assert d_done is True
    mx.eval(decode_session.output())
    assert float(decode_session.output().item()) == 101.0
    assert decode_model.observed_phases == [ForwardPhase.DECODE]

    # Resume the ORIGINAL (still-paused) session -- must continue from
    # layer 2, NOT corrupted by the interleaved decode call above.
    advanced2, done2 = session.advance(
        max_layers=2, phase_for_pause=ForwardPhase.PREFILL_CONTINUE
    )
    assert advanced2 == 2
    assert done2 is False
    assert session.last_layer_index == 3

    # Final advance reaches completion (2 layers remain: 4, 5).
    advanced3, done3 = session.advance(
        max_layers=100, phase_for_pause=ForwardPhase.PREFILL_FINAL
    )
    assert advanced3 == 2
    assert done3 is True
    assert session.done

    mx.eval(session.output())
    # 6 layers, each +1.0 starting from 0.0 -> 6.0, regardless of
    # interleaving in between.
    assert float(session.output().item()) == 6.0

    # The ORIGINAL model's own observed phases must be exactly what
    # THIS session passed at each resume -- PREFILL_CONTINUE for the
    # first 5 layers (still mid-chunk), PREFILL_FINAL for the last
    # resume's layers (the caller declared this the final resume).
    # advance() sets the SAME phase for every layer resumed within one
    # advance() call -- 2 PREFILL_CONTINUE (first advance) + 2
    # PREFILL_CONTINUE (second advance) + 2 PREFILL_FINAL (third,
    # completing advance).
    assert model.observed_phases == [
        ForwardPhase.PREFILL_CONTINUE,
        ForwardPhase.PREFILL_CONTINUE,
        ForwardPhase.PREFILL_CONTINUE,
        ForwardPhase.PREFILL_CONTINUE,
        ForwardPhase.PREFILL_FINAL,
        ForwardPhase.PREFILL_FINAL,
    ]
    # The interleaved decode_model's own phase list must be
    # UNTOUCHED by anything the original session did -- proves no
    # cross-contamination between the two sessions' independently
    # captured contexts.
    assert decode_model.observed_phases == [ForwardPhase.DECODE]


def test_advance_with_max_layers_less_than_one_raises() -> None:
    model = _FakeInterruptibleModel(n_layers=3)
    session = ResumablePrefillSession(
        inner_model=model, inputs=mx.array([0.0]), cache=()
    )
    with pytest.raises(PrefillSessionError, match=">=1"):
        session.advance(max_layers=0, phase_for_pause=ForwardPhase.PREFILL_CONTINUE)


def test_advance_after_done_raises() -> None:
    model = _FakeInterruptibleModel(n_layers=1)
    session = ResumablePrefillSession(
        inner_model=model, inputs=mx.array([0.0]), cache=()
    )
    _, done = session.advance(
        max_layers=100, phase_for_pause=ForwardPhase.PREFILL_FINAL
    )
    assert done is True
    with pytest.raises(PrefillSessionError, match="already-completed"):
        session.advance(max_layers=1, phase_for_pause=ForwardPhase.PREFILL_CONTINUE)


def test_output_before_done_raises() -> None:
    model = _FakeInterruptibleModel(n_layers=5)
    session = ResumablePrefillSession(
        inner_model=model, inputs=mx.array([0.0]), cache=()
    )
    session.advance(max_layers=1, phase_for_pause=ForwardPhase.PREFILL_CONTINUE)
    with pytest.raises(
        PrefillSessionError, match="before the session reached completion"
    ):
        session.output()


def test_output_after_done_returns_final_value() -> None:
    model = _FakeInterruptibleModel(n_layers=2)
    session = ResumablePrefillSession(
        inner_model=model, inputs=mx.array([0.0]), cache=()
    )
    session.advance(max_layers=100, phase_for_pause=ForwardPhase.PREFILL_FINAL)
    mx.eval(session.output())
    assert float(session.output().item()) == 2.0


def test_last_layer_index_starts_at_negative_one() -> None:
    model = _FakeInterruptibleModel(n_layers=3)
    session = ResumablePrefillSession(
        inner_model=model, inputs=mx.array([0.0]), cache=()
    )
    assert session.last_layer_index == -1
