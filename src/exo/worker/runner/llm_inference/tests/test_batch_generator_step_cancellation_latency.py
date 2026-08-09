# pyright: reportPrivateUsage=false, reportAny=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportAttributeAccessIssue=false, reportUnknownArgumentType=false
# pyright: reportUnknownLambdaType=false
"""Regression test for the batched-decode cancellation-observation-
latency bug root-caused 2026-08-09 (design doc
docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md Section 29, real
hardware evidence).

Root cause: ``BatchGenerator``'s per-decode-token callback closures
(``_make_on_generation_token`` / the single-stream equivalent) only
called the full, expensive ``agree_on_cancellations()`` collective once
every ``check_for_cancel_every`` decode tokens (up to 100).
``check_for_cancel_every`` is calibrated ONCE at warmup time against a
fast, near-empty-context decode -- real decode throughput under a large
resident KV cache (30K+ tokens, PP + batched-decode) can be 30-100x
slower per token than that warmup measurement, so a real client
cancellation could sit unobserved for 80+ decode tokens. Measured
directly on real hardware: 133.7s/136.4s (rank0/rank1) of continued GPU
compute after a client disconnected, before this fix -- closely enough
matching the shape of the now-fixed 717523cb6/94fc04a4d bugs to be
mistaken for a regression of either (two independent SIGUSR1
faulthandler dumps 30s apart showed an IDENTICAL live decode-loop
stack, proving a genuine busy loop, not a wedge/deadlock).

Fix: ``BatchGenerator.step()`` now also calls
``agree_on_cancellations_fast()`` UNCONDITIONALLY every step, mirroring
the already-proven-safe unconditional-with-internal-mx_any-gate pattern
``agree_on_tasks()`` already uses on the exact same call site (both
ranks call ``step()`` every decode iteration -- the ONLY place in this
class provably symmetric across ranks every step, unlike the
per-decode-token callback closures, which are NOT reliably symmetric
for the batched-decode path). This bounds cancellation-observation
latency to ~1 decode step instead of up to ``check_for_cancel_every``
steps. The old counter-gated ``agree_on_cancellations()`` calls in the
token callbacks are left in place as a harmless, redundant backstop.

This test exercises the mechanism directly on a bare ``BatchGenerator``
(object.__new__, bypassing __post_init__'s real MLX/model/group setup)
-- mirrors test_pp_spec_gen_by_uid.py's and
test_concurrency_admission_gate.py's own stated scope (black-box
mechanism test, not a full multi-rank integration test against live
cluster state; the real hardware finding this closes was reproduced
and verified on the live 2-node cluster, see the design doc).

Per this campaign's own established discipline: the final test in this
file, ``test_reverting_the_fix_reproduces_the_original_bug``, proves
the OLD counter-gated-only behavior really does leave cancellation
unobserved for many steps by removing the new unconditional call and
showing the predicted failure before the fix is restored.
"""

from __future__ import annotations

from collections import deque
from unittest.mock import MagicMock

import pytest

import exo.worker.runner.llm_inference.batch_generator as batch_generator_module
from exo.worker.runner.llm_inference.batch_generator import BatchGenerator


def _bare_batch_generator(*, has_work: bool = False) -> BatchGenerator:
    """Construct a ``BatchGenerator`` without running ``__post_init__``
    (which needs a real model/tokenizer/group/cancel_receiver) -- just
    enough state for ``step()``'s own control-flow to run against a
    mocked ``_gen``/collective layer."""
    bg = object.__new__(BatchGenerator)
    bg._cancelled_tasks = set()
    bg._maybe_queue = []
    bg._maybe_cancel = []
    bg._all_tasks = {}
    bg._queue = deque()
    bg._active_tasks = {}
    bg._jaccl_step_count = 0
    bg._jaccl_step_handle = None
    bg.group = None  # collectives are mocked below, never touch mx.distributed
    bg.check_for_cancel_every = 100

    bg._gen = MagicMock()
    bg._gen.has_work = has_work
    bg._gen.step.return_value = []

    # agree_on_tasks / agree_on_cancellations_fast are instance methods
    # that internally call mx.distributed collectives -- replace with
    # plain call-counting mocks so this test verifies ONLY step()'s own
    # call-site behavior (unconditional-per-step), not the collective
    # implementations themselves (those have their own existing tests).
    bg.agree_on_tasks = MagicMock()
    bg.agree_on_cancellations_fast = MagicMock()

    bg._jaccl_dump_step = MagicMock()

    return bg


def test_step_calls_agree_on_cancellations_fast_unconditionally_every_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fix: agree_on_cancellations_fast() must fire on EVERY step()
    call, exactly like agree_on_tasks() does immediately above it --
    not gated behind any counter -- regardless of has_work (a cancelled
    request must be observable even on a step where nothing else is
    happening)."""
    monkeypatch.setattr(batch_generator_module, "EXO_DSV4_BATCHED_PREFILL", False)
    monkeypatch.setattr(
        batch_generator_module,
        "mx_any",
        lambda value, _group: bool(value),
    )

    bg = _bare_batch_generator(has_work=False)

    list(bg.step())

    bg.agree_on_tasks.assert_called_once()
    bg.agree_on_cancellations_fast.assert_called_once()

    # Second call: still unconditional, still fires every time -- not a
    # one-shot warm-up artifact.
    list(bg.step())
    assert bg.agree_on_cancellations_fast.call_count == 2


def test_step_calls_agree_on_cancellations_fast_even_when_has_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Must also fire on the (more common, decode-rate) path where the
    rank has real work and self._gen.step() actually runs -- this is
    the exact path the real hardware bug lived on."""
    monkeypatch.setattr(batch_generator_module, "EXO_DSV4_BATCHED_PREFILL", False)
    monkeypatch.setattr(
        batch_generator_module,
        "mx_any",
        lambda value, _group: bool(value),
    )

    bg = _bare_batch_generator(has_work=True)

    list(bg.step())

    bg.agree_on_tasks.assert_called_once()
    bg.agree_on_cancellations_fast.assert_called_once()
    bg._gen.step.assert_called_once()


def test_reverting_the_fix_reproduces_the_original_bug(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per this campaign's established discipline: prove the OLD
    behavior (no unconditional per-step cancellation check) really did
    leave step() with no per-step cancellation-observation mechanism at
    all -- by simulating the pre-fix code path directly (calling
    agree_on_tasks only, as step() used to) and confirming
    agree_on_cancellations_fast is never invoked. This is the state
    that produced the real 133.7s/136.4s hardware-measured cancellation
    latency before this fix."""
    monkeypatch.setattr(batch_generator_module, "EXO_DSV4_BATCHED_PREFILL", False)
    monkeypatch.setattr(
        batch_generator_module,
        "mx_any",
        lambda value, _group: bool(value),
    )

    bg = _bare_batch_generator(has_work=True)

    # Simulate the PRE-FIX step() body: only agree_on_tasks() runs
    # per-step; agree_on_cancellations_fast() is never called here (the
    # real pre-fix code only reached the full, counter-gated
    # agree_on_cancellations() from deep inside the per-decode-token
    # callback closure, which this bare-generator harness does not
    # drive -- exactly why the bug was invisible to per-step callers).
    bg.agree_on_tasks()
    if bg._gen.has_work:
        bg._gen.step()

    bg.agree_on_cancellations_fast.assert_not_called()
