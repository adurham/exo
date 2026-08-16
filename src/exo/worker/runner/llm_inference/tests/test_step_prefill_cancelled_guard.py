"""Regression test: a cancel during DEFERRED prefill must not kill the runner.

THE BUG (design doc Section 58, found on real hardware 2026-08-15).
Cancelling a request whose prefill was in flight on the batched-decode
path terminated the entire runner process with an uncaught exception:

    runner.py:592           results = self.generator.step()
    batch_generator.py:838  results = self._gen.step()
    batch_generate.py:4155  responses = self._step_batched_decode()
    batch_generate.py:4024  self._run_deferred_prefill_for_grant(...)
    batch_generate.py:3813  deferred.run_prefill()
    batch_generate.py:1417  prefill(...)
    generate.py:848         for _ in _sg:
    generate.py:776         distributed_prompt_progress_callback()
    batch_generator.py:1119 raise PrefillCancelled()
    -> PrefillCancelled   [UNCAUGHT -> process death]

WHY IT SURVIVED REVIEW. `PrefillCancelled` subclasses `BaseException`,
not `Exception`, deliberately: an in-tree comment on
`PPSpecAlreadyActiveError` (batch_generate.py) contrasts the two,
noting that a plain `RuntimeError` subclass *is* caught by the runner's
generic `except Exception` handling and so "surfaces as a clean task
failure the caller can retry, not an uncaught crash." That design was
correct while the ONLY way to raise `PrefillCancelled` was from a
submit path that handled it explicitly -- and two such handlers do
exist (`_batched_start_task` and `_start_task`).

The deferred-prefill redesign then moved prefill's EXECUTION out of
`submit()` and into `tick()`-granted work reached from `step()`. The
exception's contract did not move with it. No handler existed on the
new path, and `except Exception` could not help by construction.

WHAT THIS TEST PINS. `step()` must translate `PrefillCancelled` into the
same outcome both submit-path handlers already chose -- the request goes
away, the runner survives -- rather than letting it propagate.

Deliberately tests at the seam (`BatchGenerator.step`) rather than
reproducing the whole 2-rank deferred-prefill handshake: the bug is
"this exception escapes this call", and that is exactly what is
asserted, with no cluster, no MLX, and no distributed group required.
"""

from __future__ import annotations

import re
from pathlib import Path

_BATCH_GENERATOR = Path(__file__).resolve().parents[1] / "batch_generator.py"


def _step_method_source() -> str:
    """Source text of the outer `def step(` that wraps `self._gen.step()`.

    The file defines more than one `step`; this selects the one whose
    body actually CALLS the engine. Matching on the bare string
    `self._gen.step()` is not enough -- several docstrings and comments
    mention it in prose, so an earlier version of this test selected
    `SequentialGenerator.step` (which merely references it in a comment)
    instead of `BatchGenerator.step`, and failed against a file that was
    already correctly patched. Require the call to appear as a real
    assignment statement.
    """
    source = _BATCH_GENERATOR.read_text()
    starts = [m.start() for m in re.finditer(r"\n    def step\(", source)]
    assert starts, "no `def step(` found -- test's parsing assumption broke"

    call_re = re.compile(r"^\s+results = self\._gen\.step\(\)", re.M)
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(source)
        body = source[start:end]
        if call_re.search(body):
            return body
    raise AssertionError(
        "no `def step(` containing a real `results = self._gen.step()` call"
    )


def test_engine_step_call_is_guarded_against_prefill_cancelled() -> None:
    """The `self._gen.step()` call must sit inside a PrefillCancelled guard.

    Without it, a cancel arriving during deferred prefill propagates out
    of `step()`, past the runner's `except Exception` (PrefillCancelled
    is a BaseException), and kills the process.
    """
    body = _step_method_source()

    # Locate the REAL call, not the first prose mention of it. The
    # method's docstrings/comments reference `self._gen.step()` several
    # times before the actual statement, so a naive `.index()` measures
    # from the wrong offset and reports a missing guard on a file that
    # has one.
    call = re.search(r"^\s+results = self\._gen\.step\(\)", body, re.M)
    assert call is not None, "real `results = self._gen.step()` call not found"
    before = body[: call.start()]

    # The guard must be the innermost enclosing block, so compare
    # positions rather than mere presence.
    last_try = before.rfind("try:")
    assert last_try != -1, (
        "self._gen.step() is not inside a try block. Under batched decode, "
        "deferred prefill runs INSIDE step(), so a cancel raises "
        "PrefillCancelled from here. It is a BaseException, so the runner's "
        "generic `except Exception` cannot catch it and the process dies "
        "(design doc Section 58)."
    )

    handler = body.find("except PrefillCancelled", call.end())
    assert handler != -1, (
        "no `except PrefillCancelled` AFTER the self._gen.step() call. Both "
        "submit-path handlers (_batched_start_task, _start_task) already "
        "treat it as 'this request is cancelled, carry on'; step() must do "
        "the same now that prefill executes here."
    )


def test_step_reports_the_cancellation_rather_than_swallowing_it() -> None:
    """Handling it must not mean silently dropping it.

    The request has to be reported cancelled, or the client hangs waiting
    for a response that will never come. `_apply_cancellations()` is the
    existing mechanism -- step()'s own no-work path already returns it,
    and it knows how to defer finalization for requests whose glue still
    holds state instead of reporting them cancelled too early.
    """
    body = _step_method_source()

    call = re.search(r"^\s+results = self\._gen\.step\(\)", body, re.M)
    assert call is not None, "real `results = self._gen.step()` call not found"
    handler_pos = body.find("except PrefillCancelled", call.end())
    assert handler_pos != -1, "no PrefillCancelled handler after the call"
    handler = body[handler_pos:]

    assert "_apply_cancellations" in handler, (
        "the PrefillCancelled handler in step() does not route through "
        "_apply_cancellations(). Swallowing the exception without reporting "
        "the cancellation leaves the client waiting forever for a response "
        "that will never arrive."
    )


def test_prefill_cancelled_is_still_a_baseexception() -> None:
    """Guard the property that makes this class of bug possible at all.

    If someone later 'fixes' this by demoting PrefillCancelled to
    Exception, every `except Exception` in the stack starts swallowing
    real cancellations as generic task errors -- which is a different and
    worse bug. The correct fix is an explicit handler at each site that
    can raise it, which is what the tests above pin.
    """
    from exo.worker.engines.mlx.generator.generate import PrefillCancelled

    assert issubclass(PrefillCancelled, BaseException)
    assert not issubclass(PrefillCancelled, Exception), (
        "PrefillCancelled was demoted to Exception. That makes it "
        "catchable by every generic `except Exception` in the runner, "
        "turning agreed cross-rank cancellations into opaque task "
        "failures. Handle it explicitly at raise-reachable call sites "
        "instead."
    )
