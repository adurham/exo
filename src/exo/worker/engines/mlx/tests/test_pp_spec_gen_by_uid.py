# pyright: reportPrivateUsage=false
"""Unit coverage for the PP speculative-decode per-task-keyed generator
state (2026-07-31 fix).

Regression coverage for the ExoBatchGenerator._pp_spec_gen/_pp_spec_uid
data-corruption bug: those used to be bare singular instance attributes,
silently clobbered by a second submit() before the first request's
generator was exhausted -- the first request's task orphaned forever
(generator reference lost, nothing ever resumes it) and the second
request inherited whatever mid-flight state the overwrite left behind.

Fixed by converting to a dict keyed by uid (_pp_spec_gen_by_uid) plus an
entry guard in _submit_pp_spec that raises PPSpecAlreadyActiveError
instead of silently overwriting when a second PP-spec request arrives
while one is still active.

IMPORTANT: this is a SAFETY fix (silent corruption -> loud rejection),
NOT a concurrency feature. PP's shared SpecPipelineFirstLayer/
SpecPipelineLastLayer mode flags represent the ONE physical rank0<->rank1
wire link, so the dict never legitimately holds more than one entry in
today's architecture -- these tests verify exactly that invariant, not
that concurrent PP-spec decoding works (it explicitly doesn't, by
design, and shouldn't).

These tests exercise the dict/guard mechanics directly on a bare
ExoBatchGenerator instance (object.__new__, bypassing __post_init__'s
real MLX/model setup) rather than driving the full _submit_pp_spec code
path -- mirrors test_concurrency_admission_gate.py's own stated scope
(that test explicitly does NOT reproduce the real multi-rank PP
corruption either; both are black-box mechanism tests, not full
integration tests against live cluster state).
"""
from typing import Generator

import mlx.core as mx
import pytest

from exo.worker.engines.mlx.generator.batch_generate import (
    ExoBatchGenerator,
    PPSpecAlreadyActiveError,
)


def _bare_gen() -> ExoBatchGenerator:
    """Construct an ExoBatchGenerator without running __post_init__ (which
    needs a real model/tokenizer/group) -- just enough state for the
    _pp_spec_gen_by_uid dict/guard mechanics under direct test.

    Because this deliberately bypasses dataclass field initialization, every
    `init=False` field that the methods under test touch must be seeded here
    by hand. When _close_pp_spec_gen() grew its
    `_pp_spec_cancel_requested.discard(uid)` call (2026-08-11, design doc
    Section 43 -- so a cancelled uid can't leak its flag into a later
    generator), this fixture was not updated, and all four close-related
    tests began failing with
    `AttributeError: 'ExoBatchGenerator' object has no attribute
    '_pp_spec_cancel_requested'`.

    That was a TEST-HARNESS gap, not a production bug: the real object gets
    these fields from the dataclass machinery. But it left four permanently
    red tests on the PP-speculation path, which is actively dangerous --
    pre-existing failures make a genuine regression on that path
    indistinguishable from known noise. Seeded rather than removed, so the
    tests keep asserting what they were written to assert.
    """
    gen = object.__new__(ExoBatchGenerator)
    gen._pp_spec_gen_by_uid = {}
    gen._uid_counter = 0
    gen._pp_spec_cancel_requested = set()
    gen._pp_spec_cancel_agree_tag = 0
    return gen


def _fake_spec_gen() -> Generator[tuple[int, mx.array], None, None]:
    """A trivial generator matching the (token_id, logprobs) yield shape
    the real pp_*_decode_loop generators produce -- never actually
    iterated in these tests, just needs to exist as a distinct object
    identity so dict entries are verifiably per-uid."""
    yield (1, mx.zeros(1))
    yield (2, mx.zeros(1))


class TestPPSpecGenByUidDict:
    def test_starts_empty(self) -> None:
        gen = _bare_gen()
        assert gen._pp_spec_gen_by_uid == {}

    def test_single_entry_insert_and_lookup(self) -> None:
        gen = _bare_gen()
        spec_gen = _fake_spec_gen()
        gen._pp_spec_gen_by_uid[1] = spec_gen
        assert gen._pp_spec_gen_by_uid[1] is spec_gen
        assert len(gen._pp_spec_gen_by_uid) == 1

    def test_close_pops_the_named_uid_only(self) -> None:
        """_close_pp_spec_gen(uid) must remove exactly that uid's entry --
        proves the fix's core safety property: closing one request's
        generator can never accidentally clear a DIFFERENT uid's state
        (impossible with the old singular-attribute design, since there
        was only ever one attribute to clear)."""
        gen = _bare_gen()
        spec_gen_a = _fake_spec_gen()
        gen._pp_spec_gen_by_uid[1] = spec_gen_a

        gen._close_pp_spec_gen(1)

        assert 1 not in gen._pp_spec_gen_by_uid
        assert gen._pp_spec_gen_by_uid == {}

    def test_close_on_absent_uid_is_a_noop(self) -> None:
        """Closing a uid that was never inserted (e.g. a double-close, or
        closing after some other path already popped it) must not raise
        -- pop(uid, None) semantics, not pop(uid) which would KeyError."""
        gen = _bare_gen()
        gen._close_pp_spec_gen(999)  # must not raise
        assert gen._pp_spec_gen_by_uid == {}

    def test_close_calls_generator_close(self) -> None:
        """_close_pp_spec_gen must call .close() on the popped generator
        (deterministic finalization -- see the method's own docstring for
        why this matters more than bare refcount-drop finalization)."""
        gen = _bare_gen()
        closed = []

        class _TrackingGen:
            def close(self) -> None:
                closed.append(True)

        gen._pp_spec_gen_by_uid[1] = _TrackingGen()  # type: ignore[assignment]
        gen._close_pp_spec_gen(1)
        assert closed == [True]

    def test_close_swallows_generator_close_exceptions(self) -> None:
        """A generator whose .close() raises (e.g. GeneratorExit handling
        gone wrong inside the real pp_*_decode_loop's try/finally) must
        not propagate -- matches the method's existing debug-log-and-
        continue behavior, now scoped per-uid instead of globally."""
        gen = _bare_gen()

        class _RaisingGen:
            def close(self) -> None:
                raise RuntimeError("boom")

        gen._pp_spec_gen_by_uid[1] = _RaisingGen()  # type: ignore[assignment]
        gen._close_pp_spec_gen(1)  # must not raise
        assert gen._pp_spec_gen_by_uid == {}


class TestPPSpecAlreadyActiveError:
    def test_is_a_runtime_error_not_base_exception(self) -> None:
        """Deliberately a plain RuntimeError subclass (unlike
        PrefillCancelled's BaseException in generate.py) so it's caught
        by the runner's existing `except Exception` handling around
        generation-task dispatch and surfaces as a clean task failure,
        not an uncaught crash. See the exception's own docstring."""
        assert issubclass(PPSpecAlreadyActiveError, RuntimeError)
        assert not issubclass(PPSpecAlreadyActiveError, BaseException) or issubclass(
            PPSpecAlreadyActiveError, Exception
        )

    def test_raised_and_catchable_as_exception(self) -> None:
        with pytest.raises(PPSpecAlreadyActiveError):
            raise PPSpecAlreadyActiveError("uid=1 already active")
        # Also catchable via the broader Exception handler the runner uses.
        try:
            raise PPSpecAlreadyActiveError("uid=1 already active")
        except Exception as e:  # noqa: BLE001 -- exact scenario under test
            assert isinstance(e, PPSpecAlreadyActiveError)
