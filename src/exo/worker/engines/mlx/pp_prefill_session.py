# pyright: reportPrivateUsage=false
"""Resumable chunked-prefill session (Phase 2 Stage 2, 2026-08-06
scoping session) -- the real caller that DRIVES
``DeepseekV4Model._forward_steps`` (the generator-core split built in
Stage 1b, ``mlx-lm`` fork commit ``26eb90f0b``) with
``interruptible=True``, pausing between transformer layers so a
concurrent decode step can run in the gap.

See ``docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md``'s
"Layer-surgery Stage 2" entry for the full design history. Consult-
reviewed before implementation -- key decisions this module
implements:

1. **A session object is required, not an in-place generator drive.**
   ``Rank0BatchedDecodeGlue.tick()`` is a single synchronous call that
   returns immediately every real pipeline step -- it cannot itself
   block inside a Python generator's for-loop waiting for a decode
   step to run "in between." A live, suspended generator (the paused
   prefill's ``_forward_steps`` instance) must therefore persist
   ACROSS separate ``tick()`` calls as real object state, not as a
   local variable inside one call. ``ResumablePrefillSession`` is that
   persisted state.

2. **ONE captured ``contextvars.Context``, reused for every resume --
   never a fresh ``copy_context()`` per call.** Per
   ``pp_metaframe.py``'s ``ForwardStepInfo`` docstring (the reentrancy
   hazard this whole mechanism exists to avoid): a bare
   ``next()``/``send()`` on the AMBIENT context would let an
   interleaved decode step's own ``set_forward_step_info`` call leak
   into this session's later reads after it resumes. The fix is NOT
   "copy a fresh context every resume" (that would silently reset
   whatever this session itself set inside its own captured context on
   a PRIOR resume) -- it is "capture ONE context once, at session
   construction, and always resume via THAT SAME context's
   ``.run(...)``." This module's ``_ctx: contextvars.Context`` field
   and ``_resume_one_layer``/``_run_to_completion`` methods are the
   only place that discipline needs to be enforced; every other caller
   just calls this session's own methods.

3. **``mx.eval()`` happens HERE, at the point of genuinely pausing --
   never inside ``_forward_steps`` itself.** ``_forward_steps``'s own
   docstring is explicit that materializing at the yield point is the
   CALLER's decision, not baked into the generator, specifically so
   the non-interrupted eager `__call__` path's kernel fusion is
   unaffected. This module IS that caller, and DOES call
   ``mx.eval(h)`` on every layer-boundary yield it decides to actually
   pause at (see ``advance`` below) -- a pause that never materializes
   the lazy graph would be a no-op for real GPU/wire scheduling (the
   activation simply never actually computes until whenever the WHOLE
   chunk eventually gets drained), defeating the entire point of
   interrupting.

4. **Policy (how many layers per real pause) lives HERE, not in the
   generator.** ``_forward_steps`` yields at EVERY layer boundary
   unconditionally when interruptible -- a generator resume costs
   microseconds against a transformer layer's milliseconds, so
   fine-grained yield points are free. This session's ``advance(...)``
   decides how many of those cheap yields to just re-enter immediately
   (no real pause) versus genuinely stopping (mx.eval + return control
   to the caller) -- the actual segment-size tuning knob belongs to
   whoever calls ``advance``, not to this module or to
   ``_forward_steps``.

5. **Multi-rank symmetry is NOT this module's job to enforce -- it is
   a real constraint on whoever DRIVES sessions on every rank.** Per
   the consult review: if rank 0 interleaves a decode step mid-chunk,
   every OTHER rank in the pipeline must pause its own local prefill
   work at the SAME protocol point, or ranks desynchronize on the
   wire. This module only manages ONE rank's own local session state;
   the cross-rank agreement on WHEN to pause is a
   ``Rank0BatchedDecodeGlue``/``Rank1BatchedDecodeGlue`` wiring
   concern (Stage 3, not yet built) that must drive a session on
   EVERY rank in lockstep via the wire protocol
   (``pp_scheduler_protocol.py``'s ``PrefillChunkAdvancedEvent``),
   never left to per-rank local timing.
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
from typing import (
    Generator,
    Iterator,
    Literal,
    Protocol,
    Tuple,
    Union,
    cast,
    runtime_checkable,
)

import mlx.core as mx

from exo.worker.engines.mlx.pp_metaframe import ForwardPhase, set_forward_step_info
from exo.worker.engines.mlx.types import KVCacheType


class PrefillSessionError(RuntimeError):
    """Raised on any misuse of ``ResumablePrefillSession`` -- fail-stop,
    matching this fork's own established discipline
    (``pp_scheduler_protocol.py``'s ``ProtocolViolationError``,
    ``pp_batched_decode_glue.py``'s ``GlueError``) rather than silently
    guessing at recovery."""


ForwardStep = Union[
    Tuple[Literal["layer"], int, "mx.array"],
    Tuple[Literal["done"], None, "mx.array"],
]


@runtime_checkable
class _InterruptibleForward(Protocol):
    """Structural capability check for "this inner model supports
    chunked-prefill interruption" -- matches this fork's OWN
    established capability-gating discipline
    (``generate.py``'s ``_has_pipeline_communication_layer`` checks
    ``isinstance(layer, (PipelineFirstLayer, PipelineLastLayer))``
    rather than branching on model architecture by name). Any inner
    model exposing a correctly-shaped ``_forward_steps`` satisfies
    this protocol structurally -- DSv4 today, potentially other
    architectures later, with zero changes needed here."""

    def _forward_steps(
        self,
        inputs: mx.array,
        cache: "KVCacheType | None" = None,
        *,
        interruptible: bool = False,
    ) -> Iterator[ForwardStep]: ...


def supports_chunked_prefill_interruption(inner_model: object) -> bool:
    """Structural (Protocol-based) capability check, not an
    architecture-name check -- see ``_InterruptibleForward``'s own
    docstring for why. Callers (the not-yet-built
    ``pipeline_parallel_prefill`` generator-ification, Stage 2's
    remaining piece) use this to decide whether a given loaded model
    can ever be driven interruptibly at all; a model that fails this
    check falls back to the existing eager, non-interruptible chunk
    loop unchanged."""
    return isinstance(inner_model, _InterruptibleForward)


@dataclass
class ResumablePrefillSession:
    """Owns ONE in-flight, possibly-paused chunked prefill's live
    generator state across multiple real ``tick()`` calls. See module
    docstring points 1-4 for the design rationale; this class is the
    concrete implementation.

    Constructed once per chunked-prefill admission (mirrors
    ``NewChunkedPrefillRequestEvent``'s own one-per-admission
    lifecycle in ``pp_scheduler_protocol.py``) and driven via
    ``advance(...)`` on each real tick that decides to make progress
    on this request's prefill.
    """

    inner_model: _InterruptibleForward
    inputs: mx.array
    cache: KVCacheType
    # 2026-08-06: captured ONCE at construction, reused for EVERY
    # resume -- see module docstring point 2 for why a fresh
    # copy_context() per resume would be WRONG here (it would discard
    # whatever this session's own prior resume already set inside its
    # captured context).
    _ctx: contextvars.Context = field(init=False)
    _gen: Iterator[ForwardStep] | None = field(default=None, init=False)
    _last_layer_index: int = field(default=-1, init=False)
    _done: bool = field(default=False, init=False)
    _final_output: mx.array | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._ctx = contextvars.copy_context()

    @property
    def done(self) -> bool:
        return self._done

    @property
    def last_layer_index(self) -> int:
        """The index of the last layer whose output this session has
        actually observed (via a real, materialized pause) -- NOT
        necessarily the last layer the underlying generator has
        internally produced, since ``advance`` may re-enter several
        cheap yields without genuinely pausing at each one (module
        docstring point 4). ``-1`` before any progress has been made."""
        return self._last_layer_index

    def _start(self) -> None:
        if self._gen is not None:
            raise PrefillSessionError(
                "ResumablePrefillSession._start called twice -- a session's "
                "generator must be created exactly once, at the first real "
                "advance() call, not re-created mid-session"
            )

        def _create() -> Iterator[ForwardStep]:
            return self.inner_model._forward_steps(
                self.inputs, self.cache, interruptible=True
            )

        self._gen = self._ctx.run(_create)

    def advance(
        self, *, max_layers: int, phase_for_pause: ForwardPhase
    ) -> tuple[int, bool]:
        """Drive this session's generator forward by up to
        ``max_layers`` real layer-boundary yields, THEN genuinely pause
        (``mx.eval()`` the paused activation, per module docstring
        point 3) unless the generator reaches its own ``("done", ...)``
        step first. ``max_layers`` is the CALLER's segment-size policy
        (module docstring point 4) -- this method imposes no default
        of its own.

        ``phase_for_pause``: the ``ForwardPhase`` this session's OWN
        resumes should present to ``MetaFramedPipelineLastLayer`` via
        ``set_forward_step_info`` -- always ``PREFILL_CONTINUE`` for a
        genuine mid-chunk pause (more work remains after this call),
        matching ``ForwardPhase``'s own documented wire semantics. The
        CALLER decides this (not hardcoded here) because the FINAL
        resume that reaches actual completion needs ``PREFILL_FINAL``
        instead -- see the ``done`` return value below, which the
        caller must check to know which phase was ACTUALLY relevant
        for the resume that just ran.

        Returns ``(layers_advanced, done)`` -- ``layers_advanced`` is
        how many real layer-boundary yields this call consumed
        (<= max_layers, less if the generator reached "done" first);
        ``done`` is ``True`` iff the underlying forward pass has fully
        completed (the caller must then read ``self.output()`` and
        retire this session, mirroring ``PrefillChunkAdvancedEvent``'s
        own "final chunk" completion semantics on the protocol side).
        """
        if max_layers < 1:
            raise PrefillSessionError(
                f"ResumablePrefillSession.advance: max_layers={max_layers} "
                f"must be >=1 -- a zero/negative-layer advance is not a "
                f"real occurrence this method should represent"
            )
        if self._done:
            raise PrefillSessionError(
                "ResumablePrefillSession.advance called on an already-"
                "completed session -- the caller must check .done and "
                "retire a finished session instead of advancing it further"
            )
        if self._gen is None:
            self._start()
        gen = self._gen
        assert gen is not None  # narrowed by _start() above

        def _set_phase_and_resume() -> ForwardStep:
            # 2026-08-06: set INSIDE the captured context's .run() call,
            # not before it -- so this write lands in THIS session's own
            # isolated context, never leaking into (or being leaked
            # into by) whatever context an interleaved decode step's
            # OWN forward pass is using on this same thread. See module
            # docstring point 2.
            #
            # 2026-08-07 (wire-ordering bug fix): defer_header=True --
            # this class is used EXCLUSIVELY by the chunk-drive path
            # (never by decode's own, separate use of queue_sends=True
            # in pp_batched_decode_runtime.py), which needs BOTH the
            # metaframe header AND activation deferred together until
            # Rank0BatchedDecodeGlue.tick()'s HANDOFF/RANK1_DRAINING
            # phases explicitly decide it's safe to put them on the
            # wire -- see ForwardStepInfo.defer_header's own docstring
            # for the full wire-ordering invariant this closes.
            set_forward_step_info(
                phase=phase_for_pause, queue_sends=True, defer_header=True
            )
            return next(gen)

        layers_advanced = 0
        last_h: mx.array | None = None
        while layers_advanced < max_layers:
            try:
                kind, idx, h = self._ctx.run(_set_phase_and_resume)
            except StopIteration as exc:  # pragma: no cover - defensive;
                # _forward_steps always yields a ("done", ...) tuple
                # before returning, per its own contract -- a bare
                # StopIteration here would mean that contract broke.
                raise PrefillSessionError(
                    "ResumablePrefillSession: underlying _forward_steps "
                    "generator raised StopIteration without first "
                    "yielding a ('done', ...) step -- this violates "
                    "_forward_steps's own documented contract"
                ) from exc
            if kind == "done":
                self._done = True
                self._final_output = h
                mx.eval(h)
                return layers_advanced, True
            assert kind == "layer"
            self._last_layer_index = (
                int(idx) if idx is not None else self._last_layer_index
            )
            layers_advanced += 1
            last_h = h

        # Genuinely pause here: materialize the most recent layer's
        # output NOW, per module docstring point 3 -- an un-evaluated
        # pause would leave the lazy graph (including whatever this
        # rank has already recv'd) open while a concurrent decode step
        # builds its OWN graph on the same device, which is exactly
        # the interleaved-lazy-graph hazard the design doc's earlier
        # eval-boundary-overhead benchmark was scoped to rule out one
        # layer at a time, not left unevaluated across an entire
        # multi-layer pause. `max_layers >= 1` is enforced above, so
        # this loop always runs at least once and `last_h` is always
        # populated by the time control reaches here -- the explicit
        # None check below is for the type checker, not a real runtime
        # possibility.
        if last_h is None:
            raise PrefillSessionError(
                "internal invariant violation: advance()'s loop exited "
                "without ever setting last_h -- this is a bug in this "
                "method itself (max_layers>=1 is validated at entry, so "
                "the loop body must run at least once), not a caller error"
            )
        mx.eval(last_h)
        return layers_advanced, False

    def output(self) -> mx.array:
        """The completed prefill's final hidden state. Raises
        ``PrefillSessionError`` if called before ``advance`` has
        returned ``done=True`` -- fail-stop, matching this module's
        own discipline throughout."""
        if not self._done or self._final_output is None:
            raise PrefillSessionError(
                "ResumablePrefillSession.output() called before the "
                "session reached completion -- check .done first"
            )
        return self._final_output

    def abort(self) -> None:
        """Genuinely close this session's suspended generator --
        2026-08-07, real cancel/abort mechanism.

        MUST be routed through ``self._ctx.run(...)``, never a bare
        ``self._gen.close()`` -- per a `consult` review: ``.close()``
        throws ``GeneratorExit`` at the suspension point and runs the
        generator's cleanup path (any ``finally``/``except`` blocks
        inside ``_forward_steps``) in WHATEVER CONTEXT THE CALL SITE
        IS RUNNING UNDER. A bare call from the glue's own ``tick()``
        would run that cleanup in the AMBIENT context, not this
        session's own captured one (module docstring point 2's exact
        reentrancy hazard, now applying to teardown as much as it
        applies to resume) -- a silent-corruption risk if
        ``_forward_steps``'s cleanup ever reads/writes
        ``ForwardStepInfo`` (e.g. resetting ``defer_header``/phase
        state), not merely a crash risk.

        Idempotent-safe to call on an already-``done`` session (a
        harmless no-op via ``.close()``'s own documented behavior on
        an exhausted generator) or one that never started
        (``self._gen is None`` -- nothing to close). Raises
        ``PrefillSessionError`` (not a bare propagated exception) if
        the generator's own cleanup path raises something OTHER than
        ``GeneratorExit`` while closing, or if the generator
        unexpectedly yields again during close (Python's own
        ``RuntimeError: generator ignored GeneratorExit``) --
        fail-stop, matching this module's own discipline throughout,
        rather than letting an unusual cleanup-path failure propagate
        as a confusing, differently-shaped exception from deep inside
        this method.
        """
        if self._gen is None:
            return
        gen = self._gen

        def _close() -> None:
            # 2026-08-07: real Python generator objects always expose
            # .close() at runtime -- the Iterator[ForwardStep] type on
            # self._gen is deliberately the narrower Protocol-facing
            # type (matches _InterruptibleForward._forward_steps's own
            # declared Iterator return type, which mirrors the REAL
            # mlx-lm _forward_steps's actual annotation exactly -- see
            # this module's own capability-check discipline). The cast
            # here is a type-level acknowledgment of a runtime fact
            # already guaranteed by _start()'s own construction (this
            # field is ALWAYS assigned a genuine generator object,
            # never an arbitrary Iterator), not a new assumption.
            cast("Generator[ForwardStep, None, None]", gen).close()

        try:
            self._ctx.run(_close)
        except RuntimeError as exc:
            raise PrefillSessionError(
                "ResumablePrefillSession.abort(): the underlying "
                "_forward_steps generator raised RuntimeError while "
                "closing (most likely 'generator ignored GeneratorExit' "
                "-- it yielded again instead of returning/raising after "
                "receiving the close signal). This violates "
                "_forward_steps's own documented contract."
            ) from exc
        except Exception as exc:  # noqa: BLE001 - fail-stop wrapper, not a
            # swallow: re-raised immediately as a typed, attributable
            # PrefillSessionError rather than letting an arbitrary
            # cleanup-path exception (e.g. from a finally block inside
            # _forward_steps) propagate in whatever shape it happened
            # to have, matching this module's own discipline of never
            # silently guessing at recovery.
            raise PrefillSessionError(
                "ResumablePrefillSession.abort(): the underlying "
                "_forward_steps generator's cleanup path raised while "
                "closing -- see the chained exception for the real "
                "cause"
            ) from exc
        finally:
            self._gen = None
            self._done = True
