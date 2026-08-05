# pyright: reportPrivateUsage=false, reportAny=false
"""Phase 0 correctness-baseline tooling for the batched-PP sharding design.

See ``docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md`` Section 9,
"Phase 0 — Correctness baseline". This module provides the reusable
harness Phase 0.5+ will drive against once real batched-PP code exists:

- ``SimPipelineTransport``: an in-process, copy-through fake for
  ``mx.distributed.send`` / ``recv_like``, standing in for real jaccl/RDMA
  transport between two simulated PP ranks in a single OS process.
- ``build_two_rank_split``: wraps two already-constructed model
  instances' local first/last layer with the REAL ``PipelineFirstLayer``/
  ``PipelineLastLayer`` classes from ``auto_parallel.py`` (not a stub) —
  the same production wiring ``pipeline_auto_parallel`` uses for a real
  2-node PP split.
- ``run_two_rank_pp_forward``: drives both simulated ranks' forward
  passes through the fake transport with correct cross-rank blocking
  semantics.
- ``compare_logits``: shared argmax-mismatch + float-tolerance comparator,
  matching the convention already used by ``test_prefill_batched.py``.

Design rationale (informed by an external consult review, 2026-08-05,
before this module was written):

1. **Only one simulated rank ever executes an MLX op at a time.** MLX
   lazy graph construction and ``mx.eval`` are not documented as
   thread-safe for concurrent callers. Two real threads freely building
   lazy graphs / calling ``mx.eval`` against the same Metal device could
   produce intermittent wrong results or crashes — disqualifying for a
   correctness oracle. Real OS threads ARE used here (this pipeline
   topology's decode-only handoff has a genuine same-call
   send-then-block-on-recv dependency that a naive
   "run rank0 fully, then rank1 fully" ordering cannot satisfy without
   deadlocking), but a single global ``_MLX_CALL_LOCK`` ensures only one
   simulated rank is ever actually executing MLX ops at any instant — the
   lock is held for a rank's entire forward-driving body and released
   ONLY while blocked inside the fake transport's ``recv_like`` (a
   ``queue.Queue.get()`` call, not an MLX op). This is a deliberate,
   narrower pattern than ``test_pipeline_prefix_hit_agreement.py``'s
   existing thread+queue precedent, which only ever exercised a tiny
   control-flow helper (no real tensor forward passes) and did not need
   this guarantee.
2. **The golden reference is the PLAIN unsharded forward, not "trust the
   simulated split."** The simulation harness is new, unproven code —
   anchoring against serial PP-as-shipped would just be validating one
   unproven thing against another. The companion test file's
   ``test_two_rank_pp_matches_plain_forward_prefill_and_decode`` asserts
   the simulated split reproduces the plain forward's GREEDY-TOKEN
   output exactly (argmax mismatches==0). It does NOT assert float-
   tolerance-level agreement: the real ``PipelineFirstLayer``/
   ``PipelineLastLayer`` classes cast activations to bf16 before every
   cross-rank send (a genuine production requirement — JACCL/RDMA
   transport requires bf16), a cost the plain unsharded forward never
   pays at all — so some real, expected numerical drift is inherent to
   the split itself, not a harness bug. That greedy-token-agreement
   assertion is Phase 0's actual deliverable: it proves the harness
   itself is valid BEFORE any batched-PP code is diffed against it.
3. **The fake transport materializes and copies, it does not alias.**
   ``send`` evals the array then round-trips it through ``numpy`` to
   produce an independent copy before enqueueing — passing the same
   ``mx.array`` object across "ranks" would (a) hide sender-side
   mutation-after-send bugs, (b) share lazy graph nodes across simulated
   ranks, and (c) fail to exercise the metadata-framing correctness the
   real batched-PP design depends on. ``recv_like`` asserts the received
   payload's shape against the caller's template.

Explicitly NOT claimed: this harness validates NUMERICS (does a 2-rank
PP split compute the same thing as an unsharded forward), not
deadlock-freedom, real-transport timing, or async/concurrent-request
scheduling correctness. Those remain the real 2-node cluster's job —
this only exists to keep GPU/cluster time off the "did I break the
math" question, per the doc's Phase 0 scoping ("largely tooling").
"""

from __future__ import annotations

import queue
import threading
from typing import Any, Protocol, cast
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from exo.worker.engines.mlx.auto_parallel import (
    PipelineFirstLayer,
    PipelineLastLayer,
    _set_layers,
    clear_prefill_sends,
    get_inner_model,
    get_layers,
    set_pipeline_prefill,
)

# Global lock: held for the ENTIRE body of a simulated rank's forward-
# driving code, released only inside SimPipelineTransport.recv_like's
# blocking queue.get(). Guarantees only one simulated rank ever executes
# an MLX op at any instant, even though two real OS threads are involved.
_MLX_CALL_LOCK = threading.Lock()

# How long a simulated rank will block waiting for its peer's send
# before concluding the protocol deadlocked (a real bug, not a slow
# machine — this harness runs small random-weight models with no I/O).
_RECV_TIMEOUT_SECONDS = 30.0


class _RankGroup:
    """Minimal ``mx.distributed.Group`` stand-in — only ``rank()`` and
    ``size()`` are read by ``PipelineFirstLayer``/``PipelineLastLayer``
    (both only ever pass ``group=`` through to send/recv_like, which are
    globally monkeypatched to ``SimPipelineTransport`` for the duration
    of a simulated run, so no other Group method is ever invoked)."""

    def __init__(self, rank: int, size: int) -> None:
        self._rank = rank
        self._size = size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


class _ModelLike(Protocol):
    def __call__(
        self, x: mx.array, cache: Any = None, **kwargs: object
    ) -> mx.array: ...


class SimPipelineTransport:
    """In-process, copy-through fake for ``mx.distributed.send`` /
    ``recv_like``, keyed by ``(src_rank, dst_rank)``.

    ``send`` is non-blocking (materializes + copies + enqueues).
    ``recv_like`` blocks on the queue, releasing ``_MLX_CALL_LOCK`` for
    the duration of the wait so the peer rank's thread can make progress.
    """

    def __init__(self) -> None:
        self._queues: dict[tuple[int, int], "queue.Queue[Any]"] = {}

    def _q(self, src: int, dst: int) -> "queue.Queue[Any]":
        key = (src, dst)
        if key not in self._queues:
            self._queues[key] = queue.Queue()
        return self._queues[key]

    def send(
        self, arr: mx.array, dst: int, *, group: _RankGroup, **_: object
    ) -> mx.array:
        src = group.rank()
        mx.eval(arr)
        # numpy roundtrip copy: an independent buffer, not an alias of
        # ``arr`` — mirrors a real cross-process transport, and forces
        # any metadata-framing/shape bug to actually surface here rather
        # than being masked by object identity. bf16 has no native numpy
        # dtype (real PP casts activations to bf16 before every send —
        # see PipelineFirstLayer/PipelineLastLayer — so this path IS
        # exercised in practice), so upcast to fp32 for the roundtrip and
        # let ``recv_like`` cast back down to the caller's template dtype.
        original_dtype = arr.dtype
        copied = mx.array(np.array(arr.astype(mx.float32)))
        self._q(src, dst).put((copied.shape, original_dtype, np.array(copied)))
        return arr

    def recv_like(
        self, template: mx.array, src: int, *, group: _RankGroup, **_: object
    ) -> mx.array:
        dst = group.rank()
        q = self._q(src, dst)
        _MLX_CALL_LOCK.release()
        try:
            shape, _sent_dtype, payload = q.get(timeout=_RECV_TIMEOUT_SECONDS)
        except queue.Empty as e:
            raise RuntimeError(
                f"SimPipelineTransport: rank {dst} timed out waiting on "
                f"rank {src} (protocol deadlock, not a real bug in the "
                f"model under test)"
            ) from e
        finally:
            _MLX_CALL_LOCK.acquire()
        if shape != template.shape:
            raise RuntimeError(
                f"SimPipelineTransport shape mismatch: rank {dst} expected "
                f"{template.shape} from rank {src}, got {shape}"
            )
        return mx.array(payload, dtype=template.dtype)


def build_two_rank_split(
    rank0_model: _ModelLike,
    rank1_model: _ModelLike,
) -> tuple[_ModelLike, _ModelLike, SimPipelineTransport]:
    """Wrap two ALREADY-CONSTRUCTED model instances' local first/last
    layer with the REAL ``PipelineFirstLayer``/``PipelineLastLayer``
    classes, exactly as ``pipeline_auto_parallel`` does for a real
    2-node PP split — using the SAME ``get_inner_model``/``get_layers``/
    ``_set_layers`` helpers the production code path uses, rather than
    assuming a bare ``model.layers`` list attribute (which is a
    read-only property on some model wrapper classes, e.g. mlx-lm's
    ``llama.Model`` — the real inner list lives on ``model.model.layers``).

    The caller is responsible for constructing both model instances with
    IDENTICAL weights (e.g. by copying ``model.parameters()`` from one
    instance to the other) — this function only handles the PP-split
    wrapping, not weight identity.

    Returns ``(rank0_model, rank1_model, transport)`` (the same two
    input objects, mutated in place, returned for chaining convenience).
    """
    inner0 = get_inner_model(cast(nn.Module, cast(Any, rank0_model)))
    inner1 = get_inner_model(cast(nn.Module, cast(Any, rank1_model)))
    layers0 = get_layers(inner0)
    layers1 = get_layers(inner1)

    n_layers = len(layers0)
    if len(layers1) != n_layers:
        raise ValueError(
            f"build_two_rank_split requires both model instances to have "
            f"the same layer count, got {n_layers} vs {len(layers1)}"
        )
    mid = n_layers // 2
    if mid == 0 or mid == n_layers:
        raise ValueError(
            f"build_two_rank_split needs >=2 layers to split into two "
            f"non-empty ranks, got {n_layers}"
        )

    transport = SimPipelineTransport()
    # ``_RankGroup`` only needs to satisfy ``rank()``/``size()`` — the
    # real ``mx.distributed.Group`` type is never otherwise touched
    # because send/recv_like are globally monkeypatched to
    # ``SimPipelineTransport`` for the duration of a simulated run (see
    # ``run_two_rank_pp_forward``). Double-cast bridges the structural-
    # vs-nominal typing gap for the type checker.
    group0 = cast(mx.distributed.Group, cast(Any, _RankGroup(0, 2)))
    group1 = cast(mx.distributed.Group, cast(Any, _RankGroup(1, 2)))

    r0_layers = list(layers0[:mid])
    r1_layers = list(layers1[mid:])

    r0_layers[0] = PipelineFirstLayer(r0_layers[0], r=0, group=group0)
    r0_layers[-1] = PipelineLastLayer(r0_layers[-1], r=0, s=2, group=group0)
    r1_layers[0] = PipelineFirstLayer(r1_layers[0], r=1, group=group1)
    r1_layers[-1] = PipelineLastLayer(r1_layers[-1], r=1, s=2, group=group1)

    _set_layers(cast(nn.Module, cast(Any, rank0_model)), r0_layers)
    _set_layers(cast(nn.Module, cast(Any, rank1_model)), r1_layers)

    return rank0_model, rank1_model, transport


def run_two_rank_pp_forward(
    rank0_model: _ModelLike,
    rank1_model: _ModelLike,
    transport: SimPipelineTransport,
    tokens: mx.array,
    rank0_cache: list[Any],
    rank1_cache: list[Any],
    *,
    is_prefill: bool,
) -> mx.array:
    """Drive one simulated 2-rank PP forward pass for ``tokens`` (shape
    ``(1, L)`` or ``(1, 1)`` for a single decode step), returning the
    final logits (only meaningful on rank 0's side, matching real PP's
    "rank 0 samples" contract).

    Uses two real OS threads (one per simulated rank) so the genuine
    same-call send-then-block-on-recv dependency in
    ``PipelineLastLayer.__call__``'s decode-only handoff (rank 0's last
    layer sends to rank 1, THEN blocks receiving rank 1's final answer,
    all within rank 0's own single forward call) can be satisfied without
    deadlocking a strictly sequential "run rank0 then rank1" ordering.
    ``_MLX_CALL_LOCK`` (module-level) ensures only one thread ever
    executes an MLX op at a time — see module docstring.

    ``mlx.core.distributed.send``/``recv_like`` are patched ONCE around
    the whole two-thread run (not per-thread) — matching
    ``test_pipeline_prefix_hit_agreement.py``'s documented rationale:
    ``unittest.mock.patch`` as a context manager mutates a single shared
    module attribute, so nested per-thread ``with patch(...)`` blocks
    race on setup/teardown across real OS threads (one thread's
    context-manager exit un-patches the function out from under another
    thread still mid-call).
    """
    set_pipeline_prefill(cast(nn.Module, cast(Any, rank0_model)), is_prefill=is_prefill)
    set_pipeline_prefill(cast(nn.Module, cast(Any, rank1_model)), is_prefill=is_prefill)

    # Force ``tokens`` fully materialized on the CALLING thread before
    # handing it to the two worker threads — an un-evaled lazy array
    # built on one thread and first evaluated inside another has been
    # observed to raise "There is no Stream(gpu, N) in current thread."
    # (MLX's per-thread stream/command-queue context is thread-local; a
    # graph node's originating thread matters even though the ARRAY
    # object itself is just Python data). This is exactly the kind of
    # MLX-thread-interaction hazard the module docstring's design
    # rationale flags — this eval is a workaround for it, on top of (not
    # instead of) ``_MLX_CALL_LOCK`` serializing actual op execution.
    mx.eval(tokens)

    result: dict[str, Any] = {}

    def _rank0() -> None:
        _MLX_CALL_LOCK.acquire()
        try:
            # Touch MLX's stream machinery on THIS thread first (see
            # comment above) before running the real forward pass.
            mx.eval(mx.zeros(1))
            out = rank0_model(tokens, cache=rank0_cache)
            mx.eval(out)
            result["logits"] = out
        except BaseException as e:  # noqa: BLE001 - surface on join
            result["error0"] = e
        finally:
            _MLX_CALL_LOCK.release()

    def _rank1() -> None:
        _MLX_CALL_LOCK.acquire()
        try:
            mx.eval(mx.zeros(1))
            out = rank1_model(tokens, cache=rank1_cache)
            mx.eval(out)
        except BaseException as e:  # noqa: BLE001 - surface on join
            result["error1"] = e
        finally:
            _MLX_CALL_LOCK.release()

    clear_prefill_sends()
    with (
        patch("mlx.core.distributed.send", transport.send),
        patch("mlx.core.distributed.recv_like", transport.recv_like),
    ):
        t0 = threading.Thread(target=_rank0)
        t1 = threading.Thread(target=_rank1)
        t0.start()
        t1.start()
        t0.join(timeout=_RECV_TIMEOUT_SECONDS + 5)
        t1.join(timeout=_RECV_TIMEOUT_SECONDS + 5)
        if t0.is_alive() or t1.is_alive():
            raise RuntimeError(
                "run_two_rank_pp_forward: simulated rank thread deadlocked "
                "(protocol bug, not a timing flake — small random-weight "
                "models with no I/O should never take this long)"
            )
    if "error0" in result:
        raise result["error0"]
    if "error1" in result:
        raise result["error1"]
    if "logits" not in result:
        raise RuntimeError(
            "run_two_rank_pp_forward: rank 0 produced no logits — check "
            "the PP topology wiring (rank 0 must be the sampling rank)"
        )
    return cast(mx.array, result["logits"])


def compare_logits(
    baseline: list[mx.array],
    candidate: list[mx.array],
    label: str = "",
) -> tuple[float, int]:
    """Shared comparator: max abs float diff + argmax-mismatch count.

    Matches the convention already used by
    ``tests/test_prefill_batched.py``'s ``_compare_logits`` — kept as a
    free function here (rather than importing that test-local helper)
    since this module is production tooling other test files will import
    from, not itself a test module.
    """
    if len(baseline) != len(candidate):
        raise ValueError(
            f"compare_logits: length mismatch baseline={len(baseline)} "
            f"candidate={len(candidate)}"
        )
    max_diff = 0.0
    mismatches = 0
    for step in range(len(baseline)):
        diff = float(
            mx.max(
                mx.abs(
                    baseline[step].astype(mx.float32)
                    - candidate[step].astype(mx.float32)
                )
            ).item()
        )
        max_diff = max(max_diff, diff)
        if int(mx.argmax(baseline[step]).item()) != int(
            mx.argmax(candidate[step]).item()
        ):
            mismatches += 1
    if label:
        print(
            f"[{label}] max_diff={max_diff:.4e} mismatches={mismatches}/{len(baseline)}"
        )
    return max_diff, mismatches
