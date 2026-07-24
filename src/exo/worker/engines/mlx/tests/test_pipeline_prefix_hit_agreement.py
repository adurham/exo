# pyright: reportPrivateUsage=false
# pyright: reportAny=false
"""Unit tests for ``pipeline_agree_prefix_hit_length`` (utils_mlx.py).

Real ``mx.distributed`` groups require a multi-process cluster launch, so
these tests simulate a 2 (and 3) rank pipeline in-process: each simulated
rank runs the function on its own thread, and ``mx.distributed.send`` /
``mx.distributed.recv_like`` are monkeypatched to route through in-memory
``queue.Queue`` channels keyed by ``(src, dst)`` -- enough to exercise the
real reduce+broadcast control flow and catch protocol bugs (wrong
message count/order, tag mismatches, wrong agreed value) without needing
real RDMA/jaccl transport.

Covers:
1. ``group=None`` / ``group.size() <= 1`` passthrough (no exchange).
2. 2-rank unanimous agreement -> agreed value is the shared hit-length.
3. 2-rank MISMATCH -> falls back to 0 on BOTH ranks (never partial/wrong).
4. A rank reporting 0 (fresh/cold cache) forces 0 cluster-wide, even if
   its peer had a large local hit.
5. Tag mismatch on receipt raises immediately (never silently proceeds).
6. 3-rank linear chain (reduce-then-broadcast) agrees correctly, proving
   the topology generalizes beyond the 2-node case this cluster runs
   today.
"""

from __future__ import annotations

import queue
import threading
from typing import Any, cast
from unittest.mock import patch

import mlx.core as mx
import pytest

from exo.worker.engines.mlx.utils_mlx import pipeline_agree_prefix_hit_length


class _FakeGroup:
    """Minimal stand-in for ``mx.distributed.Group`` — only ``rank()`` and
    ``size()`` are used by the function under test."""

    def __init__(self, rank: int, size: int) -> None:
        self._rank = rank
        self._size = size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


class _FakeTransport:
    """Routes ``mx.distributed.send``/``recv_like`` calls through in-memory
    queues keyed by ``(src, dst)``, so N simulated ranks (real Python
    threads) can exchange int32 arrays exactly like the real jaccl p2p
    transport, blocking on queue.get() the same way a real recv blocks."""

    def __init__(self) -> None:
        self._queues: dict[tuple[int, int], "queue.Queue[Any]"] = {}

    def _q(self, src: int, dst: int) -> "queue.Queue[Any]":
        key = (src, dst)
        if key not in self._queues:
            self._queues[key] = queue.Queue()
        return self._queues[key]

    def send(self, arr: mx.array, dst: int, *, group: _FakeGroup) -> mx.array:
        src = group.rank()
        self._q(src, dst).put(arr.tolist())
        return arr

    def recv_like(self, template: mx.array, src: int, *, group: _FakeGroup) -> mx.array:
        dst = group.rank()
        payload = self._q(src, dst).get(timeout=5)
        return mx.array(payload, dtype=template.dtype)


def _run_ranks(
    world_size: int,
    local_hit_lengths: list[int],
    request_tag: int = 1,
) -> list[int | BaseException]:
    """Run ``pipeline_agree_prefix_hit_length`` concurrently for every rank
    in ``world_size``, each on its own thread, wired through a shared
    ``_FakeTransport``. Returns each rank's result (or the exception it
    raised, in call order) so tests can assert per-rank outcomes.

    Patches ``mx.distributed.send``/``recv_like`` exactly ONCE around the
    whole multi-threaded run (not per-thread) — ``unittest.mock.patch`` as
    a context manager mutates a single shared module attribute, so nested
    per-thread ``with patch(...)`` blocks race on setup/teardown across
    real OS threads (one thread's context-manager exit un-patches the
    function out from under another thread still mid-call). That's a
    test-harness hazard, not something the function under test could ever
    see for real — each cluster rank is a separate OS *process* with its
    own independent import of ``mx.distributed``.
    """
    transport = _FakeTransport()
    results: list[int | BaseException] = [0] * world_size

    def _worker(rank: int) -> None:
        group = _FakeGroup(rank, world_size)
        try:
            results[rank] = pipeline_agree_prefix_hit_length(
                local_hit_lengths[rank],
                cast(Any, group),
                request_tag,
            )
        except BaseException as e:  # noqa: BLE001 - capture for assertion
            results[rank] = e

    with (
        patch("mlx.core.distributed.send", transport.send),
        patch("mlx.core.distributed.recv_like", transport.recv_like),
    ):
        threads = [
            threading.Thread(target=_worker, args=(r,)) for r in range(world_size)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
            if t.is_alive():
                pytest.fail("simulated rank thread deadlocked (protocol bug)")
    return results


def test_group_none_passthrough() -> None:
    assert pipeline_agree_prefix_hit_length(12345, None, 1) == 12345


def test_group_size_one_passthrough() -> None:
    group = _FakeGroup(0, 1)
    assert (
        pipeline_agree_prefix_hit_length(999, cast(Any, group), 1) == 999
    )


def test_two_rank_unanimous_agreement() -> None:
    results = _run_ranks(world_size=2, local_hit_lengths=[41074, 41074])
    assert results == [41074, 41074]


def test_two_rank_unanimous_zero() -> None:
    # Both ranks cold -- should agree on 0 cleanly (not a special case).
    results = _run_ranks(world_size=2, local_hit_lengths=[0, 0])
    assert results == [0, 0]


def test_two_rank_mismatch_falls_back_to_zero_on_both() -> None:
    # Classic bug #3 scenario: rank 0 hit, rank 1 missed (or hit a
    # different depth) -- must NOT proceed with either rank's value.
    results = _run_ranks(world_size=2, local_hit_lengths=[50036, 0])
    assert results == [0, 0]


def test_two_rank_partial_mismatch_falls_back_to_zero() -> None:
    # Both ranks hit *something*, but at different depths -- still must
    # be treated as a disagreement, not "use the smaller one" (a smaller
    # non-zero value could still exceed what one rank's non-sliceable
    # layers can actually restore to).
    results = _run_ranks(world_size=2, local_hit_lengths=[30000, 25000])
    assert results == [0, 0]


def test_one_rank_cold_forces_cluster_wide_cold() -> None:
    # A crash/reconnect leaves rank 1 with an empty cache (0) while rank 0
    # still has a large hit -- must force 0 everywhere, never trim rank 0
    # down to some intermediate value.
    results = _run_ranks(world_size=2, local_hit_lengths=[100000, 0])
    assert results == [0, 0]


def test_tag_mismatch_raises_on_receiving_rank() -> None:
    """Simulate a tag mismatch by having rank 1 send under a different
    tag than rank 0 expects — this must raise on rank 0 (the receiver),
    never silently pair with the mismatched message."""
    transport = _FakeTransport()
    results: list[int | BaseException] = [0, 0]

    def _rank0() -> None:
        group = _FakeGroup(0, 2)
        try:
            results[0] = pipeline_agree_prefix_hit_length(
                100, cast(Any, group), request_tag=1
            )
        except BaseException as e:  # noqa: BLE001
            results[0] = e

    def _rank1_wrong_tag() -> None:
        group = _FakeGroup(1, 2)
        try:
            results[1] = pipeline_agree_prefix_hit_length(
                100, cast(Any, group), request_tag=999  # mismatched tag
            )
        except BaseException as e:  # noqa: BLE001
            results[1] = e

    with (
        patch("mlx.core.distributed.send", transport.send),
        patch("mlx.core.distributed.recv_like", transport.recv_like),
    ):
        t0 = threading.Thread(target=_rank0)
        t1 = threading.Thread(target=_rank1_wrong_tag)
        t0.start()
        t1.start()
        t0.join(timeout=10)
        t1.join(timeout=10)
        assert not t0.is_alive() and not t1.is_alive(), "deadlocked, not a clean raise"

    # Rank 0 receives rank 1's tag=999 message while expecting tag=1 ->
    # must raise rather than silently proceed.
    assert isinstance(results[0], RuntimeError)
    assert "tag mismatch" in str(results[0])


def test_three_rank_linear_chain_unanimous() -> None:
    results = _run_ranks(world_size=3, local_hit_lengths=[8192, 8192, 8192])
    assert results == [8192, 8192, 8192]


def test_three_rank_linear_chain_mismatch_falls_back_to_zero() -> None:
    results = _run_ranks(world_size=3, local_hit_lengths=[8192, 8192, 4096])
    assert results == [0, 0, 0]


def test_three_rank_one_cold_forces_cluster_wide_cold() -> None:
    results = _run_ranks(world_size=3, local_hit_lengths=[20000, 20000, 0])
    assert results == [0, 0, 0]
