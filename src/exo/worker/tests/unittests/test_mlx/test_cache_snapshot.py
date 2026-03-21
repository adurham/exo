"""Tests for cache snapshot/restore functionality used by PP idle-time speculation."""
import mlx.core as mx
import pytest
from mlx_lm.models.cache import (
    ArraysCache,
    KVCache,
    restore_cache,
    snapshot_cache,
)


def test_kvcache_snapshot_restore():
    """KVCache snapshot saves offset; restore rolls back offset."""
    cache = KVCache()
    # Simulate prefill: write some data
    keys = mx.random.normal((1, 4, 10, 64))
    values = mx.random.normal((1, 4, 10, 64))
    cache.update_and_fetch(keys, values)
    assert cache.offset == 10

    # Take snapshot
    snap = cache.snapshot()
    assert snap == 10

    # Simulate decode step: add one more token
    k1 = mx.random.normal((1, 4, 1, 64))
    v1 = mx.random.normal((1, 4, 1, 64))
    cache.update_and_fetch(k1, v1)
    assert cache.offset == 11

    # Restore: offset goes back to 10
    cache.restore(snap)
    assert cache.offset == 10


def test_arrayscache_snapshot_restore():
    """ArraysCache snapshot saves references; restore replaces the list."""
    cache = ArraysCache(size=2)
    # Simulate initial state
    state_0 = mx.ones((1, 1, 128))
    state_1 = mx.zeros((1, 1, 128))
    cache[0] = state_0
    cache[1] = state_1
    mx.eval(state_0, state_1)

    # Take snapshot (shallow copy of list)
    snap = cache.snapshot()
    assert len(snap) == 2
    # Snapshot references same arrays
    assert snap[0] is state_0
    assert snap[1] is state_1

    # Simulate DeltaNet step: new arrays replace old ones
    new_state_0 = mx.ones((1, 1, 128)) * 2
    new_state_1 = mx.zeros((1, 1, 128)) * 3
    cache[0] = new_state_0
    cache[1] = new_state_1
    mx.eval(new_state_0, new_state_1)

    # Snapshot originals are untouched (no in-place mutation)
    assert mx.array_equal(snap[0], state_0)
    assert mx.array_equal(snap[1], state_1)

    # Restore: cache goes back to original arrays
    cache.restore(snap)
    assert cache[0] is state_0
    assert cache[1] is state_1


def test_snapshot_cache_restore_cache_helpers():
    """Top-level helpers snapshot/restore a mixed cache list."""
    kv = KVCache()
    arr = ArraysCache(size=2)

    # Populate
    kv.update_and_fetch(mx.random.normal((1, 4, 5, 64)), mx.random.normal((1, 4, 5, 64)))
    arr[0] = mx.ones((1, 1, 64))
    arr[1] = mx.zeros((1, 1, 64))
    mx.eval(arr[0], arr[1])

    cache_list = [kv, arr]
    snaps = snapshot_cache(cache_list)

    # Advance both caches
    kv.update_and_fetch(mx.random.normal((1, 4, 1, 64)), mx.random.normal((1, 4, 1, 64)))
    arr[0] = mx.ones((1, 1, 64)) * 99
    arr[1] = mx.zeros((1, 1, 64)) * 99
    assert kv.offset == 6
    assert mx.array_equal(arr[0], mx.ones((1, 1, 64)) * 99)

    # Restore
    restore_cache(cache_list, snaps)
    assert kv.offset == 5
    assert mx.array_equal(arr[0], mx.ones((1, 1, 64)))


def test_kvcache_snapshot_idempotent():
    """Multiple snapshots don't interfere with each other."""
    cache = KVCache()
    cache.update_and_fetch(mx.random.normal((1, 4, 5, 64)), mx.random.normal((1, 4, 5, 64)))
    snap1 = cache.snapshot()  # offset=5

    cache.update_and_fetch(mx.random.normal((1, 4, 3, 64)), mx.random.normal((1, 4, 3, 64)))
    snap2 = cache.snapshot()  # offset=8

    cache.update_and_fetch(mx.random.normal((1, 4, 2, 64)), mx.random.normal((1, 4, 2, 64)))
    assert cache.offset == 10

    # Restore to snap2
    cache.restore(snap2)
    assert cache.offset == 8

    # Restore to snap1
    cache.restore(snap1)
    assert cache.offset == 5
