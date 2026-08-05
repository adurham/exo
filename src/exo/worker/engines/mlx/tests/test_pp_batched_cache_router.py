# pyright: reportPrivateUsage=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
"""Tests for pp_batched_cache_router.py -- Phase 1 per-request cache
routing.

See ``docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md`` Section
6.2 item 3.

Two test classes:
1. ``BatchedCacheRouter`` slot-lifecycle bookkeeping -- pure Python,
   no MLX arrays, fast unit tests.
2. ``merge_request_caches``/``extract_request_cache`` -- REAL mlx-lm
   cache objects (KVCache, CacheList(RotatingKVCache, PoolingCache)),
   verifying actual data round-trips correctly through merge/extract
   at heterogeneous per-request lengths -- not just that the functions
   don't crash, but that request A's tokens never leak into request
   B's extracted cache (the exact silent-cross-request-corruption
   failure mode the design doc's Risk #5 warns about).
"""

from __future__ import annotations

import mlx.core as mx
import pytest
from mlx_lm.models.cache import CacheList, KVCache, PoolingCache

from exo.worker.engines.mlx.pp_batched_cache_router import (
    BatchedCacheRouter,
    CacheRouterError,
    extract_request_cache,
    merge_request_caches,
)

# ---------------------------------------------------------------------
# BatchedCacheRouter -- pure bookkeeping, no MLX
# ---------------------------------------------------------------------


def test_new_router_all_slots_free() -> None:
    router = BatchedCacheRouter(max_concurrency=2)
    assert not router.is_occupied(0)
    assert not router.is_occupied(1)
    assert router.occupied_slots() == ()


def test_assign_slot_marks_occupied_at_length_zero() -> None:
    router = BatchedCacheRouter(max_concurrency=2)
    router.assign_slot(0)
    assert router.is_occupied(0)
    assert router.length(0) == 0
    assert router.occupied_slots() == (0,)


def test_assign_already_occupied_slot_raises() -> None:
    router = BatchedCacheRouter(max_concurrency=2)
    router.assign_slot(0)
    with pytest.raises(CacheRouterError, match="already"):
        router.assign_slot(0)


def test_out_of_range_slot_raises() -> None:
    router = BatchedCacheRouter(max_concurrency=2)
    with pytest.raises(CacheRouterError, match="out of range"):
        router.assign_slot(5)


def test_advance_slot_increments_length() -> None:
    router = BatchedCacheRouter(max_concurrency=2)
    router.assign_slot(0)
    router.advance_slot(0)
    assert router.length(0) == 1
    router.advance_slot(0)
    assert router.length(0) == 2


def test_advance_unoccupied_slot_raises() -> None:
    router = BatchedCacheRouter(max_concurrency=2)
    with pytest.raises(CacheRouterError, match="not occupied"):
        router.advance_slot(0)


def test_advance_with_zero_or_negative_tokens_raises() -> None:
    router = BatchedCacheRouter(max_concurrency=2)
    router.assign_slot(0)
    with pytest.raises(CacheRouterError, match="n_tokens"):
        router.advance_slot(0, n_tokens=0)
    with pytest.raises(CacheRouterError, match="n_tokens"):
        router.advance_slot(0, n_tokens=-1)


def test_release_slot_resets_occupancy_and_length() -> None:
    router = BatchedCacheRouter(max_concurrency=2)
    router.assign_slot(0)
    router.advance_slot(0)
    router.advance_slot(0)
    router.release_slot(0)
    assert not router.is_occupied(0)
    assert router.length(0) == 0


def test_release_unoccupied_slot_raises() -> None:
    router = BatchedCacheRouter(max_concurrency=2)
    with pytest.raises(CacheRouterError, match="not occupied"):
        router.release_slot(0)


def test_slot_reuse_after_release_starts_at_length_zero() -> None:
    """The core reset-on-assign invariant (module docstring point 4):
    after release + reassign, a slot's length is genuinely 0, not
    stale from the prior occupant."""
    router = BatchedCacheRouter(max_concurrency=2)
    router.assign_slot(0)
    router.advance_slot(0, n_tokens=50)
    assert router.length(0) == 50
    router.release_slot(0)
    router.assign_slot(0)
    assert router.length(0) == 0


def test_occupied_slots_ascending_order() -> None:
    router = BatchedCacheRouter(max_concurrency=3)
    router.assign_slot(2)
    router.assign_slot(0)
    router.assign_slot(1)
    assert router.occupied_slots() == (0, 1, 2)


def test_invalid_max_concurrency_raises() -> None:
    with pytest.raises(ValueError, match="max_concurrency"):
        BatchedCacheRouter(max_concurrency=0)


# ---------------------------------------------------------------------
# merge_request_caches / extract_request_cache -- real mlx-lm cache
# objects, real data, verifying no cross-request corruption.
# ---------------------------------------------------------------------


def _make_plain_kv_request_cache(fill_value: float, n_tokens: int) -> list[KVCache]:
    """Two-layer plain-KVCache request cache (like mlx-lm's Llama),
    filled with an identifiable constant so cross-request leakage is
    trivially detectable."""
    layers: list[KVCache] = []
    for _ in range(2):
        c = KVCache()
        k = mx.ones((1, 2, n_tokens, 8), dtype=mx.float32) * fill_value
        v = mx.ones((1, 2, n_tokens, 8), dtype=mx.float32) * fill_value
        c.update_and_fetch(k, v)
        layers.append(c)
    return layers


def test_merge_then_extract_roundtrips_correct_data_same_length() -> None:
    cache_a = _make_plain_kv_request_cache(fill_value=1.0, n_tokens=4)
    cache_b = _make_plain_kv_request_cache(fill_value=2.0, n_tokens=4)

    merged = merge_request_caches([cache_a, cache_b])
    assert len(merged) == 2  # 2 layers

    extracted_a = extract_request_cache(merged, 0)
    extracted_b = extract_request_cache(merged, 1)

    for layer in extracted_a:
        assert bool(mx.all(layer.keys == 1.0))  # type: ignore[union-attr]
        assert bool(mx.all(layer.values == 1.0))  # type: ignore[union-attr]
    for layer in extracted_b:
        assert bool(mx.all(layer.keys == 2.0))  # type: ignore[union-attr]
        assert bool(mx.all(layer.values == 2.0))  # type: ignore[union-attr]


def test_merge_then_extract_roundtrips_correct_data_heterogeneous_length() -> None:
    """THE critical no-cross-request-corruption test: two requests at
    DIFFERENT cache lengths (mixed prefill/decode-progress state, the
    realistic Phase 1 scenario) must extract back to exactly their own
    data, at exactly their own length -- not padded/truncated/mixed
    with the other request's tokens."""
    cache_a = _make_plain_kv_request_cache(fill_value=7.0, n_tokens=3)
    cache_b = _make_plain_kv_request_cache(fill_value=9.0, n_tokens=10)

    merged = merge_request_caches([cache_a, cache_b])

    extracted_a = extract_request_cache(merged, 0)
    extracted_b = extract_request_cache(merged, 1)

    for layer in extracted_a:
        assert layer.keys.shape[2] == 3  # type: ignore[union-attr]
        assert bool(mx.all(layer.keys == 7.0))  # type: ignore[union-attr]
    for layer in extracted_b:
        assert layer.keys.shape[2] == 10  # type: ignore[union-attr]
        assert bool(mx.all(layer.keys == 9.0))  # type: ignore[union-attr]


def test_merge_three_requests_no_cross_contamination() -> None:
    """N>2 requests (still well within a single batched cache, even
    though N>2 CONCURRENCY is out of scope for the scheduler per the
    design doc -- this exercises the cache-merge primitive's own
    generality, independent of the scheduler's N=2 policy limit)."""
    cache_a = _make_plain_kv_request_cache(fill_value=1.0, n_tokens=2)
    cache_b = _make_plain_kv_request_cache(fill_value=2.0, n_tokens=5)
    cache_c = _make_plain_kv_request_cache(fill_value=3.0, n_tokens=8)

    merged = merge_request_caches([cache_a, cache_b, cache_c])

    for idx, expected_value, expected_len in (
        (0, 1.0, 2),
        (1, 2.0, 5),
        (2, 3.0, 8),
    ):
        extracted = extract_request_cache(merged, idx)
        for layer in extracted:
            assert layer.keys.shape[2] == expected_len  # type: ignore[union-attr]
            assert bool(mx.all(layer.keys == expected_value))  # type: ignore[union-attr]


def test_merge_empty_list_raises() -> None:
    with pytest.raises(CacheRouterError, match="empty"):
        merge_request_caches([])


def test_merge_mismatched_layer_counts_raises() -> None:
    cache_a = _make_plain_kv_request_cache(fill_value=1.0, n_tokens=2)
    cache_b = cache_a[:1]  # only 1 layer instead of 2
    with pytest.raises(CacheRouterError, match="layers"):
        merge_request_caches([cache_a, cache_b])


def _make_cachelist_request_cache(
    fill_value: float, n_tokens: int, *, ratio: int = 4
) -> list[CacheList]:
    """Two-layer CacheList(RotatingKVCache, PoolingCache) request
    cache, mirroring DSv4's actual make_cache() structure for layers
    with a nonzero compress_ratio (the SparseCompressedAttention case
    minus the second indexer PoolingCache, kept to 2 sub-caches for a
    focused test)."""
    from mlx_lm.models.cache import RotatingKVCache

    layers: list[CacheList] = []
    for _ in range(2):
        rotating = RotatingKVCache(max_size=64)
        k = mx.ones((1, 2, n_tokens, 8), dtype=mx.float32) * fill_value
        v = mx.ones((1, 2, n_tokens, 8), dtype=mx.float32) * fill_value
        rotating.update_and_fetch(k, v)
        pooling = PoolingCache(ratio)
        layers.append(CacheList(rotating, pooling))
    return layers


def test_merge_then_extract_cachelist_structure_roundtrips() -> None:
    """DSv4's REAL cache shape (CacheList wrapping RotatingKVCache +
    PoolingCache) -- confirms merge()/extract() compose correctly
    through the CacheList recursive structure, not just for a bare
    KVCache layer list."""
    cache_a = _make_cachelist_request_cache(fill_value=4.0, n_tokens=6)
    cache_b = _make_cachelist_request_cache(fill_value=5.0, n_tokens=9)

    merged = merge_request_caches([cache_a, cache_b])
    assert len(merged) == 2

    extracted_a = extract_request_cache(merged, 0)
    extracted_b = extract_request_cache(merged, 1)

    for cache_list_layer in extracted_a:
        rotating_sub = cache_list_layer.caches[0]  # type: ignore[union-attr]
        assert rotating_sub.keys.shape[2] == 6
        assert bool(mx.all(rotating_sub.keys == 4.0))
    for cache_list_layer in extracted_b:
        rotating_sub = cache_list_layer.caches[0]  # type: ignore[union-attr]
        assert rotating_sub.keys.shape[2] == 9
        assert bool(mx.all(rotating_sub.keys == 5.0))
