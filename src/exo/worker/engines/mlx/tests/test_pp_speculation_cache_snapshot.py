# pyright: reportPrivateUsage=false
"""Unit tests for pp_speculation._snapshot_cache / _restore_cache.

ROOT CAUSE (2026-07-19, live cluster debugging): the ORIGINAL
implementation only recognized bare ``ArraysCache``/``KVCache`` via
isinstance and silently no-op'd (via a fallthrough ``snap.append(None)``
+ "if s is None: continue" restore) for every other cache type --
including ``RotatingKVCache`` and ``CacheList``, which is EXACTLY what
DeepseekV4Model.make_cache() returns for every layer. So step 3a's
"snapshot before speculative forward, restore on both HIT and MISS"
invariant was a complete no-op on the actual model this code runs
against: the speculative forward's real KV writes never got rolled
back. Live-tested consequences: a degenerate repeat-loop and a jaccl
transport fault, both reproduced on isolated single requests and both
absent when the speculative-forward flag was off.

These tests exercise the FIXED _snapshot_cache/_restore_cache directly
against real RotatingKVCache/CacheList instances (the types that were
silently falling through before) with real MLX arrays -- no cluster
needed, this is exactly the kind of test that would have caught the
original bug before it ever touched hardware.
"""

from __future__ import annotations

import mlx.core as mx
from mlx_lm.models.cache import (
    ArraysCache,
    CacheList,
    KVCache,
    PoolingCache,  # pyright: ignore
    RotatingKVCache,
)

from exo.worker.engines.mlx.pp_speculation import _restore_cache, _snapshot_cache

# ---------------------------------------------------------------------------
# Shared fixtures: tiny real key/value tensors, no real model needed.
# ---------------------------------------------------------------------------

_B, _H, _D = 1, 2, 4  # batch, heads, head_dim -- kept tiny for fast tests


def _kv(seq_len: int, fill: float) -> tuple[mx.array, mx.array]:
    """A (B, H, seq_len, D) key/value pair filled with a distinct constant
    so post-corruption reads are trivially distinguishable from the
    pre-corruption snapshot."""
    k = mx.full((_B, _H, seq_len, _D), fill, dtype=mx.float32)
    v = mx.full((_B, _H, seq_len, _D), fill + 0.5, dtype=mx.float32)
    mx.eval(k, v)
    return k, v


def _keys(cache: RotatingKVCache | KVCache) -> mx.array:
    """Narrow ``cache.keys`` (typed ``array | None`` in the mlx-lm stub)
    for tests -- always populated by the time these tests read it."""
    assert cache.keys is not None
    return cache.keys


def _values(cache: RotatingKVCache | KVCache) -> mx.array:
    """Narrow ``cache.values`` the same way as `_keys`."""
    assert cache.values is not None
    return cache.values


# ---------------------------------------------------------------------------
# RotatingKVCache -- the type EVERY DSv4 layer uses (ratio == 0 layers)
# ---------------------------------------------------------------------------


def test_rotating_kv_cache_restore_undoes_speculative_write_pre_wrap() -> None:
    """Before the ring wraps: write real tokens, snapshot, speculatively
    write more, restore, and confirm the cache is back to EXACTLY the
    pre-speculative state (not just offset-truncated -- byte-identical
    keys/values, matching RotatingKVCache.save_spec_state's own
    materialized-copy contract)."""
    cache = RotatingKVCache(max_size=16)
    k1, v1 = _kv(4, fill=1.0)
    cache.update_and_fetch(k1, v1)  # pyright: ignore[reportUnknownMemberType]
    assert cache.offset == 4

    snap = _snapshot_cache([cache])

    # Speculative write: 3 more "fake" tokens that must be fully undone.
    k_spec, v_spec = _kv(3, fill=99.0)
    cache.update_and_fetch(k_spec, v_spec)  # pyright: ignore[reportUnknownMemberType]
    assert cache.offset == 7
    # Sanity: the speculative write really did land in the live buffer.
    assert float(_keys(cache)[0, 0, 4, 0].item()) == 99.0

    _restore_cache([cache], snap)

    assert cache.offset == 4, "restore must undo the speculative offset advance"
    assert _keys(cache).shape[2] >= 4
    for i in range(4):
        assert float(_keys(cache)[0, 0, i, 0].item()) == 1.0, (
            f"restored key at position {i} must match the pre-speculative "
            f"value -- a stale/aliased restore would leave the speculative "
            f"write's 99.0 fill visible"
        )
        assert float(_values(cache)[0, 0, i, 0].item()) == 1.5


def test_rotating_kv_cache_restore_undoes_speculative_write_post_wrap() -> None:
    """After the ring has wrapped at least once: trim() cannot undo this
    (RotatingKVCache.trim() is documented elsewhere in this codebase as
    only valid pre-wraparound) -- save_spec_state/restore_spec_state's
    full materialized-array-copy approach is REQUIRED here. This is the
    scenario the original bug's isinstance-based fallthrough silently
    ignored."""
    cache = RotatingKVCache(max_size=8)
    k_fill, v_fill = _kv(8, fill=2.0)
    cache.update_and_fetch(k_fill, v_fill)  # fills to max_size, ring active  # pyright: ignore[reportUnknownMemberType]
    assert cache.offset == 8

    snap = _snapshot_cache([cache])
    pre_restore_key_sample = float(_keys(cache)[0, 0, cache._idx - 1, 0].item())

    # Speculative write that wraps the ring (overwrites oldest rows).
    k_spec, v_spec = _kv(2, fill=77.0)
    cache.update_and_fetch(k_spec, v_spec)  # pyright: ignore[reportUnknownMemberType]
    assert cache.offset == 10
    # The ring wrapped -- some physical row now holds the speculative fill.
    assert float(_keys(cache)[0, 0, cache._idx - 1, 0].item()) == 77.0

    _restore_cache([cache], snap)

    assert cache.offset == 8, "restore must undo the wrapped speculative write"
    assert float(_keys(cache)[0, 0, cache._idx - 1, 0].item()) == pre_restore_key_sample


# ---------------------------------------------------------------------------
# CacheList -- DSv4's sparse/compressed-attention layer wrapper
# ---------------------------------------------------------------------------


def test_cache_list_restore_recurses_into_wrapped_rotating_kv_cache() -> None:
    """DeepseekV4Model.make_cache() wraps sparse/compressed layers in
    CacheList(RotatingKVCache(...), PoolingCache(...), [PoolingCache(...)]).
    The ORIGINAL bug's isinstance check never recognized CacheList at
    all -- confirm the fix recurses into .caches and restores the
    wrapped RotatingKVCache correctly."""
    inner = RotatingKVCache(max_size=16)
    wrapped = CacheList(inner)

    k1, v1 = _kv(5, fill=3.0)
    inner.update_and_fetch(k1, v1)  # pyright: ignore[reportUnknownMemberType]
    assert inner.offset == 5

    snap = _snapshot_cache([wrapped])

    k_spec, v_spec = _kv(4, fill=88.0)
    inner.update_and_fetch(k_spec, v_spec)  # pyright: ignore[reportUnknownMemberType]
    assert inner.offset == 9

    _restore_cache([wrapped], snap)

    assert inner.offset == 5, (
        "restoring a CacheList must recurse into the wrapped RotatingKVCache "
        "and undo ITS speculative write, not just no-op at the CacheList level"
    )
    for i in range(5):
        assert float(_keys(inner)[0, 0, i, 0].item()) == 3.0


# ---------------------------------------------------------------------------
# Multi-layer cache list (shape of the real prompt_cache argument)
# ---------------------------------------------------------------------------


def test_snapshot_restore_round_trips_a_realistic_mixed_cache_list() -> None:
    """A prompt_cache list mixing RotatingKVCache (plain layers) and
    CacheList-wrapped RotatingKVCache (sparse/compressed layers) --
    the EXACT shape DeepseekV4Model.make_cache() returns. Confirms the
    fix handles a full multi-layer list, not just a single entry."""
    plain_layer = RotatingKVCache(max_size=16)
    wrapped_layer = CacheList(RotatingKVCache(max_size=16))
    prompt_cache: list[object] = [plain_layer, wrapped_layer]

    k1, v1 = _kv(3, fill=5.0)
    plain_layer.update_and_fetch(k1, v1)  # pyright: ignore[reportUnknownMemberType]
    inner = wrapped_layer.caches[0]  # type: ignore[attr-defined]
    assert isinstance(inner, RotatingKVCache)
    inner.update_and_fetch(k1, v1)  # pyright: ignore[reportUnknownMemberType]

    snap = _snapshot_cache(prompt_cache)

    k_spec, v_spec = _kv(2, fill=66.0)
    plain_layer.update_and_fetch(k_spec, v_spec)  # pyright: ignore[reportUnknownMemberType]
    inner.update_and_fetch(k_spec, v_spec)  # pyright: ignore[reportUnknownMemberType]
    assert plain_layer.offset == 5
    assert inner.offset == 5

    _restore_cache(prompt_cache, snap)

    assert plain_layer.offset == 3
    assert inner.offset == 3
    assert float(_keys(plain_layer)[0, 0, 0, 0].item()) == 5.0
    assert float(_keys(inner)[0, 0, 0, 0].item()) == 5.0


# ---------------------------------------------------------------------------
# Generic fallback path (ArraysCache / bare KVCache) -- the ORIGINAL bug's
# only correctly-handled types. Confirm the fix doesn't regress these.
# ---------------------------------------------------------------------------


def test_arrays_cache_still_round_trips_via_generic_state_fallback() -> None:
    """ArraysCache has no save_spec_state and isn't a CacheList, so it
    goes through the generic state/meta_state fallback path. Confirm
    this still works (it was one of the two types the OLD code handled
    correctly -- must not regress)."""
    cache = ArraysCache(1)
    k1, _v1 = _kv(1, fill=9.0)
    cache.cache = [k1]  # type: ignore[attr-defined]

    snap = _snapshot_cache([cache])

    k_spec, _v_spec = _kv(1, fill=44.0)
    cache.cache = [k_spec]  # type: ignore[attr-defined]

    _restore_cache([cache], snap)

    assert float(cache.cache[0][0, 0, 0, 0].item()) == 9.0, (  # type: ignore[attr-defined]
        "generic-fallback restore must produce an INDEPENDENT copy, not an "
        "aliased reference back to the (since-mutated) live list"
    )


def test_bare_kv_cache_still_round_trips_via_generic_state_fallback() -> None:
    """Bare KVCache (no max_size cap) also goes through the generic
    fallback -- confirm offset + keys/values restore correctly and
    that the copy is independent (tree_map(mx.array, ...) must not
    alias the live buffer)."""
    cache = KVCache()
    k1, v1 = _kv(4, fill=6.0)
    cache.update_and_fetch(k1, v1)  # pyright: ignore[reportUnknownMemberType]
    assert cache.offset == 4

    snap = _snapshot_cache([cache])

    k_spec, v_spec = _kv(3, fill=55.0)
    cache.update_and_fetch(k_spec, v_spec)  # pyright: ignore[reportUnknownMemberType]
    assert cache.offset == 7

    _restore_cache([cache], snap)

    assert cache.offset == 4
    for i in range(4):
        assert float(_keys(cache)[0, 0, i, 0].item()) == 6.0


# ---------------------------------------------------------------------------
# The exact no-corruption invariant EXECUTE=1 depends on: after a
# snapshot+speculative-write+restore cycle, a completely independent
# fresh cache (never touched by the speculative path) must be
# numerically indistinguishable from the restored one.
# ---------------------------------------------------------------------------


def test_restored_cache_is_indistinguishable_from_a_cache_that_never_ran_spec_fwd() -> (
    None
):
    """This is the actual behavior-preservation property step 3a's design
    doc claims EXECUTE=1 alone provides. Build two caches with identical
    real history; run the speculative-forward-then-restore cycle on only
    one; confirm both end up byte-identical."""
    real_k, real_v = _kv(6, fill=1.0)

    baseline = RotatingKVCache(max_size=20)
    baseline.update_and_fetch(real_k, real_v)  # pyright: ignore[reportUnknownMemberType]

    executed = RotatingKVCache(max_size=20)
    executed.update_and_fetch(real_k, real_v)  # pyright: ignore[reportUnknownMemberType]

    snap = _snapshot_cache([executed])
    spec_k, spec_v = _kv(2, fill=999.0)
    executed.update_and_fetch(spec_k, spec_v)  # the "speculative forward"  # pyright: ignore[reportUnknownMemberType]
    _restore_cache([executed], snap)  # step 3a's restore-on-both-paths

    assert executed.offset == baseline.offset
    assert bool(mx.array_equal(_keys(executed)[0, 0, :6, :], _keys(baseline)[0, 0, :6, :]))
    assert bool(mx.array_equal(_values(executed)[0, 0, :6, :], _values(baseline)[0, 0, :6, :]))


# ---------------------------------------------------------------------------
# PoolingCache -- the generic-fallback type whose .state legitimately
# contains None leaves. Second bug found (2026-07-19, same session as
# the isinstance-fallthrough root cause): the FIRST fix attempt's
# tree_map(mx.array, state) crashed both runners on the first live
# request with "Invoked with types: mlx.core.array, NoneType", because
# a fresh/untouched PoolingCache's .state is (None, None, None). This
# reproduces that exact crash directly, with no cluster needed.
# ---------------------------------------------------------------------------


def test_pooling_cache_with_untouched_buffers_does_not_crash_snapshot() -> None:
    """DeepseekV4Model.make_cache() wraps EVERY sparse/compressed layer's
    CacheList with one or two PoolingCache instances. A freshly
    constructed PoolingCache (buf_kv/buf_gate/pooled all still None,
    before any accumulate_windows() call -- exactly the state a cache
    is in for the first several decode steps of any request) must
    snapshot and restore without crashing."""
    cache = PoolingCache(ratio=4)  # pyright: ignore
    assert cache.state == (None, None, None)  # pyright: ignore

    snap = _snapshot_cache([cache])  # must NOT raise

    _restore_cache([cache], snap)  # must NOT raise either

    assert cache.state == (None, None, None)  # pyright: ignore


def test_pooling_cache_with_partial_remainder_round_trips() -> None:
    """Once accumulate_windows() has run at least once (buf_kv/buf_gate
    allocated, possibly still ratio-unfilled so pooled stays None),
    confirm the generic fallback still round-trips real array data
    correctly alongside the None `pooled` leaf."""
    cache = PoolingCache(ratio=4)  # pyright: ignore
    kv = mx.zeros((1, 2, 4), dtype=mx.float32)
    gate = mx.zeros((1, 2, 1), dtype=mx.float32)
    mx.eval(kv, gate)
    cache.accumulate_windows(kv, gate, offset=2)  # pyright: ignore[reportUnknownMemberType]
    assert cache.remainder == 2  # pyright: ignore
    assert cache.pooled is None  # ratio=4 not yet reached -- still None  # pyright: ignore

    snap = _snapshot_cache([cache])

    kv2 = mx.ones((1, 2, 4), dtype=mx.float32)
    gate2 = mx.ones((1, 2, 1), dtype=mx.float32)
    mx.eval(kv2, gate2)
    cache.accumulate_windows(kv2, gate2, offset=4)  # pyright: ignore[reportUnknownMemberType]

    _restore_cache([cache], snap)

    assert cache.remainder == 2, "restore must undo the speculative accumulate"
    assert cache.pooled is None  # pyright: ignore

