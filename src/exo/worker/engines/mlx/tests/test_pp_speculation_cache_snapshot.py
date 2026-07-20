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
    BatchPoolingCache,  # pyright: ignore
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


# ---------------------------------------------------------------------------
# PoolingCache growth-path corruption bug (2026-07-20, ELEVENTH UPDATE).
#
# ROOT CAUSE: PoolingCache previously had no save_spec_state, so
# _snapshot_one fell through to the generic .state/.meta_state protocol.
# That protocol's `pooled` property setter unconditionally reallocates
# `_pool_storage` and does NOT round-trip `_pending_offset_bump` (a bare
# attribute used by the W4 deferred-update path, update_and_fetch_deferred /
# commit_pending -- not part of `.state`/`.meta_state` at all). A snapshot
# taken while a bump was staged, then restored after a rejection, left a
# STALE bump on the live object. The next commit_pending() (called
# unconditionally at the top of every Compressor.__call__) applied that
# orphaned bump on top of the freshly-restored (smaller) _pool_offset with
# no corresponding storage resize, corrupting _pool_offset past
# _pool_storage.shape[1]. The following deferred write's growth branch then
# sliced `old[:, :self._pool_offset]` against a too-small `old`, producing
# a `[broadcast_shapes]` crash on the very first decode cycle after a
# large-context growth event (observed live at ~500K context).
#
# FIX: PoolingCache/BatchPoolingCache now expose save_spec_state/
# restore_spec_state (aliasing the already-correct save_meta/restore_meta,
# which DOES round-trip the pending bump), so _snapshot_one's
# hasattr(c, "save_spec_state") branch picks up the purpose-built rollback
# instead of falling through to the generic protocol.
# ---------------------------------------------------------------------------


def test_pooling_cache_snapshot_uses_pool_meta_not_generic_fallback() -> None:
    """After the fix, PoolingCache must dispatch through the dedicated
    'pool_meta' branch (isinstance-checked save_meta/restore_meta), not
    the generic .state/.meta_state fallback, so _pending_offset_bump is
    round-tripped. Regression guard: if this ever starts returning a
    "generic" snapshot kind again, the growth-path bug below is silently
    reopened.

    NOTE: this is dispatched via isinstance, NOT via a save_spec_state
    duck-type -- adding save_spec_state to PoolingCache would collide
    with dsv4_mtp.py's OWN hasattr(sub, "save_spec_state") ring-cache
    collection (which expects rollback_spec_write, absent on
    PoolingCache) and crash that unrelated path. Confirm no such
    attribute leaked onto PoolingCache by this fix."""
    cache = PoolingCache(ratio=4)  # pyright: ignore
    assert not hasattr(cache, "save_spec_state"), (
        "PoolingCache must NOT gain save_spec_state -- that would get it "
        "swept into dsv4_mtp.py's ring-cache rollback collection "
        "(hasattr(sub, 'save_spec_state')) and crash with "
        "AttributeError: 'PoolingCache' object has no attribute "
        "'rollback_spec_write'"
    )

    snap = _snapshot_cache([cache])
    assert snap[0][0] == "pool_meta", (
        "PoolingCache must snapshot via the dedicated 'pool_meta' branch, "
        "not 'generic' -- the generic .state/.meta_state protocol does not "
        "round-trip _pending_offset_bump and reintroduces the growth-path "
        "corruption bug"
    )


def test_pooling_cache_growth_after_reject_does_not_corrupt_offset() -> None:
    """Direct repro of the ELEVENTH UPDATE crash: snapshot mid-verify while
    a deferred bump is staged, simulate more speculative draft steps that
    grow pool storage, reject (restore), then push the real committed
    continuation through. Before the fix this corrupted _pool_offset past
    _pool_storage.shape[1] and crashed on the next deferred write with
    [broadcast_shapes]; after the fix, _pool_offset must never exceed
    _pool_storage.shape[1] and the write must succeed."""
    cache = PoolingCache(ratio=4)  # pyright: ignore
    cache.step = 8  # small step so growth boundaries are reachable cheaply

    # Build a committed baseline near a storage boundary (P=7, storage=8).
    cache.update_and_fetch_deferred(mx.ones((1, 7, 4)))  # pyright: ignore[reportUnknownMemberType]
    cache.commit_pending()
    assert cache._pool_offset == 7  # pyright: ignore
    assert cache._pool_storage.shape[1] == 8  # pyright: ignore

    snap = _snapshot_cache([cache])

    # Simulate a speculative verify forward: stage a large deferred write
    # that would force growth once committed (offset 7 -> 12, past storage=8).
    cache.commit_pending()
    cache.update_and_fetch_deferred(mx.ones((1, 5, 4)) * 9)  # pyright: ignore[reportUnknownMemberType]
    assert cache._pool_storage.shape[1] == 16  # pyright: ignore # grew for the speculative write

    # REJECTED: restore from snapshot (the EXECUTE=1 rollback path).
    _restore_cache([cache], snap)

    # The next Compressor.__call__ always calls commit_pending() first.
    cache.commit_pending()

    assert cache._pool_offset <= cache._pool_storage.shape[1], (  # pyright: ignore
        f"_pool_offset ({cache._pool_offset}) exceeds "  # pyright: ignore
        f"_pool_storage.shape[1] ({cache._pool_storage.shape[1]}) after "  # pyright: ignore
        f"restore+commit -- this is the exact corrupted state that crashed "
        f"the next deferred write with [broadcast_shapes] at ~500K context"
    )
    assert cache._pool_offset == 7, "restore must fully undo the rejected draft's staged bump"  # pyright: ignore

    # The real committed continuation must write cleanly (this is the
    # exact call that crashed live before the fix).
    result = cache.update_and_fetch_deferred(mx.ones((1, 1, 4)) * 99)  # pyright: ignore[reportUnknownMemberType]
    mx.eval(result)  # pyright: ignore[reportUnknownMemberType]


def test_pooling_cache_not_swept_into_dsv4_mtp_ring_cache_collection() -> None:
    """dsv4_mtp.py's OWN speculative-rollback path (separate from
    pp_speculation.py's EXECUTE=1 path tested above) collects "ring-like"
    caches via `hasattr(sub, "save_spec_state")` and calls
    `rollback_spec_write` on every cache it collects that way (see
    dsv4_mtp.py ~line 3583 and ~line 4218). If PoolingCache were given a
    save_spec_state method (the WRONG fix, rejected during review), it
    would get swept into that collection too and crash with
    AttributeError the next time dsv4_mtp's rollback runs, since
    PoolingCache has no rollback_spec_write. This test guards against
    that regression directly."""
    cache = PoolingCache(ratio=4)  # pyright: ignore
    assert not hasattr(cache, "save_spec_state")
    assert not hasattr(cache, "restore_spec_state")
    # PoolingCache correctly lacks rollback_spec_write -- if a future
    # change added save_spec_state without also adding this, dsv4_mtp's
    # ring-cache rollback would crash on the first PoolingCache it swept up.
    assert not hasattr(cache, "rollback_spec_write")


def test_batch_pooling_cache_snapshot_preserves_pending_bumps() -> None:
    """BatchPoolingCache twin of the PoolingCache growth-corruption test:
    confirms the 'pool_meta' dispatch branch also covers the batched
    (c>=2) pool path, whose staged state is a PER-STREAM ``_pending_bumps``
    list rather than a single ``_pending_offset_bump`` int, but has the
    identical generic-.state/.meta_state-fallback gap before the fix
    (that protocol doesn't round-trip ``_pending_bumps`` either).

    Drives ``_pool_lengths``/``_pending_bumps`` directly rather than via
    the full ``accumulate_windows``/``update_and_fetch_deferred``
    production call sequence (which additionally requires ``prepare()``-
    supplied per-stream ``_lengths``/``_processed`` bookkeeping unrelated
    to this bug) -- this isolates exactly the state this fix is
    responsible for round-tripping.
    """
    cache = BatchPoolingCache(ratio=4, left_padding=[0])  # pyright: ignore

    # Simulate committed state: one stream at pool length 7.
    cache._pool_lengths = [7]  # pyright: ignore
    assert not hasattr(cache, "save_spec_state"), (
        "BatchPoolingCache must NOT gain save_spec_state either -- same "
        "ring-cache-collection collision risk as PoolingCache"
    )

    snap = _snapshot_cache([cache])
    assert snap[0][0] == "pool_meta", (
        "BatchPoolingCache must also dispatch via the dedicated "
        "'pool_meta' branch, not the generic fallback"
    )

    # Simulate a speculative draft step staging a per-stream bump.
    cache._pending_bumps = [3]  # pyright: ignore

    # REJECTED: restore from the pre-bump snapshot.
    _restore_cache([cache], snap)

    assert cache._pending_bumps == [0], (  # pyright: ignore
        "restore must clear the per-stream staged bumps, not leave the "
        "rejected draft's bump stale on the live object -- a stale bump "
        "here is the exact corruption mechanism that crashed the single-"
        "stream PoolingCache path at ~500K context"
    )
    assert cache._pool_lengths == [7]  # pyright: ignore

