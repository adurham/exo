"""Wire round-trip for DSv4-Flash composite caches.

DeepSeek-V4-Flash builds one ``CacheList(RotatingKVCache, PoolingCache,
PoolingCache)`` per layer. These tests exercise the send -> receive path for
that shape so a prefill process can hand its cache to a separate decode
process, and assert tensor EQUALITY plus ``meta_state`` equality (not merely
"no exception raised").
"""

import io
from collections.abc import Sequence
from typing import cast

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.cache import (
    ArraysCache,
    CacheList,
    KVCache,
    PoolingCache,
    RotatingKVCache,
)

from exo.worker.disaggregated.protocol import (
    ArraysState,
    Done,
    Header,
    KVChunk,
    TensorBlob,
    write_done,
    write_header,
)
from exo.worker.engines.mlx.disaggregated.adapter import (
    UnsupportedCacheStateError,
    decode_composite_cache,
    encode_composite_cache,
    send_mlx_kv_cache,
)
from exo.worker.engines.mlx.disaggregated.client import (
    PrefillResult,
    ingest_into_mlx_cache,
)
from exo.worker.engines.mlx.disaggregated.tests.test_mlx_adapter import (
    _decode_payload,  # pyright: ignore[reportPrivateUsage]
    _equal,  # pyright: ignore[reportPrivateUsage]
)


def _members(cache: CacheList) -> tuple[object, ...]:
    """``CacheList.caches`` is untyped in mlx_lm; narrow it once, here."""
    return tuple(cast(Sequence[object], getattr(cache, "caches")))  # noqa: B009


def _member(cache: CacheList, index: int) -> object:
    return _members(cache)[index]


def _ratio(cache: object) -> int:
    return int(cast(int, getattr(cache, "ratio")))  # noqa: B009


def _remainder(cache: object) -> int:
    return int(cast(int, getattr(cache, "remainder")))  # noqa: B009


def _offset(cache: object) -> int:
    return int(cast(int, getattr(cache, "offset")))  # noqa: B009


def _state_of(cache: CacheList | PoolingCache) -> object:
    return cast(object, cache.state)


def _meta_of(cache: CacheList | PoolingCache) -> object:
    return cast(object, cache.meta_state)


_RATIO_A = 4
_RATIO_B = 8
_N_HEADS = 2
_HEAD_DIM = 4
_SEQ_LEN = 6
_POOL_DIM = 5


def _fixed(shape: tuple[int, ...], seed: int, dtype: mx.Dtype = mx.bfloat16) -> mx.array:
    mx.random.seed(seed)
    return (mx.random.uniform(shape=shape) * 10).astype(dtype)


def _make_pooling_cache(ratio: int, pooled_len: int, remainder: int, seed: int) -> PoolingCache:
    pooled = _fixed((1, pooled_len, _POOL_DIM), seed) if pooled_len > 0 else None
    buffered_kv = (
        _fixed((1, remainder, _POOL_DIM), seed + 1) if remainder > 0 else None
    )
    buffered_gate = (
        _fixed((1, remainder, _POOL_DIM), seed + 2) if remainder > 0 else None
    )
    cache = PoolingCache(ratio)
    cache.state = (buffered_kv, buffered_gate, pooled)
    return cache


def _make_rotating_cache() -> RotatingKVCache:
    cache = RotatingKVCache(max_size=16, keep=0)
    cache.keys = _fixed((1, _N_HEADS, _SEQ_LEN, _HEAD_DIM), 11)
    cache.values = _fixed((1, _N_HEADS, _SEQ_LEN, _HEAD_DIM), 12)
    cache.offset = _SEQ_LEN
    cache._idx = _SEQ_LEN
    return cache


def _make_dsv4_layer_cache() -> CacheList:
    return CacheList(
        _make_rotating_cache(),
        _make_pooling_cache(_RATIO_A, pooled_len=3, remainder=2, seed=21),
        _make_pooling_cache(_RATIO_B, pooled_len=2, remainder=5, seed=31),
    )


def _make_empty_dsv4_layer_cache() -> CacheList:
    return CacheList(
        RotatingKVCache(max_size=16, keep=0),
        PoolingCache(_RATIO_A),
        PoolingCache(_RATIO_B),
    )


def _assert_state_equal(expected: object, actual: object) -> None:
    if isinstance(expected, mx.array):
        assert isinstance(actual, mx.array)
        assert _equal(expected, actual)
        return
    if expected is None:
        assert actual is None
        return
    assert isinstance(expected, tuple | list) and isinstance(actual, tuple | list)
    expected_items = cast(Sequence[object], expected)
    actual_items = cast(Sequence[object], actual)
    assert len(expected_items) == len(actual_items)
    for e, a in zip(expected_items, actual_items, strict=True):
        _assert_state_equal(e, a)


def _normalize_meta(meta: object) -> object:
    if isinstance(meta, tuple | list):
        return tuple(_normalize_meta(m) for m in cast(Sequence[object], meta))
    return meta


def test_composite_blob_roundtrip_preserves_tensors_and_meta_state() -> None:
    source = _make_dsv4_layer_cache()
    blobs = encode_composite_cache(source)

    destination = _make_empty_dsv4_layer_cache()
    decode_composite_cache(destination, blobs)

    _assert_state_equal(_state_of(source), _state_of(destination))
    assert _normalize_meta(_meta_of(source)) == _normalize_meta(_meta_of(destination))


def test_composite_member_order_and_identity_preserved() -> None:
    source = _make_dsv4_layer_cache()
    destination = _make_empty_dsv4_layer_cache()
    decode_composite_cache(destination, encode_composite_cache(source))

    assert [type(c).__name__ for c in _members(destination)] == [
        "RotatingKVCache",
        "PoolingCache",
        "PoolingCache",
    ]
    # Order matters: the two PoolingCaches have DIFFERENT ratios, so a swap
    # would be caught here even though both are the same class.
    assert _ratio(_member(destination, 1)) == _RATIO_A
    assert _ratio(_member(destination, 2)) == _RATIO_B
    assert _offset(_member(destination, 0)) == _SEQ_LEN


def test_meta_state_drop_would_be_detected() -> None:
    """Negative control: a receiver that ignores meta_state fails this test."""
    source = _make_dsv4_layer_cache()
    blobs = encode_composite_cache(source)

    # A destination built with the WRONG ratios / rotation bookkeeping. If the
    # codec dropped meta_state, these stale values would survive the restore.
    destination = CacheList(
        RotatingKVCache(max_size=999, keep=7),
        PoolingCache(_RATIO_B),  # deliberately swapped
        PoolingCache(_RATIO_A),  # deliberately swapped
    )
    decode_composite_cache(destination, blobs)

    assert _ratio(_member(destination, 1)) == _RATIO_A
    assert _ratio(_member(destination, 2)) == _RATIO_B
    rotating = _member(destination, 0)
    assert isinstance(rotating, RotatingKVCache)
    assert (rotating.max_size, rotating.keep, rotating.offset) == (16, 0, _SEQ_LEN)


def test_pooling_cache_alone_roundtrips() -> None:
    source = _make_pooling_cache(_RATIO_A, pooled_len=4, remainder=3, seed=41)
    destination = PoolingCache(1)
    decode_composite_cache(destination, encode_composite_cache(source))

    _assert_state_equal(_state_of(source), _state_of(destination))
    assert _ratio(destination) == _RATIO_A
    assert _remainder(destination) == _remainder(source)
    assert _offset(destination) == _offset(source)


def test_full_send_receive_roundtrip_over_stream() -> None:
    sources = [_make_dsv4_layer_cache(), _make_dsv4_layer_cache()]

    buf = io.BytesIO()
    write_header(
        buf, Header(request_id="r", model_id="dsv4", num_layers=2, dtype="bfloat16")
    )
    tokens_sent = send_mlx_kv_cache(buf, sources, dtype="bfloat16")
    write_done(buf, tokens_sent)

    result = _decode_payload(buf.getvalue())
    assert tokens_sent == _SEQ_LEN
    assert result.total_tokens == _SEQ_LEN
    assert set(result.arrays) == {0, 1}
    assert not result.kv_chunks

    destinations = [_make_empty_dsv4_layer_cache(), _make_empty_dsv4_layer_cache()]
    final_offset = ingest_into_mlx_cache(result, list(destinations))

    assert final_offset == _SEQ_LEN
    for source, destination in zip(sources, destinations, strict=True):
        _assert_state_equal(_state_of(source), _state_of(destination))
        assert _normalize_meta(_meta_of(source)) == _normalize_meta(
            _meta_of(destination)
        )


def test_existing_kv_and_arrays_wire_format_unchanged() -> None:
    """Regression guard: non-composite caches must keep their old framing."""
    kv = KVCache()
    kv.keys = _fixed((1, _N_HEADS, _SEQ_LEN, _HEAD_DIM), 51)
    kv.values = _fixed((1, _N_HEADS, _SEQ_LEN, _HEAD_DIM), 52)
    kv.offset = _SEQ_LEN

    arrays = ArraysCache(size=2)
    arrays.state = [_fixed((3,), 61), _fixed((2, 4), 62)]

    buf = io.BytesIO()
    write_header(
        buf, Header(request_id="r", model_id="m", num_layers=2, dtype="bfloat16")
    )
    tokens_sent = send_mlx_kv_cache(buf, [kv, arrays], dtype="bfloat16")
    write_done(buf, tokens_sent)
    result = _decode_payload(buf.getvalue())

    # Layer 0 is a KVChunk (not an ArraysState); layer 1 has NO descriptor blob.
    assert list(result.kv_chunks) == [0]
    assert isinstance(result.kv_chunks[0][0], KVChunk)
    assert list(result.arrays) == [1]
    assert len(result.arrays[1]) == 2

    dst_kv = KVCache()
    dst_arrays = ArraysCache(size=2)
    assert ingest_into_mlx_cache(result, [dst_kv, dst_arrays]) == _SEQ_LEN
    restored_keys = dst_kv.keys
    original_keys = kv.keys
    assert restored_keys is not None and original_keys is not None
    assert _equal(restored_keys, original_keys)
    expected_arrays = cast(Sequence[mx.array], arrays.state)
    restored_arrays = cast(Sequence[mx.array], dst_arrays.state)
    for expected, actual in zip(expected_arrays, restored_arrays, strict=True):
        assert _equal(expected, actual)


def test_wrong_destination_type_raises_rather_than_guessing() -> None:
    blobs = encode_composite_cache(_make_dsv4_layer_cache())
    with pytest.raises(UnsupportedCacheStateError):
        decode_composite_cache(PoolingCache(_RATIO_A), blobs)


def test_arity_mismatch_raises() -> None:
    blobs = encode_composite_cache(_make_dsv4_layer_cache())
    with pytest.raises(UnsupportedCacheStateError):
        decode_composite_cache(
            CacheList(RotatingKVCache(max_size=16, keep=0), PoolingCache(_RATIO_A)),
            blobs,
        )


def test_non_descriptor_payload_raises() -> None:
    plain = [TensorBlob(dtype="float32", shape=(2,), data=np.zeros(2, np.float32).tobytes())]
    with pytest.raises(UnsupportedCacheStateError):
        decode_composite_cache(_make_empty_dsv4_layer_cache(), plain)


def test_composite_send_rejects_incremental_start_pos() -> None:
    buf = io.BytesIO()
    with pytest.raises(UnsupportedCacheStateError):
        send_mlx_kv_cache(
            buf, [_make_dsv4_layer_cache()], dtype="bfloat16", start_pos=3
        )


def test_decoded_message_types_are_arrays_state() -> None:
    buf = io.BytesIO()
    write_header(
        buf, Header(request_id="r", model_id="dsv4", num_layers=1, dtype="bfloat16")
    )
    tokens_sent = send_mlx_kv_cache(buf, [_make_dsv4_layer_cache()], dtype="bfloat16")
    write_done(buf, tokens_sent)
    result = _decode_payload(buf.getvalue())
    assert isinstance(result, PrefillResult)
    assert all(isinstance(b, TensorBlob) for b in result.arrays[0])
    assert isinstance(ArraysState(layer_idx=0, arrays=[]), ArraysState)
    assert isinstance(Done(total_tokens=0), Done)
