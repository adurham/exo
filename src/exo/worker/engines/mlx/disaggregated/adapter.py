from collections.abc import Sequence
from typing import BinaryIO, Final, Literal, TypeAlias, cast

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import (
    ArraysCache,
    CacheList,
    KVCache,
    PoolingCache,
    QuantizedKVCache,
    RotatingKVCache,
)

# DSv4 layers no longer use a unified DeepseekV4Cache (removed in
# Blaizzy PR #1192's cache refactor). Each layer now uses a CacheList of
# (RotatingKVCache + 2× PoolingCache). The disaggregated wire protocol
# raises NotImplementedError for these complex cache types anyway, so
# the only behavioral change is the type-match arm.
from exo.worker.disaggregated.protocol import (
    DType,
    Header,
    KVChunk,
    TensorBlob,
    write_arrays_state,
    write_done,
    write_header,
    write_kv_chunk,
)
from exo.worker.engines.mlx.types import KVCacheType
from exo.worker.runner.bootstrap import logger

_STR_TO_MX: dict[DType, mx.Dtype] = {
    "bfloat16": mx.bfloat16,
    "float16": mx.float16,
    "float32": mx.float32,
}

_MX_TO_STR: dict[mx.Dtype, DType] = {v: k for k, v in _STR_TO_MX.items()}


def mx_dtype_to_str(dtype: mx.Dtype) -> DType:
    if dtype not in _MX_TO_STR:
        raise ValueError(f"Unsupported mlx dtype on wire: {dtype}")
    return _MX_TO_STR[dtype]


def wire_dtype_from_cache(caches: KVCacheType) -> DType:
    for c in caches:
        keys: mx.array | None = getattr(c, "keys", None)
        if keys is None:
            continue
        if keys.dtype in _MX_TO_STR:
            return _MX_TO_STR[keys.dtype]
        break
    return "bfloat16"


def str_to_mx_dtype(dtype: DType) -> mx.Dtype:
    if dtype not in _STR_TO_MX:
        raise ValueError(f"Unsupported wire dtype: {dtype!r}")
    return _STR_TO_MX[dtype]


def array_to_bytes(t: mx.array) -> bytes:
    # bf16 has no native numpy dtype; bitcast through uint16.
    if t.dtype == mx.bfloat16:
        return np.asarray(t.view(mx.uint16)).tobytes()
    if t.dtype in (mx.float16, mx.float32):
        return np.asarray(t).tobytes()
    raise ValueError(f"Unsupported mlx dtype for wire: {t.dtype}")


def bytes_to_array(data: bytes, shape: tuple[int, ...], dtype: DType) -> mx.array:
    match dtype:
        case "bfloat16":
            arr = np.frombuffer(data, dtype=np.uint16).reshape(shape).copy()
            return mx.array(arr).view(mx.bfloat16)
        case "float16":
            arr = np.frombuffer(data, dtype=np.float16).reshape(shape).copy()
            return mx.array(arr)
        case "float32":
            arr = np.frombuffer(data, dtype=np.float32).reshape(shape).copy()
            return mx.array(arr)


def bhsd_to_nhd(t: mx.array) -> mx.array:
    if t.ndim != 4 or int(t.shape[0]) != 1:
        raise ValueError(f"Expected BHSD with B=1, got shape={tuple(t.shape)}")
    return mx.transpose(t[0], (1, 0, 2))


def nhd_to_bhsd(t: mx.array) -> mx.array:
    if t.ndim != 3:
        raise ValueError(f"Expected NHD (3D), got shape={tuple(t.shape)}")
    return mx.expand_dims(mx.transpose(t, (1, 0, 2)), 0)


# ---------------------------------------------------------------------------
# Composite-cache (CacheList / PoolingCache) wire codec.
#
# DSv4-Flash builds one CacheList(RotatingKVCache, PoolingCache, PoolingCache)
# per layer. Those caches are not sliceable into per-token KVChunks, so they
# ride the existing ``ArraysState`` message instead: the first TensorBlob of
# the layer's array list is a DESCRIPTOR encoding the cache tree structure
# (types, nesting, state shape, and the full non-tensor ``meta_state``), and
# the remaining blobs are the tensor leaves in depth-first order.
#
# The plain ArraysCache path predates this and emits NO descriptor, so it stays
# byte-for-byte compatible on the wire. The receive side disambiguates on the
# LOCAL destination cache type, and the descriptor's magic word is verified so
# a mismatched pairing fails loudly instead of silently mis-restoring.
# ---------------------------------------------------------------------------

_COMPOSITE_MAGIC: Final[int] = 0x584C31  # "XL1"; < 2**24 so float32-exact
_COMPOSITE_VERSION: Final[int] = 1

CompositeCacheType: TypeAlias = CacheList | PoolingCache | RotatingKVCache | ArraysCache

_CacheTypeCode: TypeAlias = Literal[1, 2, 3, 4]

_CACHE_TYPE_CODES: Final[dict[str, _CacheTypeCode]] = {
    "CacheList": 1,
    "PoolingCache": 2,
    "RotatingKVCache": 3,
    "ArraysCache": 4,
}

# State-tree node tags.
_STATE_NONE: Final[int] = 0
_STATE_ARRAY: Final[int] = 1
_STATE_SEQUENCE: Final[int] = 2

# Meta-state tree node tags. ``meta_state`` is plain Python data: mlx caches
# use nested tuples/lists of ints, of decimal-int strings, or the empty string.
_META_EMPTY_STRING: Final[int] = 0
_META_INTEGER: Final[int] = 1
_META_INTEGER_STRING: Final[int] = 2
_META_SEQUENCE: Final[int] = 3


def _cache_list_members(cache: CacheList) -> tuple[object, ...]:
    """``CacheList.caches`` is untyped in mlx_lm; narrow it once, here."""
    return tuple(cast(Sequence[object], getattr(cache, "caches")))  # noqa: B009


def _read_state(cache: CompositeCacheType) -> object:
    return cast(object, cache.state)


def _read_meta_state(cache: CompositeCacheType) -> object:
    return cast(object, cache.meta_state)


class UnsupportedCacheStateError(RuntimeError):
    """A cache's state/meta_state contains something the wire cannot carry.

    Raised instead of guessing: a cache that arrives with correct tensors but
    wrong bookkeeping silently produces wrong tokens.
    """


def _encode_meta_state(meta: object, out: list[int]) -> None:
    match meta:
        case str() as text:
            if text == "":
                out.append(_META_EMPTY_STRING)
                return
            try:
                value = int(text)
            except ValueError as exc:
                raise UnsupportedCacheStateError(
                    f"meta_state string is not an integer: {text!r}"
                ) from exc
            out.append(_META_INTEGER_STRING)
            out.append(value)
        case bool():
            raise UnsupportedCacheStateError("meta_state booleans are not supported")
        case int() as value:
            out.append(_META_INTEGER)
            out.append(value)
        case tuple() | list():
            meta_items = cast(Sequence[object], meta)
            out.append(_META_SEQUENCE)
            out.append(len(meta_items))
            for item in meta_items:
                _encode_meta_state(item, out)
        case _:
            raise UnsupportedCacheStateError(
                f"Unsupported meta_state element of type {type(meta).__name__}"
            )


def _decode_meta_state(words: list[int], pos: int) -> tuple[object, int]:
    tag = words[pos]
    pos += 1
    match tag:
        case 0:  # _META_EMPTY_STRING
            return "", pos
        case 1:  # _META_INTEGER
            return words[pos], pos + 1
        case 2:  # _META_INTEGER_STRING
            return str(words[pos]), pos + 1
        case 3:  # _META_SEQUENCE
            count = words[pos]
            pos += 1
            items: list[object] = []
            for _ in range(count):
                item, pos = _decode_meta_state(words, pos)
                items.append(item)
            return tuple(items), pos
        case _:
            raise UnsupportedCacheStateError(f"Bad meta_state tag {tag}")


def _encode_state_tree(
    state: object, words: list[int], blobs: list[TensorBlob]
) -> None:
    match state:
        case None:
            words.append(_STATE_NONE)
        case mx.array() as array:
            with mx.stream(mx.Device(mx.cpu)):
                array_cpu = mx.array(array)
                mx.eval(array_cpu)
            words.append(_STATE_ARRAY)
            blobs.append(
                TensorBlob(
                    dtype=mx_dtype_to_str(array_cpu.dtype),
                    shape=tuple(int(d) for d in array_cpu.shape),
                    data=array_to_bytes(array_cpu),
                )
            )
        case tuple() | list():
            state_items = cast(Sequence[object], state)
            words.append(_STATE_SEQUENCE)
            words.append(len(state_items))
            for item in state_items:
                _encode_state_tree(item, words, blobs)
        case _:
            raise UnsupportedCacheStateError(
                f"Unsupported cache state element of type {type(state).__name__}"
            )


def _decode_state_tree(
    words: list[int], pos: int, blobs: list[TensorBlob], blob_index: int
) -> tuple[object, int, int]:
    tag = words[pos]
    pos += 1
    match tag:
        case 0:  # _STATE_NONE
            return None, pos, blob_index
        case 1:  # _STATE_ARRAY
            if blob_index >= len(blobs):
                raise UnsupportedCacheStateError("Descriptor references missing blob")
            return blob_to_mlx(blobs[blob_index]), pos, blob_index + 1
        case 2:  # _STATE_SEQUENCE
            count = words[pos]
            pos += 1
            items: list[object] = []
            for _ in range(count):
                item, pos, blob_index = _decode_state_tree(
                    words, pos, blobs, blob_index
                )
                items.append(item)
            return list(items), pos, blob_index
        case _:
            raise UnsupportedCacheStateError(f"Bad state tag {tag}")


def _encode_cache_tree(
    cache: CompositeCacheType, words: list[int], blobs: list[TensorBlob]
) -> None:
    type_name = type(cache).__name__
    if type_name not in _CACHE_TYPE_CODES:
        raise UnsupportedCacheStateError(f"Cannot serialize cache type {type_name}")
    words.append(_CACHE_TYPE_CODES[type_name])
    if isinstance(cache, CacheList):
        members = _cache_list_members(cache)
        words.append(len(members))
        for member in members:
            if not isinstance(
                member, CacheList | PoolingCache | RotatingKVCache | ArraysCache
            ):
                raise UnsupportedCacheStateError(
                    f"CacheList member of type {type(member).__name__} "
                    "is not supported on the wire"
                )
            _encode_cache_tree(member, words, blobs)
        return
    _encode_state_tree(_read_state(cache), words, blobs)
    _encode_meta_state(_read_meta_state(cache), words)


def _decode_into_cache_tree(
    cache: CompositeCacheType,
    words: list[int],
    pos: int,
    blobs: list[TensorBlob],
    blob_index: int,
) -> tuple[int, int]:
    type_name = type(cache).__name__
    expected = _CACHE_TYPE_CODES.get(type_name)
    received = words[pos]
    pos += 1
    if expected is None or received != expected:
        raise UnsupportedCacheStateError(
            f"Wire cache type code {received} does not match "
            f"destination cache {type_name}"
        )
    if isinstance(cache, CacheList):
        member_count = words[pos]
        pos += 1
        members = _cache_list_members(cache)
        if member_count != len(members):
            raise UnsupportedCacheStateError(
                f"CacheList arity mismatch: wire has {member_count} members, "
                f"destination has {len(members)}"
            )
        for member in members:
            if not isinstance(
                member, CacheList | PoolingCache | RotatingKVCache | ArraysCache
            ):
                raise UnsupportedCacheStateError(
                    f"CacheList member of type {type(member).__name__} "
                    "is not supported on the wire"
                )
            pos, blob_index = _decode_into_cache_tree(
                member, words, pos, blobs, blob_index
            )
        return pos, blob_index
    state, pos, blob_index = _decode_state_tree(words, pos, blobs, blob_index)
    meta, pos = _decode_meta_state(words, pos)
    # meta_state FIRST: PoolingCache.state's setter re-buffers the remainder
    # through accumulate_windows(), which needs the restored ``ratio``.
    cache.meta_state = meta
    cache.state = state
    return pos, blob_index


def encode_composite_cache(cache: CompositeCacheType) -> list[TensorBlob]:
    """Serialize a composite cache into a descriptor blob + tensor blobs."""
    words: list[int] = [_COMPOSITE_MAGIC, _COMPOSITE_VERSION]
    blobs: list[TensorBlob] = []
    _encode_cache_tree(cache, words, blobs)
    descriptor = np.asarray(words, dtype=np.float32)
    if not np.array_equal(
        descriptor.astype(np.int64), np.asarray(words, dtype=np.int64)
    ):
        raise UnsupportedCacheStateError(
            "Cache descriptor contains an integer too large to round-trip"
        )
    return [
        TensorBlob(
            dtype="float32",
            shape=(len(words),),
            data=descriptor.tobytes(),
        ),
        *blobs,
    ]


def decode_composite_cache(cache: CompositeCacheType, blobs: list[TensorBlob]) -> None:
    """Restore a composite cache in place from ``encode_composite_cache`` output."""
    if not blobs:
        raise UnsupportedCacheStateError("Composite cache payload has no descriptor")
    descriptor_blob = blobs[0]
    if descriptor_blob.dtype != "float32":
        raise UnsupportedCacheStateError(
            f"Composite descriptor must be float32, got {descriptor_blob.dtype}"
        )
    descriptor = np.frombuffer(descriptor_blob.data, dtype=np.float32)
    words = [int(w) for w in cast(list[float], descriptor.tolist())]
    if len(words) < 2 or words[0] != _COMPOSITE_MAGIC:
        raise UnsupportedCacheStateError(
            "Payload is not a composite cache descriptor (bad magic)"
        )
    if words[1] != _COMPOSITE_VERSION:
        raise UnsupportedCacheStateError(
            f"Unsupported composite cache descriptor version {words[1]}"
        )
    pos, blob_index = _decode_into_cache_tree(cache, words, 2, blobs, 1)
    if pos != len(words) or blob_index != len(blobs):
        raise UnsupportedCacheStateError(
            f"Composite payload not fully consumed "
            f"(words {pos}/{len(words)}, blobs {blob_index}/{len(blobs)})"
        )


def composite_cache_offset(cache: CompositeCacheType) -> int:
    """Token offset a composite cache represents, from its attention member."""
    if isinstance(cache, RotatingKVCache):
        return int(cache.offset)
    if isinstance(cache, CacheList):
        for member in _cache_list_members(cache):
            if isinstance(member, CacheList | RotatingKVCache):
                nested = composite_cache_offset(member)
                if nested > 0:
                    return nested
    return 0


def send_mlx_kv_cache(
    stream: BinaryIO,
    caches: KVCacheType,
    *,
    dtype: DType,
    start_pos: int = 0,
    max_tokens: int | None = None,
) -> int:
    tokens_sent = 0
    for layer_idx, c in enumerate(caches):
        match c:
            case QuantizedKVCache():
                raise NotImplementedError
            case CacheList() | PoolingCache():
                # Composite/pooled caches (DSv4-Flash) cannot be sliced into
                # per-token KVChunks; ship the whole state tree instead.
                if start_pos != 0:
                    raise UnsupportedCacheStateError(
                        "Composite caches are sent whole; incremental "
                        f"start_pos={start_pos} cannot be honoured"
                    )
                write_arrays_state(stream, layer_idx, encode_composite_cache(c))
                offset = composite_cache_offset(c)
                if offset > 0:
                    if tokens_sent != 0 and offset != tokens_sent:
                        logger.critical(
                            f"Unexpected number of tokens sent {offset} != {tokens_sent}"
                        )
                    tokens_sent = offset
            case KVCache() | RotatingKVCache():
                keys = c.keys
                values = c.values
                if keys is None or values is None:
                    continue
                offset = int(c.offset)
                if max_tokens is not None:
                    offset = min(offset, max_tokens)
                if offset <= start_pos:
                    continue
                with mx.stream(mx.Device(mx.cpu)):
                    k = mx.array(keys[:, :, start_pos:offset, :])
                    v = mx.array(values[:, :, start_pos:offset, :])
                    k_nhd = bhsd_to_nhd(k)
                    v_nhd = bhsd_to_nhd(v)
                    mx.eval(k_nhd, v_nhd)
                num_tokens = int(k_nhd.shape[0])
                n_heads = int(k_nhd.shape[1])
                head_dim = int(k_nhd.shape[2])
                write_kv_chunk(
                    stream,
                    layer_idx=layer_idx,
                    num_tokens=num_tokens,
                    n_heads=n_heads,
                    head_dim=head_dim,
                    dtype=dtype,
                    keys=array_to_bytes(k_nhd),
                    values=array_to_bytes(v_nhd),
                )
                if tokens_sent != 0 and num_tokens != tokens_sent:
                    logger.critical(
                        f"Unexpected number of tokens sent {num_tokens} != {tokens_sent}"
                    )
                tokens_sent = num_tokens
            case ArraysCache():
                blobs: list[TensorBlob] = []
                for a in c.state:
                    if a is None:
                        continue
                    with mx.stream(mx.Device(mx.cpu)):
                        a_cpu = mx.array(a)
                        mx.eval(a_cpu)
                    blobs.append(
                        TensorBlob(
                            dtype=mx_dtype_to_str(a_cpu.dtype),
                            shape=tuple(int(d) for d in a_cpu.shape),
                            data=array_to_bytes(a_cpu),
                        )
                    )
                if blobs:
                    write_arrays_state(stream, layer_idx, blobs)
    return tokens_sent


def chunk_to_mlx_nhd(chunk: KVChunk) -> tuple[mx.array, mx.array]:
    shape = chunk.shape
    return (
        bytes_to_array(chunk.keys, shape, chunk.dtype),
        bytes_to_array(chunk.values, shape, chunk.dtype),
    )


def blob_to_mlx(blob: TensorBlob) -> mx.array:
    return bytes_to_array(blob.data, blob.shape, blob.dtype)


def inject_kv_chunk(
    cache: KVCache,
    keys_nhd: mx.array,
    values_nhd: mx.array,
    offset: int,
    *,
    start_pos: int = 0,
    existing_k: mx.array | None = None,
    existing_v: mx.array | None = None,
) -> None:
    k_bhsd = nhd_to_bhsd(keys_nhd)
    v_bhsd = nhd_to_bhsd(values_nhd)
    if start_pos > 0 and existing_k is not None and existing_v is not None:
        cache.keys = mx.concatenate([existing_k[:, :, :start_pos, :], k_bhsd], axis=2)
        cache.values = mx.concatenate([existing_v[:, :, :start_pos, :], v_bhsd], axis=2)
    else:
        cache.keys = k_bhsd
        cache.values = v_bhsd
    cache.offset = offset


def inject_rotating_kv_chunk(
    cache: RotatingKVCache,
    keys_nhd: mx.array,
    values_nhd: mx.array,
    offset: int,
) -> None:
    k_bhsd = nhd_to_bhsd(keys_nhd)
    v_bhsd = nhd_to_bhsd(values_nhd)
    cache.keys = k_bhsd
    cache.values = v_bhsd
    cache.offset = offset
    cache._idx = int(k_bhsd.shape[2])


def inject_arrays_cache(cache: ArraysCache, blobs: list[TensorBlob]) -> None:
    cache.state = [blob_to_mlx(b) for b in blobs]


def write_cache_to_wire(
    wfile: BinaryIO,
    cache: KVCacheType,
    *,
    request_id: str = "",
    model_id: str = "",
    start_pos: int = 0,
) -> int:
    dtype = wire_dtype_from_cache(cache)
    write_header(
        wfile,
        Header(
            request_id=request_id,
            model_id=model_id,
            num_layers=len(cache),
            dtype=dtype,
            start_pos=start_pos,
        ),
    )
    tokens_sent = send_mlx_kv_cache(wfile, cache, dtype=dtype, start_pos=start_pos)
    write_done(wfile, tokens_sent)
    wfile.flush()
    return tokens_sent
