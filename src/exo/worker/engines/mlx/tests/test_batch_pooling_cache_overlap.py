"""CI-resident regression guard for the ``BatchPoolingCache`` overlap-carry bug.

Mirrors ``mlx-lm/tests/test_batch_pooling_cache_overlap.py`` from the
adurham/mlx-lm fork (commit ``37260bb``) into exo's OWN test tree.

Why the mirror exists
---------------------
The upstream copy lives inside the ``mlx-lm`` submodule. ``pipeline.yml``
scopes pytest to ``src`` and never checks out submodules, so the fork copy
has never executed in CI even once. Rather than pull mlx-lm's whole suite in
(slow, noisy, and mostly upstream behaviour we do not own), the narrow
overlap-carry guard is mirrored here.

What it actually guards
-----------------------
It imports ``BatchPoolingCache`` from the **installed** ``mlx_lm`` — the one
resolved by ``uv.lock`` / ``[tool.uv.sources]`` — not from the submodule
source tree. That makes it strictly stronger than the submodule copy: it
fails both when the fix is reverted in mlx-lm AND when exo's mlx-lm pin
drifts back to a revision that predates the fix. The pin silently trailing
the fix by nine commits is exactly the failure this file is meant to catch.

The defect (fixed in mlx-lm ``37260bb``): the four overlap-carry structures
(``_overlap_carry_valid``, ``_overlap_kv_carry``, ``_overlap_gate_carry``,
``_overlap_windows_this_call``) were sized once at construction and never
resized by the mid-decode structural ops. When a stream joined (``extend``)
or left (``filter``), the decode batch width changed but the carry lists did
not, so ``fetch_overlap_carry``'s reshape raised
``ValueError: [reshape] Cannot reshape array of size 1 into shape (2,1,1,1)``
and killed the runner on any mid-decode batch-width change with a persisted
overlap carry.
"""

from collections.abc import Callable
from typing import cast

import mlx.core as mx
from mlx_lm.models.cache import BatchPoolingCache

# ---------------------------------------------------------------------------
# Typed accessor shims.
#
# ``mlx_lm`` ships no type information, so every attribute touch on a
# BatchPoolingCache would otherwise trip basedpyright strict
# (reportUnknownMemberType / reportAny). Narrow each one exactly once here and
# route the tests through these, per the repo convention, instead of
# sprinkling per-call-site ignores.
# ---------------------------------------------------------------------------

RATIO = 4
HALF_DIM = 8
DTYPE = mx.float16


def _make_cache(width: int) -> object:
    """Construct a BatchPoolingCache of the given decode batch width."""
    constructor = cast(Callable[[int, list[int]], object], BatchPoolingCache)
    return constructor(RATIO, [0] * width)


def _carry_valid(cache: object) -> list[bool]:
    return cast(list[bool], getattr(cache, "_overlap_carry_valid"))  # noqa: B009


def _windows_this_call(cache: object) -> list[int]:
    return cast(list[int], getattr(cache, "_overlap_windows_this_call"))  # noqa: B009


def _set_windows_this_call(cache: object, windows: list[int]) -> None:
    cache._overlap_windows_this_call = windows  # pyright: ignore[reportAttributeAccessIssue]


def _kv_carry(cache: object) -> mx.array | None:
    return cast("mx.array | None", getattr(cache, "_overlap_kv_carry"))  # noqa: B009


def _remainder_width(cache: object) -> int:
    return len(cast(list[object], getattr(cache, "remainder")))  # noqa: B009


def _store_overlap_carry(cache: object, last_kv: mx.array, last_gate: mx.array) -> None:
    store = cast(
        Callable[[mx.array, mx.array], None],
        getattr(cache, "store_overlap_carry"),  # noqa: B009
    )
    store(last_kv, last_gate)


def _fetch_overlap_carry(cache: object, batch_size: int) -> tuple[mx.array, mx.array]:
    fetch = cast(
        Callable[[int, int, int, mx.Dtype], tuple[mx.array, mx.array]],
        getattr(cache, "fetch_overlap_carry"),  # noqa: B009
    )
    return fetch(batch_size, RATIO, HALF_DIM, DTYPE)


def _extend(cache: object, other: object) -> None:
    extend = cast(Callable[[object], None], getattr(cache, "extend"))  # noqa: B009
    extend(other)


def _filter(cache: object, surviving_indices: list[int]) -> None:
    filter_fn = cast(Callable[[list[int]], None], getattr(cache, "filter"))  # noqa: B009
    filter_fn(surviving_indices)


def _all_equal(tensor: mx.array, value: float) -> bool:
    """``tensor == value`` narrows to ``array | bool`` under basedpyright
    strict, which ``mx.all`` will not accept. Cast the elementwise comparison
    back to an array once, here, instead of at every call site."""
    return bool(mx.all(cast(mx.array, tensor == value)))


def _store_stream0_carry(cache: object, value: float) -> mx.array:
    """Force a persisted per-stream overlap carry.

    Stream 0 stores ``value`` in every channel; stream 1 produces no window on
    this call, so only ``valid[0]`` flips to True.
    """
    _set_windows_this_call(cache, [2, 0])
    width = _remainder_width(cache)
    last_kv = mx.full((width, 1, RATIO, HALF_DIM), value, dtype=DTYPE)
    last_gate = mx.full((width, 1, RATIO, HALF_DIM), -20.0, dtype=DTYPE)
    _store_overlap_carry(cache, last_kv, last_gate)
    return last_kv


def test_extend_widens_carry_and_preserves_surviving_stream() -> None:
    """A stream joining mid-decode must widen, not invalidate, the carry."""
    cache = _make_cache(2)
    stored = _store_stream0_carry(cache, 2.0)
    # 2 real streams in flight + 1 new stream joins mid-decode.
    _extend(cache, _make_cache(1))

    assert len(_carry_valid(cache)) == 3, (
        "extend must grow _overlap_carry_valid to the new batch width"
    )
    assert len(_windows_this_call(cache)) == 3, (
        "extend must grow _overlap_windows_this_call to the new batch width"
    )
    # Surviving stream 0's carry preserved in row 0.
    assert _carry_valid(cache) == [True, False, False]
    kv_carry_tensor = _kv_carry(cache)
    assert kv_carry_tensor is not None
    assert kv_carry_tensor.shape[0] == 3

    kv_carry, _gate_carry = _fetch_overlap_carry(cache, 3)
    # fetch_overlap_carry at the widened width must NOT raise, and row 0 must
    # still be the persisted carry (not zeroed).
    assert bool(mx.array_equal(kv_carry[0], stored[0].astype(DTYPE))), (
        "surviving stream 0's overlap carry was corrupted across extend"
    )
    # Stream 1 (in flight, no carry) and stream 2 (newly joined) get the
    # sequence-start pad.
    assert _all_equal(kv_carry[1], 0.0)
    assert _all_equal(kv_carry[2], 0.0)


def test_filter_narrows_carry_and_preserves_surviving_stream() -> None:
    """A stream leaving mid-decode must reindex, not stale-size, the carry."""
    cache = _make_cache(2)
    stored = _store_stream0_carry(cache, 3.0)
    _extend(cache, _make_cache(1))  # now width 3
    # Streams 1 and 2 leave; keep ONLY stream 0 -> resulting batch width (1)
    # differs from the original construction width (2), which is what trips
    # the stale-list reshape on the unfixed code.
    _filter(cache, [0])

    assert len(_carry_valid(cache)) == 1, (
        "filter must reindex _overlap_carry_valid to the surviving batch"
    )
    assert len(_windows_this_call(cache)) == 1, (
        "filter must reindex _overlap_windows_this_call to the surviving batch"
    )
    assert _carry_valid(cache) == [True]
    kv_carry_tensor = _kv_carry(cache)
    assert kv_carry_tensor is not None
    assert kv_carry_tensor.shape[0] == 1

    kv_carry, _gate_carry = _fetch_overlap_carry(cache, 1)
    assert bool(mx.array_equal(kv_carry[0], stored[0].astype(DTYPE))), (
        "surviving stream 0's overlap carry was corrupted across filter"
    )


def test_filter_respects_surviving_index_order() -> None:
    """filter keeps the caller's index order: row i corresponds to stream[i]."""
    cache = _make_cache(3)
    # Stream 1 stores a carry; streams 0 and 2 produce none.
    _set_windows_this_call(cache, [0, 2, 0])
    width = _remainder_width(cache)
    last_kv = mx.full((width, 1, RATIO, HALF_DIM), 5.0, dtype=DTYPE)
    last_gate = mx.full((width, 1, RATIO, HALF_DIM), -20.0, dtype=DTYPE)
    _store_overlap_carry(cache, last_kv, last_gate)  # valid[1] -> True

    # Reorder: surviving stream set is {1 (carry), 2 (none)}, ordered as
    # [2, 1]. Row 0 must be stream 2 (no carry), row 1 stream 1 (carry).
    _filter(cache, [2, 1])
    assert _carry_valid(cache) == [False, True]

    kv_carry, _gate_carry = _fetch_overlap_carry(cache, 2)
    assert _all_equal(kv_carry[0], 0.0)
    assert _all_equal(kv_carry[1], 5.0), (
        "carry must follow the stream, not its old row position"
    )


def test_fetch_overlap_carry_no_carry_untouched_by_extend() -> None:
    """extend on a cache that never stored a carry keeps the tensors None.

    fetch_overlap_carry at the widened width then returns the placeholder.
    """
    cache = _make_cache(1)
    assert _kv_carry(cache) is None
    _extend(cache, _make_cache(1))
    assert _kv_carry(cache) is None
    assert len(_carry_valid(cache)) == 2

    kv_carry, gate_carry = _fetch_overlap_carry(cache, 2)
    assert _all_equal(kv_carry, 0.0)
    assert _all_equal(gate_carry, -float("inf"))
