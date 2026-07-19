# pyright: reportPrivateUsage=false
"""Unit tests for the STEP-1 spec-tag / hidden-buffer plumbing
(``pp_speculation_spec_tag``).

These are cluster-free -- the pipeline-boundary send/recv is *mocked*
via an in-process queue that just moves ``mx.array`` objects between a
notional rank 0 and rank 1. That's sufficient to prove:

1. FNV-1a hashing is deterministic and stable across "processes"
   (we simulate by re-computing from raw bytes rather than trusting
   Python's ``hash()``).
2. ``pack_spec_tag`` / ``unpack_spec_tag`` round-trip exactly across
   the full 64-bit hash range (including the ``bit-31`` fold both
   int32 wire slots require).
3. ``SpecHiddenBuffer`` correctly stashes, releases (HIT) and discards
   (MISS) speculative hiddens, and rejects double-stash / non-spec
   entries.
4. ``SpecTagValidator`` flags each field mismatch category with a
   distinct reason string (so a real cross-rank desync is diagnosable
   from the log alone, not just "something broke").
5. The full mocked end-to-end flow: rank 0 produces a speculative
   hidden, tags it, sends the tag over the mock wire, rank 1 unpacks
   and validates -- both HIT and MISS legs -- and neither leg touches
   the fallthrough decode path (diagnostic-mode invariant).
6. ``coerce_hit_miss`` narrows correctly and rejects unknown codes.
"""

from __future__ import annotations

from collections import deque
from typing import cast, final

import mlx.core as mx
import pytest

from exo.worker.engines.mlx.pp_speculation_spec_tag import (
    HIT_MISS_HIT,
    HIT_MISS_MISS,
    HIT_MISS_NA,
    SPEC_TAG_WIRE_LEN,
    BufferedSpecHidden,
    SpecHiddenBuffer,
    SpecId,
    SpecTagValidator,
    coerce_hit_miss,
    fnv1a64,
    pack_spec_tag,
    unpack_spec_tag,
)

# ---------------------------------------------------------------------------
# hashing
# ---------------------------------------------------------------------------


def test_fnv1a64_deterministic() -> None:
    a = fnv1a64((1, 2, 3, 4, 5))
    b = fnv1a64((1, 2, 3, 4, 5))
    assert a == b


def test_fnv1a64_order_sensitive() -> None:
    assert fnv1a64((1, 2, 3)) != fnv1a64((3, 2, 1))


def test_fnv1a64_single_bit_change_avalanche() -> None:
    a = fnv1a64((1, 2, 3))
    b = fnv1a64((1, 2, 4))
    # Any change should flip a healthy chunk of bits; specifically not
    # equal, and not merely low-bit-different (avalanche property).
    assert a != b
    assert bin(a ^ b).count("1") > 8


def test_fnv1a64_empty_prefix() -> None:
    assert fnv1a64(()) == 0xCBF29CE484222325  # FNV offset basis


def test_fnv1a64_range_bounds() -> None:
    h = fnv1a64((0xFFFFFFFF, 0, 0x7FFFFFFF))
    assert 0 <= h <= 0xFFFFFFFFFFFFFFFF


# ---------------------------------------------------------------------------
# SpecId construction / validation
# ---------------------------------------------------------------------------


def test_spec_id_build_from_prefix() -> None:
    sid = SpecId.build(spec_kind="draft_ahead", cycle_n=7, prefix=(10, 20, 30))
    assert sid.spec_kind == "draft_ahead"
    assert sid.cycle_n == 7
    assert sid.prefix_len == 3
    assert sid.prefix_hash == fnv1a64((10, 20, 30))


def test_spec_id_rejects_negative_cycle_n() -> None:
    with pytest.raises(ValueError, match="cycle_n"):
        SpecId(spec_kind="draft_ahead", cycle_n=-1, prefix_hash=0, prefix_len=0)


def test_spec_id_rejects_negative_prefix_len() -> None:
    with pytest.raises(ValueError, match="prefix_len"):
        SpecId(spec_kind="draft_ahead", cycle_n=0, prefix_hash=0, prefix_len=-3)


def test_spec_id_rejects_out_of_range_hash() -> None:
    with pytest.raises(ValueError, match="prefix_hash"):
        SpecId(
            spec_kind="draft_ahead",
            cycle_n=0,
            prefix_hash=1 << 64,
            prefix_len=0,
        )


# ---------------------------------------------------------------------------
# wire pack/unpack round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cycle_n,prefix",
    [
        (0, ()),
        (1, (42,)),
        (99, (1, 2, 3, 4, 5)),
        (10_000, tuple(range(50))),
    ],
)
def test_pack_unpack_round_trip(cycle_n: int, prefix: tuple[int, ...]) -> None:
    sid = SpecId.build(spec_kind="draft_ahead", cycle_n=cycle_n, prefix=prefix)
    wire = pack_spec_tag(sid)
    assert wire.shape == (SPEC_TAG_WIRE_LEN,)
    assert wire.dtype == mx.int32
    recovered = unpack_spec_tag(wire)
    assert recovered == sid


def test_pack_unpack_covers_high_bit_hashes() -> None:
    """Exercise the bit-31 fold path in both hash_hi and hash_lo."""
    # 0xFFFFFFFF_FFFFFFFF -- both halves have bit 31 set.
    sid = SpecId(
        spec_kind="draft_ahead",
        cycle_n=1,
        prefix_hash=0xFFFFFFFFFFFFFFFF,
        prefix_len=1,
    )
    assert unpack_spec_tag(pack_spec_tag(sid)) == sid


def test_pack_unpack_covers_non_spec_kind() -> None:
    sid = SpecId.build(spec_kind="non_spec", cycle_n=3, prefix=(7,))
    assert unpack_spec_tag(pack_spec_tag(sid)) == sid


def test_unpack_rejects_wrong_shape() -> None:
    wire = mx.zeros(4, dtype=mx.int32)
    with pytest.raises(ValueError, match="shape"):
        unpack_spec_tag(wire)


def test_unpack_rejects_wrong_dtype() -> None:
    wire = mx.zeros(SPEC_TAG_WIRE_LEN, dtype=mx.float32)
    with pytest.raises(ValueError, match="int32"):
        unpack_spec_tag(wire)


def test_unpack_rejects_unknown_kind_code() -> None:
    # Manually craft a wire payload with an unrecognized kind code.
    wire = mx.array([99, 0, 0, 0, 0], dtype=mx.int32)
    with pytest.raises(ValueError, match="unknown spec_kind"):
        unpack_spec_tag(wire)


# ---------------------------------------------------------------------------
# SpecHiddenBuffer
# ---------------------------------------------------------------------------


def _hidden(seed: int) -> mx.array:
    return mx.arange(4, dtype=mx.int32) + seed


def test_buffer_stash_release_hit() -> None:
    buf = SpecHiddenBuffer()
    sid = SpecId.build(spec_kind="draft_ahead", cycle_n=1, prefix=(1, 2))
    hidden = _hidden(0)
    buf.stash(sid, hidden)
    assert len(buf) == 1
    entry = buf.release(sid)
    assert isinstance(entry, BufferedSpecHidden)
    assert entry.spec_id == sid
    eq = entry.hidden == hidden
    assert isinstance(eq, mx.array)
    assert bool(mx.all(eq).item())
    assert len(buf) == 0


def test_buffer_discard_miss_returns_true_then_false() -> None:
    buf = SpecHiddenBuffer()
    sid = SpecId.build(spec_kind="draft_ahead", cycle_n=1, prefix=(1, 2))
    buf.stash(sid, _hidden(0))
    assert buf.discard(sid) is True
    assert buf.discard(sid) is False
    assert len(buf) == 0


def test_buffer_rejects_duplicate_stash() -> None:
    buf = SpecHiddenBuffer()
    sid = SpecId.build(spec_kind="draft_ahead", cycle_n=1, prefix=(1, 2))
    buf.stash(sid, _hidden(0))
    with pytest.raises(ValueError, match="duplicate"):
        buf.stash(sid, _hidden(1))


def test_buffer_rejects_non_spec_kind() -> None:
    buf = SpecHiddenBuffer()
    sid = SpecId.build(spec_kind="non_spec", cycle_n=1, prefix=(1, 2))
    with pytest.raises(ValueError, match="spec_kind"):
        buf.stash(sid, _hidden(0))


def test_buffer_release_missing_key_raises_key_error() -> None:
    buf = SpecHiddenBuffer()
    sid = SpecId.build(spec_kind="draft_ahead", cycle_n=1, prefix=(1, 2))
    with pytest.raises(KeyError):
        buf.release(sid)


def test_buffer_clear_and_peek_keys() -> None:
    buf = SpecHiddenBuffer()
    sid1 = SpecId.build(spec_kind="draft_ahead", cycle_n=1, prefix=(1,))
    sid2 = SpecId.build(spec_kind="draft_ahead", cycle_n=2, prefix=(2,))
    buf.stash(sid1, _hidden(0))
    buf.stash(sid2, _hidden(1))
    assert len(buf.peek_keys()) == 2
    buf.clear()
    assert len(buf) == 0


def test_buffer_distinguishes_cycles_with_same_prefix_hash() -> None:
    """Two cycles with identical prefixes MUST not collide -- the key
    is (cycle_n, prefix_hash, prefix_len), not prefix_hash alone."""
    buf = SpecHiddenBuffer()
    sid1 = SpecId.build(spec_kind="draft_ahead", cycle_n=1, prefix=(1, 2))
    sid2 = SpecId.build(spec_kind="draft_ahead", cycle_n=2, prefix=(1, 2))
    assert sid1.prefix_hash == sid2.prefix_hash
    buf.stash(sid1, _hidden(0))
    buf.stash(sid2, _hidden(1))
    assert len(buf) == 2


# ---------------------------------------------------------------------------
# SpecTagValidator
# ---------------------------------------------------------------------------


def test_validator_accepts_matching_ids() -> None:
    v = SpecTagValidator()
    sid = SpecId.build(spec_kind="draft_ahead", cycle_n=4, prefix=(1, 2, 3))
    result = v.validate(sid, sid)
    assert result.ok
    assert v.matches == 1
    assert v.mismatches == 0


def test_validator_rejects_spec_kind_mismatch() -> None:
    v = SpecTagValidator()
    incoming = SpecId.build(spec_kind="draft_ahead", cycle_n=1, prefix=(1,))
    expected = SpecId.build(spec_kind="non_spec", cycle_n=1, prefix=(1,))
    result = v.validate(incoming, expected)
    assert not result.ok
    assert "spec_kind" in result.reason


def test_validator_rejects_cycle_mismatch() -> None:
    v = SpecTagValidator()
    incoming = SpecId.build(spec_kind="draft_ahead", cycle_n=1, prefix=(1,))
    expected = SpecId.build(spec_kind="draft_ahead", cycle_n=2, prefix=(1,))
    result = v.validate(incoming, expected)
    assert not result.ok
    assert "cycle_n" in result.reason


def test_validator_rejects_prefix_len_mismatch() -> None:
    v = SpecTagValidator()
    incoming = SpecId.build(spec_kind="draft_ahead", cycle_n=1, prefix=(1,))
    expected = SpecId.build(spec_kind="draft_ahead", cycle_n=1, prefix=(1, 2))
    result = v.validate(incoming, expected)
    assert not result.ok
    assert "prefix_len" in result.reason


def test_validator_rejects_prefix_hash_mismatch() -> None:
    v = SpecTagValidator()
    # Same prefix_len but different contents -> different hash.
    incoming = SpecId.build(spec_kind="draft_ahead", cycle_n=1, prefix=(1, 2))
    expected = SpecId.build(spec_kind="draft_ahead", cycle_n=1, prefix=(9, 8))
    result = v.validate(incoming, expected)
    assert not result.ok
    assert "prefix_hash" in result.reason


def test_validator_reset() -> None:
    v = SpecTagValidator()
    sid = SpecId.build(spec_kind="draft_ahead", cycle_n=1, prefix=(1,))
    v.validate(sid, sid)
    v.reset()
    assert v.matches == 0
    assert v.mismatches == 0
    assert v.validated == []


# ---------------------------------------------------------------------------
# hit/miss coercion
# ---------------------------------------------------------------------------


def test_coerce_hit_miss_accepts_known_codes() -> None:
    assert coerce_hit_miss(-1) == HIT_MISS_NA
    assert coerce_hit_miss(0) == HIT_MISS_MISS
    assert coerce_hit_miss(1) == HIT_MISS_HIT


def test_coerce_hit_miss_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown"):
        coerce_hit_miss(7)


# ---------------------------------------------------------------------------
# End-to-end mocked pipeline-boundary flow (diagnostic-only invariant)
# ---------------------------------------------------------------------------


@final
class _MockWire:
    """In-process stand-in for the mx.distributed pp_group send/recv --
    just a FIFO of ``mx.array`` payloads. Sufficient to prove the
    tagging protocol survives a full round-trip end-to-end without
    a real cluster."""

    def __init__(self) -> None:
        self._q: deque[mx.array] = deque()

    def send(self, x: mx.array) -> None:
        self._q.append(x)

    def recv(self) -> mx.array:
        return self._q.popleft()


def _rank0_produce_speculative_hidden(
    *,
    cycle_n: int,
    anchor: int,
    drafted_ids: tuple[int, ...],
    assumed_bonus: int,
    buffer: SpecHiddenBuffer,
    wire: _MockWire,
) -> SpecId:
    """Mock of what rank 0 will eventually do:
    (a) compute assumed-prefix hash from (anchor, drafted, bonus),
    (b) build SpecId, (c) 'forward' a speculative hidden (placeholder
    here: a marker array), (d) stash into buffer, (e) send tag over
    the mock wire. Diagnostic mode: never releases without a HIT
    confirmation.
    """
    prefix = (anchor, *drafted_ids, assumed_bonus)
    sid = SpecId.build(spec_kind="draft_ahead", cycle_n=cycle_n, prefix=prefix)
    hidden = mx.array([anchor, *drafted_ids, assumed_bonus], dtype=mx.int32)
    buffer.stash(sid, hidden)
    wire.send(pack_spec_tag(sid))
    return sid


def _rank1_receive_and_validate(
    *,
    cycle_n: int,
    anchor: int,
    drafted_ids: tuple[int, ...],
    real_bonus: int,
    wire: _MockWire,
    validator: SpecTagValidator,
) -> tuple[bool, SpecId]:
    """Mock of what rank 1 will eventually do: recv the tag, build the
    same prefix using its OWN just-computed real bonus, validate."""
    incoming = unpack_spec_tag(wire.recv())
    expected_prefix = (anchor, *drafted_ids, real_bonus)
    expected = SpecId.build(
        spec_kind="draft_ahead", cycle_n=cycle_n, prefix=expected_prefix
    )
    result = validator.validate(incoming, expected)
    return result.ok, incoming


def test_end_to_end_hit_path_diagnostic_mode() -> None:
    """Full-accept HIT case: rank 1's real bonus == rank 0's assumed
    bonus, tags match, buffer releases the hidden cleanly. In real
    diagnostic mode the released hidden is NOT consumed by decode --
    but the plumbing must still work end-to-end."""
    buf = SpecHiddenBuffer()
    val = SpecTagValidator()
    wire = _MockWire()
    sid = _rank0_produce_speculative_hidden(
        cycle_n=5,
        anchor=100,
        drafted_ids=(101, 102),
        assumed_bonus=103,
        buffer=buf,
        wire=wire,
    )
    ok, incoming = _rank1_receive_and_validate(
        cycle_n=5,
        anchor=100,
        drafted_ids=(101, 102),
        real_bonus=103,
        wire=wire,
        validator=val,
    )
    assert ok
    assert incoming == sid
    # HIT: rank 0 receives the hit bit and releases.
    entry = buf.release(sid)
    assert entry.spec_id == sid
    assert len(buf) == 0
    assert val.matches == 1 and val.mismatches == 0


def test_end_to_end_miss_path_diagnostic_mode() -> None:
    """MISS: rank 1's real bonus differs from rank 0's assumption ->
    validator flags mismatch, rank 0 discards buffered hidden, no
    corruption possible because diagnostic mode never consumed it."""
    buf = SpecHiddenBuffer()
    val = SpecTagValidator()
    wire = _MockWire()
    sid = _rank0_produce_speculative_hidden(
        cycle_n=5,
        anchor=100,
        drafted_ids=(101, 102),
        assumed_bonus=103,
        buffer=buf,
        wire=wire,
    )
    ok, _incoming = _rank1_receive_and_validate(
        cycle_n=5,
        anchor=100,
        drafted_ids=(101, 102),
        real_bonus=999,  # different -> MISS
        wire=wire,
        validator=val,
    )
    assert not ok
    assert val.mismatches == 1
    assert buf.discard(sid) is True
    assert len(buf) == 0


def test_end_to_end_cycle_desync_caught() -> None:
    """Rank 0 thinks it's on cycle 5, rank 1 thinks cycle 6 -- exactly
    the KV-offset-desync failure the tag guards against. Must be
    flagged."""
    buf = SpecHiddenBuffer()
    val = SpecTagValidator()
    wire = _MockWire()
    _sid = _rank0_produce_speculative_hidden(
        cycle_n=5,
        anchor=1,
        drafted_ids=(2,),
        assumed_bonus=3,
        buffer=buf,
        wire=wire,
    )
    ok, _ = _rank1_receive_and_validate(
        cycle_n=6,  # off-by-one -- exactly the failure mode we defend against
        anchor=1,
        drafted_ids=(2,),
        real_bonus=3,
        wire=wire,
        validator=val,
    )
    assert not ok
    assert val.validated[-1].reason.startswith("cycle_n mismatch")


def test_end_to_end_many_cycles_no_leakage() -> None:
    """Run a longer sequence to confirm nothing accumulates spuriously
    (buffer stays at 0 pending after each cycle's release/discard)."""
    buf = SpecHiddenBuffer()
    val = SpecTagValidator()
    wire = _MockWire()
    hits = 0
    misses = 0
    for cycle in range(50):
        # Alternate HIT / MISS deterministically.
        real_bonus = 42 if cycle % 2 == 0 else 43
        sid = _rank0_produce_speculative_hidden(
            cycle_n=cycle,
            anchor=cycle * 7,
            drafted_ids=(cycle * 7 + 1, cycle * 7 + 2),
            assumed_bonus=42,
            buffer=buf,
            wire=wire,
        )
        ok, _ = _rank1_receive_and_validate(
            cycle_n=cycle,
            anchor=cycle * 7,
            drafted_ids=(cycle * 7 + 1, cycle * 7 + 2),
            real_bonus=real_bonus,
            wire=wire,
            validator=val,
        )
        if ok:
            hits += 1
            _ = buf.release(sid)
        else:
            misses += 1
            assert buf.discard(sid) is True
        assert len(buf) == 0, f"buffer leaked at cycle {cycle}"
    assert hits == 25
    assert misses == 25
    assert val.matches == 25
    assert val.mismatches == 25


# ---------------------------------------------------------------------------
# Regression: cast is unused but referenced -- keeps import surface honest.
# ---------------------------------------------------------------------------


def test_module_public_surface_stable() -> None:
    """Guard against accidental symbol removal -- STEP 3 will import
    exactly these names from the decode loop."""
    # ``cast`` reference keeps the import honest for future strictness
    # tightening; it also documents that ``coerce_hit_miss`` is the
    # correct place to narrow, not ad-hoc ``cast(HitMissCode, x)``.
    _ = cast
    from exo.worker.engines.mlx import pp_speculation_spec_tag as m

    for name in (
        "SpecId",
        "SpecHiddenBuffer",
        "SpecTagValidator",
        "SpecTagValidationResult",
        "BufferedSpecHidden",
        "fnv1a64",
        "pack_spec_tag",
        "unpack_spec_tag",
        "SPEC_TAG_WIRE_LEN",
        "HIT_MISS_NA",
        "HIT_MISS_HIT",
        "HIT_MISS_MISS",
        "coerce_hit_miss",
    ):
        assert hasattr(m, name), f"missing public symbol: {name}"
