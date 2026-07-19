"""Speculation tagging + hidden-state buffering primitives for PP+DSpark
draft-ahead overlap (STEP 1 of the design; diagnostic-only, not wired to
the decode branch yet).

Purpose
-------
When rank 0 begins speculatively forwarding the extension block during
its own idle window (a later step), the resulting hidden state is
`SPECULATIVE`: it assumes a specific assumed-prefix (current cycle's
anchor + full acceptance of the drafted positions + a specific bonus
token). If rank 1 consumed that hidden without validating the
assumption, a mismatch would silently corrupt every subsequent cycle
(same failure class as the DSpark draft-width-truncation bug: no crash,
no error, just garbage output at a high reported accept rate).

The plumbing below implements the *safer* of the two options called out
in the design doc: **buffer-until-confirmed on rank 0**, layered with
explicit ``SpecId`` + prefix-hash tagging as defence-in-depth so rank 1
*also* validates before consuming. Both mechanisms are cheap; wearing
belt and braces here costs ~0.5ms on the hit path and eliminates a
whole class of silent-desync bugs.

This module is deliberately pure & side-effect-free -- no MLX
distributed calls, no logging in hot paths, no globals. The decode
loop imports these primitives, integrates them behind
``EXO_PP_DSPARK_DRAFT_AHEAD=1``, and (in diagnostic mode) uses them
only to *validate* that tag matching is sound under real cross-rank
timing; the actual speculative forward / hit-miss branch is a later
step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal, final

import mlx.core as mx

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# FNV-1a 64-bit parameters. Chosen over the built-in ``hash()`` because
# ``hash()`` is salted per-process (PYTHONHASHSEED), so rank 0 and rank 1
# would compute *different* hashes for the same prefix -- exactly the
# silent-desync failure this whole subsystem is here to prevent.
_FNV_OFFSET: Final[int] = 0xCBF29CE484222325
_FNV_PRIME: Final[int] = 0x100000001B3
_U64_MASK: Final[int] = 0xFFFFFFFFFFFFFFFF
_U32_MASK: Final[int] = 0xFFFFFFFF

# Wire header layout (int32 slots). Kept small and fixed so it can ride
# alongside the existing msg1/msg2 payloads without a new distributed op.
#   [0] packed_kind:  low byte = spec_kind code, bits 8/9 = recovered
#                     high-bits of hash_hi/hash_lo (int32 wire slots
#                     can't hold values >= 2**31 without ambiguity).
#   [1] cycle_n:      monotonic per stream, wraps at int32
#   [2] hash_hi_wire: high 32 bits of the 64-bit FNV-1a prefix hash
#                     (top bit stashed in packed_kind bit 8)
#   [3] hash_lo_wire: low  32 bits (top bit stashed in packed_kind bit 9)
#   [4] prefix_len:   number of tokens contributing to the hash (int32)
SPEC_TAG_WIRE_LEN: Final[int] = 5

SpecKind = Literal["non_spec", "draft_ahead"]

_SPEC_KIND_TO_CODE: Final[dict[SpecKind, int]] = {
    "non_spec": 0,
    "draft_ahead": 1,
}
_CODE_TO_SPEC_KIND: Final[dict[int, SpecKind]] = {
    v: k for k, v in _SPEC_KIND_TO_CODE.items()
}


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def fnv1a64(values: tuple[int, ...]) -> int:
    """Deterministic, cross-process-stable 64-bit hash of a token tuple.

    Used to derive ``SpecId.prefix_hash`` from an assumed-prefix so that
    rank 0 (sender) and rank 1 (validator) can agree on whether they're
    talking about the *same* speculative branch even though they never
    exchange the prefix itself. Standard FNV-1a over 4 little-endian
    bytes per token id.
    """
    h = _FNV_OFFSET
    for v in values:
        vv = int(v) & _U32_MASK
        for shift in (0, 8, 16, 24):
            h ^= (vv >> shift) & 0xFF
            h = (h * _FNV_PRIME) & _U64_MASK
    return h


# ---------------------------------------------------------------------------
# SpecId
# ---------------------------------------------------------------------------


@final
@dataclass(frozen=True, slots=True)
class SpecId:
    """Identifier attached to every hidden state crossing the PP boundary.

    Attributes
    ----------
    spec_kind:
        Which branch of the protocol produced this hidden. ``"non_spec"``
        is the existing (default, always-safe) path; ``"draft_ahead"``
        marks a speculatively-forwarded hidden that rank 1 must validate
        before consuming.
    cycle_n:
        The decode cycle index the sender believed it was on when
        producing this hidden. Wraps at 2**31 (int32 on the wire); wrap
        is fine because the validator only checks *equality* against its
        own current-cycle expectation.
    prefix_hash:
        64-bit FNV-1a of the ordered token prefix the sender *assumed*
        when producing this hidden -- for ``draft_ahead`` this is
        (anchor, accepted_ids..., assumed_bonus_token). For
        ``non_spec`` it is the actual current-cycle prefix. Mismatch
        against the validator's own computed hash means the assumption
        was violated and the hidden MUST be discarded.
    prefix_len:
        Length of the token sequence hashed. Recorded separately from
        the hash so a length mismatch (structural bug) is distinguishable
        from a value mismatch (contents diverged).
    """

    spec_kind: SpecKind
    cycle_n: int
    prefix_hash: int
    prefix_len: int

    def __post_init__(self) -> None:
        if self.cycle_n < 0:
            raise ValueError(f"cycle_n must be >= 0, got {self.cycle_n}")
        if self.prefix_len < 0:
            raise ValueError(f"prefix_len must be >= 0, got {self.prefix_len}")
        if not (0 <= self.prefix_hash <= _U64_MASK):
            raise ValueError(f"prefix_hash must fit in u64, got {self.prefix_hash}")

    @staticmethod
    def build(
        *,
        spec_kind: SpecKind,
        cycle_n: int,
        prefix: tuple[int, ...],
    ) -> SpecId:
        """Construct a ``SpecId`` from a raw token prefix (canonical path)."""
        return SpecId(
            spec_kind=spec_kind,
            cycle_n=cycle_n,
            prefix_hash=fnv1a64(prefix),
            prefix_len=len(prefix),
        )


# ---------------------------------------------------------------------------
# Wire encoding (round-trip via mx.array so it can hitch a ride on the
# existing pp_group send/recv path with no new distributed op)
# ---------------------------------------------------------------------------


def pack_spec_tag(spec_id: SpecId) -> mx.array:
    """Encode a ``SpecId`` into a fixed-shape ``int32`` array of length
    :data:`SPEC_TAG_WIRE_LEN`. Idempotent, no side effects."""
    kind_code = _SPEC_KIND_TO_CODE[spec_id.spec_kind]
    # Signed int32 range: fold cycle_n via bit-truncation. Recovery is
    # simply the same bit-pattern; equality checks still work since both
    # ranks apply the identical fold.
    cycle_wire = spec_id.cycle_n & 0x7FFFFFFF
    hash_hi = (spec_id.prefix_hash >> 32) & _U32_MASK
    hash_lo = spec_id.prefix_hash & _U32_MASK
    # int32 can't hold values >= 2**31; stash the dropped high bit in
    # the packed_kind slot so bit-exact round-trip is preserved.
    hash_hi_wire = hash_hi & 0x7FFFFFFF
    hash_lo_wire = hash_lo & 0x7FFFFFFF
    hash_hi_bit31 = (hash_hi >> 31) & 0x1
    hash_lo_bit31 = (hash_lo >> 31) & 0x1
    packed_kind = (kind_code & 0xFF) | (hash_hi_bit31 << 8) | (hash_lo_bit31 << 9)
    return mx.array(
        [packed_kind, cycle_wire, hash_hi_wire, hash_lo_wire, spec_id.prefix_len],
        dtype=mx.int32,
    )


def unpack_spec_tag(wire: mx.array) -> SpecId:
    """Inverse of :func:`pack_spec_tag`. Raises ``ValueError`` on any
    field the constructor rejects, so a corrupted/mistranscribed wire
    payload fails loudly rather than silently."""
    if wire.shape != (SPEC_TAG_WIRE_LEN,):
        raise ValueError(
            f"spec tag wire must have shape ({SPEC_TAG_WIRE_LEN},), got {wire.shape!r}"
        )
    if wire.dtype != mx.int32:
        raise ValueError(f"spec tag wire must be int32, got {wire.dtype!r}")
    raw = wire.tolist()
    if not isinstance(raw, list):
        raise ValueError(f"spec tag wire.tolist() must be a list, got {type(raw)!r}")
    raw_values: list[int] = []
    for v in raw:
        if isinstance(v, list):
            raise ValueError(f"spec tag wire has nested list element: {v!r}")
        raw_values.append(int(v))
    packed_kind, cycle_wire, hash_hi_wire, hash_lo_wire, prefix_len = raw_values
    kind_code = packed_kind & 0xFF
    hash_hi_bit31 = (packed_kind >> 8) & 0x1
    hash_lo_bit31 = (packed_kind >> 9) & 0x1
    hash_hi = (hash_hi_wire & 0x7FFFFFFF) | (hash_hi_bit31 << 31)
    hash_lo = (hash_lo_wire & 0x7FFFFFFF) | (hash_lo_bit31 << 31)
    if kind_code not in _CODE_TO_SPEC_KIND:
        raise ValueError(f"unknown spec_kind code: {kind_code}")
    prefix_hash = ((hash_hi & _U32_MASK) << 32) | (hash_lo & _U32_MASK)
    return SpecId(
        spec_kind=_CODE_TO_SPEC_KIND[kind_code],
        cycle_n=cycle_wire,
        prefix_hash=prefix_hash,
        prefix_len=prefix_len,
    )


# ---------------------------------------------------------------------------
# Rank-0 buffer: hold speculative hiddens until the hit/miss bit arrives
# ---------------------------------------------------------------------------


@final
@dataclass(frozen=False, slots=True)
class BufferedSpecHidden:
    """A speculative hidden state parked on rank 0 awaiting confirmation."""

    spec_id: SpecId
    hidden: mx.array
    # Sentinel from the send-op ``mx.eval()`` result rank 0 committed
    # for this hidden. Held so a later ``release()`` / ``discard()``
    # can enforce explicit ordering wrt msg2's trim decision
    # (design-doc failure mode #3: eval/scheduling ordering must NOT be
    # implicit from lazy MLX evaluation timing).
    eval_sentinel: mx.array | None = None


@final
class SpecHiddenBuffer:
    """Buffer of pending speculative hiddens keyed by ``SpecId``.

    Rank 0 stashes each speculatively-forwarded hidden here immediately
    after producing it, then either :meth:`release` (msg2 says HIT --
    hidden gets sent to rank 1) or :meth:`discard` (msg2 says MISS --
    hidden dropped, KV rollback path takes over).

    In *diagnostic mode* the release/discard step still happens, but
    the hidden is never actually consumed by the decode branch -- we
    just validate that the buffer contained exactly the SpecId the
    rank 1 validator expected. That confirms the tagging plumbing is
    sound under real cross-rank timing before the actual
    speculative-forward branch is wired in.
    """

    _pending: dict[tuple[int, int, int], BufferedSpecHidden]

    def __init__(self) -> None:
        # Key: (cycle_n, prefix_hash, prefix_len) -- spec_kind alone
        # would collide across cycles.
        self._pending = {}

    def __len__(self) -> int:
        return len(self._pending)

    @staticmethod
    def _key(spec_id: SpecId) -> tuple[int, int, int]:
        return (spec_id.cycle_n, spec_id.prefix_hash, spec_id.prefix_len)

    def stash(
        self,
        spec_id: SpecId,
        hidden: mx.array,
        eval_sentinel: mx.array | None = None,
    ) -> None:
        """Park a speculative hidden. Rejects duplicate keys so a bug
        in the caller (double-stash under one id) fails loudly."""
        if spec_id.spec_kind == "non_spec":
            raise ValueError(
                "SpecHiddenBuffer only stores speculative hiddens; "
                f"got spec_kind={spec_id.spec_kind!r}"
            )
        key = self._key(spec_id)
        if key in self._pending:
            raise ValueError(f"duplicate spec_id stash: {spec_id!r}")
        self._pending[key] = BufferedSpecHidden(
            spec_id=spec_id, hidden=hidden, eval_sentinel=eval_sentinel
        )

    def release(self, spec_id: SpecId) -> BufferedSpecHidden:
        """Pop the buffered hidden matching ``spec_id`` (HIT path).
        Raises ``KeyError`` if nothing matches -- caller must handle."""
        key = self._key(spec_id)
        entry = self._pending.pop(key, None)
        if entry is None:
            raise KeyError(
                f"no buffered speculative hidden for {spec_id!r}; "
                f"pending keys: {list(self._pending.keys())}"
            )
        return entry

    def discard(self, spec_id: SpecId) -> bool:
        """Drop the buffered hidden matching ``spec_id`` (MISS path).
        Returns True if something was dropped, False if nothing matched
        (both are legitimate outcomes -- e.g. never stashed because
        speculation was gated off this cycle)."""
        return self._pending.pop(self._key(spec_id), None) is not None

    def clear(self) -> None:
        """Drop every pending entry (used on decode-loop teardown)."""
        self._pending.clear()

    def peek_keys(self) -> tuple[tuple[int, int, int], ...]:
        """Snapshot of current buffer keys, for logging/assertions."""
        return tuple(self._pending.keys())


# ---------------------------------------------------------------------------
# Rank-1 validator
# ---------------------------------------------------------------------------


@final
@dataclass(frozen=True, slots=True)
class SpecTagValidationResult:
    """Outcome of comparing an incoming SpecId to the local expectation."""

    ok: bool
    incoming: SpecId
    expected: SpecId
    reason: str = ""


@final
@dataclass(frozen=False, slots=True)
class SpecTagValidator:
    """Rank-1-side validator. Cheap, stateless w.r.t. hidden contents.

    In diagnostic mode the caller *always* falls through to the
    non-speculative path regardless of validation outcome; validation
    failures are logged (and, in tests, asserted on) but don't corrupt
    the decode path because the speculative hidden is never consumed.
    """

    mismatches: int = 0
    matches: int = 0
    validated: list[SpecTagValidationResult] = field(default_factory=list)

    def validate(self, incoming: SpecId, expected: SpecId) -> SpecTagValidationResult:
        if incoming.spec_kind != expected.spec_kind:
            result = SpecTagValidationResult(
                ok=False,
                incoming=incoming,
                expected=expected,
                reason=(
                    f"spec_kind mismatch: incoming={incoming.spec_kind!r} "
                    f"expected={expected.spec_kind!r}"
                ),
            )
        elif incoming.cycle_n != expected.cycle_n:
            result = SpecTagValidationResult(
                ok=False,
                incoming=incoming,
                expected=expected,
                reason=(
                    f"cycle_n mismatch: incoming={incoming.cycle_n} "
                    f"expected={expected.cycle_n}"
                ),
            )
        elif incoming.prefix_len != expected.prefix_len:
            result = SpecTagValidationResult(
                ok=False,
                incoming=incoming,
                expected=expected,
                reason=(
                    f"prefix_len mismatch: incoming={incoming.prefix_len} "
                    f"expected={expected.prefix_len}"
                ),
            )
        elif incoming.prefix_hash != expected.prefix_hash:
            result = SpecTagValidationResult(
                ok=False,
                incoming=incoming,
                expected=expected,
                reason=(
                    "prefix_hash mismatch "
                    f"(incoming=0x{incoming.prefix_hash:016x} "
                    f"expected=0x{expected.prefix_hash:016x})"
                ),
            )
        else:
            result = SpecTagValidationResult(
                ok=True, incoming=incoming, expected=expected
            )
        if result.ok:
            self.matches += 1
        else:
            self.mismatches += 1
        self.validated.append(result)
        return result

    def reset(self) -> None:
        self.matches = 0
        self.mismatches = 0
        self.validated.clear()


# ---------------------------------------------------------------------------
# msg2 wire extension: hit/miss bit
# ---------------------------------------------------------------------------

# The existing msg2 payload is:
#   [0]         n_accepted
#   [1]         bonus_token
#   [2..vw]     padded accepted ids (length vw-1)
# STEP 1 extends it by exactly one int32 at the end:
#   [vw+1]     hit/miss/na code
#              -1 = draft-ahead was not attempted this cycle (default in
#                   diagnostic mode -- nothing produces a speculative
#                   hidden yet, so rank 0 has nothing to release/discard)
#               0 = MISS: rank 0 must discard its buffered spec hidden
#               1 = HIT:  rank 0 must release its buffered spec hidden
#
# Adding the slot now (rather than in the step that first *uses* it) is
# deliberate per the design doc: it forces both ranks to speak the same
# expanded wire protocol at commit time, so the actual speculative
# branch (STEP 3) can be enabled with a pure logic change and no
# further wire-shape churn.

HitMissCode = Literal[-1, 0, 1]

HIT_MISS_NA: Final[HitMissCode] = -1
HIT_MISS_MISS: Final[HitMissCode] = 0
HIT_MISS_HIT: Final[HitMissCode] = 1


def coerce_hit_miss(code: int) -> HitMissCode:
    """Narrow an int received off the wire into the ``HitMissCode``
    Literal set. Raises ``ValueError`` on unknown values -- any decode
    path checking the bit should be forced to handle the exhaustive
    set, not accept arbitrary garbage."""
    if code == HIT_MISS_NA:
        return HIT_MISS_NA
    if code == HIT_MISS_MISS:
        return HIT_MISS_MISS
    if code == HIT_MISS_HIT:
        return HIT_MISS_HIT
    raise ValueError(f"unknown hit/miss code: {code}")
