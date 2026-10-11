"""Cross-rank park/restore coordination for DSv4.1 (restart-survival plan §G4).

WHY. Under tensor parallelism every rank holds a SHARD of each conversation's
cache and parks/restores it to its own node-local store. Before G4 each rank
decided independently (audit A7): rank A could restore a parked cache while
rank B cold-prefilled the same request, so the two ranks would issue different
forward passes and the model's TP collectives would hang (or, worse, pair up
wrong). G4 makes the decision a CLUSTER decision: every rank restores the SAME
generation of the SAME conversation, or every rank goes cold. Never mixed.

THE MECHANISM. One fixed-width int vector per rank, exchanged with a SINGLE
``all_gather`` over the engine's coord subgroup (``get_coord_group`` -- the
QP-less TCP control channel ``RankAgreement`` already uses), on the CPU stream
with an immediate ``mx.eval`` (``utils_mlx.mx_all_gather_ints``). Every rank
then runs the same PURE function over the identical gathered matrix, so the
decision is rank-invariant by construction -- no rank ever branches on a value
its peer cannot see. Payload widths are constants, so there is no
count-agreement round and no rank-dependent branching before any collective.

The four exchanges, each reached UNCONDITIONALLY by every rank at the same
between-turn point (the callers make participation independent of local
store state, including "no store" and "parking disabled"):

* restore #1 ``RESTORE``: each rank's admissible candidates (bounded to
  :data:`MAX_CANDIDATES`) -> :func:`decide_restore` -> one ``(key, gen)`` or
  COLD;
* restore #2 ``RESTORE_ACK`` (only when #1 said RESTORE, which is itself
  rank-invariant): each rank reports whether its restore of exactly that
  generation succeeded; any failure -> every rank discards -> COLD;
* park #1 ``PARK_GEN``: each rank's local max generation for the
  conversation -> every rank writes the SAME ``N = 1 + max``;
* park #2 ``PARK_ACK``: each rank reports whether it COMMITTED gen ``N``
  (written AND admissible on re-scan). Only when EVERY rank acks ``N`` does
  any rank prune, and then only generations ``G <= N - keep`` -- so gen ``N``
  (and ``N-1``) survive on every rank and a common restorable generation is
  never deleted by the prune. Any missing ack -> nobody prunes this turn (the
  plan's "lazy prune"; the byte/count LRU still bounds disk).

FAILURE SEMANTICS. Local work (scanning the store, reading manifests,
restoring) never raises out of the drivers: an error degrades that rank's
contribution to "nothing" (empty candidates -> COLD; unknown committed gen ->
no prune). A malformed/desynchronised row (bad magic, wrong width) is treated
the same way by every rank, because every rank sees the same rows. Only a
transport failure of the collective itself propagates (that is a real fault
the runner must surface, exactly as for ``RankAgreement``).

This module is MLX-free except :meth:`ParkCoordinator.from_group`, which
imports the collective helper lazily; :func:`decide_restore` and the codecs
are pure and are what test T8 drives directly.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar

__all__ = [
    "COLD",
    "MAX_CANDIDATES",
    "Candidate",
    "ParkCoordinator",
    "RankParkState",
    "RestoreDecision",
    "agree_generation",
    "build_identity_digest",
    "decide_restore",
    "decode_restore_payload",
    "encode_restore_payload",
    "key_int",
    "prune_floor",
    "run_park",
    "run_restore",
    "CheckpointOutcome",
    "IdleDecision",
    "decide_idle",
    "decide_poll",
    "run_checkpoint",
]

#: Candidates each rank reports per restore exchange (fixed payload width).
#: Two retained generations x two conversations covers the realistic case; a
#: truncation can only make a restore COLD, never mixed.
MAX_CANDIDATES = 4

#: 30-bit halves keep every payload int a non-negative int32.
_HALF_BITS = 30
_HALF_MASK = (1 << _HALF_BITS) - 1
#: Keys and digests are 60-bit ints (two 30-bit halves on the wire).
_WIDE_MASK = (1 << (2 * _HALF_BITS)) - 1

#: Per-exchange magic words: a row with the wrong magic means the ranks are
#: not in the same exchange (a desync); every rank treats it identically.
MAGIC_RESTORE = 0x4734_5201
MAGIC_RESTORE_ACK = 0x4734_5202
MAGIC_PARK_GEN = 0x4734_5203
MAGIC_PARK_ACK = 0x4734_5204
#: §G1 idle checkpoint: the "may we checkpoint now" row and the per-unit poll.
MAGIC_IDLE = 0x4734_5205
MAGIC_CKPT_POLL = 0x4734_5206

_ENTRY_WIDTH = 6  # key_lo, key_hi, gen, pref, digest_lo, digest_hi
RESTORE_WIDTH = 2 + MAX_CANDIDATES * _ENTRY_WIDTH
RESTORE_ACK_WIDTH = 5  # magic, ok, key_lo, key_hi, gen
PARK_WIDTH = 4  # magic, key_lo, key_hi, value
IDLE_WIDTH = 5  # magic, ok, key_lo, key_hi, offset
POLL_WIDTH = 4  # magic, pending, ok, more

#: ``PARK_ACK`` value meaning "this rank did NOT commit gen N" (write refused,
#: write failed, store unreadable): blocks pruning on every rank this turn.
NOT_COMMITTED = 0

#: Fields of :class:`~exo.worker.engines.mlx.dsv41.fingerprint.BuildFingerprint`
#: that must be EQUAL across ranks. ``layout_fp`` is deliberately excluded: it
#: hashes the rank and its built layer range, so it legitimately differs
#: between TP ranks (each rank's local ``_admit`` already gates it).
_BUILD_IDENTITY_FIELDS = ("model_fp", "tokenizer_fp", "kv_code_fp", "mlx_version")

Gather = Callable[[list[int]], list[list[int]]]


# ---------------------------------------------------------------------------
# small pure helpers
# ---------------------------------------------------------------------------

def _wide(value: str) -> int:
    """A 60-bit int digest of ``value`` (blake2b, platform independent)."""
    h = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "little") & _WIDE_MASK


def _split(v: int) -> tuple[int, int]:
    v &= _WIDE_MASK
    return v & _HALF_MASK, (v >> _HALF_BITS) & _HALF_MASK


def _join(lo: int, hi: int) -> int:
    return ((int(hi) & _HALF_MASK) << _HALF_BITS) | (int(lo) & _HALF_MASK)


def key_int(prefix_hash: str) -> int:
    """The 60-bit wire key of a conversation's ``prefix_hash`` (hex digest).

    ``prefix_hash`` is a deterministic function of the token stream, which is
    identical on every TP rank, so the same conversation has the same key on
    every rank (the audit's mirrored-store observation, 44/44).
    """
    try:
        return int(prefix_hash[:15], 16) & _WIDE_MASK
    except ValueError:
        return _wide(prefix_hash)


def build_identity_digest(fingerprint: Any) -> int:
    """60-bit digest of the CROSS-RANK-INVARIANT build fields of a fingerprint.

    Accepts a ``BuildFingerprint``, its ``as_dict()`` mapping, or ``None``
    (legacy: a fixed sentinel digest). Only model / tokenizer / KV-code / MLX
    identity is hashed -- never ``layout_fp``, which embeds the rank.
    """
    if fingerprint is None:
        return _wide("<no-fingerprint>")
    if isinstance(fingerprint, Mapping):
        src: Mapping[str, Any] = fingerprint
    else:
        src = {f: getattr(fingerprint, f, None) for f in _BUILD_IDENTITY_FIELDS}
    canon = json.dumps(
        {f: str(src.get(f)) for f in _BUILD_IDENTITY_FIELDS},
        sort_keys=True, separators=(",", ":"),
    )
    return _wide(canon)


# ---------------------------------------------------------------------------
# the pure decision (T8's unit under test)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Candidate:
    """One restorable generation as one rank sees it.

    ``key`` -- 60-bit conversation key (:func:`key_int` of its prefix hash);
    ``gen`` -- the generation number; ``pref`` -- the rank-invariant preference
    (LCP with the prompt, or token count when the client named the
    conversation); ``digest`` -- :func:`build_identity_digest` of the manifest's
    build fingerprint.
    """

    key: int
    gen: int
    pref: int
    digest: int


@dataclass(frozen=True)
class RankParkState:
    """One rank's admissible candidates for the restore under consideration."""

    candidates: tuple[Candidate, ...] = ()

    @classmethod
    def for_key(
        cls, key: int, admissible: Mapping[int, int], *, pref: int = 0
    ) -> "RankParkState":
        """The plan's ``{key, admissible: {gen -> digest}}`` shape."""
        return cls(tuple(
            Candidate(key=int(key), gen=int(g), pref=int(pref), digest=int(d))
            for g, d in sorted(admissible.items())
        ))

    def admissible(self, key: int) -> dict[int, int]:
        """``{gen -> digest}`` for one key (empty when the rank has none)."""
        return {c.gen: c.digest for c in self.candidates if c.key == key}

    @property
    def empty(self) -> bool:
        return not self.candidates


@dataclass(frozen=True)
class RestoreDecision:
    """COLD, or RESTORE of exactly one ``(key, gen)`` -- on EVERY rank.

    There is deliberately no per-rank field: the type cannot express a mixed
    outcome.
    """

    kind: Literal["cold", "restore"]
    key: int | None = None
    gen: int | None = None
    reason: str = ""

    @property
    def cold(self) -> bool:
        return self.kind == "cold"

    @classmethod
    def restore(cls, key: int, gen: int) -> "RestoreDecision":
        return cls("restore", int(key), int(gen))

    @classmethod
    def make_cold(cls, reason: str) -> "RestoreDecision":
        return cls("cold", None, None, reason)


COLD = RestoreDecision.make_cold("")


def decide_restore(rank_states: Sequence[RankParkState | None]) -> RestoreDecision:
    """The cluster restore decision from every rank's state. PURE.

    ``None`` stands for a rank whose state could not be read/decoded. Rules:

    * any rank ``None`` or empty -> COLD (all-or-nothing);
    * common = the ``(key, gen)`` pairs present on ALL ranks;
    * a common pair is eligible only if its digest is EQUAL on every rank and
      unambiguous within each rank;
    * pick the eligible pair with the highest ``(min pref, gen, key)``;
      none eligible -> COLD.

    The result is a single ``(key, gen)`` or COLD -- never a per-rank map --
    and, since every rank evaluates it over the same gathered rows, it is the
    same value on every rank.
    """
    if not rank_states:
        return RestoreDecision.make_cold("no-ranks")
    per_rank: list[dict[tuple[int, int], tuple[int, int]]] = []
    ambiguous: set[tuple[int, int]] = set()
    for st in rank_states:
        if st is None:
            return RestoreDecision.make_cold("rank-state-unreadable")
        if st.empty:
            return RestoreDecision.make_cold("rank-empty")
        seen: dict[tuple[int, int], tuple[int, int]] = {}
        for c in st.candidates:
            kg = (c.key, c.gen)
            if kg in seen and seen[kg][1] != c.digest:
                ambiguous.add(kg)
            seen.setdefault(kg, (c.pref, c.digest))
        per_rank.append(seen)
    common = set(per_rank[0])
    for seen in per_rank[1:]:
        common &= set(seen)
    if not common:
        return RestoreDecision.make_cold("no-common-generation")
    best: tuple[int, int, int] | None = None
    for kg in common - ambiguous:
        digests = {seen[kg][1] for seen in per_rank}
        if len(digests) != 1:
            continue  # same gen, different build on some rank: never mixed
        pref = min(seen[kg][0] for seen in per_rank)
        cand = (pref, kg[1], kg[0])
        if best is None or cand > best:
            best = cand
    if best is None:
        return RestoreDecision.make_cold("digest-mismatch")
    return RestoreDecision.restore(key=best[2], gen=best[1])


def agree_generation(rows: Sequence[Sequence[int]]) -> int:
    """The park generation every rank writes: ``1 + max`` local max gen. PURE.

    A malformed row contributes nothing; the result is >= 1 and identical on
    every rank (same rows in, same int out).
    """
    best = 0
    for row in rows:
        if len(row) == PARK_WIDTH and int(row[0]) == MAGIC_PARK_GEN:
            best = max(best, int(row[3]))
    return best + 1


def prune_floor(rows: Sequence[Sequence[int]], key: int, gen: int) -> int | None:
    """``gen`` iff EVERY rank acked committing ``(key, gen)``; else ``None``. PURE.

    ``None`` means "nobody prunes this turn". A malformed row, a row for a
    different conversation key or generation (a desync), or any
    :data:`NOT_COMMITTED` ack all yield ``None`` -- identically on every rank.
    """
    if not rows or gen < 1:
        return None
    for row in rows:
        if len(row) != PARK_WIDTH or int(row[0]) != MAGIC_PARK_ACK:
            return None
        if _join(row[1], row[2]) != key or int(row[3]) != int(gen):
            return None
    return int(gen)


# ---------------------------------------------------------------------------
# codecs (fixed width; decode never raises)
# ---------------------------------------------------------------------------

def encode_restore_payload(cands: Sequence[Candidate]) -> list[int]:
    """Fixed-width RESTORE row. Never raises (bad input -> an empty row)."""
    row = [MAGIC_RESTORE, 0] + [0] * (MAX_CANDIDATES * _ENTRY_WIDTH)
    try:
        take = list(cands)[:MAX_CANDIDATES]
        for i, c in enumerate(take):
            k_lo, k_hi = _split(int(c.key))
            d_lo, d_hi = _split(int(c.digest))
            gen, pref = int(c.gen), int(c.pref)
            if not (0 <= gen < 2**31 and 0 <= pref < 2**31):
                raise ValueError(f"out-of-range candidate {c!r}")
            base = 2 + i * _ENTRY_WIDTH
            row[base:base + _ENTRY_WIDTH] = [k_lo, k_hi, gen, pref, d_lo, d_hi]
        row[1] = len(take)
    except Exception:  # noqa: BLE001 - an unencodable state is just "empty"
        row = [MAGIC_RESTORE, 0] + [0] * (MAX_CANDIDATES * _ENTRY_WIDTH)
    return row


def decode_restore_payload(row: Sequence[int]) -> RankParkState | None:
    """Inverse of :func:`encode_restore_payload`; ``None`` for a malformed row."""
    try:
        if len(row) != RESTORE_WIDTH or int(row[0]) != MAGIC_RESTORE:
            return None
        n = int(row[1])
        if not 0 <= n <= MAX_CANDIDATES:
            return None
        out: list[Candidate] = []
        for i in range(n):
            base = 2 + i * _ENTRY_WIDTH
            k_lo, k_hi, gen, pref, d_lo, d_hi = (int(v) for v in row[base:base + _ENTRY_WIDTH])
            if gen < 0 or pref < 0:
                return None
            out.append(Candidate(_join(k_lo, k_hi), gen, pref, _join(d_lo, d_hi)))
        return RankParkState(tuple(out))
    except Exception:  # noqa: BLE001
        return None


def _park_row(magic: int, key: int, value: int) -> list[int]:
    lo, hi = _split(key)
    return [magic, lo, hi, int(value)]


# ---------------------------------------------------------------------------
# the coordinator + drivers
# ---------------------------------------------------------------------------

class ParkCoordinator:
    """Exchanges fixed-width int rows across the ranks of one coord group.

    ``gather(row)`` must return one row per rank, in rank order, and every rank
    must call it with a row of the same width at the same program point. In
    production it is ``utils_mlx.mx_all_gather_ints`` over the coord subgroup;
    the tests inject a deterministic in-process fake.
    """

    def __init__(self, gather: Gather, *, rank: int, world: int) -> None:
        self._gather = gather
        self.rank = int(rank)
        self.world = int(world)

    @classmethod
    def from_group(cls, group: Any, *, rank: int | None = None) -> "ParkCoordinator | None":
        """A coordinator over ``group`` (the COORD subgroup), or ``None``.

        ``None`` (single rank, PP mode's ``group=None``) means local-only
        park/restore -- exactly the pre-G4 single-rank behaviour.
        """
        if group is None:
            return None
        try:
            world = int(group.size())
        except Exception:  # noqa: BLE001
            return None
        if world <= 1:
            return None
        from exo.worker.engines.mlx.utils_mlx import mx_all_gather_ints

        def gather(row: list[int]) -> list[list[int]]:
            return mx_all_gather_ints(row, group)

        return cls(gather, rank=int(group.rank()) if rank is None else int(rank),
                   world=world)

    def exchange(self, row: list[int]) -> list[list[int]]:
        """One collective. Rows of the wrong count/width are returned as ``[]``
        (malformed -> every rank's pure decision treats them identically)."""
        rows = self._gather(list(row))
        width = len(row)
        out = [list(r) if len(r) == width else [] for r in rows]
        if len(out) != self.world:
            return [[] for _ in range(self.world)]
        return out

    def __repr__(self) -> str:
        return f"ParkCoordinator(rank={self.rank}, world={self.world})"


T = TypeVar("T")


@dataclass
class RestoreOutcome(Generic[T]):
    """What :func:`run_restore` did (for logging / tests)."""

    decision: RestoreDecision
    value: T | None
    acked: bool


def run_restore(
    coord: ParkCoordinator,
    local: Callable[[], Sequence[Candidate]],
    restore: Callable[[RestoreDecision], T | None],
    discard: Callable[[T], None],
) -> RestoreOutcome[T]:
    """The two-exchange all-or-nothing restore. Never raises on LOCAL errors.

    1. ``local()`` -> this rank's candidates (an exception -> empty); one
       RESTORE exchange; :func:`decide_restore` over the gathered rows.
    2. COLD -> return. RESTORE -> ``restore(decision)`` (an exception or
       ``None`` -> failure), then one RESTORE_ACK exchange; unless EVERY rank
       restored exactly that ``(key, gen)``, the restored value is passed to
       ``discard`` and the outcome is COLD on every rank.

    Whether step 2's exchange happens is a function of the decision, which is
    identical on every rank -- so the exchange count is rank-invariant.
    """
    try:
        cands = list(local())
    except Exception:  # noqa: BLE001 - a store error makes this rank empty
        cands = []
    rows = coord.exchange(encode_restore_payload(cands))
    decision = decide_restore([decode_restore_payload(r) for r in rows])
    if decision.cold:
        return RestoreOutcome(decision, None, False)
    assert decision.key is not None and decision.gen is not None
    value: T | None
    try:
        value = restore(decision)
    except Exception:  # noqa: BLE001
        value = None
    k_lo, k_hi = _split(decision.key)
    ack = [MAGIC_RESTORE_ACK, 1 if value is not None else 0, k_lo, k_hi, decision.gen]
    acks = coord.exchange(ack)
    all_ok = all(
        len(a) == RESTORE_ACK_WIDTH and a[0] == MAGIC_RESTORE_ACK and a[1] == 1
        and a[2] == k_lo and a[3] == k_hi and a[4] == decision.gen
        for a in acks
    )
    if not all_ok:
        if value is not None:
            with contextlib.suppress(Exception):
                discard(value)
        return RestoreOutcome(RestoreDecision.make_cold("restore-failed-on-a-rank"),
                              None, False)
    return RestoreOutcome(decision, value, True)


@dataclass
class ParkOutcome:
    """What :func:`run_park` did (for logging / tests)."""

    generation: int
    written: bool
    floor: int | None


def run_park(
    coord: ParkCoordinator,
    key: int,
    local_max: Callable[[], int],
    write: Callable[[int], bool],
    committed: Callable[[int], bool],
    prune: Callable[[int], None],
) -> ParkOutcome:
    """The two-exchange park: agree ``N``, write, ack, ack-gated prune.

    1. PARK_GEN exchange of ``local_max()`` (error -> 0) -> ``N`` =
       :func:`agree_generation` (identical on every rank).
    2. ``write(N)`` (error -> False). Ranks that cannot park still take part.
    3. PARK_ACK exchange of ``N if write ok and committed(N) else 0`` ->
       :func:`prune_floor`; when every rank acked, ``prune(N)`` (the caller
       deletes generations ``<= N - keep``).
    """
    try:
        mine = max(0, int(local_max()))
    except Exception:  # noqa: BLE001
        mine = 0
    rows = coord.exchange(_park_row(MAGIC_PARK_GEN, key, mine))
    gen = agree_generation(rows)
    try:
        written = bool(write(gen))
    except Exception:  # noqa: BLE001
        written = False
    try:
        done = gen if (written and bool(committed(gen))) else NOT_COMMITTED
    except Exception:  # noqa: BLE001
        done = NOT_COMMITTED
    acks = coord.exchange(_park_row(MAGIC_PARK_ACK, key, done))
    floor = prune_floor(acks, key, gen)
    if floor is not None:
        with contextlib.suppress(Exception):  # pruning is best-effort
            prune(floor)
    return ParkOutcome(gen, written, floor)


# ---------------------------------------------------------------------------
# §G1: the coordinated, cancellable after-turn idle checkpoint
# ---------------------------------------------------------------------------
#
# A checkpoint is a park that KEEPS the conversation resident, written one
# unit (~one layer) at a time while the engine thread is idle. It reuses the
# G4 generation protocol verbatim (PARK_GEN -> write -> PARK_ACK -> ack-gated
# prune) and adds two exchanges:
#
# * ``IDLE`` (once, first): every rank reports whether it may checkpoint NOW
#   (no request pending, a usable store, a park-safe conversation) and WHICH
#   conversation (key + offset). Go only when every rank says ok for the SAME
#   (key, offset); anything else is a rank-invariant no-go.
# * ``CKPT_POLL`` (before every unit and once after the last): every rank
#   reports ``pending`` (a request arrived), ``ok`` (its local writes are
#   healthy) and ``more`` (it still has units to write). Any pending / not-ok
#   / malformed row -> EVERY rank aborts (removes its temp dir, keeps the
#   previous generation) -- so a request waits for at most the one unit that
#   was in flight. No rank has more -> commit + PARK_ACK. Otherwise each rank
#   with more writes exactly one unit and polls again.
#
# The number of exchanges is a pure function of the gathered rows, so it is
# identical on every rank; a local failure is reported as ``ok=0`` in the NEXT
# row, never as a skipped exchange.

#: Hard cap on poll rounds (rank-invariant: every rank counts the same rounds).
MAX_CKPT_POLLS = 1 << 20


@dataclass(frozen=True)
class IdleDecision:
    """Whether every rank agreed to checkpoint exactly one ``(key, offset)``."""

    go: bool
    key: int = 0
    offset: int = 0
    reason: str = ""


def _idle_row(ok: bool, key: int, offset: int) -> list[int]:
    lo, hi = _split(key)
    return [MAGIC_IDLE, 1 if ok else 0, lo, hi, int(offset)]


def decide_idle(rows: Sequence[Sequence[int]]) -> IdleDecision:
    """GO iff every row is a valid IDLE row with ok=1 and identical key/offset. PURE."""
    if not rows:
        return IdleDecision(False, reason="no-ranks")
    first: tuple[int, int] | None = None
    for row in rows:
        if len(row) != IDLE_WIDTH or int(row[0]) != MAGIC_IDLE:
            return IdleDecision(False, reason="malformed")
        if int(row[1]) != 1:
            return IdleDecision(False, reason="rank-not-ready")
        kv = (_join(row[2], row[3]), int(row[4]))
        if first is None:
            first = kv
        elif kv != first:
            return IdleDecision(False, reason="candidate-mismatch")
    assert first is not None
    return IdleDecision(True, key=first[0], offset=first[1])


PollVerdict = Literal["continue", "done", "abort-pending", "abort-failed"]


def _poll_row(pending: bool, ok: bool, more: bool) -> list[int]:
    return [MAGIC_CKPT_POLL, 1 if pending else 0, 1 if ok else 0, 1 if more else 0]


def decide_poll(rows: Sequence[Sequence[int]]) -> PollVerdict:
    """The per-unit verdict, identical on every rank. PURE.

    Any malformed row or any ``ok=0`` -> ``abort-failed``; else any
    ``pending=1`` -> ``abort-pending``; else nobody has more -> ``done``; else
    ``continue``.
    """
    if not rows:
        return "abort-failed"
    pending = False
    more = False
    for row in rows:
        if len(row) != POLL_WIDTH or int(row[0]) != MAGIC_CKPT_POLL:
            return "abort-failed"
        if int(row[2]) != 1:
            return "abort-failed"
        pending = pending or int(row[1]) == 1
        more = more or int(row[3]) == 1
    if pending:
        return "abort-pending"
    return "continue" if more else "done"


@dataclass
class CheckpointOutcome:
    """What :func:`run_checkpoint` did. ``all_committed`` is the AGREED result
    (every rank acked gen N) and is identical on every rank; ``written`` is
    this rank's local commit only."""

    go: bool
    reason: str = ""
    generation: int = 0
    written: bool = False
    floor: int | None = None
    aborted: bool = False
    pending_abort: bool = False
    units: int = 0
    polls: int = 0

    @property
    def all_committed(self) -> bool:
        return self.floor is not None


def _safe(fn: Callable[[], Any], default: Any) -> Any:
    try:
        return fn()
    except Exception:  # noqa: BLE001 - local errors become row values
        return default


def run_checkpoint(
    coord: ParkCoordinator,
    key: int,
    offset: int,
    *,
    ready: Callable[[], bool],
    pending: Callable[[], bool],
    local_max: Callable[[], int],
    begin: Callable[[int], bool],
    write_next: Callable[[], bool],
    has_more: Callable[[], bool],
    commit: Callable[[], bool],
    committed: Callable[[int], bool],
    prune: Callable[[int], None],
    abort: Callable[[], None],
) -> CheckpointOutcome:
    """The coordinated, cancellable checkpoint. Never raises on LOCAL errors.

    ``ready()`` -> the IDLE ``ok`` bit; ``begin(N)`` opens gen N's temp dir;
    ``write_next()`` writes ONE unit and returns whether more remain;
    ``has_more()`` is consulted after ``begin``; ``commit()`` writes the
    manifest + renames; ``abort()`` removes the temp dir. See the section
    comment above for the exchange sequence.
    """
    ok_now = bool(_safe(ready, False)) and not bool(_safe(pending, True))
    rows = coord.exchange(_idle_row(ok_now, key, offset))
    idle = decide_idle(rows)
    if not idle.go:
        return CheckpointOutcome(go=False, reason=idle.reason)

    mine = max(0, int(_safe(local_max, 0)))
    gen = agree_generation(coord.exchange(_park_row(MAGIC_PARK_GEN, key, mine)))
    ok = bool(_safe(lambda: begin(gen), False))
    more = bool(_safe(has_more, False)) if ok else False
    units = 0
    polls = 0
    verdict: PollVerdict = "abort-failed"
    while polls < MAX_CKPT_POLLS:
        pend = bool(_safe(pending, True))
        polls += 1
        verdict = decide_poll(coord.exchange(_poll_row(pend, ok, more)))
        if verdict != "continue":
            break
        if ok and more:
            try:
                more = bool(write_next())
                units += 1
            except Exception:  # noqa: BLE001 - reported as ok=0 next poll
                ok = False
                more = False
    else:
        verdict = "abort-failed"  # cap reached on EVERY rank (same count)
    if verdict != "done":
        with contextlib.suppress(Exception):
            abort()
        return CheckpointOutcome(
            go=True, reason=verdict, generation=gen, aborted=True,
            pending_abort=verdict == "abort-pending", units=units, polls=polls,
        )
    written = ok and bool(_safe(commit, False))
    if not written:
        with contextlib.suppress(Exception):
            abort()
    done = gen if (written and bool(_safe(lambda: committed(gen), False))) else NOT_COMMITTED
    acks = coord.exchange(_park_row(MAGIC_PARK_ACK, key, done))
    floor = prune_floor(acks, key, gen)
    if floor is not None:
        with contextlib.suppress(Exception):  # pruning is best-effort
            prune(floor)
    return CheckpointOutcome(
        go=True, reason="committed" if floor is not None else "not-acked",
        generation=gen, written=written, floor=floor, units=units, polls=polls,
    )
