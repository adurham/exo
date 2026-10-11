"""CPU-only tests for the DSv4.1 multi-rank park/restore coordination (§G4).

No GPU, no cluster, no MLX collectives. Two layers:

* the PURE decision (``coord.decide_restore`` / ``agree_generation`` /
  ``prune_floor``) driven directly with hand-built per-rank states -- this is
  where T8's "never mixed" is proven;
* two simulated TP ranks = two ``ParkedStore`` instances rooted at two tmp
  dirs, each driven on its own thread through the REAL coordinated park/restore
  path (``Dsv41Sessions`` -> ``coordinated_park`` / ``coordinated_restore`` ->
  ``run_park`` / ``run_restore``), with an in-process lockstep gather standing
  in for ``mx_all_gather_ints`` over the coord subgroup. The gather is a
  barrier with a timeout, so a rank that skipped an exchange FAILS the test
  instead of hanging it.

T8 (plan §Tests): one rank missing gen N -> everyone gen N-1 or all-cold;
NEVER mixed.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from exo.worker.engines.mlx.dsv41 import coord as co
from exo.worker.engines.mlx.dsv41 import fingerprint as fp
from exo.worker.engines.mlx.dsv41 import park as pk
from exo.worker.engines.mlx.dsv41 import session as s_

from ._stub_harness import StubBody, run_turn

CONV_KW: dict[str, Any] = dict(chunk=4, long_chunk=4, long_threshold=10**9)
IDS = np.arange(10, 82, dtype=np.int64)
KEY = 0xABCDEF


def _fp(layout: str = "layout", **over: str) -> fp.BuildFingerprint:
    base = dict(model_fp="model", tokenizer_fp="tok", kv_code_fp="kv",
                mlx_version="mlx-v1", layout_fp=layout)
    base.update(over)
    return fp.BuildFingerprint(**base)


D = co.build_identity_digest(_fp())


def _st(admissible: dict[int, int], key: int = KEY) -> co.RankParkState:
    return co.RankParkState.for_key(key, admissible)


# ---------------------------------------------------------------------------
# T8 -- the pure decision
# ---------------------------------------------------------------------------

def _assert_single(dec: co.RestoreDecision) -> None:
    """The "never mixed" invariant: one (key, gen) for all ranks, or COLD."""
    assert isinstance(dec, co.RestoreDecision)
    assert dec.kind in ("cold", "restore")
    if dec.cold:
        assert dec.key is None and dec.gen is None
    else:
        assert isinstance(dec.key, int) and isinstance(dec.gen, int)
    # The type has no per-rank field at all.
    assert not any(isinstance(v, (dict, list, tuple)) for v in vars(dec).values())


@pytest.mark.parametrize(
    "a,b,expect",
    [
        ({1: D, 2: D}, {1: D, 2: D}, 2),      # both ranks {1,2} -> RESTORE(2)
        ({1: D, 2: D}, {1: D}, 1),            # B missing gen N -> everyone N-1
        ({1: D}, {1: D, 2: D}, 1),            # symmetric
        ({2: D}, {1: D}, None),               # no common gen -> COLD
        ({2: D}, {2: D + 1}, None),           # same gen, digests differ -> COLD
        ({1: D, 2: D}, {1: D, 2: D + 1}, 1),  # gen 2 mismatched -> common gen 1
        ({1: D, 2: D}, {}, None),             # one rank empty -> COLD
        ({}, {}, None),
    ],
)
def test_t8_decide_restore(a: dict[int, int], b: dict[int, int], expect: Any) -> None:
    for states in ([_st(a), _st(b)], [_st(b), _st(a)]):  # rank-order invariant
        dec = co.decide_restore(states)
        _assert_single(dec)
        if expect is None:
            assert dec.cold
        else:
            assert not dec.cold and dec.gen == expect and dec.key == KEY


def test_t8_unreadable_rank_state_is_cold() -> None:
    dec = co.decide_restore([_st({1: D}), None])
    assert dec.cold and dec.reason == "rank-state-unreadable"


def test_t8_key_mismatch_across_ranks_is_cold() -> None:
    dec = co.decide_restore([_st({1: D}, key=1), _st({1: D}, key=2)])
    assert dec.cold


def test_t8_single_rank_is_local_best() -> None:
    dec = co.decide_restore([_st({1: D, 2: D, 3: D})])
    assert dec.gen == 3


def test_t8_never_mixed_exhaustive() -> None:
    """Every 2-rank combination of gens {1,2,3} x 2 digests: the decision is a
    single gen present on BOTH ranks with EQUAL digests, the highest such, or
    COLD exactly when none exists."""
    import itertools

    options: list[dict[int, int]] = [{}]
    for r in range(1, 4):
        for gens in itertools.combinations((1, 2, 3), r):
            for digs in itertools.product((D, D + 7), repeat=r):
                options.append(dict(zip(gens, digs, strict=True)))
    for a in options:
        for b in options:
            dec = co.decide_restore([_st(a), _st(b)])
            _assert_single(dec)
            ok = [g for g in set(a) & set(b) if a[g] == b[g]]
            if ok:
                assert dec.gen == max(ok), (a, b, dec)
            else:
                assert dec.cold, (a, b, dec)


def test_payload_roundtrip_and_malformed_rows() -> None:
    cands = [co.Candidate(key=(1 << 59) + 5, gen=3, pref=100, digest=D),
             co.Candidate(key=7, gen=2, pref=0, digest=co.build_identity_digest(None))]
    row = co.encode_restore_payload(cands)
    assert len(row) == co.RESTORE_WIDTH
    assert all(0 <= v < 2**31 for v in row)
    st = co.decode_restore_payload(row)
    assert st is not None and list(st.candidates) == cands
    # Bounded payload: excess candidates are truncated, never a wider row.
    many = [co.Candidate(KEY, g, 0, D) for g in range(1, 10)]
    assert len(co.encode_restore_payload(many)) == co.RESTORE_WIDTH
    # Malformed rows decode to None (-> COLD on every rank).
    assert co.decode_restore_payload([]) is None
    assert co.decode_restore_payload([0] * co.RESTORE_WIDTH) is None
    bad = list(row)
    bad[1] = co.MAX_CANDIDATES + 1
    assert co.decode_restore_payload(bad) is None
    # An unencodable candidate degrades to an EMPTY row, never raises.
    assert co.decode_restore_payload(
        co.encode_restore_payload([co.Candidate(KEY, -1, 0, D)])) == co.RankParkState()


def test_build_identity_digest_ignores_layout() -> None:
    a = co.build_identity_digest(_fp(layout="rank0-layers0-30"))
    b = co.build_identity_digest(_fp(layout="rank1-layers0-30"))
    assert a == b
    assert a == co.build_identity_digest(_fp().as_dict())
    for f in ("model_fp", "tokenizer_fp", "kv_code_fp", "mlx_version"):
        assert co.build_identity_digest(_fp(**{f: "other"})) != a, f


def test_agree_generation_and_prune_floor_pure() -> None:
    rows = [[co.MAGIC_PARK_GEN, 0, 0, 3], [co.MAGIC_PARK_GEN, 0, 0, 1]]
    assert co.agree_generation(rows) == 4
    assert co.agree_generation(list(reversed(rows))) == 4
    assert co.agree_generation([[co.MAGIC_PARK_GEN, 0, 0, 0]] * 2) == 1
    lo, hi = KEY & ((1 << 30) - 1), KEY >> 30
    ack = [co.MAGIC_PARK_ACK, lo, hi, 4]
    assert co.prune_floor([ack, ack], KEY, 4) == 4
    assert co.prune_floor([ack, [co.MAGIC_PARK_ACK, lo, hi, 0]], KEY, 4) is None
    assert co.prune_floor([ack, []], KEY, 4) is None
    assert co.prune_floor([ack, [co.MAGIC_PARK_ACK, lo + 1, hi, 4]], KEY, 4) is None


# ---------------------------------------------------------------------------
# the driver with a scripted peer (no threads): fault injection
# ---------------------------------------------------------------------------

def _scripted(peer_rows: list[list[int]]) -> co.ParkCoordinator:
    """Rank 0 of 2; the peer's rows are replayed in order."""
    q = list(peer_rows)
    return co.ParkCoordinator(lambda row: [row, q.pop(0)], rank=0, world=2)


def test_t8_fault_injection_local_error_is_cold_not_raise() -> None:
    peer = co.encode_restore_payload([co.Candidate(KEY, 1, 0, D)])
    restored: list[Any] = []

    def boom() -> list[co.Candidate]:
        raise OSError("disk on fire")

    out = co.run_restore(_scripted([peer]), boom, restored.append, lambda v: None)
    assert out.decision.cold and out.value is None and not restored


def test_t8_restore_failure_on_one_rank_discards_everywhere() -> None:
    """Both ranks decide RESTORE(1); the PEER's restore fails -> this rank
    discards its successfully restored value and goes COLD too."""
    cand = co.Candidate(KEY, 1, 0, D)
    peer_rows = [co.encode_restore_payload([cand]),
                 [co.MAGIC_RESTORE_ACK, 0, KEY & ((1 << 30) - 1), KEY >> 30, 1]]
    discarded: list[Any] = []
    out = co.run_restore(_scripted(peer_rows), lambda: [cand],
                         lambda dec: ("conv", dec.gen), discarded.append)
    assert out.decision.cold and out.value is None
    assert discarded == [("conv", 1)]


# ---------------------------------------------------------------------------
# two simulated ranks over real ParkedStores (threads + lockstep gather)
# ---------------------------------------------------------------------------

class _Lockstep:
    """An in-process all_gather for ``world`` threads (fails, never hangs)."""

    def __init__(self, world: int, timeout: float = 30.0) -> None:
        self.world = world
        self.rows: list[list[int] | None] = [None] * world
        self.barrier = threading.Barrier(world, timeout=timeout)
        self.calls = [0] * world

    def coordinator(self, rank: int) -> co.ParkCoordinator:
        def gather(row: list[int]) -> list[list[int]]:
            self.calls[rank] += 1
            self.rows[rank] = list(row)
            self.barrier.wait()
            out = [list(r or []) for r in self.rows]
            self.barrier.wait()
            return out

        return co.ParkCoordinator(gather, rank=rank, world=self.world)


#: MLX streams are THREAD-LOCAL in this build: a lazy array built on one thread
#: cannot be evaluated on another ("There is no Stream(cpu, 0) in current
#: thread"). So each simulated rank owns ONE persistent worker thread and does
#: ALL of its MLX work there -- exactly like a real rank's single runner thread.
_POOLS: dict[int, ThreadPoolExecutor] = {}


def _pool(rank: int) -> ThreadPoolExecutor:
    if rank not in _POOLS:
        _POOLS[rank] = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"rank{rank}")
        _POOLS[rank].submit(lambda: mx.set_default_device(mx.cpu)).result()
    return _POOLS[rank]


def _on(rank: int, fn: Callable[[], Any]) -> Any:
    return _pool(rank).submit(fn).result(timeout=60)


def _both(fns: list[Callable[[], Any]]) -> list[Any]:
    """Run fns[i] on rank i's thread, concurrently; fail (never hang) on a
    rank stuck in an exchange; re-raise any rank's error."""
    futs = [_pool(i).submit(fn) for i, fn in enumerate(fns)]
    return [f.result(timeout=60) for f in futs]


class _Rank:
    def __init__(self, root: Path, rank: int, ls: _Lockstep | None,
                 fingerprint: fp.BuildFingerprint | None = None) -> None:
        self.root = root
        self.rank = rank
        self.coord = ls.coordinator(rank) if ls is not None else None
        _on(rank, lambda: self._build(fingerprint))

    def _build(self, fingerprint: fp.BuildFingerprint | None) -> None:
        self.body = StubBody(initial_capacity=4, engram_layer_ids=(0,))
        self.store = pk.ParkedStore(
            self.body, None, root=self.root, conv_kw=CONV_KW, min_tokens=8,
            fingerprint=fingerprint or _fp(layout=f"rank{self.rank}"),
            coordinator=self.coord, rank=self.rank,
        )
        self.conv = self._conv()

    def _conv(self) -> s_.Conversation:
        sess = s_.Dsv41Sessions(self.body, None, max_seq_len=256, **CONV_KW)
        conv, _fed, _gen = run_turn(sess, IDS, key="A")
        return conv

    def gens(self) -> set[int]:
        out = set()
        for d in self.root.iterdir():
            if (d / pk.MANIFEST).is_file():
                parsed = pk._gen_from_name(d.name)
                if parsed is not None:
                    out.add(parsed[1])
        return out

    def drop_gen(self, gen: int) -> None:
        for d in list(self.root.iterdir()):
            parsed = pk._gen_from_name(d.name)
            if parsed is not None and parsed[1] == gen:
                pk.shutil.rmtree(d)


def _two(tmp_path: Path, **kw: Any) -> tuple[_Lockstep, _Rank, _Rank]:
    ls = _Lockstep(2)
    return ls, _Rank(tmp_path / "r0", 0, ls, **kw), _Rank(tmp_path / "r1", 1, ls)


def _park_both(a: _Rank, b: _Rank) -> list[bool]:
    return _both([lambda: a.store.park(a.conv, key="A"),
                  lambda: b.store.park(b.conv, key="A")])


def _restore_both(a: _Rank, b: _Rank) -> list[Any]:
    return _both([lambda: a.store.try_restore(IDS, key="A"),
                  lambda: b.store.try_restore(IDS, key="A")])


def test_park_agrees_identical_generation_and_rank_tags(tmp_path: Path) -> None:
    ls, a, b = _two(tmp_path)
    # Rank 1 already lags with a stray local gen 3 (e.g. a pre-G4 park): the
    # agreed N must be 4 on BOTH ranks, not 1 on rank 0.
    assert _park_both(a, b) == [True, True]
    assert a.gens() == b.gens() == {1}
    b.drop_gen(1)
    assert _on(1, lambda: b.store.write_generation(b.conv, 3, key="A")) is not None
    assert b.gens() == {3}
    assert _park_both(a, b) == [True, True]
    assert 4 in a.gens() and 4 in b.gens()
    for root, rank in ((a.root, 0), (b.root, 1)):
        for d in root.iterdir():
            if d.name.endswith(".bad"):
                continue
            assert pk._rank_from_name(d.name) == rank
            man = json.loads((d / pk.MANIFEST).read_text())
            assert man["rank"] == rank
    # Exactly two exchanges per park on each rank (rank-invariant count).
    assert ls.calls == [4, 4]


def test_ack_gated_prune_waits_for_lagging_rank(tmp_path: Path) -> None:
    ls, a, b = _two(tmp_path)
    assert _park_both(a, b) == [True, True]   # gen 1 everywhere
    assert _park_both(a, b) == [True, True]   # gen 2
    assert a.gens() == b.gens() == {1, 2}

    # Rank 1 fails to commit gen 3 (disk full): rank 0 must NOT prune gen 1,
    # because the common restorable set is still {1, 2}.
    real = pk.park_conversation
    tls = threading.local()

    def flaky(conv: Any, d: Any, **kw: Any) -> bool:
        if getattr(tls, "fail", False):
            return False
        return real(conv, d, **kw)

    pk.park_conversation = flaky  # type: ignore[assignment]
    try:
        def b_fail() -> bool:
            tls.fail = True
            return b.store.park(b.conv, key="A")

        res = _both([lambda: a.store.park(a.conv, key="A"), b_fail])
    finally:
        pk.park_conversation = real  # type: ignore[assignment]
    assert res == [True, False]
    assert a.gens() == {1, 2, 3}, "pruned a generation the lagging rank still needs"
    assert b.gens() == {1, 2}

    # A cold-restart restore now picks the highest COMMON gen (2), never 3.
    ls2 = _Lockstep(2)
    a2 = _Rank(a.root, 0, ls2)
    b2 = _Rank(b.root, 1, ls2)
    ra, rb = _restore_both(a2, b2)
    assert ra is not None and rb is not None
    actives = [
        {pk._gen_from_name(d.name)[1] for d in r.iterdir()
         if (d / pk.ACTIVE_MARKER).is_file()}
        for r in (a.root, b.root)
    ]
    assert actives == [{2}, {2}]

    # Once every rank commits gen 4 (N=4 agreed from max 3), both prune <= 2.
    assert _park_both(a2, b2) == [True, True]
    assert a.gens() == {3, 4}
    assert b.gens() == {4}


def test_t8_two_ranks_missing_gen_n_restores_n_minus_1(tmp_path: Path) -> None:
    _ls, a, b = _two(tmp_path)
    _park_both(a, b)
    _park_both(a, b)
    b.drop_gen(2)  # rank 1 lost gen N (e.g. crash before commit)
    ra, rb = _restore_both(a, b)
    assert ra is not None and rb is not None
    for root in (a.root, b.root):
        act = [pk._gen_from_name(d.name)[1] for d in root.iterdir()
               if (d / pk.ACTIVE_MARKER).is_file()]
        assert act == [1]


def test_t8_two_ranks_no_common_gen_all_cold(tmp_path: Path) -> None:
    _ls, a, b = _two(tmp_path)
    _park_both(a, b)
    _park_both(a, b)
    a.drop_gen(1)
    b.drop_gen(2)
    assert _restore_both(a, b) == [None, None]


def test_t8_two_ranks_one_empty_all_cold(tmp_path: Path) -> None:
    _ls, a, b = _two(tmp_path)
    _park_both(a, b)
    b.drop_gen(1)
    assert _restore_both(a, b) == [None, None]
    assert not any((d / pk.ACTIVE_MARKER).exists() for d in a.root.iterdir())


def test_t8_two_ranks_digest_mismatch_all_cold(tmp_path: Path) -> None:
    root_a = tmp_path / "r0"
    ls = _Lockstep(2)
    # Rank 0 parked under a different kv-code build than rank 1; each rank
    # admits its OWN park locally, but the build identities differ -> COLD.
    a = _Rank(root_a, 0, ls, fingerprint=_fp(layout="rank0", kv_code_fp="kvX"))
    b = _Rank(tmp_path / "r1", 1, ls)
    _park_both(a, b)
    assert a.gens() == b.gens() == {1}
    assert _restore_both(a, b) == [None, None]


def test_t8_two_ranks_corrupt_blob_on_one_rank_all_cold(tmp_path: Path) -> None:
    _ls, a, b = _two(tmp_path)
    _park_both(a, b)
    d = next(d for d in b.root.iterdir() if (d / pk.MANIFEST).is_file())
    (d / "win_kv.0.bin").write_bytes(b"short")  # passes admission, fails restore
    assert _restore_both(a, b) == [None, None]
    # Rank 1 quarantined its corrupt dir; rank 0's dir is intact and NOT active.
    assert d.with_name(d.name + ".bad").exists()
    assert not any((x / pk.ACTIVE_MARKER).exists() for x in a.root.iterdir())


def test_t8_rank_with_store_error_is_cold_not_hang(tmp_path: Path) -> None:
    _ls, a, b = _two(tmp_path)
    _park_both(a, b)

    def broken(*_a: Any, **_k: Any) -> Any:
        raise OSError("EIO")

    b.store.local_restore_candidates = broken  # type: ignore[method-assign]
    assert _restore_both(a, b) == [None, None]


def test_rank_with_parking_disabled_still_takes_part(tmp_path: Path) -> None:
    """A rank whose store is disabled must still reach both exchanges (else the
    peer would block): the cluster goes cold, nothing hangs."""
    ls, a, b = _two(tmp_path)
    _park_both(a, b)
    b.store.enabled = False
    assert _restore_both(a, b) == [None, None]
    assert ls.calls[0] == ls.calls[1]


def test_sessions_layer_without_store_still_exchanges(tmp_path: Path, monkeypatch) -> None:
    """The real call site: Dsv41Sessions on rank 1 with parking disabled by env
    still joins the restore exchange; both ranks open cold."""
    ls, a, _b = _two(tmp_path)
    _park_both(a, _b)
    sess_a = s_.Dsv41Sessions(a.body, None, max_seq_len=256, park_store=a.store,
                              coordinator=a.coord, rank=0, **CONV_KW)
    sess_b = s_.Dsv41Sessions(_b.body, None, max_seq_len=256, park_store=None,
                              coordinator=_b.coord, rank=1, **CONV_KW)
    sess_b._park_failed = True  # this rank's store "could not be built"
    ra, rb = _both([lambda: sess_a._restore_parked(IDS, key="A"),
                    lambda: sess_b._restore_parked(IDS, key="A")])
    assert ra is None and rb is None


def test_sessions_restore_through_get(tmp_path: Path) -> None:
    """End-to-end through ``Dsv41Sessions.get`` on both ranks."""
    ls, a, b = _two(tmp_path)
    _park_both(a, b)
    sessions = [
        s_.Dsv41Sessions(r.body, None, max_seq_len=256, park_store=r.store,
                         coordinator=r.coord, rank=i, **CONV_KW)
        for i, r in enumerate((a, b))
    ]
    convs = _both([lambda: sessions[0].get(IDS.tolist(), "A"),
                   lambda: sessions[1].get(IDS.tolist(), "A")])
    assert all(c.offset == a.conv.offset for c in convs)
    assert [s.stats["restored"] for s in sessions] == [1, 1]


# ---------------------------------------------------------------------------
# single rank / no coordinator: unchanged behaviour
# ---------------------------------------------------------------------------

def test_single_rank_no_coordinator_unchanged(tmp_path: Path) -> None:
    r = _Rank(tmp_path / "solo", 0, None)
    for _ in range(3):
        assert _on(0, lambda: r.store.park(r.conv, key="A")) is True
    assert r.gens() == {2, 3}  # local newest-two retention, as pre-G4
    conv = _on(0, lambda: r.store.try_restore(IDS, key="A"))
    assert conv is not None
    assert co.ParkCoordinator.from_group(None) is None


# ---------------------------------------------------------------------------
# backward-compatible dir names
# ---------------------------------------------------------------------------

def test_dir_name_parsing_old_and_new() -> None:
    assert pk._gen_from_name("abc123.gen7.deadbeef") == ("abc123", 7)
    assert pk._rank_from_name("abc123.gen7.deadbeef") is None
    assert pk._gen_from_name("abc123.gen7.r1.deadbeef") == ("abc123", 7)
    assert pk._rank_from_name("abc123.gen7.r1.deadbeef") == 1
    assert pk._gen_from_name("abc123.gen7.r1.deadbeef.tmp") is None
    assert pk._gen_from_name("abc123__deadbeef") is None
    name = pk._gen_dir_name("abc123", 4, 0)
    assert pk._gen_from_name(name) == ("abc123", 4) and pk._rank_from_name(name) == 0
    name = pk._gen_dir_name("abc123", 4, None)
    assert pk._gen_from_name(name) == ("abc123", 4) and pk._rank_from_name(name) is None


def test_old_name_generation_is_restorable_by_coordinated_path(tmp_path: Path) -> None:
    """A pre-G4 (rank-less) gen dir still takes part in the cluster decision."""
    _ls, a, b = _two(tmp_path)
    _park_both(a, b)
    for r in (a, b):
        d = next(d for d in r.root.iterdir() if (d / pk.MANIFEST).is_file())
        parsed = pk._gen_from_name(d.name)
        d.rename(d.with_name(f"{parsed[0]}.gen{parsed[1]}.0000abcd"))
    ra, rb = _restore_both(a, b)
    assert ra is not None and rb is not None


def test_t8_prefix_mode_two_ranks(tmp_path: Path) -> None:
    """No client key: candidates are LCP-matched; the same rule holds."""
    _ls, a, b = _two(tmp_path)
    _park_both(a, b)
    _park_both(a, b)
    longer = np.concatenate([IDS, np.arange(500, 510, dtype=np.int64)])
    ra, rb = _both([lambda: a.store.try_restore(longer, key=None),
                    lambda: b.store.try_restore(longer, key=None)])
    assert ra is not None and rb is not None
    b.drop_gen(2)
    b.drop_gen(1)
    assert _both([lambda: a.store.try_restore(longer, key=None),
                  lambda: b.store.try_restore(longer, key=None)]) == [None, None]


# ---------------------------------------------------------------------------
# audit fixes: A (never raise out of coordinated_park / _evict), B (int32
# overflow before the collective), C (foreign-rank dir must not be restored)
# ---------------------------------------------------------------------------

def _fake_dir(
    root: Path, name: str, *, prefix_key: str, rank: int | None,
    key: str | None = "A", gen: int = 1, created: Any = 1.0,
) -> Path:
    """A minimal but ADMISSIBLE generation dir (manifest only; key mode)."""
    d = root / name
    d.mkdir(parents=True)
    man = {
        "magic": pk.PARK_MAGIC,
        "version": pk.PARK_VERSION,
        "created": created,
        "key": key,
        "generation": gen,
        "prefix_key": prefix_key,
        "rank": rank,
        fp.FINGERPRINT_FIELD: _fp().as_dict(),
    }
    (d / pk.MANIFEST).write_text(json.dumps(man))
    return d


# -- FIX A ------------------------------------------------------------------

def test_coordinated_park_enforce_budget_raise_never_propagates(
    tmp_path: Path,
) -> None:
    """A rank-LOCAL ``enforce_budget`` error (corrupt manifest) after the two
    exchanges must NOT escape ``coordinated_park`` -- else this rank skips
    ``session.close()`` and the peer, already in the TP forward pass, hangs."""
    ls, a, b = _two(tmp_path)

    def boom(*_a: Any, **_k: Any) -> None:
        raise OSError("manifest unreadable")

    a.store.enforce_budget = boom  # type: ignore[method-assign]
    assert _park_both(a, b) == [True, True]  # returns the local commit status
    assert ls.calls == [2, 2]                # both exchanges still ran
    assert a.gens() == b.gens() == {1}       # the generation was committed


def test_enforce_budget_tolerates_non_numeric_created(tmp_path: Path) -> None:
    """Root cause A: ``float(man.get("created", 0.0))`` must not raise on a
    non-numeric field; the entry is treated as the OLDEST (evicted first)."""
    body = StubBody(initial_capacity=4, engram_layer_ids=(0,))
    store = pk.ParkedStore(body, None, root=tmp_path, conv_kw=CONV_KW,
                           min_tokens=8, fingerprint=_fp(), rank=None)
    key_hash = pk.prefix_hash(np.asarray(IDS, dtype=np.int64))
    d = _fake_dir(tmp_path, f"{key_hash}.gen1.0000abcd", prefix_key=key_hash,
                  rank=None)
    man = json.loads((d / pk.MANIFEST).read_text())
    man["created"] = "not-a-number"
    (d / pk.MANIFEST).write_text(json.dumps(man))

    store.enforce_budget()  # must NOT raise
    assert [e.created for e in store._usage_entries()] == [0.0]


def test_candidates_quarantine_non_utf8_manifest(tmp_path: Path) -> None:
    """Root cause A: a non-UTF-8 manifest must not raise out of a scan (the
    eviction hot path); it is quarantined as corrupt instead."""
    body = StubBody(initial_capacity=4, engram_layer_ids=(0,))
    store = pk.ParkedStore(body, None, root=tmp_path, conv_kw=CONV_KW,
                           min_tokens=8, fingerprint=_fp(), rank=None)
    bad = tmp_path / "deadbeef.gen1.0000abcd"
    bad.mkdir()
    (bad / pk.MANIFEST).write_bytes(b"\xff\xfe\x00\x80 not utf-8")

    assert list(store._candidates()) == []  # no raise
    assert bad.with_name("deadbeef.gen1.0000abcd.bad").exists()


def test_evict_guards_against_coordinated_park_raise(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    """Defense-in-depth A2: even if ``coordinated_park`` raised (it must not),
    ``_evict`` must not propagate and ``session.close()`` must still run."""
    from exo.worker.engines.mlx.dsv41 import park as parkmod

    _ls, a, b = _two(tmp_path)
    sessions = [
        s_.Dsv41Sessions(r.body, None, max_seq_len=256, park_store=r.store,
                         coordinator=r.coord, rank=i, **CONV_KW)
        for i, r in enumerate((a, b))
    ]

    def boom(*_a: Any, **_k: Any) -> bool:
        raise RuntimeError("injected coordinated_park failure")

    monkeypatch.setattr(parkmod, "coordinated_park", boom)
    closed: list[Any] = []
    orig_close = s_.Conversation.close

    def rec_close(self: Any) -> None:
        closed.append(self)
        orig_close(self)

    monkeypatch.setattr(s_.Conversation, "close", rec_close)
    _both([lambda: sessions[0]._evict(a.conv, key="A"),
           lambda: sessions[1]._evict(b.conv, key="A")])
    assert len(closed) == 2                                  # close() still ran
    assert [s.stats["evicted"] for s in sessions] == [1, 1]
    assert [s.stats["park_failed"] for s in sessions] == [1, 1]


# -- FIX B ------------------------------------------------------------------

def test_mx_all_gather_ints_never_raises_on_out_of_range(
    monkeypatch: Any,
) -> None:
    """``mx.array(..., int32)`` raises BEFORE the collective for a value >=
    2**31, on that rank only -- the peer then blocks. ``mx_all_gather_ints``
    must sanitize every value first so the exchange always completes."""
    from exo.worker.engines.mlx import utils_mlx as um

    # The hazard this fix closes (documented, empirically true).
    with pytest.raises(RuntimeError):
        mx.eval(mx.array([2**31], dtype=mx.int32))

    # group=None: the row is still sanitized (before any collective).
    assert um.mx_all_gather_ints([2**31, 2**40, -1, 7], None) == [[0, 0, 0, 7]]

    # With a group, the exchange still COMMITS with a corrupt value present.
    sent: list[list[int]] = []

    def fake_all_gather(arr: Any, group: Any = None, stream: Any = None) -> Any:
        sent.append(list(arr.tolist()))
        return arr

    monkeypatch.setattr(mx.distributed, "all_gather", fake_all_gather)

    class _G:
        def size(self) -> int:
            return 1

        def rank(self) -> int:
            return 0

    assert um.mx_all_gather_ints([2**31, 5], _G()) == [[0, 5]]
    assert sent == [[0, 5]]  # the sanitized row is what crossed the wire


def test_absurd_generation_dir_name_is_ignored(tmp_path: Path) -> None:
    """A corrupt ``gen<N>`` dir name (N > int32 max) is not a real generation:
    it must not reach the int32 park payload."""
    body = StubBody(initial_capacity=4, engram_layer_ids=(0,))
    store = pk.ParkedStore(body, None, root=tmp_path, conv_kw=CONV_KW,
                           min_tokens=8, fingerprint=_fp(), rank=0)
    name = "deadbeef.gen2147483648.r0.cafebabe"
    (tmp_path / name).mkdir()
    (tmp_path / name / pk.MANIFEST).write_text(
        json.dumps({"magic": pk.PARK_MAGIC, "version": pk.PARK_VERSION}))

    assert pk._gen_from_name(name) is None
    assert pk._gen_from_name("deadbeef.gen2147483647.r0.cafebabe") is not None
    assert store.max_generation("deadbeef") == 0
    assert store._generation_dirs("deadbeef") == []


# -- FIX C ------------------------------------------------------------------

def test_foreign_rank_dir_ignored_rankless_considered(tmp_path: Path) -> None:
    """A dir KNOWN to belong to another rank (by name ``r<rank>`` OR by a
    manifest ``rank``) is not restorable here; a rank-less pre-G4 dir is."""
    body = StubBody(initial_capacity=4, engram_layer_ids=(0,))
    store = pk.ParkedStore(body, None, root=tmp_path, conv_kw=CONV_KW,
                           min_tokens=8, fingerprint=_fp(), rank=0)
    key_hash = pk.prefix_hash(np.asarray(IDS, dtype=np.int64))
    # rank 1 by NAME, manifest silent -> refused on the name tag.
    _fake_dir(tmp_path, f"{key_hash}.gen1.r1.0000abcd", prefix_key=key_hash,
              rank=None, gen=1)
    # rank 1 by MANIFEST, name rank-less -> refused on the manifest tag.
    _fake_dir(tmp_path, f"{key_hash}.gen2.0000beef", prefix_key=key_hash,
              rank=1, gen=2)
    # rank-less name AND no manifest rank -> pre-G4, still considered.
    _fake_dir(tmp_path, f"{key_hash}.gen3.0000cafe", prefix_key=key_hash,
              rank=None, gen=3)

    cands = store.local_restore_candidates(IDS, key="A")
    assert [c.gen for c in cands] == [3]
    wk = pk.key_int(key_hash)
    assert store.find_generation(wk, 1) is None
    assert store.find_generation(wk, 2) is None
    assert store.find_generation(wk, 3) is not None
    assert store.has_committed(key_hash, 1) is False
    assert store.has_committed(key_hash, 2) is False
    assert store.has_committed(key_hash, 3) is True

    # A store that does NOT know its rank stays fully permissive.
    blind = pk.ParkedStore(body, None, root=tmp_path, conv_kw=CONV_KW,
                           min_tokens=8, fingerprint=_fp(), rank=None)
    assert {c.gen for c in blind.local_restore_candidates(IDS, key="A")} == {
        1, 2, 3}
