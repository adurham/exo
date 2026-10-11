"""CPU-only tests for the DSv4.1 park fingerprint policy (§G2) and durability
hygiene (§G5).

No GPU, no checkpoint, no cluster, no network: the stub body over the fork's REAL
``ModelCache`` (``_stub_harness``) is parked to a ``tmp_path`` root and restored
into a fresh conversation, exactly as ``test_dsv41_park`` does. Pins the G2/G5
contract:

* **T1** boundary zero-fill: park/restore byte-identity at a mid-HCA-block offset
  and at an exact block-boundary offset, with the poison invariant (rows past the
  live rows are zero after a restore);
* **T2** the fingerprint gate: a matching fingerprint restores; each single-field
  mismatch (model/tokenizer/kv_code/layout) and an MLX-only mismatch are REFUSED
  with the dir left UNTOUCHED (``park.skip reason=...``); a legacy (no / pre-v2)
  manifest is refused; ``ALLOW_MLX_DRIFT`` and ``ALLOW_STALE`` override exactly
  what they should;
* **T3** hygiene: a truncated blob and a garbage manifest quarantine to ``.bad``;
  a dir with no manifest is ignored; a ``.tmp`` dir present before store
  construction is swept;
* **T4** crash atomicity: a child process is SIGKILLed mid-write of gen N+1, 50x;
  a fresh store always restores a clean generation and sweeps the ``.tmp``;
* **T6** cross-build gate: same ``kv_code_fp`` plus a changed
  model/tokenizer/layout is refused; all-equal is an exact restore;
* non-consuming restore + two-generation retention + foreign/legacy-first LRU.
"""

from __future__ import annotations

import contextlib
import json
import logging
import multiprocessing
import os
import random
import signal
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from exo.worker.engines.mlx.dsv41 import fingerprint as fp
from exo.worker.engines.mlx.dsv41 import park as pk
from exo.worker.engines.mlx.dsv41 import session as s_

from ._stub_harness import StubBody, fill_buffers, run_turn

CONV_KW: dict[str, Any] = dict(chunk=4, long_chunk=4, long_threshold=10**9)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _fp(**over: str) -> fp.BuildFingerprint:
    """A small reference build fingerprint with optional field overrides."""
    base = fp.BuildFingerprint(
        model_fp="model",
        tokenizer_fp="tok",
        kv_code_fp="kv",
        mlx_version="mlx-v1",
        layout_fp="layout",
    )
    return replace(base, **over) if over else base


def _np(a: Any) -> np.ndarray:
    """Exact numpy view (bf16 through uint16, so the bit pattern is compared)."""
    if a is None:
        return np.zeros(0)
    return np.asarray(a.view(mx.uint16) if a.dtype == mx.bfloat16 else a)


def _eq(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return (
        a.dtype == b.dtype
        and tuple(a.shape) == tuple(b.shape)
        and np.array_equal(_np(a), _np(b))
    )


def _live_rows(offset: int, ratio: int) -> int:
    return -(-offset // max(ratio, 1))


def _cmp_conv(a: s_.Conversation, b: s_.Conversation) -> None:
    """Every live buffer of two conversations must be bit-equal."""
    assert a.offset == b.offset
    assert a.draft_ctx() == b.draft_ctx()
    assert [int(t) for t in a.tokens] == [int(t) for t in b.tokens]
    ma, mb = a.cache.cache, b.cache.cache
    assert ma.offset == mb.offset
    assert (ma.engram_ids is None) == (mb.engram_ids is None)
    if ma.engram_ids is not None:
        assert np.array_equal(ma.engram_ids, mb.engram_ids)
    for li, (la, lb) in enumerate(zip(ma.layers, mb.layers, strict=True)):
        assert _eq(la.win_kv, lb.win_kv), f"win_kv {li}"
        assert _eq(la.comp_kv, lb.comp_kv), f"comp_kv {li}"
        assert _eq(la.index_k, lb.index_k), f"index_k {li}"
        if la.comp_state is None:
            assert lb.comp_state is None
        else:
            assert _eq(la.comp_state.kv_state, lb.comp_state.kv_state), f"carry kv {li}"
            assert _eq(la.comp_state.score_state, lb.comp_state.score_state), f"carry sc {li}"
    assert list(a.cache._snaps.keys()) == list(b.cache._snaps.keys())
    for pos in a.cache._snaps:
        sa, sb = a.cache._snaps[pos], b.cache._snaps[pos]
        for ring_a, ring_b in zip(sa.rings, sb.rings, strict=True):
            assert _eq(ring_a, ring_b), f"snapshot ring {pos}"
        for car_a, car_b in zip(sa.carries, sb.carries, strict=True):
            if car_a is None:
                assert car_b is None
            else:
                assert car_a[2] == car_b[2]
                assert _eq(car_a[0], car_b[0])
                assert _eq(car_a[1], car_b[1])
    assert list(a._anchor_at.keys()) == list(b._anchor_at.keys())
    for pos in a._anchor_at:
        assert _eq(a._anchor_at[pos], b._anchor_at[pos]), f"anchor {pos}"


def _conv(body: Any, ids: Any, key: str = "A") -> s_.Conversation:
    """A finished conversation over ``ids`` (deterministic; no park store)."""
    sess = s_.Dsv41Sessions(body, None, max_seq_len=256, **CONV_KW)
    conv, _fed, _gen = run_turn(sess, np.asarray(ids, dtype=np.int64), key=key)
    return conv


def _make_store(root: Any, body: Any, fingerprint: Any, **kw: Any) -> pk.ParkedStore:
    return pk.ParkedStore(
        body, None, root=root, conv_kw=CONV_KW, min_tokens=8,
        fingerprint=fingerprint, **kw,
    )


def _parked(root: Path) -> list[Path]:
    """Parked generation dirs (a manifest present), sorted by name."""
    return [
        d for d in sorted(root.iterdir())
        if d.is_dir() and (d / pk.MANIFEST).is_file()
    ]


def _gens(root: Path) -> dict[int, Path]:
    """generation number -> dir, for every gen-named parked dir."""
    out: dict[int, Path] = {}
    for d in _parked(root):
        m = pk._GEN_DIR_RE.match(d.name)
        if m is not None:
            out[int(m.group("gen"))] = d
    return out


class _Capture:
    """A logger stand-in that records messages AND forwards them to stdlib
    logging, so ``caplog`` can see what the park code emitted (the repo's loguru
    singleton is not wired into stdlib logging)."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def _emit(self, message: Any) -> None:
        self.messages.append(str(message))
        logging.getLogger("dsv41.park").info(str(message))

    def info(self, message: Any) -> None:
        self._emit(message)

    def warning(self, message: Any) -> None:
        self._emit(message)

    def debug(self, message: Any) -> None:
        self._emit(message)


# ---------------------------------------------------------------------------
# T1 -- boundary zero-fill (mid-HCA-block and exact block boundary)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_tokens,boundary", [(72, False), (71, True)])
def test_t1_boundary_zero_fill(
    tmp_path: Path, n_tokens: int, boundary: bool
) -> None:
    ids = np.arange(10, 10 + n_tokens, dtype=np.int64)

    def mk() -> StubBody:
        return StubBody(initial_capacity=4, engram_layer_ids=(0,))

    conv = _conv(mk(), ids)
    mc = conv.cache.cache
    offset = int(mc.offset)
    ratio = max(int(mc.layers[0].ratio or 0), 1)
    # One case lands mid-HCA-block, one exactly on a block boundary.
    assert (offset % ratio == 0) is boundary, (offset, ratio)

    fill_buffers(conv, seed=11)
    # Poison every row a live read can NOT reach; the park must drop them.
    for lc in mc.layers:
        r = max(int(lc.ratio or 0), 1)
        used = _live_rows(min(offset, int(mc.capacity)), r)
        if lc.comp_kv is not None and lc.comp_kv.shape[1] > used:
            lc.comp_kv[:, used:] = 5.0
        if lc.index_k is not None and lc.index_k.shape[1] > used:
            lc.index_k[:, used:] = 5.0
    assert mc.engram_ids is not None
    mc.engram_ids[:, offset:] = 7

    # The never-parked twin (same seed): identical live rows, zero tail.
    twin = _conv(mk(), ids)
    fill_buffers(twin, seed=11)

    d = tmp_path / "gen"
    assert pk.park_conversation(conv, d, fingerprint=_fp()) is True
    restored = pk.restore_conversation(d, mk(), None, **CONV_KW)
    assert restored is not None
    _cmp_conv(twin, restored)

    rmc = restored.cache.cache
    for lc in rmc.layers:
        r = max(int(lc.ratio or 0), 1)
        used = _live_rows(min(offset, int(rmc.capacity)), r)
        if lc.comp_kv is not None and lc.comp_kv.shape[1] > used:
            assert not np.any(_np(lc.comp_kv[:, used:])), "comp_kv tail not zero"
        if lc.index_k is not None and lc.index_k.shape[1] > used:
            assert not np.any(_np(lc.index_k[:, used:])), "index_k tail not zero"
    assert not np.any(rmc.engram_ids[:, offset:]), "engram tail not zero"


# ---------------------------------------------------------------------------
# T2 -- fingerprint gate
# ---------------------------------------------------------------------------

def _park_fp(root: Path, body: Any, fingerprint: Any, ids: np.ndarray) -> Path:
    store = _make_store(root, body, fingerprint)
    assert store.park(_conv(body, ids), key="A") is True
    dirs = _parked(root)
    assert len(dirs) == 1, dirs
    return dirs[0]


def test_t2_matching_fingerprint_restores(tmp_path: Path) -> None:
    root = tmp_path / "root"
    body = StubBody(initial_capacity=4, engram_layer_ids=(0,))
    ids = np.arange(10, 82, dtype=np.int64)
    d = _park_fp(root, body, _fp(), ids)
    conv = _make_store(root, body, _fp()).try_restore(ids, key="A")
    assert conv is not None and conv.offset > 0
    # Non-consuming: the dir is still here, marked active, not quarantined.
    assert d.exists() and (d / pk.ACTIVE_MARKER).is_file()
    assert not (d.with_name(d.name + ".bad")).exists()


@pytest.mark.parametrize(
    "field", ["model_fp", "tokenizer_fp", "kv_code_fp", "layout_fp"]
)
def test_t2_each_field_mismatch_refused_dir_untouched(
    tmp_path: Path,
    field: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    root = tmp_path / "root"
    body = StubBody(initial_capacity=4, engram_layer_ids=(0,))
    ids = np.arange(10, 82, dtype=np.int64)
    d = _park_fp(root, body, _fp(), ids)

    rec = _Capture()
    monkeypatch.setattr(pk, "logger", rec)
    caplog.set_level(logging.INFO)
    assert _make_store(root, body, _fp(**{field: "OTHER"})).try_restore(
        ids, key="A"
    ) is None
    # Refused, not quarantined: the dir is valid, just foreign.
    assert d.exists() and not (d.with_name(d.name + ".bad")).exists()
    assert any("park.skip reason=fp_mismatch" in m for m in rec.messages)
    assert "park.skip reason=fp_mismatch" in caplog.text


def test_t2_legacy_no_fingerprint_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    body = StubBody(initial_capacity=4, engram_layer_ids=(0,))
    ids = np.arange(10, 82, dtype=np.int64)
    # A fingerprintless park (version 2, no fingerprint dict) is "legacy".
    d = root / "legacy.gen1.deadbeef"
    assert pk.park_conversation(_conv(body, ids), d, key="A") is True

    rec = _Capture()
    monkeypatch.setattr(pk, "logger", rec)
    caplog.set_level(logging.INFO)
    assert _make_store(root, body, _fp()).try_restore(ids, key="A") is None
    assert d.exists() and not (d.with_name(d.name + ".bad")).exists()
    assert any("park.skip reason=legacy" in m for m in rec.messages)
    assert "park.skip reason=legacy" in caplog.text


def test_t2_pre_v2_manifest_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    root = tmp_path / "root"
    body = StubBody(initial_capacity=4, engram_layer_ids=(0,))
    ids = np.arange(10, 82, dtype=np.int64)
    d = _park_fp(root, body, _fp(), ids)
    # Downgrade the committed manifest to the pre-fingerprint format.
    man = json.loads((d / pk.MANIFEST).read_text())
    man["version"] = 1
    man.pop(pk.FINGERPRINT_FIELD, None)
    (d / pk.MANIFEST).write_text(json.dumps(man))

    rec = _Capture()
    monkeypatch.setattr(pk, "logger", rec)
    caplog.set_level(logging.INFO)
    assert _make_store(root, body, _fp()).try_restore(ids, key="A") is None
    assert d.exists() and not (d.with_name(d.name + ".bad")).exists()
    assert any("park.skip reason=legacy" in m for m in rec.messages)
    assert "park.skip reason=legacy" in caplog.text


def test_t2_mlx_drift_refused_unless_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    body = StubBody(initial_capacity=4, engram_layer_ids=(0,))
    ids = np.arange(10, 82, dtype=np.int64)
    monkeypatch.delenv(fp.ENV_ALLOW_MLX_DRIFT, raising=False)
    d = _park_fp(root, body, _fp(), ids)

    assert _make_store(root, body, _fp(mlx_version="OTHER")).try_restore(
        ids, key="A"
    ) is None
    assert d.exists()  # untouched

    monkeypatch.setenv(fp.ENV_ALLOW_MLX_DRIFT, "1")
    conv = _make_store(root, body, _fp(mlx_version="OTHER")).try_restore(
        ids, key="A"
    )
    assert conv is not None and conv.offset > 0


def test_t2_allow_stale_overrides_a_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    body = StubBody(initial_capacity=4, engram_layer_ids=(0,))
    ids = np.arange(10, 82, dtype=np.int64)
    d = _park_fp(root, body, _fp(), ids)

    monkeypatch.delenv(fp.ENV_ALLOW_STALE, raising=False)
    assert _make_store(root, body, _fp(model_fp="OTHER", kv_code_fp="OTHER")
                       ).try_restore(ids, key="A") is None

    monkeypatch.setenv(fp.ENV_ALLOW_STALE, "1")
    conv = _make_store(root, body, _fp(model_fp="OTHER", kv_code_fp="OTHER")
                       ).try_restore(ids, key="A")
    assert conv is not None and conv.offset > 0
    assert d.exists() and (d / pk.ACTIVE_MARKER).is_file()


# ---------------------------------------------------------------------------
# T3 -- hygiene
# ---------------------------------------------------------------------------

def test_t3_truncated_blob_quarantined(tmp_path: Path) -> None:
    root = tmp_path / "root"
    body = StubBody(initial_capacity=4, engram_layer_ids=(0,))
    ids = np.arange(10, 82, dtype=np.int64)
    d = _park_fp(root, body, _fp(), ids)
    blob = d / "comp_kv.0.bin"
    blob.write_bytes(blob.read_bytes()[:-4])  # truncate

    assert _make_store(root, body, _fp()).try_restore(ids, key="A") is None
    assert not d.exists()
    assert (d.with_name(d.name + ".bad")).exists()


def test_t3_missing_manifest_ignored(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "nomanifest.gen1.deadbeef").mkdir()  # no manifest.json
    body = StubBody(initial_capacity=4, engram_layer_ids=(0,))
    store = _make_store(root, body, _fp())
    assert list(store._candidates()) == []
    assert store.try_restore(np.arange(10, 82, dtype=np.int64), key="A") is None
    assert (root / "nomanifest.gen1.deadbeef").exists()  # ignored, untouched


def test_t3_tmp_dirs_swept_at_init(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "abc.gen1.deadbeef.tmp").mkdir()
    (root / "abc.gen1.deadbeef.tmp" / "part.bin").write_bytes(b"x")
    (root / "stray.tmp").write_text("junk")
    (root / "manifest.json.tmp.d").mkdir()
    body = StubBody(initial_capacity=4, engram_layer_ids=(0,))
    _make_store(root, body, _fp())  # construction sweeps
    assert not (root / "abc.gen1.deadbeef.tmp").exists()
    assert not (root / "stray.tmp").exists()
    assert not (root / "manifest.json.tmp.d").exists()


def test_t3_garbage_manifest_quarantined(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    d = root / "garbage.gen1.deadbeef"
    d.mkdir()
    (d / pk.MANIFEST).write_text("this is not json {")
    body = StubBody(initial_capacity=4, engram_layer_ids=(0,))
    store = _make_store(root, body, _fp())
    assert list(store._candidates()) == []
    assert not d.exists()
    assert (d.with_name(d.name + ".bad")).exists()


# ---------------------------------------------------------------------------
# T4 -- crash atomicity (SIGKILL mid-write of gen N+1)
# ---------------------------------------------------------------------------

#: Fixed fingerprint for the crash test: injected into BOTH the child store and
#: the parent's fresh store so the parked generation is admissible without either
#: side recomputing (and without depending on fork-shared compute state).
_T4_FP = fp.BuildFingerprint(
    model_fp="t4-model",
    tokenizer_fp="t4-tok",
    kv_code_fp="t4-kv",
    mlx_version="t4-mlx",
    layout_fp="t4-layout",
)


def _t4_child(root: Any, ids: Any, delay_s: float, conn: Any) -> None:
    """Commit gen 1, then start gen 2 with a per-file sleep so it can be killed."""
    try:
        body = StubBody(initial_capacity=4, engram_layer_ids=(0,))
        sess = s_.Dsv41Sessions(body, None, max_seq_len=256, **CONV_KW)
        conv, _fed, _gen = run_turn(sess, np.asarray(ids, dtype=np.int64), key="A")
        store = pk.ParkedStore(
            body, None, root=root, conv_kw=CONV_KW, min_tokens=8,
            fingerprint=_T4_FP,
        )
        ok1 = store.park(conv, key="A")  # gen 1: fast, must commit
        conn.send(("gen1", bool(ok1)))
        # Inject a per-file sleep so the parent has a window to kill mid-write.
        real = pk._write_blob

        def slow(d: Path, name: str, arr: Any) -> Any:
            time.sleep(delay_s)
            return real(d, name, arr)

        pk._write_blob = slow
        ok2 = store.park(conv, key="A")  # gen 2: slow, killed mid-write
        conn.send(("gen2", bool(ok2)))
    except BaseException as exc:  # noqa: BLE001 - report, never hang the parent
        with contextlib.suppress(Exception):  # pragma: no cover - pipe closed
            conn.send(("error", repr(exc)))
        raise


def test_t4_crash_atomicity_50x(tmp_path: Path) -> None:
    ids = np.arange(10, 82, dtype=np.int64)
    mk = lambda: StubBody(initial_capacity=4, engram_layer_ids=(0,))  # noqa: E731
    twin = _conv(mk(), ids)

    ctx = multiprocessing.get_context("spawn")
    # spawn, NOT fork: forking pytest's multi-threaded process segfaults the
    # child on an MLX allocation (the exo import graph pulls in torch et al.).
    # A spawned child starts a fresh interpreter with MLX initialized cleanly.
    # Children are run in small concurrent batches purely to amortize the ~2 s
    # per-child import; each still gets its own random kill delay.
    rng = random.Random(20261010)
    delay_s = 0.02  # ~10 writes -> a wide, deterministic kill window
    total, batch = 50, 10

    for start in range(0, total, batch):
        specs: list[list[Any]] = []
        for i in range(start, min(start + batch, total)):
            root = tmp_path / f"root{i}"
            root.mkdir()
            parent_conn, child_conn = ctx.Pipe(duplex=False)
            proc = ctx.Process(
                target=_t4_child, args=(root, ids, delay_s, child_conn))
            proc.start()
            child_conn.close()
            specs.append([i, root, parent_conn, proc, rng.uniform(0.01, 0.15)])

        # Every child commits gen 1 (importing + parking in parallel), then
        # starts gen 2 with the injected sleep.
        for i, _root, parent_conn, _proc, _d in specs:
            assert parent_conn.poll(120), f"iter {i}: child never reported gen1"
            msg = parent_conn.recv()
            assert msg == ("gen1", True), f"iter {i}: {msg!r}"

        # SIGKILL each child at its own random delay into the gen-2 write.
        t0 = time.perf_counter()
        for _i, _root, _conn, proc, delay in sorted(specs, key=lambda s: s[4]):
            remaining = delay - (time.perf_counter() - t0)
            if remaining > 0:
                time.sleep(remaining)
            if proc.is_alive() and proc.pid is not None:
                os.kill(proc.pid, signal.SIGKILL)

        for _i, _root, parent_conn, proc, _d in specs:
            proc.join(30)
            parent_conn.close()
            if proc.is_alive():  # pragma: no cover - safety net
                proc.kill()
                proc.join(30)

        # A fresh store sweeps any ``.tmp`` and must restore a clean generation.
        for i, root, _conn, _proc, _d in specs:
            fresh = pk.ParkedStore(
                mk(), None, root=root, conv_kw=CONV_KW, min_tokens=8,
                fingerprint=_T4_FP,
            )
            restored = fresh.try_restore(ids, key="A")
            assert restored is not None, f"iter {i}: no restorable generation"
            _cmp_conv(twin, restored)
            assert not any(x.name.endswith(".tmp") for x in root.iterdir()), (
                f"iter {i}: a .tmp survived the sweep"
            )
            # No partial generation is ever a candidate.
            for d in _parked(root):
                assert (d / pk.MANIFEST).is_file()


# ---------------------------------------------------------------------------
# T6 -- cross-build gate
# ---------------------------------------------------------------------------

def test_t6_same_fields_exact_restore(tmp_path: Path) -> None:
    root = tmp_path / "root"
    mk = lambda: StubBody(initial_capacity=4, engram_layer_ids=(0,))  # noqa: E731
    ids = np.arange(10, 82, dtype=np.int64)
    _park_fp(root, mk(), _fp(), ids)
    twin = _conv(mk(), ids)
    restored = _make_store(root, mk(), _fp()).try_restore(ids, key="A")
    assert restored is not None
    _cmp_conv(twin, restored)


@pytest.mark.parametrize(
    "over",
    [
        {"model_fp": "OTHER"},       # different build, same kv code
        {"tokenizer_fp": "OTHER"},   # same kv code, changed tokenizer
        {"layout_fp": "OTHER"},      # same kv code, changed layout
    ],
)
def test_t6_cross_build_refused(
    tmp_path: Path, over: dict[str, str]
) -> None:
    root = tmp_path / "root"
    body = StubBody(initial_capacity=4, engram_layer_ids=(0,))
    ids = np.arange(10, 82, dtype=np.int64)
    d = _park_fp(root, body, _fp(), ids)
    assert _make_store(root, body, _fp(**over)).try_restore(ids, key="A") is None
    assert d.exists() and not (d.with_name(d.name + ".bad")).exists()


# ---------------------------------------------------------------------------
# non-consuming restore + generations + LRU
# ---------------------------------------------------------------------------

def test_non_consuming_restore_leaves_the_base(tmp_path: Path) -> None:
    root = tmp_path / "root"
    body = StubBody(initial_capacity=4, engram_layer_ids=(0,))
    ids = np.arange(10, 82, dtype=np.int64)
    store = _make_store(root, body, _fp())
    assert store.park(_conv(body, ids), key="A") is True
    d = _parked(root)[0]

    first = store.try_restore(ids, key="A")
    assert first is not None
    # The dir is still there and marked active -- it survives the next restart.
    assert d.exists() and (d / pk.ACTIVE_MARKER).is_file()
    # A SECOND restore from the same store still works (non-consuming).
    second = store.try_restore(ids, key="A")
    assert second is not None and second.offset == first.offset


def test_two_generation_retention(tmp_path: Path) -> None:
    root = tmp_path / "root"
    body = StubBody(initial_capacity=4, engram_layer_ids=(0,))
    ids = np.arange(10, 82, dtype=np.int64)
    store = _make_store(root, body, _fp())
    conv = _conv(body, ids)

    assert store.park(conv, key="A") is True       # gen 1
    assert set(_gens(root)) == {1}
    assert store.try_restore(ids, key="A") is not None  # non-consuming base

    assert store.park(conv, key="A") is True       # gen 2 (1+2 retained)
    assert set(_gens(root)) == {1, 2}

    assert store.park(conv, key="A") is True       # gen 3 -> prune gen 1
    assert set(_gens(root)) == {2, 3}


def test_eviction_evicts_foreign_and_legacy_first(tmp_path: Path) -> None:
    root = tmp_path / "root"
    body = StubBody(initial_capacity=4, engram_layer_ids=(0,))
    ids = np.arange(10, 82, dtype=np.int64)
    live = _park_fp(root, body, _fp(), ids)  # L: restorable under F

    # X: foreign fingerprint; Y: legacy (no fingerprint). Both NEWER than L, so
    # plain LRU would evict L first -- foreign/legacy-first must win instead.
    import shutil

    def _clone(src: Path, dst: Path, created: float, mutate: Any) -> Path:
        shutil.copytree(src, dst)
        man = json.loads((dst / pk.MANIFEST).read_text())
        man["created"] = created
        mutate(man)
        (dst / pk.MANIFEST).write_text(json.dumps(man))
        return dst

    l_man = json.loads((live / pk.MANIFEST).read_text())
    l_man["created"] = 1_000.0
    (live / pk.MANIFEST).write_text(json.dumps(l_man))

    foreign = _clone(live, root / "foreign.gen1.0badc0de", 3_000.0,
                     lambda m: m.__setitem__(pk.FINGERPRINT_FIELD,
                                             _fp(model_fp="X").as_dict()))
    legacy = _clone(live, root / "legacy2.gen1.0badc0dd", 2_000.0,
                    lambda m: (m.__setitem__("version", 1),
                               m.pop(pk.FINGERPRINT_FIELD, None)))

    size = pk.park_dir_size(live)
    # Budget fits only the single live generation: some eviction MUST happen.
    store = _make_store(root, body, _fp(), max_gb=(1.5 * size) / 1e9)
    store.enforce_budget()

    assert live.exists(), "the restorable generation was wrongly evicted"
    assert not foreign.exists(), "a foreign dir outlived a restorable one"
    assert not legacy.exists(), "a legacy dir outlived a restorable one"


def test_park_never_evicts_the_just_written_generation(tmp_path: Path) -> None:
    root = tmp_path / "root"
    body = StubBody(initial_capacity=4, engram_layer_ids=(0,))
    ids = np.arange(10, 82, dtype=np.int64)
    # A budget far below a single park: enforce_budget must still keep the new
    # generation it just committed.
    store = _make_store(root, body, _fp(), max_gb=1e-9)
    assert store.park(_conv(body, ids), key="A") is True
    dirs = _parked(root)
    assert len(dirs) == 1 and (dirs[0] / pk.MANIFEST).is_file()
