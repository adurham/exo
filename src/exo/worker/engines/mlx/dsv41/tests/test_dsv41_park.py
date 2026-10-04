"""CPU-only tests for the DSv4.1 SSD park/restore serializer.

No GPU, no checkpoint, no cluster: a stub body over the fork's REAL
``ModelCache`` (``_stub_harness``) is parked to a ``tmp_path`` root and restored
into a fresh conversation. Pins:

* bit-exact round-trip (offset, token ids, every per-layer buffer, engram ids,
  carries, snapshots, draft windows) and that ``rewind`` still works after;
* the grow-boundary case -- a cache that crossed a capacity doubling restores
  as the never-parked twin and continues identically;
* the eviction policy -- K resident, the (K+1)-th parks the oldest, and a later
  matching request restores it (a real restore, not a cold open); sessions below
  ``EXO_DSV41_PARK_MIN_TOKENS`` are closed, not parked;
* failure injection -- truncated blob, bad magic, shape mismatch, unwritable
  root, a writer that raises mid-park -- all fall back cleanly, never raise, and
  a bad directory is quarantined to ``<dir>.bad``;
* the zero-fill (poison) invariant -- rows past the live rows are zero after a
  restore even when the source cache had nonzero values there.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from exo.worker.engines.mlx.dsv41 import park as pk
from exo.worker.engines.mlx.dsv41 import session as s_

from ._stub_harness import StubBody, StubHead, fill_buffers, run_turn

CONV_KW: dict[str, Any] = dict(chunk=4, long_chunk=4, long_threshold=10**9)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

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


def _new_store(body: Any, head: Any | None = None, **kw: Any) -> s_.Dsv41Sessions:
    return s_.Dsv41Sessions(body, head, max_seq_len=256, **CONV_KW, **kw)


def _park_store(root: Any, body: Any, head: Any | None = None, **kw: Any):
    return pk.ParkedStore(body, head, root=root, conv_kw=CONV_KW, **kw)


def _align_draft(conv: s_.Conversation) -> None:
    """Make the draft window reach ``offset`` (test-only: the stub's decode
    rounds bypass the draft feed the engine wires through ``rounds``)."""
    if conv.draft_state is None:
        return
    gap = conv.offset - conv.draft_ctx()
    if gap > 0:
        ids = list(conv.model.args.dspark_target_layer_ids)
        taps = {layer: mx.zeros((1, gap, 4)) for layer in ids}
        conv._feed_taps([taps])
        conv._checkpoint()


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


def _live_rows(offset: int, ratio: int) -> int:
    return -(-offset // max(ratio, 1))


# ---------------------------------------------------------------------------
# a. bit-exact round trip (buffer + draft + engram + snapshots + rewind)
# ---------------------------------------------------------------------------

def test_park_restore_roundtrip_bit_exact(tmp_path: Path) -> None:
    body = StubBody(initial_capacity=4, engram_layer_ids=(0,),
                    dspark_target_layer_ids=(0,))
    head = StubHead()
    store = _new_store(body, head)
    p1 = np.arange(10, 82, dtype=np.int64)
    conv, _fed, _gen = run_turn(store, p1)
    _align_draft(conv)
    fill_buffers(conv, seed=1)

    d = tmp_path / "sess"
    assert pk.park_conversation(conv, d) is True
    man = json.loads((d / "manifest.json").read_text())
    assert man["magic"] == pk.PARK_MAGIC and man["version"] == pk.PARK_VERSION
    assert man["offset"] == conv.offset and man["capacity"] == conv.cache.cache.capacity
    assert man["token_count"] == len(conv.tokens)
    assert man["has_draft"] is True

    conv2 = pk.restore_conversation(d, body, head, **CONV_KW)
    assert conv2 is not None
    _cmp_conv(conv, conv2)
    assert conv2.draft_state is not None
    for wa, wb in zip(conv.draft_state, conv2.draft_state, strict=True):
        assert _eq(wa.win_kv, wb.win_kv)
        assert int(wa.n_ctx) == int(wb.n_ctx)

    # A restored checkpoint must still rewind -- the whole snapshot set is on disk.
    pos = sorted(conv2.cache._snaps.keys())[0]
    dropped = conv2.cache.rewind(pos)
    assert dropped == conv.offset - pos and conv2.offset == pos


# ---------------------------------------------------------------------------
# b. growth boundary
# ---------------------------------------------------------------------------

def test_restore_across_a_growth_boundary_is_the_twin(tmp_path: Path) -> None:
    # initial_capacity 64, ~100 tokens: the cache crosses 64 -> 128 while live
    # (the small analog of the study's 65536 -> 131072 boundary).
    body = StubBody(initial_capacity=64, engram_layer_ids=(0,))
    store = _new_store(body)
    p1 = np.arange(10, 110, dtype=np.int64)  # 100 tokens
    conv, _f, _g = run_turn(store, p1)
    assert conv.cache.cache.capacity > 64, conv.cache.cache.capacity
    before_cap = conv.cache.cache.capacity

    d = tmp_path / "grown"
    assert pk.park_conversation(conv, d) is True
    restored = pk.restore_conversation(d, body, None, **CONV_KW)
    assert restored is not None
    assert restored.cache.cache.capacity == before_cap
    _cmp_conv(conv, restored)

    # A never-parked twin built with the same turns must match the restore.
    twin_store = _new_store(StubBody(initial_capacity=64, engram_layer_ids=(0,)))
    twin, _f2, _g2 = run_turn(twin_store, p1)
    _cmp_conv(twin, restored)

    # Continuing both feeds the restored (grown) buffers: still identical.
    p2 = np.concatenate([p1, np.asarray([90, 91, 92, 93], dtype=np.int64)])
    restored.prefill(p2)
    twin.prefill(p2)
    _cmp_conv(twin, restored)


# ---------------------------------------------------------------------------
# c. eviction policy: park the evicted, restore on a matching get
# ---------------------------------------------------------------------------

def test_eviction_parks_the_oldest_and_get_restores_it(tmp_path: Path) -> None:
    body = StubBody(initial_capacity=4, engram_layer_ids=(0,))
    ps = _park_store(tmp_path, body, min_tokens=8)
    store = _new_store(body, park_store=ps, max_sessions=2)

    prompts = {
        "A": np.arange(10, 82, dtype=np.int64),
        "B": np.arange(110, 182, dtype=np.int64),
        "C": np.arange(210, 282, dtype=np.int64),
    }
    offsets: dict[str, int] = {}
    tokens: dict[str, list[int]] = {}
    for name, p in prompts.items():
        conv, _f, _g = run_turn(store, p, key=name)
        offsets[name] = conv.offset
        tokens[name] = conv.tokens

    # A is the oldest and was evicted -> exactly one parked dir, one closed.
    dirs = [x for x in tmp_path.iterdir() if x.is_dir() and (x / pk.MANIFEST).is_file()]
    assert len(dirs) == 1, list(tmp_path.iterdir())
    assert store.stats["parked"] == 1 and store.stats["evicted"] == 1

    # A matching get must restore A's cache (offset > 0), not open a cold one.
    conv_a = store.get(prompts["A"].tolist(), "A")
    assert store.stats["restored"] == 1
    assert conv_a.offset == offsets["A"] > 0
    assert [int(t) for t in conv_a.tokens] == tokens["A"]
    assert not any((x / pk.MANIFEST).is_file() for x in tmp_path.iterdir())


def test_below_min_tokens_is_closed_not_parked(tmp_path: Path) -> None:
    body = StubBody(initial_capacity=4)
    ps = _park_store(tmp_path, body, min_tokens=10_000)  # nothing qualifies
    store = _new_store(body, park_store=ps, max_sessions=1)

    run_turn(store, np.arange(10, 42, dtype=np.int64), key="A")
    run_turn(store, np.arange(110, 142, dtype=np.int64), key="B")  # evicts A
    assert store.stats["evicted"] == 1
    assert store.stats.get("parked", 0) == 0
    assert not any((x / pk.MANIFEST).is_file() for x in tmp_path.iterdir())


def test_parking_disabled_by_env_closes_outright(tmp_path: Path) -> None:
    body = StubBody(initial_capacity=4)
    ps = _park_store(tmp_path, body, min_tokens=1)
    store = _new_store(body, park_store=ps, max_sessions=1)
    import os

    prev = os.environ.get(s_.PARK_ENV)
    os.environ[s_.PARK_ENV] = "0"
    try:
        run_turn(store, np.arange(10, 82, dtype=np.int64), key="A")
        run_turn(store, np.arange(110, 182, dtype=np.int64), key="B")
    finally:
        if prev is None:
            os.environ.pop(s_.PARK_ENV, None)
        else:
            os.environ[s_.PARK_ENV] = prev
    assert store.stats["evicted"] == 1
    assert store.stats.get("parked", 0) == 0
    assert not any((x / pk.MANIFEST).is_file() for x in tmp_path.iterdir())


# ---------------------------------------------------------------------------
# d. failure injection: never raise, always fall back, quarantine bad dirs
# ---------------------------------------------------------------------------

def _park_one(tmp_path: Path, body: Any) -> tuple[Path, s_.Dsv41Sessions]:
    store = _new_store(body)
    conv, _f, _g = run_turn(store, np.arange(10, 82, dtype=np.int64), key="A")
    d = tmp_path / "s"
    assert pk.park_conversation(conv, d, key="A") is True
    return d, store


def test_truncated_blob_and_bad_dirs_are_quarantined(tmp_path: Path) -> None:
    body = StubBody(initial_capacity=4)
    d, _ = _park_one(tmp_path, body)
    blob = d / "comp_kv.0.bin"
    blob.write_bytes(blob.read_bytes()[:-4])  # truncate

    assert pk.restore_conversation(d, body, None, **CONV_KW) is None
    ps = _park_store(tmp_path, body)
    # A matching query (the parked prompt's prefix) reaches try_restore, which
    # quarantines the corrupt directory instead of leaving it to be retried.
    assert ps.try_restore(np.arange(10, 82, dtype=np.int64)) is None
    assert not d.exists() and (d.with_name(d.name + ".bad")).exists()


def test_bad_magic_manifest_refuses_restore(tmp_path: Path) -> None:
    body = StubBody(initial_capacity=4)
    d, _ = _park_one(tmp_path, body)
    man = json.loads((d / pk.MANIFEST).read_text())
    man["magic"] = "NOT-A-PARK"
    (d / pk.MANIFEST).write_text(json.dumps(man))
    assert pk.restore_conversation(d, body, None, **CONV_KW) is None


def test_shape_mismatch_refuses_restore(tmp_path: Path) -> None:
    body = StubBody(initial_capacity=4)
    d, _ = _park_one(tmp_path, body)
    man = json.loads((d / pk.MANIFEST).read_text())
    man["win_kv"][0]["shape"][1] += 1
    (d / pk.MANIFEST).write_text(json.dumps(man))
    assert pk.restore_conversation(d, body, None, **CONV_KW) is None


def test_unwritable_park_root_falls_back_to_close(tmp_path: Path) -> None:
    body = StubBody(initial_capacity=4)
    root_file = tmp_path / "rootfile"
    root_file.write_text("not a directory")  # mkdir under it must fail
    ps = pk.ParkedStore(body, None, root=root_file, conv_kw=CONV_KW, min_tokens=1)
    store = _new_store(body, park_store=ps, max_sessions=1)
    run_turn(store, np.arange(10, 82, dtype=np.int64), key="A")
    run_turn(store, np.arange(110, 182, dtype=np.int64), key="B")  # evict A
    assert store.stats["evicted"] == 1 and store.stats.get("parked", 0) == 0


def test_writer_failure_mid_park_leaves_no_tmp_dir(tmp_path: Path, monkeypatch) -> None:
    body = StubBody(initial_capacity=4)
    store = _new_store(body)
    conv, _f, _g = run_turn(store, np.arange(10, 82, dtype=np.int64), key="A")

    real = pk._write_blob
    calls = {"n": 0}

    def boom(d, name, arr):
        calls["n"] += 1
        if calls["n"] > 2:  # fail partway through
            raise OSError("disk full (injected)")
        return real(d, name, arr)

    monkeypatch.setattr(pk, "_write_blob", boom)
    d = tmp_path / "s"
    assert pk.park_conversation(conv, d) is False  # never raises
    assert not d.exists()
    assert not any(x.name.startswith("s.tmp-") for x in tmp_path.iterdir())


def test_restore_missing_draft_head_refuses(tmp_path: Path) -> None:
    """A session parked WITH a draft window cannot restore without a head."""
    body = StubBody(initial_capacity=4, dspark_target_layer_ids=(0,))
    head = StubHead()
    store = _new_store(body, head)
    conv, _f, _g = run_turn(store, np.arange(10, 82, dtype=np.int64), key="A")
    _align_draft(conv)
    d = tmp_path / "s"
    assert pk.park_conversation(conv, d) is True
    assert pk.restore_conversation(d, body, None, **CONV_KW) is None  # no head


# ---------------------------------------------------------------------------
# e. zero-fill (poison) invariant
# ---------------------------------------------------------------------------

def test_restore_zero_fills_past_the_live_rows(tmp_path: Path) -> None:
    body = StubBody(initial_capacity=4, engram_layer_ids=(0,))
    store = _new_store(body)
    p1 = np.arange(10, 82, dtype=np.int64)
    conv, _f, _g = run_turn(store, p1)
    mc = conv.cache.cache
    offset = int(mc.offset)

    # Poison every row a read can NOT reach in the source; the park must drop
    # them (only live rows are persisted) and the restore must zero them.
    for lc in mc.layers:
        ratio = max(int(lc.ratio or 0), 1)
        used = _live_rows(min(offset, int(mc.capacity)), ratio)
        if lc.comp_kv is not None and lc.comp_kv.shape[1] > used:
            lc.comp_kv[:, used:] = 5.0
        if lc.index_k is not None and lc.index_k.shape[1] > used:
            lc.index_k[:, used:] = 5.0
    assert mc.engram_ids is not None
    mc.engram_ids[:, offset:] = 7
    # Sanity: the poison really is there in the source.
    poisoned = [lc for lc in mc.layers if lc.comp_kv is not None and lc.comp_kv.shape[1] > _live_rows(offset, max(int(lc.ratio or 0), 1))]
    assert poisoned, "test setup: expected at least one growable latent buffer"

    d = tmp_path / "s"
    assert pk.park_conversation(conv, d) is True
    restored = pk.restore_conversation(d, body, None, **CONV_KW)
    assert restored is not None
    rmc = restored.cache.cache
    for lc in rmc.layers:
        ratio = max(int(lc.ratio or 0), 1)
        used = _live_rows(min(offset, int(rmc.capacity)), ratio)
        if lc.comp_kv is not None and lc.comp_kv.shape[1] > used:
            assert not np.any(_np(lc.comp_kv[:, used:])), "comp_kv tail not zeroed"
        if lc.index_k is not None and lc.index_k.shape[1] > used:
            assert not np.any(_np(lc.index_k[:, used:])), "index_k tail not zeroed"
    assert rmc.engram_ids is not None
    assert not np.any(rmc.engram_ids[:, offset:]), "engram tail not zeroed"


# ---------------------------------------------------------------------------
# anchors survive: an exact-repeat prompt after a restore still anchors
# ---------------------------------------------------------------------------

def test_post_restore_exact_repeat_uses_the_restored_anchor(tmp_path: Path) -> None:
    """Parking now persists the prompt-end anchor logits, so the zero-new-rows
    exact-repeat path (a resubmitted identical prompt) still succeeds on a
    restored conversation -- it used to fall back to a cold prefill, and a naive
    restore without the anchor would hard-refuse."""
    body = StubBody(initial_capacity=4)
    store = _new_store(body)
    p1 = np.arange(10, 82, dtype=np.int64)
    conv, fed, _g = run_turn(store, p1)
    d = tmp_path / "s"
    assert pk.park_conversation(conv, d) is True
    restored = pk.restore_conversation(d, body, None, **CONV_KW)
    assert restored is not None

    # Same prompt, zero new rows: the restored anchor supplies the logits.
    out = restored.prefill(p1)
    assert out.prefill_tokens == 0 and out.reused_tokens == len(p1)
    assert _eq(out.anchor_logits, fed.anchor_logits)

    # A prompt that adds rows is the normal path and works on the restored cache.
    p2 = np.concatenate([p1, np.asarray([90, 91, 92], dtype=np.int64)])
    out2 = restored.prefill(p2)
    assert out2.prefill_tokens == 3 and out2.reused_tokens == len(p1)
