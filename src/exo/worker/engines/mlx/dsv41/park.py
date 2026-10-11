"""Park idle DSv4.1 conversations to SSD (Option E of the SSD-offload study).

WHAT THIS IS. ``Dsv41Sessions`` keeps a bounded LRU of live ``Conversation``s,
each holding a full ``ModelCache`` (~3.18 GiB at 1M context: window ring +
compressed KV + compressor carry + index keys + engram-id history) plus the
DSpark draft windows and the rewind checkpoints. Eviction used to call
``Conversation.close()``, which throws the whole cache away; this module
serializes it to a per-session directory instead, so a later request whose
prompt extends the parked conversation's token history can restore it and
prefill only the delta -- amortizing the restore against a full re-prefill,
which at 1M context is minutes.

FILE FORMAT (one directory per session, one file per buffer, plus a JSON
manifest; see the study's section 3 file list):

    <dir>/manifest.json              # magic + version, cache offset/capacity/
                                     # max_seq_len, token-id history key, per-
                                     # buffer dtype+shape, snapshot positions,
                                     # n_turns/base/gen + draft bookkeeping.
                                     # WRITTEN LAST -- the commit record.
    <dir>/ids.bin                    # int32 LE token-id history (LCP matching)
    <dir>/win_kv.<layer>.bin         # window ring, every layer
    <dir>/comp_kv.<src>.bin          # compressed KV latents, kv-source layers
    <dir>/index_k.<owner>.bin        # index keys, index-owner layers
    <dir>/carry.<src>.bin            # comp_state.kv_state || score_state (fp32)
    <dir>/engram_ids.bin             # int64 engram-id history (used rows)
    <dir>/draft.<stage>.bin          # live DSpark draft-window rings
    <dir>/snapshot.<pos>.ring.<layer>.bin
    <dir>/snapshot.<pos>.carrykv.<layer>.bin
    <dir>/snapshot.<pos>.carrysc.<layer>.bin
    <dir>/draftsnap.<pos>.<stage>.bin
    <dir>/anchor.<pos>.bin           # prompt-end anchor logits (exact-repeat)

Buffers use the ``disaggregated/adapter.py`` byte codec verbatim (bf16 through
a uint16 bitcast) and the ``mx.stream(mx.Device(mx.cpu))`` copy-to-host idiom.

INVARIANTS. (1) Only rows a live read can reach are persisted -- ``comp_kv`` /
``index_k`` to ``ceil(offset / ratio)`` latents, ``engram_ids`` to ``offset`` --
and the restore ZERO-FILLS everything past them. This is the poison invariant:
a stale nonzero row above a rewind point would corrupt a future read, and a
freshly grown buffer is all-zero, so the restored cache is byte-identical to a
freshly built one. (2) The prompt-end anchor logits (``Conversation._anchor_at``)
ARE persisted, so a post-restore exact-repeat prompt (zero new rows) still has an
anchor and does not hard-refuse. (3) Park/restore only ever run between turns
(never mid-flight, never for ``drop``/``close``/``cancel_all``, which stay
hard-discard) and only when the draft window's ``n_ctx`` equals the cache offset.

FAILURES NEVER RAISE. ``park_conversation`` returns ``False`` and removes its
temporary directory on any error; ``restore_conversation`` returns ``None`` on
any manifest/geometry/data mismatch, and the store moves the bad directory to
``<dir>.bad`` so the caller falls back to a cold prefill instead of mis-restoring.

BUILD FINGERPRINT (plan §G2). Every manifest written by a store carries the
store's build :class:`~exo.worker.engines.mlx.dsv41.fingerprint.BuildFingerprint`
(model / tokenizer / KV-code / MLX / layout). ``find`` / ``try_restore`` refuse
any candidate that does not match the current build (``park.skip reason=...``)
WITHOUT quarantining it -- the dir is valid, just foreign or old, so the LRU
reclaims it. Legacy manifests (no fingerprint, or a pre-v2 format) are refused
the same way. The fingerprint is computed ONCE, at store construction.

DURABILITY + GENERATIONS (plan §G5). Every file is fsynced; the manifest is
committed as ``manifest.json.tmp`` -> fsync -> rename -> fsync dir; the whole
generation is built under ``<name>.tmp/`` and renamed into place (fsync root),
so a crash leaves only a ``.tmp`` that the next store construction sweeps.
A generation dir is ``<prefix_hash>.gen<N>.<uuid8>`` with ``N`` starting at 1 and
the newest TWO generations retained per conversation (older ones pruned only
after the new generation commits). Restore is NON-CONSUMING: a restored
generation stays on disk, marked the ACTIVE BASE with an empty ``.active`` file,
so the conversation survives the NEXT restart too.
"""

from __future__ import annotations

import collections
import contextlib
import json
import os
import re
import shutil
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import mlx.core as mx
import numpy as np
from mlx_lm.models.deepseek_v41.session_cache import (
    Snapshot,
    common_prefix_len,
    prefix_hash,
)

from exo.worker.disaggregated.protocol import DType
from exo.worker.engines.mlx.disaggregated.adapter import (
    array_to_bytes,
    bytes_to_array,
    mx_dtype_to_str,
)
from exo.worker.engines.mlx.dsv41.fingerprint import (
    FINGERPRINT_FIELD,
    BuildFingerprint,
    allow_mlx_drift,
    allow_stale,
    check_manifest,
    compute_fingerprint,
)
from exo.worker.engines.mlx.dsv41.session import MIN_REUSE_TOKENS
from exo.worker.runner.bootstrap import logger

if TYPE_CHECKING:  # pragma: no cover - typing only (avoids the import cycle)
    from exo.worker.engines.mlx.dsv41.session import Conversation

#: Refusal reason for a pre-v2 manifest or one without a usable fingerprint.
_REASON_LEGACY = "legacy"

__all__ = [
    "PARK_MAGIC",
    "PARK_VERSION",
    "ParkedStore",
    "park_conversation",
    "park_safe",
    "restore_conversation",
]

#: Manifest magic + version. A mismatch refuses the restore (never guesses).
PARK_MAGIC = "DSV41-PARK"
#: Format version. v2 adds the build-fingerprint manifest field (§G2); a v1
#: manifest is refused as legacy (not restored, not quarantined).
PARK_VERSION = 2
MANIFEST = "manifest.json"
#: Empty marker file written into a generation dir after a successful restore
#: (the non-consuming ACTIVE BASE, §G5.5). Reclaimed later by LRU / generations.
ACTIVE_MARKER = ".active"
#: Suffix a corrupt generation is renamed to, so it is never retried.
_QUARANTINE_SUFFIX = ".bad"
#: Generations of one conversation retained on disk (the newest N, §G5.6).
_KEEP_GENERATIONS = 2

#: A generation directory: ``<prefix_hash>.gen<N>.<8 hex uuid>`` (§G5.6).
_GEN_DIR_RE = re.compile(
    r"^(?P<prefix>[0-9a-f]+)\.gen(?P<gen>\d+)\.(?P<uuid>[0-9a-f]{8})$"
)

ENV_ENABLE = "EXO_DSV41_PARK"
ENV_DIR = "EXO_DSV41_PARK_DIR"
ENV_MAX_GB = "EXO_DSV41_PARK_MAX_GB"
ENV_MIN_TOKENS = "EXO_DSV41_PARK_MIN_TOKENS"
ENV_MAX_COUNT = "EXO_DSV41_PARK_MAX_COUNT"

DEFAULT_PARK_DIR = "~/.exo/dsv41_park"
DEFAULT_MAX_GB = 12.0
#: Sessions shorter than this are closed, not parked: the restore cost (and the
#: disk it occupies) is not worth it below a multi-chunk prefill.
DEFAULT_MIN_TOKENS = 4096
#: Belt-and-braces count cap beside the byte budget, so a flood of tiny parks
#: cannot swamp the index scan.
DEFAULT_MAX_COUNT = 64

_ITEMSIZE = {"bfloat16": 2, "float16": 2, "float32": 4}


# ---------------------------------------------------------------------------
# small env / fs helpers
# ---------------------------------------------------------------------------

def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except ValueError:
        logger.warning(f"[DSV41] ignoring non-integer {name}={raw!r}; using {default}")
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        logger.warning(f"[DSV41] ignoring non-float {name}={raw!r}; using {default}")
        return float(default)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _numel(shape: "list[int] | tuple[int, ...]") -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n


def park_dir_size(d: Path) -> int:
    """Total bytes of a parked session directory's files."""
    total = 0
    try:
        for p in d.iterdir():
            if p.is_file():
                total += p.stat().st_size
    except OSError:  # pragma: no cover - racing eviction
        return total
    return total


def _is_tmp_name(name: str) -> bool:
    """Whether ``name`` is a park temp artifact (``*.tmp`` / ``*.tmp.*``)."""
    return name.endswith(".tmp") or ".tmp." in name


def _write_bytes_synced(path: Path, data: bytes) -> None:
    """Write ``data`` then fsync the fd: a crash must not leave a short file."""
    with open(path, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def _fsync_dir(d: Path) -> None:
    """fsync a directory fd so a create/rename inside it is durable.

    Best-effort: a platform or filesystem that refuses fsync on a directory must
    not break a park, so failures are swallowed (the data fsync on each file is
    the durability guarantee; this orders the directory entries).
    """
    try:
        fd = os.open(d, os.O_RDONLY)
    except OSError:  # pragma: no cover - dir vanished
        return
    with contextlib.suppress(OSError):  # pragma: no cover - fs may not fsync a dir
        os.fsync(fd)
    os.close(fd)


def _gen_from_name(name: str) -> "tuple[str, int] | None":
    """Parse ``<prefix_hash>.gen<N>.<uuid8>`` into ``(prefix_hash, N)``."""
    m = _GEN_DIR_RE.match(name)
    if m is None:
        return None
    return m.group("prefix"), int(m.group("gen"))


def _generation_of(d: Path, man: "dict[str, Any]") -> int:
    """The generation number of a parked dir: manifest field, else name, else 0."""
    gen = man.get("generation")
    if isinstance(gen, int):
        return gen
    parsed = _gen_from_name(d.name)
    return parsed[1] if parsed is not None else 0


# ---------------------------------------------------------------------------
# byte plumbing (adapter codec + the CPU-copy idiom)
# ---------------------------------------------------------------------------

def _to_cpu(a: mx.array) -> mx.array:
    """Materialize ``a`` as a committed CPU copy (never aliases the GPU buffer)."""
    with mx.stream(mx.Device(mx.cpu)):
        cpu = mx.array(a)
        mx.eval(cpu)
    return cpu


def _write_blob(d: Path, name: str, arr: mx.array) -> dict[str, Any]:
    """Write one array (fsynced) and return its manifest descriptor."""
    cpu = _to_cpu(arr)
    _write_bytes_synced(d / name, array_to_bytes(cpu))
    return {
        "file": name,
        "dtype": mx_dtype_to_str(cpu.dtype),
        "shape": [int(s) for s in cpu.shape],
    }


def _read_blob(d: Path, desc: dict[str, Any]) -> mx.array:
    """Read one array back, validating the byte length against shape/dtype."""
    dtype = cast(DType, str(desc["dtype"]))
    shape = tuple(int(s) for s in desc["shape"])
    data = (d / desc["file"]).read_bytes()
    if len(data) != _numel(shape) * _ITEMSIZE[dtype]:
        raise ValueError(
            f"park blob {desc['file']} has {len(data)} bytes, expected "
            f"{_numel(shape) * _ITEMSIZE[dtype]} for {shape}/{dtype}"
        )
    return bytes_to_array(data, shape, dtype)


def _write_framed(d: Path, name: str, a: mx.array, b: mx.array) -> dict[str, Any]:
    """Write two same-dtype arrays back-to-back (fsynced); descriptors in manifest."""
    a_cpu, b_cpu = _to_cpu(a), _to_cpu(b)
    _write_bytes_synced(d / name, array_to_bytes(a_cpu) + array_to_bytes(b_cpu))
    return {
        "file": name,
        "a": {"dtype": mx_dtype_to_str(a_cpu.dtype),
              "shape": [int(s) for s in a_cpu.shape]},
        "b": {"dtype": mx_dtype_to_str(b_cpu.dtype),
              "shape": [int(s) for s in b_cpu.shape]},
    }


def _read_framed(d: Path, desc: dict[str, Any]) -> tuple[mx.array, mx.array]:
    """Inverse of :func:`_write_framed`, validating the total byte length."""
    a_d, b_d = desc["a"], desc["b"]
    a_dt, b_dt = cast(DType, str(a_d["dtype"])), cast(DType, str(b_d["dtype"]))
    a_n = _numel(a_d["shape"]) * _ITEMSIZE[a_dt]
    b_n = _numel(b_d["shape"]) * _ITEMSIZE[b_dt]
    data = (d / desc["file"]).read_bytes()
    if len(data) != a_n + b_n:
        raise ValueError(f"park blob {desc['file']} truncated/short")
    a = bytes_to_array(data[:a_n], tuple(int(s) for s in a_d["shape"]), a_dt)
    b = bytes_to_array(data[a_n:], tuple(int(s) for s in b_d["shape"]), b_dt)
    return a, b


# ---------------------------------------------------------------------------
# safety predicate
# ---------------------------------------------------------------------------

def park_safe(conv: "Conversation") -> bool:
    """Whether ``conv`` may be serialized right now.

    Between turns only: no in-flight turn (``cancel`` semantics live on the live
    cache), and the draft window -- if any -- must hold exactly the cache's rows
    (``n_ctx == offset``), because a draft window that has drifted cannot be
    reconstructed from the body cache alone.
    """
    if conv.cache is None:
        return False
    if conv.cache.cache is None:
        return False
    if conv._inflight:
        return False
    if getattr(conv, "draft_state", None) is not None:
        try:
            if conv.draft_ctx() != conv.offset:
                return False
        except Exception:  # pragma: no cover - broken draft window
            return False
    return True


# ---------------------------------------------------------------------------
# serialize
# ---------------------------------------------------------------------------

def park_conversation(
    conv: "Conversation",
    dir: "str | Path",  # noqa: A002
    *,
    key: str | None = None,
    fingerprint: "BuildFingerprint | None" = None,
) -> bool:
    """Serialize ``conv``'s full cache state into session directory ``dir``.

    Returns ``True`` on a committed park (manifest written, directory renamed
    into place) and ``False`` on any failure -- never raises. The directory is
    built under ``<name>.tmp/`` (each file fsynced) and atomically renamed, so a
    crash leaves only a ``.tmp`` the next store construction sweeps.

    ``fingerprint`` -- the store's build fingerprint (computed once at store
    construction, never here) -- is written into the manifest under
    :data:`FINGERPRINT_FIELD` so a later build can refuse a foreign park.
    """
    d = Path(dir)
    tmp = d.with_name(f"{d.name}.tmp")
    try:
        if not park_safe(conv):
            return False
        if d.exists():  # never clobber an existing session
            return False
        mc = conv.cache.cache
        layers = list(mc.layers)
        offset = int(mc.offset)
        capacity = int(mc.capacity)
        tmp.mkdir(parents=True, exist_ok=False)

        ids = np.ascontiguousarray(
            np.asarray(conv.cache.tokens, dtype=np.int64).astype("<i4"))
        _write_bytes_synced(tmp / "ids.bin", ids.tobytes())

        gen_parsed = _gen_from_name(d.name)
        man: dict[str, Any] = {
            "magic": PARK_MAGIC,
            "version": PARK_VERSION,
            "created": time.time(),
            "key": key,
            # The build fingerprint (plan §G2): computed once by the store and
            # passed in; never computed here. None makes the dir "legacy".
            FINGERPRINT_FIELD: (
                fingerprint.as_dict() if fingerprint is not None else None
            ),
            "generation": gen_parsed[1] if gen_parsed is not None else 0,
            "offset": offset,
            "capacity": capacity,
            "max_seq_len": int(mc.max_seq_len),
            "n_layers": len(layers),
            "window": int(getattr(mc.args, "window_size", 0)),
            "bsz": int(mc.bsz),
            "token_count": int(ids.shape[0]),
            "prefix_key": prefix_hash(ids),
            "ids": {"file": "ids.bin", "dtype": "int32", "count": int(ids.shape[0])},
            "n_turns": int(conv.n_turns),
            "cache_turns": int(conv.cache.n_turns),
            "base": int(conv._base),
            "gen": [int(t) for t in conv._gen],
            "win_kv": [],
            "comp_kv": [],
            "index_k": [],
            "carry": [],
            "engram": None,
            "draft": [],
            "has_draft": conv.draft_state is not None,
            "snapshots": [],
            "draft_snaps": [],
        }

        for i, lc in enumerate(layers):
            man["win_kv"].append(_write_blob(tmp, f"win_kv.{i}.bin", lc.win_kv))

        for i, lc in enumerate(layers):
            ratio = max(int(lc.ratio or 0), 1)
            used = -(-min(offset, capacity) // ratio)
            if lc.comp_kv is not None:
                rows = min(used, int(lc.comp_kv.shape[1]))
                desc = _write_blob(tmp, f"comp_kv.{i}.bin", lc.comp_kv[:, :rows])
                desc["used_rows"] = rows
                desc["ratio"] = ratio
                man["comp_kv"].append(desc)
            else:
                man["comp_kv"].append(None)
            if lc.index_k is not None:
                rows = min(used, int(lc.index_k.shape[1]))
                desc = _write_blob(tmp, f"index_k.{i}.bin", lc.index_k[:, :rows])
                desc["used_rows"] = rows
                man["index_k"].append(desc)
            else:
                man["index_k"].append(None)
            cs = lc.comp_state
            if cs is not None:
                desc = _write_framed(tmp, f"carry.{i}.bin", cs.kv_state, cs.score_state)
                desc["m"] = int(offset % ratio)
                desc["ratio"] = ratio
                man["carry"].append(desc)
            else:
                man["carry"].append(None)

        if mc.engram_ids is not None:
            used = min(offset, int(mc.engram_ids.shape[1]))
            arr = np.ascontiguousarray(mc.engram_ids[:, :used], dtype=np.int64)
            _write_bytes_synced(tmp / "engram_ids.bin", arr.tobytes())
            man["engram"] = {
                "file": "engram_ids.bin",
                "dtype": "int64",
                "shape": [int(s) for s in mc.engram_ids.shape],
                "used_rows": used,
            }

        draft = conv.draft_state
        if draft is not None:
            for stage, w in enumerate(draft):
                desc = _write_blob(tmp, f"draft.{stage}.bin", w.win_kv)
                desc["n_ctx"] = int(w.n_ctx)
                man["draft"].append(desc)

        for pos, snap in conv.cache._snaps.items():
            entry: dict[str, Any] = {"pos": int(pos), "rings": [], "carries": []}
            for j, ring in enumerate(snap.rings):
                entry["rings"].append(
                    None if ring is None
                    else _write_blob(tmp, f"snapshot.{pos}.ring.{j}.bin", ring))
            for j, carry in enumerate(snap.carries):
                if carry is None:
                    entry["carries"].append(None)
                elif carry[0] is None:  # spec.snap's m==0 sentinel: (None, None, 0)
                    entry["carries"].append({"empty": True, "m": int(carry[2])})
                else:
                    kv_rows, sc_rows, m = carry
                    desc = _write_framed(
                        tmp, f"snapshot.{pos}.carry.{j}.bin", kv_rows, sc_rows)
                    desc["m"] = int(m)
                    entry["carries"].append(desc)
            man["snapshots"].append(entry)

        for pos, saved in conv._draft_snaps.items():
            entry = {"pos": int(pos), "windows": None}
            if saved is not None:
                entry["windows"] = [
                    dict(_write_blob(tmp, f"draftsnap.{pos}.{stage}.bin", kv),
                         n_ctx=int(n_ctx))
                    for stage, (kv, n_ctx) in enumerate(saved)
                ]
            man["draft_snaps"].append(entry)

        # Persist the prompt-end anchor logits: without them an exact resubmit
        # (zero new rows) would hard-refuse after a restore, whereas before
        # parking it fell back to a cold prefill. Small if unused, correct when
        # the client re-sends an identical prompt or re-requests after eviction.
        anchors: list[dict[str, Any]] = []
        for pos, logits in conv._anchor_at.items():
            anchors.append(dict(
                _write_blob(tmp, f"anchor.{pos}.bin", logits), pos=int(pos)))
        man["anchors"] = anchors

        # The manifest is the commit record: written LAST as
        # ``manifest.json.tmp`` -> fsync -> rename -> fsync the dir, then the
        # whole generation directory is renamed into place atomically and the
        # containing (root) directory fsynced (plan §G5.2/§G5.3). A crash at any
        # point leaves either a ``.tmp`` (swept at startup) or a fully committed
        # generation -- never a half-valid one.
        tmp_man = tmp / f"{MANIFEST}.tmp"
        _write_bytes_synced(tmp_man, json.dumps(man, indent=1).encode("utf-8"))
        os.replace(tmp_man, tmp / MANIFEST)
        _fsync_dir(tmp)
        os.replace(tmp, d)
        _fsync_dir(d.parent)
        return True
    except Exception as exc:  # noqa: BLE001 - a park must never break a turn
        logger.warning(f"[DSV41] park failed ({type(exc).__name__}: {exc}); discarding")
        shutil.rmtree(tmp, ignore_errors=True)
        return False


# ---------------------------------------------------------------------------
# restore
# ---------------------------------------------------------------------------

def _grow_to(mc: Any, target: int) -> None:
    """Grow a freshly built ``ModelCache`` to hold ``target`` positions.

    ``ensure_capacity`` sets ``capacity = min(max(target, 2*capacity), max_seq_len)``,
    so from the small initial capacity a single call lands on the parked
    capacity exactly; when the fresh cache is already at least that big it is
    left alone. (Only when ``capacity < target < 2*capacity`` can it overshoot,
    and the overshoot buffer is still semantically identical: the extra rows are
    zero.)
    """
    if int(mc.capacity) < int(target):
        mc.ensure_capacity(int(target))


def restore_conversation(
    dir: "str | Path",  # noqa: A002
    model: Any,
    head: Any | None,
    *,
    progress: Any = None,
    **conv_kw: Any,
) -> "Conversation | None":
    """Rebuild a ``Conversation`` from a parked session directory.

    Returns ``None`` on any manifest/geometry/data mismatch (the caller then
    falls back to a cold prefill); the bad directory is quarantined by the
    store. Never raises.
    """
    from exo.worker.engines.mlx.dsv41.session import Conversation

    d = Path(dir)
    try:
        man = json.loads((d / MANIFEST).read_text())
        if man.get("magic") != PARK_MAGIC or man.get("version") != PARK_VERSION:
            return None
        conv = Conversation(
            model, head, max_seq_len=int(man["max_seq_len"]),
            progress=progress, **conv_kw,
        )
        mc = conv.cache.cache
        if len(mc.layers) != int(man["n_layers"]):
            return None
        # Geometry guard: a window/bsz mismatch means a different model layout.
        if int(man["window"]) != int(getattr(mc.args, "window_size", 0)) or \
                int(man["bsz"]) != int(mc.bsz):
            return None
        _grow_to(mc, int(man["capacity"]))
        if int(mc.capacity) < int(man["capacity"]):
            return None
        offset = int(man["offset"])

        if len(man["win_kv"]) != len(mc.layers):
            return None
        for lc, desc in zip(mc.layers, man["win_kv"], strict=True):
            arr = _read_blob(d, desc)
            if tuple(arr.shape) != tuple(lc.win_kv.shape):
                return None
            lc.win_kv = mx.array(arr.astype(lc.win_kv.dtype))

        for i, lc in enumerate(mc.layers):
            if not _restore_latent(d, lc, man["comp_kv"][i], "comp_kv"):
                return None
            if not _restore_latent(d, lc, man["index_k"][i], "index_k"):
                return None
            if not _restore_carry(d, lc, man["carry"][i]):
                return None

        if not _restore_engram(d, mc, man["engram"]):
            return None

        raw = (d / man["ids"]["file"]).read_bytes()
        ids = np.frombuffer(raw, dtype="<i4").astype(np.int64)
        if ids.shape[0] != int(man["token_count"]) or ids.shape[0] != offset:
            return None

        # Commit the bookkeeping only after every buffer validated.
        mc.offset = offset
        conv.cache._ids = np.ascontiguousarray(ids)
        conv.cache.n_turns = int(man.get("cache_turns", 0))
        conv._base = int(man.get("base", 0))
        conv._gen = [int(t) for t in man.get("gen", [])]
        conv.n_turns = int(man.get("n_turns", 0))
        conv._inflight = False

        snaps: "collections.OrderedDict[int, Snapshot]" = collections.OrderedDict()
        for entry in man["snapshots"]:
            rings: list[Any] = []
            for j, rdesc in enumerate(entry["rings"]):
                if rdesc is None:
                    rings.append(None)
                else:
                    ring = _read_blob(d, rdesc)
                    if tuple(ring.shape) != tuple(mc.layers[j].win_kv.shape):
                        return None
                    rings.append(ring)
            carries: list[Any] = []
            for cdesc in entry["carries"]:
                if cdesc is None:
                    carries.append(None)
                elif cdesc.get("empty"):
                    carries.append((None, None, int(cdesc.get("m", 0))))
                else:
                    kv_rows, sc_rows = _read_framed(d, cdesc)
                    carries.append((kv_rows, sc_rows, int(cdesc["m"])))
            pos = int(entry["pos"])
            snaps[pos] = Snapshot(pos=pos, rings=rings, carries=carries)
        conv.cache._snaps = snaps

        draft_snaps: dict[int, Any] = {}
        for entry in man["draft_snaps"]:
            pos = int(entry["pos"])
            wins = entry["windows"]
            draft_snaps[pos] = (
                None if wins is None
                else [(_read_blob(d, w), int(w["n_ctx"])) for w in wins]
            )
        conv._draft_snaps = draft_snaps

        anchors: dict[int, Any] = {}
        for entry in man.get("anchors", []):
            anchors[int(entry["pos"])] = _read_blob(d, entry)
        conv._anchor_at = anchors

        if man.get("has_draft"):
            if head is None:
                return None
            windows = head.make_cache(1)
            for w, desc in zip(windows, man["draft"], strict=True):
                arr = _read_blob(d, desc)
                if tuple(arr.shape) != tuple(w.win_kv.shape):
                    return None
                w.win_kv = mx.array(arr.astype(w.win_kv.dtype))
                w.n_ctx = int(desc["n_ctx"])
            conv.draft_state = windows

        return conv
    except Exception as exc:  # noqa: BLE001 - a bad file must not break a turn
        logger.warning(
            f"[DSV41] restore failed from {d.name} ({type(exc).__name__}: {exc})")
        return None


def _restore_latent(d: Path, lc: Any, desc: Any, which: str) -> bool:
    """Restore one latent buffer (comp_kv/index_k) or confirm its absence."""
    dest = getattr(lc, which)
    if desc is None:
        return dest is None
    if dest is None:
        return False
    rows = min(int(desc["used_rows"]), int(dest.shape[1]))
    arr = _read_blob(d, desc)
    if arr.shape[0] != int(dest.shape[0]) or arr.shape[1] != rows or \
            arr.shape[2] != int(dest.shape[2]):
        return False
    # Fresh zeros then fill only the live rows: the poison invariant.
    buf = mx.zeros_like(dest)
    if rows:
        buf[:, :rows] = arr.astype(buf.dtype)
    setattr(lc, which, buf)
    return True


def _restore_carry(d: Path, lc: Any, desc: Any) -> bool:
    """Restore a compressor carry (full fp32 kv/score rows) or confirm absence."""
    cs = lc.comp_state
    if desc is None:
        return cs is None
    if cs is None:
        return False
    kv, sc = _read_framed(d, desc)
    if tuple(kv.shape) != tuple(cs.kv_state.shape) or \
            tuple(sc.shape) != tuple(cs.score_state.shape):
        return False
    cs.kv_state = mx.array(kv.astype(cs.kv_state.dtype))
    cs.score_state = mx.array(sc.astype(cs.score_state.dtype))
    return True


def _restore_engram(d: Path, mc: Any, desc: Any) -> bool:
    """Restore the engram-id history, zero-filling past the live rows."""
    if desc is None:
        return mc.engram_ids is None
    if mc.engram_ids is None:
        return False
    used = min(int(desc["used_rows"]), int(mc.engram_ids.shape[1]))
    raw = (d / desc["file"]).read_bytes()
    if len(raw) != used * 8:
        return False
    new = np.zeros((int(mc.engram_ids.shape[0]), int(mc.engram_ids.shape[1])),
                   dtype=np.int64)
    if used:
        new[:, :used] = np.frombuffer(raw, dtype=np.int64)
    mc.engram_ids = new
    return True


# ---------------------------------------------------------------------------
# the store: index, LCP match, LRU budget
# ---------------------------------------------------------------------------

class _Usage(NamedTuple):
    """One parked generation's size + restore eligibility (for eviction order)."""

    created: float
    dir: Path
    size: int
    foreign: bool


class ParkedStore:
    """The on-disk set of parked sessions + the policy around it.

    Owns the root directory (``EXO_DSV41_PARK_DIR``, default
    ``~/.exo/dsv41_park``), the enable gate (``EXO_DSV41_PARK``), the min-token
    policy (``EXO_DSV41_PARK_MIN_TOKENS``) and the LRU budget
    (``EXO_DSV41_PARK_MAX_GB`` by summed file size + a count cap). Holds the
    model/head/construction kwargs so it can rebuild a ``Conversation`` on
    restore.

    Also owns (plan §G2) the build fingerprint -- computed ONCE at construction
    and stamped into every manifest this store writes, and used to refuse any
    foreign/legacy candidate on restore -- and (plan §G5) the durability
    protocol: startup ``.tmp`` sweep, per-generation atomic commit with fsync,
    two-generation retention, and a non-consuming restore that leaves the
    restored generation on disk as the ACTIVE BASE.
    """

    def __init__(
        self,
        model: Any,
        head: Any | None,
        *,
        root: "str | Path | None" = None,
        conv_kw: dict[str, Any] | None = None,
        progress: Any = None,
        enabled: bool | None = None,
        max_gb: float | None = None,
        min_tokens: int | None = None,
        max_count: int | None = None,
        fingerprint: "BuildFingerprint | None" = None,
    ) -> None:
        self.model = model
        self.head = head
        self.conv_kw = dict(conv_kw or {})
        self.progress = progress
        env_root = os.environ.get(ENV_DIR)
        self.root = Path(
            root if root is not None else (env_root or DEFAULT_PARK_DIR)
        ).expanduser()
        self.enabled = (
            _env_flag(ENV_ENABLE, True) if enabled is None else bool(enabled)
        )
        # §G5.4: sweep stale temp artifacts BEFORE any scan, so a crash mid-park
        # cannot leave a partial generation for this run to pick up.
        self._sweep_tmp()
        # §G2: the build fingerprint, computed ONCE here and reused for every
        # manifest write and every refusal. Never recomputed per-park. When the
        # caller provides none it degrades from whatever the store has (its
        # model); ``compute_fingerprint`` never raises.
        self._fingerprint: BuildFingerprint = (
            fingerprint
            if fingerprint is not None
            else compute_fingerprint(model=model)
        )
        gb = max_gb if max_gb is not None else _env_float(ENV_MAX_GB, DEFAULT_MAX_GB)
        self.max_bytes = int(max(0.0, gb) * 1_000_000_000)
        self.min_tokens = (
            int(min_tokens) if min_tokens is not None
            else _env_int(ENV_MIN_TOKENS, DEFAULT_MIN_TOKENS)
        )
        self.max_count = (
            int(max_count) if max_count is not None
            else _env_int(ENV_MAX_COUNT, DEFAULT_MAX_COUNT)
        )

    # -- internals --------------------------------------------------------

    def _sweep_tmp(self) -> None:
        """Delete stale ``*.tmp`` / ``*.tmp.*`` dirs and stray ``*.tmp`` files.

        The crash-atomicity backstop (plan §G5.4): a SIGKILL mid-park leaves
        ``<name>.tmp/`` (and possibly ``manifest.json.tmp`` inside a committed
        dir); none of those is a valid generation, so they are removed before
        the first scan.
        """
        if not self.root.is_dir():
            return
        for child in sorted(self.root.iterdir()):
            if not _is_tmp_name(child.name):
                continue
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                with contextlib.suppress(OSError):  # pragma: no cover - racing removal
                    child.unlink()

    def _candidates(self) -> Iterator[tuple[Path, dict[str, Any]]]:
        """Every directory under the root that holds a valid-magic manifest.

        A quarantined dir (``*.bad``) is never a candidate -- quarantine
        renames, it does not delete, so without this it would be re-scanned and
        re-quarantined forever. A directory whose manifest is unreadable or not
        a JSON object is corrupt and quarantined here. A dir with no manifest at
        all (or a non-park magic) is ignored.
        """
        if not self.root.is_dir():
            return
        for child in sorted(self.root.iterdir()):
            if not child.is_dir():
                continue
            if child.name.endswith(_QUARANTINE_SUFFIX):
                continue
            mpath = child / MANIFEST
            if not mpath.is_file():
                continue
            try:
                raw = mpath.read_text()
            except OSError:  # pragma: no cover - racing removal
                continue
            try:
                man = json.loads(raw)
            except ValueError:
                self._quarantine(child)  # garbage manifest: never retried
                continue
            if not isinstance(man, dict):
                self._quarantine(child)
                continue
            if man.get("magic") != PARK_MAGIC:
                continue
            yield child, man

    @staticmethod
    def _load_ids(d: Path, man: dict[str, Any]) -> np.ndarray:
        raw = (d / man["ids"]["file"]).read_bytes()
        return np.frombuffer(raw, dtype="<i4").astype(np.int64)

    def _quarantine(self, d: Path) -> None:
        """Move a bad session aside so it is never retried."""
        bad = d.with_name(f"{d.name}.bad")
        shutil.rmtree(bad, ignore_errors=True)
        try:
            os.replace(d, bad)
        except OSError:  # pragma: no cover - already gone
            shutil.rmtree(d, ignore_errors=True)

    def remove(self, d: Path) -> None:
        shutil.rmtree(d, ignore_errors=True)

    # -- policy -----------------------------------------------------------

    def should_park(self, conv: "Conversation") -> bool:
        """Whether ``conv`` is a parking candidate under the store's policy."""
        if not self.enabled or not park_safe(conv):
            return False
        try:
            return len(conv.tokens) >= self.min_tokens
        except Exception:  # pragma: no cover - broken conversation
            return False

    # -- fingerprint / version admission (plan §G2) -----------------------

    def _admit(self, man: dict[str, Any]) -> tuple[bool, str | None]:
        """Whether a manifest may be restored: version gate + fingerprint gate.

        A pre-v2 (or unparseable-version) manifest is refused as ``legacy``
        UNCONDITIONALLY -- it has no fingerprint the policy can trust, and
        ``restore_conversation`` refuses it too, so admitting it would only
        quarantine a valid-by-format-but-old dir (the audit's 44 ``key:null``
        dirs). Otherwise the fingerprint policy applies via ``check_manifest``
        (honouring the env escape hatches). Pure: it never logs or mutates, and
        never raises.
        """
        try:
            if int(man.get("version", -1)) != PARK_VERSION:
                return False, _REASON_LEGACY
            return check_manifest(
                man,
                self._fingerprint,
                allow_stale=allow_stale(),
                allow_mlx_drift=allow_mlx_drift(),
            )
        except Exception:  # noqa: BLE001 - a refusal must never raise
            return False, _REASON_LEGACY

    # -- generation bookkeeping (plan §G5.6) ------------------------------

    def _generation_dirs(self, prefix_hash: str) -> list[tuple[int, Path]]:
        """``(generation, dir)`` for every valid generation of one conversation."""
        out: list[tuple[int, Path]] = []
        for d, _man in self._candidates():
            parsed = _gen_from_name(d.name)
            if parsed is None or parsed[0] != prefix_hash:
                continue
            out.append((parsed[1], d))
        return out

    def _next_generation(self, key_hash: str) -> int:
        """The next generation number: max existing for ``key_hash`` + 1 (or 1)."""
        gens = [g for g, _d in self._generation_dirs(key_hash)]
        return (max(gens) + 1) if gens else 1

    def _prune_generations(
        self, prefix_hash: str, *, keep: int, keep_dir: Path | None = None
    ) -> None:
        """Delete all but the newest ``keep`` generations of one conversation.

        Called only AFTER a new generation has committed (plan §G5.6), so the
        conversation is never left without a valid generation on disk.
        ``keep_dir`` (the just-written generation) is never deleted.
        """
        gens = sorted(self._generation_dirs(prefix_hash), key=lambda t: t[0])
        doomed = gens[:-keep] if keep > 0 else gens
        for _gen, d in doomed:
            if keep_dir is not None and d == keep_dir:
                continue
            self.remove(d)

    # -- LRU / budget (plan §G5.7, §G5.8) ---------------------------------

    def _usage_entries(self) -> list[_Usage]:
        """Size + restore eligibility of every parked generation."""
        out: list[_Usage] = []
        for d, man in self._candidates():
            ok, _reason = self._admit(man)
            out.append(_Usage(
                created=float(man.get("created", 0.0)),
                dir=d,
                size=park_dir_size(d),
                foreign=not ok,
            ))
        return out

    @staticmethod
    def _evict_key(entry: _Usage) -> tuple[int, float]:
        """Sort key: foreign/legacy dirs FIRST, then oldest-first (LRU)."""
        return (0 if entry.foreign else 1, entry.created)

    def _reserve_headroom(self, *, protect_prefix: str | None = None) -> None:
        """Evict until free space can hold a new generation beside its old one.

        Headroom = 2 x (largest parked generation), so the new generation can be
        written BEFORE the one it supersedes is deleted (plan §G5.7). Eviction
        order is foreign/legacy first, then oldest. A generation belonging to
        ``protect_prefix`` (the conversation being re-parked) is never evicted
        here -- its old generations are pruned only after the new one commits.
        """
        entries = self._usage_entries()
        if not entries:
            return
        reserve = 2 * max(e.size for e in entries)
        total = sum(e.size for e in entries)
        for entry in sorted(entries, key=self._evict_key):
            if total + reserve <= self.max_bytes:
                break
            if protect_prefix is not None:
                parsed = _gen_from_name(entry.dir.name)
                if parsed is not None and parsed[0] == protect_prefix:
                    continue
            self.remove(entry.dir)
            total -= entry.size

    def enforce_budget(self, *, keep: Path | None = None) -> None:
        """Evict until under the byte and count caps (plan §G5.8).

        Foreign-fingerprint and legacy dirs go FIRST (this build can never
        restore them), then plain LRU (oldest first). ``keep`` -- the just
        written generation -- is never evicted.
        """
        entries = self._usage_entries()
        total = sum(e.size for e in entries)
        count = len(entries)
        for entry in sorted(entries, key=self._evict_key):
            if total <= self.max_bytes and count <= self.max_count:
                break
            if keep is not None and entry.dir == keep:
                continue
            self.remove(entry.dir)
            total -= entry.size
            count -= 1

    # -- park / find / restore --------------------------------------------

    def park(self, conv: "Conversation", *, key: str | None = None) -> bool:
        """Serialize ``conv`` as a new generation and enforce the disk budget.

        Named ``<prefix_hash>.gen<N>.<uuid8>`` with ``N`` = max existing + 1
        (plan §G5.6). Headroom is reserved before the write (§G5.7); older
        generations are pruned, and the budget enforced, only after the new
        generation has committed. Never raises.
        """
        try:
            if not self.should_park(conv):
                return False
            self.root.mkdir(parents=True, exist_ok=True)
            key_hash = prefix_hash(np.asarray(conv.tokens, dtype=np.int64))
            gen = self._next_generation(key_hash)
            d = self.root / f"{key_hash}.gen{gen}.{uuid.uuid4().hex[:8]}"
            # Reserve room for the new generation BEFORE writing it, so the old
            # one is never deleted to make space for a write that could fail.
            self._reserve_headroom(protect_prefix=key_hash)
            if not park_conversation(
                conv, d, key=key, fingerprint=self._fingerprint
            ):
                return False
            # New generation is on disk: now it is safe to prune the older ones.
            self._prune_generations(
                key_hash, keep=_KEEP_GENERATIONS, keep_dir=d)
            self.enforce_budget(keep=d)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[DSV41] park store error: {type(exc).__name__}: {exc}")
            return False

    def find(
        self, ids: Any, *, key: str | None = None
    ) -> tuple[Path, int] | None:
        """Best RESTORABLE parked session for ``ids``.

        With ``key`` (a client conversation id), the manifest key must match
        exactly and no prefix threshold applies. Without it, the longest common
        token prefix wins, and it must reach :data:`MIN_REUSE_TOKENS`. Ties on
        the same conversation are broken by the HIGHEST generation.

        A candidate the current build may not restore (foreign fingerprint or a
        legacy/pre-v2 manifest) is SKIPPED -- logged ``park.skip reason=<reason>``
        and left on disk for the LRU to reclaim, never quarantined (the dir is
        valid, just foreign/old).
        """
        ids_arr = np.asarray(ids, dtype=np.int64)
        best_dir, best_lcp, best_gen = None, 0, -1
        for d, man in self._candidates():
            ok, reason = self._admit(man)
            if not ok:
                logger.info(f"park.skip reason={reason}")
                continue
            gen = _generation_of(d, man)
            if key is not None:
                if man.get("key") == key and gen > best_gen:
                    best_dir, best_lcp, best_gen = d, int(ids_arr.shape[0]), gen
                continue
            try:
                lcp = common_prefix_len(ids_arr, self._load_ids(d, man))
            except (OSError, KeyError, ValueError):
                continue
            if lcp > best_lcp or (lcp == best_lcp and gen > best_gen):
                best_dir, best_lcp, best_gen = d, lcp, gen
        if best_dir is None or (key is None and best_lcp < MIN_REUSE_TOKENS):
            return None
        return best_dir, best_lcp

    def _mark_active(self, d: Path) -> None:
        """Mark a restored generation as the ACTIVE BASE (plan §G5.5).

        The non-consuming restore leaves the generation on disk; this empty,
        fsynced marker records that it is the live base so the LRU / generation
        policy can reclaim it later and an observer can see which generation a
        conversation is currently riding.
        """
        try:
            _write_bytes_synced(d / ACTIVE_MARKER, b"")
            _fsync_dir(d)
        except OSError:  # pragma: no cover - best-effort marker
            pass

    def try_restore(
        self, ids: Any, *, key: str | None = None
    ) -> "Conversation | None":
        """Find and restore a parked session; ``None`` on any miss.

        NON-CONSUMING (plan §G5.5): a successful restore leaves the generation
        on disk (marked active), so the conversation also survives the NEXT
        restart; the LRU / generation policy reclaims it. Only a CORRUPT dir is
        quarantined to ``<dir>.bad``.
        """
        if not self.enabled:
            return None
        try:
            hit = self.find(ids, key=key)
        except Exception:  # noqa: BLE001
            return None
        if hit is None:
            return None
        d, _lcp = hit
        conv = restore_conversation(
            d, self.model, self.head, progress=self.progress, **self.conv_kw)
        if conv is None:
            self._quarantine(d)
            return None
        self._mark_active(d)
        return conv
