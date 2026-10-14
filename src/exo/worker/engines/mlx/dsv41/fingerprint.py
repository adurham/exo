"""DSv4.1 park build fingerprint + cross-build refusal policy (plan §G2).

WHY THIS EXISTS. The park store (``dsv41/park.py``) serializes a conversation's
full composite KV cache to disk and restores it after a process restart. Until
now it keyed only on a FORMAT version (``PARK_VERSION``) -- nothing tied the
parked bytes to the engine build that wrote them. The phase-0 audit
(``docs/dsv41-restart-survival-phase0-audit-2026-10-10.md``, A5) found that is
not hypothetical: ~12 of 15 observed restores crossed a build switch, so caches
written by one checkout were silently served by another, and exact-repeat
anchors meant sampling from old-build logits. A change in the model, the
tokenizer, the KV/forward math, MLX, or the tensor-parallel layout can therefore
serve stale bytes.

``compute_fingerprint`` captures five things ONCE at startup:

* ``model_fp``     -- config.json + weight index + tensor file names/sizes +
                      any quant config (NOT the weights themselves);
* ``tokenizer_fp`` -- tokenizer files + chat template;
* ``kv_code_fp``   -- sha256 over the EXPLICIT allowlist of KV-affecting sources
                      (:data:`KV_ALLOWLIST`), path + content, sorted;
* ``mlx_version``  -- the mlx.core version;
* ``layout_fp``    -- (world_size, rank, layer range, dtype, window, bsz).

``check_manifest`` decides, from a parked manifest plus the current
fingerprint, whether a restore is safe. The policy is strict on what can change
KV bytes and permissive elsewhere:

* any mismatch in model / tokenizer / layout / kv-code -> refuse ``fp_mismatch``;
* MLX-only mismatch -> refuse ``mlx_drift`` (``allow_mlx_drift`` overrides, a
  driver/MLX bump is often benign);
* a manifest with no (or a malformed) fingerprint -> refuse ``legacy`` (the
  audit's existing 44 ``key:null`` dirs);
* ``allow_stale`` (the ``EXO_DSV41_PARK_ALLOW_STALE`` debug escape hatch)
  overrides every refusal.

``check_manifest`` deliberately does NOT check magic or ``PARK_VERSION`` --
``park.py`` owns the format version and validates it itself.

NEVER RAISES. A park/restore must never break a turn because fingerprinting
hiccuped: with a missing input the fingerprint degrades to the canonical
representation of whatever IS available, and the missing case hashes
differently from a complete one so a degraded build cannot collide with a good
one. Every degradation is documented at its helper.

WHY AN EXPLICIT ALLOWLIST. The plan's risk R1 is the inverse of oversharing:
missing a KV-affecting file would let a stale park serve silently. So the
default is ALWAYS to allowlist a module -- an over-broad allowlist only costs a
needless cold start, whereas a missed file is silent staleness -- and only
genuinely neutral modules (no tensors, never on the forward path for a
restorable conversation) go in :data:`KV_NEUTRAL`, each with a one-line WHY.
:func:`allowlist_problems` is the CI guard: it imports the forward-pass entry
modules and reports any module in one of the three KV-affecting families that is
neither allowlisted nor declared neutral. An empty list means the allowlist is
complete.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx

__all__ = [
    "BuildFingerprint",
    "FINGERPRINT_FIELD",
    "KV_ALLOWLIST",
    "KV_NEUTRAL",
    "allow_mlx_drift",
    "allow_stale",
    "allowlist_problems",
    "check_manifest",
    "compute_fingerprint",
    "layout_fingerprint",
]

# ---------------------------------------------------------------------------
# manifest / env constants
# ---------------------------------------------------------------------------

#: The manifest key the park layer stores the fingerprint under. ``park.py``
#: embeds ``BuildFingerprint.as_dict()`` here; ``check_manifest`` reads it back.
FINGERPRINT_FIELD = "fingerprint"

#: Debug escape hatch: override every refusal (park.py passes this through).
ENV_ALLOW_STALE = "EXO_DSV41_PARK_ALLOW_STALE"
#: Override ONLY an mlx_version mismatch (e.g. a driver/MLX patch bump).
ENV_ALLOW_MLX_DRIFT = "EXO_DSV41_PARK_ALLOW_MLX_DRIFT"

#: The exact refusal reasons. ``check_manifest`` returns one of these (or None).
REASON_FP_MISMATCH = "fp_mismatch"
REASON_LEGACY = "legacy"
REASON_MLX_DRIFT = "mlx_drift"


# ---------------------------------------------------------------------------
# the allowlist: KV-affecting vs explicitly neutral
# ---------------------------------------------------------------------------
#
# Every module in the three KV-affecting package families is either in
# KV_ALLOWLIST (its bytes participate in kv_code_fp, so a change refuses old
# parks) or in KV_NEUTRAL (a change cannot invalidate a restorable KV). The
# guard in allowlist_problems() fails if a family module is in neither.

#: Modules whose source can affect KV bytes, forward math, or the park codec.
#: Every entry carries a one-line WHY.
KV_ALLOWLIST: frozenset[str] = frozenset(
    {
        # --- exo DSv4.1 engine (owns the composite cache + the park codec) -------
        "exo.worker.engines.mlx.dsv41.engine",  # drives the forward pass; owns the decode round
        "exo.worker.engines.mlx.dsv41.session",  # owns Conversation + the cache wrapper park serializes
        "exo.worker.engines.mlx.dsv41.park",  # the park/restore codec itself (file layout, zero-fill)
        "exo.worker.engines.mlx.dsv41.rounds",  # request gating + the decode step that grows KV
        "exo.worker.engines.mlx.dsv41.load",  # TP geometry + build wiring (rank/world, layer subset)
        "exo.worker.engines.mlx.dsv41.builder",  # installs the prefill patch + engine wiring
        "exo.worker.engines.mlx.dsv41.engram",  # engram token map -> engram_ids history (KV-affecting)
        # Degeneration loop-guard on the committed-token stream: it can terminate
        # a turn and roll the session back, so its presence changes what survives
        # to a park (allowlisted by policy default -- it is on the decode path).
        "exo.worker.engines.mlx.dsv41.degeneration",
        # Reasoning-budget limiter (guard PHASE 2): it cuts a round mid-batch and
        # rolls the cache back to the committed prefix, then forces the think-END
        # delimiter on the next round, so it is on the decode path and can change
        # what survives to a park (allowlisted by policy default, same rationale
        # as the loop-guard above).
        "exo.worker.engines.mlx.dsv41.reasoning_budget",
        # --- exo disaggregated wire (KV serialization/transport) ----------------
        "exo.worker.engines.mlx.disaggregated.adapter",  # the byte codec park uses verbatim (bf16 bitcast)
        "exo.worker.engines.mlx.disaggregated.client",  # decodes/injects composite KV from the wire
        "exo.worker.engines.mlx.disaggregated.serve",  # produces/serves composite KV to the peer
        # --- mlx-lm DSv4.1 model stack (the KV math itself) ---------------------
        "mlx_lm.models.deepseek_v41.attention",  # MLA attention; produces the KV rows
        "mlx_lm.models.deepseek_v41.sparse_attention",  # CSA/HCA sparse kernel selection
        "mlx_lm.models.deepseek_v41._gates",  # fence row-count gates (_FENCE_MIN_ROWS) read by indexer/sparse_attention (present in the next20-vision mlx-lm pin)
        "mlx_lm.models.deepseek_v41.compressor",  # compressed-KV latents + carry state
        "mlx_lm.models.deepseek_v41.indexer",  # index top-k keys
        "mlx_lm.models.deepseek_v41.indexer_hierarchical",  # hierarchical index path
        "mlx_lm.models.deepseek_v41.hyper_connections",  # hyper-connection mixing in the block
        "mlx_lm.models.deepseek_v41.hc_fused",  # fused hyper-connection kernels
        "mlx_lm.models.deepseek_v41.mtp",  # MTP/DSpark draft head
        "mlx_lm.models.deepseek_v41.spec",  # speculative-decode snapshots/rollback
        "mlx_lm.models.deepseek_v41.engram",  # engram hasher + id stream
        "mlx_lm.models.deepseek_v41.cache",  # the per-layer composite cache layout
        "mlx_lm.models.deepseek_v41.session_cache",  # ModelCache/Snapshot; park serializes this
        "mlx_lm.models.deepseek_v41.prefill",  # fenced prefill that builds the cache
        "mlx_lm.models.deepseek_v41.collective",  # all_sum / cross-rank reductions over KV
        "mlx_lm.models.deepseek_v41.moe",  # routed/shared experts (forward math)
        "mlx_lm.models.deepseek_v41.layers",  # RMSNorm/RoPE/rope_tail primitives
        "mlx_lm.models.deepseek_v41.model",  # the model + per-layer wiring
        "mlx_lm.models.deepseek_v41.dequant",  # quantized-weight dequant math
        "mlx_lm.models.deepseek_v41.exl3_build",  # builds the EXL3 model (quant groups)
        "mlx_lm.models.deepseek_v41.fakequant",  # ue8m0/e4m3 fake-quant used by attn/indexer
        "mlx_lm.models.deepseek_v41.config",  # ModelArgs -> geometry (layers/window/ratios)
    }
)

#: Modules that are genuinely KV-neutral: a change to them cannot alter the
#: bytes of a restorable (non-image) parked cache. Every entry carries a WHY.
KV_NEUTRAL: frozenset[str] = frozenset(
    {
        # --- exo DSv4.1 engine --------------------------------------------------
        "exo.worker.engines.mlx.dsv41",  # package docstring only, no code
        "exo.worker.engines.mlx.dsv41.dispatch",  # which card gets the engine; no KV/forward math
        "exo.worker.engines.mlx.dsv41.dsml",  # DSML text parsing; touches no tensors
        "exo.worker.engines.mlx.dsv41.output",  # output pipeline (thinking split / DSML); no cache
        "exo.worker.engines.mlx.dsv41.thinking",  # reasoning-marker resolution; no cache
        "exo.worker.engines.mlx.dsv41.errors",  # exception types only
        "exo.worker.engines.mlx.dsv41.agreement",  # rank task bookkeeping; "nothing touches model state"
        "exo.worker.engines.mlx.dsv41.coord",  # §G4 cross-rank park/restore decisions; int control rows, no KV bytes
        "exo.worker.engines.mlx.dsv41.shutdown",  # §G3 signal state + shutdown budget + orchestration; no cache, no KV bytes
        "exo.worker.engines.mlx.dsv41.vision",  # engine-side ViT adapter; image convos forced cold, never parked
        "exo.worker.engines.mlx.dsv41.fingerprint",  # this module: hashes sources, touches no tensors/KV
        # --- exo disaggregated wire --------------------------------------------
        "exo.worker.engines.mlx.disaggregated",  # empty package __init__
        # --- mlx-lm DSv4.1 model stack -----------------------------------------
        "mlx_lm.models.deepseek_v41",  # import shim (Model/ModelArgs) only
        "mlx_lm.models.deepseek_v41.sampling",  # token sampler; does not change stored KV bytes
        "mlx_lm.models.deepseek_v41.logprobs",  # logprob computation; reads no cache state
        "mlx_lm.models.deepseek_v41.serve",  # fork's standalone server/CLI, unused by exo's forward
        "mlx_lm.models.deepseek_v41.vision",  # ViT tower; image convos never parked
        "mlx_lm.models.deepseek_v41.image_processor",  # ViT preprocessing; image convos never parked
    }
)

#: The three package families whose modules can affect parked KV. A module is
#: "in a family" when it equals a prefix or lives under ``<prefix>.``.
_FAMILY_PREFIXES = (
    "exo.worker.engines.mlx.dsv41",
    "exo.worker.engines.mlx.disaggregated",
    "mlx_lm.models.deepseek_v41",
)

#: Forward-pass entry modules imported by :func:`allowlist_problems` before it
#: walks ``sys.modules``. Importing these pulls in the whole live KV stack (and
#: every module an unreviewed addition would shadow). A missing one (e.g. the
#: MLX-lM fork absent) is skipped, never raised.
_ENTRY_MODULES = (
    "exo.worker.engines.mlx.dsv41.engine",
    "exo.worker.engines.mlx.dsv41.park",
    "exo.worker.engines.mlx.dsv41.session",
    "exo.worker.engines.mlx.dsv41.builder",
    "exo.worker.engines.mlx.disaggregated.adapter",
    "mlx_lm.models.deepseek_v41.model",
    "mlx_lm.models.deepseek_v41.session_cache",
    "mlx_lm.models.deepseek_v41.prefill",
    "mlx_lm.models.deepseek_v41.config",
)


# ---------------------------------------------------------------------------
# digest primitives
# ---------------------------------------------------------------------------


def _canonical(obj: Any) -> bytes:
    """Canonical JSON bytes: stable key order, no incidental whitespace."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    """Streamed sha256 of a file (config/tokenizer files can be a few MB)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe(fn: Callable[[], str], what: str) -> str:
    """Run a fingerprint helper, degrading to a deterministic hash on error.

    A hashing hiccup (a racing file, a permissions error, a partial model dir)
    must never break a park or a restore, so any exception becomes a stable
    placeholder derived from what failed -- distinct from a successful hash, so
    a broken build cannot collide with a good one.
    """
    try:
        return fn()
    except Exception:  # noqa: BLE001 - fingerprinting must never raise
        return _digest(_canonical({"fingerprint_error": what}))


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _field(obj: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` off a Mapping or an attribute object, else ``default``."""
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


# ---------------------------------------------------------------------------
# model / tokenizer / kv-code / mlx digests
# ---------------------------------------------------------------------------

#: Weight/quant file suffixes whose SIZES identify a checkpoint (not contents).
_WEIGHT_EXTS = (".safetensors", ".bin", ".gguf", ".npz", ".pth", ".pt")
#: Names that are config-like (hashed by CONTENT, not size).
_INDEX_SUFFIX = ".index.json"


def _is_quant_config(name_low: str) -> bool:
    return name_low.endswith(".json") and "quant" in name_low


def _model_fp(model_path: Any) -> str:
    """Hash of config.json + weight index + tensor names/sizes + quant config.

    NOT the weights themselves (the plan is explicit: a multi-GB checkpoint must
    not be re-read at startup). Tensor files contribute name + size, which
    catches a different/renamed/truncated checkpoint; config, index and quant
    JSON contribute content.
    """
    if model_path is None:
        return _digest(_canonical({"model_path": "<none>"}))
    p = Path(str(model_path))
    if p.is_dir():
        entries: list[list[Any]] = []
        for f in sorted(p.rglob("*")):
            if not f.is_file():
                continue
            try:
                rel = f.relative_to(p).as_posix()
                low = rel.lower()
                if (
                    rel == "config.json"
                    or _is_quant_config(low)
                    or low.endswith(_INDEX_SUFFIX)
                ):
                    entries.append([rel, "sha256", _sha256_file(f)])
                elif low.endswith(_WEIGHT_EXTS):
                    entries.append([rel, "size", f.stat().st_size])
            except OSError:  # pragma: no cover - racing file
                continue
        return _digest(_canonical({"model_path_kind": "dir", "entries": entries}))
    if p.is_file():
        return _digest(
            _canonical(
                {"model_path_kind": "file", "name": p.name, "sha256": _sha256_file(p)}
            )
        )
    return _digest(_canonical({"model_path_kind": "missing", "name": p.name}))


#: Substrings that mark a tokenizer/chat-template file inside a tokenizer dir.
_TOKENIZER_HINTS = (
    "tokenizer",
    "token",
    "vocab",
    "merges",
    "chat_template",
    "special_tokens",
    "added_tokens",
)


def _tokenizer_fp(tokenizer: Any) -> str:
    """Hash of tokenizer files + chat template.

    Accepts a directory, a single file, or a tokenizer object. The object case
    is best-effort (it hashes the chat template plus any path hint the wrapper
    exposes), because a pre-loaded ``TokenizerWrapper`` does not expose its
    files; production passes the checkpoint path, which is fully hashed.
    """
    if tokenizer is None:
        return _digest(_canonical({"tokenizer": "<none>"}))
    if isinstance(tokenizer, (str, os.PathLike)):
        p = Path(str(tokenizer))
        if p.is_dir():
            entries: list[list[Any]] = []
            for f in sorted(p.rglob("*")):
                if not f.is_file():
                    continue
                try:
                    rel = f.relative_to(p).as_posix()
                    low = rel.lower()
                    if any(hint in low for hint in _TOKENIZER_HINTS) or low.endswith(
                        ".jinja"
                    ):
                        entries.append([rel, "sha256", _sha256_file(f)])
                except OSError:  # pragma: no cover - racing file
                    continue
            return _digest(_canonical({"tokenizer_kind": "dir", "entries": entries}))
        if p.is_file():
            return _digest(
                _canonical(
                    {
                        "tokenizer_kind": "file",
                        "name": p.name,
                        "sha256": _sha256_file(p),
                    }
                )
            )
        return _digest(_canonical({"tokenizer_kind": "missing", "name": p.name}))
    # Object (e.g. a TokenizerWrapper): hash the chat template + a path hint.
    parts: dict[str, Any] = {
        "tokenizer_kind": "object",
        "class": type(tokenizer).__name__,
    }
    template = getattr(tokenizer, "chat_template", None)
    if template is not None:
        parts["chat_template"] = str(template)
    for attr in ("name_or_path", "_name_or_path", "_path"):
        val = getattr(tokenizer, attr, None)
        if isinstance(val, (str, os.PathLike)):
            parts[attr] = str(val)
            break
    return _digest(_canonical(parts))


_PACKAGE_ROOT_CACHE: dict[str, Path | None] = {}


def _package_root(prefix: str) -> Path | None:
    """Filesystem dir of a package, resolved WITHOUT executing its ``__init__``.

    ``find_spec`` imports only the parent packages; that avoids importing (and
    therefore executing) the model stack just to *locate* source files. Falls
    back to importing the package's origin when a spec is unavailable.
    """
    if prefix in _PACKAGE_ROOT_CACHE:
        return _PACKAGE_ROOT_CACHE[prefix]
    root: Path | None = None
    try:
        spec = importlib.util.find_spec(prefix)
        if spec is not None:
            locs = list(spec.submodule_search_locations or [])
            if locs:
                root = Path(locs[0])
            elif spec.origin:
                root = Path(spec.origin).parent
    except Exception:  # noqa: BLE001 - a broken import path degrades, never raises
        root = None
    if root is None:
        try:
            mod = importlib.import_module(prefix)
            f = getattr(mod, "__file__", None)
            root = Path(f).parent if f else None
        except Exception:  # noqa: BLE001
            root = None
    _PACKAGE_ROOT_CACHE[prefix] = root
    return root


def _source_file_for(dotted: str) -> Path | None:
    """Locate an allowlisted module's source file, or ``None`` if unresolved."""
    for prefix in _FAMILY_PREFIXES:
        if dotted == prefix or dotted.startswith(prefix + "."):
            root = _package_root(prefix)
            if root is None:
                return None
            rel = dotted[len(prefix) :].lstrip(".")
            parts = rel.split(".") if rel else []
            candidate = root.joinpath(*parts).with_suffix(".py")
            if candidate.is_file():
                return candidate
            pkg_init = root.joinpath(*parts, "__init__.py")
            if pkg_init.is_file():
                return pkg_init
            return None
    return None


def _kv_code_fp() -> str:
    """sha256 over the allowlisted sources, sorted, ``path + content``.

    The dotted name is folded in alongside the file hash, so adding or removing
    an allowlisted module is itself a build change (and a module whose file
    cannot be resolved contributes a stable ``unresolved`` marker rather than
    silently vanishing).
    """
    parts: list[str] = []
    for dotted in sorted(KV_ALLOWLIST):
        src = _source_file_for(dotted)
        if src is not None and src.is_file():
            try:
                parts.append(f"{dotted}\x00sha256\x00{_sha256_file(src)}")
            except OSError:  # pragma: no cover - racing file
                parts.append(f"{dotted}\x00unreadable")
        else:
            parts.append(f"{dotted}\x00unresolved")
    return _digest("\n".join(parts).encode("utf-8"))


def _mlx_version() -> str:
    """mlx.core version string, or ``""`` when unavailable (documented degrade)."""
    try:
        version = getattr(mx, "__version__", None)
        if version:
            return str(version)
    except Exception:  # noqa: BLE001 - a build without a version attr is fine
        pass
    return ""


# ---------------------------------------------------------------------------
# layout fingerprint
# ---------------------------------------------------------------------------


def layout_fingerprint(
    *,
    world_size: int,
    rank: int,
    layers: Any,
    dtype: str,
    window: int = 0,
    bsz: int = 1,
) -> str:
    """Deterministic sha256 over the tensor-parallel layout.

    Covers world size, rank, the (sorted, deduped) layer range, activation
    dtype, the sliding window and the batch size -- every axis that changes
    which bytes a rank's cache holds. ``layers=None`` means the full stack;
    an unordered or duplicate-bearing iterable is normalized before hashing.
    """
    norm_layers: Any
    if layers is None:
        norm_layers = None
    else:
        try:
            norm_layers = sorted({int(x) for x in layers})
        except (TypeError, ValueError):
            norm_layers = str(layers)
    payload = {
        "world_size": _coerce_int(world_size, 1),
        "rank": _coerce_int(rank, 0),
        "layers": norm_layers,
        "dtype": "" if dtype is None else str(dtype),
        "window": _coerce_int(window, 0),
        "bsz": _coerce_int(bsz, 1),
    }
    return _digest(_canonical(payload))


def _layout_fp_from(layout: Any, model: Any) -> str:
    """Build ``layout_fp`` from a layout mapping, falling back to ``model``.

    ``layout`` is the plan's "layout mapping" (world_size/rank/layers/dtype/
    window/bsz); any field MISSING from it falls back to ``model``'s own
    attributes. With neither, the defaults (world 1, rank 0, full stack,
    empty dtype) are used and the degradation is implicit in the result.

    ``rank`` is DELIBERATELY excluded from the park layout fingerprint: the
    DeepSeek MLA cache is a single shared latent per token (rank-independent), so
    rank is not a cache-content axis, and the two nodes' TP device ranks are
    assigned by launch order and SWAP across a graceful relaunch -- gating a park
    on rank made every dir foreign after a swap. Neither ``layout`` nor ``model``
    is consulted for rank here; :func:`layout_fingerprint` (which stays
    rank-sensitive) is simply called with rank 0.
    """
    ws = _field(layout, "world_size", None)
    layers = _field(layout, "layers", None)
    dtype = _field(layout, "dtype", None)
    window = _field(layout, "window", None)
    bsz = _field(layout, "bsz", None)
    if model is not None:
        args = getattr(model, "args", None)
        if ws is None:
            ws = _field(args, "world_size", None) or getattr(model, "world", None)
        if window is None:
            window = _field(args, "window_size", None)
        if bsz is None:
            bsz = getattr(model, "bsz", None) or _field(args, "bsz", None)
        if dtype is None:
            dtype = _field(args, "dtype", None)
    return layout_fingerprint(
        world_size=_coerce_int(ws, 1),
        rank=0,  # rank-invariant park layout (see the docstring above)
        layers=layers,
        dtype="" if dtype is None else str(dtype),
        window=_coerce_int(window, 0),
        bsz=_coerce_int(bsz, 1),
    )


# ---------------------------------------------------------------------------
# the fingerprint record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BuildFingerprint:
    """The five hashes that identify the build a parked cache was written by."""

    model_fp: str
    tokenizer_fp: str
    kv_code_fp: str
    mlx_version: str
    layout_fp: str

    def as_dict(self) -> dict[str, str]:
        """The five fields as a JSON-safe dict (the manifest sub-object)."""
        return {
            "model_fp": self.model_fp,
            "tokenizer_fp": self.tokenizer_fp,
            "kv_code_fp": self.kv_code_fp,
            "mlx_version": self.mlx_version,
            "layout_fp": self.layout_fp,
        }

    @classmethod
    def from_dict(cls, d: Any) -> "BuildFingerprint | None":
        """Parse a manifest sub-object; ``None`` when absent or malformed.

        Every field must be present and a string. A partial or non-mapping
        input is "legacy" (the caller refuses), never a guessed fingerprint.
        """
        if not isinstance(d, Mapping):
            return None
        keys = ("model_fp", "tokenizer_fp", "kv_code_fp", "mlx_version", "layout_fp")
        values: list[str] = []
        for key in keys:
            val = d.get(key)
            if not isinstance(val, str):
                return None
            values.append(val)
        return cls(*values)


def compute_fingerprint(
    *,
    model_path: Any = None,
    tokenizer: Any = None,
    layout: Any = None,
    model: Any = None,
) -> BuildFingerprint:
    """Compute the build fingerprint ONCE at startup. Never raises.

    Callers pass whatever they have (the loaded model's path and tokenizer, the
    TP layout mapping); a missing input degrades to a canonical representation
    of what IS available rather than failing, so a park/restore never breaks on
    a fingerprinting hiccup (see the module docstring).
    """
    return BuildFingerprint(
        model_fp=_safe(lambda: _model_fp(model_path), "model"),
        tokenizer_fp=_safe(lambda: _tokenizer_fp(tokenizer), "tokenizer"),
        kv_code_fp=_safe(_kv_code_fp, "kv_code"),
        mlx_version=_safe(_mlx_version, "mlx_version"),
        layout_fp=_safe(lambda: _layout_fp_from(layout, model), "layout"),
    )


# ---------------------------------------------------------------------------
# policy
# ---------------------------------------------------------------------------


def _env_flag(name: str, default: bool) -> bool:
    """Truthy unless explicitly set to 0/false/no/off (park.py's convention)."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def allow_stale() -> bool:
    """``EXO_DSV41_PARK_ALLOW_STALE`` (default False): override every refusal."""
    return _env_flag(ENV_ALLOW_STALE, False)


def allow_mlx_drift() -> bool:
    """``EXO_DSV41_PARK_ALLOW_MLX_DRIFT`` (default False): tolerate MLX bumps."""
    return _env_flag(ENV_ALLOW_MLX_DRIFT, False)


def check_manifest(
    man: Mapping[str, Any],
    current: BuildFingerprint,
    *,
    allow_stale: bool = False,
    allow_mlx_drift: bool = False,
) -> tuple[bool, str | None]:
    """Decide whether ``man`` may be restored under ``current``.

    Returns ``(ok, reason)`` with ``reason`` None when ok, else exactly one of
    ``"fp_mismatch"`` / ``"legacy"`` / ``"mlx_drift"``. Magic and ``PARK_VERSION``
    are NOT checked here -- ``park.py`` owns the format version.
    """
    if allow_stale:
        return True, None
    if not isinstance(man, Mapping):
        return False, REASON_LEGACY
    fp = BuildFingerprint.from_dict(man.get(FINGERPRINT_FIELD))
    if fp is None:
        return False, REASON_LEGACY
    if (
        fp.model_fp != current.model_fp
        or fp.tokenizer_fp != current.tokenizer_fp
        or fp.kv_code_fp != current.kv_code_fp
        or fp.layout_fp != current.layout_fp
    ):
        return False, REASON_FP_MISMATCH
    if fp.mlx_version != current.mlx_version:
        if allow_mlx_drift:
            return True, None
        return False, REASON_MLX_DRIFT
    return True, None


# ---------------------------------------------------------------------------
# the CI guard
# ---------------------------------------------------------------------------


def _in_family(dotted: str) -> bool:
    return any(
        dotted == prefix or dotted.startswith(prefix + ".")
        for prefix in _FAMILY_PREFIXES
    )


def _is_test_or_support(dotted: str) -> bool:
    """Whether a family module is test/support scaffolding, not forward code.

    Such a module can never run in a serving forward pass, so a new test must
    not be able to fail the allowlist guard.
    """
    parts = dotted.split(".")
    last = parts[-1]
    return (
        "tests" in parts
        or last == "conftest"
        or last == "_stub_harness"
        or last.startswith("test_")
    )


def allowlist_problems() -> list[str]:
    """CI guard: family modules neither allowlisted nor declared neutral.

    Imports the forward-pass entry modules (best-effort; a missing fork is not
    an error), then returns the sorted dotted names of every module under the
    three KV-affecting families that is in neither :data:`KV_ALLOWLIST` nor
    :data:`KV_NEUTRAL`. An empty list means the allowlist is complete.
    """
    for name in _ENTRY_MODULES:
        try:
            importlib.import_module(name)
        except Exception:  # noqa: BLE001 - a missing fork is a degraded CI env
            continue
    problems = [
        name
        for name in list(sys.modules)
        if _in_family(name)
        and name not in KV_ALLOWLIST
        and name not in KV_NEUTRAL
        and not _is_test_or_support(name)
    ]
    return sorted(problems)
