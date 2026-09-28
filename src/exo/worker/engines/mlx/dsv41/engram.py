"""Engram token-map resolution for the DSv4.1 engine.

The EXL3 checkpoint (~211 GB) does NOT carry the engram tables: layers 1 and 14
of the native release each own a ~384M-row fp8 embedding table (~100 GB per
layer, ~40% of the release). Both checkpoints were built from the same
tokenizer (verified equal by sha256 of ``tokenizer.json``), so the compressed
token map the tables are hashed through can be derived from the checkpoint
tokenizer exo already has in hand.

The map is 129,280 int64 entries and is cached as JSON next to the model
weights so serving never pays the derivation twice. The file is derived data
(not a weight), so it lives in a cache directory rather than in the model dir,
which exo may mount read-only.
"""

from __future__ import annotations

import json
from pathlib import Path

from exo.shared.constants import EXO_CACHE_HOME
from exo.worker.runner.bootstrap import logger

#: Committed copy produced by the DSv4.1 harnesses (``~/dsv41-test``), used as
#: a fallback when the derivation path is unavailable. Kept out of the model
#: directory on purpose -- see the module docstring.
_TOKEN_MAP_ENV = "EXO_DSV41_ENGRAM_TOKEN_MAP"


def default_token_map_path(model_path: Path) -> Path:
    return EXO_CACHE_HOME / "dsv41" / f"engram_token_map_{model_path.name}.json"


def token_map_path(model_path: Path) -> Path | None:
    """Where the engram token map for ``model_path`` should live.

    ``EXO_DSV41_ENGRAM_TOKEN_MAP`` (an explicit file path) wins when set.
    """
    import os

    override = os.environ.get(_TOKEN_MAP_ENV)
    if override:
        return Path(override).expanduser()
    return default_token_map_path(model_path)


def ensure_engram_token_map(
    model_path: Path, tokenizer: object
) -> Path | None:
    """Return a path to the engram compressed-token map, deriving it if needed.

    ``tokenizer`` is the HF tokenizer behind exo's ``TokenizerWrapper``. Returns
    ``None`` (with a loud warning) when the map cannot be produced -- callers
    decide whether that is fatal, and for a layer-subset build that excludes the
    engram layers it is not.
    """
    target = token_map_path(model_path)
    if target is None:
        return None
    if target.exists():
        logger.info(f"[DSV41] engram token map: reusing {target}")
        return target

    try:
        # Imported lazily: these modules only exist on a host with the DSv4.1
        # mlx-lm fork checked out (the Macs), never on a control-plane box.
        from mlx_lm.models.deepseek_v41.engram import build_compressed_token_map

        lookup, n_compressed = build_compressed_token_map(tokenizer)
    except Exception as e:  # noqa: BLE001 -- load-time asset, not control flow
        logger.warning(
            f"[DSV41] could not derive the engram token map from {model_path} "
            f"({type(e).__name__}: {e}); set {_TOKEN_MAP_ENV} to a prebuilt map "
            "if this checkpoint's layers need engram lookups"
        )
        return None

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(list(lookup)))
    tmp.replace(target)
    logger.info(
        f"[DSV41] wrote engram token map ({len(lookup)} ids -> {n_compressed} "
        f"compressed) to {target}"
    )
    return target
