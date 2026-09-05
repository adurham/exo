"""Local tokenizer/model discovery for the seam-rule harness.

OFFLINE ONLY. Never downloads anything. Looks in the two places exo/HF
already keep DeepSeek-V4-Flash tokenizer files on this machine:

  1. ~/.exo/models/deepseek-ai--DeepSeek-V4-Flash/  (exo's own model store)
  2. ~/.cache/huggingface/hub/models--*DeepSeek-V4-Flash*/snapshots/*/

If neither is found, `find_tokenizer_dir()` returns None and callers must
skip with a clear message rather than fabricate a result.
"""

from __future__ import annotations

import os
from pathlib import Path


def _has_tokenizer_files(d: Path) -> bool:
    return (d / "tokenizer.json").is_file() and (d / "tokenizer_config.json").is_file()


def find_tokenizer_dir() -> Path | None:
    """Return a local directory containing tokenizer.json + tokenizer_config.json,
    or None if nothing usable is present on disk. Never touches the network."""
    home = Path(os.path.expanduser("~"))

    # 1. exo's own model store (checked first: this is what production actually loads)
    exo_models = home / ".exo" / "models"
    if exo_models.is_dir():
        for candidate in sorted(exo_models.glob("*DeepSeek-V4-Flash*")):
            if candidate.is_dir() and _has_tokenizer_files(candidate):
                return candidate

    # 2. HF hub cache, any DeepSeek-V4-Flash snapshot
    hf_hub = home / ".cache" / "huggingface" / "hub"
    if hf_hub.is_dir():
        for model_dir in sorted(hf_hub.glob("models--*DeepSeek-V4-Flash*")):
            snapshots = model_dir / "snapshots"
            if not snapshots.is_dir():
                continue
            for snap in sorted(snapshots.iterdir()):
                if snap.is_dir() and _has_tokenizer_files(snap):
                    return snap

    return None


def find_vendored_encoder_path() -> Path | None:
    """Locate the vendored DSv4 pure-python encoder module inside this repo
    checkout, by walking up from this file to find repos/exo/src/... .
    Returns None if the repo layout has moved (do not guess)."""
    here = Path(__file__).resolve()
    # tmp/perf-campaign-2/round12/seam_harness/tokenizer_paths.py -> walk up to repo root
    repo_root = here.parents[4] if len(here.parents) >= 4 else None
    if repo_root is None:
        return None
    candidate = (
        repo_root
        / "src"
        / "exo"
        / "worker"
        / "engines"
        / "mlx"
        / "vendor"
        / "deepseek_v4_encoding.py"
    )
    if candidate.is_file():
        return candidate
    return None
