"""Load the vendored DSv4 encoder module by file path (it lives under
src/exo/worker/engines/mlx/vendor/ with no package __init__, and this
harness must not add anything to sys.path that could accidentally shadow
or get imported by production code -- so we load it via importlib from an
explicit path instead of `import exo...`)."""

from __future__ import annotations

import importlib.util
import types
from pathlib import Path

from tokenizer_paths import find_vendored_encoder_path


def load_vendored_encoder() -> types.ModuleType | None:
    path = find_vendored_encoder_path()
    if path is None:
        return None
    spec = importlib.util.spec_from_file_location("deepseek_v4_encoding_readonly", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def render(module: types.ModuleType, messages: list[dict], thinking_mode: str = "chat") -> str:
    """Thin wrapper documenting the exact call shape used by this harness."""
    return module.encode_messages(messages, thinking_mode=thinking_mode)


def vendored_source_path() -> Path | None:
    return find_vendored_encoder_path()
