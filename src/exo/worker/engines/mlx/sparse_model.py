"""Sparse model loader for self-speculative decoding.

Loads every Nth layer of a large model (e.g. 235B) to create a lightweight
draft model that shares the same weight distribution. Runs on a separate
device (MacBook) so there's zero GPU contention with the primary model.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import (
    load_config as _load_config,  # pyright: ignore[reportUnknownVariableType]
)
from mlx_lm.utils import load_model

from exo.worker.engines.mlx.auto_parallel import get_inner_model, get_layers
from exo.worker.runner.bootstrap import logger


def load_sparse_model(
    model_path: Path,
    skip_factor: int,
    on_layer_loaded: Callable[[int, int], None] | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a model keeping only every Nth layer (plus the last layer).

    Uses load_model(lazy=True) so all weights start as unmaterialized mmap
    references. After replacing the layers list with just the selected subset,
    only those layers' weights get evaluated into GPU memory. Dropped layers'
    lazy arrays are GC'd without ever hitting disk.

    Args:
        model_path: Path to the full model directory.
        skip_factor: Keep every Nth layer (e.g. 6 → layers 0,6,12,...,last).
        on_layer_loaded: Progress callback(current_index, total_selected).

    Returns:
        (model, config) tuple with only selected layers loaded and evaluated.
    """
    config: dict[str, Any] = _load_config(model_path)  # pyright: ignore[reportUnknownVariableType]
    n_layers: int = int(config["num_hidden_layers"])  # pyright: ignore[reportAny]

    # Compute which layer indices to keep: every skip_factor-th + always the last
    kept_indices = list(range(0, n_layers, skip_factor))
    if kept_indices[-1] != n_layers - 1:
        kept_indices.append(n_layers - 1)

    logger.info(
        f"Sparse loading: {len(kept_indices)}/{n_layers} layers "
        f"(skip_factor={skip_factor}, indices={kept_indices[:5]}...{kept_indices[-2:]})"
    )

    # Load full model lazily — no weights materialized yet (all mmap references).
    # This handles quantization, sanitization, and model class lookup correctly.
    model, config = load_model(model_path, lazy=True, strict=False)

    inner = get_inner_model(model)
    all_layers = get_layers(inner)

    # Extract only the selected layers
    selected_layers = [all_layers[i] for i in kept_indices]

    # Replace the layers list — dropped layers (and their lazy weights) get GC'd
    if hasattr(inner, "layers"):
        inner.layers = selected_layers
    elif hasattr(inner, "h"):
        inner.h = selected_layers

    # Eval selected layers one by one for progress + memory control
    total_selected = len(selected_layers)
    for i, layer in enumerate(selected_layers):
        mx.eval(layer)  # pyright: ignore[reportArgumentType]
        if on_layer_loaded is not None:
            on_layer_loaded(i, total_selected)

    # Eval remaining non-layer params (embed_tokens, norm, lm_head)
    mx.eval(model)

    logger.info(
        f"Sparse model loaded: {total_selected} layers "
        f"(original indices: {kept_indices})"
    )

    return model, config
