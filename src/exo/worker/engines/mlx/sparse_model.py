"""Sparse model loader for self-speculative decoding.

Loads every Nth layer of a large model (e.g. 235B) to create a lightweight
draft model that shares the same weight distribution. Runs on a separate
device (MacBook) so there's zero GPU contention with the primary model.

Key design: never loads unneeded weight files or creates unneeded layers.
Only the selected layer shards are read from disk, and the model skeleton
is created with num_hidden_layers = len(kept_layers).
"""

# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false, reportAttributeAccessIssue=false, reportMissingParameterType=false, reportUnknownParameterType=false

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import _get_classes
from mlx_lm.utils import load_config as _load_config

from exo.worker.engines.mlx.auto_parallel import get_inner_model, get_layers
from exo.worker.runner.bootstrap import logger


def load_sparse_model(
    model_path: Path,
    skip_factor: int,
    on_layer_loaded: Callable[[int, int], None] | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a model keeping only every Nth layer (plus the last layer).

    Unlike the lazy load_model() approach, this never touches unneeded weight
    files or creates unneeded layer objects. Only the selected shard files are
    read, and the model is built with a reduced num_hidden_layers.

    Args:
        model_path: Path to the full model directory.
        skip_factor: Keep every Nth layer (e.g. 6 → layers 0,6,12,...,last).
        on_layer_loaded: Progress callback(current_index, total_selected).

    Returns:
        (model, config) tuple with only selected layers loaded and evaluated.
    """
    config: dict[str, Any] = _load_config(model_path)
    n_layers: int = int(config["num_hidden_layers"])

    # Compute which layer indices to keep: every skip_factor-th + always the last
    kept_indices = list(range(0, n_layers, skip_factor))
    if kept_indices[-1] != n_layers - 1:
        kept_indices.append(n_layers - 1)
    n_kept = len(kept_indices)

    logger.info(
        f"Sparse loading: {n_kept}/{n_layers} layers "
        f"(skip_factor={skip_factor}, indices={kept_indices[:5]}...{kept_indices[-2:]})"
    )

    # Build set of weight key prefixes we need
    needed_layer_prefixes: set[str] = set()
    for idx in kept_indices:
        needed_layer_prefixes.add(f"model.layers.{idx}.")

    def _key_needed(key: str) -> bool:
        if not key.startswith("model.layers."):
            return True  # embed_tokens, norm, lm_head, etc.
        return any(key.startswith(p) for p in needed_layer_prefixes)

    # Parse safetensors index to load only needed shard files
    index_path = model_path / "model.safetensors.index.json"
    raw_weights: dict[str, mx.array] = {}

    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        weight_map: dict[str, str] = index.get("weight_map", {})

        # Find which shard files contain at least one needed key
        needed_files: set[str] = set()
        for key, shard_file in weight_map.items():
            if _key_needed(key):
                needed_files.add(shard_file)

        total_shards = len(set(weight_map.values()))
        logger.info(f"Sparse loading: {len(needed_files)}/{total_shards} shard files needed")

        for shard_file in sorted(needed_files):
            shard_path = str(model_path / shard_file)
            shard_weights = mx.load(shard_path)
            for key, value in shard_weights.items():
                if _key_needed(key):
                    raw_weights[key] = value
    else:
        logger.warning("No safetensors index found, loading all weights and filtering")
        all_weights = mx.load(str(model_path / "model.safetensors"))
        raw_weights = {k: v for k, v in all_weights.items() if _key_needed(k)}

    # Remap layer indices: model.layers.{orig} → model.layers.{new}
    idx_map = {orig: new for new, orig in enumerate(kept_indices)}
    weights: dict[str, mx.array] = {}
    for key, value in raw_weights.items():
        if key.startswith("model.layers."):
            parts = key.split(".", 3)  # ["model", "layers", "{idx}", ...]
            orig_idx = int(parts[2])
            if orig_idx in idx_map:
                new_key = f"model.layers.{idx_map[orig_idx]}.{parts[3]}"
                weights[new_key] = value
        else:
            weights[key] = value
    del raw_weights

    logger.info(f"Loaded {len(weights)} weight tensors for {n_kept}-layer sparse model")

    # Override num_hidden_layers so the model skeleton only has the kept layers
    config["num_hidden_layers"] = n_kept

    # Build model skeleton (same as load_model but with reduced layer count)
    model_class, model_args_class = _get_classes(config=config)
    model_args = model_args_class.from_dict(config)
    model = model_class(model_args)

    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)

    # Apply quantization (matches load_model logic)
    quantization = config.get("quantization")
    if quantization is not None:
        def _class_predicate(p, m):
            if p in config["quantization"]:
                return config["quantization"][p]
            if not hasattr(m, "to_quantized"):
                return False
            return f"{p}.scales" in weights

        nn.quantize(
            model,
            group_size=quantization["group_size"],
            bits=quantization["bits"],
            mode=quantization.get("mode", "affine"),
            class_predicate=_class_predicate,
        )

    model.eval()
    model.load_weights(list(weights.items()), strict=False)

    # Eval layers one by one for progress + memory control
    inner = get_inner_model(model)
    layers = get_layers(inner)
    total = len(layers)
    for i, layer in enumerate(layers):
        mx.eval(layer)  # pyright: ignore[reportArgumentType]
        if on_layer_loaded is not None:
            on_layer_loaded(i, total)

    # Eval remaining non-layer params (embed_tokens, norm, lm_head)
    mx.eval(model)

    logger.info(
        f"Sparse model loaded: {n_kept} layers "
        f"(original indices: {kept_indices})"
    )

    return model, config


def load_sparse_tp_model(
    model_path: Path,
    skip_factor: int,
    group: mx.distributed.Group,
    on_layer_loaded: Callable[[int, int], None] | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a sparse model and TP-shard it using an existing distributed group.

    Used for local TP draft models that share the JACCL group with the primary.
    Skips patch_tensor_model() since the primary already patched cls.__call__
    (both models share the same class).
    """
    from exo.worker.engines.mlx.auto_parallel import tensor_auto_parallel

    model, config = load_sparse_model(model_path, skip_factor, on_layer_loaded)

    logger.info(f"TP-sharding sparse draft model (group size={group.size()})...")
    model = tensor_auto_parallel(
        model, group, timeout_seconds=120.0,
        on_timeout=None, on_layer_loaded=None,
        patch_model=False,
    )

    mx.eval(model)
    logger.info("Sparse TP draft model ready")

    return model, config
