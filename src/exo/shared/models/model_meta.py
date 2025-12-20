"""Model metadata fetching from Hugging Face.

This module provides functions for fetching model metadata (storage size,
layer count) from Hugging Face Hub.
"""

from typing import Annotated

import aiofiles
import aiofiles.os as aios
from huggingface_hub import model_info
from loguru import logger
from pydantic import BaseModel, Field

from exo.shared.types.memory import Memory
from exo.shared.types.models import ModelId, ModelMetadata
from exo.worker.download.download_utils import (
    ModelSafetensorsIndex,
    download_file_with_retry,
    ensure_models_dir,
)


class ConfigData(BaseModel):
    """Model configuration data from config.json.

    Parses model configuration files, extracting layer count from
    various field names used across different architectures.

    Attributes:
        num_hidden_layers: Layer count (BERT, etc.).
        num_layers: Layer count (generic).
        n_layer: Layer count (GPT-style).
        n_layers: Layer count (alternative naming).
        num_decoder_layers: Decoder layer count (encoder-decoder models).
        decoder_layers: Decoder layers (alternative naming).
    """

    model_config = {"extra": "ignore"}

    num_hidden_layers: Annotated[int, Field(ge=0)] | None = None
    num_layers: Annotated[int, Field(ge=0)] | None = None
    n_layer: Annotated[int, Field(ge=0)] | None = None
    n_layers: Annotated[int, Field(ge=0)] | None = None
    num_decoder_layers: Annotated[int, Field(ge=0)] | None = None
    decoder_layers: Annotated[int, Field(ge=0)] | None = None

    @property
    def layer_count(self) -> int:
        """Extract layer count from configuration.

        Checks common field names used by different model architectures.

        Returns:
            Number of layers in the model.

        Raises:
            ValueError: If no layer count field is found.
        """
        layer_fields = [
            self.num_hidden_layers,
            self.num_layers,
            self.n_layer,
            self.n_layers,
            self.num_decoder_layers,
            self.decoder_layers,
        ]

        for layer_count in layer_fields:
            if layer_count is not None:
                return layer_count

        raise ValueError(
            f"No layer count found in config.json: {self.model_dump_json()}"
        )


async def get_config_data(model_id: str) -> ConfigData:
    """Download and parse config.json for a Hugging Face model.

    Args:
        model_id: Hugging Face model identifier.

    Returns:
        ConfigData with parsed configuration.
    """
    target_dir = (await ensure_models_dir()) / str(model_id).replace("/", "--")
    await aios.makedirs(target_dir, exist_ok=True)
    config_path = await download_file_with_retry(
        model_id,
        "main",
        "config.json",
        target_dir,
        lambda curr_bytes, total_bytes, is_renamed: logger.info(
            f"Downloading config.json for {model_id}: {curr_bytes}/{total_bytes} ({is_renamed=})"
        ),
    )
    async with aiofiles.open(config_path, "r") as f:
        return ConfigData.model_validate_json(await f.read())


async def get_safetensors_size(model_id: str) -> Memory:
    """Get model storage size from safetensors index or Hugging Face API.

    Attempts to get size from the safetensors index file first, falling
    back to the Hugging Face API if metadata is not available.

    Args:
        model_id: Hugging Face model identifier.

    Returns:
        Memory object with model storage size.

    Raises:
        ValueError: If safetensors info cannot be found.
    """
    target_dir = (await ensure_models_dir()) / str(model_id).replace("/", "--")
    await aios.makedirs(target_dir, exist_ok=True)
    index_path = await download_file_with_retry(
        model_id,
        "main",
        "model.safetensors.index.json",
        target_dir,
        lambda curr_bytes, total_bytes, is_renamed: logger.info(
            f"Downloading model.safetensors.index.json for {model_id}: {curr_bytes}/{total_bytes} ({is_renamed=})"
        ),
    )
    async with aiofiles.open(index_path, "r") as f:
        index_data = ModelSafetensorsIndex.model_validate_json(await f.read())

    metadata = index_data.metadata
    if metadata is not None:
        return Memory.from_bytes(metadata.total_size)

    info = model_info(model_id)
    if info.safetensors is None:
        raise ValueError(f"No safetensors info found for {model_id}")
    return Memory.from_bytes(info.safetensors.total)


_model_meta_cache: dict[str, ModelMetadata] = {}
"""Cache for fetched model metadata."""


async def get_model_meta(model_id: str) -> ModelMetadata:
    """Get model metadata with caching.

    Fetches metadata from cache if available, otherwise fetches and caches it.

    Args:
        model_id: Hugging Face model identifier.

    Returns:
        ModelMetadata with storage size and layer count.
    """
    if model_id in _model_meta_cache:
        return _model_meta_cache[model_id]
    model_meta = await _get_model_meta(model_id)
    _model_meta_cache[model_id] = model_meta
    return model_meta


async def _get_model_meta(model_id: str) -> ModelMetadata:
    """Fetch storage size and layer count for a Hugging Face model.

    Downloads config.json and safetensors index to extract metadata.

    Args:
        model_id: Hugging Face model identifier.

    Returns:
        ModelMetadata with storage size and layer count.
    """
    config_data = await get_config_data(model_id)
    num_layers = config_data.layer_count
    mem_size_bytes = await get_safetensors_size(model_id)

    return ModelMetadata(
        model_id=ModelId(model_id),
        pretty_name=model_id,
        storage_size=mem_size_bytes,
        n_layers=num_layers,
    )
