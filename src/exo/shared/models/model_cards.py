from enum import Enum
from typing import Annotated, Any

import aiofiles
import aiofiles.os as aios
import tomlkit
from anyio import Path, open_file
from huggingface_hub import model_info
from loguru import logger
from pydantic import (
    AliasChoices,
    BaseModel,
    Field,
    PositiveInt,
    ValidationError,
    field_validator,
    model_validator,
)
from tomlkit.exceptions import TOMLKitError

from exo.shared.constants import (
    EXO_CUSTOM_MODEL_CARDS_DIR,
    EXO_ENABLE_IMAGE_MODELS,
    RESOURCES_DIR,
)
from exo.shared.types.common import ModelId
from exo.shared.types.memory import Memory
from exo.utils.pydantic_ext import CamelCaseModel

# kinda ugly...
# TODO: load search path from config.toml
_custom_cards_dir = Path(str(EXO_CUSTOM_MODEL_CARDS_DIR))
CARD_SEARCH_PATH = [
    Path(RESOURCES_DIR) / "inference_model_cards",
    Path(RESOURCES_DIR) / "image_model_cards",
    _custom_cards_dir,
]

_card_cache: dict[ModelId, "ModelCard"] = {}


async def _refresh_card_cache():
    for path in CARD_SEARCH_PATH:
        logger.info(f"Scanning for model cards in: {path}")
        if not await aios.path.exists(path):
            logger.warning(f"Model card path does not exist: {path}")
            continue
        async for toml_file in path.rglob("*.toml"):
            logger.info(f"Found model card file: {toml_file}")
            try:
                card = await ModelCard.load_from_path(toml_file)
                _card_cache[card.model_id] = card
                logger.info(f"Loaded model card: {card.model_id}")
            except (ValidationError, TOMLKitError) as e:
                logger.error(f"Failed to load model card from {toml_file}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error loading model card from {toml_file}: {e}")


def _is_image_card(card: "ModelCard") -> bool:
    return any(t in (ModelTask.TextToImage, ModelTask.ImageToImage) for t in card.tasks)


async def get_model_cards() -> list["ModelCard"]:
    if len(_card_cache) == 0:
        await _refresh_card_cache()
    if EXO_ENABLE_IMAGE_MODELS:
        return list(_card_cache.values())
    return [c for c in _card_cache.values() if not _is_image_card(c)]


class ModelTask(str, Enum):
    TextGeneration = "TextGeneration"
    TextToImage = "TextToImage"
    ImageToImage = "ImageToImage"


class ComponentInfo(CamelCaseModel):
    component_name: str
    component_path: str
    storage_size: Memory
    n_layers: PositiveInt | None = None
    can_shard: bool
    safetensors_index_filename: str | None = None


class ModelCard(CamelCaseModel):
    model_id: ModelId
    storage_size: Memory
    n_layers: PositiveInt
    hidden_size: PositiveInt
    num_kv_heads: PositiveInt = 8  # Default for GQA in most large models
    head_dim: PositiveInt = 128    # Standard transformer head dimension
    max_context_length: PositiveInt = 2048 # Default if not specified
    supports_tensor: bool
    tasks: list[ModelTask]
    components: list[ComponentInfo] | None = None
    family: str = ""
    quantization: str = ""
    base_model: str = ""
    capabilities: list[str] = []
    uses_cfg: bool = False

    @field_validator("tasks", mode="before")
    @classmethod
    def _validate_tasks(cls, v: list[str | ModelTask]) -> list[ModelTask]:
        return [item if isinstance(item, ModelTask) else ModelTask(item) for item in v]

    async def save(self, path: Path) -> None:
        async with await open_file(path, "w") as f:
            py = self.model_dump(exclude_none=True)
            data = tomlkit.dumps(py)  # pyright: ignore[reportUnknownMemberType]
            await f.write(data)

    async def save_to_custom_dir(self) -> None:
        await aios.makedirs(str(_custom_cards_dir), exist_ok=True)
        await self.save(_custom_cards_dir / (self.model_id.normalize() + ".toml"))

    @staticmethod
    async def load_from_path(path: Path) -> "ModelCard":
        async with await open_file(path, "r") as f:
            py = tomlkit.loads(await f.read())
            return ModelCard.model_validate(py)

    # Is it okay that model card.load defaults to network access if the card doesn't exist? do we want to be more explicit here?
    @staticmethod
    async def load(model_id: ModelId) -> "ModelCard":
        if model_id not in _card_cache:
            await _refresh_card_cache()
        if (mc := _card_cache.get(model_id)) is not None:
            return mc

        return await ModelCard.fetch_from_hf(model_id)

    @staticmethod
    async def fetch_from_hf(model_id: ModelId) -> "ModelCard":
        """Fetches storage size and number of layers for a Hugging Face model, returns Pydantic ModelMeta."""
        # TODO: failure if files do not exist
        config_data = await fetch_config_data(model_id)
        num_layers = config_data.layer_count
        mem_size_bytes = await fetch_safetensors_size(model_id)

        mc = ModelCard(
            model_id=ModelId(model_id),
            storage_size=mem_size_bytes,
            n_layers=num_layers,
            hidden_size=config_data.hidden_size or 0,
            num_kv_heads=config_data.num_kv_heads or 8,
            max_context_length=config_data.max_context_length or 2048,
            supports_tensor=config_data.supports_tensor,
            tasks=[ModelTask.TextGeneration],
            family=_infer_family(model_id),
        )
        await mc.save_to_custom_dir()
        _card_cache[model_id] = mc
        return mc


def _infer_family(model_id: str) -> str:
    """Infers the model family from the model ID."""
    model_id = model_id.lower()
    if "llama" in model_id:
        return "llama"
    elif "qwen" in model_id:
        return "qwen"
    elif "deepseek" in model_id:
        return "deepseek"
    elif "gemma" in model_id:
        return "gemma"
    elif "mistral" in model_id:
        return "mistral"
    elif "phi" in model_id:
        return "phi"
    elif "glm" in model_id:
        return "glm"
    elif "minimax" in model_id:
        return "minimax"
    elif "kimi" in model_id or "moonshot" in model_id:
        return "kimi"
    elif "step" in model_id:
        return "step"
    elif "gpt-oss" in model_id or "gpt" in model_id:
        return "gpt-oss"
    elif "flux" in model_id:
        return "flux"
    return ""


async def delete_custom_card(model_id: ModelId) -> bool:
    """Delete a user-added custom model card. Returns True if deleted."""
    card_path = _custom_cards_dir / (ModelId(model_id).normalize() + ".toml")
    if await card_path.exists():
        await card_path.unlink()
        _card_cache.pop(model_id, None)
        return True
    return False


def is_custom_card(model_id: ModelId) -> bool:
    """Check if a model card exists in the custom cards directory."""
    import os

    card_path = Path(str(EXO_CUSTOM_MODEL_CARDS_DIR)) / (
        ModelId(model_id).normalize() + ".toml"
    )
    return os.path.isfile(str(card_path))


class ConfigData(BaseModel):
    model_config = {"extra": "ignore"}  # Allow unknown fields

    architectures: list[str] | None = None
    hidden_size: Annotated[int, Field(ge=0)] | None = None
    num_kv_heads: int | None = Field(
        None,
        validation_alias=AliasChoices(
            "num_key_value_heads",
            "multi_query_group_num",
            "num_kv_heads",
        ),
    )
    max_context_length: int | None = Field(
        None,
        validation_alias=AliasChoices(
            "max_position_embeddings",
            "max_sequence_length",
        ),
    )
    layer_count: int = Field(
        validation_alias=AliasChoices(
            "num_hidden_layers",
            "num_layers",
            "n_layer",
            "n_layers",
            "num_decoder_layers",
            "decoder_layers",
        )
    )

    @property
    def supports_tensor(self) -> bool:
        return self.architectures in [
            ["Glm4MoeLiteForCausalLM"],
            ["GlmMoeDsaForCausalLM"],
            ["DeepseekV32ForCausalLM"],
            ["DeepseekV3ForCausalLM"],
            ["Qwen3NextForCausalLM"],
            ["Qwen3MoeForCausalLM"],
            ["MiniMaxM2ForCausalLM"],
            ["LlamaForCausalLM"],
            ["GptOssForCausalLM"],
            ["Step3p5ForCausalLM"],
        ]

    @model_validator(mode="before")
    @classmethod
    def defer_to_text_config(cls, data: dict[str, Any]):
        text_config = data.get("text_config")
        if text_config is None:
            return data

        for field in [
            "architectures",
            "hidden_size",
            "num_hidden_layers",
            "num_layers",
            "n_layer",
            "n_layers",
            "num_decoder_layers",
            "decoder_layers",
        ]:
            if (val := text_config.get(field)) is not None:  # pyright: ignore[reportAny]
                data[field] = val

        return data


async def fetch_config_data(model_id: ModelId) -> ConfigData:
    """Downloads and parses config.json for a model."""
    from exo.download.download_utils import (
        download_file_with_retry,
        ensure_models_dir,
    )

    target_dir = (await ensure_models_dir()) / model_id.normalize()
    await aios.makedirs(target_dir, exist_ok=True)
    config_path = await download_file_with_retry(
        model_id,
        "main",
        "config.json",
        target_dir,
        lambda curr_bytes, total_bytes, is_renamed: logger.debug(
            f"Downloading config.json for {model_id}: {curr_bytes}/{total_bytes} ({is_renamed=})"
        ),
    )
    async with aiofiles.open(config_path, "r") as f:
        return ConfigData.model_validate_json(await f.read())


async def fetch_safetensors_size(model_id: ModelId) -> Memory:
    """Gets model size from safetensors index or falls back to HF API."""
    from exo.download.download_utils import (
        download_file_with_retry,
        ensure_models_dir,
    )
    from exo.shared.types.worker.downloads import ModelSafetensorsIndex

    target_dir = (await ensure_models_dir()) / model_id.normalize()
    await aios.makedirs(target_dir, exist_ok=True)
    index_path = await download_file_with_retry(
        model_id,
        "main",
        "model.safetensors.index.json",
        target_dir,
        lambda curr_bytes, total_bytes, is_renamed: logger.debug(
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
