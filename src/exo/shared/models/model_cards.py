from exo.shared.types.memory import Memory
from exo.shared.types.models import ModelId, ModelMetadata
from exo.utils.pydantic_ext import CamelCaseModel


class ModelCard(CamelCaseModel):
    short_id: str
    model_id: ModelId
    name: str
    description: str
    tags: list[str]
    metadata: ModelMetadata


MODEL_CARDS: dict[str, ModelCard] = {
    # Only Qwen3-235B (4-bit) - the single model for this deployment
    "qwen3-235b-a22b-4bit": ModelCard(
        short_id="qwen3-235b-a22b-4bit",
        model_id=ModelId("mlx-community/Qwen3-235B-A22B-Instruct-2507-4bit"),
        name="Qwen3 235B A22B (4-bit)",
        description="""Qwen3 235B (Active 22B) is a large language model trained on the Qwen3 235B dataset.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/Qwen3-235B-A22B-Instruct-2507-4bit"),
            pretty_name="Qwen3 235B A22B (4-bit)",
            storage_size=Memory.from_gb(132),
            n_layers=94,
        ),
    ),
}
