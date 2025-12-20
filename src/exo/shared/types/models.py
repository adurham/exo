"""Model metadata type definitions.

This module defines types for model identification and metadata.
"""

from pydantic import PositiveInt

from exo.shared.types.common import Id
from exo.shared.types.memory import Memory
from exo.utils.pydantic_ext import CamelCaseModel


class ModelId(Id):
    """Identifier for models in the system."""

    pass


class ModelMetadata(CamelCaseModel):
    """Metadata describing a model's properties.

    Attributes:
        model_id: Unique identifier for the model.
        pretty_name: Human-readable name for the model.
        storage_size: Size of the model files in storage.
        n_layers: Number of layers in the model.
    """

    model_id: ModelId
    pretty_name: str
    storage_size: Memory
    n_layers: PositiveInt
