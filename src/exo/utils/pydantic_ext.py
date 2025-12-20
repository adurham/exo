"""Pydantic model extensions for EXO.

This module provides custom Pydantic model base classes with special
serialization and validation behavior for the EXO system.
"""

# pyright: reportAny=false, reportUnknownArgumentType=false, reportUnknownVariableType=false

from typing import Any, Self

from pydantic import BaseModel, ConfigDict, model_serializer, model_validator
from pydantic.alias_generators import to_camel
from pydantic_core.core_schema import (
    SerializerFunctionWrapHandler,
    ValidatorFunctionWrapHandler,
)


class CamelCaseModel(BaseModel):
    """Pydantic model with camelCase field aliases.

    Automatically converts snake_case field names to camelCase in JSON
    serialization/deserialization, matching JavaScript naming conventions.
    Useful for API responses that need to be consumed by web clients.

    Configuration:
        - Fields are aliased to camelCase automatically
        - Strict validation enabled
        - Extra fields are forbidden
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        validate_by_name=True,
        extra="forbid",
        strict=True,
    )


class TaggedModel(CamelCaseModel):
    """Tagged union model with type discrimination.

    Serializes models wrapped in a dict with the class name as the key,
    enabling discriminated unions for pattern matching. Example:
    `{"TaskCreated": {"task_id": "..."}}`

    Also supports deserializing from either the wrapped format or unwrapped,
    providing flexibility in API compatibility.
    """

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Serialize with type tag wrapper.

        Args:
            handler: Serialization handler.

        Returns:
            Dict with class name as key and serialized data as value.
        """
        inner = handler(self)
        return {self.__class__.__name__: inner}

    @model_validator(mode="wrap")
    @classmethod
    def _validate(cls, v: Any, handler: ValidatorFunctionWrapHandler) -> Self:
        """Validate with support for wrapped and unwrapped formats.

        Args:
            v: Input value (dict or wrapped dict).
            handler: Validation handler.

        Returns:
            Validated model instance.
        """
        if isinstance(v, dict) and len(v) == 1 and cls.__name__ in v:
            return handler(v[cls.__name__])

        return handler(v)

    def __str__(self) -> str:
        """String representation with class name prefix."""
        return f"{self.__class__.__name__}({super().__str__()})"
