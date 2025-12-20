"""Utility functions for type checking and placeholder implementations.

This module provides utility functions for runtime type checking and
placeholder implementations for unimplemented code.
"""

from typing import Any, Type

from .phantom import PhantomData


def ensure_type[T](obj: Any, expected_type: Type[T]) -> T:  # type: ignore
    """Ensure an object is of the expected type, raising TypeError if not.

    Args:
        obj: Object to type-check.
        expected_type: Expected type.

    Returns:
        The object, typed as T.

    Raises:
        TypeError: If obj is not an instance of expected_type.
    """
    if not isinstance(obj, expected_type):
        raise TypeError(f"Expected {expected_type}, got {type(obj)}")  # type: ignore
    return obj


def todo[T](
    msg: str = "This code has not been implemented yet.",
    _phantom: PhantomData[T] = None,
) -> T:
    """Placeholder function for unimplemented code.

    Raises NotImplementedError with a message. Used as a type-safe placeholder
    for functions that need to be implemented later.

    Args:
        msg: Error message to display.
        _phantom: Phantom type parameter for type inference (unused at runtime).

    Returns:
        Never returns (always raises).

    Raises:
        NotImplementedError: Always raised with the provided message.
    """
    raise NotImplementedError(msg)
