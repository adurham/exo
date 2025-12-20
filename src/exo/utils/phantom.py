"""Phantom type utilities for generic type hints.

This module provides phantom types for use in generic function signatures
where the generic type parameter is needed for type checking but no
runtime value of that type is actually stored.
"""

class _PhantomData[*T]:
    """Internal phantom type marker (not meant for direct use)."""

    pass


type PhantomData[*T] = _PhantomData[*T] | None
"""Phantom type for generic signatures without runtime storage.

Allows using generics in function signatures for type checking purposes
without actually storing a value of that type. Use `None` as the value.
Useful for type-level programming where the type matters but the value doesn't.
"""
