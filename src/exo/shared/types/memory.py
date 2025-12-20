"""Memory size representation with multiple unit conversions.

This module provides a Memory class that represents memory sizes with
convenient conversion between bytes, KB, MB, and GB.
"""

from math import ceil
from typing import Self

from exo.utils.pydantic_ext import CamelCaseModel


class Memory(CamelCaseModel):
    """Represents a memory size with unit conversions.

    Supports conversion between bytes, kilobytes, megabytes, and gigabytes.
    All conversions use binary (1024-based) units. Supports arithmetic and
    comparison operations.

    Attributes:
        in_bytes: Memory size in bytes.
    """

    in_bytes: int = 0

    @classmethod
    def from_bytes(cls, val: int) -> Self:
        """Construct a Memory object from bytes.

        Args:
            val: Memory size in bytes.

        Returns:
            New Memory instance.
        """
        return cls(in_bytes=val)

    @property
    def in_kb(self) -> int:
        """Get memory size in kilobytes (rounded up).

        Returns:
            Memory size in KB, rounded up to the nearest integer.
        """
        return ceil(self.in_bytes / 1024)

    @in_kb.setter
    def in_kb(self, val: int) -> None:
        """Set memory size from kilobytes.

        Args:
            val: Memory size in KB.
        """
        self.in_bytes = val * 1024

    @classmethod
    def from_kb(cls, val: int) -> Self:
        """Construct a Memory object from kilobytes.

        Args:
            val: Memory size in KB.

        Returns:
            New Memory instance.
        """
        return cls(in_bytes=val * 1024)

    @classmethod
    def from_float_kb(cls, val: float) -> Self:
        """Construct a Memory object from fractional kilobytes.

        Args:
            val: Memory size in KB (may be fractional).

        Returns:
            New Memory instance with bytes rounded to nearest integer.
        """
        return cls(in_bytes=round(val * 1024))

    @property
    def in_mb(self) -> float:
        """Get memory size in megabytes.

        Returns:
            Memory size in MB as a float.
        """
        return self.in_bytes / (1024**2)

    @in_mb.setter
    def in_mb(self, val: float) -> None:
        """Set memory size from megabytes.

        Args:
            val: Memory size in MB.
        """
        self.in_bytes = round(val * (1024**2))

    @classmethod
    def from_mb(cls, val: float) -> Self:
        """Construct a Memory object from megabytes.

        Args:
            val: Memory size in MB.

        Returns:
            New Memory instance with bytes rounded to nearest integer.
        """
        return cls(in_bytes=round(val * (1024**2)))

    @classmethod
    def from_gb(cls, val: float) -> Self:
        """Construct a Memory object from gigabytes.

        Args:
            val: Memory size in GB.

        Returns:
            New Memory instance with bytes rounded to nearest integer.
        """
        return cls(in_bytes=round(val * (1024**3)))

    @property
    def in_gb(self) -> float:
        """Get memory size in gigabytes.

        Returns:
            Memory size in GB as a float.
        """
        return self.in_bytes / (1024**3)

    def __add__(self, other: "Memory") -> "Memory":
        """Add two Memory values together.

        Args:
            other: Memory value to add.

        Returns:
            New Memory instance with combined size.
        """
        return Memory.from_bytes(self.in_bytes + other.in_bytes)

    def __lt__(self, other: Self) -> bool:
        """Compare if this memory is less than other."""
        return self.in_bytes < other.in_bytes

    def __le__(self, other: Self) -> bool:
        """Compare if this memory is less than or equal to other."""
        return self.in_bytes <= other.in_bytes

    def __gt__(self, other: Self) -> bool:
        """Compare if this memory is greater than other."""
        return self.in_bytes > other.in_bytes

    def __ge__(self, other: Self) -> bool:
        """Compare if this memory is greater than or equal to other."""
        return self.in_bytes >= other.in_bytes
