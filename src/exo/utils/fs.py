"""File system utilities.

This module provides convenience functions for common filesystem operations.
"""

import contextlib
import os
import pathlib
import tempfile
from typing import LiteralString

type StrPath = str | os.PathLike[str]
"""Type for string-based file paths."""

type BytesPath = bytes | os.PathLike[bytes]
"""Type for bytes-based file paths."""

type StrOrBytesPath = str | bytes | os.PathLike[str] | os.PathLike[bytes]
"""Type for paths that can be either string or bytes."""


def delete_if_exists(filename: StrOrBytesPath) -> None:
    """Delete a file if it exists.

    Silently does nothing if the file doesn't exist (no exception raised).

    Args:
        filename: Path to file to delete.
    """
    with contextlib.suppress(FileNotFoundError):
        os.remove(filename)


def ensure_parent_directory_exists(filename: StrPath) -> None:
    """Ensure the parent directory of a file path exists.

    Creates the parent directory (and any necessary parent directories)
    if it doesn't exist. Useful before writing files to ensure the
    directory structure exists.

    Args:
        filename: File path whose parent directory should exist.
    """
    pathlib.Path(filename).parent.mkdir(parents=True, exist_ok=True)


def ensure_directory_exists(dirname: StrPath) -> None:
    """Ensure a directory exists.

    Creates the directory (and any necessary parent directories) if
    it doesn't exist.

    Args:
        dirname: Directory path that should exist.
    """
    pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)


def make_temp_path(name: LiteralString) -> str:
    """Create a temporary directory and return a path within it.

    Creates a new temporary directory and returns a path to a file
    with the given name within it. The directory will need to be
    cleaned up manually.

    Args:
        name: Name of the file path to create.

    Returns:
        Full path to the file in the temporary directory.
    """
    return os.path.join(tempfile.mkdtemp(), name)
