"""Hugging Face API and repository utilities.

This module provides utilities for interacting with Hugging Face, including
filtering repository objects, authentication, and pattern matching.
"""

import os
from fnmatch import fnmatch
from pathlib import Path
from typing import Callable, Generator, Iterable

import aiofiles
import aiofiles.os as aios
from loguru import logger

from exo.shared.types.worker.shards import ShardMetadata


def filter_repo_objects[T](
    items: Iterable[T],
    *,
    allow_patterns: list[str] | str | None = None,
    ignore_patterns: list[str] | str | None = None,
    key: Callable[[T], str] | None = None,
) -> Generator[T, None, None]:
    """Filter repository objects by filename patterns.

    Filters items based on allow/ignore patterns using fnmatch. Directory
    patterns ending with "/" are automatically expanded to include all
    contents with "*".

    Args:
        items: Items to filter.
        allow_patterns: Patterns to allow (if provided, items must match one).
        ignore_patterns: Patterns to ignore (items matching are excluded).
        key: Function to extract path string from items (defaults to str/Path).

    Yields:
        Filtered items.

    Raises:
        ValueError: If key is needed but item is not str or Path.
    """
    if isinstance(allow_patterns, str):
        allow_patterns = [allow_patterns]
    if isinstance(ignore_patterns, str):
        ignore_patterns = [ignore_patterns]
    if allow_patterns is not None:
        allow_patterns = [_add_wildcard_to_directories(p) for p in allow_patterns]
    if ignore_patterns is not None:
        ignore_patterns = [_add_wildcard_to_directories(p) for p in ignore_patterns]

    if key is None:

        def _identity(item: T) -> str:
            if isinstance(item, str):
                return item
            if isinstance(item, Path):
                return str(item)
            raise ValueError(
                f"Please provide `key` argument in `filter_repo_objects`: `{item}` is not a string."
            )

        key = _identity

    for item in items:
        path = key(item)
        if allow_patterns is not None and not any(
            fnmatch(path, r) for r in allow_patterns
        ):
            continue
        if ignore_patterns is not None and any(
            fnmatch(path, r) for r in ignore_patterns
        ):
            continue
        yield item


def _add_wildcard_to_directories(pattern: str) -> str:
    """Add wildcard to directory patterns.

    Args:
        pattern: Pattern string.

    Returns:
        Pattern with "*" appended if it ends with "/".
    """
    if pattern[-1] == "/":
        return pattern + "*"
    return pattern


def get_hf_endpoint() -> str:
    """Get Hugging Face API endpoint.

    Returns:
        Endpoint URL from HF_ENDPOINT env var or default.
    """
    return os.environ.get("HF_ENDPOINT", "https://huggingface.co")


def get_hf_home() -> Path:
    """Get the Hugging Face home directory.

    Returns:
        Path from HF_HOME env var or default ~/.cache/huggingface.
    """
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))


async def get_hf_token() -> str | None:
    """Retrieve the Hugging Face token from the user's HF_HOME directory.

    Returns:
        Token string if found, None otherwise.
    """
    token_path = get_hf_home() / "token"
    if await aios.path.exists(token_path):
        async with aiofiles.open(token_path, "r") as f:
            return (await f.read()).strip()
    return None


async def get_auth_headers() -> dict[str, str]:
    """Get authentication headers for Hugging Face API.

    Returns:
        Dictionary with Authorization header if token available, empty dict otherwise.
    """
    token = await get_hf_token()
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def extract_layer_num(tensor_name: str) -> int | None:
    """Extract layer number from tensor name.

    Args:
        tensor_name: Tensor name (e.g., "layers.5.attention.qkv").

    Returns:
        Layer number if found, None otherwise.
    """
    parts = tensor_name.split(".")
    for part in parts:
        if part.isdigit():
            return int(part)
    return None


def get_allow_patterns(weight_map: dict[str, str], shard: ShardMetadata) -> list[str]:
    """Get file patterns to download for a shard.

    Includes default patterns (config, tokenizer files) plus shard-specific
    safetensors files from the weight map.

    Args:
        weight_map: Mapping from weight names to safetensors filenames.
        shard: Shard metadata.

    Returns:
        List of filename patterns to download.
    """
    default_patterns = set(
        ["*.json", "*.py", "tokenizer.model", "*.tiktoken", "*.txt", "*.jinja"]
    )
    shard_specific_patterns: set[str] = set()
    if weight_map:
        for tensor_name, filename in weight_map.items():
            layer_num = extract_layer_num(tensor_name)
            if (
                layer_num is not None
                and shard.start_layer <= layer_num <= shard.end_layer
            ):
                shard_specific_patterns.add(filename)
        layer_independent_files = set(
            [v for k, v in weight_map.items() if extract_layer_num(k) is None]
        )
        shard_specific_patterns.update(layer_independent_files)
        logger.debug(f"get_allow_patterns {shard=} {layer_independent_files=}")
    else:
        shard_specific_patterns = set(["*.safetensors"])
    logger.info(f"get_allow_patterns {shard=} {shard_specific_patterns=}")
    return list(default_patterns | shard_specific_patterns)
