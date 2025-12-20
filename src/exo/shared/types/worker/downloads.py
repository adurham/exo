"""Download progress types for model shard downloads.

This module defines types for tracking the progress of downloading
model shards from Hugging Face or other sources.
"""

from exo.shared.types.common import NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.worker.shards import ShardMetadata
from exo.utils.pydantic_ext import CamelCaseModel, TaggedModel


class DownloadProgressData(CamelCaseModel):
    """Detailed download progress information.

    Attributes:
        total_bytes: Total size of the download.
        downloaded_bytes: Total bytes downloaded so far.
        downloaded_bytes_this_session: Bytes downloaded in this session.
        completed_files: Number of files that have completed downloading.
        total_files: Total number of files to download.
        speed: Download speed in bytes per second.
        eta_ms: Estimated time until completion in milliseconds.
        files: Per-file download progress (recursive structure).
    """

    total_bytes: Memory
    downloaded_bytes: Memory
    downloaded_bytes_this_session: Memory
    completed_files: int
    total_files: int
    speed: float
    eta_ms: int
    files: dict[str, "DownloadProgressData"]


class BaseDownloadProgress(TaggedModel):
    """Base class for download progress states.

    Attributes:
        node_id: Node ID where the download is occurring.
        shard_metadata: Shard being downloaded.
    """

    node_id: NodeId
    shard_metadata: ShardMetadata


class DownloadPending(BaseDownloadProgress):
    """Download is queued but not yet started."""

    pass


class DownloadCompleted(BaseDownloadProgress):
    """Download has completed successfully."""

    pass


class DownloadFailed(BaseDownloadProgress):
    """Download has failed with an error.

    Attributes:
        error_message: Human-readable error message.
    """

    error_message: str


class DownloadOngoing(BaseDownloadProgress):
    """Download is currently in progress.

    Attributes:
        download_progress: Detailed progress information.
    """

    download_progress: DownloadProgressData


DownloadProgress = (
    DownloadPending | DownloadCompleted | DownloadFailed | DownloadOngoing
)
"""Discriminated union of all download progress states."""
