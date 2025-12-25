from pathlib import Path

import pytest

from exo.shared.types.memory import Memory
from exo.shared.types.models import ModelId, ModelMetadata
from exo.shared.types.worker.shards import PipelineShardMetadata, ShardMetadata
from exo.worker.download.download_utils import RepoDownloadProgress
from exo.worker.download.shard_downloader import NoopShardDownloader, ShardDownloader


@pytest.fixture
def sample_shard():
    return PipelineShardMetadata(
        model_meta=ModelMetadata(
            model_id=ModelId("test-model"),
            pretty_name="Test Model",
            storage_size=Memory.from_mb(100),
            n_layers=12,
        ),
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=12,
        n_layers=12,
    )


class TestNoopShardDownloader:
    @pytest.mark.asyncio
    async def test_ensure_shard(self, sample_shard):
        downloader = NoopShardDownloader()
        result = await downloader.ensure_shard(sample_shard)
        assert result == Path("/tmp/noop_shard")

    @pytest.mark.asyncio
    async def test_ensure_shard_config_only(self, sample_shard):
        downloader = NoopShardDownloader()
        result = await downloader.ensure_shard(sample_shard, config_only=True)
        assert result == Path("/tmp/noop_shard")

    def test_on_progress(self, sample_shard):
        downloader = NoopShardDownloader()
        callback_calls = []

        def callback(shard: ShardMetadata, progress: RepoDownloadProgress) -> None:
            callback_calls.append((shard, progress))

        # Should not raise
        downloader.on_progress(callback)
        # Callback should not be called (noop implementation)
        assert len(callback_calls) == 0

    @pytest.mark.asyncio
    async def test_get_shard_download_status(self):
        downloader = NoopShardDownloader()
        statuses = []
        async for path, progress in downloader.get_shard_download_status():
            statuses.append((path, progress))
        
        assert len(statuses) == 1
        assert statuses[0][0] == Path("/tmp/noop_shard")
        assert isinstance(statuses[0][1], RepoDownloadProgress)
        assert statuses[0][1].repo_id == "noop"

    @pytest.mark.asyncio
    async def test_get_shard_download_status_for_shard(self, sample_shard):
        downloader = NoopShardDownloader()
        progress = await downloader.get_shard_download_status_for_shard(sample_shard)
        
        assert isinstance(progress, RepoDownloadProgress)
        assert progress.repo_id == "noop"
        assert progress.shard == sample_shard
        assert progress.status == "complete"


class TestShardDownloader:
    def test_shard_downloader_is_abstract(self):
        # Cannot instantiate abstract class
        with pytest.raises(TypeError):
            ShardDownloader()

    @pytest.mark.asyncio
    async def test_shard_downloader_get_shard_download_status_abstract(self):
        # The abstract method has a default implementation that yields a noop result
        # But we can't call it directly since the class is abstract
        # This test verifies the abstract method exists
        assert hasattr(ShardDownloader, "get_shard_download_status")

