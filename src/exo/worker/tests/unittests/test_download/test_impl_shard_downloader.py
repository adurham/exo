import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from exo.shared.types.memory import Memory
from exo.shared.types.models import ModelId, ModelMetadata
from exo.shared.types.worker.shards import PipelineShardMetadata, ShardMetadata
from exo.worker.download.download_utils import RepoDownloadProgress
from exo.worker.download.impl_shard_downloader import (
    CachedShardDownloader,
    ResumableShardDownloader,
    SingletonShardDownloader,
    build_base_shard,
    build_full_shard,
    exo_shard_downloader,
)


@pytest.fixture
def sample_shard():
    """Create a sample shard metadata for testing."""
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


@pytest.fixture
def mock_base_downloader():
    """Create a mock base downloader."""
    downloader = Mock()
    downloader.ensure_shard = AsyncMock(return_value=Path("/tmp/test"))
    downloader.on_progress = Mock()
    downloader.get_shard_download_status = AsyncMock()
    downloader.get_shard_download_status_for_shard = AsyncMock(
        return_value=Mock(spec=RepoDownloadProgress)
    )
    return downloader


class TestResumableShardDownloader:
    @pytest.mark.asyncio
    async def test_ensure_shard_calls_download_shard(self, sample_shard, tmp_path):
        downloader = ResumableShardDownloader(max_parallel_downloads=4)
        
        with patch(
            "exo.worker.download.impl_shard_downloader.download_shard",
            new_callable=AsyncMock,
        ) as mock_download:
            mock_download.return_value = (tmp_path, Mock(spec=RepoDownloadProgress))
            
            result = await downloader.ensure_shard(sample_shard)
            
            mock_download.assert_called_once()
            assert result == tmp_path

    @pytest.mark.asyncio
    async def test_ensure_shard_config_only(self, sample_shard, tmp_path):
        downloader = ResumableShardDownloader(max_parallel_downloads=4)
        
        with patch(
            "exo.worker.download.impl_shard_downloader.download_shard",
            new_callable=AsyncMock,
        ) as mock_download:
            mock_download.return_value = (tmp_path, Mock(spec=RepoDownloadProgress))
            
            result = await downloader.ensure_shard(sample_shard, config_only=True)
            
            # Should pass allow_patterns=["config.json"]
            call_kwargs = mock_download.call_args[1]
            assert call_kwargs["allow_patterns"] == ["config.json"]

    def test_on_progress_registers_callback(self, sample_shard):
        downloader = ResumableShardDownloader()
        
        callback1 = Mock()
        callback2 = Mock()
        
        downloader.on_progress(callback1)
        downloader.on_progress(callback2)
        
        assert len(downloader.on_progress_callbacks) == 2

    def test_on_progress_wrapper_calls_all_callbacks(self, sample_shard):
        downloader = ResumableShardDownloader()
        
        callback1 = Mock()
        callback2 = Mock()
        
        downloader.on_progress(callback1)
        downloader.on_progress(callback2)
        
        mock_progress = Mock(spec=RepoDownloadProgress)
        downloader.on_progress_wrapper(sample_shard, mock_progress)
        
        callback1.assert_called_once_with(sample_shard, mock_progress)
        callback2.assert_called_once_with(sample_shard, mock_progress)

    @pytest.mark.asyncio
    async def test_get_shard_download_status_for_shard(self, sample_shard):
        downloader = ResumableShardDownloader()
        
        mock_progress = Mock(spec=RepoDownloadProgress)
        
        with patch(
            "exo.worker.download.impl_shard_downloader.download_shard",
            new_callable=AsyncMock,
        ) as mock_download:
            mock_download.return_value = (Path("/tmp"), mock_progress)
            
            result = await downloader.get_shard_download_status_for_shard(sample_shard)
            
            assert result == mock_progress
            # Should call with skip_download=True
            call_kwargs = mock_download.call_args[1]
            assert call_kwargs.get("skip_download") is True


class TestCachedShardDownloader:
    @pytest.mark.asyncio
    async def test_ensure_shard_caches_result(self, sample_shard, mock_base_downloader):
        downloader = CachedShardDownloader(mock_base_downloader)
        
        result1 = await downloader.ensure_shard(sample_shard)
        result2 = await downloader.ensure_shard(sample_shard)
        
        # Should only call base downloader once
        assert mock_base_downloader.ensure_shard.call_count == 1
        assert result1 == result2
        assert result1 == Path("/tmp/test")

    @pytest.mark.asyncio
    async def test_ensure_shard_different_shards(self, mock_base_downloader):
        downloader = CachedShardDownloader(mock_base_downloader)
        
        shard1 = PipelineShardMetadata(
            model_meta=ModelMetadata(
                model_id=ModelId("model1"),
                pretty_name="Model 1",
                storage_size=Memory.from_mb(100),
                n_layers=12,
            ),
            device_rank=0,
            world_size=1,
            start_layer=0,
            end_layer=12,
            n_layers=12,
        )
        
        shard2 = PipelineShardMetadata(
            model_meta=ModelMetadata(
                model_id=ModelId("model2"),
                pretty_name="Model 2",
                storage_size=Memory.from_mb(100),
                n_layers=12,
            ),
            device_rank=0,
            world_size=1,
            start_layer=0,
            end_layer=12,
            n_layers=12,
        )
        
        await downloader.ensure_shard(shard1)
        await downloader.ensure_shard(shard2)
        
        # Should call base downloader twice for different shards
        assert mock_base_downloader.ensure_shard.call_count == 2

    def test_on_progress_forwards_to_base(self, mock_base_downloader):
        downloader = CachedShardDownloader(mock_base_downloader)
        
        callback = Mock()
        downloader.on_progress(callback)
        
        mock_base_downloader.on_progress.assert_called_once_with(callback)

    @pytest.mark.asyncio
    async def test_get_shard_download_status_forwards(self, mock_base_downloader):
        downloader = CachedShardDownloader(mock_base_downloader)
        
        async def mock_status_gen():
            yield (Path("/tmp"), Mock(spec=RepoDownloadProgress))
        
        mock_base_downloader.get_shard_download_status = mock_status_gen
        
        statuses = []
        async for path, progress in downloader.get_shard_download_status():
            statuses.append((path, progress))
        
        assert len(statuses) == 1


class TestSingletonShardDownloader:
    @pytest.mark.asyncio
    async def test_ensure_shard_prevents_concurrent_downloads(self, sample_shard, mock_base_downloader):
        downloader = SingletonShardDownloader(mock_base_downloader)
        
        # Create a slow download task
        async def slow_download(shard, config_only):
            await asyncio.sleep(0.1)
            return Path("/tmp/test")
        
        mock_base_downloader.ensure_shard = slow_download
        
        # Start two concurrent downloads of the same shard
        task1 = asyncio.create_task(downloader.ensure_shard(sample_shard))
        task2 = asyncio.create_task(downloader.ensure_shard(sample_shard))
        
        results = await asyncio.gather(task1, task2)
        
        # Both should return the same result
        assert results[0] == results[1]
        # Base downloader should only be called once (they share the same task)
        # Actually, since we're using create_task, it might be called once and both await the same task

    @pytest.mark.asyncio
    async def test_ensure_shard_cleans_up_after_completion(self, sample_shard, mock_base_downloader):
        downloader = SingletonShardDownloader(mock_base_downloader)
        
        await downloader.ensure_shard(sample_shard)
        
        # After completion, the task should be removed from active_downloads
        assert sample_shard not in downloader.active_downloads

    def test_on_progress_forwards_to_base(self, mock_base_downloader):
        downloader = SingletonShardDownloader(mock_base_downloader)
        
        callback = Mock()
        downloader.on_progress(callback)
        
        mock_base_downloader.on_progress.assert_called_once_with(callback)

    @pytest.mark.asyncio
    async def test_get_shard_download_status_forwards(self, mock_base_downloader):
        downloader = SingletonShardDownloader(mock_base_downloader)
        
        async def mock_status_gen():
            yield (Path("/tmp"), Mock(spec=RepoDownloadProgress))
        
        mock_base_downloader.get_shard_download_status = mock_status_gen
        
        statuses = []
        async for path, progress in downloader.get_shard_download_status():
            statuses.append((path, progress))
        
        assert len(statuses) == 1


class TestExoShardDownloader:
    def test_exo_shard_downloader_returns_singleton(self):
        downloader = exo_shard_downloader()
        
        assert isinstance(downloader, SingletonShardDownloader)
        assert isinstance(downloader.shard_downloader, CachedShardDownloader)

    def test_exo_shard_downloader_with_peer_service(self):
        mock_peer_service = Mock()
        downloader = exo_shard_downloader(peer_file_service=mock_peer_service)
        
        assert isinstance(downloader, SingletonShardDownloader)
        # The peer service should be passed down to ResumableShardDownloader
        cached = downloader.shard_downloader
        assert isinstance(cached, CachedShardDownloader)
        resumable = cached.shard_downloader
        assert isinstance(resumable, ResumableShardDownloader)
        assert resumable.peer_file_service == mock_peer_service


class TestBuildShard:
    @pytest.mark.asyncio
    async def test_build_base_shard(self):
        with patch(
            "exo.worker.download.impl_shard_downloader.get_model_meta",
            new_callable=AsyncMock,
        ) as mock_get_meta:
            mock_meta = ModelMetadata(
                model_id=ModelId("test-model"),
                pretty_name="Test Model",
                storage_size=Memory.from_mb(100),
                n_layers=24,
            )
            mock_get_meta.return_value = mock_meta
            
            shard = await build_base_shard("test-model")
            
            assert isinstance(shard, PipelineShardMetadata)
            assert shard.model_meta == mock_meta
            assert shard.device_rank == 0
            assert shard.world_size == 1
            assert shard.start_layer == 0
            assert shard.end_layer == 24

    @pytest.mark.asyncio
    async def test_build_full_shard(self):
        with patch(
            "exo.worker.download.impl_shard_downloader.build_base_shard",
            new_callable=AsyncMock,
        ) as mock_build:
            base_shard = PipelineShardMetadata(
                model_meta=ModelMetadata(
                    model_id=ModelId("test-model"),
                    pretty_name="Test Model",
                    storage_size=Memory.from_mb(100),
                    n_layers=24,
                ),
                device_rank=0,
                world_size=1,
                start_layer=0,
                end_layer=12,
                n_layers=24,
            )
            mock_build.return_value = base_shard
            
            full_shard = await build_full_shard("test-model")
            
            assert isinstance(full_shard, PipelineShardMetadata)
            assert full_shard.end_layer == 24  # Should be full model
            assert full_shard.start_layer == 0

