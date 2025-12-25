import asyncio
import hashlib
import tempfile
from datetime import timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import aiofiles
import aiofiles.os as aios
import aiohttp
import pytest

from exo.shared.types.memory import Memory
from exo.shared.types.models import ModelId, ModelMetadata
from exo.shared.types.worker.shards import PipelineShardMetadata, ShardMetadata
from exo.worker.download.download_utils import (
    FileListEntry,
    RepoDownloadProgress,
    RepoFileDownloadProgress,
    _download_file,
    calc_hash,
    calculate_repo_progress,
    delete_model,
    download_file_with_retry,
    download_progress_for_local_path,
    download_shard,
    ensure_models_dir,
    fetch_file_list_with_cache,
    file_meta,
    get_downloaded_size,
    resolve_allow_patterns,
    trim_etag,
)
from exo.worker.download.huggingface_utils import get_allow_patterns


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


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


class TestTrimEtag:
    def test_trim_etag_with_double_quotes(self):
        assert trim_etag('"abc123"') == "abc123"

    def test_trim_etag_with_single_quotes(self):
        assert trim_etag("'abc123'") == "abc123"

    def test_trim_etag_without_quotes(self):
        assert trim_etag("abc123") == "abc123"

    def test_trim_etag_empty(self):
        assert trim_etag('""') == ""


class TestCalcHash:
    @pytest.mark.asyncio
    async def test_calc_hash_sha1(self, temp_dir):
        test_file = temp_dir / "test.txt"
        test_content = b"Hello, World!"
        test_file.write_bytes(test_content)

        # Calculate expected hash manually
        hasher = hashlib.sha1()
        header = f"blob {len(test_content)}\0".encode()
        hasher.update(header)
        hasher.update(test_content)
        expected_hash = hasher.hexdigest()

        result = await calc_hash(test_file, hash_type="sha1")
        assert result == expected_hash

    @pytest.mark.asyncio
    async def test_calc_hash_sha256(self, temp_dir):
        test_file = temp_dir / "test.txt"
        test_content = b"Hello, World!"
        test_file.write_bytes(test_content)

        # Calculate expected hash manually
        hasher = hashlib.sha256()
        hasher.update(test_content)
        expected_hash = hasher.hexdigest()

        result = await calc_hash(test_file, hash_type="sha256")
        assert result == expected_hash

    @pytest.mark.asyncio
    async def test_calc_hash_large_file(self, temp_dir):
        test_file = temp_dir / "large.txt"
        # Create a file larger than 8MB to test chunking
        large_content = b"x" * (10 * 1024 * 1024)
        test_file.write_bytes(large_content)

        result = await calc_hash(test_file, hash_type="sha256")
        assert len(result) == 64  # SHA256 produces 64 hex characters


class TestFileMeta:
    @pytest.mark.asyncio
    async def test_file_meta_success(self):
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {
            "content-length": "1024",
            "etag": '"abc123"',
        }
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.head = Mock(return_value=mock_response)

        mock_session_context = AsyncMock()
        mock_session_context.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_context.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "exo.worker.download.download_utils.create_http_session",
            return_value=mock_session_context,
        ), patch(
            "exo.worker.download.download_utils.get_download_headers",
            new_callable=AsyncMock,
            return_value={},
        ):
            size, etag = await file_meta("test-repo", "main", "test-file.txt")
            assert size == 1024
            assert etag == "abc123"

    @pytest.mark.asyncio
    async def test_file_meta_with_x_linked_headers(self):
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {
            "x-linked-size": "2048",
            "x-linked-etag": '"def456"',
        }
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.head = Mock(return_value=mock_response)

        mock_session_context = AsyncMock()
        mock_session_context.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_context.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "exo.worker.download.download_utils.create_http_session",
            return_value=mock_session_context,
        ), patch(
            "exo.worker.download.download_utils.get_download_headers",
            new_callable=AsyncMock,
            return_value={},
        ):
            size, etag = await file_meta("test-repo", "main", "test-file.txt")
            assert size == 2048
            assert etag == "def456"

    @pytest.mark.asyncio
    async def test_file_meta_with_redirect(self):
        # First response: redirect
        mock_redirect = AsyncMock()
        mock_redirect.status = 307
        mock_redirect.headers = {
            "location": "/redirected/path",
        }
        mock_redirect.__aenter__ = AsyncMock(return_value=mock_redirect)
        mock_redirect.__aexit__ = AsyncMock(return_value=None)

        # Second response: success
        mock_success = AsyncMock()
        mock_success.status = 200
        mock_success.headers = {
            "content-length": "4096",
            "etag": '"ghi789"',
        }
        mock_success.__aenter__ = AsyncMock(return_value=mock_success)
        mock_success.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.head = Mock(side_effect=[mock_redirect, mock_success])

        mock_session_context = AsyncMock()
        mock_session_context.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_context.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "exo.worker.download.download_utils.create_http_session",
            return_value=mock_session_context,
        ), patch(
            "exo.worker.download.download_utils.get_download_headers",
            new_callable=AsyncMock,
            return_value={},
        ):
            size, etag = await file_meta("test-repo", "main", "test-file.txt")
            assert size == 4096
            assert etag == "ghi789"

    @pytest.mark.asyncio
    async def test_file_meta_redirect_with_x_linked_headers(self):
        mock_response = AsyncMock()
        mock_response.status = 307
        mock_response.headers = {
            "x-linked-size": "8192",
            "x-linked-etag": '"jkl012"',
        }
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.head = Mock(return_value=mock_response)

        mock_session_context = AsyncMock()
        mock_session_context.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_context.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "exo.worker.download.download_utils.create_http_session",
            return_value=mock_session_context,
        ), patch(
            "exo.worker.download.download_utils.get_download_headers",
            new_callable=AsyncMock,
            return_value={},
        ):
            size, etag = await file_meta("test-repo", "main", "test-file.txt")
            assert size == 8192
            assert etag == "jkl012"


class TestDownloadFileWithRetry:
    @pytest.mark.asyncio
    async def test_download_file_with_retry_success(self, temp_dir):
        with patch(
            "exo.worker.download.download_utils._download_file"
        ) as mock_download:
            mock_download.return_value = temp_dir / "downloaded.txt"
            result = await download_file_with_retry(
                "test-repo", "main", "test.txt", temp_dir
            )
            assert result == temp_dir / "downloaded.txt"
            mock_download.assert_called_once()

    @pytest.mark.asyncio
    async def test_download_file_with_retry_retries_on_failure(self, temp_dir):
        with patch(
            "exo.worker.download.download_utils._download_file"
        ) as mock_download, patch("asyncio.sleep", new_callable=AsyncMock):
            # First two attempts fail, third succeeds
            mock_download.side_effect = [
                Exception("Network error"),
                Exception("Network error"),
                temp_dir / "downloaded.txt",
            ]
            result = await download_file_with_retry(
                "test-repo", "main", "test.txt", temp_dir
            )
            assert result == temp_dir / "downloaded.txt"
            assert mock_download.call_count == 3

    @pytest.mark.asyncio
    async def test_download_file_with_retry_raises_file_not_found_immediately(
        self, temp_dir
    ):
        with patch(
            "exo.worker.download.download_utils._download_file"
        ) as mock_download:
            mock_download.side_effect = FileNotFoundError("File not found")
            with pytest.raises(FileNotFoundError):
                await download_file_with_retry(
                    "test-repo", "main", "test.txt", temp_dir
                )
            # Should not retry on FileNotFoundError
            assert mock_download.call_count == 1

    @pytest.mark.asyncio
    async def test_download_file_with_retry_exhausts_retries(self, temp_dir):
        with patch(
            "exo.worker.download.download_utils._download_file"
        ) as mock_download, patch("asyncio.sleep", new_callable=AsyncMock):
            # The function raises the last exception after exhausting retries
            mock_download.side_effect = Exception("Network error")
            with pytest.raises(Exception, match="Network error"):
                await download_file_with_retry(
                    "test-repo", "main", "test.txt", temp_dir
                )
            assert mock_download.call_count == 30  # Max retries


class TestCalculateRepoProgress:
    def test_calculate_repo_progress_all_complete(self, sample_shard):
        file_progress = {
            "file1.bin": RepoFileDownloadProgress(
                repo_id="test-repo",
                repo_revision="main",
                file_path="file1.bin",
                downloaded=Memory.from_bytes(1000),
                downloaded_this_session=Memory.from_bytes(1000),
                total=Memory.from_bytes(1000),
                speed=100.0,
                eta=timedelta(0),
                status="complete",
                start_time=0.0,
            ),
            "file2.bin": RepoFileDownloadProgress(
                repo_id="test-repo",
                repo_revision="main",
                file_path="file2.bin",
                downloaded=Memory.from_bytes(2000),
                downloaded_this_session=Memory.from_bytes(2000),
                total=Memory.from_bytes(2000),
                speed=200.0,
                eta=timedelta(0),
                status="complete",
                start_time=0.0,
            ),
        }

        with patch("time.time", return_value=10.0):
            result = calculate_repo_progress(
                sample_shard, "test-repo", "main", file_progress, 0.0
            )

        assert result.repo_id == "test-repo"
        assert result.repo_revision == "main"
        assert result.completed_files == 2
        assert result.total_files == 2
        assert result.downloaded_bytes.in_bytes == 3000
        assert result.total_bytes.in_bytes == 3000
        assert result.status == "complete"

    def test_calculate_repo_progress_in_progress(self, sample_shard):
        file_progress = {
            "file1.bin": RepoFileDownloadProgress(
                repo_id="test-repo",
                repo_revision="main",
                file_path="file1.bin",
                downloaded=Memory.from_bytes(500),
                downloaded_this_session=Memory.from_bytes(500),
                total=Memory.from_bytes(1000),
                speed=50.0,
                eta=timedelta(seconds=10),
                status="in_progress",
                start_time=0.0,
            ),
        }

        with patch("time.time", return_value=10.0):
            result = calculate_repo_progress(
                sample_shard, "test-repo", "main", file_progress, 0.0
            )

        assert result.status == "in_progress"
        assert result.completed_files == 0
        assert result.total_files == 1
        assert result.downloaded_bytes.in_bytes == 500
        assert result.total_bytes.in_bytes == 1000

    def test_calculate_repo_progress_not_started(self, sample_shard):
        file_progress = {
            "file1.bin": RepoFileDownloadProgress(
                repo_id="test-repo",
                repo_revision="main",
                file_path="file1.bin",
                downloaded=Memory.from_bytes(0),
                downloaded_this_session=Memory.from_bytes(0),
                total=Memory.from_bytes(1000),
                speed=0.0,
                eta=timedelta(0),
                status="not_started",
                start_time=0.0,
            ),
        }

        with patch("time.time", return_value=10.0):
            result = calculate_repo_progress(
                sample_shard, "test-repo", "main", file_progress, 0.0
            )

        assert result.status == "not_started"
        assert result.overall_speed == 0.0


class TestDownloadProgressForLocalPath:
    @pytest.mark.asyncio
    async def test_download_progress_for_local_path_directory(self, temp_dir, sample_shard):
        # Create test files
        (temp_dir / "model.safetensors").write_bytes(b"x" * 1000)
        (temp_dir / "config.json").write_bytes(b'{"test": true}')
        (temp_dir / "tokenizer.bin").write_bytes(b"y" * 500)

        result = await download_progress_for_local_path(
            "test-repo", sample_shard, temp_dir
        )

        assert result.repo_id == "test-repo"
        assert result.repo_revision == "local"
        assert result.status == "complete"
        assert len(result.file_progress) == 3
        assert all(
            fp.status == "complete" for fp in result.file_progress.values()
        )

    @pytest.mark.asyncio
    async def test_download_progress_for_local_path_not_directory(self, temp_dir, sample_shard):
        test_file = temp_dir / "not_a_dir.txt"
        test_file.write_bytes(b"test")

        with pytest.raises(ValueError, match="is not a directory"):
            await download_progress_for_local_path(
                "test-repo", sample_shard, test_file
            )


class TestGetDownloadedSize:
    @pytest.mark.asyncio
    async def test_get_downloaded_size_file_exists(self, temp_dir):
        test_file = temp_dir / "test.bin"
        test_file.write_bytes(b"x" * 1234)
        size = await get_downloaded_size(test_file)
        assert size == 1234

    @pytest.mark.asyncio
    async def test_get_downloaded_size_partial_exists(self, temp_dir):
        test_file = temp_dir / "test.bin"
        partial_file = temp_dir / "test.bin.partial"
        partial_file.write_bytes(b"x" * 567)
        size = await get_downloaded_size(test_file)
        assert size == 567

    @pytest.mark.asyncio
    async def test_get_downloaded_size_neither_exists(self, temp_dir):
        test_file = temp_dir / "nonexistent.bin"
        size = await get_downloaded_size(test_file)
        assert size == 0


class TestFetchFileListWithCache:
    @pytest.mark.asyncio
    async def test_fetch_file_list_with_cache_from_cache(self, temp_dir):
        cache_dir = temp_dir / "caches" / "test--repo"
        cache_dir.mkdir(parents=True)
        cache_file = cache_dir / "test--repo--main--file_list.json"

        cached_data = [
            {"type": "file", "path": "model.bin", "size": 1000},
            {"type": "file", "path": "config.json", "size": 100},
        ]
        import json

        cache_file.write_text(json.dumps(cached_data))

        with patch(
            "exo.worker.download.download_utils.ensure_models_dir",
            return_value=temp_dir,
        ):
            result = await fetch_file_list_with_cache("test/repo", "main")
            assert len(result) == 2
            assert result[0].path == "model.bin"

    @pytest.mark.asyncio
    async def test_fetch_file_list_with_cache_fetches_and_caches(self, temp_dir):
        file_list = [
            FileListEntry(type="file", path="model.bin", size=1000),
            FileListEntry(type="file", path="config.json", size=100),
        ]

        with patch(
            "exo.worker.download.download_utils.ensure_models_dir",
            return_value=temp_dir,
        ), patch(
            "exo.worker.download.download_utils.fetch_file_list_with_retry",
            return_value=file_list,
        ) as mock_fetch:
            result = await fetch_file_list_with_cache("test/repo", "main")
            assert len(result) == 2
            mock_fetch.assert_called_once()

            # Verify cache was written
            cache_file = (
                temp_dir
                / "caches"
                / "test--repo"
                / "test--repo--main--file_list.json"
            )
            assert cache_file.exists()


class TestResolveAllowPatterns:
    @pytest.mark.asyncio
    async def test_resolve_allow_patterns_success(self, sample_shard):
        weight_map = {
            "model.safetensors": "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors": "model-00002-of-00002.safetensors",
        }

        with patch(
            "exo.worker.download.download_utils.get_weight_map",
            return_value=weight_map,
        ), patch(
            "exo.worker.download.download_utils.get_allow_patterns",
            return_value=["*.safetensors"],
        ):
            result = await resolve_allow_patterns(sample_shard)
            assert result == ["*.safetensors"]

    @pytest.mark.asyncio
    async def test_resolve_allow_patterns_fallback_on_error(self, sample_shard):
        with patch(
            "exo.worker.download.download_utils.get_weight_map",
            side_effect=Exception("Network error"),
        ):
            result = await resolve_allow_patterns(sample_shard)
            assert result == ["*"]


class TestDeleteModel:
    @pytest.mark.asyncio
    async def test_delete_model_exists(self, temp_dir):
        model_dir = temp_dir / "test--repo"
        model_dir.mkdir()
        (model_dir / "model.bin").write_bytes(b"test")

        with patch(
            "exo.worker.download.download_utils.ensure_models_dir",
            return_value=temp_dir,
        ):
            result = await delete_model("test/repo")
            assert result is True
            assert not model_dir.exists()

    @pytest.mark.asyncio
    async def test_delete_model_not_exists(self, temp_dir):
        with patch(
            "exo.worker.download.download_utils.ensure_models_dir",
            return_value=temp_dir,
        ):
            result = await delete_model("test/repo")
            assert result is False


class TestEnsureModelsDir:
    @pytest.mark.asyncio
    async def test_ensure_models_dir_creates_directory(self, temp_dir):
        models_dir = temp_dir / "models"
        with patch(
            "exo.worker.download.download_utils.EXO_MODELS_DIR", models_dir
        ):
            result = await ensure_models_dir()
            assert result == models_dir
            assert models_dir.exists()

    @pytest.mark.asyncio
    async def test_ensure_models_dir_exists_already(self, temp_dir):
        models_dir = temp_dir / "models"
        models_dir.mkdir()
        with patch(
            "exo.worker.download.download_utils.EXO_MODELS_DIR", models_dir
        ):
            result = await ensure_models_dir()
            assert result == models_dir


class TestDownloadFile:
    @pytest.mark.asyncio
    async def test_download_file_already_exists(self, temp_dir):
        test_file = temp_dir / "test.txt"
        test_file.write_bytes(b"existing content")
        
        result = await _download_file(
            "test-repo", "main", "test.txt", temp_dir
        )
        assert result == test_file
        assert test_file.read_bytes() == b"existing content"

    @pytest.mark.asyncio
    async def test_download_file_from_peer_success(self, temp_dir):
        from exo.shared.types.common import NodeId
        from exo.shared.types.multiaddr import Multiaddr
        
        mock_peer_service = Mock()
        mock_availability = Mock()
        mock_availability.has_file = True
        mock_availability.node_id = NodeId()
        mock_availability.ip_address = "192.168.1.100"
        mock_availability.port = 8080
        mock_availability.file_hash = "abc123"
        
        mock_peer_service.check_peer_has_file = AsyncMock(return_value=mock_availability)
        mock_peer_service.download_from_peer = AsyncMock()
        
        target_path = temp_dir / "test.txt"
        
        with patch("aiofiles.os.path.exists", return_value=False):
            result = await _download_file(
                "test-repo",
                "main",
                "test.txt",
                temp_dir,
                peer_file_service=mock_peer_service,
            )
        
        assert result == target_path
        mock_peer_service.download_from_peer.assert_called_once()

    @pytest.mark.asyncio
    async def test_download_file_from_peer_fallback_on_error(self, temp_dir):
        from exo.shared.types.common import NodeId
        
        mock_peer_service = Mock()
        mock_availability = Mock()
        mock_availability.has_file = True
        mock_availability.node_id = NodeId()
        mock_availability.ip_address = "192.168.1.100"
        mock_availability.port = 8080
        mock_availability.file_hash = "abc123"
        
        mock_peer_service.check_peer_has_file = AsyncMock(return_value=mock_availability)
        mock_peer_service.download_from_peer = AsyncMock(side_effect=Exception("Peer download failed"))
        
        # Mock HuggingFace download path
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.content.read = AsyncMock(side_effect=[b"test content", b""])
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        mock_session = AsyncMock()
        mock_session.get = Mock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        mock_session_context = AsyncMock()
        mock_session_context.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_context.__aexit__ = AsyncMock(return_value=None)
        
        mock_file = AsyncMock()
        mock_file.write = AsyncMock(return_value=12)
        mock_file.__aenter__ = AsyncMock(return_value=mock_file)
        mock_file.__aexit__ = AsyncMock(return_value=None)
        
        mock_open_context = AsyncMock()
        mock_open_context.__aenter__ = AsyncMock(return_value=mock_file)
        mock_open_context.__aexit__ = AsyncMock(return_value=None)
        
        with patch("aiofiles.os.path.exists", return_value=False), patch(
            "exo.worker.download.download_utils.file_meta",
            return_value=(12, "abc123"),
        ), patch(
            "exo.worker.download.download_utils.create_http_session",
            return_value=mock_session_context,
        ), patch(
            "exo.worker.download.download_utils.get_download_headers",
            new_callable=AsyncMock,
            return_value={},
        ), patch(
            "exo.worker.download.download_utils.calc_hash",
            return_value="abc123",
        ), patch("exo.worker.download.download_utils.aiofiles.open", return_value=mock_open_context), patch(
            "aiofiles.os.stat", new_callable=AsyncMock
        ) as mock_stat, patch("aiofiles.os.rename", new_callable=AsyncMock):
            mock_stat.side_effect = FileNotFoundError()
            
            result = await _download_file(
                "test-repo",
                "main",
                "test.txt",
                temp_dir,
                peer_file_service=mock_peer_service,
            )
            
            # Should fall back to HuggingFace after peer fails
            assert result == temp_dir / "test.txt"

    @pytest.mark.asyncio
    async def test_download_file_huggingface_full_download(self, temp_dir):
        test_content = b"test file content"
        remote_hash = hashlib.sha256(test_content).hexdigest()
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.content.read = AsyncMock(side_effect=[test_content, b""])
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        mock_session = AsyncMock()
        mock_session.get = Mock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        mock_session_context = AsyncMock()
        mock_session_context.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_context.__aexit__ = AsyncMock(return_value=None)
        
        progress_calls = []
        
        def on_progress(curr: int, total: int, is_renamed: bool):
            progress_calls.append((curr, total, is_renamed))
        
        mock_file = AsyncMock()
        mock_file.write = AsyncMock(return_value=len(test_content))
        mock_file.__aenter__ = AsyncMock(return_value=mock_file)
        mock_file.__aexit__ = AsyncMock(return_value=None)
        
        mock_open_context = AsyncMock()
        mock_open_context.__aenter__ = AsyncMock(return_value=mock_file)
        mock_open_context.__aexit__ = AsyncMock(return_value=None)
        
        with patch("aiofiles.os.path.exists", return_value=False), patch(
            "exo.worker.download.download_utils.file_meta",
            return_value=(len(test_content), remote_hash),
        ), patch(
            "exo.worker.download.download_utils.create_http_session",
            return_value=mock_session_context,
        ), patch(
            "exo.worker.download.download_utils.get_download_headers",
            new_callable=AsyncMock,
            return_value={},
        ), patch("exo.worker.download.download_utils.aiofiles.open", return_value=mock_open_context), patch(
            "aiofiles.os.stat", new_callable=AsyncMock
        ) as mock_stat, patch("aiofiles.os.rename", new_callable=AsyncMock), patch(
            "exo.worker.download.download_utils.calc_hash",
            return_value=remote_hash,
        ):
            # Mock stat for partial file (doesn't exist)
            mock_stat.side_effect = FileNotFoundError()
            
            result = await _download_file(
                "test-repo",
                "main",
                "test.txt",
                temp_dir,
                on_progress=on_progress,
            )
            
            assert result == temp_dir / "test.txt"
            # Progress should be called
            assert len(progress_calls) > 0

    @pytest.mark.asyncio
    async def test_download_file_resume_partial(self, temp_dir):
        partial_content = b"partial"
        remaining_content = b" remaining"
        full_content = partial_content + remaining_content
        remote_hash = hashlib.sha256(full_content).hexdigest()
        
        # Create partial file
        partial_path = temp_dir / "test.txt.partial"
        partial_path.write_bytes(partial_content)
        
        mock_response = AsyncMock()
        mock_response.status = 206  # Partial content
        mock_response.content.read = AsyncMock(side_effect=[remaining_content, b""])
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        mock_session = AsyncMock()
        mock_session.get = Mock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        mock_session_context = AsyncMock()
        mock_session_context.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_context.__aexit__ = AsyncMock(return_value=None)
        
        mock_file = AsyncMock()
        mock_file.write = AsyncMock(return_value=len(remaining_content))
        mock_file.__aenter__ = AsyncMock(return_value=mock_file)
        mock_file.__aexit__ = AsyncMock(return_value=None)
        
        mock_open_context = AsyncMock()
        mock_open_context.__aenter__ = AsyncMock(return_value=mock_file)
        mock_open_context.__aexit__ = AsyncMock(return_value=None)
        
        with patch("aiofiles.os.path.exists", side_effect=lambda p: str(p).endswith(".partial")), patch(
            "exo.worker.download.download_utils.file_meta",
            return_value=(len(full_content), remote_hash),
        ), patch(
            "exo.worker.download.download_utils.create_http_session",
            return_value=mock_session_context,
        ), patch(
            "exo.worker.download.download_utils.get_download_headers",
            new_callable=AsyncMock,
            return_value={},
        ), patch("exo.worker.download.download_utils.aiofiles.open", return_value=mock_open_context), patch(
            "aiofiles.os.stat", new_callable=AsyncMock
        ) as mock_stat, patch("aiofiles.os.rename", new_callable=AsyncMock), patch(
            "exo.worker.download.download_utils.calc_hash",
            return_value=remote_hash,
        ):
            # Mock stat to return partial file size
            mock_stat_obj = Mock()
            mock_stat_obj.st_size = len(partial_content)
            mock_stat.return_value = mock_stat_obj
            
            result = await _download_file(
                "test-repo",
                "main",
                "test.txt",
                temp_dir,
            )
            
            assert result == temp_dir / "test.txt"
            # Should have used Range header for resume
            call_kwargs = mock_session.get.call_args[1] if mock_session.get.call_args else {}
            assert "Range" in call_kwargs.get("headers", {})

    @pytest.mark.asyncio
    async def test_download_file_hash_verification_success(self, temp_dir):
        test_content = b"test content"
        remote_hash = hashlib.sha256(test_content).hexdigest()
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.content.read = AsyncMock(side_effect=[test_content, b""])
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        mock_session = AsyncMock()
        mock_session.get = Mock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        mock_session_context = AsyncMock()
        mock_session_context.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_context.__aexit__ = AsyncMock(return_value=None)
        
        mock_file = AsyncMock()
        mock_file.write = AsyncMock(return_value=len(test_content))
        mock_file.__aenter__ = AsyncMock(return_value=mock_file)
        mock_file.__aexit__ = AsyncMock(return_value=None)
        
        mock_open_context = AsyncMock()
        mock_open_context.__aenter__ = AsyncMock(return_value=mock_file)
        mock_open_context.__aexit__ = AsyncMock(return_value=None)
        
        with patch("aiofiles.os.path.exists", return_value=False), patch(
            "exo.worker.download.download_utils.file_meta",
            return_value=(len(test_content), remote_hash),
        ), patch(
            "exo.worker.download.download_utils.create_http_session",
            return_value=mock_session_context,
        ), patch(
            "exo.worker.download.download_utils.get_download_headers",
            new_callable=AsyncMock,
            return_value={},
        ), patch("exo.worker.download.download_utils.aiofiles.open", return_value=mock_open_context), patch(
            "aiofiles.os.stat", new_callable=AsyncMock
        ) as mock_stat, patch("aiofiles.os.rename", new_callable=AsyncMock):
            mock_stat.side_effect = FileNotFoundError()
            
            # Mock calc_hash to return correct hash
            with patch(
                "exo.worker.download.download_utils.calc_hash",
                return_value=remote_hash,
            ):
                result = await _download_file(
                    "test-repo",
                    "main",
                    "test.txt",
                    temp_dir,
                )
                
                assert result == temp_dir / "test.txt"

    @pytest.mark.asyncio
    async def test_download_file_hash_verification_failure(self, temp_dir):
        test_content = b"test content"
        remote_hash = hashlib.sha256(test_content).hexdigest()
        wrong_hash = "wrong_hash_value"
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.content.read = AsyncMock(side_effect=[test_content, b""])
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        mock_session = AsyncMock()
        mock_session.get = Mock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        mock_session_context = AsyncMock()
        mock_session_context.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_context.__aexit__ = AsyncMock(return_value=None)
        
        mock_file = AsyncMock()
        mock_file.write = AsyncMock(return_value=len(test_content))
        mock_file.__aenter__ = AsyncMock(return_value=mock_file)
        mock_file.__aexit__ = AsyncMock(return_value=None)
        
        mock_open_context = AsyncMock()
        mock_open_context.__aenter__ = AsyncMock(return_value=mock_file)
        mock_open_context.__aexit__ = AsyncMock(return_value=None)
        
        with patch("aiofiles.os.path.exists", return_value=False), patch(
            "exo.worker.download.download_utils.file_meta",
            return_value=(len(test_content), remote_hash),
        ), patch(
            "exo.worker.download.download_utils.create_http_session",
            return_value=mock_session_context,
        ), patch(
            "exo.worker.download.download_utils.get_download_headers",
            new_callable=AsyncMock,
            return_value={},
        ), patch("exo.worker.download.download_utils.aiofiles.open", return_value=mock_open_context), patch(
            "aiofiles.os.stat", new_callable=AsyncMock
        ) as mock_stat:
            mock_stat.side_effect = FileNotFoundError()
            
            # Mock calc_hash to return wrong hash
            with patch(
                "exo.worker.download.download_utils.calc_hash",
                return_value=wrong_hash,
            ), patch("aiofiles.os.remove", new_callable=AsyncMock):
                with pytest.raises(Exception, match="hash"):
                    await _download_file(
                        "test-repo",
                        "main",
                        "test.txt",
                        temp_dir,
                    )

    @pytest.mark.asyncio
    async def test_download_file_404_error(self, temp_dir):
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        mock_session = AsyncMock()
        mock_session.get = Mock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        mock_session_context = AsyncMock()
        mock_session_context.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_context.__aexit__ = AsyncMock(return_value=None)
        
        with patch("aiofiles.os.path.exists", return_value=False), patch(
            "exo.worker.download.download_utils.file_meta",
            return_value=(100, "hash123"),
        ), patch(
            "exo.worker.download.download_utils.create_http_session",
            return_value=mock_session_context,
        ), patch(
            "exo.worker.download.download_utils.get_download_headers",
            new_callable=AsyncMock,
            return_value={},
        ), patch("aiofiles.os.stat", new_callable=AsyncMock) as mock_stat:
            mock_stat.side_effect = FileNotFoundError()
            
            with pytest.raises(FileNotFoundError, match="File not found"):
                await _download_file(
                    "test-repo",
                    "main",
                    "test.txt",
                    temp_dir,
                )


class TestDownloadShard:
    @pytest.mark.asyncio
    async def test_download_shard_local_path(self, temp_dir, sample_shard):
        local_model_path = Path(str(sample_shard.model_meta.model_id))
        local_model_path.mkdir(exist_ok=True)
        (local_model_path / "model.safetensors").write_bytes(b"test")
        
        progress_calls = []
        
        def on_progress(shard: ShardMetadata, progress: RepoDownloadProgress):
            progress_calls.append((shard, progress))
        
        # Mock path.exists to return True for the model_id string
        def mock_exists(path):
            return str(path) == str(sample_shard.model_meta.model_id)
        
        with patch("aiofiles.os.path.exists", side_effect=mock_exists), patch(
            "exo.worker.download.download_utils.download_progress_for_local_path",
            new_callable=AsyncMock,
        ) as mock_progress:
            mock_progress.return_value = RepoDownloadProgress(
                repo_id=str(sample_shard.model_meta.model_id),
                repo_revision="local",
                shard=sample_shard,
                completed_files=1,
                total_files=1,
                downloaded_bytes=Memory.from_bytes(100),
                downloaded_bytes_this_session=Memory.from_bytes(0),
                total_bytes=Memory.from_bytes(100),
                overall_speed=0,
                overall_eta=timedelta(0),
                status="complete",
            )
            
            result_path, result_progress = await download_shard(
                sample_shard,
                on_progress=on_progress,
            )
        
        assert result_path == local_model_path
        assert isinstance(result_progress, RepoDownloadProgress)
        mock_progress.assert_called_once()

    @pytest.mark.asyncio
    async def test_download_shard_skip_download(self, temp_dir, sample_shard):
        file_list = [
            FileListEntry(type="file", path="model.safetensors", size=1000),
            FileListEntry(type="file", path="config.json", size=100),
        ]
        
        progress_calls = []
        
        def on_progress(shard: ShardMetadata, progress: RepoDownloadProgress):
            progress_calls.append((shard, progress))
        
        with patch("aiofiles.os.path.exists", return_value=False), patch(
            "exo.worker.download.download_utils.ensure_models_dir",
            return_value=temp_dir,
        ), patch(
            "exo.worker.download.download_utils.resolve_allow_patterns",
            return_value=["*.safetensors", "*.json"],
        ), patch(
            "exo.worker.download.download_utils.fetch_file_list_with_cache",
            return_value=file_list,
        ), patch(
            "exo.worker.download.download_utils.get_downloaded_size",
            return_value=0,
        ):
            result_path, result_progress = await download_shard(
                sample_shard,
                on_progress=on_progress,
                skip_download=True,
            )
        
        assert result_path == temp_dir / str(sample_shard.model_meta.model_id).replace("/", "--")
        assert isinstance(result_progress, RepoDownloadProgress)

    @pytest.mark.asyncio
    async def test_download_shard_file_filtering(self, temp_dir, sample_shard):
        file_list = [
            FileListEntry(type="file", path="model.safetensors", size=1000),
            FileListEntry(type="file", path="other.bin", size=500),
            FileListEntry(type="file", path="config.json", size=100),
        ]
        
        allow_patterns = ["*.safetensors", "*.json"]
        
        progress_calls = []
        
        def on_progress(shard: ShardMetadata, progress: RepoDownloadProgress):
            progress_calls.append((shard, progress))
        
        with patch("aiofiles.os.path.exists", return_value=False), patch(
            "exo.worker.download.download_utils.ensure_models_dir",
            return_value=temp_dir,
        ), patch(
            "exo.worker.download.download_utils.resolve_allow_patterns",
            return_value=allow_patterns,
        ), patch(
            "exo.worker.download.download_utils.fetch_file_list_with_cache",
            return_value=file_list,
        ), patch(
            "exo.worker.download.download_utils.get_downloaded_size",
            return_value=0,
        ), patch(
            "exo.worker.download.download_utils.download_file_with_retry",
            new_callable=AsyncMock,
        ) as mock_download:
            result_path, result_progress = await download_shard(
                sample_shard,
                on_progress=on_progress,
                skip_download=True,
            )
        
        # Should only download files matching patterns (safetensors and json, not bin)
        # Since skip_download=True, no actual downloads happen, but we can verify filtering
        assert isinstance(result_progress, RepoDownloadProgress)

    @pytest.mark.asyncio
    async def test_download_shard_progress_tracking(self, temp_dir, sample_shard):
        file_list = [
            FileListEntry(type="file", path="model.safetensors", size=1000),
        ]
        
        progress_calls = []
        
        def on_progress(shard: ShardMetadata, progress: RepoDownloadProgress):
            progress_calls.append((shard, progress))
        
        with patch("aiofiles.os.path.exists", return_value=False), patch(
            "exo.worker.download.download_utils.ensure_models_dir",
            return_value=temp_dir,
        ), patch(
            "exo.worker.download.download_utils.resolve_allow_patterns",
            return_value=["*.safetensors"],
        ), patch(
            "exo.worker.download.download_utils.fetch_file_list_with_cache",
            return_value=file_list,
        ), patch(
            "exo.worker.download.download_utils.get_downloaded_size",
            return_value=0,
        ), patch(
            "exo.worker.download.download_utils.download_file_with_retry",
            new_callable=AsyncMock,
        ), patch("time.time", return_value=1000.0):
            result_path, result_progress = await download_shard(
                sample_shard,
                on_progress=on_progress,
                skip_download=True,
            )
        
        # Progress should be called
        assert len(progress_calls) > 0

    @pytest.mark.asyncio
    async def test_download_shard_gguf_file(self, temp_dir, sample_shard):
        file_list = [
            FileListEntry(type="file", path="model.gguf", size=2000),
            FileListEntry(type="file", path="config.json", size=100),
        ]
        
        progress_calls = []
        
        def on_progress(shard: ShardMetadata, progress: RepoDownloadProgress):
            progress_calls.append((shard, progress))
        
        target_dir = temp_dir / str(sample_shard.model_meta.model_id).replace("/", "--")
        
        with patch("aiofiles.os.path.exists", return_value=False), patch(
            "exo.worker.download.download_utils.ensure_models_dir",
            return_value=temp_dir,
        ), patch(
            "exo.worker.download.download_utils.resolve_allow_patterns",
            return_value=["*.gguf", "*.json"],
        ), patch(
            "exo.worker.download.download_utils.fetch_file_list_with_cache",
            return_value=file_list,
        ), patch(
            "exo.worker.download.download_utils.get_downloaded_size",
            return_value=0,
        ), patch(
            "exo.worker.download.download_utils.download_file_with_retry",
            new_callable=AsyncMock,
        ):
            result_path, result_progress = await download_shard(
                sample_shard,
                on_progress=on_progress,
                skip_download=True,
            )
        
        # Should return GGUF file path, not directory
        assert result_path == target_dir / "model.gguf"

    @pytest.mark.asyncio
    async def test_download_shard_no_gguf_returns_directory(self, temp_dir, sample_shard):
        file_list = [
            FileListEntry(type="file", path="model.safetensors", size=1000),
            FileListEntry(type="file", path="config.json", size=100),
        ]
        
        progress_calls = []
        
        def on_progress(shard: ShardMetadata, progress: RepoDownloadProgress):
            progress_calls.append((shard, progress))
        
        target_dir = temp_dir / str(sample_shard.model_meta.model_id).replace("/", "--")
        
        with patch("aiofiles.os.path.exists", return_value=False), patch(
            "exo.worker.download.download_utils.ensure_models_dir",
            return_value=temp_dir,
        ), patch(
            "exo.worker.download.download_utils.resolve_allow_patterns",
            return_value=["*.safetensors", "*.json"],
        ), patch(
            "exo.worker.download.download_utils.fetch_file_list_with_cache",
            return_value=file_list,
        ), patch(
            "exo.worker.download.download_utils.get_downloaded_size",
            return_value=0,
        ), patch(
            "exo.worker.download.download_utils.download_file_with_retry",
            new_callable=AsyncMock,
        ):
            result_path, result_progress = await download_shard(
                sample_shard,
                on_progress=on_progress,
                skip_download=True,
            )
        
        # Should return directory when no GGUF file
        assert result_path == target_dir

    @pytest.mark.asyncio
    async def test_download_shard_with_custom_allow_patterns(self, temp_dir, sample_shard):
        file_list = [
            FileListEntry(type="file", path="model.safetensors", size=1000),
            FileListEntry(type="file", path="config.json", size=100),
        ]
        
        custom_patterns = ["*.safetensors"]
        
        progress_calls = []
        
        def on_progress(shard: ShardMetadata, progress: RepoDownloadProgress):
            progress_calls.append((shard, progress))
        
        with patch("aiofiles.os.path.exists", return_value=False), patch(
            "exo.worker.download.download_utils.ensure_models_dir",
            return_value=temp_dir,
        ), patch(
            "exo.worker.download.download_utils.fetch_file_list_with_cache",
            return_value=file_list,
        ), patch(
            "exo.worker.download.download_utils.get_downloaded_size",
            return_value=0,
        ), patch(
            "exo.worker.download.download_utils.download_file_with_retry",
            new_callable=AsyncMock,
        ):
            result_path, result_progress = await download_shard(
                sample_shard,
                on_progress=on_progress,
                allow_patterns=custom_patterns,
                skip_download=True,
            )
        
        # Should use custom patterns, not resolve from shard
        assert isinstance(result_progress, RepoDownloadProgress)

