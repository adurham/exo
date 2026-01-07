
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call
import pytest
from exo.worker.download.download_utils import _download_file

@pytest.mark.asyncio
async def test_download_file_reports_cumulative_progress(tmp_path):
    """
    Verify that _download_file reports cumulative downloaded bytes to on_progress callback.
    This ensures the fix for negative/incorrect progress bars is tested.
    """
    target_dir = tmp_path / "models"
    target_dir.mkdir()
    path = "model.safetensors"
    
    # Mock file metadata
    async def mock_file_meta(*args, **kwargs):
        return 200, "fake-etag"

    # Mock session and response
    mock_session = MagicMock()
    mock_response = AsyncMock()
    mock_response.status = 200
    
    # Simulate 2 chunks of 10 bytes each
    chunk1 = b"0123456789"
    chunk2 = b"abcdefghij"
    
    # Setup iter_chunked to yield chunks
    async def iter_chunked(size):
        yield chunk1
        yield chunk2
        
    mock_response.content.iter_chunked = iter_chunked
    mock_session.get.return_value.__aenter__.return_value = mock_response

    # Mock other dependencies
    with patch("exo.worker.download.download_utils.file_meta", side_effect=mock_file_meta), \
         patch("exo.worker.download.download_utils.calc_hash", return_value="fake-etag"), \
         patch("exo.worker.download.download_utils.get_hf_token", return_value="token"), \
         patch("exo.worker.download.download_utils.aios.rename", new_callable=AsyncMock) as mock_rename:
        
        on_progress = MagicMock()
        
        await _download_file(
            repo_id="model1",
            revision="main",
            path=path,
            target_dir=target_dir,
            on_progress=on_progress,
            session=mock_session
        )

        # Total length expected is 200 (from mock_file_meta return value)
        # But our chunks are only 20 bytes total. Ideally mocking length to match chunks is cleaner,
        # but logic uses length from metadata.
        
        # Verify calls
        # 1st call: 10 bytes (chunk1)
        # 2nd call: 20 bytes (chunk1 + chunk2) -> cumulative!
        # Final call: 200 bytes (completion)
        
        expected_calls = [
            call(10, 200, False),
            call(20, 200, False),
            call(200, 200, True),
        ]
        
        on_progress.assert_has_calls(expected_calls)
