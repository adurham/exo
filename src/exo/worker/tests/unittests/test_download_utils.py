
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

@pytest.mark.asyncio
async def test_delete_model_instant_trash(tmp_path):
    """
    Verify that delete_model renames the directory immediately (instant for user)
    and then deletes it in the background.
    """
    from exo.worker.download.download_utils import delete_model
    import shutil
    import os
    
    # Needs to patch ensure_models_dir to use tmp_path
    with patch("exo.worker.download.download_utils.EXO_MODELS_DIR", tmp_path):
        # Create a dummy model
        repo_id = "test/model"
        model_name = repo_id.replace("/", "--")
        model_dir = tmp_path / model_name
        model_dir.mkdir()
        (model_dir / "weights.bin").write_bytes(b"0" * 1024)
        
        assert model_dir.exists()
        
        # Call delete_model
        # We need to verify it returns quickly, but in a unit test checking filesystem state is better
        result = await delete_model(repo_id)
        
        assert result is True
        assert not model_dir.exists(), "Model directory should be gone immediately (renamed)"
        
        # Check that a trash directory exists (briefly) or was deleted
        # Since the background task runs immediately, it might be gone already or still there.
        # But we want to ensure the background task was at least scheduled.
        
        # Let's wait for background tasks to finish.
        # Since _background_delete is a fire-and-forget task, we can't await it directly easily without returning it.
        # But we can wait for the file to disappear.
        
        async def wait_for_trash_cleanup():
            for _ in range(50):
                trash_dirs = list(tmp_path.glob(f".trash_*_{model_name}"))
                if not trash_dirs:
                    return True
                await asyncio.sleep(0.01)
            return False

        cleanup_success = await wait_for_trash_cleanup()
        assert cleanup_success, "Trash directory was not cleaned up in background"

