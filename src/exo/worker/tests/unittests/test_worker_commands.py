import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from exo.worker.main import Worker
from exo.shared.types.common import NodeId, SessionId
from exo.shared.types.commands import DeviceDownloadModel, DeleteModel

@pytest.fixture
def mock_worker():
    shard_downloader = MagicMock()
    shard_downloader.get_shard_download_status_for_shard = AsyncMock()
    shard_downloader.ensure_shard = AsyncMock()
    shard_downloader.delete_model = AsyncMock()
    shard_downloader.on_progress = MagicMock()

    worker = Worker(
        node_id=NodeId("node1"),
        session_id=SessionId(master_node_id=NodeId("node1"), election_clock=0),
        shard_downloader=shard_downloader,
        connection_message_receiver=MagicMock(),
        global_event_receiver=MagicMock(),
        local_event_sender=MagicMock(),
        command_sender=MagicMock(),
        command_receiver=MagicMock(),
    )
    worker.event_sender = MagicMock()
    worker.event_sender.send_nowait = MagicMock()
    
    # Mock TaskGroup
    worker._tg = MagicMock()
    worker._tg.start_soon = MagicMock()

    # Mock State
    worker.state = MagicMock()
    worker.state.downloads = {}
    worker.download_status = {}
    
    return worker

@pytest.mark.asyncio
async def test_handle_device_download_model(mock_worker):
    """Test handling of DeviceDownloadModel command."""
    command = DeviceDownloadModel(model_id="test-model")
    
    # We need to mock build_full_shard because it interacts with model registry/network
    # Note: We patch it where it is imported/used in worker.main. Since it is imported inside the method
    # in _handle_device_download_model, checking the import path there:
    # "from exo.worker.download.impl_shard_downloader import build_full_shard"
    # So we patch 'exo.worker.download.impl_shard_downloader.build_full_shard'
    # Wait, the code I wrote was:
    # try:
    #    from exo.worker.download.impl_shard_downloader import build_full_shard
    #    shard = await build_full_shard(command.model_id)
    # So patching exo.worker.main.build_full_shard won't work if it's not imported at top level.
    # I should patch 'exo.worker.download.impl_shard_downloader.build_full_shard'.
    
    with patch("exo.worker.download.impl_shard_downloader.build_full_shard", new_callable=AsyncMock) as mock_build:
        mock_shard = MagicMock()
        mock_build.return_value = mock_shard
        
        await mock_worker._handle_device_download_model(command)
        
        mock_build.assert_called_once_with("test-model")
        mock_worker._tg.start_soon.assert_called_once_with(
            mock_worker.shard_downloader.ensure_shard, mock_shard
        )

@pytest.mark.asyncio
async def test_handle_delete_model(mock_worker):
    """Test handling of DeleteModel command."""
    from exo.shared.types.events import NodeDownloadRemoved
    
    mock_worker.event_sender.send = AsyncMock()
    
    command = DeleteModel(model_id="test-model")
    mock_worker.download_status = {"test-model": "some_status"}
    
    await mock_worker._handle_delete_model(command)
    
    # Verify delete called
    mock_worker.shard_downloader.delete_model.assert_called_once_with("test-model")
    # Verify status cleared
    assert "test-model" not in mock_worker.download_status
    
    # Verify event emission
    mock_worker.event_sender.send.assert_called_once()
    event = mock_worker.event_sender.send.call_args[0][0]
    assert isinstance(event, NodeDownloadRemoved)
    assert event.node_id == mock_worker.node_id
    assert event.model_id == "test-model"

