import pytest
from unittest.mock import AsyncMock, MagicMock
from exo.master.api import API, DownloadModelRequest
from exo.shared.types.commands import DeviceDownloadModel, DeleteModel
from exo.shared.types.common import NodeId, SessionId

@pytest.fixture
def mock_api():
    command_sender = MagicMock()
    command_sender.send = AsyncMock()
    
    api = API(
        node_id=NodeId("master"),
        session_id=SessionId(master_node_id=NodeId("master"), election_clock=0),
        port=8080,
        global_event_receiver=MagicMock(),
        command_sender=command_sender,
        election_receiver=MagicMock(),
    )
    
    # Mock paused_ev since _send waits on it if paused
    api.paused_ev = AsyncMock()
    api.paused_ev.wait = AsyncMock()
    
    return api

@pytest.mark.asyncio
async def test_start_download_model(mock_api):
    request = DownloadModelRequest(model_id="test-model")
    response = await mock_api.start_download_model(request)
    
    assert response["status"] == "ok"
    assert "Download started" in response["message"]
    
    # Verify command sent
    mock_api.command_sender.send.assert_called_once()
    forwarder_command = mock_api.command_sender.send.call_args[0][0]
    assert isinstance(forwarder_command.command, DeviceDownloadModel)
    assert forwarder_command.command.model_id == "test-model"

@pytest.mark.asyncio
async def test_delete_model_command(mock_api):
    model_id = "test-model-to-delete"
    response = await mock_api.delete_model_command(model_id)
    
    assert response["status"] == "ok"
    assert "Deletion started" in response["message"]
    
    # Verify command sent
    mock_api.command_sender.send.assert_called_once()
    forwarder_command = mock_api.command_sender.send.call_args[0][0]
    assert isinstance(forwarder_command.command, DeleteModel)
    assert forwarder_command.command.model_id == model_id
