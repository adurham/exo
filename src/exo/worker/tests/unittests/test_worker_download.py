import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
import pytest
from exo.worker.main import Worker
from exo.shared.types.worker.shards import ShardMetadata
from exo.shared.types.models import ModelId
from exo.shared.types.worker.downloads import DownloadOngoing, DownloadProgressData

from exo.shared.types.common import NodeId, SessionId
from exo.shared.types.tasks import DownloadModel
from exo.shared.types.worker.runners import RunnerId
from exo.shared.types.worker.instances import BoundInstance, Instance
from exo.shared.types.worker.shards import ShardMetadata
from exo.shared.types.models import ModelMetadata
from exo.shared.types.memory import Memory
from exo.worker.download.shard_downloader import RepoDownloadProgress
from datetime import timedelta
from exo.worker.tests.unittests.conftest import (
    get_pipeline_shard_metadata,
)

@pytest.fixture
def mock_worker():
    shard_downloader = MagicMock()
    shard_downloader.get_shard_download_status_for_shard = AsyncMock()
    shard_downloader.ensure_shard = AsyncMock()
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
    
    return worker

@pytest.mark.asyncio
async def test_coordinator_download_fallback(mock_worker):
    """Test that the coordinator falls back to HF download if P2P discovery fails."""
    mock_worker.node_id = NodeId("master")
    mock_worker.session_id.master_node_id = NodeId("master")
    
    shard = get_pipeline_shard_metadata(model_id="model1", device_rank=0)
    task = DownloadModel(instance_id="inst1", shard_metadata=shard)
    initial_progress = RepoDownloadProgress(
        repo_id="model1",
        repo_revision="main",
        shard=shard,
        completed_files=0,
        total_files=10,
        downloaded_bytes=Memory.from_bytes(0),
        downloaded_bytes_this_session=Memory.from_bytes(0),
        total_bytes=Memory.from_bytes(100),
        overall_speed=0,
        overall_eta=timedelta(seconds=0),
        status="not_started",
    )
    
    # Simulate discovery timeout
    mock_worker.peer_locations = {}
    mock_worker.discovery_start_times = {"model1": 0} 
    
    # This should break loop because node_id == master_node_id and time > 1.0
    await mock_worker._handle_shard_download_process(task, initial_progress)
    
    # Coordinator should proceed to download (ensure_shard called)
    mock_worker.shard_downloader.ensure_shard.assert_called_once()
    args = mock_worker.shard_downloader.ensure_shard.call_args
    assert args[0][0] == task.shard_metadata
    assert args[0][2] is None # endpoint is None

@pytest.mark.asyncio
async def test_worker_waits_for_coordinator(mock_worker):
    """Test that a non-coordinator node waits and does not download from HF."""
    mock_worker.node_id = NodeId("worker1")
    mock_worker.session_id.master_node_id = NodeId("master")
    
    shard = get_pipeline_shard_metadata(model_id="model1", device_rank=0)
    task = DownloadModel(instance_id="inst1", shard_metadata=shard)
    initial_progress = RepoDownloadProgress(
        repo_id="model1",
        repo_revision="main",
        shard=shard,
        completed_files=0,
        total_files=10,
        downloaded_bytes=Memory.from_bytes(0),
        downloaded_bytes_this_session=Memory.from_bytes(0),
        total_bytes=Memory.from_bytes(100),
        overall_speed=0,
        overall_eta=timedelta(seconds=0),
        status="not_started",
    )
    
    mock_worker.peer_locations = {}
    mock_worker.discovery_start_times = {"model1": 0}

    # Patch sleep to verify waiting logic and then break loop
    async def side_effect_sleep(seconds):
        # Verify haven't downloaded yet
        mock_worker.shard_downloader.ensure_shard.assert_not_called()
        # "Discover" the peer to break the loop
        mock_worker.peer_locations["model1"] = "http://master:8080"
    
    with patch("exo.worker.main.anyio.sleep", side_effect=side_effect_sleep) as mock_sleep:
        await mock_worker._handle_shard_download_process(task, initial_progress)
        
    mock_sleep.assert_called()
    
    # Worker should download from PEER now
    mock_worker.shard_downloader.ensure_shard.assert_called_once()
    args = mock_worker.shard_downloader.ensure_shard.call_args
    assert args[0][2] == "http://master:8080"

@pytest.mark.asyncio
async def test_worker_p2p_download(mock_worker):
    """Test that any node downloads if P2P endpoint is available."""
    mock_worker.node_id = NodeId("worker1")
    
    shard = get_pipeline_shard_metadata(model_id="model1", device_rank=0)
    task = DownloadModel(instance_id="inst1", shard_metadata=shard)
    initial_progress = RepoDownloadProgress(
        repo_id="model1",
        repo_revision="main",
        shard=shard,
        completed_files=0,
        total_files=10,
        downloaded_bytes=Memory.from_bytes(0),
        downloaded_bytes_this_session=Memory.from_bytes(0),
        total_bytes=Memory.from_bytes(100),
        overall_speed=0,
        overall_eta=timedelta(seconds=0),
        status="not_started",
    )
    
    # Simulate discovery success
    mock_worker.peer_locations = {"model1": "http://peer:8080"}
    
    await mock_worker._handle_shard_download_process(task, initial_progress)
    
    # Should download via P2P
    mock_worker.shard_downloader.ensure_shard.assert_called_once()
    args = mock_worker.shard_downloader.ensure_shard.call_args
    assert args[0][2] == "http://peer:8080"

@pytest.mark.asyncio
async def test_worker_waits_if_peer_in_state(mock_worker):
    """Test that a worker waits (does not fallback) if global state shows peer has it."""
    mock_worker.node_id = NodeId("worker1")
    shard = get_pipeline_shard_metadata(model_id="model1", device_rank=0)
    task = DownloadModel(instance_id="inst1", shard_metadata=shard)
    initial_progress = RepoDownloadProgress(
        repo_id="model1",
        repo_revision="main",
        shard=shard,
        completed_files=0,
        total_files=10,
        downloaded_bytes=Memory.from_bytes(0),
        downloaded_bytes_this_session=Memory.from_bytes(0),
        total_bytes=Memory.from_bytes(100),
        overall_speed=0,
        overall_eta=timedelta(seconds=0),
        status="not_started",
    )

    # Mock state showing another node has it
    mock_worker.state.downloads = {
        NodeId("peer"): [
            DownloadOngoing(
                node_id=NodeId("peer"),
                shard_metadata=shard,
                download_progress=DownloadProgressData(
                    completed_files=0,
                    total_files=10,
                    downloaded_bytes=Memory.from_bytes(0),
                    downloaded_bytes_this_session=Memory.from_bytes(0),
                    total_bytes=Memory.from_bytes(100),
                    speed=0.0,
                    eta_ms=0,
                    files={},
                )
            )
        ]
    }
    
    mock_worker.peer_locations = {} # Not discovered yet
    mock_worker.discovery_start_times = {}

    async def side_effect_sleep(seconds):
        mock_worker.command_sender.send_nowait.assert_called()
        # Break loop
        mock_worker.peer_locations["model1"] = "http://peer:8080"
    
    with patch("exo.worker.main.anyio.sleep", side_effect=side_effect_sleep):
         await mock_worker._handle_shard_download_process(task, initial_progress)
    
    # Should trigger discovery 
    assert mock_worker.command_sender.send_nowait.called
    # Should mark discovery start
    assert "model1" in mock_worker.discovery_start_times
    
    # And eventually download once loop broke
    mock_worker.shard_downloader.ensure_shard.assert_called_once()
