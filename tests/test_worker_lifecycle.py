import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from exo.worker.main import Worker
from exo.shared.types.common import NodeId, SessionId
from exo.shared.types.tasks import Task, CreateRunner, TaskStatus
from exo.shared.types.events import Event, TaskCreated, TaskStatusUpdated
from exo.shared.types.worker.instances import BoundInstance, MlxRingInstance, InstanceId
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.shared.types.models import ModelMetadata, ModelId
from exo.worker.runner.runner_supervisor import RunnerSupervisor
from exo.shared.types.memory import Memory

class MockSender:
    def __init__(self):
        self.sent = []
    async def send(self, obj):
        self.sent.append(obj)
    def send_nowait(self, obj):
        self.sent.append(obj)
    def close(self): pass
    def clone(self): return self

class MockReceiver:
    def __init__(self, items=None):
        self.items = items or []
    def __enter__(self): return self
    def __exit__(self, *args): pass
    async def __aiter__(self):
        for item in self.items:
            yield item
    def clone_sender(self): return MockSender()
    def close(self): pass

@pytest.fixture
def mock_channel():
    sender = MockSender()
    receiver = MockReceiver()
    return sender, receiver

class MockChannelFactory:
    def __init__(self, sender, receiver):
        self.sender = sender
        self.receiver = receiver
    def __getitem__(self, item):
        return lambda: (self.sender, self.receiver)

@pytest.mark.asyncio
async def test_worker_create_runner_lifecycle(mock_channel):
    # Setup
    node_id = NodeId("node-1")
    session_id = SessionId(master_node_id=node_id, election_clock=0)
    
    sender, receiver = mock_channel
    
    with pytest.MonkeyPatch.context() as m:
        # Patch channel with a mock that supports []
        m.setattr("exo.worker.main.channel", MockChannelFactory(sender, receiver))
        
        worker = Worker(
            node_id=node_id,
            session_id=session_id,
            shard_downloader=MagicMock(),
            connection_message_receiver=MockReceiver(),
            global_event_receiver=MockReceiver(),
            local_event_sender=MockSender(),
            command_sender=MockSender(),
        )
        
        # Construct a valid Instance
        runner_id = RunnerId("runner-1")
        model_id = ModelId("model-1")
        shard = PipelineShardMetadata(
            model_meta=ModelMetadata(
                model_id=model_id,
                storage_size=Memory(in_bytes=1024),
                supports_tensor=False,
                pretty_name="Mock Model",
                n_layers=1,
                hidden_size=1024
            ),
            start_layer=0,
            end_layer=1,
            n_layers=1,
            device_rank=0,
            world_size=1
        )
        
        instance = MlxRingInstance(
            instance_id=InstanceId("inst-1"),
            shard_assignments=ShardAssignments(
                model_id=model_id,
                node_to_runner={node_id: runner_id},
                runner_to_shard={runner_id: shard}
            ),
            hosts_by_node={node_id: []},
            ephemeral_port=8080
        )

        mock_task = CreateRunner(
            instance_id=instance.instance_id,
            bound_instance=BoundInstance(
                instance=instance,
                bound_runner_id=runner_id,
                bound_node_id=node_id
            )
        )
        
        m.setattr("exo.worker.main.plan", lambda *args: mock_task)
        
        mock_supervisor = MagicMock(spec=RunnerSupervisor)
        mock_supervisor.run = AsyncMock()
        m.setattr(worker, "_create_supervisor", lambda t: mock_supervisor)
        m.setattr("exo.worker.main.start_polling_node_metrics", AsyncMock())
        m.setattr("exo.worker.main.start_polling_memory_metrics", AsyncMock())
        m.setattr("exo.worker.main.check_reachable", AsyncMock(return_value={}))
        
        sleep_count = 0
        async def mock_sleep(sec):
            nonlocal sleep_count
            if sleep_count > 0:
                raise KeyboardInterrupt("Break Loop")
            sleep_count += 1
            
        m.setattr("anyio.sleep", mock_sleep)
        
        with pytest.raises(KeyboardInterrupt, match="Break Loop"):
            await worker.plan_step()
            
        # Verifications
        assert len(sender.sent) >= 2
        assert isinstance(sender.sent[0], TaskCreated)
        assert isinstance(sender.sent[1], TaskStatusUpdated)
        assert sender.sent[1].task_status == TaskStatus.Complete


@pytest.mark.asyncio
async def test_worker_download_lifecycle(mock_channel):
    # Setup
    node_id = NodeId("node-1")
    session_id = SessionId(master_node_id=node_id, election_clock=0)
    mock_downloader = MagicMock()
    # Mock get_shard_download_status_for_shard to return pending then complete
    mock_downloader.get_shard_download_status_for_shard = AsyncMock()
    
    sender, receiver = mock_channel
    
    with pytest.MonkeyPatch.context() as m:
        m.setattr("exo.worker.main.channel", MockChannelFactory(sender, receiver))
        
        worker = Worker(
            node_id=node_id,
            session_id=session_id,
            shard_downloader=mock_downloader,
            connection_message_receiver=MockReceiver(),
            global_event_receiver=MockReceiver(),
            local_event_sender=MockSender(),
            command_sender=MockSender(),
        )
        worker._tg = MagicMock()

        # Valid shard metadata
        shard = PipelineShardMetadata(
            model_meta=ModelMetadata(
                model_id=ModelId("model-1"),
                storage_size=Memory(in_bytes=1024),
                supports_tensor=False,
                pretty_name="Mock Model",
                n_layers=1,
                hidden_size=1024
            ),
            start_layer=0,
            end_layer=1,
            n_layers=1,
            device_rank=0,
            world_size=1
        )

        # Mock Task: DownloadModel
        from exo.shared.types.tasks import DownloadModel
        mock_task = DownloadModel(
            instance_id="inst-1",
            shard_metadata=shard
        )
        from datetime import timedelta
        
        from exo.shared.types.events import NodeDownloadProgress
        from exo.worker.download.download_utils import RepoDownloadProgress

        # CASE 1: Download Incomplete (Running)
        mock_downloader.get_shard_download_status_for_shard.return_value = RepoDownloadProgress(
            status="in_progress",
            repo_id="model-1",
            repo_revision="main",
            shard=shard,
            completed_files=0,
            total_files=1,
            downloaded_bytes=Memory(in_bytes=0),
            downloaded_bytes_this_session=Memory(in_bytes=0),
            total_bytes=Memory(in_bytes=100),
            overall_speed=0.0,
            overall_eta=timedelta(seconds=100)
        )

        m.setattr("exo.worker.main.plan", lambda *args: mock_task)
        m.setattr("exo.worker.main.start_polling_node_metrics", AsyncMock())
        m.setattr("exo.worker.main.start_polling_memory_metrics", AsyncMock())
        m.setattr("exo.worker.main.check_reachable", AsyncMock(return_value={}))
        
        sleep_count_2 = 0
        async def mock_sleep_2(sec):
            nonlocal sleep_count_2
            if sleep_count_2 > 0:
                raise KeyboardInterrupt("Break Loop")
            sleep_count_2 += 1
            
        m.setattr("anyio.sleep", mock_sleep_2)
        
        with pytest.raises(KeyboardInterrupt, match="Break Loop"):
            await worker.plan_step()
        
        # Verifications
        # 1. TaskCreated
        # 2. NodeDownloadProgress (Pending - before check)
        # 3. NodeDownloadProgress (In Progress - after check)
        # 4. TaskStatusUpdated (Running)
        
        print(f"DEBUG SENT: {sender.sent}")
        assert len(sender.sent) >= 3
        assert isinstance(sender.sent[0], TaskCreated)
        # We might have duplicates or slight ordering changes depending on implementation
        # The key is that we have TaskStatus.Running eventually
        
        has_running = any(
            isinstance(e, TaskStatusUpdated) and e.task_status == TaskStatus.Running 
            for e in sender.sent
        )
        assert has_running, "Did not find TaskStatus.Running event"
        assert isinstance(sender.sent[0], TaskCreated)
        assert isinstance(sender.sent[1], NodeDownloadProgress)
        assert isinstance(sender.sent[2], TaskStatusUpdated)
        assert sender.sent[2].task_status == TaskStatus.Running
