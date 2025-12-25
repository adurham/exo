import asyncio
from multiprocessing import Process
from unittest.mock import AsyncMock, Mock, patch

import anyio
import pytest

from exo.shared.types.common import NodeId
from exo.shared.types.events import Event, RunnerStatusUpdated, TaskAcknowledged
from exo.shared.types.memory import Memory
from exo.shared.types.models import ModelId, ModelMetadata
from exo.shared.types.tasks import Task, TaskId
from exo.shared.types.worker.instances import BoundInstance, InstanceId, MlxRingInstance
from exo.shared.types.worker.runners import (
    RunnerId,
    RunnerStatus,
    RunnerWaitingForModel,
    ShardAssignments,
)
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.utils.channels import Sender
from exo.worker.runner.runner_supervisor import RunnerSupervisor


@pytest.fixture
def sample_bound_instance():
    """Create a sample bound instance for testing."""
    runner_id = RunnerId()
    shard = PipelineShardMetadata(
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
    instance = MlxRingInstance(
        instance_id=InstanceId(),
        shard_assignments=ShardAssignments(
            model_id=ModelId("test-model"),
            node_to_runner={NodeId(): runner_id},
            runner_to_shard={runner_id: shard},
        ),
        hosts=[],
    )
    return BoundInstance(
        bound_runner_id=runner_id,
        bound_node_id=NodeId(),
        instance=instance,
    )


@pytest.fixture
def mock_event_sender():
    """Create a mock event sender."""
    sender = Mock(spec=Sender)
    sender.send = AsyncMock()
    sender.close = Mock()
    return sender


class TestRunnerSupervisorCreate:
    def test_create_initializes_supervisor(self, sample_bound_instance, mock_event_sender):
        """Test that create() properly initializes a RunnerSupervisor."""
        mock_ev_send = Mock()
        mock_ev_recv = Mock()
        mock_task_send = Mock()
        mock_task_recv = Mock()
        
        call_count = [0]
        def mock_mp_channel(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return (mock_ev_send, mock_ev_recv)
            else:
                return (mock_task_send, mock_task_recv)
        
        with patch("exo.utils.channels.mp_channel.__new__", side_effect=mock_mp_channel), patch(
            "exo.worker.runner.runner_supervisor.Process"
        ) as mock_process_class:
            
            mock_process = Mock(spec=Process)
            mock_process_class.return_value = mock_process
            
            supervisor = RunnerSupervisor.create(
                bound_instance=sample_bound_instance,
                event_sender=mock_event_sender,
                initialize_timeout=400.0,
            )
            
            assert supervisor.bound_instance == sample_bound_instance
            assert supervisor.shard_metadata == sample_bound_instance.bound_shard
            assert supervisor.runner_process == mock_process
            assert supervisor.initialize_timeout == 400.0
            assert supervisor._ev_recv == mock_ev_recv
            assert supervisor._task_sender == mock_task_send
            assert supervisor._event_sender == mock_event_sender
            assert supervisor.status == RunnerWaitingForModel()
            assert supervisor.pending == {}

    def test_create_sets_process_target(self, sample_bound_instance, mock_event_sender):
        """Test that create() sets the process target to entrypoint."""
        mock_ev_send = Mock()
        mock_ev_recv = Mock()
        mock_task_send = Mock()
        mock_task_recv = Mock()
        
        call_count = [0]
        def mock_mp_channel(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return (mock_ev_send, mock_ev_recv)
            else:
                return (mock_task_send, mock_task_recv)
        
        with patch("exo.utils.channels.mp_channel.__new__", side_effect=mock_mp_channel), patch(
            "exo.worker.runner.runner_supervisor.Process"
        ) as mock_process_class:
            
            mock_process = Mock(spec=Process)
            mock_process_class.return_value = mock_process
            
            RunnerSupervisor.create(
                bound_instance=sample_bound_instance,
                event_sender=mock_event_sender,
            )
            
            # Verify Process was created with entrypoint as target
            mock_process_class.assert_called_once()
            call_kwargs = mock_process_class.call_args
            assert call_kwargs.kwargs["target"] is not None
            assert call_kwargs.kwargs["daemon"] is True


class TestRunnerSupervisorRun:
    @pytest.mark.asyncio
    async def test_run_starts_process(self, sample_bound_instance, mock_event_sender):
        """Test that run() starts the runner process."""
        with patch("exo.worker.runner.runner_supervisor.mp_channel") as mock_channel, patch(
            "exo.worker.runner.runner_supervisor.Process"
        ) as mock_process_class, patch(
            "exo.worker.runner.runner_supervisor.create_task_group"
        ) as mock_tg, patch("anyio.sleep", new_callable=AsyncMock):
            mock_ev_send = Mock()
            mock_ev_recv = Mock()
            mock_task_send = Mock()
            mock_task_recv = Mock()
            
            call_count = [0]
            def channel_new(cls, *args, **kwargs):
                call_count[0] += 1
                if call_count[0] == 1:
                    return (mock_ev_send, mock_ev_recv)
                else:
                    return (mock_task_send, mock_task_recv)
            
            mock_channel.__new__ = channel_new
            
            mock_process = Mock(spec=Process)
            mock_process.is_alive.return_value = True
            mock_process.pid = 12345
            mock_process_class.return_value = mock_process
            
            mock_tg_instance = AsyncMock()
            mock_tg_instance.__aenter__ = AsyncMock(return_value=mock_tg_instance)
            mock_tg_instance.__aexit__ = AsyncMock(return_value=None)
            mock_tg_instance.start_soon = Mock()
            mock_tg.return_value = mock_tg_instance
            
            supervisor = RunnerSupervisor.create(
                bound_instance=sample_bound_instance,
                event_sender=mock_event_sender,
            )
            
            # Mock the event forwarding to complete immediately
            async def mock_forward_events():
                pass
            
            supervisor._forward_events = mock_forward_events
            
            # Mock channels to be async iterable
            async def mock_async_iter():
                return
                yield  # Make it an async generator
            
            mock_ev_recv.__aiter__ = Mock(return_value=mock_async_iter())
            mock_ev_recv.close = Mock()
            mock_task_send.close = Mock()
            mock_event_sender.close = Mock()
            
            # Mock to_thread.run_sync for join
            with patch("exo.worker.runner.runner_supervisor.to_thread.run_sync", new_callable=AsyncMock):
                # Run should complete quickly since we're mocking everything
                try:
                    await asyncio.wait_for(supervisor.run(), timeout=0.1)
                except asyncio.TimeoutError:
                    # Expected - we'll cancel it
                    supervisor.shutdown()
            
            # Verify process was started
            mock_process.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_handles_process_crash(self, sample_bound_instance, mock_event_sender):
        """Test that run() handles process crash detection."""
        with patch("exo.worker.runner.runner_supervisor.mp_channel") as mock_channel, patch(
            "exo.worker.runner.runner_supervisor.Process"
        ) as mock_process_class, patch(
            "exo.worker.runner.runner_supervisor.create_task_group"
        ) as mock_tg, patch("anyio.sleep", new_callable=AsyncMock):
            mock_ev_send = Mock()
            mock_ev_recv = Mock()
            mock_task_send = Mock()
            mock_task_recv = Mock()
            
            call_count = [0]
            def channel_new(cls, *args, **kwargs):
                call_count[0] += 1
                if call_count[0] == 1:
                    return (mock_ev_send, mock_ev_recv)
                else:
                    return (mock_task_send, mock_task_recv)
            
            mock_channel.__new__ = channel_new
            
            mock_process = Mock(spec=Process)
            mock_process.is_alive.return_value = False  # Process crashed
            mock_process.pid = 12345
            mock_process.exitcode = 1
            mock_process_class.return_value = mock_process
            
            mock_tg_instance = AsyncMock()
            mock_tg_instance.__aenter__ = AsyncMock(return_value=mock_tg_instance)
            mock_tg_instance.__aexit__ = AsyncMock(return_value=None)
            mock_tg_instance.start_soon = Mock()
            mock_tg.return_value = mock_tg_instance
            
            supervisor = RunnerSupervisor.create(
                bound_instance=sample_bound_instance,
                event_sender=mock_event_sender,
            )
            
            async def mock_forward_events():
                pass
            
            supervisor._forward_events = mock_forward_events
            
            mock_ev_recv.close = Mock()
            mock_task_send.close = Mock()
            mock_event_sender.close = Mock()
            
            with patch("exo.worker.runner.runner_supervisor.to_thread.run_sync", new_callable=AsyncMock):
                try:
                    await asyncio.wait_for(supervisor.run(), timeout=0.1)
                except asyncio.TimeoutError:
                    supervisor.shutdown()
            
            # Process should have been checked for crash
            assert mock_process.is_alive.call_count > 0


class TestRunnerSupervisorStartTask:
    @pytest.mark.asyncio
    async def test_start_task_sends_task_and_waits(self, sample_bound_instance, mock_event_sender):
        """Test that start_task() sends a task and waits for acknowledgment."""
        with patch("exo.worker.runner.runner_supervisor.mp_channel") as mock_channel, patch(
            "exo.worker.runner.runner_supervisor.Process"
        ) as mock_process_class:
            mock_ev_send = Mock()
            mock_ev_recv = Mock()
            mock_task_send = Mock()
            mock_task_recv = Mock()
            
            call_count = [0]
            def channel_new(cls, *args, **kwargs):
                call_count[0] += 1
                if call_count[0] == 1:
                    return (mock_ev_send, mock_ev_recv)
                else:
                    return (mock_task_send, mock_task_recv)
            
            mock_channel.__new__ = channel_new
            
            mock_process = Mock(spec=Process)
            mock_process_class.return_value = mock_process
            
            supervisor = RunnerSupervisor.create(
                bound_instance=sample_bound_instance,
                event_sender=mock_event_sender,
            )
            
            task = Task(task_id=TaskId(), command=Mock())
            
            # Mock task sender
            mock_task_send.send = Mock()
            
            # Create an event that will be set when task is acknowledged
            ack_event = anyio.Event()
            supervisor.pending[task.task_id] = ack_event
            
            # Start task in background and set event after a delay
            async def set_ack_event():
                await anyio.sleep(0.01)
                ack_event.set()
            
            async def start_task_wrapper():
                await supervisor.start_task(task)
            
            # Run both concurrently
            await asyncio.gather(
                start_task_wrapper(),
                set_ack_event(),
            )
            
            # Verify task was sent
            mock_task_send.send.assert_called_once_with(task)

    @pytest.mark.asyncio
    async def test_start_task_handles_closed_channel(self, sample_bound_instance, mock_event_sender):
        """Test that start_task() handles ClosedResourceError gracefully."""
        with patch("exo.worker.runner.runner_supervisor.mp_channel") as mock_channel, patch(
            "exo.worker.runner.runner_supervisor.Process"
        ) as mock_process_class:
            mock_ev_send = Mock()
            mock_ev_recv = Mock()
            mock_task_send = Mock()
            mock_task_recv = Mock()
            
            call_count = [0]
            def channel_new(cls, *args, **kwargs):
                call_count[0] += 1
                if call_count[0] == 1:
                    return (mock_ev_send, mock_ev_recv)
                else:
                    return (mock_task_send, mock_task_recv)
            
            mock_channel.__new__ = channel_new
            
            mock_process = Mock(spec=Process)
            mock_process_class.return_value = mock_process
            
            supervisor = RunnerSupervisor.create(
                bound_instance=sample_bound_instance,
                event_sender=mock_event_sender,
            )
            
            task = Task(task_id=TaskId(), command=Mock())
            
            # Mock task sender to raise ClosedResourceError
            from anyio import ClosedResourceError
            
            mock_task_send.send = Mock(side_effect=ClosedResourceError())
            
            # Should not raise, just return
            await supervisor.start_task(task)
            
            # Task should not be in pending
            assert task.task_id not in supervisor.pending


class TestRunnerSupervisorShutdown:
    def test_shutdown_cancels_task_group(self, sample_bound_instance, mock_event_sender):
        """Test that shutdown() cancels the task group."""
        with patch("exo.worker.runner.runner_supervisor.mp_channel") as mock_channel, patch(
            "exo.worker.runner.runner_supervisor.Process"
        ) as mock_process_class:
            mock_ev_send = Mock()
            mock_ev_recv = Mock()
            mock_task_send = Mock()
            mock_task_recv = Mock()
            
            call_count = [0]
            def channel_new(cls, *args, **kwargs):
                call_count[0] += 1
                if call_count[0] == 1:
                    return (mock_ev_send, mock_ev_recv)
                else:
                    return (mock_task_send, mock_task_recv)
            
            mock_channel.__new__ = channel_new
            
            mock_process = Mock(spec=Process)
            mock_process_class.return_value = mock_process
            
            supervisor = RunnerSupervisor.create(
                bound_instance=sample_bound_instance,
                event_sender=mock_event_sender,
            )
            
            # Create a mock task group
            mock_tg = Mock()
            mock_tg.cancel_scope = Mock()
            mock_tg.cancel_scope.cancel = Mock()
            supervisor._tg = mock_tg
            
            supervisor.shutdown()
            
            # Verify cancel was called
            mock_tg.cancel_scope.cancel.assert_called_once()

