import os
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from exo.shared.types.common import NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.models import ModelId, ModelMetadata
from exo.shared.types.worker.instances import (
    BoundInstance,
    InstanceId,
    MlxJacclInstance,
    MlxRingInstance,
)
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.shared.types.worker.shards import PipelineShardMetadata, ShardMetadata
from exo.worker.runner.bootstrap import entrypoint


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
def sample_mlx_jaccl_instance():
    """Create a sample MlxJacclInstance for testing."""
    runner_id = RunnerId()
    shard = PipelineShardMetadata(
        model_meta=ModelMetadata(
            model_id=ModelId("test-model"),
            pretty_name="Test Model",
            storage_size=Memory.from_mb(100),
            n_layers=12,
        ),
        device_rank=0,
        world_size=2,
        start_layer=0,
        end_layer=12,
        n_layers=12,
    )
    instance = MlxJacclInstance(
        instance_id=InstanceId(),
        shard_assignments=ShardAssignments(
            model_id=ModelId("test-model"),
            node_to_runner={NodeId(): runner_id},
            runner_to_shard={runner_id: shard},
        ),
        ibv_devices=[["en2", None], ["en3", None]],
        ibv_coordinators={},
    )
    return BoundInstance(
        bound_runner_id=runner_id,
        bound_node_id=NodeId(),
        instance=instance,
    )


class TestBootstrapEntrypoint:
    def test_entrypoint_sets_mlx_metal_fast_synch_for_jaccl(self, sample_mlx_jaccl_instance):
        """Test that MLX_METAL_FAST_SYNCH is set for MlxJacclInstance with 2+ devices."""
        mock_event_sender = Mock()
        mock_task_receiver = Mock()
        
        with patch.dict(os.environ, {}, clear=True), patch(
            "exo.worker.runner.bootstrap.logger_setup"
        ), patch("exo.worker.runner.bootstrap.loguru.logger") as mock_logger, patch(
            "exo.worker.runner.runner.main", new_callable=Mock
        ) as mock_main:
            entrypoint(sample_mlx_jaccl_instance, mock_event_sender, mock_task_receiver)
            
            assert os.environ.get("MLX_METAL_FAST_SYNCH") == "1"
            mock_main.assert_called_once()

    def test_entrypoint_no_mlx_metal_fast_synch_for_ring(self, sample_bound_instance):
        """Test that MLX_METAL_FAST_SYNCH is not set for MlxRingInstance."""
        mock_event_sender = Mock()
        mock_task_receiver = Mock()
        
        with patch.dict(os.environ, {}, clear=True), patch(
            "exo.worker.runner.bootstrap.logger_setup"
        ), patch("exo.worker.runner.bootstrap.loguru.logger") as mock_logger, patch(
            "exo.worker.runner.runner.main", new_callable=Mock
        ) as mock_main:
            entrypoint(sample_bound_instance, mock_event_sender, mock_task_receiver)
            
            assert "MLX_METAL_FAST_SYNCH" not in os.environ
            mock_main.assert_called_once()

    def test_entrypoint_no_mlx_metal_fast_synch_single_device(self):
        """Test that MLX_METAL_FAST_SYNCH is not set for MlxJacclInstance with < 2 devices."""
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
        instance = MlxJacclInstance(
            instance_id=InstanceId(),
            shard_assignments=ShardAssignments(
                model_id=ModelId("test-model"),
                node_to_runner={NodeId(): runner_id},
                runner_to_shard={runner_id: shard},
            ),
            ibv_devices=[["en2", None]],  # Only one device
            ibv_coordinators={},
        )
        bound_instance = BoundInstance(
            bound_runner_id=runner_id,
            bound_node_id=NodeId(),
            instance=instance,
        )
        
        mock_event_sender = Mock()
        mock_task_receiver = Mock()
        
        with patch.dict(os.environ, {}, clear=True), patch(
            "exo.worker.runner.bootstrap.logger_setup"
        ), patch("exo.worker.runner.bootstrap.loguru.logger") as mock_logger, patch(
            "exo.worker.runner.runner.main", new_callable=Mock
        ) as mock_main:
            entrypoint(bound_instance, mock_event_sender, mock_task_receiver)
            
            assert "MLX_METAL_FAST_SYNCH" not in os.environ
            mock_main.assert_called_once()

    def test_entrypoint_patches_multiprocessing_flush(self, sample_bound_instance):
        """Test that multiprocessing flush is patched."""
        mock_event_sender = Mock()
        mock_task_receiver = Mock()
        
        import multiprocessing.util
        original_flush = multiprocessing.util._flush_std_streams
        
        with patch("exo.worker.runner.bootstrap.logger_setup"), patch(
            "exo.worker.runner.bootstrap.loguru.logger"
        ) as mock_logger, patch(
            "exo.worker.runner.runner.main", new_callable=Mock
        ) as mock_main:
            entrypoint(sample_bound_instance, mock_event_sender, mock_task_receiver)
            
            # Verify the flush function was patched (should be different from original)
            assert multiprocessing.util._flush_std_streams != original_flush

    def test_entrypoint_handles_exception(self, sample_bound_instance):
        """Test that exceptions are logged and re-raised."""
        mock_event_sender = Mock()
        mock_task_receiver = Mock()
        
        with patch("exo.worker.runner.bootstrap.logger_setup"), patch(
            "exo.worker.runner.bootstrap.loguru.logger"
        ) as mock_logger, patch(
            "exo.worker.runner.runner.main", side_effect=Exception("Test error")
        ):
            with pytest.raises(Exception, match="Test error"):
                entrypoint(sample_bound_instance, mock_event_sender, mock_task_receiver)
            
            # Should log the error
            mock_logger.error.assert_called()

    def test_entrypoint_logs_initialization(self, sample_bound_instance):
        """Test that initialization is logged."""
        mock_event_sender = Mock()
        mock_task_receiver = Mock()
        
        with patch("exo.worker.runner.bootstrap.logger_setup"), patch(
            "exo.worker.runner.bootstrap.loguru.logger"
        ) as mock_logger, patch(
            "exo.worker.runner.runner.main", new_callable=Mock
        ) as mock_main:
            entrypoint(sample_bound_instance, mock_event_sender, mock_task_receiver)
            
            # Should log bootstrap messages
            assert mock_logger.info.called

    def test_entrypoint_sets_verbosity(self, sample_bound_instance):
        """Test that verbosity is set from environment."""
        mock_event_sender = Mock()
        mock_task_receiver = Mock()
        
        with patch.dict(os.environ, {"EXO_VERBOSITY": "2"}, clear=False), patch(
            "exo.worker.runner.bootstrap.logger_setup"
        ) as mock_setup, patch(
            "exo.worker.runner.bootstrap.loguru.logger"
        ) as mock_logger, patch(
            "exo.worker.runner.runner.main", new_callable=Mock
        ):
            entrypoint(sample_bound_instance, mock_event_sender, mock_task_receiver)
            
            # Should call logger_setup with verbosity (2)
            mock_setup.assert_called()
            # Check that verbosity 2 was passed
            assert mock_setup.call_args[0][1] == 2

