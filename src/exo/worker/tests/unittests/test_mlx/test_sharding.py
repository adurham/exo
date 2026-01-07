import unittest
from unittest.mock import MagicMock, patch
import mlx.core as mx
import mlx.nn as nn
from exo.worker.engines.mlx.utils_mlx import shard_and_load
from exo.shared.types.worker.shards import TensorShardMetadata, PipelineShardMetadata
from exo.shared.types.models import ModelMetadata
from exo.shared.types.memory import Memory

class TestSharding(unittest.TestCase):
    @patch("exo.worker.engines.mlx.utils_mlx.load_model")
    @patch("exo.worker.engines.mlx.utils_mlx.get_tokenizer")
    @patch("exo.worker.engines.mlx.utils_mlx.tensor_auto_parallel")
    @patch("exo.worker.engines.mlx.utils_mlx.pipeline_auto_parallel")
    @patch("exo.worker.engines.mlx.utils_mlx.mx.eval")
    @patch("exo.worker.engines.mlx.utils_mlx.mx_barrier")
    @patch("gc.collect")
    @patch("exo.worker.engines.mlx.utils_mlx.set_wired_limit_for_model")
    def test_tensor_sharding_call(self, mock_limit, mock_gc, mock_barrier, mock_eval, mock_pipeline, mock_tensor, mock_get_tok, mock_load):
        # Setup
        mock_model = MagicMock(spec=nn.Module)
        # Mock 20 layers
        layers = [MagicMock() for _ in range(20)]
        mock_model.layers = layers
        # Mock tensor_auto_parallel returning the model
        mock_tensor.return_value = mock_model
        
        mock_load.return_value = (mock_model, None)
        
        mock_group = MagicMock()
        mock_group.size.return_value = 2
        mock_group.rank.return_value = 0
        
        meta = TensorShardMetadata(
            model_meta=ModelMetadata(model_id="test", pretty_name="test", storage_size=Memory(in_bytes=1000), n_layers=20, hidden_size=10, supports_tensor=True),
            device_rank=0,
            world_size=2,
            start_layer=0,
            end_layer=20,
            n_layers=20
        )
        
        # Execute
        shard_and_load(meta, mock_group)
        
        # Verify correct sharding function called
        mock_tensor.assert_called_once()
        mock_pipeline.assert_not_called()
        
        # Verify layer-wise eval
        # 20 layers. GC collected at i=0, i=10.
        self.assertEqual(mock_gc.call_count, 2)
        
        # Verify eval called for each layer params
        # Note: mx.eval is called for each layer (20 times) + maybe others?
        # In our code:
        # for i, layer in enumerate(model.layers):
        #    mx.eval(layer.parameters())
        # So at least 20 calls.
        self.assertGreaterEqual(mock_eval.call_count, 20) 

    @patch("exo.worker.engines.mlx.utils_mlx.load_model")
    @patch("exo.worker.engines.mlx.utils_mlx.get_tokenizer")
    @patch("exo.worker.engines.mlx.utils_mlx.tensor_auto_parallel")
    @patch("exo.worker.engines.mlx.utils_mlx.pipeline_auto_parallel")
    @patch("exo.worker.engines.mlx.utils_mlx.mx.eval")
    @patch("exo.worker.engines.mlx.utils_mlx.mx_barrier")
    @patch("exo.worker.engines.mlx.utils_mlx.set_wired_limit_for_model")
    def test_pipeline_sharding_call(self, mock_limit, mock_barrier, mock_eval, mock_pipeline, mock_tensor, mock_get_tok, mock_load):
        # Setup
        mock_model = MagicMock(spec=nn.Module)
        mock_model.layers = [MagicMock() for _ in range(20)]
        mock_pipeline.return_value = mock_model
        mock_load.return_value = (mock_model, None)
        mock_group = MagicMock()
        mock_group.size.return_value = 2
        mock_group.rank.return_value = 0
        
        meta = PipelineShardMetadata(
            model_meta=ModelMetadata(model_id="test", pretty_name="test", storage_size=Memory(in_bytes=1000), n_layers=20, hidden_size=10, supports_tensor=True),
            device_rank=0,
            world_size=2,
            start_layer=0,
            end_layer=10,
            n_layers=20
        )
        
        # Execute
        shard_and_load(meta, mock_group)
        
        # Verify correct sharding function called
        mock_pipeline.assert_called_once()
        mock_tensor.assert_not_called()
