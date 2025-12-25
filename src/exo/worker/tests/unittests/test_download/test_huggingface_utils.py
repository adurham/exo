import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import aiofiles
import aiofiles.os as aios
import pytest

from exo.shared.types.models import ModelId, ModelMetadata
from exo.shared.types.memory import Memory
from exo.shared.types.worker.shards import PipelineShardMetadata, ShardMetadata
from exo.worker.download.huggingface_utils import (
    _add_wildcard_to_directories,
    extract_layer_num,
    filter_repo_objects,
    get_allow_patterns,
    get_auth_headers,
    get_hf_endpoint,
    get_hf_home,
    get_hf_token,
)


class TestFilterRepoObjects:
    def test_filter_repo_objects_with_allow_patterns_string(self):
        items = ["file1.txt", "file2.bin", "file3.json"]
        result = list(
            filter_repo_objects(items, allow_patterns="*.txt")
        )
        assert result == ["file1.txt"]

    def test_filter_repo_objects_with_allow_patterns_list(self):
        items = ["file1.txt", "file2.bin", "file3.json"]
        result = list(
            filter_repo_objects(items, allow_patterns=["*.txt", "*.json"])
        )
        assert result == ["file1.txt", "file3.json"]

    def test_filter_repo_objects_with_ignore_patterns(self):
        items = ["file1.txt", "file2.bin", "file3.json"]
        result = list(
            filter_repo_objects(items, ignore_patterns=["*.bin"])
        )
        assert result == ["file1.txt", "file3.json"]

    def test_filter_repo_objects_with_both_patterns(self):
        items = ["file1.txt", "file2.bin", "file3.json", "file4.py"]
        result = list(
            filter_repo_objects(
                items, allow_patterns=["*.txt", "*.json", "*.bin"], ignore_patterns=["*.bin"]
            )
        )
        assert result == ["file1.txt", "file3.json"]

    def test_filter_repo_objects_with_key_function(self):
        items = [
            {"name": "file1.txt", "size": 100},
            {"name": "file2.bin", "size": 200},
        ]
        result = list(
            filter_repo_objects(items, allow_patterns="*.txt", key=lambda x: x["name"])
        )
        assert result == [{"name": "file1.txt", "size": 100}]

    def test_filter_repo_objects_with_path_objects(self):
        items = [Path("file1.txt"), Path("file2.bin")]
        result = list(filter_repo_objects(items, allow_patterns="*.txt"))
        assert result == [Path("file1.txt")]

    def test_filter_repo_objects_no_patterns(self):
        items = ["file1.txt", "file2.bin"]
        result = list(filter_repo_objects(items))
        assert result == items

    def test_filter_repo_objects_requires_key_for_non_string(self):
        items = [{"name": "file1.txt"}]
        with pytest.raises(ValueError, match="Please provide `key` argument"):
            list(filter_repo_objects(items))


class TestAddWildcardToDirectories:
    def test_add_wildcard_to_directories_with_slash(self):
        assert _add_wildcard_to_directories("dir/") == "dir/*"

    def test_add_wildcard_to_directories_without_slash(self):
        assert _add_wildcard_to_directories("file.txt") == "file.txt"

    def test_add_wildcard_to_directories_empty(self):
        # Empty string causes IndexError in current implementation
        with pytest.raises(IndexError):
            _add_wildcard_to_directories("")


class TestGetHfEndpoint:
    def test_get_hf_endpoint_default(self):
        with patch.dict(os.environ, {}, clear=False):
            if "HF_ENDPOINT" in os.environ:
                del os.environ["HF_ENDPOINT"]
            assert get_hf_endpoint() == "https://huggingface.co"

    def test_get_hf_endpoint_from_env(self):
        with patch.dict(os.environ, {"HF_ENDPOINT": "https://custom.hf.co"}):
            assert get_hf_endpoint() == "https://custom.hf.co"


class TestGetHfHome:
    def test_get_hf_home_default(self):
        with patch.dict(os.environ, {}, clear=False):
            if "HF_HOME" in os.environ:
                del os.environ["HF_HOME"]
            result = get_hf_home()
            assert result == Path.home() / ".cache" / "huggingface"

    def test_get_hf_home_from_env(self):
        with patch.dict(os.environ, {"HF_HOME": "/custom/path"}):
            result = get_hf_home()
            assert result == Path("/custom/path")


class TestGetHfToken:
    @pytest.mark.asyncio
    async def test_get_hf_token_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            token_path = Path(tmpdir) / "token"
            token_path.write_text("test_token_123\n")

            with patch(
                "exo.worker.download.huggingface_utils.get_hf_home",
                return_value=Path(tmpdir),
            ):
                result = await get_hf_token()
                assert result == "test_token_123"

    @pytest.mark.asyncio
    async def test_get_hf_token_not_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "exo.worker.download.huggingface_utils.get_hf_home",
                return_value=Path(tmpdir),
            ):
                result = await get_hf_token()
                assert result is None

    @pytest.mark.asyncio
    async def test_get_hf_token_strips_whitespace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            token_path = Path(tmpdir) / "token"
            token_path.write_text("  test_token_123  \n")

            with patch(
                "exo.worker.download.huggingface_utils.get_hf_home",
                return_value=Path(tmpdir),
            ):
                result = await get_hf_token()
                assert result == "test_token_123"


class TestGetAuthHeaders:
    @pytest.mark.asyncio
    async def test_get_auth_headers_with_token(self):
        with patch(
            "exo.worker.download.huggingface_utils.get_hf_token",
            return_value="test_token_123",
        ):
            result = await get_auth_headers()
            assert result == {"Authorization": "Bearer test_token_123"}

    @pytest.mark.asyncio
    async def test_get_auth_headers_without_token(self):
        with patch(
            "exo.worker.download.huggingface_utils.get_hf_token",
            return_value=None,
        ):
            result = await get_auth_headers()
            assert result == {}


class TestExtractLayerNum:
    def test_extract_layer_num_with_digit(self):
        assert extract_layer_num("layer.5.weight") == 5
        assert extract_layer_num("model.layers.12.attention") == 12

    def test_extract_layer_num_no_digit(self):
        assert extract_layer_num("embedding.weight") is None
        assert extract_layer_num("tokenizer.model") is None

    def test_extract_layer_num_multiple_digits(self):
        # Returns first digit found
        assert extract_layer_num("layer.5.12.weight") == 5


class TestGetAllowPatterns:
    @pytest.fixture
    def sample_shard(self):
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
            end_layer=6,
            n_layers=12,
        )

    def test_get_allow_patterns_with_weight_map(self, sample_shard):
        weight_map = {
            "model.layers.0.weight": "model-00001-of-00002.safetensors",
            "model.layers.5.weight": "model-00001-of-00002.safetensors",
            "model.layers.6.weight": "model-00002-of-00002.safetensors",
            "model.layers.11.weight": "model-00002-of-00002.safetensors",
            "model.embedding.weight": "model-00001-of-00002.safetensors",
        }
        result = get_allow_patterns(weight_map, sample_shard)
        
        # Should include default patterns
        assert "*.json" in result
        assert "*.py" in result
        
        # Should include files for layers 0-6 (within shard range)
        assert "model-00001-of-00002.safetensors" in result
        
        # Should include layer-independent files
        # (embedding.weight has no layer number)
        
        # Should NOT include files only for layers outside range
        # model-00002-of-00002.safetensors has layers 6-11, but shard is 0-6
        # Actually, layer 6 is in range (start_layer <= 6 <= end_layer), so it should be included
        assert "model-00002-of-00002.safetensors" in result

    def test_get_allow_patterns_empty_weight_map(self, sample_shard):
        result = get_allow_patterns({}, sample_shard)
        assert "*.safetensors" in result
        assert "*.json" in result

    def test_get_allow_patterns_layer_independent_only(self, sample_shard):
        weight_map = {
            "model.embedding.weight": "embedding.safetensors",
            "model.output.weight": "output.safetensors",
        }
        result = get_allow_patterns(weight_map, sample_shard)
        assert "embedding.safetensors" in result
        assert "output.safetensors" in result

    def test_get_allow_patterns_single_layer_shard(self):
        shard = PipelineShardMetadata(
            model_meta=ModelMetadata(
                model_id=ModelId("test-model"),
                pretty_name="Test Model",
                storage_size=Memory.from_mb(100),
                n_layers=12,
            ),
            device_rank=0,
            world_size=12,
            start_layer=5,
            end_layer=6,
            n_layers=12,
        )
        weight_map = {
            "model.layers.5.weight": "model-00005-of-00012.safetensors",
            "model.layers.6.weight": "model-00006-of-00012.safetensors",
            "model.layers.7.weight": "model-00007-of-00012.safetensors",
        }
        result = get_allow_patterns(weight_map, shard)
        assert "model-00005-of-00012.safetensors" in result
        assert "model-00006-of-00012.safetensors" in result
        assert "model-00007-of-00012.safetensors" not in result

