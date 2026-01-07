import pytest
from exo.shared.types.worker.instances import (
    InstanceConfig,
    MlxRingInstance,
    ShardAssignments,
    InstanceId,
)
from exo.shared.types.models import ModelId
from exo.shared.types.common import NodeId, Host
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.shared.types.models import ModelMetadata
from exo.shared.types.memory import Memory

def test_instance_config_defaults():
    config = InstanceConfig()
    assert config.max_input_tokens is None
    assert config.max_output_tokens is None
    assert config.temperature is None
    assert config.kv_cache_bits is None

def test_instance_config_assignment():
    config = InstanceConfig(
        max_input_tokens=1024,
        max_output_tokens=100,
        temperature=0.7,
        kv_cache_bits=4
    )
    assert config.max_input_tokens == 1024
    assert config.max_output_tokens == 100
    assert config.temperature == 0.7
    assert config.kv_cache_bits == 4

def test_mlx_ring_instance_serialization_with_config():
    model_id = ModelId("test-model")
    runner_id = "runner-1"
    node_id = NodeId("node-1")
    
    shard_assignments = ShardAssignments(
        model_id=model_id,
        runner_to_shard={
            runner_id: PipelineShardMetadata(
                model_meta=ModelMetadata(
                    model_id=model_id,
                    pretty_name="Test Model",
                    storage_size=Memory.from_bytes(1000),
                    n_layers=10,
                    hidden_size=1024,
                    supports_tensor=True
                ),
                device_rank=0,
                world_size=1,
                start_layer=0,
                end_layer=10,
                n_layers=10
            )
        },
        node_to_runner={node_id: runner_id}
    )
    
    config = InstanceConfig(
        max_input_tokens=1024,
        max_output_tokens=100,
        temperature=0.7,
        kv_cache_bits=4
    )
    
    instance = MlxRingInstance(
        instance_id=InstanceId("test-instance"),
        shard_assignments=shard_assignments,
        config=config,
        hosts_by_node={node_id: [Host(ip="127.0.0.1", port=8080)]},
        ephemeral_port=8080
    )
    
    # Test serialization
    serialized = instance.model_dump(by_alias=True)
    
    # Verify TaggedModel structure wrapping
    assert "MlxRingInstance" in serialized
    inner = serialized["MlxRingInstance"]
    
    assert "config" in inner
    serialized_config = inner["config"]
    assert serialized_config["maxInputTokens"] == 1024
    assert serialized_config["maxOutputTokens"] == 100
    assert serialized_config["temperature"] == 0.7
    assert serialized_config["kvCacheBits"] == 4
    
    # Test deserialization
    deserialized = MlxRingInstance.model_validate(serialized)
    assert deserialized.config.max_input_tokens == 1024
    assert deserialized.config.max_output_tokens == 100
    assert deserialized.config.temperature == 0.7
    assert deserialized.config.kv_cache_bits == 4

def test_instance_config_default_in_instance():
    # Verify that if config is not provided, it defaults to empty config
    model_id = ModelId("test-model")
    runner_id = "runner-1"
    node_id = NodeId("node-1")
    
    shard_assignments = ShardAssignments(
        model_id=model_id,
        runner_to_shard={
            runner_id: PipelineShardMetadata(
                model_meta=ModelMetadata(
                    model_id=model_id,
                    pretty_name="Test Model",
                    storage_size=Memory.from_bytes(1000),
                    n_layers=10,
                    hidden_size=1024,
                    supports_tensor=True
                ),
                device_rank=0,
                world_size=1,
                start_layer=0,
                end_layer=10,
                n_layers=10
            )
        },
        node_to_runner={node_id: runner_id}
    )
    
    instance = MlxRingInstance(
        instance_id=InstanceId("test-instance"),
        shard_assignments=shard_assignments,
        hosts_by_node={node_id: [Host(ip="127.0.0.1", port=8080)]},
        ephemeral_port=8080
    )
    
    assert instance.config is not None
    assert instance.config.max_input_tokens is None
    assert instance.config.temperature is None
