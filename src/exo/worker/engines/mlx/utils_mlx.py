import json
import os
import resource
import time
from pathlib import Path
from typing import Any, Callable, cast

from mlx_lm.models.cache import KVCache, QuantizedKVCache, RotatingKVCache
from mlx_lm.models.deepseek_v3 import DeepseekV3Model
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.worker.engines.mlx.constants import (
    CACHE_GROUP_SIZE,
    KV_CACHE_BITS,
    TEMPERATURE,
    TRUST_REMOTE_CODE,
)

try:
    from mlx_lm.tokenizer_utils import load_tokenizer
except ImportError:
    from mlx_lm.tokenizer_utils import load as load_tokenizer  # type: ignore
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import load_model
from pydantic import RootModel

from exo.shared.types.api import ChatCompletionMessageText
from exo.shared.types.common import Host
from exo.shared.types.memory import Memory
from exo.shared.types.tasks import ChatCompletionTaskParams
from exo.shared.types.worker.instances import (
    BoundInstance,
    MlxJacclInstance,
    MlxRingInstance,
)
from exo.shared.types.worker.shards import (
    PipelineShardMetadata,
    ShardMetadata,
    TensorShardMetadata,
)
from exo.worker.download.download_utils import build_model_path
from exo.worker.engines.mlx import Model
from exo.worker.engines.mlx.auto_parallel import (
    pipeline_auto_parallel,
    tensor_auto_parallel,
)
from exo.worker.runner.bootstrap import logger

# Needed for 8 bit model
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, 4096))


# TODO: Test this
#  ALSO https://github.com/exo-explore/exo/pull/233#discussion_r2549683673
def get_weights_size(model_shard_meta: ShardMetadata) -> Memory:
    return Memory.from_float_kb(
        (model_shard_meta.end_layer - model_shard_meta.start_layer)
        / model_shard_meta.n_layers
        * model_shard_meta.model_meta.storage_size.in_kb
        / (
            1
            if isinstance(model_shard_meta, PipelineShardMetadata)
            else model_shard_meta.world_size
        )
    )


def mx_barrier(group: mx.distributed.Group | None = None):
    mx.eval(
        mx.distributed.all_sum(
            mx.array(1.0),
            stream=mx.default_stream(mx.Device(mx.cpu)),
            group=group,
        )
    )


def broadcast_from_zero(value: int, group: mx.distributed.Group | None = None):
    if group is None:
        return value

    if group.rank() == 0:
        a = mx.array([value], dtype=mx.int32)
    else:
        a = mx.array([0], dtype=mx.int32)

    m = mx.distributed.all_sum(a, stream=mx.Device(mx.DeviceType.cpu), group=group)
    mx.eval(m)
    return int(m.item())


class HostList(RootModel[list[str]]):
    @classmethod
    def from_hosts(cls, hosts: list[Host]) -> "HostList":
        return cls(root=[str(host) for host in hosts])


def mlx_distributed_init(
    bound_instance: BoundInstance,
) -> mx.distributed.Group:
    """
    Initialize the MLX distributed (runs in thread pool).

    Either hosts or mlx_ibv_devices must be provided:
    - hosts: traditional host-based connectivity using MLX_HOSTFILE
    - mlx_ibv_devices: RDMA connectivity matrix using MLX_IBV_DEVICES
    - mlx_ibv_coordinator: coordinator address (IP:PORT) for RDMA setup
    - strict: if True, raise an error if the distributed backend is not available
    """
    rank = bound_instance.bound_shard.device_rank
    logger.info(f"Starting initialization for rank {rank}")

    # TODO: singleton instances
    match bound_instance.instance:
        case MlxRingInstance(hosts=hosts):
            hostfile = f"./hosts_{rank}.json"
            hosts_json = HostList.from_hosts(hosts).model_dump_json()

            with open(hostfile, "w") as f:
                _ = f.write(hosts_json)

            logger.info(f"rank {rank} hostfile: {hostfile} hosts: {hosts_json}")

            os.environ["MLX_HOSTFILE"] = hostfile
            os.environ["MLX_RANK"] = str(rank)
            os.environ["MLX_RING_VERBOSE"] = "1"
            group = mx.distributed.init(backend="ring", strict=True)

        case MlxJacclInstance(
            ibv_devices=ibv_devices, ibv_coordinators=ibv_coordinators
        ):
            # Use RDMA connectivity matrix
            # MLX may require the devices file to be in the current working directory
            # Use a relative path to ensure it's accessible from the process's working directory
            devices_file = f"./hosts_{rank}.json"
            devices_file_abs = os.path.abspath(devices_file)
            ibv_devices_json = json.dumps(ibv_devices)

            # Ensure the directory exists
            os.makedirs(os.path.dirname(devices_file_abs) or ".", exist_ok=True)
            
            with open(devices_file_abs, "w") as f:
                _ = f.write(ibv_devices_json)
            
            # MLX may need the relative path, not absolute
            # Try relative path first, fall back to absolute if needed

            ibv_coordinator = ibv_coordinators[bound_instance.bound_node_id]

            world_size = bound_instance.bound_shard.world_size
            logger.info(f"rank {rank} MLX_IBV_DEVICES: {ibv_devices_json}")
            logger.info(f"rank {rank} MLX_IBV_COORDINATOR: {ibv_coordinator}")
            logger.info(f"rank {rank} World size: {world_size}, Device rank: {rank}")
            
            # Verify the devices file exists and is readable
            if not os.path.exists(devices_file_abs):
                raise FileNotFoundError(f"RDMA devices file not found: {devices_file_abs}")
            
            # Read back the file to verify it was written correctly
            with open(devices_file_abs, "r") as f:
                file_content = f.read()
            logger.info(f"rank {rank} Devices file content: {file_content}")
            logger.info(f"rank {rank} Devices file path (relative): {devices_file}, (absolute): {devices_file_abs}, (cwd): {os.getcwd()}")
            
            # CRITICAL: Verify all RDMA devices in the matrix actually exist on this system
            # MLX will fail if it tries to use a device that doesn't exist
            import subprocess
            try:
                result = subprocess.run(
                    ["ibv_devices"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    available_devices = set()
                    for line in result.stdout.split("\n"):
                        line = line.strip()
                        # Skip empty lines, headers, and separator lines
                        if not line or line.startswith("device") or line.startswith("------"):
                            continue
                        # Parse device name from line (first column)
                        parts = line.split()
                        if parts and parts[0].startswith("rdma_"):
                            available_devices.add(parts[0])
                    logger.info(f"rank {rank} Available RDMA devices: {sorted(available_devices)}")
                    
                    # Check all devices in the matrix
                    missing_devices = []
                    for i, row in enumerate(ibv_devices):
                        for j, device_name in enumerate(row):
                            if device_name is not None and device_name not in available_devices:
                                missing_devices.append(f"matrix[{i}][{j}]={device_name}")
                    
                    if missing_devices:
                        raise RuntimeError(
                            f"Rank {rank}: RDMA devices file references devices that don't exist on this system: {', '.join(missing_devices)}. "
                            f"Available devices: {sorted(available_devices)}. "
                            f"This will cause MLX to fail to initialize RDMA."
                        )
                else:
                    logger.warning(f"rank {rank} Could not run 'ibv_devices' to verify devices (return code {result.returncode})")
            except FileNotFoundError:
                logger.warning(f"rank {rank} 'ibv_devices' command not found, skipping device verification")
            except subprocess.TimeoutExpired:
                logger.warning(f"rank {rank} 'ibv_devices' command timed out, skipping device verification")
            except Exception as e:
                logger.warning(f"rank {rank} Error checking RDMA devices: {e}, continuing anyway")
            
            # CRITICAL: Clear any conflicting environment variables that might cause MLX to use non-RDMA backends
            # Remove MLX_HOSTFILE if it exists - this would cause MLX to prefer 'ring' backend
            if "MLX_HOSTFILE" in os.environ:
                logger.warning(f"rank {rank} Removing MLX_HOSTFILE to ensure RDMA backend is used")
                del os.environ["MLX_HOSTFILE"]
            
            # Set RDMA-specific environment variables
            # MLX_WORLD_SIZE is required for RDMA initialization
            # Use relative path - MLX may need it relative to current working directory
            os.environ["MLX_IBV_DEVICES"] = devices_file
            os.environ["MLX_RANK"] = str(rank)
            os.environ["MLX_IBV_COORDINATOR"] = ibv_coordinator
            os.environ["MLX_WORLD_SIZE"] = str(world_size)
            
            # Log final environment state
            logger.info(
                f"rank {rank} Final environment: "
                f"MLX_IBV_DEVICES={os.environ.get('MLX_IBV_DEVICES')}, "
                f"MLX_RANK={os.environ.get('MLX_RANK')}, "
                f"MLX_IBV_COORDINATOR={os.environ.get('MLX_IBV_COORDINATOR')}, "
                f"MLX_WORLD_SIZE={os.environ.get('MLX_WORLD_SIZE')}, "
                f"MLX_HOSTFILE={os.environ.get('MLX_HOSTFILE', 'NOT SET')}"
            )
            
            # For RDMA/InfiniBand, we MUST use RDMA - no fallback to 'any' backend
            # MLX requires MLX_IBV_DEVICES to be set for RDMA, and will use RDMA when:
            # - MLX_IBV_DEVICES is set (file path to JSON matrix)
            # - MLX_IBV_COORDINATOR is set (IP:PORT of rank 0)
            # - MLX_WORLD_SIZE is set (number of ranks)
            # - MLX_HOSTFILE is NOT set (would cause ring backend)
            # - backend="any" (MLX doesn't have a separate "rdma" backend name)
            # 
            # When these conditions are met, MLX's "any" backend will use RDMA.
            # We verify RDMA is actually being used by checking group.size() == world_size.
            logger.info(f"rank {rank} Initializing MLX distributed with RDMA (backend='any' with MLX_IBV_DEVICES set)")
            try:
                group = mx.distributed.init(backend="any", strict=True)
            except RuntimeError as e:
                error_msg = str(e)
                logger.error(
                    f"rank {rank} MLX distributed.init failed with: {error_msg}. "
                    f"Environment: MLX_IBV_DEVICES={os.environ.get('MLX_IBV_DEVICES')}, "
                    f"MLX_RANK={os.environ.get('MLX_RANK')}, "
                    f"MLX_IBV_COORDINATOR={os.environ.get('MLX_IBV_COORDINATOR')}, "
                    f"MLX_WORLD_SIZE={os.environ.get('MLX_WORLD_SIZE')}, "
                    f"MLX_HOSTFILE={os.environ.get('MLX_HOSTFILE', 'NOT SET')}. "
                    f"Devices file exists: {os.path.exists(devices_file_abs)}, "
                    f"Devices file path: {devices_file} (abs: {devices_file_abs}), "
                    f"Current working directory: {os.getcwd()}, "
                    f"Devices file content: {file_content[:200] if len(file_content) > 200 else file_content}"
                )
                raise
            
            # CRITICAL: Verify RDMA is actually being used
            # If we get a singleton group, RDMA failed and we must fail immediately
            if group.size() == 1 and world_size > 1:
                raise RuntimeError(
                    f"Rank {rank}: RDMA initialization FAILED! Got singleton group (size=1) "
                    f"but expected distributed RDMA group (world_size={world_size}). "
                    f"This means MLX did not use RDMA despite environment being set correctly. "
                    f"Environment: MLX_IBV_DEVICES={os.environ.get('MLX_IBV_DEVICES')}, "
                    f"MLX_IBV_COORDINATOR={os.environ.get('MLX_IBV_COORDINATOR')}, "
                    f"MLX_WORLD_SIZE={os.environ.get('MLX_WORLD_SIZE')}, "
                    f"MLX_HOSTFILE={os.environ.get('MLX_HOSTFILE', 'NOT SET')}. "
                    f"RDMA must be working for this instance type. Check RDMA configuration and network connectivity."
                )
            
            logger.info(f"rank {rank} Successfully initialized RDMA distributed group (size={group.size()}, expected {world_size})")
            
            # CRITICAL: Verify MLX assigned the correct rank and created a distributed group
            mlx_actual_rank = group.rank()
            mlx_group_size = group.size()
            logger.info(
                f"rank {rank} MLX init result: group.rank()={mlx_actual_rank}, "
                f"group.size()={mlx_group_size}, expected_rank={rank}, expected_world_size={world_size}"
            )
            
            # CRITICAL: For multi-node instances, we MUST have a distributed group
            if mlx_group_size == 1 and world_size > 1:
                raise RuntimeError(
                    f"Rank {rank}: MLX distributed initialization failed! "
                    f"Got singleton group (size=1) but expected distributed group (world_size={world_size}). "
                    f"Environment was: MLX_RANK={os.environ.get('MLX_RANK')}, "
                    f"MLX_IBV_DEVICES={os.environ.get('MLX_IBV_DEVICES')}, "
                    f"MLX_IBV_COORDINATOR={os.environ.get('MLX_IBV_COORDINATOR')}. "
                    f"RDMA backend may not be working correctly on this system."
                )
            
            if mlx_actual_rank != rank:
                logger.error(
                    f"RANK MISMATCH in mlx_distributed_init! "
                    f"Expected rank={rank} but MLX assigned rank={mlx_actual_rank}. "
                    f"This will break pipeline parallelism!"
                )

    logger.info(f"Rank {rank} mlx distributed initialization complete")

    return group


def initialize_mlx(
    bound_instance: BoundInstance,
) -> tuple[Model, TokenizerWrapper, Callable[[mx.array], mx.array]]:
    """
    Initialize the MLX model, tokenizer, and sampler. Runs in the MLX thread.
    """
    mx.random.seed(42)

    set_wired_limit_for_model(get_weights_size(bound_instance.bound_shard))

    sampler: Callable[[mx.array], mx.array] = make_sampler(temp=TEMPERATURE)
    logger.info("Created a sampler")

    if len(bound_instance.instance.shard_assignments.node_to_runner) <= 1:
        logger.info(f"Single device used for {bound_instance.instance}")
        model_path = build_model_path(bound_instance.bound_shard.model_meta.model_id)
        start_time = time.perf_counter()
        model, _ = load_model(model_path, strict=True)
        end_time = time.perf_counter()
        logger.info(f"Time taken to load model: {(end_time - start_time):.2f}s")
        if hasattr(model, "model") and isinstance(model.model, DeepseekV3Model):  # type: ignore
            pass
            # model, config = quantize_model(
            #    model, config, group_size=KV_GROUP_SIZE, bits=ATTENTION_KV_BITS, quant_predicate=quant_predicate, mode=QUANTIZE_MODEL_MODE
            # )

        tokenizer = get_tokenizer(model_path, bound_instance.bound_shard)

    else:
        logger.info("Starting distributed init")
        group = mlx_distributed_init(bound_instance)

        start_time = time.perf_counter()
        model, tokenizer = shard_and_load(bound_instance.bound_shard, group=group)
        end_time = time.perf_counter()
        logger.info(
            f"Time taken to shard and load model: {(end_time - start_time):.2f}s"
        )

    set_wired_limit_for_model(get_weights_size(bound_instance.bound_shard))

    logger.debug(model)

    return cast(Model, model), tokenizer, sampler


def shard_and_load(
    shard_metadata: ShardMetadata,
    group: mx.distributed.Group,
) -> tuple[nn.Module, TokenizerWrapper]:
    model_path = build_model_path(shard_metadata.model_meta.model_id)

    model, _ = load_model(model_path, lazy=True, strict=False)
    logger.debug(model)
    if hasattr(model, "model") and isinstance(model.model, DeepseekV3Model):  # type: ignore
        pass
        # TODO: See if we should quantize the model.
        # def is_attention_layer(path: str) -> bool:
        #     path = path.lower()

        #     return "self_attn" in path and "layernorm" not in path

        # def quant_predicate(path: str, module: nn.Module):
        #     if not isinstance(module, nn.Linear):
        #         return False

        #     return is_attention_layer(path)
        # model, config = quantize_model(
        #        model, config, group_size=KV_GROUP_SIZE, bits=ATTENTION_KV_BITS, quant_predicate=quant_predicate, mode=QUANTIZE_MODEL_MODE
        #    )

    assert isinstance(model, nn.Module)

    tokenizer = get_tokenizer(model_path, shard_metadata)

    logger.info(f"Group size: {group.size()}, group rank: {group.rank()}")

    match shard_metadata:
        case TensorShardMetadata():
            logger.info(f"loading model from {model_path} with tensor parallelism")
            model = tensor_auto_parallel(model, group)
        case PipelineShardMetadata():
            logger.info(f"loading model from {model_path} with pipeline parallelism")
            model = pipeline_auto_parallel(model, group, shard_metadata)

    mx.eval(model.parameters())

    # TODO: Do we need this?
    mx.eval(model)

    logger.debug("SHARDED")
    logger.debug(model)

    # Synchronize processes before generation to avoid timeout
    mx_barrier(group)

    return model, tokenizer


def get_tokenizer(model_path: Path, shard_metadata: ShardMetadata):
    # TODO: Let's move away from this custom logic to mlx_lm.load()
    if "kimi-k2" in shard_metadata.model_meta.model_id.lower():
        eos_token_ids = [163586]

    elif "glm" in shard_metadata.model_meta.model_id.lower():
        eos_token_ids = [151336, 151329, 151338]

    else:
        eos_token_ids = None

    tokenizer = cast(
        TokenizerWrapper,
        load_tokenizer(
            model_path,
            tokenizer_config_extra={"trust_remote_code": TRUST_REMOTE_CODE},
            eos_token_ids=eos_token_ids,
        ),
    )
    assert isinstance(tokenizer, TokenizerWrapper)

    return tokenizer


def apply_chat_template(
    tokenizer: TokenizerWrapper,
    chat_task_data: ChatCompletionTaskParams,
) -> str:
    # Now we can properly access the messages
    messages = chat_task_data.messages

    formatted_messages: list[dict[str, Any]] = []
    for _, message in enumerate(messages):
        if isinstance(message.content, ChatCompletionMessageText):
            message.content = message.content.text
        if isinstance(message.content, list):
            if len(message.content) != 1:
                logger.warning("Received malformed prompt")
                continue

            message.content = message.content[0].text
        if message.content is None and message.thinking is None:
            continue

        # Null values are not valid when applying templates in tokenizer
        formatted_messages.append(
            {k: v for k, v in message.model_dump().items() if v is not None}  # type: ignore
        )

    prompt: str = tokenizer.apply_chat_template(  # type: ignore
        formatted_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    return prompt  # type: ignore


class NullKVCache(KVCache):
    """
    A KVCache that pretends to exist but holds zero tokens.
    It satisfies .state/.meta_state and never allocates real keys/values.
    """

    def __init__(self, dtype: mx.Dtype = mx.float16):
        super().__init__()
        # zero-length K/V so shapes/dtypes are defined but empty
        self.keys = mx.zeros((1, 1, 0, 1), dtype=dtype)
        self.values = mx.zeros((1, 1, 0, 1), dtype=dtype)
        self.offset = 0

    @property
    def state(self) -> tuple[mx.array, mx.array]:
        # matches what mx.save_safetensors / mx.eval expect
        return self.keys, self.values

    @state.setter
    def state(self, v: tuple[mx.array, mx.array]) -> None:
        raise NotImplementedError("We should not be setting a NullKVCache.")


def make_kv_cache(
    model: Model, max_kv_size: int | None = None, keep: int = 0
) -> list[KVCache | RotatingKVCache | QuantizedKVCache]:
    assert hasattr(model, "layers")

    if max_kv_size is None:
        if KV_CACHE_BITS is None:
            logger.info("Using default KV cache")
            return [KVCache() for _ in model.layers]
        else:
            logger.info("Using quantized KV cache")
            return [
                QuantizedKVCache(group_size=CACHE_GROUP_SIZE, bits=KV_CACHE_BITS)
                for _ in model.layers
            ]
    else:
        logger.info(f"Using rotating KV cache with {max_kv_size=} with {keep=}")
        return [RotatingKVCache(max_size=max_kv_size, keep=keep) for _ in model.layers]


def mlx_force_oom(size: int = 40000) -> None:
    """
    Force an Out-Of-Memory (OOM) error in MLX by performing large tensor operations.
    """
    mx.set_default_device(mx.gpu)
    a = mx.random.uniform(shape=(size, size), dtype=mx.float32)
    b = mx.random.uniform(shape=(size, size), dtype=mx.float32)
    mx.eval(a, b)
    c = mx.matmul(a, b)
    d = mx.matmul(a, c)
    e = mx.matmul(b, c)
    f = mx.sigmoid(d + e)
    mx.eval(f)


def set_wired_limit_for_model(model_size: Memory):
    """
    A context manager to temporarily change the wired limit.

    Note, the wired limit should not be changed during an async eval.  If an
    async eval could be running pass in the streams to synchronize with prior
    to exiting the context manager.
    """
    if not mx.metal.is_available():
        return

    model_bytes = model_size.in_bytes
    max_rec_size = int(mx.metal.device_info()["max_recommended_working_set_size"])
    if model_bytes > 0.9 * max_rec_size:
        model_mb = model_bytes // 2**20
        max_rec_mb = max_rec_size // 2**20
        logger.warning(
            f"Generating with a model that requires {model_mb} MB "
            f"which is close to the maximum recommended size of {max_rec_mb} "
            "MB. This can be slow. See the documentation for possible work-arounds: "
            "https://github.com/ml-explore/mlx-lm/tree/main#large-models"
        )
    kv_bytes = int(0.02 * model_bytes)
    target_cache = int(1.10 * (model_bytes + kv_bytes))
    target_cache = min(target_cache, max_rec_size)
    mx.set_cache_limit(target_cache)
    mx.set_wired_limit(max_rec_size)
    logger.info(
        f"Wired limit set to {max_rec_size}. Cache limit set to {target_cache}."
    )
