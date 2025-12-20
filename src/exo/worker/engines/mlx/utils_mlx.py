"""MLX engine utility functions.

This module provides utilities for MLX model initialization, distributed setup,
KV cache creation, chat template application, and other MLX-specific operations.
"""

import json
import os
import resource
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, cast

from mlx_lm.models.cache import KVCache, QuantizedKVCache, RotatingKVCache
from mlx_lm.models.deepseek_v3 import DeepseekV3Model
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.worker.engines.mlx.constants import (
    CACHE_GROUP_SIZE,
    KV_GROUP_SIZE,
    KV_CACHE_BITS,
    QUANTIZE_MODEL_MODE,
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
try:
    from mlx_lm.utils import quantize_model  # type: ignore
except ImportError:
    quantize_model = None
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
from exo.networking.manual_topology import NodeRole, RoleDetection, infer_role_from_ips
from exo.worker.runner.bootstrap import logger

resource.setrlimit(resource.RLIMIT_NOFILE, (2048, 4096))
"""Increase file descriptor limit for 8-bit models."""


FAST_ROLES = {"A", "B", "C"}


def _local_thunderbolt_ips() -> set[str]:
    ips: set[str] = set()
    for iface in ("en2", "en3", "en4", "en5", "en6", "en7"):
        try:
            output = subprocess.check_output(
                ["ipconfig", "getifaddr", iface], text=True
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
        if output:
            ips.add(output)
    return ips


def _detect_role() -> RoleDetection:
    role_env = os.getenv("EXO_NODE_ROLE")
    ips = _local_thunderbolt_ips()
    inferred = infer_role_from_ips(ips)
    role_raw = role_env or inferred
    if role_raw is None:
        raise RuntimeError("Cannot infer node role; set EXO_NODE_ROLE to A/B/C")
    role = role_raw.upper()
    if role not in FAST_ROLES:
        raise RuntimeError(f"Unsupported role {role}; only A/B/C are allowed")
    role_typed = cast(NodeRole, role)
    has_rdma = _has_rdma()
    if not has_rdma:
        raise RuntimeError("RDMA required on all nodes; ibv_devices missing")
    return RoleDetection(role=role_typed, has_rdma=True)


def _has_rdma() -> bool:
    try:
        output = subprocess.check_output(["ibv_devices"], text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    return any(line.strip().startswith("rdma_") for line in output.splitlines())


def _maybe_quantize_qwen(
    model: nn.Module, config: Any, model_id: str, *, bits: int = 4
) -> None:
    if "qwen3-235b" not in model_id.lower():
        return
    if quantize_model is None:
        raise RuntimeError("mlx quantize_model not available for Qwen3 4-bit path")
    group_size = KV_GROUP_SIZE or 64
    mode = QUANTIZE_MODEL_MODE or "affine"
    quantize_model(
        model,
        config,
        group_size=group_size,
        bits=bits,
        mode=mode,
    )


def get_weights_size(model_shard_meta: ShardMetadata) -> Memory:
    """Calculate the weight size for a model shard.

    For pipeline parallelism, returns size proportional to layer count.
    For tensor parallelism, divides total size by world_size.

    Args:
        model_shard_meta: Shard metadata.

    Returns:
        Memory size of weights for this shard.
    """
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


def mx_barrier(group: mx.distributed.Group | None = None) -> None:
    """Synchronize all processes in a distributed group.

    Args:
        group: Distributed group to synchronize (None for no-op).
    """
    mx.eval(
        mx.distributed.all_sum(
            mx.array(1.0),
            stream=mx.default_stream(mx.Device(mx.cpu)),
            group=group,
        )
    )


def broadcast_from_zero(value: int, group: mx.distributed.Group | None = None) -> int:
    """Broadcast a value from rank 0 to all ranks in a group.

    Args:
        value: Value to broadcast (used by rank 0).
        group: Distributed group (None returns value unchanged).

    Returns:
        The value from rank 0 on all ranks.
    """
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
    """Pydantic model for list of host strings."""

    @classmethod
    def from_hosts(cls, hosts: list[Host]) -> "HostList":
        """Create HostList from list of Host objects.

        Args:
            hosts: List of Host objects.

        Returns:
            HostList with string representations.
        """
        return cls(root=[str(host) for host in hosts])


def mlx_distributed_init(
    bound_instance: BoundInstance,
) -> mx.distributed.Group | None:
    """Initialize MLX distributed computing.

    Sets up distributed MLX based on instance type:
    - MlxRingInstance: Uses ring backend with hostfile
    - MlxJacclInstance: Uses jaccl backend with RDMA devices and coordinator

    Args:
        bound_instance: Instance configuration.

    Returns:
        Initialized distributed group.
    """
    rank = bound_instance.bound_shard.device_rank
    role_detection = _detect_role()
    logger.info(f"Starting initialization for rank {rank} role={role_detection.role}")

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
            devices_file = f"./hosts_{rank}.json"
            ibv_devices_json = json.dumps(ibv_devices)

            with open(devices_file, "w") as f:
                _ = f.write(ibv_devices_json)

            ibv_coordinator = ibv_coordinators[bound_instance.bound_node_id]

            logger.info(f"rank {rank} MLX_IBV_DEVICES: {ibv_devices_json}")
            logger.info(f"rank {rank} MLX_IBV_COORDINATOR: {ibv_coordinator}")
            os.environ["MLX_IBV_DEVICES"] = devices_file
            os.environ["MLX_RANK"] = str(rank)
            os.environ["MLX_IBV_COORDINATOR"] = ibv_coordinator
            group = mx.distributed.init(backend="jaccl", strict=True)

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
        model, config = load_model(model_path, strict=True)
        _maybe_quantize_qwen(
            model,
            config,
            bound_instance.bound_shard.model_meta.model_id,
            bits=4,
        )
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
        if group is None:
            raise RuntimeError("Distributed group initialization failed")
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

    model, config = load_model(model_path, lazy=True, strict=False)
    _maybe_quantize_qwen(model, config, shard_metadata.model_meta.model_id, bits=4)
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
    tokenizer = cast(
        TokenizerWrapper,
        load_tokenizer(
            model_path,
            tokenizer_config_extra={"trust_remote_code": TRUST_REMOTE_CODE},
            # TODO: HACK for Kimi K2 wrong eos token id
            eos_token_ids=[163586]
            if "kimi-k2" in shard_metadata.model_meta.model_id.lower()
            else None,
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
