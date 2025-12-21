from abc import ABC, abstractmethod
from functools import partial
from inspect import signature
from typing import TYPE_CHECKING, Callable, Protocol, cast

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import (
    shard_inplace,
    shard_linear,
    sum_gradients,
)
from mlx_lm.models.cache import (
    _BaseCache,  # pyright: ignore[reportPrivateUsage]
)
from mlx_lm.models.deepseek_v3 import DeepseekV3MLP
from mlx_lm.models.deepseek_v3 import Model as DeepseekV3Model
from mlx_lm.models.llama import Model as LlamaModel
from mlx_lm.models.qwen3_moe import Model as Qwen3MoeModel
from mlx_lm.models.qwen3_moe import Qwen3MoeSparseMoeBlock

from exo.shared.types.worker.shards import (
    PipelineShardMetadata,
)


class _LayerCallable(Protocol):
    """Structural type that any compatible layer must satisfy.

    We require a single positional input of type ``mx.array`` and an
    ``mx.array`` output, while permitting arbitrary *args / **kwargs so this
    protocol matches the vast majority of `mlx.nn.Module` subclasses.
    """

    def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array: ...


class CustomMlxLayer(nn.Module):
    """Base class for replacing an MLX layer with a custom implementation."""

    def __init__(self, original_layer: _LayerCallable):
        super().__init__()
        # Set twice to avoid __setattr__ recursion
        object.__setattr__(self, "_original_layer", original_layer)
        self.original_layer: _LayerCallable = original_layer

    # Calls __getattr__ for any attributes not found on nn.Module (e.g. use_sliding)
    if not TYPE_CHECKING:

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                original_layer = object.__getattribute__(self, "_original_layer")
                return object.__getattribute__(original_layer, name)


class PipelineFirstLayer(CustomMlxLayer):
    def __init__(
        self,
        original_layer: _LayerCallable,
        r: int,
        group: mx.distributed.Group,
    ):
        super().__init__(original_layer)
        self.r: int = r
        self.group = group
        # Check if group is singleton (size == 1) - distributed operations will fail
        self.is_singleton = group.size() == 1

    def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
        from exo.worker.runner.bootstrap import logger
        
        mlx_group_rank = self.group.rank()
        logger.debug(
            f"PipelineFirstLayer[rank={self.r}]: Input shape={x.shape}, "
            f"group_rank={mlx_group_rank}, group_size={self.group.size()}, singleton={self.is_singleton}"
        )
        
        if self.r != 0 and not self.is_singleton:
            logger.debug(f"PipelineFirstLayer[rank={self.r}]: Receiving from rank {self.r - 1}")
            x = mx.distributed.recv_like(x, (self.r - 1), group=self.group)
            logger.debug(f"PipelineFirstLayer[rank={self.r}]: Received shape={x.shape}")
        
        output = self.original_layer(x, *args, **kwargs)
        logger.debug(f"PipelineFirstLayer[rank={self.r}]: Output shape={output.shape}")
        return output


class PipelinePassThroughLayer(nn.Module):
    """A pass-through layer for nodes with 0 layers (KV cache only).
    
    This layer receives input from the previous rank, forwards it without processing,
    and sends it to the next rank. It's used when a node has no layers assigned
    but still needs to participate in the pipeline for KV cache purposes.
    """
    def __init__(
        self,
        r: int,
        s: int,
        group: mx.distributed.Group,
    ):
        super().__init__()
        self.r: int = r
        self.s: int = s
        self.group = group
        self.is_singleton = group.size() == 1

    def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
        from exo.worker.runner.bootstrap import logger
        
        mlx_group_rank = self.group.rank()
        logger.debug(
            f"PipelinePassThroughLayer[rank={self.r}]: Input shape={x.shape}, "
            f"group_rank={mlx_group_rank}, group_size={self.group.size()}, singleton={self.is_singleton}"
        )
        
        # Receive from previous rank if not rank 0
        if self.r != 0 and not self.is_singleton:
            logger.debug(f"PipelinePassThroughLayer[rank={self.r}]: Receiving from rank {self.r - 1}")
            x = mx.distributed.recv_like(x, (self.r - 1), group=self.group)
            logger.debug(f"PipelinePassThroughLayer[rank={self.r}]: Received shape={x.shape}")
        
        # Pass through without processing (identity function)
        output = x
        
        # Send to next rank if not last rank
        if self.r != self.s - 1 and not self.is_singleton:
            next_rank = (self.r + 1) % self.s
            logger.debug(f"PipelinePassThroughLayer[rank={self.r}]: Sending shape {output.shape} to rank {next_rank}")
            output = mx.distributed.send(
                output, next_rank, group=self.group
            )
        
        # All gather for final output
        if not self.is_singleton:
            logger.debug(f"PipelinePassThroughLayer[rank={self.r}]: Participating in all_gather")
            gathered = mx.distributed.all_gather(output, group=self.group)
            logger.debug(f"PipelinePassThroughLayer[rank={self.r}]: all_gather returned shape {gathered.shape}")
            output = gathered[-output.shape[0] :]
            logger.debug(f"PipelinePassThroughLayer[rank={self.r}]: Returning shape {output.shape}")
        return output


class PipelineLastLayer(CustomMlxLayer):
    def __init__(
        self,
        original_layer: _LayerCallable,
        r: int,
        s: int,
        group: mx.distributed.Group,
    ):
        super().__init__(original_layer)
        self.r: int = r
        self.s: int = s
        self.group = group
        self.original_layer_signature = signature(self.original_layer.__call__)
        # Check if group is singleton (size == 1) - distributed operations will fail
        self.is_singleton = group.size() == 1

    def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
        from exo.worker.runner.bootstrap import logger
        
        cache = self.original_layer_signature.bind_partial(
            x, *args, **kwargs
        ).arguments.get("cache", None)

        assert cache is None or issubclass(type(cache), _BaseCache)  # type: ignore

        output: mx.array = self.original_layer(x, *args, **kwargs)
        
        # Verify group rank matches our device rank
        mlx_group_rank = self.group.rank()
        if mlx_group_rank != self.r:
            logger.error(
                f"PipelineLastLayer: Rank mismatch! device_rank={self.r} but "
                f"group.rank()={mlx_group_rank}"
            )

        is_last_rank = self.r == self.s - 1
        
        if not is_last_rank and not self.is_singleton:
            # Send output to next rank in pipeline
            next_rank = (self.r + 1) % self.s
            logger.debug(
                f"Rank {self.r}: Sending output shape {output.shape} to rank {next_rank}"
            )
            send_result = mx.distributed.send(
                output, next_rank, group=self.group
            )
            if cache is not None:
                # This change happened upstream - check out mlx github somewhere??
                cache.keys = mx.depends(cache.keys, send_result)  # type: ignore[reportUnknownMemberType]
            # Non-last ranks: need to receive final output from last rank
            # The send creates a dependency that ensures pipeline ordering
            # We'll participate in all_gather but only use the last rank's output
            if not self.is_singleton:
                logger.debug(
                    f"Rank {self.r}: Participating in all_gather (expecting final output from rank {self.s - 1})"
                )
                # Ensure send completes before all_gather by using the output (not send_result)
                # all_gather will wait for all ranks, including the last rank to finish processing
                gathered = mx.distributed.all_gather(output, group=self.group)
                logger.debug(
                    f"Rank {self.r}: all_gather returned shape {gathered.shape}, "
                    f"extracting last {output.shape[0]} elements"
                )
                # all_gather concatenates along first dim: [rank0, rank1, ..., rankN-1]
                # Extract only the last rank's output (the final output we need)
                final_output = gathered[-output.shape[0]:]
                logger.debug(f"Rank {self.r}: Returning final output shape {final_output.shape}")
                return final_output
        else:
            # Last rank: produce final output and participate in all_gather to broadcast it
            if not self.is_singleton:
                logger.debug(
                    f"Rank {self.r} (LAST): Producing final output shape {output.shape}, "
                    f"broadcasting via all_gather"
                )
                # Broadcast final output to all ranks via all_gather
                # This ensures all ranks get the final output for next token generation
                gathered = mx.distributed.all_gather(output, group=self.group)
                logger.debug(
                    f"Rank {self.r} (LAST): all_gather returned shape {gathered.shape}, "
                    f"extracting last {output.shape[0]} elements"
                )
                # Return our output (last seq_len elements from gathered array)
                final_output = gathered[-output.shape[0]:]
                logger.debug(f"Rank {self.r} (LAST): Returning final output shape {final_output.shape}")
                return final_output
        
        return output


def _inner_model(model: nn.Module) -> nn.Module:
    inner = getattr(model, "model", None)
    if isinstance(inner, nn.Module):
        return inner

    inner = getattr(model, "transformer", None)
    if isinstance(inner, nn.Module):
        return inner

    raise ValueError("Model must either have a 'model' or 'transformer' attribute")


def _get_layers(inner_model_instance: nn.Module) -> list[_LayerCallable]:
    # Handle both model.layers and model.h cases
    layers: list[_LayerCallable]
    if hasattr(inner_model_instance, "layers"):
        layers = cast(list[_LayerCallable], inner_model_instance.layers)
    elif hasattr(inner_model_instance, "h"):
        layers = cast(list[_LayerCallable], inner_model_instance.h)
    else:
        raise ValueError("Model must have either a 'layers' or 'h' attribute")

    return layers


def _set_layers(model: nn.Module, layers: list[_LayerCallable]) -> None:
    inner_model_instance = _inner_model(model)
    if hasattr(inner_model_instance, "layers"):
        inner_model_instance.layers = layers

        # Update DeepSeek V3 specific parameters when layers are shrunk
        if isinstance(model, DeepseekV3Model) and hasattr(
            inner_model_instance, "num_layers"
        ):
            inner_model_instance.start_idx = 0
            inner_model_instance.end_idx = len(layers)
            inner_model_instance.num_layers = len(layers)
    elif hasattr(inner_model_instance, "h"):
        inner_model_instance.h = layers
    else:
        raise ValueError("Model must have either a 'layers' or 'h' attribute")


def pipeline_auto_parallel(
    model: nn.Module,
    group: mx.distributed.Group,
    model_shard_meta: PipelineShardMetadata,
) -> nn.Module:
    """
    Automatically parallelize a model across multiple devices.
    Args:
    model: The model to parallelize (must have a 'layers' or 'h' property)
    model_shard_meta: The metadata for the model shard
    Returns:
    The parallelized model
    """
    from exo.worker.runner.bootstrap import logger
    
    start_layer, end_layer = model_shard_meta.start_layer, model_shard_meta.end_layer
    device_rank, world_size = model_shard_meta.device_rank, model_shard_meta.world_size
    mlx_group_rank = group.rank()
    
    # CRITICAL: Verify MLX group rank matches our device_rank
    if mlx_group_rank != device_rank:
        logger.error(
            f"RANK MISMATCH! device_rank={device_rank} but group.rank()={mlx_group_rank}. "
            f"This will cause incorrect pipeline communication!"
        )
    else:
        logger.info(
            f"Pipeline setup: device_rank={device_rank}, group.rank()={mlx_group_rank}, "
            f"world_size={world_size}, layers=[{start_layer}, {end_layer})"
        )

    inner_model_instance: nn.Module = _inner_model(model)

    # Handle both model.layers and model.h cases
    layers: list[_LayerCallable] = _get_layers(inner_model_instance)

    # Handle nodes with 0 layers (KV cache only)
    if start_layer == end_layer:
        # This node has no layers to process, only contributes to KV cache
        # Create a single pass-through layer that handles receive/send but doesn't process
        layers = [PipelinePassThroughLayer(device_rank, world_size, group=group)]
        logger.info(f"Rank {device_rank}: Using pass-through layer (0 layers, KV cache only)")
    else:
        layers = layers[start_layer:end_layer]
        logger.info(
            f"Rank {device_rank}: Processing layers {start_layer}-{end_layer} "
            f"({len(layers)} layers)"
        )
        layers[0] = PipelineFirstLayer(layers[0], device_rank, group=group)
        layers[-1] = PipelineLastLayer(
            layers[-1],
            device_rank,
            world_size,
            group=group,
        )

    _set_layers(model, layers)

    assert isinstance(layers, list), (
        "Expected a list of layers after auto-parallel initialisation"
    )

    return model


def tensor_auto_parallel(
    model: nn.Module,
    group: mx.distributed.Group,
) -> nn.Module:
    all_to_sharded_linear = partial(
        shard_linear,
        sharding="all-to-sharded",
        group=group,
    )
    sharded_to_all_linear = partial(
        shard_linear,
        sharding="sharded-to-all",
        group=group,
    )

    all_to_sharded_linear_in_place = partial(
        shard_inplace,
        sharding="all-to-sharded",
        group=group,
    )
    sharded_to_all_linear_in_place = partial(
        shard_inplace,
        sharding="sharded-to-all",
        group=group,
    )

    if isinstance(model, LlamaModel):
        tensor_parallel_sharding_strategy = LlamaShardingStrategy(
            group,
            all_to_sharded_linear,
            sharded_to_all_linear,
            all_to_sharded_linear_in_place,
            sharded_to_all_linear_in_place,
        )
    elif isinstance(model, DeepseekV3Model):
        tensor_parallel_sharding_strategy = DeepSeekShardingStrategy(
            group,
            all_to_sharded_linear,
            sharded_to_all_linear,
            all_to_sharded_linear_in_place,
            sharded_to_all_linear_in_place,
        )
    elif isinstance(model, Qwen3MoeModel):
        tensor_parallel_sharding_strategy = QwenShardingStrategy(
            group,
            all_to_sharded_linear,
            sharded_to_all_linear,
            all_to_sharded_linear_in_place,
            sharded_to_all_linear_in_place,
        )
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")

    return tensor_parallel_sharding_strategy.shard_model(model)


class TensorParallelShardingStrategy(ABC):
    def __init__(
        self,
        group: mx.distributed.Group,
        all_to_sharded_linear: Callable[..., nn.Linear],
        sharded_to_all_linear: Callable[..., nn.Linear],
        all_to_sharded_linear_in_place: Callable[..., None],
        sharded_to_all_linear_in_place: Callable[..., None],
    ):
        self.all_to_sharded_linear = all_to_sharded_linear
        self.sharded_to_all_linear = sharded_to_all_linear
        self.all_to_sharded_linear_in_place = all_to_sharded_linear_in_place
        self.sharded_to_all_linear_in_place = sharded_to_all_linear_in_place
        self.group = group
        self.N = group.size()

    @abstractmethod
    def shard_model(self, model: nn.Module) -> nn.Module: ...


class LlamaShardingStrategy(TensorParallelShardingStrategy):
    def shard_model(self, model: nn.Module) -> nn.Module:
        model = cast(LlamaModel, model)
        for layer in model.layers:
            layer.self_attn.q_proj = self.all_to_sharded_linear(layer.self_attn.q_proj)
            layer.self_attn.k_proj = self.all_to_sharded_linear(layer.self_attn.k_proj)
            layer.self_attn.v_proj = self.all_to_sharded_linear(layer.self_attn.v_proj)
            layer.self_attn.o_proj = self.sharded_to_all_linear(layer.self_attn.o_proj)
            layer.self_attn.n_heads //= self.N
            if layer.self_attn.n_kv_heads is not None:
                layer.self_attn.n_kv_heads //= self.N

            layer.mlp.gate_proj = self.all_to_sharded_linear(layer.mlp.gate_proj)
            layer.mlp.down_proj = self.sharded_to_all_linear(layer.mlp.down_proj)
            layer.mlp.up_proj = self.all_to_sharded_linear(layer.mlp.up_proj)

        return model


class DeepSeekShardingStrategy(TensorParallelShardingStrategy):
    def shard_model(self, model: nn.Module) -> nn.Module:
        model = cast(DeepseekV3Model, model)
        for layer in model.layers:
            # Shard the self attention
            if layer.self_attn.q_lora_rank is None:  # pyright: ignore[reportUnnecessaryComparison]
                # Unfortunately, q_lora_rank can be None despite typing hints.
                layer.self_attn.q_proj = self.all_to_sharded_linear(
                    layer.self_attn.q_proj
                )
            else:
                layer.self_attn.q_b_proj = self.all_to_sharded_linear(
                    layer.self_attn.q_b_proj
                )
            layer.self_attn.kv_b_proj = self.all_to_sharded_linear(
                layer.self_attn.kv_b_proj
            )
            layer.self_attn.o_proj = self.sharded_to_all_linear(layer.self_attn.o_proj)
            layer.self_attn.num_heads //= self.N

            # Shard the MLP
            if isinstance(layer.mlp, DeepseekV3MLP):
                layer.mlp.gate_proj = self.all_to_sharded_linear(layer.mlp.gate_proj)
                layer.mlp.down_proj = self.sharded_to_all_linear(layer.mlp.down_proj)
                layer.mlp.up_proj = self.all_to_sharded_linear(layer.mlp.up_proj)

            # Shard the MoE. Shard in place since the MoE should be responsible
            # for aggregating the results.
            else:
                self.all_to_sharded_linear_in_place(layer.mlp.shared_experts.gate_proj)
                self.sharded_to_all_linear_in_place(layer.mlp.shared_experts.down_proj)
                self.all_to_sharded_linear_in_place(layer.mlp.shared_experts.up_proj)
                self.all_to_sharded_linear_in_place(layer.mlp.switch_mlp.gate_proj)
                self.sharded_to_all_linear_in_place(layer.mlp.switch_mlp.down_proj)
                self.all_to_sharded_linear_in_place(layer.mlp.switch_mlp.up_proj)
                layer.mlp = ShardedDeepseekV3MoE(layer.mlp)  # type: ignore
                layer.mlp.sharding_group = self.group

        return model


class ShardedDeepseekV3MoE(CustomMlxLayer):
    def __init__(self, layer: _LayerCallable):
        super().__init__(layer)
        self.sharding_group: mx.distributed.Group | None = None

    def __call__(self, x: mx.array) -> mx.array:
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)
        y = self.original_layer.__call__(x)
        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)
        return y


class QwenShardingStrategy(TensorParallelShardingStrategy):
    def shard_model(self, model: nn.Module) -> nn.Module:
        model = cast(Qwen3MoeModel, model)
        for layer in model.layers:
            # Shard the self attention
            layer.self_attn.q_proj = self.all_to_sharded_linear(layer.self_attn.q_proj)
            layer.self_attn.k_proj = self.all_to_sharded_linear(layer.self_attn.k_proj)
            layer.self_attn.v_proj = self.all_to_sharded_linear(layer.self_attn.v_proj)
            layer.self_attn.o_proj = self.sharded_to_all_linear(layer.self_attn.o_proj)
            layer.self_attn.n_heads //= self.N
            layer.self_attn.n_kv_heads //= self.N

            # Shard the MoE. Shard in place since the MoE should be responsible
            # for aggregating the results.
            if isinstance(layer.mlp, Qwen3MoeSparseMoeBlock):
                self.all_to_sharded_linear_in_place(layer.mlp.switch_mlp.gate_proj)
                self.sharded_to_all_linear_in_place(layer.mlp.switch_mlp.down_proj)
                self.all_to_sharded_linear_in_place(layer.mlp.switch_mlp.up_proj)
                layer.mlp = ShardedQwenMoE(layer.mlp)  # type: ignore
                layer.mlp.sharding_group = self.group

            # Shard the MLP
            else:
                layer.mlp.gate_proj = self.all_to_sharded_linear(layer.mlp.gate_proj)
                layer.mlp.down_proj = self.sharded_to_all_linear(layer.mlp.down_proj)
                layer.mlp.up_proj = self.all_to_sharded_linear(layer.mlp.up_proj)

        return model


class ShardedQwenMoE(CustomMlxLayer):
    def __init__(self, layer: _LayerCallable):
        super().__init__(layer)
        self.sharding_group: mx.distributed.Group | None = None

    def __call__(self, x: mx.array) -> mx.array:
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)
        y = self.original_layer.__call__(x)
        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)
        return y
