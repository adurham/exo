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
        if self.r != 0 and not self.is_singleton:
            x = mx.distributed.recv_like(x, (self.r - 1), group=self.group)
        return self.original_layer(x, *args, **kwargs)


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
        # Receive from previous rank if not rank 0
        if self.r != 0 and not self.is_singleton:
            x = mx.distributed.recv_like(x, (self.r - 1), group=self.group)
        
        # Pass through without processing (identity function)
        output = x
        
        # Send to next rank if not last rank
        if self.r != self.s - 1 and not self.is_singleton:
            output = mx.distributed.send(
                output, (self.r + 1) % self.s, group=self.group
            )
        
        # All gather for final output
        if not self.is_singleton:
            output = mx.distributed.all_gather(output, group=self.group)[-output.shape[0] :]
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
        cache = self.original_layer_signature.bind_partial(
            x, *args, **kwargs
        ).arguments.get("cache", None)

        assert cache is None or issubclass(type(cache), _BaseCache)  # type: ignore

        output: mx.array = self.original_layer(x, *args, **kwargs)

        if self.r != self.s - 1 and not self.is_singleton:
            # Send output to next rank in pipeline
            # Store the actual output before sending (send returns a dependency)
            actual_output = output
            send_result = mx.distributed.send(
                output, (self.r + 1) % self.s, group=self.group
            )
            if cache is not None:
                # This change happened upstream - check out mlx github somewhere??
                cache.keys = mx.depends(cache.keys, send_result)  # type: ignore[reportUnknownMemberType]
            # Non-last ranks: participate in all_gather to receive final output from last rank
            # Use the actual layer output (not the send dependency) for all_gather
            if not self.is_singleton:
                # all_gather collects outputs from all ranks
                # We use actual_output (our intermediate output) so all_gather has consistent shapes
                gathered = mx.distributed.all_gather(actual_output, group=self.group)
                # Extract the last rank's output (the final output we need)
                # all_gather concatenates in rank order, so last rank's output is at the end
                return gathered[-actual_output.shape[0]:]
        else:
            # Last rank: produce final output and broadcast to all ranks via all_gather
            if not self.is_singleton:
                # Broadcast final output to all ranks so they can use it for next token
                gathered = mx.distributed.all_gather(output, group=self.group)
                # Return our output (the last seq_len elements from the gathered array)
                return gathered[-output.shape[0]:]
        
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
    start_layer, end_layer = model_shard_meta.start_layer, model_shard_meta.end_layer
    device_rank, world_size = model_shard_meta.device_rank, model_shard_meta.world_size

    inner_model_instance: nn.Module = _inner_model(model)

    # Handle both model.layers and model.h cases
    layers: list[_LayerCallable] = _get_layers(inner_model_instance)

    # Handle nodes with 0 layers (KV cache only)
    if start_layer == end_layer:
        # This node has no layers to process, only contributes to KV cache
        # Create a single pass-through layer that handles receive/send but doesn't process
        layers = [PipelinePassThroughLayer(device_rank, world_size, group=group)]
    else:
        layers = layers[start_layer:end_layer]
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
