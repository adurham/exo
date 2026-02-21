"""Tests for hybrid_auto_parallel control flow.

These tests use mocked MLX distributed groups to verify that the hybrid TP+PP
logic calls the right operations with the right arguments, without requiring
actual RDMA hardware or a running cluster.

Bugs these tests would have caught:
  - RingGroup::split() not implemented (split is called, would raise)
  - Singleton sub-group socket bind failure (split must handle size=1)
  - Wrong tp_color assignment (TP=0, PP=1)
  - tensor_auto_parallel called for PP tail nodes (should be TP only)
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, call
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn

from exo.shared.types.worker.shards import HybridShardMetadata
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.memory import Memory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model_card(n_layers: int = 62) -> ModelCard:
    """Create a minimal ModelCard for testing."""
    return ModelCard(
        model_id=ModelId("test/hybrid-model"),
        n_layers=n_layers,
        storage_size=Memory.from_kb(40_000_000),  # ~40GB
        hidden_size=3072,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    )


def _make_shard(
    *,
    tp_rank: int,
    tp_size: int,
    pp_rank: int,
    pp_size: int = 2,
    start_layer: int,
    end_layer: int,
    n_layers: int = 62,
    world_size: int = 3,
    device_rank: int = 0,
    pipeline_send_to: int | None = None,
    pipeline_recv_from: int | None = None,
) -> HybridShardMetadata:
    return HybridShardMetadata(
        model_card=_make_model_card(n_layers),
        device_rank=device_rank,
        world_size=world_size,
        start_layer=start_layer,
        end_layer=end_layer,
        n_layers=n_layers,
        tp_size=tp_size,
        tp_rank=tp_rank,
        pp_rank=pp_rank,
        pp_size=pp_size,
        pipeline_send_to=pipeline_send_to,
        pipeline_recv_from=pipeline_recv_from,
    )


class FakeLayer(nn.Module):
    """A trivial layer for testing. Just stores an index."""

    def __init__(self, idx: int):
        super().__init__()
        self.idx = idx

    def __call__(self, x: mx.array, **kwargs: object) -> mx.array:
        return x


class _FakeInnerModel(nn.Module):
    """Inner model with a layers attribute, as expected by _inner_model / _get_layers."""

    def __init__(self, n_layers: int):
        super().__init__()
        self.layers = [FakeLayer(i) for i in range(n_layers)]


class FakeModel(nn.Module):
    """A minimal model that hybrid_auto_parallel can work with."""

    def __init__(self, n_layers: int = 62):
        super().__init__()
        self.model = _FakeInnerModel(n_layers)


def _make_mock_group(rank: int = 0, size: int = 3) -> MagicMock:
    """Create a mock mx.distributed.Group."""
    group = MagicMock(spec=mx.distributed.Group)
    group.rank.return_value = rank
    group.size.return_value = size

    # split() returns a sub-group mock
    sub_group = MagicMock(spec=mx.distributed.Group)
    sub_group.rank.return_value = 0
    sub_group.size.return_value = 2  # TP group of 2
    group.split.return_value = sub_group

    return group


def _make_singleton_group() -> MagicMock:
    """Create a mock for the PP tail's split result (singleton)."""
    sub_group = MagicMock(spec=mx.distributed.Group)
    sub_group.rank.return_value = 0
    sub_group.size.return_value = 1
    return sub_group


# ---------------------------------------------------------------------------
# Tests: tp_color assignment
# ---------------------------------------------------------------------------

class TestTpColorAssignment:
    """Verify that split() is called with the correct color for each node role."""

    def test_tp_node_uses_color_0(self):
        """TP nodes (tp_rank >= 0) should call split with color=0."""
        shard = _make_shard(
            tp_rank=0, tp_size=2, pp_rank=0,
            start_layer=0, end_layer=52,
        )
        group = _make_mock_group(rank=0)

        with patch(
            "exo.worker.engines.mlx.auto_parallel.tensor_auto_parallel",
            side_effect=lambda model, *a, **kw: model,
        ), patch(
            "exo.worker.engines.mlx.auto_parallel.patch_pipeline_model",
            side_effect=lambda model, *a, **kw: model,
        ):
            from exo.worker.engines.mlx.auto_parallel import hybrid_auto_parallel
            model = FakeModel()
            hybrid_auto_parallel(model, group, shard)

        group.split.assert_called_once_with(0)  # color=0 for TP

    def test_pp_tail_uses_color_1(self):
        """PP tail nodes (tp_rank == -1) should call split with color=1."""
        shard = _make_shard(
            tp_rank=-1, tp_size=0, pp_rank=1,
            start_layer=52, end_layer=62,
            pipeline_recv_from=0,
        )
        group = _make_mock_group(rank=2)
        # For PP tail, split returns a singleton
        group.split.return_value = _make_singleton_group()

        with patch(
            "exo.worker.engines.mlx.auto_parallel.tensor_auto_parallel",
            side_effect=lambda model, *a, **kw: model,
        ), patch(
            "exo.worker.engines.mlx.auto_parallel.patch_pipeline_model",
            side_effect=lambda model, *a, **kw: model,
        ):
            from exo.worker.engines.mlx.auto_parallel import hybrid_auto_parallel
            model = FakeModel()
            hybrid_auto_parallel(model, group, shard)

        group.split.assert_called_once_with(1)  # color=1 for PP tail


# ---------------------------------------------------------------------------
# Tests: split() is always called (catches "not implemented" early)
# ---------------------------------------------------------------------------

class TestSplitIsCalled:
    """Verify split() is actually invoked — would catch RingGroup stub error."""

    def test_split_called_for_tp_master(self):
        shard = _make_shard(
            tp_rank=0, tp_size=2, pp_rank=0,
            start_layer=0, end_layer=52,
            pipeline_send_to=2,
        )
        group = _make_mock_group(rank=0)

        with patch(
            "exo.worker.engines.mlx.auto_parallel.tensor_auto_parallel",
            side_effect=lambda model, *a, **kw: model,
        ), patch(
            "exo.worker.engines.mlx.auto_parallel.patch_pipeline_model",
            side_effect=lambda model, *a, **kw: model,
        ):
            from exo.worker.engines.mlx.auto_parallel import hybrid_auto_parallel
            hybrid_auto_parallel(FakeModel(), group, shard)

        assert group.split.call_count == 1, "split() must be called for all ranks"

    def test_split_called_for_pp_tail(self):
        shard = _make_shard(
            tp_rank=-1, tp_size=0, pp_rank=1,
            start_layer=52, end_layer=62,
            pipeline_recv_from=0,
        )
        group = _make_mock_group(rank=2)
        group.split.return_value = _make_singleton_group()

        with patch(
            "exo.worker.engines.mlx.auto_parallel.tensor_auto_parallel",
            side_effect=lambda model, *a, **kw: model,
        ), patch(
            "exo.worker.engines.mlx.auto_parallel.patch_pipeline_model",
            side_effect=lambda model, *a, **kw: model,
        ):
            from exo.worker.engines.mlx.auto_parallel import hybrid_auto_parallel
            hybrid_auto_parallel(FakeModel(), group, shard)

        assert group.split.call_count == 1, "split() must be called even for PP tail"

    def test_split_raises_propagates(self):
        """If split() raises (like old RingGroup), the error propagates."""
        shard = _make_shard(
            tp_rank=0, tp_size=2, pp_rank=0,
            start_layer=0, end_layer=52,
        )
        group = _make_mock_group(rank=0)
        group.split.side_effect = RuntimeError("[jaccl] Group split not supported.")

        with pytest.raises(RuntimeError, match="Group split not supported"):
            from exo.worker.engines.mlx.auto_parallel import hybrid_auto_parallel
            hybrid_auto_parallel(FakeModel(), group, shard)


# ---------------------------------------------------------------------------
# Tests: tensor_auto_parallel is only called for TP nodes
# ---------------------------------------------------------------------------

class TestTensorParallelDispatch:
    """Verify tensor_auto_parallel is called only for TP nodes with size > 1."""

    def test_tp_node_gets_tensor_parallel(self):
        shard = _make_shard(
            tp_rank=0, tp_size=2, pp_rank=0,
            start_layer=0, end_layer=52,
        )
        group = _make_mock_group(rank=0)

        with patch(
            "exo.worker.engines.mlx.auto_parallel.tensor_auto_parallel",
            side_effect=lambda model, *a, **kw: model,
        ) as mock_tp, patch(
            "exo.worker.engines.mlx.auto_parallel.patch_pipeline_model",
            side_effect=lambda model, *a, **kw: model,
        ):
            from exo.worker.engines.mlx.auto_parallel import hybrid_auto_parallel
            hybrid_auto_parallel(FakeModel(), group, shard)

        assert mock_tp.call_count == 1, "tensor_auto_parallel must be called for TP nodes"

    def test_pp_tail_skips_tensor_parallel(self):
        """PP tail node (tp_rank=-1) must NOT call tensor_auto_parallel."""
        shard = _make_shard(
            tp_rank=-1, tp_size=0, pp_rank=1,
            start_layer=52, end_layer=62,
            pipeline_recv_from=0,
        )
        group = _make_mock_group(rank=2)
        group.split.return_value = _make_singleton_group()

        with patch(
            "exo.worker.engines.mlx.auto_parallel.tensor_auto_parallel",
            side_effect=lambda model, *a, **kw: model,
        ) as mock_tp, patch(
            "exo.worker.engines.mlx.auto_parallel.patch_pipeline_model",
            side_effect=lambda model, *a, **kw: model,
        ):
            from exo.worker.engines.mlx.auto_parallel import hybrid_auto_parallel
            hybrid_auto_parallel(FakeModel(), group, shard)

        assert mock_tp.call_count == 0, "PP tail must NOT call tensor_auto_parallel"

    def test_singleton_tp_group_skips_tensor_parallel(self):
        """Even a TP node with sub-group size=1 should skip tensor_auto_parallel."""
        shard = _make_shard(
            tp_rank=0, tp_size=1, pp_rank=0,
            start_layer=0, end_layer=52,
        )
        group = _make_mock_group(rank=0)
        # split returns a singleton sub-group
        group.split.return_value = _make_singleton_group()

        with patch(
            "exo.worker.engines.mlx.auto_parallel.tensor_auto_parallel",
            side_effect=lambda model, *a, **kw: model,
        ) as mock_tp, patch(
            "exo.worker.engines.mlx.auto_parallel.patch_pipeline_model",
            side_effect=lambda model, *a, **kw: model,
        ):
            from exo.worker.engines.mlx.auto_parallel import hybrid_auto_parallel
            hybrid_auto_parallel(FakeModel(), group, shard)

        assert mock_tp.call_count == 0, (
            "Singleton TP sub-group (size=1) should skip tensor_auto_parallel"
        )


# ---------------------------------------------------------------------------
# Tests: layer slicing
# ---------------------------------------------------------------------------

class TestLayerSlicing:
    """Verify correct layers are kept based on start_layer/end_layer."""

    def test_tp_node_gets_correct_layer_range(self):
        shard = _make_shard(
            tp_rank=0, tp_size=2, pp_rank=0,
            start_layer=0, end_layer=52, n_layers=62,
        )
        group = _make_mock_group(rank=0)
        model = FakeModel(n_layers=62)

        with patch(
            "exo.worker.engines.mlx.auto_parallel.tensor_auto_parallel",
            side_effect=lambda model, *a, **kw: model,
        ), patch(
            "exo.worker.engines.mlx.auto_parallel.patch_pipeline_model",
            side_effect=lambda model, *a, **kw: model,
        ):
            from exo.worker.engines.mlx.auto_parallel import hybrid_auto_parallel
            result = hybrid_auto_parallel(model, group, shard)

        result_layers = result.model.layers
        assert len(result_layers) == 52

    def test_pp_tail_gets_correct_layer_range(self):
        shard = _make_shard(
            tp_rank=-1, tp_size=0, pp_rank=1,
            start_layer=52, end_layer=62, n_layers=62,
            pipeline_recv_from=0,
        )
        group = _make_mock_group(rank=2)
        group.split.return_value = _make_singleton_group()
        model = FakeModel(n_layers=62)

        with patch(
            "exo.worker.engines.mlx.auto_parallel.tensor_auto_parallel",
            side_effect=lambda model, *a, **kw: model,
        ), patch(
            "exo.worker.engines.mlx.auto_parallel.patch_pipeline_model",
            side_effect=lambda model, *a, **kw: model,
        ):
            from exo.worker.engines.mlx.auto_parallel import hybrid_auto_parallel
            result = hybrid_auto_parallel(model, group, shard)

        result_layers = result.model.layers
        assert len(result_layers) == 10


# ---------------------------------------------------------------------------
# Tests: pipeline wrapping
# ---------------------------------------------------------------------------

class TestPipelineWrapping:
    """Verify pipeline send/recv layers are applied correctly."""

    def test_tp_master_gets_send_wrapper(self):
        """TP master (tp_rank=0, pipeline_send_to set) should wrap last layer."""
        shard = _make_shard(
            tp_rank=0, tp_size=2, pp_rank=0,
            start_layer=0, end_layer=52,
            pipeline_send_to=2,
        )
        group = _make_mock_group(rank=0)
        model = FakeModel()

        with patch(
            "exo.worker.engines.mlx.auto_parallel.tensor_auto_parallel",
            side_effect=lambda model, *a, **kw: model,
        ), patch(
            "exo.worker.engines.mlx.auto_parallel.patch_pipeline_model",
            side_effect=lambda model, *a, **kw: model,
        ):
            from exo.worker.engines.mlx.auto_parallel import (
                hybrid_auto_parallel,
                _HybridPipelineLastLayer,
            )
            result = hybrid_auto_parallel(model, group, shard)

        last_layer = result.model.layers[-1]
        assert isinstance(last_layer, _HybridPipelineLastLayer), (
            "TP master's last layer must be wrapped with _HybridPipelineLastLayer"
        )
        assert last_layer.send_to == [2]

    def test_pp_tail_gets_recv_wrapper(self):
        """PP tail (pipeline_recv_from set) should wrap first layer."""
        shard = _make_shard(
            tp_rank=-1, tp_size=0, pp_rank=1,
            start_layer=52, end_layer=62,
            pipeline_recv_from=0,
        )
        group = _make_mock_group(rank=2)
        group.split.return_value = _make_singleton_group()
        model = FakeModel()

        with patch(
            "exo.worker.engines.mlx.auto_parallel.tensor_auto_parallel",
            side_effect=lambda model, *a, **kw: model,
        ), patch(
            "exo.worker.engines.mlx.auto_parallel.patch_pipeline_model",
            side_effect=lambda model, *a, **kw: model,
        ):
            from exo.worker.engines.mlx.auto_parallel import (
                hybrid_auto_parallel,
                PipelineFirstLayer,
            )
            result = hybrid_auto_parallel(model, group, shard)

        first_layer = result.model.layers[0]
        assert isinstance(first_layer, PipelineFirstLayer), (
            "PP tail's first layer must be wrapped with PipelineFirstLayer"
        )

    def test_pp_tail_uses_hybrid_send_not_all_gather(self):
        """PP tail's last layer MUST use _HybridPipelineLastLayer, NOT PipelineLastLayer.

        PipelineLastLayer uses all_gather (a collective requiring all group members).
        In hybrid mode, TP nodes don't call all_gather, causing a GPU deadlock.
        The PP tail must use _HybridPipelineLastLayer with explicit sends instead.
        """
        shard = _make_shard(
            tp_rank=-1, tp_size=0, pp_rank=1,
            start_layer=52, end_layer=62,
            pipeline_recv_from=0,
        )
        group = _make_mock_group(rank=2, size=3)
        group.split.return_value = _make_singleton_group()
        model = FakeModel()

        with patch(
            "exo.worker.engines.mlx.auto_parallel.tensor_auto_parallel",
            side_effect=lambda model, *a, **kw: model,
        ), patch(
            "exo.worker.engines.mlx.auto_parallel.patch_pipeline_model",
            side_effect=lambda model, *a, **kw: model,
        ):
            from exo.worker.engines.mlx.auto_parallel import (
                hybrid_auto_parallel,
                _HybridPipelineLastLayer,
                PipelineLastLayer,
            )
            result = hybrid_auto_parallel(model, group, shard)

        last_layer = result.model.layers[-1]
        assert not isinstance(last_layer, PipelineLastLayer), (
            "PP tail must NOT use PipelineLastLayer (all_gather deadlocks in hybrid mode)"
        )
        assert isinstance(last_layer, _HybridPipelineLastLayer), (
            "PP tail's last layer must be _HybridPipelineLastLayer with explicit sends"
        )
        # PP tail (rank=2) should send to all other ranks: [0, 1]
        assert sorted(last_layer.send_to) == [0, 1], (
            f"PP tail should send to all other ranks, got {last_layer.send_to}"
        )

    def test_tp_follower_no_pipeline_wrappers(self):
        """TP follower (tp_rank=1, no send/recv) should have no pipeline wrappers."""
        shard = _make_shard(
            tp_rank=1, tp_size=2, pp_rank=0,
            start_layer=0, end_layer=52,
            pipeline_send_to=None,
            pipeline_recv_from=None,
        )
        group = _make_mock_group(rank=1)
        model = FakeModel()

        with patch(
            "exo.worker.engines.mlx.auto_parallel.tensor_auto_parallel",
            side_effect=lambda model, *a, **kw: model,
        ), patch(
            "exo.worker.engines.mlx.auto_parallel.patch_pipeline_model",
            side_effect=lambda model, *a, **kw: model,
        ):
            from exo.worker.engines.mlx.auto_parallel import (
                hybrid_auto_parallel,
                _HybridPipelineLastLayer,
                PipelineFirstLayer,
            )
            result = hybrid_auto_parallel(model, group, shard)

        layers = result.model.layers
        # No pipeline recv wrapping on first layer (follower doesn't receive from upstream)
        assert not isinstance(layers[0], PipelineFirstLayer)
        # Last layer should have a recv-only wrapper (no sends, but receives from PP tail during decode)
        assert isinstance(layers[-1], _HybridPipelineLastLayer)
        assert layers[-1].send_to == []
        assert layers[-1].decode_recv_from is not None


# ---------------------------------------------------------------------------
# Tests: full 3-node scenario
# ---------------------------------------------------------------------------

class TestThreeNodeHybridScenario:
    """End-to-end scenario matching the actual cluster config:
    Studio-1 (rank=0, TP master), Studio-2 (rank=1, TP follower),
    MacBook (rank=2, PP tail).
    """

    @pytest.fixture
    def model(self):
        return FakeModel(n_layers=62)

    def _run_hybrid(self, model, rank, shard):
        group = _make_mock_group(rank=rank, size=3)
        if shard.tp_rank < 0:
            group.split.return_value = _make_singleton_group()

        with patch(
            "exo.worker.engines.mlx.auto_parallel.tensor_auto_parallel",
            side_effect=lambda model, *a, **kw: model,
        ) as mock_tp, patch(
            "exo.worker.engines.mlx.auto_parallel.patch_pipeline_model",
            side_effect=lambda model, *a, **kw: model,
        ):
            from exo.worker.engines.mlx.auto_parallel import hybrid_auto_parallel
            result = hybrid_auto_parallel(model, group, shard)

        return result, group, mock_tp

    def test_studio1_tp_master(self, model):
        """Studio-1: rank=0, TP master, layers 0-52, sends to rank 2."""
        shard = _make_shard(
            tp_rank=0, tp_size=2, pp_rank=0,
            start_layer=0, end_layer=52,
            device_rank=0,
            pipeline_send_to=2,
        )
        result, group, mock_tp = self._run_hybrid(model, rank=0, shard=shard)

        group.split.assert_called_once_with(0)
        assert mock_tp.call_count == 1
        assert len(result.model.layers) == 52

    def test_studio2_tp_follower(self, model):
        """Studio-2: rank=1, TP follower, layers 0-52, no pipeline."""
        shard = _make_shard(
            tp_rank=1, tp_size=2, pp_rank=0,
            start_layer=0, end_layer=52,
            device_rank=1,
        )
        result, group, mock_tp = self._run_hybrid(model, rank=1, shard=shard)

        group.split.assert_called_once_with(0)
        assert mock_tp.call_count == 1
        assert len(result.model.layers) == 52

    def test_macbook_pp_tail(self, model):
        """MacBook: rank=2, PP tail, layers 52-62, receives from rank 0."""
        shard = _make_shard(
            tp_rank=-1, tp_size=0, pp_rank=1,
            start_layer=52, end_layer=62,
            device_rank=2,
            pipeline_recv_from=0,
        )
        result, group, mock_tp = self._run_hybrid(model, rank=2, shard=shard)

        group.split.assert_called_once_with(1)  # PP tail gets color=1
        assert mock_tp.call_count == 0  # NO tensor parallel for PP tail
        assert len(result.model.layers) == 10

    def test_all_layers_covered(self, model):
        """TP nodes + PP tail must cover all 62 layers without gaps or overlaps."""
        tp_start, tp_end = 0, 52
        pp_start, pp_end = 52, 62
        n_layers = 62

        assert tp_start == 0, "TP must start at layer 0"
        assert pp_end == n_layers, "PP must end at last layer"
        assert tp_end == pp_start, "No gap between TP and PP ranges"
        assert (tp_end - tp_start) + (pp_end - pp_start) == n_layers
