# pyright: reportPrivateUsage=false
"""Tests for the JIT request-path hook in the API server.

Covers the behavior of ``_validate_model_has_instance`` and the single-flight
``_jit_ensure_instance`` coalescing without standing up a full cluster:
  - JIT disabled → unchanged 404 behavior.
  - JIT enabled, model already resident → returns immediately.
  - JIT enabled, model not downloaded → 404 (no auto-place).
  - Single-flight: concurrent first-requests trigger exactly ONE placement.
"""

import anyio
import pytest
from fastapi import HTTPException

from exo.api.main import API
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.common import NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.state import State
from exo.shared.types.worker.downloads import (
    DownloadCompleted,
)
from exo.shared.types.worker.instances import (
    InstanceId,
    MlxRingInstance,
)
from exo.shared.types.worker.runners import RunnerId, RunnerReady, ShardAssignments
from exo.shared.types.worker.shards import PipelineShardMetadata


def _make_api(state: State) -> API:
    api = object.__new__(API)
    api.state = state
    api._jit_inflight_loads = {}
    return api


def _card() -> ModelCard:
    return ModelCard(
        model_id=ModelId("aux-model"),
        storage_size=Memory.from_gb(1),
        n_layers=10,
        hidden_size=64,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal],
    )


def _resident_instance(model: str, runner_id: RunnerId) -> MlxRingInstance:
    node_id = NodeId()
    shard = PipelineShardMetadata(
        model_card=_card(),
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=10,
        n_layers=10,
    )
    return MlxRingInstance(
        instance_id=InstanceId(),
        shard_assignments=ShardAssignments(
            model_id=ModelId(model),
            runner_to_shard={runner_id: shard},
            node_to_runner={node_id: runner_id},
        ),
        hosts_by_node={},
        ephemeral_port=50000,
        jit=True,
    )


def _downloaded(model: str) -> DownloadCompleted:
    shard = PipelineShardMetadata(
        model_card=_card(),
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=10,
        n_layers=10,
    )
    return DownloadCompleted(
        node_id=NodeId(),
        shard_metadata=shard,
        total=Memory.from_gb(1),
    )


async def test_validate_returns_when_instance_present() -> None:
    runner_id = RunnerId()
    inst = _resident_instance("aux-model", runner_id)
    state = State(
        instances={inst.instance_id: inst},
        runners={runner_id: RunnerReady()},
    )
    api = _make_api(state)
    result = await api._validate_model_has_instance(ModelId("aux-model"))
    assert result == ModelId("aux-model")


async def test_validate_404_when_jit_disabled_and_no_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EXO_JIT_ENABLED", raising=False)
    state = State()
    api = _make_api(state)
    with pytest.raises(HTTPException) as exc:
        await api._validate_model_has_instance(ModelId("missing"))
    assert exc.value.status_code == 404


async def test_validate_404_when_jit_enabled_but_not_downloaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXO_JIT_ENABLED", "1")
    state = State()
    api = _make_api(state)

    notified: list[ModelId] = []

    async def _notify(model_id: ModelId) -> None:
        notified.append(model_id)

    api._trigger_notify_user_to_download_model = _notify

    with pytest.raises(HTTPException) as exc:
        await api._validate_model_has_instance(ModelId("missing"))
    assert exc.value.status_code == 404
    assert notified == [ModelId("missing")]


async def test_single_flight_one_placement_for_concurrent_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent first-requests for the same model trigger ONE placement."""
    monkeypatch.setenv("EXO_JIT_ENABLED", "1")
    model = ModelId("aux-model")
    state = State(
        downloads={NodeId(): [_downloaded("aux-model")]},
    )
    api = _make_api(state)

    place_calls = 0
    ready_after_place = anyio.Event()

    async def _fake_place_and_wait(model_id: ModelId) -> None:
        nonlocal place_calls
        place_calls += 1
        # Simulate the load taking a moment, then becoming ready.
        await ready_after_place.wait()
        runner_id = RunnerId()
        inst = _resident_instance(str(model_id), runner_id)
        api.state = State(
            instances={inst.instance_id: inst},
            runners={runner_id: RunnerReady()},
            downloads=state.downloads,
        )

    api._jit_place_and_wait = _fake_place_and_wait

    results: list[ModelId | None] = [None, None, None]

    async with anyio.create_task_group() as tg:

        async def _req(idx: int) -> None:
            results[idx] = await api._validate_model_has_instance(model)

        tg.start_soon(_req, 0)
        tg.start_soon(_req, 1)
        tg.start_soon(_req, 2)
        # Let all three coalesce on the single-flight event before completing.
        await anyio.sleep(0.05)
        ready_after_place.set()

    assert place_calls == 1
    assert all(r == model for r in results)
    # Single-flight map is cleared after the load completes.
    assert api._jit_inflight_loads == {}
