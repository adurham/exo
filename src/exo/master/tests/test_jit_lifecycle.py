"""Tests for the JIT model lifecycle admission/config primitives.

Covers the pure, unit-testable surface of the JIT feature:
  - ``jit_memory_reserve`` env parsing (default / override / malformed)
  - ``jit_enabled`` / ``jit_load_timeout_seconds`` / ``jit_idle_unload_seconds``
  - ``cycle_admits_with_reserve`` per-node semantics (lopsided, boundary, disable)
  - ``weight_share_per_node`` for tensor + pipeline sharding
  - ``place_instance(jit=True)`` hard-refusal when the reserve can't be met,
    and that ``jit=False`` placement is unaffected by the reserve.
"""

from collections.abc import Mapping

import pytest

from exo.master.placement import place_instance
from exo.master.placement_utils import (
    cycle_admits_with_reserve,
    jit_enabled,
    jit_idle_unload_seconds,
    jit_instances_to_reap,
    jit_load_timeout_seconds,
    jit_memory_reserve,
    weight_share_per_node,
)
from exo.master.tests.conftest import (
    create_node_memory,
    create_node_network,
    create_socket_connection,
)
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.topology import Topology
from exo.shared.types.backends import Backend
from exo.shared.types.commands import CommandId, PlaceInstance
from exo.shared.types.common import NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import MemoryUsage, NodeNetworkInfo
from exo.shared.types.topology import Connection, Cycle
from exo.shared.types.worker.instances import (
    Instance,
    InstanceId,
    InstanceMeta,
    MlxRingInstance,
)
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.shared.types.worker.shards import PipelineShardMetadata, Sharding


# --------------------------------------------------------------------------- #
# jit_memory_reserve
# --------------------------------------------------------------------------- #
def test_jit_memory_reserve_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EXO_JIT_MEMORY_RESERVE_GB", raising=False)
    assert jit_memory_reserve() == Memory.from_gb(18.0)


def test_jit_memory_reserve_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXO_JIT_MEMORY_RESERVE_GB", "24.5")
    assert jit_memory_reserve() == Memory.from_gb(24.5)


def test_jit_memory_reserve_zero_disables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXO_JIT_MEMORY_RESERVE_GB", "0")
    assert jit_memory_reserve() == Memory()


def test_jit_memory_reserve_malformed_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXO_JIT_MEMORY_RESERVE_GB", "not-a-number")
    assert jit_memory_reserve() == Memory.from_gb(18.0)


# --------------------------------------------------------------------------- #
# jit_enabled / timeouts
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "value,expected",
    [
        ("1", True),
        ("true", True),
        ("TRUE", True),
        ("yes", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("", False),
        ("nonsense", False),
    ],
)
def test_jit_enabled(
    monkeypatch: pytest.MonkeyPatch, value: str, expected: bool
) -> None:
    monkeypatch.setenv("EXO_JIT_ENABLED", value)
    assert jit_enabled() is expected


def test_jit_enabled_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EXO_JIT_ENABLED", raising=False)
    assert jit_enabled() is False


def test_jit_load_timeout_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EXO_JIT_LOAD_TIMEOUT_SECONDS", raising=False)
    assert jit_load_timeout_seconds() == 120.0


def test_jit_load_timeout_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXO_JIT_LOAD_TIMEOUT_SECONDS", "45")
    assert jit_load_timeout_seconds() == 45.0


def test_jit_load_timeout_malformed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXO_JIT_LOAD_TIMEOUT_SECONDS", "soon")
    assert jit_load_timeout_seconds() == 120.0


def test_jit_load_timeout_negative_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXO_JIT_LOAD_TIMEOUT_SECONDS", "-5")
    assert jit_load_timeout_seconds() == 120.0


def test_jit_idle_unload_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EXO_JIT_IDLE_UNLOAD_SECONDS", raising=False)
    assert jit_idle_unload_seconds() == 300.0


def test_jit_idle_unload_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXO_JIT_IDLE_UNLOAD_SECONDS", "600")
    assert jit_idle_unload_seconds() == 600.0


# --------------------------------------------------------------------------- #
# cycle_admits_with_reserve
# --------------------------------------------------------------------------- #
def _mem(gb: float) -> MemoryUsage:
    return MemoryUsage.from_bytes(
        ram_total=128 * 1024**3,
        ram_available=int(gb * 1024**3),
        swap_total=0,
        swap_available=0,
    )


def test_cycle_admits_with_reserve_both_nodes_pass() -> None:
    a, b = NodeId(), NodeId()
    cycle = Cycle(node_ids=[a, b])
    node_memory: Mapping[NodeId, MemoryUsage] = {a: _mem(40), b: _mem(40)}
    weights = {a: Memory.from_gb(10), b: Memory.from_gb(10)}
    # 40 - 10 = 30 free >= 18 reserve on both nodes.
    assert cycle_admits_with_reserve(
        cycle, node_memory, weights, Memory.from_gb(18)
    )


def test_cycle_admits_with_reserve_lopsided_one_node_fails() -> None:
    a, b = NodeId(), NodeId()
    cycle = Cycle(node_ids=[a, b])
    # Node b is tight: 25 - 10 = 15 free < 18 reserve, even though node a is fine.
    node_memory: Mapping[NodeId, MemoryUsage] = {a: _mem(40), b: _mem(25)}
    weights = {a: Memory.from_gb(10), b: Memory.from_gb(10)}
    assert not cycle_admits_with_reserve(
        cycle, node_memory, weights, Memory.from_gb(18)
    )


def test_cycle_admits_with_reserve_exact_boundary_passes() -> None:
    a = NodeId()
    cycle = Cycle(node_ids=[a])
    # 28 - 10 = 18 free == 18 reserve → the >= check admits exactly at the edge.
    node_memory: Mapping[NodeId, MemoryUsage] = {a: _mem(28)}
    weights = {a: Memory.from_gb(10)}
    assert cycle_admits_with_reserve(cycle, node_memory, weights, Memory.from_gb(18))


def test_cycle_admits_with_reserve_just_below_boundary_fails() -> None:
    a = NodeId()
    cycle = Cycle(node_ids=[a])
    # 27.9 - 10 = 17.9 free < 18 reserve.
    node_memory: Mapping[NodeId, MemoryUsage] = {a: _mem(27.9)}
    weights = {a: Memory.from_gb(10)}
    assert not cycle_admits_with_reserve(
        cycle, node_memory, weights, Memory.from_gb(18)
    )


def test_cycle_admits_with_reserve_zero_disables() -> None:
    a = NodeId()
    cycle = Cycle(node_ids=[a])
    # With reserve=0 the check degrades to weights-fit: 10 - 10 = 0 >= 0.
    node_memory: Mapping[NodeId, MemoryUsage] = {a: _mem(10)}
    weights = {a: Memory.from_gb(10)}
    assert cycle_admits_with_reserve(cycle, node_memory, weights, Memory())


def test_cycle_admits_with_reserve_missing_node_memory_fails() -> None:
    a, b = NodeId(), NodeId()
    cycle = Cycle(node_ids=[a, b])
    node_memory: Mapping[NodeId, MemoryUsage] = {a: _mem(40)}  # b missing
    weights = {a: Memory.from_gb(10), b: Memory.from_gb(10)}
    assert not cycle_admits_with_reserve(
        cycle, node_memory, weights, Memory.from_gb(1)
    )


# --------------------------------------------------------------------------- #
# weight_share_per_node
# --------------------------------------------------------------------------- #
def _card(storage_gb: float, n_layers: int = 10) -> ModelCard:
    return ModelCard(
        model_id=ModelId("test-model"),
        storage_size=Memory.from_gb(storage_gb),
        n_layers=n_layers,
        hidden_size=64,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal],
    )


def test_weight_share_single_node_holds_whole_model() -> None:
    a = NodeId()
    cycle = Cycle(node_ids=[a])
    shares = weight_share_per_node(_card(20), cycle, {a: _mem(40)}, Sharding.Pipeline)
    assert shares[a] == Memory.from_gb(20)


def test_weight_share_tensor_splits_evenly() -> None:
    a, b = NodeId(), NodeId()
    cycle = Cycle(node_ids=[a, b])
    node_memory = {a: _mem(40), b: _mem(40)}
    shares = weight_share_per_node(_card(20), cycle, node_memory, Sharding.Tensor)
    assert shares[a] == Memory.from_gb(20) / 2
    assert shares[b] == Memory.from_gb(20) / 2


def test_weight_share_pipeline_proportional_to_memory() -> None:
    a, b = NodeId(), NodeId()
    cycle = Cycle(node_ids=[a, b])
    # Equal memory → 10 layers split 5/5 → half the storage each.
    node_memory = {a: _mem(40), b: _mem(40)}
    shares = weight_share_per_node(
        _card(20, n_layers=10), cycle, node_memory, Sharding.Pipeline
    )
    # storage * 5 // 10 = storage/2 each (integer-byte floor).
    expected = (Memory.from_gb(20) * 5) // 10
    assert shares[a] == expected
    assert shares[b] == expected


# --------------------------------------------------------------------------- #
# place_instance(jit=True) hard-refusal
# --------------------------------------------------------------------------- #
def _metal_only(node_memory: Mapping[NodeId, object]) -> dict[NodeId, list[Backend]]:
    return {node_id: [Backend.MlxMetal] for node_id in node_memory}


def _two_node_topology(
    mem_a: int, mem_b: int
) -> tuple[
    Topology,
    dict[NodeId, MemoryUsage],
    dict[NodeId, NodeNetworkInfo],
    NodeId,
    NodeId,
]:
    topology = Topology()
    a, b = NodeId(), NodeId()
    topology.add_node(a)
    topology.add_node(b)
    topology.add_connection(
        Connection(source=a, sink=b, edge=create_socket_connection(1))
    )
    topology.add_connection(
        Connection(source=b, sink=a, edge=create_socket_connection(2))
    )
    node_memory = {a: create_node_memory(mem_a), b: create_node_memory(mem_b)}
    node_network = {a: create_node_network(), b: create_node_network()}
    return topology, node_memory, node_network, a, b


def _place_cmd(card: ModelCard, *, jit: bool) -> PlaceInstance:
    return PlaceInstance(
        command_id=CommandId(),
        model_card=card,
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        min_nodes=1,
        jit=jit,
    )


def test_place_instance_jit_refuses_when_reserve_unmet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXO_JIT_MEMORY_RESERVE_GB", "18")
    # Weights fit (model 20GB across 2x40GB nodes) but leave only ~30GB/node —
    # wait, that DOES clear 18. Make it tight: each node has 25GB, model 20GB
    # single-node won't fit; 2-node pipeline puts ~10GB/node leaving 15 < 18.
    card = _card(20, n_layers=10)
    topology, node_memory, node_network, _a, _b = _two_node_topology(
        25 * 1024**3, 25 * 1024**3
    )
    with pytest.raises(ValueError, match="JIT memory reserve"):
        place_instance(
            _place_cmd(card, jit=True),
            topology,
            {},
            node_memory,
            node_network,
            _metal_only(node_memory),
        )


def test_place_instance_non_jit_ignores_reserve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Same tight layout, but an explicit (non-JIT) placement must still succeed:
    # the reserve only gates JIT auto-placements.
    monkeypatch.setenv("EXO_JIT_MEMORY_RESERVE_GB", "18")
    card = _card(20, n_layers=10)
    topology, node_memory, node_network, _a, _b = _two_node_topology(
        25 * 1024**3, 25 * 1024**3
    )
    placements = place_instance(
        _place_cmd(card, jit=False),
        topology,
        {},
        node_memory,
        node_network,
        _metal_only(node_memory),
    )
    assert len(placements) == 1
    instance = next(iter(placements.values()))
    assert instance.jit is False


def test_place_instance_jit_admits_with_headroom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Plenty of headroom: 2x80GB nodes, 20GB model → ~10GB/node weights leaving
    # ~70GB free >> 18 reserve. JIT placement succeeds and is tagged jit=True.
    monkeypatch.setenv("EXO_JIT_MEMORY_RESERVE_GB", "18")
    card = _card(20, n_layers=10)
    topology, node_memory, node_network, _a, _b = _two_node_topology(
        80 * 1024**3, 80 * 1024**3
    )
    placements = place_instance(
        _place_cmd(card, jit=True),
        topology,
        {},
        node_memory,
        node_network,
        _metal_only(node_memory),
    )
    assert len(placements) == 1
    instance = next(iter(placements.values()))
    assert instance.jit is True


# --------------------------------------------------------------------------- #
# jit_instances_to_reap (idle reaper policy)
# --------------------------------------------------------------------------- #
def _instance(*, jit: bool, model: str = "m") -> Instance:
    node_id = NodeId()
    runner_id = RunnerId()
    shard = PipelineShardMetadata(
        model_card=_card(1),
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
        jit=jit,
    )


def test_reap_skips_non_jit_pinned_instance() -> None:
    inst = _instance(jit=False)
    instances = {inst.instance_id: inst}
    # Way past the idle window, but jit=False → never reaped.
    reap = jit_instances_to_reap(
        instances, {}, {inst.instance_id: 0.0}, now=10_000.0, idle_window=300.0
    )
    assert reap == []


def test_reap_skips_inflight_jit_instance() -> None:
    inst = _instance(jit=True)
    instances = {inst.instance_id: inst}
    # Idle long enough, but a request is in flight → skip (closes the race).
    reap = jit_instances_to_reap(
        instances,
        {inst.instance_id: 1},
        {inst.instance_id: 0.0},
        now=10_000.0,
        idle_window=300.0,
    )
    assert reap == []


def test_reap_unloads_idle_jit_instance() -> None:
    inst = _instance(jit=True)
    instances = {inst.instance_id: inst}
    reap = jit_instances_to_reap(
        instances,
        {},
        {inst.instance_id: 0.0},
        now=400.0,
        idle_window=300.0,
    )
    assert reap == [inst.instance_id]


def test_reap_keeps_recently_used_jit_instance() -> None:
    inst = _instance(jit=True)
    instances = {inst.instance_id: inst}
    # Used 100s ago, window is 300s → not yet eligible.
    reap = jit_instances_to_reap(
        instances,
        {},
        {inst.instance_id: 300.0},
        now=400.0,
        idle_window=300.0,
    )
    assert reap == []


def test_reap_skips_instance_without_last_use_record() -> None:
    inst = _instance(jit=True)
    instances = {inst.instance_id: inst}
    # No last-use record yet (placed but never observed) → not reaped this pass.
    reap = jit_instances_to_reap(
        instances, {}, {}, now=10_000.0, idle_window=300.0
    )
    assert reap == []


def test_reap_exact_window_boundary_unloads() -> None:
    inst = _instance(jit=True)
    instances = {inst.instance_id: inst}
    # now - last_use == idle_window exactly → the >= comparison reaps it.
    reap = jit_instances_to_reap(
        instances, {}, {inst.instance_id: 100.0}, now=400.0, idle_window=300.0
    )
    assert reap == [inst.instance_id]


def test_reap_mixed_only_eligible_jit() -> None:
    pinned = _instance(jit=False, model="interactive")
    idle_jit = _instance(jit=True, model="aux-idle")
    busy_jit = _instance(jit=True, model="aux-busy")
    instances = {
        pinned.instance_id: pinned,
        idle_jit.instance_id: idle_jit,
        busy_jit.instance_id: busy_jit,
    }
    last_use = {
        pinned.instance_id: 0.0,
        idle_jit.instance_id: 0.0,
        busy_jit.instance_id: 0.0,
    }
    inflight = {busy_jit.instance_id: 2}
    reap = jit_instances_to_reap(
        instances, inflight, last_use, now=10_000.0, idle_window=300.0
    )
    assert reap == [idle_jit.instance_id]

