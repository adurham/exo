"""DSv4.1 engine dispatch: which card gets which builder, and the model card.

The dispatch contract under test (see ``dsv41.dispatch``):

* a card that declares ``engine = "dsv41"`` is served by ``Dsv41Builder``;
* every other card is served by the generic ``MlxBuilder`` (unchanged);
* the shipped model card for this checkpoint parses, carries the real geometry,
  and asks for this engine.

Nothing here loads a checkpoint: the builders are compared by type, and
``bootstrap``'s selection is exercised through the same helper it calls.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import tomlkit
from pydantic import ValidationError

from exo.shared.models.model_cards import (
    Backend,
    ModelCard,
    ModelEngine,
    ModelId,
    ModelTask,
)
from exo.shared.types.common import NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.worker.instances import (
    BoundInstance,
    InstanceId,
    MlxRingInstance,
)
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.shared.types.worker.shards import (
    PipelineShardMetadata,
    TensorShardMetadata,
)
from exo.worker.engines.mlx.builder import MlxBuilder
from exo.worker.engines.mlx.dsv41.builder import Dsv41Builder
from exo.worker.engines.mlx.dsv41.dispatch import (
    DSV41_ENGINE,
    is_dsv41_card,
    is_dsv41_instance,
    make_mlx_family_builder,
)


def _card_file() -> Path:
    """Locate the shipped card without depending on a fixed parents[] depth.

    The worktree keeps `resources/` in a different place than the installed
    layout, so walk up from this file until the card is found.
    """
    name = "dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw.toml"
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "resources" / "inference_model_cards" / name
        if candidate.is_file():
            return candidate
    raise AssertionError(f"card not found above {Path(__file__).resolve()}")


CARD_FILE = _card_file()
DSV41_MODEL_ID = ModelId("dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw")


def _card(**overrides) -> ModelCard:
    base: dict[str, Any] = {
        "model_id": ModelId("some/model"),
        "storage_size": Memory.from_mb(1000),
        "n_layers": 32,
        "hidden_size": 2048,
        "supports_tensor": True,
        "tasks": [ModelTask.TextGeneration],
        "backends": [Backend.MlxMetal],
    }
    base.update(overrides)
    return ModelCard(**base)


def _bound(card: ModelCard, *, tensor: bool = True) -> BoundInstance:
    shard_cls = TensorShardMetadata if tensor else PipelineShardMetadata
    shard = shard_cls(
        model_card=card,
        device_rank=0,
        world_size=2,
        start_layer=0,
        end_layer=card.n_layers,
        n_layers=card.n_layers,
    )
    instance = MlxRingInstance(
        instance_id=InstanceId("instance"),
        shard_assignments=ShardAssignments(
            model_id=card.model_id,
            node_to_runner={NodeId("node"): RunnerId("runner")},
            runner_to_shard={RunnerId("runner"): shard},
        ),
        hosts_by_node={},
        ephemeral_port=50000,
    )
    return BoundInstance(
        instance=instance,
        bound_runner_id=RunnerId("runner"),
        bound_node_id=NodeId("node"),
    )


# --------------------------------------------------------------- the predicate


def test_engine_field_defaults_to_the_generic_mlx_path():
    assert _card().engine is None
    assert is_dsv41_card(_card()) is False


def test_engine_field_accepts_the_toml_string_and_the_enum():
    assert _card(engine="dsv41").engine is ModelEngine.Dsv41
    assert _card(engine=ModelEngine.Dsv41).engine is ModelEngine.Dsv41
    with pytest.raises(ValidationError):
        _card(engine="not-an-engine")


def test_is_dsv41_instance_follows_the_card():
    assert is_dsv41_instance(_bound(_card(engine=DSV41_ENGINE))) is True
    assert is_dsv41_instance(_bound(_card())) is False


def test_a_deepseek_exl3_card_without_the_field_is_not_dsv41():
    """The predicate must not be inferred from quantization/family.

    A future DeepSeek EXL3 card (a different weight layout) must NOT be routed
    into this engine by accident -- only an explicit ``engine`` field does it.
    """
    card = _card(family="deepseek", quantization="exl3")
    assert is_dsv41_card(card) is False


# --------------------------------------------------------------- the builder


def test_dispatch_returns_the_dsv41_builder_for_a_dsv41_card():
    builder = make_mlx_family_builder(
        _bound(_card(engine=DSV41_ENGINE, model_id=DSV41_MODEL_ID)),
        event_sender=object(),
        cancel_receiver=object(),
    )
    assert isinstance(builder, Dsv41Builder)
    assert isinstance(builder, MlxBuilder)  # shares connect()/close()
    assert builder.model_id == DSV41_MODEL_ID


def test_dispatch_returns_the_generic_builder_for_everything_else():
    builder = make_mlx_family_builder(
        _bound(_card()), event_sender=object(), cancel_receiver=object()
    )
    assert isinstance(builder, MlxBuilder)
    assert not isinstance(builder, Dsv41Builder)


def test_dsv41_builder_builds_the_engine_for_its_loaded_checkpoint(monkeypatch):
    """The builder's build() must produce the DSv4.1 engine, using the loader."""
    from exo.worker.engines.mlx.dsv41 import builder as builder_module
    from exo.worker.engines.mlx.dsv41.engine import Dsv41Engine
    from exo.worker.engines.mlx.dsv41.load import Dsv41Loaded
    from exo.worker.engines.mlx.dsv41.tests.conftest import FakeTokenizer, model_id

    loaded = Dsv41Loaded(
        model=type("FakeModel", (), {"layers": list(range(40))})(),
        tokenizer=FakeTokenizer(),
        args=type("A", (), {"max_seq_len": 4096, "engram_layer_ids": ()})(),
        model_path=Path("/nonexistent"),
        built_layers=list(range(40)),
        full_stack=True,
        rank=0,
        world=1,
        load_seconds=0.0,
    )

    def fake_load_dsv41(shard, card, *, group=None, model_path=None):
        del shard, card, group, model_path
        yield None  # a ModelLoadingResponse-shaped progress item
        return loaded

    monkeypatch.setattr(builder_module, "load_dsv41", fake_load_dsv41)
    monkeypatch.setattr(
        builder_module, "build_draft_head", lambda _loaded, _group: None
    )
    monkeypatch.setattr(builder_module, "speculation_enabled", lambda: False)
    warmed: list[object] = []
    monkeypatch.setattr(builder_module, "load_warmup", warmed.append)

    builder = Dsv41Builder(
        model_id=model_id(),  # type: ignore[arg-type]
        event_sender=object(),  # type: ignore[arg-type]
        cancel_receiver=object(),  # type: ignore[arg-type]
    )
    bound = _bound(_card(engine=DSV41_ENGINE, model_id=DSV41_MODEL_ID))
    assert list(builder.load(bound)) == [None]  # drives the loader generator
    assert builder.loaded_dsv41 is loaded
    # the compile-storm warmup runs at load, before any request (p114/p115)
    assert warmed == [loaded]

    engine = builder.build()
    assert isinstance(engine, Dsv41Engine)
    assert engine.model_id == DSV41_MODEL_ID
    assert engine.speculative is False


# --------------------------------------------------------------- the card


def test_shipped_card_parses_and_declares_this_engine():
    card = ModelCard.model_validate(tomlkit.loads(CARD_FILE.read_text()))
    assert card.model_id == DSV41_MODEL_ID
    assert card.engine is ModelEngine.Dsv41
    assert is_dsv41_card(card) is True


def test_card_file_is_found_next_to_the_repo_root():
    """The card path is derived from the test file, so a rename is caught here."""
    assert CARD_FILE.is_file(), CARD_FILE
    assert CARD_FILE.parent.name == "inference_model_cards"


def test_shipped_card_geometry_matches_the_checkpoint():
    card = ModelCard.model_validate(tomlkit.loads(CARD_FILE.read_text()))
    # From the checkpoint's config.json -> text_config (not the nested
    # num_hidden_layers-free top level): 40 served layers, hidden 5120,
    # ONE shared 512-d KV head per position (MLA), vocab 129280, 1M context.
    assert card.n_layers == 40
    assert card.hidden_size == 5120
    assert card.num_key_value_heads == 1
    assert card.context_length == 1048576
    assert card.supports_tensor is True
    assert card.backends == [Backend.MlxMetal]


def test_shipped_card_storage_size_is_the_checkpoints_index_total():
    card = ModelCard.model_validate(tomlkit.loads(CARD_FILE.read_text()))
    # metadata.total_size from the checkpoint's model.safetensors.index.json on
    # macstudio-m4-1 (39 shards). Pinned so a card edit cannot silently change
    # the number placement uses to decide whether the model fits.
    assert card.storage_size.in_bytes == 210713013432


def test_shipped_card_is_greedy_by_default():
    """The engine is greedy-only, so the card must not invite sampling."""
    card = ModelCard.model_validate(tomlkit.loads(CARD_FILE.read_text()))
    assert card.sampling_defaults.temperature == 0.0
    assert card.sampling_defaults.min_p == 0.0


def test_shipped_card_does_not_advertise_vision_yet():
    """The checkpoint HAS a vision tower; the exo image path is not wired."""
    card = ModelCard.model_validate(tomlkit.loads(CARD_FILE.read_text()))
    assert "vision" not in card.capabilities
    assert card.capabilities == ["text", "thinking", "thinking_toggle"]
    assert card.reasoning_dialect == "tool_conditional"
