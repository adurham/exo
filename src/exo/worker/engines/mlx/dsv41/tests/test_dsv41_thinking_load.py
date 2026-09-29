"""Thinking-marker resolution and the loader's shard geometry.

The marker part pins the CONTRACT with mlx-lm's ``TokenizerWrapper``: the
checkpoint's own tokenizer already resolves `` thinking``/``</think>`` (verified
on macstudio-m4-1), so ``ensure_thinking_markers`` must be a no-op there; if a
tokenizer does NOT resolve them, it must install them (or warn loudly when the
vocab has no such tokens). The attribute guard raising is deliberate: mlx-lm
renaming those private attributes must fail at load, not silently leave
reasoning unparsed.

The geometry part covers ``tp_geometry`` (which shard placements DSv4.1 can
serve) and ``_requested_layers`` (the layer-subset harness switch).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from exo.shared.models.model_cards import Backend, ModelCard, ModelId, ModelTask
from exo.shared.types.memory import Memory
from exo.shared.types.worker.shards import (
    PipelineShardMetadata,
    TensorShardMetadata,
)
from exo.worker.engines.mlx.dsv41 import thinking
from exo.worker.engines.mlx.dsv41.errors import Dsv41UnsupportedPlacement
from exo.worker.engines.mlx.dsv41.load import _requested_layers, tp_geometry
from exo.worker.engines.mlx.dsv41.tests.conftest import (
    DSML_SENTINEL,
    THINK_END,
    THINK_START,
    FakeTokenizer,
)

VOCAB = {DSML_SENTINEL: 128825, THINK_START: 128821, THINK_END: 128822}


def _card() -> ModelCard:
    return ModelCard(
        model_id=ModelId("dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"),
        storage_size=Memory.from_mb(1000),
        n_layers=40,
        hidden_size=5120,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal],
    )


def _shard(cls, *, device_rank: int, world_size: int):
    return cls(
        model_card=_card(),
        device_rank=device_rank,
        world_size=world_size,
        start_layer=0,
        end_layer=40,
        n_layers=40,
    )


# --------------------------------------------------------------- geometry


def test_tensor_shard_world_1_and_2_are_supported():
    assert tp_geometry(_shard(TensorShardMetadata, device_rank=0, world_size=1)) == (
        0,
        1,
    )
    assert tp_geometry(_shard(TensorShardMetadata, device_rank=1, world_size=2)) == (
        1,
        2,
    )


def test_tensor_shard_world_3_is_refused():
    with pytest.raises(Dsv41UnsupportedPlacement, match="world_size 1 or 2"):
        tp_geometry(_shard(TensorShardMetadata, device_rank=0, world_size=3))


def test_single_rank_pipeline_shard_is_the_whole_model_on_one_node():
    assert tp_geometry(_shard(PipelineShardMetadata, device_rank=0, world_size=1)) == (
        0,
        1,
    )


def test_multi_rank_pipeline_shard_is_refused():
    with pytest.raises(Dsv41UnsupportedPlacement, match="Pipeline"):
        tp_geometry(_shard(PipelineShardMetadata, device_rank=0, world_size=2))


# --------------------------------------------------------------- layer subsets


def test_requested_layers_empty_means_full_stack(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("EXO_DSV41_LAYERS", raising=False)
    assert _requested_layers(40) is None


def test_requested_layers_parses_and_sorts(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EXO_DSV41_LAYERS", "0,1,2,3,20,21,24,25")
    assert _requested_layers(40) == [0, 1, 2, 3, 20, 21, 24, 25]
    monkeypatch.setenv("EXO_DSV41_LAYERS", "3,1,3")
    assert _requested_layers(40) == [1, 3]


def test_requested_layers_rejects_out_of_range(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EXO_DSV41_LAYERS", "0,40")
    with pytest.raises(ValueError, match="outside 0..39"):
        _requested_layers(40)


# --------------------------------------------------------------- thinking markers


def test_ensure_markers_is_a_noop_when_the_tokenizer_already_resolves_them(
    monkeypatch: pytest.MonkeyPatch,
):
    """The real checkpoint tokenizer reports the markers; do not touch it."""
    monkeypatch.delenv("EXO_DSV41_THINK_MARKERS", raising=False)
    tokenizer = FakeTokenizer(vocab=VOCAB)
    assert tokenizer.has_thinking is True
    before = (tokenizer.think_start, tokenizer.think_end, tokenizer.think_start_tokens)
    assert thinking.ensure_thinking_markers(tokenizer) is False
    assert (
        tokenizer.think_start,
        tokenizer.think_end,
        tokenizer.think_start_tokens,
    ) == before


def test_ensure_markers_installs_them_when_the_wrapper_did_not_resolve(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("EXO_DSV41_THINK_MARKERS", raising=False)
    tokenizer = FakeTokenizer(vocab=VOCAB, has_thinking=False)
    assert tokenizer.has_thinking is False
    assert thinking.ensure_thinking_markers(tokenizer) is True
    assert tokenizer.think_start == THINK_START
    assert tokenizer.think_end == THINK_END
    assert tokenizer.think_start_tokens == (128821,)
    assert tokenizer.think_end_tokens == (128822,)
    assert tokenizer.has_thinking is True
    # Idempotent: a second call is a no-op.
    assert thinking.ensure_thinking_markers(tokenizer) is False
    assert thinking.markers_match_checkpoint(tokenizer) is True


def test_ensure_markers_warns_and_declines_when_the_vocab_lacks_them(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("EXO_DSV41_THINK_MARKERS", raising=False)
    tokenizer = FakeTokenizer(vocab={"hello": 7}, has_thinking=False)
    assert thinking.ensure_thinking_markers(tokenizer) is False
    assert tokenizer.has_thinking is False


def test_env_var_disables_the_patch(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EXO_DSV41_THINK_MARKERS", "0")
    tokenizer = FakeTokenizer(vocab=VOCAB, has_thinking=False)
    assert thinking.ensure_thinking_markers(tokenizer) is False
    assert tokenizer.has_thinking is False


def test_attribute_guard_raises_when_mlx_lm_renames_the_field(
    monkeypatch: pytest.MonkeyPatch,
):
    """A silent no-op here would leave reasoning parsed as content forever."""
    monkeypatch.delenv("EXO_DSV41_THINK_MARKERS", raising=False)
    tokenizer = FakeTokenizer(vocab=VOCAB, has_thinking=False)
    assert hasattr(tokenizer, "_think_start")
    # Emulate the rename: the private attributes are gone from the instance.
    for attr in (
        "_think_start_tokens",
        "_think_end_tokens",
    ):
        del tokenizer.__dict__[attr]
    with pytest.raises(RuntimeError, match="_think_start"):
        thinking.ensure_thinking_markers(tokenizer)


def test_markers_match_checkpoint_detects_a_foreign_pair():
    tokenizer = FakeTokenizer(
        vocab=VOCAB, think_start="<|think|>", think_end="<|/think|>"
    )
    assert tokenizer.has_thinking is True
    assert thinking.markers_match_checkpoint(tokenizer) is False
    assert thinking.markers_match_checkpoint(FakeTokenizer(vocab=VOCAB)) is True
    assert (
        thinking.markers_match_checkpoint(
            FakeTokenizer(vocab=VOCAB, has_thinking=False)
        )
        is False
    )


def test_checkpoint_style_vocab_is_the_source_of_the_constants():
    """If the constants ever drift, this fails with the checkpoint's own values."""
    payload = json.loads(
        Path(__file__).with_name("checkpoint_sentinels.json").read_text()
    )
    think = payload["thinking"]
    # The checkpoint's markers are pinned as utf-8 hex (see the JSON's note);
    # decoding them is the same byte-exact check the module-level test does.
    assert bytes.fromhex(think["start_utf8_hex"]).decode() == thinking.THINK_START
    assert bytes.fromhex(think["end_utf8_hex"]).decode() == thinking.THINK_END
    assert think["start_id"] == thinking.THINK_START_ID
    assert think["end_id"] == thinking.THINK_END_ID
    assert payload["dsml"]["sentinel_id"] == 128825
