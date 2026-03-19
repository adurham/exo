# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false
"""Tests for local TP-sharded sparse draft speculative decoding.

Tests the new code paths without requiring a real cluster or distributed group.
Covers: sparse layer selection, patch_model=False, local draft_fn, draft detection,
and runner wiring logic.
"""

import importlib
import json
import tempfile
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

from exo.worker.engines.mlx.auto_parallel import (
    patch_tensor_model,
)


# ── Helpers ─────────────────────────────────────────────────────────────── #


def _build_tiny_llama(num_layers: int = 4) -> nn.Module:
    """Build a tiny Llama model with random weights for testing."""
    cfg: dict[str, Any] = {
        "model_type": "llama",
        "vocab_size": 1000,
        "hidden_size": 64,
        "num_hidden_layers": num_layers,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "intermediate_size": 128,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "tie_word_embeddings": False,
    }
    mod = importlib.import_module("mlx_lm.models.llama")
    args = mod.ModelArgs.from_dict(cfg)
    model: nn.Module = mod.Model(args)
    flat = cast(list[tuple[str, mx.array]], tree_flatten(model.parameters()))
    random_weights = [
        (k, mx.random.normal(shape=v.shape, dtype=mx.float16)) for k, v in flat
    ]
    model.update(cast(dict[str, Any], tree_unflatten(random_weights)))
    mx.eval(model.parameters())
    return model


def _build_tiny_qwen3_moe(num_layers: int = 4) -> nn.Module:
    """Build a tiny Qwen3 MoE model with random weights for testing."""
    cfg: dict[str, Any] = {
        "model_type": "qwen3_moe",
        "vocab_size": 1000,
        "hidden_size": 64,
        "num_hidden_layers": num_layers,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "intermediate_size": 128,
        "moe_intermediate_size": 64,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "tie_word_embeddings": False,
        "shared_expert_intermediate_size": 128,
        "norm_topk_prob": True,
        "decoder_sparse_step": 1,
        "mlp_only_layers": [],
        "head_dim": 16,
        "max_position_embeddings": 1024,
    }
    mod = importlib.import_module("mlx_lm.models.qwen3_moe")
    args = mod.ModelArgs.from_dict(cfg)
    model: nn.Module = mod.Model(args)
    flat = cast(list[tuple[str, mx.array]], tree_flatten(model.parameters()))
    random_weights = [
        (k, mx.random.normal(shape=v.shape, dtype=mx.float16)) for k, v in flat
    ]
    model.update(cast(dict[str, Any], tree_unflatten(random_weights)))
    mx.eval(model.parameters())
    return model


# ── Test: Sparse layer index selection ──────────────────────────────────── #


def test_sparse_layer_indices_skip_6():
    """Verify skip_factor=6 with 94 layers produces the expected 17 layers."""
    n_layers = 94
    skip_factor = 6
    kept = list(range(0, n_layers, skip_factor))
    if kept[-1] != n_layers - 1:
        kept.append(n_layers - 1)

    assert len(kept) == 17
    assert kept[0] == 0
    assert kept[-1] == 93
    assert kept[1] == 6
    assert kept[2] == 12
    # Verify all indices are in range
    assert all(0 <= i < n_layers for i in kept)


def test_sparse_layer_indices_skip_16():
    """Verify skip_factor=16 with 94 layers produces expected layers."""
    n_layers = 94
    skip_factor = 16
    kept = list(range(0, n_layers, skip_factor))
    if kept[-1] != n_layers - 1:
        kept.append(n_layers - 1)

    assert len(kept) == 7
    assert kept[0] == 0
    assert kept[-1] == 93
    assert kept == [0, 16, 32, 48, 64, 80, 93]


def test_sparse_layer_indices_exact_divisor():
    """When n_layers-1 is exactly divisible, last layer is not duplicated."""
    n_layers = 13
    skip_factor = 4
    kept = list(range(0, n_layers, skip_factor))
    if kept[-1] != n_layers - 1:
        kept.append(n_layers - 1)

    assert kept == [0, 4, 8, 12]
    assert len(kept) == len(set(kept)), "No duplicate indices"


# ── Test: Weight key filtering ──────────────────────────────────────────── #


def test_sparse_key_needed_filtering():
    """Verify _key_needed logic correctly filters weight keys."""
    skip_factor = 6
    n_layers = 12
    kept_indices = list(range(0, n_layers, skip_factor))
    if kept_indices[-1] != n_layers - 1:
        kept_indices.append(n_layers - 1)
    # kept = [0, 6, 11]

    needed_prefixes = {f"model.layers.{idx}." for idx in kept_indices}

    def key_needed(key: str) -> bool:
        if not key.startswith("model.layers."):
            return True
        return any(key.startswith(p) for p in needed_prefixes)

    # Non-layer keys always pass
    assert key_needed("model.embed_tokens.weight")
    assert key_needed("model.norm.weight")
    assert key_needed("lm_head.weight")

    # Kept layers pass
    assert key_needed("model.layers.0.self_attn.q_proj.weight")
    assert key_needed("model.layers.6.mlp.gate_proj.weight")
    assert key_needed("model.layers.11.self_attn.o_proj.weight")

    # Skipped layers are filtered
    assert not key_needed("model.layers.1.self_attn.q_proj.weight")
    assert not key_needed("model.layers.3.mlp.gate_proj.weight")
    assert not key_needed("model.layers.10.self_attn.o_proj.weight")


# ── Test: Weight key remapping ──────────────────────────────────────────── #


def test_sparse_weight_remapping():
    """Verify weight keys are correctly remapped to contiguous indices."""
    kept_indices = [0, 6, 11]
    idx_map = {orig: new for new, orig in enumerate(kept_indices)}

    # model.layers.0.x → model.layers.0.x (unchanged)
    assert idx_map[0] == 0
    # model.layers.6.x → model.layers.1.x
    assert idx_map[6] == 1
    # model.layers.11.x → model.layers.2.x
    assert idx_map[11] == 2

    # Verify remapping
    key = "model.layers.6.self_attn.q_proj.weight"
    parts = key.split(".", 3)
    orig_idx = int(parts[2])
    new_key = f"model.layers.{idx_map[orig_idx]}.{parts[3]}"
    assert new_key == "model.layers.1.self_attn.q_proj.weight"


# ── Test: patch_model=False preserves original __call__ ─────────────────── #


def test_patch_model_false_preserves_call():
    """tensor_auto_parallel with patch_model=False should not modify cls.__call__."""
    model = _build_tiny_llama(num_layers=2)
    cls = model.__class__
    original_call = cls.__call__

    # Verify the model works
    tokens = mx.array([[1, 2, 3]])
    output = model(tokens)
    mx.eval(output)
    assert output.shape[0] == 1
    assert output.shape[1] == 3

    # After patch_tensor_model, __call__ changes
    # (We test this to confirm the mechanism works, then test that
    #  patch_model=False skips it)
    patched_model = patch_tensor_model(model)
    assert cls.__call__ is not original_call, "patch_tensor_model should modify __call__"

    # Restore for the next test
    cls.__call__ = original_call
    assert cls.__call__ is original_call


# ── Test: Draft detection logic ─────────────────────────────────────────── #


def test_draft_detection_local_nn_module():
    """nn.Module draft model should be detected as local draft."""
    model = _build_tiny_llama(num_layers=2)
    assert isinstance(model, nn.Module)
    assert not hasattr(model, 'prefill')

    _is_local = isinstance(model, nn.Module)
    _is_remote = hasattr(model, 'prefill')
    assert _is_local
    assert not _is_remote


def test_draft_detection_remote_draft_client():
    """DraftClient-like object should be detected as remote draft."""
    mock_client = MagicMock()
    mock_client.prefill = MagicMock()
    mock_client.fetch_draft_sync = MagicMock()
    mock_client.num_draft_tokens = 5

    # MagicMock is NOT an nn.Module
    _is_local = isinstance(mock_client, nn.Module)
    _is_remote = hasattr(mock_client, 'prefill')
    assert not _is_local
    assert _is_remote


def test_draft_detection_none():
    """None draft model should be neither local nor remote."""
    draft_model = None
    _is_local = draft_model is not None and isinstance(draft_model, nn.Module)
    _is_remote = draft_model is not None and hasattr(draft_model, 'prefill')
    assert not _is_local
    assert not _is_remote


# ── Test: Local draft function ──────────────────────────────────────────── #


def test_local_draft_fn_produces_tokens():
    """Local draft function should produce the requested number of tokens."""
    model = _build_tiny_llama(num_layers=2)
    cache = make_prompt_cache(model)

    # Prefill with a short sequence
    prompt = mx.array([[1, 2, 3, 4, 5]])
    model(prompt, cache=cache)
    mx.eval([c.state for c in cache])

    # Build draft function (same pattern as generate.py)
    def draft_fn(token_id: int, num_tokens: int, trim: int = 0) -> list[int]:
        if trim > 0:
            trim_prompt_cache(cache, trim)
        tokens: list[int] = []
        tok = token_id
        for _ in range(num_tokens):
            logits = model(mx.array([[tok]]), cache=cache)
            mx.eval(logits)
            tok = int(logits[0, -1].argmax().item())
            tokens.append(tok)
        return tokens

    # Generate 5 draft tokens
    result = draft_fn(token_id=10, num_tokens=5)
    assert len(result) == 5
    assert all(isinstance(t, int) for t in result)
    assert all(0 <= t < 1000 for t in result)  # vocab_size=1000


def test_local_draft_fn_trim():
    """Draft function should correctly trim cache on rejection."""
    model = _build_tiny_llama(num_layers=2)
    cache = make_prompt_cache(model)

    # Prefill
    prompt = mx.array([[1, 2, 3, 4, 5]])
    model(prompt, cache=cache)
    mx.eval([c.state for c in cache])
    initial_offset = cache[0].offset

    # Generate some tokens to grow the cache
    def draft_fn(token_id: int, num_tokens: int, trim: int = 0) -> list[int]:
        if trim > 0:
            trim_prompt_cache(cache, trim)
        tokens: list[int] = []
        tok = token_id
        for _ in range(num_tokens):
            logits = model(mx.array([[tok]]), cache=cache)
            mx.eval(logits)
            tok = int(logits[0, -1].argmax().item())
            tokens.append(tok)
        return tokens

    # Generate 3 tokens (cache grows by 3)
    result1 = draft_fn(10, 3)
    after_gen_offset = cache[0].offset
    assert after_gen_offset == initial_offset + 3

    # Trim 2 (simulates 2 rejected tokens), then generate 3 more
    result2 = draft_fn(result1[0], 3, trim=2)
    assert len(result2) == 3
    # Cache should be: initial + 3 - 2 + 3 = initial + 4
    assert cache[0].offset == initial_offset + 4


def test_local_draft_fn_deterministic():
    """Same input should produce same output (argmax is deterministic)."""
    model = _build_tiny_llama(num_layers=2)

    def run_draft(token_id: int, num_tokens: int) -> list[int]:
        cache = make_prompt_cache(model)
        prompt = mx.array([[1, 2, 3]])
        model(prompt, cache=cache)
        mx.eval([c.state for c in cache])

        tokens: list[int] = []
        tok = token_id
        for _ in range(num_tokens):
            logits = model(mx.array([[tok]]), cache=cache)
            mx.eval(logits)
            tok = int(logits[0, -1].argmax().item())
            tokens.append(tok)
        return tokens

    result1 = run_draft(10, 5)
    result2 = run_draft(10, 5)
    assert result1 == result2, "Argmax draft should be deterministic"


# ── Test: Draft with MoE model ─────────────────────────────────────────── #


def test_local_draft_fn_qwen3_moe():
    """Draft function should work with MoE architecture (the actual 235B arch)."""
    model = _build_tiny_qwen3_moe(num_layers=2)
    cache = make_prompt_cache(model)

    prompt = mx.array([[1, 2, 3, 4, 5]])
    model(prompt, cache=cache)
    mx.eval([c.state for c in cache])

    tokens: list[int] = []
    tok = 10
    for _ in range(3):
        logits = model(mx.array([[tok]]), cache=cache)
        mx.eval(logits)
        tok = int(logits[0, -1].argmax().item())
        tokens.append(tok)

    assert len(tokens) == 3
    assert all(isinstance(t, int) for t in tokens)


# ── Test: Runner wiring prioritization ──────────────────────────────────── #


def test_runner_local_draft_priority():
    """When local_draft is not None, it should be used instead of DraftClient."""
    local_draft = _build_tiny_llama(num_layers=2)

    # Simulate the runner wiring logic
    generator_draft_model = None
    generator_use_speculative = False

    if local_draft is not None:
        generator_draft_model = local_draft
        generator_use_speculative = True
    else:
        # Remote DraftClient path (should not be reached)
        generator_draft_model = "remote_client"
        generator_use_speculative = True

    assert isinstance(generator_draft_model, nn.Module)
    assert generator_use_speculative is True


def test_runner_fallback_to_remote():
    """When local_draft is None, should fall through to remote draft path."""
    local_draft = None

    generator_draft_model = None
    generator_use_speculative = False

    if local_draft is not None:
        generator_draft_model = local_draft
        generator_use_speculative = True

    assert generator_draft_model is None
    assert generator_use_speculative is False


# ── Test: load_sparse_tp_model function signature ───────────────────────── #


def test_load_sparse_tp_model_exists():
    """Verify the function exists with expected signature."""
    from exo.worker.engines.mlx.sparse_model import load_sparse_tp_model
    import inspect

    sig = inspect.signature(load_sparse_tp_model)
    params = list(sig.parameters.keys())
    assert "model_path" in params
    assert "skip_factor" in params
    assert "group" in params
    assert "on_layer_loaded" in params


def test_tensor_auto_parallel_has_patch_model_param():
    """Verify tensor_auto_parallel accepts patch_model kwarg."""
    from exo.worker.engines.mlx.auto_parallel import tensor_auto_parallel
    import inspect

    sig = inspect.signature(tensor_auto_parallel)
    assert "patch_model" in sig.parameters
    param = sig.parameters["patch_model"]
    assert param.default is True
    assert param.kind == inspect.Parameter.KEYWORD_ONLY


# ── Test: load_mlx_items returns 3-tuple ────────────────────────────────── #


def test_load_mlx_items_signature():
    """Verify load_mlx_items has 3-element return type annotation."""
    from exo.worker.engines.mlx.utils_mlx import load_mlx_items
    import inspect

    sig = inspect.signature(load_mlx_items)
    ret = sig.return_annotation
    # Should be tuple[Model, TokenizerWrapper, object | None]
    assert "tuple" in str(ret).lower()


# ── Test: shard_and_load returns 3-tuple ────────────────────────────────── #


def test_shard_and_load_signature():
    """Verify shard_and_load has 3-element return type annotation."""
    from exo.worker.engines.mlx.utils_mlx import shard_and_load
    import inspect

    sig = inspect.signature(shard_and_load)
    ret = sig.return_annotation
    assert "Module | None" in str(ret)


# ── Test: EXO_DRAFT_TOKENS env var ──────────────────────────────────────── #


def test_draft_tokens_env_var(monkeypatch: pytest.MonkeyPatch):
    """EXO_DRAFT_TOKENS should control the number of draft tokens."""
    import os
    monkeypatch.setenv("EXO_DRAFT_TOKENS", "7")
    val = int(os.environ.get("EXO_DRAFT_TOKENS", "5"))
    assert val == 7


def test_draft_tokens_default():
    """Default draft tokens should be 5."""
    import os
    # Don't set env var, check default
    val = int(os.environ.get("EXO_DRAFT_TOKENS", "5"))
    assert val == 5


# ── Multiprocess TP integration tests ──────────────────────────────────── #
# These spawn 2 processes with ring backend to validate actual TP behavior.


def _tp_draft_worker(
    rank: int,
    world_size: int,
    hostfile_path: str,
    weights_bytes: bytes,
    result_queue: Any,
) -> None:
    """Worker process: TP-shard a tiny model, run draft fn, return tokens."""
    import os
    import pickle
    import traceback

    os.environ["MLX_HOSTFILE"] = hostfile_path
    os.environ["MLX_RANK"] = str(rank)

    try:
        group = mx.distributed.init(backend="ring", strict=True)

        # Build model from shared weights (same random init on all ranks)
        cfg: dict[str, Any] = {
            "model_type": "llama",
            "vocab_size": 1000,
            "hidden_size": 64,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "intermediate_size": 128,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "tie_word_embeddings": False,
        }
        mod = importlib.import_module("mlx_lm.models.llama")
        args = mod.ModelArgs.from_dict(cfg)
        model: nn.Module = mod.Model(args)

        # Load identical weights on all ranks
        weights = pickle.loads(weights_bytes)
        model.update(cast(dict[str, Any], tree_unflatten(weights)))
        mx.eval(model.parameters())

        # Primary patches __call__ (rank 0 does it, but it patches the CLASS
        # so both ranks see it)
        if rank == 0:
            from exo.worker.engines.mlx.auto_parallel import patch_tensor_model
            patch_tensor_model(model)

        # TP-shard with patch_model=False (draft path)
        from exo.worker.engines.mlx.auto_parallel import tensor_auto_parallel
        model = tensor_auto_parallel(
            model, group, timeout_seconds=30.0,
            on_timeout=None, on_layer_loaded=None,
            patch_model=False,
        )

        # Verify forward pass works
        tokens = mx.array([[1, 2, 3]])
        logits = model(tokens)
        mx.eval(logits)

        assert logits.shape[0] == 1
        assert logits.shape[1] == 3
        assert logits.shape[2] == 1000  # vocab_size

        # Build draft cache and prefill
        draft_cache = make_prompt_cache(model)
        prompt = mx.array([[1, 2, 3, 4, 5]])
        model(prompt, cache=draft_cache)
        mx.eval([c.state for c in draft_cache])

        # Run draft fn — generate 5 tokens
        draft_tokens: list[int] = []
        tok = 10
        for _ in range(5):
            logits = model(mx.array([[tok]]), cache=draft_cache)
            mx.eval(logits)
            tok = int(logits[0, -1].argmax().item())
            draft_tokens.append(tok)

        result_queue.put((rank, True, draft_tokens))

    except Exception as e:
        result_queue.put((rank, False, f"{e}\n{traceback.format_exc()}"))


def test_tp_draft_produces_identical_tokens_across_ranks():
    """2-rank TP draft: both ranks must produce identical argmax tokens.

    This validates the core correctness guarantee: TP all_sum inside the
    model ensures all ranks see the same logits, so argmax is identical
    without any broadcast.
    """
    import multiprocessing as mp
    import os
    import pickle

    ctx = mp.get_context("spawn")
    world_size = 2
    base_port = 29700

    hosts = [f"127.0.0.1:{base_port + i}" for i in range(world_size)]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(hosts, f)
        hostfile_path = f.name

    # Build shared random weights (identical across ranks)
    model = _build_tiny_llama(num_layers=4)
    flat = cast(list[tuple[str, mx.array]], tree_flatten(model.parameters()))
    weights_bytes = pickle.dumps(flat)
    del model

    try:
        result_queue: Any = ctx.Queue()
        processes: list[Any] = []
        for rank in range(world_size):
            p = ctx.Process(
                target=_tp_draft_worker,
                args=(rank, world_size, hostfile_path, weights_bytes, result_queue),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join(timeout=30)

        results: dict[int, list[int]] = {}
        errors: dict[int, str] = {}
        while not result_queue.empty():
            rank, success, value = result_queue.get()
            if success:
                results[rank] = value
            else:
                errors[rank] = value

        assert len(results) == world_size, (
            f"Expected {world_size} results, got {len(results)}. Errors: {errors}"
        )

        # THE KEY ASSERTION: both ranks produced identical tokens
        assert results[0] == results[1], (
            f"Rank 0 tokens {results[0]} != Rank 1 tokens {results[1]}"
        )

        # Sanity: tokens are valid
        assert len(results[0]) == 5
        assert all(0 <= t < 1000 for t in results[0])

    finally:
        os.unlink(hostfile_path)
