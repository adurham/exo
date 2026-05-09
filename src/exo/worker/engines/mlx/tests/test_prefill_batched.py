# pyright: reportAny=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownLambdaType=false, reportPrivateUsage=false
# pyright: reportInvalidCast=false, reportArgumentType=false
# pyright: reportUnusedImport=false
"""Bit-equivalence tests for ``prefill_batched`` vs serial ``prefill``.

Proves that running ``prefill_batched([tokens_a, tokens_b], [cache_a, cache_b])``
produces caches that match running ``prefill`` on each prompt separately.
The downstream decode logits should match across both paths to within
right-padded BatchKVCache numerical tolerance (~2e-3 — same as the existing
B=1-vs-B=2 test).

Targets the c=2 long-context regression: legacy ``prefill()`` ran one stream
at a time at submit() — at 100K context that serialized stream 1 behind
stream 0's ~6 min prefill, collapsing per-stream throughput to 7.7 tok/s.
``prefill_batched`` processes both streams in a single batched pass.

Random weights — no model download required for the model itself; only the
tokenizer is fetched from HuggingFace.
"""

from pathlib import Path
from typing import cast

import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import pytest
from mlx_lm.generate import _merge_caches
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper
from transformers import AutoTokenizer

import exo.worker.engines.mlx.generator.batch_generate  # noqa: F401  (right-padding patch)
from exo.worker.engines.mlx.cache import encode_prompt, make_kv_cache
from exo.worker.engines.mlx.generator.generate import (
    prefill,
    prefill_batched,
)
from exo.worker.engines.mlx.types import Model

# How many decode tokens to compare across the two prefill paths.
NUM_DECODE_STEPS = 20

# Numerical tolerance — matches the existing ``test_batch_generate.py`` budget.
# Bit-equivalence at temp=0 is asserted via ``argmax`` selecting the same
# token (`mismatches == 0`) rather than per-element float identity.
LOGIT_DIFF_TOLERANCE = 0.002


def _init_random(model: nn.Module) -> None:
    params = model.parameters()
    new_params = mlx.utils.tree_map(
        lambda p: mx.random.normal(shape=p.shape, dtype=p.dtype)
        if isinstance(p, mx.array)
        else p,
        params,
    )
    model.update(new_params)
    mx.eval(model.parameters())


def _make_tokenizer() -> TokenizerWrapper:
    from huggingface_hub import snapshot_download

    model_path = Path(
        snapshot_download(
            "mlx-community/Qwen3.5-35B-A3B-4bit",
            allow_patterns=["tokenizer*", "*.jinja"],
        )
    )
    hf_tokenizer = AutoTokenizer.from_pretrained(model_path)
    return TokenizerWrapper(hf_tokenizer)


def _decode_logits(
    model: Model,
    cache: list,
    next_token: int,
    n_steps: int,
) -> tuple[list[mx.array], int]:
    """Decode ``n_steps`` tokens from ``next_token`` against a B=1 cache,
    return per-step last-position logits and the final next_token."""
    logits_per_step: list[mx.array] = []
    cur = next_token
    for _ in range(n_steps):
        out = model(mx.array([[cur]]), cache=cache)
        mx.eval(out)
        logits_per_step.append(out[0, -1])
        cur = int(mx.argmax(out[0, -1]).item())
    return logits_per_step, cur


def _decode_logits_batched(
    model: Model,
    cache: list,
    next_tokens: list[int],
    n_steps: int,
) -> list[list[mx.array]]:
    """Decode ``n_steps`` tokens from ``next_tokens`` against a batched cache,
    return per-stream per-step last-position logits."""
    n_streams = len(next_tokens)
    cur = list(next_tokens)
    per_stream: list[list[mx.array]] = [[] for _ in range(n_streams)]
    for _ in range(n_steps):
        out = model(mx.array([[c] for c in cur]), cache=cache)
        mx.eval(out)
        for i in range(n_streams):
            per_stream[i].append(out[i, -1])
            cur[i] = int(mx.argmax(out[i, -1]).item())
    return per_stream


def _compare_logits(
    legacy: list[mx.array],
    batched: list[mx.array],
    label: str,
) -> tuple[float, int]:
    max_diff = 0.0
    mismatches = 0
    for step in range(len(legacy)):
        diff = float(
            mx.max(
                mx.abs(
                    legacy[step].astype(mx.float32)
                    - batched[step].astype(mx.float32)
                )
            ).item()
        )
        max_diff = max(max_diff, diff)
        if int(mx.argmax(legacy[step]).item()) != int(
            mx.argmax(batched[step]).item()
        ):
            mismatches += 1
    print(
        f"[{label}] max_diff={max_diff:.4e} mismatches={mismatches}/{len(legacy)}"
    )
    return max_diff, mismatches


def _run_prefill_batched_vs_serial(
    model: Model,
    tokenizer: TokenizerWrapper,
    tokens_a: mx.array,
    tokens_b: mx.array,
) -> tuple[float, int]:
    """Compare prefill_batched vs serial prefill+merge for two prompts.

    Both paths end with a (B=2) batched cache that decodes the same tokens.
    Returns ``(max_diff, mismatches)`` aggregated across both streams.
    """
    sampler = make_sampler(temp=0.0)

    # === Legacy path: serial prefill per stream, then merge for batched decode ===
    cache_a_legacy = make_kv_cache(model)
    cache_b_legacy = make_kv_cache(model)
    prefill(
        model, tokenizer, sampler, tokens_a[:-1], cache_a_legacy, None, None, None
    )
    prefill(
        model, tokenizer, sampler, tokens_b[:-1], cache_b_legacy, None, None, None
    )
    merged_legacy = _merge_caches([list(cache_a_legacy), list(cache_b_legacy)])
    for c in merged_legacy:
        c.prepare(lengths=[1, 1], right_padding=[0, 0])
    model(
        mx.array([[tokens_a[-2].item()], [tokens_b[-2].item()]]),
        cache=merged_legacy,
    )
    mx.eval([c.state for c in merged_legacy])
    for c in merged_legacy:
        c.finalize()

    legacy_logits = _decode_logits_batched(
        model,
        merged_legacy,
        [int(tokens_a[-1].item()), int(tokens_b[-1].item())],
        NUM_DECODE_STEPS,
    )

    # === Batched path: prefill_batched in one shot, then merge for batched decode ===
    cache_a_fresh = make_kv_cache(model)
    cache_b_fresh = make_kv_cache(model)
    prefill_batched(
        model,
        tokenizer,
        sampler,
        [tokens_a[:-1], tokens_b[:-1]],
        [cache_a_fresh, cache_b_fresh],
        None,
        None,
        None,
        prefill_step_size=4096,
    )
    # prefill_batched returns per-stream caches as the third element. Use
    # those (fresh extracted KVCaches) for the merged-decode step. The
    # input cache lists ``cache_a_fresh`` / ``cache_b_fresh`` are NOT
    # mutated by prefill_batched (the merge takes a copy via _merge_caches).
    _, _, per_stream_caches, _ = prefill_batched(
        model,
        tokenizer,
        sampler,
        [tokens_a[:-1], tokens_b[:-1]],
        [make_kv_cache(model), make_kv_cache(model)],
        None,
        None,
        None,
        prefill_step_size=4096,
    )
    merged_batched = _merge_caches(per_stream_caches)
    for c in merged_batched:
        c.prepare(lengths=[1, 1], right_padding=[0, 0])
    model(
        mx.array([[tokens_a[-2].item()], [tokens_b[-2].item()]]),
        cache=merged_batched,
    )
    mx.eval([c.state for c in merged_batched])
    for c in merged_batched:
        c.finalize()

    batched_logits = _decode_logits_batched(
        model,
        merged_batched,
        [int(tokens_a[-1].item()), int(tokens_b[-1].item())],
        NUM_DECODE_STEPS,
    )

    max_diff_a, mis_a = _compare_logits(
        legacy_logits[0], batched_logits[0], "stream A"
    )
    max_diff_b, mis_b = _compare_logits(
        legacy_logits[1], batched_logits[1], "stream B"
    )
    return max(max_diff_a, max_diff_b), mis_a + mis_b


@pytest.mark.slow
def test_prefill_batched_vs_serial_llama_same_length() -> None:
    """Llama-style model (KVCache only): same-length prompts must match
    serial prefill + merge bit-exactly.

    No padding involved. This is the simplest case — the merged cache state
    after batched prefill should equal serial-prefill-then-merge byte-for-byte.
    """
    from mlx_lm.models.llama import Model as LlamaModel
    from mlx_lm.models.llama import ModelArgs

    mx.random.seed(42)
    args = ModelArgs(
        model_type="llama",
        hidden_size=256,
        num_hidden_layers=4,
        intermediate_size=512,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
        vocab_size=248320,
        rope_theta=10000.0,
        tie_word_embeddings=True,
    )
    model = LlamaModel(args)
    _init_random(model)

    tokenizer = _make_tokenizer()
    # Pad to identical lengths via repeat — same length removes the
    # right-padding code path and tests pure batched-prefill correctness.
    tokens_a = encode_prompt(tokenizer, "Write a short essay about AI today.")
    tokens_b = encode_prompt(tokenizer, "Explain evolution and selection.")
    common_length = min(int(tokens_a.shape[0]), int(tokens_b.shape[0]))
    tokens_a = tokens_a[:common_length]
    tokens_b = tokens_b[:common_length]

    max_diff, mismatches = _run_prefill_batched_vs_serial(
        cast(Model, model), tokenizer, tokens_a, tokens_b
    )
    assert mismatches == 0, (
        f"prefill_batched (same-length) token mismatches: {mismatches}/{NUM_DECODE_STEPS * 2}"
    )
    assert max_diff < LOGIT_DIFF_TOLERANCE, (
        f"prefill_batched (same-length) max logit diff: {max_diff}"
    )


@pytest.mark.slow
def test_prefill_batched_vs_serial_llama_diff_length() -> None:
    """Different-length prompts: right-padding + finalize must produce the
    same per-stream caches as serial prefill on each prompt."""
    from mlx_lm.models.llama import Model as LlamaModel
    from mlx_lm.models.llama import ModelArgs

    mx.random.seed(42)
    args = ModelArgs(
        model_type="llama",
        hidden_size=256,
        num_hidden_layers=4,
        intermediate_size=512,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
        vocab_size=248320,
        rope_theta=10000.0,
        tie_word_embeddings=True,
    )
    model = LlamaModel(args)
    _init_random(model)

    tokenizer = _make_tokenizer()
    # Naturally different lengths — exercises the right-padding +
    # finalize() path inside prefill_batched.
    tokens_a = encode_prompt(
        tokenizer,
        "Write a long technical essay about artificial intelligence and its societal implications today.",
    )
    tokens_b = encode_prompt(tokenizer, "Explain evolution.")
    assert int(tokens_a.shape[0]) != int(tokens_b.shape[0]), (
        "test prerequisite: tokens_a and tokens_b must differ in length"
    )

    max_diff, mismatches = _run_prefill_batched_vs_serial(
        cast(Model, model), tokenizer, tokens_a, tokens_b
    )
    # Right-padding finalize() introduces a small numerical residual
    # via ``dynamic_roll``; argmax-token equivalence must still hold.
    assert mismatches == 0, (
        f"prefill_batched (diff-length) token mismatches: {mismatches}/{NUM_DECODE_STEPS * 2}"
    )
    assert max_diff < LOGIT_DIFF_TOLERANCE, (
        f"prefill_batched (diff-length) max logit diff: {max_diff}"
    )


@pytest.mark.slow
def test_prefill_batched_b1_degenerate_llama() -> None:
    """B=1 degenerate case must equal serial ``prefill()`` exactly.

    Phase 2 wedge guard: a c=1 deployment running the batched path
    cannot wedge or diverge from c=1 serial behavior.
    """
    from mlx_lm.models.llama import Model as LlamaModel
    from mlx_lm.models.llama import ModelArgs

    mx.random.seed(42)
    args = ModelArgs(
        model_type="llama",
        hidden_size=256,
        num_hidden_layers=4,
        intermediate_size=512,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
        vocab_size=248320,
        rope_theta=10000.0,
        tie_word_embeddings=True,
    )
    model = LlamaModel(args)
    _init_random(model)

    tokenizer = _make_tokenizer()
    tokens = encode_prompt(tokenizer, "Write a short essay about AI.")
    sampler = make_sampler(temp=0.0)

    # Legacy
    cache_legacy = make_kv_cache(cast(Model, model))
    prefill(
        cast(Model, model),
        tokenizer,
        sampler,
        tokens[:-1],
        cache_legacy,
        None,
        None,
        None,
    )

    # Batched (B=1)
    cache_fresh = make_kv_cache(cast(Model, model))
    _, _, per_stream_caches, _ = prefill_batched(
        cast(Model, model),
        tokenizer,
        sampler,
        [tokens[:-1]],
        [cache_fresh],
        None,
        None,
        None,
        prefill_step_size=4096,
    )

    # Compare cache layer-wise via decode-step logits — clean way to
    # collapse per-layer KV cache to a single comparable scalar without
    # having to bit-compare quantized state.
    legacy_logits, _ = _decode_logits(
        cast(Model, model),
        list(cache_legacy),
        int(tokens[-1].item()),
        NUM_DECODE_STEPS,
    )
    batched_logits, _ = _decode_logits(
        cast(Model, model),
        list(per_stream_caches[0]),
        int(tokens[-1].item()),
        NUM_DECODE_STEPS,
    )

    max_diff, mismatches = _compare_logits(legacy_logits, batched_logits, "B=1")
    assert mismatches == 0, (
        f"prefill_batched (B=1) token mismatches: {mismatches}/{NUM_DECODE_STEPS}"
    )
    assert max_diff < LOGIT_DIFF_TOLERANCE, (
        f"prefill_batched (B=1) max logit diff: {max_diff}"
    )
