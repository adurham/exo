# pyright: reportAny=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownLambdaType=false, reportPrivateUsage=false
# pyright: reportInvalidCast=false, reportArgumentType=false
"""Batch=1 (well, sequential-submit) regression test for the Phase 1
batched-decode wiring in ExoBatchGenerator (design doc Section 9,
prerequisite item 3): with EXO_PP_BATCHED_DECODE unset (the shipped
default), submit()/step() must be byte-identical to before this
session's edit.

METHODOLOGY NOTE (why this test asserts against a fixed token
sequence rather than an independently-computed "golden" reference):
two earlier drafts of this test hand-reconstructed an "equivalent"
forward-pass sequence to compare against, and got mlx-lm's internal
insert()/prefill() token-consumption accounting wrong TWICE (a subtle
implementation detail, not a bug in the code under test). Per a
`consult` review, the responsible fix was differential (A/B) testing
against the actual PRE-EDIT code, not continuing to fight mlx-lm's
internals with a third hand-rolled reconstruction: using `git
worktree` to run this exact scenario against origin/main (commit
9c9ab9623, before this session's ExoBatchGenerator edit) and the
patched tree side-by-side, and diffing their JSON output byte-for-
byte. That confirmed EVERY token across a 2-request sequential-submit
scenario was IDENTICAL between the two trees (the only diff was the
new `_batched_decode_active` attribute existing at all, which does
not exist pre-edit and is expected). The exact token sequence that
A/B run produced (verified against real pre-edit code, not derived
independently) is committed here as a fixture -- this is a real,
externally-verified regression baseline, not a self-referential
tautology (the test does not compute its own expected output using
the same code under test).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import pytest
from mlx_lm.tokenizer_utils import TokenizerWrapper
from transformers import AutoTokenizer

from exo.shared.types.common import ModelId
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.worker.engines.mlx.generator.batch_generate import ExoBatchGenerator

# Verified byte-for-byte against real pre-edit code (git worktree A/B,
# origin/main commit 9c9ab9623) -- see module docstring.
_EXPECTED_REQUEST_1_TOKENS = [1686, 1475, 1851, 1927, 4062, 3800]
_EXPECTED_REQUEST_1_FINISH_REASONS = [None, None, None, None, None, "length"]
_EXPECTED_REQUEST_2_TOKENS = [2178, 1179, 3212, 4086, 3494]
_EXPECTED_REQUEST_2_FINISH_REASONS = [None, None, None, None, "length"]


def _make_tiny_llama() -> nn.Module:
    from mlx_lm.models.llama import Model as LlamaModel
    from mlx_lm.models.llama import ModelArgs

    args = ModelArgs(
        model_type="llama",
        hidden_size=256,
        num_hidden_layers=4,
        intermediate_size=512,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
        vocab_size=4096,
        rope_theta=10000.0,
        tie_word_embeddings=True,
    )
    mx.random.seed(42)
    model = LlamaModel(args)
    params = model.parameters()
    new_params = mlx.utils.tree_map(
        lambda p: mx.random.normal(shape=p.shape, dtype=p.dtype)
        if isinstance(p, mx.array)
        else p,
        params,
    )
    model.update(new_params)
    mx.eval(model.parameters())
    return model


def _make_tokenizer() -> TokenizerWrapper:
    from huggingface_hub import snapshot_download

    model_path = snapshot_download(
        "mlx-community/Qwen3.5-35B-A3B-4bit",
        allow_patterns=["tokenizer*", "*.jinja"],
    )
    hf_tokenizer = AutoTokenizer.from_pretrained(model_path)
    return TokenizerWrapper(hf_tokenizer)


def _run_request(
    gen: ExoBatchGenerator, prompt: str, n_steps: int
) -> tuple[list[int], list[str | None]]:
    task_params = TextGenerationTaskParams(
        model=ModelId("test-model"),
        input=[],
        max_output_tokens=n_steps,
        temperature=0.0,
        seed=0,
    )
    uid = gen.submit(task_params, prompt)
    produced: list[int] = []
    finish_reasons: list[str | None] = []
    finished = False
    for _ in range(n_steps + 5):
        if finished:
            break
        for response_uid, response in gen.step():
            if response_uid != uid:
                continue
            produced.append(int(response.token))
            finish_reasons.append(response.finish_reason)
            if response.finish_reason is not None:
                finished = True
    return produced, finish_reasons


def test_exobatchgenerator_flag_off_matches_verified_pre_edit_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With EXO_PP_BATCHED_DECODE unset (the shipped default), a real
    ExoBatchGenerator instance running TWO sequential submit()/step()
    cycles on the same instance (proving both single-request
    correctness AND that a second, later submit() still works
    identically -- a real invariant this class's existing per-uid
    bookkeeping must preserve) produces token-for-token IDENTICAL
    output to the real pre-edit code, per the git-worktree A/B run
    documented in this module's own docstring."""
    monkeypatch.delenv("EXO_PP_BATCHED_DECODE", raising=False)

    model = _make_tiny_llama()
    tokenizer = _make_tokenizer()

    gen = ExoBatchGenerator(
        model=model, tokenizer=tokenizer, group=None, kv_prefix_cache=None
    )

    # Confirm the flag is genuinely off, not just unset-by-accident --
    # the whole point of this test is proving this exact invariant.
    assert gen._batched_decode_active is False
    assert gen._batched_decode_rank0_glue is None
    assert gen._batched_decode_rank1_glue is None

    tokens_1, finish_1 = _run_request(
        gen, "Write a short essay about AI.", len(_EXPECTED_REQUEST_1_TOKENS)
    )
    assert tokens_1 == _EXPECTED_REQUEST_1_TOKENS
    assert finish_1 == _EXPECTED_REQUEST_1_FINISH_REASONS

    tokens_2, finish_2 = _run_request(
        gen, "Explain evolution briefly.", len(_EXPECTED_REQUEST_2_TOKENS)
    )
    assert tokens_2 == _EXPECTED_REQUEST_2_TOKENS
    assert finish_2 == _EXPECTED_REQUEST_2_FINISH_REASONS
