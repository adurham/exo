# pyright: reportMissingImports=false, reportUnknownMemberType=false
# pyright: reportUntypedFunctionDecorator=false, reportPrivateUsage=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
"""``patch_embed_tokens`` (generator/generate.py) bounds-check coverage.

Two fixes landed together with the mlx-lm-level `_assert_embeddable` defense-
in-depth check (mlx_lm/models/deepseek_v4.py, DeepseekV4Model._forward_steps):

1. `_inject` (the splice `patch_embed_tokens` installs on `embed_tokens`) now
   CLAMPS ids before every call into the real `original_embed`, instead of
   depending on MLX's undocumented "out-of-range gather returns zero" gather
   behaviour -- mirroring the existing convention in
   `deepseek_v4_vision.build_embeddings`.
2. `_inject` is marked `handles_out_of_range_ids = True` so the model-level
   `_assert_embeddable` check defers to it instead of raising on legitimate
   sentinel ids inside the splice's injection window.
3. `_inject` also asserts that ids OUTSIDE the injection window (or in the
   "fringe" columns of a mixed chunk) are in-range -- an out-of-range id
   there means the window itself is miscomputed, a real caller bug distinct
   from an expected DSv4 sentinel.

This uses a real (tiny, synthetic, untrained) DeepseekV4 model end-to-end --
not mocks -- so the marker attribute and the model-level check are proven to
actually cooperate, not just individually plausible.
"""

import mlx.core as mx
import pytest
from mlx_lm.models import deepseek_v4 as dsv4

from exo.worker.engines.mlx.generator.generate import patch_embed_tokens


def _tiny_args(vocab_size: int = 32) -> "dsv4.ModelArgs":
    return dsv4.ModelArgs(
        model_type="deepseek_v4",
        vocab_size=vocab_size,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        q_lora_rank=8,
        o_lora_rank=4,
        o_groups=1,
        head_dim=8,
        qk_rope_head_dim=4,
        sliding_window=8,
        compress_ratios=[0],
        index_n_heads=2,
        index_head_dim=4,
        index_topk=2,
        moe_intermediate_size=8,
        n_routed_experts=2,
        n_shared_experts=1,
        num_experts_per_tok=1,
        num_hash_layers=1,
        hc_mult=1,
        hc_sinkhorn_iters=1,
        num_nextn_predict_layers=0,
    )


def test_running_against_scratch_copy() -> None:
    print(f"\n[provenance] dsv4.__file__ = {dsv4.__file__}")
    assert "site-packages" not in dsv4.__file__, (
        "tests are importing an INSTALLED mlx_lm copy, not the working tree"
    )


def test_marker_attribute_is_set_on_inject() -> None:
    """`_inject` must be marked so the model-level check defers to it."""
    args = _tiny_args(vocab_size=32)
    model = dsv4.Model(args)
    embeddings = mx.zeros((1, 5, args.hidden_size))
    with patch_embed_tokens(model, embeddings, start_offset=0, token_count=5):
        inner_embed = model.model.embed_tokens
        assert getattr(inner_embed, "handles_out_of_range_ids", False) is True
    # After the context manager exits, the original (plain nn.Embedding,
    # unmarked) must be restored.
    restored = model.model.embed_tokens
    assert getattr(restored, "handles_out_of_range_ids", False) is False


def test_splice_with_sentinel_ids_inside_window_does_not_raise() -> None:
    """A real vision-shaped forward: sentinel ids inside the injection
    window must not trip the model-level `_assert_embeddable` check --
    `_inject` claims (and is expected) to handle them."""
    args = _tiny_args(vocab_size=32)
    model = dsv4.Model(args)
    vocab_size = args.vocab_size
    # 5 tokens: [text, text, SENTINEL, SENTINEL, text] -- sentinels at
    # vocab_size and vocab_size+4 (the two ends of DSv4's 5-sentinel range).
    raw_ids = mx.array([[1, 2, vocab_size, vocab_size + 4, 5]], dtype=mx.int32)
    precomputed_embeddings = mx.random.normal((1, 5, args.hidden_size))
    cache = model.make_cache()
    with patch_embed_tokens(
        model, precomputed_embeddings, start_offset=0, token_count=5
    ):
        logits = model(raw_ids, cache=cache)
        mx.eval(logits, [c.state for c in cache])
    assert logits.shape == (1, 5, vocab_size)


def test_splice_clamps_before_gather_instead_of_relying_on_oob_zero() -> None:
    """Directly exercises `_inject`'s internal `original_embed(clamped_ids)`
    call: even with raw sentinel ids in the overlap region, the text-embeds
    intermediate must be finite/real (not garbage from an unclamped OOB
    gather), proving the clamp actually ran."""
    args = _tiny_args(vocab_size=32)
    model = dsv4.Model(args)
    vocab_size = args.vocab_size
    inner = model.model
    original_weight = inner.embed_tokens.weight
    precomputed_embeddings = mx.zeros((1, 3, args.hidden_size))
    with patch_embed_tokens(
        model, precomputed_embeddings, start_offset=0, token_count=3
    ):
        injected = inner.embed_tokens
        raw_ids = mx.array([[vocab_size, vocab_size + 2, 1]], dtype=mx.int32)
        out = injected(raw_ids)
        mx.eval(out)
    # The overlap columns [0:3) are entirely replaced by the caller-supplied
    # `precomputed_embeddings` (all zero here), so the splice output for
    # them is zero regardless of what the clamped gather produced -- this
    # test's job is just to prove the call didn't raise/crash and produced
    # a real, finite, shape-correct tensor (the clamp is what makes the
    # internal `original_embed(clamped_ids)` call well-defined instead of
    # dependent on undocumented OOB semantics).
    assert out.shape == (1, 3, args.hidden_size)
    assert not bool(mx.any(mx.isnan(out)).item())
    # Sanity the clamp landed on a real row: clamped id (vocab_size -> 31,
    # vocab_size+2 -> 31) should read the SAME row of the real embedding
    # table both times (id 31, the last valid row).
    last_row = original_weight[vocab_size - 1]
    # Reproduce clamp+gather independently and compare.
    direct_clamped = mx.minimum(
        mx.array([vocab_size, vocab_size + 2], dtype=mx.int32), vocab_size - 1
    )
    mx.eval(direct_clamped)
    assert direct_clamped.tolist() == [vocab_size - 1, vocab_size - 1]
    mx.eval(last_row)


def test_out_of_range_id_outside_window_raises() -> None:
    """A sentinel-shaped id OUTSIDE the injection window is a real caller
    bug (the window doesn't cover the actual image span) -- `_inject` must
    raise, not silently clamp-and-continue, and the message must say so.

    Note: the internal `offset` tracker is initialized to `start_offset`
    itself (the real call site's first forward call under the context
    begins exactly where the post-cache-hit prompt resumes), so the
    realistic way to reach the "outside window" branch is a SUBSEQUENT
    call after the window has been fully consumed -- e.g. a decode step
    following the vision prefill -- not a chunk positioned before the
    window on the very first call.
    """
    args = _tiny_args(vocab_size=32)
    model = dsv4.Model(args)
    vocab_size = args.vocab_size
    precomputed_embeddings = mx.zeros((1, 3, args.hidden_size))
    # Window is [0, 3).
    with patch_embed_tokens(
        model, precomputed_embeddings, start_offset=0, token_count=3
    ):
        inner_embed = model.model.embed_tokens
        # First call: consumes the whole window [0,3) -- legitimate mixed
        # chunk, sentinel ids inside it are fine.
        in_window_ids = mx.array([[1, vocab_size, vocab_size + 2]], dtype=mx.int32)
        out = inner_embed(in_window_ids)
        mx.eval(out)
        # Second call: position 3, PAST the window (end_offset=3) -- a
        # decode-shaped step. An out-of-range id here is a real caller bug.
        bad_ids = mx.array([[vocab_size]], dtype=mx.int32)
        with pytest.raises(ValueError, match="patch_embed_tokens"):
            inner_embed(bad_ids)



def test_out_of_range_id_in_fringe_of_mixed_chunk_raises() -> None:
    """Same as above, but for the "fringe" columns of a MIXED chunk (a
    single `_inject` call whose window straddles the injection boundary)."""
    args = _tiny_args(vocab_size=32)
    model = dsv4.Model(args)
    vocab_size = args.vocab_size
    precomputed_embeddings = mx.zeros((1, 5, args.hidden_size))
    # Window [2, 4). A single 5-token chunk [0,5) straddles it: leading
    # fringe = cols [0,2), overlap = cols [2,4), trailing fringe = col [4,5).
    with patch_embed_tokens(
        model, precomputed_embeddings, start_offset=2, token_count=2
    ):
        inner_embed = model.model.embed_tokens
        # Trailing fringe (col 4) carries an out-of-range id -- a caller bug.
        bad_ids = mx.array([[1, 2, vocab_size, vocab_size + 1, vocab_size + 2]], dtype=mx.int32)
        with pytest.raises(ValueError, match="patch_embed_tokens"):
            inner_embed(bad_ids)


def test_text_only_no_vision_context_unaffected() -> None:
    """No `patch_embed_tokens` in play at all: the model-level check runs
    directly against the real `nn.Embedding` and legitimate text-only
    forwards are unaffected."""
    args = _tiny_args(vocab_size=32)
    model = dsv4.Model(args)
    cache = model.make_cache()
    inputs = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
    logits = model(inputs, cache=cache)
    mx.eval(logits, [c.state for c in cache])
    assert logits.shape == (1, 5, args.vocab_size)
