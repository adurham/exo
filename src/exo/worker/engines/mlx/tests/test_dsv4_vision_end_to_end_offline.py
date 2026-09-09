# pyright: reportAny=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownParameterType=false, reportMissingParameterType=false
# pyright: reportUntypedFunctionDecorator=false, reportPrivateUsage=false
# pyright: reportCallIssue=false, reportArgumentType=false
# pyright: reportMissingImports=false, reportAttributeAccessIssue=false
# pyright: reportIndexIssue=false, reportOptionalMemberAccess=false
# pyright: reportUnknownLambdaType=false
"""Phase 4 acceptance: offline end-to-end vision pipeline, no cluster involved.

Runs a REAL image all the way through:

    image bytes
      -> Phase 2 processor (load_image, build_image_block)
      -> placeholder expansion into sentinel token ids
      -> Phase 1 ViT + Aligner (MLX)
      -> merge into the token embedding stream
      -> Phase 4d chunk plan

and asserts shapes, token counts and merge placement against the reference's
own arithmetic. Runs on CPU/one machine with a randomly-initialised small
model -- it exercises the PLUMBING, not checkpoint numerics (Phase 1 already
established parity vs torch to 4.29e-06).

Deliberately touches no cluster, no network, and no 167 GB checkpoint.
"""

import io
import sys

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, "/tmp/dsv4-phase4-scratch/mlx-lm")

from mlx_lm.models import deepseek_v4 as dsv4  # noqa: E402

from exo.worker.engines.mlx.prefill_chunking import (  # noqa: E402
    ImageSpan,
    plan_prefill_chunks,
)
from exo.worker.engines.mlx.vendor.deepseek_v4_image_processor import (  # noqa: E402
    IMAGE,
    IMAGE_END,
    IMAGE_NEW_LINE,
    IMAGE_PAD,
    IMAGE_START,
    VisionEncoderConfig,
    expand_image_placeholders,
)

# Mirrors the real checkpoint's vision geometry (config.json), at a hidden size
# small enough to build and run in a couple of seconds.
PATCH_SIZE = 14
DOWNSAMPLE = 3
MAX_TOKENS = 384
MIN_PIXELS = 147456
MAX_WH_RATIO = 8
VOCAB = 2048
HIDDEN = 64
PLACEHOLDER_ID = 1999


@pytest.fixture(scope="module")
def encoder_config() -> VisionEncoderConfig:
    return VisionEncoderConfig(
        patch_size=PATCH_SIZE,
        downsample_ratio=DOWNSAMPLE,
        max_token_count=MAX_TOKENS,
        min_pixel_count=MIN_PIXELS,
        max_width_height_ratio=MAX_WH_RATIO,
        vocabulary_size=VOCAB,
    )


@pytest.fixture(scope="module")
def image_bytes() -> bytes:
    """A real, non-uniform 640x480 PNG (gradient + shapes, so not degenerate)."""
    rng = np.random.default_rng(20260909)
    height, width = 480, 640
    ys = np.linspace(0, 255, height, dtype=np.float32)[:, None]
    xs = np.linspace(0, 255, width, dtype=np.float32)[None, :]
    red = np.broadcast_to(ys, (height, width))
    green = np.broadcast_to(xs, (height, width))
    blue = (ys + xs) / 2.0
    pixels = np.stack([red, green, blue], axis=-1)
    pixels[100:200, 150:400] = 255.0
    pixels[300:400, 200:500] = 0.0
    pixels += rng.normal(0, 8, pixels.shape)
    image = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture(scope="module")
def model():
    config = dsv4.ModelArgs(
        model_type="deepseek_v4",
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        head_dim=32,
        q_lora_rank=32,
        o_lora_rank=32,
        o_groups=2,
        qk_rope_head_dim=16,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        num_hash_layers=1,
        sliding_window=128,
        compress_ratios=[0, 4],
        index_topk=8,
        index_n_heads=4,
        index_head_dim=16,
        vision_n_layers=2,
        vision_dim=32,
        vision_inter_dim=64,
        vision_n_heads=4,
        vision_patch_size=PATCH_SIZE,
        vision_downsample_ratio=DOWNSAMPLE,
        vision_max_n_token=MAX_TOKENS,
        vision_rope_theta=10000.0,
    )
    built = dsv4.Model(config)
    rng = np.random.default_rng(4242)

    def fill(tree):
        if isinstance(tree, dict):
            return {k: fill(v) for k, v in tree.items()}
        if isinstance(tree, list):
            return [fill(v) for v in tree]
        if isinstance(tree, mx.array):
            if tree.dtype == mx.int32:
                return mx.array(rng.integers(0, 8, size=tree.shape).astype(np.int32))
            return mx.array((rng.standard_normal(tree.shape) * 0.05).astype(np.float32))
        return tree

    built.update(fill(built.parameters()))
    return built


def test_end_to_end_vision_pipeline(encoder_config, image_bytes, model, capsys):
    from exo.worker.engines.mlx import deepseek_v4_vision as dsvision

    # ---------------------------------------------- 1. expand placeholders
    prompt_token_ids = [10, 11, 12, PLACEHOLDER_ID, 13, 14]
    records = [{"data": image_bytes}]
    expanded, image_inputs = expand_image_placeholders(
        prompt_token_ids, records, PLACEHOLDER_ID, encoder_config
    )

    assert len(image_inputs) == 1
    image = image_inputs[0]
    block_len = len(image.types)

    print("\n=== Phase 2: image -> patches -> sentinel block ===")
    print(f"source image      : 640x480 RGB PNG, {len(image_bytes)} bytes")
    print(f"patches           : {image.patches.shape} dtype={image.patches.dtype}")
    print(f"ViT grid          : n_vit_h={image.n_vit_h} n_vit_w={image.n_vit_w}")
    print(f"block start       : {image.start}")
    print(f"block length      : {block_len} tokens")
    print(f"prompt tokens     : {len(prompt_token_ids)} -> {len(expanded)}")

    # Placeholder replaced by exactly one block.
    assert len(expanded) == len(prompt_token_ids) - 1 + block_len
    assert PLACEHOLDER_ID not in expanded

    # Sentinel ids are vocab_size + type, deliberately out of table range.
    span_ids = np.asarray(expanded[image.start : image.start + block_len])
    assert (span_ids >= VOCAB).all(), "every span token must be a sentinel"
    assert span_ids[len(span_ids) - 1] == VOCAB + IMAGE_END
    types_seen = sorted({int(t) - VOCAB for t in span_ids})
    print(f"sentinel types    : {types_seen}")
    assert set(types_seen) <= {
        IMAGE_START,
        IMAGE_PAD,
        IMAGE,
        IMAGE_NEW_LINE,
        IMAGE_END,
    }

    # Text tokens outside the span are untouched.
    assert expanded[: image.start] == prompt_token_ids[: image.start]
    assert expanded[image.start + block_len :] == [13, 14]

    # -------------------------------------------- 2. ViT + Aligner encode
    encoded = model.encode_image(mx.array(image.patches), image.n_vit_h, image.n_vit_w)
    mx.eval(encoded)
    n_llm_h = -(-image.n_vit_h // DOWNSAMPLE)
    n_llm_w = -(-image.n_vit_w // DOWNSAMPLE)
    print("\n=== Phase 1: ViT + Aligner ===")
    print(f"aligner output    : {encoded.shape}")
    print(f"expected rows     : {n_llm_h} x {n_llm_w} = {n_llm_h * n_llm_w}")
    assert encoded.shape == (n_llm_h * n_llm_w, HIDDEN)

    n_image_slots = int((span_ids == VOCAB + IMAGE).sum())
    print(f"IMAGE slots       : {n_image_slots}")
    assert n_image_slots == encoded.shape[0] == len(image.perm)

    # ------------------------------------------------------- 3. merge
    embeddings = dsvision.build_embeddings(model, expanded, image_inputs, VOCAB)
    mx.eval(embeddings)
    array = np.asarray(embeddings, dtype=np.float32)

    print("\n=== Phase 4b: merged embedding stream ===")
    print(f"embeddings        : {array.shape} dtype={embeddings.dtype}")
    print(f"finite            : {bool(np.isfinite(array).all())}")
    print(f"abs max / mean    : {np.abs(array).max():.6f} / {array.mean():.6f}")

    assert array.shape == (1, len(expanded), HIDDEN)
    assert np.isfinite(array).all()

    # Text rows must equal a plain embedding lookup (untouched by the merge).
    text_ids = mx.array([expanded[0], expanded[1], expanded[2]])[None]
    text_reference = np.asarray(model.model.embed_tokens(text_ids), dtype=np.float32)
    assert np.array_equal(array[:, :3, :], text_reference), (
        "merge perturbed text rows outside the image span"
    )

    # IMAGE rows must equal the permuted aligner output, exactly.
    permuted = np.asarray(encoded[mx.array(image.perm)], dtype=np.float32)
    image_positions = np.nonzero(span_ids == VOCAB + IMAGE)[0] + image.start
    merged_image_rows = array[0, image_positions, :]
    max_diff = float(np.abs(merged_image_rows - permuted).max())
    print(f"IMAGE-row max diff vs aligner output: {max_diff:.3e}")
    assert max_diff == 0.0, "merged IMAGE rows must be the aligner output verbatim"

    # Sentinel rows must equal their parameter vectors.
    for type_value, name in (
        (IMAGE_START, "image_start"),
        (IMAGE_NEW_LINE, "image_newline"),
        (IMAGE_END, "image_end"),
        (IMAGE_PAD, "image_pad"),
    ):
        positions = np.nonzero(span_ids == VOCAB + type_value)[0]
        if positions.size == 0:
            continue
        expected = np.asarray(getattr(model, name), dtype=np.float32)
        rows = array[0, positions + image.start, :]
        diff = float(np.abs(rows - expected[None, :]).max())
        print(f"{name:14s} rows={positions.size:4d} max diff={diff:.3e}")
        assert diff == 0.0

    # ------------------------------------------------- 4. chunk schedule
    spans = [ImageSpan(image.start, image.start + block_len)]
    assert dsvision.image_span_bounds(image_inputs) == [
        (image.start, image.start + block_len)
    ]
    chunks = plan_prefill_chunks(
        total_tokens=len(expanded) - 1, prefill_step_size=1024, image_spans=spans
    )
    print("\n=== Phase 4d: chunk plan ===")
    print(f"span              : [{spans[0].start}, {spans[0].end})")
    print(f"chunks @1024      : {chunks}")
    assert spans[0].end <= chunks[0], "image span must be inside chunk 0"
    assert sum(chunks) == len(expanded) - 1

    # A long text prefix would push the image past a boundary; the planner
    # must stretch chunk 0 rather than let it land at a non-zero offset.
    long_prefix = 3000
    shifted = [ImageSpan(long_prefix, long_prefix + block_len)]
    long_chunks = plan_prefill_chunks(
        total_tokens=long_prefix + block_len + 500,
        prefill_step_size=1024,
        image_spans=shifted,
    )
    print(f"chunks w/ 3000-token prefix: {long_chunks}")
    assert long_chunks[0] == long_prefix + block_len

    # Forward pass with the merged embeddings actually runs.
    inner = model.model
    original_embed = inner.embed_tokens
    try:
        inner.embed_tokens = lambda _ids: embeddings
        logits = model(mx.array(expanded)[None], cache=model.make_cache())
        mx.eval(logits)
    finally:
        inner.embed_tokens = original_embed
    logits_array = np.asarray(logits, dtype=np.float32)
    print("\n=== forward pass with merged embeddings ===")
    print(f"logits            : {logits_array.shape}")
    print(f"finite            : {bool(np.isfinite(logits_array).all())}")
    print(f"abs max           : {np.abs(logits_array).max():.6f}")
    assert logits_array.shape == (1, len(expanded), VOCAB)
    assert np.isfinite(logits_array).all()

    with capsys.disabled():
        pass


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
