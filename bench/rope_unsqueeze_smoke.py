#!/usr/bin/env python3
"""Smoke test: verify the compressor rope unsqueeze/squeeze removal
produces identical output."""
import mlx.core as mx
from mlx_lm.models.deepseek_v4 import DeepseekV4RoPE


def main():
    rope = DeepseekV4RoPE(
        dims=64,
        base=160000.0,
        max_position_embeddings=1048576,
        freq_scale=4,
    )

    B, L, D = 1, 100, 128
    x = mx.random.normal(shape=(B, L, D)).astype(mx.bfloat16)
    offset = 0

    # Old form
    out_old = rope(x[:, None], offset=offset).squeeze(1)
    mx.eval(out_old)
    print(f"old shape: {out_old.shape}")

    # New form
    out_new = rope(x, offset=offset)
    mx.eval(out_new)
    print(f"new shape: {out_new.shape}")

    diff = float((out_old.astype(mx.float32) - out_new.astype(mx.float32)).abs().max())
    print(f"max abs diff: {diff}")
    assert diff == 0, f"NOT IDENTICAL: diff={diff}"
    print("IDENTICAL ✓")


if __name__ == "__main__":
    main()
