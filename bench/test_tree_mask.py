#!/usr/bin/env python3
"""Phase 5.2 mask correctness test for token-tree drafting.

Validates two things WITHOUT loading the full DSv4-Flash model:

1. _build_tree_mask_and_positions produces the right (parent_idx, depth) ->
   (mask, positions) mapping. Sub-mask has 1s on diagonal + ancestor entries.

2. _rope_with_positions(x, positions) and the equivalent batch-axis trick
   round-trip correctly: applying rope with positions=[L_kv, L_kv+1] to a
   (1, H, 2, D) tensor should match independently applying rope at each
   single-token position.

3. Tree-mask SDPA output at each tree node matches the equivalent linear
   forward output for the path leading to that node. Uses RANDOM weights
   so we don't need to load DSv4. This is a structural test, not a quality
   test.

Run from the laptop (no cluster needed). Pass = mask + RoPE + SDPA are
correctly wired so we won't burn cluster time on a structurally-broken
config.
"""
from __future__ import annotations

import sys
import mlx.core as mx

from exo.worker.engines.mlx.speculative.dsv4_mtp import (
    _build_tree_mask_and_positions,
)
from mlx_lm.models.deepseek_v4 import (
    _rope_with_positions,
    DeepseekV4RoPE,
)


def test_tree_mask_structure() -> None:
    """K=2 gamma=2 tree:
        0 = root (depth 0)
        1, 2 = depth-1 children of root  (call them 'a', 'b')
        3, 4 = depth-2 children of node 1 (a_x, a_y)
        5, 6 = depth-2 children of node 2 (b_x, b_y)
    """
    parent = [-1, 0, 0, 1, 1, 2, 2]
    depth = [0, 1, 1, 2, 2, 2, 2]
    l_kv = 100

    mask, positions = _build_tree_mask_and_positions(parent, depth, l_kv)

    # Shape checks
    assert mask.shape == (7, l_kv + 7), f"bad mask shape {mask.shape}"
    assert positions.shape == (7,), f"bad pos shape {positions.shape}"

    # Positions check: node-depth maps to l_kv + depth.
    expected_pos = [l_kv + d for d in depth]
    actual_pos = positions.tolist()
    assert actual_pos == expected_pos, (
        f"positions mismatch: {actual_pos} vs {expected_pos}"
    )

    # Mask check: prefix (l_kv columns) is all 0 (attend-all).
    prefix = mask[:, :l_kv]
    assert mx.all(prefix == 0).item(), "prefix columns must be 0 (attend-all)"

    # Sub-mask: per-tree-node ancestor check.
    sub = mask[:, l_kv:].astype(mx.float32)  # (7, 7)
    expected_attend = {
        0: {0},
        1: {0, 1},
        2: {0, 2},
        3: {0, 1, 3},
        4: {0, 1, 4},
        5: {0, 2, 5},
        6: {0, 2, 6},
    }
    for i in range(7):
        for j in range(7):
            v = float(sub[i, j].item())
            should_attend = j in expected_attend[i]
            if should_attend:
                assert v == 0.0, (
                    f"node {i} should attend to {j} but mask = {v}"
                )
            else:
                assert v < -1e8, (
                    f"node {i} should NOT attend to {j} but mask = {v}"
                )
    print("PASS test_tree_mask_structure (7 nodes, K=2 g=2)")


def test_rope_per_token_positions() -> None:
    """RoPE with per-token positions matches per-token RoPE at scalar offsets."""
    # Small test setup: 1 batch, 4 heads, 5 tokens, 16 head_dim.
    B, H, L, D = 1, 4, 5, 16
    rope = DeepseekV4RoPE(
        dims=D,
        base=10000.0,
        scaling_config=None,
        max_position_embeddings=1024,
        freq_scale=1,
    )

    # Random Q tensor.
    mx.random.seed(7749)
    x = mx.random.normal((B, H, L, D))

    # Apply RoPE with per-token positions [100, 100, 101, 101, 102].
    positions = mx.array([100, 100, 101, 101, 102], dtype=mx.int32)
    out_tree = _rope_with_positions(rope, x, positions, inverse=False)

    # Compare against per-token rope: rotate token i at scalar offset positions[i].
    # The standard rope expects (B, H, L, D); we can slice each L=1 and call.
    out_ref = mx.zeros_like(x)
    for i in range(L):
        slc = x[:, :, i:i + 1, :]   # (1, H, 1, D)
        rotated = rope(slc, int(positions[i].item()))
        # Manually assemble out_ref.
        out_ref = mx.concatenate(
            [out_ref[:, :, :i, :], rotated, out_ref[:, :, i + 1:, :]],
            axis=2,
        )

    diff = mx.max(mx.abs(out_tree - out_ref)).item()
    print(f"  RoPE per-token vs scalar-loop max abs diff: {diff:.2e}")
    assert diff < 1e-4, f"RoPE per-token mismatch (max diff {diff})"
    print("PASS test_rope_per_token_positions")


def test_same_depth_siblings_share_rope() -> None:
    """Two same-depth tree nodes must rotate identically under RoPE."""
    B, H, L, D = 1, 4, 2, 16
    rope = DeepseekV4RoPE(
        dims=D, base=10000.0, scaling_config=None,
        max_position_embeddings=1024, freq_scale=1,
    )
    mx.random.seed(42)
    x = mx.random.normal((B, H, L, D))
    # Two siblings at the same depth d => same position.
    positions = mx.array([200, 200], dtype=mx.int32)
    out = _rope_with_positions(rope, x, positions)
    # If positions match, the rotation matrices are identical → out[:,:,0]
    # and out[:,:,1] should be the rope-rotation of x[:,:,0] and x[:,:,1]
    # respectively (NOT equal to each other since x is random, but the
    # rotation applied is the same. Apply rope independently and check.
    out_0 = rope(x[:, :, 0:1, :], 200)
    out_1 = rope(x[:, :, 1:2, :], 200)
    diff_0 = mx.max(mx.abs(out[:, :, 0:1, :] - out_0)).item()
    diff_1 = mx.max(mx.abs(out[:, :, 1:2, :] - out_1)).item()
    print(f"  sibling 0 diff: {diff_0:.2e}, sibling 1 diff: {diff_1:.2e}")
    assert diff_0 < 1e-4 and diff_1 < 1e-4
    print("PASS test_same_depth_siblings_share_rope")


def test_sdpa_with_tree_mask() -> None:
    """Tree-mask SDPA at a leaf node should equal linear-mask SDPA for that path.

    We construct a tiny attention (1 head, 4 dim, no real model) with random
    K/V and run SDPA twice:
      1. (a) Single linear path: query at root + a + a_x positions, KV
         at the same. Standard causal mask.
      2. (b) Tree query at the 7 tree nodes; tree mask. Pluck out the
         a_x logit (node 3).
    The two outputs at node 3 should match within fp32 noise.
    """
    # Tree: 0=root, 1=a, 2=b, 3=a_x, 4=a_y, 5=b_x, 6=b_y
    H, D = 2, 8
    L_kv = 5  # 5 prefill tokens
    n_nodes = 7

    mx.random.seed(0)
    # Fake K/V buffer of prefill + tree-token positions.
    # Linear test (a): KV at positions 0..L_kv (prefill) + L_kv (root) +
    # L_kv+1 (node-a) + L_kv+2 (node-a_x). 3 tree positions out of 7.
    # We just need each tree node's Q/K/V to differ.

    # Build a fake (1, H, L_kv + n_nodes, D) K/V tensor.
    kv_total = mx.random.normal((1, H, L_kv + n_nodes, D))

    # Q for each tree node. Shape (1, H, n_nodes, D).
    q_tree = mx.random.normal((1, H, n_nodes, D))

    # Tree mask.
    parent = [-1, 0, 0, 1, 1, 2, 2]
    depth = [0, 1, 1, 2, 2, 2, 2]
    mask_2d, _positions = _build_tree_mask_and_positions(parent, depth, L_kv)
    # SDPA expects (B, n_heads, T_q, T_kv) broadcastable.
    mask_tree = mask_2d[None, None, :, :].astype(q_tree.dtype)

    out_tree = mx.fast.scaled_dot_product_attention(
        q_tree, kv_total, kv_total,
        scale=1.0 / (D ** 0.5),
        mask=mask_tree,
    )

    # Linear test: pluck the a_x path = root (node 0), a (node 1), a_x (node 3).
    # Q at those 3 positions = same q_tree rows. KV is prefill + tokens at
    # tree positions L_kv (root), L_kv+1 (a), L_kv+3 (a_x). Standard causal.
    path_qs = mx.stack([q_tree[0, :, 0], q_tree[0, :, 1], q_tree[0, :, 3]], axis=1)
    path_qs = path_qs[None]  # (1, H, 3, D)
    # KV: prefill (L_kv tokens) + (root_kv, a_kv, a_x_kv).
    kv_linear_q = mx.concatenate(
        [
            kv_total[:, :, :L_kv, :],
            kv_total[:, :, L_kv:L_kv + 1, :],   # root
            kv_total[:, :, L_kv + 1:L_kv + 2, :],  # a (node 1)
            kv_total[:, :, L_kv + 3:L_kv + 4, :],  # a_x (node 3)
        ],
        axis=2,
    )

    out_linear = mx.fast.scaled_dot_product_attention(
        path_qs, kv_linear_q, kv_linear_q,
        scale=1.0 / (D ** 0.5),
        mask="causal",
    )

    # Compare a_x output: out_tree[:, :, 3, :] vs out_linear[:, :, 2, :].
    diff = mx.max(mx.abs(out_tree[:, :, 3, :] - out_linear[:, :, 2, :])).item()
    print(f"  Tree-mask vs linear-causal SDPA, a_x logit max abs diff: {diff:.2e}")
    assert diff < 1e-3, f"tree-mask SDPA mismatch at leaf a_x (diff {diff})"
    print("PASS test_sdpa_with_tree_mask")


if __name__ == "__main__":
    print("Phase 5.2 mask correctness test")
    print("================================")
    test_tree_mask_structure()
    test_rope_per_token_positions()
    test_same_depth_siblings_share_rope()
    test_sdpa_with_tree_mask()
    print("\nAll Phase 5.2 tests PASSED")
