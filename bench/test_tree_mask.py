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
    _tree_pmask,
    _dispatch_pmask,
    _set_tree_verify_ctx,
    DeepseekV4RoPE,
)
from mlx_lm.models.cache import PoolingCache


class _FakeCache:
    """Stand-in for RotatingKVCache in the standalone microbench. Mirrors
    the make_mask + offset + max_size surface that
    _build_tree_mask_and_positions reads."""
    def __init__(self, offset: int, max_size: int) -> None:
        self.offset = offset
        self.max_size = max_size

    def make_mask(self, N: int, window_size=None, return_array: bool = False):
        # Mirrors RotatingKVCache.make_mask for N>1.
        from mlx_lm.models.base import create_causal_mask
        ws = window_size or self.max_size
        offset = min(self.max_size - 1, self.offset)
        return create_causal_mask(N, offset, window_size=ws)


def test_tree_mask_structure() -> None:
    """K=2 gamma=2 tree:
        0 = root (depth 0)
        1, 2 = depth-1 children of root  (call them 'a', 'b')
        3, 4 = depth-2 children of node 1 (a_x, a_y)
        5, 6 = depth-2 children of node 2 (b_x, b_y)
    """
    parent = [-1, 0, 0, 1, 1, 2, 2]
    depth = [0, 1, 1, 2, 2, 2, 2]
    # Use a cache whose max_size > offset so kv_window = offset (no clamp).
    l_kv = 100
    cache = _FakeCache(offset=l_kv, max_size=1024)

    mask, positions = _build_tree_mask_and_positions(parent, depth, cache)

    # Shape: at offset=100, max_size=1024, kv-axis = offset + L_q = 107.
    assert mask.shape == (7, l_kv + 7), f"bad mask shape {mask.shape}"
    assert positions.shape == (7,), f"bad pos shape {positions.shape}"

    # Positions: depth-d node gets position offset + d (the REAL offset).
    expected_pos = [l_kv + d for d in depth]
    actual_pos = positions.tolist()
    assert actual_pos == expected_pos, (
        f"positions mismatch: {actual_pos} vs {expected_pos}"
    )

    # Mask check: prefix (l_kv columns) attend (0.0). Same-depth siblings
    # all see the entire prefill block; create_causal_mask emits True for
    # rinds < offset+row_depth, which for the lowest row covers cols 0..offset+0,
    # for higher rows covers cols 0..offset+row -- ALL include 0..offset.
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

    # Tree mask. Use a fake cache with offset=L_kv, max_size large so we
    # get a kv_window = L_kv mask (matching the non-sliding path).
    parent = [-1, 0, 0, 1, 1, 2, 2]
    depth = [0, 1, 1, 2, 2, 2, 2]
    cache = _FakeCache(offset=L_kv, max_size=1024)
    mask_2d, _positions = _build_tree_mask_and_positions(parent, depth, cache)
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


def test_tree_mask_sliding_window_clamp() -> None:
    """Mask must clamp KV-axis to sliding_window when offset > sliding_window.

    Reproduces the 2026-05-19 cluster crash setup: offset=69321 (100K-ish
    context, well past the 128 sliding window). The tree mask MUST come
    out (L_q, sliding_window - 1 + L_q) -- NOT (L_q, offset + L_q) --
    or else the DSv4 SparseCompressedAttention local-attention SDPA will
    fail to broadcast against the (1, 64, L_q, ~134) scores tensor.
    """
    parent = [-1, 0, 0, 1, 1, 2, 2]
    depth = [0, 1, 1, 2, 2, 2, 2]
    cache = _FakeCache(offset=69321, max_size=128)  # production sliding window

    mask, positions = _build_tree_mask_and_positions(parent, depth, cache)

    # Expected kv-axis width: (max_size - 1) + L_q = 127 + 7 = 134.
    # (RotatingKVCache uses `offset = min(max_size - 1, self.offset)` then
    # create_causal_mask(N, offset, ws) emits shape (N, offset+N) bools.)
    expected_kv = (cache.max_size - 1) + 7
    assert mask.shape == (7, expected_kv), (
        f"sliding-window mask shape: expected (7,{expected_kv}) "
        f"got {mask.shape}"
    )

    # Positions are NOT clamped (RoPE is independent of sliding window).
    expected_pos = [69321 + d for d in depth]
    assert positions.tolist() == expected_pos, (
        f"positions clamped? {positions.tolist()} vs {expected_pos}"
    )

    # Last L_q columns: tree sub-mask.
    sub = mask[:, -7:].astype(mx.float32)
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
                assert v == 0.0, f"tail sub-mask node {i}->{j}: expected 0, got {v}"
            else:
                assert v < -1e8, f"tail sub-mask node {i}->{j}: expected -inf, got {v}"
    print("PASS test_tree_mask_sliding_window_clamp (offset=69321 sw=128)")


class _FakeCache4D(_FakeCache):
    """Emits a 4-D mask like BatchRotatingKVCache.make_mask does."""
    def make_mask(self, N: int, window_size=None, return_array: bool = False):
        m = super().make_mask(N, window_size=window_size, return_array=return_array)
        if m is None:
            return None
        # (L_q, kv) -> (1, 1, L_q, kv) to mimic BatchRotatingKVCache.
        return m[None, None]


def test_tree_pmask_same_depth_siblings_identical() -> None:
    """Tree-aware pmask: same-depth siblings get IDENTICAL rows.

    The stock PoolingCache.make_mask is row-causal-by-row-index. For tree
    input that wrongly differentiates siblings sharing a depth. _tree_pmask
    fixes this by using per-token positions.
    """
    # Set up a PoolingCache with ratio=4 ("overlap" layer; same default the
    # 2026-05-20 bisect implicates) and pre-populate the pool with 8 entries
    # so make_mask returns non-None.
    ratio = 4
    pool_cache = PoolingCache(ratio=ratio)
    # Pre-populate pool via update_and_fetch. Shape (B=1, P, D=16).
    fake_pool = mx.zeros((1, 8, 16))
    pool_cache.update_and_fetch(fake_pool)
    assert pool_cache.pooled is not None
    assert pool_cache.pooled.shape[1] == 8

    # Tree positions for K=2 g=2 at offset 27 (matching the 6A.2 diagnostic).
    # depth=[0,1,1,2,2,2,2] -> positions=[27,28,28,29,29,29,29].
    positions = mx.array([27, 28, 28, 29, 29, 29, 29], dtype=mx.int32)

    pmask = _tree_pmask(pool_cache, positions)
    assert pmask is not None
    assert pmask.shape == (7, 8), f"bad pmask shape {pmask.shape}"

    # Rows 1 and 2 share position 28 -> rows MUST be identical.
    diff_12 = mx.max(mx.abs(
        pmask[1].astype(mx.int32) - pmask[2].astype(mx.int32)
    )).item()
    assert diff_12 == 0, f"siblings at depth 1 differ in pmask: {pmask[1].tolist()} vs {pmask[2].tolist()}"

    # Rows 3,4,5,6 share position 29 -> all four rows MUST be identical.
    for i in (4, 5, 6):
        diff = mx.max(mx.abs(
            pmask[3].astype(mx.int32) - pmask[i].astype(mx.int32)
        )).item()
        assert diff == 0, f"depth-2 sibling row {i} differs from row 3"

    # Sanity: per-row cutoff matches (positions + 1) // ratio.
    # ratio=4 -> cutoffs [28,29,29,30,30,30,30] // 4 = [7,7,7,7,7,7,7].
    # So every row should attend to pool indices [0..6] and NOT to 7.
    expected_attend_count_per_row = 7  # cols 0..6
    for i in range(7):
        attended = int(pmask[i].astype(mx.int32).sum().item())
        assert attended == expected_attend_count_per_row, (
            f"row {i}: expected {expected_attend_count_per_row} attended, "
            f"got {attended}; pmask[{i}]={pmask[i].tolist()}"
        )

    print("PASS test_tree_pmask_same_depth_siblings_identical")


def test_tree_pmask_matches_make_mask_for_linear_path() -> None:
    """For LINEAR positions (no sibling sharing), tree_pmask == make_mask.

    Linear-causal positions for L=4 starting at offset=10 are [10,11,12,13].
    The tree-aware pmask built from those positions must equal what
    make_mask(L=4, offset=10) returns (the stock row-causal path).
    """
    ratio = 4
    pool_cache = PoolingCache(ratio=ratio)
    pool_cache.update_and_fetch(mx.zeros((1, 12, 8)))

    L, offset = 4, 10
    stock = pool_cache.make_mask(L=L, offset=offset)
    assert stock is not None, "make_mask returned None unexpectedly"

    linear_positions = mx.array([10, 11, 12, 13], dtype=mx.int32)
    tree = _tree_pmask(pool_cache, linear_positions)
    assert tree is not None

    diff = mx.max(mx.abs(
        stock.astype(mx.int32) - tree.astype(mx.int32)
    )).item()
    assert diff == 0, (
        f"tree_pmask diverges from make_mask on linear input:\n"
        f"  stock={stock.tolist()}\n  tree ={tree.tolist()}"
    )
    print("PASS test_tree_pmask_matches_make_mask_for_linear_path")


def test_dispatch_pmask_falls_through_when_ctx_unset() -> None:
    """_dispatch_pmask must call stock make_mask when the side channel is unset.

    Critical correctness invariant: production cluster restarts WITHOUT the
    tree side channel must produce bit-exact linear behavior. This guards
    against regressions where _dispatch_pmask accidentally still routes to
    _tree_pmask when EXO_DSV4_TREE_DRAFT is off.
    """
    ratio = 4
    pool_cache = PoolingCache(ratio=ratio)
    pool_cache.update_and_fetch(mx.zeros((1, 12, 8)))

    # Ensure the side channel is cleared (defensive — other tests may have
    # set it; they should clear at the end but be belt-and-braces here).
    _set_tree_verify_ctx(None, None)
    fallthrough = _dispatch_pmask(pool_cache, L=4, offset=10)
    stock = pool_cache.make_mask(L=4, offset=10)
    assert fallthrough is not None and stock is not None
    diff = mx.max(mx.abs(
        fallthrough.astype(mx.int32) - stock.astype(mx.int32)
    )).item()
    assert diff == 0, "dispatch_pmask should equal stock make_mask when ctx is None"

    # Sanity: empty pool_cache returns None from both.
    empty = PoolingCache(ratio=ratio)
    assert _dispatch_pmask(empty, L=4, offset=10) is None

    print("PASS test_dispatch_pmask_falls_through_when_ctx_unset")


def test_dispatch_pmask_uses_tree_pmask_when_ctx_set() -> None:
    """_dispatch_pmask must route to _tree_pmask when positions match L_q.

    Verifies the L-vs-positions-length gate: if L doesn't match, we still
    fall through to make_mask (defensive — wrong-shape positions in the
    side channel must NOT crash the linear path).
    """
    ratio = 4
    pool_cache = PoolingCache(ratio=ratio)
    pool_cache.update_and_fetch(mx.zeros((1, 12, 8)))

    positions = mx.array([100, 101, 101, 102, 102, 102, 102], dtype=mx.int32)
    _set_tree_verify_ctx(mask=mx.zeros((7, 1)), positions=positions)

    try:
        # Matched length: should return tree-pmask.
        got_tree = _dispatch_pmask(pool_cache, L=7, offset=99)
        ref_tree = _tree_pmask(pool_cache, positions)
        assert got_tree is not None and ref_tree is not None
        diff = mx.max(mx.abs(
            got_tree.astype(mx.int32) - ref_tree.astype(mx.int32)
        )).item()
        assert diff == 0, "dispatch_pmask did not route to _tree_pmask when L matched"

        # Mismatched length: should fall through to make_mask (so that a
        # stray prefill / cache-read with the side channel still set won't
        # crash; not expected in practice — dsv4_mtp clears the channel in
        # a finally — but cheap insurance).
        got_fallthrough = _dispatch_pmask(pool_cache, L=4, offset=10)
        ref_fallthrough = pool_cache.make_mask(L=4, offset=10)
        assert got_fallthrough is not None and ref_fallthrough is not None
        diff = mx.max(mx.abs(
            got_fallthrough.astype(mx.int32) - ref_fallthrough.astype(mx.int32)
        )).item()
        assert diff == 0, "dispatch_pmask did not fall through when L != L_q"
    finally:
        # Always clear so subsequent tests are unaffected.
        _set_tree_verify_ctx(None, None)

    print("PASS test_dispatch_pmask_uses_tree_pmask_when_ctx_set")


def test_tree_mask_4d_base_input() -> None:
    """Reproduces the 2026-05-19 production crash: BatchRotatingKVCache
    emits a 4-D base mask via mx.expand_dims, _build_tree_mask must
    squeeze it back to 2-D before splicing in the tree sub-mask.
    """
    parent = [-1, 0, 0, 1, 1, 2, 2]
    depth = [0, 1, 1, 2, 2, 2, 2]
    cache = _FakeCache4D(offset=69321, max_size=128)

    mask, positions = _build_tree_mask_and_positions(parent, depth, cache)

    # Should land back at 2-D, kv-axis clamped to sliding_window.
    expected_kv = (cache.max_size - 1) + 7
    assert mask.shape == (7, expected_kv), (
        f"4D-input mask shape: expected (7,{expected_kv}) got {mask.shape}"
    )
    # Positions still un-clamped.
    assert positions.tolist() == [69321 + d for d in depth]
    print("PASS test_tree_mask_4d_base_input (4-D base mask squeezes correctly)")


if __name__ == "__main__":
    print("Phase 5.2 mask correctness test")
    print("================================")
    test_tree_mask_structure()
    test_rope_per_token_positions()
    test_same_depth_siblings_share_rope()
    test_sdpa_with_tree_mask()
    test_tree_mask_sliding_window_clamp()
    test_tree_mask_4d_base_input()
    test_tree_pmask_same_depth_siblings_identical()
    test_tree_pmask_matches_make_mask_for_linear_path()
    test_dispatch_pmask_falls_through_when_ctx_unset()
    test_dispatch_pmask_uses_tree_pmask_when_ctx_set()
    print("\nAll Phase 5.2 + tree-pmask tests PASSED")
