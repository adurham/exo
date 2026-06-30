# type: ignore
import time
from typing import cast
from unittest.mock import patch

import mlx.core as mx
import pytest
from mlx_lm.models.cache import CacheList, KVCache, PoolingCache, RotatingKVCache
from mlx_lm.sample_utils import make_sampler

from exo.shared.types.common import ModelId
from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from exo.worker.engines.mlx.cache import (
    CacheSnapshot,
    KVPrefixCache,
    _find_nearest_snapshot,
    _select_spaced_snapshots,
    cache_length,
    encode_prompt,
    get_prefix_length,
    make_kv_cache,
)
from exo.worker.engines.mlx.generator.generate import mlx_generate, prefill
from exo.worker.engines.mlx.types import Model
from exo.worker.engines.mlx.utils_mlx import apply_chat_template
from exo.worker.tests.unittests.test_mlx.conftest import (
    DEFAULT_GPT_OSS_CONFIG,
    DEFAULT_GPT_OSS_MODEL_ID,
)


def _check_model_exists() -> bool:
    return DEFAULT_GPT_OSS_CONFIG.model_path.exists()


class TestGetPrefixLength:
    def test_identical_arrays(self):
        a = mx.array([1, 2, 3, 4, 5])
        b = mx.array([1, 2, 3, 4, 5])
        assert get_prefix_length(a, b) == 5

    def test_no_common_prefix(self):
        a = mx.array([1, 2, 3])
        b = mx.array([4, 5, 6])
        assert get_prefix_length(a, b) == 0

    def test_partial_prefix(self):
        a = mx.array([1, 2, 3, 4, 5])
        b = mx.array([1, 2, 3, 7, 8])
        assert get_prefix_length(a, b) == 3

    def test_prompt_longer_than_cached(self):
        a = mx.array([1, 2, 3, 4, 5])
        b = mx.array([1, 2, 3])
        assert get_prefix_length(a, b) == 3

    def test_cached_longer_than_prompt(self):
        a = mx.array([1, 2, 3])
        b = mx.array([1, 2, 3, 4, 5])
        assert get_prefix_length(a, b) == 3

    def test_single_token_match(self):
        a = mx.array([1, 2, 3])
        b = mx.array([1, 5, 6])
        assert get_prefix_length(a, b) == 1

    def test_empty_prompt(self):
        a = mx.array([]).astype(mx.int32)
        b = mx.array([1, 2, 3])
        assert get_prefix_length(a, b) == 0

    def test_empty_cached(self):
        a = mx.array([1, 2, 3])
        b = mx.array([]).astype(mx.int32)
        assert get_prefix_length(a, b) == 0

    def test_both_empty(self):
        a = mx.array([]).astype(mx.int32)
        b = mx.array([]).astype(mx.int32)
        assert get_prefix_length(a, b) == 0


class TestKVPrefix:
    @pytest.fixture
    def mock_tokenizer(self):
        """Create a minimal mock tokenizer for tests that don't need real tokenization."""
        from unittest.mock import MagicMock

        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3]
        return tokenizer

    def test_starts_empty(self, mock_tokenizer):
        cache = KVPrefixCache(None)
        assert len(cache.prompts) == 0
        assert len(cache.caches) == 0

    def test_clear_empties_cache(self, mock_tokenizer):
        cache = KVPrefixCache(None)
        cache.add_kv_cache(mx.array([1, 2, 3]), [KVCache()])
        assert len(cache.prompts) == 1
        cache.clear()
        assert len(cache.prompts) == 0
        assert len(cache.caches) == 0

    def test_clear_on_empty_cache(self, mock_tokenizer):
        cache = KVPrefixCache(None)
        cache.clear()
        assert len(cache.prompts) == 0


def _fake_kv_cache(num_layers: int, num_tokens: int) -> list[KVCache]:
    """Build a KV cache of `num_layers` KVCache entries each holding a small
    but valid K/V tensor covering `num_tokens` tokens along the sequence axis.
    Shape [B=1, H=2, S=num_tokens, D=4].
    """
    caches: list[KVCache] = []
    for layer_idx in range(num_layers):
        c = KVCache()
        k = mx.arange(num_tokens, dtype=mx.float32)
        k = mx.broadcast_to(k.reshape(1, 1, num_tokens, 1), (1, 2, num_tokens, 4))
        # Layer-dependent offset so different layers aren't identical.
        k = k + float(layer_idx)
        v = k + 0.5
        c.keys = mx.array(k)
        c.values = mx.array(v)
        c.offset = num_tokens
        caches.append(c)
    return caches


class TestRadixTrieStorage:
    """Verify that storage is actually deduplicated across sessions sharing a prefix."""

    def test_two_sessions_share_prefix_node(self):
        cache = KVPrefixCache(None)
        # Two prompts sharing [1,2,3,4,5] then diverging.
        shared = [1, 2, 3, 4, 5]
        tokens_a = mx.array(shared + [10, 11], dtype=mx.int32)
        tokens_b = mx.array(shared + [20, 21], dtype=mx.int32)

        id_a = cache.add_kv_cache(tokens_a, _fake_kv_cache(num_layers=2, num_tokens=7))
        id_b = cache.add_kv_cache(tokens_b, _fake_kv_cache(num_layers=2, num_tokens=7))

        # Both leaves reachable under the shared node at depth 5.
        root = cache._root  # pyright: ignore[reportPrivateUsage]
        first = root.children.get(1)
        assert first is not None
        # The first child's edge should cover exactly the shared prefix, then
        # split into two children (10 and 20).
        assert first.depth == 5
        assert first.edge_length == 5
        assert set(first.children.keys()) == {10, 20}
        # Ref count == 2 leaves under the shared node.
        assert first.ref_count == 2
        assert id_a != id_b

    def test_prefix_hit_reuses_stored_kv(self):
        cache = KVPrefixCache(None)
        shared = [1, 2, 3, 4, 5]
        tokens_a = mx.array(shared + [10, 11], dtype=mx.int32)
        cache.add_kv_cache(tokens_a, _fake_kv_cache(num_layers=2, num_tokens=7))

        # Query with a longer prompt sharing the prefix.
        tokens_b = mx.array(shared + [20, 21, 22], dtype=mx.int32)
        # Model mock: _materialize_full_leaf_cache doesn't need the model, and
        # get_kv_cache constructs a fresh cache for remaining tokens via
        # make_kv_cache(model). Provide a stub with .layers that routes to
        # make_kv_cache's fallback branch.
        #
        # In practice, miss case is what matters here: verify the hit branch
        # returns the expected match depth and leaf id.
        from unittest.mock import MagicMock

        model = MagicMock()
        model.layers = [None, None]

        result_cache, remaining, matched_id, is_exact = cache.get_kv_cache(
            model, tokens_b
        )
        assert matched_id == 0
        assert int(remaining.shape[0]) == 3  # [20, 21, 22]
        assert is_exact is False
        # Materialized cache should have 5 tokens along the sequence axis.
        assert result_cache[0].offset == 5

    def test_eviction_of_one_leaf_frees_only_unique_branch(self):
        cache = KVPrefixCache(None, max_sessions=2)
        shared = [1, 2, 3, 4, 5]
        tokens_a = mx.array(shared + [10, 11], dtype=mx.int32)
        tokens_b = mx.array(shared + [20, 21], dtype=mx.int32)
        tokens_c = mx.array(shared + [30, 31], dtype=mx.int32)

        id_a = cache.add_kv_cache(tokens_a, _fake_kv_cache(num_layers=2, num_tokens=7))
        id_b = cache.add_kv_cache(tokens_b, _fake_kv_cache(num_layers=2, num_tokens=7))
        # Adding C forces eviction of A (the LRU non-pinned).
        id_c = cache.add_kv_cache(tokens_c, _fake_kv_cache(num_layers=2, num_tokens=7))

        assert id_a not in cache.prompts
        assert id_b in cache.prompts
        assert id_c in cache.prompts

        # Shared prefix node must still exist (referenced by B and C).
        root = cache._root  # pyright: ignore[reportPrivateUsage]
        first = root.children.get(1)
        assert first is not None
        assert first.ref_count == 2
        # A's unique branch (first token 10) is gone; B and C's branches remain.
        assert set(first.children.keys()) == {20, 30}

    def test_low_priority_leaf_evicted_before_interactive(self):
        """A background (low_priority) leaf is dropped before an interactive
        one, even when the interactive leaf is older (LRU would pick it).

        Mirrors the production scenario: a long-lived conversation (interactive)
        plus a one-shot background aux call (e.g. compression, tagged via a
        non-default service_tier) sharing a capped DSv4 instance. Plain LRU
        would evict the older interactive session; priority-aware eviction must
        drop the background leaf instead.
        """
        cache = KVPrefixCache(None, max_sessions=2)
        # Interactive session added FIRST → it is the LRU by last_used.
        interactive = mx.array([1, 2, 3, 4, 5, 10, 11], dtype=mx.int32)
        background = mx.array([90, 91, 92, 93, 94, 95, 96], dtype=mx.int32)
        third = mx.array([50, 51, 52, 53, 54, 55, 56], dtype=mx.int32)

        id_interactive = cache.add_kv_cache(
            interactive, _fake_kv_cache(num_layers=2, num_tokens=7),
            low_priority=False,
        )
        id_background = cache.add_kv_cache(
            background, _fake_kv_cache(num_layers=2, num_tokens=7),
            low_priority=True,
        )
        # Adding a third leaf forces ONE eviction (cap=2). LRU alone would pick
        # the interactive leaf (oldest); priority must pick the background leaf.
        cache.add_kv_cache(
            third, _fake_kv_cache(num_layers=2, num_tokens=7),
            low_priority=False,
        )

        assert id_interactive in cache.prompts, (
            "interactive session was wrongly evicted before the background leaf"
        )
        assert id_background not in cache.prompts, (
            "background (low_priority) leaf should have been evicted first"
        )

    def test_interactive_leaf_evicted_when_no_low_priority_remains(self):
        """When no low_priority leaf exists, eviction falls back to plain LRU
        over interactive leaves (no starvation / no-op)."""
        cache = KVPrefixCache(None, max_sessions=2)
        a = mx.array([1, 2, 3, 4, 5, 10, 11], dtype=mx.int32)
        b = mx.array([1, 2, 3, 4, 5, 20, 21], dtype=mx.int32)
        c = mx.array([1, 2, 3, 4, 5, 30, 31], dtype=mx.int32)

        id_a = cache.add_kv_cache(a, _fake_kv_cache(num_layers=2, num_tokens=7))
        cache.add_kv_cache(b, _fake_kv_cache(num_layers=2, num_tokens=7))
        cache.add_kv_cache(c, _fake_kv_cache(num_layers=2, num_tokens=7))

        # No low_priority leaves → LRU (oldest = A) is evicted.
        assert id_a not in cache.prompts

    def test_update_kv_cache_refreshes_priority(self):
        """A leaf's priority class follows the latest request: a leaf created
        interactive that is later updated as low_priority becomes evict-first
        (and vice-versa)."""
        cache = KVPrefixCache(None, max_sessions=2)
        conv = mx.array([1, 2, 3, 4, 5, 10, 11], dtype=mx.int32)
        leaf_id = cache.add_kv_cache(
            conv, _fake_kv_cache(num_layers=2, num_tokens=7), low_priority=False
        )
        assert cache._leaves[leaf_id].low_priority is False  # pyright: ignore[reportPrivateUsage]

        # Continue the same conversation but tagged low_priority this turn.
        conv2 = mx.array([1, 2, 3, 4, 5, 10, 11, 12, 13], dtype=mx.int32)
        cache.update_kv_cache(
            leaf_id, conv2, _fake_kv_cache(num_layers=2, num_tokens=9),
            snapshots=None, restore_pos=7, low_priority=True,
        )
        assert cache._leaves[leaf_id].low_priority is True  # pyright: ignore[reportPrivateUsage]

    def test_high_priority_leaf_survives_when_normal_competes(self):
        """A protected (high_priority) interactive leaf is NOT evicted to make
        room for an untagged normal leaf, even when the high-priority leaf is the
        LRU. This is the cross-turn protection for the user's hot 100K+ session
        against an untagged co-equal session — the gap that plain low_priority
        tagging (which only the cooperating background caller sets) didn't cover.
        """
        cache = KVPrefixCache(None, max_sessions=2)
        # Protected session added FIRST → it is the LRU by last_used.
        protected = mx.array([1, 2, 3, 4, 5, 10, 11], dtype=mx.int32)
        untagged = mx.array([90, 91, 92, 93, 94, 95, 96], dtype=mx.int32)
        third = mx.array([50, 51, 52, 53, 54, 55, 56], dtype=mx.int32)

        id_protected = cache.add_kv_cache(
            protected, _fake_kv_cache(num_layers=2, num_tokens=7),
            high_priority=True,
        )
        cache.add_kv_cache(
            untagged, _fake_kv_cache(num_layers=2, num_tokens=7),
        )
        # cap=2; adding a third forces ONE eviction. LRU alone would pick the
        # protected leaf (oldest); priority must pick the untagged normal leaf.
        cache.add_kv_cache(
            third, _fake_kv_cache(num_layers=2, num_tokens=7),
        )

        assert id_protected in cache.prompts, (
            "high_priority interactive leaf was wrongly evicted before a normal leaf"
        )

    def test_eviction_order_low_then_normal_then_high(self):
        """Full three-class ordering: low_priority evicts first, then normal,
        then high_priority — each only when the cheaper classes are exhausted."""
        cache = KVPrefixCache(None, max_sessions=3)
        low = mx.array([10, 11, 12, 13, 14, 15, 16], dtype=mx.int32)
        normal = mx.array([20, 21, 22, 23, 24, 25, 26], dtype=mx.int32)
        high = mx.array([30, 31, 32, 33, 34, 35, 36], dtype=mx.int32)
        # Insert high FIRST (oldest), then normal, then low (newest) so that
        # plain LRU would evict high → normal → low, the exact REVERSE of the
        # priority order we require.
        id_high = cache.add_kv_cache(
            high, _fake_kv_cache(num_layers=2, num_tokens=7), high_priority=True
        )
        id_normal = cache.add_kv_cache(
            normal, _fake_kv_cache(num_layers=2, num_tokens=7)
        )
        id_low = cache.add_kv_cache(
            low, _fake_kv_cache(num_layers=2, num_tokens=7), low_priority=True
        )

        # 4th leaf → evict the low_priority leaf first.
        cache.add_kv_cache(
            mx.array([40, 41, 42, 43, 44, 45, 46], dtype=mx.int32),
            _fake_kv_cache(num_layers=2, num_tokens=7),
        )
        assert id_low not in cache.prompts, "low_priority should evict first"
        assert id_normal in cache.prompts and id_high in cache.prompts

        # 5th leaf → no low_priority left → evict the normal leaf next.
        cache.add_kv_cache(
            mx.array([50, 51, 52, 53, 54, 55, 56], dtype=mx.int32),
            _fake_kv_cache(num_layers=2, num_tokens=7),
        )
        assert id_normal not in cache.prompts, "normal should evict before high_priority"
        assert id_high in cache.prompts, "high_priority must survive longest"

    def test_high_priority_evicted_when_only_class_remaining(self):
        """No starvation: a high_priority leaf CAN be evicted when it is the only
        reclaimable (non-active) leaf left — protection is relative, not absolute.
        """
        cache = KVPrefixCache(None, max_sessions=1)
        a = mx.array([1, 2, 3, 4, 5, 10, 11], dtype=mx.int32)
        b = mx.array([1, 2, 3, 4, 5, 20, 21], dtype=mx.int32)
        id_a = cache.add_kv_cache(
            a, _fake_kv_cache(num_layers=2, num_tokens=7), high_priority=True
        )
        # Only one slot; a second leaf forces the high_priority leaf out since it
        # is the sole non-active candidate (no cheaper class exists).
        cache.add_kv_cache(b, _fake_kv_cache(num_layers=2, num_tokens=7))
        assert id_a not in cache.prompts

    def test_update_kv_cache_refreshes_high_priority(self):
        """high_priority follows the latest request, surviving the in-place
        extend AND the rebuild path (via _rebuild_leaf_in_place preservation)."""
        cache = KVPrefixCache(None, max_sessions=2)
        conv = mx.array([1, 2, 3, 4, 5, 10, 11], dtype=mx.int32)
        leaf_id = cache.add_kv_cache(
            conv, _fake_kv_cache(num_layers=2, num_tokens=7), high_priority=False
        )
        assert cache._leaves[leaf_id].high_priority is False  # pyright: ignore[reportPrivateUsage]
        # Continue the same conversation, now tagged high_priority.
        conv2 = mx.array([1, 2, 3, 4, 5, 10, 11, 12, 13], dtype=mx.int32)
        cache.update_kv_cache(
            leaf_id, conv2, _fake_kv_cache(num_layers=2, num_tokens=9),
            snapshots=None, restore_pos=7, high_priority=True,
        )
        assert cache._leaves[leaf_id].high_priority is True  # pyright: ignore[reportPrivateUsage]

    def test_pin_prevents_eviction(self):
        cache = KVPrefixCache(None, max_sessions=1)
        tokens_a = mx.array([1, 2, 3, 4, 5], dtype=mx.int32)
        tokens_b = mx.array([6, 7, 8, 9, 10], dtype=mx.int32)
        id_a = cache.add_kv_cache(tokens_a, _fake_kv_cache(num_layers=2, num_tokens=5))
        cache.pin(id_a)
        cache.add_kv_cache(tokens_b, _fake_kv_cache(num_layers=2, num_tokens=5))
        # Pinned A survives; the cap is "soft" under pinning.
        assert id_a in cache.prompts

    def test_active_leaf_survives_memory_pressure_eviction(self, monkeypatch):
        """The in-flight session (marked active by get_kv_cache) must NOT be
        evicted under memory pressure — evicting it guarantees a full re-prefill
        of the exact context being served next turn (the re-prefill loop).

        Repro of the production bug: one large session, memory pegged above the
        eviction threshold. Before the fix, _evict_lru_once picked the only leaf
        (the active one) and dropped it. After the fix, the active leaf is
        excluded and eviction is a clean no-op.
        """
        import exo.worker.engines.mlx.cache as cache_mod

        cache = KVPrefixCache(None)
        # The "big continuing session".
        tokens = mx.array(list(range(1, 51)), dtype=mx.int32)
        leaf_id = cache.add_kv_cache(
            tokens, _fake_kv_cache(num_layers=2, num_tokens=50)
        )
        # Simulate the next turn's lookup hitting this leaf → marks it active.
        cache._active_leaf_id = leaf_id  # pyright: ignore[reportPrivateUsage]

        # Peg memory above threshold so the pressure loop wants to evict.
        monkeypatch.setattr(
            cache_mod, "get_memory_used_percentage", lambda: 0.99
        )
        # Also stub mx.clear_cache (no-op) so the test doesn't depend on real
        # Metal buffer reclamation changing the patched pressure value.
        monkeypatch.setattr(cache_mod.mx, "clear_cache", lambda: None)

        cache._evict_if_needed(reserve_slot=False)  # pyright: ignore[reportPrivateUsage]

        # Active leaf must still be present — never self-evicted.
        assert leaf_id in cache.prompts, "active session was wrongly evicted"

    def test_active_leaf_excluded_but_idle_leaf_evicted(self, monkeypatch):
        """Under pressure with multiple sessions, eviction drops the idle LRU
        leaf and spares the active one (graceful order: idle-first, never the
        in-flight session)."""
        import exo.worker.engines.mlx.cache as cache_mod

        cache = KVPrefixCache(None)
        idle = cache.add_kv_cache(
            mx.array([1, 2, 3, 4, 5], dtype=mx.int32),
            _fake_kv_cache(num_layers=2, num_tokens=5),
        )
        active = cache.add_kv_cache(
            mx.array([6, 7, 8, 9, 10], dtype=mx.int32),
            _fake_kv_cache(num_layers=2, num_tokens=5),
        )
        # Make the IDLE leaf the LRU, and mark the other active.
        cache._leaves[idle].last_used = 0  # pyright: ignore[reportPrivateUsage]
        cache._leaves[active].last_used = 100  # pyright: ignore[reportPrivateUsage]
        cache._active_leaf_id = active  # pyright: ignore[reportPrivateUsage]

        # Pressure stays high so the loop evicts until only the protected
        # active leaf remains, then stops (can't evict the active one).
        monkeypatch.setattr(
            cache_mod, "get_memory_used_percentage", lambda: 0.99
        )
        monkeypatch.setattr(cache_mod.mx, "clear_cache", lambda: None)

        cache._evict_if_needed(reserve_slot=False)  # pyright: ignore[reportPrivateUsage]

        assert idle not in cache.prompts, "idle LRU leaf should be evicted"
        assert active in cache.prompts, "active leaf must survive"

    def test_get_kv_cache_hit_marks_active_leaf(self):
        """A get_kv_cache hit sets _active_leaf_id to the matched leaf so the
        subsequent same-turn add/update can't evict it; a fresh lookup clears
        the stale marker first."""
        cache = KVPrefixCache(None)
        tokens = mx.array([1, 2, 3, 4, 5, 6], dtype=mx.int32)
        leaf_id = cache.add_kv_cache(
            tokens, _fake_kv_cache(num_layers=2, num_tokens=6)
        )
        # add does not mark active.
        assert cache._active_leaf_id is None  # pyright: ignore[reportPrivateUsage]
        # A continuing-turn lookup that shares the prefix marks it active.
        # An exact hit materializes from the trie and never touches `model`,
        # so passing None is safe here.
        _cache, _remaining, matched, _exact = cache.get_kv_cache(
            None,  # pyright: ignore[reportArgumentType]
            tokens,
        )
        assert matched == leaf_id
        assert cache._active_leaf_id == leaf_id  # pyright: ignore[reportPrivateUsage]

    def test_update_extends_existing_leaf(self):
        cache = KVPrefixCache(None)
        tokens_short = mx.array([1, 2, 3], dtype=mx.int32)
        id_s = cache.add_kv_cache(
            tokens_short, _fake_kv_cache(num_layers=2, num_tokens=3)
        )
        tokens_long = mx.array([1, 2, 3, 4, 5], dtype=mx.int32)
        cache.update_kv_cache(
            leaf_id=id_s,
            prompt_tokens=tokens_long,
            cache=_fake_kv_cache(num_layers=2, num_tokens=5),
            snapshots=None,
            restore_pos=3,
        )
        assert list(cache.prompts.keys()) == [id_s]
        stored = cache.prompts[id_s]
        assert int(stored.shape[0]) == 5

    def test_update_fast_path_extends_without_touching_prefix(self):
        """Conversation-extending update should attach a suffix edge rather than
        rebuild the trie. The shared-prefix node keeps its identity (same Python
        object) so we know no re-slicing happened.
        """
        cache = KVPrefixCache(None)
        tokens_turn_1 = mx.array([1, 2, 3, 4, 5], dtype=mx.int32)
        id_c = cache.add_kv_cache(
            tokens_turn_1, _fake_kv_cache(num_layers=2, num_tokens=5)
        )
        root = cache._root  # pyright: ignore[reportPrivateUsage]
        original_first_edge = root.children[1]  # node covering [1..5]

        tokens_turn_2 = mx.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=mx.int32)
        cache.update_kv_cache(
            leaf_id=id_c,
            prompt_tokens=tokens_turn_2,
            cache=_fake_kv_cache(num_layers=2, num_tokens=8),
            snapshots=None,
            restore_pos=5,
        )

        # Same shared-prefix node, now with a child edge covering [6,7,8].
        first_edge_after = root.children[1]
        assert first_edge_after is original_first_edge, (
            "extend-in-place must not rebuild the shared prefix node"
        )
        assert set(first_edge_after.children.keys()) == {6}
        suffix_edge = first_edge_after.children[6]
        assert suffix_edge.edge_length == 3  # [6, 7, 8]
        assert int(cache.prompts[id_c].shape[0]) == 8

    def test_update_rebuilds_when_prefix_diverges(self):
        """If the new prompt doesn't start with the old leaf's tokens, the
        leaf's trie anchor must be re-rooted. Cache stays functional.
        """
        cache = KVPrefixCache(None)
        tokens_a = mx.array([1, 2, 3, 4, 5], dtype=mx.int32)
        id_c = cache.add_kv_cache(
            tokens_a, _fake_kv_cache(num_layers=2, num_tokens=5)
        )
        tokens_b = mx.array([9, 9, 9, 1, 2, 3], dtype=mx.int32)
        cache.update_kv_cache(
            leaf_id=id_c,
            prompt_tokens=tokens_b,
            cache=_fake_kv_cache(num_layers=2, num_tokens=6),
            snapshots=None,
            restore_pos=0,
        )
        assert list(cache.prompts.keys()) == [id_c]
        assert int(cache.prompts[id_c].shape[0]) == 6

    def test_media_region_mismatch_truncates_match(self):
        from exo.worker.engines.mlx.vision import MediaRegion

        cache = KVPrefixCache(None)
        tokens = mx.array([1, 2, 3, 4, 5, 6, 7], dtype=mx.int32)
        regions_a = [MediaRegion(content_hash="img-A", start_pos=2, end_pos=5)]
        cache.add_kv_cache(
            tokens, _fake_kv_cache(num_layers=1, num_tokens=7), media_regions=regions_a
        )

        from unittest.mock import MagicMock

        model = MagicMock()
        model.layers = [None]

        # Query with the same tokens but a different image hash at pos 2.
        regions_b = [MediaRegion(content_hash="img-B", start_pos=2, end_pos=5)]
        _, remaining, _, _ = cache.get_kv_cache(model, tokens, media_regions=regions_b)
        # Match should be truncated to pos 2 (start of the mismatching region).
        assert int(remaining.shape[0]) == 5  # tokens 2..6


def _load_gpt_oss() -> tuple[Model, object]:
    from mlx_lm.utils import load_model

    from exo.worker.engines.mlx.utils_mlx import load_tokenizer_for_model_id

    model_path = DEFAULT_GPT_OSS_CONFIG.model_path
    model_id = ModelId(DEFAULT_GPT_OSS_MODEL_ID)

    model, _ = load_model(model_path, lazy=False)
    tokenizer = load_tokenizer_for_model_id(model_id, model_path)
    return cast(Model, model), tokenizer


@pytest.mark.slow
@pytest.mark.skipif(
    not _check_model_exists(),
    reason=f"GPT-OSS model not found at {DEFAULT_GPT_OSS_CONFIG.model_path}",
)
class TestKVPrefixCacheWithModel:
    @pytest.fixture(scope="class")
    def model_and_tokenizer(self):
        model, tokenizer = _load_gpt_oss()
        return model, tokenizer

    def test_prefill_populates_cache(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer

        task = TextGenerationTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            input=[InputMessage(role="user", content="Hello!!")],
            max_output_tokens=1,
        )
        prompt = apply_chat_template(tokenizer, task)
        tokens = encode_prompt(tokenizer, prompt)
        cache = make_kv_cache(model)

        _, _, snapshots = prefill(
            model,
            tokenizer,
            make_sampler(0.0),
            tokens,
            cache,
            group=None,
            on_prefill_progress=None,
            distributed_prompt_progress_callback=None,
        )

        # Cache should now hold the prompt tokens minus one
        assert cache_length(cache) == len(tokens) - 1
        # Snapshots should be available for models with non-KV caches
        assert len(snapshots) > 0

    def test_add_and_get_exact_match(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer

        task = TextGenerationTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            input=[InputMessage(role="user", content="Test exact")],
            max_output_tokens=1,
        )
        prompt = apply_chat_template(tokenizer, task)
        tokens = encode_prompt(tokenizer, prompt)
        cache = make_kv_cache(model)

        _, _, snapshots = prefill(
            model,
            tokenizer,
            make_sampler(0.0),
            tokens,
            cache,
            group=None,
            on_prefill_progress=None,
            distributed_prompt_progress_callback=None,
        )

        kv_prefix_cache = KVPrefixCache(None)
        kv_prefix_cache.add_kv_cache(tokens, cache, snapshots)

        assert len(kv_prefix_cache.prompts) == 1
        stored_length = cache_length(kv_prefix_cache.caches[0])
        assert stored_length > 0

        # Retrieve with same prompt: exact match
        result_cache, remaining_tokens, matched_index, _ = kv_prefix_cache.get_kv_cache(
            model, tokens
        )
        assert matched_index == 0

        # Exact match returns last token(s) — for models with SSM/rotating caches,
        # snapshot availability constrains how far back we can trim, so remaining
        # may be 1 or 2 tokens depending on the model.
        assert len(remaining_tokens) >= 1
        assert mx.array_equal(remaining_tokens, tokens[-len(remaining_tokens) :])

    def test_add_and_get_prefix_match(self, model_and_tokenizer):
        """get_kv_cache with a longer prompt sharing prefix should return partial match."""
        model, tokenizer = model_and_tokenizer

        short_task = TextGenerationTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            input=[InputMessage(role="user", content="Hi")],
            max_output_tokens=1,
        )
        short_prompt = apply_chat_template(tokenizer, short_task)
        short_tokens = encode_prompt(tokenizer, short_prompt)
        cache = make_kv_cache(model)

        _, _, snapshots = prefill(
            model,
            tokenizer,
            make_sampler(0.0),
            short_tokens,
            cache,
            group=None,
            on_prefill_progress=None,
            distributed_prompt_progress_callback=None,
        )

        kv_prefix_cache = KVPrefixCache(None)
        kv_prefix_cache.add_kv_cache(short_tokens, cache, snapshots)

        # Query with longer prompt that shares the chat template prefix
        long_task = TextGenerationTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            input=[InputMessage(role="user", content="Hi there, how are you?")],
            max_output_tokens=1,
        )
        long_prompt = apply_chat_template(tokenizer, long_task)
        long_tokens = encode_prompt(tokenizer, long_prompt)

        # The prompts share a prefix (chat template preamble + "Hi")
        expected_prefix = get_prefix_length(long_tokens, short_tokens)
        assert expected_prefix > 0, (
            "Prompts should share a prefix from the chat template"
        )

        result_cache, remaining_tokens, matched_index, _ = kv_prefix_cache.get_kv_cache(
            model, long_tokens
        )
        assert matched_index == 0

        # remaining_tokens covers from snapshot restore position to end
        assert len(remaining_tokens) >= len(long_tokens) - expected_prefix

    def test_stored_cache_not_mutated_after_get_and_generation(
        self, model_and_tokenizer
    ):
        """Getting a cache and then mutating it (as generation does) must not corrupt stored cache."""
        model, tokenizer = model_and_tokenizer

        task = TextGenerationTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            input=[InputMessage(role="user", content="Mutation test")],
            max_output_tokens=1,
        )
        prompt = apply_chat_template(tokenizer, task)
        tokens = encode_prompt(tokenizer, prompt)
        cache = make_kv_cache(model)

        _, _, snapshots = prefill(
            model,
            tokenizer,
            make_sampler(0.0),
            tokens,
            cache,
            group=None,
            on_prefill_progress=None,
            distributed_prompt_progress_callback=None,
        )

        kv_prefix_cache = KVPrefixCache(None)
        kv_prefix_cache.add_kv_cache(tokens, cache, snapshots)

        stored_length = cache_length(kv_prefix_cache.caches[0])

        # Get cache and mutate it (simulating what generation does)
        result_cache, _, matched_index, _ = kv_prefix_cache.get_kv_cache(model, tokens)
        assert matched_index == 0

        # Simulate generation: feed many additional tokens through the cache
        head_dim = result_cache[0].keys.shape[-1]
        num_heads = result_cache[0].keys.shape[1]
        extra_keys = mx.random.normal((1, num_heads, 50, head_dim))
        extra_values = mx.random.normal((1, num_heads, 50, head_dim))
        for layer_cache in result_cache:
            layer_cache.update_and_fetch(extra_keys, extra_values)
        mx.eval([c.keys for c in result_cache])

        # Stored cache must be unchanged
        assert cache_length(kv_prefix_cache.caches[0]) == stored_length

    def test_stored_cache_survives_repeated_get_mutate_cycles(
        self, model_and_tokenizer
    ):
        """Multiple get+mutate cycles (like repeated user requests) must not corrupt cache."""
        model, tokenizer = model_and_tokenizer

        task = TextGenerationTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            input=[InputMessage(role="user", content="Repeat test")],
            max_output_tokens=1,
        )
        prompt = apply_chat_template(tokenizer, task)
        tokens = encode_prompt(tokenizer, prompt)
        cache = make_kv_cache(model)

        _, _, snapshots = prefill(
            model,
            tokenizer,
            make_sampler(0.0),
            tokens,
            cache,
            group=None,
            on_prefill_progress=None,
            distributed_prompt_progress_callback=None,
        )

        kv_prefix_cache = KVPrefixCache(None)
        kv_prefix_cache.add_kv_cache(tokens, cache, snapshots)

        stored_length = cache_length(kv_prefix_cache.caches[0])

        for i in range(3):
            result_cache, _, _, _ = kv_prefix_cache.get_kv_cache(model, tokens)

            head_dim = result_cache[0].keys.shape[-1]
            num_heads = result_cache[0].keys.shape[1]
            extra = mx.random.normal((1, num_heads, 30, head_dim))
            for layer_cache in result_cache:
                layer_cache.update_and_fetch(extra, extra)
            mx.eval([c.keys for c in result_cache])

            assert cache_length(kv_prefix_cache.caches[0]) == stored_length, (
                f"Failed on loop {i}"
            )

    def test_mlx_generate_populates_cache(self, model_and_tokenizer):
        """mlx_generate should save the post-prefill cache (before the decode loop)."""
        model, tokenizer = model_and_tokenizer

        kv_prefix_cache = KVPrefixCache(None)
        task = TextGenerationTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            input=[InputMessage(role="user", content="Hello")],
            max_output_tokens=5,
        )
        prompt = apply_chat_template(tokenizer, task)
        prompt_tokens = encode_prompt(tokenizer, prompt)

        # Consume the entire generator so the cache-saving code after yield runs
        for _response in mlx_generate(
            model=model,
            tokenizer=tokenizer,
            task=task,
            prompt=prompt,
            kv_prefix_cache=kv_prefix_cache,
            group=None,
        ):
            pass

        assert len(kv_prefix_cache.prompts) == 1
        assert len(kv_prefix_cache.caches) == 1
        # add_kv_cache is called before the decode loop and stores a deepcopy of
        # the cache as it is just after prefill + trim(2). Generation tokens are
        # never written into the stored entry.
        assert cache_length(kv_prefix_cache.caches[0]) == len(prompt_tokens) - 2

    def test_mlx_generate_second_call_gets_prefix_hit(self, model_and_tokenizer):
        """Second mlx_generate call with same prompt should get a prefix hit from stored cache."""
        model, tokenizer = model_and_tokenizer

        kv_prefix_cache = KVPrefixCache(None)
        task = TextGenerationTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            input=[InputMessage(role="user", content="Reuse test")],
            max_output_tokens=5,
        )
        prompt = apply_chat_template(tokenizer, task)
        prompt_tokens = encode_prompt(tokenizer, prompt)

        # First generation populates cache
        for _response in mlx_generate(
            model=model,
            tokenizer=tokenizer,
            task=task,
            prompt=prompt,
            kv_prefix_cache=kv_prefix_cache,
            group=None,
        ):
            pass

        assert len(kv_prefix_cache.prompts) == 1

        # Second call should find a prefix match (the stored cache contains
        # prompt + generated tokens, which shares the prompt prefix)
        result_cache, remaining_tokens, matched_index, _ = kv_prefix_cache.get_kv_cache(
            model, prompt_tokens
        )
        # The stored cache is longer than the prompt (it includes generated tokens),
        # so this is a prefix match where our prompt is fully contained
        assert matched_index == 0
        # Exact match: remaining_tokens is just the last token and the one before
        assert len(remaining_tokens) == 2
        assert mx.array_equal(remaining_tokens, prompt_tokens[-2:])

    def test_mlx_generate_long_prompt_updates_cache_in_place(self, model_and_tokenizer):
        """With a prompt > 1000 tokens, second generation should update the cache entry in-place."""
        model, tokenizer = model_and_tokenizer

        kv_prefix_cache = KVPrefixCache(None)

        # Build a long user message (> 1000 tokens) to exceed _MIN_PREFIX_HIT_TO_UPDATE
        base_text = "The quick brown fox jumps over the lazy dog. "
        base_tokens = tokenizer.encode(base_text)
        repeats = (1200 // len(base_tokens)) + 2
        long_content = base_text * repeats

        task1 = TextGenerationTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            input=[InputMessage(role="user", content=long_content)],
            max_output_tokens=5,
        )
        prompt1 = apply_chat_template(tokenizer, task1)
        prompt1_tokens = encode_prompt(tokenizer, prompt1)
        assert len(prompt1_tokens) > 1000, (
            "Prompt must exceed _MIN_PREFIX_HIT_TO_UPDATE"
        )

        # First generation populates the cache (must prefill all tokens)
        t0 = time.perf_counter()
        for _response in mlx_generate(
            model=model,
            tokenizer=tokenizer,
            task=task1,
            prompt=prompt1,
            kv_prefix_cache=kv_prefix_cache,
            group=None,
        ):
            pass
        first_gen_time = time.perf_counter() - t0

        assert len(kv_prefix_cache.prompts) == 1
        first_cache_length = cache_length(kv_prefix_cache.caches[0])

        # Second generation: same long prompt + extra content (simulating multi-turn)
        task2 = TextGenerationTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            input=[
                InputMessage(role="user", content=long_content),
                InputMessage(role="assistant", content="Sure, I can help."),
                InputMessage(role="user", content="Tell me more."),
            ],
            max_output_tokens=5,
        )
        prompt2 = apply_chat_template(tokenizer, task2)
        prompt2_tokens = encode_prompt(tokenizer, prompt2)

        # Verify the prompts share a long prefix
        prefix_len = get_prefix_length(prompt2_tokens, prompt1_tokens)
        assert prefix_len > 1000, "Prompts must share > 1000 token prefix"

        # Second generation should reuse the cached prefix (only prefill new tokens)
        t0 = time.perf_counter()
        for _response in mlx_generate(
            model=model,
            tokenizer=tokenizer,
            task=task2,
            prompt=prompt2,
            kv_prefix_cache=kv_prefix_cache,
            group=None,
        ):
            pass
        second_gen_time = time.perf_counter() - t0

        # Second generation should be significantly faster due to prefix cache hit - hopefully not flaky
        assert second_gen_time < first_gen_time * 0.5, (
            f"Expected prefix cache speedup: "
            f"first={first_gen_time:.2f}s, second={second_gen_time:.2f}s"
        )

        # With prefix_hit > 1000, should update in-place (not add a second entry)
        assert len(kv_prefix_cache.prompts) == 1
        # Updated cache should be longer (prompt2 + generated > prompt1 + generated)
        updated_cache_length = cache_length(kv_prefix_cache.caches[0])
        assert updated_cache_length > first_cache_length

    def test_mlx_generate_stored_cache_not_mutated(self, model_and_tokenizer):
        """After mlx_generate saves a cache, a second generation must not corrupt the stored copy."""
        model, tokenizer = model_and_tokenizer

        kv_prefix_cache = KVPrefixCache(None)
        task = TextGenerationTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            input=[InputMessage(role="user", content="Immutable test")],
            max_output_tokens=5,
        )
        prompt = apply_chat_template(tokenizer, task)

        # First generation populates cache
        for _response in mlx_generate(
            model=model,
            tokenizer=tokenizer,
            task=task,
            prompt=prompt,
            kv_prefix_cache=kv_prefix_cache,
            group=None,
        ):
            pass

        firstcache_length = cache_length(kv_prefix_cache.caches[0])

        # Second generation gets the cache and mutates it during generation
        for _response in mlx_generate(
            model=model,
            tokenizer=tokenizer,
            task=task,
            prompt=prompt,
            kv_prefix_cache=kv_prefix_cache,
            group=None,
        ):
            pass

        # The first stored cache must not have been mutated by the second generation
        assert cache_length(kv_prefix_cache.caches[0]) == firstcache_length

    def test_evicts_lru_entry_under_memory_pressure(self, model_and_tokenizer):
        """Under memory pressure, adding a new cache entry evicts the least recently used one."""
        model, tokenizer = model_and_tokenizer

        kv_prefix_cache = KVPrefixCache(None)

        # Add three cache entries with different prompts
        prompts = ["First entry", "Second entry", "Third entry"]
        for i, content in enumerate(prompts):
            task = TextGenerationTaskParams(
                model=DEFAULT_GPT_OSS_MODEL_ID,
                input=[InputMessage(role="user", content=content)],
                max_output_tokens=1,
            )
            prompt = apply_chat_template(tokenizer, task)
            tokens = encode_prompt(tokenizer, prompt)
            cache = make_kv_cache(model)
            prefill(
                model,
                tokenizer,
                make_sampler(0.0),
                tokens,
                cache,
                group=None,
                on_prefill_progress=None,
                distributed_prompt_progress_callback=None,
            )
            kv_prefix_cache.add_kv_cache(tokens, cache)
            # Stagger _last_used so LRU order is deterministic
            kv_prefix_cache._last_used[i] = float(i)

        assert len(kv_prefix_cache.prompts) == 3

        # Access the third entry to make it most recently used
        kv_prefix_cache._last_used[2] = 100.0
        # Entry 0 (_last_used=0.0) is LRU, entry 1 (_last_used=1.0) is next

        # Simulate memory pressure: return usage above _MEMORY_THRESHOLD (0.9)
        with patch(
            "exo.worker.engines.mlx.cache.get_memory_used_percentage",
            return_value=0.95,
        ):
            # Trigger eviction by adding a new entry
            task = TextGenerationTaskParams(
                model=DEFAULT_GPT_OSS_MODEL_ID,
                input=[InputMessage(role="user", content="New entry")],
                max_output_tokens=1,
            )
            prompt = apply_chat_template(tokenizer, task)
            tokens = encode_prompt(tokenizer, prompt)
            cache = make_kv_cache(model)
            prefill(
                model,
                tokenizer,
                make_sampler(0.0),
                tokens,
                cache,
                group=None,
                on_prefill_progress=None,
                distributed_prompt_progress_callback=None,
            )
            kv_prefix_cache.add_kv_cache(tokens, cache)

        # LRU entries should have been evicted (entries 0, 1, 2 in order of _last_used)
        # Since fake_active stays above threshold after each eviction (we don't change it),
        # all old entries get evicted, leaving only the newly added one
        assert len(kv_prefix_cache.prompts) == 1
        # The surviving entry should be the newly added one
        assert get_prefix_length(kv_prefix_cache.prompts[0], tokens) == len(tokens)


class TestMultiTurnSnapshotLeak:
    """Regression guard for the DSv4-Flash multi-turn memory leak.

    A continuing conversation on DSv4-Flash (non-trimmable
    ``CacheList(RotatingKVCache, PoolingCache, ...)`` layers) accumulated one
    full per-sparse-layer snapshot set in the leaf EVERY turn. Two sites leaked:

      1. ``update_kv_cache`` merged the leaf's old snapshots with the new turn's
         filtered by ``token_count <= restore_pos``. Since ``restore_pos`` climbs
         monotonically across turns, the filter never dropped anything → the
         leaf's ``leaf_snapshots`` grew unbounded (+1 set/turn).
      2. ``_build_edge_node`` stored a per-layer snapshot on every new suffix
         edge — write-only state (the restore path never reads node snapshots),
         pinning a full PoolingCache set per edge, never pruned while the leaf
         lived.

    On DSv4 each snapshot holds a ``PoolingCache`` per compress_ratio=128 layer,
    so the leak was ~+21 ``(1,P,512)``/``(1,P,128)`` bf16 tensors per turn
    (~0.2-0.4 GB/turn, ~29 GB over a long Hermes session; survived idle AND
    session-end because the leaf lives in the persistent trie).

    These tests use lightweight stand-ins (no model) that reproduce the
    non-trimmable cache shape and assert both the leaf snapshot count and the
    live ``PoolingCache`` object count reach a flat steady state across turns.
    """

    @staticmethod
    def _pooling_cache(p_len: int, ratio: int = 128) -> PoolingCache:
        # Set pooled storage so is_trimmable() -> False (the DSv4 steady state
        # that routes the CacheList through the snapshot path).
        pc = PoolingCache(ratio)
        pc._pool_storage = mx.zeros((1, p_len, 512), dtype=mx.bfloat16)
        pc._pool_offset = p_len
        pc.buf_kv = mx.zeros((1, ratio, 512), dtype=mx.bfloat16)
        pc.buf_gate = mx.zeros((1, ratio, 128), dtype=mx.bfloat16)
        return pc

    @staticmethod
    def _rotating(num_tokens: int) -> RotatingKVCache:
        r = RotatingKVCache(max_size=4096)
        r.keys = mx.zeros((1, 2, num_tokens, 4), dtype=mx.bfloat16)
        r.values = mx.zeros((1, 2, num_tokens, 4), dtype=mx.bfloat16)
        r.offset = num_tokens
        r._idx = num_tokens
        return r

    @classmethod
    def _dsv4_like_cache(cls, num_tokens: int, n_sparse_layers: int = 21):
        p_len = max(1, num_tokens // 128)
        return [
            CacheList(cls._rotating(num_tokens), cls._pooling_cache(p_len))
            for _ in range(n_sparse_layers)
        ]

    @staticmethod
    def _count_live_pooling() -> int:
        import gc

        gc.collect()
        return sum(1 for o in gc.get_objects() if isinstance(o, PoolingCache))

    def test_continuing_conversation_does_not_accumulate_snapshots(self):
        from exo.worker.engines.mlx.cache import (
            _LEAF_SNAPSHOT_RETENTION,
            snapshot_ssm_states,
        )

        cache = KVPrefixCache(None)
        base = list(range(1, 201))
        c0 = self._dsv4_like_cache(len(base))
        leaf_id = cache.add_kv_cache(
            mx.array(base, dtype=mx.int32),
            c0,
            ssm_snapshots=[snapshot_ssm_states(c0)],
        )

        pooling_trace: list[int] = []
        prompt = list(base)
        for turn in range(1, 8):
            restore_pos = len(prompt)
            prompt = prompt + list(range(1000 + turn * 100, 1000 + turn * 100 + 80))
            new_cache = self._dsv4_like_cache(len(prompt))
            cache.update_kv_cache(
                leaf_id,
                mx.array(prompt, dtype=mx.int32),
                new_cache,
                [snapshot_ssm_states(new_cache)],
                restore_pos=restore_pos,
            )
            pooling_trace.append(self._count_live_pooling())

        leaf = cache._leaves[leaf_id]  # pyright: ignore[reportPrivateUsage]
        n_snaps = len(leaf.leaf_snapshots or [])

        # Snapshot list bounded by retention (was unbounded: +1/turn).
        assert n_snaps <= _LEAF_SNAPSHOT_RETENTION, (
            f"leaf_snapshots unbounded: {n_snaps} > {_LEAF_SNAPSHOT_RETENTION}"
        )
        # Live PoolingCache count reaches a flat steady state — no +21/turn.
        tail = pooling_trace[-3:]
        assert len(set(tail)) == 1, (
            f"PoolingCache still leaking across turns: {pooling_trace}"
        )


def _snap(tc: int) -> CacheSnapshot:
    """A minimal CacheSnapshot at a given token_count (states irrelevant here)."""
    return CacheSnapshot(states=[], token_count=tc)


class TestSpacedSnapshotRetention:
    """Regression tests for _select_spaced_snapshots — the spaced (stride-from-
    tip) retention that replaced deepest-N. The deepest-N bug clustered all
    retained snapshots at the tip, so a prompt that diverged a few thousand
    tokens below the tip (reasoning-trace drop / compaction) found NO snapshot
    at/below the divergence and cold re-prefilled the whole 100K+ prompt.
    """

    def test_under_budget_returns_all(self):
        snaps = [_snap(1000), _snap(2000), _snap(3000)]
        out = _select_spaced_snapshots(snaps, keep=4)
        assert [s.token_count for s in out] == [1000, 2000, 3000]

    def test_keeps_shallowest_and_deepest_endpoints(self):
        # Simulate the observed failure: dense near-tip snapshots from small
        # extends, tip ~112873. Deepest-4 kept [111631,112354,112639,112873] —
        # nothing at/below a 108130 divergence. Even-spread must keep BOTH the
        # shallowest (106670) and deepest (112873), so a below-tip divergence
        # resolves.
        tips = [106670, 107556, 108246, 109787, 111631, 112354, 112639, 112873]
        out = _select_spaced_snapshots([_snap(t) for t in tips], keep=4)
        depths = [s.token_count for s in out]
        assert depths[0] == 106670, depths   # shallowest always kept
        assert depths[-1] == 112873, depths  # deepest always kept
        assert len(depths) == 4
        assert depths == sorted(set(depths))

    def test_divergence_below_tip_now_finds_snapshot(self):
        # End-to-end intent: after spaced selection, _find_nearest_snapshot at
        # the exact observed divergence depth returns a usable restore point
        # (not None) — the whole point of the fix.
        tips = [106670, 107556, 108246, 109787, 111631, 112354, 112639, 112873]
        retained = _select_spaced_snapshots([_snap(t) for t in tips], keep=4)
        snap = _find_nearest_snapshot(retained, 108130)
        assert snap is not None
        assert snap.token_count <= 108130

    def test_even_spacing_bounds_worst_case_gap(self):
        # 20 snapshots at 1000-token spacing (1000..20000), keep=4 → endpoints
        # 1000 & 20000 plus 2 interior; the largest gap between retained
        # snapshots must be bounded near range/(keep-1) ~= 6333, never the whole
        # range. This bounds the worst-case re-prefill.
        snaps = [_snap(1000 * i) for i in range(1, 21)]
        out = _select_spaced_snapshots(snaps, keep=4)
        depths = sorted(s.token_count for s in out)
        assert depths[0] == 1000 and depths[-1] == 20000
        assert len(depths) == 4
        gaps = [b - a for a, b in zip(depths, depths[1:], strict=False)]
        # Allow one snap-to-nearest rounding step (1000) above the ideal.
        assert max(gaps) <= (20000 - 1000) // (4 - 1) + 1000, depths

    def test_sparse_set_backfills_to_full_budget(self):
        # Fewer distinct depths than keep slots after spacing collisions:
        # backfill must still use the full budget with deepest snapshots.
        snaps = [_snap(t) for t in (1000, 1100, 1200, 20000, 20100)]
        out = _select_spaced_snapshots(snaps, keep=4)
        assert len(out) == 4
        assert out[0].token_count == 1000   # shallowest
        assert out[-1].token_count == 20100  # deepest

    def test_keep_one_returns_deepest(self):
        snaps = [_snap(t) for t in (1000, 2000, 3000)]
        out = _select_spaced_snapshots(snaps, keep=1)
        assert [s.token_count for s in out] == [3000]
