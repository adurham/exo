# KV Cache Architecture

Authoritative reference for how exo manages KV state across a request,
between turns of a conversation, and across different sessions that share
system-prompt prefixes.

Scope:
- [Cache types](#cache-types) — the per-layer cache objects exo consumes from
  mlx-lm.
- [Radix prefix cache](#radix-prefix-cache) — exo's `KVPrefixCache`, which
  stores past prompts and their KV so a follow-up request with a shared
  prefix doesn't re-prefill that prefix.
- [Per-instance configuration](#per-instance-configuration) — the fields on
  `BaseInstance` that control caps, quantization, and prefix-cache behavior.
- [Runtime code paths](#runtime-code-paths) — which file owns which part.

File map:
| What | Where |
|---|---|
| Per-instance config fields | `src/exo/shared/types/worker/instances.py` (`BaseInstance`) |
| Runner → Builder wiring | `src/exo/worker/runner/llm_inference/runner.py` |
| Prefix cache, radix trie, `make_kv_cache` | `src/exo/worker/engines/mlx/cache.py` |
| mlx-lm cache types (upstream) | `mlx-lm/mlx_lm/models/cache.py` (`KVCache`, `QuantizedKVCache`, `RotatingKVCache`, `ArraysCache`) |

## Cache types

A KV cache is a list of per-layer cache objects. Mlx-lm ships several and exo
treats them differently based on whether each supports positional slicing
(keeping tokens `[a:b)` independently of the rest).

| Type | Sliceable? | Where it comes from | Notes |
|---|---|---|---|
| `KVCache` | Yes | mlx-lm default for pure-attention layers | Grows lazily in steps (`step=16384` in exo) |
| `QuantizedKVCache` | Yes, group-aligned | `EXO_KV_CACHE_BITS>0` or `kv_cache_bits>0` | Group-quantized K/V; merge dequantizes into a `BatchKVCache` for `_merge_caches` |
| `RotatingKVCache` | Only before rotation | exo when `max_kv_tokens` is set | After rotation, token N doesn't live at index N — treat as non-sliceable |
| `ArraysCache` | No (SSM state) | GatedDeltaNet layers in Qwen3.5, etc. | Store snapshots at specific depths; can't positionally slice |
| `TurboQuantKVCache` | Yes | `EXO_TURBOQUANT_ENABLED=1` | Experimental, off by default |

The radix prefix cache dedupes sliceable layers across edges and keeps
non-sliceable layers per-leaf (with the caller's snapshots).

## Radix prefix cache

### Data model

`KVPrefixCache` (in `cache.py`) is a radix trie whose edges carry per-layer
K/V slices for the tokens they cover. Each node knows the edge tokens, the
edge K/V slices (for sliceable layer types), an optional `CacheSnapshot` for
the end-of-edge depth (used by non-sliceable layers), and a ref-count.

Leaves map to sessions. An internal node represents a prefix shared by ≥ 2
leaves. Ref-counts let eviction free a leaf's unique-suffix subtree without
disturbing the shared prefix.

Key attributes (read side):
- `KVPrefixCache.prompts: dict[int, mx.array]` — leaf-id → full token sequence.
- `KVPrefixCache.prefill_tps: dict[int, float]` — leaf-id → prefill tok/s when the leaf was inserted.
- `KVPrefixCache.caches: _CacheProxy` — dict-like; `caches[leaf_id]` materialises a contiguous per-layer cache on demand by concat'ing the slices along the leaf's path.

### add_kv_cache

Inserts a session. Walks the trie matching tokens. On divergence it splits the existing edge at the divergence point, attaches a new leaf on a new child edge, and slices only the unique-suffix tokens of the caller's cache onto the new leaf's edge. No re-slicing of the shared prefix.

The log line at this path includes `shared_prefix=N tok (P%)` — that's the bytes the trie reused vs. what this session added.

### update_kv_cache (fast path vs rebuild)

Called when a session has run another turn and the caller wants to extend
its leaf. Two paths:

1. **Extend-in-place (common)** — the old leaf's stored tokens are a prefix
   of the new prompt. A new edge carrying just the suffix tokens and the
   suffix K/V slice is attached under the old leaf's anchor; the leaf
   pointer moves to the new terminal. Ref-count bookkeeping stays local to
   the new edge (see the comment block in `_extend_leaf_suffix`) — the
   shared prefix node is never touched, re-sliced, or re-copied. Per-turn
   work is O(suffix), not O(full prompt).

2. **Rebuild** — the new prompt doesn't start with the old leaf's tokens
   (e.g., system-prompt edit). The old path is detached and re-inserted
   from scratch.

### get_kv_cache

Returns `(cache, remaining_tokens, matched_leaf_id, is_exact)`:

- `cache` — a per-layer cache usable by the generator. For sliceable
  layers it's built by concat'ing the slices along the matched path, then
  trimmed to the hit depth. For non-sliceable layers the leaf's own cache
  is deep-copied and restored to the nearest snapshot ≤ the target depth.
- `remaining_tokens` — the suffix the caller still needs to prefill
  (empty/one-token on exact hit).
- `matched_leaf_id` — the leaf id whose node was used as the donor for
  non-sliceable recovery; passed back into `update_kv_cache` to extend the
  same leaf post-turn.
- `is_exact` — full-prompt match.

Media-region consistency: if a cached region at position P has a different
`content_hash` from the query's region at the same position, the match is
truncated to P so we don't reuse KV derived from a different image.

### Copy semantics

Per-edge K/V slices are stored via `_detached_copy` which is an `mx.array(a)`
alias. MLX arrays are copy-on-write under `__setitem__`: if the caller's
cache mutates after `add_kv_cache`, MLX forks the backing allocation and our
slice keeps pointing at the unmutated data. `_detached_copy_numpy` (numpy
round-trip) is kept only for `copy_rotating_kv_cache`, where the rotating-
buffer metadata forces a full-detached store to avoid a memory leak.

### Eviction

LRU over leaves, triggered by `_evict_if_needed`. Caps:

1. `max_sessions` — preemptively drops the LRU leaf to make room for a new
   insert (`len(leaves) < max_sessions` after insert is the invariant).
2. `max_bytes` — optional hard byte cap over the whole trie (edges + leaf
   non-sliceable holdings).
3. Memory-threshold eviction — cluster-wide pressure via
   `EXO_MEMORY_THRESHOLD` (default scales by node RAM).

Pinning a leaf (`pin(leaf_id)`) marks it non-evictable; ancestors stay alive
transitively while the pinned leaf is alive.

## Per-instance configuration

All fields live on `BaseInstance` and inherit into every concrete instance
type (`MlxRingInstance`, `MlxJacclInstance`).

| Field | Effect |
|---|---|
| `max_kv_tokens: int \| None` | Per-active-request token cap. When set, plain `KVCache` layers are wrapped in `RotatingKVCache(max_size=N)`. |
| `max_prefix_sessions: int \| None` | Leaf count cap for the radix trie. |
| `max_prefix_bytes: int \| None` | Hard byte cap for the radix trie. |
| `kv_cache_bits: int \| None` | Per-instance KV quantization override. `None` = defer to `EXO_KV_CACHE_BITS` env; `0` = explicitly disable (overrides env); positive N = quantize to N bits. |
| `default_temperature/top_p/top_k/min_p/presence_penalty/repetition_penalty` | Sampling defaults; resolution order is request → instance → cluster env → hardcoded fallback. |

### kv_cache_bits resolution order

1. `instance.kv_cache_bits` if not `None`.
2. `EXO_KV_CACHE_BITS` env (backward compatible global default).
3. No quantization.

A value of `0` at step 1 explicitly disables quantization even when the env
is set — the sentinel exists specifically for instances like the Huihui
scouts that coexist on a node with a quantization-hungry tenant but don't
themselves benefit from 4-bit KV.

### start_cluster.sh knobs

Defaults wired as of commit `65655ced`:
- `MINIMAX_KV_CACHE_BITS=4` — MiniMax-M2.7 runs 4-bit KV so 196K × 2 sessions fits per node.
- `HUIHUI_KV_CACHE_BITS=0` — Huihui scouts run bf16 KV.
- `MINIMAX_MAX_PREFIX_SESSIONS=2` — two-turn reuse for the active Hermes conversation.
- `HUIHUI_MAX_PREFIX_SESSIONS=1` — scouts overwrite the previous turn (no reuse).
- `HUIHUI_MAX_KV_TOKENS=36864` — rotating-buffer cap on scouts; keeps their working set small.

## Runtime code paths

Concrete chain from a request landing on the runner:

1. Master places the instance → Worker spawns a Runner process
   (`src/exo/worker/runner/llm_inference/runner.py`).
2. `Runner.__init__` builds a `Builder` with every per-instance field,
   including `kv_cache_bits`.
3. On `LoadModel`, `Builder.build()` constructs the `KVPrefixCache` and
   passes it plus the resolved `kv_cache_bits` into either
   `BatchGenerator` or `SequentialGenerator`.
4. Per request, the generator calls `KVPrefixCache.get_kv_cache(model, …)`.
   - Miss: `make_kv_cache(model, max_kv_size=…, kv_cache_bits=…)` builds a
     fresh cache honoring the instance override.
   - Hit: trie materialises the hit-depth cache.
5. Generation runs. Post-generation the caller invokes
   `update_kv_cache(leaf_id, …)` or `add_kv_cache(…)` which feeds the
   updated KV back into the trie for the next turn.

## Known follow-ups

- **MTPBatchGenerator port.** `src/exo/worker/engines/mlx/speculative/mtp_batch_generator.py`
  subclasses mlx-lm's old `BatchGenerator` and uses `active_batch` /
  `unprocessed_prompts` / `_next()` returning a list. Upstream rewrote the
  class into `PromptProcessingBatch` + `GenerationBatch` (with `next()`
  returning a tuple). The exo runtime auto-detects the new API and falls
  back to plain `BatchGenerator`, logging
  `EXO_SPECULATIVE=1 but … MTPBatchGenerator is not ported yet` — Huihui
  still runs, just without the ~40% MTP throughput win. See
  `docs/fork-notes.md` for the port plan.
- **jaccl refactor revert on the mlx fork.** Upstream's #3412 breaks 2-rank
  RDMA init for our config (`ValueError: vector` from `mx.distributed.init`).
  Reverted on `adurham/mlx` main at `1cfcb5b6`. See `docs/fork-notes.md`.
