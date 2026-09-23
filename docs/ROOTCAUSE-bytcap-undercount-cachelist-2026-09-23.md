# ROOT CAUSE (CONFIRMED, 2026-09-23): byte-cap accounting can't see DSv4's KV — `CacheList` reports 0 bytes

## The bug, proven

`KVPrefixCache._total_bytes()` sizes every leaf via `_cache_object_nbytes(layer)`,
which hand-rolled the attribute walk `("keys", "values", "state")`. For
**DeepSeek-V4 every sparse layer is a `CacheList`** (see mlx-lm
`DeepseekV4Model.make_cache()`), and that walk returns **zero** for `CacheList`:

- `CacheList` exposes no `.keys` / `.values` -> `getattr` returns None -> skipped.
- `CacheList.state` returns a **list of tuples** `[(sub.state, type_name), ...]`.
  `_array_like_nbytes` sees `isinstance(list)`, iterates the items, and reads
  `.nbytes` off each item — but the items are **tuples**, which have no
  `.nbytes`, so each contributes 0.

So every DSv4 layer contributes 0 bytes, and `_total_bytes()` reports a rounding
error for what is actually gigabytes of retained KV.

### Measurement on the real structure

Built DSv4's actual cache layout (`RotatingKVCache` for ratio-0 layers, and
`CacheList(RotatingKVCache, PoolingCache, PoolingCache)` for sparse layers) and
filled it to a 273,626-token prompt:

| method | bytes | GiB |
|---|---|---|
| old `_cache_object_nbytes` (+ `_total_bytes` path) | 786,432 | 0.0007 |
| ground truth (all reachable `mx.array.nbytes`) | 706,506,752 | 0.658 |
| standard `.nbytes` (fixed) | 2,807,939,072 | 2.62 |

Undercount factor **~3,570x** (2,807,939,072 / 786,432).

Note the middle row differs from the third because the ground-truth walk only
followed a subset of attrs; the correct value is the `.nbytes` one — every
mlx-lm cache class implements `_BaseCache.nbytes`, and `CacheList.nbytes`
recurses over sub-caches (`sum(c.nbytes for c in self.caches)`).

## Consequence

The `max_bytes` ("byte cap") branch in `_evict_if_needed`:

```python
if self._max_bytes is not None:
    while self._leaves and self._total_bytes() > self._max_bytes:
        if not self._evict_lru_once(f"byte cap {self._max_bytes}"):
            break
```

can **never fire for DSv4**, because the quantity it tests is ~3,570x too small.

This explains the observed eviction log: across a whole boot with four resident
deep leaves (273,626 / 273,626 / 279,228 / 279,226 tokens), the ONLY evictions
logged were `— session cap 4`. No `byte cap` eviction ever appeared, even after
`DSV4_MAX_PREFIX_BYTES=12884901888` was set and confirmed live on the instance
(`maxPrefixBytes: 12884901888` in `/state`).

**The cap was configured, wired end-to-end, and structurally unreachable.** My
earlier note that it was merely "inert because retention is only 0.3 GB" was
also wrong — retention is 2.4-5.4 GiB per deep leaf; the accounting just
couldn't see it.

## The fix

`_cache_object_nbytes` now prefers the standard `.nbytes` property, falling back
to the legacy walk for objects that lack it (test doubles, unknown cache types):

```python
nb = getattr(cache_entry, "nbytes", None)
if nb is not None:
    try:
        return int(nb)
    except (NotImplementedError, TypeError, ValueError):
        pass
# legacy fallback ...
```

Verified: after the fix, exo's accounting equals `.nbytes` exactly (undercount
factor 1x), restoring 2.68 GiB of visibility per 273K leaf.

## Corrected sizing (this is what the cap SHOULD be)

With working accounting, one leaf costs:

| depth | leaf KV |
|---|---|
| 30K | 0.29 GiB |
| 100K | 0.96 GiB |
| 250K | 2.39 GiB |
| 273K | 2.62 GiB |
| 565K | 5.41 GiB |

With session cap 4, worst-case retention is 4 x 5.41 = **21.6 GiB at 565K** but
only 4 x 0.29 = **1.2 GiB at 30K**. So:

- `cap = 12 GiB` (my earlier value) -> **never fires** (4 deep leaves at 273K =
  9.6 GiB < 12). Too high.
- `cap = 6 GiB` -> holds ~2 deep leaves at 273K; still never fires at 250K (4 x
  2.39 = 9.6 > 6 -> fires). Small sessions (1.2 GiB) untouched.
- `cap = 2 GiB` -> holds 0-1 deep leaves; fires even at 250K.

The cap should be sized so the SMALL-session workload is unchanged (count cap
still binds: 4 x 0.29 = 1.2 GiB) while the DEEP case is bounded.

## Relation to the two user targets

- **500K collapse:** at 565K the run peaked 117.24 GB against a 115.4 GB wired
  limit. Four retained 5.41 GiB leaves = 21.6 GiB is a real contributor to that
  overshoot. Bounding retention here is a genuine headroom win.
- **250K > 30 t/s:** retention at 250K is 9.6 GiB out of a ~104 GB footprint,
  and the measured depth decline is CYCLE COST (+22.9% ms/cycle from 30K to
  273K), not memory. So this fix does NOT address the 250K gap; that needs the
  per-cycle cost investigated (indexer scoring scales with depth).
