# DSv4.1 park/restore — rank-continuity fix (P20) — buildnote (2026-10-11)

Branch: `p20/dsv41-restart-rankfix` (off `origin/deploy/next21-restart` @ 3afce11d7).
Scope: the `.r<rank>`-keyed-dir vs launch-order-rank defect found during the live
restart-survival A/B. Fully offline: no cluster contact, no node ssh, no boots.

## The defect (from the live run log)

After a graceful relaunch the two nodes' TP **device ranks swap** (ranks are
assigned by launch order and the node ids regenerate), so every parked dir —
tagged `.r<rank>` — was refused on both nodes
(`park.skip reason=foreign-rank` x2, then `park.restore cold reason=rank-empty`),
and each post-relaunch request paid the full cold prefill (~265 s for 69K rows).
The rank gate (`park.py:_rank_matches`) ran **before** the fingerprint/LCP gates.
G4's all-or-nothing held (both ranks cold, never mixed) but the feature delivered
zero benefit on any relaunch that permuted ranks.

## Design + rationale

The fix has three parts, all keyed on the observation that the parked KV cache
is **rank-INDEPENDENT**, so the launch-order rank is not a cache-validity axis.

1. **Stable SITE identity replaces the rank gate.**
   `park.site_identity()` = `EXO_DSV41_SITE_ID` (stripped) → else
   `socket.gethostname()` → else `None`. Each `ParkedStore` carries `self.site`;
   the manifest records `site`. A new `_identity_matches` admits a dir iff the
   manifest's `site` equals the store's `site` — a logical node label that
   survives a rank permutation, unlike `.r<rank>`. `_rank_matches` is retained as
   the **legacy fallback** for dirs with no `site` field.
   - Fail-safe is absolute: site present but unequal, or site present but the
     store's identity unknown (`None`) → **refuse (COLD)**. Never admit by
     guessing. A rank-tag mismatch is never used to *accept* a foreign site.

2. **Rank-invariant park layout.** The engine no longer passes `rank` into the
   park fingerprint layout (`engine.py`), and `_layout_fp_from` no longer
   consults a model's `rank` when a layout mapping is supplied
   (`fingerprint.py`). `layout_fingerprint()` itself is unchanged (still
   rank-sensitive). This is what makes a swapped-rank dir pass the G2 `_admit`
   gate; every other layout axis (world_size, layers, dtype, window, bsz) is still
   gated.

3. **Startup identity agreement (G4-style).** A single `MAGIC_IDENTITY` exchange
   (new: `coord.encode/decode_identity_row`, pure `decide_identity`,
   `ParkCoordinator.identity_confirmed`, cached) runs **once, first**, at a
   program point every rank reaches (`coordinated_restore`, before the RESTORE
   exchange). `ok` requires every rank to report a well-formed row, a KNOWN
   (non-zero) site tag, and the SAME `world`; otherwise the cluster goes COLD
   identically. `decide_identity` is pure/rank-invariant, so the exchange count is
   rank-invariant and no rank branches on a value a peer cannot see. Duplicate
   sites are tolerated on purpose (a single host / single-machine test may host
   both ranks).

### Why restoring a swapped-rank dir is content-safe (the load-bearing claim)

DeepSeek's MLA keeps **one shared KV latent per token**, and DSv4.1's in-loader
TP shards weights (routed experts, 128-wide dense/Hadamard blocks, vocab head,
`wo_a`/`wq_b`/`wo_b`) — **not** the cache:

- `mlx-lm/mlx_lm/models/deepseek_v41/cache.py`: `win_kv`/`comp_kv` are
  `(bsz, …, head_dim)`, `index_k` is `(bsz, n_comp, index_head_dim)` — a single
  head_dim latent per position, not `n_heads*head_dim`.
- `exl3_build.build_block`: the `attn_tp` slice set is `{attn.wq_b, attn.wo_b}`
  plus per-head `attn.wo_a`; `attn.wkv` (the source of the cached latent) is
  built **full** on both ranks (and `attn.group` all-sums only the attention
  *output*, never the cache).

So each rank's parked cache holds identical bytes regardless of its rank; only
the weights (rebuilt per current rank at load) differ. Hence admitting a dir
written under a different rank on the same physical node is content-correct. This
is the assumption the fix relies on; it is pinned by
`test_park_layout_is_rank_invariant` plus the bit-exact swapped restore.

### Legacy `.r<rank>` dirs (44+/node) — treatment

Not migrated, not rewritten, not deleted (no destructive migration on a path that
must never corrupt a parked shard). A dir with **no** `site` field keeps the
pre-existing rank rule: `test_legacy_no_site_dir_keeps_the_rank_rule` documents
that a matching rank is admitted and a foreign rank refused — i.e. such dirs are
**cold-accepted** after a rank swap exactly as today. New dirs (written by this
build) carry `site` and restore across a swap. The first post-deploy relaunch
therefore still cold-restores the pre-fix dirs and only thereafter benefits.

### G2 / G5 preserved

G2: model/tokenizer/kv-code/mlx identity is compared exactly as before; only the
`rank` axis is dropped from `layout_fp` (see above). G5: restore stays
NON-CONSUMING (dir kept, marked `.active`); fsync/tmp→rename/retention unchanged;
`_identity_matches` runs before `_admit` at all three local admission sites.

## Files changed

- `src/exo/worker/engines/mlx/dsv41/park.py` — `ENV_SITE`/`site_identity()`,
  `ParkedStore(site=…)`, manifest `site` (written only when known), `site`
  threaded through `_GenerationWriter`/`park_conversation`/`park`/
  `write_generation`/`open_generation`/`checkpoint_conversation`, new
  `_identity_matches` at the 3 admission sites (`foreign-site` vs legacy
  `foreign-rank` logging), identity gate in `coordinated_restore`.
- `src/exo/worker/engines/mlx/dsv41/coord.py` — `MAGIC_IDENTITY`,
  `IDENTITY_WIDTH`, `site_tag`, `encode/decode_identity_row`, `IdentityDecision`,
  pure `decide_identity`, `ParkCoordinator.identity_confirmed`.
- `src/exo/worker/engines/mlx/dsv41/engine.py` — park layout omits `rank`.
- `src/exo/worker/engines/mlx/dsv41/fingerprint.py` — `_layout_fp_from` ignores
  rank when a layout mapping is given.
- `src/exo/worker/engines/mlx/dsv41/tests/test_dsv41_park_g4.py` — +11 CPU-only
  tests (swap both directions bit-exact; site mismatch → cold both; no-swap
  unchanged & exchange counts symmetric; one rank missing → cold both; ambiguous
  same-gen different-digest → cold; `decide_identity` unit cases + rank-order
  invariance; `site_tag`; foreign-site refused locally; legacy no-site keeps the
  rank rule; park layout rank-invariant; `site_identity` env→hostname).

## Test evidence

Command (from the worktree):

```
PYTHONPATH=<wt>/src:<wt>/mlx-lm EXO_DASHBOARD_DIR=<live>/dashboard/build \
  ~/repos/exo/.venv/bin/python -m pytest src/exo/worker/engines/mlx/dsv41/tests -q
```

- Baseline at construct tip (3afce11d7): **354 passed**.
- With the fix: **365 passed** (354 + 11 new), locally reproduced by the PM on the
  committed tree.

## Adversarial audit

See the "Adversarial audit findings" section appended below (independent audit of
wrong-identity mapping, startup-agreement race/order, partial restore, and the
rank-independence claim).

## Deferred / not done (out of scope here)

- Live re-validation on the cluster (this work is offline by constraint; the
  deploy + A/B remains the orchestrator's call).
- Promoting the fix onto the live construct / launcher allow-list (no new env is
  required: the default is the hostname, which the launcher already forwards as
  the process env — `EXO_DSV41_SITE_ID` is only an override).
