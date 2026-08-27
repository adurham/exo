# DSpark 14K verify cliff — code-reading investigation — 2026-08-27

**STATUS: code-only investigation (cluster untouched — 352.6K protocol running).**
This doc is the code-reading + fix-proposal deliverable for task (2) of the
2026-08-27 DSpark decode optimization round. No cluster runs were performed;
all findings below are from source reading of `mlx-lm/mlx_lm/models/deepseek_v4.py`
(submodule `dda9237`) and `src/exo/worker/engines/mlx/speculative/dsv4_mtp.py`
(parent `42be34951`).

## The claim under investigation

The campaign brief states a "14K verify cliff: 1455ms @14K vs 99ms @100K —
non-monotonic (likely Metal kernel dispatch threshold)". The two numbers
come from **two different measurement regimes at two different dates**:

| Number | Source | Regime | Date |
|---|---|---|---|
| 1455.8 ms @14K (`r1_verify_fwd`) | `docs/dspark-fullblock-context-scaling-cliff-2026-08-04.md` line 86 | **FULLBLOCK per-row** verify (`EXO_DSV4_ROWSEQ_FULLBLOCK=1`) — 5 separate attention calls per verify cycle | 2026-08-04 |
| 99.0 ms @100K (per-cycle verify) | `docs/dspark-cs-profile-2026-08-26.md` line 26 | **Batched verify** (`EXO_DSV4_VERIFY_BATCH=1`, `MIN_CTX=8192`) — current production | 2026-08-26 |

**These are not apples-to-apples.** The 1455ms was measured under a regime
that is now OFF in production (FULLBLOCK default OFF; batched verify is the
promoted default per commit `42be34951`). The cs-profile doc itself notes
this lineage (line 117-121): "the `dspark-fullblock-cliff` doc measured
`r1_verify_fwd=1455.8 ms` at ~14K context … This is the same mechanism that
produced the 15.9x collapse between depth 500 and 14K in the 2026-08-04
investigation."

**The actual open question** is narrower than the brief implies: under the
*current* batched-verify regime, does a 14K-context verify cycle cost
materially more than the 99ms measured at 100K? That comparison has not
been measured under the current regime (the 352.6K protocol, still running,
will produce the 100K + 352K points but not a fresh 14K point). This doc
identifies the code-path candidates that *could* produce a non-monotonic
14K>100K cost under the current regime, and proposes a cheap env-gated
fix for the most likely one.

## Code-path map (the verify forward at small L, large context)

The DSpark verify forward for a γ=3 cycle runs `dsv4_speculative_forward`
(`dsv4_mtp.py:1419-1420`) → `model(inputs, cache)` at L=γ+1=4 rows. Inside
`DeepseekV4Model.__call__` each `DeepseekV4Block` runs attention. For the
~21 sparse layers the attention is `SparseCompressedAttention.__call__`
(`deepseek_v4.py:4660+`). The dispatch at `:4695-4721`:

```
if pooled.shape[1] == 0:                      # no pool yet → local-only SDPA
elif pooled.shape[1] <= self.indexer.index_topk:   # (default 512) compressed attn
else:                                        # sparse compressed attn (full Indexer)
```

**The Indexer** (`deepseek_v4.py:3858+`) for the sparse branch runs, per
layer per verify cycle:
1. `compressor(x, pool_cache, offset)` — builds/extends the pool (`:3874`)
2. `_dispatch_pmask(pool_cache, L_full, offset)` — row-causal pmask (`:3890`)
3. `_indexer_score` or `_indexer_score_tiled` — the score GEMM `(B,L,D)@(D,P)` (`:3908-3925`)
4. pmask apply (`:3933-3980`) — `_TAIL_PMASK` band-restricted (default ON)
5. top-K search (`:3995-4042`) — `_EXACT_TOPK` (default ON) for L≤16
6. gathered SDPA over k=512 entries (`:4749+`)

`P = context // compress_ratio`. `compress_ratios` alternates 4 and 128
across the ~21 sparse layers (`:881-893`). So:
- **ratio=4 layers** (~half): P = ctx/4. At 14K → P≈3500; at 100K → P≈25000.
  Both are >512 → sparse branch (full Indexer) at both depths.
- **ratio=128 layers** (~half): P = ctx/128. At 14K → P≈110; at 100K → P≈781.
  At 14K these are <512 → cheap compressed branch. At 100K P≈781>512 → these
  flip INTO the sparse branch somewhere between 14K and 100K.

**This is the first structural candidate for non-monotonicity**: between
14K and 100K, the ratio=128 layers cross P=512 (at ctx≈65536) and flip from
the cheap compressed branch to the expensive sparse/Indexer branch. That
would make 100K *more* expensive than 14K for those layers — the opposite
of the observed 14K>100K direction. So this dispatch boundary alone does
NOT explain a 14K>100K cliff; it predicts 100K>14K.

## Candidate mechanisms for a 14K>100K cost (non-monotonic)

Since the structural dispatch predicts 100K≥14K, a genuine 14K>100K
non-monotonicity (if it exists under the current regime) must come from a
**stateful / compile / allocator effect**, not the steady-state op count.
The code-reading candidates, ranked by likelihood:

### Candidate 1 (MOST LIKELY): `_exact_topk` params-cache thrash near a P boundary
`_exact_topk` (`deepseek_v4.py:3647-3674`) is the verify-path top-K (L≤16,
default ON). Its params array is cached by `(P, k)` in
`_EXACT_TOPK_PARAM_CACHE` (`:3491, 3660-3666`) with a **hard cap of 64
entries** — when exceeded, the cache is **cleared** (`:3665`).

`P` grows by 1 every `compress_ratio` decode tokens. For ratio=4 layers at
14K, P≈3500 and increments every 4 tokens; the verify cycle itself advances
P by γ+1=4 rows per cycle. Over a decode window the (P,k) key sweeps a
*range* of P values. If that sweep straddles the 64-entry cap, the cache
clears mid-window and the next call rebuilds `mx.array([P,k])` — a cheap
Python-side op, but the **clear happens on the cycle that crosses the
boundary**, and the kernel dispatch grid `(1024, L, B)` re-evaluates. This
is a Python-side hiccup (microseconds), not a 1455ms cliff. **Likely too
small to explain the magnitude, but is a real non-monotonic perturbation.**

### Candidate 2 (LIKELY): `_pool_storage` realloc boundary (PoolingCache step=256)
`PoolingCache.update_and_fetch` (`cache.py:1494-1516`) grows `_pool_storage`
in steps of `self.step=256` (`:1271-1332` comment block, `:1508`). When
`new_offset > _pool_storage.shape[1]`, it allocates a fresh `mx.zeros`
buffer of `current_size + max(step, delta)` and copies the old prefix
(`:1510-1512`). This is an **amortized 1-in-256 realloc**, but the realloc
cycle itself is a full-buffer allocate + copy.

For ratio=4 at 14K (P≈3500), the pool storage is sized ~3584 (14×256). For
ratio=4 at 100K (P≈25000), it's ~25088. The realloc *rate* is identical
(1 per 256 new pooled entries = 1 per 1024 decode tokens). So steady-state
realloc frequency is the same at both depths — **not** a depth-dependent
cost. BUT: a realloc lands whenever P crosses a 256 boundary, and a verify
cycle that straddles a boundary pays the copy. This is per-cycle noise,
not a sustained cliff. **Ruled out as the sustained-cliff cause.**

### Candidate 3 (POSSIBLE): MLX `shapeless=True` compile-cache first-touch
`_indexer_score` is `@partial(mx.compile, shapeless=True)` (`:3677`) — one
compiled kernel serves all P sizes, explicitly to avoid the per-P
recompile OOM documented at `:3689-3691` (~24K cached pipelines at 94K
tokens). The compressor inner kernels (`_overlap_compress_kv` `:1389`,
`:1444`, `:1474`) are also `shapeless=True`. So the **steady-state** path
is recompile-free.

The residual recompile risk is in ops that are NOT `shapeless`-wrapped.
`_dispatch_pmask` (`:660`) builds `pool_idx = mx.arange(P)` and
`query_idx = (positions+1)` — `mx.arange(P)` is shape-specialized (a
fresh array per P). Under `mx.compile` this would retrace per-P; the pmask
path is NOT compiled (it's a plain function), so each distinct P builds a
fresh `mx.arange(P)` — a cheap Python+alloc op, not a Metal recompile.
**Small, per-cycle, not a sustained cliff.**

### Candidate 4 (UNLIKELY under current regime): `_SPARSE_VERIFY_FOLD` mask rebuild
`_sparse_verify_rows_batched` (`:2384+`) folds the L verify rows into one
`(B*L,H,1,S)` SDPA. When `_DECODE_NODE_DIET` (default ON) and the mask is
the canonical 2-D bool causal local mask, it's served from
`_cached_verify_mask` (`:2354`) — a cached build, first-touch cost only.
The pooled_mask path (`:2459-2493`) builds a fresh `(B,H,L,k_dim)` mask per
call when `pooled_mask is not None`. In the verify forward `pooled_mask`
derives from pmask (`:4751-4756`), which is non-None when the row-causal
pmask is built. This is per-cycle work proportional to L·k_dim (small,
L≤16, k=512) — **not depth-dependent, not a cliff.**

### Candidate 5 (RULED OUT by direction): ratio=128 layer flip into sparse branch
As computed above, ratio=128 layers cross P=512 at ctx≈65536 — between
14K and 100K. This flips ~half the remaining sparse layers from cheap
(compressed) to expensive (sparse/Indexer) and would make 100K *more*
expensive than 14K. **Predicts 100K>14K, opposite of the observed
direction. Ruled out as the 14K>100K cause** (it may be a real 100K cost
additive on top of the baseline, but it cannot produce 14K>100K).

## Verdict on the non-monotonicity claim

**Under the current batched-verify regime, a sustained 1455ms cliff at 14K
is NOT supported by the code-reading evidence.** The 1455ms number is from
the FULLBLOCK per-row regime (OFF in current production). The candidates
that could produce a *small* non-monotonic 14K>100K perturbation under the
current regime (params-cache thrash, pool realloc boundary) are
per-cycle-magnitude (microseconds to low-milliseconds), not 15x.

**The most probable explanation is that the 1455ms@14K and 99ms@100K
numbers are simply from different regimes** (FULLBLOCK vs batched), and the
"non-monotonic" framing conflates them. A genuine same-regime 14K-vs-100K
comparison under the current batched path has not been measured; the
running 352.6K protocol will produce a 100K point but not a fresh 14K
point, so a clean same-regime A/B remains for cluster validation after the
protocol completes.

## Fix proposal

### Cheap env-gated fix (shipped this round): `_EXACT_TOPK` params-cache cap raise

The one candidate that is both (a) genuinely non-monotonic and (b) cheap to
fix is the `_EXACT_TOPK_PARAM_CACHE` 64-entry cap + clear (`:3664-3666`).
Raising the cap (env-gated, default unchanged at 64 for A/B isolation) to a
large value eliminates the mid-window clear-thrash entirely. The params
array is 2 uint32s (8 bytes); even 65536 entries is 512KB — negligible.
This is a zero-risk, env-gated knob:

- **Env:** `EXO_DSV4_EXACT_TOPK_PARAM_CAP` (default 64 = current behavior).
  Set to e.g. 65536 to disable the clear entirely.
- **Risk:** zero. The cache is a dict of 8-byte arrays; the kernel itself
  is L-cached and unaffected. The cap only existed as a naive memory guard.
- **Gain expected:** small (eliminates a per-window Python hiccup), NOT a
  1455ms fix — but it's the only candidate with a concrete code site and a
  clean env gate.

**Implemented:** see the submodule diff in this round's commit (env-gated,
default OFF via `EXO_DSV4_EXACT_TOPK_PARAM_CAP` unset = 64 = unchanged).

### The real fix (requires cluster validation, NOT shipped this round)

If a same-regime 14K>100K cliff is confirmed after the 352.6K protocol, the
next experiment is a **targeted 14K-vs-100K same-regime A/B** with the
section-time harness (`EXO_DSV4_SECTION_TIME=1`) to localize which sub-op
spikes. The code-reading above says the likely spike site is the
`_indexer_score` GEMM for ratio=4 layers at the P≈3500 boundary IF (and
only if) the `shapeless=True` compile is not actually serving that P
(which would be an MLX-internal regression, not a fork bug). That experiment
needs the cluster.

## What remains for cluster validation (after 352.6K protocol)

1. **Same-regime 14K-vs-100K A/B** under the current batched verify config
   (`EXO_DSV4_VERIFY_BATCH=1`, `MIN_CTX=8192`), `EXO_DSV4_SECTION_TIME=1`
   to get per-sub-op timing. If 14K verify >100K verify under this regime,
   the cliff is real and the section-time data localizes it.
2. **If a cliff is confirmed at 14K under batched verify**, bisect with the
   `EXO_DSV4_NOP_SPARSE_LAYERS` per-layer NOP toggle (`:4722-4745`) to find
   which layer(s) spike — the ratio=4 layers are the prime suspects.
3. **Validate the `EXO_DSV4_EXACT_TOPK_PARAM_CAP` fix**: A/B with the cap
   raised to 65536 vs default 64 at the confirmed-cliff depth.
4. **Re-measure the 352.6K verify cost** the protocol is collecting — if
   352K verify is WORSE than 100K (non-monotonic at the top end), that's a
   different cliff (memory pressure / RDMA contention, per the GLM consult's
   352K regression-risk list) and points at the ratio=128 sparse-branch flip
   (Candidate 5) which IS expected to add cost past 65K.

## Sources (code locations, all read this session)

- `mlx-lm/mlx_lm/models/deepseek_v4.py`:
  - `SparseCompressedAttention.__call__` dispatch `:4695-4721`
  - `Indexer.__call__` `:3858-4042` (score GEMM `:3908`, pmask `:3933-3980`, topk `:3995-4042`)
  - `_indexer_score` (shapeless compiled) `:3677-3743`; `_indexer_score_tiled` `:3763-3810`
  - `_exact_topk` + `_EXACT_TOPK_PARAM_CACHE` (64-cap clear) `:3491, 3647-3674`
  - `_sparse_verify_rows_batched` (fold path) `:2384-2523`
  - `_SPARSE_SDPA_TILE` `:316`, `_INDEXER_PBLOCK` `:327`
  - `_VERIFY_BATCH` / `_VERIFY_BATCH_CTX` / `MIN_CTX=8192` `:1642-1677`
  - `_LMHEAD_LASTROW` + `_LMHEAD_LASTROW_MIN_L=32` `:329-334, 7063-7082`
  - `compress_ratios` default `[4 if i%2 else 128 ...]` `:881-893`; `index_topk=512` `:864`
- `mlx-lm/mlx_lm/models/cache.py`:
  - `PoolingCache.update_and_fetch` (step=256 growth) `:1494-1516`
  - `accumulate_windows` `:1429-1492`
- `docs/dspark-fullblock-context-scaling-cliff-2026-08-04.md` (1455ms@14K, FULLBLOCK regime)
- `docs/dspark-cs-profile-2026-08-26.md` (99ms@100K, batched-verify regime, `:117-121` lineage)
- `docs/prefill-cliff-mechanism-2026-08-24.md` (prefill-side cliff, separate mechanism)