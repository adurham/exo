# DSv4.1 P1 — indexer query-band split: buildnote (2026-10-11)

**Campaign:** P20 indexer-qsplit, STEP-1 (BUILD). **Status:** built, tested,
pushed; **NOT deployed** (deployment is user-gated and later). Gate-independent:
this work does not depend on the STEP-0 indexer-share reconciliation outcome.

- **exo branch:** `p20/dsv41-indexer-qsplit` (off `deploy/next22-rankfix` @
  `ade481853`), worktree `/private/tmp/p20-idxsplit-wt`.
- **mlx-lm branch:** `perf/dsv41-indexer-qsplit` @ `f728bb6810da763b3b63b1be05aa20d59e210ff8`
  (off `689e4eafd3cf2925a1cf78ba81375a3c098a0eef`), pushed to `adurham/mlx-lm`.
- **Design of record:** `docs/dsv41-attention-sharding-scope-2026-10-11.md`
  §2 / §3b / §5 (branch `p20/attention-sharding-scope` @ `0adf3f0fd`).

## What was built

The DSv4.1 indexer (`mlx_lm/models/deepseek_v41/indexer.py::Indexer.__call__`,
call site `attention.py:197`) is fully **replicated** and **O(nb)**. P1 splits its
**query side** across the 2 tensor-parallel ranks:

- each rank scores / top-ks / emits the rows `[rank*n/2, (rank+1)*n/2)` through
  the **whole** indexer (hierarchical, tiled and untiled branches all banded);
- the **KEY side stays FULL** on every rank (`index_k`, the compressor, `wkv`,
  `wq_a`, the rope/cos-sin tables) — key coherence;
- one small **`cidx` `[b, n, index_topk=512]` int32 exchange** reconstructs the
  full selection on both ranks, so `sparse_attn` and the `shared.candidates`
  consumer are **unchanged**.

### Exactness argument

Per-row top-k is row-independent: the softmax/relu normalization is over **keys**,
never queries, and every score element is a dot of one query row with one shared
key. So the band's `cidx` equals the full run's row slice. The reconstruction is a
zero-padded `all_sum` over the TP group — the two bands cover disjoint row ranges,
so each row has exactly one non-zero contributor and `0 + x == x` elementwise
(exact for int32, including the `-1` masked sentinel). This is the v4-default
seq-split path (`deepseek_v4.py:5777-5796`), chosen over a bare `all_gather`
(`exl3_build.py:390-392`: the JACCL mesh rejects some `all_gather` sizes;
`int32` `all_sum` is supported — the jaccl `Dtype::Int32` reducer). **The MoE
`all_sum` path is untouched; P2 (SP-hybrid) was not built.**

### Source ↔ consumer band coupling

The band is computed once from `(n, self.group)` and is therefore **identical at
every index-source layer** of a forward. Layer 20 (candidate source) publishes its
candidate mask **produced banded** (rows `[lo,hi)`, shape `[b, n/2, nb]`); consumers
24/28/32/36 run the **same** band and read exactly those rows. No mask exchange is
needed and the mask is never full-then-sliced.

### History hazard (B>1)

The band is a slice of the **QUERY axis (axis 1) only** — batch is never folded
into rows — so a `b > 1` forward bands each batch identically and the padded
exchange preserves axis 0. That is the guard against the historical seq-split
"batch-unsafe" corruption (`docs/PERFORMANCE_HISTORY.md` §"seq-split all_gather
batch-unsafe bug", 2026-06-24). A dedicated `b=2` test is included.

### Gating + launcher

- Env flag **`DSV41_INDEXER_QSPLIT`** (default OFF), read once at import
  (`indexer.py`), matching the existing `DSV41_INDEXER_*` family. The **effective
  value is logged once to stderr** (`[DSV41] indexer qsplit: DSV41_INDEXER_QSPLIT=0
  (effective at import)`) — stderr, because the runner's root logger filters
  `mlx_lm` INFO records.
- `start_cluster.sh` forwards it inside the `DSV41_INDEXER_*` allow-list block
  (after the `DSV41_INDEXER_SMALLN_ROW_BF16` line). Audited `grep -rn
  DSV41_INDEXER_QSPLIT` across repo + shell rc: **no stale value**.
- **Decode / small-n stays replicated**: `query_band` returns `None` for
  `n <= _FENCE_MIN_ROWS` (16), `world < 2`, no group, or `n % world != 0`.
- **Flag-off byte-identity**: with the gate off the call site passes `band=None`
  (never a collective); the `band is None` branch collapses to the original
  expressions. The whole existing dsv41 suite runs on that path and is unchanged
  (see below).

## Files + line ranges

mlx-lm (`689e4eafd..f728bb681`):

| file | change |
|---|---|
| `mlx_lm/models/deepseek_v41/indexer.py` | `_DistributedGroup` Protocol shim (L143-156); `DSV41_INDEXER_QSPLIT` read + one-time stderr log (L162-173); `query_band` (L176-198); `_reconstruct_rows` / `_finish_rows` (L200-226); `Indexer.__call__` band branch + `n_full` (L646-680); the three return sites wrapped in `_finish_rows` (L742, L765, L792) |
| `mlx_lm/models/deepseek_v41/attention.py` | import `query_band` (L41); compute `band = query_band(n, self.group)` and pass `band=/group=` to the indexer (L193-207) |
| `tests/test_dsv41_indexer_qsplit.py` | new, 301 lines, 15 tests |

exo (`87d0bb12d`, plus the docs commit):

| file | change |
|---|---|
| `mlx-lm` (gitlink) | `689e4eafd` → `f728bb681` |
| `uv.lock` | pin advanced surgically to the same sha (line 1501) |
| `start_cluster.sh` | `DSV41_INDEXER_QSPLIT` forwarding line (~L2713-2723) |
| `docs/dsv41-indexer-qsplit-buildnote-2026-10-11.md` | this note |
| `docs/dsv41-indexer-qsplit-deploy-readiness-2026-10-11.md` | deploy-readiness note |

## Tests + sabotage

`tests/test_dsv41_indexer_qsplit.py` — **15 passed** (scoped):

1. gate shape (world=2 rank 0/1 bands; `None` when gate off / no group / world 1 /
   `n <= 16` / uneven `n`);
2. row-independence + reconstruction on the SOURCE role and a plain role, on both
   the hierarchical and tiled branches, incl. uneven `nb` — asserts
   `np.array_equal(reconstruction, full)` and per-rank pads equal the full run's
   slice, top-k overlap == 1.0;
3. `b=2` batch-safety (hier + tiled);
4. negative-sentinel (`-1`) survival through the exchange;
5. flag-off identity (`query_band` is `None` for every shape; `band=None` never
   issues a collective — the `all_sum` is spied to raise).

**Sabotage (RED-proof):** changing the band's rope base to `mx.arange(start_pos,
start_pos + (hi-lo))` (the band-relative-vs-absolute coordinate-frame bug class)
makes **6 of 15 fail** ("reconstruction != full", "rank N band != full slice");
restored clean.

**Regression control (dsv41 suite, scoped, `-k dsv41`, ignoring the pre-existing
sibling-import file):** BEFORE (base sources) 11 failed / **268 passed**; AFTER
(this branch) 11 failed / **283 passed** (= 268 + the 15 new). The same 11 fail
either way — pre-existing local-mlx precision artifacts, not this change. Zero
regressions.

## Lint — BEFORE vs AFTER (basedpyright default mode, mlx-lm)

| file | BEFORE | AFTER |
|---|---|---|
| `indexer.py` | 82 | 83 |
| `attention.py` | 46 | 46 |
| `tests/test_dsv41_indexer_qsplit.py` | (new) | 211 |

`ruff check`: **All checks passed** on all three (BEFORE and AFTER). The single
indexer +1 is `Type of "all_sum" is partially unknown` on the new `_coll.all_sum`
call — the **identical** diagnostic the package already reports at all 9 existing
`_coll.all_sum` call sites (`attention.py`, `exl3_build.py` ×3, `moe.py` ×2,
`mtp.py`, `sampling.py`); the `_DistributedGroup` Protocol removed the ~25
otherwise-new `Unknown`-propagation errors. The test file's errors are the
`IX._QSPLIT` / `IX._HIER` private-access pattern every sibling dsv41 test uses
(control: `test_dsv41_hier_integration.py` = 413 errors).

## NOT verified

- **Not launched, not A/B'd, not quality-gated.** No boot, no node contact, no
  generation request — offline build only. The live A/B (flag ON vs OFF on
  identical shapes, `[DSV41] turn reuse:` / prefill-controls lines matched) and
  the live quality battery are LATER, user-gated.
- **The `int32` `all_sum` has not run on the real 2-rank JACCL mesh** (verified
  statically against `mlx/distributed/jaccl/`: `Dtype::Int32` → `int32_t` `SumOp`).
- **The banded candidate-mask path has not run end-to-end in a real 2-rank
  forward** (unit-level: source publish + consumer read on one band are covered by
  construction + the row-independence test).
- basedpyright ran in mlx-lm's default mode (no `pyrightconfig`); mlx-lm does not
  configure a basedpyright gate (the submodule carries 82 pre-existing errors).
- Per the chain-build convention (checked against `p20/dsv41-restart-g3`), a
  `PERFORMANCE_HISTORY.md` entry is **not** added for a gated, unmeasured build;
  the measured entry lands with the live A/B + battery.
