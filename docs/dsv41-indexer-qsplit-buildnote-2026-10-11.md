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

## N-node implications

The band split is **parametric in rank count by construction** — it generalizes in
shape to any N (verified against the shipped code at `f728bb681`):

- `query_band` computes `world = group.size()` (`indexer.py:193`), `rows = n // world`
  (`indexer.py:196`), `lo = group.rank() * rows` (`indexer.py:197`) → band
  `(lo, lo + rows)` (`indexer.py:198`). Nothing is hard-coded to 2 ranks.
- The reconstruction is an **N-way zero-padded `all_sum` on the full query axis**:
  `full = mx.zeros((banded.shape[0], n_full, *banded.shape[2:]))`, then
  `full[:, lo:hi, :] = banded`, then `_coll.all_sum(full, group=group)`
  (`indexer.py:211-214`, `_reconstruct_rows`); driven from `_finish_rows`
  (`indexer.py:217-223`) at each return site (`indexer.py:745,768,795`). The pad is
  applied to axis 1 (query) only, so **batch (axis 0) is preserved** — the exchange
  is batch-safe for any N.
- The collect payload per layer is the full `[b, n, k]` int32 (`k = index_topk = 512`),
  i.e. ~4.19 MB/layer at `n=2048`, **regardless of N**; the collective's own cost grows
  with N, as any allreduce.

**Caveat (honest).** The gate engages only when `n % world == 0` and
`n > _FENCE_MIN_ROWS` (`_FENCE_MIN_ROWS = 16`, `_gates.py:43`); a non-zero remainder
returns `None` → replicated fallback (`indexer.py:194`). At `w=2` both production chunk
sizes (2048 low-ctx / 128 high-ctx — `start_cluster.sh:102,142`) divide evenly, so the
split engages. At **`w=3`** `2048 % 3 = 2` and `128 % 3 = 2` → `query_band` returns
`None` on every forward, so the P1 split **does not engage** and the indexer stays
replicated. As-shipped, P1 is therefore w=2-only in practice; generalizing to w=3 needs
a **query-axis pad to a multiple of `world`** (or a chunk size divisible by 3). The
N-way collective machinery itself already supports any N.

## TP3 relevance

The TP3 study's flat-to-worse verdict at w=3
(`docs/dsv41-tp3-scaling-readiness-2026-10-11.md`, branch `p20/tp3-scaling-readiness`
@ `11fe07ff7`) was dominated by the **replicated indexer at depth** — the O(context)
term a 3rd rank cannot split under the current design. `attn.indexer.score` rises from
3.6% of a 16K chunk to **42.9 / 44.2%** at ~340K (that doc's M3/M4), and with the indexer
replicated the shardable fraction `S ≈ 0.31` (experts only) models the band at
**0.78×–0.86×** (a regression). P1 removes the 2-rank duplication of that dominant
replicated term (`indexer.py:196-197`), and — per the section above — its `all_sum`
reconstruction is already N-way by shape (`indexer.py:211-214`), so P1 is the
**precondition that de-risks** a future 3-node case.

**P1 is necessary, not sufficient.** 3-node *support* still requires the separate TP3
readiness items from that doc: transport N>2 — the ARQ/reliable layer is hard-gated
`size_==2` and *throws* for N>2 (`mesh_impl.h:706-709,1171-1175,1590`); the `w=3` raise
tensors — `_block_bounds` `n % (8·world)` (`exl3_build.py:324`; `attn.wq_b` 2048t /
`attn.wo_b` 512t / `head` 8080t all RAISE) and `n_heads % world or n_groups % world`
(`exl3_build.py:703`; 64%3 and 8%3); and the launcher hardcodes (~15 two-node
assumptions in `start_cluster.sh`). P1 clears the dominant *performance* blocker; the
transport / geometry / launcher prerequisites remain open.

## Post-audit fixes (2026-10-11)

Adversarial audit (STEP-3) of mlx-lm `perf/dsv41-indexer-qsplit` @ `f728bb6` /
exo `p20/dsv41-indexer-qsplit` @ `01ddc7bf3` — report
`~/.hermes/cache/scratch/p20-idxsplit-audit-report.md`. The band math itself was
confirmed exact (reconstruction byte-exact, batch-safe, source↔consumer on one
band, flag-off value-identical); the audit found **two defects** and three
mutation-coverage gaps. All are fixed here.

**mlx-lm** `perf/dsv41-indexer-qsplit`: `f728bb681` → `e03406282` (new tip).

### D1 (HIGH) — attention.py call signature

The new `band=`/`group=` keywords were passed **unconditionally**, so with
`DSV41_INDEXER_QSPLIT` OFF the call still bound the keyword and raised
`TypeError` against a **fixed positional-signature** monkeypatch of
`Indexer.__call__` — the supported `DSV41_NEXT18_CAPTURE` capture hook
(`bench/next18_capture.py:222`), installed on every rank when that env is set
(`dsv41/engine.py`, forwarded by `start_cluster.sh`). The flag-off call now passes
**no band keyword at all** (the literal base call); the kwargs are passed only
when `band is not None` (`attention.py:196-218`).

### D2 (MEDIUM) — `query_band` fences on the BAND row count

`query_band` gated on the FULL `n` (`n <= _FENCE_MIN_ROWS`) while the indexer
routes on the BAND's row count (`indexer.py` `n = qpos.shape[0]`, driving both
`_HIER and n > _FENCE_MIN_ROWS` and `_row_dtype(n)`). At world=2 with
`_FENCE_MIN_ROWS < n_full <= _FENCE_MIN_ROWS * world` (17..32) each rank's band
(9..16 rows) took the small-n fp32 fallback while the replicated run took the
hier/bf16 path, so the "band == full slice, byte-exact" invariant broke (not
corruption: both ranks share the regime, and the fp32 row is the more precise
one). The fence now applies to the band's own rows
(`n % world or n // world <= _FENCE_MIN_ROWS`, `indexer.py:190-198`), so a band
engages only where the replicated run would have taken the same path.

### Coverage fills (the audit's mutation escapes)

- **M5** (`lens` band-relative while the rope stays absolute) — every case used
  `sp = 2*nb` (all columns visible), so `lens` was inert. Added a causal-frame
  case (`sp=0`, `nb=n/2`).
- **M10** (untiled return site not wrapped) — the suite only forced tiling or
  the hierarchy. Added a banded **untiled** case (which fails with `KeyError
  'padded'` if the wrap is dropped).
- **M12** (attention call site never bands) — only `Indexer`/`query_band` were
  called directly. Added an **Attention-forward** test asserting the band
  (`(0, 20)` at n=40, world=2) is actually passed, plus the D1 fixed-arity
  wrapper test driven through a real Attention forward.

`tests/test_dsv41_indexer_qsplit.py` 15 → **21**; `ruff` clean. All five
sabotages (D1, D2, M5, M10, M12) are **CAUGHT (RED)** when re-applied.

> Note (out of lane, flagged): `tests/test_dsv41_indexer_bf16_row.py` ends with
> an `importlib.reload(indexer)`, leaving a *second* `Indexer` class bound while
> `attention` still holds the first. The new Attention-level tests therefore
> patch `type(att.indexer)`, not `IX.Indexer`. Not fixed here — a future cleanup
> could make that test restore the module.

### exo pins + alignment

The `mlx-lm` gitlink and the `uv.lock` pin (`uv.lock:1501`) both advance to the
same sha; the alignment guard
(`src/exo/shared/tests/test_submodule_lock_pin_alignment.py`) is **green**
(2 passed).

> **Advisory (audit I1, unresolved — offline):** the `uv.lock` pin reads
> `?branch=main#<sha>` while `<sha>` lives only on the feature branch
> `perf/dsv41-indexer-qsplit` (precedent: next11's pin `d740764`). uv resolves a
> locked git source by the **exact rev**, and the commit is pushed/reachable, so
> it is fetchable; only CI/uv2nix fetchers that verify `rev ∈ ref` are at risk
> (deploy force-reinstalls from the submodule, so the lock does not drive the
> cluster). Re-pin to the **merge sha** when this lands on `main`.

### Bench impact

None. Bench shapes are n=2048 (far above the fence) so neither fix changes the
band geometry there — **no re-bench**.


---

## Post-mortem remediation (token-identity divergence)

Source: live A/B post-mortem `docs/dsv41-indexer-qsplit-live-ab-postmortem-2026-10-11.md`
(origin/main `01fd85f33`): `identity_3` (2126 tok, chunks 2048 + 78) diverged at token
4 with the flag ON; only chunk 2 (even n=78 -> bands of 39) engaged the band.
Fixed offline on the SAME (never-shipped) branches, in place: mlx-lm
`perf/dsv41-indexer-qsplit` `e0340628` -> `de6caa2e` -> **`6a5c78f`** (`de6caa2e` =
token-identity; `6a5c78f` = the residual **D3** route fix, below); exo
`p20/dsv41-indexer-qsplit` (this commit) advances the gitlink + `uv.lock:1501` to
`6a5c78f`. 0 boots, 0 deploys, 0 generation requests.

### Root cause (neither named hypothesis)

`mlx_lm/models/deepseek_v41/indexer.py` (pre-fix `e0340628`, L678-692): the band
branch sliced `x`/`qr` to the band rows **before** the query projections,
`q = self.wq_b(qr[:, lo:hi])` (L687) and `w = self.weights_proj(x[:, lo:hi])` (L692).
Neither GEMM is row-count invariant:

- production `indexer.wq_b` is an `EXL3Linear` (fused with `attn.wq_b`,
  `exl3_build.py:_GROUPS`) whose kernel is picked by the call's row count
  (`mlx_lm/models/exl3/exl3_linear.py` `__call__`: rows<=16 simd GEMM,
  rows<=`FUSED_GEMM_ROW_LIMIT`=64 fused trellis GEMM, else `decode_full` + native
  matmul). Band 39 rows -> fused GEMM; full 78 rows -> decode_full + matmul;
- MLX native bf16 matmul (`weights_proj`, 4096->64) is also M-dependent.

So the same query row got a different `q`/`w` banded vs replicated; fp4 fake-quant
(`fake_quant_fp4_ue8m0`) amplifies a 1-ulp difference into a scale-step change, and
the k=512 boundary flips. Single-chunk 2048 is exposed too (band 1024 vs 2048:
`weights_proj` 41 rows differ); n=64 fp32 `nn.Linear` (the old suite) is M-invariant,
which is why 21 tests stayed green.

Decisive experiments (production indexer geometry: q_lora 1024 -> EXL3 wq_b 8192,
64 heads x 128, k=512, ratio 4, start_pos=2048, n=78, nb=531; harness
`~/.hermes/cache/scratch/p20-idxsplit-fix2/`):

| measurement (pre-fix) | result |
|---|---|
| band-vs-full intermediates, seed 0 | `wq_b` out: 2027 elems / 78 rows differ (max 0.5); fp4 q: 9 elems / 8 rows (max 16); score: 2035 elems / 8 rows; `weights_proj` 0 |
| same at n=2048, sp=0 | `wq_b` 0; `weights_proj` 41 rows; score 4713 elems / 40 rows |
| EXL3Linear alone, rows n vs n/2 | n=34/64/130/256/2048: 0 diff; **n=66/78/128 (band crosses 64): 6460/7207/13682 diff** |
| selection sweep, 32 seeds | **8/32 seeds diverge, 12 rows** (e.g. seed 4 row 38, 1 swapped slot) |
| score expressions with IDENTICAL q/w (shared fp32/bf16, gathered fp32), n=78 & 2048 | **0 diff** -> routing/score math is M-invariant |

Knob sweep on the 8 diverging seeds (pre-fix, one knob each):

| knob | diverging seeds |
|---|---|
| baseline | 8/8 |
| `EXL3_FUSED_ROW_LIMIT=16` (39 & 78 both decode_full) | **0/8** |
| `EXL3_FUSED_ROW_LIMIT=100` (39 & 78 both fused) | **0/8** |
| `DSV41_INDEXER_HIER=0` (tiled) | 4/8 |
| `HIER=0 TILE=0` (untiled) | 4/8 |
| `HIER=0 TILE_FORCE=1` | 4/8 |
| `HIER=0 ROW_BF16=0` | 8/8 |
| `L2_FULL=0` | 8/8 |
| `COMPILE=0` | 8/8 |

Adjudication: **(a) position/offset/mask frame — refuted** (qpos/lens/rope/offset are
absolute; with q/w equalised every band reproduces the full selection, the candidate
mask, and the consumer layer, incl. `start_pos=2048/16384`). **(b) as stated
(tiled/untiled/tile width/row dtype/compiled body keyed on band n) — refuted**: no
indexer routing knob clears it, and the score expressions are bitwise M-invariant.
The actual cause is (b)'s *class* (shape-dependent rounding) one stage earlier: the
query **projection** GEMMs, selected by row count. Only the EXL3 row-limit knob,
which makes both row counts take the same kernel, clears it.

### Fix (mlx-lm `de6caa2e`; residual D3 at `6a5c78f`)

`Indexer.__call__` runs `wq_b(qr)` and `weights_proj(x)` on all `n_full` rows and
slices the band **after** the projections. Everything downstream (rope at absolute
`qpos`, fp4 quant, score, top-k, candidate mask) is row-local and verified
M-invariant, so a band row is bitwise the replicated row -> identical selection,
identical argpartition tie-break (same op on identical inputs, same `[nb]` row
shape), identical downstream outputs. `band=None` is unchanged (flag-OFF
byte-identical). No tie-rule or rescoring change was needed.

### Repro before / after

| check | pre-fix `e0340628` | post-fix `de6caa2e` |
|---|---|---|
| identity_3 geometry, real EXL3 wq_b, 32 seeds | 8/32 diverge (12 rows) | **0/32** |
| source(L20)->consumer pair, n=2048 sp=16384 nb=4608, 2 seeds | seed0: src 2 / consumer 2 rows; seed1: src 3 / consumer 2 rows diverge | **0** (cand mask equal) |
| margin-0 tie class at k=512 (1/3 keys duplicated), hier + untiled, n=78/2048 | — | **0** |

### Tests (mlx-lm `tests/test_dsv41_indexer_qsplit.py` 21 -> **34**)

`NonFirstChunkBandTest` (8: chunk-2 n=78@sp=2048 with row-count-sensitive
wq_b / weights_proj on hier + tiled, even band rows n=80, plain role, even/odd
boundary gate for 3/129/79/78/80/2048/34, structural "projections see n_full rows"),
`MarginZeroTieTest` (3), `ProductionExl3ChunkTwoTest` (real EXL3 wq_b, identity_3
geometry, seed 4), `MultiChunkAttentionTest` (2-chunk prompt 40+78 through real
`Attention.__call__`, 2-rank simulated exchange: cidx AND attention output bitwise
equal on both chunks). ruff clean.

Sabotage (mutation applied to the fixed `indexer.py`, suite re-run, file restored):

| mutation | result |
|---|---|
| M1 full revert (project band rows only) | **10 RED** / 24 pass |
| M2 `wq_b` on band rows only | **8 RED** (incl. real-EXL3 identity_3, 2-chunk attention) |
| M3 `weights_proj` on band rows only | **5 RED** |
| restored | 34 passed |

Margin-0 tie controls `test_ties_hier` / `test_ties_untiled` are green pre- and
post-fix (expected: with identical inputs the tie-break is identical); their RED
proof is `test_ties_rowcount_sensitive`.

### Suites

- mlx-lm qsplit: **35 passed** (21 -> 34 -> 35; `6a5c78f` adds `D3TiledRouteTest`).
- exo dsv41 `src/exo/worker/engines/mlx/dsv41/tests/`: **367 passed**.
- alignment guard `src/exo/shared/tests/test_submodule_lock_pin_alignment.py`: green after commit (see commit log).
- Out of lane, pre-existing (identical at `e0340628`, not touched): sibling mlx-lm
  suites `test_dsv41_indexer_smallm_hier` (5), `test_dsv41_hier_integration` (2),
  `test_dsv41_consumer_skip` (1), `test_dsv41_indexer_bf16_row` (1) fail standalone
  on this host both before and after this fix.

### D3 (residual, audit 2) — fixed at mlx-lm `6a5c78f`

Audit 2 (`~/.hermes/cache/scratch/p20-idxsplit-audit2/REPORT.md`) found the
token-identity fix left one gap on the **non-default `DSV41_INDEXER_HIER=0`**
path. `indexer.py:716` keyed the tiled/untiled ROUTE on the **band's** row count,
so the replicated run (n_full rows) could take TILED while each band (n rows) took
UNTILED; the two paths are not equivalent on consumer layers (L1: untiled has no
`isfinite` guard), so the band's consumer selection diverged (audit sweep 5/84
shapes; prod h=64 b=1 n=78, nb in [6670,13339] ~ start_pos 26.6K-53K). Fixed by
`tile = tiled(bsz, n_full, ...)` — the audit's verified control (0/84; re-confirmed
on the fixed tree, `sweep h0_default`: cases=84 diverging=0). `band is None`
(flag OFF) is unchanged, and HIER=1 returns in the hier branch **before** the
line-716 consumer, so neither the flag-off default nor the re-bench geometry moves.

`tests/test_dsv41_indexer_qsplit.py` 34 -> **35** (`D3TiledRouteTest`, production
tile geometry b=2 n=78 sp=16384 nb=4115); sabotage-proven: revert line 716 ->
**RED** (`cons_diff_rows` 134, the audit probe's exact 67/78-per-batch), restore ->
green. The projection-revert mutation is still **10 RED**. `ruff` clean.

### Speed

The fix does **not** change the band geometry: the score / top-k pass (the part the
split halves) still runs `n/2` rows at n=2048. What moves is the two small query
projections, now at `n_full` rows per rank (= the replicated cost; the pre-fix band
had halved them too). Local measurement (EXL3 1024->8192 + bf16 4096->64):
**2.06 ms @1024 rows -> 3.78 ms @2048 rows, +1.72 ms per index layer per chunk**;
x 8 index layers ~ **+14 ms per 2048-row chunk** (local M-series; node numbers may
differ). Against the bench's indexer saving (e.g. L20 @25K 132.3 -> 61.1 ms, ~71 ms
on one layer) this is a small fraction of the win; the re-bench below is the
authority.

### Re-bench (idle-guarded, offline, node `macstudio-m4-2`; fixed mlx-lm; wall-fold primary)

| depth | new d% wall-fold | pre-fix d% | change pp |
|---|---|---|---|
| 25K | 7.69 | 7.79 | -0.10 |
| 50K | 10.22 | 10.36 | -0.14 |
| 100K | 14.38 | 14.47 | -0.09 |
| 130K | 16.74 | 16.81 | -0.07 |

Absolute wall-fold full->band (ms): 6649.2->6137.8 / 7106.1->6380.2 / 7986.7->6838.3 / 8574.1->7138.4. `indexer_fold` full->band: 964.10->448.17 / 1404.51->669.38 / 2250.74->1091.30 / 2824.27->1377.67. Added projection cost: band-arm per-layer indexer +0.96..+1.73 ms vs pre-fix band; banded fraction f 0.543->0.535 / 0.528->0.523 / 0.518->0.515 / 0.515->0.512. All 4 non-index layer types byte-identical between arms. Full arm reproduced within +0.38..+0.58% of the prior run. Guard exit 0, idle ok before AND after, no `ABORTED_*`; sentinel `mlx_lm/models/exl3/gemv_metal.py` md5 `649d2d6a0a68c8d6eab2a8a5394b7ab9` BEFORE==AFTER both nodes; prod `deepseek_v41/indexer.py` `af7be506...` unchanged; all_sum census 2->3/forward; node writes confined to `/tmp/p20_idxsplit_rebench/`; raw blobs there + local `/private/tmp/p20-idxsplit-rebench/raw/`. Caveat: single-node all_sum stubbed identity, so the +1 cidx exchange (+2.0% wire, ~0.04-0.08% of chunk) is not charged -- optimistic bound by that amount (same as pre-fix).

### Audit findings (audit 2)

Report `~/.hermes/cache/scratch/p20-idxsplit-audit2/REPORT.md` (independent out-of-repo re-derivation):

- **D3** (MEDIUM, non-default HIER=0 path) — **FIXED** in this commit + test (details above).
- **L1** (LOW, pre-existing untiled `isfinite`) — documented, not fixed (out of lane).
- Root cause **CONFIRMED**: `wq_b` kernel by row count at `exl3_linear.py:123/146/157/162`; downstream M-invariance 0/1206 cases (b in {1,2}; n 34..2048; tile 8..4096).
- Shape sweep banded==replicated: hier 0/252; COMPILE=0 0/84; HIER=0 TILE_FORCE 0/84; HIER=0 untiled 0/84; small-geometry 0/276; world=3 0/8; **HIER=0 default-tiling 5/84** (the D3 gap).
- flag-off byte-identical 24/24; D1 call site intact; memory: full-width `q_full` is 32 MiB at n=2048 (peak unchanged vs replicated); 23/23 mutations caught, 0 escaped (old escapes M5/M10/M12 now caught); the new tests are 10 RED pre-fix.
- Advisory: keep the full projection (kernel-pinning unproven; exchanging q ~32 MiB vs ~0.5 MiB for the cidx exchange would likely cancel the win; regime guards rely on undocumented MLX thresholds).

Unverified here: the exact live node-side cost of the full-row projections, and a
live re-run of the 5-prompt identity gate (needs a user-gated deploy).
