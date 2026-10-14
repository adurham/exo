# DSv4.1 indexer — QSPLIT-path reference-divergence extension (2026-10-14)

**Scope.** Extend the wave-1 reference-divergence harness
(`audit/dsv41-indexer-refdiv-2026-10-14` @ `3255ba4`, base `a0573e3`) from the
`DSV41_INDEXER_QSPLIT=0` surface in §1 of
`docs/dsv41-indexer-refdiv-audit-2026-10-14.md` to the **`QSPLIT=1` path** — the path
ENGAGED during the 2026-10-13 degeneration event, whose first live A/B aborted with a
token-identity divergence on a banded non-first prefill chunk and whose post-fix
re-verification passed only a 5-probe gate. **No cluster contact, no relaunch, no
studio writes.** Worktree `~/.hermes/cache/scratch/mlx-lm-qsplit` (mlx-lm fork @
`a0573e3`), harness pushed on
**`audit/dsv41-indexer-qsplit-refdiv-2026-10-14` = `128c23d`**.

**Verdict (headline).** The `QSPLIT=1` path is **selection-clean across 485 cells /
84,810 compared query rows** (30,991 of them banded), spanning non-first prefill
chunks, band edges, 32K–262K contexts with 213 cells inside the live 50K–80K band, TP
world 2 and 4, all three indexer roles, all three routes, `index_topk=512`, the
`merge` implementation, bf16 small-n rows and a deliberately-built 900-way exact tie
across the k-th boundary: **0 selection divergences, 0 duplicate indices, 0 candidate-
mask bit differences.** Two of the three "band must reproduce the chunk" decisions (D2
fence, D3 route) are confirmed aligned *and* load-bearing; the third — the hierarchical
exact pass's **strip width**, derived from the band's row count while the chunk's is
derived from the full chunk (`indexer.py:796-799`) — is **not** aligned, is a
demonstrated tie-break hazard in the shipped top-k primitive, and **could not be
reached** by any tie structure the harness can build. That is a ranked, unshipped
finding with a zero-cost env mitigation, not a blocker. The original live divergence
class (a row-count-dependent query projection) is now **directly testable and the fix
is proven structurally immune** — the gap that let that bug through every predecessor
suite.

---

## 1. What the harness covers (and why each shape is there)

`bench/dsv41_indexer_qsplit_refdiv.py` (new, 1,338 lines) — production geometry:
`index_n_heads=32`, `index_head_dim=128`, `index_topk=512`, `DSV41_INDEXER_TILE=512`,
`TILE_MB=128`, `TILE_MIN_NB=0`, `_FENCE_MIN_ROWS=16`, `_HIER_BLOCK=8`, `_L2_FULL=1`,
`_HIER=1`. `dim`/`q_lora_rank` are shrunk — neither enters the score GEMM
`(M = bsz*n, K = index_head_dim, N = tile)` — so the row dtype, row width, top-k and
mask grid are production-exact. Context is `nb * ratio * NB_SCALE` with `NB_SCALE=16`,
so nb is 512–8192 for realistic `(n, nb)` pairs.

| section | what it probes |
|---|---|
| **A** | `query_band` over the (n, world) grid incl. rank-boundary straddles and partial last bands; and the **band-vs-chunk asymmetry census** — every shape-derived quantity the two arms resolve separately |
| **B** | the prefill-chunk matrix: **non-first** prefill chunks (`start_pos = nb*ratio - n + phase`), band seams on/just-off a block and a tile boundary, 32K–262K contexts dense over 50K–80K, world 2/4, source/consumer/plain, HIER and tiled routes, `lattice` (random fp4) and `ties` (periodic keys → exact ties) |
| **C** | decode (n=1) and verify-shaped (n=2..17,32,33) forwards at 32K–262K: the gate must refuse (no band, and a RAISING `all_sum` spy proves no collective), plus the **D2 hazard** — an explicitly injected below-fence band, measured rather than assumed |
| **D** | `index_topk=512` edges (nb = 512/513/515/520/1024), the duplicate-index census on the reconstructed selection, band disjointness, and `_TILED_IMPL=merge` |
| **E** | bf16 small-n rows (`L2_FULL=0`, `SMALLN_ROW_BF16=1`) |
| **F** | the ORIGINAL divergence class: a **row-count-dependent projection stub** (mimicking `EXL3Linear`'s ≤16 / ≤64 / else kernel crossover and MLX's M-dependent bf16 matmul) + a faithful model of the pre-`de6caa2` expression, transcribed from that commit's own diff |
| **G** | the residual class: the exact-pass strip width (band-derived vs chunk-derived), a direct estrip A/B, **wide-tie reachability** (a 900-way exact tie straddling k=512), and a probe of the shipped top-k primitive's own tie pick |
| **H** | the D3 read-out: the row count each projection is **called** with, spied at the call site (behavioural, not a code read) |
| **I** | non-vacuity: a margin-sized perturbation must flip a selection; the tie lattice must put exact ties on rank k |
| **J** | the matrix re-run with a row-count-dependent projection installed — the arm a plain-`nn.Linear` harness is structurally blind to |

Comparison levels per cell: index-set **exact equality** per row (plus mean per-row
top-k overlap over non-degenerate rows), the stored `[b,n,nb]` score row / the dense
fp32 row rebuilt through the shipped score body, the candidate mask bits, the padded-band
zero-support and contributor-disjointness, and the duplicate-index census. A row-level
ulp difference is reported, never asserted away; **exit status 1 on any selection
divergence.**

Run (mlx-lm worktree root; NOT from the vendored `~/repos/exo/mlx-lm`, where the parent
repo's `tests/` package wins the import race):

```bash
cd ~/.hermes/cache/scratch/mlx-lm-qsplit
PYTHONPATH=$PWD ~/repos/exo/.venv/bin/python bench/dsv41_indexer_qsplit_refdiv.py \
    --out /tmp/dsv41_indexer_qsplit_refdiv.json          # --quick for the ~1/3 matrix
PYTHONPATH=$PWD ~/repos/exo/.venv/bin/python -m pytest \
    tests/test_dsv41_indexer_qsplit_refdiv.py -q          # 20 tests, ~4.2 s
```

## 2. Measured results (this Mac, `a0573e3` + the audit branch, `_INDEXER_COMPILE=1`)

| quantity | result |
|---|---|
| matrix cells / records | **485** cells of 490 records; 84,810 query rows compared, 30,991 banded |
| context / n / world / route coverage | 32K–262K (213 cells in 50K–80K); n ∈ {1..2048}; world 2 and 4; hier 258 / tiled 139 / untiled 88 |
| **selection divergences** | **0** (index sets, candidate-mask bits, non-finite mismatches) |
| duplicate real indices in the reconstructed selection | **0** in all 485 cells |
| band contributor overlaps / padded writes outside the band | **0 / 0** |
| banded in-band slice ≠ the replicated run's row slice | **0** (the core invariant, byte-exact) |
| `max|Δ|` on the stored row / the dense fp32 row | 0.0 on every cell except the 2 deliberately-injected D2-hazard cells |
| gate census (234 shapes, all ranks) | **0 mismatches**; first banded n = 34 (w2) / 68 (w4); odd or non-divisible n → replicated; `world<2` → replicated |
| band **dtype** ≠ chunk dtype | 0 of 40 asymmetry cells (D2 holds) |
| band **estrip** ≠ chunk estrip | **40 of 40** asymmetry cells |
| band **strip count** ≠ chunk strip count | **24 of 40** (e.g. n=78/nb=3000 → 1 vs 2; n=132/w4/nb=3000 → 1 vs 3; n=512/nb=4096 → 7 vs 13) |
| pre-D3 route rule (`tiled(band_rows)`) ≠ shipped rule (`tiled(n_full)`) | **6 of 40** cells — what `6a5c78f` fixed |
| projection call row counts (section H) | `wq_b`/`weights_proj` called with **n_full only** — 78, 1024, 2048 rows at band rows 39, 512, 1024 |
| **row-count-dependent projection** (section F/J) | stub changes **159,744 / 159,744** band-q elements between 39-row and 78-row calls; the shipped banded path is **byte-identical** to the replicated run at eps = 1e-7, 1e-5, 1e-4, 1e-3, **1e-2** — and in every one of the 6 matrix cells of section J |
| the pre-fix expression, same stub | 0 rows differ at eps=1e-7, **4** at 1e-5, **29** at 1e-4, **78** at 1e-3 → non-vacuous; **threshold eps = 2⁻¹⁶ (1 row)** |
| the pre-fix expression, **plain** `nn.Linear` | **0 rows differ, byte-equal** — the live bug's invisibility to every predecessor suite, reproduced |
| D2 hazard (injected below-fence band) | tiled fallback: **2/20 and 3/26 rows separate**; hier route: 0/20, 0/26 (the hierarchy prunes nothing at that nb, so it agrees with the fallback) |
| direct estrip A/B (same replicated forward, estrip 4456 vs 2224 etc.) | byte-equal, 0 differing rows, with every row exactly tied at k |
| band vs chunk on different strip counts (1v2, 1v3) with every row tied | **0 differing rows** |
| wide-tie reachability (900-way exact tie at k=512; 1v2, 1v3, 7v13 strips) | **0 differing rows**, `band_slice_equal` in all 6 cells |
| the tie-break PRIMITIVE (shipped `topk_from_row` 1 pass vs shipped `merge_topk` 2 rounds over the same 4224 columns, 900 exactly tied) | sets **differ**, overlap **0.771** |
| non-vacuity | margin flip on **78/78** rows, min margin 1.055e-05; tie lattice puts exact ties on rank k on 69/78 rows |
| sabotage: revert `de6caa2`'s call order | **5 RED / 20** — both new projection arms and the D3 read-out — while the plain-`nn.Linear` matrix arms stay green in **9 of 10** cases |
| gates | `ruff check` clean (2 files); `ruff format` not clean, in parity with the sibling `tests/test_dsv41_*.py` (55 files under `tests/` are already format-dirty; `nix fmt` unavailable on this Mac) |

## 3. Ranked findings

**1 — MAJOR (latent, no reachable reproducer, unshipped): the hierarchical exact pass
derives its strip width from the BAND's row count while the replicated run derives it
from the full chunk.** `indexer/__call__` passes `n` — the band's rows — to
`hier_strip_for_budget(bsz, n, …)` at `indexer.py:796-799`, so `estrip` differs between
the arms in **40 of 40** census cells and the strip **count** differs in 24 of 40 (1 vs
2 at n=78, 1 vs 13 as the row halves at n=1024). The stage it feeds merges with
`argpartition(-v, k-1)` over `concat([best_v, strip_v])` — a length of `k + bp*block` —
and the shipped primitive's pick **among exactly tied values is length-sensitive**
(measured: the shipped one-pass and two-round forms agree on only **0.771** of k on a
900-way tie). The D2/D3 principle ("the band must resolve the same regime the chunk
would") was applied to the *route* and the *row dtype* but not to the *strip*. At
`DSV41_INDEXER_HIER_EXACT_STRIP=0` (the shipped default) this asymmetry is live whenever
`QSPLIT=1`; at a non-zero value it disappears, because both arms then read the same
constant. **No fix shipped** — a precision-path change needs the live quality battery
and an owner decision (the repo's standing rule), and this one is unproven in practice:
it survives the whole matrix *including* a deliberately constructed 900-way exact tie at
the k-th boundary. Two candidate actions, in order of preference:
* **zero-cost, env-only, no code change:** add `DSV41_INDEXER_HIER_EXACT_STRIP` to the
  frozen env in §1.3 of the pre-registered A/B and set it to the chunk-derived value at
  the largest `n` the run will see. Both arms then use one constant strip and the class
  is **removed from the A/B entirely**. (It is memory-safe: the band with fewer rows
  simply runs more, smaller strips.)
* **code (needs the quality battery):** pass `n_full` at `indexer.py:796-799`. Strictly
  more conservative than the band-derived value (estrip ∝ 1/n, so `n_full` gives smaller
  strips), so it cannot raise the peak; it costs the band some host fences.

**2 — MAJOR (measured, by design): the D2 fence is load-bearing, and by how much.** With
a band injected below the fence on a chunk above it — the shape `query_band` refuses —
the banded run separates from the chunk's run on **2 of 20 and 3 of 26 rows (~10%)** on
the tiled fallback route, because the band resolves `_row_dtype(band_rows)` (fp32 under
L2-full) while the chunk resolves bf16. On the hierarchical route the same injection
separates nothing at nb=2048 (the top-(k+overfetch) block prune admits every block, so
hier and the fallback agree). *Any* future relaxation of the predicate in `query_band`
— for instance gating on the full `n` instead of the band's rows — reintroduces a
selection divergence on roughly a tenth of the rows of every affected forward. This is
now pinned by a test rather than by a comment.

**3 — MINOR (already known for `merge`, now also true of the HIER exact pass): the
shipped top-k's tie pick depends on the merge layout.** `argpartition` over an
exactly-tied top group returns a different subset at different array lengths
(0.771/512 overlap in the probe). Wave 1 flagged this for `_TILED_IMPL=merge`
(default `buffer`, hence unshipped); the finding here is that the *production* hier path
reaches the same primitive through its strip merge. Keep `DSV41_INDEXER_TILED_IMPL=buffer`
frozen (already §1.3) **and** apply finding 1's mitigation if a digests-differ result
appears in the A/B with no other explanation.

**4 — NOT findings.** The `de6caa2` projection fix (structurally immune to any
deterministic row-count-dependent kernel, and the pre-fix model is non-vacuously
divergent at eps ≥ 2⁻¹⁶); the D3 route key; the D2 dtype; the padded-`all_sum`
reconstruction (contributor disjointness, zero support outside the band, byte-exact
in-band slices); the candidate mask; duplicate indices; `publish_keys` (query-side
invariant, re-pinned by the sibling suite). 485 cells, 0 divergences.

## 4. NOT verified (do not read these as claims)

1. **Real trained weights.** `wq_b`/`weights_proj` are random-init and the keys are
   fp4_ue8m0 fake-quantized on the harness's own lattice. Shapes, dtypes, row widths and
   top-k are production-exact; the *value distribution* that decides whether two scores
   straddle the k-th boundary is not the trained model's. The exact-tie density under
   real weights (and under a real `EXL3Linear` at real row counts) is unmeasured — it is
   the single quantity finding 1 turns on.
2. **No real `mx.distributed`/JACCL exchange.** Both arms stub `collective.all_sum`
   in-process (identity for the reconstruction, raising for the `band=None` contract).
   The cross-rank int32 reducer, its ARQ-retry behaviour and the mesh's size restrictions
   are not exercised. This harness is not a substitute for the cluster A/B.
3. **The real `EXL3Linear` kernel crossover.** Modelled by a per-row-count-bucket scale
   stub, not by the shipped quantized kernel; the *shape-dependence* is modelled
   faithfully, the *magnitude* (2⁻¹⁶ measured as the selection-flip threshold through the
   pre-fix expression) is a synthetic bound, not the production q-delta.
4. **The production prefill path.** `n ≤ 2048`, `nb ≤ 8192`, `bsz ≤ 2`, toy-shrunk
   `dim`/`q_lora_rank`. The hierarchical pass at production prefill `n`/`nb` is
   untouched here; wave 1's NOT-verified item 4 stands.
5. **bf16 small-n rows are unreachable through the gate.** `query_band` never bands at
   `n//world ≤ 16`, so the bf16 small-n row was exercised only via an explicitly injected
   band (section E) — the shipped configuration cannot produce that shape.
6. **Token/logit level.** Nothing here compares tokens; that is by design (greedy chaos)
   and is what the pre-registered cluster A/B is for.
7. **`ruff format`.** Both new files are `ruff check`-clean but not `ruff format`-clean,
   in parity with every sibling `tests/test_dsv41_*.py` in this tree; `nix fmt` was not
   available on this Mac.

## 5. How this feeds the pre-registered relaunch A/B

(`docs/dsv41-indexer-relaunch-ab-prereg-2026-10-14.md`; nothing there needs rewriting —
this note closes its §1 items and adds one env to freeze.)

* **Arms A0/A1 (`QSPLIT=0`)** — unchanged. This work adds nothing to them and does not
  disturb the wide-tile evidence in §2 of the wave-1 audit.
* **Arm A2 (`QSPLIT=1`, wide tile OFF)** — this is the arm this note is about. It is now
  backed by a 485-cell selection-clean matrix at production geometry rather than a
  5-probe gate, so a clean A2 is a meaningful result and a dirty one is worth
  investigating rather than discounting. **Launch precondition to add to §1.3:**
  `DSV41_INDEXER_HIER_EXACT_STRIP` frozen to the chunk-derived value at the run's largest
  `n` (finding 1) — env-only, no code change, no quality battery, and it removes the one
  shape-derived quantity in which the two arms currently disagree.
* **The digest criterion (§3.3) needs one refinement:** the pre-registered criterion 1
  ("exact digest equality required for every call whose geometry shows `route=untiled`
  or `wide=0`") is correct and now locally corroborated. If a digest *does* differ on a
  banded tiled call, read the strip geometry before treating it as a blocker: this audit
  predicts the only reachable cause is a k-th-boundary exact tie whose break the two
  strip layouts took differently, and its signature is a difference confined to rows with
  a zero k-th gap. Everything else is a real divergence.
* **Arm A3 (`QSPLIT=1` + wide tile)** — do not run it for attribution; run A2 first, as
  pre-registered. Nothing in this note changes the A3 precondition.
* **What this note does NOT change:** the A/B still cannot prove the qsplit path correct
  on the box (real weights, real collective, real prefill), and a clean A2 does not
  exonerate the wide tile.

## 6. Provenance

* mlx-lm fork branch **`audit/dsv41-indexer-qsplit-refdiv-2026-10-14` = `128c23d`**
  (base `3255ba4`, itself on `a0573e3`), pushed to `adurham/mlx-lm`.
  Files: `bench/dsv41_indexer_qsplit_refdiv.py` (new), `tests/test_dsv41_indexer_qsplit_refdiv.py`
  (new, 20 tests, ~4.2 s). No source file modified — the sabotage in §2 was applied,
  measured and reverted (`indexer.py` byte-identical to `a0573e3` afterwards).
* Wave-1 harness left untouched at `~/.hermes/cache/scratch/mlx-lm-audit` @ `3255ba4`.
* Local run artefacts (this Mac, not committed): `~/.hermes/cache/scratch/dsv41_indexer_qsplit_refdiv.json`
  (per-cell records), `qsplit-refdiv-notes.md` (incremental notes), `qs_full.log`,
  `qs_pytest.log`, `qs_sabotage.log`.
* Gates run: `ruff check` clean (2 files); `pytest` 20 passed for the new pin, and
  **56 passed** for the three indexer harness files together
  (`..._wide_tile_refdiv.py` 11 + `..._qsplit_band_matrix.py` 25 + `..._qsplit_refdiv.py` 20),
  25.6 s.
