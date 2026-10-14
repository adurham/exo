# DSv4.1 indexer — kernel-correctness audit + reference-divergence harness (2026-10-14)

**Scope.** Kernel-level audit of the mlx-lm serving-math change surface between the clean
baseline pin `689e4ea` (`deploy/next22-rankfix`) and the live pin `a0573e3`
(`deploy/next26-limiter`), and a reference-divergence harness for it. **No cluster contact,
no relaunch, no studio writes.** Worktree: `~/.hermes/cache/scratch/mlx-lm-audit` (mlx-lm fork
@ `a0573e3`). Harness pushed to the mlx-lm fork on
**`audit/dsv41-indexer-refdiv-2026-10-14` = `3255ba4`**.

**Verdict (headline).** With `DSV41_INDEXER_QSPLIT=0` — the config of **both** live
degeneration events — the whole 6-commit surface reduces to exactly **one** rule that can
move serving math: the `tile_width()` small-n wide-tile branch added by `a0573e3`. That rule
is **proven** to (a) never change the tiled/untiled ROUTE, (b) change the stored score ROW at
the ulp level at one specific shape class (a one-column tail tile), and (c) **never move a
selection** — index set, candidate mask, or `Indexer.__call__` output — in any of 97 harness
cells spanning its whole engagement region. Its engagement is also far narrower than the
"context inversion" hypothesis assumed: at a γ≤3 verify shape (n≤4) it **cannot engage below
254.2K context at all**, i.e. in neither the clean-era 110-241K band nor the loop events'
50-80K band. **No fix is proposed: the mechanism is not proven, and a guess-fix on a
bit-exactness-sensitive path is the highest-cost mistake available here.**

---

## 1. The change surface, reduced (evidence: hunk-by-hunk read of `689e4ea..a0573e3`)

Files touched: `deepseek_v41/indexer.py`, `deepseek_v41/attention.py`, and the two new test
files. Nothing else in the package. Six commits: `dc0e79c` (query-band split) → `f728bb6`
(typed group shim) → `e034062` (D1 call-signature + D2 band fence) → `de6caa2` (project the
FULL chunk, then slice) → `6a5c78f` (D3 route keyed on the full chunk) → `a0573e3` (small-n
wide tile).

Disposition of every hunk **with the gate off** (`DSV41_INDEXER_QSPLIT=0`):

| hunk | file:line (a0573e3) | live at QSPLIT=0? |
|---|---|---|
| `query_band` call + the D1 conditional kwargs | `attention.py:211-219` | **no** — `query_band` returns `None`, so the call is the original 7-positional-arg form |
| `_DistributedGroup` Protocol, `query_band`, `_reconstruct_rows`, `_finish_rows` | `indexer.py:158-242` | **no** — every one returns `cidx` unchanged when `band is None` |
| project-then-slice refactor (`q_full`/`w_full`/`qpos`) | `indexer.py:725-743` | **no** — expression-identical: `wq_b(qr).reshape(...)`, `weights_proj(x) * scale`, `arange(start_pos, start_pos+n_full)`, `((qpos+1)//ratio)` |
| `tile = tiled(bsz, n_full, …)` (D3) | `indexer.py:759` | **no** — with `band is None`, `n_full == n` |
| the three `_finish_rows(...)` returns | `indexer.py:812, 835, 862` | **no** — pass-through |
| **`tile_width()` small-n wide branch** | `indexer.py:368-375` (+ env read `:128`) | **YES — the only one** |
| `publish_keys` (cache write) | `indexer.py:668-685` | not in the diff at all (cache writes unchanged) |

So event 2 (QSPLIT=0), if it is in this diff at all, must be the wide tile. That is the
reduction the rest of this audit tests.

## 2. Ranked findings

**BLOCKER — the audit's own conclusion, not a code defect: the wide tile cannot be the
mechanism of the two recorded loops under the documented verify shape.** The width rule is
observable only on a TILED route (`indexer.py:398` returns `nb` — untiled — whenever
`tile >= nb` or the budget clause `indexer.py:402` holds). With `nb = (start_pos+n)//ratio`
(`attention.py:178,195`) a width change therefore needs `nb > cap(n)` ⟺
`context > cap(n) * ratio`, and production `cap(n) = 134,217,728 // (n * (32*4+4))`
(`indexer.py:367`, 32 index heads, fp32 small-n row under `_L2_FULL=1`):

| n | 1 | 2 | 3 | 4 | 5 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|---|---|
| crossing context (ratio 1) | 1,016,800 | 508,400 | 338,933 | **254,200** | 203,360 | 127,100 | 84,733 | 63,550 |

`EXO_SPECULATIVE_GAMMA` defaults to 2 for the DSv4 MTP path (γ≤3 ⇒ n≤4 in the verify
forward). At n≤4 the rule cannot engage below 254.2K, so **it did not fire in either loop
run (56K, 78K)** and it did not fire anywhere in the clean baseline's 110-241K either. Only
**n=16** reaches into a 48-82K band (measured: of 5,850 cells in 48-82K, 82 changed, and every
one is n=16). Consequence: **before spending a relaunch on this arm, read the actual small-n
row count off the runner logs.** If the loop-era verify shape was γ≤3 (n=4), the wide tile is
exonerated by arithmetic and the regression lives elsewhere (the exo-side guards / limiter /
park-restore changes, or the prompt payloads) — see the sibling forensics workstreams.
`indexer.py:128` reads `DSV41_INDEXER_SMALLN_WIDE_TILE` but **never reports it** (contrast
`indexer.py:185-187`, which prints the qsplit flag), and no line reports the per-call indexer
geometry `(n, nb, ratio, tile, route)`. That is the single highest-value instrumentation gap
in this question; the pre-registered A/B below makes closing it a launch precondition.

**MAJOR — documented as "index sets identical in every offline case"; now measured, localized
and mechanistically explained.** The wide tile *does* move the stored score row at the ulp
level, in exactly one shape class: **a one-column tail tile** (`nb = cap + 1`, where the new
arm's last tile is 1 column wide while the old arm's is 63-502 columns wide, so the tail
GEMM's N collapses and a different kernel handles it). Measured: 1 element differs, |Δ| ≤
4.5e-08, and that element is **always the row's last column** (`col = nb-1`, the newest
compressed group). The affected element's rank inside its own row is 12,856 … 793,180 of up
to 1.02M columns — i.e. it never comes near the top-`index_topk`(512) selection. Hence 0
selection divergences across 97 cells (below). This corroborates the code comment's own
"2-5 fp32 row elements (|d| ≤ 1.5e-8) differing across widths at some deep shapes"
(`indexer.py:45-49`) — the numbers here are the same class, at the rule's own engagement
shapes, and the shape class is now identified rather than "some deep shapes".

**MAJOR — the evidence base for `a0573e3`'s exactness claim was thinner than its buildnote
implies.** The wide-tile buildnote's "index sets + candidate masks identical across
new/old/untiled in 67/67 harness cases" was gathered partly on a bench whose
`build_indexer` "never installed real `wq_b`/`wk` (random-init `nn.Linear`)" — the note says
so itself. Random-init weights under-represent the exact ties that make a top-k boundary
flip. The harness below closes that gap with (a) fp4_ue8m0 fake-quantized keys on a coarse
lattice (what `publish_keys` actually stores), (b) a tie-injected lattice that puts exact
ties on rank k for **every** row, (c) the production widths themselves (512 vs 254,200 —
the existing suite's identity grid shrinks `_TILE_BUDGET` to test *shape classes*, never the
production pair), and (d) a non-vacuity control that proves the comparison can see a
margin-sized difference. All three of the sibling suite's claims survive; what changes is
that they are now measured where the change actually engages.

**MINOR — the non-default `merge` implementation is the one place a width change can move a
tie pick.** `_tiled_scores_merge` runs a per-tile `argpartition` over `concat([best_v, tile_v])`
(`indexer.py:616-623`), whose LENGTH is `k + tile_width` — width-dependent by construction —
so two arms can break a tie differently even with bitwise-identical per-tile scores. It is
**not shipped** (`_TILED_IMPL` defaults to `"buffer"`, `indexer.py:106`, where the top-k is a
single global pass over one identically-shaped row) and the harness found 0 divergences on
it too, but it is a documented A/B/rollback arm: any A/B that runs with
`DSV41_INDEXER_TILED_IMPL=merge` must not inherit the wide tile's exactness claim.

**MINOR — `_TILE_BUDGET`/`_TILE` are read at import (as designed), so the kill switch needs a
process restart, not a config reload.** Already true of every `DSV41_INDEXER_*` knob and
already the documented rollback (`=0` env flip at relaunch), so this is a note for the
runbook rather than a defect. `_SMALLN_WIDE_TILE` is read at `indexer.py:128`.

**MINOR — the 128 MiB budget is a *host* constant while the transient it bounds is a *GPU*
allocation.** The wide tile's whole justification is `cap` columns of `[b, n, heads, cap]`
fp32 ≈ one budget; the existing sibling test asserts that bound
(`test_wide_tile_within_budget`). No new finding, but a relaunch that changes
`DSV41_INDEXER_TILE_MB` changes the width AND the engagement threshold together — keep it
frozen across the A/B arms.

**NOT a finding — suspect items 3 (`overlap=512` dedupe/ties), 4 (offset arithmetic), 5
(cache writes).** The default buffer path's top-k is one exact global `argpartition` over a
single `[b, n, nb]` row (`topk_from_row`, `indexer.py:425-434`) — there is no cross-band
merge to dedupe and no overlap parameter in this path; `merge_topk` is only reachable via
`_TILED_IMPL=merge`. The offset arithmetic (`qpos`, `lens`, `i + offset`) is unchanged with
the gate off, and `publish_keys` is untouched by the whole range. Item 6 (park/restore) is
outside this task by instruction.

## 3. The harness (pushed, runnable)

* `bench/dsv41_indexer_refdiv.py` — the measurement. Production geometry (32 index heads,
  `index_head_dim=128`, `index_topk=512`, `TILE=512`, 128 MiB, `_TILE_MIN_NB=0`,
  `_FENCE_MIN_ROWS=16`, `_L2_FULL=1` fp32 small-n row, `_HIER=1`). `dim`/`q_lora_rank` are
  shrunk: neither enters the score GEMM `(M = bsz*n, K = 128, N = tile)`, so the row dtype,
  row width, top-k and mask grid are production-exact.
  Sections: **A** route/engagement over a context grid with a dense 48-82K inset; **B/C**
  the stored row rebuilt at each arm's exact width through the shipped compiled score body,
  per arm ROUTE, against the untiled dense fp32 reference; **D** the real
  `Indexer.__call__` in NEW/OLD/UNT arms (plus consumer layers, so the published mask is
  consumed); **E** sensitivity/margin controls and the `merge` implementation; **F** the env
  kill switch; **G** a tail-width shape census.
  `--quick` skips C/G. Exit status is a **selection** gate: a row-level ulp difference is
  reported but is not a failure.
* `tests/test_dsv41_indexer_wide_tile_refdiv.py` — the bounded regression pin (11 tests,
  ~21 s).
* `tests/test_dsv41_indexer_qsplit_band_matrix.py` — 256-cell shape matrix for the
  query-band split vs the replicated run and the dense reference (index sets, stored row
  deltas, k-th gaps, `publish_keys` byte-equality), over offsets mod band, `n` from 1 to 512
  including the `n//world > 16` gate edges, TP world 2 and 4, partial last bands, all three
  indexer roles and both routes. No divergence found there either.

Run (mlx-lm worktree root):
```bash
cd ~/.hermes/cache/scratch/mlx-lm-audit      # or the fork checkout of the branch
PYTHONPATH=$PWD ~/repos/exo/.venv/bin/python bench/dsv41_indexer_refdiv.py \
    --out /tmp/dsv41_indexer_refdiv.json
PYTHONPATH=$PWD ~/repos/exo/.venv/bin/python -m pytest \
    tests/test_dsv41_indexer_wide_tile_refdiv.py tests/test_dsv41_indexer_qsplit_band_matrix.py -q
```

### Measured results (this Mac, `a0573e3`, `_INDEXER_COMPILE=1`, 97 cells)

| quantity | result |
|---|---|
| route (tiled/untiled) mismatches, NEW vs OLD | **0 / 10,374 cells** (and the transcribed rules equal the shipped `tile_width` in every cell) |
| cells where the width rule is observable at all | 1,416 / 10,374 (2,574 of them ≥ 254K context; **82 in 48-82K, all n=16**) |
| row-level ulp differences | 12 cells, all `nb = cap+1` (one-column tail): 1 element, \|Δ\| ≤ 4.5e-08, always the row's **last** column |
| rank of the differing element inside its row | 12,856 … 793,180 (top-k selects 512) |
| index-set differences NEW vs OLD / NEW vs UNT | **0** |
| candidate-mask bit differences | **0** |
| real `Indexer.__call__` NEW vs OLD vs UNT | **byte-identical `cidx`** in all 6 arm cells (plain/source/consumer, incl. tie lattice) |
| non-vacuity | a margin-sized perturbation flips the set for **4/4** rows; min margin 8.95e-05; tie lattice puts exact ties on rank k for 4/4 rows |
| kill switch | unset → ON, `=1` → ON, `=0` → OFF and reproduces the old formula byte-for-byte |

Interpretation: the change is **provably route-preserving**, its only numerical effect is a
one-element ulp coincidence in a one-column tail tile at a column the top-k never selects,
and it moved no selection in any cell of its engagement region. It is therefore **not
exonerated as a class** (a greedy trajectory is chaotic, and a 4.5e-08 difference that *did*
land inside the top-k would still be invisible to this harness), but there is **no proven
mechanism**, which is exactly the condition under which no fix should be written.

## 4. NOT verified (do not read these as claims)

1. **Real trained weights.** All cells use random-init `wq_b`/`weights_proj`/`wk` with
   fp4-quantized keys. The GEMM *shapes and dtypes* are production-exact, so the
   shape-dependent kernel selection (which is what the ulp class comes from) is faithfully
   exercised — but the *value* distribution that decides whether two scores straddle the
   boundary (and how often exact ties occur) is not the trained model's. The 12
   `row_elems_differing` cells are value-dependent rounding coincidences; their rate under
   real weights is unmeasured.
2. **No real `mx.distributed` / JACCL exchange.** Both harnesses stub `collective.all_sum`
   in-process (identity for the replicated run, a summing spy for the band). The cross-rank
   collective and its int32 reducer are not exercised; a 2-node TP run is required for that.
   Neither harness is a substitute for the cluster A/B.
3. **bf16 small-n rows.** Production runs fp32 there (`_L2_FULL=1`, `_SMALLN_ROW_BF16` unset).
   The bf16 small-n path was not measured (it is the pre-L2-full behaviour, reachable only
   by `DSV41_INDEXER_SMALLN_ROW_BF16=1` or `DSV41_INDEXER_L2_FULL=0`), and it is where the
   *documented* tie-creation risk lives (bf16 rounds distinct fp32 scores into exact ties).
   If the relaunch runs with either of those flipped, this audit's neutrality evidence does
   not carry over.
4. **The production widths of the *prefill* path.** All cells here are `n ≤ 16`. The
   hierarchical pass (`n > _FENCE_MIN_ROWS`) is untouched by the range and out of scope.
5. **Index-set tolerance at production `nb` on the cluster**: unmeasured. Locally the digest
   question ("did any *other* serving-math hunk move anything?") is answered only for the
   small-n path; the reduction in §1 is a code-reading argument, verified by no live capture.
6. **Token/logit level.** Nothing here compares tokens; that is by design (greedy chaos) and
   is what the pre-registered cluster A/B is for.
7. **The qsplit path.** Gate OFF in production; `qsplit`'s own 2-node collective and its
   earlier live token-identity divergence are **not** re-verified here.
8. **`ruff format`.** Both new files are `ruff check`-clean but NOT `ruff format`-clean, nor
   are any of the sibling `tests/test_dsv41_*.py` files in this tree (measured, same
   before/after); `nix fmt` was not available on this Mac. Left consistent with the
   neighbours rather than reformatted in isolation.

## 5. Provenance

* mlx-lm fork branch `audit/dsv41-indexer-refdiv-2026-10-14` @ `3255ba4` (base `a0573e3`),
  pushed to `adurham/mlx-lm`.
* Local run artefacts (this Mac, not committed): `~/.hermes/cache/scratch/dsv41_indexer_refdiv.json`
  (per-cell records), `refdiv.out`, `test_refdiv.out`, `qsplit_band_matrix.json`.
* Gates run: `ruff check` clean (3 files); `pytest` 11 (wide-tile pin) + 25 (qsplit band
  matrix) + 9 (sibling `test_dsv41_indexer_smalln_wide_tile.py`) = **45 passed**.
  No source file modified; `nix fmt` unavailable.

See `docs/dsv41-indexer-relaunch-ab-prereg-2026-10-14.md` for the pre-registered
verification spec for the one owner-approved relaunch.
