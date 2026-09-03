# D1 — Regression Guard Teeth Verification (exo-side shipped wins)

Verification only. No permanent code changes, no commits, nothing staged.
mlx-lm submodule located at `/Users/adam.durham/repos/exo/mlx-lm` (git submodule per
`.gitmodules`, separate git repo, pinned at `37260bb` / tag
`known-good-poolgrow-default-20260823-10-g37260bb`).

## Summary table

| Win | Guard | Has teeth? (verified how) | Runs in CI? |
|---|---|---|---|
| 1. BatchPoolingCache overlap-carry resize fix | `mlx-lm/tests/test_batch_pooling_cache_overlap.py` (4 tests) | **YES.** Deleted the `filter()` reindex block in `mlx_lm/models/cache.py` (the actual fix, commit `37260bb`); 2 of 4 tests failed with the exact assertion messages the fix is meant to prevent. Restored file byte-for-byte (md5 verified), all 4 pass again. | **NO.** Neither exo's `.github/workflows/pipeline.yml` nor mlx-lm's own `.github/workflows/pull_request.yml` job (`mac_build_and_test`, gated `if: github.repository == 'ml-explore/mlx-lm'`) executes this file — see §4 below. Test only runs when a human/agent invokes it locally. |
| 2. Profiler unit metadata (`dsv4_mtp.py`, `_PhaseTimer`/`_ProfUnit`) | **NONE FOUND.** | N/A — **UNGUARDED.** Repo-wide grep for `PhaseTimer` and `_ProfUnit` outside `dsv4_mtp.py` itself returns zero hits in any test file. | N/A |
| 3. Telemetry runner filter (`collect_telemetry.py`, `parse_ps_runner`) | **NONE FOUND.** | N/A — **UNGUARDED.** The file itself only exists under `tmp/research-v1v2v3-20260901/telemetry/collect_telemetry.py` (a scratch/tooling script, not under `src/`, so even if a test existed it would be outside the `pytest src` scope CI runs). No test file references `parse_ps_runner` anywhere in the repo. | N/A |

---

## 1. BatchPoolingCache — exact command + result

```
cd /Users/adam.durham/repos/exo/mlx-lm
uv run --project /Users/adam.durham/repos/exo python -m pytest tests/test_batch_pooling_cache_overlap.py -v
```
Result (pristine source): **4 passed in 2.01s** (initial run: 4 passed in 3.85s, `real 0m5.708s` wall incl. interpreter startup). All 4 tests:
- `test_extend_widens_carry_and_preserves_surviving_stream` — PASSED
- `test_fetch_overlap_carry_no_carry_untouched_by_extend` — PASSED
- `test_filter_narrows_carry_and_preserves_surviving_stream` — PASSED
- `test_filter_respects_surviving_index_order` — PASSED

## 2. Teeth proof

**Mutation applied:** in `mlx-lm/mlx_lm/models/cache.py`, inside `BatchPoolingCache.filter()`
(around line 2656-2671), deleted the block that reindexes the four overlap-carry
structures (`_overlap_carry_valid`, `_overlap_windows_this_call`, `_overlap_kv_carry`,
`_overlap_gate_carry`) to the surviving stream set, replacing it with a no-op comment.
This is the literal root-cause fix from commit `37260bb` ("keep overlap-carry structs
in sync with batch width").

**Result: 2 failed, 2 passed** (`2.01s`… actually `1.98s`).

Failed tests and assertion messages:
- `test_filter_narrows_carry_and_preserves_surviving_stream`:
  `AssertionError: 3 != 1 : filter must reindex _overlap_carry_valid to the surviving batch`
- `test_filter_respects_surviving_index_order`:
  `AssertionError: Lists differ: [False, True, False] != [False, True]` (extra stale element retained)

The 2 `extend()`-only tests still passed since the mutation only touched `filter()` —
this is expected and does not weaken the finding; it confirms the mutation is targeted
and the failing tests are precisely the ones exercising the reverted code path.

## 3. Post-restore verification

Source restored via `cp` from a pre-mutation copy; md5 checksums matched
(`46c840cc6bdc831aa1d0c59df778ce74` before and after).

Re-run: **4 passed in 2.01s** (all four tests PASSED again).

```
$ git -C /Users/adam.durham/repos/exo/mlx-lm status --porcelain
(empty)
$ git -C /Users/adam.durham/repos/exo/mlx-lm diff --stat
(empty)
```

Exo repo top-level `git status --porcelain` / `git diff --stat` were **not** touched by
this task's mutations (only `mlx-lm/mlx_lm/models/cache.py` was ever mutated, and it was
restored). Note: the exo repo working tree already carried pre-existing, unrelated local
modifications from before this session started — `start_cluster.sh` and
`src/exo/worker/engines/mlx/speculative/dsv4_mtp.py` (methodology-comment additions,
committed upstream already at `bbb0e93` per `git log`, so these are local-only diffs
predating this task) — plus a large set of untracked files under `tmp/` being managed by
a concurrent agent. None of these were created, modified, or touched by this
verification task; flagging for triage since the instructions require a clean-repo
report, not because this task caused them.

## 4. CI coverage — definitive answer: NO

- exo repo CI config: `.github/workflows/pipeline.yml`. The only test step is:
  ```
  - name: Run pytest (macOS only)
    if: runner.os == 'macOS'
    run: |
      ...
      $TEST_ENV/bin/python -m pytest src -m "not slow" --import-mode=importlib
  ```
  This scopes pytest to the `src` directory only — the mlx-lm submodule (checked out at
  repo root as `mlx-lm/`) is never included, and the checkout step
  (`actions/checkout@v4` with default settings) does **not** pass
  `submodules: true`/`recursive`, so the submodule content wouldn't even be present in
  CI's checkout.
- mlx-lm's own upstream CI: `mlx-lm/.github/workflows/pull_request.yml`, job
  `mac_build_and_test`. That job is gated:
  ```
  if: github.repository == 'ml-explore/mlx-lm'
  ```
  Since exo's submodule points at a fork (`adurham/mlx-lm`, per `.gitmodules`), this
  workflow's condition never evaluates true for the fork — and in any case it is not
  triggered as part of the exo repo's own pipeline; it would only fire on pushes/PRs
  directly against the upstream `ml-explore/mlx-lm` repo.
- **Conclusion: the BatchPoolingCache test never runs in CI for this project as configured.**
  It is a guard with real teeth locally (proven above) but zero teeth against silent
  regression via the actual CI gate — a regression in this code could merge to `main`
  today with all CI checks green.

## 5. Profiler unit metadata — UNGUARDED

Source: `src/exo/worker/engines/mlx/speculative/dsv4_mtp.py` (`_ProfUnit = Literal["ms",
"count"]`, `class _PhaseTimer`, `record(..., unit: _ProfUnit = "ms")`), shipped in commit
`bbb0e9341` ("fix(profiler,telemetry): per-series unit metadata, runner comm filter,
stale comments"), documented in `docs/PERFORMANCE_HISTORY.md` under "SIDE-FIXES SHIPPED
— profiler units, telemetry filter, stale comments (2026-09-01)".

Repo-wide search (`grep -rln "PhaseTimer\|_ProfUnit" --include="*.py" .`) finds these
symbols used only inside `dsv4_mtp.py` itself — zero test files reference them. No unit
test, no integration test, nothing exercises the unit-tagging logic or the formatter
that consumes it.

**Guard exists: NO. Status: UNGUARDED.** If a future edit reintroduces the hardcoded
`"ms"` stamp (or mislabels a new count-type series), nothing would catch it.

## 6. Telemetry runner filter — UNGUARDED

Source: `collect_telemetry.py` (`parse_ps_runner`, now requiring the `comm` basename to
start with `"python"`), shipped in the same commit `bbb0e9341`, documented in
`docs/PERFORMANCE_HISTORY.md` under the same "SIDE-FIXES SHIPPED" entry (item 2:
"Telemetry restart-detector false positives").

This file only exists at `tmp/research-v1v2v3-20260901/telemetry/collect_telemetry.py`
— it is a one-off research/tooling script under `tmp/`, not part of `src/`, so it is
excluded from CI's `pytest src` scope by construction even if a test did exist. Repo-wide
search for `parse_ps_runner` and any `test_collect_telemetry*` filename returns zero
matches outside that one source file itself.

**Guard exists: NO. Status: UNGUARDED.** No test, and the source file itself lives
outside the tree CI or any conventional test discovery would ever scan.

## 7. Zero permanent modifications

```
$ git -C /Users/adam.durham/repos/exo/mlx-lm diff --stat
(empty)
$ git -C /Users/adam.durham/repos/exo/mlx-lm status --porcelain
(empty)
```
The only file this task ever mutated (`mlx-lm/mlx_lm/models/cache.py`) is confirmed
byte-identical to its pre-task state (md5 match) and shows zero diff. No files in the
exo repo were created, edited, staged, or committed by this task (findings file written
under `tmp/hardening-round1-20260903/findings/` per instructions is the sole new file).
