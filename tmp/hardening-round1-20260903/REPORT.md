# Hardening Round 1 — 2026-09-03

Perf campaign is CLOSED. This round protects the shipped wins, cuts launch cost, and
consolidates the record. No perf hunting, no attribution work, no cluster relaunch.

Every claim below was verified by the PM independently of the worker self-reports.
Two worker claims were corrected during verification (see "Corrections" at the end).

---

## A. tmp/ prune + rsync exclude

### Before / after

| Metric | Before | After |
|---|---|---|
| `du -sh ~/repos/exo/tmp` | **16 GB** | **34 MB** |
| Reclaimed | — | **16.88 GB** (145 files) |
| `*.md` files under tmp/ | 84 | **84** (unchanged) |
| REPORT/RESULTS/PRE-REGISTRATION | 18 | **18** (unchanged) |
| `*.patch` files | 2 tracked + 1 defective | **2** (defective one removed on purpose, see B) |

The evidence trail is fully intact. Deletion was manifest-driven, file-by-file, never
`rm -rf`. Manifest: `findings/DELETION_MANIFEST.tsv` (145 rows), inventory:
`findings/tmp_inventory.tsv` (1256 rows).

PM audit of the manifest: 0 `.md` files, 0 `.patch` files, and every path confined to
`~/repos/exo/tmp/`. Largest single contributor was an **11 GB Metal GPU trace capture**
(`p01-20260829/laptop_smoke/moe_capture.gputrace/`) — it did not match the task's literal
extension list but is unambiguously the class of bulk artifact the task targeted.

### The rsync exclude — IMPLEMENTED

`start_cluster.sh:1314`, inside the repo-sync rsync:

```
        --exclude 'tmp/' \
```

**Safety verified by the PM, not just asserted.** Grepping the launcher for repo-relative
`tmp/` references returns exactly one hit — line 592, a *comment* citing a finding path.
Nothing under `tmp/` is read during launch: not `config_examples/`, not `run_exo_on.sh`,
not `set_rdma_network_config.sh`, not `run_llm.sh`, not `prompt.txt`. The directory was
being copied and never consumed. `bash -n` passes.

### Expected launch-time saving

The 16 GB was the dominant term in the ~8 min rsync (`.venv`, `__pycache__`,
`node_modules`, `.pytest_cache` were already excluded; `mlx/build` ~1.0 GB is
deliberately *kept* because excluding it turns every relaunch into a full ~8 min MLX
rebuild). Excluding `tmp/` removes that transfer permanently and is robust to tmp/
regrowth — the prune alone would not have been.

**Estimate, not a measurement.** No relaunch was performed (per constraints), so the new
wall-clock was not observed. Expect the large majority of the ~8 min rsync cost to
disappear; confirm on the next real launch.

### ⚠ Finding the workers missed: the remote copies are stranded

`rsync --exclude` combined with `--delete` **protects** excluded paths from deletion on
the receiver. Verified read-only on both nodes:

```
macstudio-m4-1   16G  ~/repos/exo/tmp
macstudio-m4-2   16G  ~/repos/exo/tmp
```

So the exclude correctly stops *future* transfer cost, but 16 GB per node stays on disk
indefinitely. This costs no launch time — only ~32 GB of node disk. **Recommended
follow-up (not done, needs a write op on the nodes and this round was read-only):** a
one-time manual `rm -rf ~/repos/exo/tmp` on each Studio. Safe precisely because the
launcher never reads it.

---

## B. Defective instrumentation artifact — REMOVED (superseded)

**Disposition: deleted the defective copy, kept `instrumentation_as_run.patch` as the
single canonical artifact.**

Both documented defects were confirmed against the actual file bytes, not taken on faith:
an orphaned `_t0` (removal with no replacement → `NameError`) and 6 `[PROF]` sites using
stdlib `%`-arg style that loguru silently never interpolates.

Removal beat repair here: a repaired-in-place patch would be a *second* plausible-looking
copy of the same instrumentation, and the failure mode being defended against is exactly
"next person grabs the wrong one and burns a cluster launch." One canonical copy removes
the choice.

- Deleted: `tmp/prefill-round3-20260902/artifacts/decode_instrumentation.patch`
- Pointer left behind: `decode_instrumentation.patch.REMOVED.md` (same dir) → redirects
  to the canonical patch
- Canonical: `tmp/prefill-round4-exec-askb-20260902/revert/instrumentation_as_run.patch`
  — verified `_t0` properly restored (not orphaned), all 6 `[PROF]` sites pre-format
  their strings so loguru emits them correctly

The `revert/` directory is fully intact (all 4 original files).

---

## C. mx.eval bracket-close requirement — now in the code

Requirement recorded at all four bracket-close sites in
`src/exo/worker/engines/mlx/speculative/dsv4_mtp.py` (in-repo, not the submodule):

| Site | Phase | Comment at |
|---|---|---|
| batch cycle path | `accept` close | **dsv4_mtp.py:3089–3091** |
| batch cycle path | `rollback`/`total` close | **dsv4_mtp.py:3416–3418** |
| linear / temp>0 verify path | `accept` close | **dsv4_mtp.py:4960–4962** |
| linear / temp>0 verify path | `rollback`/`total` close | **dsv4_mtp.py:5478–5480** |

Each comment states that any phase-attribution profiling added there MUST force
materialization via `mx.eval`, explains why (MLX lazy eval otherwise bills deferred
compute to whichever phase happens to materialize it — the multi-round phantom), and
names both `instrumentation_as_run.patch` and `docs/PERFORMANCE_HISTORY.md`.

**No production behavior changed — proven, not asserted.** PM ran an AST comparison of
HEAD vs working tree:

```
AST mx.eval() CALL SITES   HEAD: 30   CURRENT: 30
FULL AST IDENTICAL (proves comment-only): True
```

The parsed syntax trees are byte-for-byte identical, so the 36-line diff is provably
comments only. (The worker's own `grep -c mx.eval` went 44→52 and would have looked
alarming; the AST check is the honest oracle.)

Skill-pitfalls edit was skipped as non-cheap. The code comments cannot be missed, which
was the task's stated preference.

---

## D. Regression guard table — does each guard have TEETH?

Standard applied: a test existing is **not** a guard. It has teeth only if it actually
FAILS when the fix is reverted — demonstrated by mutation, then restored.

| Win | Guard | Has teeth? | Verified how | Runs in CI? |
|---|---|---|---|---|
| **BatchPoolingCache overlap-carry resize** | `mlx-lm/tests/test_batch_pooling_cache_overlap.py` (4 tests) | ✅ **YES** | Deleted the reindex block in `BatchPoolingCache.filter()` (the actual fix from `37260bb`) → 2/4 failed on the `_overlap_carry_valid` assertions. Restored (md5-identical) → 4/4 green | ❌ **NO** |
| **Canonical serializer + golden bytes** | `hermes-agent tests/agent/test_exo_canonical_serializer.py` (29 tests) | ✅ **YES** | Swapped `content`/`reasoning_content` key order → 4 tests failed incl. the golden-bytes drift test. Restored → 29/29 green | ✅ yes |
| **Pad-strip** | `hermes-agent tests/run_agent/test_exo_reasoning_pad_omission.py` (16 tests) | ✅ **YES** | Two mutations: full predicate revert → 19 failures; realistic narrow revert (`build_assistant_message` pad only) → 2 failures | ✅ yes |
| **Profiler unit metadata** | none | ❌ **UNGUARDED** | No test anywhere references `_PhaseTimer` / `_ProfUnit` | n/a |
| **Telemetry runner filter** | none | ❌ **UNGUARDED** | No test for `parse_ps_runner`; the source only lives under `tmp/research-v1v2v3-20260901/`, outside pytest discovery entirely | n/a |

### Pad-strip: the task's open question, answered

The brief asked whether pad-strip is "covered by the serializer tests?" — **it is not**,
but it is guarded, by its own dedicated file. This distinction has teeth: under the
narrow, realistic mutation, **zero serializer tests failed** while the pad-omission suite
caught it. Had we relied on proximity and assumed serializer coverage, we would have
believed pad-strip was guarded by tests that demonstrably do not catch its regression.
Verdict: **GUARDED**, by `test_exo_reasoning_pad_omission.py`, not by the serializer suite.

---

## Not guarded / should be

1. **`BatchPoolingCache` test never runs in CI.** `pipeline.yml` scopes pytest to `src`
   only and does not checkout submodules recursively; mlx-lm's own upstream workflow is
   gated to `ml-explore/mlx-lm`, not the fork exo points at. The test has real local
   teeth and **zero teeth against drift** — nothing will ever run it unattended. Highest
   priority of the gaps here: the guard exists and is good, it's simply never fired.
2. **Profiler unit metadata — UNGUARDED.** No coverage at all.
3. **Telemetry runner filter — UNGUARDED**, and worse, its source sits under `tmp/`,
   which is now excluded from rsync and outside test discovery. If it matters, it needs
   to move into `src/` before a guard is even possible.
4. **🔴 HF tokens are in committed git history, and `.gitignore` only covers half of it.**
   The ignore rule is `research-v1v2v3-20260901/**/raw/`, but token patterns are present
   in *tracked, committed* blobs at:
   - `tmp/verify-decomposition-20260901/raw/accept_250k_n1.txt` — 2 pattern hits
   - `tmp/research-v1v2v3-20260901/v3/run1/exo_m4-1.log` — 8 pattern hits (a raw node-log
     copy that is **not** inside a `raw/` dir, so the ignore rule never applied)

   Deleting them from the working tree does **nothing** about history. Root-cause fix is a
   `git-filter-repo` purge + token rotation — out of scope for this round and it rewrites
   every SHA, so it needs the user's explicit go-ahead. Counts only were checked; no token
   content was printed, copied, or staged.

---

## Commit boundary

Committed (autonomous authorization for exo-repo changes):
- `src/.../dsv4_mtp.py` — the 4 mx.eval comments (AST-proven comment-only)
- `start_cluster.sh` — the `--exclude 'tmp/'` line. **Note:** this file also carried two
  *pre-existing* uncommitted env allow-list lines (`EXO_DSV4_SDPA_CALL_PROFILE`,
  `EXO_DSV4_DECODE_COLLECTIVE_PROFILING`) from earlier campaign work that were never
  committed; they ride along in this commit and are called out here rather than silently
  bundled.
- Deletion of the defective patch + its `.REMOVED.md` pointer
- This report and `findings/`

**Deliberately NOT committed:** the 5 tracked deletions of raw node-log copies. The task
constraint is absolute ("never commit anything from a `raw/` directory") and these are
exactly the files carrying token patterns. They remain deleted in the working tree but
unstaged, pending the filter-repo decision in item 4 above — which would remove them from
history entirely and subsume the question.

---

## Corrections to worker self-reports

Recorded because both were caught by verification rather than by the worker:

1. **rsync-vs-git deploy.** The prune worker claimed "the rsync (not git) is the actual
   src/ deploy mechanism," contradicting `exo-cluster-operations` pitfall #51. Reading the
   launcher directly: the rsync comments state it *replaces* the old `git reset --hard`
   semantics via `--delete`, and `.git` is synced so the git-based consistency checks keep
   working. The worker's conclusion was right, but the skill's pitfall #51 is now **stale**
   on this point and should be updated by whoever next touches that skill.
2. **`grep -c mx.eval` 44→52** looked like added eval calls; the AST comparison shows the
   real call-site count is unchanged at 30 and the trees are identical. Grep was the wrong
   oracle for that assertion.
