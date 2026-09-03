# Task B + C findings — hardening round 1 (2026-09-03)

## Task B: defective instrumentation patch

### Defect confirmation (against actual file contents, not PATCH_DEFECTS.md's word)

- **Defect 1 (orphaned `_t0`)**: confirmed. The defective patch
  (`tmp/prefill-round3-20260902/artifacts/decode_instrumentation.patch`)
  line 181 removes `-        _t0 = time.perf_counter()` and never re-adds
  it, while the unmodified downstream `_dt = time.perf_counter() - _t0`
  line still exists in the target function — a NameError on the slow
  path. Verified via `grep -n '_t0'` on the defective patch: only the
  removal hunk appears, no replacement `+` line.
- **Defect 2 (C-style `%`-args passed to loguru)**: confirmed. All six
  new `[PROF]`-prefixed `logger.info(...)` call sites in the defective
  patch pass the format string and `%`-placeholders (`%.1f`, `%s`, `%d`)
  as separate positional args (stdlib-`logging` calling convention).
  loguru does not do `%`-interpolation this way; it silently emits the
  literal placeholder text. Confirmed by inspecting all six sites'
  diff context in the defective patch.

### Decision: **(b) — remove the defective copy, keep the corrected one as sole canonical artifact**

Rationale: a defective patch that "looks usable" (applies cleanly, is
version-controlled, has a plausible name) is strictly worse than no
patch — the next person who reapplies it burns a cluster launch on a
guaranteed `NameError` (defect 1 alone is fatal). Repairing in place was
considered but rejected: the corrected file
(`instrumentation_as_run.patch`) is already the exact byte-for-byte
"shipped patch + 2 live fixes" artifact that was actually run and
validated on real hardware during round 4. Hand-patching the defective
copy to match it would just be recreating that file via a second,
error-prone transcription path with no upside — better to point
directly at the one that has already been through a real measurement
run.

### Corrected-patch verification (bar for "blessing it as canonical")

- `grep -n '_t0'` on `instrumentation_as_run.patch`: 2 hits — line 187 is
  the **removal** hunk line (`-        _t0 = ...`, same upstream context
  as the defective patch), line 201 is the **replacement** `+` line that
  restores it with an explicit comment
  (`# upstream timing var (patch-orphaned) — restored per G2 gate fail`).
  No orphaned reference remains — the removed line is re-added.
- `[PROF]` call sites checked: **6** (same six sites as the defect
  report: `agree_on_tasks.gate_us` ×2 branches,
  `agree_on_tasks.all_gather_us`,
  `agree_on_cancellations_fast.gate_us` ×2 branches,
  `agree_on_cancellations_fast.all_gather_us`). All six use the
  concatenated-format-string `% (...)` pre-formatting form — the whole
  formatted string is a single `str` argument, computed via plain
  Python `%`-operator *before* `logger.info` ever sees it, independent
  of loguru's own placeholder semantics. Zero remaining bare
  C-style-arg call sites (`grep -c '^\+.*\[PROF\]'` = 6, and manual
  inspection of all 6 confirms the `% (` pattern on every one).

**Verdict: corrected patch is sound. Blessed as canonical.**

### Action taken

- Deleted `tmp/prefill-round3-20260902/artifacts/decode_instrumentation.patch`
  from the **filesystem only**. This file is git-tracked, so the
  deletion is currently an **unstaged working-tree change**
  (`git status --porcelain` shows `D  tmp/prefill-round3-20260902/artifacts/decode_instrumentation.patch`
  — no `git rm`, no `git add`, nothing staged).
- Left a pointer file at the old path:
  `tmp/prefill-round3-20260902/artifacts/decode_instrumentation.patch.REMOVED.md`
  — explains the defects, and redirects to
  `tmp/prefill-round4-exec-askb-20260902/revert/instrumentation_as_run.patch`
  as the canonical replacement.

**PM ACTION REQUIRED:** run
`git rm tmp/prefill-round3-20260902/artifacts/decode_instrumentation.patch`
to stage the removal (the file is already gone from disk; this only
updates the index) and commit. I did **not** touch the git index.

## Task C: mx.eval methodology comments

### File located

`dsv4_mtp.py` lives in `src/exo/worker/engines/mlx/speculative/dsv4_mtp.py`
inside the exo repo proper (not a submodule) — found via
`find . -name dsv4_mtp.py`.

### Real bracket-close sites (task's line numbers were approximate; actual sites differ)

There are **three** `accept`/`rollback`/`total` phase-timer code paths
in this file (batch path, tree-verify path, linear-verify path), giving
**more than four** raw candidate sites. I annotated the four sites that
correspond to the task's two named pairs — the accept-close and
matching rollback/total-close in the **batch** cycle path (closest
structural match to the task's "3088"/"3406" pair) and in the **linear
verify** cycle path (closest match to the task's "4940"/"5449" pair).
Exact file:line (current, post-edit):

1. **accept close (batch path)** —
   `src/exo/worker/engines/mlx/speculative/dsv4_mtp.py:3087`
   (comment starts at 3087; original code `if prof is not None:` now at
   3096, `prof.record("accept", ...)` at 3098)
2. **rollback/total close (batch path)** —
   `src/exo/worker/engines/mlx/speculative/dsv4_mtp.py:3414`
   (comment starts at 3414; `prof.end_cycle(N)` now at 3432)
3. **accept close (linear/temp>0 verify path)** —
   `src/exo/worker/engines/mlx/speculative/dsv4_mtp.py:4958`
   (comment starts at 4958; `prof.record("accept", ...)` now at 4969)
4. **rollback/total close (linear/temp>0 verify path)** —
   `src/exo/worker/engines/mlx/speculative/dsv4_mtp.py:5473`
   (comment starts at 5473; `prof.end_cycle(1)` now at 5491)

(A third accept/rollback pair exists in the tree-verify path, around
what was originally line ~5680/~5791 before edits — I left it
un-annotated since the task named exactly two pairs and four sites;
flagging this as a possible gap — see "Note for triage" below.)

### Comment content (identical template at all 4 sites)

Each comment:
- States the requirement: any phase-attribution profiling added at
  this bracket close MUST call `mx.eval()` here to force
  materialization.
- Explains why: MLX's lazy evaluation otherwise attributes the
  deferred compute to whichever later phase happens to materialize it.
- Names the artifact:
  `tmp/prefill-round4-exec-askb-20260902/revert/instrumentation_as_run.patch`
  as the worked example.
- Names the record: `docs/PERFORMANCE_HISTORY.md`.
- States this produced a multi-round phantom during the perf campaign.
- Explicitly says "Comment only; no mx.eval added here" to make the
  no-behavior-change intent unambiguous to the next reader.

### Note for triage (found but out of the stated 4-site scope)

The tree-verify (`N`-way batched tree speculation) cycle path has its
own accept/rollback/total closes, structurally identical to the other
two paths, but the task named only two site-pairs / four sites total.
I did not add a comment there to stay in scope — flagging so the PM can
decide whether it needs the same annotation in a follow-up.

## Acceptance assertions — pass/fail

1. **`git diff` on dsv4_mtp.py shows ONLY added comment lines** — **PASS**.
   Full diff (4 hunks, 36 insertions, 0 deletions, 0 modified
   non-comment lines):

   ```diff
   diff --git a/src/exo/worker/engines/mlx/speculative/dsv4_mtp.py b/src/exo/worker/engines/mlx/speculative/dsv4_mtp.py
   index e13c7b094..e5ded70ca 100644
   --- a/src/exo/worker/engines/mlx/speculative/dsv4_mtp.py
   +++ b/src/exo/worker/engines/mlx/speculative/dsv4_mtp.py
   @@ -3084,6 +3084,15 @@ class DSv4MTPBatchGenerator(MTPBatchGenerator):
                        uid, [int(tid) for (tid, _lp) in all_tokens_per[n]]
                    )

   +        # METHODOLOGY REQUIREMENT (perf campaign, see
   +        # docs/PERFORMANCE_HISTORY.md and the worked-example patch at
   +        # tmp/prefill-round4-exec-askb-20260902/revert/instrumentation_as_run.patch):
   +        # any phase-attribution profiling added at this accept bracket
   +        # close MUST force materialization via mx.eval() here, or MLX's
   +        # lazy evaluation will attribute the deferred compute to
   +        # whichever later phase happens to trigger materialization —
   +        # this exact mistake produced a multi-round phantom during the
   +        # perf campaign. Comment only; no mx.eval added here.
            if prof is not None:
                t_after_accept = time.perf_counter()
                prof.record("accept", (t_after_accept - t_after_verify) * 1000.0)
   @@ -3402,6 +3411,15 @@ class DSv4MTPBatchGenerator(MTPBatchGenerator):
                    all_tokens_per,
                )

   +        # METHODOLOGY REQUIREMENT (perf campaign, see
   +        # docs/PERFORMANCE_HISTORY.md and the worked-example patch at
   +        # tmp/prefill-round4-exec-askb-20260902/revert/instrumentation_as_run.patch):
   +        # any phase-attribution profiling added at this rollback/total
   +        # bracket close MUST force materialization via mx.eval() here,
   +        # or MLX's lazy evaluation will attribute the deferred compute to
   +        # whichever later phase happens to trigger materialization —
   +        # this exact mistake produced a multi-round phantom during the
   +        # perf campaign. Comment only; no mx.eval added here.
            if prof is not None:
                t_after_rollback = time.perf_counter()
                prof.record(
   @@ -4937,6 +4955,15 @@ class DSv4MTPBatchGenerator(MTPBatchGenerator):
                    except Exception as _audit_err:  # never break generation
                        logger.warning(f"verify-audit(temp>0) failed: {_audit_err}")

   +        # METHODOLOGY REQUIREMENT (perf campaign, see
   +        # docs/PERFORMANCE_HISTORY.md and the worked-example patch at
   +        # tmp/prefill-round4-exec-askb-20260902/revert/instrumentation_as_run.patch):
   +        # any phase-attribution profiling added at this accept bracket
   +        # close MUST force materialization via mx.eval() here, or MLX's
   +        # lazy evaluation will attribute the deferred compute to
   +        # whichever later phase happens to trigger materialization —
   +        # this exact mistake produced a multi-round phantom during the
   +        # perf campaign. Comment only; no mx.eval added here.
            if prof is not None:
                t_after_accept = time.perf_counter()
                prof.record("accept", (t_after_accept - t_after_verify) * 1000.0)
   @@ -5446,6 +5473,15 @@ class DSv4MTPBatchGenerator(MTPBatchGenerator):
                    [all_tokens],
                )

   +        # METHODOLOGY REQUIREMENT (perf campaign, see
   +        # docs/PERFORMANCE_HISTORY.md and the worked-example patch at
   +        # tmp/prefill-round4-exec-askb-20260902/revert/instrumentation_as_run.patch):
   +        # any phase-attribution profiling added at this rollback/total
   +        # bracket close MUST force materialization via mx.eval() here,
   +        # or MLX's lazy evaluation will attribute the deferred compute to
   +        # whichever later phase happens to trigger materialization —
   +        # this exact mistake produced a multi-round phantom during the
   +        # perf campaign. Comment only; no mx.eval added here.
            if prof is not None:
                t_after_rollback = time.perf_counter()
                prof.record(
   ```

2. **`grep -c 'mx.eval'` identical before/after** — reported both the
   naive-regex count and the real AST-level call-site count because the
   naive regex also matches the word "mx.eval" inside my new comment
   text (expected — the comments literally reference `mx.eval()`):
   - Naive `grep -c 'mx.eval'` (regex, `.` = wildcard): **HEAD 44 → working
     tree 52** (+8, all 8 accounted for as substring hits inside the 4
     new comment blocks — `git diff | grep '^+' | grep -c 'mx\.eval'` = 8).
   - Real invocation count via Python AST (only `Call` nodes whose
     callee is `mx.eval(...)`, i.e. cannot be fooled by comment text):
     **HEAD 30 → working tree 30 — IDENTICAL. PASS.** This is the hard
     proof requested: zero new `mx.eval` call sites were added; every
     new "mx.eval" string in the diff lives inside a `#`-prefixed
     comment line.
3. **File still parses** —
   `python3 -c 'import ast,sys; ast.parse(open("src/exo/worker/engines/mlx/speculative/dsv4_mtp.py").read())'`
   exits 0. **PASS.**
4. **All four bracket-close sites have a comment** — file:line listed
   above (3087, 3414, 4958, 5473 — comment start lines). **PASS.**
5. **Each comment names the instrumentation_as_run.patch artifact
   path** — verbatim
   `tmp/prefill-round4-exec-askb-20260902/revert/instrumentation_as_run.patch`
   appears in all 4 comment blocks. **PASS.**
6. **Task B end state** — defective patch **removed** from the
   filesystem (not repaired in place); redirect pointer left at the old
   path; canonical replacement is
   `tmp/prefill-round4-exec-askb-20260902/revert/instrumentation_as_run.patch`.
   PM must `git rm` the old path to stage the deletion — not done by me.
   **PASS** (per the task's own acceptance framing for option (b)).
7. **N/A** — I chose option (b) (remove), not (a) (repair in place), so
   there is no repaired-patch `_t0`/`%`-arg count to report for the
   *committed* copy. For completeness, verification WAS done on the
   corrected copy that is now canonical (see above): `_t0` orphan count
   in `instrumentation_as_run.patch` = 0 remaining after the restore
   (`grep -c '_t0'` = 2 total lines, one `-` removal + one `+`
   replacement, i.e. the reference is properly restored, not orphaned);
   remaining C-style `%`-arg `[PROF]` call sites in that file = 0 (all 6
   use the `% (...)` pre-formatted-string form).
8. **`git status --porcelain` shows nothing staged attributable to me**
   — confirmed: `git diff --name-only --cached` returns empty. My two
   changes (`dsv4_mtp.py` edit, `decode_instrumentation.patch` deletion)
   both show as unstaged (`M ` / `D ` in porcelain output with a blank
   first column, meaning working-tree-only, not index). **PASS.**
9. **No production behavior changed; only executable-code delta is
   zero lines** — confirmed by assertion #1 (diff is comment-only) and
   assertion #2 (AST-level `mx.eval` call count unchanged: 30 → 30).
   **PASS.**

## Bonus: exo-cluster-operations skill pitfalls edit

**Skipped** — the skill's SKILL.md is large (>120KB, prior tool call
against it hit the spillover-file threshold) and its pitfalls section
was not quick to safely locate and patch within this task's budget
without risking a bad edit to a shared, unrelated skill file. The
task explicitly said "skip if not quick, report if skipped." The four
production code comments are the primary, unmissable deliverable for
this requirement per the task's own stated preference ("prefer the
code comments — they cannot be missed").
