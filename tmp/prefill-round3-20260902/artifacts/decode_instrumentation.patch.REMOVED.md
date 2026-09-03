# decode_instrumentation.patch — REMOVED (was defective)

The file that used to live here (`decode_instrumentation.patch`) has been
removed from disk because it was **defective**: applying it would raise a
`NameError` from an orphaned `_t0` reference in
`BatchGeneratorRunner.agree_on_tasks`'s slow path, and all six new
`[PROF]` logger.info calls it added used C-style `%`-style args that
loguru silently fails to interpolate (the log lines would show literal
`%.1f`/`%s`/`%d` text instead of real numbers).

Full defect analysis: `tmp/prefill-round4-exec-askb-20260902/revert/PATCH_DEFECTS.md`

**Use the corrected, verified-working copy instead:**

```
tmp/prefill-round4-exec-askb-20260902/revert/instrumentation_as_run.patch
```

That file is the shipped patch plus the two live fixes applied during the
round-4 measurement run (orphaned `_t0` restored; all six `[PROF]` sites
converted to pre-formatted `%` strings so loguru actually interpolates
them). It is the single canonical copy of this instrumentation patch going
forward.

## Action needed from whoever finds this

The defective file was deleted from the filesystem only. It is still
**git-tracked** (committed in history), so removing it from the git index
requires:

```
git rm tmp/prefill-round3-20260902/artifacts/decode_instrumentation.patch
```

This has **NOT** been run — no git index/staging changes were made as part
of this cleanup. A human/PM must run the `git rm` (and can also `git rm`
this pointer file, or keep it, depending on their preference for how this
directory should look post-cleanup) and commit.
