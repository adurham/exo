# dsv41 restart-survival — `deploy/next22-rankfix` construct — 2026-10-11

## What this is

`deploy/next22-rankfix` = `deploy/next21-restart` (production-ready restart-survival
line) **plus the audited rank-continuity fix**. It is the offline *construct* for the
next22 live redeploy + restart-survival re-validation: it must carry the rank-continuity
behaviour (stable SITE identity admission + rank-invariant park layout + startup identity
agreement) on top of the already-deployed next21 line. Verified only with the scoped
dsv41 suite (no cluster, no node contact, no boots).

- **Base**: `origin/deploy/next21-restart` @ `3afce11d78a3a8b13e7438631fe90d9cedb4fb94`
  ("deploy(next21): forward EXO_DSV41_CKPT*/SHUTDOWN_* knobs through the launcher allow-list").
- **Fix**: `origin/p20/dsv41-restart-rankfix` @
  `bebeb006629573a22c10048f489835ce83114194` — 3 commits, linear, each a strict descendant
  of the base:

  | # | sha | summary |
  |---|-----|---------|
  | 1 | `ea3175a6b` | rank-continuity — SITE identity admission + rank-invariant park layout + startup identity agreement |
  | 2 | `30f32e448` | rank-continuity post-audit fix — don't latch a negative identity verdict; keep window/bsz in the park layout |
  | 3 | `bebeb0066` | buildnote for the rank-continuity fix (design, audit findings, test evidence) |

- **Integration method**: `git worktree add -b deploy/next22-rankfix <wt> bebeb0066` —
  i.e. branch **pointed directly at the fix tip**, verbatim. No cherry-pick, no rebase,
  no merge, no patch application, no construct-only edits.

## Conflict / zero-conflict statement

**Zero conflicts by construction: none was possible.** `deploy/next22-rankfix` is a
branch ref at `bebeb0066` — the exact commit of the audited fix tip — with no
integration step that could conflict. `git merge-base --is-ancestor
3afce11d7 bebeb0066` → **true** (the fix is a clean 3-commit fast-forward off the
next21 base). No `<<<<<<<`/`=======`/`>>>>>>>` markers exist, and the tree is
byte-identical to the fix tip (`git rev-parse HEAD` → `bebeb0066…` before the
docs-only note commit below).

## Verification

Scoped dsv41 suite on THIS construct worktree (`/private/tmp/p20-next22`), mlx-lm
submodule at the pinned `689e4eafd3cf2925a1cf78ba81375a3c098a0eef`:

```
EXO_DASHBOARD_DIR=/Users/adam.durham/repos/exo/dashboard/build \
PYTHONPATH=/private/tmp/p20-next22/src:/private/tmp/p20-next22/mlx-lm \
/Users/adam.durham/repos/exo/.venv/bin/python -m pytest \
  src/exo/worker/engines/mlx/dsv41/tests -q
```

- **`367 passed in 51.63s`** (0 failed, 0 skipped) — matches the orchestrator's
  independent re-run of 367 at the fix tip. ≥ 354 (next21 construct baseline) + 13
  new rank-continuity cases. ✅

## Residual risk / NOT verified here

- **No hardware / no cluster**: this construct is validated only by the scoped
  synthetic suite. The live park→restart→restore round trip (graceful + `kill -9`
  crash cases, site-match restore, TTFT bar, needles, top-1, DSpark) is the live
  Phase B and is NOT covered by this document.
- **Pre-existing dirs cold-accept once**: the first post-deploy relaunch still
  cold-restores the pre-fix `.r<rank>` dirs (documented in the fix buildnote); only
  dirs written by this build restore across a rank swap.
- **Not merged to `main`**: branch `deploy/next22-rankfix` only.
