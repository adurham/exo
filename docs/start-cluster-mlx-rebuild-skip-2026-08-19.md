start_cluster.sh: skip mlx C++ rebuild when nothing changed (2026-08-19)
==============================================================================

Problem
----------

Every single cluster relaunch tonight -- regardless of whether the mlx
submodule's C++ source actually changed -- triggered a full ~5-8 minute
rebuild (28 parallel clang processes) via an unconditional
`--force-reinstall`. This included relaunches that only changed an env
var, restored a config, or touched an unrelated file. Real, repeated
friction across the session.

Root cause
-------------

`start_cluster.sh` (line ~1233) ran
`uv pip install --no-deps --force-reinstall ./mlx` unconditionally on
every relaunch, with no check for whether the compiled extension was
already up to date.

Fix
------

Made the mlx rebuild step conditional on ALL of:
- `MLX_FORCE_REINSTALL` unset (escape hatch to force a rebuild)
- `git -C mlx diff --quiet` AND `git -C mlx diff --cached --quiet`
  (tracked-file dirty check -- deliberately does NOT check untracked
  files, so build artifacts inside the mlx submodule don't force a
  rebuild every time)
- a stamp file (`.venv/.mlx-installed-sha`) matching the current mlx
  submodule HEAD SHA
- `import mlx` succeeding in the current venv (catches a venv that
  lost the package without the stamp knowing)

The stamp lives INSIDE `.venv/` specifically so it's destroyed along
with the venv on any `uv sync --reinstall` / fresh clone -- it can
never go stale relative to what's actually installed. Stamp is written
only after a successful install.

The mlx-lm step's own `--force-reinstall` (a separate, cheap
Python-only reinstall, no C++ compile) was deliberately left
untouched -- it has its own documented staleness-bug history (an
`uv`-version-string quirk that silently kept an old mlx_lm import alive
after a submodule bump, causing a real production warmup crash). This
fix only touches the expensive mlx C++ step.

Safety review
----------------

Consulted twice before landing (once on the general pattern, once
specifically on whether it's safe to touch only the mlx step and leave
mlx-lm's force-reinstall alone). Both consults flagged concrete failure
modes, all addressed:
- dirty-tree false-negative (untracked build/ dirs) -> tracked-only diff
- stamp outliving a venv recreation -> stamp lives inside .venv/
- stamp written on a failed build -> only written after success
- venv losing the package without stamp knowing -> import check added
- per-node correctness (2-node cluster) -> each node's SSH command runs
  its own independent check, verified separately on both nodes before
  committing

Verification (before AND after committing)
-----------------------------------------------

1. Dry-run tested all three logic branches manually against BOTH live
   nodes before ever touching the real script path:
   - no stamp -> NEED_BUILD=1 (correct, first run)
   - matching stamp + clean tree -> NEED_BUILD=0 (correct, skip)
   - matching stamp + DIRTY tree (manually dirtied a tracked file,
     `git -C mlx diff` non-empty) -> NEED_BUILD=1 (correct, forces
     rebuild -- this is the safety-critical case)
   Cleaned up the dirty-tree test artifact and removed the test stamp
   before committing, so the real first relaunch would exercise the
   genuine first-build path end to end.
2. Real end-to-end test, TWO full relaunches back to back on the live
   cluster (not simulated):
   - Relaunch 1 (no stamp yet): real mlx build ran, cluster came up
     READY (2/2), healthy. Total time: 10:02.
   - Relaunch 2 (stamp now matches, tree clean): "mlx unchanged...
     skipping rebuild" printed on BOTH nodes, cluster came up READY
     (2/2), healthy. Total time: 6:38 -- a real ~34% reduction, and
     this number still includes rsync, the exo/rust rebuild, and the
     dashboard build, none of which this fix touches. The mlx-specific
     savings (minutes of clang) are a larger fraction of what remains
     to optimize.

Standing config confirmed healthy at both checkpoints
(`EXO_PREFILL_STEP_SIZE=2048`, no experimental flags).

What this does NOT fix
---------------------------

The mlx-lm pin step, the exo/rust rebuild, dashboard build, and rsync
sync step are all unaffected -- there's likely further relaunch-speed
headroom there, not investigated tonight. `MLX_FORCE_REINSTALL=1` is
available as a manual override if this skip logic is ever suspected of
masking a real staleness bug.
