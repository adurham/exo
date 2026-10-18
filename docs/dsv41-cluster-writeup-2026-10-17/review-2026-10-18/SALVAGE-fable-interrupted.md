# Salvage note — interrupted Fable batch (deleg_2fafe007), 2026-10-18

The 3 Fable subagents were stopped mid-run when routing was corrected to
"Fable consults, DSv4.1 executes". They wrote no findings files, but their live
transcripts contain real verification output that is worth keeping as a
cross-check against the DSv4.1 findings (deleg_712f14d2, same ground).

Transcripts (raw, complete tool/assistant trace):
- `~/.hermes/cache/delegation/live/deleg_2fafe007/task-0.log` (§0-§6 fact-check)
- `~/.hermes/cache/delegation/live/deleg_2fafe007/task-1.log` (§7-§12 fact-check)
- `~/.hermes/cache/delegation/live/deleg_2fafe007/task-2.log` (copy-edit)

## What they actually verified (raw results, not yet triaged)

**task-1 — §12 code claims:**
- `start_cluster.sh:61-66` knob defaults read live: `CONTIG_REPEAT_MAX_PERIOD=64`,
  `MIN_BLOCKS=4`, `WORD_ONLY=1`, `SOFT_MULTIPLIER=2.0`, `SOFT_BASE=1.75`,
  `SOFT_ALLOWED=4` — matches §12.6's stated "64 / 4 / 1 / 2.0 / 1.75 / 4". SUPPORTED.
- Commit ancestry probe: `579183af6`, `f3010c4e2`, `cd54a361a`, `4c0fdb35b`,
  `d6ff8ad9e`, `032641805` all NOT in `b88d65dd3`; `991cf11` IS in it. Consistent
  with §11.2 marking `b88d65dd3` post-capture — needs a read to confirm the
  writeup states it that way.
- `f3010c4e2` commit message says `MIN_BLOCKS=3` while the code says `4`. Code is
  authoritative and §12.6 quotes the code; flag only if the writeup cites the
  commit message instead.
- `reasoning_budget.py:62-63,75-76` limiter defaults read live.
- `docs/PERFORMANCE_HISTORY.md:12176` — the 2026-10-17 decode-liveness entry exists.

**task-0 — §0-§6 sources:**
- `mlx/mlx/distributed/jaccl/lib/jaccl/mesh.cpp` exists, 2089 lines (the
  upstream-refactored `lib/jaccl` layout, not a top-level `mlx/jaccl/`).
- `docs/dual-cable-topology-and-qp-budget-2026-08-21.md` exists, 220 lines.
- `src/exo/worker/runner/supervisor.py:89` `HANG_TIMEOUT_SECONDS = 45.0`,
  `:123` probe interval `20.0` — matches the writeup's watchdog timing.
- `mlx-lm/mlx_lm/models/deepseek_v41/config.py:34` `n_layers: int = 40`.
- `docs/deepseek-v41-memory-budget-correction-2026-09-27.md` carries the
  108.76 GB running-code figure.
- `docs/moe-all-sum-dominant-cost-2026-08-19.md:18-19` carries the 62.0% / 61.1%
  all_sum shares.

**task-2 — mechanical sweeps (raw output in the transcript):**
- Absolute-quantifier hits with line numbers.
- Full quantity map (GB/GiB/MiB/tok-s/tokens) with every line number.
- Hex-id inventory with lengths and line numbers.
- `t/s` lines: 553, 554, 569, 572, 1406, 1414.
- Candidate: corpus doc pins `115000 MB` (120.6 GB) while the live value is
  `120000 MiB` (125.83 GB) — §5.4's headroom recompute rests on the 120000 MiB
  reading. Worth a read to confirm the writeup states the basis.

## Status

Not triaged. Fold anything that survives into the DSv4.1 findings before the
Fable ruling pass. Do not treat these as findings — they are raw verification
output from an interrupted run.
