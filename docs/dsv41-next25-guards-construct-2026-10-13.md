# dsv41 degeneration loop-guard — `deploy/next25-guards` construct — 2026-10-13

## What this is

`deploy/next25-guards` = `deploy/next24-routing` @ `9fe1d593` (the LIVE build: banked
small-n wide-tile route fix) **plus the dsv41 degeneration loop-guard** (the
committed-token repetition detectors ported from the generic MLX path) + the
launcher forwarding for its knobs. Construct for a STAGED live rollout: first launch
with `EXO_LOOP_DETECT_ACTION=warn` (observe + capture, no behavior change), then
relaunch without it (engine default `error`) to arm termination.

Why: 2026-10-13 live incident — an interactive agentic session on this model entered
an exact-repetition loop; the dsv41 engine (greedy-only, deliberately bypasses the
generic path) had NO loop detector at all, so the generation streamed 3m41s of
unbroken repetition until the client killed it. The generic path's detectors
(`_detect_token_loop`, `_detect_long_period_loop`) never covered this engine.

- **Base**: `origin/deploy/next24-routing` @ `9fe1d5930` ("indexer small-n wide tile
  (R-B) — advance mlx-lm pin + buildnote + kill-switch forward line").
- **Guard commits** (from origin/main @ `a9d4700f4`, cherry-picked with `-x`):

  | cherry sha | source sha | summary |
  |---|---|---|
  | `3eef8fdbf` | `12548e373` | port: both detectors wired into `engine._generate`'s committed-token choke point (every committed token, in order; termination through the existing terminal plumbing; rollback via `_abort_turn`); per-request observability (first-token latency, ~10s progress, turn-end line, guard-fire WARNING with cross-rank-comparable env digest, <=2048-id dump on guard-fire OR client cancel); 45 tests |
  | `0fed048d3` | `a9d4700f4` | adversarial-audit fixup: separator-only period-1 exemption (shared detector `token_text`), startup cross-rank guard-env equality check (fail-fast at the construction lockstep), buildnote rewrite to the REAL client path (partial-delivery no-retry -> `length` stub -> bounded auto-continue), minor fixes |

- **Integration method**: `git cherry-pick -x` of the two commits onto `9fe1d593`.
  ONE conflict (import block in engine.py: next24's `ParkCoordinator` import vs the
  guard-module import) — resolved by keeping both; everything else auto-merged.
  Plus two construct-only hunks:
  - `start_cluster.sh`: `EXO_LOOP_DETECT*` forward block (family:
    DETECT / _WINDOW / _MAX_PERIOD / _MIN_REPEATS / _ACTION / _LONG /
    _LONG_BLOCK / _LONG_MAX_PERIOD / _LONG_MIN_REPEATS) — the runner env is an
    allow-list, so unforwarded knobs are silently absent.
  - `dsv41/fingerprint.py`: `dsv41.degeneration` added to `KV_ALLOWLIST` (the CI
    guard correctly flagged the new module; allowlisted per policy default — it is
    on the decode path and can roll a session back).

## Verification

Scoped suite on the construct worktree (mlx-lm submodule at the base's `a0573e3`,
mlx at `603f16eb`):

```
EXO_DASHBOARD_DIR=<wt>/dashboard/build PYTHONPATH=<wt>/src:<wt>/mlx-lm \
/Users/adam.durham/repos/exo/.venv/bin/python -m pytest \
  src/exo/worker/engines/mlx/dsv41/tests \
  src/exo/worker/tests/unittests/test_runner/test_loop_detection.py -q -p no:cacheprovider
```

- **452 passed, 0 failed** (451 before the fingerprint-allowlist fix).
- `zsh -n start_cluster.sh` clean (launcher change is syntax-valid).
- Real-text replay re-verified in the port tests: sessions 90d257 (43,892 tok) and
  180a8c (32,588 tok) -> 0 detections (no false positives).

## Staged rollout (this construct's purpose)

1. **Warn stage** — launch with `EXO_LOOP_DETECT_ACTION=warn` (forwarded by the
   launcher): the guard LOGS + DUMPS but never terminates. Observe real fire
   rate/content with no behavior change.
   Verify live: `ps eww <runner-pid>` shows the guard vars on BOTH nodes; the
   runner log's startup guard-env banner prints the effective values once per rank
   (`guard_env_digest=` must match between ranks).
2. **Arm stage** — relaunch WITHOUT the override (engine default `error`): a
   detected loop terminates the turn at its onset (~150-900 committed tokens,
   ~1-2 min) instead of the 3m41s runaway. Client side then takes the
   partial-delivery path (no retry -> `length` stub -> bounded auto-continue).
3. **Live proof to capture during the warn stage** — degenerate prompt ->
   `DEGENERATION DETECTED` WARNING in the runner log; client-cancel -> the
   <=2048-id committed-token dump under the exo log dir.

## Rollback

- `EXO_LOOP_DETECT=0` at relaunch disables the guard entirely (shared knob).
- `EXO_LOOP_DETECT_ACTION` unset = `error` (engine default); `=stop` = clean exit.
- Abandoning the construct = relaunch `deploy/next24-routing` @ `9fe1d593`.

## Residual / not verified here

- No live-cluster validation yet — that IS the staged rollout above.
- Restart-survival integration: the guard's early-return turn-ending path bypasses
  `_end_turn`/`finish`; before merging the guard with `deploy/next21-restart` /
  `next22-rankfix` (which rewrite session.py/park.py), their per-turn
  park/checkpoint invariants must be audited to tolerate a turn ending via
  `_abort_turn`. (This construct is based on next24, which does not carry them.)
- Paraphrase-drift loops (period 60-400+) remain uncaught (phase-2
  reasoning-budget limiter, out of scope).
- The already-streamed degenerate tail cannot be retracted; the guard only
  controls what happens at detection time.
