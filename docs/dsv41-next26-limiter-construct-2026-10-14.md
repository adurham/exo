# dsv41 reasoning-budget limiter — `deploy/next26-limiter` construct — 2026-10-14

## What this is

`deploy/next26-limiter` = `deploy/next25-guards` @ `0e9c7d93d` (the LIVE build: banked
small-n wide-tile route fix + dsv41 degeneration loop-guard, live-armed) **plus PHASE 2:
the reasoning-budget limiter** ported into the dsv41 round-batched greedy decode, plus the
launcher forwarding for its knobs.

Why: phase-2 of the degeneration-guard work. A still-open reasoning block that keeps running
(a paraphrase-drift / self-doubt loop, period 60-400+) could consume the entire generation
budget and leave the answer channel empty; the generic MLX path already forced the think-END
delimiter at `0.75 x max_tokens` (capped 16384) and the dsv41 engine had no such limiter.

- **Base**: `deploy/next25-guards` @ `0e9c7d93d` (live since 2026-10-13 22:50, armed).
- **Phase-2 commit** (from `origin/main` @ `7286f8979`, cherry-picked with `-x`):

  | cherry sha | source sha | summary |
  |---|---|---|
  | `eb582f151` | `7286f8979` | port: `ReasoningBudgetLimiter` on the committed-token stream; `rounds._one_round` cuts at the threshold + rolls the cache back to the committed prefix + one forced think-END round (off-by-one KV fact covered by tests); structured `error.code='repetition_loop'` on the guard clean-fail; optional per-request `reasoning_budget_tokens`; buildnote `docs/dsv41-reasoning-budget-limiter-2026-10-13.md`; 36 + 27 tests |

- **Construct-only hunks:**
  - `start_cluster.sh`: `EXO_REASONING_BUDGET*` forward block (family: `FRACTION`, `MAX_TOKENS`)
    — the runner env is an allow-list, so unforwarded knobs are silently absent.
    `MAX_SECONDS` is deliberately NOT forwarded: the dsv41 port rejects the generic path's
    wall-clock trigger (rank-local time is a hang class here).
  - `dsv41/fingerprint.py`: `KV_ALLOWLIST` entry for the new `dsv41.reasoning_budget` module
    (the CI guard flagged it, exactly like next25's `dsv41.degeneration`; it can cut + roll
    back the cache).

## Verification

Scoped suite on the construct worktree (mlx-lm submodule at `a0573e3`; mlx from the laptop venv):

```
EXO_DASHBOARD_DIR=<wt>/dashboard/build PYTHONPATH=<wt>/src:<wt>/mlx-lm \
/Users/adam.durham/repos/exo/.venv/bin/python -m pytest \
  src/exo/worker/engines/mlx/dsv41/tests \
  src/exo/worker/tests/unittests/test_runner/test_loop_detection.py \
  src/exo/worker/engines/mlx/tests/test_reasoning_budget_limiter.py -q -p no:cacheprovider
```

- **515 passed, 0 failed** (514 before the fingerprint-allowlist entry; the CI guard correctly
  flagged the new module first — same sequence as next25's `dsv41.degeneration`).
- `zsh -n start_cluster.sh` clean.

## Rollout

No staged rollout required. The limiter acts only INSIDE a still-open reasoning block once it
crosses `min(max_tokens * EXO_REASONING_BUDGET_FRACTION, EXO_REASONING_BUDGET_MAX_TOKENS)`
(defaults 0.75 / 16384); normal agentic calls never reach the threshold. Both TP ranks compute
the same cut/force from byte-identical committed ids (no collective, no new hang class).

Launch (established pattern; background + log):

```
cd ~/repos/exo && EXO_TARGET_BRANCH=deploy/next26-limiter ./start_cluster.sh
```

## Rollback

- Relaunch `deploy/next25-guards` @ `0e9c7d93d` (guards armed, no limiter).
- Knob off-switches at relaunch: `EXO_REASONING_BUDGET_FRACTION<=0` disables the limiter;
  per-request `reasoning_budget_tokens` overrides the default.

## Residual / not verified here

- No live-cluster validation: TP2 in-situ behaviour (both ranks cutting + forcing together) is
  argued from purity + byte-identical committed ids, not measured live.
- The post-force round is a cold start; the draft window is rolled to the committed prefix
  rather than re-primed (hint-only inconsistency; the target verifies every draft).
- `reasoning_budget_tokens` is not sent by any client yet — the env default governs.
