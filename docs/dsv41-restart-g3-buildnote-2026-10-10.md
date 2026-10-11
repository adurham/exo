# DSv4.1 restart survival — G3 build note (2026-10-10)

Gap **G3** of `docs/dsv41-restart-survival-plan-2026-10-10.md` (park-on-shutdown
/ graceful-restart fast path), as corrected by the Phase-0 audit
(`docs/dsv41-restart-survival-phase0-audit-2026-10-10.md`, A8). Branch
`p20/dsv41-restart-g3`, stacked on G1 (`b2f009e69`), worktree
`/private/tmp/p20-restart-g3`. **OFFLINE / laptop-only**: no cluster contact of
any kind (no ssh, no boots, no relaunches/kills, no model requests, no
benchmarks, no GUI automation), nothing deployed, production untouched. Commit
sha: see the push record (`git ls-remote`).

Behind **`EXO_DSV41_SHUTDOWN_PARK=1` (default OFF)**. With the gate unset:
`shutdown_park_enabled()` is False, `bootstrap.entrypoint` installs its exact
pre-G3 `_graceful_sigterm` (SystemExit(0) → RDMA teardown), the runner keeps its
plain blocking `_work_queue.get()`, and no §G3 code runs. The 322 pre-G3 dsv41
tests stay green (see evidence).

## What changed (line numbers in the committed tree)

| File | What |
|---|---|
| `src/exo/worker/engines/mlx/dsv41/shutdown.py` (NEW) | Env + budget + signal + orchestrator. `SHUTDOWN_PARK_ENV` :72, the four knobs :74-82, defaults :84-92 (`DRAIN` 10, `PARK` 4, `MARGIN` 1, `GRACE` 15, poll 0.25). **`shutdown_park_enabled` :95**, **`ShutdownBudget` :115** (`fits` :122, **`park_deadline` :131**), **`shutdown_budget` :144**, **`ShutdownSignal` :165** (**`on_signal` :197** idempotent, raises the pre-G3 `SystemExit` when park-off or before the loop exists; **`install` :226**), **`ShutdownOutcome` :244**, **`run_shutdown_park` :258** (steps 3+4). |
| `src/exo/worker/engines/mlx/dsv41/session.py` | **`checkpoint_candidate` :1335** gains keyword `force=False` / `exclude=None` (force relaxes ONLY the time gates — idle-since-turn, global cooldown, per-conv back-off; the between-turns + `min_tokens` + dirty gates are NOT relaxed). **`abort_inflight` :1504** (§G3 step 3: drop every `_inflight` conversation from residency, never park it). **`force_checkpoint_all` :1536** (§G3 step 4: most-recent-first, one attempt each, `maybe_checkpoint` per unit, wall-clock deadline, stops on the first non-committed attempt / a `precheck` stand-down). |
| `src/exo/worker/engines/mlx/dsv41/engine.py` | `_shutting_down` field :374; **`_activate_next` guard :630** (refuses to start a queued turn while shutting down). **`begin_shutdown` :680** (step 1). **`shutdown_park` :689** (steps 3+4: release `_active` + `_draft_windows`, then `run_shutdown_park` with an `agree_on_tasks` `precheck`). |
| `src/exo/worker/engines/base.py` | `Engine.begin_shutdown` :59 and `Engine.shutdown_park` :68 — no-op defaults (the runner reaches them generically; a non-DSv4.1 engine opts out). |
| `src/exo/worker/runner/runner.py` | `_shutdown_state` / `_shutdown_budget` :164 (None unless bootstrap arms it). **`main` :339**: flips `runner_active`, checks `_shutdown_pending` at the top of the loop, runs `_run_shutdown_park` in the finally. **Helpers :382-453**: `_shutdown_pending` :384, `_shutdown_budget_or_default` :393, `_arm_shutdown_engine` :402, `_shutdown_drain_expired` :411, `_abort_inflight_for_shutdown` :418, `_run_shutdown_park` :431. **`_next_work_item` :455** uses a bounded wait (`poll_s`) ONLY while armed. **`handle_generation_tasks` drain check :785** (keep stepping the in-flight turn; abort at the drain deadline). |
| `src/exo/worker/runner/bootstrap.py` | Park-aware SIGTERM install :225-249 (default-off keeps `_graceful_sigterm` byte-for-byte); attaches `runner._shutdown_state = shutdown_signal` :362-365. |
| `src/exo/worker/engines/mlx/dsv41/fingerprint.py` | `dsv41.shutdown` added to `KV_NEUTRAL` :167 (control-plane; required by the existing `allowlist_problems()` guard). |
| `src/exo/worker/engines/mlx/dsv41/tests/test_dsv41_park_g3.py` (NEW) | T5 (23 test functions, 23 cases). |

## Shutdown sequence (plan §G3 steps 1-5, all on the runner/engine main thread)

There is NO asyncio loop in this process. `runner.main()` is a synchronous loop
on the *main* thread that blocks in `_work_queue.get()`, and that thread is also
the one that owns every MLX call. So the "minimal safe equivalent" of
`loop.add_signal_handler` is a `signal.signal` handler that runs *on the same
thread*, between bytecodes — no cross-thread marshalling is required.

1. **Stop admitting.** The runner's main loop checks `_shutdown_pending()` at the
   top and `break`s; `engine.begin_shutdown()` sets `_shutting_down` so
   `_activate_next` will not start a queued turn while the current one drains.
2. **Drain.** `handle_generation_tasks` keeps stepping the in-flight turn through
   its normal loop. If the turn completes inside `DRAIN_S`, it commits exactly as
   it would have (its task is Completed). The drain check itself issues no
   collective.
3. **Abort.** If the drain window expires, the runner fails the in-flight streams
   and the engine-thread `shutdown_park` → `abort_inflight` DROPS the in-flight
   conversation from residency. Nothing is parked mid-turn (absolute invariant:
   `checkpoint_candidate` skips `_inflight`); the last §G1 generation stays the
   on-disk restore point and only the aborted turn's delta is lost.
4. **Force checkpoint.** Remaining dirty *between-turns* conversations,
   most-recent-first, each attempted once, via the EXISTING §G1/§G4 path
   (`Dsv41Sessions.maybe_checkpoint` → `park.checkpoint_conversation` →
   `coord.run_checkpoint`). A `precheck` mirrors the §G1 idle tick's
   `agree_on_tasks` pairing so a rank does not enter a checkpoint collective while
   a peer is inside `step()`; the loop stops on the first non-committed attempt.
5. **Exit.** `main()` returns; `bootstrap.entrypoint`'s finally releases the MLX
   pool and the interpreter exits normally (JACCL/RDMA static destructors run).
   Unchanged from before — G3 does not regress this.

**Signal handling / idempotency / clean exit.** The handler records the request
(idempotent: the first signal arms the sequence, every later one is a logged
no-op). If the fast path is off, OR the signal lands before `runner.main()` has
flipped `runner_active` (during builder setup / load — no loop will observe the
flag), it raises `SystemExit(0)` — exactly the pre-G3 handler — so the
RDMA-teardown guarantee holds in every case. A duplicate during a parked shutdown
never re-enters the sequence.

## Budget math (audit A8)

The launcher's real budget is the graceful-shutdown script's `GRACE`
(`scripts/exo_graceful_shutdown.sh`, default 15 s): SIGTERM → wait → SIGKILL.
There is **no launchd / no ExitTimeOut** for exo (it runs under `screen -dmS`),
and the kill → rsync ordering is already correct (**R3 is struck**; nothing to
fix). The §G1-era plan's "0.7 × supervisor ExitTimeOut" was wrong. So:

```
DRAIN_S + PARK_S + EXIT_MARGIN_S <= GRACE
  10.0  +   4.0   +      1.0     <= 15.0      (defaults; each knob env-overridable)
park_deadline = min(park_start + PARK_S, signal_time + GRACE - EXIT_MARGIN_S)
```

The **hard wall-clock cap** (`signal_time + GRACE - EXIT_MARGIN_S`) is what
actually protects the SIGKILL boundary: even if the drain overran its own
nominal budget, the park phase can never push the process past `GRACE - MARGIN`,
so it always exits (and tears down RDMA) before the script escalates to `kill
-9`. `shutdown_budget()` logs a loud warning if the three nominal phases do not
fit GRACE, but still hard-caps. Env: `EXO_DSV41_SHUTDOWN_DRAIN_S`,
`EXO_DSV41_SHUTDOWN_PARK_S`, `EXO_DSV41_SHUTDOWN_EXIT_MARGIN_S`, and the budget
it is checked against `EXO_SHUTDOWN_GRACE_SECONDS`.

## Failure modes (all COLD-safe)

- **SIGKILL before completion** → the in-flight generation's `*.tmp` dir is left
  and the previous generation is intact; §G5 sweeps `*.tmp` at the next startup.
- **Half-written turn** → IMPOSSIBLE: step 3 never serializes in-flight state,
  and the writer's `_check_unchanged` refuses to write if a conversation moved.
- **Repeat / duplicate signals** → idempotent (one sequence).
- **Coord deadline exceeding GRACE** (peer slow / desynced) → SIGKILL → temp
  swept, previous generation kept (COLD-safe). No new collective was invented.

## Multi-rank (TP2)

Both ranks receive SIGTERM from the same deploy script and run the same
sequence. The final force checkpoint reuses the G4 ALL-OR-NOTHING coordinated
path VERBATIM, so it is rank-invariant and degrades to COLD on ANY disagreement
(candidate mismatch → the IDLE exchange is a no-go on both ranks; the loop then
stops). A rank whose `precheck` sees a peer task stands down. Nothing new is
coordinated. See OPEN-QUESTION 1.

## Test evidence (real runs, this worktree)

```
$ EXO_DASHBOARD_DIR=…/dashboard PYTHONPATH=…/src:…/mlx-lm …/python \
      -m pytest src/exo/worker/engines/mlx/dsv41/tests/ -q
322 passed in 31.27s     (baseline, before any change — given)
345 passed in 32.75s     (after: 322 + 23 new)

$ … -m pytest src/exo/worker/engines/mlx/dsv41/tests/test_dsv41_park_g3.py -q
23 passed in 1.13s       (T5 file; 23 test functions, 23 cases)

$ … -m pytest src/exo/worker/tests/unittests/test_runner -q   (runner.py/bootstrap.py touched)
174 passed in 16.71s

$ ruff check src/exo/worker/engines/mlx/dsv41/shutdown.py …/session.py …/engine.py \
             …/fingerprint.py src/exo/worker/engines/base.py \
             src/exo/worker/runner/runner.py src/exo/worker/runner/bootstrap.py \
             src/exo/worker/engines/mlx/dsv41/tests/test_dsv41_park_g3.py
All checks passed!
```

**T5** (23 cases):

- **Budget / gate.** Default-off; the four value spellings; defaults 10/4/1/15
  and `fits`; an overrun `not fits`; `park_deadline` = nominal when the drain
  finished early and = the hard GRACE cap when it overran.
- **Signal.** Two/three signals → one armed sequence, frozen `signal_time`; a
  signal before `runner_active` → `SystemExit`; park-off → always `SystemExit`.
- **Runner hooks.** Off-state no-ops; the drain predicate (open window → keep
  draining, `signal_time - 11 s` → abort); the idle wait is the plain blocking get
  when off and bounded by `poll_s` when armed; the abort path fails exactly the
  in-flight streams.
- **Engine wiring.** `shutdown_park` sets `_shutting_down`, releases `_active`,
  calls `abort_inflight` then `force_checkpoint_all` (with an `agree_on_tasks`
  precheck); `begin_shutdown` → `_activate_next` does not start the queued task.
- **Sequence over the real cache.** Mid-turn **PREFILL** and mid-turn **DECODE**
  → the in-flight conversation is aborted + dropped, NO generation with
  `n_ctx != offset` is written, the last §G1 generation is untouched and restores
  byte-identically to the pre-turn state. One in-flight conversation is dropped
  while a different between-turns dirty conversation still gets its force
  checkpoint and restores byte-identically.
- **Deadline.** `deadline` in the past → no attempt, `deadline_hit`; `PARK_S`
  allows exactly one attempt then stops; most-recent-first ordering; one attempt
  per conversation; an in-flight conversation is skipped by `force=True`.
- **Flag-off.** Gate false; runner hooks false; `_run_shutdown_park` does not call
  the engine.
- **Two ranks (G4 lockstep).** Both ranks force-checkpoint the same conversation →
  the SAME agreed generation on both roots, equal exchange counts; a candidate
  disagreement (same count, different streams) → the IDLE exchange is a no-go on
  both, one exchange each, nothing written (COLD-safe).

## Deviations from the plan / brief

1. **The raise from the signal handler is limited to the no-loop case.** The brief
   suggested the handler raise to unwind out of the blocking `get()`. Raising
   mid-turn would unwind *through the generator frame* (`next(active.
   output_generator)`), killing the turn and defeating the drain. So the handler
   only raises `SystemExit` when no loop will observe the flag (park-off, or
   before `main()` starts); once the loop is live it sets the flag and the
   runner's BOUNDED idle wait observes it. This keeps the drain intact and needs
   no thread.
2. **Drain lives in the runner loop, not the orchestrator.** Only the runner can
   advance a turn, so `handle_generation_tasks` drains (step 2) and
   `run_shutdown_park` owns abort + force-checkpoint (steps 3+4).
3. **The force loop stops on the first non-committed attempt.** A no-go means the
   ranks are not in lockstep this shutdown; continuing would issue another
   collective at a possibly divergent point. Strictly safer, COLD-safe.
4. **`PARK_S` is a nominal per-phase budget; the GRACE cap is the hard one.**

## OPEN-QUESTIONs (cannot be settled offline)

1. **Multi-rank shutdown simultaneity (the requested OQ).** Both ranks get
   SIGTERM together, but their drains can finish at different times, so they may
   reach the force-checkpoint `precheck`/IDLE exchange at different program
   points. The loop is designed to stand down symmetrically (a `precheck`
   `agree_on_tasks` pairs with a peer in `step()`; the IDLE exchange is a no-go on
   any disagreement; the loop stops on the first non-committed attempt), and every
   outcome is COLD-safe. **Residual risk:** if the two ranks hold a DIFFERENT
   *number* of eligible dirty between-turns candidates, the loop could issue a
   different number of exchanges and desync. Mitigation is the G4 all-or-nothing
   decision plus the GRACE hard cap (a desync past GRACE is absorbed by SIGKILL →
   temp swept). The exact simultaneity race needs live TP2 validation; it was NOT
   attempted offline.
2. **Launcher / GRACE change is DESIGN-ONLY; deferred to deploy time.** G3 needs
   NO launcher change to be correct (it never overruns GRACE). It is only worth
   RAISING `GRACE` (so a large dirty set has more than ~4 s of park budget) or
   wiring `EXO_SHUTDOWN_GRACE_SECONDS` into `start_cluster.sh`'s env allowlist so
   the runner and the script agree. Any such edit MUST land on the **production
   branch** at deploy time (per A8: `origin/main`'s `start_cluster.sh` lacks
   `DSV41_DENSE`; the production branch carries it). **Nothing was changed here**
   — no production or launcher file was touched in this worktree.
3. **Park cost vs budget.** Audit A9 measured ~471 MiB per 92 K-row session and
   found no timing instrumentation; whether a typical dirty set parks inside the
   ~4 s `PARK_S` on NVMe is unmeasured. If not, the deadline truncates the set
   (most-recent-first), which is acceptable and COLD-safe. Needs an instrumented
   run.
4. **Donation trap.** The coordination touches only Python ints, and `abort_inflight`
   drops the `_active` request and clears `_draft_windows` before any checkpoint, so
   no MLX array is held across the next forward. The `park.checkpoint_conversation`
   unit writer already drops every reference per unit (G1 T9). Not re-measured here.

**No-engine-contact affirmation:** no ssh, network, cluster, boot, relaunch, kill,
model request, benchmark, or exo-server contact of any kind was made; the optional
instrumented park/GRACE measurements in OQ-2/3 were NOT run. All work is
laptop-local in `/private/tmp/p20-restart-g3`; nothing was deployed and no
production or launcher file was modified.
