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
| `src/exo/worker/engines/mlx/dsv41/shutdown.py` (NEW) | Env + budget + signal + orchestrator. `SHUTDOWN_PARK_ENV` :72, the four knobs :74-82, defaults :84-92 (`DRAIN` 10, `PARK` 4, `MARGIN` 1, `GRACE` 15, poll 0.25). **`shutdown_park_enabled` :95**, **`ShutdownBudget` :115** (`fits` :122, **`park_deadline` :131**), **`shutdown_budget` :144**, **`ShutdownSignal` :165** (**`on_signal` :197** idempotent, raises the pre-G3 `SystemExit` when park-off or before the loop exists; **`install` :226**), **`ShutdownOutcome` :244**, **`run_shutdown_park` :258** (steps 3+4; builds a deadline-aware `request_pending` gate `now() >= deadline` unless the caller supplies one, so the EXISTING §G1 per-unit `CKPT_POLL` bounds a checkpoint's overshoot to ONE unit — audit MED-A). |
| `src/exo/worker/engines/mlx/dsv41/session.py` | **`checkpoint_candidate` :1335** gains keyword `force=False` / `exclude=None` (force relaxes ONLY the time gates — idle-since-turn, global cooldown, per-conv back-off; the between-turns + `min_tokens` + dirty gates are NOT relaxed). **`abort_inflight` :1504** (§G3 step 3: drop every `_inflight` conversation from residency, never park it). **`force_checkpoint_all` :1536** (§G3 step 4: most-recent-first, one attempt each, `maybe_checkpoint` per unit, wall-clock deadline, stops on the first non-committed attempt / a `precheck` stand-down). |
| `src/exo/worker/engines/mlx/dsv41/engine.py` | `_shutting_down` field :374; **`_activate_next` guard :630** (refuses to start a queued turn while shutting down). **`begin_shutdown` :680** (step 1). **`shutdown_park` :689** (steps 3+4: release `_active` + `_draft_windows`, then `run_shutdown_park` with an `agree_on_tasks` `precheck`). |
| `src/exo/worker/engines/base.py` | `Engine.begin_shutdown` :59 and `Engine.shutdown_park` :68 — no-op defaults (the runner reaches them generically; a non-DSv4.1 engine opts out). |
| `src/exo/worker/runner/runner.py` | `_shutdown_state` / `_shutdown_budget` :164 (None unless bootstrap arms it). **`main` :339**: flips `runner_active`, checks `_shutdown_pending` at the top of the loop, runs `_run_shutdown_park` in the finally. **Helpers :382-453**: `_shutdown_pending` :384, `_shutdown_budget_or_default` :393, `_arm_shutdown_engine` :402, `_shutdown_drain_expired` :411, `_abort_inflight_for_shutdown` :418, `_run_shutdown_park` :431. **`_next_work_item` :455** uses a bounded wait (`poll_s`) ONLY while armed. **`handle_generation_tasks` drain check :785** (keep stepping the in-flight turn; abort at the drain deadline). |
| `src/exo/worker/runner/bootstrap.py` | Park-aware SIGTERM install :225-249 (default-off keeps `_graceful_sigterm` byte-for-byte); attaches `runner._shutdown_state = shutdown_signal` :362-365. |
| `src/exo/worker/engines/mlx/dsv41/fingerprint.py` | `dsv41.shutdown` added to `KV_NEUTRAL` :167 (control-plane; required by the existing `allowlist_problems()` guard). |
| `src/exo/worker/engines/mlx/dsv41/tests/test_dsv41_park_g3.py` (NEW) | T5 (24 test functions, 24 cases). |

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
   conversation from residency. Nothing is parked mid-turn (the between-turns
   invariant is enforced by `checkpoint_candidate` skipping `_inflight`; see
   *Failure modes → Half-written turn* for its one inherited caveat); the last §G1
   generation stays the on-disk restore point and only the aborted turn's delta is
   lost.
4. **Force checkpoint.** Remaining dirty *between-turns* conversations,
   most-recent-first, each attempted once, via the EXISTING §G1/§G4 path
   (`Dsv41Sessions.maybe_checkpoint` → `park.checkpoint_conversation` →
   `coord.run_checkpoint`). A `precheck` mirrors the §G1 idle tick's
   `agree_on_tasks` pairing so a rank does not enter a checkpoint collective while
   a peer is inside `step()`; the loop stops on the first non-committed attempt.
   The park **deadline** is threaded into the checkpoint as its
   `request_pending` gate (`now() >= deadline`; at shutdown there are no new
   requests), so the EXISTING §G1 per-unit `CKPT_POLL` aborts an *in-progress*
   checkpoint within ONE unit of the deadline (audit MED-A) — the deadline is no
   longer only checked *between* attempts.
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

The **hard wall-clock cap** (`signal_time + GRACE - EXIT_MARGIN_S`) protects the
SIGKILL boundary, and it is now enforced at per-unit granularity: the deadline is
the checkpoint's `request_pending` gate, so the §G1 `CKPT_POLL` aborts the
in-progress write within **one unit** of the deadline on every rank (audit
MED-A). **Residual:** the deadline is still a *soft* Python-level gate, not a
preemptive timer. The overshoot is bounded to **one unit's write** (one layer's
blobs + fsync), not zero: a single unit slower than `EXIT_MARGIN_S` could still
push past `GRACE - MARGIN`, and only at that point does the launcher's SIGKILL
land. With the default 1 s margin and a unit that is a fraction of a second this
holds with a wide gap, but it is not a hard interrupt — the process does NOT
"always exit before `kill -9`". A unit larger than the margin is a *design*
limit; a launcher `GRACE` raise was NOT made here (`OPEN-QUESTION 2`).
`shutdown_budget()` logs a loud warning if the three nominal phases do not fit
GRACE, but still hard-caps. Env: `EXO_DSV41_SHUTDOWN_DRAIN_S`,
`EXO_DSV41_SHUTDOWN_PARK_S`, `EXO_DSV41_SHUTDOWN_EXIT_MARGIN_S`, and the budget
it is checked against `EXO_SHUTDOWN_GRACE_SECONDS`.

## Failure modes (all COLD-safe)

- **SIGKILL before completion** → the in-flight generation's `*.tmp` dir is left
  and the previous generation is intact; §G5 sweeps `*.tmp` at the next startup.
- **Deadline overshoot** → the park deadline is a per-unit `request_pending` gate,
  so a checkpoint is aborted within ONE unit of it (temp dir removed, previous
  generation kept). It is a soft gate, not a preemptive timer: the residual is one
  unit's write, not zero (see *Budget math*).
- **Half-written turn** → not serialized: step 3 never parks in-flight state,
  `checkpoint_candidate` skips `conv._inflight`, and the writer's
  `_check_unchanged` refuses to write if a conversation moved. **The
  between-turns-only invariant is enforced SOLELY by `conv._inflight`** (set only
  at the very end of `Conversation.prefill`, `session.py:780`). A *failed* prefill
  (an error raised from a mid-stream `append_turn`) re-raises without rewinding
  (`session_cache.py:619-633`; `_generate` only cancels on `PrefillCancelled`), so
  it can leave a partial cache with `_inflight == False` that a later park WOULD
  serialize. This is **inherited from G1, not introduced by G3**, is benign in
  *bytes* (the parked `n_ctx == offset` is a valid prefix of the same stream, so a
  restore reproduces exactly the live bytes — no torn/mixed state), and is **not
  reachable from the runner**: prefill is one synchronous call inside `step()`, so
  no park (idle or shutdown) can observe a conversation between the start of
  prefill and its final `_inflight = True`.
- **Repeat / duplicate signals** → idempotent (one sequence).
- **Coord deadline exceeding GRACE** (peer slow / desynced) → SIGKILL → temp
  swept, previous generation kept (COLD-safe). No new collective was invented.

## Multi-rank (TP2) — BEST-EFFORT / COLD, needs live validation

Both ranks receive SIGTERM from the same deploy script and run the same
sequence, and the final force checkpoint is **BEST-EFFORT / COLD-safe, NOT
rank-invariant**: it MUST be validated on a live TP2 pair. The G4 protocol
*inside* one checkpoint (IDLE agreement, agreed N, per-unit `CKPT_POLL`,
ack-gated prune) is rank-invariant — a candidate disagreement is a no-go on both
ranks. But the force LOOP around it is not: every per-iteration break
(`now() >= deadline`, `precheck()` False, candidate None, attempt
non-committed) is a RANK-LOCAL decision taken with no agreeing exchange, so the
two ranks can issue a different NUMBER of collectives. Compounding this, G3's
own step-1 `begin_shutdown` flips `_shutting_down` at a slightly different time
on each rank, so `_activate_next` can pop a queued task on one rank and leave it
in `_agreement.queue` on the other — which makes even `precheck()`
(`not agree_on_tasks() and not self._agreement.queue`) asymmetric. On a real
coord group that is a deterministic unmatched-collective hang, not merely a
timing race. See OPEN-QUESTION 1.

## Test evidence (real runs, this worktree)

```
$ EXO_DASHBOARD_DIR=…/dashboard PYTHONPATH=…/src:…/mlx-lm …/python \
      -m pytest src/exo/worker/engines/mlx/dsv41/tests/ -q
322 passed in 31.27s     (baseline, before any change — given)
346 passed in 26.19s     (after: 322 + 24 new)

$ … -m pytest src/exo/worker/engines/mlx/dsv41/tests/test_dsv41_park_g3.py -q
24 passed in 0.96s       (T5 file; 24 test functions, 24 cases)

$ … -m pytest src/exo/worker/tests/unittests/test_runner -q   (runner.py/bootstrap.py touched)
174 passed in 16.26s

$ ruff check src/exo/worker/engines/mlx/dsv41/shutdown.py …/session.py …/engine.py \
             …/fingerprint.py src/exo/worker/engines/base.py \
             src/exo/worker/runner/runner.py src/exo/worker/runner/bootstrap.py \
             src/exo/worker/engines/mlx/dsv41/tests/test_dsv41_park_g3.py
All checks passed!
```

**T5** (24 cases):

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
  per conversation; **the park deadline aborts an in-progress checkpoint after
  exactly ONE unit** (per-unit `CKPT_POLL`, MED-A regression — a whole-conversation
  write does not overshoot); an in-flight conversation is skipped by `force=True`
  (now exercised with a COMMITTED prior turn present, so the `_inflight` gate
  alone rejects it — LOW-D regression).
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

1. **Multi-rank shutdown simultaneity (the requested OQ) — a DETERMINISTIC
   rank-local-break risk, NOT merely a race.** Both ranks get SIGTERM together,
   but the force-checkpoint LOOP is a sequence of RANK-LOCAL branches with no
   agreeing exchange between them: `now() >= deadline` (a per-rank monotonic
   clock), `precheck()` False, candidate None (each rank's own residency/LRU),
   and "attempt not committed". Any one of these can be taken by one rank and not
   its peer, so the two ranks can issue a different NUMBER of collectives and the
   peer blocks in an unmatched `all_gather`. It is deterministic in the code
   (probe I exercised candidate-count 50 vs 0, standdown 50 vs 0, deadline 50 vs 0
   on the simulated pair), not a rare timing accident. It is compounded by G3's
   own step-1 `begin_shutdown`: the `_shutting_down` flip lands at slightly
   different times on the two ranks, so `_activate_next` can pop the queued task
   on one rank and leave it in `_agreement.queue` on the other — making
   `precheck()` itself asymmetric. **The final shutdown checkpoint is therefore
   BEST-EFFORT / COLD and must be validated on a live TP2 pair; the loop does NOT
   "stand down symmetrically" and is NOT rank-invariant** (the G4 protocol inside
   a single checkpoint is; the loop around it is not). On a real coord group a
   desync hangs until the launcher's SIGKILL, which also skips the clean RDMA
   teardown (the QP-leak / "No device connected" wedge `bootstrap.py:208-217`
   exists to prevent). **Robust fix (DESIGN-ONLY, deferred to deploy):** put a
   *symmetric agreeing gate* before any shutdown collective — an exchange in
   which both ranks commit to "we are at the shutdown point" and to the SAME next
   action (proceed / stop), so a rank can never act on a local-only break. Per the
   original brief this multi-rank simultaneity question is an explicit
   OPEN-QUESTION and was NOT attempted offline; only the DOCUMENTATION is
   corrected here.
2. **Launcher / GRACE change is DESIGN-ONLY; deferred to deploy time.** G3 needs
   no launcher change to be correct at the default budget: the park phase is
   bounded to ONE unit past the deadline (audit MED-A), inside `GRACE - MARGIN`
   with the default 1 s margin. It is only worth RAISING `GRACE` (so a large dirty
   set has more than ~4 s of park budget) or
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
