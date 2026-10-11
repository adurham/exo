# DSv4.1 restart survival — G1 build note (2026-10-10)

Gap **G1** of `docs/dsv41-restart-survival-plan-2026-10-10.md` (after-turn idle
checkpoint), as corrected by the Phase-0 audit. Branch `p20/dsv41-restart-g1`,
stacked on G4 (`360fadbf6`), worktree `/private/tmp/p20-restart-g1`. Behind
**`EXO_DSV41_CKPT=1` (default OFF)**. With the flag unset, the runner keeps its
plain blocking `work_queue.get()`, `Engine.idle_tick_seconds()` returns `None`,
and no G1 code runs. The one change that is live regardless of the flag: the
eviction park now goes through the unit writer. It writes the same files and
manifest (checked by `test_park_conversation_format_unchanged_by_split`).

## Design finding that changed the hook point

The task brief assumed the idle hook was `Dsv41Engine.step()` with `_active is
None`. That assumption is wrong. `Runner.handle_generation_tasks` calls `step()`
only `while self.active_tasks` (`runner.py:649` pre-change). Between requests the
runner blocks in `Runner.main` on `self._work_queue.get()` (`runner.py:336`
pre-change), so `step()` is never called while idle. The idle tick therefore
lives in the runner's idle wait. It still runs on the same runner/engine thread
that owns every MLX call, never a separate thread.

## Files / functions (line numbers in the committed tree)

| File | What |
|---|---|
| `src/exo/worker/runner/runner.py` | `main` :336 now calls `_next_work_item` :360. With the runner Ready and the engine opted in, the wait becomes `get(timeout=tick)`. On timeout it calls `engine.idle_tick(request_pending=lambda: not work_queue.empty())` on the runner thread. Otherwise it does the old blocking `get()`. |
| `src/exo/worker/engines/base.py` | `Engine.idle_tick_seconds` :42 (default `None`) and `Engine.idle_tick` :48 (default no-op). |
| `dsv41/engine.py` | `idle_tick_seconds` :464 and **`idle_tick` :468** (local gate → `agree_on_tasks` → `Dsv41Sessions.maybe_checkpoint`, then `mx.clear_cache()`). `_last_step_monotonic` :370 is stamped at `step()` entry :530. |
| `dsv41/agreement.py` | `agree_on_tasks` :48 now **returns** the `mx_any` result (rank-invariant). The collectives are unchanged and `step()` ignores the return value. |
| `dsv41/session.py` | Env + helpers: `CKPT_ENV` :95, defaults :98-108 (`MIN_DELTA` 2048, `IDLE_S` 20, `TICK_S` 2, `RETRY_S` 120), `ckpt_enabled` :111, `ckpt_tick_seconds` :130. Per-conversation `ckpt_offset` / `last_turn_end_ts` :636, `dirty_tokens` :641. `last_turn_end_ts` is set at the turn-commit point in `Conversation.finish(checkpoint=True)` :864, where draft `n_ctx == offset` holds. A restored conversation is marked clean :1185. `_evict` skips the rewrite of a clean conversation :1228 (G4 exchanges still run). **`Dsv41Sessions.checkpoint_candidate` :1296** (pure local gate) and **`Dsv41Sessions.maybe_checkpoint` :1343** (new manager-level entry point). `Conversation.maybe_checkpoint` (in-memory DSpark ladder, :814) is untouched. |
| `dsv41/park.py` | **`_GenerationWriter` :411** splits serialization from eviction: `begin` :460, `write_next` :550 (one unit), `commit` :560, `abort` :581, `_check_unchanged` :591, `_layer_unit` :602 (plus engram / draft / snapshot / draft-snap / anchor units). `park_conversation` :692 is now a one-shot driver over it (same format). **`ParkedStore.open_generation` :1525** is the keep-resident path: headroom reserve plus begin, no close. **`checkpoint_conversation` :1701** is the G1 driver over `run_checkpoint`. `_local_coordinator` :1697 is a world-of-one coordinator, so single-rank runs the same sequence. |
| `dsv41/coord.py` | `MAGIC_IDLE` / `MAGIC_CKPT_POLL` :102-103, `IDLE_WIDTH` / `POLL_WIDTH` :109-110. `IdleDecision` :575, **`decide_idle` :589**, **`decide_poll` :615** (pure). `CheckpointOutcome` :639. **`run_checkpoint` :666**: IDLE → PARK_GEN (G4 `agree_generation`) → per-unit CKPT_POLL → commit → PARK_ACK (G4 `prune_floor`, ack-gated prune). |
| `dsv41/tests/test_dsv41_park_g1.py` | NEW: T7, T9, cross-rank and gate tests (25 functions, 28 collected cases). |

`fingerprint.py` is unchanged. `coord`, `park`, `session` and `engine` are
already classified, and the `allowlist_problems()` guard still passes in the
suite.

## Trigger design (and why)

1. **Local gate, no collectives** (`checkpoint_candidate` plus the `idle_tick`
   preconditions). The tick proceeds only if all of these hold:
   - no `_active` request, empty agreed queue, empty `maybe_queue`;
   - this rank last entered `step()` at least `CKPT_IDLE_S` ago;
   - some resident conversation is between turns, has `offset >=
     PARK_MIN_TOKENS`, is dirty (`dirty_tokens >= CKPT_MIN_DELTA`) or never
     checkpointed, has been idle `>= CKPT_IDLE_S` since its turn committed, and
     is not in an agreed back-off.

   Most-recent turn first, ONE conversation per tick. The common idle tick
   costs nothing and issues no collective. Every gate input is shared state:
   conversation state (same request stream, same deterministic LRU), env, and
   the monotonic clock. `PARK_MIN_TOKENS` is read from env, not from the lazily
   built store, because a store build can fail on one rank only.
2. **`agree_on_tasks()`**: exactly the collective prefix `step()` issues with no
   active request.
   - If a peer is already in `step()` with a task, this rank's `mx_any` pairs
     with the peer's `mx_any`. Both then run `mx_all_gather_tasks`; this rank
     contributes no task, the intersection is empty, nothing is agreed, and the
     tick stands down without issuing any checkpoint collective. The runner then
     fetches the task and both ranks meet in `step()`. Tested in
     `test_idle_tick_pairs_with_peer_in_step`.
   - If `mx_any` is False, no rank holds a task. The agreed queues are
     identical and empty, so the peer cannot be in `step()` with work and must
     be at the same point of its own idle tick.
3. **IDLE exchange** `[magic, ok, key, offset]`. Go only if every rank says ok
   for the SAME (key, offset). ok means: no local request pending, usable
   store, park-safe conversation. Anything else is an agreed no-go on every
   rank (followed by a short symmetric cooldown).
4. **PARK_GEN** (G4, unchanged rule `N = 1 + max`), then `open_generation`.
5. **CKPT_POLL before every unit and once after the last unit**:
   `[magic, pending, ok, more]`.
   - Any `ok=0` or malformed row: every rank aborts.
   - Else any `pending=1`: every rank aborts and serves the request.
   - Else nobody has more units: commit.
   - Else each rank with units left writes ONE unit.

   The number of exchanges is a pure function of the gathered rows, so it is
   identical on every rank (asserted via `ls.calls` in the 2-rank tests).
6. **Commit**: G5 manifest commit, then **PARK_ACK** (G4). Prune only on a
   unanimous ack. Bookkeeping follows the AGREED outcome only:
   - unanimous ack: `ckpt_offset = offset` (clean);
   - agreed failure: per-conversation back-off (`RETRY_S`);
   - abort because a request is pending: no back-off.

**Cancellation.** A request (or a peer's request) waits for at most the one
unit in flight. A unit is one layer (`win_kv` + live `comp_kv` / `index_k` rows
+ carry) or one small tail item. On abort the temp dir is `rmtree`'d and the
previous generation is untouched; nothing partial is ever committed, because
the manifest is written only by `commit`.

**Donation trap.** Each unit looks its arrays up on the live cache when it
runs. It copies them to the host on the CPU stream, writes, fsyncs, and drops
every reference before returning. The writer holds only ints, the manifest
dict, and unit closures over indices/positions. `_check_unchanged` refuses to
write if the conversation moved, which cannot happen while idle but is checked
anyway. Nothing is referenced across a unit boundary, and the whole checkpoint
finishes or aborts before the next forward pass.

## Test evidence (real runs, this worktree)

The baseline needed `git submodule update --init mlx-lm`: the worktree's
`mlx-lm` was empty, and the suite's own landmine guard refuses to run against
the venv's `mlx_lm`. I pinned it to the same commit as G4 (`6cc9c1e8`) and
detached it from the reference repo with repack + removed alternates. No source
change.

```
$ EXO_DASHBOARD_DIR=…/dashboard PYTHONPATH=…/src:…/mlx-lm …/python -m pytest src/exo/worker/engines/mlx/dsv41/tests/ -q
290 passed in 24.03s     (baseline, before any change)
318 passed in 25.75s     (after: 290 + 28 new)

$ … -m pytest src/exo/worker/engines/mlx/dsv41/tests/test_dsv41_park_g1.py -q
28 passed in 2.38s
$ … -k t7      ->  7 passed      $ … -k t9      ->  2 passed

$ … -m pytest src/exo/worker/tests/unittests/test_runner -q      (runner.py touched)
174 passed in 16.97s

$ ruff check src/exo/worker/engines/mlx/dsv41/ src/exo/worker/engines/mlx/utils_mlx.py \
             src/exo/worker/engines/base.py src/exo/worker/runner/runner.py
All checks passed!
```

**T7** (7 cases):

- `test_t7_request_during_checkpoint_aborts_within_one_unit`: the request lands
  during unit 1. Result: exactly 1 unit written and none after; no new
  committed dir; temp dir gone; previous generation byte-identical. The served
  turn's tokens and cache bytes are identical to a never-checkpointed control,
  and the next idle period commits.
- `test_t7_abort_point_sweep[0-3]`: abort after k units means exactly k units
  written and no generation. With k=0, the IDLE exchange itself says no-go.
- `test_t7_pause_bounded_by_one_unit_write`: units slowed to 50 ms; the time
  from request to return is under one unit.
- `test_t7_two_ranks_request_on_one_rank_aborts_both`: the request is seen only
  on rank 1. Both ranks abort at the same poll, with no generation and no temp
  dir on either rank.

**T9** (2 cases). I chose the synthetic harness over the optional studio2
microbench; reasoning below.

- `test_t9_donation_survives_checkpoint_commit_and_abort`: a real fork
  `ModelCache` (2 layers, 131072-row latents). Each decode step does in-place
  slice writes exactly as `attention.py:200` / `indexer.py:503` /
  `write_window` do. The data pointer of every cache buffer is unchanged step
  to step, both after a COMMITTED and after an ABORTED checkpoint. The negative
  control holds one slice reference, and the next step's buffer address
  changes. That proves the detector sees a non-donated whole-buffer copy.
- `test_t9_decode_step_time_unchanged_after_checkpoint`: decode-step medians
  over 5 alternating blocks of 80 steps. The assertion is after ≤ before × 1.02
  + 2 µs (min-of-blocks). Eight observed runs: ratio 1.0129, 0.9963, 0.9969,
  1.0117, 0.9974, 1.0002, 1.0061, 0.9984 (all within 2%). The leaked-reference control is 2.9–3.3× slower on
  the same harness, so the harness can see the regression T9 is meant to
  catch.

**Why synthetic instead of studio2.** Donation is a refcount property of MLX
arrays (an in-place `slice_update` reuses the buffer iff its refcount is 1). It
does not depend on the device, and the address-identity check observes it
directly rather than through noisy throughput. The real fork cache and the real
write ops are used. An offline studio2 run would add Metal-specific timing but
needs node contact and the idle-guard/canary procedure, and the address test
is the stronger oracle. Residual: see OQ-6.

Other G1 tests:

- Pure `decide_idle` / `decide_poll`.
- Keep-resident: the same object, not closed, cache bytes unchanged, restores
  byte-identically.
- Split preserves format (file set + bytes + manifest minus `created`).
- Gate: min tokens / delta / idle / most-recent-first / mid-turn / flag off.
- One session per tick.
- ENOSPC on a blob: previous generation intact, no temp dir, back-off, the
  conversation still serves.
- Disabled store: agreed no-go.
- Clean eviction is not rewritten and restores.
- Two ranks:
  - same generation N on both, equal exchange counts;
  - write failure on one rank: both abort, gen 1 survives on both, no gen 2,
    symmetric back-off;
  - candidate mismatch: 1 exchange and no-go on both;
  - unusable store on one rank: no-go;
  - fresh sessions restore the checkpoint generation (G4 agreed).
- Engine `idle_tick` default-off + gating; runner idle-wait unchanged when off,
  one tick on the runner thread when on.

## Hardening (independent adversarial audit)

An independent adversarial audit of the committed G1 change found three real
defects and one pre-default-on gate concern. Each defect was reproduced before
the fix with a standalone script (`repro_headroom.py`, `repro_dirty.py`,
`repro_keyed.py`), fixed, and guarded by a mutation-checked regression test (each
new test fails against the pre-fix behaviour, verified in-process). Suite:
**318 → 322 passed** (+4). `ruff check` clean on every touched path.

### HIGH-1 — an aborted checkpoint could delete the conversation's only generation

*Finding.* `_reserve_headroom(protect_prefix=key_hash)` protected only dirs whose
NAME prefix equals `prefix_hash(conv.tokens)` — the CURRENT, grown stream. A
conversation that grew has its previous generation under an EARLIER (shorter)
stream's hash, so it was NOT protected, and headroom eviction deleted it BEFORE
the new generation existed (`open_generation`, `write_generation`, `park`). If
the write was then aborted, zero generations remained. Repro (pre-fix):
`prev ckpt dirs ['…gen1…']` → `aborted True dirs after abort []`.

*Fix.* `ParkedStore._reserve_headroom` now also takes a `protect` predicate.
`ParkedStore._protect_own(conv, key)` matches EVERY generation of the
conversation, using the identity rules `find` / `local_restore_candidates` already
use: a client `key` match, or the stored ids being a strict PREFIX (ancestor) of
the live stream. All three call sites pass it. Under byte-budget pressure the
previous generation now survives an abort (or an ENOSPC write failure).

*Regression test.* `test_audit_h1_budget_abort_keeps_previous_generation`. Mutation
(protect-predicate forced to always-False) → the test fails at
`assert set(_gens(tmp_path)) == {1}` (dirs after the abort are empty).

### MED-1 — offset-only dirty tracking missed a rewind+refeed to the same offset

*Finding.* `dirty_tokens = offset - max(ckpt_offset, 0)`; a turn that rewinds to a
lower row and refeeds back to the SAME offset on a different branch gave
`dirty_tokens == 0`, so `checkpoint_candidate()` returned `None` and eviction's
clean-skip never rewrote the generation: the on-disk `ids.bin` kept the
pre-rewind branch. Repro (pre-fix): `candidate None`; dirs unchanged after a
same-offset token change.

*Fix.* `Conversation.ckpt_hash` records the `prefix_hash` of the checkpointed
token stream at commit (`maybe_checkpoint`) and at restore (`_restore_parked`).
`Conversation.stream_changed` compares the live stream's hash against it
(`False` when no hash was ever recorded — a manual/legacy state — so the
offset-only behaviour is a strict fallback). The idle gate treats a stream change
as dirty regardless of the offset delta, and eviction's `clean` check now also
requires `not stream_changed`.

*Regression test.*
`test_audit_med1_same_offset_branch_change_is_dirty_and_rewritten`. Mutation
(`stream_changed` forced always-False) → the candidate is `None` and the test
fails.

### MED-2 — a keyed restore picked an arbitrary generation; a growing conversation never pruned

*Finding.* Key-mode `find` broke ties by generation alone and
`local_restore_candidates` set `pref = 0`. Every turn of a grown stream produced a
fresh `prefix_hash` with generation 1, so the restore picked the FIRST dir by name
(offset 152 when the newest was 177), and `prune_acked` / `_KEEP_GENERATIONS`
(keyed on `prefix_hash`) never fired — one dir accumulated per turn up to
`max_count`. Repro (pre-fix): 5 checkpoints → 5 dirs ALL named `…gen1…`; the keyed
restore returned 152, not 177.

*Fix (both parts).*
- **Newest wins.** Key-mode `find` and `local_restore_candidates` now order by the
  stored stream LENGTH (`token_count`, then `created`, then generation).
  `Candidate.pref` in key mode is now the stream length — its long-standing
  documented meaning (G4 OQ-6). `find` also uses `created` to break a same-length
  tie (a rewind+refeed to the same offset), so a single-round restart
  deterministically restores the newest branch.
- **Bounded accumulation (contained fix, NOT merely documented).**
  `ParkedStore.retain_keyed(key, keep=_KEEP_GENERATIONS, keep_dir=…)` groups a
  conversation's generations by the manifest's stable client `key` and keeps the
  newest `_KEEP_GENERATIONS` (2) by `(created, generation, name)`. It runs at the
  ack-gated prune point of both the §G1 checkpoint (`checkpoint_conversation`) and
  the §G4 park (`coordinated_park`), and after `_prune_generations` in the local
  `park`. Measured: a keyed conversation's dirs are now bounded to 2 (5
  checkpoints → 2 dirs).

*Regression tests.* `test_audit_med2_keyed_restore_newest_and_bounded` (single
rank; also asserts the dir count ≤ `_KEEP_GENERATIONS`) and
`test_audit_med2_two_ranks_keyed_restore_newest` (G4 agreed restore returns the
newest on BOTH ranks). Mutation (pre-fix key ordering + no `retain_keyed`) → both
fail.

### MED-3 — idle-tick `mx_any` mispairing (pre-default-on gate)

Left unchanged, exactly as before: the hot path is untouched. This is the same
pre-existing invariant, with the same mitigations already recorded in
OPEN-QUESTION 2 below (an idle collective only after `CKPT_IDLE_S` since this
rank's last `step()`, plus strict magic validation of the IDLE / POLL rows; a
magic tag on `agree_on_tasks`' `mx_any` is recommended before default-on).

### Residual OPEN-QUESTION (from MED-2, cross-rank)

`decide_restore` orders the wire candidates by `(pref, gen, key)`, and `pref` is
the stored stream length. Two generations of the SAME keyed conversation that
differ only by branch but share a length (a rewind+refeed to the same offset,
considered on two ranks) therefore tie on `(pref, gen)` and are broken by the wire
key (`key_int(prefix_hash)`) — arbitrary. The single-rank `find` breaks this tie
deterministically by `created`; the wire cannot carry `created` safely (per-rank
`created` timestamps differ by milliseconds, so a rank-invariant `min pref`
packing is not sound). *Recommendation:* give a keyed conversation's generations a
stable prefix (so generation numbers order them) before default-on, or carry
recency on the wire explicitly. Probability is low (a same-length branch rewrite
on both ranks) and the outcome is COLD-safe per G4 OQ-4.

## Deliberately NOT done

- **G3** park-on-shutdown (signal handling, drain, forced checkpoint). The
  `maybe_checkpoint` entry point is shaped so G3 can call it.
- **G6** rewind-to-LCP (deprioritised by audit A4).
- **G7** incremental / append-only checkpoints. Each checkpoint rewrites the
  full generation, roughly 1–3 GiB per qualifying turn at ≥2048-token deltas,
  per the plan's estimate.
- No live, TP2, or node run of any kind (see the affirmation).

## Deviations from the plan / brief

1. The idle hook is in the runner's idle wait, not `step()` (finding above).
   `step()` itself is unchanged apart from one timestamp.
2. "PARK_MIN_TOKENS" uses the store's own `min_tokens` policy
   (`EXO_DSV41_PARK_MIN_TOKENS`, default 4096). There is no separate knob.
3. A unit is a layer plus small tail units (engram, draft, each snapshot, each
   anchor). The plan's "per layer" holds for the bulk of the bytes.
4. An extra `CKPT_IDLE_S`-since-last-`step()` engine gate, so a rank that just
   served a request never enters an idle collective while its peer could still
   be finishing the same request.
5. Clean-eviction skip: evicting an agreed-clean conversation does not rewrite
   it (G4 exchanges still run with `store=None`). This avoids double-writing
   the same state.

## OPEN-QUESTIONs (cannot be settled offline)

1. **Asymmetric eligibility → bounded or unbounded block.** The gate is local.
   Both ranks share the conversation state, but a rank that passes the gate
   while its peer does not sits in the idle tick's `mx_any` until one of these
   happens:
   - the peer's own tick, which normally comes within one tick (2 s) plus
     clock skew;
   - the peer's next coord collective, i.e. the next request, which pairs
     correctly per step 2 above;
   - the coord-socket deadline.

   Possible causes of a peer that never gets there: resident-set divergence
   (relied on, not enforced, as in G4 OQ-1), or a per-rank store min_tokens
   override. In that case the deadline raises and the runner crashes, which is
   the same failure class as any coord desync.

   *Conservative choice:* the gate uses only shared inputs (env, not the lazy
   store), and any disagreement detected at the IDLE exchange is a symmetric
   no-go plus cooldown. *Not yet:* a coord-level timeout or heartbeat on the
   idle `mx_any`.

   The supervisor's 45 s hang watchdog is gated on `in_progress`, so it does
   not fire while the runner is idle. Once a task is in progress, the tick has
   already aborted within one unit.
2. **Runner leaving generation while the peer is still active.** Examples are a
   locally decided cancel, or a rank-local error that drops the task on one
   rank only. If a rank returns to its idle wait while the peer is still inside
   a request, the idle tick's `mx_any` would pair, silently and with the same
   shape, with the peer's cancellation-check `mx_any`.

   This is a pre-existing invariant: that state already hangs without G1. With
   G1 the failure mode could become silent mispairing instead of a hang.
   *Mitigations in place:* no idle collective unless this rank's last `step()`
   was ≥ `CKPT_IDLE_S` ago, plus strict magic validation of the IDLE and POLL
   rows (a mispaired row decodes as malformed and gives no-go / abort).
   *Not done:* tagging `agree_on_tasks`' `mx_any` with a magic, which would
   change a hot-path collective. Recommended before default-on.
3. **Per-rank budget eviction** (`enforce_budget` after a commit) can still
   evict a generation on one rank only. This is COLD-safe, as G4 OQ-4 says:
   restore falls to a lower common generation or to COLD, never to a mixed
   state.
4. **Coord wait vs I/O.** Each POLL exchange waits for the slowest rank's unit
   write. Units are small (per-layer MB scale; tens of ms expected on NVMe), so
   this is far below both the coord deadline and the hang window. Unmeasured
   on hardware (A9/A10 still NEEDS-GO).
5. **SSD wear and the two-generation headroom** (plan R4). A checkpoint
   reserves `2 × largest` like a park. With checkpointing on, `MAX_GB=12` may
   evict other parked conversations sooner. Consider `EXO_DSV41_PARK_MAX_GB≈24`
   at rollout.
6. **T9 on Metal.** Donation is device-independent and the address oracle is
   exact, but end-to-end decode tok/s with checkpointing on was not measured on
   studio hardware. The plan's rollout gate ("watch logs 1 day: count, bytes,
   pauses") should include one before/after tok/s sample. The log line is
   `park.ckpt … secs=`.

**No-engine-contact affirmation:** I made no ssh, network, cluster, boot, model
request, or exo-server contact of any kind; the optional studio2 microbench was
NOT run. All work is laptop-local in `/private/tmp/p20-restart-g1`, plus a
local `git submodule update --init mlx-lm` referencing
`/private/tmp/p20-restart-g4/mlx-lm` read-only, then detached.
