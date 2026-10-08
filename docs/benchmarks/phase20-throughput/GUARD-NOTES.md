# GUARD-NOTES — `bench/phase20_guard.py` (Phase-20 R1/R2/R3/R4 safety layer)

Author: guard worker (branch `p20/guard`).  Contract: `GUARD-CONTRACT.md` (implemented verbatim:
names, signatures, exit codes).  Everything below is derived from real artifacts and pasted
live output, not memory.

## 1. Generation-triggering POST routes

Enumerated by reading `src/exo/api/main.py::_setup_routes` in the **shared checkout**
`/Users/adam.durham/repos/exo` (deploy/next13 @ f4bb14746, read-only).  Every route that emits a
`TextGeneration` / image-generation command is listed; the log line is
`exo.api.main:_log_requests:340 ] API request: POST <path>`.

Generation (S2 fires on these):

- `/v1/chat/completions`  (`chat_completions` → `_send_text_generation_with_images`)
- `/bench/chat/completions`
- `/v1/images/generations`, `/bench/images/generations`
- `/v1/images/edits`, `/bench/images/edits`
- `/v1/messages`  (Claude-compatible, → TextGeneration)
- `/v1/responses` (OpenAI Responses, → TextGeneration)
- `/ollama/v1/chat/completions`
- `/ollama/api/chat`, `/ollama/api/api/chat`, `/ollama/api/v1/chat`
- `/ollama/api/generate`

NOT generation (never fire S2): `GET /state`, `GET /metrics`, `GET /node_id`, `GET /v1/models`,
`GET /models`, `GET /events`, the dashboard static mount, and control-plane POSTs
`/instance`, `/place_instance`, `/v1/instance-links`, `/models/add`, `/download/*`,
`/v1/traces/delete`, `/onboarding`, `/v1/cancel/{id}` (a cancellation, not a request),
`/ollama/api/show`.

Real evidence: the whole saved boot carries exactly two POST paths —
`/v1/chat/completions ×75` and `/instance ×1` (fixture `real_post_routes.log`).

## 2. In-flight marker proof across the 42 real calls

Marker source: per node, `… exo.worker.runner.runner:handle_generation_tasks:571 ] runner running`
(start) and `… :860/:865 ] runner idle: reclaimed MLX allocator pool` + `… :865 ] runner ready`
(end).  Line numbers vary across builds, so the parser matches **message text only**.
Definition: `in-flight = last of {runner running, runner ready} on either node is "runner running"`.

Proof (real session `20261007_092009_9a2ed7`, sum(latency)=1530.59 s, m4-1 boot 09:01–14:00):

- runner markers: **72 running, 73 ready** → **72 closed `running→ready` intervals**.
- Bracket test (a call is bracketed when some interval satisfies
  `started_at ≤ interval_end AND ended_at ≥ interval_start`, i.e. the generation window
  overlaps the call window): **42 / 42** calls bracketed, **0 misses**.
- Distance from each call's `started_at` (state.db, UTC) to the nearest bracketing
  `runner running` (node CDT, converted): **median 0.168 s, max 4.600 s** (the max is one
  call where the caller queued ~4.6 s before the runner picked the task up).

The earlier "41/42, miss call 28" result was produced by a too-tight rule
(requiring `runner_running − started_at ∈ [0,3]`); call 28 is a *sequential* call whose
next POST arrives 0.98 s *after* the previous `runner ready`, so its own `running` marker
fires ~4.6 s after `started_at`.  The overlap rule above (the one the module uses) brackets
all 42.  Both numbers are reported; the module implements the overlap rule.

## 3. Clock offsets (live, this run)

Measured with `ssh node 'date +%s.%N'`, RTT-midpoint corrected, min-RTT of 3:

```
studio1 (m4-1, rank 1): +0.102 s   (min RTT 0.210 s)
studio2 (m4-2, rank 0): +0.104 s   (min RTT 0.215 s)
```

Both nodes are ~0.1 s *ahead* of the laptop — small (well under the 1.5 s abort gate), so
own-request matching at ±2 s is safe.  Offsets are re-measured and printed in every
`IdleReport.detail.clock_offset_s`.

## 4. S4 early-signal verification (real data)

Hermes writes the user message *before* the request is sent:

- `messages` row id **529750** (role user) ts **1791382820.275**
- first `api_calls.started_at` **1791382821.046**
- delta = **0.771 s < 1 s** ✓

So S4 (message newer than chunk start on an exo-model session) is a genuine early signal.
`session_turn_leases` is **empty** on this DB (the brief's "2 rows" no longer holds) and,
having no model/session column, is not used.

## 5. Live read-only smoke (verbatim)

`PYTHONPATH=bench python bench/phase20_guard.py idle` → exit **0**:

```
ok= True reasons= []
clock_offset_s= {'studio1': 0.1024017333984375, 'studio2': 0.10402345657348633}
state_active_tasks= 0
studio1 state= ready running= False last_post= [1791413084.9355984, '/v1/chat/completions'] post_routes= {'/instance': 1, '/v1/chat/completions': 15} n_events= 77
studio2 state= ready running= False last_post= None post_routes= {} n_events= 46
state_db= {'ok': True, 'error': None, 'recent_completed': []} s4= {'messages': [], 'sessions': []}
```

`PYTHONPATH=bench python bench/phase20_guard.py canary` → exit **0**:

```
{"ok": true, "state": "healthy",
 "per_node": {"studio1": [14.85, 14.83, 14.87], "studio2": [14.85, 14.86, 14.88]},
 "median": {"studio1": 14.85, "studio2": 14.86}}
```

(The last POST in the live log is 17:44:45 CDT, >10 min before the check, so idle=true is
correct; `/state` shows both runners `RunnerReady`, zero active tasks.)

## 6. Known limitations / false-negative analysis

- **S2 cannot discriminate for non-registering harnesses.**  `watch -- <script>` has no way
  to know which POSTs are the child's own, so S2 is *not* applied there; only S1
  (concurrency ≥ 2), S3 and S4 gate.  `signals_seen` records this.  A harness that runs one
  request at a time and registers each call (`register_own_request`) gets full S2 coverage.
- **`api_calls.ended_at` is NOT NULL** (rows written at completion), so the plan's R1(b)
  "ended_at is null" check is vacuous — D1 replaces it.  A long *in-flight* user call is
  caught by S1/S2 (runner running / unregistered POST), not by state.db.
- **Log scan is bounded to `tail -n 400`** of the matched lines.  A user arrival older than
  400 generation-marker lines but still inside the window could be missed by S2; S3/S4
  (state.db) and S1 (`/state`) are unbounded.  Within a ≤15-min chunk with a handful of
  feeds this bound is far larger than the activity produced.
- **Log rotation**: at relaunch a fresh `exo.log` replaces the old one.  Only the current
  boot is scanned; that is exactly the chunk window.
- **ssh failure / unparsable source ⇒ NOT idle** (conservative), unless the log file is
  simply absent — a healthy node with an empty match set yields `state=None`, treated as
  no activity (not a failure).  `rc != 0` always ⇒ not idle.
- **Own-request matching window ±2 s** after offset correction; a harness whose own POST is
  delayed >2 s from its `register_own_request()` call would false-positive.  Register
  immediately before the HTTP call (contract).
- **Never verified live**: behaviour when a *real* user request arrives mid-chunk (would
  require a live POST — forbidden).  Covered only by the recorded-fixture + fake-clock
  integration test (`test_watch_arrival_exit_75_and_sigint`).

## 7. Fixtures (`bench/phase20_tests/fixtures/guard/`)

| file | provenance |
|---|---|
| `real_inflight_window.log` | 8 real marker lines, m4-1 09:20:21.067–09:22:00.159 (call 1 running + call 2 running) |
| `real_completed_window.log` | real marker lines, m4-1 call-1 window 09:04:50–09:04:52 |
| `real_idle_window.log` | 338 real raw lines, 10:00:00–10:01:00 (zero guard markers) |
| `real_post_routes.log` | POST path → count over the whole boot (`/v1/chat/completions 75`, `/instance 1`) |
| `real_turn_42calls.json` | the 42 real `api_calls` (started_at, ended_at) of session `20261007_092009_9a2ed7` + all real runner running/ready markers |
