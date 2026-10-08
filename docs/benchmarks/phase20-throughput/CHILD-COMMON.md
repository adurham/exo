# CHILD-COMMON — read this FIRST, in full (shared by every Phase-20 worker)

You are a leaf worker in a throughput campaign (prefill tok/s + decode tok/s) on a 2-node exo cluster. You cannot ask anyone questions and
cannot delegate. Your PM verifies every claim you make against the artifact (diff, test output, file contents) — report only what you ran.

## Facts (verified 2026-10-07 ~21:30 CDT)
* Cluster: 2x Mac Studio M4 Max 128 GB, TP2 over jaccl RDMA. ssh aliases `studio1` (m4-1, 192.168.86.48, master + API node, **rank 1**) and
  `studio2` (m4-2, 192.168.86.47, jaccl coordinator, **rank 0**). Model `dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw`, engine `dsv41`
  (DSpark draft head, greedy only, gamma 3 default, per-request `spec_gamma` already exists on branch deploy/next14-gamma).
* API: `http://macstudio-m4-1.tail19c543.ts.net:52415` (tailnet name, stable) or `http://192.168.86.48:52415` (LAN, DHCP-pinned). `GET /state` (27 KB JSON),
  `GET /metrics` (Prometheus), `POST /v1/chat/completions`. Nodes run the production build deploy/next13 @ f4bb14746 and are IDLE right now.
* Node paths: repo `~/repos/exo`, venv python `~/repos/exo/.venv/bin/python`, current-boot log `~/.exo/exo_log/exo.log` (rotated `*.zst` siblings; also
  `~/exo.log` which is the launcher-redirected stdout), runner stderr `~/.exo/exo_log/runner_log/stderr.log` (APPEND-ONLY across boots - anchor on pid/lstart).
  Runner pid = the `multiprocessing.spawn ... spawn_main` child of `python -m exo -v` (`pgrep -f 'multiprocessing.spawn import spawn_main'`).
* exo.log line shapes (timestamps are node-local CDT, no tz marker; state.db epochs are UTC; CDT = UTC-5 today):
  `API request: POST /v1/chat/completions` (also GET /state, /metrics, /node_id, /v1/models polls - NOT requests),
  `Starting task TextGeneration(task_id=...`, `[DSV41] session reuse: this N-token prompt matches a resident conversation on M rows`,
  `[DSV41] prefill controls: ... (rows=M, base=2048)` (prefill entry), `[DSV41] turn reuse: prompt=N prefill=M reuse=R cache=C rewind=W UNCOMMITTED`
  (emitted when prefill returns; ONLY when reuse>0, so a fully cold feed has no such line), `[DSV41] reuse undershoot: refed=M rows (reused=R)` (WARNING when refed>256 and reuse>0),
  `Executing command: TaskFinished(command_id=..., finished_command_id=...)`. A real 42-call session's log window is saved locally (zstd):
  `/Users/adam.durham/.hermes/cache/scratch/phase20/m41_0901_1400.log.zst` (m4-1, boot 09:01-14:00; use `zstdcat`; **never** write its decompressed form into the repo).
* state.db (READ-ONLY, always `sqlite3.connect("file:/Users/adam.durham/.hermes/state.db?mode=ro", uri=True)`): `api_calls(session_id, call_seq, started_at, ended_at, latency_seconds,
  input_tokens, cache_read_tokens, output_tokens, reasoning_tokens, prompt_tokens_total, provider, model, ...)`; **`ended_at` is NOT NULL (rows are written at call completion)** so
  "ended_at is null" never matches anything. **The exo model rows are `provider='custom'` AND `model='dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw'`.
  PITFALL: SQLite LIKE is case-insensitive; `LIKE '%DeepSeek-V4.1%'` also matches the unrelated ollama model `deepseek-v4.1-flash` (the PM/agent sessions themselves). Use exact equality.**
  The real turn used throughout: session `20261007_092009_9a2ed7`, 42 calls, sum(latency)=1530.59 s.
* Prior art you may read (do not trust blindly; re-derive what you report): `/Users/adam.durham/repos/exo/docs/benchmarks/phase19-latency/` (restore-snapshot-live.md,
  levers-postreboot-2026-10-07.md, raw/phase1-accounting.md, raw/phase2-delta-audit.md), `/private/tmp/next14-gamma/docs/benchmarks/phase19-latency/` (round-loop-map.md, raw/A1-real-turn-split.md,
  gamma-depth-2026-10-07.md), harnesses `/private/tmp/levers-wt/bench/phase19_{round,agentic}_measure.py`, `phase19_gamma_matrix.py` (SSE parsing, `: generation_stats` comment frame, salted filler builders).
* Campaign docs: `/private/tmp/phase20-campaign/docs/benchmarks/phase20-throughput/` - read `PREREG.md` (frozen gates + deviations D1-D7) and, if your task touches the guard, `GUARD-CONTRACT.md`.
* The full plan is `/Users/adam.durham/.hermes/cache/scratch/PHASE20-PLAN.md` (do not edit).

## Hard rules
1. **The cluster is READ-ONLY for you.** Allowed: `ssh studioN '<read-only command>'` (ps, pgrep, cat, grep, tail, ls, zstdcat, scp FROM a node), `GET` endpoints. FORBIDDEN: any generation request
   (`POST /v1/chat/completions` etc.), starting/stopping/killing any process on a node, `start_cluster.sh`, reboot, writing to `~/repos/exo` on a node, `sudo`. (Writing a scratch file under `/tmp` on a node for a
   canary/parser fixture is OK only if your brief says so.) The PM runs all live measurements.
2. **Do not touch the shared checkout `/Users/adam.durham/repos/exo`** (it is the rsync source for deploys; its branch/worktree state is load-bearing). No checkout, commit, add, stash, reset there. Read it freely.
   Do not touch global Hermes config (`hermes config set` is out of scope). Never push to `upstream` / `exo-explore`; **do not push at all** (the PM pushes after review).
3. Git hygiene: work only in your assigned worktree; `git branch --show-current` before every commit; stage EXACT paths (`git add <file> <file>`; never `git add -A`/`.`); never `git stash`; if a commit fails on
   `index.lock` (siblings share the repo), sleep 5 s and retry. `bench/**/*.json` can be gitignored: force-add with `git add -f <path>`.
4. Python: stdlib only unless told otherwise; `from __future__ import annotations`; run with the shared venv python `/Users/adam.durham/repos/exo/.venv/bin/python` (3.13). Never `export PYTHONPATH`
   (it leaks into sibling sessions) - use inline `PYTHONPATH=... python ...` per command. Never run the full test suite (600 s) - scoped paths only. For dsv41 tests the incantation is
   `EXO_DASHBOARD_DIR=/Users/adam.durham/repos/exo/dashboard/build PYTHONPATH=<worktree>/src:/Users/adam.durham/repos/exo/mlx-lm /Users/adam.durham/repos/exo/.venv/bin/python -m pytest <scoped path> -q -p no:cacheprovider`.
   (`mlx-lm` inside a fresh worktree is an EMPTY submodule dir - use the shared checkout's `mlx-lm` via PYTHONPATH, same pinned sha 6cc9c1e.)
5. No fabricated numbers. If a field cannot be derived from a real log/db/file, write `null`/`UNKNOWN` and say why. A partial honest result beats a complete invented one.
6. Scratch files go in `/Users/adam.durham/.hermes/cache/scratch/phase20/` (create subdirs freely), never `/tmp` on the laptop root beyond throwaway.
7. Other agents are working concurrently on other files in the campaign worktree. Touch only the files in your brief.

## Report format (your final message - keep it tight)
Outcome first (done / partial / blocked), then bullets: files created/modified with absolute paths, commit SHA(s) + `git status --short` of your worktree (clean for your files), the exact test/command output
lines proving it works (paste, don't paraphrase), key numbers/findings, and an explicit **NOT verified** list. If you hit a wall, say what you tried and stop.

## Process addenda (PM, 2026-10-07 ~21:50 CDT) - these OVERRIDE anything contrary in your brief
A1. **You have your OWN worktree + branch** (named in your task goal), cut from `deploy/phase20-campaign`. Wherever your brief says `/private/tmp/phase20-campaign`,
    use YOUR worktree path instead. Do not touch any other worktree. Sibling workers are building sibling modules in their own worktrees; to read a sibling's file
    (e.g. the guard) read it from the sibling's worktree path given in your goal - never copy-edit it.
A2. **Commit with an explicit pathspec** so a stray staged file can never ride along: `git add <paths> && git commit -m "<msg>" -- <paths>`. Verify with `git show --stat HEAD` that ONLY your files are in the commit.
A3. **Tests:** the repo-root conftest has a landmine guard that refuses to run from a worktree whose `mlx-lm/` is empty. Your bench/ tests do not import exo, so run them with
    `--noconftest`:  `cd <your worktree> && PYTHONPATH=bench /Users/adam.durham/repos/exo/.venv/bin/python -m pytest --noconftest bench/phase20_tests/<file> -q -p no:cacheprovider`
    (verified working). Do NOT try to bypass the guard for src/ tests - that is a different brief.
A4. Budgets are in TOOL CALLS (you cannot see a clock): stop and report by the budget in your goal even if incomplete; a partial, honest, committed result beats an overrun.
A5. `bench/phase20_tests/__init__.py` and `bench/phase20_tests/fixtures/` already exist on your branch (PM-captured idle `powermetrics`/`sample` fixtures under `fixtures/`).
