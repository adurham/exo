# JIT Model Lifecycle — Handoff (2026-06-30)

Branch: `feat/jit-model-lifecycle` (base commit `e70390e2`, branched from
`main@3c7b700f`). The admission-safety **primitive** is committed; the rest of
the feature is scoped here and NOT yet built.

## Goal (user's words)

LM-Studio-style just-in-time model loading in **exo itself**, transparent to the
client: a request for a model that isn't resident causes exo to auto-place it,
serve it, and **unload it after an idle window** — with **protections so an
auto-load can't overload / OOM the cluster**. The client (dashboard,
`claude-exo`, Hermes aux) does NOT manage lifecycle. Idle policy: unload after a
configurable idle timeout (not immediately-after-each-call).

Motivating use case: Qwen3.6-35B-A3B-8bit exists almost entirely to serve Hermes
**auxiliary** tasks (compression routes to DSv4, everything else to Qwen) while
DeepSeek-V4-Flash is the interactive model. Keeping Qwen resident costs
~17.5 GB/node on a 128 GB box that is already tight under deep DSv4 sessions.
JIT-loading Qwen on demand frees that headroom when it's not in use.

It was explicitly decided to put this in **exo** (transparent to all clients),
NOT in the hermes-agent exo provider. Hermes-side glue is acceptable only if
some piece genuinely can't live in exo.

## Why this is needed — measured facts (2026-06-30 session)

- Co-hosted 128 GB/node box. DSv4-Flash weights ~77.5 GB/node; Qwen ~17.5 GB/node.
- DSv4 **active** memory (process-only, `cache=0.00`, NOT reclaimable pool)
  climbs to **~85 GB/node at ~140K ctx**, with **~10 GB transient prefill peaks**
  on top. Settles at the `iogpu.wired_limit_mb=115000` (115 GB) wired limit.
- At ~140K ctx with both models resident, the box hit **98% on both nodes** and a
  generation **stream interrupted** (`finish_reason='length' on
  partial-stream-stub`) — peak-memory reclaim stalling the HTTP stream. It
  self-recovered (Hermes continuation stitch), but it's disruptive and worsens
  as context climbs.
- Per-snapshot KV cost measured ~0.72 GB at ~108K ctx.

The danger that drives the whole safety design:
**`ModelCard.storage_size` is WEIGHTS ONLY.** exo's existing admission
(`filter_cycles_by_memory`) checks weights vs `MemoryUsage.ram_available`. But
`ram_available` at load time reflects the resident interactive model's *weights*
and does **NOT** account for its *future KV/working-set growth*. So a naive
weights-fit check admits Qwen (35 GB into ~30 GB free) and then OOMs the box on
DSv4's next deep prefill. **Any auto-load must reserve growth headroom for what
is already resident.**

## What exo already provides (building blocks)

- `master/placement.py::place_instance()` — placement with memory filtering via
  `filter_cycles_by_memory()` (raises `ValueError("No cycles found with
  sufficient memory")`).
- `master/placement_utils.py::get_shard_assignments()` — computes per-node shard
  layout; source of the **per-node weight share** the reserve check needs.
- `api/main.py`:
  - `place_instance` / `create_instance` endpoints (create_instance already does
    a coarse `required > available` 400 check on TOTAL memory).
  - `get_placement()` (dry-run placement; raises HTTP 400 on ValueError).
  - `delete_instance(instance_id)` endpoint + `DeleteInstance` command → teardown.
  - `_validate_model_has_instance(model_id)` — **currently raises HTTP 404** when
    no instance exists for the requested model. **This is the hook for
    auto-place.** Called at the top of `chat_completions` and
    `bench_chat_completions`.
- `/state` exposes live `instances` (with `shardAssignments.modelId`),
  `nodeMemory` (`ram_available`/`ram_total`), and runner `taskStatus`
  (`RunnerReady` vs `RunnerShuttingDown`).
- `master/main.py` command loop dispatches `PlaceInstance` / `DeleteInstance`
  (event-sourced; placement is a pure function over state).

## What's DONE on this branch (commit `e70390e2`)

`src/exo/master/placement_utils.py`:

- **`jit_memory_reserve() -> Memory`** — reads `EXO_JIT_MEMORY_RESERVE_GB`
  (float GB, default **18.0**), malformed → warn + default (never raises in the
  master loop). `0` disables the reserve (pre-JIT behavior). 18 GB ≈ DSv4's
  measured growth headroom (active ~85 GB/node + ~10 GB transient peak above its
  ~77.5 GB weights).
- **`cycle_admits_with_reserve(cycle, node_memory, weight_share_per_node,
  reserve) -> bool`** — **PER-NODE** check (cycles can be lopsided): placing the
  instance must leave `>= reserve` free on EVERY node, i.e.
  `ram_available[n] - weight_share[n] >= reserve` for all n.

Both functions are currently **inert** — no callers yet. Merging this branch
alone changes no behavior.

## Remaining work (the actual feature)

### Step 1 — wire admission into placement
- In `place_instance()` (or a thin wrapper), after the existing
  `filter_cycles_by_memory`, additionally filter candidate cycles by
  `cycle_admits_with_reserve(...)` **when the placement is JIT/auto-triggered**
  (don't change behavior for explicit user placements unless desired — decide).
- Get `weight_share_per_node` from `get_shard_assignments` for the candidate
  model on the candidate cycle (weights only, sharded).
- Decision needed: on no admissible cycle → **hard-refuse** (raise, surfaced as a
  clean 503) vs. fall back to a smaller/single-node cycle. Recommend hard-refuse
  for JIT (predictable; caller falls back) — do NOT silently degrade.

### Step 2 — auto-place in the request path
- `_validate_model_has_instance()`: instead of 404 when no instance, **place the
  model and wait for `RunnerReady`** (poll `/state` taskStatus, bounded by an
  `EXO_JIT_LOAD_TIMEOUT_SECONDS`, default ~120s). On success, proceed; on
  admission refusal or timeout → clean **503** (NOT a hang, NOT a 404).
- **Single-flight lock keyed by model_id**: concurrent first-requests for the
  same model must trigger ONE placement and all wait on it — never double-place.
  An `asyncio.Lock` per model_id (or a `dict[ModelId, asyncio.Event]`) in the API
  server. Hold across the place→ready wait.
- Only auto-place models that are **downloaded** (the existing
  `model_is_downloaded` check) — otherwise keep the download-notify + 404/clean
  error; do not block a request for minutes on a cold model download.
- **Client-visible latency change**: the first request for an unloaded model
  blocks ~30-60s (Qwen) while it loads. Acceptable for aux; document it. The
  dashboard/`claude-exo` get the same transparent behavior.

### Step 3 — idle reaper
- Background task in the master (or API server) that tracks **per-instance
  last-use timestamp** (update on each `chat_completions` dispatch for that
  instance's model).
- After `EXO_JIT_IDLE_UNLOAD_SECONDS` (default e.g. 300) of no use, issue
  `DeleteInstance` for **JIT-placed** instances only.
- **Must never unload**: an instance with an in-flight request, or a
  non-JIT/"pinned" instance (the interactive model). Tag JIT-placed instances
  distinctly from explicit user placements — e.g. an `InstanceMeta` flag or a
  master-side set of JIT instance ids. The interactive model (DSv4) must be
  immune.
- Guard the in-flight race: a request can arrive between the idle check and the
  delete — refcount in-flight requests per instance and skip teardown if > 0.

### Step 4 — admission-refusal UX
- When a needed model can't be admitted (reserve fails) or load times out,
  return a clean **503** with a clear message. Hermes aux should treat this as a
  fall-back trigger (e.g. route that aux task elsewhere), not an error that
  wedges the turn. Confirm the OpenAI-compat error shape Hermes' exo provider
  expects.

### Step 5 — tests
- `cycle_admits_with_reserve`: lopsided-node cycle (one node fails), exact
  reserve boundary, env override via `EXO_JIT_MEMORY_RESERVE_GB`, reserve=0
  disables.
- `jit_memory_reserve`: default, valid override, malformed → default + warn.
- Single-flight: two concurrent first-requests → one placement.
- Idle reaper: skips in-flight, skips pinned/interactive, unloads idle JIT
  instance after window.
- Refusal path: no admissible cycle → 503, not OOM, not hang.

## Config knobs (all new; document in start_cluster.sh allow-list like
`EXO_LEAF_SNAPSHOT_RETENTION`)

- `EXO_JIT_MEMORY_RESERVE_GB` (default 18.0; 0 disables reserve) — **implemented**.
- `EXO_JIT_LOAD_TIMEOUT_SECONDS` (default ~120) — TODO.
- `EXO_JIT_IDLE_UNLOAD_SECONDS` (default ~300) — TODO.
- Possibly `EXO_JIT_ENABLED` (master kill-switch; default off until proven) — TODO.

## Deploy / safety notes (exo cluster, see exo-cluster-operations skill)

- exo `src/` deploys via **git only**: edit local → commit → push
  `origin/main` → `start_cluster.sh` does `git reset --hard origin/main` +
  `uv sync` on both nodes. NEVER scp/edit-on-node.
- `start_cluster.sh` env is an **allow-list** (~line 920-945): a new `EXO_*` var
  needs BOTH a default declaration (~line 283 area) AND a passthrough line, or it
  won't reach the runner. Verify with
  `ssh m4-1 'ps -axo command | grep -oE "EXO_JIT[^ ]+"'`.
- Minimize `start_cluster.sh` restarts — repeated runs leak RoCE queue-pairs and
  wedge the TB link (reboot-only recovery). Batch changes into one deploy.
- Pre-commit gates (AGENTS.md): `uv run basedpyright` (diff added-errors against
  baseline; cache/placement files have many pre-existing), `uv run ruff check`,
  `uv run pytest`. Format with `nix fmt` (NOT bare `ruff format` — it rewrites
  unrelated code in this repo since files aren't ruff-formatted).
- Pyright walking untyped mlx objects: mirror the existing
  `# type: ignore[reportUnknownVariableType]` / `[reportUnknownArgumentType]`
  convention rather than fighting the types.

## Cross-references

- Cache work that shipped same day (context for the memory pressure):
  `main@0888c0e2` — high-priority KV eviction (`1d680db5`), evenly-spaced
  snapshot retention (`3c7b700f`), `EXO_LEAF_SNAPSHOT_RETENTION` 4→3
  (`0888c0e2`). See `src/exo/worker/engines/mlx/cache.py`.
- Memory measurement method: `~/.hermes/skills/exo/exo-cluster-debugging`
  (the `[MEM] before prefill` active number is authoritative; dashboard/`/state`
  over-read by the reclaimable Metal pool + co-hosted Qwen).
