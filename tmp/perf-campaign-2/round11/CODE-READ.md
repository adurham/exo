# CAMPAIGN 2 / ROUND 11 — Task 1: END-TO-END CODE READ of the c=1 chat-completion path

**Scope:** pure code read. No source edits, no cluster time, no test-suite run, no runner restart.
**Repo:** `/Users/adam.durham/repos/exo` @ `76294c3d4` (branch `main`).
**Submodules read:** `mlx-lm` @ `7f146542811dd774d2cb3ab38b13c4b25a8e8063`, `mlx` @ `e40a416b20851d118b061b3a57d8cab70f5756de` (`git submodule status`).
**Citation convention:** `path:LINE` or `path:START-END`, paths relative to repo root. `mlx-lm/...` = the vendored submodule.

## Live-configuration assumptions (established from source, cite each)

These determine *which* of several branches is the c=1 path. Every claim below is conditioned on them.

- `DSV4_SHARDING` defaults to `Tensor` — `start_cluster.sh:401`. Therefore the Pipeline-only branches
  (`pp_no_coord_collective`, `EXO_PP_BATCHED_DECODE`) are **off**: `EXO_PP_BATCHED_DECODE` is only defaulted inside the
  Pipeline branch at `start_cluster.sh:2496`, and `pp_no_coord_collective` requires `EXO_PP_NO_COORD_COLLECTIVE=1`
  (`src/exo/worker/engines/mlx/generator/batch_generate.py:2375-2379`).
- `EXO_NO_BATCH` is **not** set by `start_cluster.sh` (grep of the file returns no hits), so the engine is
  `BatchGenerator`, not `SequentialGenerator` — `src/exo/worker/engines/mlx/builder.py:163-190`.
- `EXO_DSV4_BATCHED_PREFILL` defaults to `1` (`start_cluster.sh:129`, `src/exo/shared/constants.py:115`) but
  `EXO_BATCHED_PREFILL_RENDEZVOUS_MS` was shipped to `0` in R10 (`start_cluster.sh:145`), overriding the source default of
  `200` at `src/exo/shared/constants.py:138-140`.
- Master, worker, API and router all live in ONE process per node (`src/exo/main.py:57-122`); the *runner* is a separate
  OS process (`src/exo/worker/runner/supervisor.py:297-306`).

---

## (a) WHERE IS `server_received_ts` STAMPED — and what is already paid by then

### The exact emit site

```
src/exo/worker/runner/runner.py:563
    logger.info(f"received chat request: {truncate_for_log(starting_task)}")
```

This is the ONLY emitter of that string in the tree (`grep -rn "received chat request" --include=*.py .` → one hit).

**VERDICT: the stamp is NOT in the API process at all. It is in the RUNNER process**, at the top of
`Runner.handle_generation_tasks` (`src/exo/worker/runner/runner.py:559-563`). The study's "client→server transit"
(0.191 s median) is therefore **client-serialize + HTTP + FastAPI/pydantic + master-index + gossip round-trip + worker
plan-poll + mp.Queue IPC into the runner process** — it is emphatically not network transit.

### Ordered list: what is ALREADY DONE before the stamp

1. **Full HTTP body read off the socket — YES, done.** hypercorn is the ASGI server (`src/exo/api/main.py:20-22`,
   `src/exo/api/main.py:2425-2441`). The route is declared with a pydantic body model
   (`src/exo/api/main.py:451-453` → handler `src/exo/api/main.py:1129-1131`, signature
   `payload: ChatCompletionRequest`), so FastAPI cannot call the handler until the body is fully received.
2. **JSON parse — YES, done.** Same mechanism; FastAPI decodes the body before validation.
3. **Pydantic validation — YES, done.** `ChatCompletionRequest` is defined at `src/exo/api/types/api.py:243-283`; the
   handler receives an already-constructed model.
4. **Message normalization / `chat_template_messages` construction — YES, done.**
   `chat_request_to_text_generation` runs at `src/exo/api/adapters/chat_completions.py:62-189`, called from
   `src/exo/api/main.py:1133`. This walks all ~55 messages and `model_dump`s each one
   (`src/exo/api/adapters/chat_completions.py:144-147`).
5. **Command dispatch to master — YES, done.** `src/exo/api/main.py:1137` → `_send_text_generation_with_images`
   (`src/exo/api/main.py:1057-1088`) → `await self._send(command)` at `src/exo/api/main.py:1087` →
   `src/exo/api/main.py:2646-2651` publishes a `ForwarderCommand` on the `COMMANDS` topic
   (`src/exo/routing/topics.py:42`, `PublishPolicy.Always` → JSON-serialized and gossiped,
   `src/exo/routing/router.py:59-60`, `:92-96`, `src/exo/routing/topics.py:33-34`).
6. **Master indexed the command into a `TextGenerationTask` — YES, done.**
   `src/exo/master/main.py:209-280` (the `case TextGeneration()` arm; `TaskCreated` emitted at
   `src/exo/master/main.py:268-279`), then indexed/applied/broadcast at `src/exo/master/main.py:646-671`.
7. **Worker's `plan_step` poll observed the task and dispatched it — YES, done.**
   `src/exo/worker/main.py:193-208` (poll), `plan()` → `_pending_tasks` at `src/exo/worker/plan.py:347-390`,
   dispatch at `src/exo/worker/main.py:383` (`self._tg.start_soon(self._start_runner_task, task)`).
8. **IPC into the runner process — YES, done.** `RunnerSupervisor.start_task`
   (`src/exo/worker/runner/supervisor.py:361-382`) → `MpSender.send_async`
   (`src/exo/utils/channels.py:269-272`, a `to_thread.run_sync` over a blocking `mp.Queue.put`), read by the runner's
   task-reader thread (`src/exo/worker/runner/runner.py:255-288`) and popped by the main loop at
   `src/exo/worker/runner/runner.py:335`.

### What is STILL AHEAD of the stamp

1. **Chat-template render (Jinja / DSv4 encoder) — NOT yet done.** Happens later, at
   `src/exo/worker/runner/llm_inference/batch_generator.py:1158-1159`.
2. **Tokenization — NOT yet done.** Happens at `src/exo/worker/engines/mlx/generator/batch_generate.py:2194-2198`.
3. **Prefix-trie walk + KV restore — NOT yet done** (`src/exo/worker/engines/mlx/generator/batch_generate.py:2386-2392`).
4. **Prefill, decode, SSE — NOT yet done.**

### Consequence for the study's split

The study's `server_received_ts − client_started_ts` = [0.077, 0.394] s, median 0.191 s **already contains**
body-receive + json + pydantic + adapter message-dump + master indexing + a full gossip publish→index→broadcast
round-trip + the worker's 0-100 ms `plan_step` tick + mp.Queue IPC. It contains **NO** tokenization and **NO** chat-template
render — those land in the *residual*, on the far side of the stamp. Reviewer expectation #7 ("if stamped after
tokenization, #1/#6 are already hiding inside transit") is **REFUTED for tokenization** and **CONFIRMED for HTTP
body/json/pydantic**.

---

## (b) EVERY sleep / timeout / poll-interval — critical path vs. background

Search performed over `src/exo/` with: `asyncio.sleep(`, `time.sleep(`, `anyio.sleep(`, `await sleep(`, `.get(timeout=`,
`wait_for(`, `move_on_after(`, `fail_after(`, `.wait(timeout=`, `POLL_INTERVAL`, `_INTERVAL`, `_MS`, `sleep_ms`,
`backoff`, `select.select(`, `epoll` — plus a targeted sweep of `src/exo/worker/engines/` and `mlx-lm/mlx_lm/`.

### LIST 1 — ON the c=1 critical path

| # | file:line | tick | env var / default | worst-case added | expected added |
|---|-----------|------|-------------------|------------------|----------------|
| 1 | `src/exo/worker/main.py:195` | `await anyio.sleep(0.1)` at the TOP of `plan_step`'s `while True` | none (hardcoded 0.1 s) | **100 ms** | **~50 ms** |
| 2 | `src/exo/worker/runner/runner.py:580-604` | rendezvous drain, `self._work_queue.get(timeout=remaining)` at `:602` | `EXO_BATCHED_PREFILL_RENDEZVOUS_MS`, source default `200` (`src/exo/shared/constants.py:138-140`), **shipped 0** (`start_cluster.sh:145`) | 0 ms as shipped (200 ms if the override is ever lost) | 0 ms |

**#1 is the biggest live one and is a genuine, unfixed, per-request tick.** The task lands in `self.state.tasks` via the
event applier (`src/exo/worker/main.py:140-144`), but nothing wakes `plan_step`; it sleeps 0.1 s *first*, then calls
`plan()` (`src/exo/worker/main.py:194-196`). `_pending_tasks` (`src/exo/worker/plan.py:347-390`) is what actually
selects the task. It gates exactly one hop (master-broadcast → runner dispatch), so worst-case 100 ms / expected 50 ms
per request. This same loop is already documented as a pure poll elsewhere in the tree —
`src/exo/api/main.py:850-851` calls it "a plain 100ms poll loop, confirmed by reading it — NOT event-triggered".

**#2 is the R10 rendezvous.** Confirmed neutralized at the launcher (`start_cluster.sh:145` sets it to 0), and the code
guards on `EXO_BATCHED_PREFILL_RENDEZVOUS_MS > 0` (`src/exo/worker/runner/runner.py:580`), so the whole block is skipped.

### Explicitly NOT ticks (checked and cleared — do not re-litigate)

- `src/exo/api/keepalive.py:26` `with anyio.move_on_after(interval)` (interval `10.0`, `src/exo/api/keepalive.py:12`).
  `move_on_after` is a *deadline*, not a poll: `recv.receive()` returns immediately when an item is available. Adds
  **0 ms**. It also yields the first keepalive byte eagerly at `src/exo/api/keepalive.py:14`, before the generator is
  even started, so SSE headers flush without waiting for the first token.
- `src/exo/utils/channels.py:332-346` `MpReceiver.receive` / `:348-351` `receive_async` — a **blocking** `mp.Queue.get`
  run on a worker thread. Event-driven, no interval.
- `src/exo/worker/runner/runner.py:335` `self._work_queue.get()` — blocking, no timeout.
- `src/exo/worker/runner/runner.py:794` `self._work_queue.get_nowait()` — non-blocking; on `queue.Empty` it `continue`s
  straight into the next `generator.step()` (`src/exo/worker/runner/runner.py:795-796`). No sleep in the decode loop.
- `src/exo/api/main.py:2647-2648` `while self.paused: await self.paused_ev.wait()` — an `Event`, not a poll.
- `src/exo/worker/runner/supervisor.py:377-382` `await event.wait()` — an `anyio.Event` set by `TaskAcknowledged`
  (`src/exo/worker/runner/supervisor.py:414-416`). Event-driven. It is also invoked via `start_soon`
  (`src/exo/worker/main.py:383`), so it never blocks the planner loop.
- `mlx-lm`: the only `time.sleep` in the submodule is `mlx-lm/mlx_lm/benchmark.py:160` — not on the serving path.
- `src/exo/worker/engines/`: the only two hits are `src/exo/worker/engines/mlx/utils_mlx.py:131` (jaccl init backoff,
  startup only, `src/exo/worker/engines/mlx/utils_mlx.py:108-131`) and
  `src/exo/worker/engines/mlx/pp_batched_correctness.py:175` (`_RECV_TIMEOUT_SECONDS = 30.0`,
  `src/exo/worker/engines/mlx/pp_batched_correctness.py:105`) which is a *simulated* transport used only by
  correctness tests. **Neither is on the c=1 path.**

### LIST 2 — NOT on the c=1 critical path (background / idle / health / error-only)

| file:line | tick | role |
|-----------|------|------|
| `src/exo/api/main.py:1358` | `anyio.sleep(0.25)` | JIT load wait; only when `_validate_model_has_instance` misses (`src/exo/api/main.py:1233-1238`) |
| `src/exo/api/main.py:1429` | `anyio.sleep(min(_JIT_PLACEMENT_POLL_SECONDS, remaining))`, `_JIT_PLACEMENT_POLL_SECONDS = 2.0` (`src/exo/api/main.py:246`) | JIT placement wait |
| `src/exo/api/main.py:755`, `:765` | `anyio.sleep(0.1)` | `/await_instance` SSE endpoint only |
| `src/exo/api/main.py:908` | `anyio.sleep(0.05)` | cancel-ack wait, `cancel_command` only |
| `src/exo/api/main.py:2641` | `anyio.sleep(cleanup_interval_seconds)` | background cleanup |
| `src/exo/master/main.py:520` | `anyio.sleep(10)` | master `_plan` |
| `src/exo/master/main.py:537` | `anyio.sleep(5)` | JIT idle reaper |
| `src/exo/worker/main.py:182` | `anyio.sleep(1)` | custom-model-card reconcile |
| `src/exo/worker/main.py:477` | `anyio.sleep(10)` | `_poll_connection_updates` |
| `src/exo/worker/runner/supervisor.py:445` | `anyio.sleep(5)` | `_watch_runner` hang watchdog |
| `src/exo/worker/runner/supervisor.py:390` | `move_on_after(0.5)` | cancel pipe guard |
| `src/exo/worker/runner/runner.py:196` | `time.sleep(1.0)` | stall sampler; **disabled unless `EXO_STALL_SAMPLER_SECONDS>0`** (`src/exo/worker/runner/runner.py:181-183`) |
| `src/exo/worker/runner/runner.py:131` | `time.sleep(timeout)` | test-only `should_timeout` fixture |
| `src/exo/worker/runner/runner.py:688` | `time.sleep(300)` | `MLX_DIAG_HOLD_WEDGE` diagnostic only |
| `src/exo/worker/runner/runner.py:237`, `:243` | `PREFILL_PICKUP/FINISH_TIMEOUT_SECONDS` | disaggregated prefill server; gated on `ENABLE_DISAGGREGATION` (`src/exo/worker/runner/runner.py:222`, default false at `src/exo/shared/constants.py:106`) |
| `src/exo/worker/runner/llm_inference/batch_generator.py:90` | `time.sleep(100)` | `EXO_RUNNER_MUST_TIMEOUT` debug-prompt trigger |
| `src/exo/routing/event_router.py:78` | `anyio.sleep(1 + random())` | `_simple_retry` for undelivered events |
| `src/exo/routing/event_router.py:169` | `anyio.sleep(delay)`, 0.5 s × 2^n capped 10 s (`src/exo/routing/event_router.py:62-63,165-167`) | NACK request; **fires only on an out-of-order/dropped event** (`src/exo/routing/event_router.py:136-142`) |
| `src/exo/routing/router.py:170` | `move_on_after(1, shield=True)` | shutdown unsubscribe |
| `src/exo/shared/election.py:163`, `:214` | `anyio.sleep(0.2)` / campaign timeout | election |
| `src/exo/utils/async_process.py:165` | `await sleep(0.01)` | process-exit wait |
| `src/exo/utils/info_gatherer/*`, `src/exo/utils/power_sampler.py:39`, `src/exo/download/*` | various | telemetry / download |

### Structural (non-sleep) latency on the critical path, flagged but NOT quantified here

Every hop between API and runner traverses the gossip topic layer, JSON-serialized:
`src/exo/routing/topics.py:33-34` (`model_dump_json().encode`), published at `src/exo/routing/router.py:92-96` and
`:221-229`. `LOCAL_EVENTS`/`GLOBAL_EVENTS`/`COMMANDS` are all `PublishPolicy.Always`
(`src/exo/routing/topics.py:40-42`), so they hit the network even when a local receiver exists
(`src/exo/routing/router.py:59-60`). The API only accepts events that came back from the **master**
(`src/exo/routing/event_router.py:122-123`), and they pass through an `OrderedBuffer`
(`src/exo/routing/event_router.py:125-130`). If the elected master is the *other* node, a request pays two
cross-node hops before dispatch and every token chunk pays two more. **NOT MEASURED here — flagged as a Task-2 target.**

---

## (c) TOKENIZATION

### Full prompt tokenized on EVERY turn — YES. No tokenization-level prefix cache.

```
src/exo/worker/engines/mlx/generator/batch_generate.py:2194-2198
        with T("submit.encode_prompt"):
            all_prompt_tokens = encode_prompt(self.tokenizer, prompt)
            all_prompt_tokens = fix_unmatched_think_end_tokens(
                all_prompt_tokens, self.tokenizer
            )
```
```
src/exo/worker/engines/mlx/cache.py:2285-2294
def encode_prompt(tokenizer: TokenizerWrapper, prompt: str) -> mx.array:
    ...
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    return mx.array(prompt_tokens)
```

One unconditional `tokenizer.encode()` over the **entire** rendered prompt string, every request. Searched for a
memo/cache around it (`lru_cache`, `@cache`, `cached_property`, `_token_cache`, `tokenize_cache` across
`utils_mlx.py`, `cache.py`, `batch_generate.py`) — **zero hits. NOT FOUND: there is no tokenization-level prefix cache.**
The prefix cache is purely a *KV* cache keyed on already-computed token ids (see below).

### HF fast (Rust) tokenizer — YES, and effectively single-threaded per call

`load_tokenizer_for_model_id` (`src/exo/worker/engines/mlx/utils_mlx.py:1246-1322`) delegates to mlx-lm's
`load_tokenizer` at `src/exo/worker/engines/mlx/utils_mlx.py:1315-1319`, which calls
`AutoTokenizer.from_pretrained(...)` at `mlx-lm/mlx_lm/tokenizer_utils.py:704-706`. The DSv4 checkpoint declares
`"tokenizer_class": "PreTrainedTokenizerFast"` (read from
`/Users/adam.durham/.exo/models/deepseek-ai--DeepSeek-V4-Flash/tokenizer_config.json`), i.e. the Rust `tokenizers`
backend. `TokenizerWrapper.__getattr__` forwards `.encode` straight through
(`mlx-lm/mlx_lm/tokenizer_utils.py:547-555`). The call site uses the **single-string** `encode()`, not `encode_batch`
(which exists at `mlx-lm/mlx_lm/tokenizer_utils.py:584-585` but is unused here) — so no intra-call parallelism.
Kimi is the only special case with a Python/tiktoken path (`src/exo/worker/engines/mlx/utils_mlx.py:1301-1303`), not
applicable to DSv4.

### Node 2 tokenizes INDEPENDENTLY — no multi-MB token array on the wire

Every rank runs the identical `submit()` on the identical `TextGenerationTaskParams`. The task is broadcast as an
**event/command carrying the messages, not tokens**: `TaskCreated` with `task_params`
(`src/exo/master/main.py:268-279`), and the runner-side `_start_task` renders + encodes locally
(`src/exo/worker/runner/llm_inference/batch_generator.py:1158-1159`,
`src/exo/worker/engines/mlx/generator/batch_generate.py:2194-2195`). Rank 1 reaching the same code is confirmed by the
rank-guards that exist precisely because both ranks run it — e.g.
`src/exo/worker/runner/runner.py:876-878` (only rank 0 emits `ChunkGenerated`) and
`src/exo/worker/engines/mlx/generator/batch_generate.py:2280-2288` ("each rank's KVPrefixCache is an independent
per-process radix trie"). **Verdict: parallel and cheap on the wire; the CPU cost is paid twice but concurrently.**

### Chat template rendered per turn over the FULL conversation — YES

`apply_chat_template` (`src/exo/worker/engines/mlx/utils_mlx.py:1569-1610`) rebuilds the whole `messages` list
(`:1584-1605`) and calls `render_chat_template` (`:1607`). For DSv4 this does **not** go through Jinja: the
`_needs_v4_encoding` branch (`src/exo/worker/engines/mlx/utils_mlx.py:1489-1492`, predicate at `:1400-1401`
`"deepseek-v4" in model.lower()`) routes to the vendored pure-Python encoder
`src/exo/worker/engines/mlx/vendor/deepseek_v4_encoding.py:598` (`encode_messages`, 872-line module). It re-does
`merge_tool_messages` / `sort_tool_results_by_call_order` / `_drop_thinking_messages` over **all** messages every turn
(`src/exo/worker/engines/mlx/vendor/deepseek_v4_encoding.py:630-650`). The Jinja path
(`tokenizer.apply_chat_template`, `src/exo/worker/engines/mlx/utils_mlx.py:1551-1558`) is the *non*-DSv4 fallback.
Either way: **full re-render every turn, no incremental reuse.**

### The prefix trie keys on TOKEN IDS, materialized to numpy — not bytes, not a hash

**Lookup:**
```
src/exo/worker/engines/mlx/cache.py:1190-1192
        match_node, match_length = self._longest_prefix_match(
            prompt_tokens, query_regions
        )
```
`_longest_prefix_match` is `src/exo/worker/engines/mlx/cache.py:1595-1673`. It converts the query to numpy once
(`:1604`, via `_tokens_to_np` at `src/exo/worker/engines/mlx/cache.py:2010-2017`), then per trie edge does
`int(prompt_np[matched])` for a dict lookup (`:1616-1617`), converts the edge's tokens to numpy (`:1621`) and runs a
vectorised compare (`:1622` → `_np_prefix_length` at `src/exo/worker/engines/mlx/cache.py:2020-2036`). So the walk is
**O(number of edges) in Python, with numpy-vectorised per-edge comparison** — not a Python per-token loop over 150K ids.
Reviewer expectation #2's "Python per-token trie walk over 150K ids = 50-150 ms" is therefore **structurally overstated**;
the per-token compare is in numpy.

An additional **whole-prefix numpy compare runs on every hit** as a permanent integrity check:
`src/exo/worker/engines/mlx/cache.py:1248-1257` (`np.array_equal(_donor_tokens_np[:match_length], _query_np[:match_length])`),
deliberately left ungated (`:1245-1247`).

**Insert:** `src/exo/worker/engines/mlx/cache.py:1441-1503` (`_insert_path`), which re-walks the trie the same way
(`:1452`, `:1458-1459`, `:1474-1475`). It is called from `add_kv_cache` at `src/exo/worker/engines/mlx/cache.py:828-834`.
`add_kv_cache` **also** runs a second full `_longest_prefix_match` purely for a log line
(`src/exo/worker/engines/mlx/cache.py:823-825`, consumed at `:861-868`) — i.e. **the trie is walked twice on a commit**,
once for diagnostics. The extend path is `_extend_leaf_suffix` (`src/exo/worker/engines/mlx/cache.py:1027-1113`), which
slices only `[old_depth, new_length)` (`:1079-1088`) and does **not** re-slice the shared prefix (`:1038-1041`).

---

## (d) KV RESTORE — THE `mx.eval` QUESTION

### The restore implementation

`get_kv_cache` (`src/exo/worker/engines/mlx/cache.py:1157`) ends in
`_materialize_cache_to_depth` (`src/exo/worker/engines/mlx/cache.py:1705-1787`), invoked at
`src/exo/worker/engines/mlx/cache.py:1315-1320`. The body:

```
src/exo/worker/engines/mlx/cache.py:1728-1751
        # Walk root→donor_leaf.node collecting edges up to target_depth.
        path = _collect_path(self._root, donor_leaf.node)
        num_layers = len(donor_leaf.leaf_layer_caches)

        per_layer_keys: list[list[Any]] = [[] for _ in range(num_layers)]
        per_layer_values: list[list[Any]] = [[] for _ in range(num_layers)]

        depth_so_far = 0
        for edge in path:
            if depth_so_far >= target_depth:
                break
            take = min(edge.edge_length, target_depth - depth_so_far)
            if edge.edge_keys is not None and edge.edge_values is not None:
                for layer_idx in range(num_layers):
                    k = edge.edge_keys[layer_idx]
                    v = edge.edge_values[layer_idx]
                    if not _has_tokens(k):
                        continue
                    if take < edge.edge_length:
                        k = _slice_seq_axis(k, 0, take)
                        v = _slice_seq_axis(v, 0, take)
                    per_layer_keys[layer_idx].append(k)
                    per_layer_values[layer_idx].append(v)
            depth_so_far += take
```
```
src/exo/worker/engines/mlx/cache.py:1774-1785
            ks = per_layer_keys[layer_idx]
            vs = per_layer_values[layer_idx]
            cache_entry = KVCache()
            if ks:
                concat_k = _concat_seq_axis(ks)
                concat_v = _concat_seq_axis(vs)
                # Detach to break MLX graph references; keeps the stored slice
                # safe from caller mutation (generation writes in place).
                cache_entry.keys = _detached_copy(concat_k)
                cache_entry.values = _detached_copy(concat_v)
                cache_entry.offset = int(cache_entry.keys.shape[2])
            new_cache.append(cache_entry)
```

### VERDICT: the sliceable-layer restore is **LAZY — it does NOT call `mx.eval`.**

- `_detached_copy` is an explicitly documented **near-free lazy COW alias**:
  ```
  src/exo/worker/engines/mlx/cache.py:150-160
  def _detached_copy(a: mx.array) -> mx.array:
      """Return an array that is safe to hold across in-place mutations of `a`.
      ... `mx.array(a)` is a near-free lazy alias that rides that same COW
      guarantee. Reserved for callers in `copy_rotating_kv_cache` that require
      fully-detached storage (numpy round-trip below) — the slice path used by
      the radix trie uses this fast alias."""
      return mx.array(a)
  ```
- `_concat_seq_axis` (`src/exo/worker/engines/mlx/cache.py:2096-2110`) is `mx.concatenate` — lazy.
- `_slice_seq_axis` (`src/exo/worker/engines/mlx/cache.py:2082-2093`) is array slicing — lazy.
- There is **no `mx.eval` anywhere in `_materialize_cache_to_depth`, in `get_kv_cache`, or in
  `_resolve_restore_position`** (`src/exo/worker/engines/mlx/cache.py:1685-1703`). Verified by enumerating every
  `mx.eval` occurrence in `cache.py`: lines **189, 253, 267, 293** — where `:253` is prose inside
  `_copy_pooling_cache`'s docstring, not a call. The three real calls are `:189` (inside `copy_rotating_kv_cache`),
  `:267-269` (inside `_copy_pooling_cache`), `:293` (inside `_detached_copy_or_none`). **None of the three is reached
  from the sliceable restore path**; the line ranges 1157-1360 (`get_kv_cache`) and 1705-1787
  (`_materialize_cache_to_depth`) contain zero `mx.eval` occurrences of any kind.
- Only `int(cache_entry.keys.shape[2])` at `src/exo/worker/engines/mlx/cache.py:1784` touches the array — and `.shape`
  is metadata, not a materialization.

**CONSEQUENCE (this is the brief's flagged TRAP, and it BITES): for a purely-sliceable model, the restore cost is
deferred into the first prefill `mx.eval`, i.e. it lands in `prefill_uncached`, NOT in the residual.** Chasing it as a
residual term would be chasing a phantom.

### The eager exception — snapshot layers

If the model has non-sliceable layers, `_materialize_cache_to_depth` takes a different branch:
```
src/exo/worker/engines/mlx/cache.py:1756-1763
            leaf_layer = donor_leaf.leaf_layer_caches[layer_idx]
            if leaf_layer is not None:
                snap_state = (
                    snapshot.states[layer_idx] if snapshot is not None else None
                )
                if snap_state is not None:
                    new_cache.append(deepcopy(snap_state))  # type: ignore[arg-type]
                    continue
```
This is a **`deepcopy` per non-sliceable layer, executed in Python, on the critical path**. `deepcopy` of a
`RotatingKVCache`/`ArraysCache`/`CacheList` is documented at `src/exo/worker/engines/mlx/cache.py:176-183` as copying
metadata + a `shared_ptr` (lazy in itself), but it is real Python object-graph work per layer.

Whether DSv4-Flash actually takes this branch on a live c=1 turn: `has_non_sliceable` is
`any(c is not None for c in donor_leaf.leaf_layer_caches)` (`src/exo/worker/engines/mlx/cache.py:1278-1279`), and
`leaf_layer_caches` is populated by `_extract_non_sliceable_layers` (`src/exo/worker/engines/mlx/cache.py:836`,
implementation `:2207-2219`) from `_sliceable_layer_mask` (`:2039`). The code strongly implies DSv4 DOES have such layers
— `src/exo/worker/engines/mlx/cache.py:1725-1726`: *"DSv4 hits this on EVERY partial hit: snapshot_ssm_states stores
None for trimmable CacheLists, which is what DSv4 layers use."* **NOT FULLY RESOLVED from source alone — I could not
determine the DSv4 layer-type composition without loading the model config (`config.json` is not present in
`/Users/adam.durham/.exo/models/deepseek-ai--DeepSeek-V4-Flash/`, which holds only `tokenizer.json` +
`tokenizer_config.json`). This is a Task-2 measurement question, not a code-read question.**

### Deep-copy / snapshot protection of the stored trie entry — YES, and it is EAGER at COMMIT time (not restore time)

The trie's protection against the live request mutating a stored entry lives on the **write** side:

- `_slice_layer_kv` (`src/exo/worker/engines/mlx/cache.py:2139-2161`) uses `_detached_copy` (lazy COW alias) — its
  docstring at `:2142-2143` claims "numpy-round-tripped", but the code at `:2156-2160` calls `_detached_copy`, **not**
  `_detached_copy_numpy`. **The docstring is stale relative to the code.**
- The genuinely eager, `mx.eval`-forcing copies are in the snapshot helpers:
  `copy_rotating_kv_cache` → `_detached_copy_numpy` + `mx.eval(k_slice, v_slice)`
  (`src/exo/worker/engines/mlx/cache.py:187-189`), and `_copy_pooling_cache` → three `_detached_copy_numpy` +
  `mx.eval(...)` (`src/exo/worker/engines/mlx/cache.py:260-269`). Both are documented as correctness fixes against
  MLX buffer donation (`src/exo/worker/engines/mlx/cache.py:225-254`). They run from `copy_snapshot_entry`
  (`:297-309`) / `snapshot_ssm_states` (`:312`), i.e. from `prefill`'s progress callback at
  `src/exo/worker/engines/mlx/generator/generate.py:859` — **inside prefill, not inside restore.**
- `_extract_non_sliceable_layers` `deepcopy`s per layer at commit (`src/exo/worker/engines/mlx/cache.py:2218`), but
  the extend path deliberately does **not** re-deepcopy (`src/exo/worker/engines/mlx/cache.py:1051-1058`, `:1101-1111`).

### Python-op count per restore, as a function of layer count `L`

Let `L = num_layers` (`src/exo/worker/engines/mlx/cache.py:1730`), `E` = trie edges on the path
(`_collect_path`, `src/exo/worker/engines/mlx/cache.py:2241-2251`), `S` = number of non-sliceable layers.

**It is a per-layer Python loop, NOT one buffer op.** Per restore:
- `2L` list allocations (`src/exo/worker/engines/mlx/cache.py:1732-1733`).
- The gather loop is `O(E × L)`: `src/exo/worker/engines/mlx/cache.py:1736-1751`, with 2 indexing ops + a `_has_tokens`
  call per (edge, layer), plus 2 `_slice_seq_axis` calls per (edge, layer) **only on the final partial edge**.
- The build loop is `O(L)`: `src/exo/worker/engines/mlx/cache.py:1755-1785` — per sliceable layer, 1 `KVCache()`
  construction, 2 `_concat_seq_axis` (each 1 `mx.concatenate` when `E>1`, `src/exo/worker/engines/mlx/cache.py:2100-2101`
  short-circuits when `E==1`), 2 `_detached_copy`, 1 `.shape` read.
- Plus `S` × `deepcopy` for non-sliceable layers (`src/exo/worker/engines/mlx/cache.py:1762`).

**Total ≈ `L × (2E + 6) + S × deepcopy` Python-level operations, all lazy except the `deepcopy`s.**

Additionally on the same restore path, `_pick_leaf_under` (`src/exo/worker/engines/mlx/cache.py:1675-1683`) is a
**linear scan of every leaf in the cache**, each with an `_is_ancestor` parent-chain walk
(`src/exo/worker/engines/mlx/cache.py:2232-2238`) — `O(leaves × depth)` in pure Python. Worth a Task-2 mark.

---

## (e) WHAT THE `done` / FINAL SSE EVENT WAITS ON

### VERDICT: the prefix-cache COMMIT is **NOT** on the critical path before `done`. It happens at SUBMIT time, before the first token.

Ordering, cited:

```
src/exo/worker/engines/mlx/generator/batch_generate.py:2586-2601
        with T("submit.save_prefix_cache"):
            if not is_bench:
                min_prefix_hit_length = max(
                    1000, system_prompt_token_count(task_params, self.tokenizer)
                )
                self._save_prefix_cache(
                    all_prompt_tokens,
                    list(cache),
                    cache_snapshots,
                    prefix_hit_length,
                    matched_index,
                    ...
                )
```

That block sits **inside `ExoBatchGenerator.submit()`** (which begins at
`src/exo/worker/engines/mlx/generator/batch_generate.py:2179`), immediately after prefill
(`:2531-2570`) and the rotating-cache clamp (`:2573-2584`), and **before** the sampler/logits-processor construction
(`:2607-2623`) and before `_active_tasks[uid]` is registered (`:2766-2784`). Decode has not started yet.
`_save_prefix_cache` itself is `src/exo/worker/engines/mlx/generator/batch_generate.py:5256-5309`, dispatching to
`update_kv_cache` (`:5289`) or `add_kv_cache` (`:5300`).

`submit()` is reached from `_start_task` (`src/exo/worker/runner/llm_inference/batch_generator.py:1280-1287`), called
from `BatchGenerator.step()`'s dispatch block (`src/exo/worker/runner/llm_inference/batch_generator.py:803-807`).
The first decode does not begin until `self._gen.step()` at
`src/exo/worker/runner/llm_inference/batch_generator.py:853`.

**Therefore: the commit cost (trie re-walk + edge build + any `deepcopy`) is paid BEFORE the first token, i.e. it is a
TTFT term, not a `done`-event term.** Reviewer expectation #5's escape clause ("unless the `done` event waits on
committing new KV into the trie, then it inherits #2") is **REFUTED as stated** — but the cost is real and simply
lands earlier than predicted, in the pre-first-token window that the study attributes to residual (since it is after
`server_received_ts` and outside `prefill_uncached`/`decode`).

### What `done` DOES wait on

1. The generator sets `finish_reason`, then builds `stats`/`usage`
   (`src/exo/worker/engines/mlx/generator/batch_generate.py:4549-4625`) and attaches them to the `GenerationResponse`
   (`:4630-4638`).
2. `BatchGenerator.step()` pushes it through the parser chain and emits `FinishedResponse`
   (`src/exo/worker/runner/llm_inference/batch_generator.py:912-940`).
3. `runner.py` sends `TaskStatus.Complete` and the terminal chunk
   (`src/exo/worker/runner/runner.py:720-724`, `send_chunk` at `:863-878`).
4. The chunk crosses mp.Queue → supervisor `_forward_events` (`src/exo/worker/runner/supervisor.py:402-434`) →
   `LOCAL_EVENTS` → master index (`src/exo/master/main.py:646-671`) → `GLOBAL_EVENTS` → API `_apply_state`
   (`src/exo/api/main.py:2460-2478`) → `_token_chunk_stream` (`src/exo/api/main.py:961-967`).
5. SSE: the final `data:` frame carries `usage` (`src/exo/api/adapters/chat_completions.py:293-297`), then the stats
   comment (`:300-301`), then `data: [DONE]` (`:302`), then `return` (`:303`).
6. Teardown: `finally:` sends `TaskFinished` and drops the queue (`src/exo/api/main.py:976-979`).

**There is no cache-commit, no snapshot copy, and no trie re-walk anywhere in steps 1-6.** The tail is: stop-detection
(already timed at `src/exo/worker/engines/mlx/generator/batch_generate.py:4522`), optional logprob extraction
(`:4530-4543`), stats/usage construction, and the same 4-hop event path every token already pays.

---

## (f) HOW `generation_tps` IS COMPUTED — and whether a slow first decode lands in the residual

### The computation site (non-PP-spec path, i.e. the c=1 TP path)

```
src/exo/worker/engines/mlx/generator/batch_generate.py:4567-4576
                else:
                    gen_time_delta = (
                        _mlx_gen_elapsed_seconds(self._mlx_gen)
                        - state.generation_time_at_start
                    )
                    generation_tps = (
                        state.completion_tokens / gen_time_delta
                        if gen_time_delta > 0
                        else 0.0
                    )
```
Written into `GenerationStats.generation_tps` at
`src/exo/worker/engines/mlx/generator/batch_generate.py:4597-4599`, field declared at
`src/exo/api/types/api.py:177-179`.

### What the interval actually is

`_mlx_gen_elapsed_seconds`:
```
src/exo/worker/engines/mlx/generator/batch_generate.py:471-484
def _mlx_gen_elapsed_seconds(mlx_gen: Any) -> float:
    """Best-effort cumulative generation time for an mlx-lm BatchGenerator.

    Older mlx-lm forks kept a ``_stats.generation_time`` counter. The new
    BatchGenerator tracks timing through a ``stats()`` context manager and
    exposes only a monotonic ``_steps_counter``. Fall through a known-good
    order; if nothing fits, use wall clock — tok/s stays meaningful.
    """
    stats = getattr(mlx_gen, "_stats", None)
    if stats is not None:
        gen_time = getattr(stats, "generation_time", None)
        if gen_time is not None:
            return float(gen_time)
    return time.perf_counter()
```

**The vendored mlx-lm `BatchGenerator` has NO `_stats` attribute.** `grep -c "_stats" mlx-lm/mlx_lm/generate.py` → **0**;
the class is `mlx-lm/mlx_lm/generate.py:1817` and its `__init__` sets only `_prompt_tokens_counter`,
`_prompt_time_counter`, `_gen_tokens_counter`, `_steps_counter`
(`mlx-lm/mlx_lm/generate.py:1869-1872`). Its `generation_time` accounting lives entirely inside the
`stats()` context manager (`mlx-lm/mlx_lm/generate.py:1894-1913`), which **exo never enters** — the only occurrences of
`_stats` in exo's engine tree are the doc comment and the `getattr` itself
(`src/exo/worker/engines/mlx/generator/batch_generate.py:474`, `:479`, `:2778`).

**Therefore `_mlx_gen_elapsed_seconds` ALWAYS falls through to the `return time.perf_counter()` fallback at
`src/exo/worker/engines/mlx/generator/batch_generate.py:484`.** So:

```
gen_time_delta = perf_counter()_at_done  −  generation_time_at_start
```
and `generation_time_at_start` was itself `_mlx_gen_elapsed_seconds(...)` = `perf_counter()` captured at task
registration, i.e. at the END of `submit()`:
`src/exo/worker/engines/mlx/generator/batch_generate.py:2782` (serial path; the same pattern at `:1257`, `:1554`,
`:2782`, `:3231`, `:3526`).

### VERDICT on (f)

`generation_tps` is **WALL-CLOCK from end-of-submit to done**, divided into `completion_tokens`. It therefore:

- **INCLUDES** the first decode step, any `mx.compile`/graph warm-up, the first Metal allocation after restore, and the
  lazily-deferred KV-restore evaluation that (d) showed is paid inside the first eval;
- **INCLUDES** all inter-step Python overhead (`agree_on_tasks`/`agree_on_cancellations_fast` collectives at
  `src/exo/worker/runner/llm_inference/batch_generator.py:682-724`, the `send_chunk` event emission, the
  runner-loop `get_nowait`);
- **EXCLUDES** everything before end-of-submit — which by construction is prefill, tokenization, template render,
  trie walk and restore.

So the study's `decode = completion_tokens / generation_tps` is **not** a steady-state rate that hides a slow first
step — the slow first step is already **inside** the study's `decode` term. Reviewer expectation #4 ("if it's
`tokens/mean_tps`, a slow first step lands in the residual") is **REFUTED**: graph warm-up and the deferred-restore eval
are absorbed by `decode`, not by the residual.

The only caveat: this is the wall-clock **fallback**, and the docstring at
`src/exo/worker/engines/mlx/generator/batch_generate.py:472-478` describes it as "best-effort". If a future mlx-lm bump
reintroduces `_stats.generation_time`, the semantics silently flip to a prefill-excluded steady-state rate and the
warm-up cost would migrate into the residual. **Worth a Task-2 assertion.**

(The PP-spec branch at `src/exo/worker/engines/mlx/generator/batch_generate.py:4550-4564` uses
`time.perf_counter() - state.generation_start_time`, which is the same wall-clock semantics by a different route.
`generation_start_time` is set at `src/exo/worker/engines/mlx/generator/batch_generate.py:2776`. That branch is gated on
`was_pp_spec_step` (`:4192`), i.e. PP only — not the TP c=1 path.)

---

## Task 2 mark sites

`perf_counter` is `mach_absolute_time`-based: comparable across the API process and the local runner process on ONE
host; **not** comparable to node 2's runner. That constraint is real here because master/worker/API share a process
(`src/exo/main.py:57-122`) while the runner is a separate `AsyncProcess`
(`src/exo/worker/runner/supervisor.py:297-306`).

### API process

| wishlist mark | natural boundary? | file:line |
|---|---|---|
| `recv_headers` | **NO — DOES NOT EXIST.** hypercorn owns this; the first exo-owned frame is the already-validated handler. Would need an ASGI middleware or a hypercorn hook. | server config at `src/exo/api/main.py:2425-2441`; first exo frame `src/exo/api/main.py:1129` |
| `body_read_done` | **NO — DOES NOT EXIST.** Same reason; FastAPI reads the body before dispatch. | — (`src/exo/api/main.py:1129-1131` is already post-read) |
| `json_validated` | **YES** — handler entry is by definition post-parse, post-validate. | `src/exo/api/main.py:1129` (entry) or `:1133` (first statement) |
| `task_dispatched` | **YES** | `src/exo/api/main.py:1087` (`await self._send(command)` returns) or the call site `:1137` |
| `first_token_from_runner` | **YES** (needs a first-flag) | `src/exo/api/main.py:2476` (`await queue.send(event.chunk)`) or `src/exo/api/main.py:962-963` |
| `first_sse_written` | **YES** (needs a first-flag) | `src/exo/api/adapters/chat_completions.py:297` |
| `last_sse_written` | **YES** | `src/exo/api/adapters/chat_completions.py:302` (`yield "data: [DONE]"`) |
| `stream_closed` | **YES** | `src/exo/api/main.py:976-979` (`finally:` of `_token_chunk_stream`) |

**Two of eight do not exist** (`recv_headers`, `body_read_done`). A cheap partial substitute for both: a single ASGI
middleware mark at request entry, giving `recv_headers`-ish; `body_read_done` is genuinely unavailable without a
custom `Request` body reader.

### Runner (tag by node / `device_rank`, `src/exo/worker/runner/runner.py:125`)

| wishlist mark | natural boundary? | file:line |
|---|---|---|
| `task_received` | **YES** — this is the study's join line | `src/exo/worker/runner/runner.py:563` |
| `template_rendered` | **YES** — existing `T()` span | `src/exo/worker/runner/llm_inference/batch_generator.py:1158-1159` |
| `tokenized` | **YES** — existing `T()` span | `src/exo/worker/engines/mlx/generator/batch_generate.py:2194-2198` |
| `trie_matched` | **PARTIAL.** An existing outer span covers lookup+restore together (`src/exo/worker/engines/mlx/generator/batch_generate.py:2386-2392`). Separating the walk needs a mark inside `cache.py`. No restructure required — just two marks. | walk returns at `src/exo/worker/engines/mlx/cache.py:1190-1192` |
| `kv_restored` | **PARTIAL**, same span as above. | `src/exo/worker/engines/mlx/cache.py:1315-1320` (`_materialize_cache_to_depth` returns) |
| `prefill_start` | **YES** — existing `T()` span opens here | `src/exo/worker/engines/mlx/generator/batch_generate.py:2531` |
| `prefill_done` | **YES** — same span closes after the `prefill(...)` call | `src/exo/worker/engines/mlx/generator/batch_generate.py:2559-2570` (also `generate.py:975-980` computes `elapsed` already) |
| `first_decode_done` | **NO NATURAL BOUNDARY.** `self._gen.step()` (`src/exo/worker/runner/llm_inference/batch_generator.py:853`) → `self._mlx_gen.next()` (`src/exo/worker/engines/mlx/generator/batch_generate.py:4218`) returns a *batch* of responses with no per-uid "is this your first" flag. Needs a per-`_EngineTask` boolean. **Small addition, not a restructure.** | closest existing: `src/exo/worker/engines/mlx/generator/batch_generate.py:4218-4220` (`_next_elapsed` already computed) |
| `first_token_emitted` | **YES** (needs a first-flag) | `src/exo/worker/runner/llm_inference/batch_generator.py:915-916`, or the emit at `src/exo/worker/runner/runner.py:724` |
| `last_token` | **YES** | `src/exo/worker/engines/mlx/generator/batch_generate.py:4524` (`is_done = finish_reason is not None`); runner-side `src/exo/worker/runner/runner.py:720-722` |
| `stop_detected` | **YES** — already `perf_counter`-timed | `src/exo/worker/engines/mlx/generator/batch_generate.py:4522` (`_t_stop_total += ...`) |
| `cache_committed` | **YES**, but **it is at SUBMIT time, not after last token** — see (e). Label it honestly or the phase order will look wrong. | `src/exo/worker/engines/mlx/generator/batch_generate.py:2586-2601` |

**One of twelve does not exist** (`first_decode_done`); **two of twelve are partial** (`trie_matched`, `kv_restored` —
they share one existing span and need two new marks inside `cache.py`); **one is misplaced relative to the wishlist's
implied ordering** (`cache_committed`).

### Where the `usage` / `stats` dict is built (marks must ship inside it)

```
src/exo/worker/engines/mlx/generator/batch_generate.py:4597-4613   GenerationStats(...)
src/exo/worker/engines/mlx/generator/batch_generate.py:4615-4625   Usage(...)
src/exo/worker/engines/mlx/generator/batch_generate.py:4630-4638   attached to GenerationResponse(stats=..., usage=...)
```

Type definitions: `GenerationStats` at `src/exo/api/types/api.py:177-189`; carried on the chunk at
`src/exo/shared/types/chunks.py:29` (`TokenChunk.stats`) and `:52` (`ToolCallChunk.stats`), `usage` at
`src/exo/shared/types/chunks.py:27` / `:50`.

Wire-out to the client (both must be considered — 44/55 of the user's real requests end in `tool_calls`):
- **TokenChunk finish:** `usage` merged into the last data frame at
  `src/exo/api/adapters/chat_completions.py:293-297`; `stats` emitted as an SSE **comment** at
  `src/exo/api/adapters/chat_completions.py:300-301`.
- **ToolCallChunk finish:** `usage` at `src/exo/api/adapters/chat_completions.py:281-283`; `stats` comment at
  `src/exo/api/adapters/chat_completions.py:284-285`.

**Recommendation:** add the marks dict as a new optional field on `GenerationStats`
(`src/exo/api/types/api.py:177-189`) so it rides the existing `: generation_stats {...}` SSE comment on BOTH the token
and tool-call finishes without touching the OpenAI-shaped `usage` object — the study's existing client-wall join then
works unchanged. Note `GenerationStats` already carries optional-with-default fields
(`src/exo/api/types/api.py:183-189`), so an additive field is consistent with existing practice and backward-compatible
with a client that ignores it.

---

## Confidence and gaps

**HIGH confidence (direct code, single unambiguous path):**
- (a) the stamp location and the before/after ordering.
- (b) list membership; in particular `src/exo/worker/main.py:195` being a real, live, unfixed 100 ms tick.
- (c) that the full prompt is re-tokenized and the template fully re-rendered every turn, with no tokenization cache.
- (d) that the sliceable restore path contains **no** `mx.eval`.
- (e) that cache commit is at submit time, before the first token.
- (f) that `_mlx_gen_elapsed_seconds` falls through to wall clock because the vendored mlx-lm `BatchGenerator` has no
  `_stats` (verified by a zero-hit grep on the submodule).

**MEDIUM confidence / stated assumptions:**
- The live path is TP + `BatchGenerator` + serial (non-batched-decode) submit. Derived from `start_cluster.sh:401`,
  `:129`, `:145`, `:2496` and the absence of `EXO_NO_BATCH`. **If the cluster is actually running Pipeline, sections
  (d)/(e)/(f) shift branches** — specifically `_submit_batched_decode_deferred`
  (`src/exo/worker/engines/mlx/generator/batch_generate.py:1263-1318`) skips the KVPrefixCache entirely
  (`:2318-2324`), and (f) would use the PP-spec branch (`:4550-4564`).
- Whether the elected master is node 1 or node 2 — this changes the gossip hop count per request. Determined at
  runtime by election (`src/exo/shared/election.py`), not readable from source.

**COULD NOT DETERMINE (explicitly not guessed):**
1. **DSv4-Flash's layer-type composition** (`L`, and how many layers are non-sliceable / snapshot-restored). Needed to
   turn (d)'s `L × (2E + 6) + S × deepcopy` into a number, and to decide whether the eager `deepcopy` branch at
   `src/exo/worker/engines/mlx/cache.py:1762` fires on a real turn. **NOT FOUND, searched:**
   `/Users/adam.durham/.exo/models/deepseek-ai--DeepSeek-V4-Flash/` (contains only `tokenizer.json`,
   `tokenizer_config.json`), `/Users/adam.durham/.exo/models/mlx-community--DeepSeek-V4-Flash-4bit/`,
   `src/exo/shared/models/model_cards.py` (no `V4-Flash` literal).
2. **The real cost of the gossip round trips.** Structurally present and cited
   (`src/exo/routing/topics.py:33-34,40-42`, `src/exo/routing/router.py:59-60,92-96,221-229`) but not measured;
   serialization size and cross-node hop count are runtime facts.
3. **Whether `_pick_leaf_under`'s `O(leaves × depth)` scan (`src/exo/worker/engines/mlx/cache.py:1675-1683`) is
   material.** It is unambiguously a full Python scan of every leaf on every cache hit, but the live leaf count is a
   runtime quantity.
4. **Whether the second, diagnostics-only trie walk in `add_kv_cache`
   (`src/exo/worker/engines/mlx/cache.py:823-825`) is material.** It is a confirmed duplicate walk feeding only a log
   line (`:861-868`); its cost depends on trie shape.

**Discrepancy found while reading (reported, not fixed — this was a read-only task):**
`_slice_layer_kv`'s docstring at `src/exo/worker/engines/mlx/cache.py:2142-2143` says the slices are
"detached (numpy-round-tripped) copies", but the implementation at `src/exo/worker/engines/mlx/cache.py:2156-2160`
calls `_detached_copy` (the lazy COW alias), **not** `_detached_copy_numpy`. The docstring is stale. This does not
change any verdict above (it reinforces (d): the trie write path is also lazy for sliceable layers) but it is a live
trap for anyone reading that function in isolation, and `_copy_pooling_cache`'s docstring
(`src/exo/worker/engines/mlx/cache.py:225-254`) documents exactly the class of correctness bug that a COW alias caused
before.
