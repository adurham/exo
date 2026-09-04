# Round-11 Phase Marks — implementation notes

## Out-of-scope marks (explicitly, per PM's fixed mark set)

`recv_headers` and `body_read_done` are NOT implemented. CODE-READ.md
established that hypercorn is the ASGI server and body-receive + JSON
parse + pydantic validation all happen inside hypercorn/FastAPI machinery
BEFORE the `chat_completions` handler is ever called (see CODE-READ.md
section (a), item 1-3, citing `src/exo/api/main.py:20-22`, `:2425-2441`,
`:451-453`, `:1129-1131`). Timing those two sub-phases would require
hypercorn-level ASGI middleware lifecycle hooks (`http.request` /
`http.disconnect` events, or a custom `Config`/protocol wrapper) — a
different, heavier instrumentation surface than a `perf_counter()` mark at
an existing Python statement boundary. Out of scope this round.

`a1 handler_entered` is therefore honestly labelled **post-validation**: it
is the first mark taken inside `chat_completions()`, after FastAPI has
already parsed the body and validated it into a `ChatCompletionRequest`.

## Mark-to-code-site map (for the PM's independent verification)

### Group A — API process (`src/exo/api/`)

| mark | field name | site |
|---|---|---|
| a1 handler_entered | (recorder start, not itself a delta) | `main.py::chat_completions`, first statement |
| a2 messages_serialized | `messages_serialized_ms` | after `chat_request_to_text_generation(payload)` |
| a3 command_published | `command_published_ms` | after `_send_text_generation_with_images(task_params)` returns |
| a4 first_chunk_received | `first_chunk_received_ms` | `generate_chat_stream()`, top of the `async for chunk in chunk_stream:` loop, fires once on the FIRST chunk received (before the `match` dispatch, so it fires even if the first chunk is a `PrefillProgressChunk`) |
| a5 first_sse_written | `first_sse_written_ms` | `generate_chat_stream()`, first data-carrying SSE frame (both TokenChunk and ToolCallChunk branches) |
| a6 last_sse_written | `last_sse_written_ms` | `generate_chat_stream()`, last data-carrying frame on the finish branch (both TokenChunk and ToolCallChunk) |
| a7 stream_closed | `stream_closed_ms` | `generate_chat_stream()`, immediately before the `[DONE]` frame + return, both finish branches |

All four of a4-a7 are populated fields in `api_phase_marks_ms`; the
`dispatch_and_ipc_gap` formula in `analyze_marks.py` has real data to work
with.

### Group B — Runner process (`src/exo/worker/runner/` + `src/exo/worker/engines/mlx/`)

| mark | field name | site |
|---|---|---|
| b1 task_received | (recorder `begin()`, not itself a delta) | `runner.py::handle_generation_tasks`, first statement, coincident with the `"received chat request"` log line (runner.py:563 pre-instrumentation) |
| b2 template_rendered | `template_rendered_ms` | `batch_generator.py::_start_task`, after `apply_chat_template(...)` |
| b3 tokenized | `tokenized_ms` | `batch_generate.py::submit`, after `encode_prompt` + `fix_unmatched_think_end_tokens` |
| b4 trie_matched | `trie_matched_ms` | `cache.py::get_kv_cache`, after `_pick_leaf_under(match_node)` (covers both `_longest_prefix_match` and the O(leaves×depth) `_pick_leaf_under` scan CODE-READ flagged) |
| b5 kv_restored | `kv_restored_lazy_no_eval_ms` | `cache.py::get_kv_cache`, after `_materialize_cache_to_depth(...)` returns. Named `lazy_no_eval` per PM's mandatory labeling rule — CODE-READ proved no `mx.eval` on this path |
| b6 prefill_start | `prefill_start_ms` | `batch_generate.py::submit`, immediately before `with vision_ctx, T("submit.prefill"):` |
| b7 prefill_done | `prefill_done_ms` | `batch_generate.py::submit`, immediately after the prefill block (both remote and local prefill arms) |
| b8 save_prefix_cache_done | `cache_commit_pre_first_token_ms` | `batch_generate.py::submit`, after the `with T("submit.save_prefix_cache"):` block. Named to make pre-first-token ordering explicit per PM instruction |
| b9 first_token_emitted | `first_token_emitted_ms` | `batch_generate.py::step`, in the per-response loop, fires once when `state.completion_tokens == 0` |
| b10 last_token | `last_token_ms` | `batch_generate.py::step`, same loop, fires every decode iteration (final value in the snapshot is the LAST step's delta) |
| b11 stop_detected | `stop_detected_ms` | `batch_generate.py::step`, right after `is_done = finish_reason is not None` evaluates True |

Runner marks are attached to `GenerationStats.phase_marks_ms` at the same
site the pre-existing `stats = GenerationStats(...)` object is built
(`batch_generate.py`, `is_done` branch), via
`runner_phase_marks.snapshot_and_clear()`.

## G6 verification (both SSE finish paths carry marks)

Verified live (not just by inspection) via a stdlib-only async harness that
drives `generate_chat_stream()` directly against synthetic `ToolCallChunk`
and `TokenChunk` streams with `EXO_PHASE_MARKS=1`:

```
tool_calls OK -> : generation_stats {...,"api_phase_marks_ms":{...}}
token OK -> : generation_stats {...,"api_phase_marks_ms":{...}}
```

Both branches emit `api_phase_marks_ms` inside the existing
`: generation_stats {...}` SSE comment. `chunk.stats` (the runner-attached
`phase_marks_ms`) rides through unchanged on both branches since it was
never touched by the API-side edit — it flows straight from
`ToolCallChunk.stats` / `TokenChunk.stats` into the same
`stats_to_emit.model_copy(update={"api_phase_marks_ms": api_marks})` call
on both branches.

## Design note: why `first_sse_written`/`last_sse_written`/`stream_closed`
share a code path between TokenChunk and ToolCallChunk branches

The three marks are placed at structurally symmetric points in both `match`
arms (first data frame, last data frame, right before `[DONE]`) rather than
factored into a shared helper, to keep each arm's mark-insertion diff
minimal and easy to audit against CODE-READ's citations line-by-line. A
future refactor could extract a `_finish_stream(command_id, stats)` helper
if this instrumentation becomes permanent.
