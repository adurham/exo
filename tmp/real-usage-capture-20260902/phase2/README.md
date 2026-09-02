# passive_capture_proxy — client-side latency truth for Hermes-on-exo

A passive, non-buffering reverse proxy that sits between the Hermes client and
your exo OpenAI-compatible endpoint and **measures, purely as a side effect**, the
real per-request timing a live session experiences. It answers the benchmark-vs-
perception gap: the server benchmark reports decode-only tokens/sec, but you
*feel* a slower rate because your wall clock includes TTFT on very large prompts,
tool-call round trips, and streaming overhead. This tool captures both rate
conventions side by side so the difference is visible directly.

## One-line start

```bash
python3 /Users/adam.durham/repos/exo/tmp/real-usage-capture-20260902/phase2/passive_capture_proxy.py
```

Then, in a second terminal, point Hermes at the proxy and use it normally:

```bash
hermes config set providers.exo.base_url http://127.0.0.1:52416/v1
```

When the capture session is over, restore the original endpoint:

```bash
hermes config set providers.exo.base_url http://192.168.86.201:52415/v1
```

*(These are the exact two values; the proxy defaults to listening on
`127.0.0.1:52416` and forwarding to your current exo endpoint
`http://192.168.86.201:52415`.)*

Options: `--port` (default `52416`), `--upstream` (default
`http://192.168.86.201:52415`), `--jsonl` (default `./capture.jsonl` next to this
script), `--listen` (default `127.0.0.1`). The tool reads your
`~/.hermes/config.yaml` **read-only** — it never edits it.

## Where the data lands

Every request that flows through the proxy appends exactly one JSON line to
`capture.jsonl` in this directory. Stop the proxy with `Ctrl-C` whenever you
like; there is no setup/teardown beyond that.

## Non-buffering + fail-open guarantees

- **Streaming is never buffered.** Each upstream socket read is flushed to the
  client *before* any parsing/measurement happens. The proxy body relay uses
  `HTTPResponse.read1()` (single-chunk reads) so a streamed body is forwarded in
  one-chunk-at-a-time granularity — never accumulated into an `amt`-sized window.
  If it buffered even a small amount, TTFT measurement would be corrupted and
  your real session would feel slower. It does neither.
- **Fail-open measurement.** Every measurement/parse/JSONL-write step is wrapped
  in try/except and any failure is recorded in the `capture_errors` / `relay_errors`
  fields of that request's line. A crash in the capture path can never break the
  session; the bytes are still relayed.
- **Passive.** Response bytes are relayed byte-for-byte unmodified.

## Dependencies

**Standard library only** (Python ≥ 3.9). Zero pip/uv installs.

## Field dictionary

Identity / wall clock:

| field | meaning |
|---|---|
| `request_id` | per-request uuid (12 hex chars) |
| `ts_start_epoch` / `ts_end_epoch` | wall-clock (epoch) timestamps of request start / relay end |
| `wall_duration_s` | total client-visible duration (time between start and last byte) |
| `method` / `path` | the proxied HTTP method and request path (e.g. `POST /v1/chat/completions`) |
| `status` | HTTP status returned by upstream |
| `model` / `stream` | model id and `stream` flag from the request body |

Streaming timing:

| field | meaning |
|---|---|
| `streaming` | true if the response was an SSE `text/event-stream` |
| `ttft_s` | seconds from request sent to the **first SSE chunk carrying actual content** (the real time-to-first-token as seen by the client) |
| `ts_first_content_epoch` | wall-clock epoch of that first content chunk |

Tokens — two separate, clearly named counts:

| field | meaning |
|---|---|
| `completion_tokens_streamed` | count of SSE chunks whose `delta.content` was non-empty — i.e. the tokens the client actually *streamed in* (client-perceived truth) |
| `completion_tokens_usage` | server-reported `usage.completion_tokens`, if the response carried a usage block |
| `prompt_tokens` / `total_tokens` | server-reported prompt / total token counts, if present |
| `cached_tokens` / `prompt_tokens_details` | server-reported prefix-cache hit tokens, if reported |
| `n_sse_events` | total `data:` events parsed (content + finish + usage + others) |

Streaming stall visibility:

| field | meaning |
|---|---|
| `inter_chunk_gaps_s` | full list of per-pair arrival gaps (s) between consecutive content chunks — a stall is a big outlier here |
| `inter_chunk_gap_summary` | `{mean_s, median_s, max_s, min_s, count}` summarising the gaps |

Rate conventions (the core of the benchmark-vs-perception question), both
separately named:

| field | meaning |
|---|---|
| `post_ttft_rate_toks_per_s` | `completion_tokens_streamed / (wall − TTFT)` — decode-throughput-style, excludes the initial wait |
| `full_wall_rate_toks_per_s` | `completion_tokens_streamed / wall` — everything you actually experience |

Completion semantics:

| field | meaning |
|---|---|
| `finish_reason` | e.g. `stop` / `tool_calls` / `length`, from the stream |
| `has_tool_calls` | whether any tool-call delta appeared in the stream |
| `n_tool_calls` / `tool_call_names` | count and function names of the tool calls |
| `ended_in_tool_call` | `true` when the turn finished at a tool call (finish_reason `tool_calls` or tool-call deltas present) — the harness uses this to reconstruct tool-call round-trip gaps between consecutive requests |

Diagnostics:

| field | meaning |
|---|---|
| `capture_errors` | any errors caught in the measurement path (should stay `[]`; fail-open means they're non-fatal) |
| `relay_errors` | any errors from relaying to/from the client (e.g. client disconnected) |

## Self-test

```bash
python3 /Users/adam.durham/repos/exo/tmp/real-usage-capture-20260902/phase2/self_test.py
```

Starts a local fake OpenAI-compatible SSE server (stdlib `http.server`) that emits
a realistic chat.completions stream — a deliberate 0.5 s pause before the first
content chunk (simulated TTFT), 12 content chunks with ~60 ms gaps, then a usage
block and `[DONE]` — drives a request through the proxy, and asserts the captured
JSONL has TTFT within tolerance of the injected delay, streamed token count equal
to the number of content chunks, and full-wall rate lower than post-TTFT rate. A
sample of its actual output is in `capture.jsonl` in this directory.
