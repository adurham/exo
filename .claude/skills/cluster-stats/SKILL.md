---
name: cluster-stats
description: Analyze exo cluster and opencode logs for performance stats, KV cache hit rates, TPS, request flow, and issues. Run after an opencode session to evaluate effectiveness of the dual-model setup.
argument-hint: [latest|all]
---

# Exo Cluster Stats Analysis

Gather and analyze logs from all three sources — local opencode logs and remote exo logs on the cluster nodes — then produce a performance report.

## Cluster Nodes

- **macstudio-m4-1** (128GB) — MiniMax-M2.5-6bit (229B MoE) TP worker, rank 1 (API gateway at 192.168.86.201:52415)
- **macstudio-m4-2** (128GB) — MiniMax-M2.5-6bit (229B MoE) TP worker, rank 0
- **macbook-m4** (36GB) — Qwen3-Coder-30B-A3B-Instruct-5bit (single node)

## Current Known State

Before making recommendations, verify these against the actual logs — do NOT recommend something that's already in place:

- **KV cache entries**: 1 on Studios (MiniMax), 2 on MacBook (Qwen)
- **Opencode explore agent steps**: 10
- **Qwen context limit**: 50K tokens, output limit 8192
- **MiniMax context limit**: 120K tokens
- **Tracing**: `EXO_TRACING_ENABLED=true` is active on all nodes — tracing output goes to stderr (mixed into ~/exo.log)
- **Known volatile cache issue**: title-agent vs explore-subagent system prompts diverge at token 6 on Qwen — every first explore starts cold
- **Known cross-subagent miss**: different explore sessions diverge at ~token 4774 (different task content), so cache entries don't help across sessions

## Data Collection Steps

### 1. Opencode Logs

Find the most recent opencode log files in `~/.local/share/opencode/log/`. If "$ARGUMENTS" is "latest", use only the most recent log file. Otherwise use all logs from today.

Extract from these logs:
- **LLM calls**: grep for `service=llm` lines — note the `providerID`, `modelID`, `agent`, `mode` (primary vs subagent)
- **Session flow**: grep for `service=session.prompt` lines — note `step=N`, `sessionID`, `loop`/`exiting loop`/`cancel`
- **Subagent sessions**: look for `parentID=` in session creation lines
- **Timing**: compute wall-clock time between steps from timestamps

### 2. Exo Logs (Remote)

SSH to each node and read `~/exo.log`. **CRITICAL**: exo logs contain full request bodies (including system prompts and prior conversation) embedded in `Starting task TextGeneration(...)` lines. These are **enormous** (often 100KB+) and will false-match on ANY keyword (KV cache, TPS, prefill, etc.). Simple `grep -v` is NOT sufficient because the request body is all on one line.

**Correct filtering approach** — use Python to filter by line length, or use specific log source patterns:

```bash
# Option A: Python filter — only lines < 1000 chars (real metrics are short)
ssh <node> 'python3 -c "
import sys
for line in open(\"/Users/adam.durham/exo.log\"):
    if len(line) < 1000 and \"PATTERN\" in line:
        print(line.rstrip()[:300])
"'

# Option B: Match on logger source (not content) to avoid request body lines
# Real metric lines come from these loggers:
#   exo.worker.engines.mlx.generator.batch_generate:submit  — KV cache hit
#   exo.worker.engines.mlx.generator.generate:prefill       — prefill stats
#   exo.worker.engines.mlx.cache:update_kv_cache            — cache updates
#   exo.worker.engines.mlx.cache:add_kv_cache               — cache adds
#   exo.worker.engines.mlx.cache:_evict_if_needed           — evictions
#   exo.worker.engines.mlx.cache:get_prefix_length          — VOLATILE warnings
#   exo.master.api:_token_chunk_stream                      — request summaries
#   exo.worker.runner.llm_inference.model_output_parsers     — tool call parsing
# Tracing lines from mlx-lm go to stderr with [generate_step] or [decode] prefix

ssh <node> 'grep -E "batch_generate:submit|update_kv_cache|add_kv_cache|_evict_if_needed|get_prefix_length|_token_chunk_stream|parse_tool_calls|\\[generate_step\\]|\\[decode\\]|prefill:2|prefill:3|warmed up" ~/exo.log'
```

#### On macstudio-m4-1 and macstudio-m4-2 (MiniMax):

Extract (use logger source patterns from above, NOT content grep):
- **KV cache hits**: `batch_generate:submit` → `KV cache hit: X/Y tokens cached (Z%)`
- **KV cache updates**: `update_kv_cache` → `KV cache updated (index N): X tokens`
- **KV cache adds**: `add_kv_cache` → `KV cache added: X tokens`
- **KV cache evictions**: `_evict_if_needed` → `KV cache evicted entry N`
- **Prefill TPS**: `prefill:368` → `Prefill complete: N tokens in Xs (Z tok/s)`
- **Prefill chunks**: `[generate_step] prefill chunk: N tokens in Xms` (stderr)
- **Decode latency**: `[decode] n=N X.Xms` (stderr, per-token)
- **TTFT**: `[generate_step] first eval (TTFT barrier): Xms` (stderr)
- **Request summary**: `_token_chunk_stream` → `chunk_stream: cmd=... prefill=X tok/s decode=Y tok/s prompt=N gen=M mem=XGB`
- **Tool call parsing**: `parse_tool_calls` → `parsed tool_call_text_parts`
- **Warmup**: `warmed up by generating N tokens`

#### On macbook-m4 (Qwen):

Same as above, plus:
- **VOLATILE PATTERN DETECTED** warnings (`get_prefix_length`) — KV prefix cache busting from system prompt changes
- **Cache evictions** caused by volatile patterns
- **Model reload cycles**: `Shutdown` → `LoadModel` → `StartWarmup` after volatile detection

### 3. Tracing / TPS Data

`EXO_TRACING_ENABLED=true` is set in `start_cluster.sh` and IS active on all nodes. Tracing output goes to **stderr** (mixed into `~/exo.log`). The tracing lines come from two places:

#### mlx-lm `generate_step` (stderr, all nodes):
- `[generate_step] prefill start: N tokens` — marks beginning of prefill
- `[generate_step] prefill chunk: N tokens in Xms` — per-chunk prefill timing (chunk_size typically 512)
- `[generate_step] first _step graph built in Xms` — graph compilation after prefill
- `[generate_step] first eval (TTFT barrier): Xms` — time-to-first-token barrier
- `[decode] n=N X.Xms` — per-token decode latency (every token!)

#### exo pipeline prefill (logger, TP nodes only):
- `Prefill progress: X/Y tokens (Z tok/s)` — running prefill throughput
- `Prefill complete: N tokens in Xs (Z tok/s)` — final prefill summary with TPS
- `[R{rank}] Prefill post-loop: Xms (cache eval: Xms)` — post-prefill overhead

#### exo API chunk_stream (logger, API gateway node — macstudio-m4-1):
- `chunk_stream: cmd=UUID prefill=X tok/s decode=Y tok/s prompt=N gen=M mem=XGB` — end-to-end summary per request

Use these to compute:
- **Prefill TPS**: from `Prefill complete` lines or `[generate_step] prefill chunk` math
- **Decode TPS**: from `[decode] n=N` latencies (1000/ms = tok/s) or `chunk_stream` summary
- **TTFT**: from `first eval (TTFT barrier)` lines

## Report Format

Produce a report with these sections:

### Session Overview
- Total wall-clock time
- Number of agentic steps (main agent + subagents)
- Models used and their roles

### Request Flow Table
| Step | Time | Model | Agent | Prompt Tokens | Duration | Notes |
|------|------|-------|-------|--------------|----------|-------|

### MiniMax Performance (TP on Studios)
- KV cache hit rates per step (table)
- Context growth over session
- Any cache misses > 10% and why
- Estimated prefill TPS if data available

### Qwen Performance (MacBook)
- KV cache hit rates
- Volatile pattern warnings (count and impact)
- Cache evictions and rebuilds

### Issues Found
- Volatile cache patterns
- Slow steps (> 30s)
- Context growth concerns
- Model contention (MiniMax and Qwen requests overlapping)
- Compaction or streaming errors

### Recommendations
- Actionable suggestions based on findings
- **CRITICAL: Before listing any recommendation, verify it against the "Current Known State" section above AND the actual log data. Do NOT recommend changes that are already implemented. Only recommend genuinely new/unresolved issues.**
