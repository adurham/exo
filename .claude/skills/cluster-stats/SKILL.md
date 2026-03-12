---
name: cluster-stats
description: Analyze exo cluster and opencode logs for performance stats, KV cache hit rates, TPS, request flow, and issues. Run after an opencode session to evaluate effectiveness of the dual-model setup.
argument-hint: [latest|all]
---

# Exo Cluster Stats Analysis

Gather and analyze logs from all three sources — local opencode logs and remote exo logs on the cluster nodes — then produce a performance report.

## Cluster Nodes

- **macstudio-m4-1** (128GB) — MiniMax-M2.5-6bit (229B MoE) TP worker, rank 1
- **macstudio-m4-2** (128GB) — MiniMax-M2.5-6bit (229B MoE) TP worker, rank 0 (master node, API at 192.168.86.201:52415)
- **macbook-m4** (36GB) — Qwen3.5-9B-8bit (single node)

## Data Collection Steps

### 1. Opencode Logs

Find the most recent opencode log files in `~/.local/share/opencode/log/`. If "$ARGUMENTS" is "latest", use only the most recent log file. Otherwise use all logs from today.

Extract from these logs:
- **LLM calls**: grep for `service=llm` lines — note the `providerID`, `modelID`, `agent`, `mode` (primary vs subagent)
- **Session flow**: grep for `service=session.prompt` lines — note `step=N`, `sessionID`, `loop`/`exiting loop`/`cancel`
- **Subagent sessions**: look for `parentID=` in session creation lines
- **Timing**: compute wall-clock time between steps from timestamps

### 2. Exo Logs (Remote)

SSH to each node and read `~/exo.log`. **CRITICAL**: the exo logs contain full request bodies (including system prompts) that will match on keywords like "TPS", "KV cache", etc. Always filter these out first:

```bash
# Filter out request body noise, then search for real metrics
ssh <node> 'grep -v "TextGeneration\|TextGenerationTaskParams\|InputMessage\|instructions=" ~/exo.log | grep -iE "<pattern>"'
```

#### On macstudio-m4-1 and macstudio-m4-2 (MiniMax):

Extract:
- **KV cache hits**: `KV cache hit: X/Y tokens cached (Z%)`
- **KV cache updates**: `KV cache updated (index N): X tokens`
- **KV cache misses/evictions**: `KV cache evicted`
- **Prefill times**: time between `Starting task TextGeneration` and first `KV cache updated`
- **Tool call parsing**: `parsed tool_call_text_parts`
- **Model load time**: `Time taken to shard and load model`
- **Warmup**: `warmed up by generating N tokens`
- **Context clamping**: `Clamping max_tokens`

#### On macbook-m4 (Qwen):

Same as above, plus:
- **VOLATILE PATTERN DETECTED** warnings — these indicate KV prefix cache busting from system prompt changes
- **Cache evictions** caused by volatile patterns
- **Cache rebuilds** after eviction

### 3. Check for TPS Logging

If `EXO_TRACING_ENABLED=true` is set on nodes, look for prefill/decode TPS stats. If `false`, note this gap and estimate TPS from timing:
- Prefill TPS = (new tokens prefilled) / (time between cache hit log and cache updated log)
- Decode TPS = rough estimate from (output tokens) / (time between steps in opencode log)

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
- Missing telemetry (e.g., TPS not logged)
- Context growth concerns
- Model contention (MiniMax and Qwen requests overlapping)

### Recommendations
- Actionable suggestions based on findings
