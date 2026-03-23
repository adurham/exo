"""Stress test for multi-turn tool call generation on the exo cluster.

Simulates the opencode agentic pattern:
  1. User asks a question with tools defined
  2. Model calls a tool (e.g. explore subagent)
  3. Tool result is sent back
  4. Model either calls another tool or produces final answer

Measures tok/s, TTFT, and detects deadlocks at each step.

Usage:
  uv run python bench/stress_tool_calls.py [--turns N] [--api URL] [--model MODEL]
"""

import argparse
import json
import time
import sys
import urllib.request
from dataclasses import dataclass

API = "http://192.168.86.201:52415"
MODEL_397B = "mlx-community/Qwen3.5-397B-A17B-4bit"
MODEL_35B = "mlx-community/Qwen3.5-35B-A3B-4bit"

TOOLS = [
    {"type": "function", "function": {"name": "task", "description": "Dispatch a task to a subagent (explore, code, or general)", "parameters": {"type": "object", "properties": {"agent": {"type": "string", "enum": ["explore", "code", "general"], "description": "Which subagent to use"}, "description": {"type": "string", "description": "What to search for or do"}}, "required": ["agent", "description"]}}},
    {"type": "function", "function": {"name": "read", "description": "Read a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "glob", "description": "Search for files by pattern", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "grep", "description": "Search file contents", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "edit", "description": "Edit a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}}, "required": ["path", "old", "new"]}}},
    {"type": "function", "function": {"name": "bash", "description": "Run a shell command", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
]

SYSTEM_PROMPT = """You are the primary reasoning agent. You think, plan, and analyze — but you NEVER directly use grep, glob, read, or bash tools yourself. ALL file searches and file reads MUST go through the explore subagent via the Task tool.

Use sequential Task calls with the explore agent by default — it is fast and efficient for codebase lookups.

Once you understand the code from explore summaries, apply edits directly with edit/write/patch. Be concise — bullet points, not paragraphs.

For large multi-file implementations (3+ files), use the code subagent instead of editing each file yourself.

Your tools are: Task (subagent lookup), edit, write, patch, and bash."""

# Fake tool results that look like real codebase exploration output
FAKE_TOOL_RESULTS = [
    '{"files_found": ["src/exo/worker/engines/mlx/cache.py", "src/exo/worker/engines/mlx/generator/generate.py"], "summary": "KVPrefixCache class at cache.py:90. Main methods: get_kv_cache (lookup), add_kv_cache (store), get_prefix_length (matching). Uses LRU eviction with token_count * sqrt(access_count) scoring. Supports DeltaNet ArraysCache via CacheSnapshot."}',
    '{"file": "src/exo/worker/engines/mlx/cache.py", "lines": "215-311", "summary": "get_kv_cache: finds best matching prefix using mx.cumprod for mismatch detection. Returns deepcopy (or move if EXO_KV_CACHE_MOVE=1). For exact matches, trims to max_length-1. Handles SSM snapshot restore for DeltaNet models."}',
    '{"pattern_matches": 12, "files": ["cache.py", "generate.py", "batch_generator.py"], "summary": "KV prefix cache is created per-worker in the runner builder. Saved after prefill (generate.py:820), loaded at start of each generation (generate.py:673). Cache key uses normalized tokens to handle volatile patterns in first 100 tokens."}',
    '{"file": "src/exo/worker/engines/mlx/generator/generate.py", "lines": "666-725", "summary": "Cache lookup flow: 1) get_kv_cache finds best prefix match. 2) If hit ratio >= 70%, update existing entry. 3) If miss, add new entry. 4) Dedup entries with >90% overlap. Memory pressure check uses local GPU memory (not distributed) to avoid RDMA deadlocks."}',
]

PROMPTS = [
    "Explain how KV prefix caching works in this codebase",
    "How does the PP idle-time speculation work? Show me the key code paths.",
    "What happens when a client disconnects mid-generation? Trace the cancel flow.",
    "How does the adaptive throttle work for SDPA at high context?",
    "Explain the DeltaNet ArraysCache and why it can't be trimmed arbitrarily.",
]


@dataclass
class StepResult:
    step: int
    turn: int
    model: str
    prompt_tokens: int
    output_tokens: int
    ttft_ms: float
    decode_tps: float
    total_ms: float
    finish_reason: str
    tool_call_name: str | None = None
    error: str | None = None


def chat_completion(
    api: str,
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    max_tokens: int = 2000,
    temperature: float = 0.6,
    timeout: int = 180,
) -> tuple[dict, float]:
    """Send a chat completion request. Returns (response_dict, elapsed_seconds)."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if tools:
        payload["tools"] = tools

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{api}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read())
    elapsed = time.perf_counter() - t0
    return result, elapsed


def run_multi_turn(
    api: str,
    model: str,
    prompt: str,
    max_turns: int = 4,
    timeout: int = 180,
) -> list[StepResult]:
    """Run a multi-turn tool call conversation and collect metrics."""
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    results: list[StepResult] = []
    step = 0
    fake_result_idx = 0

    for turn in range(max_turns):
        step += 1
        print(f"  Turn {turn + 1}/{max_turns} (step {step})...", end=" ", flush=True)

        try:
            resp, elapsed = chat_completion(
                api, model, messages, tools=TOOLS, max_tokens=2000, timeout=timeout
            )
        except Exception as e:
            error_msg = str(e)
            print(f"ERROR: {error_msg}")
            results.append(StepResult(
                step=step, turn=turn + 1, model=model,
                prompt_tokens=0, output_tokens=0,
                ttft_ms=0, decode_tps=0, total_ms=elapsed * 1000 if 'elapsed' in dir() else 0,
                finish_reason="error", error=error_msg,
            ))
            break

        choice = resp["choices"][0]
        message = choice["message"]
        finish_reason = choice["finish_reason"]
        usage = resp.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        # Calculate tok/s from total time and output tokens
        decode_tps = output_tokens / elapsed if elapsed > 0 and output_tokens > 0 else 0

        tool_call_name = None
        if message.get("tool_calls"):
            tool_call_name = message["tool_calls"][0]["function"]["name"]

        result = StepResult(
            step=step, turn=turn + 1, model=model,
            prompt_tokens=prompt_tokens, output_tokens=output_tokens,
            ttft_ms=0,  # can't measure from non-streaming
            decode_tps=decode_tps, total_ms=elapsed * 1000,
            finish_reason=finish_reason, tool_call_name=tool_call_name,
        )
        results.append(result)

        if finish_reason == "tool_calls" and message.get("tool_calls"):
            tc = message["tool_calls"][0]
            print(f"tool_call={tc['function']['name']} args={tc['function']['arguments'][:60]}... "
                  f"({output_tokens} tok, {decode_tps:.1f} tok/s, {elapsed:.1f}s)")

            # Add assistant message with tool calls
            messages.append({
                "role": "assistant",
                "tool_calls": message["tool_calls"],
            })
            # Add fake tool result
            fake_result = FAKE_TOOL_RESULTS[fake_result_idx % len(FAKE_TOOL_RESULTS)]
            fake_result_idx += 1
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": fake_result,
            })
        else:
            # Final answer or stop
            content = message.get("content", "") or ""
            reasoning = message.get("reasoning_content", "") or ""
            total_text = len(content) + len(reasoning)
            print(f"finish={finish_reason} ({output_tokens} tok, {decode_tps:.1f} tok/s, "
                  f"{elapsed:.1f}s, {total_text} chars)")
            break

    return results


def main():
    parser = argparse.ArgumentParser(description="Stress test multi-turn tool calls")
    parser.add_argument("--api", default=API, help=f"API base URL (default: {API})")
    parser.add_argument("--model", default=MODEL_397B, help=f"Model ID (default: {MODEL_397B})")
    parser.add_argument("--turns", type=int, default=4, help="Max turns per conversation (default: 4)")
    parser.add_argument("--prompts", type=int, default=3, help="Number of prompts to test (default: 3)")
    parser.add_argument("--timeout", type=int, default=180, help="Per-request timeout in seconds (default: 180)")
    args = parser.parse_args()

    print(f"Stress test: {args.model}")
    print(f"API: {args.api}")
    print(f"Max turns: {args.turns}, Prompts: {args.prompts}, Timeout: {args.timeout}s")
    print("=" * 70)

    all_results: list[StepResult] = []

    for i, prompt in enumerate(PROMPTS[: args.prompts]):
        print(f"\n--- Prompt {i + 1}/{args.prompts}: {prompt[:60]}... ---")
        results = run_multi_turn(
            api=args.api,
            model=args.model,
            prompt=prompt,
            max_turns=args.turns,
            timeout=args.timeout,
        )
        all_results.extend(results)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    successful = [r for r in all_results if r.error is None]
    errors = [r for r in all_results if r.error is not None]
    tool_calls = [r for r in successful if r.finish_reason == "tool_calls"]
    completions = [r for r in successful if r.finish_reason in ("stop", "length")]

    if successful:
        avg_tps = sum(r.decode_tps for r in successful) / len(successful)
        total_tokens = sum(r.output_tokens for r in successful)
        total_time = sum(r.total_ms for r in successful) / 1000
        overall_tps = total_tokens / total_time if total_time > 0 else 0

        print(f"Total steps:     {len(all_results)}")
        print(f"  Successful:    {len(successful)}")
        print(f"  Tool calls:    {len(tool_calls)}")
        print(f"  Completions:   {len(completions)}")
        print(f"  Errors:        {len(errors)}")
        print(f"")
        print(f"Tokens generated: {total_tokens}")
        print(f"Total time:       {total_time:.1f}s")
        print(f"Overall tok/s:    {overall_tps:.1f}")
        print(f"Avg per-step:     {avg_tps:.1f} tok/s")

        if tool_calls:
            avg_tc_tps = sum(r.decode_tps for r in tool_calls) / len(tool_calls)
            print(f"Avg tool call:    {avg_tc_tps:.1f} tok/s")
        if completions:
            avg_comp_tps = sum(r.decode_tps for r in completions) / len(completions)
            print(f"Avg completion:   {avg_comp_tps:.1f} tok/s")

        print(f"\nPer-step breakdown:")
        print(f"{'Step':>4} {'Turn':>4} {'Tokens':>6} {'tok/s':>7} {'Time':>7} {'Finish':>12} {'Tool':>10}")
        print(f"{'-'*4:>4} {'-'*4:>4} {'-'*6:>6} {'-'*7:>7} {'-'*7:>7} {'-'*12:>12} {'-'*10:>10}")
        for r in all_results:
            tool = r.tool_call_name or ""
            finish = r.error[:12] if r.error else r.finish_reason
            print(f"{r.step:4d} {r.turn:4d} {r.output_tokens:6d} {r.decode_tps:7.1f} {r.total_ms / 1000:7.1f}s {finish:>12} {tool:>10}")
    else:
        print("No successful completions!")
        for r in errors:
            print(f"  Step {r.step}: {r.error}")


if __name__ == "__main__":
    main()
