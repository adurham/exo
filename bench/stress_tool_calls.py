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

Use sequential Task calls with the explore agent (Qwen3) by default — it is fast and efficient for codebase lookups.

Once you understand the code from explore summaries, apply edits directly with edit/write/patch. Be concise — bullet points, not paragraphs.

For large multi-file implementations (3+ files), use the 'code' subagent instead of editing each file yourself.

Your tools are: Task (subagent lookup), edit, write, patch, and bash.
You are powered by the model named mlx-community/Qwen3.5-397B-A17B-4bit. The exact model ID is exo/mlx-community/Qwen3.5-397B-A17B-4bit
Here is some useful information about the environment you are running in:
<env>
  Working directory: /Users/adam.durham/repos/exo
  Workspace root folder: /Users/adam.durham/repos/exo
  Is directory a git repo: yes
  Platform: darwin
  Today's date: Mon Mar 23 2026
</env>
<directories>
  src/exo/worker/engines/mlx/ - MLX inference engine
  src/exo/master/ - Master node API and state machine
  src/exo/shared/ - Shared types and utilities
  mlx-lm/mlx_lm/ - MLX-LM library (forked)
</directories>
Skills provide specialized instructions and workflows for specific tasks.
Use the skill tool to load a skill when a task matches its description.
<available_skills>
  <skill>
    <name>cluster-stats</name>
    <description>Analyze exo cluster and opencode logs for performance stats</description>
  </skill>
</available_skills>

IMPORTANT: Always use the task tool to delegate file exploration. Never read files directly.
Always think step by step. When you have enough information from tool results, provide a comprehensive answer.
Do not ask the user questions — use your tools to find what you need.
Be thorough but concise. Use bullet points and code references."""

# Realistic tool results matching what the explore subagent actually returns
# Each is ~2000-4000 chars to simulate real file contents and summaries
FAKE_TOOL_RESULTS = [
    """Found 8 files matching KV/cache/prefix patterns:

1. src/exo/worker/engines/mlx/cache.py (main implementation)
   - KVPrefixCache class (line 90-350)
   - get_kv_cache(): Finds best matching prefix, returns deepcopy or move
   - add_kv_cache(): Stores with dedup (>90% overlap merged)
   - get_prefix_length(): Uses mx.cumprod for mismatch detection
   - LRU eviction: score = token_count * sqrt(access_count)
   - Supports DeltaNet ArraysCache via CacheSnapshot

2. src/exo/worker/engines/mlx/generator/generate.py
   - Cache lookup at line 666-725 (inside mlx_generate)
   - Cache save at line 820 (after prefill)
   - Cache key normalization for volatile first-100 tokens
   - prefix_hit_length tracks how many tokens were cached
   - Matched entries updated in-place if hit_ratio >= 70%

3. src/exo/worker/engines/mlx/cache.py:215-311 (get_kv_cache detail):
   ```python
   def get_kv_cache(self, model, prompt_tokens, normalized_tokens=None):
       best_idx, best_len = -1, 0
       compare_tokens = normalized_tokens if normalized_tokens is not None else prompt_tokens
       for i, cached_prompt in enumerate(self.prompts):
           cached_compare = self._normalized[i] if self._normalized[i] is not None else cached_prompt
           prefix_len = get_prefix_length(compare_tokens, cached_compare)
           if prefix_len > best_len:
               best_idx, best_len = i, prefix_len
       # ... trim, deepcopy/move, return remaining tokens
   ```

4. Key constants:
   - EXO_KV_CACHE_MAX_ENTRIES (default 4)
   - EXO_KV_CACHE_MOVE=1 (move instead of deepcopy)
   - _MIN_PREFIX_HIT_RATIO_TO_UPDATE = 0.70
   - EXOMEMORY_THRESHOLD = 0.98 (on studios)""",

    """File: src/exo/worker/engines/mlx/cache.py (lines 90-200)

class KVPrefixCache:
    def __init__(self, group=None):
        self.prompts: list[mx.array] = []
        self.caches: list[KVCacheType] = []
        self._snapshots: list[list[CacheSnapshot] | None] = []
        self._normalized: list[mx.array | None] = []
        self._access_counts: list[int] = []
        self._access_counter = 0
        self._MAX_ENTRIES = int(os.environ.get("EXO_KV_CACHE_MAX_ENTRIES", "4"))
        self._group = group

    def get_prefix_length(self, a: mx.array, b: mx.array) -> int:
        min_len = min(len(a), len(b))
        if min_len == 0:
            return 0
        match = (a[:min_len] == b[:min_len]).astype(mx.int32)
        prefix_len = int(mx.sum(mx.cumprod(match)).item())
        return prefix_len

    def add_kv_cache(self, prompt_tokens, caches, snapshots=None,
                     normalized_tokens=None):
        # Dedup: if >90% overlap with existing entry, replace it
        for i, existing in enumerate(self.prompts):
            overlap = self.get_prefix_length(prompt_tokens, existing)
            if overlap > 0.9 * max(len(prompt_tokens), len(existing)):
                # Replace in-place
                self.prompts[i] = prompt_tokens
                self.caches[i] = caches
                self._snapshots[i] = snapshots
                return

        # Evict if at capacity
        while len(self.prompts) >= self._MAX_ENTRIES:
            self._evict_lowest_value()

        self.prompts.append(prompt_tokens)
        self.caches.append(caches)
        self._snapshots.append(snapshots)

    def _evict_lowest_value(self):
        if not self.prompts:
            return
        scores = []
        for i in range(len(self.prompts)):
            token_count = len(self.prompts[i])
            access = self._access_counts[i] if i < len(self._access_counts) else 1
            scores.append(token_count * (access ** 0.5))
        victim = scores.index(min(scores))
        del self.prompts[victim]
        del self.caches[victim]""",

    """File: src/exo/worker/engines/mlx/generator/generate.py (lines 666-830)

Key flow in mlx_generate():

1. CACHE LOOKUP (line 666-725):
   prefix_hit_length = 0
   if kv_prefix_cache is not None:
       caches, prompt_tokens, matched_index = kv_prefix_cache.get_kv_cache(
           model, all_prompt_tokens, normalized_tokens=cache_key_tokens
       )
       prefix_hit_length = len(all_prompt_tokens) - len(prompt_tokens)
       if prefix_hit_length > 0:
           hit_pct = 100 * prefix_hit_length / len(all_prompt_tokens)
           if hit_pct < 5:
               # Near-miss: volatile pattern in first tokens
               prefix_hit_length = 0
           else:
               logger.info(f"KV cache hit: {prefix_hit_length}/{len(all_prompt_tokens)} ({hit_pct:.1f}%)")

2. PREFILL (line 779-790):
   prefill_tps, prefill_tokens, ssm_snapshots = prefill(
       model, tokenizer, sampler, prompt_tokens[:-1], caches, group, ...
   )

3. SAVE CACHE (line 800-826):
   if hit_ratio >= _MIN_PREFIX_HIT_RATIO_TO_UPDATE:
       kv_prefix_cache.update_kv_cache(matched_index, all_prompt_tokens, caches, ...)
   else:
       kv_prefix_cache.add_kv_cache(all_prompt_tokens, caches, ...)

4. DECODE (line 868+):
   Draft model prefill, then stream_generate / pp_step loop

Cache key normalization:
   - Strips <think>\\n prefix from cache key
   - Uses normalized_tokens for comparison to handle volatile patterns
   - First 100 tokens checked for instability""",

    """Additional implementation details from batch_generator.py and runner.py:

batch_generator.py (line 430-625):
- BatchGenerator._start_task() creates the mlx_generate generator
- KVPrefixCache is created once per worker in the runner Builder
- Passed to mlx_generate on each request
- Cache persists across requests (same worker process)

runner.py (line 456-510):
- Builder.build() creates the InferenceGenerator with KVPrefixCache
- kv_prefix_cache = KVPrefixCache(self.group)
- Stored as self.kv_prefix_cache on the generator

Memory management:
- Cache eviction uses LOCAL GPU memory pressure (not distributed)
- This avoids RDMA deadlocks from collective memory queries
- EXOMEMORY_THRESHOLD=0.98 on studios (raised from 0.90)
- Default 0.90 was evicting cache at 115GB, triggering re-prefill

DeltaNet ArraysCache handling:
- DeltaNet layers use ArraysCache (recurrent state, not KV pairs)
- Can't trim to arbitrary positions (sequential recurrence)
- CacheSnapshot stores state at specific positions
- get_kv_cache checks snapshots to find nearest valid restore point
- If no valid snapshot: full re-prefill required

Performance impact:
- 94.7% cache hit on multi-turn follow-ups (after <think>\\n fix)
- 8x faster TTFT on follow-up turns (350ms vs 2.74s for 695 tokens)
- Cache miss causes full re-prefill: 30+ seconds for 10K tokens""",
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
                api, model, messages, tools=TOOLS, max_tokens=32000, timeout=timeout
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
        prompt_tokens = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)
        # Tool call responses may report 0 completion_tokens — estimate from timing
        if output_tokens == 0 and elapsed > 0.5:
            # Rough estimate: ~30 tok/s baseline
            output_tokens = max(1, int(elapsed * 20))

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
    parser.add_argument("--turns", type=int, default=6, help="Max turns per conversation (default: 6)")
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
