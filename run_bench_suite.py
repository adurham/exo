#!/usr/bin/env python3
"""
Exo Distributed Inference Benchmark Suite

Industry-standard benchmark for distributed LLM inference.
Tests TTFT and decode throughput at multiple prompt sizes:
  - Short  (~128 tokens)  — quick queries
  - Medium (~2k tokens)   — moderate context
  - Large  (~8k tokens)   — heavy context
  - XL     (~16k tokens)  — Claude Code style

Usage:
  python3 run_bench_suite.py [--label "description"] [--output results.json]
"""

import json, sys, time, urllib.request, os, argparse, textwrap

API_BASE = os.environ.get("EXO_API", "http://192.168.86.201:52415")
MODEL_ID = "mlx-community/MiniMax-M2.5-5bit"

# ── Prompt Generation ─────────────────────────────────────────────────

def make_code_context(target_chars: int) -> str:
    """Build a realistic code-context prompt from real exo source files."""
    exo_dir = os.path.expanduser("~/repos/exo/src/exo")
    files = []
    for root, _, filenames in os.walk(exo_dir):
        for f in filenames:
            if f.endswith('.py') and '__pycache__' not in root:
                fp = os.path.join(root, f)
                try:
                    with open(fp) as fh:
                        content = fh.read()
                        if len(content) > 100:
                            files.append((fp.replace(exo_dir, "src/exo"), content))
                except:
                    pass

    parts = ["You are a senior software engineer. Analyze the following code and suggest optimizations.\n\n"]
    total = len(parts[0])
    for path, content in sorted(files):
        block = f"### {path}\n```python\n{content}\n```\n\n"
        if total + len(block) > target_chars:
            # Truncate the last file to fit
            remaining = target_chars - total - 50
            if remaining > 200:
                parts.append(f"### {path}\n```python\n{content[:remaining]}\n```\n\n")
            break
        parts.append(block)
        total += len(block)

    parts.append("\nBased on the code above, identify the top 3 performance bottlenecks and propose specific fixes with code examples.\n")
    return "".join(parts)


PROMPTS = {
    "short": {
        "target_tokens": 128,
        "content": textwrap.dedent("""\
            You are a senior distributed systems architect. Design a fault-tolerant
            real-time bidding system handling 1M requests/sec with <10ms p99 latency.
            Cover: feature store lookups, ML inference, budget pacing, frequency capping,
            and online learning feedback. Provide specific technology choices and
            capacity planning. Be thorough and specific."""),
    },
    "medium": {
        "target_tokens": 2048,
        "content": None,  # generated at runtime
        "target_chars": 8000,
    },
    "large": {
        "target_tokens": 8192,
        "content": None,
        "target_chars": 32000,
    },
    "xl": {
        "target_tokens": 16384,
        "content": None,
        "target_chars": 64000,
    },
}


# ── Inference Runner ──────────────────────────────────────────────────

def run_inference(prompt: str, max_tokens: int = 2048) -> dict:
    """Send streaming inference, return timing metrics."""
    payload = json.dumps({
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }).encode()

    req = urllib.request.Request(
        f"{API_BASE}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    t_start = time.time()
    first_token = None
    first_content = None
    token_count = 0
    reasoning_count = 0
    content_count = 0
    prompt_tokens = 0
    completion_tokens = 0
    prefill_updates = []

    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            buffer = b""
            while True:
                chunk = resp.read(1)
                if not chunk:
                    break
                buffer += chunk
                if buffer.endswith(b"\n"):
                    line = buffer.decode("utf-8").strip()
                    buffer = b""

                    if "prefill_progress" in line:
                        try:
                            pdata = json.loads(line.split(" ", 2)[2])
                            chunk_data = pdata.get("PrefillProgressChunk", {})
                            prefill_updates.append({
                                "processed": chunk_data.get("processed_tokens", 0),
                                "total": chunk_data.get("total_tokens", 0),
                                "time": time.time() - t_start,
                            })
                        except:
                            pass
                        continue

                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "") or ""
                        reasoning = delta.get("reasoning_content", "") or ""
                        text = content or reasoning
                        if text:
                            if first_token is None:
                                first_token = time.time()
                            if content and first_content is None:
                                first_content = time.time()
                            token_count += 1
                            if reasoning:
                                reasoning_count += 1
                            if content:
                                content_count += 1

                            # Progress indicator
                            if token_count % 200 == 0:
                                elapsed = time.time() - first_token
                                tps = token_count / elapsed if elapsed > 0 else 0
                                print(f"    [{token_count} tokens, {tps:.1f} t/s]", flush=True)

                        usage = data.get("usage")
                        if usage:
                            prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
                            completion_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0))
                    except json.JSONDecodeError:
                        pass
    except Exception as e:
        print(f"    ERROR: {e}", flush=True)

    t_end = time.time()
    ttft = (first_token - t_start) if first_token else -1
    decode_time = (t_end - first_token) if first_token else 0
    tps = (token_count / decode_time) if decode_time > 0 else 0
    total = t_end - t_start

    # Calculate prefill throughput
    prefill_tps = 0
    if prefill_updates and first_token:
        prefill_time = first_token - t_start
        total_prefill_tokens = prefill_updates[-1].get("total", 0) if prefill_updates else 0
        prefill_tps = total_prefill_tokens / prefill_time if prefill_time > 0 else 0

    return {
        "ttft": round(ttft, 2),
        "prefill_tps": round(prefill_tps, 1),
        "decode_tps": round(tps, 2),
        "total_time": round(total, 2),
        "total_tokens": token_count,
        "reasoning_tokens": reasoning_count,
        "content_tokens": content_count,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "completed": token_count > 0,
    }


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Exo Benchmark Suite")
    parser.add_argument("--label", default="", help="Run label")
    parser.add_argument("--output", default=None, help="Output JSON file")
    parser.add_argument("--sizes", default="short,medium,large,xl",
                        help="Comma-separated prompt sizes to test")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Max output tokens per test")
    args = parser.parse_args()

    sizes = [s.strip() for s in args.sizes.split(",")]

    # Check cluster health
    try:
        state = json.loads(urllib.request.urlopen(f"{API_BASE}/state", timeout=5).read())
        node_count = len(state.get("topology", {}).get("nodes", []))
    except:
        print("ERROR: Cannot reach cluster API")
        sys.exit(1)

    print("=" * 65)
    print(f"  Exo Benchmark Suite — {args.label or 'unnamed run'}")
    print("=" * 65)
    print(f"  Model:  {MODEL_ID}")
    print(f"  Nodes:  {node_count}")
    print(f"  Sizes:  {', '.join(sizes)}")
    print(f"  Max output tokens: {args.max_tokens}")
    print("=" * 65)
    print()

    # Generate prompts
    for name, cfg in PROMPTS.items():
        if cfg["content"] is None and "target_chars" in cfg:
            print(f"  Generating {name} prompt (~{cfg['target_chars']//1000}k chars)...", flush=True)
            cfg["content"] = make_code_context(cfg["target_chars"])
            print(f"    → {len(cfg['content'])} chars (~{len(cfg['content'])//4} est. tokens)")

    all_results = {}

    for size in sizes:
        if size not in PROMPTS:
            print(f"  Skipping unknown size: {size}")
            continue

        prompt = PROMPTS[size]["content"]
        est_tokens = len(prompt) // 4

        print(f"\n{'─' * 65}")
        print(f"  Test: {size.upper()} (~{est_tokens} est. prompt tokens)")
        print(f"{'─' * 65}")

        result = run_inference(prompt, max_tokens=args.max_tokens)
        all_results[size] = result

        print(f"  ✓ TTFT: {result['ttft']}s")
        print(f"    Prefill: {result['prefill_tps']} tok/s")
        print(f"    Decode:  {result['decode_tps']} tok/s")
        print(f"    Tokens:  {result['total_tokens']} ({result['reasoning_tokens']}r + {result['content_tokens']}c)")
        print(f"    Prompt tokens (API): {result['prompt_tokens']}")
        print(f"    Status: {'✅' if result['completed'] else '❌ FAILED'}")

        if not result["completed"]:
            print("    ⚠️  Generation failed or timed out — possible deadlock")

    # Summary table
    print(f"\n{'=' * 65}")
    print(f"  SUMMARY — {args.label or 'unnamed run'}")
    print(f"{'=' * 65}")
    print(f"  {'Size':<8} {'Prompt':>8} {'TTFT':>8} {'Prefill':>10} {'Decode':>10} {'Status':>8}")
    print(f"  {'─'*8} {'─'*8} {'─'*8} {'─'*10} {'─'*10} {'─'*8}")
    for size, r in all_results.items():
        status = "✅" if r["completed"] else "❌"
        print(f"  {size:<8} {r['prompt_tokens']:>7}t {r['ttft']:>7}s {r['prefill_tps']:>8} t/s {r['decode_tps']:>8} t/s {status:>8}")
    print(f"{'=' * 65}")

    # Save results
    output = {
        "label": args.label,
        "model": MODEL_ID,
        "nodes": node_count,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": all_results,
    }
    fname = args.output or f"/tmp/bench_suite_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {fname}")


if __name__ == "__main__":
    main()
