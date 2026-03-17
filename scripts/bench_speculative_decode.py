#!/usr/bin/env python3
"""Benchmark speculative decoding: compare decode throughput with and without draft model.

Sends requests at multiple context sizes to the Qwen3-235B instance,
first without speculative decoding (baseline), then with the draft model
enabled. Produces a comparison table.

Usage:
  uv run python scripts/bench_speculative_decode.py [--api http://192.168.86.201:52415]

Requirements:
  - Cluster running with Qwen3-235B instance on Studios
  - Draft model instance (Qwen3-1.7B-8bit) on MacBook
  - EXO_DRAFT_SERVER and EXO_DRAFT_MODEL set on Studios (for draft runs)
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

MODEL = "mlx-community/Qwen3-235B-A22B-Instruct-2507-6bit"
DRAFT_MODEL = "mlx-community/Qwen3-1.7B-8bit"
DECODE_TOKENS = 100
CONTEXT_SIZES = [1000, 5000, 10000, 25000, 50000]

# Source text for building prompts at various sizes.
# Repeating natural text is better than random chars for realistic tokenization.
FILLER = (
    "The architecture of modern distributed systems relies on careful coordination "
    "between multiple nodes. Each node maintains its own local state while participating "
    "in global consensus protocols. The challenge lies in balancing consistency with "
    "availability, as described by the CAP theorem. Network partitions are inevitable "
    "in any sufficiently large system, requiring careful design of failure recovery "
    "mechanisms. Replication strategies must account for both synchronous and asynchronous "
    "communication patterns, with trade-offs in latency and durability. "
)


def build_prompt(target_tokens: int) -> str:
    """Build a prompt string that tokenizes to approximately target_tokens."""
    # ~4 chars per token is a rough estimate for English text
    chars_needed = target_tokens * 4
    repetitions = chars_needed // len(FILLER) + 1
    text = (FILLER * repetitions)[:chars_needed]
    return f"Summarize the following text in exactly {DECODE_TOKENS} words:\n\n{text}"


def run_inference(
    api: str,
    model: str,
    prompt: str,
    max_tokens: int = DECODE_TOKENS,
    timeout: int = 600,
) -> dict:
    """Send a streaming chat completion request and measure TTFT + decode TPS."""
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "stream": True,
        "enable_thinking": False,
    }).encode()

    req = urllib.request.Request(
        f"{api}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    t_start = time.perf_counter()
    t_first_token = None
    token_count = 0
    generated_text = ""

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            buf = b""
            while True:
                chunk = resp.read(1)
                if not chunk:
                    break
                buf += chunk
                if buf.endswith(b"\n"):
                    line = buf.decode().strip()
                    buf = b""
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        choices = data.get("choices", [])
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})
                        content = delta.get("content") or delta.get("reasoning_content") or ""
                        if content:
                            if t_first_token is None:
                                t_first_token = time.perf_counter()
                            token_count += 1
                            generated_text += content
                    except json.JSONDecodeError:
                        pass
    except Exception as e:
        return {"status": "FAIL", "error": str(e)}

    t_end = time.perf_counter()
    ttft = t_first_token - t_start if t_first_token else -1
    decode_time = t_end - t_first_token if t_first_token else 0
    tps = token_count / decode_time if decode_time > 0 else 0
    total_time = t_end - t_start

    return {
        "status": "OK",
        "ttft_s": round(ttft, 2),
        "decode_tps": round(tps, 1),
        "tokens": token_count,
        "total_s": round(total_time, 2),
    }


def reset_draft_cache(api: str) -> None:
    """Reset the draft model's KV cache between runs."""
    try:
        req = urllib.request.Request(
            f"{api}/v1/draft/reset",
            data=json.dumps({"model": DRAFT_MODEL}).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass


def check_draft_available(api: str) -> bool:
    """Check if the draft endpoint is responding on the MacBook's API."""
    try:
        req = urllib.request.Request(
            f"{api}/v1/draft/health",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def run_benchmark(
    api: str,
    draft_api: str,
    context_sizes: list[int],
) -> None:
    print(f"=== Speculative Decode Benchmark ===")
    print(f"API:          {api}")
    print(f"Draft API:    {draft_api}")
    print(f"Model:        {MODEL}")
    print(f"Draft Model:  {DRAFT_MODEL}")
    print(f"Decode Tokens: {DECODE_TOKENS} per run")
    print(f"Context Sizes: {context_sizes}")
    print()

    # Check cluster health
    try:
        state = json.loads(urllib.request.urlopen(f"{api}/state", timeout=5).read())
        instance_count = len(state.get("instances", {}))
        print(f"Cluster: {instance_count} instances")
    except Exception as e:
        print(f"ERROR: Cannot reach cluster at {api}: {e}")
        sys.exit(1)

    draft_available = check_draft_available(draft_api)
    print(f"Draft server: {'available' if draft_available else 'NOT available'}")
    print()

    # Results storage
    baseline_results: list[dict] = []
    draft_results: list[dict] = []

    # Run baseline (no draft)
    print("=" * 60)
    print("PHASE 1: Baseline (no speculative decoding)")
    print("=" * 60)
    for ctx in context_sizes:
        prompt = build_prompt(ctx)
        ctx_label = f"{ctx // 1000}K" if ctx >= 1000 else str(ctx)
        print(f"\n  ~{ctx_label} tokens...", end=" ", flush=True)
        result = run_inference(api, MODEL, prompt)
        baseline_results.append({"ctx": ctx, **result})
        if result["status"] == "OK":
            print(f"TTFT={result['ttft_s']}s  Decode={result['decode_tps']} tok/s  Total={result['total_s']}s")
        else:
            print(f"FAILED: {result.get('error', 'unknown')}")
        time.sleep(2)

    # Run with draft
    if draft_available:
        print()
        print("=" * 60)
        print("PHASE 2: With speculative decoding (draft model)")
        print("=" * 60)
        for ctx in context_sizes:
            prompt = build_prompt(ctx)
            ctx_label = f"{ctx // 1000}K" if ctx >= 1000 else str(ctx)
            print(f"\n  ~{ctx_label} tokens...", end=" ", flush=True)
            reset_draft_cache(draft_api)
            result = run_inference(api, MODEL, prompt)
            draft_results.append({"ctx": ctx, **result})
            if result["status"] == "OK":
                print(f"TTFT={result['ttft_s']}s  Decode={result['decode_tps']} tok/s  Total={result['total_s']}s")
            else:
                print(f"FAILED: {result.get('error', 'unknown')}")
            time.sleep(2)
    else:
        print("\nSkipping draft phase — draft server not available")

    # Generate report
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    report = []
    report.append(f"# Speculative Decode Benchmark — {time.strftime('%Y-%m-%d %H:%M')}")
    report.append(f"")
    report.append(f"- **Model**: {MODEL}")
    report.append(f"- **Draft Model**: {DRAFT_MODEL}")
    report.append(f"- **Decode Tokens**: {DECODE_TOKENS} per run")
    report.append(f"")

    if draft_results:
        report.append("| Context | Baseline TPS | Draft TPS | Speedup | Baseline TTFT | Draft TTFT |")
        report.append("|---------|-------------|-----------|---------|---------------|------------|")
        for b, d in zip(baseline_results, draft_results):
            ctx = b["ctx"]
            ctx_label = f"~{ctx // 1000}K" if ctx >= 1000 else f"~{ctx}"
            b_tps = b.get("decode_tps", "?")
            d_tps = d.get("decode_tps", "?")
            b_ttft = b.get("ttft_s", "?")
            d_ttft = d.get("ttft_s", "?")
            if isinstance(b_tps, (int, float)) and isinstance(d_tps, (int, float)) and b_tps > 0:
                speedup = f"{d_tps / b_tps:.2f}x"
            else:
                speedup = "?"
            report.append(f"| {ctx_label} | {b_tps} | {d_tps} | {speedup} | {b_ttft}s | {d_ttft}s |")
    else:
        report.append("| Context | TTFT (s) | Decode (tok/s) | Total (s) | Status |")
        report.append("|---------|----------|----------------|-----------|--------|")
        for r in baseline_results:
            ctx = r["ctx"]
            ctx_label = f"~{ctx // 1000}K" if ctx >= 1000 else f"~{ctx}"
            report.append(f"| {ctx_label} | {r.get('ttft_s', '?')} | {r.get('decode_tps', '?')} | {r.get('total_s', '?')} | {r.get('status', '?')} |")

    report_text = "\n".join(report)
    print(report_text)

    # Save report
    output_path = Path(f"/tmp/bench_speculative_{time.strftime('%Y%m%d_%H%M%S')}.md")
    output_path.write_text(report_text)
    print(f"\nSaved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark speculative decoding")
    parser.add_argument("--api", default="http://192.168.86.201:52415", help="Primary API URL")
    parser.add_argument("--draft-api", default="http://192.168.86.203:52415", help="Draft model API URL")
    parser.add_argument("--contexts", default=",".join(str(c) for c in CONTEXT_SIZES),
                        help="Comma-separated context sizes")
    parser.add_argument("--decode-tokens", type=int, default=DECODE_TOKENS,
                        help="Number of tokens to generate per run")
    args = parser.parse_args()

    global DECODE_TOKENS
    DECODE_TOKENS = args.decode_tokens
    context_sizes = [int(c.strip()) for c in args.contexts.split(",")]

    run_benchmark(args.api, args.draft_api, context_sizes)


if __name__ == "__main__":
    main()
