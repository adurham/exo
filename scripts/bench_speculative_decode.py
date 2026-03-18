#!/usr/bin/env python3
"""Benchmark decode throughput at various context sizes.

Usage:
  uv run python scripts/bench_speculative_decode.py
  uv run python scripts/bench_speculative_decode.py --label baseline --contexts 1000,5000,10000
  uv run python scripts/bench_speculative_decode.py --decode-tokens 200 --contexts 1000,50000,100000
"""
import argparse
import json
import sys
import time
import urllib.request

MODEL = "mlx-community/Qwen3-235B-A22B-Instruct-2507-6bit"
DECODE_TOKENS = 100
CONTEXT_SIZES = [1000, 5000, 10000, 25000, 50000, 100000]

FILLER = (
    "The architecture of modern distributed systems relies on careful coordination "
    "between multiple nodes. Each node maintains its own local state while participating "
    "in global consensus protocols. The challenge lies in balancing consistency with "
    "availability, as described by the CAP theorem. Network partitions are inevitable "
    "in any sufficiently large system, requiring careful design of failure recovery "
    "mechanisms. Replication strategies must account for both synchronous and asynchronous "
    "communication patterns, with trade-offs in latency and durability. "
)


def build_prompt(target_tokens: int, decode_tokens: int) -> str:
    """Build a prompt string that tokenizes to approximately target_tokens."""
    chars_needed = target_tokens * 4
    repetitions = chars_needed // len(FILLER) + 1
    text = (FILLER * repetitions)[:chars_needed]
    return f"Summarize the following text in exactly {decode_tokens} words:\n\n{text}"


def run_inference(
    api: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: int = 900,
) -> dict:
    """Send a streaming chat completion request and measure TTFT + decode TPS."""
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
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

    return {
        "status": "OK",
        "ttft_s": round(ttft, 2),
        "decode_tps": round(tps, 1),
        "tokens": token_count,
        "total_s": round(t_end - t_start, 2),
        "text": generated_text[:80],
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark decode throughput")
    parser.add_argument("--api", default="http://192.168.86.201:52415")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--label", default="", help="Annotation for this run")
    parser.add_argument("--contexts", default=",".join(str(c) for c in CONTEXT_SIZES))
    parser.add_argument("--decode-tokens", type=int, default=DECODE_TOKENS)
    args = parser.parse_args()

    decode_tokens = args.decode_tokens
    context_sizes = [int(c.strip()) for c in args.contexts.split(",")]
    label = args.label or "unlabeled"

    try:
        state = json.loads(urllib.request.urlopen(f"{args.api}/state", timeout=5).read())
        instance_count = len(state.get("instances", {}))
    except Exception as e:
        print(f"ERROR: Cannot reach cluster at {args.api}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[{label}] model={args.model}  instances={instance_count}  decode={decode_tokens}tok")
    print()
    print(f"{'Context':>10} {'TTFT':>8} {'TPS':>8} {'Tokens':>8} {'Total':>8}  Output")
    print("-" * 80)

    results = []
    for ctx in context_sizes:
        prompt = build_prompt(ctx, decode_tokens)
        r = run_inference(args.api, args.model, prompt, decode_tokens)
        results.append({"ctx": ctx, **r})

        ctx_label = f"~{ctx // 1000}K" if ctx >= 1000 else f"~{ctx}"
        if r["status"] == "OK":
            text = r["text"][:40].replace("\n", " ")
            print(f"{ctx_label:>10} {r['ttft_s']:>7.2f}s {r['decode_tps']:>7.1f} {r['tokens']:>8} {r['total_s']:>7.2f}s  {text}...")
        else:
            print(f"{ctx_label:>10} {'':>8} {'':>8} {'':>8} {'':>8}  FAIL: {r.get('error', '?')}")

        time.sleep(2)

    ok_results = [r for r in results if r["status"] == "OK"]
    if ok_results:
        avg_tps = sum(r["decode_tps"] for r in ok_results) / len(ok_results)
        print(f"\n[{label}] avg decode: {avg_tps:.1f} tok/s across {len(ok_results)} runs")


if __name__ == "__main__":
    main()
