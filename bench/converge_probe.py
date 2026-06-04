#!/usr/bin/env python3
"""Convergence diagnostic for DSv4-Flash thinking termination.

Distinguishes a SERVING bug (thinking block never terminates) from a
TASK-induced loop (model legitimately fails to converge on an ambiguous task).

For each probe x iters: record finish_reason, content_len, reasoning_len, and
whether the model CONVERGED (emitted a real final answer with finish_reason=stop).
A healthy serving stack should converge cleanly on the UNAMBIGUOUS probes.
"""
import argparse, json, sys, time
import httpx

# Unambiguous probes: hard enough to require real reasoning, but each has a
# definite endpoint. A healthy model MUST be able to stop and answer these.
PROBES = {
    "count_divis": (
        "How many positive integers less than 1000 are divisible by neither 5 "
        "nor 7? Work through it step by step, then end with a line exactly of "
        "the form 'ANSWER: <number>'."
    ),
    "trailing_zeros": (
        "How many trailing zeros are in 100! (100 factorial)? Show your "
        "reasoning, then end with a line exactly of the form 'ANSWER: <number>'."
    ),
    "logic_puzzle": (
        "Three switches in a room each control one of three bulbs in another "
        "room you can only enter once. Describe a procedure to determine which "
        "switch controls which bulb, then end with a line exactly of the form "
        "'ANSWER: <one-sentence summary>'."
    ),
    # Ambiguous control: the cart task's own test contradicts its bug spec, so
    # legitimate non-convergence is EXPECTED here. Used as a contrast.
    "cart_ambiguous": (
        "Fix this Python bug. The function should add two numbers but the spec "
        "is contradictory: the docstring says 'return the sum' but the included "
        "test asserts add(2,3)==6. Resolve it and return only the corrected "
        "function in a python code block, then a line 'ANSWER: done'.\n\n"
        "def add(a, b):\n    \"\"\"return the sum\"\"\"\n    return a - b"
    ),
}


def converged(content: str, finish_reason: str) -> bool:
    c = (content or "").strip()
    return finish_reason == "stop" and len(c) >= 10 and "ANSWER" in c.upper()


def run(base_url, model, probe_id, prompt, max_tokens, temperature, timeout):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    t0 = time.time()
    with httpx.Client(timeout=httpx.Timeout(timeout, connect=30)) as cl:
        r = cl.post(base_url.rstrip("/") + "/v1/chat/completions", json=body)
        r.raise_for_status()
        d = r.json()
    dt = time.time() - t0
    ch = d["choices"][0]
    msg = ch.get("message", {}) or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    fr = ch.get("finish_reason") or ""
    usage = d.get("usage", {}) or {}
    # BOS-spam guard
    leak = any(t in (content + reasoning) for t in (
        "<\uff5cbegin\u2581of\u2581sentence\uff5c>", "<|begin_of_sentence|>",
        "<\uff5cend\u2581of\u2581sentence\uff5c>"))
    return {
        "probe": probe_id, "finish_reason": fr,
        "content_len": len(content), "reasoning_len": len(reasoning),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "converged": converged(content, fr), "leak": leak,
        "latency_s": round(dt, 1),
        "content_head": content.strip()[:120],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://192.168.86.201:52415")
    ap.add_argument("--model", default="mlx-community/DeepSeek-V4-Flash-8bit")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--probes", default="count_divis,trailing_zeros,logic_puzzle,cart_ambiguous")
    ap.add_argument("--tag", default="specON")
    ap.add_argument("--out", default="/Users/adam.durham/repos/exo/bench/converge_probe_results.json")
    args = ap.parse_args()

    chosen = [p.strip() for p in args.probes.split(",") if p.strip()]
    all_results = []
    print(f"[converge_probe tag={args.tag}] model={args.model} iters={args.iters} "
          f"max_tokens={args.max_tokens} temp={args.temperature}")
    for pid in chosen:
        prompt = PROBES[pid]
        rows = []
        for i in range(args.iters):
            try:
                row = run(args.base_url, args.model, pid, prompt,
                          args.max_tokens, args.temperature, args.timeout)
            except Exception as e:
                row = {"probe": pid, "finish_reason": f"ERROR:{e}",
                       "content_len": 0, "reasoning_len": 0,
                       "completion_tokens": 0, "converged": False,
                       "leak": False, "latency_s": 0, "content_head": ""}
            rows.append(row)
            all_results.append(row)
            print(f"  {pid:16} it{i+1}/{args.iters}: conv={row['converged']!s:5} "
                  f"fr={row['finish_reason']:8} ctok={row['completion_tokens']:5} "
                  f"reason_len={row['reasoning_len']:7} ans={row['content_head'][:50]!r}")
        nconv = sum(1 for r in rows if r["converged"])
        rlens = [r["reasoning_len"] for r in rows]
        print(f"  --> {pid}: converged {nconv}/{len(rows)}  "
              f"reason_len min/med/max={min(rlens)}/{sorted(rlens)[len(rlens)//2]}/{max(rlens)}")
    # summary
    print("\n=== SUMMARY (" + args.tag + ") ===")
    for pid in chosen:
        rows = [r for r in all_results if r["probe"] == pid]
        nconv = sum(1 for r in rows if r["converged"])
        nleak = sum(1 for r in rows if r["leak"])
        rlens = [r["reasoning_len"] for r in rows]
        print(f"  {pid:16} converged {nconv}/{len(rows)}  leaks {nleak}  "
              f"reason_len[min/med/max]={min(rlens)}/{sorted(rlens)[len(rlens)//2]}/{max(rlens)}")
    with open(args.out, "w") as f:
        json.dump({"tag": args.tag, "model": args.model,
                   "max_tokens": args.max_tokens, "results": all_results}, f, indent=2)
    print(f"[out] {args.out}")


if __name__ == "__main__":
    sys.exit(main())
