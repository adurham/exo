#!/usr/bin/env python3
"""Benchmark speculative decoding acceptance rate across temperatures."""

import json
import sys
import time
import urllib.request

API = sys.argv[1] if len(sys.argv) > 1 else "http://192.168.86.201:52415"
MODEL = "mlx-community/Qwen3-235B-A22B-Instruct-2507-6bit"
MAX_TOKENS = 200

PROMPTS = [
    "List the first 20 elements of the periodic table with their atomic numbers.",
    "Write a Python function that implements binary search on a sorted list.",
    "Explain how TCP three-way handshake works step by step.",
    "What are the differences between compiled and interpreted programming languages?",
    "Describe the water cycle in detail, including all major stages.",
]

TEMPERATURES = [0.0, 0.3, 0.5, 0.7, 1.0]

print(f"=== Speculative Decoding Benchmark ===")
print(f"API: {API}")
print(f"Model: {MODEL}")
print(f"Max tokens: {MAX_TOKENS}")
print(f"Prompts: {len(PROMPTS)}")
print()

for temp in TEMPERATURES:
    print(f"--- Temperature: {temp} ---")
    total_accepted = 0
    total_generated = 0

    for i, prompt in enumerate(PROMPTS):
        body = json.dumps({
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAX_TOKENS,
            "temperature": temp,
            "stream": False,
        }).encode()

        try:
            req = urllib.request.Request(
                f"{API}/v1/chat/completions",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read())

            usage = result["usage"]
            gen = usage["completion_tokens"]
            accepted = usage["completion_tokens_details"]["accepted_prediction_tokens"]
            rate = accepted / gen * 100 if gen > 0 else 0
            print(f"  Prompt {i+1}: gen={gen} accepted={accepted} rate={rate:.1f}%")
            total_accepted += accepted
            total_generated += gen
        except Exception as e:
            print(f"  Prompt {i+1}: FAILED ({e})")

        time.sleep(1)

    if total_generated > 0:
        overall = total_accepted / total_generated * 100
        print(f"  TOTAL: temp={temp} gen={total_generated} accepted={total_accepted} rate={overall:.1f}%")
    print()

print("Done.")
