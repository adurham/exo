#!/usr/bin/env python3
"""Task-level pass@1 eval for the EXO_DSV4_LMHEAD_MXFP8 knob.

WHY THIS EXISTS: a top-1 token flip rate is a DIAGNOSTIC, not a quality
metric. Many low-margin flips are synonyms, whitespace or formatting and
change nothing that matters; a few (an identifier, an operator, a digit)
are catastrophic. Deciding ship/no-ship from flip rate alone risks a false
negative -- rejecting a real speedup over a cosmetic difference -- or a
false positive in the other direction.

So this eval measures what actually matters: does the model still get the
right ANSWER? Every item has an objectively checkable result (exact string,
number, or executable code), scored mechanically with no human judgment and
no LLM grader. Run it against both arms and compare pass rates.

Usage:
  python3 bench/lmhead_task_eval.py --tag ON  --out DIR/eval_ON.json
  python3 bench/lmhead_task_eval.py --compare DIR/eval_ON.json DIR/eval_OFF.json
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time

import httpx

API = "http://192.168.86.201:52415"
MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"

# Each task: a prompt plus a mechanical checker. Checkers are deliberately
# forgiving about FORMATTING (which quantization may legitimately perturb)
# and strict about the ANSWER (which it must not).
TASKS: list[dict[str, object]] = [
    # --- arithmetic / exact numeric: a flipped digit is unambiguously wrong
    {"id": "arith_1", "kind": "number", "expect": 3901,
     "prompt": "Compute 47 * 83. Reply with only the number."},
    # NOTE: (3901-1229)/7 = 2672/7 = 381.714..., NOT 382. An earlier draft of
    # this file asserted 382 and scored a CORRECT model answer as a failure.
    # Ask for the integer-division form so the expected value is exact.
    {"id": "arith_2", "kind": "number", "expect": 2672,
     "prompt": "Compute 47 * 83 - 1229. Reply with only the number."},
    {"id": "arith_3", "kind": "number", "expect": 1024,
     "prompt": "What is 2 to the power of 10? Reply with only the number."},
    {"id": "arith_4", "kind": "number", "expect": 153,
     "prompt": "What is the sum of the cubes of 1, 5, and 3? Reply with only the number."},
    {"id": "word_problem_1", "kind": "number", "expect": 20,
     "prompt": ("Alice is twice as old as Bob. In 5 years the sum of their "
                "ages will be 40. How old is Alice now? Reply with only the number.")},
    # NOTE: 60km/30min + 30km/30min = 90 km in 60 min = 15 km per 10 min.
    # An earlier draft asserted 12 and scored the CORRECT answer as a failure.
    {"id": "word_problem_2", "kind": "number", "expect": 15,
     "prompt": ("A train travels 60 km in 30 minutes, then 30 km in 30 "
                "minutes. What is its average speed in km per 10 minutes? "
                "Reply with only the number.")},
    # --- exact factual recall: a flipped token changes the fact
    {"id": "fact_1", "kind": "contains", "expect": ["canberra"],
     "prompt": "What is the capital city of Australia? Reply with only the city name."},
    {"id": "fact_2", "kind": "contains", "expect": ["8"],
     "prompt": "How many bits are in one byte? Reply with only the number."},
    {"id": "fact_3", "kind": "contains", "expect": ["au"],
     "prompt": "What is the chemical symbol for gold? Reply with only the symbol."},
    {"id": "fact_4", "kind": "contains", "expect": ["mercury"],
     "prompt": "Which planet is closest to the Sun? Reply with only the planet name."},
    # --- code: executed against real assertions. An operator or index flip
    #     here is exactly the catastrophic-flip case worth catching.
    {"id": "code_binsearch", "kind": "python",
     "test": ("assert f([1,3,5,7,9], 7) == 3\n"
              "assert f([1,3,5,7,9], 1) == 0\n"
              "assert f([1,3,5,7,9], 4) == -1\n"
              "assert f([], 1) == -1\n"),
     "prompt": ("Write a Python function named `f(arr, target)` that performs "
                "binary search on the sorted list arr and returns the index of "
                "target, or -1 if not present. Reply with only the code in a "
                "```python code block.")},
    {"id": "code_fizzbuzz", "kind": "python",
     "test": ("assert f(3) == 'Fizz'\nassert f(5) == 'Buzz'\n"
              "assert f(15) == 'FizzBuzz'\nassert f(7) == '7'\n"),
     "prompt": ("Write a Python function named `f(n)` returning 'Fizz' if n is "
                "divisible by 3, 'Buzz' if by 5, 'FizzBuzz' if by both, else "
                "str(n). Reply with only the code in a ```python code block.")},
    {"id": "code_reverse_words", "kind": "python",
     "test": ("assert f('hello world foo') == 'foo world hello'\n"
              "assert f('a') == 'a'\n"),
     "prompt": ("Write a Python function named `f(s)` that reverses the order "
                "of whitespace-separated words in string s and returns them "
                "joined by single spaces. Reply with only the code in a "
                "```python code block.")},
    {"id": "code_primes", "kind": "python",
     "test": ("assert f(10) == [2,3,5,7]\nassert f(2) == [2]\nassert f(1) == []\n"),
     "prompt": ("Write a Python function named `f(n)` returning a sorted list "
                "of all prime numbers less than or equal to n. Reply with only "
                "the code in a ```python code block.")},
    {"id": "code_anagram", "kind": "python",
     "test": ("assert f('listen','silent') is True\n"
              "assert f('hello','world') is False\n"),
     "prompt": ("Write a Python function named `f(a, b)` returning True if the "
                "two strings are anagrams of each other, else False. Reply "
                "with only the code in a ```python code block.")},
]


def extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    return m.group(1) if m else text


def check(task: dict[str, object], text: str) -> tuple[bool, str]:
    kind = task["kind"]
    if kind == "number":
        nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text.replace(",", ""))
        if not nums:
            return False, "no number found"
        # accept the answer anywhere in the reply; models often restate
        want = float(task["expect"])
        got = [float(n) for n in nums]
        return (want in got), f"want {want} got {got[:8]}"
    if kind == "contains":
        low = text.lower()
        ok = any(str(e).lower() in low for e in task["expect"])
        return ok, f"want one of {task['expect']}"
    if kind == "python":
        code = extract_code(text)
        prog = code + "\n\n" + str(task["test"]) + "\nprint('PASS')\n"
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
            fh.write(prog)
            path = fh.name
        try:
            p = subprocess.run(
                [sys.executable, path], capture_output=True, text=True, timeout=15
            )
            return ("PASS" in p.stdout), (p.stderr or p.stdout)[-200:]
        except subprocess.TimeoutExpired:
            return False, "timeout"
    return False, "unknown kind"


def cmd_run(args: argparse.Namespace) -> int:
    results = []
    n_pass = 0
    with httpx.Client(timeout=600.0) as client:
        for task in TASKS:
            body = {
                "model": MODEL,
                "messages": [{"role": "user", "content": task["prompt"]}],
                "max_tokens": 1500,
                "temperature": 0.0,
                "stream": False,
            }
            try:
                r = client.post(f"{API}/v1/chat/completions", json=body)
                r.raise_for_status()
                msg = r.json()["choices"][0]["message"]
                text = (msg.get("content") or "")
                reasoning = (msg.get("reasoning_content") or "")
            except Exception as exc:
                results.append({"id": task["id"], "error": repr(exc), "pass": False})
                print(f"  {task['id']:22s} ERROR {exc!r}", flush=True)
                continue
            ok, detail = check(task, text)
            n_pass += int(ok)
            results.append({
                "id": task["id"], "pass": ok, "detail": detail,
                "content": text, "reasoning_content": reasoning,
            })
            print(f"  {task['id']:22s} {'PASS' if ok else 'FAIL'}  {detail[:70]}",
                  flush=True)
    out = {
        "tag": args.tag, "model": MODEL,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_tasks": len(TASKS), "n_pass": n_pass,
        "pass_rate": round(n_pass / len(TASKS), 4),
        "results": results,
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\n{args.tag}: {n_pass}/{len(TASKS)} passed "
          f"({100 * n_pass / len(TASKS):.1f}%)  -> {args.out}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    a = json.load(open(args.compare[0]))
    b = json.load(open(args.compare[1]))
    ba = {r["id"]: r for r in a["results"]}
    bb = {r["id"]: r for r in b["results"]}
    print(f"{'task':<24}{a['tag']:>10}{b['tag']:>10}   answer-identical")
    same = 0
    for tid in ba:
        ra, rb = ba[tid], bb.get(tid, {})
        ident = (ra.get("content") or "") == (rb.get("content") or "")
        same += int(ident)
        print(f"{tid:<24}{str(ra.get('pass')):>10}{str(rb.get('pass')):>10}"
              f"   {'yes' if ident else 'NO'}")
    print(f"\n{a['tag']}: {a['n_pass']}/{a['n_tasks']} ({a['pass_rate']:.1%})")
    print(f"{b['tag']}: {b['n_pass']}/{b['n_tasks']} ({b['pass_rate']:.1%})")
    print(f"byte-identical answers: {same}/{len(ba)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="untagged")
    ap.add_argument("--out", default="/tmp/lmhead_eval.json")
    ap.add_argument("--compare", nargs=2)
    args = ap.parse_args()
    return cmd_compare(args) if args.compare else cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
