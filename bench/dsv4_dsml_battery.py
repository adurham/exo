#!/usr/bin/env python3
"""DSML tool-call fidelity battery for DSv4 serving.

Regression gate for the 2026-07-09 verify-drift corruption class
(``</DSML-inv>`` / leaked tool-call markup / unterminated invoke): drives
tool-call-heavy generations at a context ladder against a live exo
endpoint and fails if ANY response leaks DSML/template markup into
content, drops a malformed tool call, or if the node's exo.log records
new DSML parse failures during the run.

Needle recall (dsv4_quality_harness.py) cannot see one-token structural
flips; this battery is the missing complement. Run it after ANY change to
the verify path (MTP, rowseq, pooling, indexer, top-k).

Usage (from the repo root on a Studio, cluster serving):
    .venv/bin/python bench/dsv4_dsml_battery.py \
        [--api http://192.168.86.201:52415/v1] \
        [--ctx 4096 65536 122880] [--turns 6] [--log ~/exo.log]

Exit 0 = battery clean; exit 1 = corruption detected (details on stdout).
"""
import argparse
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

DSML_SENTINEL = "｜DSML｜"
LEAK_PATTERNS = re.compile(
    r"<｜DSML｜|</?\s*tool_call|</parameter>|</invoke>|<\|im_"
)
LOG_FAIL_PATTERNS = (
    "DSML tool call parsing failed",
    "unterminated invoke",
    "invoke body was corrupt",
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "terminal",
            "description": "Run a shell command and return its output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from disk.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "integer"},
                    "limit": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
]

_FILLER_TOPICS = [
    "cluster utilization trends", "thermal envelope headroom",
    "storage growth projections", "the migration backlog",
    "network fabric saturation", "backup verification cadence",
    "certificate rotation schedules", "capacity planning inputs",
    "incident postmortem actions", "dependency upgrade windows",
    "observability gaps", "quota reconciliation",
]
_FILLER_VERBS = [
    "reviews", "summarizes", "quantifies", "contrasts", "projects",
    "audits", "flags regressions in", "tracks month-over-month",
]


def _filler(n_sentences: int) -> str:
    # Deterministic but VARIED prose. A single sentence repeated thousands
    # of times is degenerate context that biases the model toward early
    # EOS / repetition suppression — that tests model robustness, not
    # serving correctness, and produced a false-positive empty-response
    # finding on the first battery revision.
    out = []
    for i in range(n_sentences):
        t = _FILLER_TOPICS[i % len(_FILLER_TOPICS)]
        v = _FILLER_VERBS[(i * 7 + i // 12) % len(_FILLER_VERBS)]
        out.append(
            f"Section {i + 1} of the quarterly infrastructure report {v} "
            f"{t}, with appendix {chr(65 + i % 26)}{i % 100} holding the "
            "raw counters. "
        )
    return "".join(out)

SYSTEM = (
    "You are an infrastructure agent. Use the provided tools to inspect "
    "and edit files. Always respond with a tool call when a tool applies."
)

TURN_PROMPTS = [
    "Check the git status of /srv/repo with the terminal tool.",
    "Read the first 40 lines of /srv/repo/config.yaml.",
    "Run 'df -h /srv' and report.",
    "Write a file /tmp/battery_note.txt containing the single line 'ok'.",
    "Read /var/log/app.log lines 100-160.",
    "Run 'uname -a' with a 10 second timeout.",
]


def call(api: str, payload: dict, timeout: int = 1800) -> dict:
    req = urllib.request.Request(
        f"{api}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def log_fail_count(log_path: str) -> int:
    if not Path(log_path).expanduser().exists():
        return 0
    n = 0
    grep = subprocess.run(
        ["grep", "-c", "-E", "|".join(LOG_FAIL_PATTERNS),
         str(Path(log_path).expanduser())],
        capture_output=True, text=True,
    )
    try:
        n = int(grep.stdout.strip() or 0)
    except ValueError:
        n = 0
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://192.168.86.201:52415/v1")
    ap.add_argument("--model", default="mlx-community/DeepSeek-V4-Flash")
    ap.add_argument("--ctx", type=int, nargs="+",
                    default=[4096, 65536, 122880])
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--log", default="~/exo.log")
    args = ap.parse_args()

    failures: list[str] = []
    log_before = log_fail_count(args.log)

    for ctx in args.ctx:
        # Pad the system context up to ~ctx tokens with varied filler prose
        # (~1.3 tokens/word, ~22 words/sentence).
        n_fill = max(0, int((ctx - 600) / 1.3 / 22))
        messages = [
            {"role": "system", "content": SYSTEM + "\n\n" + _filler(n_fill)}
        ]
        t0 = time.time()
        for i in range(args.turns):
            messages.append(
                {"role": "user", "content": TURN_PROMPTS[i % len(TURN_PROMPTS)]}
            )
            try:
                resp = call(args.api, {
                    "model": args.model,
                    "messages": messages,
                    "tools": TOOLS,
                    "max_tokens": 2048,
                    "stream": False,
                })
            except Exception as exc:
                failures.append(f"ctx={ctx} turn={i}: request failed: {exc}")
                break
            choice = resp.get("choices", [{}])[0]
            msg = choice.get("message", {})
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or ""
            tool_calls = msg.get("tool_calls") or []
            finish = choice.get("finish_reason")

            for label, text in (("content", content), ("reasoning", reasoning)):
                if LEAK_PATTERNS.search(text):
                    failures.append(
                        f"ctx={ctx} turn={i}: markup leak in {label}: "
                        f"{text[:160]!r}"
                    )
            if finish == "tool_calls" and not tool_calls:
                failures.append(
                    f"ctx={ctx} turn={i}: finish=tool_calls but no tool_calls"
                )
            for tc in tool_calls:
                try:
                    json.loads(tc["function"]["arguments"])
                except Exception as exc:
                    failures.append(
                        f"ctx={ctx} turn={i}: unparseable tool args: {exc}"
                    )
            if not tool_calls and not content.strip():
                dump = Path(f"/tmp/dsml_battery_fail_ctx{ctx}_turn{i}.json")
                dump.write_text(json.dumps(resp, indent=1)[:20000])
                failures.append(
                    f"ctx={ctx} turn={i}: empty response (finish={finish}, "
                    f"reasoning_len={len(reasoning)}, dump={dump})"
                )

            # Feed the turn back so context accumulates naturally.
            assistant_msg = {"role": "assistant", "content": content}
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)
            for tc in tool_calls:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", "call_0"),
                    "content": '{"output": "ok", "exit_code": 0}',
                })
        print(f"ctx={ctx}: {args.turns} turns in {time.time()-t0:.0f}s")

    new_log_failures = log_fail_count(args.log) - log_before
    if new_log_failures > 0:
        failures.append(
            f"exo.log recorded {new_log_failures} new DSML parse failure(s) "
            "during the battery"
        )

    if failures:
        print(f"\nBATTERY FAILED — {len(failures)} finding(s):")
        for f in failures:
            print("  *", f)
        return 1
    print("\nBATTERY CLEAN — no DSML leaks, no malformed tool calls, "
          "no new parser failures.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
