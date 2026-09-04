#!/usr/bin/env /usr/bin/python3
"""Round-11 replay driver: >=20 sequential (c=1) chat-completion requests at
90-150K prompt depth against the cluster, through the passive-capture proxy,
mostly prefix-cache HITS (one shared long conversation with a short appended
turn each time), several ending in tool_calls (mirroring the study's real
44/55 tool_calls mix).

PM RUNS THIS, NOT THE IMPLEMENTING SUBAGENT. Do not invoke start_cluster.sh,
ssh, or touch the live runners from this script -- it is a pure HTTP client.

Interpreter: /usr/bin/python3 (R10 lesson: Homebrew python3 lacks httpx).
Stdlib only (urllib) -- no pip/uv install required, matching the passive
capture proxy's own zero-dependency policy.

Points at the EXISTING study capture path
(tmp/real-usage-capture-20260902/phase2/passive_capture_proxy.py), which
must already be running and listening on 127.0.0.1:52416, forwarding to
http://192.168.86.201:52415. This driver does NOT start the proxy -- start
it yourself first:

    /usr/bin/python3 tmp/real-usage-capture-20260902/phase2/passive_capture_proxy.py

Usage:
    /usr/bin/python3 tmp/perf-campaign-2/round11/replay_c1.py [--requests N]

Writes one JSON line per request to
tmp/perf-campaign-2/round11/results/replay.jsonl (appended, not overwritten,
so a crash mid-run doesn't lose earlier progress).
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROXY_URL = "http://127.0.0.1:52416/v1/chat/completions"
MODEL = "deepseek-ai/DeepSeek-V4-Flash"
RESULTS_PATH = (
    Path(__file__).resolve().parent / "results" / "replay.jsonl"
)

# Padding text repeated to build a long shared prefix. Not meaningful content
# -- purpose is prompt depth (90-150K tokens) with a stable, cacheable prefix
# across every turn appended below it.
_FILLER_PARAGRAPH = (
    "This is background context for a long-running engineering conversation "
    "about a distributed inference system. It exists purely to occupy token "
    "budget so the shared conversation prefix reaches the target depth for "
    "this round's prefix-cache-hit measurement. "
)

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_node_status",
            "description": "Get the current status of a cluster node by id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {
                        "type": "string",
                        "description": "The node identifier to query.",
                    }
                },
                "required": ["node_id"],
            },
        },
    }
]


def build_shared_prefix(target_tokens: int) -> str:
    """Build a filler prefix roughly target_tokens long (rough word-based
    estimate: ~1.3 tokens/word for English filler text)."""
    words_needed = int(target_tokens / 1.3)
    reps = max(1, words_needed // len(_FILLER_PARAGRAPH.split()))
    return (_FILLER_PARAGRAPH * reps)[: words_needed * 6]  # rough char cap too


def post_chat_completion(
    messages: list[dict], tools: list[dict] | None = None, stream: bool = True
) -> tuple[float, dict]:
    """POST one chat completion through the proxy. Returns (wall_seconds,
    parsed_summary_dict). Streams the response so the proxy's TTFT capture
    behaves as designed; does not depend on any non-stdlib package."""
    payload: dict = {
        "model": MODEL,
        "messages": messages,
        "stream": stream,
    }
    if tools is not None:
        payload["tools"] = tools

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        PROXY_URL,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )

    t0 = time.monotonic()
    finish_reason = None
    tool_call_names: list[str] = []
    content_parts: list[str] = []
    n_sse_events = 0
    stats_comment_lines: list[str] = []
    error: str | None = None

    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").rstrip("\n")
                if not line:
                    continue
                if line.startswith(": generation_stats"):
                    stats_comment_lines.append(line)
                    continue
                if line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    continue
                n_sse_events += 1
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    content_parts.append(delta["content"])
                for tc in delta.get("tool_calls") or []:
                    fn = (tc.get("function") or {}).get("name")
                    if fn:
                        tool_call_names.append(fn)
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
    except urllib.error.URLError as e:
        error = str(e)
    except TimeoutError as e:
        error = f"timeout: {e}"

    wall = time.monotonic() - t0
    return wall, {
        "finish_reason": finish_reason,
        "content_len": len("".join(content_parts)),
        "tool_call_names": tool_call_names,
        "n_sse_events": n_sse_events,
        "generation_stats_raw": stats_comment_lines,
        "error": error,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=24)
    parser.add_argument("--prompt-tokens", type=int, default=110_000)
    parser.add_argument(
        "--tool-call-fraction",
        type=float,
        default=0.8,
        help="Fraction of turns that should induce a tool call (mirrors "
        "the study's 44/55 ~= 0.8 real-usage mix).",
    )
    args = parser.parse_args()

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[replay_c1] building shared prefix (~{args.prompt_tokens} tokens)...",
        file=sys.stderr,
    )
    shared_prefix = build_shared_prefix(args.prompt_tokens)
    conversation: list[dict] = [
        {"role": "system", "content": "You are a helpful cluster-ops assistant."},
        {"role": "user", "content": shared_prefix},
        {
            "role": "assistant",
            "content": "Understood, I have the background context loaded.",
        },
    ]

    n_tool_turns = round(args.requests * args.tool_call_fraction)
    print(
        f"[replay_c1] {args.requests} requests, target ~{n_tool_turns} "
        f"ending in tool_calls, strictly sequential (c=1)",
        file=sys.stderr,
    )

    for i in range(args.requests):
        want_tool_call = i < n_tool_turns
        if want_tool_call:
            turn_text = (
                f"Turn {i}: please check the status of node studio-2 using "
                "the get_node_status tool."
            )
        else:
            turn_text = (
                f"Turn {i}: briefly acknowledge you received this short "
                "follow-up, in one sentence."
            )
        conversation.append({"role": "user", "content": turn_text})

        tools = TOOLS_SCHEMA if want_tool_call else None
        wall, summary = post_chat_completion(conversation, tools=tools)

        ended_in_tool_call = (
            summary["finish_reason"] == "tool_calls"
            or bool(summary["tool_call_names"])
        )
        record = {
            "turn_index": i,
            "wall_s": wall,
            "wanted_tool_call": want_tool_call,
            "ended_in_tool_call": ended_in_tool_call,
            **summary,
        }
        with RESULTS_PATH.open("a") as f:
            f.write(json.dumps(record) + "\n")

        status = "ERROR" if summary["error"] else "OK"
        print(
            f"[replay_c1] turn {i + 1}/{args.requests} {status} "
            f"wall={wall:.2f}s finish_reason={summary['finish_reason']} "
            f"tool_calls={summary['tool_call_names']} "
            f"n_sse_events={summary['n_sse_events']} "
            f"has_stats={bool(summary['generation_stats_raw'])}",
            file=sys.stderr,
        )

        if summary["error"]:
            # Keep going -- a single failed turn shouldn't abort the whole
            # replay; the analysis script's "no marks" counter will surface
            # it.
            conversation.append(
                {"role": "assistant", "content": "(error placeholder)"}
            )
            continue

        # Append a short synthetic assistant reply to extend the shared
        # conversation for the next turn's prefix-cache hit, whether or not
        # this turn produced visible content (tool-call turns produce none).
        conversation.append(
            {
                "role": "assistant",
                "content": "Acknowledged."
                if not ended_in_tool_call
                else "Calling get_node_status.",
            }
        )

    print(f"[replay_c1] done. Results appended to {RESULTS_PATH}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
