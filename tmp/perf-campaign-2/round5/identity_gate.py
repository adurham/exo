#!/usr/bin/env python3
"""Byte-identity + quality gate for a speculative-decode ship candidate.

Capture mode fires 3 fixed short prompts (reused verbatim from
bench/spec_degen_capture.py's PROMPTS list -- the three that reach
finish_reason=stop without truncation: sys_primary_colors,
sys_capital_france, sys_count_to_five) plus 2 tool-call-eliciting prompts
(reusing bench/dsv4_dsml_battery.py's TOOLS list) at temperature=0, and
saves the full response (content, reasoning_content, finish_reason,
token_ids if available).

Compare mode diffs two capture files prompt-by-prompt for byte identity,
after stripping server-generated tool_call.id fields (fresh random UUIDs
per response -- comparing them raw guarantees false positives).

Usage:
  identity_gate.py --capture --tag A --out A.json [--base-url ...] [--model ...]
  identity_gate.py --compare A.json B.json
"""
from __future__ import annotations

import argparse
import http.client
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[3]
assert (REPO_ROOT / "bench" / "spec_degen_capture.py").exists(), (
    f"expected repo root at {REPO_ROOT}, but bench/spec_degen_capture.py "
    "not found there -- adjust parents[N] if this script moved"
)

_spec_sdc = importlib.util.spec_from_file_location(
    "sdc", str(REPO_ROOT / "bench" / "spec_degen_capture.py")
)
sdc = importlib.util.module_from_spec(_spec_sdc)  # type: ignore[arg-type]
sys.modules["sdc"] = sdc
_spec_sdc.loader.exec_module(sdc)  # type: ignore[union-attr]

_spec_battery = importlib.util.spec_from_file_location(
    "dsml_battery", str(REPO_ROOT / "bench" / "dsv4_dsml_battery.py")
)
dsml_battery = importlib.util.module_from_spec(_spec_battery)  # type: ignore[arg-type]
sys.modules["dsml_battery"] = dsml_battery
_spec_battery.loader.exec_module(dsml_battery)  # type: ignore[union-attr]

# The 3 short prompts from spec_degen_capture.PROMPTS that reach
# finish_reason=stop without truncation (per harness-map.md section 5(iii)).
SHORT_PROMPT_LABELS = {"sys_primary_colors", "sys_capital_france", "sys_count_to_five"}
SHORT_PROMPTS = [
    (label, messages) for label, messages in sdc.PROMPTS if label in SHORT_PROMPT_LABELS
]
assert len(SHORT_PROMPTS) == 3, (
    f"expected exactly 3 short prompts from spec_degen_capture.PROMPTS, "
    f"found {len(SHORT_PROMPTS)} matching {SHORT_PROMPT_LABELS}"
)

DEFAULT_BASE_URL = "http://192.168.86.201:52415"
# NOTE: ab_probe_tier1.py hardcodes MODEL="mlx-community/DeepSeek-V4-Flash",
# but a live /state query during this dispatch's verification (2026-09-03)
# showed the ACTUALLY PLACED model id is "deepseek-ai/DeepSeek-V4-Flash-0731"
# -- the harness-map's assumed model id is stale. Use --model to override.
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"

# 2 tool-call-eliciting prompts, built from dsv4_dsml_battery.TOOLS + its
# own TURN_PROMPTS (first 2 turns -- both are tool-eliciting instructions).
TOOL_PROMPT_LABELS_AND_TEXT = list(zip(
    ["tool_git_status", "tool_read_config"],
    dsml_battery.TURN_PROMPTS[:2],
))
TOOL_PROMPTS = [
    (
        label,
        [
            {"role": "system", "content": dsml_battery.SYSTEM},
            {"role": "user", "content": text},
        ],
    )
    for label, text in TOOL_PROMPT_LABELS_AND_TEXT
]


def _post(base_url: str, body: dict, timeout: float) -> dict:
    u = urlparse(base_url)
    conn_cls = (
        http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
    )
    host = u.hostname or "localhost"
    conn = conn_cls(host, u.port or (443 if u.scheme == "https" else 80), timeout=timeout)
    payload = json.dumps(body)
    conn.request("POST", "/v1/chat/completions", payload, {"Content-Type": "application/json"})
    resp = conn.getresponse()
    raw = resp.read().decode("utf-8")
    conn.close()
    if resp.status != 200:
        raise RuntimeError(f"HTTP {resp.status}: {raw[:500]}")
    return json.loads(raw)


def _extract_token_ids(choice: dict) -> list | None:
    lp = choice.get("logprobs")
    if isinstance(lp, dict) and isinstance(lp.get("content"), list):
        return [tok.get("token") for tok in lp["content"] if isinstance(tok, dict)]
    return None


def capture_one(
    base_url: str, model: str, max_tokens: float, timeout: float,
    label: str, messages: list[dict], tools: list[dict] | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": 0,
    }
    if tools is not None:
        body["tools"] = tools
    try:
        d = _post(base_url, body, timeout)
    except Exception as e:  # noqa: BLE001
        return {"label": label, "messages": messages, "error": str(e)}

    choice = d["choices"][0]
    msg = choice.get("message", {})
    return {
        "label": label,
        "messages": messages,
        "content": msg.get("content", "") or "",
        "reasoning_content": msg.get("reasoning_content", "") or "",
        "finish_reason": choice.get("finish_reason"),
        "tool_calls": msg.get("tool_calls"),
        "token_ids": _extract_token_ids(choice),
        "usage": d.get("usage"),
    }


def do_capture(args: argparse.Namespace) -> int:
    results = []
    for label, messages in SHORT_PROMPTS:
        print(f"[capture:{args.tag}] {label} (short, no tools)...", file=sys.stderr)
        results.append(
            capture_one(args.base_url, args.model, args.max_tokens, args.timeout,
                        label, messages, tools=None)
        )
    for label, messages in TOOL_PROMPTS:
        print(f"[capture:{args.tag}] {label} (tool-eliciting)...", file=sys.stderr)
        results.append(
            capture_one(args.base_url, args.model, args.max_tokens, args.timeout,
                        label, messages, tools=dsml_battery.TOOLS)
        )

    out = {"tag": args.tag, "model": args.model, "base_url": args.base_url,
           "max_tokens": args.max_tokens, "results": results}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    n_err = sum(1 for r in results if "error" in r)
    print(f"[capture:{args.tag}] wrote {len(results)} results "
          f"({n_err} error(s)) to {args.out}", file=sys.stderr)
    return 0 if n_err == 0 else 1


def _strip_tool_call_ids(obj: Any) -> Any:
    """Recursively strip 'id' keys from tool_call dicts (fresh random UUIDs
    per response -- must be excluded before any identity comparison)."""
    if isinstance(obj, dict):
        return {
            k: (None if k == "id" else _strip_tool_call_ids(v))
            for k, v in obj.items()
            if k != "index"  # server-assigned position, not content
        }
    if isinstance(obj, list):
        return [_strip_tool_call_ids(x) for x in obj]
    return obj


def _digest_bytes(result: dict) -> bytes:
    """Canonical byte representation of a captured result for comparison,
    with tool_call ids stripped and key order normalized."""
    normalized = {
        "content": result.get("content"),
        "reasoning_content": result.get("reasoning_content"),
        "finish_reason": result.get("finish_reason"),
        "tool_calls": _strip_tool_call_ids(result.get("tool_calls")),
        "token_ids": result.get("token_ids"),
    }
    return json.dumps(normalized, sort_keys=True, ensure_ascii=False).encode("utf-8")


def _first_diff_context(a: bytes, b: bytes, ctx: int = 40) -> tuple[int, str]:
    n = min(len(a), len(b))
    offset = n
    for i in range(n):
        if a[i] != b[i]:
            offset = i
            break
    lo = max(0, offset - ctx)
    a_ctx = a[lo:offset + ctx]
    b_ctx = b[lo:offset + ctx]
    return offset, (
        f"...A={a_ctx!r}...\n" f"...B={b_ctx!r}..."
    )


def do_compare(args: argparse.Namespace) -> int:
    with open(args.file_a) as f:
        cap_a = json.load(f)
    with open(args.file_b) as f:
        cap_b = json.load(f)

    results_a = {r["label"]: r for r in cap_a["results"]}
    results_b = {r["label"]: r for r in cap_b["results"]}
    labels = list(results_a.keys())
    if list(results_b.keys()) != labels:
        print(f"FAIL: prompt label sets differ: "
              f"A={list(results_a)} B={list(results_b)}")
        return 1

    all_identical = True
    for label in labels:
        ra, rb = results_a[label], results_b[label]
        if "error" in ra or "error" in rb:
            print(f"[{label}] SKIP (capture error: A={ra.get('error')} "
                  f"B={rb.get('error')})")
            all_identical = False
            continue
        ba, bb = _digest_bytes(ra), _digest_bytes(rb)
        if ba == bb:
            print(f"[{label}] BYTE-IDENTICAL ({len(ba)} bytes)")
        else:
            all_identical = False
            offset, ctx = _first_diff_context(ba, bb)
            print(f"[{label}] DIFFERS at byte offset {offset} "
                  f"(A={len(ba)}B, B={len(bb)}B)")
            print(ctx)

    print()
    if all_identical:
        print(f"PASS: all {len(labels)} prompts byte-identical between "
              f"{cap_a.get('tag')} and {cap_b.get('tag')}")
        return 0
    else:
        print(f"FAIL: one or more of {len(labels)} prompts differ between "
              f"{cap_a.get('tag')} and {cap_b.get('tag')}")
        return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--capture", action="store_true")
    mode.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"))
    ap.add_argument("--tag", default="untagged", help="capture mode only")
    ap.add_argument("--out", help="capture mode only")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()

    if args.capture:
        if not args.out:
            ap.error("--capture requires --out")
        return do_capture(args)
    else:
        args.file_a, args.file_b = args.compare
        return do_compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
