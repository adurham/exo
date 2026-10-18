#!/usr/bin/env python3
"""Execute one §11 request row and record BOTH timing channels.

What it does
  POSTs a prepared §11 body (``--data-binary @file`` equivalent) to the
  cluster's OpenAI-compatible endpoint, timestamps every SSE line as it
  arrives, and writes:
    out/<label>.sse   raw stream, one "epoch<TAB>line" record per line
    out/<label>.json  parsed result: usage, finish_reason, per-frame token
                      epochs, client decode window, prefill-progress bands,
                      generation_stats comment, generated text

Why
  §9.1 of docs/dsv41-cluster-writeup-2026-10-17/WRITEUP.md measures two
  independent channels: client (per-frame line stamps) and server (the
  runner's ``[DSV41] obs`` lines). This script produces the client channel
  and every artifact the server-side correlation needs (response id,
  prompt-token count, start/end epochs). ``s11-correlate.py`` joins them.

Decode-rate definition (matches §9.1)
  window = last committed-token frame epoch - first committed-token frame
  epoch; rate = (usage.completion_tokens - 1) / window. The first committed
  token starts the window and is therefore excluded. Comment frames
  (``: keep-alive``, ``: prefill_progress``, ``: generation_stats``) are not
  committed tokens.

Usage
  python3 run-s11.py --body ../../prompts/depth_d100k.json --label d100k \
      --url http://node:52415/v1/chat/completions --out ./out --max-time 3600
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import sys
import time
from datetime import datetime, timezone
from urllib.parse import urlparse


def iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat(timespec="milliseconds")


def post(url: str, body: bytes, max_time: int):
    u = urlparse(url)
    if u.scheme not in ("http", "https"):
        raise SystemExit(f"bad url scheme: {u.scheme}")
    cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
    conn = cls(u.hostname, u.port or (443 if u.scheme == "https" else 80), timeout=max_time)
    path = u.path or "/v1/chat/completions"
    t0 = time.time()
    conn.request(
        "POST",
        path,
        body=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Connection": "keep-alive",
        },
    )
    resp = conn.getresponse()
    status = resp.status
    headers = dict(resp.getheaders())
    frames = []
    while True:
        line = resp.readline()
        if not line:
            break
        frames.append((time.time(), line.rstrip(b"\r\n").rstrip(b"\n")))
    t_end = time.time()
    conn.close()
    return status, headers, t0, t_end, frames


def extract_token_text(chunk: dict):
    """Return (content, reasoning) carried by one OpenAI chat chunk."""
    out_c, out_r = "", ""
    for ch in chunk.get("choices") or []:
        d = ch.get("delta") or {}
        if isinstance(d.get("content"), str):
            out_c += d["content"]
        if isinstance(d.get("reasoning_content"), str):
            out_r += d["reasoning_content"]
    return out_c, out_r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--url", default="http://macstudio-m4-1.tail19c543.ts.net:52415/v1/chat/completions")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
    ap.add_argument("--max-time", type=int, default=3600)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with open(args.body, "rb") as fh:
        body = fh.read()
    req = json.loads(body)
    streaming = bool(req.get("stream", False))

    sse_path = os.path.join(args.out, f"{args.label}.sse")
    json_path = os.path.join(args.out, f"{args.label}.json")

    try:
        status, headers, t0, t_end, frames = post(args.url, body, args.max_time)
        error = None
    except Exception as exc:  # timeout, reset, connection refused
        status, headers, t0, t_end, frames, error = 0, {}, time.time(), time.time(), [], repr(exc)

    content_parts, reasoning_parts = [], []
    token_epochs = []
    progress = []
    gen_stats = None
    usage = None
    finish_reason = None
    response_id = None
    error_events = []
    prefill_frames = 0
    keepalive_frames = 0

    with open(sse_path, "w", encoding="utf-8") as sse:
        for ts, raw in frames:
            sse.write(f"{ts:.6f}\t{raw.decode('utf-8', 'replace')}\n")
            line = raw.strip()
            if not line:
                continue
            if line.startswith(b": keep-alive"):
                keepalive_frames += 1
                continue
            if line.startswith(b": prefill_progress "):
                prefill_frames += 1
                try:
                    payload = json.loads(line[len(b": prefill_progress "):])
                    progress.append(
                        {
                            "epoch": ts,
                            "processed_tokens": payload.get("processed_tokens"),
                            "total_tokens": payload.get("total_tokens"),
                        }
                    )
                except json.JSONDecodeError:
                    pass
                continue
            if line.startswith(b": generation_stats "):
                try:
                    gen_stats = json.loads(line[len(b": generation_stats "):])
                except json.JSONDecodeError:
                    pass
                continue
            if not line.startswith(b"data: "):
                continue
            payload = line[len(b"data: "):]
            if payload.strip() == b"[DONE]":
                continue
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if chunk.get("error"):
                error_events.append(chunk["error"])
                continue
            response_id = response_id or chunk.get("id")
            if chunk.get("usage"):
                usage = chunk["usage"]
            c, r = extract_token_text(chunk)
            if c or r:
                token_epochs.append(ts)
                content_parts.append(c)
                reasoning_parts.append(r)
            for ch in chunk.get("choices") or []:
                if ch.get("finish_reason"):
                    finish_reason = ch["finish_reason"]

    if not streaming and frames:
        # non-stream: one JSON body across the lines
        joined = b"\n".join(raw for _, raw in frames)
        try:
            resp = json.loads(joined)
            response_id = response_id or resp.get("id")
            usage = resp.get("usage") or usage
            for ch in resp.get("choices") or []:
                finish_reason = ch.get("finish_reason") or finish_reason
                msg = ch.get("message") or {}
                content_parts.append(msg.get("content") or "")
                reasoning_parts.append(msg.get("reasoning_content") or "")
        except json.JSONDecodeError:
            pass

    first = token_epochs[0] if token_epochs else None
    last = token_epochs[-1] if token_epochs else None
    window = (last - first) if (first is not None and last is not None) else None
    completion = (usage or {}).get("completion_tokens")
    decode_client = None
    if window and completion and completion > 1:
        decode_client = (completion - 1) / window

    cached = None
    if usage:
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        if cached is None:
            cached = usage.get("cached_tokens")

    bands = []
    for a, b in zip(progress, progress[1:]):
        dt = b["epoch"] - a["epoch"]
        dp = (b["processed_tokens"] or 0) - (a["processed_tokens"] or 0)
        if dt > 0 and dp > 0:
            bands.append(
                {
                    "t": a["epoch"],
                    "from_token": a["processed_tokens"],
                    "to_token": b["processed_tokens"],
                    "seconds": round(dt, 3),
                    "tok_s": round(dp / dt, 2),
                }
            )

    result = {
        "label": args.label,
        "body_path": os.path.abspath(args.body),
        "url": args.url,
        "stream": streaming,
        "http_status": status,
        "error": error,
        "error_events": error_events,
        "started_epoch": t0,
        "started_iso": iso(t0),
        "ended_epoch": t_end,
        "ended_iso": iso(t_end),
        "wall_s": round(t_end - t0, 3),
        "response_id": response_id,
        "first_token_epoch": first,
        "first_token_iso": iso(first) if first else None,
        "ttft_s": round(first - t0, 3) if first else None,
        "last_token_epoch": last,
        "decode_window_s": round(window, 3) if window else None,
        "token_frames": len(token_epochs),
        "completion_tokens": completion,
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "cached_tokens": cached,
        "usage": usage,
        "finish_reason": finish_reason,
        "client_decode_tok_s": round(decode_client, 2) if decode_client else None,
        "prefill_frames": prefill_frames,
        "keepalive_frames": keepalive_frames,
        "progress": progress,
        "prefill_bands": bands,
        "generation_stats": gen_stats,
        "content": "".join(content_parts),
        "reasoning": "".join(reasoning_parts),
        "sse_path": os.path.abspath(sse_path),
        "headers": {k: v for k, v in headers.items() if k.lower() in ("content-type", "transfer-encoding", "date")},
    }
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)

    summary = {
        k: result[k]
        for k in (
            "label",
            "http_status",
            "wall_s",
            "ttft_s",
            "prompt_tokens",
            "cached_tokens",
            "completion_tokens",
            "finish_reason",
            "token_frames",
            "decode_window_s",
            "client_decode_tok_s",
            "error",
        )
    }
    if error_events:
        summary["error_events"] = error_events
    print(json.dumps(summary))
    print(f"OUT {json_path}")
    return 0 if status == 200 and not error else 1


if __name__ == "__main__":
    sys.exit(main())
