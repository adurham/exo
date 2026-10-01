#!/usr/bin/env python3
"""p22_api_baseline -- served-path TTFT baseline for DSv4.1 (production up).

Sends the FIXED phase-22 prompts (byte-for-byte files built by p22_prep.py)
through exo's /bench/chat/completions with use_prefix_cache=false (cold
session, dropped at end) and stream=true, and records:
  * wall to first streamed content token  (the user-visible TTFT)
  * usage.prompt_tokens / completion_tokens
  * per-chunk prefill progress events (PrefillProgressChunk, if surfaced)
Run from the gateway; the API lives on the Macs.
Writes JSONL + a summary table.
"""
import json
import os
import sys
import time
import urllib.request

API = os.environ.get("P22_API", "http://100.91.246.26:52415")
MODEL = "dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
PROMPTS = os.environ.get("P22_PROMPTS", "/home/hermes/.hermes/cache/scratch/exl3patch/p22_prompts")
OUT = os.environ.get("P22_API_OUT", "/home/hermes/.hermes/cache/scratch/exl3patch/p22_api_baseline.jsonl")
MAXTOK = int(os.environ.get("P22_API_MAXTOK", "16"))

recs = []
for tag in ("2k", "8k", "16k"):
    text = open(os.path.join(PROMPTS, f"prompt_{tag}_text.txt")).read()
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": text}],
        "max_tokens": MAXTOK,
        "temperature": 0.0,
        "enable_thinking": False,
        "stream": True,
        "use_prefix_cache": False,
    }
    req = urllib.request.Request(
        API + "/bench/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    tf = None
    text_out = []
    usage = None
    stats = None
    progress = []
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            ev = json.loads(payload)
            if ev.get("usage"):
                usage = ev["usage"]
            if ev.get("generation_stats"):
                stats = ev["generation_stats"]
            if ev.get("prefill_progress"):
                progress.append(ev["prefill_progress"])
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                piece = (d.get("content") or "") + (d.get("reasoning_content") or "")
                if piece:
                    if tf is None:
                        tf = time.time()
                    text_out.append(piece)
    wall = time.time() - t0
    rec = {
        "prompt": tag,
        "wall_s": round(wall, 3),
        "ttft_s": None if tf is None else round(tf - t0, 3),
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "completion_tokens": (usage or {}).get("completion_tokens"),
        "stats": stats,
        "prefill_progress_events": len(progress),
        "text": "".join(text_out)[:200],
    }
    if rec["ttft_s"] and rec["prompt_tokens"]:
        rec["prompt_tps_upper"] = round(rec["prompt_tokens"] / rec["ttft_s"], 1)
    recs.append(rec)
    print(json.dumps(rec, indent=1), flush=True)

with open(OUT, "a") as f:
    for rec in recs:
        f.write(json.dumps(rec) + "\n")
print("P22_API_BASELINE_DONE", flush=True)
