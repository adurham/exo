#!/usr/bin/env python3
"""300K + concurrent second stream, triggered IN-PROCESS (no file polling).

The first attempt failed because run-s11's .sse is buffered: the poller saw the first
token ~20 s late. Here the big row and the second client run in one process, so the
second request is fired from the same thread that parses the first content frame.

Writes out/contention2-300k.json, out/contention2-300k.sse (line-stamped, same format
as run-s11), out/contention2-second.json.
"""
import json
import statistics
import sys
import threading
import time
import urllib.request

URL = "http://macstudio-m4-1.tail19c543.ts.net:52415/v1/chat/completions"
MODEL = "dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
OUT = "/Users/adam.durham/.hermes/cache/scratch/ab300k-20261018/out"
BIG = "/Users/adam.durham/.hermes/cache/scratch/s11-run-20261018/prompts/depth_d300k.json"

SECOND_PROMPT = ("Write a detailed technical explanation of how Thunderbolt 5 RDMA works "
                 "between two Apple Silicon Macs: PCIe tunnelling, the UC completion model, "
                 "queue pairs, and what fails when a completion is lost. Be concrete, go long.")

first_token = threading.Event()
second_result = {}


def stream(body, on_first=None, sse_path=None, collect_text=False):
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    stamps, text, errors, first = [], [], [], None
    fh = open(sse_path, "w") if sse_path else None
    try:
        with urllib.request.urlopen(req, timeout=5400) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    if fh and line:
                        fh.write("%.6f\t%s\n" % (time.time(), line))
                    continue
                payload = line[5:].strip()
                if payload in ("", "[DONE]"):
                    continue
                now = time.time()
                if fh:
                    fh.write("%.6f\t%s\n" % (now, line))
                try:
                    d = json.loads(payload)
                except Exception:
                    continue
                if "error" in d:
                    errors.append(str(d["error"])[:200])
                    continue
                for ch in d.get("choices") or []:
                    delta = ch.get("delta") or {}
                    txt = delta.get("content") or delta.get("reasoning_content") or ""
                    if txt:
                        if first is None:
                            first = now
                            if on_first:
                                on_first()
                        stamps.append(now)
                        if collect_text:
                            text.append(txt)
                    if ch.get("finish_reason"):
                        break
    finally:
        if fh:
            fh.close()
    return {"stamps": stamps, "first": first, "text": "".join(text), "errors": errors}


big_body = json.load(open(BIG))
big_result = {}


def run_big():
    res = stream(big_body, on_first=first_token.set, sse_path=f"{OUT}/contention2-300k.sse")
    big_result.update(res)


def rounds(stamps, gap=0.05):
    if len(stamps) < 2:
        return [], 0
    rl, cur, rgaps = [], 1, []
    for a, b in zip(stamps, stamps[1:]):
        d = b - a
        if d > gap:
            rl.append(cur)
            rgaps.append(d)
            cur = 1
        else:
            cur += 1
    rl.append(cur)
    return rgaps, rl


t0 = time.time()
th = threading.Thread(target=run_big)
th.start()

fired = first_token.wait(timeout=3600)
t_fire = time.time() if fired else None
second = {}
if fired:
    second = stream({"model": MODEL,
                     "messages": [{"role": "user", "content": SECOND_PROMPT}],
                     "max_tokens": 3000, "temperature": 0.7, "stream": True,
                     "enable_thinking": False})
th.join()

bs = big_result.get("stamps") or []
rgaps, rl = rounds(bs)
res_big = {
    "label": "contention2-300k",
    "wall_s": round(time.time() - t0, 2),
    "ttft_s": round((big_result["first"] - t0), 2) if big_result.get("first") else None,
    "frames": len(bs),
    "decode_s": round(bs[-1] - bs[0], 3) if len(bs) > 1 else None,
    "tok_s": round(len(bs) / (bs[-1] - bs[0]), 2) if len(bs) > 1 else None,
    "rounds": len(rl),
    "tok_per_round": round(statistics.mean(rl), 2) if rl else None,
    "median_gap_s": round(statistics.median(rgaps), 3) if rgaps else None,
    "second_fired_at": t_fire,
    "second_ttft_s": round(second["first"] - t_fire, 2) if second.get("first") else None,
    "second_frames": len(second.get("stamps") or []),
    "second_tok_s": (round(len(second["stamps"]) / (second["stamps"][-1] - second["stamps"][0]), 2)
                     if len(second.get("stamps") or []) > 1 else None),
    "second_errors": second.get("errors") or [],
    "errors": big_result.get("errors") or [],
}
if bs and second.get("stamps"):
    # gaps that overlap the second stream vs gaps before it started
    s0 = second["stamps"][0]
    over, before = [], []
    for a, b in zip(bs, bs[1:]):
        if b - a <= 0.05:
            continue
        (over if b >= s0 else before).append(b - a)
    res_big["median_gap_overlap_s"] = round(statistics.median(over), 3) if over else None
    res_big["median_gap_solo_s"] = round(statistics.median(before), 3) if before else None
    res_big["overlap_gap_count"] = len(over)

json.dump(res_big, open(f"{OUT}/contention2-300k.json", "w"), indent=1)
json.dump({"fired": fired, "second": second.get("errors") or [],
           "frames": len(second.get("stamps") or [])},
          open(f"{OUT}/contention2-second.json", "w"), indent=1)
print(json.dumps(res_big, indent=1))
