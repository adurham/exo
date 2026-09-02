#!/usr/bin/env python3
"""
self_test.py
============
Verifies that passive_capture_proxy.py truly measures without buffering.

Starts:
  1. a LOCAL fake OpenAI-compatible SSE server (stdlib http.server) that emits a
     realistic chat.completions streaming response:
         - a deliberate delay before the first content chunk (simulated TTFT),
         - N content chunks each with a small inter-chunk delay,
         - a usage block, then [DONE].
  2. the capture proxy pointed at that fake server.

Then it drives a chat.completions request through the proxy and asserts the
captured JSONL line contains:
    - ttft_s within tolerance of the injected TTFT delay,
    - completion_tokens_streamed == number of content chunks emitted,
    - full_wall_rate_toks_per_s < post_ttft_rate_toks_per_s,
    - finish_reason == "stop", a usage block, and no tool calls.

Run:  python3 self_test.py
"""
from __future__ import annotations

import http.server
import json
import statistics
import sys
import threading
import time
import urllib.request

PROXY_DIR = __file__.rsplit("/", 1)[0]
sys.path.insert(0, PROXY_DIR)
import passive_capture_proxy as cap  # noqa: E402

INJECTED_TTFT_S = 0.50
N_CHUNKS = 12
INTER_CHUNK_S = 0.06
TTFT_TOLERANCE_S = 0.25
MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"

RESULTS = []


class FakeSSEHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _chunk(self, data):
        b = data.encode("utf-8")
        return ("%x\r\n" % len(b) + data + "\r\n").encode("utf-8")

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length)
        try:
            req = json.loads(body)
        except Exception:
            req = {}

        stream = req.get("stream", False)
        if not stream:
            # non-streaming path: return immediately (not exercised here)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(2))
            self.end_headers()
            self.wfile.write(b"{}")
            return

        # streaming chat.completions SSE response
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Transfer-Encoding", "chunked")   # body is chunk-framed
        self.end_headers()
        self.wfile.flush()

        # TTFT: no bytes at all for the injected delay
        time.sleep(INJECTED_TTFT_S)

        def ev(ch):
            payload = json.dumps(ch)
            return "data: %s\n\n" % payload

        # first content chunk -- the first byte of actual content arrives
        # right at the injected TTFT mark
        self.wfile.write(self._chunk(ev({
            "id": "chatcmpl-fake",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": MODEL,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hel"}, "finish_reason": None}],
        })))
        self.wfile.flush()

        for i in range(1, N_CHUNKS):
            time.sleep(INTER_CHUNK_S)
            word = "lo%dthere" % i
            self.wfile.write(self._chunk(ev({
                "id": "chatcmpl-fake",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": MODEL,
                "choices": [{"index": 0, "delta": {"content": word}, "finish_reason": None}],
            })))
            self.wfile.flush()

        time.sleep(INTER_CHUNK_S)
        # final chunk with finish_reason
        self.wfile.write(self._chunk(ev({
            "id": "chatcmpl-fake",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": MODEL,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        })))
        self.wfile.flush()

        time.sleep(INTER_CHUNK_S)
        # usage block
        self.wfile.write(self._chunk(ev({
            "id": "chatcmpl-fake",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": MODEL,
            "choices": [],
            "usage": {
                "prompt_tokens": 150_000,
                "completion_tokens": N_CHUNKS,
                "total_tokens": 150_000 + N_CHUNKS,
                "prompt_tokens_details": {"cached_tokens": 148_000},
            },
        })))
        self.wfile.flush()

        self.wfile.write(self._chunk("data: [DONE]\n\n"))
        self.wfile.flush()
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


def drive_request(proxy_port):
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:%d/v1/chat/completions" % proxy_port,
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()          # consume the stream end-to-end
    return resp.status, data


def main():
    jsonl_path = PROXY_DIR + "/capture.jsonl"
    try:
        os_remove(jsonl_path)
    except FileNotFoundError:
        pass

    # --- 1. fake SSE server on an ephemeral port ---
    fake = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeSSEHandler)
    fake_port = fake.server_address[1]
    t = threading.Thread(target=fake.serve_forever, daemon=True)
    t.start()
    RESULTS.append("fake sse server  : http://127.0.0.1:%d" % fake_port)

    # --- 2. capture proxy pointed at the fake server ---
    proxy = cap.CaptureServer(("127.0.0.1", 0), "127.0.0.1", fake_port, jsonl_path)
    proxy_port = proxy.server_address[1]
    pt = threading.Thread(target=proxy.serve_forever, daemon=True)
    pt.start()
    RESULTS.append("capture proxy    : http://127.0.0.1:%d  upstream=127.0.0.1:%d" % (proxy_port, fake_port))

    # --- 3. drive a streamed request through the proxy ---
    st = time.perf_counter()
    status, body = drive_request(proxy_port)
    drive_s = time.perf_counter() - st
    RESULTS.append("request driven   : status=%s  client read took %.3fs" % (status, round(drive_s, 3)))

    fake.shutdown()
    proxy.shutdown()
    fake.server_close()
    proxy.server_close()

    # --- 4. read + assert on the captured JSONL ---
    with open(jsonl_path) as f:
        lines = [ln for ln in f if ln.strip()]
    if not lines:
        print("FAIL: no JSONL lines captured")
        sys.exit(1)
    rec = json.loads(lines[-1])
    print("\n--- captured JSONL record (pretty-printed) ---")
    print(json.dumps(rec, indent=2))
    print("--- raw JSONL line ---")
    print(lines[-1].rstrip())

    failures = []
    checks = []

    # TTFT
    ttft = rec.get("ttft_s")
    ok = ttft is not None and abs(ttft - INJECTED_TTFT_S) <= TTFT_TOLERANCE_S
    checks.append("ttft_s within tolerance of injected %.2fs" % INJECTED_TTFT_S)
    if not ok:
        failures.append("TTFT %r not within %.2f +/- %.2f" % (ttft, INJECTED_TTFT_S, TTFT_TOLERANCE_S))

    # token count == number of content chunks
    streamed = rec.get("completion_tokens_streamed")
    ok = streamed == N_CHUNKS
    checks.append("completion_tokens_streamed == %d content chunks" % N_CHUNKS)
    if not ok:
        failures.append("streamed tokens %r != expected %d" % (streamed, N_CHUNKS))

    # server usage + cached tokens
    ok = rec.get("completion_tokens_usage") == N_CHUNKS and rec.get("cached_tokens") == 148_000
    checks.append("usage block: completion_tokens_usage == %d, cached_tokens == 148000" % N_CHUNKS)
    if not ok:
        failures.append("usage block mismatch: %r" % (rec.get("completion_tokens_usage"),))

    # rates: full-wall < post-TTFT
    fw = rec.get("full_wall_rate_toks_per_s")
    pt_rate = rec.get("post_ttft_rate_toks_per_s")
    ok = fw is not None and pt_rate is not None and fw < pt_rate
    checks.append("full_wall_rate_toks_per_s < post_ttft_rate_toks_per_s (%s < %s)" % (fw, pt_rate))
    if not ok:
        failures.append("rate ordering broken: full_wall=%r post_ttft=%r" % (fw, pt_rate))

    # finish_reason + tool-call flags
    ok = rec.get("finish_reason") == "stop" and rec.get("has_tool_calls") is False and rec.get("ended_in_tool_call") is False
    checks.append("finish_reason == 'stop', no tool calls, ended_in_tool_call == False")
    if not ok:
        failures.append("finish/tool flags wrong: %r" % rec.get("finish_reason"))

    # inter-chunk gap summary sanity
    gs = rec.get("inter_chunk_gap_summary") or {}
    ok = (gs.get("count") == N_CHUNKS - 1) and (gs.get("max_s", 0) >= 0.02)
    checks.append("inter_chunk_gap_summary: count==%d, max_s>=0.02 (real gaps recorded)" % (N_CHUNKS - 1))
    if not ok:
        failures.append("gap summary wrong: %r" % (gs,))

    # prompt/cached reporting
    checks.append("prompt_tokens == 150000 recorded")
    if rec.get("prompt_tokens") != 150_000:
        failures.append("prompt_tokens %r != 150000" % rec.get("prompt_tokens"))

    # fail-open: any capture errors?
    if rec.get("capture_errors"):
        failures.append("capture_errors present: %r" % rec["capture_errors"])
    checks.append("no capture_errors (fail-open clean)")

    # non-buffering is implied by ttft_s matching the injected delay; also the
    # wall duration should be approx N*inter_chunk + TTFT (not zeroed out)
    approx_expect = INJECTED_TTFT_S + N_CHUNKS * INTER_CHUNK_S
    ok = abs(rec["wall_duration_s"] - approx_expect) <= 0.6
    checks.append("wall_duration_s ~= TTFT + N*inter_chunk (%.2fs, captured %.2fs)" % (approx_expect, rec["wall_duration_s"]))
    if not ok:
        failures.append("wall duration %r not close to expected %r" % (rec["wall_duration_s"], approx_expect))

    print("\n=== ASSERTIONS ===")
    for c in checks:
        print("  [PASS] " + c)
    if failures:
        print("\n=== FAILURES ===")
        for f in failures:
            print("  [FAIL] " + f)
        sys.exit(1)

    print("\nRESULT: ALL CHECKS PASSED (%d)" % len(checks))
    print("\nSetup trace:")
    for r in RESULTS:
        print("  " + r)


def os_remove(p):
    import os
    os.remove(p)


if __name__ == "__main__":
    main()
