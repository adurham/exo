#!/usr/bin/env python3
"""Mock exo server for the Phase-20 0c delta ladder (BRIEF L).

A tiny ``http.server`` that plays the role of the exo engine so the ladder can
be proven OFFLINE (the PM runs the live chunks; this worker never sends a
generation request to the real cluster).

It serves ``POST /v1/chat/completions`` as an OpenAI-compatible SSE stream that
contains BOTH a ``: generation_stats {json}`` COMMENT frame (which does NOT start
with ``data:``) and a final ``usage`` chunk, and it APPENDS real-shaped engine log
lines to a temp log file that the ladder reads through its ``LocalFileLog`` seam
instead of ssh:

    [ ts | INFO  | ...engine_prefill:333 ] [DSV41] prefill controls: ... (rows=M, base=2048)
    [ ts | INFO  | ...engine:_start_turn:957 ] [DSV41] turn reuse: prompt=N prefill=M reuse=R cache=C rewind=W UNCOMMITTED
    [ ts | WARNING | ...engine:_start_turn:964 ] [DSV41] reuse undershoot: refed=M rows (reused=R)

It models the two behaviours the ladder depends on:
  * BRANCHING (multi-turn) shape  messages=[user base, assistant reply, user delta]
    -> the LCP reaches the base's prompt-end checkpoint: reuse = base_rows,
       prefill = delta_rows.  (reps branch from the same base depth)
  * COLLAPSE (single-message-append) shape messages=[user base+delta]
    -> the trailing template tokens change and the cache rewinds to a rung:
       reuse = floor(base_rows, rung), prefill = the remainder (looks collapsed).

Timing is made checkable: the server sleeps ``rows / rows_per_s`` before the
first token, so ``prefill_s_log ~= prefill_rows / rows_per_s``.
"""
from __future__ import annotations

import argparse
import http.server
import json
import os
import re
import socketserver
import threading
import time

DEFAULT_CHARS_PER_TOKEN = 5.111
SALT_RE = re.compile(r"\[SESSION-SALT\s+([^\]\s]+)\]")
RUNG = 1024          # checkpoint ladder rung (SessionCache.plan)


def _est_rows(text: str, chars_per_token: float) -> int:
    return max(1, int(len(text) / chars_per_token))


class MockExo:
    """Stateful engine model (tracks resident conversations by head salt).

    rows_per_s       configurable prefill speed -> checkable prefill_s_log / ttft
    chars_per_token  how the mock tokenises char counts
    base_rows        the token depth registered when a base conversation is first fed
    """

    def __init__(self, log_path: str, *, rows_per_s: float = 200.0,
                 chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
                 sleep: bool = True):
        self.log_path = log_path
        self.rows_per_s = rows_per_s
        self.chars_per_token = chars_per_token
        self.sleep = sleep
        self.resident: dict[str, int] = {}      # salt -> base prompt-end rows
        self.lock = threading.Lock()

    # ---------------------------------------------------------------- log writer
    def _append(self, line: str) -> None:
        with open(self.log_path, "a") as fh:
            fh.write(line + "\n")

    def _ts(self, t: float | None = None) -> str:
        t = time.time() if t is None else t
        lt = time.localtime(t)
        frac = f"{t % 1:.3f}"[1:]
        return time.strftime("%Y-%m-%d %H:%M:%S", lt) + frac

    def _log_prefill_controls(self, rows: int, ts: float | None = None) -> None:
        self._append(
            f"[ {self._ts(ts)} | INFO     | exo.worker.engines.mlx.dsv41.session:"
            f"engine_prefill:333 ] [DSV41] prefill controls: fence_every=2 "
            f"transient_budget_mb=2048 score_row_bytes=1 fence_hook=on "
            f"(rows={rows}, base=2048)"
        )

    def _log_turn_reuse(self, prompt: int, prefill: int, reuse: int, cache: int,
                        rewind: int | None, ts: float | None = None) -> None:
        rw = f" rewind={rewind}" if rewind is not None else ""
        self._append(
            f"[ {self._ts(ts)} | INFO     | exo.worker.engines.mlx.dsv41.engine:"
            f"_start_turn:957 ] [DSV41] turn reuse: prompt={prompt} prefill={prefill} "
            f"reuse={reuse} cache={cache}{rw} UNCOMMITTED"
        )
        if prefill > 256 and reuse > 0:
            self._append(
                f"[ {self._ts(ts)} | WARNING  | exo.worker.engines.mlx.dsv41.engine:"
                f"_start_turn:964 ] [DSV41] reuse undershoot: refed={prefill} rows "
                f"(reused={reuse})"
            )

    # -------------------------------------------------------------- engine model
    def plan(self, messages: list[dict]) -> dict:
        """Return {prompt, prefill, reuse, cache, rewind} for this request."""
        head = messages[0].get("content", "") if messages else ""
        m = SALT_RE.search(head)
        salt = m.group(1) if m else None
        delta = ""
        branching = len(messages) >= 3 and messages[1].get("role") == "assistant"
        if branching:
            delta = messages[-1].get("content", "")
        with self.lock:
            base_rows = self.resident.get(salt) if salt else None
            head_rows = _est_rows(head, self.chars_per_token)
            delta_rows = _est_rows(delta, self.chars_per_token) if delta else 0

            if salt is not None and base_rows is None:
                # FIRST feed of this salt -> cold; register the base depth.
                base_rows = head_rows
                self.resident[salt] = base_rows
                prefill = head_rows + delta_rows
                return {"prompt": head_rows + delta_rows, "prefill": prefill,
                        "reuse": 0, "cache": head_rows + delta_rows, "rewind": None}

            if branching:
                # rewind to the base's prompt-end checkpoint; branch from base depth
                reuse = base_rows
                prompt = base_rows + delta_rows
                return {"prompt": prompt, "prefill": delta_rows, "reuse": reuse,
                        "cache": prompt, "rewind": _rung(prompt)}

            # single-message-append -> collapse to a rung
            reuse = (base_rows // RUNG) * RUNG
            prompt = head_rows
            return {"prompt": prompt, "prefill": prompt - reuse, "reuse": reuse,
                    "cache": prompt, "rewind": reuse}

    def handle_request(self, messages: list[dict]) -> dict:
        plan = self.plan(messages)
        rows = plan["prefill"]
        # The log timestamps are DERIVED, not slept: prefill_s_log = rows/rows_per_s
        # exactly (checkable).  sleep=True additionally burns real wall so an E2E
        # caller can see ttft; tests run with sleep=False.
        prefill_s = (rows / self.rows_per_s) if self.rows_per_s > 0 else 0.0
        if self.sleep and prefill_s > 0:
            time.sleep(prefill_s)
        t_start = time.time() - prefill_s
        t_end = time.time()
        self._log_prefill_controls(rows, ts=t_start)
        if plan["reuse"] > 0:
            self._log_turn_reuse(plan["prompt"], plan["prefill"], plan["reuse"],
                                 plan["cache"], plan["rewind"], ts=t_end)
        return plan


def _rung(prompt: int) -> int:
    return (prompt // RUNG) * RUNG


class _Handler(http.server.BaseHTTPRequestHandler):
    engine: MockExo = None            # set on the server instance
    model = "dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"

    def log_message(self, *a):       # silence
        pass

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):                 # /state /metrics /node_id /v1/models
        self._json(200, {"ok": True, "path": self.path})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n) if n else b"{}"
        try:
            req = json.loads(raw)
        except json.JSONDecodeError:
            self._json(400, {"error": "bad json"})
            return
        messages = req.get("messages", [])
        plan = self.engine.handle_request(messages)
        usage = {"prompt_tokens": plan["prompt"],
                 "completion_tokens": req.get("max_tokens", 1),
                 "total_tokens": plan["prompt"] + req.get("max_tokens", 1)}
        # stream: a stats COMMENT frame, one content chunk, then usage + [DONE]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        stats = {"prompt_tps": round(self.engine.rows_per_s, 1),
                 "prefix_cache_hit": plan["reuse"],
                 "mtp_cycles_cumulative": 3, "mtp_accepted_drafts_cumulative": 2}
        frames = [
            f": generation_stats {json.dumps(stats)}\n\n",
            "data: " + json.dumps({"id": "cmpl-mock", "created": int(time.time()),
                                   "choices": [{"index": 0, "delta": {"content": "ok"},
                                                "finish_reason": None}]}) + "\n\n",
            "data: " + json.dumps({"id": "cmpl-mock", "created": int(time.time()),
                                   "choices": [{"index": 0, "delta": {},
                                                "finish_reason": "stop"}],
                                   "usage": usage}) + "\n\n",
            "data: [DONE]\n\n",
        ]
        for fr in frames:
            self.wfile.write(fr.encode())
        self.wfile.flush()


class MockExoServer:
    """Context manager running the mock on an ephemeral port."""

    def __init__(self, log_path: str, *, rows_per_s: float = 200.0,
                 chars_per_token: float = DEFAULT_CHARS_PER_TOKEN, sleep: bool = True,
                 host: str = "127.0.0.1"):
        self.engine = MockExo(log_path, rows_per_s=rows_per_s,
                              chars_per_token=chars_per_token, sleep=sleep)
        self.host = host
        self.httpd = None
        self.thread = None

    def start(self) -> str:
        handler = type("_H", (_Handler,), {"engine": self.engine})
        self.httpd = socketserver.ThreadingTCPServer((self.host, 0), handler)
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        return self.base_url

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.httpd.server_address[1]}"

    def stop(self) -> None:
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()

    def __enter__(self) -> "MockExoServer":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


def main() -> int:
    ap = argparse.ArgumentParser(description="mock exo server for the 0c ladder")
    ap.add_argument("--log", required=True)
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--rows-per-s", type=float, default=200.0)
    a = ap.parse_args()
    srv = MockExoServer(a.log, rows_per_s=a.rows_per_s)
    handler = type("_H", (_Handler,), {"engine": srv.engine})
    srv.httpd = socketserver.ThreadingTCPServer((srv.host, a.port), handler)
    srv.httpd.daemon_threads = True
    print(f"mock exo listening on {srv.base_url}  log={a.log}", flush=True)
    try:
        srv.httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
