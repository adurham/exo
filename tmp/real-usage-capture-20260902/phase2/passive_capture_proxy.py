#!/usr/bin/env python3
"""
passive_capture_proxy.py
========================
Passive reverse proxy + per-request latency measurer for a Hermes-on-exo
OpenAI-compatible session.

It sits between the Hermes client and the exo API endpoint and:
  * forwards every request to the upstream verbatim,
  * relays the HTTP response body UNMODIFIED and UNBUFFERED (bytes are written
    to the client the instant they arrive -- chunk-by-chunk, never accumulated),
  * measures purely as a side effect, appending one JSON line per request to a
    JSONL file.

Two hard guarantees:
  1. STREAMING IS NEVER BUFFERED. Each socket read from upstream is flushed to
     the client before any parsing happens. Destroying TTFT by buffering is the
     one thing this tool must never do.
  2. FAIL-OPEN MEASUREMENT. Every measurement/parsing/logging step is wrapped so
     that a crash in the capture path can never break the user's session; the
     bytes are still relayed.

STDLIB ONLY. Python >= 3.9. No pip/uv install needed.

Usage:
    python3 passive_capture_proxy.py [--port 52416] [--upstream http://192.168.86.201:52415] [--jsonl capture.jsonl]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
import uuid
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_HOP_BY_HOP = {
    "connection", "keep-alive", "transfer-encoding", "content-length",
    "upgrade", "proxy-connection", "te", "trailer", "proxy-authenticate",
    "proxy-authorization",
}

_READ_SIZE = 65536


# --------------------------------------------------------------------------- #
# Measurement state kept per request. Pure data; relay never depends on it.    #
# --------------------------------------------------------------------------- #
def _new_meas(path, method, start_monotonic, start_epoch):
    return {
        "request_id": uuid.uuid4().hex[:12],
        "start_monotonic": start_monotonic,   # perf_counter, for durations
        "start_ts_epoch": start_epoch,        # wall clock, for humans
        "path": path,
        "method": method,
        "streaming": False,
        "sse_buffer": "",
        "n_sse_events": 0,
        "streamed_content_chunks": 0,         # SSE chunks whose delta.content != ""
        "first_content_chunk_time": None,     # monotonic ts of first content chunk
        "content_chunk_times": [],            # monotonic ts of every content chunk
        "finish_reason": None,
        "tool_calls": [],                     # [{id, name, arguments}]
        "usage": None,
        "capture_errors": [],
        "relay_errors": [],
    }


class CaptureHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    do_GET = lambda self: self._dispatch("GET")      # noqa: E731
    do_POST = lambda self: self._dispatch("POST")    # noqa: E731
    do_PUT = lambda self: self._dispatch("PUT")      # noqa: E731
    do_DELETE = lambda self: self._dispatch("DELETE")# noqa: E731
    do_PATCH = lambda self: self._dispatch("PATCH")  # noqa: E731
    do_OPTIONS = lambda self: self._dispatch("OPTIONS")  # noqa: E731
    do_HEAD = lambda self: self._dispatch("HEAD")    # noqa: E731

    # -- logging noise suppressor -------------------------------------------
    def log_message(self, *args):  # keep stderr clean during a real session
        pass

    # ------------------------------------------------------------------ #
    # Request body reading (supports Content-Length and chunked uploads)  #
    # ------------------------------------------------------------------ #
    def _read_body(self):
        te = (self.headers.get("Transfer-Encoding") or "").lower()
        if te == "chunked":
            return self._read_chunked_body()
        length = self.headers.get("Content-Length")
        if not length:
            return b""
        return self.rfile.read(int(length))

    def _read_chunked_body(self):
        chunks = []
        while True:
            line = self.rfile.readline().strip()
            if not line:
                continue
            try:
                size = int(line, 16)
            except ValueError:                     # trailer / end marker
                return b"".join(chunks)
            if size == 0:
                while self.rfile.readline().strip():
                    pass                            # drain trailers
                return b"".join(chunks)
            chunks.append(self.rfile.read(size))
            self.rfile.read(2)                      # trailing CRLF

    # ------------------------------------------------------------------ #
    # Main relay loop                                                     #
    # ------------------------------------------------------------------ #
    def _dispatch(self, method):
        start_mon = time.perf_counter()
        start_epoch = time.time()
        meas = _new_meas(self.path, method, start_mon, start_epoch)
        status = 502
        model = None
        stream = None
        try:
            # 1) read the request body (not latency-critical: uploads are not
            #    the direction we measure)
            body = self._read_body()
            try:
                req = json.loads(body.decode("utf-8", "replace")) if body else {}
                model = req.get("model")
                stream = req.get("stream", False)
            except Exception:
                pass

            # 2) forward to upstream
            host, port = self.server.upstream
            conn = HTTPConnection(host, port, timeout=600)
            headers = {}
            for k, v in self.headers.items():
                if k.lower() in _HOP_BY_HOP:
                    continue
                headers[k] = v
            headers["Host"] = "%s:%d" % (host, port)

            try:
                conn.request(method, self.path, body=body, headers=headers)
            except Exception as e:                      # upstream connect/request failed
                self._raw_error(502, "upstream request failed: %s" % e)
                status = 502
                conn.close()
                return

            resp = conn.getresponse()
            status = resp.status

            # 3) relay response headers (re-framed as chunked since we stream)
            self.send_response(resp.status, resp.reason)
            for k, v in resp.getheaders():
                if k.lower() in _HOP_BY_HOP:
                    continue
                self.send_header(k, v)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            ct = (resp.getheader("Content-Type") or "").lower()
            meas["streaming"] = "event-stream" in ct
            if not meas["streaming"]:
                meas["streamed_content_chunks"] = 0  # non-SSE: no token counting

            # 4) relay body chunk-by-chunk, UNBUFFERED. The write to the client
            #    happens BEFORE any parsing so parsing latency never delays bytes.
            try:
                while True:
                    # read1() reads at most ONE upstream chunk and never
                    # accumulates across chunk boundaries -- so we relay bytes
                    # the instant they arrive instead of buffering an amt-sized
                    # window. (read(amt) would aggregate the whole small body
                    # before returning, destroying both TTFT and streaming.)
                    chunk = resp.read1(_READ_SIZE)
                    if not chunk:
                        break
                    chunk_mon = time.perf_counter()
                    # --- relay first (this is what matters) ---
                    try:
                        self.wfile.write(b"%x\r\n" % len(chunk))
                        self.wfile.write(chunk)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError) as e:
                        meas["relay_errors"].append(repr(e))
                        break                            # client went away
                    # --- measure (fail-open) ---
                    if meas["streaming"]:
                        try:
                            self._feed_measure(chunk, chunk_mon, meas)
                        except Exception as e:
                            meas["capture_errors"].append(repr(e))
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass                                     # don't crash on client disconnect
            finally:
                try:
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                except Exception:
                    pass
                conn.close()
        except Exception as e:
            meas["relay_errors"].append("handler: %r" % (e,))
            try:
                self._raw_error(500, "proxy internal error")
            except Exception:
                pass
        finally:
            self._finalize(meas, status, model, stream)

    # ------------------------------------------------------------------ #
    # SSE tokenization + measurement (side effect only, fail-open)        #
    # ------------------------------------------------------------------ #
    def _feed_measure(self, chunk, chunk_mon, meas):
        meas["sse_buffer"] += chunk.decode("utf-8", "replace")
        # keep runaway buffers bounded even if a server sends no \n\n
        if len(meas["sse_buffer"]) > 1_000_000:
            meas["sse_buffer"] = meas["sse_buffer"][-1_000_000:]
            return
        while True:
            idx = meas["sse_buffer"].find("\n\n")
            if idx == -1:
                break
            block = meas["sse_buffer"][:idx]
            meas["sse_buffer"] = meas["sse_buffer"][idx + 2:]
            self._track_event(block, chunk_mon, meas)

    def _track_event(self, block, chunk_mon, meas):
        data_lines = [ln[5:].strip() for ln in block.splitlines() if ln.startswith("data:")]
        if not data_lines:
            return
        payload = "".join(data_lines).strip()
        if payload == "[DONE]":
            return
        meas["n_sse_events"] += 1
        try:
            ev = json.loads(payload)
        except Exception:
            return

        try:
            choices = ev.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                fr = choices[0].get("finish_reason")
                if fr is not None:
                    meas["finish_reason"] = fr
                content = delta.get("content")
                if isinstance(content, str) and content != "":
                    meas["streamed_content_chunks"] += 1
                    meas["content_chunk_times"].append(chunk_mon)
                    if meas["first_content_chunk_time"] is None:
                        meas["first_content_chunk_time"] = chunk_mon
                tcs = delta.get("tool_calls")
                if isinstance(tcs, list):
                    for tc in tcs:
                        fn = tc.get("function") or {}
                        meas["tool_calls"].append({
                            "id": tc.get("id"),
                            "name": fn.get("name"),
                            "arguments": fn.get("arguments"),
                        })
        except Exception:
            pass
        if "usage" in ev:
            meas["usage"] = ev.get("usage")

    # ------------------------------------------------------------------ #
    # Emit the one JSONL line for this request                            #
    # ------------------------------------------------------------------ #
    def _finalize(self, meas, status, model, stream):
        try:
            wall = time.perf_counter() - meas["start_monotonic"]
            end_epoch = time.time()
            first = meas["first_content_chunk_time"]
            ttft = (first - meas["start_monotonic"]) if first is not None else None

            streamed = meas["streamed_content_chunks"]
            ct = meas["content_chunk_times"]
            gaps = [ct[i] - ct[i - 1] for i in range(1, len(ct))] if len(ct) > 1 else []

            gap_summary = {}
            if gaps:
                gap_summary = {
                    "mean_s": round(statistics.mean(gaps), 6),
                    "median_s": round(statistics.median(gaps), 6),
                    "max_s": round(max(gaps), 6),
                    "min_s": round(min(gaps), 6),
                    "count": len(gaps),
                }

            usage = meas["usage"] or {}
            denom_pt = wall if wall > 0 else 1e-9
            if ttft is not None and wall > ttft:
                denom_pt = wall - ttft

            tool_names = [t["name"] for t in meas["tool_calls"] if t.get("name")]
            ended_in_tool = (
                meas["finish_reason"] == "tool_calls"
                or bool(meas["tool_calls"])  # and no plain text completion follows
            )

            record = {
                # identity & wall clock
                "request_id": meas["request_id"],
                "ts_start_epoch": round(meas["start_ts_epoch"], 6),
                "ts_end_epoch": round(end_epoch, 6),
                "wall_duration_s": round(wall, 6),
                # request metadata
                "method": meas["method"],
                "path": meas["path"],
                "status": status,
                "model": model,
                "stream": stream,
                # streaming timing
                "streaming": meas["streaming"],
                "ttft_s": round(ttft, 6) if ttft is not None else None,
                "ts_first_content_epoch": round(meas["start_ts_epoch"] + ttft, 6) if ttft is not None else None,
                # tokens: streamed count (client-perceived) vs server-reported usage
                "completion_tokens_streamed": streamed,
                "completion_tokens_usage": usage.get("completion_tokens"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens") if usage.get("prompt_tokens_details") else None,
                "prompt_tokens_details": usage.get("prompt_tokens_details"),
                # inter-chunk arrival gaps (streaming stalls are visible here)
                "inter_chunk_gaps_s": [round(g, 6) for g in gaps],
                "inter_chunk_gap_summary": gap_summary,
                # rates -- two conventions side by side
                "post_ttft_rate_toks_per_s": round(streamed / denom_pt, 3) if streamed else None,
                "full_wall_rate_toks_per_s": round(streamed / wall, 3) if streamed and wall > 0 else None,
                # completion semantics
                "finish_reason": meas["finish_reason"],
                "has_tool_calls": bool(meas["tool_calls"]),
                "n_tool_calls": len(meas["tool_calls"]),
                "tool_call_names": tool_names,
                "ended_in_tool_call": ended_in_tool,
                # misc / diagnostics
                "n_sse_events": meas["n_sse_events"],
                "capture_errors": meas["capture_errors"],
                "relay_errors": meas["relay_errors"],
            }
            if hasattr(self.server, "capture_write"):
                self.server.capture_write(json.dumps(record))
        except Exception as e:
            sys.stderr.write("[capture] finalize failed: %r\n" % (e,))

    # ------------------------------------------------------------------ #
    def _raw_error(self, code, msg):
        """Send a plain-text error response (used when upstream is unreachable)."""
        body = msg.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
            self.wfile.flush()
        except Exception:
            pass


class CaptureServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True                       # Ctrl-C exits cleanly

    def __init__(self, addr, upstream_host, upstream_port, jsonl_path):
        super().__init__(addr, CaptureHandler)
        self.upstream = (upstream_host, upstream_port)
        self.jsonl_path = jsonl_path
        self._lock = threading.Lock()

    def capture_write(self, record):
        try:
            with self._lock, open(self.jsonl_path, "a") as f:
                f.write(record + "\n")
                f.flush()
        except Exception as e:
            sys.stderr.write("[capture] jsonl write failed: %r\n" % (e,))


# --------------------------------------------------------------------------- #
def parse_upstream(s):
    """Returns (scheme, host, port). Only http is supported."""
    s = s.rstrip("/")
    if "://" not in s:
        s = "http://" + s
    scheme, _, rest = s.partition("://")
    scheme = scheme.lower()
    if scheme != "http":
        raise SystemExit("only http upstream is supported (got %r)" % scheme)
    hostport = rest.split("/", 1)[0]
    if ":" in hostport:
        host, _, port_s = hostport.rpartition(":")
        port = int(port_s)
    else:
        host, port = hostport, 80
    return scheme, host, port


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Passive non-buffering capture proxy for Hermes-on-exo.")
    ap.add_argument("--listen", default="127.0.0.1", help="proxy bind address (default 127.0.0.1)")
    ap.add_argument("--port", type=int, default=52416, help="proxy listen port (default 52416)")
    ap.add_argument("--upstream", default="http://192.168.86.201:52415",
                    help="upstream exo API (default http://192.168.86.201:52415)")
    ap.add_argument("--jsonl", default=os.path.join(here, "capture.jsonl"),
                    help="JSONL output path (default ./capture.jsonl)")
    args = ap.parse_args()

    scheme, host, port = parse_upstream(args.upstream)
    srv = CaptureServer((args.listen, args.port), host, port, args.jsonl)

    print("=" * 72)
    print("passive_capture_proxy  (stdlib only, non-buffering)")
    print("  proxy      : http://%s:%d  (this is what Hermes points at)" % (args.listen, args.port))
    print("  upstream   : %s://%s:%d%s" % (scheme, host, port, ""))
    print("  jsonl out  : %s" % args.jsonl)
    print("-" * 72)
    print("ONE-LINE CONFIG SWAP (point the exo provider at the proxy):")
    print("  exo provider base_url should be set to:")
    print("    http://%s:%d/v1" % (args.listen, args.port))
    print("  (currently configured in ~/.hermes/config.yaml as: http://192.168.86.201:52415/v1)")
    print("-" * 72)
    print("Start capturing, use your Hermes session normally, then Ctrl-C to stop.")
    print("Every request appends one JSON line to %s" % args.jsonl)
    print("=" * 72)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping.")
        srv.server_close()


if __name__ == "__main__":
    main()
