"""Lightweight draft token server for speculative decoding.

Runs as a thread inside the runner process. Serves draft predictions
via simple HTTP endpoints. The primary model's decode loop queries this
to get draft tokens for multi-token batch verification.

Endpoints:
  POST /v1/draft          {"token_id": int, "num_tokens": int, "trim": int}
                           → {"tokens": [int...], "elapsed_ms": float}
  POST /v1/draft/prefill   {"token_ids": [int...]}
                           → {"cache_len": int, "elapsed_ms": float}
  POST /v1/draft/reset     {}
                           → {"status": "ok"}
  GET  /v1/draft/health    → {"status": "ok", "cache_len": int}
"""
import json
import os
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

from exo.worker.runner.bootstrap import logger

DRAFT_SERVER_PORT = int(os.environ.get("EXO_DRAFT_SERVER_PORT", "52417"))


def start_draft_server(model, model_id: str, tokenizer=None) -> int | None:
    """Start a draft HTTP server in a daemon thread. Returns the port or None on failure."""
    lock = threading.Lock()
    cache = make_prompt_cache(model)
    cache_len = 0

    # Ban thinking/structural tokens from draft predictions — the verifier
    # (with enable_thinking=False) never produces these, so the draft shouldn't either.
    banned_ids: set[int] = set()
    if tokenizer is not None:
        for attr in ("think_start_id", "think_end_id"):
            tid = getattr(tokenizer, attr, None)
            if tid is not None:
                banned_ids.add(tid)
        logger.info(f"Draft server banning {len(banned_ids)} thinking token IDs: {banned_ids}")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/v1/draft/health":
                self._json({"status": "ok", "cache_len": cache_len, "model": model_id})
            else:
                self.send_error(404)

        def do_POST(self):
            nonlocal cache, cache_len
            body = self._read()
            if self.path == "/v1/draft":
                token_id = body.get("token_id", 1)
                num_tokens = body.get("num_tokens", 5)
                trim = body.get("trim", 0)
                t0 = time.perf_counter()
                with lock:
                    if trim > 0:
                        trim_prompt_cache(cache, trim)
                        cache_len = max(0, cache_len - trim)
                    tokens = []
                    tok = token_id
                    for _ in range(num_tokens):
                        logits = model(mx.array([[tok]]), cache=cache)
                        mx.eval(logits)
                        cache_len += 1
                        if banned_ids:
                            logit_vec = logits[0, -1]
                            for bid in banned_ids:
                                logit_vec[bid] = -float('inf')
                            mx.eval(logit_vec)
                            tok = logit_vec.argmax().item()
                        else:
                            tok = logits[0, -1].argmax().item()
                        tokens.append(tok)
                self._json({"tokens": tokens, "elapsed_ms": (time.perf_counter() - t0) * 1000})
            elif self.path == "/v1/draft/prefill":
                token_ids = body.get("token_ids", [])
                t0 = time.perf_counter()
                with lock:
                    if token_ids:
                        logits = model(mx.array([token_ids]), cache=cache)
                        mx.eval(logits)
                        cache_len += len(token_ids)
                self._json({"cache_len": cache_len, "elapsed_ms": (time.perf_counter() - t0) * 1000})
            elif self.path == "/v1/draft/reset":
                with lock:
                    cache = make_prompt_cache(model)
                    cache_len = 0
                self._json({"status": "ok"})
            else:
                self.send_error(404)

        def _read(self) -> dict:
            n = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(n)) if n else {}

        def _json(self, data: dict):
            out = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, format, *args):
            pass

    port = DRAFT_SERVER_PORT
    try:
        class ReuseServer(HTTPServer):
            allow_reuse_address = True
        server = ReuseServer(("0.0.0.0", port), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        logger.info(f"Draft server started on port {port}")
        return port
    except OSError as e:
        logger.warning(f"Could not start draft server on port {port}: {e}")
        return None
