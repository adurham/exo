#!/usr/bin/env python3
"""Lightweight draft token server for speculative decoding.

Runs on any node with the draft model. Serves draft predictions via
simple HTTP endpoints. The primary model's decode loop queries this
to get draft tokens for multi-token verification.

Usage:
  uv run python -m exo.worker.engines.mlx.draft_server \
    --model mlx-community/Qwen3-1.7B-8bit \
    --port 8199

Then set EXO_DRAFT_SERVER=http://<host>:8199 on primary model nodes.

Endpoints:
  POST /draft   {"token_id": int, "num_tokens": int, "trim": int}
                 → {"tokens": [int...], "elapsed_ms": float}
  POST /prefill  {"token_ids": [int...]}
                 → {"cache_len": int, "elapsed_ms": float}
  POST /reset    {}
                 → {"status": "ok"}
  GET  /health   → {"status": "ok", "cache_len": int, "model": str}
"""
import argparse
import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

import mlx.core as mx
from mlx_lm.utils import load
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache


class DraftModel:
    def __init__(self, model_id: str):
        print(f"Loading draft model: {model_id}")
        t0 = time.perf_counter()
        self.model, self.tokenizer = load(model_id)
        mx.eval(self.model.parameters())
        self.model_id = model_id
        self.cache = make_prompt_cache(self.model)
        self.cache_len = 0

        # Warmup
        self._generate_one(1)
        self.reset()

        print(f"Draft model ready in {time.perf_counter() - t0:.1f}s")

    def _generate_one(self, token_id: int) -> int:
        """Feed one token through model. Returns predicted next token ID."""
        tokens = mx.array([[token_id]])
        logits = self.model(tokens, cache=self.cache)
        mx.eval(logits)
        self.cache_len += 1
        return logits[0, -1].argmax().item()

    def prefill(self, token_ids: list[int]) -> int:
        """Prefill KV cache with token sequence. Returns cache length."""
        if not token_ids:
            return self.cache_len
        tokens = mx.array([token_ids])
        logits = self.model(tokens, cache=self.cache)
        mx.eval(logits)
        self.cache_len += len(token_ids)
        return self.cache_len

    def draft(self, token_id: int, num_tokens: int, trim: int = 0) -> list[int]:
        """Generate num_tokens draft predictions starting from token_id.

        If trim > 0, trims that many tokens from the KV cache first
        (for rejected draft tokens from the previous verification step).
        """
        if trim > 0:
            trim_prompt_cache(self.cache, trim)
            self.cache_len = max(0, self.cache_len - trim)

        results = []
        tok = token_id
        for _ in range(num_tokens):
            tok = self._generate_one(tok)
            results.append(tok)
        return results

    def reset(self):
        """Reset KV cache."""
        self.cache = make_prompt_cache(self.model)
        self.cache_len = 0


class DraftHandler(BaseHTTPRequestHandler):
    draft_model: DraftModel | None = None

    def do_GET(self):
        if self.path == "/health":
            assert self.draft_model is not None
            self._json_response({
                "status": "ok",
                "cache_len": self.draft_model.cache_len,
                "model": self.draft_model.model_id,
            })
        else:
            self.send_error(404)

    def do_POST(self):
        assert self.draft_model is not None
        body = self._read_body()

        if self.path == "/draft":
            token_id = body.get("token_id", 1)
            num_tokens = body.get("num_tokens", 10)
            trim = body.get("trim", 0)

            t0 = time.perf_counter()
            tokens = self.draft_model.draft(token_id, num_tokens, trim=trim)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            self._json_response({
                "tokens": tokens,
                "elapsed_ms": elapsed_ms,
            })

        elif self.path == "/prefill":
            token_ids = body.get("token_ids", [])

            t0 = time.perf_counter()
            cache_len = self.draft_model.prefill(token_ids)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            self._json_response({
                "cache_len": cache_len,
                "elapsed_ms": elapsed_ms,
            })

        elif self.path == "/reset":
            self.draft_model.reset()
            self._json_response({"status": "ok"})

        else:
            self.send_error(404)

    def _read_body(self) -> dict:
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length == 0:
            return {}
        return json.loads(self.rfile.read(content_length))

    def _json_response(self, data: dict, status: int = 200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, format, *args):
        pass  # suppress request logs


def main():
    parser = argparse.ArgumentParser(description="Draft token server for speculative decoding")
    parser.add_argument("--model", default="mlx-community/Qwen3-1.7B-8bit")
    parser.add_argument("--port", type=int, default=8199)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    DraftHandler.draft_model = DraftModel(args.model)

    server = HTTPServer((args.host, args.port), DraftHandler)
    print(f"Draft server listening on {args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    server.server_close()


if __name__ == "__main__":
    main()
