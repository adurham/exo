#!/usr/bin/env python3
"""Lightweight draft token server for speculative decoding.

Runs on the MacBook (or any node with the draft model). Serves draft
predictions via a simple HTTP endpoint. Studios call this during
their decode loop to get draft tokens.

Usage:
  uv run python -m exo.worker.engines.mlx.draft_server \
    --model mlx-community/Qwen3-0.6B-8bit \
    --port 8199

Then Studios set EXO_DRAFT_SERVER=http://macbook-m4:8199
"""
import argparse
import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
import mlx.core as mx
from mlx_lm.utils import load
from mlx_lm.models.cache import make_prompt_cache


class DraftModel:
    def __init__(self, model_id: str, max_cache_size: int = 4096):
        print(f"Loading draft model: {model_id}")
        t0 = time.perf_counter()
        self.model, self.tokenizer = load(model_id)
        mx.eval(self.model.parameters())
        self.cache = make_prompt_cache(self.model)
        self.cache_len = 0
        self.max_cache_size = max_cache_size

        # Warmup
        self._generate_one(1)
        self.reset()

        print(f"Draft model ready in {time.perf_counter() - t0:.1f}s")

    def _generate_one(self, token_id: int) -> int:
        """Generate one token. Returns predicted next token ID."""
        tokens = mx.array([[token_id]])
        logits = self.model(tokens, cache=self.cache)
        mx.eval(logits)
        self.cache_len += 1
        return logits[0, -1].argmax().item()

    def draft(self, token_id: int, num_tokens: int) -> list[int]:
        """Generate num_tokens draft predictions starting from token_id."""
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
    draft_model = None  # Set by main()

    def do_POST(self):
        if self.path == "/draft":
            content_length = int(self.headers.get('Content-Length', 0))
            body = json.loads(self.rfile.read(content_length))

            token_id = body.get("token_id", 1)
            num_tokens = body.get("num_tokens", 2)

            t0 = time.perf_counter()
            tokens = self.draft_model.draft(token_id, num_tokens)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                "tokens": tokens,
                "elapsed_ms": elapsed_ms,
            }).encode())

        elif self.path == "/reset":
            self.draft_model.reset()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')

        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass  # suppress logs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Qwen3-0.6B-8bit")
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
