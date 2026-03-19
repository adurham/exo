"""Draft client: requests draft tokens from a remote draft server.

Blocking API — fetch_draft_sync makes a synchronous HTTP call and returns
the draft tokens. Used by the speculative decode loop in generate.py.

Uses persistent HTTP/1.1 keep-alive connection over Thunderbolt to minimize
TCP overhead (~2ms round-trip).
"""
import http.client
import json
import time
import threading
from urllib.parse import urlparse

from exo.worker.runner.bootstrap import logger


class DraftClient:
    """Blocking client for the draft token server."""

    def __init__(self, server_url: str, num_draft_tokens: int = 5, draft_model: str = ""):
        self.server_url = server_url.rstrip('/')
        self.num_draft_tokens = num_draft_tokens
        self.draft_model = draft_model
        self._step = 0
        self._total_exchange_ms = 0.0

        parsed = urlparse(self.server_url)
        self._host = parsed.hostname or "localhost"
        self._port = parsed.port or 52417
        self._conn: http.client.HTTPConnection | None = None
        self._conn_lock = threading.Lock()

    def _get_conn(self) -> http.client.HTTPConnection:
        if self._conn is None:
            self._conn = http.client.HTTPConnection(self._host, self._port, timeout=5.0)
        return self._conn

    def _post(self, path: str, body: dict, timeout: float = 5.0) -> dict | None:
        data = json.dumps(body).encode()
        headers = {"Content-Type": "application/json", "Connection": "keep-alive"}
        with self._conn_lock:
            for attempt in range(2):
                try:
                    conn = self._get_conn()
                    conn.timeout = timeout
                    conn.request("POST", path, body=data, headers=headers)
                    resp = conn.getresponse()
                    return json.loads(resp.read())
                except Exception:
                    try:
                        if self._conn is not None:
                            self._conn.close()
                    except Exception:
                        pass
                    self._conn = None
                    if attempt > 0:
                        raise
        return None

    def prefill(self, token_ids: list[int]) -> int | None:
        """Prefill the draft model's KV cache with prompt tokens. Blocking."""
        try:
            t0 = time.perf_counter()
            result = self._post("/v1/draft/prefill", {"model": self.draft_model, "token_ids": token_ids}, timeout=30.0)
            if result is None:
                return None
            elapsed_ms = (time.perf_counter() - t0) * 1000
            cache_len = result.get("cache_len", 0)
            logger.info(f"Draft prefill: {len(token_ids)} tokens → cache_len={cache_len} in {elapsed_ms:.1f}ms")
            return cache_len
        except Exception as e:
            logger.warning(f"Draft prefill failed: {e}")
            return None

    def fetch_draft_sync(self, token_id: int, num_tokens: int = 0, trim: int = 0) -> list[int]:
        """Blocking draft request — returns draft token IDs."""
        num = num_tokens or self.num_draft_tokens
        self._step += 1
        step = self._step
        t0 = time.perf_counter()
        try:
            result = self._post("/v1/draft", {
                "model": self.draft_model,
                "token_id": token_id,
                "num_tokens": num,
                "trim": trim,
            })
            if result is not None:
                tokens = result.get("tokens", [])
                elapsed_ms = (time.perf_counter() - t0) * 1000
                self._total_exchange_ms += elapsed_ms
                if step <= 5 or step % 50 == 0:
                    logger.info(f"Draft exchange step {step}: {elapsed_ms:.1f}ms (token={token_id}, trim={trim}, got {len(tokens)} drafts)")
                return tokens
        except Exception as e:
            if step <= 5:
                logger.warning(f"Draft exchange failed at step {step}: {e}")
        return []

    def reset_cache(self):
        """Reset the remote draft model's KV cache."""
        try:
            self._post("/v1/draft/reset", {"model": self.draft_model})
        except Exception:
            pass

    def shutdown(self):
        """Log stats and close persistent connection."""
        if self._step > 0:
            avg_ms = self._total_exchange_ms / self._step
            logger.info(f"Draft client stats: {self._step} exchanges, avg {avg_ms:.1f}ms per exchange")
        with self._conn_lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None
