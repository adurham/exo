"""Draft client: requests draft tokens from a remote draft server.

Non-blocking API — request_draft starts the HTTP call in a background
thread, get_result checks if the response arrived.

The HTTP request goes over TCP to whichever node runs the draft server.
Zero GPU contention on the primary model's nodes.

Uses persistent HTTP/1.1 keep-alive connection to minimize TCP overhead.
"""
import http.client
import json
import threading
import time
import urllib.request
from urllib.parse import urlparse

from exo.worker.runner.bootstrap import logger


class DraftClient:
    """Non-blocking client for the draft token server."""

    def __init__(self, server_url: str, num_draft_tokens: int = 10, draft_model: str = ""):
        self.server_url = server_url.rstrip('/')
        self.num_draft_tokens = num_draft_tokens
        self.draft_model = draft_model
        self._thread: threading.Thread | None = None
        self._result: list[int] | None = None
        self._pending = False
        self._num_to_trim = 0
        self._step = 0
        self._total_exchange_ms = 0.0

        # Persistent HTTP connection for keep-alive
        parsed = urlparse(self.server_url)
        self._host = parsed.hostname or "localhost"
        self._port = parsed.port or 52415
        self._conn: http.client.HTTPConnection | None = None
        self._conn_lock = threading.Lock()

    def _get_conn(self) -> http.client.HTTPConnection:
        """Get or create persistent HTTP connection."""
        if self._conn is None:
            self._conn = http.client.HTTPConnection(self._host, self._port, timeout=5.0)
        return self._conn

    def _post(self, path: str, body: dict, timeout: float = 5.0) -> dict | None:
        """POST with persistent connection, fallback to new connection on error."""
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
                    # Connection stale — reconnect once
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
            logger.info(
                f"Draft prefill: {len(token_ids)} tokens → cache_len={cache_len} "
                f"in {elapsed_ms:.1f}ms"
            )
            return cache_len
        except Exception as e:
            logger.warning(f"Draft prefill failed: {e}")
            return None

    def request_draft(self, token_id: int, num_tokens: int = 0):
        """Start a draft request in a background thread. Non-blocking."""
        if self._thread is not None and self._thread.is_alive():
            return  # previous request still running

        num = num_tokens or self.num_draft_tokens
        trim = self._num_to_trim
        self._num_to_trim = 0
        self._result = None
        self._pending = True
        self._step += 1
        step = self._step

        def _fetch():
            t0 = time.perf_counter()
            try:
                result = self._post("/v1/draft", {
                    "model": self.draft_model,
                    "token_id": token_id,
                    "num_tokens": num,
                    "trim": trim,
                })
                if result is not None:
                    self._result = result.get("tokens")
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    self._total_exchange_ms += elapsed_ms
                    if step <= 5 or step % 50 == 0:
                        logger.info(
                            f"Draft exchange step {step}: {elapsed_ms:.1f}ms "
                            f"(token={token_id}, trim={trim}, got {len(self._result)} drafts)"
                        )
            except Exception as e:
                if step <= 5:
                    logger.warning(f"Draft exchange failed at step {step}: {e}")
                self._result = None

        self._thread = threading.Thread(target=_fetch, daemon=True)
        self._thread.start()

    def fetch_draft_sync(self, token_id: int, num_tokens: int = 0, trim: int = 0) -> list[int]:
        """Blocking draft request — no background thread. Use when you need the result immediately."""
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
                    logger.info(
                        f"Draft exchange step {step}: {elapsed_ms:.1f}ms "
                        f"(token={token_id}, trim={trim}, got {len(tokens)} drafts)"
                    )
                return tokens
        except Exception as e:
            if step <= 5:
                logger.warning(f"Draft exchange failed at step {step}: {e}")
        return []

    def set_trim(self, num_rejected: int):
        """Set how many tokens the draft server should trim on next request."""
        self._num_to_trim = num_rejected

    def get_result(self) -> list[int] | None:
        """Get draft result. Non-blocking — returns None if not ready."""
        if not self._pending:
            return None
        if self._thread is not None and self._thread.is_alive():
            return None  # still fetching
        self._pending = False
        self._thread = None
        return self._result

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
            logger.info(
                f"Draft client stats: {self._step} exchanges, "
                f"avg {avg_ms:.1f}ms per exchange"
            )
        with self._conn_lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None

    @property
    def is_alive(self) -> bool:
        try:
            urllib.request.urlopen(f"{self.server_url}/v1/draft/health", timeout=2.0)
            return True
        except Exception:
            return False
