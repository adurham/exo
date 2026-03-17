"""Draft client: requests draft tokens from a remote draft server.

Non-blocking API — request_draft starts the HTTP call in a background
thread, get_result checks if the response arrived.

The HTTP request goes over TCP to whichever node runs the draft server.
Zero GPU contention on the primary model's nodes.

Set EXO_DRAFT_SERVER=http://<host>:8199 to enable.
"""
import json
import threading
import time
import urllib.request

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

    def prefill(self, token_ids: list[int]) -> int | None:
        """Prefill the draft model's KV cache with prompt tokens. Blocking."""
        try:
            t0 = time.perf_counter()
            data = json.dumps({"model": self.draft_model, "token_ids": token_ids}).encode()
            req = urllib.request.Request(
                f"{self.server_url}/v1/draft/prefill",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=30.0) as resp:
                result = json.loads(resp.read())
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
                data = json.dumps({
                    "model": self.draft_model,
                    "token_id": token_id,
                    "num_tokens": num,
                    "trim": trim,
                }).encode()
                req = urllib.request.Request(
                    f"{self.server_url}/v1/draft",
                    data=data,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=30.0) as resp:
                    result = json.loads(resp.read())
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
            req = urllib.request.Request(
                f"{self.server_url}/v1/draft/reset",
                data=json.dumps({"model": self.draft_model}).encode(),
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=5.0)
        except Exception:
            pass

    def shutdown(self):
        """Log stats. Draft server stays running independently."""
        if self._step > 0:
            avg_ms = self._total_exchange_ms / self._step
            logger.info(
                f"Draft client stats: {self._step} exchanges, "
                f"avg {avg_ms:.1f}ms per exchange"
            )

    @property
    def is_alive(self) -> bool:
        try:
            urllib.request.urlopen(f"{self.server_url}/v1/draft/health", timeout=2.0)
            return True
        except Exception:
            return False
