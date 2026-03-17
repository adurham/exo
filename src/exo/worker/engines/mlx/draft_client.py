"""Draft client: requests draft tokens from a remote draft server.

Non-blocking API — request_draft starts the HTTP call in a background
thread, get_result checks if the response arrived.

The HTTP request goes over the Thunderbolt network to the MacBook,
which runs the draft server with the 0.6B model on its own GPU.
Zero GPU contention on the Studios.
"""
import json
import threading
import urllib.request


class DraftClient:
    """Non-blocking client for the draft token server."""

    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip('/')
        self._thread = None
        self._result = None
        self._pending = False

    def request_draft(self, token_id: int, num_tokens: int):
        """Start a draft request in a background thread. Non-blocking."""
        if self._thread is not None and self._thread.is_alive():
            return  # previous request still running

        self._result = None
        self._pending = True

        def _fetch():
            try:
                data = json.dumps({
                    "token_id": token_id,
                    "num_tokens": num_tokens,
                }).encode()
                req = urllib.request.Request(
                    f"{self.server_url}/draft",
                    data=data,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=5.0) as resp:
                    result = json.loads(resp.read())
                    self._result = result.get("tokens")
            except Exception:
                self._result = None

        self._thread = threading.Thread(target=_fetch, daemon=True)
        self._thread.start()

    def get_result(self) -> list | None:
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
                f"{self.server_url}/reset",
                data=b'{}',
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=5.0)
        except Exception:
            pass

    @property
    def is_alive(self) -> bool:
        try:
            urllib.request.urlopen(f"{self.server_url}/draft",
                data=json.dumps({"token_id": 1, "num_tokens": 0}).encode(),
                timeout=2.0)
            return True
        except Exception:
            return False
