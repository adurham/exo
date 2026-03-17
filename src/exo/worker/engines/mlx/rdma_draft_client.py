"""RDMA Draft Client: sends tokens to draft node and receives predictions via JACCL RDMA.

Uses mx.distributed.send/recv over the existing hybrid JACCL group.
The draft node (MacBook) is at a specific rank in the group.

Has the same request_draft/get_result API as DraftClient/CPUDraftWrapper
so it plugs into the ExoBatchGenerator callback.
"""
import threading
import mlx.core as mx


class RDMADraftClient:
    """Sends/receives draft tokens via JACCL RDMA send/recv."""

    def __init__(self, group: mx.distributed.Group, draft_rank: int, num_draft_tokens: int = 2):
        self.group = group
        self.draft_rank = draft_rank
        self.num_draft_tokens = num_draft_tokens
        self._thread = None
        self._result = None
        self._pending = False

    def request_draft(self, token_id: int, num_tokens: int = 0):
        """Send token to draft node and start receiving predictions.
        Non-blocking — runs in background thread."""
        if self._thread is not None and self._thread.is_alive():
            return

        num = num_tokens or self.num_draft_tokens
        self._result = None
        self._pending = True

        def _exchange():
            try:
                # Send current token to draft node
                token = mx.array([token_id], dtype=mx.int32)
                mx.distributed.send(token, self.draft_rank, group=self.group)
                mx.eval(token)

                # Receive draft predictions
                preds = mx.distributed.recv(
                    shape=(num,), dtype=mx.int32,
                    src=self.draft_rank, group=self.group)
                mx.eval(preds)
                self._result = preds.tolist()
            except Exception:
                self._result = None

        self._thread = threading.Thread(target=_exchange, daemon=True)
        self._thread.start()

    def get_result(self) -> list | None:
        """Get draft result. Non-blocking."""
        if not self._pending:
            return None
        if self._thread is not None and self._thread.is_alive():
            return None
        self._pending = False
        self._thread = None
        return self._result

    @property
    def is_alive(self) -> bool:
        return True

    def reset_cache(self):
        """Send reset signal to draft node."""
        pass  # TODO: send negative token as reset signal
