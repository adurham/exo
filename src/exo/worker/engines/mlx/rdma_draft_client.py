"""RDMA Draft Client: sends tokens to draft node and receives predictions via JACCL RDMA.

Uses mx.distributed.send/recv over the existing hybrid JACCL group.
The draft node (MacBook) is at a specific rank in the group.

Has the same request_draft/get_result API as DraftClient/CPUDraftWrapper
so it plugs into the ExoBatchGenerator callback.
"""
import threading
import time

import mlx.core as mx

from exo.worker.runner.bootstrap import logger


class RDMADraftClient:
    """Sends/receives draft tokens via JACCL RDMA send/recv."""

    def __init__(self, group: mx.distributed.Group, draft_rank: int, num_draft_tokens: int = 10):
        self.group = group
        self.draft_rank = draft_rank
        self.num_draft_tokens = num_draft_tokens
        self._thread: threading.Thread | None = None
        self._result: list[int] | None = None
        self._pending = False
        self._num_to_trim = 0  # rejected tokens from last step
        self._step = 0
        self._total_exchange_ms = 0.0

    def request_draft(self, token_id: int, num_tokens: int = 0):
        """Send accepted token + trim count to draft node, receive predictions.
        Non-blocking — runs in background thread."""
        if self._thread is not None and self._thread.is_alive():
            return

        num = num_tokens or self.num_draft_tokens
        trim = self._num_to_trim
        self._num_to_trim = 0
        self._result = None
        self._pending = True
        self._step += 1
        step = self._step

        def _exchange():
            t0 = time.perf_counter()
            try:
                # Send control: [accepted_token_id, num_to_trim]
                ctrl = mx.array([token_id, trim], dtype=mx.int32)
                mx.distributed.send(ctrl, self.draft_rank, group=self.group)
                mx.eval(ctrl)

                # Receive draft predictions
                preds = mx.distributed.recv(
                    shape=(num,), dtype=mx.int32,
                    src=self.draft_rank, group=self.group)
                mx.eval(preds)
                self._result = preds.tolist()
                elapsed_ms = (time.perf_counter() - t0) * 1000
                self._total_exchange_ms += elapsed_ms
                if step <= 5 or step % 50 == 0:
                    logger.info(
                        f"RDMA draft exchange step {step}: {elapsed_ms:.1f}ms "
                        f"(token={token_id}, trim={trim}, got {len(self._result)} drafts)"
                    )
            except Exception as e:
                logger.warning(f"RDMA draft exchange failed at step {step}: {e}")
                self._result = None

        self._thread = threading.Thread(target=_exchange, daemon=True)
        self._thread.start()

    def set_trim(self, num_rejected: int):
        """Set how many tokens the draft node should trim on next request."""
        self._num_to_trim = num_rejected

    def get_result(self) -> list[int] | None:
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
        pass

    def shutdown(self):
        """Send shutdown signal to draft node (negative token)."""
        try:
            logger.info(f"Sending shutdown to draft node (rank={self.draft_rank})")
            ctrl = mx.array([-1, 0], dtype=mx.int32)
            mx.distributed.send(ctrl, self.draft_rank, group=self.group)
            mx.eval(ctrl)
            if self._step > 0:
                avg_ms = self._total_exchange_ms / self._step
                logger.info(
                    f"RDMA draft stats: {self._step} exchanges, "
                    f"avg {avg_ms:.1f}ms per exchange"
                )
        except Exception as e:
            logger.warning(f"Failed to send shutdown to draft node: {e}")
