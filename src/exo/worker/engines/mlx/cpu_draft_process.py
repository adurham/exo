"""CPU Draft wrapper: synchronous draft during GPU async eval.

Runs the CPU draft model DURING the GPU's async eval wait period.
No threads, no fork, no separate process. Just synchronous calls
timed to overlap with GPU execution.

Timeline:
  GPU: [async_eval dispatched] ........... [y.item() returns]
  CPU:    [draft_sync 16ms]   [idle 19ms]

The draft runs on CPU while GPU processes. When y.item() blocks,
the CPU draft is already done. Zero extra latency.
"""


class CPUDraftWrapper:
    """Wraps CPUDraftEngine with request/result API for async-style usage."""

    def __init__(self, engine):
        self.engine = engine
        self._last_result = None
        self._pending = False

    @classmethod
    def from_engine(cls, engine):
        return cls(engine)

    def request_draft(self, token_id: int, num_tokens: int):
        """Run draft synchronously. Called during GPU async eval period."""
        try:
            self._last_result = self.engine.draft_sync(token_id, num_tokens)
            self._pending = True
        except Exception:
            self._last_result = None
            self._pending = False

    def get_result(self) -> list | None:
        """Get the last draft result. Always ready (synchronous)."""
        if not self._pending:
            return None
        result = self._last_result
        self._pending = False
        self._last_result = None
        return result

    def reset_cache(self):
        """Reset draft KV cache."""
        self.engine.reset_cache()

    @property
    def is_alive(self) -> bool:
        return True

    def stop(self):
        pass
