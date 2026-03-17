"""CPU Draft Process: runs draft model in a separate process to avoid fork+cblas segfault.

The draft process loads weights once at startup, then sits in a loop:
  1. Receive token ID from main process via pipe
  2. Run C/Accelerate forward pass (cblas_sgemv)
  3. Send predicted tokens back via pipe

Separate process = separate address space = no fork+Accelerate conflicts.
Apple Silicon unified memory means no data copy overhead.

Usage:
    draft = CPUDraftProcess("mlx-community/Qwen3-0.6B-8bit")
    draft.start()
    draft.request_draft(token_id=1234, num_tokens=2)
    # ... GPU does work ...
    tokens = draft.get_result()  # non-blocking check
    draft.stop()
"""
import multiprocessing as mp
import os
import time


def _draft_worker(model_id: str, request_pipe, result_pipe, ready_event):
    """Worker function that runs in a separate process."""
    # Set BLAS thread limits BEFORE importing anything
    os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"

    try:
        from exo.worker.engines.mlx.cpu_draft_engine import CPUDraftEngine

        engine = CPUDraftEngine(model_id)
        # Warmup
        engine.draft_sync(start_token=1, num_tokens=1)
        engine.reset_cache()

        ready_event.set()  # signal main process we're ready

        while True:
            if not request_pipe.poll(timeout=1.0):
                continue

            msg = request_pipe.recv()
            if msg is None:  # shutdown signal
                break

            cmd, token_id, num_tokens = msg
            if cmd == "draft":
                tokens = engine.draft_sync(token_id, num_tokens)
                result_pipe.send(tokens)
            elif cmd == "reset":
                engine.reset_cache()
                result_pipe.send("ok")
            elif cmd == "feed":
                # Feed a token to the cache without drafting
                engine._forward_one(token_id)
                result_pipe.send("ok")

    except Exception as e:
        ready_event.set()  # unblock main even on error
        result_pipe.send(f"ERROR: {e}")


class CPUDraftProcess:
    """Manages a separate process for CPU draft model inference."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._process = None
        self._request_pipe = None
        self._result_pipe = None
        self._ready = None
        self._pending = False

    def start(self, timeout: float = 30.0) -> bool:
        """Start the draft process. Returns True if ready."""
        if self._process is not None:
            return True

        req_recv, self._request_pipe = mp.Pipe(duplex=False)
        self._result_pipe, res_send = mp.Pipe(duplex=False)
        self._ready = mp.Event()

        self._process = mp.Process(
            target=_draft_worker,
            args=(self.model_id, req_recv, res_send, self._ready),
            daemon=True,
        )
        self._process.start()

        # Wait for worker to load and warm up
        if not self._ready.wait(timeout=timeout):
            self.stop()
            return False

        return self._process.is_alive()

    def stop(self):
        """Stop the draft process."""
        if self._request_pipe is not None:
            try:
                self._request_pipe.send(None)  # shutdown signal
            except (BrokenPipeError, OSError):
                pass
        if self._process is not None:
            self._process.join(timeout=5.0)
            if self._process.is_alive():
                self._process.kill()
            self._process = None
        self._request_pipe = None
        self._result_pipe = None
        self._pending = False

    def request_draft(self, token_id: int, num_tokens: int):
        """Request draft tokens. Non-blocking — starts the draft."""
        if self._process is None or not self._process.is_alive():
            return
        # Drain any old results
        while self._result_pipe.poll():
            self._result_pipe.recv()
        self._request_pipe.send(("draft", token_id, num_tokens))
        self._pending = True

    def get_result(self) -> list | None:
        """Get draft result. Non-blocking — returns None if not ready."""
        if not self._pending or self._result_pipe is None:
            return None
        if not self._result_pipe.poll():
            return None  # not ready yet
        result = self._result_pipe.recv()
        self._pending = False
        if isinstance(result, str) and result.startswith("ERROR"):
            return None
        return result

    def reset_cache(self):
        """Reset the draft model's KV cache."""
        if self._process is None or not self._process.is_alive():
            return
        while self._result_pipe.poll():
            self._result_pipe.recv()
        self._request_pipe.send(("reset", 0, 0))
        # Wait for ack
        self._result_pipe.recv()

    def feed_token(self, token_id: int):
        """Feed a token to the draft cache without drafting."""
        if self._process is None or not self._process.is_alive():
            return
        while self._result_pipe.poll():
            self._result_pipe.recv()
        self._request_pipe.send(("feed", token_id, 0))
        self._result_pipe.recv()  # wait for ack

    @property
    def is_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()
