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
import os
import select
import signal
import struct
import time


class CPUDraftProcess:
    """Manages a separate process for CPU draft model inference.

    Uses os.fork() directly instead of multiprocessing.Process to avoid
    the 'daemonic processes cannot have children' restriction.
    Communication via raw pipes (os.pipe) with simple binary protocol.
    """

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._pid = None
        self._to_child_w = None   # write end: parent → child
        self._from_child_r = None  # read end: child → parent
        self._pending = False

    def start(self, timeout: float = 30.0) -> bool:
        """Start the draft process via os.fork(). Returns True if ready."""
        if self._pid is not None:
            return True

        # Create pipes: parent writes to child, child writes to parent
        to_child_r, to_child_w = os.pipe()
        from_child_r, from_child_w = os.pipe()

        pid = os.fork()
        if pid == 0:
            # === CHILD PROCESS ===
            os.close(to_child_w)
            os.close(from_child_r)
            # Ignore SIGINT in child (parent handles cleanup)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            self._child_loop(self.model_id, to_child_r, from_child_w)
            os._exit(0)
        else:
            # === PARENT PROCESS ===
            os.close(to_child_r)
            os.close(from_child_w)
            self._pid = pid
            self._to_child_w = to_child_w
            self._from_child_r = from_child_r

            # Wait for "ready" signal from child
            start = time.time()
            while time.time() - start < timeout:
                r, _, _ = select.select([self._from_child_r], [], [], 1.0)
                if r:
                    data = os.read(self._from_child_r, 1)
                    if data == b'R':
                        return True
                    elif data == b'E':
                        self.stop()
                        return False
            self.stop()
            return False

    @staticmethod
    def _child_loop(model_id, read_fd, write_fd):
        """Child process main loop."""
        os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
        os.environ["OPENBLAS_NUM_THREADS"] = "4"

        try:
            from exo.worker.engines.mlx.cpu_draft_engine import CPUDraftEngine
            engine = CPUDraftEngine(model_id)
            engine.draft_sync(start_token=1, num_tokens=1)
            engine.reset_cache()
            os.write(write_fd, b'R')  # ready signal
        except Exception:
            os.write(write_fd, b'E')  # error signal
            return

        while True:
            try:
                r, _, _ = select.select([read_fd], [], [], 1.0)
                if not r:
                    continue
                header = os.read(read_fd, 9)  # 1 byte cmd + 4 byte token + 4 byte num
                if len(header) < 9:
                    break  # pipe closed
                cmd = header[0:1]
                token_id = struct.unpack('i', header[1:5])[0]
                num_tokens = struct.unpack('i', header[5:9])[0]

                if cmd == b'Q':  # quit
                    break
                elif cmd == b'D':  # draft
                    tokens = engine.draft_sync(token_id, num_tokens)
                    # Send: 4 byte count + 4 bytes per token
                    os.write(write_fd, struct.pack('i', len(tokens)))
                    for t in tokens:
                        os.write(write_fd, struct.pack('i', t))
                elif cmd == b'R':  # reset cache
                    engine.reset_cache()
                    os.write(write_fd, b'K')
                elif cmd == b'F':  # feed token
                    engine._forward_one(token_id)
                    os.write(write_fd, b'K')
            except (OSError, BrokenPipeError):
                break

    def stop(self):
        """Stop the draft process."""
        if self._to_child_w is not None:
            try:
                os.write(self._to_child_w, b'Q' + struct.pack('ii', 0, 0))
            except (OSError, BrokenPipeError):
                pass
            try:
                os.close(self._to_child_w)
            except OSError:
                pass
            self._to_child_w = None
        if self._from_child_r is not None:
            try:
                os.close(self._from_child_r)
            except OSError:
                pass
            self._from_child_r = None
        if self._pid is not None:
            try:
                os.waitpid(self._pid, os.WNOHANG)
            except ChildProcessError:
                pass
            self._pid = None
        self._pending = False

    def request_draft(self, token_id: int, num_tokens: int):
        """Request draft tokens. Non-blocking — starts the draft."""
        if self._to_child_w is None:
            return
        # Drain any old results
        while self._from_child_r is not None:
            r, _, _ = select.select([self._from_child_r], [], [], 0)
            if not r:
                break
            os.read(self._from_child_r, 4096)
        try:
            os.write(self._to_child_w, b'D' + struct.pack('ii', token_id, num_tokens))
            self._pending = True
        except (OSError, BrokenPipeError):
            self._pending = False

    def get_result(self) -> list | None:
        """Get draft result. Non-blocking — returns None if not ready."""
        if not self._pending or self._from_child_r is None:
            return None
        r, _, _ = select.select([self._from_child_r], [], [], 0)
        if not r:
            return None
        try:
            count_data = os.read(self._from_child_r, 4)
            if len(count_data) < 4:
                self._pending = False
                return None
            count = struct.unpack('i', count_data)[0]
            tokens = []
            for _ in range(count):
                t_data = os.read(self._from_child_r, 4)
                tokens.append(struct.unpack('i', t_data)[0])
            self._pending = False
            return tokens
        except (OSError, BrokenPipeError):
            self._pending = False
            return None

    def reset_cache(self):
        """Reset the draft model's KV cache."""
        if self._to_child_w is None:
            return
        try:
            os.write(self._to_child_w, b'R' + struct.pack('ii', 0, 0))
            os.read(self._from_child_r, 1)  # wait for ack
        except (OSError, BrokenPipeError):
            pass

    @property
    def is_alive(self) -> bool:
        if self._pid is None:
            return False
        try:
            os.kill(self._pid, 0)  # check if process exists
            return True
        except ProcessLookupError:
            return False
